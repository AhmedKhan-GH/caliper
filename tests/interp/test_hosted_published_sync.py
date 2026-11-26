"""Bounded CPU checks for status at published action completion."""
from copy import deepcopy

import pytest
import torch

from caliper import _wire
from caliper.interp import _publish
from .test_hosted import _trace_at_boundary
from .test_replication_groups import replicated_hosted


def queue(route, name, request_id):
    reply = route.on_control(None, _wire.Control(request_id, "run", name=name).to_meta())
    assert reply["status"] == "queued"


def slider(run):
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.boundary()
    run.clamp.set(2.)
    run.boundary()
    assert run._owned_clamp is not None and "Live slider clamp:" in run.clamp_status.value
    return graph


@pytest.mark.parametrize("status", ["clamp", "output"])
def test_success_syncs_status_before_action_reply(replicated_hosted, status):
    run, route = replicated_hosted
    graph = slider(run)
    run.status.set("Previous output refusal.")
    observed = []
    run.actions.append(run.controls.action("observe", lambda: observed.append(
        (run.clamp_status.value, run.status.value))))
    queue(route, "published_dallas_v1.cell18", 70)
    queue(route, "observe", 71)
    run.boundary()
    assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(70, "ok"), (71, "ok")]
    assert run._owned_clamp is run._owned_applied_spec is None
    assert run.service.graph is graph
    assert run._comparison["case_id"] == "published_dallas_v1.cell18"
    assert run._output_key == run._input_state()
    if status == "clamp":
        assert "No slider clamp." in observed[0][0]
        assert "Live slider clamp:" not in observed[0][0]
        assert "Clamp input: 2 times" in observed[0][0]
    else:
        assert observed[0][1] == "ready"
        run.boundary()
        assert run.status.value == "ready"


@pytest.mark.parametrize("enabled_before", [True, False])
def test_published_retrace_reports_limit_without_graph_replacement(replicated_hosted, monkeypatch, enabled_before):
    run, route = replicated_hosted
    graph = _trace_at_boundary(run, prompt="AB", max_targets=0)
    traced = (run.traced_prompt, deepcopy(run.traced_specifications), deepcopy(run.traced_state))
    public = {name: (tensor, tensor.clone(), _publish._VERSIONS[name])
              for name, tensor in _publish._TENSORS.items() if name.startswith("trace.")}
    if enabled_before:
        run.retrace.set(True)
        run.boundary()
        assert run._retrace
    prepare = run.output.prepare_logits

    def prepared(*args, **kwargs):
        result = prepare(*args, **kwargs)
        if not enabled_before:
            run.retrace.set(True)
        return result

    monkeypatch.setattr(run.output, "prepare_logits", prepared)
    monkeypatch.setattr(run.service, "_retrace", lambda *a, **k: pytest.fail("Published action requested an unsupported trace"))
    queue(route, "published_dallas_v1.cell18", 70)
    run.boundary()
    assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(70, "ok")]
    assert "automatic retrace" in run.status.value.lower()
    assert "unsupported" in run.status.value.lower()
    assert run._retrace
    comparison = deepcopy(run._comparison)
    result = deepcopy(run.replication.last_result)
    run.boundary()
    assert run._comparison == comparison and run.replication.last_result == result
    assert run.service.graph is graph
    assert (run.traced_prompt, run.traced_specifications, run.traced_state) == traced
    for name, (tensor, value, version) in public.items():
        assert _publish._resolve(name) is tensor and _publish._VERSIONS[name] == version
        torch.testing.assert_close(tensor, value, rtol=0, atol=0)


@pytest.mark.parametrize("cell", [10, 18])
def test_action_preserves_a_late_trace_request_and_its_status(replicated_hosted, monkeypatch, cell):
    run, route = replicated_hosted
    graph = _trace_at_boundary(run, prompt="AB", max_targets=0)
    prepare = run.output.prepare_logits
    replies = []
    params = {"prompt": "AB", "max_targets": 0}

    def prepared(*args, **kwargs):
        result = prepare(*args, **kwargs)
        run.retrace.set(True)
        accepted = run.service.submit({"version": 1, "method": "trace.run", "params": params},
                                      lambda *reply: replies.append(reply))
        assert accepted[0] == "queued"
        return result

    monkeypatch.setattr(run.output, "prepare_logits", prepared)
    monkeypatch.setattr(run.service, "_retrace", lambda *a, **k: pytest.fail("Automatic trace displaced a request"))
    queue(route, f"published_dallas_v1.cell{cell}", 70)
    run.boundary()
    assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(70, "ok")]
    assert run.service.graph is graph and not replies
    assert run.service.status == {"state": "pending", "seconds": 0., "step": None}
    assert len(run.service.queue) == 1 and run.service.queue[0][1] == params
    assert "unsupported" in run.status.value and run._retrace
    assert run._comparison["case_id"] == f"published_dallas_v1.cell{cell}"


def test_action_preserves_pending_slider_edit_and_single_output(replicated_hosted, monkeypatch):
    run, route = replicated_hosted
    slider(run)
    prepare = run.output.prepare_logits
    forwards, candidates = [], []

    def prepared(*args, **kwargs):
        result = prepare(*args, **kwargs)
        candidates.append(result)
        edited = route.on_control(None, _wire.Control(72, "set", name="clamp", value=3.).to_meta())
        assert edited["status"] == "ok"
        return result

    monkeypatch.setattr(run.output, "prepare_logits", prepared)
    monkeypatch.setattr(run.output, "compare", lambda *a, **k: pytest.fail("Action repeated the output comparison"))
    from caliper.interp import _replication_actions as actions
    forward = actions.forward_logits

    def counted(*args, **kwargs):
        forwards.append(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(actions, "forward_logits", counted)
    run.apply_published_case(18)
    assert forwards == [1] and len(candidates) == 1
    assert run.clamp.pending == 3. and run.clamp.value == 2.
    assert "No slider clamp." in run.clamp_status.value
    assert "Clamp input: 2 times" in run.clamp_status.value
    assert run._owned_clamp is run._owned_applied_spec is None
    assert run._output_key == run._input_state() and run.status.value == "ready"
    assert run._comparison["argmax_token_ids"] == run.replication.last_result["argmax_token_ids"]
    assert run._comparison["case_id"] == run.replication.last_result["case_id"]


def test_failed_action_preserves_slider_status_and_output_identity(replicated_hosted, monkeypatch):
    run, _ = replicated_hosted
    graph = slider(run)
    run.status.set("Previous output refusal.")
    run.retrace.set(True)
    owned, applied = run._owned_clamp, run._owned_applied_spec
    before = (run.clamp_status.value, run.status.value, run._retrace,
              deepcopy(run._comparison), deepcopy(run._output_key), run.interventions.state_dict())
    public = {name: (tensor, tensor.clone(), _publish._VERSIONS[name])
              for name, tensor in _publish._TENSORS.items()}

    def refuse(*args, **kwargs):
        raise ValueError("Candidate output refusal.")

    monkeypatch.setattr(run.output, "prepare_logits", refuse)
    with pytest.raises(ValueError, match="Candidate output refusal"):
        run.apply_published_case(18)
    assert run._owned_clamp is owned and run._owned_applied_spec is applied
    assert (run.clamp_status.value, run.status.value, run._retrace,
            run._comparison, run._output_key, run.interventions.state_dict()) == before
    assert run.service.graph is graph and run.replication.last_attempt["status"] == "error"
    assert set(_publish._TENSORS) == set(public)
    for name, (tensor, value, version) in public.items():
        assert _publish._resolve(name) is tensor and _publish._VERSIONS[name] == version
        torch.testing.assert_close(tensor, value, rtol=0, atol=0)
