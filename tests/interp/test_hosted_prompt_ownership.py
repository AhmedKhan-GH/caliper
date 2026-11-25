"""CPU checks for prompt ownership across trace admission and refusal."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from caliper import _wire
from caliper.interp import _memory, _publish, capture
from caliper.interp._output import internal_forward
from .concurrency_isolation import isolated_test
from .test_controls_concurrency import Job, other_thread
from .test_hosted import hosted_run, _trace_at_boundary
from .test_hosted_controls import _queue_action
from .test_service import call


@pytest.fixture(autouse=True)
def local_resources(monkeypatch):
    monkeypatch.setattr("caliper.serve", lambda: SimpleNamespace(url="in-process"))
    initialize = _memory.MemoryGuard.__init__

    def initialize_cpu(self, limit_gb, device, **kwargs):
        assert torch.device(device).type == "cpu"
        kwargs["pressure"] = lambda: 80.
        initialize(self, limit_gb, device, **kwargs)

    monkeypatch.setattr(_memory.MemoryGuard, "__init__", initialize_cpu)


def checkpoint(run):
    return json.loads(Path(run.managed.checkpoint(run.model, None)).read_text())


def active_clamp(run, route):
    graph = _trace_at_boundary(run, prompt="AB CD", max_targets=32)
    row = next(i for i, node in enumerate(graph.nodes)
               if node["kind"] == "feature" and node["position"] == 5
               and run.index.show(node["layer"], node["feature"])["maximum"] > 0)
    run.selected.set(row)
    run.boundary()
    _queue_action(route, "apply_clamp", 10)
    run.boundary()
    record = run._owned_record()
    assert record.state == "active" and record.applied_spec["positions"] == [5]
    return record


@pytest.mark.parametrize("intervention", ["positional_clamp", "live_ablation"])
def test_refused_trace_preserves_prompt_hook_output_and_checkpoint(hosted_run, monkeypatch, intervention):
    run, route = hosted_run
    if intervention == "positional_clamp":
        record = active_clamp(run, route)
        original, refused, recovery = "AB CD", "AB", "AB CB"
        message = "position"
    else:
        _trace_at_boundary(run, prompt="ABCDEFGH", max_targets=0)
        with internal_forward(), torch.no_grad(), capture(run.model, ["norm"]):
            run.model(**run.tokenizer("ABCDEFGH"))
        run.interventions.set({"kind": "ablate", "point": "norm"})
        run.boundary()
        record = run.controls._ivs[("norm", "ablate")]
        original, refused, recovery = "ABCDEFGH", "CD", "EF"
        message = "ablation point"
    hook = record.hook
    assert hook is not None and not hook.removed
    graph = run.service.graph
    traced = (run.traced_prompt, deepcopy(run.traced_specifications), deepcopy(run.traced_state))
    selection, selected = deepcopy(run.selection), run.selected.value
    comparison = deepcopy(run.inference_status()["comparison"])
    output_key = deepcopy(run._output_key)
    published = {name: (tensor, tensor.clone(), _publish._VERSIONS[name])
                 for name, tensor in _publish._TENSORS.items() if name.startswith(("trace.", "output."))}
    before = checkpoint(run)
    comparisons = []
    compare = run.output.compare

    def compared(prompt):
        comparisons.append(prompt)
        return compare(prompt)

    monkeypatch.setattr(run.output, "compare", compared)
    assert call(route, "trace.run", {"prompt": refused, "max_targets": 0}, 20)["status"] == "queued"
    run.boundary()
    reply, = [reply for reply in route.transport.replies if reply.request_id == 20]
    assert reply.status == "error" and reply.error["code"] == "interpret.refused"
    assert message in reply.error["message"]
    assert record.state == "active" and record.hook is hook and not hook.removed
    assert not comparisons
    assert run.prompt.value == run.checkpoint_prompt == run.output.prompt == original
    assert run.service.graph is graph and run.service.status["state"] == "error"
    assert (run.traced_prompt, run.traced_specifications, run.traced_state) == traced
    assert run.selection == selection and run.selected.value == selected
    assert run._output_key == output_key
    assert run.inference_status()["comparison"] == comparison
    for name, (tensor, values, version) in published.items():
        assert _publish._resolve(name) is tensor and _publish._VERSIONS[name] == version
        assert torch.equal(tensor, values)
    after = checkpoint(run)
    for key in ("bindings", "params", "interventions", "selection", "owned_clamp", "generation_id"):
        assert after[key] == before[key]
    run.boundary()
    assert record.state == "active" and record.hook is hook and not hook.removed
    assert run._comparison == comparison and not comparisons
    if intervention == "positional_clamp":
        assert run._owned_record() is record and after["owned_clamp"]["positions"] == [5]
    else:
        run.interventions.clear()
        run.boundary()
    graph = _trace_at_boundary(run, prompt=recovery, max_targets=0)
    assert run._selection_graph is graph
    assert run.prompt.value == run.traced_prompt == run.checkpoint_prompt == run.output.prompt == recovery
    assert run.service.status["state"] == "done" and run.service.failure is None
    assert run.selected.value == -1 and run.selection is None
    assert run.inference_status()["comparison"]["prompt_hash"] == run.inference_status()["prompt_hash"]
    assert checkpoint(run)["params"]["prompt"] == recovery
    if intervention == "positional_clamp":
        assert run._owned_record() is record and record.hook is hook and not hook.removed


@pytest.mark.parametrize("refused", [False, True])
@pytest.mark.parametrize("edit", ["script", "equal_script", "viewer"])
@isolated_test("hosted_run", "monkeypatch")
def test_trace_preserves_concurrent_prompt_and_external_clamp(hosted_run, monkeypatch, refused, edit):
    run, route = hosted_run
    record = active_clamp(run, route)
    hook, graph = record.hook, run.service.graph
    replacement = dict(record.spec, value=record.spec["value"] + 1.)
    trace = run.service.tracer.trace
    entered, release = Event(), Event()

    def delayed(**params):
        entered.set()
        assert release.wait(5), "trace admission exceeded its finite wait"
        return trace(**params)

    monkeypatch.setattr(run.service.tracer, "trace", delayed)
    requested = "AB" if refused else "AB CB"
    edited = "AB CD" if edit == "equal_script" else "AB CC"
    assert call(route, "trace.run", {"prompt": requested, "max_targets": 0}, 20)["status"] == "queued"
    job = Job(run.boundary)
    try:
        assert entered.wait(5), "trace admission did not reach its barrier"

        def change():
            if edit == "viewer":
                assert route.on_control(None, _wire.Control(21, "set", name="prompt", value=edited).to_meta())["status"] == "ok"
            else:
                run.prompt.set(edited)
            run.interventions.set(replacement)

        other_thread(change)
        assert not job.done.is_set()
    finally:
        release.set()
        job.join()
    job.success()
    reply, = [reply for reply in route.transport.replies if reply.request_id == 20]
    assert reply.status == ("error" if refused else "ok")
    assert run.prompt.value == ("AB CD" if edit == "viewer" else edited)
    assert run.prompt.pending == (edited if edit == "viewer" else None)
    assert run.checkpoint_prompt == "AB CD"
    assert run.traced_prompt == ("AB CD" if refused else requested)
    assert (run.service.graph is graph) == refused
    assert record.replace and record.spec == replacement
    assert record.hook is hook and not hook.removed
    assert run._owned_record() is None
    assert run.inference_status()["comparison"]["prompt_hash"] == run.inference_status()["prompt_hash"]
    run.boundary()
    assert run.prompt.value == run.checkpoint_prompt == run.output.prompt == edited
    assert run.prompt.pending is None and record.applied_spec == replacement
    assert run._owned_record() is None
    assert checkpoint(run)["params"]["prompt"] == edited


def test_automatic_refusal_keeps_accepted_editor_prompt_and_recovers(hosted_run, monkeypatch):
    run, route = hosted_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=0)
    with internal_forward(), torch.no_grad(), capture(run.model, ["norm"]):
        run.model(**run.tokenizer("AB"))
    run.interventions.set({"kind": "ablate", "point": "norm"})
    run.boundary()
    record = run.controls._ivs[("norm", "ablate")]
    hook = record.hook
    trace = run.service.tracer.trace
    attempts = []

    def traced(**params):
        attempts.append(params["prompt"])
        return trace(**params)

    monkeypatch.setattr(run.service.tracer, "trace", traced)
    run.retrace.set(True)
    run.prompt.set("CD")
    run.boundary()
    assert attempts == ["CD"] and run.service.status["state"] == "error"
    assert run.service.graph is graph and run.traced_prompt == "AB"
    assert run.prompt.value == run.checkpoint_prompt == run.output.prompt == "CD"
    assert record.state == "active" and record.hook is hook and not hook.removed
    assert checkpoint(run)["params"]["prompt"] == "CD"
    comparison, output = deepcopy(run._comparison), _publish._resolve("output.probs")
    run.boundary()
    assert attempts == ["CD"] and run._comparison == comparison
    assert _publish._resolve("output.probs") is output and not hook.removed
    run.interventions.clear()
    run.boundary()
    assert attempts == ["CD", "CD"] and run.service.status["state"] == "done"
    assert run.traced_prompt == run.checkpoint_prompt == "CD"
    assert run.service.graph is not graph and run.service.failure is None
