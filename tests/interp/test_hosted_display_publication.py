"""CPU publication checks without model forwards or a device runtime."""
from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
import threading

import pytest
import torch

import caliper
from caliper import _controls, _programs
from caliper.interp import _hosted, _memory, _publish, publish
from caliper.interp._output import is_internal_forward, prompt_inputs
from caliper.interp._service import Refused, register_service


@pytest.fixture
def publication_run(monkeypatch, tmp_path):
    monkeypatch.delenv("CALIPER_FIXTURE_DISPLAY_MPS", raising=False)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    monkeypatch.setattr(_publish, "_VERSIONS", {})
    monkeypatch.setattr(_publish, "_VERSION", 0)
    monkeypatch.setattr(_programs, "_STATE", {**_programs._STATE, "model": None})
    monkeypatch.setattr(_programs, "ACTIVE", True)
    @wraps(torch.backends.mps.is_available)
    def device_query():
        pytest.fail("unexpected device query")
    monkeypatch.setattr(torch.backends.mps, "is_available", device_query)

    # A device anchor supports prompt validation. No model forward is permitted.
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))
    model.config = SimpleNamespace(vocab_size=512)
    model.forward = lambda *args, **kwargs: pytest.fail("unexpected model forward")
    index = SimpleNamespace(examples=lambda *args, **kwargs: [], close=lambda: None)
    run = _hosted.HostedInference("tiny-lm", tmp_path, "AB", resources=(
        model, _hosted.TinyTokenizer(), object(), index, None))
    controls = _controls.Controls()
    monkeypatch.setattr(caliper, "_CTL", controls)
    run.controls = controls
    for attribute, name, value in (
        ("prompt", "prompt", "AB"), ("selected", "selected_node", -1),
        ("clamp", "clamp", 0.), ("clamp_status", "clamp.status", ""),
        ("retrace", "retrace", False), ("status", "output.status", "ready"),
    ):
        parameter = controls.param(name, value)
        setattr(run, attribute, parameter)
        run.parameters.append(parameter)
    run.actions = [controls.action("apply_clamp", lambda: None)]
    run._clamp_version, run._selected_version = run.clamp._version, run.selected._version
    run._inference_ready = True
    run.interventions = SimpleNamespace(
        executor=SimpleNamespace(_dictionary_status=lambda: None, preparing_source=False,
                                 dictionary=run.dictionary, tokenizer=run.tokenizer),
        specifications=lambda: [], suspend=nullcontext, _ctl=controls, model=model, points=[],
        _operation_owner=None, _readers=0, _retired=None)
    controls._iv_model = run.interventions
    run.service = register_service(_hosted.HostedService(run), controls=controls)
    events, checkpoints, consumed, replies, sources = [], [], [], [], {}
    state = SimpleNamespace(change_prompt=False, stop_stage=None)
    failure = _memory.CleanStop("fixture publication resource stop")

    def check(stage, **kwargs):
        if stage == state.stop_stage:
            raise failure

    run.guard = _hosted._RunGuard(SimpleNamespace(check=check), on_stop=run._latch_failure)

    def compare(prompt):
        events.append("compare")
        run.guard.check("comparison")
        prompt_inputs(model, run.tokenizer, prompt)
        publish("output.probs", torch.full((2, 10), .1))
        publish("output.tokens", torch.arange(10, dtype=torch.int32))
        sources.update({name: (_publish._resolve(name), _publish._VERSIONS[name])
                        for name in ("output.probs", "output.tokens")})
        if state.change_prompt:
            run.prompt.set("AD")
        return {"argmax_token_ids": [0, 0], "baseline_top_text": "\u0020Austin", "baseline_top_probability": .1}

    run.output = SimpleNamespace(compare=compare)

    class DisplayDouble(_hosted._FixtureDisplay):
        def publish(self):
            events.append("display")
            if run.service.boundary_running:
                assert is_internal_forward()
                assert run.service.boundary_thread == threading.get_ident()
                assert run.service.step(controls.n) == 0
                with pytest.raises(Refused, match="hosted boundary"):
                    run.service.close()
            else:
                assert controls.actions["apply_clamp"].running
            return super().publish()

    run.fixture_display = DisplayDouble(run.guard, device="cpu")

    def consume(step):
        events.append("producer")
        consumed.append((step, dict(_publish._VERSIONS)))
        for name, (tensor, version) in sources.items():
            actual = _publish._resolve(name)
            assert actual.device.type == "cpu"
            if run.fixture_display is not None:
                assert actual.data_ptr() != tensor.data_ptr()
                assert _publish._VERSIONS[name] > version
            else:
                assert actual.data_ptr() == tensor.data_ptr()
                assert _publish._VERSIONS[name] == version
            torch.testing.assert_close(actual, tensor, rtol=0, atol=0)

    monkeypatch.setattr(_programs, "at_step", consume)
    run.managed = SimpleNamespace(
        inference=run, stopping=False, checkpoint=lambda *args: checkpoints.append(controls.n),
        _keep=lambda *args: events.append("keep"))

    def enqueue():
        def reply(*args):
            events.append("request")
            replies.append(args)
            sources.update({name: (_publish._resolve(name), version)
                            for name, version in _publish._VERSIONS.items() if name.startswith("examples.")})
        assert run.service.submit({"version": 1, "method": "feature.examples",
                                   "params": {"layer": 0, "feature": 0}}, reply) == ("queued", None)

    run.boundary()
    events.clear()
    consumed.clear()
    sources.clear()
    try:
        yield SimpleNamespace(run=run, events=events, checkpoints=checkpoints, consumed=consumed,
                              replies=replies, sources=sources, state=state, failure=failure, enqueue=enqueue)
    finally:
        run.close()


def assert_boundary_released(run):
    assert not run.service.running and run.service.running_thread is None
    assert not run.service.boundary_running and run.service.boundary_thread is None
    assert not is_internal_forward()


@pytest.mark.parametrize("outcome", ["success", "invalid_prompt", "changed_input", "unchanged"])
def test_completed_request_reaches_display_before_producer(publication_run, outcome):
    case = publication_run
    run = case.run
    previous = deepcopy(run.inference_status()["comparison"])
    previous_key = deepcopy(run._output_key)
    if outcome == "invalid_prompt":
        run.prompt.set("A" * 33)
    elif outcome != "unchanged":
        run.prompt.set("AC")
    case.state.change_prompt = outcome == "changed_input"
    case.enqueue()
    run.boundary()

    expected = ["request"] + ([] if outcome == "unchanged" else ["compare"])
    assert case.events == expected + ["display", "producer", "keep"]
    assert case.replies[0][0] == "ok" and case.replies[0][2] is None
    assert len(case.sources) == (7 if outcome in ("success", "changed_input") else 5)
    assert len(case.consumed) == 1
    assert_boundary_released(run)
    current = run.inference_status()
    if outcome == "success":
        assert current["comparison"]["completed_step"] == run.controls.n
        assert current["comparison"]["prompt_hash"] == current["prompt_hash"]
        assert run.status.value == "ready"
    else:
        assert current["comparison"] == previous and run._output_key == previous_key
    if outcome in ("invalid_prompt", "changed_input"):
        error = "1 to 32 integer tokens" if outcome == "invalid_prompt" else "state changed"
        assert error in run.status.value
        assert current["comparison"]["prompt_hash"] != current["prompt_hash"]
        # A later valid boundary must still compute a fresh comparison.
        case.state.change_prompt = False
        run.prompt.set("AE")
        case.events.clear()
        run.boundary()
        assert case.events == ["compare", "display", "producer", "keep"]
        assert run.inference_status()["comparison"]["completed_step"] == run.controls.n
        assert run.status.value == "ready"


@pytest.mark.parametrize("stop_stage", ["comparison", "fixture display copy start", "fixture display copy complete"])
def test_resource_stop_blocks_producer_and_releases_boundary(publication_run, stop_stage):
    case = publication_run
    run = case.run
    previous = deepcopy(run.inference_status()["comparison"])
    run.prompt.set("AC" if stop_stage == "comparison" else "A" * 33)
    case.state.stop_stage = stop_stage
    case.enqueue()
    with pytest.raises(_memory.CleanStop) as caught:
        run.boundary()
    assert caught.value is case.failure
    expected = ["request", "compare"] + ([] if stop_stage == "comparison" else ["display"])
    assert case.events == expected and not case.consumed
    assert case.replies[0][0] == "ok"
    assert run._failure is run.service.failure is run.guard.failure is case.failure
    assert run.service.status["state"] == "error"
    assert run.inference_status()["comparison"] == previous
    assert_boundary_released(run)
    for name, (tensor, version) in case.sources.items():
        assert _publish._resolve(name).data_ptr() == tensor.data_ptr()
        assert _publish._VERSIONS[name] == version
    with pytest.raises(_memory.CleanStop) as repeated:
        run.boundary()
    assert repeated.value is case.failure
    assert case.events == expected and case.checkpoints == [run.controls.n]


def test_recoverable_error_without_fixture_display_keeps_cpu_publications(publication_run):
    case = publication_run
    case.run.fixture_display = None
    case.run.prompt.set("A" * 33)
    case.enqueue()
    case.run.boundary()
    assert case.events == ["request", "compare", "producer", "keep"]
    assert len(case.consumed) == 1
    assert_boundary_released(case.run)


def test_action_comparison_error_synchronizes_before_program_boundary(publication_run):
    case = publication_run
    run = case.run
    previous = deepcopy(run.inference_status()["comparison"])
    action = run.controls.actions["apply_clamp"]

    def apply():
        case.state.change_prompt = True
        run.update_output(force=True)

    action.fn = apply
    # Queue the action through the control API to exercise its error handler.
    from caliper import _wire
    replies = []
    route = _controls.Route(run.controls, SimpleNamespace(
        may_control=lambda conn: True, reply=lambda conn, meta: replies.append(meta)))
    assert route.on_control(None, _wire.Control(1, "run", name="apply_clamp").to_meta())["status"] == "queued"
    with pytest.warns(RuntimeWarning, match="state changed"):
        run.boundary()
    assert case.events == ["display", "compare", "display", "producer", "keep"]
    assert replies[-1]["status"] == "error" and "state changed" in replies[-1]["error"]["message"]
    assert len(case.sources) == 2 and not action.running
    assert run.inference_status()["comparison"] == previous
    assert_boundary_released(run)
