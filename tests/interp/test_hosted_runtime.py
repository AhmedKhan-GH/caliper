"""Guarded CPU checks for hosted ownership and structured completion evidence."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from caliper.interp._hosted import CHECKPOINT_FORMAT
from .test_hosted import hosted_run, _trace_at_boundary


def own_clamp(run):
    graph = _trace_at_boundary(run, prompt="AB", max_targets=32)
    run.selected.set(next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"))
    run.clamp.set(0.)
    run.boundary()
    return deepcopy(run._owned_clamp)


@pytest.mark.parametrize("same_spec", [False, True])
@pytest.mark.parametrize("apply", [False, True])
def test_reset_preserves_external_replacement_at_owned_point(hosted_run, same_spec, apply):
    run, _ = hosted_run
    spec = own_clamp(run)
    replacement = dict(spec, value=spec["value"] if same_spec else 1.)
    run.interventions.set(replacement)
    if apply:
        run._apply()
    assert run.state_dict()["owned_clamp"] is None
    run.reset()
    assert run.interventions.specifications() == [replacement]


def test_reset_removes_only_owned_clamp_after_selection_loss(hosted_run):
    run, _ = hosted_run
    spec = own_clamp(run)
    run.interventions.set({"kind": "add", "point": spec["point"], "vector": [1.] * 64, "coefficient": .1})
    other_point = next(p for p in run.dictionary.metadata["write_points"] if p != spec["point"])
    run.interventions.set({"kind": "clamp", "point": other_point, "feature": 1, "value": .2})
    run._apply()
    expected = [s for s in run.interventions.specifications() if s != spec]
    run.selection = None
    run.selected.set(-1)
    run.reset()
    assert run.interventions.specifications() == expected


def test_new_slider_target_releases_prior_owned_clamp(hosted_run):
    run, _ = hosted_run
    spec = own_clamp(run)
    graph = run.service.graph
    row = next(i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"
               and run.dictionary.metadata["write_points"][node["layer"]] != spec["point"])
    run.selected.set(row)
    run.clamp.set(1.)
    run.boundary()
    current, = run.interventions.specifications()
    assert current["point"] != spec["point"]
    run.reset()
    assert not run.interventions.specifications()


def test_clamp_checkpoint_rebinds_before_fork_override(hosted_run):
    run, _ = hosted_run
    spec = own_clamp(run)
    record = json.loads(run.managed._state(run.model, None, run.controls.n))
    assert record["format"] == CHECKPOINT_FORMAT and record["owned_clamp"] == spec
    run.load_checkpoint(record)
    run.interventions.load_state_dict(record["interventions"])
    run._restore_owned_clamp(run.restored["owned_clamp"])
    assert run._owned_clamp is not spec and run._owned_record() is not None
    run.reset()
    assert not run.interventions.specifications()
    run.interventions.load_state_dict(record["interventions"])
    run._restore_owned_clamp(record["owned_clamp"])
    run.interventions.apply_overrides([spec])
    run.reset()
    assert run.interventions.specifications() == [spec]
    assert run.state_dict()["owned_clamp"] is None


def test_old_checkpoint_has_no_inferred_clamp_ownership(hosted_run):
    run, _ = hosted_run
    spec = own_clamp(run)
    record = json.loads(run.managed._state(run.model, None, run.controls.n))
    del record["owned_clamp"]
    run.load_checkpoint(record)
    run.interventions.load_state_dict(record["interventions"])
    run._restore_owned_clamp(run.restored.get("owned_clamp"))
    run.reset()
    assert run.interventions.specifications() == [spec]
    record["owned_clamp"] = dict(spec, value=100.)
    with pytest.raises(ValueError, match="ownership"):
        run.load_checkpoint(record)


@pytest.mark.parametrize("origin", ["tracer", "index", "action", "guard"])
def test_hosted_owner_checkpoints_once_and_exits_on_independent_stop(hosted_run, monkeypatch, origin):
    from caliper import _wire
    from caliper.interp._memory import CleanStop
    from caliper.interp._frozen import MemoryGuard as TraceGuard
    from types import SimpleNamespace
    run, route = hosted_run
    checkpoints, calls, replies = [], [], []
    checkpoint = run.managed.checkpoint
    failure = CleanStop("independent resource stop")

    def stop(*args, **kwargs):
        calls.append(origin)
        raise failure

    def save(*args):
        checkpoints.append(checkpoint(*args))

    monkeypatch.setattr(run.managed, "checkpoint", save)
    monkeypatch.setattr(run.managed, "_keep", lambda *args: pytest.fail("normal keep followed a stop"))
    if origin == "tracer":
        # Exercise the real admission formula without an MPS tensor or API call.
        guard = SimpleNamespace(device=SimpleNamespace(type="mps"), limit=12_000_000_000,
                                check=lambda stage: 11_800_000_000)
        def admit(**kwargs):
            calls.append(origin)
            TraceGuard.batch_size(guard, 64, layers=24, tokens=32, width=1024, mixer_width=2048)
        monkeypatch.setattr(run.service._resource("tracer"), "trace", admit)
        method, params = "trace.run", {"prompt": "AB"}
    elif origin == "index":
        monkeypatch.setattr(run.index, "examples", stop)
        method, params = "feature.examples", {"layer": 0, "feature": 0}
    elif origin == "action":
        monkeypatch.setattr(run.output, "generate", stop)
        route.on_control(None, _wire.Control(1, "run", name="generate").to_meta())
        method = None
    else:
        from caliper.interp._hosted import _RunGuard
        run.guard = _RunGuard(SimpleNamespace(check=stop))
        run.output.guard = run.guard
        run.prompt.set("AC")
        method = None
    if method is not None:
        run.service.submit({"version": 1, "method": method, "params": params}, lambda *a: replies.append(a))
        run.service.submit({"version": 1, "method": "feature.dictionary", "params": {}}, lambda *a: replies.append(a))
    later = run.controls.action("later", lambda: pytest.fail("later work ran"))
    route.on_control(None, _wire.Control(2, "run", name="later").to_meta())
    try:
        with pytest.raises(CleanStop) as caught:
            run.run()
        assert calls == [origin] and len(checkpoints) == 1
        assert run.service.failure is caught.value and run._failure is caught.value
        if origin != "tracer":
            assert caught.value is failure
        assert json.loads(Path(checkpoints[0]).read_text())["bindings"] == run._verified_bindings
        with pytest.raises(CleanStop) as repeated:
            run.boundary()
        assert repeated.value is caught.value and len(checkpoints) == 1
        assert not later.queued and not run.controls.queue
        assert all(reply[0] == "error" for reply in replies)
    finally:
        later.remove()


def test_failed_terminal_checkpoint_preserves_original_stop_without_retry(hosted_run, monkeypatch):
    from caliper.interp._memory import CleanStop
    run, _ = hosted_run
    failure = CleanStop("admission stop")
    calls = []
    run.service.stop(failure, run.controls.n)
    def fail(*args):
        calls.append(1)
        raise OSError("checkpoint unavailable")
    monkeypatch.setattr(run.managed, "checkpoint", fail)
    with pytest.raises(CleanStop) as caught:
        run.boundary()
    assert caught.value is failure and isinstance(caught.value.__cause__, OSError)
    with pytest.raises(CleanStop):
        run.boundary()
    assert calls == [1]


def test_latched_owner_stop_ends_queued_actions_before_boundary(hosted_run, monkeypatch):
    from caliper import _wire
    from caliper.interp._hosted import _RunGuard
    from caliper.interp._memory import CleanStop
    run, route = hosted_run
    failure = CleanStop("prior owner stop")
    run.guard = _RunGuard(None)
    run.guard.failure = failure
    run.output.guard = run.guard
    calls = []
    monkeypatch.setattr(run.managed, "checkpoint", lambda *args: calls.append("checkpoint"))
    action = run.controls.action("later", lambda: pytest.fail("later action ran"))
    route.on_control(None, _wire.Control(19, "run", name="later").to_meta())
    try:
        with pytest.raises(CleanStop) as caught:
            run.boundary()
        assert caught.value is failure and calls == ["checkpoint"]
        assert [(r.request_id, r.status) for r in route.transport.replies] == [(19, "error")]
        assert not action.queued
    finally:
        action.remove()


def inference_reply(route):
    from caliper import _wire
    reply = route.on_control(None, _wire.Control(100, "interpret", interpretation={
        "version": 1, "method": "trace.status", "params": {}}).to_meta())
    frame = _wire.decode(_wire.encode_frame(_wire.MsgType.CONTROL_REPLY, reply))
    assert frame.status == "ok"
    return frame.result["inference"]


def test_generation_evidence_is_fresh_exact_and_follows_both_text_rows(hosted_run, monkeypatch):
    import hashlib
    from caliper import _wire
    from caliper.interp._output import bounded_text
    run, route = hosted_run
    own_clamp(run)
    before = inference_reply(route)
    assert before["generation"] == {"id": 0, "prompt_hash": None, "interventions_hash": None,
                                    "token_ids": [[], []], "counts": [0, 0], "completed_step": None,
                                    "text_hashes": [None, None]}
    observations = []
    publish = run.controls._publish
    def observe(*args, **kwargs):
        observations.append((run._generation["id"], run.baseline_text.value, run.steered_text.value))
        return publish(*args, **kwargs)
    monkeypatch.setattr(run.controls, "_publish", observe)
    for action_id in (1, 2):
        step = run.controls.n + 1
        assert route.on_control(None, _wire.Control(action_id, "run", name="generate").to_meta())["status"] == "queued"
        run.boundary()
        result = inference_reply(route)
        generation = result["generation"]
        assert generation["id"] == action_id
        assert generation["counts"] == [24, 24] and [len(row) for row in generation["token_ids"]] == [24, 24]
        assert generation["completed_step"] == step
        assert generation["prompt_hash"] == result["prompt_hash"] == hashlib.sha256(b"AB").hexdigest()
        assert generation["interventions_hash"] == result["interventions_hash"]
        texts = [run.baseline_text.value, run.steered_text.value]
        assert generation["text_hashes"] == [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        assert texts == [bounded_text(run.tokenizer.decode(row)) for row in generation["token_ids"]]
        assert (action_id - 1, *texts) in observations
        assert (action_id, *texts) in observations
    assert before["generation"]["id"] == 0
    comparison = result["comparison"]
    assert comparison["prompt_hash"] == result["prompt_hash"]
    assert comparison["interventions_hash"] == result["interventions_hash"]
    from caliper.interp._publish import _resolve
    assert set(comparison["argmax_token_ids"]) <= set(_resolve("output.tokens").tolist())


@pytest.mark.parametrize("fault", ["short", "boolean", "vocabulary", "text", "prompt", "interventions", "publish"])
def test_failed_generation_cannot_advance_completion(hosted_run, monkeypatch, fault):
    run, route = hosted_run
    run.generate()
    before = inference_reply(route)["generation"]
    result = {"token_ids": [[65] * 24, [66] * 24], "baseline": "A" * 24, "steered": "B" * 24}
    if fault == "short":
        result["token_ids"][0].pop()
    elif fault == "boolean":
        result["token_ids"][0][0] = True
    elif fault == "vocabulary":
        result["token_ids"][0][0] = run.model.config.vocab_size
    elif fault == "text":
        result["baseline"] = "old text"
    def generate(prompt):
        if fault == "prompt":
            run.prompt.set("AC")
        elif fault == "interventions":
            run.interventions.set({"kind": "ablate", "point": "layers.0.mlp_out"})
            run._apply()
        return result
    monkeypatch.setattr(run.output, "generate", generate)
    if fault == "publish":
        publish = run.controls._script_set
        def fail(param, value):
            if param is run.steered_text:
                raise ValueError("publication failed")
            return publish(param, value)
        monkeypatch.setattr(run.controls, "_script_set", fail)
    with pytest.raises(ValueError):
        run.generate()
    assert inference_reply(route)["generation"] == before


def test_completion_hashes_detect_later_prompt_and_intervention_changes(hosted_run):
    run, route = hosted_run
    run.generate()
    original = inference_reply(route)
    run.prompt.set("AC")
    changed = inference_reply(route)
    assert changed["prompt_hash"] != changed["generation"]["prompt_hash"]
    run.interventions.set({"kind": "ablate", "point": "layers.0.mlp_out"})
    run._apply()
    changed = inference_reply(route)
    assert changed["interventions_hash"] != changed["generation"]["interventions_hash"]
    assert changed["generation"] == original["generation"]


def test_generation_checkpoint_retains_counter_and_clears_completion(hosted_run):
    run, route = hosted_run
    run.generate()
    record = json.loads(run.managed._state(run.model, None, run.controls.n))
    assert record["generation_id"] == 1
    run.load_checkpoint(record)
    run._restore_generation(run.restored.get("generation_id", 0))
    restored = inference_reply(route)["generation"]
    assert restored["id"] == 1 and restored["completed_step"] is None
    assert restored["counts"] == [0, 0] and restored["text_hashes"] == [None, None]
    run.generate()
    assert inference_reply(route)["generation"]["id"] == 2
    del record["generation_id"]
    run.load_checkpoint(record)
    assert run.restored.get("generation_id", 0) == 0
    record["generation_id"] = True
    with pytest.raises(ValueError, match="generation identity"):
        run.load_checkpoint(record)


@pytest.mark.parametrize("hosted_run", [True], indirect=True)
def test_caught_action_stop_still_terminates_replies_and_later_actions(hosted_run, monkeypatch):
    from caliper import _wire
    from caliper.interp._memory import CleanStop
    run, route = hosted_run
    failure = CleanStop("stop\0\ud800" + "😀" * 400)
    checkpoints = []
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(run.managed, "checkpoint", lambda *args: checkpoints.append(1))
    def caught():
        monkeypatch.setattr(run.guard.guard, "check", fail)
        try:
            run.guard.check("caught action")
        except CleanStop:
            pass
        return "caught"
    action = run.controls.action("caught", caught)
    later = run.controls.action("later", lambda: pytest.fail("later action ran"))
    for rid, name in ((61, "caught"), (62, "later")):
        route.on_control(None, _wire.Control(rid, "run", name=name).to_meta())
    try:
        with pytest.raises(CleanStop) as result:
            run.run()
        assert result.value is failure and checkpoints == [1]
        replies = {r.request_id: r for r in route.transport.replies}
        assert set(replies) == {61, 62}
        assert all(r.status == "error" and "\0" not in r.error["message"]
                   and len(r.error["message"].encode()) <= 1024 for r in replies.values())
    finally:
        action.remove()
        later.remove()


def test_periodic_checkpoint_cannot_swallow_a_separate_identity_stop(hosted_run, monkeypatch):
    from caliper.interp import _dictionary
    from caliper.interp._memory import CleanStop
    run, _ = hosted_run
    failure = CleanStop("identity admission stop")
    checks, checkpoints = [], []
    checkpoint = run.managed.checkpoint
    def identity(model, **kwargs):
        checks.append(1)
        raise failure
    def save(*args):
        checkpoints.append(checkpoint(*args))
    monkeypatch.setattr(_dictionary, "model_identity", identity)
    monkeypatch.setattr(run.managed, "checkpoint", save)
    monkeypatch.setattr(run.managed.kept, "keep", lambda step, callback: callback())
    with pytest.warns(RuntimeWarning, match="keeps no more checkpoints"):
        with pytest.raises(CleanStop) as caught:
            run.boundary()
    assert caught.value is failure and checks == [1] and len(checkpoints) == 1
    assert json.loads(Path(checkpoints[0]).read_text())["bindings"] == run._verified_bindings
