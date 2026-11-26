"""Live publication recovery at the offline MCP boundary."""
from copy import deepcopy
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from test_published_dallas import callback_boundary, driver, evidence_factory, published, write_json
from _mcp import Failure


@pytest.fixture
def live_bridge(evidence_factory):
    root, prepared, make = evidence_factory
    baseline = make(10, 1, vocab_size=10)
    evidence = published.validate_case(root, baseline, 10, previous_sequence=0)
    state = {"status": baseline, "sequence": 1, "calls": [], "resource_reads": [], "program_reads": [],
             "programs": [{"id": "probabilities", "state": "running", "last_step": 1,
                           "draw": {"state": "running"}}]}

    def resources():
        inference = state["status"]["inference"]
        record = inference["replication"]["result"]
        with np.load(root / record["evidence"]["file"], allow_pickle=False) as archive:
            probabilities = archive["probabilities"]
            state["bars"] = {"step": inference["comparison"]["completed_step"],
                "elements": [[0, 0, 1, 1, float(probabilities[0, i]), float(probabilities[1, i]), i, i]
                             for i in range(10)]}
        state["programs"][0]["last_step"] = state["bars"]["step"]

    def body(method, **args):
        if method == "run.state":
            return {"attached": True}
        if method == "program.read":
            assert args["id"] == "probabilities" and args["resource"] == "cells"
            state["resource_reads"].append(args)
            reply = deepcopy(state["bars"])
            if hook := state.pop("during_read", None):
                hook()
            return reply
        if method == "program.status":
            state["program_reads"].append(deepcopy(state["programs"]))
            return {"programs": deepcopy(state["programs"])}
        assert method == "state.set"
        state["calls"].append((method, args))

    def until(name, callback):
        # Two event boundaries model pending publication without a clock delay.
        for _ in range(2):
            if result := callback():
                return result
            if hook := state.pop("on_wait", None):
                hook()
        raise Failure(f"Timed out: {name}")

    resources()
    session = SimpleNamespace(
        out=root, replication_profile=published.PROFILE, private_output_directory=root,
        published_last_sequence=1, published_preparation_id=prepared["id"],
        published_baseline_hash=evidence["baseline_array_sha256"],
        events=SimpleNamespace(pending=[], until=until, complete=lambda method: deepcopy(state["status"])),
        mcp=SimpleNamespace(body=body))
    session.read = MethodType(driver.Session.read, session)
    session.ready_programs = MethodType(driver.Session.ready_programs, session)

    def commit(cell, *, success=True):
        state["sequence"] += 1
        if success:
            state["status"] = make(cell, state["sequence"], vocab_size=10)
            resources()
        else:
            state["status"]["inference"]["replication"]["last_attempt"] = {
                "case_id": published.action_name(cell), "sequence": state["sequence"], "status": "error"}
        event = {"kind": "run", "change": "action", "name": published.action_name(cell),
                 "status": "ok" if success else "error"}
        session.events.pending.append(event)
        return event

    return driver.Bridge(session), state, commit, make


def test_live_recovers_two_successes_then_third_success(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    commit(16)
    bridge.consume(tolerate_refusals=True)
    assert bridge.session.published_last_sequence == 3
    evidence = json.loads((bridge.session.out / "live-case.json").read_text())
    assert evidence["result"]["cell"] == 16
    commit(18)
    bridge.consume(tolerate_refusals=True)
    assert bridge.session.published_last_sequence == 4
    evidence = json.loads((bridge.session.out / "live-case.json").read_text())
    assert evidence["result"]["cell"] == 18
    assert not (bridge.session.out / "bridge-errors.jsonl").exists()


@pytest.mark.parametrize("actions", [
    [(14, False), (16, True)],
    [(14, True), (16, False)],
    [(14, False), (16, False)],
    [(14, True), (16, False), (18, True)],
    [(14, False), (16, True), (18, False)],
    [(14, True), (14, True)],
    [(14, True), (14, False)],
    [(14, False), (14, True)],
])
def test_live_interleavings_recover_only_the_latest_attempt(live_bridge, actions):
    bridge, state, commit, _ = live_bridge
    for cell, success in actions:
        event = commit(cell, success=success)
        event["text"] = "This event has no attempt sequence."
    bridge.consume(tolerate_refusals=True)
    assert bridge.session.published_last_sequence == 1 + len(actions)
    assert len(state["calls"]) == 1
    cell, success = actions[-1]
    text = state["calls"][0][1]["value"]
    assert published.action_name(cell) in text
    case_file = bridge.session.out / "live-case.json"
    if success:
        assert "After:" in text
        assert json.loads(case_file.read_text())["result"]["sequence"] == 1 + len(actions)
    else:
        assert text == f"{published.action_name(cell)}: Action failed."
        assert not case_file.exists()
    assert not (bridge.session.out / "bridge-errors.jsonl").exists()
    commit(20)
    bridge.consume()
    assert bridge.session.published_last_sequence == 2 + len(actions)
    assert json.loads(case_file.read_text())["result"]["cell"] == 20


def test_live_failure_preserves_last_valid_case_without_republishing_arrays(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    bridge.consume()
    case_file = bridge.session.out / "live-case.json"
    saved = case_file.read_bytes()
    commit(16, success=False)
    bridge.consume()
    assert bridge.session.published_last_sequence == 3
    assert case_file.read_bytes() == saved
    assert state["calls"][-1][1]["value"] == f"{published.action_name(16)}: Action failed."
    commit(18)
    bridge.consume()
    assert json.loads(case_file.read_text())["result"]["sequence"] == 4


@pytest.mark.parametrize("success", [False, True])
def test_live_duplicate_events_do_not_increment_or_republish(live_bridge, success):
    bridge, state, commit, _ = live_bridge
    event = commit(14, success=success)
    bridge.session.events.pending.extend([event, deepcopy(event)])
    bridge.consume()
    assert bridge.session.published_last_sequence == 2
    assert len(state["calls"]) == 1
    bridge.session.events.pending.append(deepcopy(event))
    bridge.consume()
    assert bridge.session.published_last_sequence == 2
    assert len(state["calls"]) == 1
    commit(14)
    bridge.consume()
    assert bridge.session.published_last_sequence == 3
    assert len(state["calls"]) == 2


def test_live_duplicate_status_must_match_the_validated_snapshot(live_bridge):
    bridge, state, commit, _ = live_bridge
    event = commit(14)
    bridge.consume()
    state["status"]["inference"]["replication"]["active_cell"] = 16
    with pytest.raises(Failure, match="stale"):
        bridge.handle(deepcopy(event))
    assert bridge.session.published_last_sequence == 2
    assert len(state["calls"]) == 1


def test_live_events_cannot_invent_sequence_increments(live_bridge):
    bridge, state, commit, _ = live_bridge
    stale = deepcopy(state["status"])
    commit(14)
    commit(16)
    current = state["status"]
    state["status"] = stale
    bridge.consume(tolerate_refusals=True)
    assert bridge.session.published_last_sequence == 1
    assert not (bridge.session.out / "live-case.json").exists()
    assert all("Case evidence error:" in args["value"] for _, args in state["calls"])
    state["status"] = current
    commit(18)
    bridge.consume()
    assert bridge.session.published_last_sequence == 4
    assert json.loads((bridge.session.out / "live-case.json").read_text())["result"]["cell"] == 18


def test_live_status_regression_cannot_replace_newer_evidence(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    older = deepcopy(state["status"])
    commit(16)
    bridge.consume()
    case_file = bridge.session.out / "live-case.json"
    saved = case_file.read_bytes()
    event = commit(18)
    state["status"] = older
    with pytest.raises(Failure, match="stale"):
        bridge.handle(event)
    assert bridge.session.published_last_sequence == 3
    assert case_file.read_bytes() == saved
    commit(20)
    bridge.consume()
    assert bridge.session.published_last_sequence == 5
    assert json.loads(case_file.read_text())["result"]["cell"] == 20


@pytest.mark.parametrize("fault", [
    "missing_result", "old_result", "active", "publication", "publication_hash", "prompt_hash",
    "baseline", "missing_baseline", "missing_preparation", "preparation_file", "metadata", "archive", "native",
])
def test_live_recovery_refuses_incomplete_or_stale_evidence(live_bridge, fault):
    bridge, state, commit, _ = live_bridge
    session = bridge.session
    commit(14)
    old_result = deepcopy(state["status"]["inference"]["replication"]["result"])
    event = commit(14)
    inference = state["status"]["inference"]
    replication = inference["replication"]
    record = replication["result"]
    if fault == "missing_result": replication["result"] = None
    if fault == "old_result": replication["result"] = old_result
    if fault == "active": replication["active_cell"] = 16
    if fault == "publication": inference["comparison"] = None
    if fault == "publication_hash": inference["comparison"]["interventions_hash"] = "d" * 64
    if fault == "prompt_hash": inference["prompt_hash"] = "d" * 64
    if fault == "baseline": session.published_baseline_hash = "d" * 64
    if fault == "missing_baseline": del session.published_baseline_hash
    if fault == "missing_preparation": del session.published_preparation_id
    if fault == "preparation_file": (session.out / replication["preparation_file"]).unlink()
    if fault == "metadata": write_json(session.out / record["metadata_file"], {})
    if fault == "archive": (session.out / record["evidence"]["file"]).write_bytes(b"stale")
    if fault == "native": (session.out / record["native_evidence"]["file"]).write_bytes(b"stale")
    with pytest.raises(Failure):
        bridge.handle(event)
    assert session.published_last_sequence == 1
    assert not state["calls"]
    assert not (session.out / "live-case.json").exists()


@pytest.mark.parametrize("attempt", [None, {},
    {"case_id": [], "sequence": 3, "status": "ok"},
    {"case_id": published.action_name(14), "sequence": True, "status": "ok"},
    {"case_id": published.action_name(14), "sequence": "3", "status": "error"},
    {"case_id": published.action_name(14), "sequence": 0, "status": "error"},
    {"case_id": published.action_name(14), "sequence": 3, "status": "pending"},
])
def test_live_recovery_requires_a_committed_attempt(live_bridge, attempt):
    bridge, state, commit, _ = live_bridge
    event = commit(14)
    state["status"]["inference"]["replication"]["last_attempt"] = attempt
    with pytest.raises(Failure, match="committed attempt"):
        bridge.handle(event)
    assert bridge.session.published_last_sequence == 1
    assert not state["calls"]


@pytest.mark.parametrize("success", [False, True])
def test_live_preparation_replacement_requires_explicit_new_baseline(live_bridge, success):
    bridge, state, commit, make = live_bridge
    original_file = bridge.session.out / "preparation.json"
    original = json.loads(original_file.read_text())
    replacement = deepcopy(original)
    replacement["model_content_hash"] = "sha256:" + "d" * 64
    replacement["id"] = published.digest({k: v for k, v in replacement.items() if k not in ("id", "seconds")})
    event = commit(14, success=success)
    state["status"]["inference"]["replication"]["preparation_id"] = replacement["id"]
    write_json(original_file, replacement)
    with pytest.raises(Failure, match="preparation identity"):
        bridge.handle(event)
    assert bridge.session.published_last_sequence == 1
    assert not state["calls"]
    write_json(original_file, original)
    state["status"] = make(10, 1)
    commit(16)
    bridge.consume()
    assert bridge.session.published_last_sequence == 3


@pytest.mark.parametrize("success", [False, True])
def test_live_commit_during_evidence_read_recovers_on_next_event(live_bridge, success):
    bridge, state, commit, _ = live_bridge
    first = commit(14, success=success)
    prior = deepcopy(state["status"])
    bars = deepcopy(state["bars"])
    second = commit(16)
    latest_bars = state["bars"]
    state["bars"] = bars
    state["programs"][0]["last_step"] = bars["step"]
    replies = iter([prior, deepcopy(state["status"])])
    bridge.session.events.complete = lambda method: next(replies)
    with pytest.raises(Failure, match="changed during case publication"):
        bridge.handle(first)
    assert bridge.session.published_last_sequence == 1
    assert not state["calls"]
    assert not (bridge.session.out / "live-case.json").exists()
    bridge.session.events.complete = lambda method: deepcopy(state["status"])
    state["bars"] = latest_bars
    state["programs"][0]["last_step"] = latest_bars["step"]
    bridge.handle(second)
    assert bridge.session.published_last_sequence == 3
    assert json.loads((bridge.session.out / "live-case.json").read_text())["result"]["cell"] == 16


def test_live_readout_failure_does_not_advance_the_sequence(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    event = commit(16)
    body = bridge.session.mcp.body
    def refuse(*args, **kwargs):
        raise Failure("The readout publication failed.")
    bridge.session.mcp.body = refuse
    with pytest.raises(Failure, match="readout publication"):
        bridge.handle(event)
    assert bridge.session.published_last_sequence == 1
    assert not (bridge.session.out / "live-case.json").exists()
    bridge.session.mcp.body = body
    bridge.handle(event)
    assert bridge.session.published_last_sequence == 3


def test_live_same_boundary_can_publish_distinct_cases_at_one_output_step(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    state["status"]["inference"]["comparison"]["completed_step"] = 2
    bridge.consume()
    commit(16)
    state["status"]["inference"]["comparison"]["completed_step"] = 2
    state["bars"]["step"] = state["programs"][0]["last_step"] = 2
    bridge.consume()
    assert bridge.session.published_last_sequence == 3
    assert json.loads((bridge.session.out / "live-case.json").read_text())["result"]["cell"] == 16


def test_live_success_reads_current_bars_and_program_status(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    bridge.consume()
    assert state["resource_reads"] and state["program_reads"]
    assert bridge.session.published_last_sequence == 2
    assert "After:" in state["calls"][-1][1]["value"]


@pytest.mark.parametrize("sequence", [1, 2])
def test_live_case_requires_a_new_result(live_bridge, sequence):
    bridge, state, commit, _ = live_bridge
    event = commit(14)
    if sequence == 1:
        bridge.session.published_last_sequence = 2
        with pytest.raises(Failure, match="stale"):
            bridge.handle(event)
        assert not state["calls"]
    else:
        bridge.handle(event)
        assert bridge.session.published_last_sequence == 2 and "Before:" in state["calls"][-1][1]["value"]


@pytest.mark.parametrize("publication_delay", [0, 4])
@pytest.mark.parametrize("fault", ["stopped", "draw_stopped", "refused", "draw_refused", "error", "draw_error",
                                  "stale", "resource_ahead_of_program", "values", "program_behind_resource", "missing"])
def test_live_resource_refusal_preserves_confirmed_case_and_recovers(live_bridge, fault, publication_delay):
    bridge, state, commit, _ = live_bridge
    commit(14)
    bridge.consume()
    case_file = bridge.session.out / "live-case.json"
    saved = case_file.read_bytes()
    confirmed = deepcopy(bridge.session.published_live_state)
    commit(16)
    completed_step = state["status"]["inference"]["comparison"]["completed_step"]
    state["bars"]["step"] += publication_delay
    state["programs"][0]["last_step"] = state["bars"]["step"]
    if fault == "stopped": state["programs"][0]["state"] = "stopped"
    if fault == "draw_stopped": state["programs"][0]["draw"]["state"] = "stopped"
    if fault == "refused": state["programs"][0]["state"] = "refused"
    if fault == "draw_refused": state["programs"][0]["draw"]["state"] = "refused"
    if fault == "error": state["programs"][0]["error"] = {"code": "fixture.error"}
    if fault == "draw_error": state["programs"][0]["draw"]["error"] = {"code": "fixture.error"}
    if fault == "stale": state["bars"]["step"] = completed_step - 1
    if fault == "resource_ahead_of_program": state["bars"]["step"] += 1
    if fault == "values": state["bars"]["elements"][0][5] += .125
    if fault == "program_behind_resource": state["programs"][0]["last_step"] -= 1
    original_programs = state["programs"]
    if fault == "missing": state["programs"] = []
    bridge.consume(tolerate_refusals=True)
    assert bridge.session.published_last_sequence == 2
    assert bridge.session.published_live_state == confirmed
    assert case_file.read_bytes() == saved
    assert state["calls"][-1][1]["value"].startswith("Case evidence error:")
    assert "After:" not in state["calls"][-1][1]["value"]
    if fault in ("resource_ahead_of_program", "program_behind_resource"):
        assert "current probabilities program status" in state["calls"][-1][1]["value"]
    if fault == "values":
        assert "private case arrays" in state["calls"][-1][1]["value"]
    state["programs"] = original_programs
    state["programs"][0].update(state="running", error=None, draw={"state": "running"})
    commit(18)
    bridge.consume()
    assert bridge.session.published_last_sequence == 4
    assert json.loads(case_file.read_text())["result"]["cell"] == 18


@pytest.mark.parametrize("publication_delay", [0, 4])
@pytest.mark.parametrize("same_step", [False, True])
def test_live_commit_during_resource_read_recovers_on_next_event(live_bridge, same_step, publication_delay):
    bridge, state, commit, _ = live_bridge
    commit(14)
    bridge.consume()
    case_file = bridge.session.out / "live-case.json"
    saved = case_file.read_bytes()
    confirmed = deepcopy(bridge.session.published_live_state)
    commit(16)
    state["bars"]["step"] += publication_delay
    state["programs"][0]["last_step"] = state["bars"]["step"]
    def later():
        step = (state["status"]["inference"]["comparison"]["completed_step"] if same_step
                else state["bars"]["step"] + 1)
        commit(18)
        state["status"]["inference"]["comparison"]["completed_step"] = step
        state["bars"]["step"] = state["programs"][0]["last_step"] = step + publication_delay
    state["during_read"] = later
    bridge.consume(tolerate_refusals=True)
    assert bridge.session.published_last_sequence == 2
    assert bridge.session.published_live_state == confirmed
    assert case_file.read_bytes() == saved
    assert "changed during case publication" in state["calls"][-1][1]["value"]
    bridge.consume()
    assert bridge.session.published_last_sequence == 4
    assert json.loads(case_file.read_text())["result"]["cell"] == 18


@pytest.mark.parametrize("half", ["derive", "draw"])
def test_live_program_stop_during_read_allows_same_event_recovery(live_bridge, half):
    bridge, state, commit, _ = live_bridge
    event = commit(14)
    program = state["programs"][0] if half == "derive" else state["programs"][0]["draw"]
    state["during_read"] = lambda: program.update(state="stopped")
    bridge.consume(tolerate_refusals=True)
    assert bridge.session.published_last_sequence == 1
    assert not (bridge.session.out / "live-case.json").exists()
    assert "probabilities program" in state["calls"][-1][1]["value"]
    program["state"] = "running"
    bridge.session.events.pending.append(event)
    bridge.consume()
    assert bridge.session.published_last_sequence == 2
    assert json.loads((bridge.session.out / "live-case.json").read_text())["result"]["cell"] == 14


def test_live_waits_for_resource_publication_by_event(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    current = deepcopy(state["bars"])
    state["bars"]["step"] -= 1
    state["on_wait"] = lambda: state.update(bars=current)
    bridge.consume()
    assert len(state["resource_reads"]) == 2
    assert bridge.session.published_last_sequence == 2


@pytest.mark.parametrize("completed_step,publication_step", [(14, 16), (1219, 1564)])
@pytest.mark.parametrize("advance_after_read", [False, True])
def test_live_delayed_derive_publication_accepts_current_case(
        live_bridge, completed_step, publication_step, advance_after_read):
    bridge, state, commit, _ = live_bridge
    commit(14)
    bridge.consume()
    prior_bars = deepcopy(state["bars"])
    commit(16)
    state["status"]["inference"]["comparison"]["completed_step"] = completed_step
    inference = deepcopy(state["status"]["inference"])
    current_bars = deepcopy(state["bars"])
    current_bars["step"] = publication_step
    state["bars"] = prior_bars
    program = state["programs"][0]
    program["last_step"] = prior_bars["step"]

    def publish():
        # A later derive pass reads the same committed model output.
        state["bars"] = current_bars
        program["last_step"] = publication_step
        if advance_after_read:
            def advance():
                state["bars"] = {**current_bars, "step": publication_step + 1}
                program["last_step"] = publication_step + 1
            state["during_read"] = advance

    state["on_wait"] = publish
    reads = len(state["resource_reads"])
    bridge.consume()
    assert len(state["resource_reads"]) == reads + 2
    assert state["status"]["inference"] == inference
    assert bridge.session.published_live_state == inference
    assert bridge.session.published_last_sequence == 3
    assert json.loads((bridge.session.out / "live-case.json").read_text())["result"]["cell"] == 16
    assert "After:" in state["calls"][-1][1]["value"]


@pytest.mark.parametrize("publication_step", [14, 16])
def test_serialized_collection_accepts_derive_progress_without_a_new_case(live_bridge, publication_step):
    bridge, state, commit, _ = live_bridge
    session = bridge.session
    session.run_id = "fixture"
    session.published_programs_ready = True

    def action(name):
        assert name == published.action_name(16)
        event = commit(16)
        state["status"]["inference"]["comparison"]["completed_step"] = 14
        state["bars"]["step"] = publication_step
        program = state["programs"][0]
        program["last_step"] = publication_step
        state["during_read"] = lambda: program.update(last_step=publication_step + 1)
        return event

    session.action = action
    entry = driver.published_action(session, 16, session.published_preparation_id)
    assert entry["completed"], entry.get("error")
    assert entry["publication_step"] == publication_step
    assert entry["status"]["inference"]["comparison"]["completed_step"] == 14
    assert session.published_live_state == state["status"]["inference"]
    assert session.published_last_sequence == 2 and len(entry["bars"]) == 10
    assert "After:" in state["calls"][-1][1]["value"]


@pytest.mark.parametrize("fault", [None, "stopped", "values", "resource_ahead_of_program"])
def test_serialized_collection_checks_the_same_current_resources(live_bridge, fault):
    bridge, state, commit, _ = live_bridge
    session = bridge.session
    session.run_id = "fixture"
    session.published_programs_ready = True
    def action(name):
        assert name == published.action_name(16)
        event = commit(16)
        if fault == "stopped": state["programs"][0]["state"] = "stopped"
        if fault == "values": state["bars"]["elements"][0][4] += .125
        if fault == "resource_ahead_of_program": state["bars"]["step"] += 1
        return event
    session.action = action
    entry = driver.published_action(session, 16, session.published_preparation_id)
    assert entry["completed"] is (fault is None)
    if fault is None:
        assert entry["publication_step"] == 2 and len(entry["bars"]) == 10
        assert state["resource_reads"] and state["program_reads"]
    else:
        assert entry["error"] and not state["calls"]


def test_serialized_acceptance_still_refuses_sequence_gaps(callback_boundary):
    root, prepared, session, state = callback_boundary
    assert driver.published_action(session, 10, prepared["id"])["completed"]
    state["sequence"] += 1
    entry = driver.published_action(session, 14, prepared["id"])
    assert not entry["completed"] and "missing sequence" in entry["error"]
    assert session.published_last_sequence == 1


def test_live_duplicate_of_serialized_event_preserves_the_validated_readout(callback_boundary):
    root, prepared, session, state = callback_boundary
    session.replication_profile = published.PROFILE
    entry = driver.published_action(session, 10, prepared["id"])
    assert entry["completed"]
    session.events.pending.append(deepcopy(entry["event"]))
    driver.Bridge(session).consume()
    assert session.published_last_sequence == 1
    assert not (root / "live-case.json").exists()


def test_live_waits_for_cached_program_status_after_current_resources(live_bridge):
    bridge, state, commit, _ = live_bridge
    commit(14)
    step = state["bars"]["step"]
    state["programs"][0]["last_step"] = step - 1
    state["on_wait"] = lambda: state["programs"][0].update(last_step=step)
    bridge.consume()
    assert bridge.session.published_last_sequence == 2
    assert "After:" in state["calls"][-1][1]["value"]


def test_serialized_collection_waits_for_cached_program_status(live_bridge):
    bridge, state, commit, _ = live_bridge
    session = bridge.session
    session.run_id = "fixture"
    session.published_programs_ready = True

    def action(name):
        event = commit(16)
        step = state["bars"]["step"]
        state["programs"][0]["last_step"] = step - 1
        state["on_wait"] = lambda: state["programs"][0].update(last_step=step)
        return event

    session.action = action
    entry = driver.published_action(session, 16, session.published_preparation_id)
    assert entry["completed"], entry.get("error")
    assert "After:" in state["calls"][-1][1]["value"]
