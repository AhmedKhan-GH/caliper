"""Live publication recovery at the offline MCP boundary."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_published_dallas import callback_boundary, driver, evidence_factory, published, write_json
from _mcp import Failure


@pytest.fixture
def live_bridge(evidence_factory):
    root, prepared, make = evidence_factory
    baseline = make(10, 1)
    evidence = published.validate_case(root, baseline, 10, previous_sequence=0)
    state = {"status": baseline, "sequence": 1, "calls": []}
    session = SimpleNamespace(
        out=root, replication_profile=published.PROFILE, private_output_directory=root,
        published_last_sequence=1, published_preparation_id=prepared["id"],
        published_baseline_hash=evidence["baseline_array_sha256"],
        events=SimpleNamespace(pending=[], complete=lambda method: deepcopy(state["status"])),
        mcp=SimpleNamespace(body=lambda method, **args: state["calls"].append((method, args))))

    def commit(cell, *, success=True):
        state["sequence"] += 1
        if success:
            state["status"] = make(cell, state["sequence"])
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
    second = commit(16)
    replies = iter([prior, deepcopy(state["status"])])
    bridge.session.events.complete = lambda method: next(replies)
    with pytest.raises(Failure, match="changed during case publication"):
        bridge.handle(first)
    assert bridge.session.published_last_sequence == 1
    assert not state["calls"]
    assert not (bridge.session.out / "live-case.json").exists()
    bridge.session.events.complete = lambda method: deepcopy(state["status"])
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
    bridge.consume()
    assert bridge.session.published_last_sequence == 3
    assert json.loads((bridge.session.out / "live-case.json").read_text())["result"]["cell"] == 16


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
