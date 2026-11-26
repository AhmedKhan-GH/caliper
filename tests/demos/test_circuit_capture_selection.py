"""CPU checks for selection frames with unchanged attribution resources."""
from copy import deepcopy

import pytest

from test_circuit_trace import CaptureResources, Failure, driver


@pytest.mark.parametrize("order,ack_step", [
    ("state_before_ack", 3), ("ack_before_state", 1), ("same_node", 3),
])
def test_capture_uses_encoded_selection_without_an_ack_resource(tmp_path, monkeypatch, order, ack_step):
    harness = CaptureResources(tmp_path, monkeypatch)
    harness.cursor.update(step=3, latest=3)
    harness.session.bridge.selection_step = ack_step
    received = deepcopy(harness.received)
    # Both deliveries precede capture. Only the control acknowledgement differs.
    if order != "same_node":
        harness.draw_params["selected_node"] = -1
    def advance():
        if harness.stage == 2:
            harness.draw_params["selected_node"] = 4
    harness.advance = advance
    png, _ = harness.capture()
    assert png.read_bytes() == b"png"
    assert harness.received == received and harness.shown == received
    assert harness.captures[0][0] == (1 if order == "same_node" else 2)
    evidence = harness.session.program_evidence["before"]
    assert evidence["expected_steps"]["attribution"] == 1
    assert evidence["resource_steps"]["attribution"] == received["attribution"]
    assert evidence["selected_node"] == 4


@pytest.mark.parametrize("selected", [-1, 7])
def test_fresh_frames_with_stale_selection_cannot_pass(tmp_path, monkeypatch, selected):
    harness = CaptureResources(tmp_path, monkeypatch)
    harness.draw_params["selected_node"] = selected
    with pytest.raises(Failure, match="Timed out: rendered circuit resources"):
        harness.capture()
    assert harness.frames > 8 and harness.stage == 5
    assert not harness.captures and harness.cursor["live"]


@pytest.mark.parametrize("params", [None, {}, {"unrelated": 4}])
def test_missing_encoded_selection_fails_closed(tmp_path, monkeypatch, params):
    harness = CaptureResources(tmp_path, monkeypatch)
    body = harness.body
    def status(command, **args):
        result = body(command, **args)
        if command == "program.status":
            result["programs"][0]["draw"]["params"] = params
        return result
    harness.body = status
    with pytest.raises(Failure, match="lacks encoded selection parameters"):
        harness.capture()
    assert harness.stage == 1 and not harness.captures
    assert harness.cursor["live"]


def test_clear_selection_requires_a_frame_with_the_clear_value(tmp_path, monkeypatch):
    harness = CaptureResources(tmp_path, monkeypatch)
    harness.session.bridge.last_selection = None
    harness.session.bridge.selection_step = None
    harness.advance = lambda: harness.draw_params.update(selected_node=-1) if harness.stage == 2 else None
    harness.capture()
    assert harness.captures[0][0] == 2
    assert harness.session.program_evidence["before"]["selected_node"] == -1


def test_selection_does_not_lower_a_required_resource_step(tmp_path, monkeypatch):
    harness = CaptureResources(tmp_path, monkeypatch)
    original = dict(harness.cursor, step=1, live=False, playing=True, speed=3)
    harness.cursor = dict(original)
    with pytest.raises(Failure, match="Timed out: capture cursor step"):
        harness.capture(minimum_step=3)
    assert harness.cursor == original and not harness.captures


@pytest.mark.parametrize("half", ["derive", "draw"])
@pytest.mark.parametrize("fault", ["refused", "stopped", "error"])
def test_terminal_program_errors_abort_before_the_cursor_hold(tmp_path, monkeypatch, half, fault):
    harness = CaptureResources(tmp_path, monkeypatch)
    body = harness.body
    def status(command, **args):
        result = body(command, **args)
        if command == "program.status":
            program = result["programs"][0]
            target = program if half == "derive" else program["draw"]
            target["error" if fault == "error" else "state"] = fault
        return result
    harness.body = status
    with pytest.raises(Failure, match="reports an error"):
        harness.capture()
    assert harness.stage == 0 and harness.cursor["live"]
    assert not harness.captures


@pytest.mark.parametrize("change", ["selection", "resource", "cursor", "stopped"])
def test_capture_completion_rejects_changed_draw_evidence(tmp_path, monkeypatch, change):
    harness = CaptureResources(tmp_path, monkeypatch)
    capture = harness.call
    def changed(command):
        reply = capture(command)
        if change == "selection":
            harness.draw_params["selected_node"] = 7
        elif change == "resource":
            harness.shown["attribution"]["strings"] = 1
        elif change == "cursor":
            harness.cursor["step"] = 1
        else:
            harness.stopped = True
        return reply
    harness.call = changed
    with pytest.raises(Failure):
        harness.capture()
    assert harness.cursor["live"]
    assert not (tmp_path / "before.png").exists()


@pytest.mark.parametrize("order", ["params_first", "control_first", "same_node"])
def test_bridge_control_and_program_params_have_independent_progress(tmp_path, monkeypatch, order):
    from types import SimpleNamespace
    from caliper import _controls as controls, _programs as programs

    harness = CaptureResources(tmp_path, monkeypatch)
    session = harness.session
    session.run_id = "capture-fixture"
    bridge = session.bridge = driver.Bridge(session)
    bridge.examples_present = True
    owner = controls.Controls()
    previous = 4 if order == "same_node" else -1
    if order == "same_node":
        bridge.last_selection = 4
        bridge.last_examples = {"examples": ["previous"]}
        bridge.selection_generation = 1
    selected = owner.param("selected_node", previous, min=-1, max=8191)
    owner.state = controls.RUN_RUNNING
    owner.n = 3
    program = programs._Program("attribution", {"params": {"selected_node": previous}}, "", None, "")
    program.state = "running"
    program.kind = "piece"
    program.runs, program.last_step = 1, 2
    program.params_dirty = False
    monkeypatch.setattr(programs, "_PROGS", {"attribution": program})
    monkeypatch.setattr(programs, "_OUTBOX", {})
    monkeypatch.setattr(programs, "_controls_ports", lambda: [])
    route = SimpleNamespace(transport=SimpleNamespace(may_control=lambda connection: True))
    received = deepcopy(harness.received)
    timeline = []
    shared = {"selected_node": previous}
    harness.draw_params["selected_node"] = previous
    session.events.complete = lambda command, **args: {"maximum": 1} if command == "feature.show" else {"examples": []}
    body = harness.body

    def program_params():
        # Native refresh sends only changed values. The real producer applies them.
        if program.params["selected_node"] != shared["selected_node"]:
            timeline.append("params")
            programs.set_params("attribution", dict(shared))

    def delivered(command, **args):
        if command == "selection.set":
            shared["selected_node"] = args["item"]
            session.events.pending.append({"kind": "pick", "panel": "graph",
                                           "fields": {"kind": 1, "index": args["item"]}})
            if order != "control_first":
                program_params()
            return {"fields": {"index": args["item"]}}
        if command == "run.set_param":
            reply = owner._on_control(route, "fixture", {"request_id": 1, "op": "set",
                                                       "name": args["name"], "value": args["value"]})
            assert reply["status"] == "ok" and selected.pending == 4
            # Use the actual checkpoint operation, without a model or derive pass.
            with owner.lock:
                assert owner._apply_pending()
            timeline.append("control")
            if order == "control_first":
                program_params()
            return reply
        if command == "run.state":
            meta = owner.controls_meta()
            return {"step": meta["run"]["step"], "params": meta["params"]}
        if command == "state.set":
            assert args["id"] == "activation_max"
            return {}
        return body(command, **args)

    harness.body = delivered
    harness.advance = lambda: harness.draw_params.update(shared)
    assert bridge.select(4) == {"examples": []}
    assert selected.value == 4 and selected.pending is None and selected._version == 1
    assert bridge.selection_step == 3
    expected = {"params_first": ["params", "control"], "control_first": ["control", "params"],
                "same_node": ["control"]}
    assert timeline == expected[order]
    assert programs.inference_pending() is (order != "same_node")
    harness.capture()
    assert harness.received == received and harness.shown == received
    assert (program.runs, program.last_step) == (1, 2)
    assert harness.session.program_evidence["before"]["selected_node"] == 4
