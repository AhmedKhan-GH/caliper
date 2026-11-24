"""Circuit math checks and the real hidden MCP acceptance driver."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demos" / "circuit_trace"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
import run as driver
from _checks import check, finish, math_checks, program_budget_checks, selected_feature, token_path, FIXTURE_PROMPT
from _mcp import Failure


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Use unchanged installed seed7 resources and actual nonlinear output math."""
    from caliper.interp._hosted import fixture_resources, load_resources
    from caliper.interp import Tracer
    from caliper import _controls
    from caliper.interp._output import Output
    from caliper.interp._validate import validate_trace
    folder = tmp_path_factory.mktemp("circuit-dictionary") / "dictionary"
    fixture_resources(folder)
    model, tokenizer, dictionary, index, _ = load_resources("tiny-lm", folder)
    owner = _controls.Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    graph = Tracer(model, tokenizer, dictionary, controls=owner).trace(FIXTURE_PROMPT)
    data = graph.read(node_limit=256, edge_limit=256)
    # Read every page because influence pruning can retain more than 256 edges.
    for kind in ("node", "edge"):
        offset = data[f"next_{kind}_offset"]
        while offset is not None:
            page = graph.read(**{f"{kind}_offset": offset,
                                 "node_limit": 256, "edge_limit": 256})
            data[f"{kind}s"].extend(page[f"{kind}s"])
            offset = page[f"next_{kind}_offset"]
        data[f"next_{kind}_offset"] = None
    selected = selected_feature(data)
    output = Output(model, tokenizer, service)
    before = output.compare(FIXTURE_PROMPT)
    service.set({"kind": "clamp", "point": dictionary.metadata["write_points"][selected["layer"]],
                 "feature": selected["feature"], "positions": [selected["position"]], "value": 0.})
    owner._apply_interventions()
    after = output.compare(FIXTURE_PROMPT)
    service.clear()
    owner._apply_interventions()
    validation = validate_trace(model, tokenizer, dictionary, service, graph, prompt=FIXTURE_PROMPT)
    def bars(output):
        return [{"token": token, "baseline": output["probabilities"][0][i],
                 "probability": output["probabilities"][1][i]} for i, token in enumerate(output["tokens"])]
    result = {"graph": data, "selected": selected, "before": bars(before), "after": bars(after),
              "examples": {"examples": index.examples(selected["layer"], selected["feature"], limit=16)},
              "validation": validation}
    # These identities exercise the pure gate on this fixture, not the installed metadata oracle.
    result["argmax_token_ids"] = [max(result["before"], key=lambda r: r["baseline"])["token"],
                                  max(result["after"], key=lambda r: r["probability"])["token"]]
    driver.save(folder.parent / "fixture-math.json", result)
    yield result
    service.remove()
    index.close()


def test_actual_fixture_selection_and_graph(measured):
    graph = measured["graph"]
    assert len(graph["nodes"]) == 52
    assert len(graph["edges"]) == 596
    assert measured["selected"]["id"] == "feature:0:30:138"
    assert graph["completeness"]["max_absolute_residual"] <= 1e-9
    assert graph["feature_share"] >= .30
    assert token_path(graph)


def test_actual_fixture_gates_keep_semantics_separate(measured):
    gates = math_checks(measured["graph"], measured["examples"], measured["before"], measured["after"],
                        measured["validation"], 0, 0, fixture=True, argmax_token_ids=measured["argmax_token_ids"])
    # Zero durations here exercise the pure gate function. They are not timing evidence.
    by_name = {g["name"]: g for g in gates}
    assert all(g["passed"] for g in gates if g["required"])
    assert by_name["examples"]["passed"]
    assert by_name["fixture_output_effect"]["passed"]
    assert not by_name["steering"]["required"]
    assert by_name["validation"]["required"]
    semantic = math_checks(measured["graph"], measured["examples"], measured["before"], measured["after"],
                           measured["validation"], 21, 2, fixture=False, argmax_token_ids=measured["argmax_token_ids"])
    by_name = {g["name"]: g for g in semantic}
    assert by_name["steering"]["required"] and by_name["validation"]["required"]
    assert not by_name["trace_time"]["passed"]
    assert not by_name["steering_time"]["passed"]


def test_missing_feature_is_a_failure(measured):
    graph = dict(measured["graph"], nodes=[n for n in measured["graph"]["nodes"] if n["kind"] != "feature"])
    with pytest.raises(Failure, match="no selectable feature"):
        selected_feature(graph)


def test_missing_top_bar_is_a_failure(measured):
    top = max(measured["before"], key=lambda r: r["baseline"])["token"]
    after = [r for r in measured["after"] if r["token"] != top]
    with pytest.raises(Failure, match="omit the baseline"):
        math_checks(measured["graph"], measured["examples"], measured["before"], after,
                    measured["validation"], 0, 0, fixture=True, argmax_token_ids=measured["argmax_token_ids"])


def test_finish_refuses_missing_evidence():
    result = {"mode": "fixture", "checks": []}
    assert finish(result) == 1
    assert result["qwen_acceptance"] is False


def inference_fixture():
    from _inference_evidence import text_hash
    prompt = "A neutral fixture prompt"
    state = {"prompt_hash": text_hash(prompt), "interventions_hash": text_hash("[]")}
    completion = dict(state, id=2, token_ids=[list(range(24)), list(range(24, 48))], counts=[24, 24],
                      completed_step=9, text_hashes=[text_hash("baseline text"), text_hash("steered text")])
    prior = dict(state, generation=dict(completion, id=1, completed_step=8))
    status = {"inference": dict(state, generation=completion)}
    params = {"prompt": prompt, "output.baseline": "baseline text", "output.steered": "steered text"}
    return prompt, prior, status, params


def test_generation_requires_fresh_identity_before_text_acceptance():
    from _inference_evidence import generation_baseline, generation_evidence
    prompt, prior, status, params = inference_fixture()
    assert generation_baseline({"inference": prior}, prompt) == prior
    assert generation_evidence(prior, {"inference": prior}, params, prompt) is None
    result = generation_evidence(prior, status, params, prompt)
    assert result["prior_id"] == 1 and result["completion"]["counts"] == [24, 24]


@pytest.mark.parametrize("fault", ["one_token", "counts", "bool_token", "negative_token", "missing_row",
                                    "missing_text", "stale_text", "missing_hashes", "prompt", "interventions",
                                    "old_completion", "old_step", "skipped_identity"])
def test_generation_rejects_incomplete_or_unbound_completion(fault):
    from _inference_evidence import generation_evidence, text_hash
    prompt, prior, status, params = inference_fixture()
    state = status["inference"]
    record = state["generation"]
    if fault == "one_token": record["token_ids"] = [[1], [2]]
    elif fault == "counts": record["counts"] = [1, 1]
    elif fault == "bool_token": record["token_ids"][0][0] = True
    elif fault == "negative_token": record["token_ids"][1][0] = -1
    elif fault == "missing_row": record["token_ids"].pop()
    elif fault == "missing_text": params.pop("output.steered")
    elif fault == "stale_text": params["output.baseline"] = "older text"
    elif fault == "missing_hashes": record.pop("text_hashes")
    elif fault == "prompt": state["prompt_hash"] = text_hash("changed prompt")
    elif fault == "interventions": state["interventions_hash"] = text_hash("changed interventions")
    elif fault == "old_completion": record["interventions_hash"] = text_hash("older interventions")
    elif fault == "old_step": record["completed_step"] = 8
    elif fault == "skipped_identity": record["id"] = 3
    with pytest.raises(Failure):
        generation_evidence(prior, status, params, prompt)


def test_comparison_uses_true_argmax_and_rejects_stale_metadata():
    from _inference_evidence import comparison_evidence, text_hash
    prompt, _, status, _ = inference_fixture()
    state = status["inference"]
    record = {key: state[key] for key in ("prompt_hash", "interventions_hash")}
    state["comparison"] = dict(record, argmax_token_ids=[1, 23], completed_step=8)
    bars = [{"token": 1}, {"token": 23}]
    assert comparison_evidence(status, prompt, bars)["argmax_token_ids"] == [1, 23]
    with pytest.raises(Failure, match="omit an actual argmax"):
        comparison_evidence(status, prompt, bars[:1])
    state["comparison"]["interventions_hash"] = text_hash("old state")
    with pytest.raises(Failure, match="stale"):
        comparison_evidence(status, prompt, bars)


def test_semantic_token_change_uses_metadata_without_relaxing_shorter_bar(measured):
    # A pure oracle case. The separate strict Qwen shorter-bar gate remains in the driver.
    old = measured["argmax_token_ids"][0]
    other = next(row["token"] for row in measured["after"] if row["token"] != old)
    gates = math_checks(measured["graph"], measured["examples"], measured["before"], measured["before"],
                        measured["validation"], 0, 0, fixture=False, argmax_token_ids=[old, other])
    steering = next(g for g in gates if g["name"] == "steering")
    assert steering["passed"] and steering["measured"]["probability_fall"] == 0


def synthetic_capture(tmp_path, *, missing_panel=None, text_fault=None):
    """Paint a tiny independent CPU scene. This is never application evidence."""
    import numpy as np
    from PIL import Image
    window = np.full((88, 216, 4), [9, 11, 13, 255], dtype=np.uint8)
    atlas = np.full((7, 9, 4), 255, dtype=np.uint8)
    atlas[:, :, 3] = 0
    glyphs = {
        "A": np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=np.uint8),
        "B": np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8),
    }
    atlas[1:6, 1:4, 3] = glyphs["A"] * 255
    atlas[1:6, 5:8, 3] = glyphs["B"] * 255
    atlas.tofile(tmp_path / "atlas.rgba")
    metadata = {"capture": 1, "origin": [10, 20], "pixel_scale": [2, 2], "width": 216, "height": 88,
                "window": "window.rgba", "font_atlas": {"width": 9, "height": 7, "file": "atlas.rgba"},
                "panels": {}, "expected_steps": {"attribution": 1, "example_tokens": 1, "probabilities": 1}}
    def points(rect):
        return [rect[0] / 2 + 10, rect[1] / 2 + 20, rect[2] / 2 + 10, rect[3] / 2 + 20]
    for index, (name, program) in enumerate((("graph", "attribution"), ("examples", "example_tokens"), ("output", "probabilities"))):
        source = np.full((32, 32, 4), [20, 24, 28, 255], dtype=np.uint8)
        ids = np.zeros((32, 32, 2), dtype="<u4")
        pick = index + 1
        if name == "graph":
            for item, x in enumerate((5, 13, 21)):
                source[9:15, x:x + 5] = [180, 80, 60, 255]
                ids[9:15, x:x + 5] = [pick, item]
        elif name == "examples":
            for row in range(8):
                source[3 + row * 3:5 + row * 3, 5:27] = [50, 90 + row * 10, 130, 255]
                ids[3 + row * 3:5 + row * 3, 5:27] = [pick, row]
        else:
            source[8:24, 10:18] = [30, 180, 220, 255]
            ids[8:24, 10:18] = [pick, 0]
        source.tofile(tmp_path / (name + ".rgba"))
        ids.tofile(tmp_path / (name + ".id"))
        left, top = 4 + index * 72, 8
        rect = points([left, top, left + 64, top + 64])
        clip = points([left + 4, top + 2, left + 64, top + 64])
        # Pillow supplies independent pixel-center scaling for this synthetic scene.
        painted = np.asarray(Image.fromarray(source).resize((64, 64), Image.Resampling.BILINEAR))
        if missing_panel != name:
            window[top + 2:top + 64, left + 4:left + 64] = painted[2:, 4:]
        matrix = [0.] * 16
        matrix[5] = 1 / .72
        panel = {"width": 32, "height": 32, "ids": name + ".id", "color": name + ".rgba",
                 "generation": 1, "programs": [{"id": program, "pick_id": pick, "step": 1}],
                 "clip_from_world": matrix, "labels": [],
                 "composition": {"rect": rect, "clip": clip, "uv": [0, 0, 1, 1], "tint": [255] * 4,
                                 "sampler": "linear", "blend": "source_alpha"}}
        if name == "graph":
            for item, text in enumerate(("A", "B")):
                x, y = left + 14 + item * 18, top + 48
                uv_left = 1 if text == "A" else 5
                quads = []
                for offset, color in ((2, [0, 0, 0, 255]), (0, [240, 240, 240, 255])):
                    quads.append({"rect": points([x + offset, y + offset, x + offset + 3, y + offset + 5]),
                                  "clip": clip, "uv": [uv_left / 9, 1 / 7, (uv_left + 3) / 9, 6 / 7], "tint": color})
                    if text_fault == "missing" or (text_fault == "selected_missing" and text == "B"):
                        continue
                    bitmap = glyphs["B"] if text_fault == "wrong" and text == "A" else glyphs[text]
                    region = window[y + offset:y + offset + 5, x + offset:x + offset + 3]
                    region[bitmap.astype(bool)] = color
                panel["labels"].append({"text": text, "box": points([x, y, x + 5, y + 7]), "quads": quads})
                if text_fault == "variation":
                    window[y:y + 7, x:x + 5] = [20, 24, 28, 255]
                    window[y:y + 7:2, x:x + 5] = [170, 100, 50, 255]
        metadata["panels"][name] = panel
    window.tofile(tmp_path / "window.rgba")
    Image.fromarray(window).save(tmp_path / "capture.png")
    graph = {"tokens": [{"text": "A"}], "nodes": [{}, {}, {}]}
    bars = [{"index": 0, "token": 7, "baseline": .5, "probability": .5}]
    return metadata, graph, bars


def synthetic_gates(tmp_path, **faults):
    from _checks import capture_checks
    metadata, graph, bars = synthetic_capture(tmp_path, **faults)
    gates, measured = capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars, selected={"label": "B"})
    return {gate["name"]: gate for gate in gates}, measured


def test_composition_oracle_handles_dpi_clip_scaled_targets_and_glyphs(tmp_path):
    gates, measured = synthetic_gates(tmp_path)
    assert all(gate["passed"] for gate in gates.values()), gates
    assert measured["height"] == 32 and measured["expected_height"] == pytest.approx(32)
    for name in ("graph", "examples", "output"):
        evidence = gates[name + "_composition"]["measured"]
        assert evidence["pixels"] == 60 * 62
        assert evidence["source_items"] == evidence["window_items"]


@pytest.mark.parametrize("panel", ["graph", "examples", "output"])
def test_missing_window_composition_fails_with_valid_offscreen_targets(tmp_path, panel):
    gates, _ = synthetic_gates(tmp_path, missing_panel=panel)
    assert gates["capture_generation"]["passed"]  # Equal copies do not prove composition.
    assert gates["rendered_nodes"]["passed"] and gates["example_page"]["passed"]
    assert not gates[panel + "_composition"]["passed"]


@pytest.mark.parametrize("fault", ["missing", "selected_missing", "wrong", "variation"])
def test_independent_text_oracle_rejects_wrong_or_missing_glyphs(tmp_path, fault):
    gates, _ = synthetic_gates(tmp_path, text_fault=fault)
    assert gates["capture_generation"]["passed"]
    assert not gates["host_labels"]["passed"]


def test_missing_panel_placement_is_a_failure(tmp_path):
    from _checks import capture_checks
    metadata, graph, bars = synthetic_capture(tmp_path)
    del metadata["panels"]["graph"]["composition"]
    with pytest.raises(Failure, match="composition evidence"):
        capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars, selected={"label": "B"})


def test_text_oracle_requires_each_repeated_graph_label():
    from _composition import required_text
    assert not required_text([{"text": "A", "passed": True}], ["A", "A"])
    assert required_text([{"text": "A", "passed": True}] * 2, ["A", "A"])


def test_text_oracle_rejects_partial_glyphs_under_the_actual_clip(tmp_path):
    from _checks import capture_checks
    metadata, graph, bars = synthetic_capture(tmp_path)
    label = metadata["panels"]["graph"]["labels"][0]
    for quad in label["quads"]:
        quad["clip"][2] = quad["rect"][0] + .5
    gates, _ = capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars, selected={"label": "B"})
    text = next(gate for gate in gates if gate["name"] == "host_labels")
    assert not text["passed"]
    assert not text["measured"]["glyphs"][0]["complete"]


def test_visible_result_keeps_pixel_evidence_optional():
    # This pure contract case is not a measured application acceptance result.
    result = {"mode": "qwen", "pixel_evidence_collected": False,
              "checks": [check("hidden_capture_evidence", False, "not collected", "hidden only", required=False)]}
    assert finish(result) == 0
    assert result["qwen_acceptance"] is False
    result["checks"][0]["required"] = True
    assert finish(result) == 1


@pytest.mark.parametrize("change,passed,evidence", [
    ({}, True, "measured"),
    ({"last_pass_ms": None}, False, "unverified"),
    ({"last_pass_ms": 50.01}, False, "measured"),
    ({"memory_bytes": (64 << 20) + 1}, False, "measured"),
    ({"frame_ms": None}, False, "unverified"),
    ({"frame_ms": 4.01}, False, "measured"),
])
def test_program_budgets_reject_missing_or_excess_measurements(change, passed, evidence):
    # This pure validation input is not saved as acceptance evidence.
    program = {"id": "attribution", "last_pass_ms": 50, "memory_bytes": 64 << 20,
               "memory_limit_bytes": 64 << 20, "draw": {"frame_ms": 4, "frame_limit_ms": 4,
               "memory_bytes": 64 << 20, "memory_limit_bytes": 64 << 20}}
    for key, value in change.items():
        (program["draw"] if key == "frame_ms" else program)[key] = value
    snapshot = {"status": {"programs": [program]}, "expected_steps": {"attribution": 1},
                "pieces": {"attribution": {"derive": {"time_ms": 50, "memory_mb": 64},
                                           "draw": {"frame_ms": 4, "memory_mb": 64}}}}
    gates = program_budget_checks(snapshot)
    assert all(gate["passed"] for gate in gates) is passed
    if change:
        failed = [gate for gate in gates if not gate["passed"]]
        assert len(failed) == 1 and failed[0]["evidence"] == evidence
        assert failed[0]["measured"] == next(iter(change.values()))
    result = {"mode": "fixture", "checks": gates}
    assert finish(result) == (0 if passed else 1)


def test_bridge_sets_selection_and_queries_examples():
    calls = []
    class Calls:
        def body(self, method, **args):
            calls.append((method, args))
    class Events:
        pending = [{"kind": "pick", "panel": "graph", "fields": {"kind": 1, "index": 10}}]
        def complete(self, method, **args):
            calls.append((method, args))
            return {"maximum": 2} if method == "feature.show" else {"examples": []}
    class Session:
        mcp = Calls()
        events = Events()
        def parameter(self, name, value):
            calls.append((name, value))
    session = Session()
    # This transport unit test writes no acceptance artifact.
    from unittest.mock import patch
    with patch.object(driver, "save"), patch.object(Session, "out", Path("unused"), create=True):
        bridge = driver.Bridge(session)
        bridge.consume()
    assert calls == [("selected_node", 10), ("feature.show", {"node": 10}),
                     ("feature.examples", {"node": 10, "limit": 16}),
                     ("state.set", {"id": "activation_max", "value": 2})]
    assert bridge.last_selection == 10


def test_visible_bridge_continues_after_first_event():
    from unittest.mock import Mock
    session = Mock()
    session.live.side_effect = [True, True, False]
    bridge = driver.Bridge(session)
    bridge.consume = Mock()
    bridge.keep_alive()
    assert session.events.wait.call_count == 2
    assert bridge.consume.call_count == 2


def test_bridge_reconciles_run_selection_without_parameter_feedback(tmp_path):
    from unittest.mock import Mock, patch
    session = Mock(out=tmp_path)
    bridge = driver.Bridge(session)
    bridge.last_selection = 10
    bridge.examples_present = True
    session.events.pending = [{"kind": "run", "change": "param", "name": "selected_node", "value": -1}]
    bridge.consume()
    session.mcp.body.assert_any_call("state.set", id="selected_node", value=-1)
    session.mcp.body.assert_any_call("program.remove", id="example_tokens")
    assert bridge.last_selection is None and bridge.examples_present is False
    session.events.pending = [{"kind": "run", "change": "param", "name": "selected_node", "value": 12}]
    session.events.complete.side_effect = [{"maximum": 2}, {"examples": []}, {"examples": []}]
    with patch.object(driver, "programs") as programs:
        bridge.consume()
    session.parameter.assert_not_called()
    programs.assert_called_once_with(session, examples=True)
    session.ready_programs.assert_called_once_with(["example_tokens"])
    session.mcp.body.assert_any_call("state.set", id="selected_node", value=12)
    assert bridge.last_selection == 12 and bridge.examples_present is True


def test_visible_bridge_records_refusal_and_consumes_next_event(tmp_path):
    from unittest.mock import Mock
    session = Mock(out=tmp_path)
    session.events.pending = [{"kind": "pick", "panel": "graph"}, {"kind": "control", "control": "page"}]
    bridge = driver.Bridge(session)
    bridge.handle = Mock(side_effect=[Failure("stale feature"), None])
    bridge.consume(tolerate_refusals=True)
    assert bridge.handle.call_count == 2
    assert json.loads((tmp_path / "bridge-errors.jsonl").read_text())["error"] == "stale feature"


def test_publication_wait_rejects_a_stale_step():
    from unittest.mock import Mock
    session = driver.Session.__new__(driver.Session)
    session.mcp = Mock()
    session.mcp.body.side_effect = [{"step": 2}, {"step": 4}]
    session.ready_programs = Mock()
    session.events = Mock()
    def until(description, read):
        assert read() is False
        return read()
    session.events.until.side_effect = until
    assert session.read("probabilities", "cells", minimum_step=3) == {"step": 4}
    session.events.complete.assert_called_once_with("feature.dictionary")
    session.ready_programs.assert_called_once_with(["probabilities"])


@pytest.mark.parametrize("platform,text,expected", [
    ("darwin", "4863754240 maximum resident set size\n12412946880 peak memory footprint\n", 12412946880),
    ("linux", "Maximum resident set size (kbytes): 12345\n", 12641280),
])
def test_process_high_water_uses_platform_measurement(tmp_path, platform, text, expected):
    (tmp_path / "run-memory.txt").write_text(text)
    assert driver.memory_bytes(tmp_path, platform=platform) == expected


def test_macos_footprint_refuses_rss_fallback(tmp_path):
    (tmp_path / "run-memory.txt").write_text("4863754240 maximum resident set size\n")
    with pytest.raises(Failure, match="No peak memory footprint"):
        driver.memory_bytes(tmp_path, platform="darwin")


def stdio_child(tmp_path, *, refuse=False):
    """Create a bounded standard-library transport child, with no Caliper import."""
    path = tmp_path / "stdio-child"
    path.write_text(f"#!{sys.executable}\n" + "\n".join([
        "import json, os, pathlib, sys",
        "pathlib.Path(sys.argv[-1]).with_suffix('.pid').write_text(str(os.getpid()))",
        "for line in sys.stdin:",
        "    request = json.loads(line)",
        f"    reply = {{'id': request['id'], {'error' if refuse else 'result'!r}: {{}}}}",
        "    print(json.dumps(reply), flush=True)",
    ]) + "\n")
    path.chmod(0o700)
    return path


def test_mcp_startup_failure_reaps_owned_standard_library_child(tmp_path):
    from _mcp import MCP
    import psutil
    path = stdio_child(tmp_path, refuse=True)
    with pytest.raises(Failure, match="server/discover"):
        MCP(path, tmp_path / "api.json", tmp_path, os.environ.copy())
    pid = int((tmp_path / "api.pid").read_text())
    assert not psutil.pid_exists(pid)


def test_mcp_close_failure_still_reaps_child_and_closes_other_files(tmp_path):
    from _mcp import MCP
    from unittest.mock import patch
    client = MCP(stdio_child(tmp_path), tmp_path / "api.json", tmp_path, os.environ.copy())
    original = client.process.stdin
    class FailedClose:
        def close(self):
            raise OSError("injected input close failure")
    try:
        with patch.object(client.process, "stdin", FailedClose()):
            with pytest.raises(Failure, match="injected input close failure"):
                client.close()
        assert client.process.poll() is not None
        assert all(f.closed for f in (client.log, client.journal, client.event_journal, client.process.stdout))
    finally:
        original.close()
        if client.process.poll() is None:
            driver.stop(client.process)


def test_session_event_and_mcp_close_faults_preserve_app_cleanup(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    import psutil
    app = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE)
    session = driver.Session.__new__(driver.Session)
    session.out, session.app, session.app_process = tmp_path, app, None
    session.app_owner = psutil.Process(app.pid)
    session.app_process = session.app_owner
    session.app_log = (tmp_path / "app.log").open("w")
    session.started, session.cleanup_error = False, None
    session.events = SimpleNamespace(history=[])
    session.mcp = Mock()
    session.mcp.close.side_effect = OSError("injected MCP close failure")
    try:
        with patch.object(driver, "save", side_effect=OSError("injected event save failure")):
            session.close()
        assert app.poll() is not None and session.app_log.closed
        assert "event save" in session.cleanup_error and "MCP close" in session.cleanup_error
    finally:
        driver.stop(app)
        app.stdin.close()


@pytest.mark.parametrize("model,display", [("tiny-lm", "1"), ("local-qwen", None)])
def test_session_ready_failure_reaps_owned_standard_library_child(tmp_path, monkeypatch, model, display):
    from types import SimpleNamespace
    from unittest.mock import patch
    import psutil
    child = tmp_path / "ready-child"
    child.write_text(f"#!{sys.executable}\n" + "\n".join([
        "import json, os, pathlib, signal",
        "folder = pathlib.Path(os.environ['CALIPER_DATA_DIR'])",
        "(folder / 'probe.json').write_text(json.dumps({'pid': os.getpid(), 'display': os.getenv('CALIPER_FIXTURE_DISPLAY_MPS')}))",
        "signal.pause()",
    ]) + "\n")
    child.chmod(0o700)
    actual_wait = driver.wait_file
    def fail_ready(path, **kwargs):
        actual_wait(path.parent / "probe.json", timeout=5, **kwargs)
        raise Failure("injected ready-file failure")
    monkeypatch.setenv("CALIPER_FIXTURE_DISPLAY_MPS", "1")
    args = SimpleNamespace(model=model, dictionary=str(tmp_path), prompt=FIXTURE_PROMPT)
    with patch.object(driver, "executable", return_value=child), patch.object(driver, "wait_file", side_effect=fail_ready):
        with pytest.raises(Failure, match="injected ready-file failure"):
            driver.Session(args, tmp_path / "owned")
    probe = json.loads((tmp_path / "owned/data/probe.json").read_text())
    assert probe["display"] == display
    assert not psutil.pid_exists(probe["pid"])


def test_instrument_uses_three_panels_and_confirmed_bindings():
    from unittest.mock import Mock
    import jsonschema
    session = Mock()
    driver.instrument(session)
    driver.programs(session)
    driver.programs(session, examples=True)
    descriptors = json.loads((ROOT / "src/builder/commands/command_table.json").read_text())["commands"]
    schemas = {d["name"]: d["args"] for d in descriptors}
    calls = session.mcp.body.call_args_list
    for call in calls:
        jsonschema.validate(call.kwargs, schemas[call.args[0]])
    panels = [call.kwargs["id"] for call in calls if call.args[0] == "panel.create"]
    assert panels == ["graph", "examples", "output"]
    graph = next(call.kwargs for call in calls if call.args[0] == "program.create" and call.kwargs["id"] == "attribution")
    assert graph["bind"]["graph_counts"] == "live:trace.counts"
    bars = next(call.kwargs for call in calls if call.args[0] == "program.create" and call.kwargs["id"] == "probabilities")
    assert set(bars["bind"].values()) == {"live:output.probs", "live:output.tokens", "live:output.label_text"}


def headless_executable():
    # Playwright 1.65 selects this registry entry for headless launch without a channel.
    # The public chromium.executablePath() selects the full Chromium artifact.
    script = """
const { registry } = require('playwright-core/lib/coreBundle').registry;
process.stdout.write(registry.findExecutable('chromium-headless-shell').executablePathOrDie('javascript'));
"""
    browser = subprocess.check_output(["node", "-e", script], cwd=ROOT / "web", text=True)
    assert Path(browser).is_file(), "The local Chromium headless shell is absent."
    return browser


def test_headless_preflight_uses_launch_artifact(tmp_path, monkeypatch):
    executable = tmp_path / "headless-shell"
    executable.write_text("artifact check only")
    def resolve(command, **kwargs):
        assert command[:2] == ["node", "-e"]
        assert "findExecutable('chromium-headless-shell')" in command[2]
        assert "executablePathOrDie('javascript')" in command[2]
        assert ".launch(" not in command[2]
        return str(executable)
    monkeypatch.setattr(subprocess, "check_output", resolve)
    assert headless_executable() == str(executable)
    executable.unlink()
    with pytest.raises(AssertionError, match="headless shell is absent"):
        headless_executable()


def test_headless_preflight_preserves_setup_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["node"])
    monkeypatch.setattr(subprocess, "check_output", fail)
    with pytest.raises(subprocess.CalledProcessError):
        headless_executable()


def resources_and_device(*, fixture):
    # Required setup errors precede the only permitted platform skip.
    for name in ("torch", "numpy", "PIL", "websockets", "watchdog", "psutil", "safetensors", "transformers",
                 "caliper._netio", "caliper._producer"):
        __import__(name)
    driver.executable("CALIPER_APP_EXE", "cmake-build-release/caliper")
    driver.executable("CALIPER_MCP_EXE", "cmake-build-release/src/builder/mcp/caliper-mcp")
    if fixture:
        driver.executable("CALIPER_CIRCUIT_PARITY_EXE", "cmake-build-release/caliper_circuit_parity")
        wasm = Path(os.environ.get("CALIPER_WASM_DIR", ROOT / "python/caliper/_web"))
        for name in ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm"):
            assert (wasm / name).is_file(), f"Required browser resource is absent: {name}"
        headless_executable()
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("Circuit fixture: verified absence of a Metal device.")


def hidden(tmp_path, model, dictionary):
    env = dict(os.environ, CALIPER_AGENT_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    out = Path(os.environ.get("CALIPER_CIRCUIT_OUT", tmp_path / "result"))
    command = [sys.executable, str(DEMO / "run.py"), "--model", model, "--dictionary", str(dictionary),
               "--hidden", "--out", str(out)]
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=540)
        except subprocess.TimeoutExpired:
            import psutil
            owned = psutil.Process(process.pid).children(recursive=True)
            process.terminate()
            try:
                process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            finally:
                for child in reversed(owned):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(owned, timeout=10)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            pytest.fail("The hidden driver timed out; its owned processes were closed.")
    report = json.loads((out / "result.json").read_text())
    assert process.returncode == 0, (report, stdout, stderr)
    assert report["passed"]
    return report


def test_hidden_fixture(tmp_path):
    resources_and_device(fixture=True)
    from caliper.interp._hosted import fixture_resources
    folder = tmp_path / "dictionary"
    fixture_resources(folder)
    report = hidden(tmp_path, "tiny-lm", folder)
    assert report["qwen_acceptance"] is False


@pytest.mark.local_model
def test_local_qwen_hidden(tmp_path):
    model, dictionary = os.environ.get("CALIPER_QWEN_MODEL"), os.environ.get("CALIPER_QWEN_DICTIONARY")
    if not model or not dictionary:
        pytest.skip("Set both explicit local Qwen resource paths to opt in.")
    assert not os.environ.get("CI"), "The local Qwen case must not run in CI."
    assert Path(model).is_dir() and Path(dictionary).is_dir()
    resources_and_device(fixture=False)
    assert hidden(tmp_path, model, dictionary)["qwen_acceptance"]
