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
from _checks import check, finish, math_checks, selected_feature, token_path, FIXTURE_PROMPT
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
    # This fixture has fewer than 256 edges. A change must not silently omit edges.
    assert data["next_node_offset"] is None and data["next_edge_offset"] is None
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
    driver.save(folder.parent / "fixture-math.json", result)
    yield result
    service.remove()
    index.close()


def test_actual_fixture_selection_and_graph(measured):
    graph = measured["graph"]
    assert len(graph["nodes"]) == 31
    assert measured["selected"]["id"] == "feature:0:30:138"
    assert graph["completeness"]["max_absolute_residual"] <= 1e-9
    assert graph["feature_share"] >= .30
    assert token_path(graph)


def test_actual_fixture_gates_keep_semantics_separate(measured):
    gates = math_checks(measured["graph"], measured["examples"], measured["before"], measured["after"],
                        measured["validation"], 0, 0, fixture=True)
    # Zero durations here exercise the pure gate function. They are not timing evidence.
    by_name = {g["name"]: g for g in gates}
    assert all(g["passed"] for g in gates if g["required"])
    assert by_name["examples"]["passed"]
    assert by_name["fixture_output_effect"]["passed"]
    assert not by_name["steering"]["required"]
    assert by_name["validation"]["required"]
    semantic = math_checks(measured["graph"], measured["examples"], measured["before"], measured["after"],
                           measured["validation"], 21, 2, fixture=False)
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
                    measured["validation"], 0, 0, fixture=True)


def test_finish_refuses_missing_evidence():
    result = {"mode": "fixture", "checks": []}
    assert finish(result) == 1
    assert result["qwen_acceptance"] is False


def test_visible_result_keeps_pixel_evidence_optional():
    # This pure contract case is not a measured application acceptance result.
    result = {"mode": "qwen", "pixel_evidence_collected": False,
              "checks": [check("hidden_capture_evidence", False, "not collected", "hidden only", required=False)]}
    assert finish(result) == 0
    assert result["qwen_acceptance"] is False
    result["checks"][0]["required"] = True
    assert finish(result) == 1


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
        browser = subprocess.check_output(["node", "-e", "process.stdout.write(require('@playwright/test').chromium.executablePath())"],
                                          cwd=ROOT / "web", text=True)
        assert Path(browser).is_file(), "The local Chromium executable is absent."
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
