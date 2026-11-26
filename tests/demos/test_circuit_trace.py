"""Circuit math checks and the real hidden MCP acceptance driver."""
from __future__ import annotations

import json
import math
from copy import deepcopy
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
from _checks import (bar_direction_check, check, finish, math_checks, program_budget_checks,
                     secondary_selection, selected_feature, token_path, FIXTURE_PROMPT)
from _mcp import Failure
from _qualification import dictionary_qualification


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
    with owner.lock:
        owner._apply_interventions()
    after = output.compare(FIXTURE_PROMPT)
    service.clear()
    with owner.lock:
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
    assert measured["selected"]["id"] == "feature:0:30:216"
    assert selected_feature(graph, rule="influence")["id"] == "feature:0:30:138"
    assert graph["completeness"]["max_absolute_residual"] <= 1e-9
    assert graph["feature_share"] >= .30
    assert token_path(graph)


def test_actual_fixture_gates_keep_semantics_separate(measured):
    gates = math_checks(measured["graph"], measured["examples"], measured["before"], measured["after"],
                        measured["validation"], 0, 0, fixture=True, argmax_token_ids=measured["argmax_token_ids"],
                        predicted_ablation_effect=-measured["selected"]["effect"])
    # Zero durations here exercise the pure gate function. They are not timing evidence.
    by_name = {g["name"]: g for g in gates}
    assert all(g["passed"] for g in gates if g["required"])
    assert by_name["examples"]["passed"]
    assert by_name["fixture_output_effect"]["passed"]
    assert not by_name["steering_change_matches_prediction"]["required"]
    assert by_name["validation"]["required"]
    semantic = math_checks(measured["graph"], measured["examples"], measured["before"], measured["after"],
                           measured["validation"], 21, 2, fixture=False, argmax_token_ids=measured["argmax_token_ids"],
                           predicted_ablation_effect=-measured["selected"]["effect"])
    by_name = {g["name"]: g for g in semantic}
    assert by_name["steering_change_matches_prediction"]["required"] and by_name["validation"]["required"]
    assert by_name["trace_time"]["passed"] and by_name["trace_time"]["measured"] == 21
    assert by_name["steering_time"]["passed"] and by_name["steering_time"]["measured"] == 2


def test_missing_feature_is_a_failure(measured):
    graph = dict(measured["graph"], nodes=[n for n in measured["graph"]["nodes"] if n["kind"] != "feature"])
    with pytest.raises(Failure, match="no selectable feature"):
        selected_feature(graph)


def test_selection_uses_top_logit_effect_and_keeps_original_measurement():
    nodes = [dict(kind="feature", layer=layer, position=0, feature=index, index=index,
                  influence=influence, effect=effect)
             for index, (layer, influence, effect) in enumerate([(0, .8, .3), (2, .2, -4.6), (3, .9, 9)])]
    graph = {"nodes": nodes, "counts": {"layers": 4}}
    assert selected_feature(graph) == nodes[1]
    assert selected_feature(graph, rule="influence") == nodes[0]
    validation = {"measurements": [dict(layer=0, position=0, feature=0, predicted=-.3, measured=.0035)]}
    record = secondary_selection(graph, validation)
    assert record["feature"] == nodes[0]
    assert record["predicted_ablation_effect"] == -.3
    assert record["measured_ablation_effect"] == .0035
    assert record["measurement_source"] == "trace.validate"
    assert secondary_selection(graph)["measured_ablation_effect"] is None


@pytest.fixture
def secondary_case():
    """Place ten final-layer features above the influence-selected feature."""
    nodes = [dict(kind="feature", layer=3, position=0, feature=index, index=index,
                  influence=1., effect=1.) for index in range(10)]
    nodes.append(dict(kind="feature", layer=1, position=2, feature=10, index=10,
                      influence=.5, effect=-.3))
    rows = [dict(layer=node["layer"], position=node["position"], feature=node["feature"],
                 predicted=-node["effect"], measured=-node["effect"] / 2, sign_match=True) for node in nodes]
    return {"nodes": nodes, "counts": {"layers": 4}}, {
        "count": 10, "correct_signs": 10, "spearman": .8,
        "measurements": rows[:10], "secondary_measurement": rows[10]}


def test_secondary_selection_uses_measurement_outside_scored_ten(secondary_case):
    graph, validation = secondary_case
    original = deepcopy(validation)
    record = secondary_selection(graph, validation)
    assert record["feature"] == graph["nodes"][10]
    assert record["predicted_ablation_effect"] == .3
    assert record["measured_ablation_effect"] == .15
    assert record["measurement_source"] == "trace.validate"
    assert validation == original and len(validation["measurements"]) == 10


def test_secondary_selection_accepts_reused_measurement(secondary_case):
    graph, validation = secondary_case
    graph["nodes"][0]["layer"] = 0
    validation["measurements"][0]["layer"] = 0
    validation["secondary_measurement"] = validation["measurements"][0]
    record = secondary_selection(graph, validation)
    assert record["feature"] == graph["nodes"][0]
    assert record["measured_ablation_effect"] == -.5
    assert record["measurement_source"] == "trace.validate"


@pytest.mark.parametrize("fault", ["layer", "position", "feature", "missing_identity", "null"])
def test_secondary_selection_rejects_mismatched_explicit_identity(secondary_case, fault):
    graph, validation = secondary_case
    # A legacy match must not hide an invalid explicit measurement.
    validation["measurements"][0] = deepcopy(validation["secondary_measurement"])
    if fault == "null":
        validation["secondary_measurement"] = None
    elif fault == "missing_identity":
        validation["secondary_measurement"].pop("position")
    else:
        validation["secondary_measurement"][fault] += 1
    with pytest.raises(Failure, match="secondary measurement does not match"):
        secondary_selection(graph, validation)


def test_secondary_selection_preserves_legacy_missing_measurement(secondary_case):
    graph, validation = secondary_case
    validation.pop("secondary_measurement")
    original = deepcopy(validation)
    record = secondary_selection(graph, validation)
    assert record["feature"] == graph["nodes"][10]
    assert record["measured_ablation_effect"] is None
    assert record["measurement_source"] is None
    assert validation == original


def test_missing_top_bar_is_a_failure(measured):
    top = max(measured["before"], key=lambda r: r["baseline"])["token"]
    after = [r for r in measured["after"] if r["token"] != top]
    with pytest.raises(Failure, match="omit the baseline"):
        math_checks(measured["graph"], measured["examples"], measured["before"], after,
                    measured["validation"], 0, 0, fixture=True, argmax_token_ids=measured["argmax_token_ids"],
                    predicted_ablation_effect=-measured["selected"]["effect"])


def test_finish_refuses_missing_evidence():
    result = {"mode": "fixture", "completed": True, "checks": []}
    assert finish(result) == 1
    assert result["accepted_replication"] is False


def test_parity_retains_the_existing_edge_side_tolerance():
    import numpy as np
    from parity import compare
    bands = np.arange(512, dtype=np.uint32) // 16
    app = np.zeros((8, 512, 4), dtype=np.uint8)
    app[:, :, :3] = ((bands % 2) * 255)[None, :, None]
    app[:, :, 3] = 255
    ids = np.zeros((8, 512, 2), dtype=np.uint32)
    ids[:, :, 0], ids[:, :, 1] = 1, bands
    web, web_ids = np.roll(app, 1, axis=1), np.roll(ids, 1, axis=1)
    # The review example changes 6.25 percent of pixels to an adjacent edge color.
    assert np.any(app != web, axis=2).mean() == .0625
    metrics = compare(app, ids, web, web_ids)
    assert metrics["interior_over"] == metrics["interior_id_mismatch"] == metrics["edge_share"] == 0


def test_parity_rejects_colors_with_no_edge_side_match():
    import numpy as np
    from parity import compare
    app = np.zeros((8, 8, 4), dtype=np.uint8)
    web = np.full_like(app, 255)
    ids = np.zeros((8, 8, 2), dtype=np.uint32)
    ids[:, :, 1] = np.indices((8, 8)).sum(axis=0) % 2
    assert compare(app, ids, web, ids)["edge_share"] == 1


def test_parity_keeps_interior_color_and_identity_checks():
    import numpy as np
    from parity import compare
    app = np.zeros((8, 8, 4), dtype=np.uint8)
    ids = np.zeros((8, 8, 2), dtype=np.uint32)
    metrics = compare(app, ids, app + 3, ids + 1)
    assert metrics["interior_over"] == metrics["interior_id_mismatch"] == 64


@pytest.mark.parametrize("parent_temp", ["absent", "external", "mixed"])
def test_parity_children_receive_private_temporary_environment(tmp_path, monkeypatch, parent_temp):
    from unittest.mock import Mock
    import parity
    monkeypatch.chdir(tmp_path)
    out = Path("parity")
    out.mkdir()
    for index, key in enumerate(("TMPDIR", "TMP", "TEMP")):
        if parent_temp == "absent" or (parent_temp == "mixed" and key == "TMPDIR"):
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(tmp_path / ("external-" + str(index))))
    for key in ("CALIPER_AGENT_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "CALIPER_HIDDEN", "CALIPER_TEST_NO_WINDOW", "CI"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("CALIPER_TEST_FORCE_NO_GPU", "1")
    monkeypatch.setenv("PARITY_INHERITED_MARKER", "retained")
    parent = dict(os.environ)
    calls = []
    def run_owned(command, *, stdout, timeout, env):
        calls.append((command, timeout, dict(env)))
        assert not stdout.closed
        assert env is not os.environ and dict(os.environ) == parent
        temporary = out.resolve() / "temp"
        assert temporary.is_dir()
        assert all(env[key] == str(temporary) for key in ("TMPDIR", "TMP", "TEMP"))
        assert all(env[key] == "1" for key in ("CALIPER_AGENT_OFFLINE", "HF_HUB_OFFLINE",
                   "TRANSFORMERS_OFFLINE", "CALIPER_HIDDEN", "CALIPER_TEST_NO_WINDOW", "CI",
                   "CALIPER_TEST_FORCE_NO_GPU"))
        assert env["PARITY_INHERITED_MARKER"] == "retained"
        if len(calls) == 2:
            raise Failure("both child environments checked")
    monkeypatch.setattr(parity, "run_owned", run_owned)
    native = tmp_path / "native-parity"
    monkeypatch.setattr(driver, "executable", Mock(return_value=native))
    monkeypatch.setattr(driver, "memory_bytes", Mock(return_value=1024))
    with pytest.raises(Failure, match="both child environments checked"):
        parity.run_parity(out)
    assert len(calls) == 2
    assert str(native) in calls[0][0] and calls[0][1] == 90
    assert "node" in calls[1][0] and calls[1][1] == 180
    assert any(arg.endswith("browser.mjs") for arg in calls[1][0])
    assert calls[0][2] == calls[1][2]
    assert dict(os.environ) == parent


def inference_fixture(prompt="A neutral fixture prompt"):
    from _inference_evidence import text_hash
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


def test_semantic_token_change_reports_zero_probability_change(measured):
    old = measured["argmax_token_ids"][0]
    other = next(row["token"] for row in measured["after"] if row["token"] != old)
    gates = math_checks(measured["graph"], measured["examples"], measured["before"], measured["before"],
                        measured["validation"], 0, 0, fixture=False, argmax_token_ids=[old, other],
                        predicted_ablation_effect=-measured["selected"]["effect"])
    steering = next(g for g in gates if g["name"] == "steering_change_matches_prediction")
    assert steering["passed"] and steering["measured"]["probability_change"] == 0
    assert not steering["measured"]["direction_matches"]


@pytest.fixture
def steering_case():
    """Supply CPU gate inputs without a model or application process."""
    graph = {"counts": {"layers": 3}, "completeness": {"max_absolute_residual": 0.}, "feature_share": .3,
             "nodes": [{"kind": "token", "index": 0},
                       {"kind": "feature", "index": 1, "layer": 0, "position": 0, "feature": 4,
                        "effect": -4.64, "influence": .2},
                       {"kind": "feature", "index": 2, "layer": 1, "position": 0, "feature": 5,
                        "effect": 1., "influence": .8},
                       {"kind": "logit", "index": 3, "probability": .5}],
             "edges": [{"source": 0, "target": 3, "weight": 1.}]}
    examples = {"examples": [{"example_id": i, "activation": 16 - i} for i in range(16)]}
    before = [{"token": token, "baseline": .5 if token == 1 else .05, "probability": .5 if token == 1 else .05}
              for token in range(1, 11)]
    return dict(graph=graph, examples=examples, before=before, after=deepcopy(before),
                validation={"count": 10, "correct_signs": 8, "spearman": .6},
                trace_seconds=0, steering_seconds=0, fixture=False, argmax_token_ids=[1, 1],
                predicted_ablation_effect=4.64)


@pytest.mark.parametrize("prediction", [0., -0., float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("probability", [.1, .9])
@pytest.mark.parametrize("new_top", [1, 2])
def test_steering_requires_finite_prediction(steering_case, prediction, probability, new_top):
    steering_case.update(predicted_ablation_effect=prediction, argmax_token_ids=[1, new_top])
    steering_case["after"][0]["probability"] = probability
    gates = math_checks(**steering_case)
    steering = next(g for g in gates if g["name"] == "steering_change_matches_prediction")
    pixels = bar_direction_check(50, int(probability * 100), predicted_ablation_effect=prediction, fixture=False)
    for gate in (steering, pixels):
        assert gate["required"] and gate["passed"] is math.isfinite(prediction)
        assert gate["measured"]["predicted_direction"] is None
        assert gate["measured"]["prediction_finite"] is math.isfinite(prediction)
        assert not gate["measured"]["direction_matches"]
        json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("before,after,prediction,passed", [
    (.05, .1, 1., True), (.05, 0., -1., True),
    (.05, .099, 1., True), (.05, .001, -1., True),
])
def test_steering_probability_change_has_no_magnitude_gate(steering_case, before, after, prediction, passed):
    steering_case["before"][0]["baseline"] = before
    steering_case["after"][0]["probability"] = after
    steering_case["predicted_ablation_effect"] = prediction
    steering = next(g for g in math_checks(**steering_case) if g["name"] == "steering_change_matches_prediction")
    assert steering["passed"] is passed


@pytest.mark.parametrize("missing", ["before", "after"])
def test_steering_requires_exact_original_token(steering_case, missing):
    steering_case[missing] = steering_case[missing][1:]
    steering_case["argmax_token_ids"] = [1, 2]
    with pytest.raises(Failure, match="omit.*baseline.*top token"):
        math_checks(**steering_case)


@pytest.mark.parametrize("count,signs,spearman,passed", [
    (10, 8, .6, True), (10, 7, 1., True), (10, 10, .599, True), (9, 9, 1., False),
    (10, 0, -1., True), (10, 11, 0., False), (10, True, 0., False), (10, 0, None, False),
    (10, 0, float("nan"), False), (10, 0, float("inf"), False),
])
def test_validation_requires_complete_finite_measurements(steering_case, count, signs, spearman, passed):
    steering_case["validation"] = dict(count=count, correct_signs=signs, spearman=spearman)
    gate = next(g for g in math_checks(**steering_case) if g["name"] == "validation")
    assert gate["required"] and gate["passed"] is passed


def synthetic_capture(tmp_path, *, missing_panel=None, text_fault=None, output_bars=None):
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
        elif output_bars is None:
            source[8:24, 10:18] = [30, 180, 220, 255]
            ids[8:24, 10:18] = [pick, 0]
        else:
            for row in output_bars:
                x = 2 + row["index"] * 3
                height = round(row["probability"] * 32)
                source[28 - height:28, x:x + 2] = [30, 180, 220, 255]
                ids[28 - height:28, x:x + 2] = [pick, row["index"]]
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
    gates, measured = capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars, selected={"label": "B"}, baseline_top_token=7)
    return {gate["name"]: gate for gate in gates}, measured


def test_composition_oracle_handles_dpi_clip_scaled_targets_and_glyphs(tmp_path):
    gates, measured = synthetic_gates(tmp_path)
    # This small composition fixture does not claim a full instrument layout.
    names = {"capture_generation", "rendered_nodes", "host_labels", "example_page", "bar_height"}
    names.update(name for name in gates if name.endswith(("_composition", "_resource_generation")))
    assert all(gates[name]["passed"] for name in names), gates
    assert measured["height"] == 32 and measured["expected_height"] == pytest.approx(32)
    for name in ("graph", "examples", "output"):
        evidence = gates[name + "_composition"]["measured"]
        assert evidence["pixels"] == 60 * 62
        assert evidence["source_items"] == evidence["window_items"]


@pytest.mark.parametrize("offset", [-24., -2.01, 0., 2.01, 24.])
def test_bar_height_preserves_large_finite_discrepancies(tmp_path, offset):
    from _checks import capture_checks
    metadata, graph, bars = synthetic_capture(tmp_path)
    bars[0]["probability"] = (32 + offset) / 64
    gates, measured = capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars, selected={"label": "B"}, baseline_top_token=7)
    gate = next(g for g in gates if g["name"] == "bar_height")
    assert gate["passed"]
    assert measured["height"] == 32
    assert gate["measured"]["absolute_difference"] == pytest.approx(abs(offset))


@pytest.mark.parametrize("token", [None, True, -1, "7", 8])
def test_capture_requires_original_top_token_identity(tmp_path, token):
    from _checks import capture_checks
    metadata, graph, bars = synthetic_capture(tmp_path)
    with pytest.raises(Failure, match="original baseline top token"):
        capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars,
                       selected={"label": "B"}, baseline_top_token=token)


@pytest.mark.parametrize("panel", ["graph", "examples", "output"])
def test_missing_window_composition_fails_with_valid_offscreen_targets(tmp_path, panel):
    gates, _ = synthetic_gates(tmp_path, missing_panel=panel)
    assert gates["capture_generation"]["passed"]  # Equal copies do not prove composition.
    assert gates["rendered_nodes"]["passed"] and gates["example_page"]["passed"]
    assert not gates[panel + "_composition"]["passed"]


@pytest.mark.parametrize("fault", ["missing", "selected_missing", "wrong", "variation"])
def test_text_inventory_preserves_missing_glyph_refusals_and_pixel_discrepancies(tmp_path, fault):
    gates, _ = synthetic_gates(tmp_path, text_fault=fault)
    assert gates["capture_generation"]["passed"]
    if fault in ("wrong", "variation"):
        assert max(row["maximum_difference"] for row in gates["host_labels"]["measured"]["glyphs"]) > 2
        assert gates["graph_composition"]["measured"]["reference_equivalence"] == "not_established"
    else:
        assert not gates["host_labels"]["passed"]


def test_missing_panel_placement_is_a_failure(tmp_path):
    from _checks import capture_checks
    metadata, graph, bars = synthetic_capture(tmp_path)
    del metadata["panels"]["graph"]["composition"]
    with pytest.raises(Failure, match="composition evidence"):
        capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars, selected={"label": "B"}, baseline_top_token=7)


def test_text_oracle_requires_each_repeated_graph_label():
    from _composition import required_text
    assert not required_text([{"text": "A", "passed": True}], ["A", "A"])
    assert required_text([{"text": "A", "passed": True}] * 2, ["A", "A"])


def fitted_label(original, visible, width):
    """Supply font measurements for a synthetic font with four-pixel advances."""
    candidates = ["…"] + [original[:i] + "…" for i in range(1, len(original)) if original[i - 1] != " "]
    return {"original_text": original, "text": visible,
            "fit": None if width is None else {"width": width, "full_width": 4 * len(original),
                                               "prefix_widths": [4 * len(text) for text in candidates]}}


def fitted_capture(tmp_path):
    """Paint complete synthetic glyphs for two repeated tokens and a selection."""
    import numpy as np
    from PIL import Image
    metadata, graph, bars = synthetic_capture(tmp_path)
    panel = metadata["panels"]["graph"]
    records = [fitted_label("[empty token 256]", "[emp…", 20),
               fitted_label("[whitespace token 32]", "[whit…", 24),
               fitted_label("[whitespace token 32]", "[whit…", 24),
               fitted_label("Output: café λ", "Out…", 16)]
    graph["tokens"] = [{"text": row["original_text"]} for row in records[:3]]
    selected = {"label": records[-1]["original_text"]}
    # Each character has its own small bitmap. These pixels do not come from
    # the composition oracle or from its provenance validation function.
    characters = sorted(set("".join(row["text"] for row in records)))
    atlas = np.full((7, len(characters) * 4 + 1, 4), 255, dtype=np.uint8)
    atlas[:, :, 3] = 0
    bitmaps = {}
    for index, character in enumerate(characters):
        bitmap = np.zeros((5, 3), dtype=bool)
        bitmap[0, 1] = bitmap[4, 1] = True
        for bit in range(6):
            bitmap[1 + bit // 3, bit % 3] = bool((index + 1) & (1 << bit))
        bitmaps[character] = bitmap
        atlas[1:6, 1 + 4 * index:4 + 4 * index, 3] = bitmap * 255
    atlas.tofile(tmp_path / "atlas.rgba")
    metadata["font_atlas"].update(width=atlas.shape[1], height=atlas.shape[0])
    window = np.fromfile(tmp_path / "window.rgba", dtype=np.uint8).reshape(88, 216, 4)
    source = np.fromfile(tmp_path / "graph.rgba", dtype=np.uint8).reshape(32, 32, 4)
    painted = np.asarray(Image.fromarray(source).resize((64, 64), Image.Resampling.BILINEAR))
    window[10:72, 8:68] = painted[2:, 4:]
    def points(rect):
        return [rect[0] / 2 + 10, rect[1] / 2 + 20, rect[2] / 2 + 10, rect[3] / 2 + 20]
    for row, record in enumerate(records):
        x, y = 18, 40 + row * 8
        record.update(box=points([x, y, x + len(record["text"]) * 4 + 1, y + 7]), quads=[])
        # Match the real pass order: the complete shadow, then the complete ink.
        for offset, color in ((2, [0, 0, 0, 255]), (0, [240, 240, 240, 255])):
            for column, character in enumerate(record["text"]):
                left = 1 + 4 * characters.index(character)
                gx, gy = x + column * 4 + offset, y + offset
                record["quads"].append({"rect": points([gx, gy, gx + 3, gy + 5]),
                    "clip": panel["composition"]["clip"], "tint": color,
                    "uv": [left / atlas.shape[1], 1 / 7, (left + 3) / atlas.shape[1], 6 / 7]})
                window[gy:gy + 5, gx:gx + 3][bitmaps[character]] = color
    panel["labels"] = records
    window.tofile(tmp_path / "window.rgba")
    Image.fromarray(window).save(tmp_path / "capture.png")
    return metadata, graph, bars, selected


def fitted_gates(tmp_path, metadata, graph, bars, selected):
    from _checks import capture_checks
    gates, _ = capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars,
                              selected=selected, baseline_top_token=7)
    return {gate["name"]: gate for gate in gates}


def test_text_oracle_proves_exact_source_fits_with_complete_glyphs(tmp_path):
    case = fitted_capture(tmp_path)
    gates = fitted_gates(tmp_path, *case)
    for name in ("host_labels", "graph_composition", "graph_label_spacing", "rendered_nodes"):
        assert gates[name]["passed"], gates[name]
    proof = gates["host_labels"]["measured"]
    assert proof["required"] == ["[empty token 256]", "[whitespace token 32]", "[whitespace token 32]", "Output: café λ"]
    assert all(row["passed"] and row["complete"] and row["ink_pixels"] > 0 for row in proof["glyphs"])


@pytest.mark.parametrize("fault", ["source", "selected_source", "visible", "short_prefix", "missing_fit",
                                    "missing_original", "empty", "ellipsis_only", "duplicate", "selected_identity"])
def test_text_oracle_rejects_unproved_fits_and_source_changes(tmp_path, fault):
    metadata, graph, bars, selected = fitted_capture(tmp_path)
    labels = metadata["panels"]["graph"]["labels"]
    if fault == "source": labels[0]["original_text"] = "[empty token 257]"
    elif fault == "selected_source": labels[-1]["original_text"] = "Output: café μ"
    elif fault == "visible": labels[0]["text"] = "[xyz…"
    elif fault == "short_prefix": labels[0]["text"] = "[e…"
    elif fault == "missing_fit": labels[0].pop("fit")
    elif fault == "missing_original": labels[0].pop("original_text")
    elif fault == "empty": labels[0]["text"] = ""
    elif fault == "ellipsis_only": labels[0].update(text="…", fit={**labels[0]["fit"], "width": 4})
    elif fault == "duplicate": labels.pop(2)
    else: selected["label"] = "Output: café μ"
    gates = fitted_gates(tmp_path, metadata, graph, bars, selected)
    assert not gates["host_labels"]["passed"]
    if fault != "duplicate":
        # Correct reference pixels alone cannot prove a different source or fit.
        assert gates["graph_composition"]["passed"]
        assert all(row["passed"] for row in gates["host_labels"]["measured"]["glyphs"])


@pytest.mark.parametrize("fault", ["wrong_glyph", "partial_clip", "no_quads", "overlap"])
def test_fitted_provenance_preserves_independent_pixel_and_overlap_checks(tmp_path, fault):
    import numpy as np
    from PIL import Image
    metadata, graph, bars, selected = fitted_capture(tmp_path)
    labels = metadata["panels"]["graph"]["labels"]
    if fault == "wrong_glyph":
        window = np.fromfile(tmp_path / "window.rgba", dtype=np.uint8).reshape(88, 216, 4)
        window[40:45, 18:21] = [20, 24, 28, 255]
        window.tofile(tmp_path / "window.rgba")
        Image.fromarray(window).save(tmp_path / "capture.png")
    elif fault == "partial_clip":
        labels[0]["quads"][0]["clip"] = [0, 0, 20.5, 100]
    elif fault == "no_quads": labels[0]["quads"] = []
    else: labels[1]["box"] = labels[0]["box"].copy()
    gates = fitted_gates(tmp_path, metadata, graph, bars, selected)
    assert gates["capture_generation"]["passed"]
    if fault == "wrong_glyph":
        assert max(row["maximum_difference"] for row in gates["host_labels"]["measured"]["glyphs"]) > 2
    else:
        assert not gates["graph_label_spacing" if fault == "overlap" else "host_labels"]["passed"]


@pytest.mark.parametrize("original,visible,width", [
    ("é你🙂 suffix", "é你…", 12), ("é你🙂 suffix", "é你🙂…", 16),
    ("word suffix", "word…", 24), ("é你🙂", "é你🙂", 12), ("literal…", "literal…", None),
])
def test_text_provenance_preserves_unicode_boundaries_spaces_and_literal_ellipsis(original, visible, width):
    from _composition import required_text
    row = dict(fitted_label(original, visible, width), passed=True)
    assert required_text([row], [original])
    assert not required_text([row], [original, original])
    assert not required_text([dict(row, passed=False)], [original])


@pytest.mark.parametrize("fault", ["width", "full_width", "prefix_widths", "nan", "negative", "bool", "surrogate"])
def test_text_provenance_rejects_incomplete_or_invalid_font_measurements(fault):
    from _composition import required_text
    row = dict(fitted_label("é你🙂 suffix", "é你…", 12), passed=True)
    if fault in ("width", "full_width", "prefix_widths"): row["fit"].pop(fault)
    elif fault == "nan": row["fit"]["prefix_widths"][0] = float("nan")
    elif fault == "negative": row["fit"]["width"] = -1
    elif fault == "bool": row["fit"]["width"] = True
    else: row["original_text"] = "\ud800 suffix"
    assert not required_text([row], ["é你🙂 suffix"])


def test_text_provenance_keeps_colliding_prefixes_and_selected_duplicates_distinct():
    from _composition import required_text
    first, second = "[empty token 256]", "[empty token 257]"
    rows = [dict(fitted_label(original, "[emp…", 20), passed=True) for original in (first, second)]
    assert required_text(rows, [first, second])
    assert not required_text(rows[:1], [first, second])
    assert not required_text(rows, [first, second, second])
    assert required_text([*rows, rows[1]], [first, second, second])


def test_text_provenance_requires_the_measured_longest_fit_with_proportional_widths():
    from _composition import required_text
    row = {"text": "Wi…", "original_text": "Wide label", "passed": True,
           "fit": {"width": 13, "full_width": 36, "prefix_widths": [3, 11, 13, 18, 22, 27, 30, 34, 36]}}
    assert required_text([row], ["Wide label"])
    for visible in ("W…", "Wid…", "Wide…", "Wide label"):
        assert not required_text([dict(row, text=visible)], ["Wide label"])


@pytest.mark.parametrize("fault", ["zero_width", "missing_measurement", "extra_measurement", "wrong_width",
                                    "wrong_full_width", "no_fit", "null_source", "source_bound", "embedded_null"])
def test_text_provenance_rejects_inconsistent_fit_evidence(fault):
    from _composition import required_text
    row = dict(fitted_label("[empty token 256]", "[emp…", 20), passed=True)
    if fault == "zero_width": row["fit"]["width"] = 0
    elif fault == "missing_measurement": row["fit"]["prefix_widths"].pop()
    elif fault == "extra_measurement": row["fit"]["prefix_widths"].append(100)
    elif fault == "wrong_width": row["fit"]["width"] = 24
    elif fault == "wrong_full_width": row["fit"]["full_width"] = 16
    elif fault == "no_fit": row["fit"] = None
    elif fault == "null_source": row["original_text"] = None
    elif fault == "source_bound": row["original_text"] = "x" * 1025
    else: row["original_text"] += "\0"
    assert not required_text([row], ["[empty token 256]"])


def test_text_provenance_legacy_capture_cannot_attest_a_fitted_source():
    from _composition import required_text
    assert not required_text([{"text": "[emp…", "passed": True}], ["[empty token 256]"])
    assert required_text([{"text": "literal…", "passed": True}], ["literal…"])


def test_text_oracle_rejects_partial_glyphs_under_the_actual_clip(tmp_path):
    from _checks import capture_checks
    metadata, graph, bars = synthetic_capture(tmp_path)
    label = metadata["panels"]["graph"]["labels"][0]
    for quad in label["quads"]:
        quad["clip"][2] = quad["rect"][0] + .5
    gates, _ = capture_checks(tmp_path, metadata, tmp_path / "capture.png", graph, bars, selected={"label": "B"}, baseline_top_token=7)
    text = next(gate for gate in gates if gate["name"] == "host_labels")
    assert not text["passed"]
    assert not text["measured"]["glyphs"][0]["complete"]


def test_visible_result_keeps_pixel_evidence_optional():
    # This pure contract case is not a measured application acceptance result.
    result = {"mode": "fixture", "completed": True, "pixel_evidence_collected": False,
              "checks": [check("hidden_capture_evidence", False, "not collected", "hidden only", required=False)]}
    assert finish(result) == 0
    assert result["accepted_replication"] is False
    result["checks"][0]["required"] = True
    assert finish(result) == 1


@pytest.mark.parametrize("change,passed,evidence", [
    ({}, True, "measured"),
    ({"last_pass_ms": None}, True, "unavailable"),
    ({"last_pass_ms": 50.01}, True, "measured"),
    ({"last_pass_ms": float("inf")}, False, "unverified"),
    ({"last_pass_ms": True}, False, "unverified"),
    ({"last_pass_ms": -1}, False, "unverified"),
    ({"memory_bytes": (64 << 20) + 1}, False, "measured"),
    ({"frame_ms": None}, True, "unavailable"),
    ({"frame_ms": 4.01}, True, "measured"),
    ({"frame_ms": "4"}, False, "unverified"),
])
def test_program_measurements_retain_memory_caps(change, passed, evidence):
    # This pure validation input is not saved as acceptance evidence.
    program = {"id": "attribution", "state": "running", "last_step": 1, "last_pass_ms": 50, "memory_bytes": 64 << 20,
               "memory_limit_bytes": 64 << 20, "draw": {"state": "running", "frames": 1, "frame_ms": 4, "frame_limit_ms": 4,
               "memory_bytes": 64 << 20, "memory_limit_bytes": 64 << 20}}
    for key, value in change.items():
        (program["draw"] if key == "frame_ms" else program)[key] = value
    snapshot = {"status": {"programs": [program]}, "expected_steps": {"attribution": 1},
                "pieces": {"attribution": {"derive": {"time_ms": 50, "memory_mb": 64},
                                           "draw": {"frame_ms": 4, "memory_mb": 64}}}}
    gates = program_budget_checks(snapshot)
    assert all(gate["passed"] for gate in gates) is passed
    if evidence == "unavailable":
        unavailable = [gate for gate in gates if gate.get("evidence") == "unavailable"]
        assert len(unavailable) == 1 and unavailable[0]["measured"] is None and unavailable[0]["reason"]
    if change and not passed:
        failed = [gate for gate in gates if not gate["passed"]]
        assert len(failed) == 1 and failed[0]["evidence"] == evidence
        assert failed[0]["measured"] == next(iter(change.values()))
    result = {"mode": "fixture", "completed": True, "checks": gates}
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
            return {"step": 7}
        def ready_programs(self, names):
            calls.append(("ready_programs", names))
    session = Session()
    # This transport unit test writes no acceptance artifact.
    from unittest.mock import patch
    with patch.object(driver, "save"), patch.object(Session, "out", Path("unused"), create=True), \
            patch.object(driver, "programs") as programs:
        bridge = driver.Bridge(session)
        bridge.consume()
    programs.assert_called_once_with(session, examples=True)
    assert calls == [("selected_node", 10), ("feature.show", {"node": 10}),
                     ("feature.examples", {"node": 10, "limit": 16}),
                     ("state.set", {"id": "activation_max", "value": 2}),
                     ("ready_programs", ["example_tokens"]),
                     ("feature.examples", {"node": 10, "limit": 16})]
    assert bridge.last_selection == 10
    assert bridge.selection_step == 7


@pytest.mark.parametrize("kind", [0, 2, 3])
def test_bridge_clears_shared_state_for_non_feature_picks(tmp_path, kind, monkeypatch):
    from unittest.mock import Mock
    session = Mock(out=tmp_path)
    edit = Mock()
    monkeypatch.setattr(driver, "scene_programs", edit)
    bridge = driver.Bridge(session)
    bridge.last_selection, bridge.examples_present = 10, True
    bridge.handle({"kind": "pick", "panel": "graph", "fields": {"kind": kind, "index": 12}})
    bridge.handle({"kind": "run", "change": "param", "name": "selected_node", "value": -1})
    session.mcp.body.assert_any_call("state.set", id="selected_node", value=-1)
    session.parameter.assert_called_once_with("selected_node", -1)
    edit.assert_called_once_with(session, [], remove_examples=True)
    assert bridge.last_selection is None and bridge.examples_present is False


def test_bridge_creates_examples_after_an_early_refusal(tmp_path):
    from unittest.mock import Mock, patch
    session = Mock(out=tmp_path)
    session.parameter.return_value = {"step": 7}
    bridge = driver.Bridge(session)
    session.events.complete.side_effect = [Failure("feature unavailable"), {"maximum": 2},
                                           {"examples": []}, {"examples": []}]
    event = {"kind": "pick", "panel": "graph", "fields": {"kind": 1, "index": 10}}
    with pytest.raises(Failure, match="unavailable"):
        bridge.handle(event)
    with patch.object(driver, "programs") as programs:
        bridge.handle(event)
    programs.assert_called_once_with(session, examples=True)
    session.ready_programs.assert_called_once_with(["example_tokens"])
    assert bridge.examples_present is True and bridge.last_selection == 10


def test_visible_bridge_continues_after_first_event():
    from unittest.mock import Mock
    session = Mock(family="fixture")
    session.live.side_effect = [True, True, False]
    bridge = driver.Bridge(session)
    bridge.consume = Mock()
    bridge.keep_alive()
    assert session.events.wait.call_count == 2
    assert bridge.consume.call_count == 2


def test_bridge_reconciles_run_selection_without_parameter_feedback(tmp_path, monkeypatch):
    from unittest.mock import Mock, patch
    session = Mock(out=tmp_path)
    edit = Mock()
    monkeypatch.setattr(driver, "scene_programs", edit)
    bridge = driver.Bridge(session)
    bridge.last_selection = 10
    bridge.examples_present = True
    session.events.pending = [{"kind": "run", "change": "param", "name": "selected_node", "value": -1}]
    bridge.consume()
    session.mcp.body.assert_any_call("state.set", id="selected_node", value=-1)
    edit.assert_called_once_with(session, [], remove_examples=True)
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


@pytest.mark.parametrize("control,value", [("clamp", 2.), ("apply_clamp", None), ("reset", None)])
def test_bridge_leaves_clamp_controls_to_their_bindings(control, value):
    from unittest.mock import Mock
    session = Mock()
    session.events.pending = [{"kind": "control", "control": control, "value": value}]
    bridge = driver.Bridge(session)
    bridge.last_selection = 10
    bridge.consume()
    session.parameter.assert_not_called()
    session.mcp.body.assert_not_called()
    session.events.complete.assert_not_called()


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
    session.events.complete.assert_not_called()
    session.ready_programs.assert_called_once_with(["probabilities"])


@pytest.mark.parametrize("page", [0, 1])
def test_example_page_waits_for_page_identity_and_one_resource_step(page):
    from unittest.mock import Mock
    session = driver.Session.__new__(driver.Session)
    old = {"step": 8, "elements": [[i, (1 - page) * 8 + i, i, 25] for i in range(8)]}
    current = {"step": 9, "elements": [[i, page * 8 + i, i, 25] for i in range(8)]}
    latest = dict(current, step=10)
    cells = {"step": 10, "elements": [[0] * 8 for _ in range(200)]}
    session.read = Mock(side_effect=[old, current, cells, latest, cells])
    session.ready_programs = Mock()
    session.events = Mock()
    def until(description, read):
        assert read() is False
        assert read() is False
        return read()
    session.events.until.side_effect = until
    assert session.example_page(page, minimum_step=7) == (latest, cells)
    session.ready_programs.assert_called_once_with(["example_tokens"])
    assert session.read.call_count == 5
    assert all(call.kwargs["minimum_step"] >= 7 for call in session.read.call_args_list)


@pytest.mark.parametrize("status", ["ok", "error"])
def test_run_action_waits_for_its_fresh_completion(status):
    from unittest.mock import Mock
    session = driver.Session.__new__(driver.Session)
    session.run_id = "circuit-trace"
    session.mcp = Mock()
    session.events = driver.Events(session.mcp)
    stale = {"kind": "run", "change": "action", "name": "apply_clamp", "status": "ok"}
    unrelated = dict(stale, name="reset")
    completed = dict(stale, status=status, text="selected feature has no positive indexed maximum")
    batches = iter([[stale], [unrelated], [completed]])
    def body(method, **args):
        if method == "run.action":
            assert args == {"run": session.run_id, "name": "apply_clamp"}
            return {"name": "apply_clamp", "queued": True}
        assert method == "events.wait"
        return {"cursor": args["cursor"] + 1, "lost": False, "events": next(batches)}
    session.mcp.body.side_effect = body
    if status == "ok":
        assert session.action("apply_clamp") == completed
    else:
        with pytest.raises(Failure, match="no positive indexed maximum"):
            session.action("apply_clamp")
    assert session.events.pending == [stale, unrelated, completed]
    requests = [call for call in session.mcp.body.call_args_list if call.args[0] == "run.action"]
    assert len(requests) == 1


@pytest.fixture
def clamp_session():
    from unittest.mock import Mock
    prompt, _, status, _ = inference_fixture()
    state = status["inference"]
    binding = {key: state[key] for key in ("prompt_hash", "interventions_hash")}
    state["comparison"] = dict(binding, argmax_token_ids=[1, 2], completed_step=9)
    session = driver.Session.__new__(driver.Session)
    session.parameter = Mock()
    session.action = Mock()
    session.events = Mock()
    session.events.until.side_effect = lambda description, read: read()
    session.read = Mock(return_value={"step": 14, "elements": [
        [0, 0, 0, 0, .1, .1, token, token - 1] for token in range(1, 11)]})
    return session, prompt, status


@pytest.mark.parametrize("minimum_step", [10, 20])
def test_zero_clamp_waits_for_action_output_and_resource(clamp_session, minimum_step):
    session, prompt, prior = clamp_session
    completed = deepcopy(prior)
    completed["inference"]["comparison"]["completed_step"] = 14
    session.read.return_value["step"] = max(minimum_step, 14)
    session.events.complete.side_effect = [prior, prior, completed, completed]
    def until(description, read):
        if description == "slider output comparison":
            session.action.assert_not_called()
            return read()
        session.action.assert_called_once_with("apply_clamp")
        assert read() is False
        session.read.assert_not_called()
        return read()
    session.events.until.side_effect = until
    reply, evidence = session.ablate_selected(prompt, minimum_step=minimum_step)
    session.parameter.assert_called_once_with("clamp", 0.)
    session.action.assert_called_once_with("apply_clamp")
    session.read.assert_called_once_with("probabilities", "cells", minimum_step=max(minimum_step, 14))
    assert reply == session.read.return_value
    assert evidence == completed["inference"]["comparison"]


@pytest.mark.parametrize("fault", ["missing", "old_state"])
def test_zero_clamp_waits_for_slider_comparison_before_action(clamp_session, fault):
    session, prompt, prior = clamp_session
    pending = deepcopy(prior)
    if fault == "missing":
        pending["inference"].pop("comparison")
    else:
        pending["inference"]["comparison"]["interventions_hash"] = "f" * 64
    completed = deepcopy(prior)
    completed["inference"]["comparison"]["completed_step"] = 14
    session.events.complete.side_effect = [pending, prior, completed, completed]
    def until(description, read):
        if description == "slider output comparison":
            session.action.assert_not_called()
            assert read() is False
            session.action.assert_not_called()
            session.read.assert_not_called()
        else:
            session.action.assert_called_once_with("apply_clamp")
        return read()
    session.events.until.side_effect = until
    _, evidence = session.ablate_selected(prompt, minimum_step=10)
    session.action.assert_called_once_with("apply_clamp")
    assert evidence == completed["inference"]["comparison"]


def test_zero_clamp_refusal_stops_before_output_read(clamp_session):
    session, prompt, prior = clamp_session
    session.events.complete.return_value = prior
    session.action.side_effect = Failure("clamp requires a selected feature")
    with pytest.raises(Failure, match="requires a selected feature"):
        session.ablate_selected(prompt, minimum_step=10)
    session.parameter.assert_called_once_with("clamp", 0.)
    session.action.assert_called_once_with("apply_clamp")
    session.read.assert_not_called()


@pytest.mark.parametrize("fault", ["missing", "old_state", "wrong_prompt", "missing_step"])
def test_zero_clamp_rejects_unbound_completion(clamp_session, fault):
    session, prompt, prior = clamp_session
    completed = deepcopy(prior)
    state = completed["inference"]
    state["comparison"]["completed_step"] = 14
    if fault == "missing": state.pop("comparison")
    elif fault == "old_state": state["comparison"]["interventions_hash"] = "f" * 64
    elif fault == "wrong_prompt": state["prompt_hash"] = "f" * 64
    elif fault == "missing_step": state["comparison"].pop("completed_step")
    session.events.complete.side_effect = [prior, completed]
    session.events.until.side_effect = lambda description, read: read()
    with pytest.raises(Failure):
        session.ablate_selected(prompt, minimum_step=10)
    session.read.assert_not_called()


def test_zero_clamp_rejects_changed_comparison_after_publication(clamp_session):
    session, prompt, prior = clamp_session
    completed = deepcopy(prior)
    completed["inference"]["comparison"]["completed_step"] = 14
    changed = deepcopy(completed)
    changed["inference"]["comparison"]["completed_step"] = 15
    session.events.complete.side_effect = [prior, completed, changed]
    session.events.until.side_effect = lambda description, read: read()
    with pytest.raises(Failure, match="changed during resource publication"):
        session.ablate_selected(prompt, minimum_step=10)


@pytest.mark.parametrize("selection_changed", [False, True])
def test_acceptance_uses_explicit_ablation_and_retains_selection_gate(tmp_path, monkeypatch, selection_changed):
    from types import SimpleNamespace
    from unittest.mock import Mock
    prompt, _, status, _ = inference_fixture()
    binding = {key: status["inference"][key] for key in ("prompt_hash", "interventions_hash")}
    prior = dict(binding, argmax_token_ids=[1, 2], completed_step=9)
    after = dict(prior, completed_step=14)
    status.update(state="done", counts={"nodes": 1})
    status["inference"]["comparison"] = prior
    graph = {"nodes": [{"kind": "feature", "layer": 0, "influence": 1, "index": 0, "effect": 1}], "counts": {"layers": 2}}
    before_reply = {"step": 9, "elements": [
        [0, 0, 0, 0, .1, .2 if selection_changed else .1, token, token - 1] for token in range(1, 11)]}
    after_reply = dict(before_reply, step=14)
    session = Mock(out=tmp_path, run_id="circuit-trace", program_evidence={"before": {}})
    session.read.return_value = before_reply
    session.example_page.return_value = before_reply, before_reply
    session.ablate_selected.return_value = after_reply, after
    def complete(method, **args):
        if method == "trace.run": return {"step": 1, "seconds": 0}
        if method == "trace.status": return status
        return {}
    session.events.complete.side_effect = complete
    def capture(name, steps):
        if name == "before": session.ablate_selected.assert_not_called()
        if name == "after":
            assert steps["probabilities"] == 14
            raise Failure("capture barrier reached")
        return None, None
    session.capture.side_effect = capture
    monkeypatch.setattr(driver, "read_graph", lambda s: graph)
    monkeypatch.setattr(driver, "programs", Mock())
    monkeypatch.setattr(driver.Bridge, "select", lambda bridge, row: {})
    monkeypatch.setattr(driver, "program_budget_checks", lambda snapshot: [])
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: 0))
    import _checks
    monkeypatch.setattr(_checks, "example_page_check", lambda *args: {})
    result = {"checks": []}
    with pytest.raises(Failure, match="capture barrier reached"):
        driver.acceptance(session, SimpleNamespace(prompt=prompt, model="tiny-lm"), result)
    session.ablate_selected.assert_called_once_with(prompt, minimum_step=10)
    selection = next(gate for gate in result["checks"] if gate.get("name") == "selection_only")
    assert selection["required"] is True and selection["passed"] is (not selection_changed)
    assert result["comparison"] == {"before": prior, "after": after}


@pytest.mark.parametrize("effect,probability,new_top,height,steering_direction,pixel_direction", [
    (-4.64, .5, 1, 50, True, True),
    (4.64, .1, 1, 10, True, True),
    (-4.64, .1, 1, 10, False, False),
    (4.64, .5, 1, 50, False, False),
    (-4.64, .32, 1, 32, True, True),
    (4.64, .28, 1, 28, True, True),
    (-4.64, .32, 2, 32, True, True),
    (4.64, .28, 2, 28, True, True),
    (-4.64, .28, 2, 28, False, False),
    (4.64, .32, 2, 32, False, False),
    (-4.64, .3, 2, 30, False, False),
    (4.64, .3, 2, 30, False, False),
    (-4.64, .5, 1, 10, True, False),
    (4.64, .1, 1, 50, True, False),
    (-4.64, .32, 2, 30, True, False),
    (4.64, .28, 2, 30, True, False),
    (0., .5, 1, 50, False, False),
    (0., .32, 2, 32, False, False),
])
def test_acceptance_wires_zero_clamp_prediction_to_probability_and_pixels(
        tmp_path, monkeypatch, steering_case, effect, probability, new_top, height, steering_direction, pixel_direction):
    from types import SimpleNamespace
    from unittest.mock import Mock
    prompt, _, status, _ = inference_fixture()
    binding = {key: status["inference"][key] for key in ("prompt_hash", "interventions_hash")}
    prior = dict(binding, argmax_token_ids=[1, 1], completed_step=9)
    after = dict(prior, argmax_token_ids=[1, new_top], completed_step=14)
    graph = steering_case["graph"]
    selected = graph["nodes"][1]
    selected["effect"] = effect
    # The secondary feature has the opposite prediction and greater influence.
    graph["nodes"][2]["effect"] = -effect / 2
    status.update(state="done", counts={"nodes": len(graph["nodes"])})
    status["inference"]["comparison"] = prior
    before_rows = [[0, 0, 0, 0, .3 if token == 1 else .02, .3 if token == 1 else .02, token, token - 1]
                   for token in range(1, 11)]
    after_rows = deepcopy(before_rows)
    after_rows[0][5] = probability
    if new_top == 2:
        after_rows[1][5] = .4
    # The original token moves to the last row, separate from the new top token.
    before_reply = {"step": 9, "elements": before_rows}
    after_reply = {"step": 14, "elements": list(reversed(after_rows))}
    folder = tmp_path / "acceptance"
    folder.mkdir()
    session = Mock(out=folder, run_id="circuit-trace", show=False,
                   program_evidence={"before": {}, "after": {}})
    session.read.return_value = before_reply
    session.example_page.return_value = before_reply, before_reply
    session.ablate_selected.return_value = after_reply, after
    def complete(method, **args):
        if method == "trace.run": return {"step": 1, "seconds": 0}
        if method == "trace.status": return status
        if method == "trace.validate": return steering_case["validation"]
        return {}
    session.events.complete.side_effect = complete
    def capture(name, steps):
        if name == "methods-expanded":
            raise Failure("steering checks recorded")
        return folder / (name + ".png"), {"name": name}
    session.capture.side_effect = capture
    def capture_checks(folder, metadata, png, graph, bars, *, selected, baseline_top_token):
        assert baseline_top_token == 1
        return [], {"row_ids": list(range(8)), "height": height if metadata["name"] == "after" else 30}
    monkeypatch.setattr(driver, "instrument", Mock())
    monkeypatch.setattr(driver, "read_graph", lambda s: graph)
    monkeypatch.setattr(driver, "programs", Mock())
    monkeypatch.setattr(driver.Bridge, "select", lambda bridge, row: steering_case["examples"])
    monkeypatch.setattr(driver.Bridge, "details", Mock())
    monkeypatch.setattr(driver, "program_budget_checks", lambda snapshot: [])
    monkeypatch.setattr(driver, "capture_checks", capture_checks)
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: 0))
    import _checks
    monkeypatch.setattr(_checks, "example_page_check", lambda *args: {})
    result = {"checks": []}
    with pytest.raises(Failure, match="steering checks recorded"):
        driver.acceptance(session, SimpleNamespace(prompt=prompt, model="local-gemma"), result)
    session.ablate_selected.assert_called_once_with(prompt, minimum_step=10)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["selected_feature"] == selected
    assert saved["secondary_selection"]["feature"] == graph["nodes"][2]
    gates = {gate["name"]: gate for gate in saved["checks"] if "name" in gate}
    steering = gates["steering_change_matches_prediction"]
    pixels = gates["bar_change_matches_prediction"]
    assert steering["required"] and steering["passed"]
    assert pixels["required"] and pixels["passed"]
    assert steering["measured"]["direction_matches"] is steering_direction
    assert pixels["measured"]["direction_matches"] is pixel_direction
    for gate in (steering, pixels):
        assert gate["measured"]["predicted_ablation_effect"] == -effect
        expected_direction = "increase" if effect < 0 else "decrease" if effect > 0 else None
        assert gate["measured"]["predicted_direction"] == expected_direction
    assert steering["measured"]["baseline_top"] == 1
    assert steering["measured"]["steered_top"] == new_top
    assert steering["measured"]["baseline_probability"] == .3
    assert steering["measured"]["steered_probability"] == probability
    assert steering["measured"]["probability_change"] == pytest.approx(probability - .3)
    assert steering["measured"]["absolute_probability_change"] == pytest.approx(abs(probability - .3))
    assert pixels["measured"]["pixel_change"] == height - 30
    assert gates["validation"]["passed"]


@pytest.mark.parametrize("tie", ["exact", "rounded"])
@pytest.mark.parametrize("missing", [None, "comparison", "before", "after"])
def test_acceptance_pixels_keep_original_argmax(tmp_path, monkeypatch, steering_case, tie, missing):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np
    import _checks
    prompt, _, status, _ = inference_fixture()
    # Float32 bar values can hide a distinct float64 argmax.
    exact = np.array([.5, .5 if tie == "exact" else .5 + 1e-9], dtype=np.float64)
    original = int(exact.argmax()) + 1
    other = 3 - original
    rounded = exact.astype(np.float32)
    assert rounded[0] == rounded[1]
    binding = {key: status["inference"][key] for key in ("prompt_hash", "interventions_hash")}
    prior = dict(binding, argmax_token_ids=[original, original], completed_step=9)
    after_comparison = dict(binding, argmax_token_ids=[original, other], completed_step=14)
    graph = steering_case["graph"]
    graph["tokens"] = [{"text": "A"}]
    graph["nodes"][1].update(effect=4.64, label="B")
    secondary = graph["nodes"][2]
    steering_case["validation"]["secondary_measurement"] = dict(
        layer=secondary["layer"], position=secondary["position"], feature=secondary["feature"],
        predicted=-secondary["effect"], measured=-.5, sign_match=True)
    status.update(state="done", counts={"nodes": len(graph["nodes"])})
    status["inference"]["comparison"] = prior
    before_rows = [[0, 0, 0, 0, float(rounded[token - 1]) if token <= 2 else 0.,
                   float(rounded[token - 1]) if token <= 2 else 0., token, token - 1]
                  for token in range(1, 11)]
    after_rows = deepcopy(before_rows)
    after_rows[original - 1][5] = .25
    after_rows[other - 1][5] = .75
    # Reorder both the resource rows and their pixel IDs after the clamp.
    after_rows.sort(key=lambda row: row[5], reverse=True)
    for index, row in enumerate(after_rows):
        row[7] = index
    if missing == "comparison":
        prior.pop("argmax_token_ids")
    elif missing in ("before", "after"):
        rows = before_rows if missing == "before" else after_rows
        next(row for row in rows if row[6] == original)[6] = 99
    before_reply = {"step": 9, "elements": before_rows}
    after_reply = {"step": 14, "elements": after_rows}
    folder = tmp_path / "acceptance"
    folder.mkdir()
    session = Mock(out=folder, run_id="circuit-trace", show=False,
                   program_evidence={"before": {}, "after": {}})
    session.read.return_value = before_reply
    session.example_page.return_value = before_reply, before_reply
    session.ablate_selected.return_value = after_reply, after_comparison
    captures = []
    def complete(method, **args):
        if method == "trace.run": return {"step": 1, "seconds": 0}
        if method == "trace.status":
            if "methods-expanded" in captures:
                raise Failure("pixel checks recorded")
            return status
        if method == "trace.validate": return steering_case["validation"]
        return {}
    session.events.complete.side_effect = complete
    def capture(name, steps):
        captures.append(name)
        target = folder / "captures" / name
        target.mkdir(parents=True)
        reply = after_reply if name in ("after", "methods-expanded") else before_reply
        metadata, _, _ = synthetic_capture(target, output_bars=driver.bar_rows(reply))
        metadata["window"] = name + "/" + metadata["window"]
        metadata["font_atlas"]["file"] = name + "/" + metadata["font_atlas"]["file"]
        for panel in metadata["panels"].values():
            for key in ("ids", "color"):
                panel[key] = name + "/" + panel[key]
        return target / "capture.png", metadata
    session.capture.side_effect = capture
    monkeypatch.setattr(driver, "instrument", Mock())
    monkeypatch.setattr(driver, "read_graph", lambda s: graph)
    monkeypatch.setattr(driver, "programs", Mock())
    monkeypatch.setattr(driver.Bridge, "select", lambda bridge, row: steering_case["examples"])
    monkeypatch.setattr(driver.Bridge, "details", Mock())
    monkeypatch.setattr(driver, "program_budget_checks", lambda snapshot: [])
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: 0))
    monkeypatch.setattr(_checks, "example_page_check", lambda *args: {})
    result = {"checks": []}
    message = {None: "pixel checks recorded", "comparison": "actual argmax token IDs",
               "before": "omit an actual argmax", "after": "omit the original baseline top token"}[missing]
    with pytest.raises(Failure, match=message):
        driver.acceptance(session, SimpleNamespace(prompt=prompt, model="local-gemma"), result)
    if missing is not None:
        assert not any(g.get("name") == "bar_change_matches_prediction" for g in result["checks"])
        return
    gates = {gate["name"]: gate for gate in result["checks"] if "name" in gate}
    assert captures == ["before", "examples-page-2", "after", "methods-expanded"]
    for name, height in (("before", 32), ("page-2", 32), ("after", 16), ("methods-expanded", 16)):
        gate = gates[name + ".bar_height"]
        assert gate["passed"] and gate["threshold"] == "finite bar measurements with exact resource identity"
        assert gate["measured"]["token"] == original
        assert gate["measured"]["pixels"] == height
        assert gate["measured"]["expected"] == pytest.approx(height)
    pixels = gates["bar_change_matches_prediction"]
    assert pixels["passed"] and pixels["measured"]["pixel_change"] == -16
    semantic = gates["steering_change_matches_prediction"]
    assert semantic["passed"] and semantic["measured"]["baseline_top"] == original
    assert semantic["measured"]["probability_change"] == -.25
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["secondary_selection"]["feature"] == secondary
    assert saved["secondary_selection"]["measured_ablation_effect"] == -.5
    assert saved["secondary_selection"]["measurement_source"] == "trace.validate"


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


@pytest.mark.skipif(os.name == "nt", reason="This integration case uses a POSIX executable and process launcher.")
def test_mcp_startup_failure_reaps_owned_standard_library_child(tmp_path):
    from _mcp import MCP
    import psutil
    path = stdio_child(tmp_path, refuse=True)
    with pytest.raises(Failure, match="server/discover"):
        MCP(path, tmp_path / "api.json", tmp_path, os.environ.copy())
    pid = int((tmp_path / "api.pid").read_text())
    assert not psutil.pid_exists(pid)


@pytest.mark.skipif(os.name == "nt", reason="This integration case uses a POSIX executable and process launcher.")
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


@pytest.mark.parametrize("model,display", [("tiny-lm", "1"), ("local-gemma", None)])
@pytest.mark.skipif(os.name == "nt", reason="This integration case uses a POSIX executable and process launcher.")
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
    session = Mock(family="fixture")
    import tomllib
    spec = {"caliper_spec": 1, "panel": []}
    def body(method, **args):
        if method == "panel.create":
            spec["panel"].append(deepcopy(args))
        elif method == "spec.get":
            return {"json": deepcopy(spec)}
        elif method == "piece.show":
            return {"piece": {"hash": "0" * 64}, "from": "library"}
        elif method == "spec.set_text":
            spec.clear()
            spec.update(tomllib.loads(args["text"]))
    session.mcp.body.side_effect = body
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
    description = next(call.kwargs["description"] for call in calls if call.args[0] == "spec.new")
    readouts = {call.kwargs["id"]: call.kwargs["options"]["format"] for call in calls
                if call.args[0] == "control.create" and call.kwargs["kind"] == "readout"}
    assert len(readouts) == 6
    for name, text in driver.METHOD_DISCLOSURES.items():
        assert text in description and readouts["method_" + name] == text
    controls = {call.kwargs["id"]: call.kwargs for call in calls if call.args[0] == "control.create"}
    assert controls["clamp"]["kind"] == "slider"
    assert controls["clamp"]["bind"] == "param:clamp"
    assert controls["clamp"]["options"] == {"min": -4, "max": 4}
    assert controls["apply_clamp"]["kind"] == "button"
    assert controls["apply_clamp"]["bind"] == "action:apply_clamp"
    assert controls["reset"]["bind"] == "action:reset"
    assert controls["clamp_status_value"]["bind"] == "param:clamp.status"
    assert "bind" not in controls["clamp_status"]
    assert controls["clamp_status"]["options"] == {
        "format": "{status:s}", "fields": {"status": {"control": "clamp_status_value"}}}
    layout = next(call.kwargs["layout"] for call in calls if call.args[0] == "layout.set")
    nodes = {node["id"]: node for node in layout["nodes"]}
    leaves = []
    def visit(identifier):
        if identifier in nodes:
            for child in nodes[identifier]["children"]:
                visit(child)
        else:
            leaves.append(identifier)
    visit(layout["root"])
    assert len(leaves) == len(set(leaves))
    hidden = {"clamp_status_value", *("method_" + name for name in driver.METHOD_DISCLOSURES)}
    assert set(leaves) == set(panels) | (set(controls) - hidden)
    assert nodes["instrument"]["ratio"] == .95
    assert nodes["body"]["children"] == ["plots", "examples"]
    assert nodes["body"]["ratio"] == .70
    assert nodes["plots"]["ratio"] == .67
    assert nodes["main"]["children"] == ["header", "body"]
    assert nodes["header"]["children"] == ["prompt", "clamp_status"]
    assert nodes["controls"]["children"][1:3] == ["apply_clamp", "reset"]
    graph = next(program for program in spec["program"] if program["id"] == "attribution")
    assert graph["bind"]["graph_counts"] == "live:trace.counts"
    bars = next(program for program in spec["program"] if program["id"] == "probabilities")
    assert set(bars["bind"].values()) == {"live:output.probs", "live:output.tokens", "live:output.label_text"}
    assert all(panel["primitive"] == "scene" for panel in spec["panel"])
    assert all(program["piece"]["hash"] == "0" * 64 for program in spec["program"])
    driver.scene_programs(session, [], remove_examples=True)
    assert next(panel for panel in spec["panel"] if panel["id"] == "examples")["primitive"] == "text"
    assert {program["id"] for program in spec["program"]} == {"attribution", "probabilities"}
    driver.instrument_layout(session, expanded=True)
    expanded = session.mcp.body.call_args.kwargs["layout"]
    groups = {node["id"]: node for node in expanded["nodes"]}
    assert groups["instrument"]["ratio"] == .74
    assert groups["body"]["ratio"] == .64
    assert groups["controls"]["split"] == "horizontal"
    assert groups["primary_controls"]["children"] == ["clamp", "apply_clamp", "reset", "retrace"]
    assert groups["secondary_controls"]["children"] == ["generate", "page", "baseline_text", "steered_text"]
    def depth(identifier):
        return 1 + max((depth(child) for child in groups[identifier]["children"]), default=0) if identifier in groups else 1
    assert depth(expanded["root"]) <= 8
    assert set(groups["methods"]["children"]) == {"method_details"} | (set(readouts) - {"clamp_status"})


def test_bridge_expands_and_collapses_method_details():
    from unittest.mock import Mock, patch
    session = Mock(family="fixture")
    bridge = driver.Bridge(session)
    with patch.object(driver, "instrument_layout") as layout:
        for value in (True, False):
            bridge.handle({"kind": "control", "control": "method_details", "value": value})
            layout.assert_called_with(session, expanded=value)
            assert bridge.details_expanded is value
    assert layout.call_count == 2


@pytest.mark.parametrize("method,arguments", [("piece.show", {"name": "attribution_graph", "version": 1}),
                                            ("run.set_param", {"run": "circuit-trace", "name": "clamp", "value": 0})])
def test_mcp_command_name_does_not_consume_named_argument(method, arguments):
    from io import StringIO
    from unittest.mock import Mock
    from _mcp import MCP
    client = MCP.__new__(MCP)
    client.journal = StringIO()
    client.request = Mock(return_value={"structuredContent": {"ok": True}})
    assert client.body(method, **arguments) == {"ok": True}
    client.request.assert_called_once_with("tools/call", {"name": method.replace(".", "_"), "arguments": arguments})
    assert json.loads(client.journal.getvalue())["args"] == arguments


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
    family = driver.acceptance_profile(model)[0]
    if family == "gemma":
        command.extend(["--replication-profile", "published_dallas_v1", "--private-output-directory", str(out.parent)])
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=1800)
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
    if family == "gemma":
        assert report["evidence_collection_complete"] and not report["accepted_replication"]
    else:
        assert report["passed"]
    return report


def test_hidden_fixture(tmp_path):
    resources_and_device(fixture=True)
    from caliper.interp._hosted import fixture_resources
    folder = tmp_path / "dictionary"
    fixture_resources(folder)
    report = hidden(tmp_path, "tiny-lm", folder)
    assert report["accepted_replication"] is False


@pytest.mark.local_model
def test_local_gemma_collection(tmp_path):
    model, dictionary = os.environ.get("CALIPER_GEMMA_MODEL"), os.environ.get("CALIPER_GEMMA_DICTIONARY")
    if not model or not dictionary:
        pytest.skip("Set both explicit local Gemma resource paths to opt in.")
    assert not os.environ.get("CI"), "The local Gemma case must not run in CI."
    assert Path(model).is_dir() and Path(dictionary).is_dir()
    resources_and_device(fixture=False)
    report = hidden(tmp_path, model, dictionary)
    assert report["evidence_collection_complete"] and not report["gemma_acceptance"]


def test_event_wait_bounds_completion_without_an_event(monkeypatch):
    from unittest.mock import Mock
    from demos.circuit_trace import _mcp
    events = _mcp.Events(Mock())
    clock = [0.]
    monkeypatch.setattr(_mcp.time, "monotonic", lambda: clock[0])
    waits = []
    def wait(timeout):
        waits.append(timeout)
        clock[0] += timeout
    monkeypatch.setattr(events, "wait", wait)
    reads = iter([False, False, {"step": 4}])
    assert events.until("resource", lambda: next(reads)) == {"step": 4}
    assert waits == [.05, .05]


class CaptureResources:
    """Separate resource versions from the proposed encoded draw parameters."""
    def __init__(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        import _mcp
        self.session = driver.Session.__new__(driver.Session)
        self.session.out = tmp_path
        self.session.show = True
        self.session.capture_number = 0
        self.session.program_evidence = {}
        self.session.bridge = SimpleNamespace(selection_step=2, last_selection=4)
        self.session.mcp = self
        self.cursor = dict(step=2, latest=2, live=True, playing=False, speed=10)
        self.received = {p: dict.fromkeys(resources, 2) for p, resources in driver.CAPTURE_RESOURCES.items()}
        self.shown = deepcopy(self.received)
        self.draw_params = {"selected_node": 4}
        self.frames = 8
        self.stage = 0
        self.clock = 0
        self.actions = []
        self.captures = []
        self.reads = []
        self.stopped = False
        self.derive_ready = True
        self.advance = lambda: None
        events = _mcp.Events(self)
        monkeypatch.setattr(_mcp, "time", SimpleNamespace(monotonic=lambda: self.clock))
        def wait(timeout):
            self.clock += .2
            self.stage += 1
            self.frames += 1
            self.advance()
        events.wait = wait
        until = events.until
        events.until = lambda description, read: until(description, read, timeout=1)
        self.session.events = events

    def body(self, command, **args):
        if command == "cursor.get":
            return {"cursor": dict(self.cursor)}
        if command == "cursor.set":
            self.actions.append(dict(args))
            self.cursor.update(args)
            return {"cursor": dict(self.cursor)}
        if command == "program.read":
            self.reads.append(dict(args))
            step = self.received[args["id"]][args["resource"]]
            return {"step": step if step is None else min(step, args["step"])}
        if command == "program.status":
            return {"programs": [
                dict(id=p, state="running",
                     last_step=max((v or 0) for v in self.received[p].values()) if self.derive_ready else None,
                     last_pass_ms=.1 if self.derive_ready else None, piece={"name": p},
                     draw=dict(resource_step=max((v or 0) for v in self.shown[p].values()),
                               resource_steps=dict(self.shown[p]), frames=self.frames, frame_ms=.2,
                               params=dict(self.draw_params) if p == "attribution" else {},
                               state="stopped" if self.stopped else "running"))
                for p in self.received]}
        if command == "piece.show":
            return {"piece": {"name": args["name"]}}
        raise AssertionError((command, args))

    def call(self, command):
        assert command == "capture.frame"
        assert not self.cursor["live"] and not self.cursor["playing"]
        assert all(len(set(steps.values())) == 1 for steps in self.shown.values())
        selected = self.session.bridge.last_selection
        assert self.draw_params["selected_node"] == (selected if selected is not None else -1)
        assert self.derive_ready
        self.captures.append((self.stage, deepcopy(self.shown)))
        # A later resource must not move the held capture to another step.
        held = self.cursor["step"]
        self.cursor["latest"] = held + 1
        assert self.cursor["step"] == held
        return {"structuredContent": {"png_base64": "cG5n"}}

    def capture(self, name="before", *, minimum_step=1):
        return self.session.capture(name, dict.fromkeys(driver.CAPTURE_RESOURCES, minimum_step))


def test_capture_waits_for_received_and_displayed_resources_after_selection(tmp_path, monkeypatch):
    harness = CaptureResources(tmp_path, monkeypatch)
    harness.received["attribution"] = dict.fromkeys(driver.CAPTURE_RESOURCES["attribution"], 1)
    harness.shown["attribution"] = dict(harness.received["attribution"])
    def advance():
        if harness.stage <= len(driver.CAPTURE_RESOURCES["attribution"]):
            resource = driver.CAPTURE_RESOURCES["attribution"][harness.stage - 1]
            harness.received["attribution"][resource] = 2
            if resource != "strings":
                harness.shown["attribution"][resource] = 2
        else:
            harness.shown["attribution"]["strings"] = 2
    harness.advance = advance
    png, metadata = harness.capture(minimum_step=2)
    assert png.read_bytes() == b"png" and metadata is None
    assert harness.captures[0][0] == len(driver.CAPTURE_RESOURCES["attribution"]) + 1
    assert harness.cursor["live"]
    evidence = harness.session.program_evidence["before"]
    assert evidence["expected_steps"]["attribution"] == 2
    assert evidence["resource_steps"] == harness.shown
    assert {item["resource"] for item in harness.reads if item["id"] == "attribution"} == {
        "nodes", "curves", "labels", "strings", "cells"}


@pytest.mark.parametrize("program,resource", [(p, r) for p, resources in driver.CAPTURE_RESOURCES.items() for r in resources])
def test_capture_waits_for_each_displayed_resource(tmp_path, monkeypatch, program, resource):
    harness = CaptureResources(tmp_path, monkeypatch)
    harness.shown[program][resource] = 1
    harness.advance = lambda: harness.shown[program].update({resource: 2}) if harness.stage == 2 else None
    harness.capture()
    assert harness.captures[0][0] == 2


@pytest.mark.parametrize("fault", ["missing_publication", "missing_frame", "mixed_frame", "old_frame", "derive_status"])
def test_capture_resource_failure_is_bounded_without_pixels_or_sleeps(tmp_path, monkeypatch, fault):
    harness = CaptureResources(tmp_path, monkeypatch)
    if fault == "missing_publication":
        harness.received["attribution"]["strings"] = None
    elif fault == "missing_frame":
        harness.shown["attribution"].pop("strings")
    elif fault == "mixed_frame":
        harness.shown["attribution"]["strings"] = 1
    elif fault == "old_frame":
        harness.advance = lambda: setattr(harness, "frames", 8)
    else:
        harness.derive_ready = False
    with pytest.raises(Failure, match="Timed out"):
        harness.capture()
    assert not harness.captures
    assert harness.stage == 5
    assert harness.cursor["live"]


def test_capture_waits_for_derive_status_at_the_rendered_step(tmp_path, monkeypatch):
    harness = CaptureResources(tmp_path, monkeypatch)
    harness.derive_ready = False
    harness.advance = lambda: setattr(harness, "derive_ready", harness.stage >= 2)
    harness.capture()
    assert harness.captures[0][0] == 2


def test_new_selection_requires_a_new_frame_after_an_existing_capture(tmp_path, monkeypatch):
    harness = CaptureResources(tmp_path, monkeypatch)
    harness.capture("first")
    first = harness.session.program_evidence["first"]["status"]["programs"][0]["draw"]["frames"]
    harness.cursor.update(step=3, latest=3)
    harness.session.bridge.selection_step = 3
    harness.session.bridge.last_selection = 7
    received = deepcopy(harness.received)
    start = harness.stage
    def advance():
        if harness.stage >= start + 2:
            harness.draw_params["selected_node"] = 7
    harness.advance = advance
    harness.capture("second")
    second = harness.session.program_evidence["second"]["status"]["programs"][0]["draw"]["frames"]
    assert second > first
    assert [stage for stage, _ in harness.captures] == [1, 3]
    assert harness.received == received
    evidence = harness.session.program_evidence["second"]
    assert evidence["resource_steps"]["attribution"] == received["attribution"]
    assert evidence["selected_node"] == 7


@pytest.mark.parametrize("failure", ["capture", "cursor", "stopped"])
def test_capture_restores_a_held_cursor_after_failure(tmp_path, monkeypatch, failure):
    harness = CaptureResources(tmp_path, monkeypatch)
    original = dict(harness.cursor, step=1, live=False, playing=True, speed=3)
    harness.cursor = dict(original)
    if failure == "capture":
        def fail(command):
            raise Failure("Capture unavailable")
        harness.call = fail
    elif failure == "cursor":
        harness.advance = lambda: harness.cursor.update(step=3)
    else:
        harness.advance = lambda: setattr(harness, "stopped", True)
    with pytest.raises(Failure):
        harness.capture()
    assert {k: harness.cursor[k] for k in original} == original
    assert not harness.captures


def test_selection_bridge_waits_for_a_fresh_pick_even_when_the_row_matches(tmp_path):
    from types import SimpleNamespace
    bridge = driver.Bridge(SimpleNamespace(out=tmp_path))
    bridge.last_selection = 4
    bridge.last_examples = {"examples": ["old"]}
    bridge.selection_generation = 1
    polls = []
    def until(description, read):
        polls.append(read())
        bridge.selection_generation += 1
        bridge.last_examples = {"examples": ["new"]}
        polls.append(read())
        return polls[-1]
    bridge.consume = lambda: None
    bridge.session.mcp = SimpleNamespace(body=lambda *a, **kw: None)
    bridge.session.events = SimpleNamespace(until=until)
    assert bridge.select(4) == {"examples": ["new"]}
    assert polls == [False, {"examples": ["new"]}]


@pytest.mark.parametrize("flow", ["ordinary", "live", "published_collection", "published_live"])
def test_demo_flows_use_the_complete_selection_capture_barrier(tmp_path, monkeypatch, flow):
    from types import SimpleNamespace
    harness = CaptureResources(tmp_path, monkeypatch)
    session = harness.session
    session.run_id = "circuit-trace"
    session.private_output_directory = tmp_path
    session.replication_profile = driver.published.PROFILE if flow.startswith("published") else None
    session.ready_programs = lambda identifiers: None
    session.read = lambda *args, **kwargs: {"step": 2}
    session.example_page = lambda page: ({"step": 2}, {"step": 2})
    selected = dict(index=0, id="feature:0:0:1", kind="feature", layer=0, feature=1, position=0,
                    influence=1, effect=1, label="Selected Feature")
    graph = dict(nodes=[selected], counts={"layers": 2}, completeness={"max_absolute_residual": 0})
    import _replication_review, _completeness
    monkeypatch.setattr(_replication_review, "_full_graph", lambda *args: {})
    monkeypatch.setattr(_replication_review, "_graph_path", lambda *args: tmp_path / "synthetic-graph.json")
    monkeypatch.setattr(_completeness, "summary_evidence", lambda graph: graph["completeness"])
    trace = dict(step=1, seconds=0, state="done", counts={"nodes": 1})
    session.events.complete = lambda *args, **kwargs: trace
    monkeypatch.setattr(driver, "check_assets", lambda s: {})
    monkeypatch.setattr(driver, "instrument", lambda s: None)
    monkeypatch.setattr(driver, "programs", lambda s: None)
    monkeypatch.setattr(driver, "read_graph", lambda s: graph)
    monkeypatch.setattr(driver, "bar_rows", lambda r: [dict(token=1, baseline=.7, probability=.7)])
    monkeypatch.setattr(driver, "comparison_evidence", lambda *a: {"argmax_token_ids": [1, 1]})
    monkeypatch.setattr(driver.published, "replication_state", lambda s: ({}, {}))
    monkeypatch.setattr(driver.published, "preparation", lambda *a: {
        "id": "prepared", "prompts": {"dallas": {"token_ids": []}}})
    monkeypatch.setattr(driver.published, "read_graph", lambda *a, **kw: (graph, []))
    monkeypatch.setattr(driver.published, "published_selection", lambda g: selected)
    monkeypatch.setattr(driver, "published_action", lambda *a: {
        "completed": True, "evidence": {"result": {"argmax_token_ids": [1, 1]}}})
    def select(bridge, row):
        bridge.selection_step = 3
        bridge.last_selection = row
        harness.draw_params["selected_node"] = -1
        harness.received["attribution"]["strings"] = 1
        harness.shown["attribution"]["strings"] = 1
        return {"examples": []}
    monkeypatch.setattr(driver.Bridge, "select", select)
    def advance():
        if harness.stage >= 2:
            harness.received["attribution"]["strings"] = 2
        if harness.stage >= 3:
            harness.shown["attribution"]["strings"] = 2
            harness.draw_params["selected_node"] = 0
    harness.advance = advance
    def reached(command):
        CaptureResources.call(harness, command)
        raise Failure("Complete resource capture reached")
    harness.call = reached
    entry = driver.live_demo if flow in ("live", "published_live") else driver.acceptance
    with pytest.raises(Failure, match="Complete resource capture reached"):
        entry(session, SimpleNamespace(prompt=driver.published.PROMPT, model="tiny-lm"), {"checks": []})
    assert harness.captures[0][0] == 3
    assert harness.cursor["live"]


@pytest.mark.parametrize("fault", ["missing_program", "refused", "stopped_draw", "error", "missing_step", "stale_step", "no_frames"])
def test_program_measurements_cannot_replace_readiness(fault):
    program = {"id": "attribution", "state": "running", "last_step": 2, "last_pass_ms": 100,
               "memory_bytes": 1024, "memory_limit_bytes": 64 << 20,
               "draw": {"state": "running", "frames": 1, "frame_ms": 20, "frame_limit_ms": 4,
                        "memory_bytes": 1024, "memory_limit_bytes": 64 << 20}}
    if fault == "refused": program["state"] = "refused"
    elif fault == "stopped_draw": program["draw"]["state"] = "stopped"
    elif fault == "error": program["draw"]["error"] = {"code": "failed"}
    elif fault == "missing_step": program.pop("last_step")
    elif fault == "stale_step": program["last_step"] = 1
    elif fault == "no_frames": program["draw"]["frames"] = 0
    snapshot = {"status": {"programs": [] if fault == "missing_program" else [program]},
                "expected_steps": {"attribution": 2}, "pieces": {"attribution": {
                    "derive": {"time_ms": 50, "memory_mb": 64}, "draw": {"frame_ms": 4, "memory_mb": 64}}}}
    checks = program_budget_checks(snapshot)
    assert not next(gate for gate in checks if gate["name"] == "attribution.ready")["passed"]
    assert finish({"mode": "fixture", "completed": True, "checks": checks}) == 1


@pytest.mark.parametrize("share,passed", [(0, True), (.01, True), (1., True), (None, False),
    (True, False), ("0.3", False), (-.1, False), (1.1, False), (float("nan"), False), (float("inf"), False)])
def test_feature_share_requires_measurements_without_minimum(steering_case, share, passed):
    steering_case["graph"]["feature_share"] = share
    checks = {gate["name"]: gate for gate in math_checks(**steering_case)}
    assert checks["feature_share"]["passed"] is passed
    assert checks["graph_size"]["passed"] and checks["graph_size"]["measured"] == 4


@pytest.mark.parametrize("value", [None, True, "0.5", -.1, 1.1, float("nan"), float("inf")])
def test_steering_rejects_invalid_actual_probabilities(steering_case, value):
    steering_case["after"][0]["probability"] = value
    try:
        checks = {gate["name"]: gate for gate in math_checks(**steering_case)}
    except (TypeError, ValueError):
        return
    assert not checks["steering_change_matches_prediction"]["passed"]


@pytest.mark.parametrize("command,completion,passed", [("trace.run", 150, True),
    ("trace.run", 630, False), ("trace.validate", 150, False)])
def test_trace_command_wait_uses_founder_safety_timeout(monkeypatch, command, completion, passed):
    from types import SimpleNamespace
    import _mcp
    clock = [0.]
    def body(name, **arguments):
        assert name == command
        if not arguments or clock[0] < completion:
            return {"pending": True, "request_id": "fixture"}
        return {"pending": False, "result": {"state": "done"}}
    events = _mcp.Events(SimpleNamespace(body=body))
    monkeypatch.setattr(_mcp, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    events.wait = lambda timeout: clock.__setitem__(0, clock[0] + 30)
    if passed:
        assert events.complete(command) == {"state": "done"}
        assert clock[0] == 150
    else:
        with pytest.raises(Failure, match="Timed out"):
            events.complete(command)
        assert clock[0] == (600 if command == "trace.run" else 120)
