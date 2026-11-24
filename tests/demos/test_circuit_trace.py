"""Circuit math checks and the real hidden MCP acceptance driver."""
from __future__ import annotations

import json
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
from _checks import check, finish, math_checks, program_budget_checks, selected_feature, token_path, FIXTURE_PROMPT
from _mcp import Failure
from _qualification import dictionary_qualification, MODEL_ID, MODEL_REVISION, MODEL_HASH


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


def qualification_reply(*, fallback=False, token_budget=20_000_000):
    """Synthetic metadata exercises gates without model or index allocation."""
    training = {"dataset_id": "test-source", "dataset_config": "test", "dataset_revision": "revision",
                "split": "train", "source_id": "test-source:revision", "tokens": token_budget,
                "token_budget": token_budget, "stop_reason": "token_budget",
                "partitions": {"algorithm": "sha256-utf8-text-mod10000-v1", "train": [0, 9400],
                               "quality": [9400, 9500], "index": [9500, 10000],
                               "quality_tokens": 200_000, "index_tokens": 2_000_000}}
    quality = {"status": "passed", "evaluated_tokens": 200_000, "partition": "quality", "stop_reason": "complete",
               "dictionary_hash": "a" * 64, "all_mlps_replaced": 24,
               "fvu": .4, "dead_fraction": .15, "cross_entropy_delta": .5,
               "per_layer_fvu": [.4] * 24, "per_layer_dead_fraction": [.15] * 24,
               "cross_entropy_tokens": 199_000, "original_cross_entropy": 2., "replaced_cross_entropy": 2.5}
    metadata = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "model_hash": MODEL_HASH,
                "layers": 24, "latents": 8192, "training": training, "quality": quality}
    identity = {"model_hash": MODEL_HASH, "dictionary_hash": "b" * 64,
                "dictionary_fingerprint": "a" * 64, "index_hash": "c" * 64, "tokenizer_hash": "d" * 64}
    index = {k: identity[k] for k in ("model_hash", "dictionary_hash", "index_hash", "tokenizer_hash")}
    index.update(complete=True, tokens=2_000_000, token_budget=2_000_000, layers=24, features=8192,
                 provenance={k: deepcopy(training[k]) for k in
                             ("dataset_id", "dataset_config", "dataset_revision", "split", "source_id", "partitions")})
    index["provenance"]["partition"] = "index"
    response = {"metadata": metadata, "identity": identity, "index": index}
    if fallback:
        source = deepcopy(metadata)
        source["quality"].update(status="failed", dictionary_hash="e" * 64,
                                  fvu=.41, per_layer_fvu=[.41] * 24)
        response["fallback_source"] = {
            "metadata": source, "identity": {"model_hash": MODEL_HASH, "dictionary_fingerprint": "e" * 64},
            "evaluation": {"version": 1, "count": 200_000, "pending_ids": None, "identity": "e" * 64,
                           "settings": {"tokens": 200_000, "scope": "all_layers", "active_layers": list(range(24))},
                           "stream": {"partition": "quality", "source_id": training["source_id"],
                                      "partitions": deepcopy(training["partitions"])}}}
        metadata["fallback_source"] = response["fallback_source"]
        training["fallback"] = {"version": 1, "mode": "upper_half", "active_layers": list(range(12, 24)),
                                "zero_layers": list(range(12)), "source_dictionary_hash": "e" * 64,
                                "source_training_tokens": token_budget, "source_id": training["source_id"],
                                "token_budget": token_budget, "warm_start": "upper_weights_fresh_adam",
                                "lower_behavior": "zero_dictionary_original_mlp"}
        del quality["all_mlps_replaced"]
        quality.update(status="fallback_passed", scope="upper_half", active_layers=list(range(12, 24)),
                       replaced_mlps=12, per_layer_status=["error_only"] * 12 + ["active"] * 12,
                       per_layer_fvu=[None] * 12 + [.4] * 12, per_layer_dead_fraction=[None] * 12 + [.15] * 12)
    return response


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("token_budget", [20_000_000, 40_000_000])
def test_qualification_accepts_exact_complete_scope(fallback, token_budget):
    response = qualification_reply(fallback=fallback, token_budget=token_budget)
    qualified = dictionary_qualification(response)
    assert qualified["qualified"] is True
    assert qualified["scope"] == ("upper_half" if fallback else "all_layers")
    assert qualified["status"] == ("fallback_passed" if fallback else "passed")
    assert qualified["identity"] == response["identity"]


@pytest.mark.parametrize("scope", ["all_layers", "upper_half", "fallback_source"])
@pytest.mark.parametrize("fault", ["unfinished_count", "pilot_budget", "wrong_stop_reason"])
def test_qualification_requires_completed_approved_training(scope, fault):
    response = qualification_reply(fallback=scope != "all_layers")
    current = response["metadata"]["training"]
    training = response["fallback_source"]["metadata"]["training"] if scope == "fallback_source" else current
    if fault == "unfinished_count":
        training["tokens"] -= 1
    elif fault == "pilot_budget":
        training.update(tokens=100_000, token_budget=100_000)
    else:
        training["stop_reason"] = "memory_stop"
    if scope == "upper_half":
        current["fallback"]["token_budget"] = training["token_budget"]
    elif scope == "fallback_source":
        current["fallback"]["source_training_tokens"] = training["tokens"]
    qualified = dictionary_qualification(response)
    assert qualified["qualified"] is False
    assert "complete training budget" in qualified["reason"]


def test_qualification_rejects_fallback_budget_mismatch():
    response = qualification_reply(fallback=True)
    response["metadata"]["training"]["token_budget"] = 40_000_000
    qualified = dictionary_qualification(response)
    assert qualified["qualified"] is False
    assert "complete training budget" in qualified["reason"]


@pytest.mark.parametrize("field,value", [
    ("status", "approved"), ("status", "partial"), ("evaluated_tokens", 199_999),
    ("evaluated_tokens", "200000"), ("stop_reason", "memory_stop"), ("partition", "train"),
    ("dictionary_hash", None), ("dictionary_hash", "f" * 64), ("all_mlps_replaced", 12),
    ("fvu", float("nan")), ("cross_entropy_delta", float("inf")), ("cross_entropy_tokens", 0),
])
def test_qualification_rejects_incomplete_or_stale_quality(field, value):
    response = qualification_reply()
    response["metadata"]["quality"][field] = value
    assert dictionary_qualification(response)["qualified"] is False


@pytest.mark.parametrize("field,value", [("fvu", .401), ("dead_fraction", .151), ("cross_entropy_delta", .501)])
def test_qualification_rejects_failed_metrics_under_passed_status(field, value):
    response = qualification_reply()
    quality = response["metadata"]["quality"]
    quality[field] = value
    if field == "cross_entropy_delta":
        quality["replaced_cross_entropy"] = quality["original_cross_entropy"] + value
    else:
        quality["per_layer_" + field] = [value] * 24
    assert dictionary_qualification(response)["qualified"] is False


@pytest.mark.parametrize("section,field,value", [
    ("index", "tokens", 1_999_999), ("index", "token_budget", 1_999_999), ("index", "complete", 1),
    ("index", "model_hash", "other-model"), ("index", "dictionary_hash", "f" * 64),
    ("index", "index_hash", "f" * 64), ("index", "tokenizer_hash", "f" * 64),
    ("identity", "dictionary_fingerprint", "f" * 64), ("identity", "dictionary_hash", None),
])
def test_qualification_rejects_index_and_live_identity_mismatch(section, field, value):
    response = qualification_reply()
    response[section][field] = value
    assert dictionary_qualification(response)["qualified"] is False


@pytest.mark.parametrize("section", ["identity", "index", "metadata"])
def test_qualification_requires_live_response_evidence(section):
    response = qualification_reply()
    del response[section]
    assert dictionary_qualification(response)["qualified"] is False


def test_qualification_rejects_overlapping_partitions():
    response = qualification_reply()
    response["metadata"]["training"]["partitions"]["quality"] = [0, 100]
    response["index"]["provenance"]["partitions"]["quality"] = [0, 100]
    assert dictionary_qualification(response)["qualified"] is False


@pytest.mark.parametrize("fault", ["missing_source", "detached_source", "stale_source", "partial_source", "fake_failure",
                                    "unfinished_training", "wrong_scope", "all_layer_claim", "lower_metrics"])
def test_qualification_requires_authorized_fallback_evidence(fault):
    response = qualification_reply(fallback=True)
    source = response["fallback_source"]
    quality = response["metadata"]["quality"]
    if fault == "missing_source":
        del response["fallback_source"]
    elif fault == "detached_source":
        del response["metadata"]["fallback_source"]
    elif fault == "stale_source":
        source["evaluation"]["identity"] = "f" * 64
    elif fault == "partial_source":
        source["evaluation"]["count"] -= 1
    elif fault == "fake_failure":
        source["metadata"]["quality"].update(fvu=.3, per_layer_fvu=[.3] * 24)
    elif fault == "unfinished_training":
        source["metadata"]["training"]["tokens"] -= 1
    elif fault == "wrong_scope":
        quality["active_layers"] = list(range(12))
    elif fault == "all_layer_claim":
        quality["all_mlps_replaced"] = 24
    elif fault == "lower_metrics":
        quality["per_layer_fvu"][0] = .3
    assert dictionary_qualification(response)["qualified"] is False


@pytest.mark.parametrize("fallback", [False, True])
def test_finish_keeps_qwen_qualification_scope(fallback):
    result = {"mode": "qwen", "checks": [check("synthetic_evidence", True, True, True)],
              "pixel_evidence_collected": True, "dictionary_evidence": qualification_reply(fallback=fallback)}
    assert finish(result) == 0
    assert result["qwen_acceptance"] is True
    assert result["qwen_full_mlp_acceptance"] is (not fallback)
    assert result["qwen_acceptance_scope"] == ("upper_half" if fallback else "all_layers")
    result["dictionary_evidence"]["metadata"]["quality"]["dictionary_hash"] = "f" * 64
    assert finish(result) == 1
    assert result["qwen_acceptance"] is False
    assert result["qwen_full_mlp_acceptance"] is False
    assert result["failed_checks"] == ["dictionary_qualification"]


def test_finish_rejects_a_cached_qualification_without_evidence():
    result = {"mode": "qwen", "pixel_evidence_collected": True,
              "dictionary_qualification": {"qualified": True, "status": "passed"},
              "checks": [check("dictionary_qualification", True, "passed", "passed")]}
    assert finish(result) == 1
    assert result["qwen_acceptance"] is False


def test_finish_preserves_fixture_without_qwen_qualification():
    result = {"mode": "fixture", "checks": [check("synthetic_fixture", True, True, True)]}
    assert finish(result) == 0
    assert result["qwen_acceptance"] is False


def failed_quality_override():
    response = qualification_reply(token_budget=40_000_000)
    quality = response["metadata"]["quality"]
    quality.update(status="failed", fvu=.329, dead_fraction=.261, cross_entropy_delta=1.18,
                   per_layer_fvu=[.329] * 24, per_layer_dead_fraction=[.261] * 24,
                   replaced_cross_entropy=quality["original_cross_entropy"] + 1.18)
    record = {"version": 1, "authority": "founder", "dictionary_fingerprint": "a" * 64,
              "evaluated_tokens": 200_000, "fvu": .329, "dead_fraction": .261,
              "cross_entropy_delta": 1.18, "failed_gates": ["dead_fraction", "cross_entropy_delta"]}
    return response, record


def test_quality_override_retains_failures_and_cannot_override_causal_checks():
    response, record = failed_quality_override()
    original = deepcopy(response)
    assert dictionary_qualification(response)["qualified"] is False
    qualified = dictionary_qualification(response, quality_override=record)
    assert qualified["qualified"] is True and qualified["status"] == "founder_override"
    assert qualified["reported_status"] == "failed" and qualified["quality_override"] == record
    assert response == original
    result = {"mode": "qwen", "checks": [check("steering", False, .01, .05)],
              "pixel_evidence_collected": True, "dictionary_evidence": response,
              "dictionary_quality_override": record}
    assert finish(result) == 1
    assert result["failed_checks"] == ["steering"] and result["qwen_acceptance"] is False
    result["checks"][0]["passed"] = True
    assert finish(result) == 0 and result["qwen_acceptance"] is True
    assert result["dictionary_evidence"]["metadata"]["quality"]["status"] == "failed"


@pytest.mark.parametrize("key,value", [
    ("version", True), ("authority", "unknown"), ("dictionary_fingerprint", "f" * 64),
    ("evaluated_tokens", 199_999), ("fvu", .33), ("dead_fraction", .262),
    ("cross_entropy_delta", 1.181), ("fvu", float("nan")), ("failed_gates", []),
    ("failed_gates", ["dead_fraction"]), ("extra", True),
])
def test_quality_override_refuses_changed_identity_or_measurements(key, value):
    response, record = failed_quality_override()
    record[key] = value
    assert dictionary_qualification(response, quality_override=record)["qualified"] is False


@pytest.mark.parametrize("fault", ["index", "training", "evaluation", "model", "layer_metrics"])
def test_quality_override_preserves_resource_and_evidence_requirements(fault):
    response, record = failed_quality_override()
    if fault == "index":
        response["index"]["tokens"] -= 1
    elif fault == "training":
        response["metadata"]["training"]["tokens"] -= 1
    elif fault == "evaluation":
        response["metadata"]["quality"]["stop_reason"] = "interrupted"
    elif fault == "model":
        response["identity"]["model_hash"] = "other"
    else:
        response["metadata"]["quality"]["per_layer_fvu"][0] = .1
    assert dictionary_qualification(response, quality_override=record)["qualified"] is False


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
    # This small composition fixture does not claim a full instrument layout.
    names = {"capture_generation", "rendered_nodes", "host_labels", "example_page", "bar_height"}
    names.update(name for name in gates if name.endswith(("_composition", "_resource_generation")))
    assert all(gates[name]["passed"] for name in names), gates
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
    session = Mock(quality_override=None)
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
    session.events.complete.assert_called_once_with("feature.dictionary")
    session.ready_programs.assert_called_once_with(["probabilities"])


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
    graph = {"nodes": [{"kind": "feature", "layer": 0, "influence": 1, "index": 0}], "counts": {"layers": 2}}
    before_reply = {"step": 9, "elements": [
        [0, 0, 0, 0, .1, .2 if selection_changed else .1, token, token - 1] for token in range(1, 11)]}
    after_reply = dict(before_reply, step=14)
    session = Mock(out=tmp_path, run_id="circuit-trace", program_evidence={"before": {}}, quality_override=None)
    session.read.return_value = before_reply
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


@pytest.mark.parametrize("override", [False, True])
def test_instrument_uses_three_panels_and_confirmed_bindings(override):
    from unittest.mock import Mock
    import jsonschema
    record = failed_quality_override()[1] if override else None
    session = Mock(quality_override=record)
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
    assert len(readouts) == (8 if override else 6)
    if override:
        assert record["dictionary_fingerprint"] in readouts["method_dictionary_identity"]
        text = readouts["method_dictionary_quality_override"]
        assert "Founder override" in text and "Quality remains failed" in text
        assert all(f"{record[key]:.9f}" in text for key in ("fvu", "dead_fraction", "cross_entropy_delta"))
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
    assert nodes["instrument"]["ratio"] == (.90 if override else .95)
    assert nodes["body"]["children"] == ["plots", "examples"]
    assert nodes["plots"]["ratio"] == .67
    assert nodes["status_body"]["children"] == ["clamp_status", "body"]
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
    assert set(groups["methods"]["children"]) == {"method_details"} | (set(readouts) - {"clamp_status"})


def test_bridge_expands_and_collapses_method_details():
    from unittest.mock import Mock, patch
    session = Mock(quality_override=None)
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
    if model != "tiny-lm" and env.get("CALIPER_QWEN_QUALITY_OVERRIDE"):
        command.extend(["--quality-override", env["CALIPER_QWEN_QUALITY_OVERRIDE"]])
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
