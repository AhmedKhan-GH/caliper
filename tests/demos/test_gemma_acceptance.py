"""Gemma release gates without a model, device, or gated weights."""
from copy import deepcopy

import pytest

from test_circuit_trace import qualification_reply
from _checks import check, finish
from _mcp import Failure
from _qualification import (GEMMA_HASH, GEMMA_ID, GEMMA_L0, GEMMA_REVISION,
                             GEMMA_SCOPE_REVISION, GEMMA_SELECTION_REVISION,
                             MODEL_REVISION, dictionary_qualification)
from run import acceptance_profile


def gemma_reply():
    response = qualification_reply()
    metadata, index, identity = response["metadata"], response["index"], response["identity"]
    source = {key: deepcopy(metadata["training"][key]) for key in
              ("dataset_id", "dataset_config", "dataset_revision", "split", "source_id", "partitions")}
    source.update(dataset_id="HuggingFaceFW/fineweb-edu", dataset_config="sample-10BT",
                  dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
    metadata.update(model_id=GEMMA_ID, revision=GEMMA_REVISION, model_hash=GEMMA_HASH,
                    layers=26, width=2304, latents=16384, activation="jumprelu", k=None,
                    evaluation_source=source,
                    training={"origin": "imported", "local_training": False, "tokens": 0,
                              "partitions": {}, "publisher_training_disjointness": "unknown"})
    imported = {
        "kind": "gemma_scope", "license": "CC-BY-4.0",
        "source": {"repo": "google/gemma-scope-2b-pt-transcoders", "revision": GEMMA_SCOPE_REVISION},
        "selection": {"repo": "mntss/gemma-scope-transcoders", "revision": GEMMA_SELECTION_REVISION,
                      "file": "config.yaml"},
        "original_input": "pure_rms_normalized_pre_mlp_before_learned_gain",
        "input_conversion": "W_enc.T / (1 + pre_feedforward_layernorm.weight)[None, :]",
        "output": "after_post_feedforward_layernorm", "output_conversion": "none",
        "layers": [{"layer": i, "width": 2304, "latents": 16384, "average_l0": l0,
                    "source_file": f"layer_{i}/width_16k/average_l0_{l0}/params.npz",
                    "file": f"layer_{i:02d}.safetensors", "source_sha256": "e" * 64,
                    "sha256": "f" * 64, "input_gain_hash": "sha256:" + "1" * 64}
                   for i, l0 in enumerate(GEMMA_L0)]}
    metadata["import_source"] = imported
    response["import_binding"] = {
        "version": 1, "model_hash": GEMMA_HASH,
        "layers": [{key: row[key] for key in ("layer", "file", "sha256", "input_gain_hash")}
                   for row in imported["layers"]],
    }
    metadata["quality"].update(fvu=.3, per_layer_fvu=[.3] * 26,
                               per_layer_dead_fraction=[.15] * 26, all_mlps_replaced=26,
                               evaluation_source=deepcopy(source), publisher_training_overlap="unknown")
    response["model_binding"] = {"version": 2, "model_hash": GEMMA_HASH,
                                 "tensor_hash": "2" * 64, "forward_settings_hash": "3" * 64}
    metadata["quality"]["model_binding"] = deepcopy(response["model_binding"])
    identity["model_hash"] = index["model_hash"] = GEMMA_HASH
    index.update(layers=26, features=16384,
                 provenance={**deepcopy(source), "partition": "index", "publisher_training_overlap": "unknown"})
    return response


def test_gemma_requires_its_own_identity_and_preserves_unknown_training():
    response = gemma_reply()
    original = deepcopy(response)
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"] and result["status"] == "passed" and result["scope"] == "all_layers"
    assert response == original
    assert not dictionary_qualification(response)["qualified"]
    assert not dictionary_qualification(qualification_reply(), family="gemma")["qualified"]


@pytest.mark.parametrize("metric,value", [("fvu", .300001), ("dead_fraction", .150001),
                                           ("cross_entropy_delta", .500001)])
def test_gemma_retains_original_quality_thresholds(metric, value):
    response = gemma_reply()
    quality = response["metadata"]["quality"]
    quality[metric] = value
    if metric == "cross_entropy_delta":
        quality["replaced_cross_entropy"] = quality["original_cross_entropy"] + value
    else:
        quality["per_layer_" + metric] = [value] * 26
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("fault", ["model", "source_revision", "selection", "l0", "missing_layer",
                                    "norm", "gain_hash", "trained", "corpus", "quality_source",
                                    "index_source", "index_overlap", "partial_quality", "partial_index"])
def test_gemma_rejects_stale_or_incomplete_release_evidence(fault):
    response = gemma_reply()
    metadata = response["metadata"]
    source = metadata["import_source"]
    if fault == "model":
        metadata["revision"] = "0" * 40
    elif fault == "source_revision":
        source["source"]["revision"] = "0" * 40
    elif fault == "selection":
        source["selection"]["revision"] = "0" * 40
    elif fault == "l0":
        source["layers"][11]["average_l0"] = 58
    elif fault == "missing_layer":
        source["layers"].pop()
    elif fault == "norm":
        source["output"] = "before_post_feedforward_layernorm"
    elif fault == "gain_hash":
        source["layers"][0]["input_gain_hash"] = "sha256:missing"
    elif fault == "trained":
        metadata["training"]["local_training"] = True
    elif fault == "corpus":
        metadata["evaluation_source"]["dataset_revision"] = "0" * 40
    elif fault == "quality_source":
        metadata["quality"]["evaluation_source"]["source_id"] = "different"
    elif fault == "index_source":
        response["index"]["provenance"]["source_id"] = "different"
    elif fault == "index_overlap":
        response["index"]["provenance"]["publisher_training_overlap"] = "excluded"
    elif fault == "partial_quality":
        metadata["quality"]["evaluated_tokens"] -= 1
    else:
        response["index"]["tokens"] -= 1
    assert not dictionary_qualification(response, family="gemma")["qualified"]


def test_qwen_override_cannot_qualify_gemma():
    from test_circuit_trace import failed_quality_override
    _, override = failed_quality_override()
    assert not dictionary_qualification(gemma_reply(), family="gemma", quality_override=override)["qualified"]


@pytest.mark.parametrize("fault", ["missing", "file_hash", "gain_hash", "reforged_record"])
def test_gemma_requires_current_import_evidence(fault):
    response = gemma_reply()
    if fault == "missing":
        del response["import_binding"]
    elif fault == "reforged_record":
        response["metadata"]["import_source"]["layers"][0]["sha256"] = "a" * 64
    else:
        field = "sha256" if fault == "file_hash" else "input_gain_hash"
        response["import_binding"]["layers"][0][field] = ("sha256:" if fault == "gain_hash" else "") + "b" * 64
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("location", ["quality", "live"])
@pytest.mark.parametrize("fault", ["missing", "tensor_hash", "forward_settings_hash", "legacy",
                                    "model_hash", "uppercase", "extra", "version_type"])
def test_gemma_requires_exact_quality_and_live_model_binding(location, fault):
    response = gemma_reply()
    target = response["metadata"]["quality"] if location == "quality" else response
    if fault == "missing":
        del target["model_binding"]
    elif fault == "legacy":
        del target["model_binding"]["forward_settings_hash"]
    elif fault == "model_hash":
        target["model_binding"][fault] = "hf:google/gemma-2-2b@main"
    elif fault == "uppercase":
        target["model_binding"]["tensor_hash"] = "A" * 64
    elif fault == "extra":
        target["model_binding"]["trusted"] = True
    elif fault == "version_type":
        target["model_binding"]["version"] = 2.0
    else:
        target["model_binding"][fault] = "9" * 64
    assert not dictionary_qualification(response, family="gemma")["qualified"]


def test_qwen_retains_legacy_quality_evidence_policy():
    response = qualification_reply()
    assert "model_binding" not in response["metadata"]["quality"]
    assert dictionary_qualification(response)["qualified"]


def test_gemma_finish_keeps_causal_failure_and_separate_model_result():
    result = {"mode": "gemma", "dictionary_evidence": gemma_reply(), "pixel_evidence_collected": True,
              "checks": [check("validation", False, {"correct_signs": 6, "spearman": .539},
                               "10 features, >=8 correct signs, Spearman >=0.6")]}
    assert finish(result) == 1 and not result["gemma_acceptance"]
    assert result["failed_checks"] == ["validation"]
    # finish aggregates checks from the internal driver. Each check owns its measurements.
    result["checks"][0] = check("validation", True, {"correct_signs": 8, "spearman": .6},
                                "10 features, >=8 correct signs, Spearman >=0.6")
    assert finish(result) == 0 and result["gemma_acceptance"]
    assert not result["qwen_acceptance"] and not result["qwen_full_mlp_acceptance"]


def test_memory_budget_requires_the_exact_gemma_snapshot(tmp_path):
    gemma = tmp_path / "models--google--gemma-2-2b" / "snapshots" / GEMMA_REVISION
    qwen = tmp_path / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / MODEL_REVISION
    assert acceptance_profile(str(gemma)) == ("gemma", 28_000_000_000)
    assert acceptance_profile(str(qwen)) == ("qwen", 12_000_000_000)
    assert acceptance_profile("tiny-lm") == ("fixture", 12_000_000_000)
    with pytest.raises(Failure, match="exact supported"):
        acceptance_profile(str(qwen.with_name(GEMMA_REVISION)))
