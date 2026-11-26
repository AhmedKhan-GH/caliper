"""Gemma release gates without a model, device, or gated weights."""
from copy import deepcopy

import pytest

from _checks import check, finish
from _mcp import Failure
from _qualification import (GEMMA_HASH, GEMMA_ID, GEMMA_L0, GEMMA_REVISION,
                             GEMMA_SCOPE_REVISION, GEMMA_SELECTION_REVISION,
                             dictionary_qualification)
from run import acceptance_profile


def qualification_reply():
    """Synthetic metadata exercises gates without model or index allocation."""
    source = {"dataset_id": "test-source", "dataset_config": "test", "dataset_revision": "revision",
                "split": "train", "source_id": "test-source:revision",
                "partitions": {"algorithm": "sha256-utf8-text-mod10000-v1", "train": [0, 9400],
                               "quality": [9400, 9500], "index": [9500, 10000],
                               "quality_tokens": 200_000, "index_tokens": 2_000_000}}
    quality = {"status": "passed", "evaluated_tokens": 200_000, "partition": "quality", "stop_reason": "complete",
               "dictionary_hash": "a" * 64, "all_mlps_replaced": 26,
               "fvu": .4, "dead_fraction": .15, "cross_entropy_delta": .5,
               "per_layer_fvu": [.4] * 26, "per_layer_dead_fraction": [.15] * 26,
               "cross_entropy_tokens": 199_000, "original_cross_entropy": 2., "replaced_cross_entropy": 2.5}
    metadata = {"model_id": GEMMA_ID, "revision": GEMMA_REVISION, "model_hash": GEMMA_HASH,
                "layers": 26, "latents": 16384, "evaluation_source": source, "quality": quality}
    identity = {"model_hash": GEMMA_HASH, "dictionary_hash": "b" * 64,
                "dictionary_fingerprint": "a" * 64, "index_hash": "c" * 64, "tokenizer_hash": "d" * 64}
    index = {k: identity[k] for k in ("model_hash", "dictionary_hash", "index_hash", "tokenizer_hash")}
    index.update(complete=True, tokens=2_000_000, token_budget=2_000_000, layers=26, features=16384,
                 provenance={k: deepcopy(source[k]) for k in
                             ("dataset_id", "dataset_config", "dataset_revision", "split", "source_id", "partitions")})
    index["provenance"]["partition"] = "index"
    response = {"metadata": metadata, "identity": identity, "index": index}
    return response


def gemma_reply(*, source_tokens=200000):
    response = qualification_reply()
    metadata, index, identity = response["metadata"], response["index"], response["identity"]
    source = {key: deepcopy(metadata["evaluation_source"][key]) for key in
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
    response["tensor_binding"] = {"version": 1, "tensor_hash": "4" * 64}
    metadata["quality"]["tensor_binding"] = deepcopy(response["tensor_binding"])
    policy = {"version": 1, "family": "gemma2", "prefix_ids": [2], "pad_id": 0,
              "model_context": 256, "content_context": 255,
              "segmentation": "contiguous_document_positions", "observations": "source_tokens_only",
              "replacement": "source_positions_only",
              "cross_entropy": "within_context_content_targets_including_bos_to_first"}
    metadata["quality"].update(context_policy=deepcopy(policy), content_segmentation={
        "contexts": 1000, "sha256": "6" * 64, "encoding": "sha256_ascii_json_rows_with_newlines",
        "fields": ["document_id", "source_position", "content_length"],
        "record": "evaluation/state.json:content_segments"})
    index.update(context_length=256, context_policy=deepcopy(policy),
                 example_context_policy=dict(policy, model_context=34, content_context=33))
    identity["model_hash"] = index["model_hash"] = GEMMA_HASH
    index.update(layers=26, features=16384,
                 provenance={**deepcopy(source), "partition": "index", "publisher_training_overlap": "unknown"})
    research = deepcopy(metadata["quality"])
    research_policy = {"version": 1, "family": "gemma2", "bos_policy": "genuine_bos_prefix",
        "prefix_ids": [2], "pad_id": 0, "eos_ids": [1], "excluded_ids": [0, 1, 2],
        "model_context": 1024, "content_context": 1023, "segmentation": "contiguous_document_positions",
        "observations": "source_positions_excluding_bos_eos_pad",
        "replacement": "metric_positions_only_preserve_special_outputs",
        "cross_entropy": "attended_predictor_and_next_nonspecial_source_target_in_same_row",
        "first_token_loss": "bos_predicts_first_nonspecial_target"}
    research.update(protocol="research_v1", protocol_version=1, source_budget_complete=True,
        source_tokens=200000, metric_observations=199000, cross_entropy_tokens=199000,
        scope="all_layers", replaced_mlps=26,
        thresholds={"fvu": .30, "dead_fraction": .15, "cross_entropy_delta": .5, "evaluated_tokens": 200000},
        checkpoint_sha256="7" * 64, context_policy=research_policy, active_layers=list(range(26)),
        per_layer_status=["active"] * 26, per_layer_l0=[20.] * 26,
        per_layer_cross_entropy=[research["original_cross_entropy"] + .1] * 26,
        per_layer_cross_entropy_delta=[.1] * 26,
        evaluation_source={**deepcopy(source), "partition": "quality"})
    research["content_segmentation"]["record"] = "evaluation-research-v1/state.json:content_segments"
    research["settings"] = {"protocol": {"name": "research_v1", "version": 1,
        "context_policy": deepcopy(research_policy), "source_budget": 200000,
        "fvu": "sum_squared_error_over_sum_squared_deviation_from_dataset_mean_vector_per_layer",
        "activity": "positive_features_per_nonspecial_source_position",
        "loss": "sum_nats_over_identical_targets_baseline_isolated_and_all_active",
        "statistics_dtype": "cpu_float64"}, "context_policy": deepcopy(research_policy),
        "tokens": 200000, "context": 1024, "batch_tokens": 1024, "chunk_tokens": 64,
        "scope": "all_layers", "active_layers": list(range(26)), "source": deepcopy(research["evaluation_source"]),
        **{key: deepcopy(research[key]) for key in ("dictionary_hash", "tensor_binding", "model_binding")}}
    if source_tokens != 200000:
        research.update(source_tokens=source_tokens, evaluated_tokens=source_tokens,
                        metric_observations=source_tokens - 1000, cross_entropy_tokens=source_tokens - 1000,
                        source_budget_complete=False, status="partial", stop_reason="signal 2")
    response["research_quality"] = research
    return response


def test_gemma_requires_its_own_identity_and_preserves_unknown_training():
    response = gemma_reply()
    original = deepcopy(response)
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"] and result["status"] == "research_reported" and result["scope"] == "all_layers"
    assert response == original
    assert not dictionary_qualification(qualification_reply(), family="gemma")["qualified"]


@pytest.mark.parametrize("source_tokens", [50000, 51237, 199999, 200000])
@pytest.mark.parametrize("failed_metrics", [False, True])
def test_gemma_comparison_reports_actual_sample_and_original_status(source_tokens, failed_metrics):
    response = gemma_reply(source_tokens=source_tokens)
    quality = response["research_quality"]
    if failed_metrics:
        quality.update(fvu=.5, per_layer_fvu=[.5] * 26)
        if source_tokens == 200000:
            quality["status"] = "failed"
    original = deepcopy(response)
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"] and result["status"] == "research_reported"
    assert result["reported_status"] == result["historical_quality_status"] == quality["status"]
    assert result["acceptance_policy"] == "research_comparison_only"
    assert not result["quality_gates_blocking"]
    assert result["acceptance_reason"] == (
        "Decision 15 reports sample size and historical quality without numerical acceptance limits.")
    assert result["source_tokens"] == result["evaluated_tokens"] == source_tokens
    assert result["metric_observations"] == result["cross_entropy_tokens"] == source_tokens - 1000
    assert result["source_budget"] == quality["settings"]["tokens"] == 200000
    assert result["source_budget_complete"] is (source_tokens == 200000)
    assert result["checkpoint_sha256"] == quality["checkpoint_sha256"]
    assert result["stop_reason"] == quality["stop_reason"]
    assert response == original


@pytest.mark.parametrize("source_tokens", [50000, 52231, 199999, 200000])
@pytest.mark.parametrize("stop_reason", ["signal 2", "signal 15"])
def test_gemma_clean_stop_requires_a_partial_sample(source_tokens, stop_reason):
    response = gemma_reply(source_tokens=source_tokens)
    response["research_quality"]["stop_reason"] = stop_reason
    original = deepcopy(response)
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"] is (source_tokens < 200000)
    if result["qualified"]:
        assert result["reported_status"] == "partial"
        assert result["source_budget_complete"] is False
        assert result["source_tokens"] == source_tokens
        assert result["stop_reason"] == stop_reason
        assert result["checkpoint_sha256"] == response["research_quality"]["checkpoint_sha256"]
    assert response == original


@pytest.mark.parametrize("source_tokens", [50000, 200000])
def test_gemma_acceptance_result_preserves_comparison_disclosure(source_tokens):
    result = {"mode": "gemma", "completed": True, "dictionary_evidence": gemma_reply(source_tokens=source_tokens),
              "pixel_evidence_collected": True, "checks": [check("validation", True, 8, ">=8")]}
    assert finish(result) == 1 and not result["gemma_acceptance"]
    qualification = result["dictionary_qualification"]
    assert qualification["source_tokens"] == source_tokens
    assert qualification["reported_status"] == ("partial" if source_tokens < 200000 else "passed")
    assert qualification["acceptance_policy"] == "research_comparison_only"
    assert "Decision 15" in qualification["acceptance_reason"]


@pytest.mark.parametrize("source_tokens", [50000, 51237, 200000])
def test_gemma_segment_bounds_use_measured_sample(source_tokens):
    response = gemma_reply(source_tokens=source_tokens)
    spans = response["research_quality"]["content_segmentation"]
    minimum = (source_tokens + 1022) // 1023
    for count, valid in ((minimum, True), (minimum - 1, False), (source_tokens, True), (source_tokens + 1, False)):
        spans["contexts"] = count
        assert dictionary_qualification(response, family="gemma")["qualified"] is valid


@pytest.mark.parametrize("source_tokens", [50000, 200000])
@pytest.mark.parametrize("field,value", [
    ("source_tokens", 49999), ("source_tokens", 200001), ("source_tokens", 50000.), ("source_tokens", True),
    ("evaluated_tokens", 200000.), ("evaluated_tokens", True), ("evaluated_tokens", 49999),
    ("metric_observations", 0), ("metric_observations", -1), ("metric_observations", 200001),
    ("metric_observations", 49999.), ("metric_observations", True),
    ("cross_entropy_tokens", 200001), ("cross_entropy_tokens", True), ("cross_entropy_tokens", 49999.),
    ("source_budget_complete", 0), ("source_budget_complete", 1),
    ("stop_reason", "signal 9"), ("stop_reason", "source_exhausted"), ("stop_reason", None),
    ("status", "invalid_partial"), ("status", None), ("partition", "index"),
    ("checkpoint_sha256", None), ("checkpoint_sha256", "A" * 64),
    ("protocol_version", True), ("protocol_version", 1.),
    ("all_mlps_replaced", 26.), ("replaced_mlps", True), ("scope", "upper_half"),
    ("original_cross_entropy", -1),
    ("original_cross_entropy", True), ("replaced_cross_entropy", -1),
    ("fvu", float("inf")), ("dead_fraction", float("nan")),
    ("cross_entropy_delta", float("nan")), ("replaced_cross_entropy", float("inf")),
])
def test_gemma_rejects_invalid_sample_scalars(source_tokens, field, value):
    response = gemma_reply(source_tokens=source_tokens)
    response["research_quality"][field] = value
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("source_tokens", [50000, 200000])
@pytest.mark.parametrize("fault", ["completion", "status", "stop_reason", "observation_bound", "target_mismatch",
                                    "forged_budget", "missing_cursor", "dictionary_identity"])
def test_gemma_rejects_inconsistent_sample_evidence(source_tokens, fault):
    response = gemma_reply(source_tokens=source_tokens)
    quality = response["research_quality"]
    complete = source_tokens == 200000
    if fault == "completion":
        quality["source_budget_complete"] = not complete
    elif fault == "status":
        quality["status"] = "partial" if complete else "passed"
    elif fault == "stop_reason":
        quality["stop_reason"] = "signal 2" if complete else "complete"
    elif fault == "observation_bound":
        quality["metric_observations"] = quality["cross_entropy_tokens"] = source_tokens + 1
    elif fault == "target_mismatch":
        quality["cross_entropy_tokens"] -= 1
    elif fault == "forged_budget":
        quality.update(source_budget_complete=True, status="passed", stop_reason="complete")
        quality["settings"]["tokens"] = 50000
        quality["settings"]["protocol"]["source_budget"] = 49999
        quality["thresholds"]["evaluated_tokens"] = 50000
    elif fault == "missing_cursor":
        del quality["checkpoint_sha256"]
    else:
        quality["dictionary_hash"] = quality["settings"]["dictionary_hash"] = "9" * 64
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("key", ["per_layer_fvu", "per_layer_dead_fraction", "per_layer_l0",
                                "per_layer_cross_entropy", "per_layer_cross_entropy_delta"])
@pytest.mark.parametrize("fault", ["absent", "short", "boolean", "nan", "infinity", "inconsistent"])
def test_gemma_partial_requires_all_layer_measurements(key, fault):
    response = gemma_reply(source_tokens=51237)
    quality = response["research_quality"]
    if fault == "absent":
        del quality[key]
    elif fault == "short":
        quality[key].pop()
    else:
        value = {"boolean": True, "nan": float("nan"), "infinity": float("inf"),
                 "inconsistent": 16385 if key == "per_layer_l0" else -1}[fault]
        quality[key][17] = value
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"] is (fault == "inconsistent" and key == "per_layer_cross_entropy_delta")


@pytest.mark.parametrize("location", ["context_policy", "settings", "protocol"])
@pytest.mark.parametrize("field,value", [
    ("version", True), ("pad_id", False), ("prefix_ids", [2.]), ("eos_ids", [True]),
    ("excluded_ids", [0, 2]), ("model_context", 1024.), ("content_context", 1023.),
    ("observations", "source_tokens_only"), ("replacement", "all_positions"),
    ("cross_entropy", "all_targets"), ("first_token_loss", "exclude_first"),
])
def test_gemma_partial_requires_exact_masks_and_context_types(location, field, value):
    response = gemma_reply(source_tokens=51237)
    quality = response["research_quality"]
    policy = (quality["context_policy"] if location == "context_policy" else
              quality["settings"]["context_policy"] if location == "settings" else
              quality["settings"]["protocol"]["context_policy"])
    policy[field] = value
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("path,value", [
    (("tokens",), 50000), (("tokens",), 200000.), (("context",), 1024.),
    (("batch_tokens",), True), (("batch_tokens",), 1023), (("batch_tokens",), 4097),
    (("chunk_tokens",), 0), (("chunk_tokens",), 129), (("chunk_tokens",), 64.),
    (("active_layers",), [False, *range(1, 26)]), (("scope",), "upper_half"),
    (("protocol", "source_budget"), 50000), (("protocol", "version"), True),
    (("protocol", "fvu"), "mean_context_ratios"), (("protocol", "statistics_dtype"), "float32"),
    (("protocol", "activity"), "all_positions"), (("protocol", "loss"), "mean_batch_loss"),
    (("source", "source_id"), "stale"), (("source", "partition"), "index"),
    (("source", "partitions", "quality"), [9400., 9500]),
    (("tensor_binding", "version"), True), (("model_binding", "version"), 2.),
])
def test_gemma_partial_requires_exact_resume_settings(path, value):
    response = gemma_reply(source_tokens=51237)
    target = response["research_quality"]["settings"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("location", ["quality", "live"])
@pytest.mark.parametrize("fault", ["missing", "version", "boolean_version", "hash", "different", "extra"])
def test_gemma_requires_exact_dictionary_tensor_evidence(location, fault):
    response = gemma_reply()
    owner = response["research_quality"] if location == "quality" else response
    if fault == "missing":
        del owner["tensor_binding"]
    else:
        field, value = {"version": ("version", 2), "boolean_version": ("version", True),
                        "hash": ("tensor_hash", "invalid"), "different": ("tensor_hash", "5" * 64),
                        "extra": ("unverified", True)}[fault]
        owner["tensor_binding"][field] = value
    result = dictionary_qualification(response, family="gemma")
    assert not result["qualified"] and "dictionary tensor" in result["reason"]


@pytest.mark.parametrize("metric,value", [("fvu", .300001), ("dead_fraction", .150001),
                                           ("cross_entropy_delta", .500001)])
def test_gemma_reports_historical_quality_status_without_a_numerical_gate(metric, value):
    response = gemma_reply()
    quality = response["research_quality"]
    quality[metric] = value
    if metric == "cross_entropy_delta":
        quality["replaced_cross_entropy"] = quality["original_cross_entropy"] + value
    else:
        quality["per_layer_" + metric] = [value] * 26
    quality["status"] = "failed"
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"] and not result["quality_gates_blocking"]
    assert result["historical_quality_status"] == "failed"
    quality["status"] = "passed"
    assert dictionary_qualification(response, family="gemma")["qualified"]


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
        response["research_quality"]["evaluation_source"]["source_id"] = "different"
    elif fault == "index_source":
        response["index"]["provenance"]["source_id"] = "different"
    elif fault == "index_overlap":
        response["index"]["provenance"]["publisher_training_overlap"] = "excluded"
    elif fault == "partial_quality":
        response["research_quality"]["evaluated_tokens"] -= 1
    else:
        response["index"]["complete"] = False
    assert not dictionary_qualification(response, family="gemma")["qualified"]


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
    target = response["research_quality"] if location == "quality" else response
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


@pytest.mark.parametrize("source_tokens", [51237, 200000])
def test_gemma_finish_refuses_legacy_causal_acceptance(source_tokens):
    result = {"mode": "gemma", "completed": True, "dictionary_evidence": gemma_reply(source_tokens=source_tokens),
              "pixel_evidence_collected": True,
              "checks": [check("validation", False, {"correct_signs": 6, "spearman": .539},
                               "10 features, >=8 correct signs, Spearman >=0.6")]}
    assert finish(result) == 1 and not result["gemma_acceptance"]
    assert result["failed_checks"] == ["validation", "published_profile", "external_replication_review"]
    # finish aggregates checks from the internal driver. Each check owns its measurements.
    result["checks"][0] = check("validation", True, {"correct_signs": 8, "spearman": .6},
                                "10 features, >=8 correct signs, Spearman >=0.6")
    assert finish(result) == 1 and not result["gemma_acceptance"]
    assert result["failed_checks"] == ["published_profile", "external_replication_review"]


def test_memory_budget_requires_the_exact_gemma_snapshot(tmp_path):
    gemma = tmp_path / "models--google--gemma-2-2b" / "snapshots" / GEMMA_REVISION
    assert acceptance_profile(str(gemma)) == ("gemma", 40_000_000_000)
    assert acceptance_profile("tiny-lm") == ("fixture", 12_000_000_000)
    with pytest.raises(Failure, match="exact supported"):
        acceptance_profile(str(gemma.with_name("unsupported")))


@pytest.mark.parametrize("owner", ["quality", "index", "examples"])
@pytest.mark.parametrize("fault", ["missing", "bos", "boolean", "segmentation", "replacement"])
def test_gemma_qualification_requires_corrected_context_policy(owner, fault):
    response = gemma_reply()
    parent = response["research_quality"] if owner == "quality" else response["index"]
    key = "example_context_policy" if owner == "examples" else "context_policy"
    if fault == "missing":
        del parent[key]
    else:
        field, value = {"bos": ("prefix_ids", []), "boolean": ("pad_id", False),
                        "segmentation": ("segmentation", "flat_content"),
                        "replacement": ("replacement", "all_positions")}[fault]
        parent[key][field] = value
    result = dictionary_qualification(response, family="gemma")
    assert not result["qualified"] and "context policy" in result["reason"]


@pytest.mark.parametrize("fault", ["missing", "hash", "count", "record"])
def test_gemma_qualification_requires_content_segmentation_evidence(fault):
    response = gemma_reply()
    quality = response["research_quality"]
    if fault == "missing":
        del quality["content_segmentation"]
    else:
        field, value = {"hash": ("sha256", "invalid"), "count": ("contexts", True),
                        "record": ("record", "different")}[fault]
        quality["content_segmentation"][field] = value
    result = dictionary_qualification(response, family="gemma")
    assert not result["qualified"] and "segmentation evidence" in result["reason"]


@pytest.mark.parametrize("fault", ["absent", "target_count", "context_count", "source_count", "layer_loss",
                                   "layer_delta", "metric_count", "threshold", "prefix", "cursor"])
def test_gemma_research_rejects_incomplete_or_inconsistent_measurements(fault):
    response = gemma_reply()
    quality = response["research_quality"]
    if fault == "absent":
        del response["research_quality"]
    elif fault == "context_count":
        quality["content_segmentation"]["contexts"] = 1
    elif fault == "layer_loss":
        quality["per_layer_cross_entropy"].pop()
    elif fault == "layer_delta":
        quality["per_layer_cross_entropy_delta"][3] += .1
    elif fault == "threshold":
        quality["thresholds"]["fvu"] = .4
    elif fault == "prefix":
        quality["context_policy"]["prefix_ids"] = [True]
    else:
        key, value = {"target_count": ("cross_entropy_tokens", 1), "source_count": ("source_tokens", 199999),
                      "metric_count": ("metric_observations", True), "cursor": ("checkpoint_sha256", "bad")}[fault]
        quality[key] = value
    assert dictionary_qualification(response, family="gemma")["qualified"] is (fault in ("layer_delta", "threshold"))


def test_legacy_quality_cannot_replace_missing_research_evidence():
    response = gemma_reply()
    assert response["metadata"]["quality"]["status"] == "passed"
    del response["research_quality"]
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("source_tokens", [51237, 200000])
def test_gemma_instrument_discloses_actual_comparison_without_blocking(source_tokens):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from run import instrument, instrument_layout

    response = gemma_reply(source_tokens=source_tokens)
    quality = response["research_quality"]
    quality.update(fvu=.5, per_layer_fvu=[.5] * 26)
    if source_tokens == 200000:
        quality["status"] = "failed"
    session = SimpleNamespace(family="gemma", mcp=Mock(), events=Mock())
    session.events.complete.return_value = response
    instrument(session)
    calls = session.mcp.body.call_args_list
    readout = next(call.kwargs for call in calls if call.args[0] == "control.create"
                   and call.kwargs["id"] == "method_quality_measurements")
    assert readout["options"]["format"] == (
        f"Research comparison uses {source_tokens:,} source positions ({quality['status']}). "
        "FVU 0.500000. Dead fraction 0.150000. All-layer loss rise 0.500000 nats. Quality bars await calibration.")
    for expanded in (False, True):
        instrument_layout(session, expanded=expanded)
        nodes = session.mcp.body.call_args.kwargs["layout"]["nodes"]
        methods = next(node for node in nodes if node["id"] == "methods")
        assert methods["children"].count("method_quality_measurements") == 1


@pytest.mark.parametrize("source_tokens", [49999, 200001])
def test_gemma_reports_a_coherent_sample_outside_former_bounds(source_tokens):
    response = gemma_reply(source_tokens=source_tokens)
    quality = response["research_quality"]
    quality["settings"]["tokens"] = max(200000, source_tokens + 1)
    quality["settings"]["protocol"]["source_budget"] = quality["settings"]["tokens"]
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"], result
    assert result["source_tokens"] == source_tokens


def test_gemma_partial_cannot_claim_failed_or_passed_quality():
    response = gemma_reply(source_tokens=51237)
    response["research_quality"].update(status="failed", fvu=.5, per_layer_fvu=[.5] * 26)
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("location", ["quality", "live"])
@pytest.mark.parametrize("binding", ["tensor_binding", "model_binding"])
def test_gemma_partial_keeps_current_resource_checks(location, binding):
    response = gemma_reply(source_tokens=51237)
    target = response["research_quality"] if location == "quality" else response
    target[binding]["tensor_hash"] = "9" * 64
    if location == "quality":
        response["research_quality"]["settings"][binding] = deepcopy(target[binding])
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("field,value", [("fvu", -1e-14), ("dead_fraction", -1e-14),
                                       ("dead_fraction", 1 + 1e-14)])
def test_gemma_aggregate_ranges_remain_validity_requirements(field, value):
    response = gemma_reply(source_tokens=51237)
    quality = response["research_quality"]
    quality[field] = value
    quality["per_layer_" + field] = [0 if value < 0 else 1] * 26
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("fault", ["import_boolean", "import_float", "index_layers", "index_features",
                                   "index_partition", "active_layers", "layer_status"])
def test_gemma_partial_rejects_scalar_aliases_in_resource_evidence(fault):
    response = gemma_reply(source_tokens=51237)
    quality = response["research_quality"]
    if fault.startswith("import_"):
        response["import_binding"]["layers"][0]["layer"] = False if fault == "import_boolean" else 0.
    elif fault in ("index_layers", "index_features"):
        response["index"][fault.removeprefix("index_")] = 26. if fault == "index_layers" else 16384.
    elif fault == "index_partition":
        response["index"]["provenance"]["partitions"]["train"][0] = False
    elif fault == "active_layers":
        quality["active_layers"][0] = False
    else:
        quality["per_layer_status"][0] = "error_only"
    assert not dictionary_qualification(response, family="gemma")["qualified"]


@pytest.mark.parametrize("prompt,probability,passed", [
    ("Fact: the capital of the state containing Dallas is", .409498900175, True),
    ("A custom fact", .40, True), ("A custom fact", .5, True),
    ("A custom fact", .500001, True),
])
def test_prompt_confidence_reports_actual_baseline_without_minimum(prompt, probability, passed):
    from _checks import prompt_confidence_check
    rows = [{"token": 2, "baseline": .99, "probability": .99},
            {"token": 7, "baseline": probability, "probability": .9}]
    gate = prompt_confidence_check(prompt, rows, 7)
    assert gate["required"] and gate["passed"] is passed
    assert gate["measured"]["baseline_probability"] == probability
    assert gate["measured"]["baseline_top"] == 7


@pytest.mark.parametrize("rows", [[], [{"token": 8, "baseline": .9}],
    [{"token": 7, "baseline": .9}] * 2,
    *[[{"token": 7, "baseline": value}] for value in (float("nan"), float("inf"), -0.1, 1.1, True)],
])
def test_prompt_confidence_refuses_missing_ambiguous_or_invalid_evidence(rows):
    from _checks import MODEL_PROMPT, prompt_confidence_check
    with pytest.raises(Failure, match="Prompt confidence"):
        prompt_confidence_check(MODEL_PROMPT, rows, 7)


def test_gemma_requires_index_source_identity():
    response = gemma_reply()
    response["index"]["provenance"].pop("source_id")
    assert not dictionary_qualification(response)["qualified"]


def test_gemma_cached_qualification_cannot_replace_evidence():
    result = {"mode": "gemma", "completed": True, "pixel_evidence_collected": True,
              "dictionary_qualification": {"qualified": True},
              "checks": [check("dictionary_qualification", True, True, True)]}
    assert finish(result) == 1
    assert not result["dictionary_qualification"]["qualified"]
    assert not result["accepted_replication"]


def test_unsupported_family_cannot_claim_acceptance():
    result = {"mode": "unsupported", "completed": True,
              "checks": [check("synthetic", True, True, True)]}
    assert finish(result) == 1
    assert not result["accepted_replication"]


def test_quality_aggregate_discrepancies_remain_measurements():
    response = gemma_reply()
    quality = response["research_quality"]
    quality["fvu"] = 90.
    quality["cross_entropy_delta"] = 80.
    result = dictionary_qualification(response, family="gemma")
    assert result["qualified"] and result["acceptance_policy"] == "research_comparison_only"
    assert result["aggregate_discrepancies"]["fvu"] == abs(90. - sum(quality["per_layer_fvu"]) / 26)
    assert result["aggregate_discrepancies"]["cross_entropy_delta"] == abs(
        80. - (quality["replaced_cross_entropy"] - quality["original_cross_entropy"]))
    assert not result["quality_gates_blocking"]
