"""Release qualification from the live feature.dictionary response."""
from __future__ import annotations

import json
import math
import re

GEMMA_ID = "google/gemma-2-2b"
GEMMA_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
GEMMA_HASH = f"hf:{GEMMA_ID}@{GEMMA_REVISION}"
GEMMA_SCOPE_REVISION = "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"
GEMMA_SELECTION_REVISION = "9250a2d4860ce5ed5c96c14d5882b7d8162809a3"
GEMMA_L0 = (76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value, expected):
    return type(value) is int and value == expected


def _same_json(actual, expected):
    """Compare evidence without boolean or floating-point integer aliases."""
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def _partitions(training):
    partitions = training["partitions"]
    _require(isinstance(training.get("source_id"), str) and bool(training["source_id"]),
             "The held-out source identity is absent.")
    intervals = [partitions[name] for name in ("train", "quality", "index")]
    _require(partitions.get("algorithm") == "sha256-utf8-text-mod10000-v1"
             and intervals == [[0, 9400], [9400, 9500], [9500, 10000]]
             and all(type(n) is int for row in intervals for n in row)
             and all(type(partitions.get(key)) is int and partitions[key] > 0
                     for key in ("quality_tokens", "index_tokens")),
             "Quality and index documents must use the held-out partitions.")


def _gemma_provenance(metadata):
    """Bind the published Scope selection and keep publisher training unknown."""
    _require(_integer(metadata.get("width"), 2304) and _integer(metadata.get("latents"), 16384)
             and metadata.get("activation") == "jumprelu" and "k" in metadata and metadata["k"] is None,
             "Gemma requires the complete JumpReLU dictionary dimensions.")
    imported = metadata["import_source"]
    _require(imported.get("kind") == "gemma_scope" and imported.get("license") == "CC-BY-4.0"
             and imported.get("source") == {"repo": "google/gemma-scope-2b-pt-transcoders",
                                            "revision": GEMMA_SCOPE_REVISION}
             and imported.get("selection") == {"repo": "mntss/gemma-scope-transcoders",
                 "revision": GEMMA_SELECTION_REVISION, "file": "config.yaml"},
             "Gemma Scope source or selection differs from the pinned set.")
    layers = imported["layers"]
    _require(isinstance(layers, list) and len(layers) == 26, "Gemma Scope requires every source layer.")
    for i, (row, l0) in enumerate(zip(layers, GEMMA_L0)):
        _require(_integer(row.get("layer"), i) and _integer(row.get("width"), 2304)
                 and _integer(row.get("latents"), 16384) and _integer(row.get("average_l0"), l0)
                 and row.get("source_file") == f"layer_{i}/width_16k/average_l0_{l0}/params.npz"
                 and row.get("file") == f"layer_{i:02d}.safetensors"
                 and _digest(row.get("source_sha256")) and _digest(row.get("sha256"))
                 and isinstance(row.get("input_gain_hash"), str)
                 and row["input_gain_hash"].startswith("sha256:") and _digest(row["input_gain_hash"][7:]),
                 "Gemma Scope layer provenance is incomplete or differs from the selected set.")
    training = metadata["training"]
    _require(training.get("origin") == "imported" and training.get("local_training") is False
             and _integer(training.get("tokens"), 0) and training.get("partitions") == {}
             and training.get("publisher_training_disjointness") == "unknown",
             "Gemma Scope must retain its imported training provenance.")
    source = metadata["evaluation_source"]
    _require(all(source.get(k) == v for k, v in {
        "dataset_id": "HuggingFaceFW/fineweb-edu", "dataset_config": "sample-10BT",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "split": "train",
    }.items()), "Gemma evaluation requires the pinned held-out text source.")
    _partitions(source)
    return source


def _gemma_source(metadata, response):
    source = _gemma_provenance(metadata)
    imported = metadata["import_source"]
    _require("reference_input" not in metadata and "reference_conversion" not in imported
             and imported.get("original_input") == "pure_rms_normalized_pre_mlp_before_learned_gain"
             and imported.get("input_conversion") == "W_enc.T / (1 + pre_feedforward_layernorm.weight)[None, :]"
             and imported.get("output") == "after_post_feedforward_layernorm"
             and imported.get("output_conversion") == "none",
             "Gemma Scope normalization coordinates differ.")
    binding = response["import_binding"]
    _require(isinstance(binding, dict) and _integer(binding.get("version"), 1)
             and binding.get("model_hash") == GEMMA_HASH
             and _same_json(binding.get("layers"), [{key: row[key] for key in
                 ("layer", "file", "sha256", "input_gain_hash")} for row in imported["layers"]]),
             "Gemma Scope import does not match the current files and verified model gains.")
    _require(metadata["quality"].get("evaluation_source") == source
             and metadata["quality"].get("publisher_training_overlap") == "unknown",
             "Gemma quality must bind the evaluation source and unknown publisher overlap.")
    return source


def _gemma_model_binding(metadata, response):
    """Require exact model evidence from quality and the current live resources."""
    fields = {"version", "model_hash", "tensor_hash", "forward_settings_hash"}
    for binding in (metadata["quality"].get("model_binding"), response.get("model_binding")):
        _require(isinstance(binding, dict) and set(binding) == fields
                 and _integer(binding.get("version"), 2) and binding.get("model_hash") == GEMMA_HASH
                 and _digest(binding.get("tensor_hash")) and _digest(binding.get("forward_settings_hash")),
                 "Gemma requires complete verified model evidence for quality and live resources.")
    _require(metadata["quality"]["model_binding"] == response["model_binding"],
             "Gemma quality model tensors or forward settings differ from the current resources.")


def _gemma_index_context(index):
    """Require the corrected source positions for the index and examples."""
    policy = {"version": 1, "family": "gemma2", "prefix_ids": [2], "pad_id": 0,
              "model_context": 256, "content_context": 255,
              "segmentation": "contiguous_document_positions",
              "observations": "source_tokens_only", "replacement": "source_positions_only",
              "cross_entropy": "within_context_content_targets_including_bos_to_first"}
    examples = dict(policy, model_context=34, content_context=33)
    for actual, expected in ((index.get("context_policy"), policy),
                             (index.get("example_context_policy"), examples)):
        _require(isinstance(actual, dict) and actual == expected
                 and all(type(actual[key]) is type(value) for key, value in expected.items())
                 and all(type(value) is int for value in actual["prefix_ids"]),
                 "Gemma quality or index context policy differs from the corrected contract.")
    _require(_integer(index.get("context_length"), 256), "Gemma index context length differs.")


def _gemma_context(metadata, index):
    _gemma_index_context(index)
    spans = metadata["quality"].get("content_segmentation")
    capacity = metadata["quality"]["context_policy"]["content_context"]
    source_tokens = metadata["quality"]["source_tokens"]
    _require(isinstance(spans, dict) and set(spans) == {"contexts", "sha256", "encoding", "fields", "record"}
             and type(spans["contexts"]) is int
             and math.ceil(source_tokens / capacity) <= spans["contexts"] <= source_tokens
             and _digest(spans["sha256"])
             and spans["encoding"] == "sha256_ascii_json_rows_with_newlines"
             and spans["fields"] == ["document_id", "source_position", "content_length"]
             and spans["record"] == "evaluation-research-v1/state.json:content_segments",
             "Gemma quality content segmentation evidence is incomplete.")


def _gemma_research(metadata, response):
    """Validate the measured comparison sample and retain its resume budget."""
    quality = response.get("research_quality")
    _require(isinstance(quality, dict) and quality.get("protocol") == "research_v1"
             and _integer(quality.get("protocol_version"), 1)
             and _digest(quality.get("checkpoint_sha256")),
             "Gemma requires the research protocol and its committed cursor.")
    source_tokens = quality.get("source_tokens")
    budget = quality["settings"]["tokens"]
    _require(type(budget) is int and budget > 0, "Gemma source budget is invalid.")
    _require(type(source_tokens) is int and 0 < source_tokens <= budget
             and _integer(quality.get("evaluated_tokens"), source_tokens),
             "Gemma research requires a valid actual sample within its recorded source budget.")
    complete = source_tokens == budget
    _require(quality.get("source_budget_complete") is complete
             and quality.get("status") in (("passed", "failed") if complete else ("partial",))
             and quality.get("stop_reason") in (("complete",) if complete else ("signal 2", "signal 15")),
             "Gemma research status or stop reason contradicts its measured source budget.")
    policy = {"version": 1, "family": "gemma2", "bos_policy": "genuine_bos_prefix",
              "prefix_ids": [2], "pad_id": 0, "eos_ids": [1], "excluded_ids": [0, 1, 2],
              "model_context": 1024, "content_context": 1023,
              "segmentation": "contiguous_document_positions",
              "observations": "source_positions_excluding_bos_eos_pad",
              "replacement": "metric_positions_only_preserve_special_outputs",
              "cross_entropy": "attended_predictor_and_next_nonspecial_source_target_in_same_row",
              "first_token_loss": "bos_predicts_first_nonspecial_target"}
    actual = quality.get("context_policy")
    _require(_same_json(actual, policy),
             "Gemma research context policy differs from the published protocol.")
    settings = quality["settings"]
    protocol = settings["protocol"]
    _require(protocol.get("name") == "research_v1" and _integer(protocol.get("version"), 1)
             and _same_json(protocol.get("context_policy"), actual)
             and _same_json(settings.get("context_policy"), actual)
             and protocol.get("fvu") == "sum_squared_error_over_sum_squared_deviation_from_dataset_mean_vector_per_layer"
             and protocol.get("activity") == "positive_features_per_nonspecial_source_position"
             and protocol.get("loss") == "sum_nats_over_identical_targets_baseline_isolated_and_all_active"
             and protocol.get("statistics_dtype") == "cpu_float64"
             and _integer(protocol.get("source_budget"), budget)
             and _integer(settings.get("tokens"), budget)
             and _integer(settings.get("context"), 1024)
             and type(settings.get("batch_tokens")) is int and 1024 <= settings["batch_tokens"] <= 4096
             and type(settings.get("chunk_tokens")) is int and 1 <= settings["chunk_tokens"] <= 128,
             "Gemma research aggregation or source budget differs.")
    observations = quality.get("metric_observations")
    _require(type(observations) is int and 0 < observations <= source_tokens
             and _integer(quality.get("cross_entropy_tokens"), observations),
             "Gemma research target count must equal its nonspecial observations.")
    _require(quality.get("active_layers") == list(range(26))
             and all(type(i) is int for i in quality["active_layers"])
             and _same_json(settings.get("active_layers"), quality["active_layers"])
             and quality.get("scope") == settings.get("scope") == "all_layers"
             and _integer(quality.get("all_mlps_replaced"), 26)
             and _integer(quality.get("replaced_mlps"), 26)
             and quality.get("per_layer_status") == ["active"] * 26,
             "Gemma research replacement scope is incomplete.")
    for field, maximum in (("fvu", None), ("dead_fraction", 1)):
        rows = quality.get("per_layer_" + field)
        _require(isinstance(rows, list) and len(rows) == 26
                 and all(_number(value) and value >= 0 and (maximum is None or value <= maximum) for value in rows),
                 "Gemma research has invalid per-layer reconstruction measurements.")
        _require(_number(quality.get(field)) and quality[field] >= 0
                 and (maximum is None or quality[field] <= maximum),
                 "Gemma research aggregate is invalid.")
    for key in ("per_layer_cross_entropy", "per_layer_cross_entropy_delta", "per_layer_l0"):
        _require(isinstance(quality.get(key), list) and len(quality[key]) == 26
                 and all(_number(value) for value in quality[key]), "Gemma lacks finite per-layer research measurements.")
    baseline, replaced = quality["original_cross_entropy"], quality["replaced_cross_entropy"]
    delta = quality["cross_entropy_delta"]
    _require(_number(baseline) and baseline >= 0 and _number(replaced) and replaced >= 0
             and _number(delta),
             "Gemma research all-layer loss measurements are invalid.")
    _require(all(value >= 0 for value in quality["per_layer_cross_entropy"])
             and all(0 <= value <= 16384 for value in quality["per_layer_l0"]),
             "Gemma isolated loss measurements are invalid.")
    source = {**metadata["evaluation_source"], "partition": "quality"}
    _require(quality.get("partition") == "quality"
             and _same_json(quality.get("evaluation_source"), source) and _same_json(settings.get("source"), source),
             "Gemma research quality source differs from the held-out partition.")
    for key in ("dictionary_hash", "tensor_binding", "model_binding"):
        label = {"dictionary_hash": "dictionary identity", "tensor_binding": "dictionary tensor",
                 "model_binding": "model binding"}[key]
        _require(_same_json(settings.get(key), quality.get(key)), f"Gemma research settings have stale {label} evidence.")
    # The existing source validator expects the source identity without its partition tag.
    return {**metadata, "quality": {**quality, "evaluation_source": metadata["evaluation_source"]}}


def _gemma_dictionary_binding(metadata, response):
    """Require the exact dictionary tensors that produced the quality metrics."""
    for binding in (metadata["quality"].get("tensor_binding"), response.get("tensor_binding")):
        _require(isinstance(binding, dict) and set(binding) == {"version", "tensor_hash"}
                 and _integer(binding.get("version"), 1) and _digest(binding.get("tensor_hash")),
                 "Gemma requires complete dictionary tensor evidence for quality and live resources.")
    _require(metadata["quality"]["tensor_binding"] == response["tensor_binding"],
             "Gemma quality dictionary tensors differ from the current resources.")


def dictionary_qualification(response, *, family="gemma", replication_profile=None):
    """Refuse absent or stale evidence without model, tensor, or file access."""
    if replication_profile is not None:
        from _published_qualification import published_dictionary_qualification
        return published_dictionary_qualification(response, family=family, profile=replication_profile)
    result = {"qualified": False, "scope": None, "status": "unverified"}
    try:
        _require(isinstance(response, dict), "The feature.dictionary response is absent.")
        _require(family == "gemma", "Unsupported acceptance model family.")
        metadata = response["metadata"]
        metadata = _gemma_research(metadata, response)
        result.update(scope="all_layers", status="unverified",
                      reported_status=metadata["quality"].get("status"))
        index, identity = response["index"], response["identity"]
        expected_id, revision, model_hash, layers = (GEMMA_ID, GEMMA_REVISION, GEMMA_HASH, 26)
        _require(metadata.get("model_id") == expected_id and metadata.get("revision") == revision
                 and metadata.get("model_hash") == identity.get("model_hash") == model_hash
                 and _integer(metadata.get("layers"), layers), "The current model identity differs from the pinned family.")
        _require(all(_digest(identity.get(key)) for key in
                     ("dictionary_hash", "dictionary_fingerprint", "index_hash", "tokenizer_hash")),
                 "Current dictionary and index identities are absent or invalid.")
        _require(all(index.get(key) == identity[key] for key in
                     ("model_hash", "dictionary_hash", "index_hash", "tokenizer_hash"))
                 and index.get("layers") == metadata["layers"] and index.get("features") == metadata["latents"],
                 "The index does not identify the current model and dictionary.")
        _require(index.get("complete") is True and type(index.get("tokens")) is int and index["tokens"] > 0
                 and type(index.get("token_budget")) is int and index["token_budget"] >= index["tokens"],
                 "Acceptance requires a complete index with valid source counts.")
        training = _gemma_source(metadata, response)
        provenance = index["provenance"]
        _partitions(training)
        _require(_integer(index.get("layers"), layers) and _integer(index.get("features"), 16384)
                 and metadata["quality"].get("dictionary_hash") == identity["dictionary_fingerprint"],
                 "Gemma research or index dictionary identity differs from the current resources.")
        _gemma_model_binding(metadata, response)
        _gemma_dictionary_binding(metadata, response)
        _gemma_context(metadata, index)
        _require(_same_json(provenance.get("partitions"), training["partitions"]),
                 "Gemma index partition types differ from the held-out source.")
        _require(provenance.get("publisher_training_overlap") == "unknown",
                 "Gemma index must retain unknown publisher training overlap.")
        _require(provenance.get("partition") == "index" and all(provenance.get(k) == training[k] for k in
                 ("dataset_id", "dataset_config", "dataset_revision", "split", "partitions")),
                 "The index source or held-out partition differs.")
        _require(provenance.get("source_id") == training["source_id"], "The index source identity differs.")
        result.update(qualified=True, status="passed", identity=dict(identity))
        quality = metadata["quality"]
        result.update(status="research_reported", quality_gates_blocking=False,
                      historical_quality_status=quality["status"], research_protocol="research_v1",
                      source_tokens=quality["source_tokens"], evaluated_tokens=quality["evaluated_tokens"],
                      metric_observations=quality["metric_observations"],
                      cross_entropy_tokens=quality["cross_entropy_tokens"],
                      source_budget=quality["settings"]["tokens"], source_budget_complete=quality["source_budget_complete"],
                      stop_reason=quality["stop_reason"], checkpoint_sha256=quality["checkpoint_sha256"],
                      acceptance_policy="research_comparison_only",
                      acceptance_reason="Decision 15 reports sample size and historical quality without numerical acceptance limits.",
                      index_measurements={key: index.get(key) for key in ("tokens", "token_budget", "documents")},
                      aggregate_discrepancies={
                          "fvu": abs(quality["fvu"] - sum(quality["per_layer_fvu"]) / 26),
                          "dead_fraction": abs(quality["dead_fraction"] - sum(quality["per_layer_dead_fraction"]) / 26),
                          "cross_entropy_delta": abs(quality["cross_entropy_delta"] -
                              (quality["replaced_cross_entropy"] - quality["original_cross_entropy"])),
                          "per_layer_cross_entropy_delta": [abs(delta - (loss - quality["original_cross_entropy"]))
                              for delta, loss in zip(quality["per_layer_cross_entropy_delta"], quality["per_layer_cross_entropy"])]})
        differences = result["aggregate_discrepancies"]
        _require(all(_number(value) for value in [differences["fvu"], differences["dead_fraction"],
                     differences["cross_entropy_delta"], *differences["per_layer_cross_entropy_delta"]]),
                 "Quality aggregate discrepancies are nonfinite.")
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        result.update(qualified=False, reason=str(error))
    return result
