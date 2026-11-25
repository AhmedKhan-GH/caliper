"""Release qualification from the live feature.dictionary response."""
from __future__ import annotations

import json
import math
import re

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_HASH = f"hf:{MODEL_ID}@{MODEL_REVISION}"
GEMMA_ID = "google/gemma-2-2b"
GEMMA_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
GEMMA_HASH = f"hf:{GEMMA_ID}@{GEMMA_REVISION}"
GEMMA_SCOPE_REVISION = "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"
GEMMA_SELECTION_REVISION = "9250a2d4860ce5ed5c96c14d5882b7d8162809a3"
GEMMA_L0 = (76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41)
QUALITY_TOKENS = 200_000
GEMMA_RESEARCH_SAMPLE_TOKENS = 50_000
INDEX_TOKENS = 2_000_000


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


def validate_quality_override(record):
    """Require an explicit decision bound to exact bytes and measured quality."""
    fields = {"version", "authority", "dictionary_fingerprint", "evaluated_tokens",
              "fvu", "dead_fraction", "cross_entropy_delta", "failed_gates"}
    _require(isinstance(record, dict) and set(record) == fields, "Invalid quality override fields.")
    _require(_integer(record["version"], 1) and record["authority"] == "founder"
             and _digest(record["dictionary_fingerprint"])
             and _integer(record["evaluated_tokens"], QUALITY_TOKENS), "Invalid quality override identity.")
    limits = {"fvu": .40, "dead_fraction": .15, "cross_entropy_delta": .5}
    _require(all(_number(record[key]) for key in limits)
             and record["fvu"] >= 0 and 0 <= record["dead_fraction"] <= 1,
             "Invalid quality override measurements.")
    failures = [key for key, limit in limits.items() if record[key] > limit]
    _require(bool(failures) and record["failed_gates"] == failures,
             "Quality override must name every failed quality gate.")
    return {**record, "failed_gates": list(failures)}


def _quality(metadata, fingerprint, *, fallback=False, failed=False, override=None, fvu_limit=.40):
    quality = metadata["quality"]
    layers = metadata["layers"]
    active = list(range(layers // 2, layers)) if fallback else list(range(layers))
    status = "failed" if failed or override is not None else "fallback_passed" if fallback else "passed"
    _require(quality.get("status") == status, "Quality status does not match the required result.")
    _require(_digest(fingerprint) and quality.get("dictionary_hash") == fingerprint,
             "Quality does not identify the exact dictionary.")
    _require(_integer(quality.get("evaluated_tokens"), QUALITY_TOKENS)
             and quality.get("partition") == "quality" and quality.get("stop_reason") == "complete",
             "Quality requires a complete evaluation of 200000 held-out tokens.")
    _require(quality.get("scope", "all_layers") == ("upper_half" if fallback else "all_layers"),
             "Quality replacement scope differs.")
    field = "replaced_mlps" if fallback else "all_mlps_replaced"
    _require(_integer(quality.get(field), len(active)), "Quality replacement coverage is incomplete.")
    if fallback:
        _require("all_mlps_replaced" not in quality and quality.get("active_layers") == active
                 and all(type(i) is int for i in quality["active_layers"])
                 and quality.get("per_layer_status") == ["active" if i in active else "error_only"
                                                         for i in range(layers)],
                 "Fallback quality must distinguish active layers from error-only layers.")
    for field, maximum in (("fvu", None), ("dead_fraction", 1)):
        rows = quality["per_layer_" + field]
        _require(isinstance(rows, list) and len(rows) == layers, "Quality lacks per-layer metrics.")
        _require(all(_number(rows[i]) and rows[i] >= 0 and (maximum is None or rows[i] <= maximum)
                     for i in active), "Quality has invalid per-layer metrics.")
        _require(all(rows[i] is None for i in range(layers) if i not in active),
                 "Fallback quality reports metrics for inactive layers.")
        mean = sum(rows[i] for i in active) / len(active)
        _require(_number(quality[field]) and math.isclose(quality[field], mean, rel_tol=1e-9, abs_tol=1e-12),
                 "Quality aggregate differs from its per-layer metrics.")
    delta = quality["cross_entropy_delta"]
    original, replaced = quality["original_cross_entropy"], quality["replaced_cross_entropy"]
    ce_tokens = quality["cross_entropy_tokens"]
    _require(type(ce_tokens) is int and 0 < ce_tokens <= QUALITY_TOKENS
             and _number(original) and original >= 0 and _number(replaced) and replaced >= 0
             and _number(delta) and math.isclose(delta, replaced - original, rel_tol=1e-9, abs_tol=1e-12),
             "Quality cross-entropy evidence is invalid.")
    passes = quality["fvu"] <= fvu_limit and quality["dead_fraction"] <= .15 and delta <= .5
    _require(passes != (failed or override is not None), "Quality metrics contradict the required status.")
    if override is not None:
        approved = validate_quality_override(override)
        _require(not fallback and approved["dictionary_fingerprint"] == fingerprint
                 and all(approved[key] == quality[key] for key in
                         ("evaluated_tokens", "fvu", "dead_fraction", "cross_entropy_delta")),
                 "Quality override differs from the exact dictionary or measured values.")


def _partitions(training):
    partitions = training["partitions"]
    _require(isinstance(training.get("source_id"), str) and bool(training["source_id"]),
             "The held-out source identity is absent.")
    intervals = [partitions[name] for name in ("train", "quality", "index")]
    _require(partitions.get("algorithm") == "sha256-utf8-text-mod10000-v1"
             and intervals == [[0, 9400], [9400, 9500], [9500, 10000]]
             and all(type(n) is int for row in intervals for n in row)
             and _integer(partitions.get("quality_tokens"), QUALITY_TOKENS)
             and _integer(partitions.get("index_tokens"), INDEX_TOKENS),
             "Quality and index documents must use the held-out partitions.")


def _completed_training(training, budget):
    _require(type(budget) is int and budget in (20_000_000, 40_000_000)
             and _integer(training.get("tokens"), budget)
             and training.get("stop_reason") == "token_budget",
             "Qwen requires a complete training budget of 20000000 or 40000000 tokens.")


def _gemma_source(metadata, response):
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
    _require(imported.get("original_input") == "pure_rms_normalized_pre_mlp_before_learned_gain"
             and imported.get("input_conversion") == "W_enc.T / (1 + pre_feedforward_layernorm.weight)[None, :]"
             and imported.get("output") == "after_post_feedforward_layernorm"
             and imported.get("output_conversion") == "none",
             "Gemma Scope normalization coordinates differ.")
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
    binding = response["import_binding"]
    _require(isinstance(binding, dict) and _integer(binding.get("version"), 1)
             and binding.get("model_hash") == GEMMA_HASH
             and _same_json(binding.get("layers"), [{key: row[key] for key in
                                            ("layer", "file", "sha256", "input_gain_hash")} for row in layers]),
             "Gemma Scope import does not match the current files and verified model gains.")
    training = metadata["training"]
    _require(training.get("origin") == "imported" and training.get("local_training") is False
             and _integer(training.get("tokens"), 0) and training.get("partitions") == {}
             and "fallback" not in training and training.get("publisher_training_disjointness") == "unknown",
             "Gemma Scope must retain its imported training provenance.")
    source = metadata["evaluation_source"]
    _require(all(source.get(k) == v for k, v in {
        "dataset_id": "HuggingFaceFW/fineweb-edu", "dataset_config": "sample-10BT",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "split": "train",
    }.items()), "Gemma evaluation requires the pinned held-out text source.")
    _partitions(source)
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


def _gemma_context(metadata, index):
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
    _require(type(source_tokens) is int and GEMMA_RESEARCH_SAMPLE_TOKENS <= source_tokens <= QUALITY_TOKENS
             and _integer(quality.get("evaluated_tokens"), source_tokens),
             "Gemma research requires the actual sample of 50000 through 200000 source positions.")
    complete = source_tokens == QUALITY_TOKENS
    _require(quality.get("source_budget_complete") is complete
             and quality.get("status") in (("passed", "failed") if complete else ("partial",))
             and quality.get("stop_reason") == ("complete" if complete else "signal 2"),
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
             and _integer(protocol.get("source_budget"), QUALITY_TOKENS)
             and _integer(settings.get("tokens"), QUALITY_TOKENS)
             and _integer(settings.get("context"), 1024)
             and type(settings.get("batch_tokens")) is int and 1024 <= settings["batch_tokens"] <= 4096
             and type(settings.get("chunk_tokens")) is int and 1 <= settings["chunk_tokens"] <= 128,
             "Gemma research aggregation or source budget differs.")
    observations = quality.get("metric_observations")
    _require(type(observations) is int and 0 < observations <= source_tokens
             and _integer(quality.get("cross_entropy_tokens"), observations),
             "Gemma research target count must equal its nonspecial observations.")
    _require(_same_json(quality.get("thresholds"), {"fvu": .30, "dead_fraction": .15,
                                                 "cross_entropy_delta": .5, "evaluated_tokens": QUALITY_TOKENS}),
             "Gemma historical quality thresholds changed.")
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
                 and (maximum is None or quality[field] <= maximum)
                 and math.isclose(quality[field], sum(rows) / 26, rel_tol=1e-9, abs_tol=1e-12),
                 "Gemma research aggregate differs from its per-layer measurements.")
    for key in ("per_layer_cross_entropy", "per_layer_cross_entropy_delta", "per_layer_l0"):
        _require(isinstance(quality.get(key), list) and len(quality[key]) == 26
                 and all(_number(value) for value in quality[key]), "Gemma lacks finite per-layer research measurements.")
    baseline, replaced = quality["original_cross_entropy"], quality["replaced_cross_entropy"]
    delta = quality["cross_entropy_delta"]
    _require(_number(baseline) and baseline >= 0 and _number(replaced) and replaced >= 0
             and _number(delta) and math.isclose(delta, replaced - baseline, rel_tol=1e-9, abs_tol=1e-12),
             "Gemma research all-layer loss measurements are inconsistent.")
    _require(all(value >= 0 for value in quality["per_layer_cross_entropy"])
             and all(0 <= value <= 16384 for value in quality["per_layer_l0"])
             and all(math.isclose(delta, loss - baseline, rel_tol=1e-9, abs_tol=1e-12)
                     for loss, delta in zip(quality["per_layer_cross_entropy"], quality["per_layer_cross_entropy_delta"])),
             "Gemma isolated loss measurements are inconsistent.")
    if complete:
        passes = quality["fvu"] <= .30 and quality["dead_fraction"] <= .15 and delta <= .5
        _require(quality["status"] == ("passed" if passes else "failed"),
                 "Gemma research measurements contradict the historical quality status.")
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


def _fallback(metadata, response):
    training = metadata["training"]
    fallback = training["fallback"]
    layers = metadata["layers"]
    upper, lower = list(range(layers // 2, layers)), list(range(layers // 2))
    _require(_integer(fallback.get("version"), 1) and fallback.get("mode") == "upper_half"
             and fallback.get("active_layers") == upper and fallback.get("zero_layers") == lower
             and all(type(i) is int for i in fallback["active_layers"] + fallback["zero_layers"])
             and fallback.get("lower_behavior") == "zero_dictionary_original_mlp"
             and fallback.get("warm_start") == "upper_weights_fresh_adam",
             "Fallback metadata does not describe the authorized upper half.")
    _completed_training(training, fallback.get("token_budget"))
    source = response["fallback_source"]
    _require(source == metadata.get("fallback_source"),
             "Fallback source evidence is not bound to the current dictionary metadata.")
    prior = source["metadata"]
    identity = source["identity"]
    fingerprint = identity["dictionary_fingerprint"]
    _require(_digest(fingerprint) and fingerprint == fallback.get("source_dictionary_hash")
             and identity.get("model_hash") == MODEL_HASH
             and all(prior.get(k) == metadata[k] for k in ("model_id", "revision", "model_hash", "layers", "latents"))
             and "fallback" not in prior["training"], "Fallback source identity differs.")
    _quality(prior, fingerprint, failed=True)
    _partitions(prior["training"])
    _completed_training(prior["training"], prior["training"].get("token_budget"))
    _require(_integer(fallback.get("source_training_tokens"), prior["training"]["tokens"])
             and fallback.get("source_id") == training["source_id"] == prior["training"]["source_id"]
             and training["partitions"] == prior["training"]["partitions"], "Fallback source provenance differs.")
    evaluation = source["evaluation"]
    settings, stream = evaluation["settings"], evaluation["stream"]
    _require(_integer(evaluation.get("version"), 1) and _integer(evaluation.get("count"), QUALITY_TOKENS)
             and "pending_ids" in evaluation and evaluation["pending_ids"] is None
             and evaluation.get("identity") == fingerprint and _integer(settings.get("tokens"), QUALITY_TOKENS)
             and settings.get("scope", "all_layers") == "all_layers"
             and settings.get("active_layers", list(range(layers))) == list(range(layers))
             and stream.get("partition") == "quality" and stream.get("source_id") == training["source_id"]
             and stream.get("partitions") == training["partitions"],
             "Fallback source evaluation is incomplete or stale.")


def dictionary_qualification(response, *, quality_override=None, family="qwen"):
    """Refuse absent or stale evidence without model, tensor, or file access."""
    result = {"qualified": False, "scope": None, "status": "unverified"}
    try:
        _require(isinstance(response, dict), "The feature.dictionary response is absent.")
        _require(family in ("qwen", "gemma"), "Unsupported acceptance model family.")
        metadata = response["metadata"]
        gemma = family == "gemma"
        if gemma:
            metadata = _gemma_research(metadata, response)
        fallback = "fallback" in metadata["training"]
        result.update(scope="upper_half" if fallback else "all_layers",
                      status="fallback_unverified" if fallback else "unverified",
                      reported_status=metadata["quality"].get("status"))
        index, identity = response["index"], response["identity"]
        expected_id, revision, model_hash, layers = ((GEMMA_ID, GEMMA_REVISION, GEMMA_HASH, 26) if gemma else
                                                    (MODEL_ID, MODEL_REVISION, MODEL_HASH, 24))
        _require(metadata.get("model_id") == expected_id and metadata.get("revision") == revision
                 and metadata.get("model_hash") == identity.get("model_hash") == model_hash
                 and _integer(metadata.get("layers"), layers), "The current model identity differs from the pinned family.")
        _require(not gemma or (quality_override is None and not fallback),
                 "The Qwen quality override and fallback do not apply to Gemma.")
        _require(all(_digest(identity.get(key)) for key in
                     ("dictionary_hash", "dictionary_fingerprint", "index_hash", "tokenizer_hash")),
                 "Current dictionary and index identities are absent or invalid.")
        _require(all(index.get(key) == identity[key] for key in
                     ("model_hash", "dictionary_hash", "index_hash", "tokenizer_hash"))
                 and index.get("layers") == metadata["layers"] and index.get("features") == metadata["latents"],
                 "The index does not identify the current model and dictionary.")
        _require(index.get("complete") is True and _integer(index.get("tokens"), INDEX_TOKENS)
                 and _integer(index.get("token_budget"), INDEX_TOKENS),
                 "Acceptance requires a complete 2000000-token index.")
        training = _gemma_source(metadata, response) if gemma else metadata["training"]
        provenance = index["provenance"]
        _partitions(training)
        if gemma:
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
        else:
            _completed_training(training, training.get("token_budget"))
        _require(provenance.get("partition") == "index" and all(provenance.get(k) == training[k] for k in
                 ("dataset_id", "dataset_config", "dataset_revision", "split", "partitions")),
                 "The index source or held-out partition differs.")
        if gemma or "source_id" in provenance:
            _require(provenance["source_id"] == training["source_id"], "The index source identity differs.")
        if not gemma:
            _quality(metadata, identity["dictionary_fingerprint"], fallback=fallback, override=quality_override)
        if fallback:
            _fallback(metadata, response)
        result.update(qualified=True, status="fallback_passed" if fallback else "passed", identity=dict(identity))
        if gemma:
            quality = metadata["quality"]
            result.update(status="research_reported", quality_gates_blocking=False,
                          historical_quality_status=quality["status"], research_protocol="research_v1",
                          source_tokens=quality["source_tokens"], evaluated_tokens=quality["evaluated_tokens"],
                          metric_observations=quality["metric_observations"],
                          cross_entropy_tokens=quality["cross_entropy_tokens"],
                          source_budget=QUALITY_TOKENS, source_budget_complete=quality["source_budget_complete"],
                          stop_reason=quality["stop_reason"], checkpoint_sha256=quality["checkpoint_sha256"],
                          acceptance_policy="research_comparison_only",
                          acceptance_reason="Decision 10 accepts a committed comparison sample of at least 50000 source positions "
                          "to avoid delay to the Gemma index and acceptance captures. The founder sets final quality bars.")
        if quality_override is not None:
            result.update(status="founder_override", quality_override=validate_quality_override(quality_override))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        result["reason"] = str(error)
    return result
