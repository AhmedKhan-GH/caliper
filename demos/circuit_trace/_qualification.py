"""Release qualification from the live feature.dictionary response."""
from __future__ import annotations

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
             and binding.get("layers") == [{key: row[key] for key in
                                            ("layer", "file", "sha256", "input_gain_hash")} for row in layers],
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
            _gemma_model_binding(metadata, response)
            _require(provenance.get("publisher_training_overlap") == "unknown",
                     "Gemma index must retain unknown publisher training overlap.")
        else:
            _completed_training(training, training.get("token_budget"))
        _require(provenance.get("partition") == "index" and all(provenance.get(k) == training[k] for k in
                 ("dataset_id", "dataset_config", "dataset_revision", "split", "partitions")),
                 "The index source or held-out partition differs.")
        if gemma or "source_id" in provenance:
            _require(provenance["source_id"] == training["source_id"], "The index source identity differs.")
        _quality(metadata, identity["dictionary_fingerprint"], fallback=fallback, override=quality_override,
                 fvu_limit=.30 if gemma else .40)
        if fallback:
            _fallback(metadata, response)
        result.update(qualified=True, status="fallback_passed" if fallback else "passed", identity=dict(identity))
        if quality_override is not None:
            result.update(status="founder_override", quality_override=validate_quality_override(quality_override))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        result["reason"] = str(error)
    return result
