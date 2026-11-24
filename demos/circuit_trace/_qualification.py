"""Release qualification from the live feature.dictionary response."""
from __future__ import annotations

import math
import re

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_HASH = f"hf:{MODEL_ID}@{MODEL_REVISION}"
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


def _quality(metadata, fingerprint, *, fallback=False, failed=False):
    quality = metadata["quality"]
    layers = metadata["layers"]
    active = list(range(layers // 2, layers)) if fallback else list(range(layers))
    status = "failed" if failed else "fallback_passed" if fallback else "passed"
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
    passes = quality["fvu"] <= .30 and quality["dead_fraction"] <= .15 and delta <= .5
    _require(passes != failed, "Quality metrics contradict the required status.")


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
    _require(type(budget) is int and budget > 0 and _integer(training.get("tokens"), budget)
             and training.get("stop_reason") == "token_budget",
             "Fallback requires a complete training budget.")


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


def dictionary_qualification(response):
    """Refuse absent or stale evidence without model, tensor, or file access."""
    result = {"qualified": False, "scope": None, "status": "unverified"}
    try:
        _require(isinstance(response, dict), "The feature.dictionary response is absent.")
        metadata = response["metadata"]
        fallback = "fallback" in metadata["training"]
        result.update(scope="upper_half" if fallback else "all_layers",
                      status="fallback_unverified" if fallback else "unverified",
                      reported_status=metadata["quality"].get("status"))
        index, identity = response["index"], response["identity"]
        _require(metadata.get("model_id") == MODEL_ID and metadata.get("revision") == MODEL_REVISION
                 and metadata.get("model_hash") == identity.get("model_hash") == MODEL_HASH
                 and _integer(metadata.get("layers"), 24), "The current model identity differs from pinned Qwen.")
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
        training, provenance = metadata["training"], index["provenance"]
        _partitions(training)
        _require(provenance.get("partition") == "index" and all(provenance.get(k) == training[k] for k in
                 ("dataset_id", "dataset_config", "dataset_revision", "split", "partitions")),
                 "The index source or held-out partition differs.")
        if "source_id" in provenance:
            _require(provenance["source_id"] == training["source_id"], "The index source identity differs.")
        _quality(metadata, identity["dictionary_fingerprint"], fallback=fallback)
        if fallback:
            _fallback(metadata, response)
        result.update(qualified=True, status="fallback_passed" if fallback else "passed", identity=dict(identity))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        result["reason"] = str(error)
    return result
