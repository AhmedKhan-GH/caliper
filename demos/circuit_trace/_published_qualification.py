"""Published resource checks over the trusted live qualification response."""
from __future__ import annotations

import hashlib
import json
import re

from _qualification import (GEMMA_HASH, GEMMA_ID, GEMMA_REVISION, INDEX_TOKENS, _digest,
                            _gemma_index_context, _gemma_provenance, _integer, _require, _same_json)

PROFILE = "published_dallas_v1"
REFERENCE_PROFILE = "circuit_tracer_82996f14"
REFERENCE_REVISION = "82996f144b3371ae858ea5713590d004c218f0c2"
REFERENCE_INPUT = {"coordinate": "gemma_scope_pre_gain", "encoder": "W_enc_original", "dtype": "bfloat16"}
INPUT_CONVERSION = "(W_enc.T / reference_gain).to(bfloat16)"
REFERENCE_CONVERSION = {
    "version": 2, "profile": REFERENCE_PROFILE, "model_hash": GEMMA_HASH,
    "source_norm_cast": "runtime_norm_weight = source_norm_weight.to(bfloat16)",
    "source_float32_gain": "1 + source_norm_weight.to(float32)",
    "runtime_float32_gain": "1 + runtime_norm_weight.to(float32)",
    "gain": "(1 + runtime_norm_weight.to(float32)).to(bfloat16)",
    "division_dtype": "promote_types(source_dtype, float32)",
    "tensors": {"W_enc_original": "W_enc.T.to(bfloat16).contiguous()", "W_enc": INPUT_CONVERSION,
                **{name: f"{name}.to(bfloat16).contiguous()" for name in ("b_enc", "W_dec", "b_dec", "threshold")}},
}
_GAIN_HASHES = ("input_gain_hash", "source_gain_hash", "float32_gain_hash",
                "runtime_norm_hash", "runtime_float32_gain_hash")
_LAYER_FIELDS = ("layer", "file", "source_file", "sha256", "source_sha256", "original_dtype",
                 "source_gain_dtype", "runtime_norm_dtype", "tensor_hashes", *_GAIN_HASHES)
_ADAPTERS = {"Gemma2ForCausalLM": "ReferenceGemmaForCausalLM", "Gemma2RMSNorm": "ReferenceRMSNorm",
             "Gemma2Attention": "ReferenceAttention", "Gemma2RotaryEmbedding": "ReferenceRotaryEmbedding"}
_MODULE = "caliper.interp._gemma_reference."
_HF = "transformers.models.gemma2.modeling_gemma2."
_PARAMETERS = {"parameter:model.embed_tokens.weight", "parameter:model.norm.weight", *(
    f"parameter:model.layers.{i}.{name}.weight" for i in range(26) for name in (
        "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))}
_BUFFERS = {"buffer:model.embed_tokens.embed_scale", "buffer:model.rotary_emb.inv_freq",
            "buffer:model.rotary_emb.original_inv_freq"}
_BACKENDS = {"", "model", "model.rotary_emb", *(f"model.layers.{i}{suffix}" for i in range(26)
                                               for suffix in ("", ".self_attn", ".mlp"))}
_OPERATIONS = {
    *(_MODULE + name + ".forward" for name in (*_ADAPTERS.values(), "ReferenceInput")),
    _MODULE + "normalized_input", _HF + "rotate_half",
    *(_HF + name + ".forward" for name in
      ("Gemma2ForCausalLM", "Gemma2Model", "Gemma2DecoderLayer", "Gemma2MLP", "Gemma2TextScaledWordEmbedding")),
    "torch.nn.modules.linear.Linear.forward", "transformers.activations.GELUTanh.forward",
}


def _state_digest(value):
    return isinstance(value, str) and value.startswith("sha256:") and _digest(value[7:])


def _import_evidence(metadata, response):
    source = _gemma_provenance(metadata)
    imported = metadata["import_source"]
    _require(_same_json(metadata.get("reference_input"), REFERENCE_INPUT)
             and _same_json(imported.get("reference_conversion"), REFERENCE_CONVERSION)
             and imported.get("original_input") == "pure_rms_normalized_pre_mlp_before_learned_gain"
             and imported.get("input_conversion") == INPUT_CONVERSION
             and imported.get("output") == "after_post_feedforward_layernorm"
             and imported.get("output_conversion") == "direct source cast to bfloat16"
             and imported.get("reference") == {"repo": "decoderesearch/circuit-tracer", "revision": REFERENCE_REVISION}
             and imported.get("source_verification") == "trusted_local_snapshot_and_file_sha256",
             "Published Gemma requires the exact version 2 source conversion and BF16 original coordinates.")
    rows = imported["layers"]
    for row in rows:
        _require(row.get("original_dtype") in ("float16", "float32", "float64")
                 and row.get("source_gain_dtype") == "float32" and row.get("runtime_norm_dtype") == "bfloat16"
                 and all(_state_digest(row.get(key)) for key in _GAIN_HASHES),
                 "Published Gemma source and runtime gain evidence is incomplete.")
        hashes = row.get("tensor_hashes")
        _require(isinstance(hashes, dict) and set(hashes) == set(REFERENCE_CONVERSION["tensors"])
                 and all(_state_digest(value) for value in hashes.values()),
                 "Published Gemma requires all six original-source tensor hashes per layer.")
    expected = {"version": 2, "profile": REFERENCE_PROFILE, "model_hash": GEMMA_HASH,
                "source": imported["source"], "reference_input": REFERENCE_INPUT,
                "reference_conversion": REFERENCE_CONVERSION,
                "layers": [{key: row[key] for key in _LAYER_FIELDS} for row in rows]}
    _require(_same_json(response.get("import_binding"), expected),
             "Published Gemma import evidence differs from the complete version 2 artifact evidence.")
    return source


def _model_evidence(response, index):
    binding = response.get("model_binding")
    _require(isinstance(binding, dict) and set(binding) ==
             {"version", "model_hash", "tensor_hash", "forward_settings_hash"}
             and _integer(binding.get("version"), 2) and binding.get("model_hash") == GEMMA_HASH
             and _digest(binding.get("tensor_hash")) and _digest(binding.get("forward_settings_hash")),
             "Published Gemma requires current verified model tensors and arithmetic evidence.")
    tensors = response.get("tensor_binding")
    _require(isinstance(tensors, dict) and set(tensors) == {"version", "tensor_hash"}
             and _integer(tensors.get("version"), 1) and _digest(tensors.get("tensor_hash")),
             "Published Gemma requires current live dictionary tensor evidence.")
    encoding = index["model_encoding"]
    _require(isinstance(encoding, dict) and set(encoding) == {"version", "autocast_dtype",
             "float32_matmul_precision", "tensors", "forward_settings_hash", "attention_backends", "reference_profile"}
             and _integer(encoding.get("version"), 1) and encoding.get("autocast_dtype") is None
             and encoding.get("float32_matmul_precision") == "highest"
             and encoding.get("forward_settings_hash") == binding["forward_settings_hash"],
             "Published index model encoding differs from the current model arithmetic.")
    records = encoding["tensors"]
    _require(isinstance(records, list) and bool(records) and all(
        isinstance(row, list) and len(row) == 3 and all(isinstance(value, str) for value in row)
        and re.fullmatch(r"(?:parameter|buffer):.+", row[0]) is not None
        and re.fullmatch(r"(?:cpu|mps|cuda)(?::[0-9]+)?", row[2]) is not None
        and (row[1] == "torch.bfloat16" if row[0].startswith("parameter:") else
             row[1] in ("torch.bfloat16", "torch.float32"))
        for row in records) and len({row[0] for row in records}) == len(records)
        and {row[0] for row in records} == _PARAMETERS | _BUFFERS,
        "Published index requires complete BF16 parameter dtype and device evidence.")
    backends = encoding["attention_backends"]
    _require(isinstance(backends, list) and bool(backends) and all(
        isinstance(row, list) and len(row) == 2 and isinstance(row[0], str) and row[1] == "eager" for row in backends)
        and len({row[0] for row in backends}) == len(backends) and {row[0] for row in backends} == _BACKENDS,
        "Published index requires the qualified eager attention backend.")
    profile = encoding["reference_profile"]
    fixed = {"profile": REFERENCE_PROFILE, "dtype": "bfloat16", "version": 1,
             "reference_revision": REFERENCE_REVISION, "reference_transformer_lens": "3.9.0",
             "normalization": "float32_divide_sqrt_then_cast_then_gain",
             "gain": "source_weight_float32_plus_one_then_cast",
             "rotary": "cpu_float32_position_divide_frequency_then_trig_then_cast_then_transfer",
             "attention": "matmul_divide_sqrt_scalar_softcap_mask_negative_inf_softmax_score_dtype",
             "encoder_coordinate": "gemma_scope_pre_gain", "capture": "pre_feedforward_layernorm.reference_input",
             "runtime": "autocast_disabled_float32_matmul_highest",
             "operations": {name: _MODULE + cls + ".forward" for name, cls in _ADAPTERS.items()}}
    _require(isinstance(profile, dict) and set(profile) == set(fixed) |
             {"operation_sha256", "torch_version", "transformers_version"}
             and _same_json({key: profile[key] for key in fixed}, fixed)
             and all(isinstance(profile[key], str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+[^\s]*", profile[key])
                     for key in ("torch_version", "transformers_version")),
             "Published index requires the complete qualified BF16 arithmetic profile.")
    hashes = profile["operation_sha256"]
    _require(isinstance(hashes, dict) and _OPERATIONS <= set(hashes)
             and all(isinstance(name, str) and re.sub(r"(?:\.__wrapped__)+$", "", name) in _OPERATIONS
                     and (not name.endswith(".__wrapped__") or name.removesuffix(".__wrapped__") in hashes)
                     and _digest(value) for name, value in hashes.items()),
             "Published index lacks complete reference operation hashes.")


def _index_evidence(metadata, response, source):
    index, identity = response["index"], response["identity"]
    _require(metadata.get("model_id") == GEMMA_ID and metadata.get("revision") == GEMMA_REVISION
             and metadata.get("model_hash") == identity.get("model_hash") == GEMMA_HASH
             and _integer(metadata.get("layers"), 26), "Published Gemma model identity differs from the pinned model.")
    keys = ("dictionary_hash", "dictionary_fingerprint", "index_hash", "tokenizer_hash")
    _require(all(_digest(identity.get(key)) for key in keys)
             and all(index.get(key) == identity[key] for key in (*keys, "model_hash"))
             and _digest(index.get("dictionary_source_hash")),
             "Published index does not bind the full current artifact fingerprint and resource identities.")
    _require(_integer(index.get("format_version"), 2) and index.get("encoder_policy") == "gemma_scope_pre_gain_v1"
             and index.get("threshold_policy") == "live_input_v1"
             and _same_json(index.get("reference_input"), REFERENCE_INPUT),
             "Published index requires version 2 and original-coordinate encoding.")
    _require(index.get("complete") is True and _integer(index.get("tokens"), INDEX_TOKENS)
             and _integer(index.get("token_budget"), INDEX_TOKENS)
             and _integer(index.get("layers"), 26) and _integer(index.get("features"), 16384)
             and _integer(index.get("top_examples"), 16) and _integer(index.get("window_tokens"), 33)
             and index.get("tie_order") == "activation_desc,global_offset_asc"
             and _integer(index.get("vocab_size"), 256000)
             and type(index.get("documents")) is int and 0 < index["documents"] <= INDEX_TOKENS,
             "Published Gemma requires a complete 2000000-position reference index and exact example policy.")
    _gemma_index_context(index)
    provenance = index.get("provenance")
    expected_source = {**source, "partition": "index", "publisher_training_overlap": "unknown"}
    _require(isinstance(provenance, dict)
             and _same_json({key: provenance.get(key) for key in expected_source}, expected_source)
             and provenance.get("source_kind") in ("pinned_local_parquet", "pinned_dataset")
             and provenance.get("duplicate_documents") == "first occurrence only",
             "Published index source or held-out partition differs from the pinned source.")
    files = index.get("files")
    _require(isinstance(files, dict) and set(files) == {"table.npy", "counts.npy", "tokens.bin", "documents.bin"}
             and all(isinstance(row, dict) and set(row) == {"bytes", "sha256"}
                     and type(row["bytes"]) is int and row["bytes"] > 0 and _digest(row["sha256"]) for row in files.values()),
             "Published index file evidence is incomplete.")
    _model_evidence(response, index)
    payload = {key: value for key, value in index.items() if key != "index_hash"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _require(index["index_hash"] == digest, "Published index manifest hash differs from its complete evidence.")
    return identity


def published_dictionary_qualification(response, *, family, profile, quality_override=None):
    """Keep resource qualification separate from historical quality and D12 gates."""
    result = {"qualified": False, "scope": None, "status": "unverified", "quality_gates_blocking": False,
              "acceptance_policy": "published_replication", "research_sample_required": False,
              "quality_disclosure": {
                  "historical_comparison": "supplied" if isinstance(response, dict) and
                      any(response.get(key) is not None for key in ("research_quality", "historical_research_quality")) else "absent",
                  "authorizes_current_resources": False, "current_bf16_quality": "not_established_by_this_qualification",
                  "reason": "Decisions 8 and 10 keep research quality as separate comparison evidence. "
                            "Historical F32 measurements do not describe these BF16 resources. No new research sample is required."}}
    try:
        _require(family == "gemma" and profile == PROFILE and quality_override is None,
                 "Published resource qualification requires the explicit Gemma Dallas profile without a quality override.")
        _require(isinstance(response, dict), "The feature.dictionary response is absent.")
        metadata = response["metadata"]
        source = _import_evidence(metadata, response)
        identity = _index_evidence(metadata, response, source)
        result.update(qualified=True, scope="all_layers", status="published_resources_verified", identity=dict(identity),
                      reference_profile=REFERENCE_PROFILE, dtype="bfloat16", import_binding_version=2,
                      acceptance_reason="Current resource evidence qualifies the published profile. "
                                        "Every Decision 12 replication and external review gate remains required.")
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        result["reason"] = str(error)
    return result
