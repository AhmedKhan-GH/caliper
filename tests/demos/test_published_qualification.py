"""Synthetic BF16 resource evidence without payloads or real acceptance claims."""
from copy import deepcopy
import hashlib
import json

import pytest

from test_circuit_trace import driver  # Add the demo modules to the import path.
from test_gemma_acceptance import gemma_reply
from _checks import check, finish
from _qualification import (GEMMA_HASH, GEMMA_ID, GEMMA_L0, GEMMA_REVISION,
                            GEMMA_SCOPE_REVISION, GEMMA_SELECTION_REVISION, dictionary_qualification)

PROFILE = "published_dallas_v1"


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def seal_index(response):
    index = response["index"]
    payload = {key: value for key, value in index.items() if key != "index_hash"}
    value = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    index["index_hash"] = response["identity"]["index_hash"] = value


def published_reply():
    """Describe the real response schema with entirely synthetic hashes and counts."""
    reference = {"coordinate": "gemma_scope_pre_gain", "encoder": "W_enc_original", "dtype": "bfloat16"}
    conversion = {"version": 2, "profile": "circuit_tracer_82996f14", "model_hash": GEMMA_HASH,
        "source_norm_cast": "runtime_norm_weight = source_norm_weight.to(bfloat16)",
        "source_float32_gain": "1 + source_norm_weight.to(float32)",
        "runtime_float32_gain": "1 + runtime_norm_weight.to(float32)",
        "gain": "(1 + runtime_norm_weight.to(float32)).to(bfloat16)",
        "division_dtype": "promote_types(source_dtype, float32)",
        "tensors": {"W_enc_original": "W_enc.T.to(bfloat16).contiguous()",
                    "W_enc": "(W_enc.T / reference_gain).to(bfloat16)",
                    "W_dec": "W_dec.to(bfloat16).contiguous()", "b_enc": "b_enc.to(bfloat16).contiguous()",
                    "b_dec": "b_dec.to(bfloat16).contiguous()", "threshold": "threshold.to(bfloat16).contiguous()"}}
    source = {"dataset_id": "HuggingFaceFW/fineweb-edu", "dataset_config": "sample-10BT",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "split": "train",
        "source_id": "synthetic-pinned-source",
        "partitions": {"algorithm": "sha256-utf8-text-mod10000-v1", "train": [0, 9400],
                       "quality": [9400, 9500], "index": [9500, 10000], "quality_tokens": 200000, "index_tokens": 2000000}}
    rows, bindings = [], []
    for i, l0 in enumerate(GEMMA_L0):
        row = {"layer": i, "width": 2304, "latents": 16384, "average_l0": l0,
            "source_file": f"layer_{i}/width_16k/average_l0_{l0}/params.npz", "file": f"layer_{i:02d}.safetensors",
            "source_sha256": digest(f"source-{i}"), "sha256": digest(f"bf16-file-{i}"),
            "original_dtype": "float32", "source_gain_dtype": "float32", "runtime_norm_dtype": "bfloat16",
            **{key: "sha256:" + digest(f"{key}-{i}") for key in
               ("input_gain_hash", "source_gain_hash", "float32_gain_hash", "runtime_norm_hash", "runtime_float32_gain_hash")},
            "tensor_hashes": {name: "sha256:" + digest(f"bf16-{i}-{name}") for name in conversion["tensors"]}}
        rows.append(row)
        bindings.append({key: deepcopy(value) for key, value in row.items() if key not in ("width", "latents", "average_l0")})
    imported = {"kind": "gemma_scope", "license": "CC-BY-4.0",
        "source": {"repo": "google/gemma-scope-2b-pt-transcoders", "revision": GEMMA_SCOPE_REVISION},
        "selection": {"repo": "mntss/gemma-scope-transcoders", "revision": GEMMA_SELECTION_REVISION, "file": "config.yaml"},
        "reference": {"repo": "decoderesearch/circuit-tracer", "revision": "82996f144b3371ae858ea5713590d004c218f0c2"},
        "source_verification": "trusted_local_snapshot_and_file_sha256",
        "original_input": "pure_rms_normalized_pre_mlp_before_learned_gain",
        "input_conversion": "(W_enc.T / reference_gain).to(bfloat16)", "output": "after_post_feedforward_layernorm",
        "output_conversion": "direct source cast to bfloat16", "reference_conversion": conversion, "layers": rows}
    metadata = {"model_id": GEMMA_ID, "revision": GEMMA_REVISION, "model_hash": GEMMA_HASH,
        "layers": 26, "width": 2304, "latents": 16384, "activation": "jumprelu", "k": None,
        "reference_input": reference, "import_source": imported, "evaluation_source": source,
        "training": {"origin": "imported", "local_training": False, "tokens": 0, "partitions": {},
                     "publisher_training_disjointness": "unknown"}, "quality": {"status": "not_evaluated"}}
    operations = {
        "Gemma2Attention": "caliper.interp._gemma_reference.ReferenceAttention.forward",
        "Gemma2ForCausalLM": "caliper.interp._gemma_reference.ReferenceGemmaForCausalLM.forward",
        "Gemma2RMSNorm": "caliper.interp._gemma_reference.ReferenceRMSNorm.forward",
        "Gemma2RotaryEmbedding": "caliper.interp._gemma_reference.ReferenceRotaryEmbedding.forward"}
    functions = [*operations.values(), "caliper.interp._gemma_reference.ReferenceInput.forward",
        "caliper.interp._gemma_reference.ReferenceMLPOutputNorm.forward",
        "caliper.interp._gemma_reference.normalized_input", "transformers.models.gemma2.modeling_gemma2.rotate_half",
        "transformers.models.gemma2.modeling_gemma2.Gemma2ForCausalLM.forward",
        "transformers.models.gemma2.modeling_gemma2.Gemma2Model.forward",
        "transformers.models.gemma2.modeling_gemma2.Gemma2DecoderLayer.forward",
        "transformers.models.gemma2.modeling_gemma2.Gemma2MLP.forward",
        "transformers.models.gemma2.modeling_gemma2.Gemma2TextScaledWordEmbedding.forward",
        "torch.nn.modules.linear.Linear.forward", "transformers.activations.GELUTanh.forward"]
    profile = {"profile": "circuit_tracer_82996f14", "version": 2, "dtype": "bfloat16",
        "reference_revision": "82996f144b3371ae858ea5713590d004c218f0c2", "reference_transformer_lens": "3.9.0",
        "torch_version": "2.12.0", "transformers_version": "5.6.1",
        "normalization": "float32_divide_sqrt_then_cast_then_gain", "gain": "source_weight_float32_plus_one_then_cast",
        "mlp_output": "model_dtype_post_norm_then_float32_before_residual_add",
        "rotary": "cpu_float32_position_divide_frequency_then_trig_then_cast_then_transfer",
        "attention": "matmul_divide_sqrt_scalar_softcap_mask_negative_inf_softmax_score_dtype",
        "encoder_coordinate": "gemma_scope_pre_gain", "capture": "pre_feedforward_layernorm.reference_input",
        "runtime": "autocast_disabled_float32_matmul_highest", "operations": operations,
        "operation_sha256": {name: digest(name) for name in functions}}
    model = {"version": 2, "model_hash": GEMMA_HASH, "tensor_hash": digest("bf16-model"),
             "forward_settings_hash": digest("bf16-forward-settings")}
    identity = {"model_hash": GEMMA_HASH, "dictionary_hash": digest("bf16-dictionary"),
                "dictionary_fingerprint": digest("bf16-full-fingerprint"), "tokenizer_hash": digest("tokenizer")}
    policy = {"version": 1, "family": "gemma2", "prefix_ids": [2], "pad_id": 0,
        "model_context": 256, "content_context": 255, "segmentation": "contiguous_document_positions",
        "observations": "source_tokens_only", "replacement": "source_positions_only",
        "cross_entropy": "within_context_content_targets_including_bos_to_first"}
    parameters = ["model.embed_tokens.weight", "model.norm.weight"]
    parameters.extend(f"model.layers.{i}.{name}.weight" for i in range(26) for name in (
        "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    tensors = [["parameter:" + name, "torch.bfloat16", "cpu"] for name in parameters]
    tensors.extend([["buffer:model.rotary_emb." + name, "torch.float32", "cpu"]
                    for name in ("inv_freq", "original_inv_freq")])
    tensors.append(["buffer:model.embed_tokens.embed_scale", "torch.float32", "cpu"])
    backends = [[name, "eager"] for name in ["", "model", "model.rotary_emb"] +
                [f"model.layers.{i}{suffix}" for i in range(26) for suffix in ("", ".self_attn", ".mlp")]]
    index = {**identity, "format_version": 2, "complete": True, "encoder_policy": "gemma_scope_pre_gain_v1",
        "threshold_policy": "live_input_v1", "reference_input": deepcopy(reference),
        "dictionary_source_hash": digest("bf16-source-files"), "tokens": 2000000, "token_budget": 2000000,
        "layers": 26, "features": 16384, "top_examples": 16, "window_tokens": 33, "vocab_size": 256000,
        "documents": 1000, "tie_order": "activation_desc,global_offset_asc", "context_length": 256,
        "context_policy": policy, "example_context_policy": dict(policy, model_context=34, content_context=33),
        "provenance": {**deepcopy(source), "partition": "index", "publisher_training_overlap": "unknown",
                       "source_kind": "pinned_local_parquet", "duplicate_documents": "first occurrence only"},
        "files": {name: {"bytes": 1024, "sha256": digest(name)} for name in
                  ("table.npy", "counts.npy", "tokens.bin", "documents.bin")},
        "model_encoding": {"version": 1, "autocast_dtype": None, "float32_matmul_precision": "highest",
            "tensors": tensors,
            "forward_settings_hash": model["forward_settings_hash"], "attention_backends": backends,
            "reference_profile": profile}}
    response = {"metadata": metadata, "index": index, "identity": identity, "model_binding": model,
        "tensor_binding": {"version": 1, "tensor_hash": digest("bf16-live-tensors")},
        "import_binding": {"version": 2, "profile": "circuit_tracer_82996f14", "model_hash": GEMMA_HASH,
            "reference_input": deepcopy(reference), "reference_conversion": deepcopy(conversion),
            "source": deepcopy(imported["source"]), "layers": bindings}}
    seal_index(response)
    return response


def qualify(response, **kwargs):
    return dictionary_qualification(response, family="gemma", replication_profile=PROFILE, **kwargs)


def test_exact_reference_response_requires_no_research_sample():
    response = published_reply()
    before = deepcopy(response)
    result = qualify(response)
    assert result["qualified"], result
    assert result["status"] == "published_resources_verified" and result["scope"] == "all_layers"
    assert not result["quality_gates_blocking"] and not result["research_sample_required"]
    assert result["quality_disclosure"]["historical_comparison"] == "absent"
    assert not result["quality_disclosure"]["authorizes_current_resources"]
    assert "F32" in result["quality_disclosure"]["reason"] and "BF16" in result["quality_disclosure"]["reason"]
    assert response == before


@pytest.mark.parametrize("wire", [False, True])
def test_live_adapter_metadata_satisfies_the_demo_contract(wire):
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM
    from caliper.interp._gemma_reference import install_reference_profile, profile_metadata
    from caliper.interp._qualification import _wire_index

    config = Gemma2Config(vocab_size=37, hidden_size=24, intermediate_size=40,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=6, max_position_embeddings=32, sliding_window=3,
                         query_pre_attn_scalar=6, hidden_activation="gelu_pytorch_tanh", use_cache=False)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        model = Gemma2ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    install_reference_profile(model, "circuit_tracer_82996f14")
    response = published_reply()
    response["index"]["model_encoding"]["reference_profile"] = profile_metadata(model)
    seal_index(response)
    if wire:
        response.update(_wire_index(response["index"]))
    before = deepcopy(response)
    result = qualify(response)
    assert result["qualified"], result
    assert response == before


@pytest.mark.parametrize("fault", ["legacy_version", "missing_output", "wrong_output", "missing_operation"])
def test_current_output_coordinate_is_required(fault):
    response = published_reply()
    profile = response["index"]["model_encoding"]["reference_profile"]
    if fault == "legacy_version":
        profile["version"] = 1
    elif fault == "missing_output":
        profile.pop("mlp_output")
    elif fault == "wrong_output":
        profile["mlp_output"] = "model_dtype_post_norm"
    else:
        profile["operation_sha256"].pop("caliper.interp._gemma_reference.ReferenceMLPOutputNorm.forward")
    seal_index(response)
    assert not qualify(response)["qualified"]


def test_full_reference_evidence_crosses_the_metadata_boundary():
    from caliper import _interpret_protocol as protocol
    from caliper.interp._qualification import _wire_index
    from _published_qualification import _restore_index
    response = published_reply()
    original = deepcopy(response)
    assert not protocol.bounded(response)  # Qualified operation names exceed key limits.
    response.update(_wire_index(response["index"]))
    encoded = deepcopy(response)
    assert protocol.bounded(response)
    assert qualify(response)["qualified"]
    assert _restore_index(response)["index"] == original["index"]
    assert response == encoded


@pytest.mark.parametrize("fault", ["format", "duplicate", "order", "hash", "missing", "mapping"])
def test_index_transport_cannot_change_qualified_identity(fault):
    from caliper.interp._qualification import _wire_index
    response = published_reply()
    response.update(_wire_index(response["index"]))
    profile = response["index"]["model_encoding"]["reference_profile"]
    rows = profile["operation_sha256"]
    if fault == "format": response["index_transport"] = "unknown"
    elif fault == "duplicate": rows.append(rows[0])
    elif fault == "order": rows.reverse()
    elif fault == "hash": rows[0][1] = "a" * 64
    elif fault == "missing": rows.pop()
    else: profile["operation_sha256"] = dict(rows)
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("field", ["metadata", "index", "identity", "model_binding", "tensor_binding", "import_binding"])
def test_missing_current_evidence_cannot_borrow_historical_quality(field):
    response = published_reply()
    response["research_quality"] = gemma_reply()["research_quality"]
    response.pop(field)
    result = qualify(response)
    assert not result["qualified"] and result["reason"]


@pytest.mark.parametrize("field", ["research_quality", "historical_research_quality"])
@pytest.mark.parametrize("historical", [None, {}, {"status": "partial", "source_tokens": 51237}, "malformed"])
def test_historical_quality_never_becomes_a_published_gate(field, historical):
    response = published_reply()
    if historical is None:
        historical = gemma_reply(source_tokens=51237)["research_quality"]
    response[field] = deepcopy(historical)
    result = qualify(response)
    assert result["qualified"], result
    assert result["quality_disclosure"]["historical_comparison"] == "supplied"
    assert result["quality_disclosure"]["current_bf16_quality"] == "not_established_by_this_qualification"
    response.pop("import_binding")
    assert not qualify(response)["qualified"]


def test_f32_comparison_and_index_cannot_authorize_bf16():
    f32 = gemma_reply(source_tokens=51237)
    assert dictionary_qualification(f32, family="gemma")["qualified"]
    assert not qualify(f32)["qualified"]
    reference = published_reply()
    assert not dictionary_qualification(reference, family="gemma")["qualified"]
    reference["research_quality"] = f32["research_quality"]
    reference["index"] = f32["index"]
    assert not qualify(reference)["qualified"]


@pytest.mark.parametrize("family,profile", [("fixture", PROFILE), ("gemma", "arbitrary"),
                                           ("gemma", "circuit_tracer_82996f14"), ("gemma", True), ("gemma", {})])
def test_profile_marker_does_not_select_or_qualify_resources(family, profile):
    result = dictionary_qualification(published_reply(), family=family, replication_profile=profile)
    assert not result["qualified"]


def test_finish_keeps_external_review_required_after_resource_qualification():
    result = {"mode": "gemma", "completed": True, "replication_profile": PROFILE,
              "dictionary_evidence": published_reply(), "pixel_evidence_collected": True,
              "checks": [check("complete_graph", True, 1, 1)]}
    assert finish(result) == 1
    assert result["dictionary_qualification"]["qualified"]
    assert not result["accepted_replication"] and "external_replication_review" in result["failed_checks"]
    gate = next(row for row in result["checks"] if row["name"] == "dictionary_qualification")
    assert gate["required"] and "50000" not in gate["threshold"]
    result["dictionary_evidence"].pop("import_binding")
    assert finish(result) == 1
    assert "dictionary_qualification" in result["failed_checks"]


def replace(response, path, value):
    parent = response
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("metadata", "model_id"), "other/gemma"),
    (("metadata", "revision"), "0" * 40),
    (("metadata", "model_hash"), "sha256:" + "0" * 64),
    (("metadata", "layers"), 26.0),
    (("metadata", "latents"), 8192),
    (("metadata", "training", "local_training"), True),
    (("metadata", "training", "publisher_training_disjointness"), "known"),
    (("metadata", "reference_input", "encoder"), "W_enc"),
    (("metadata", "reference_input", "dtype"), "float32"),
    (("metadata", "reference_input", "coordinate"), "post_gain_inverse"),
    (("metadata", "import_source", "reference_conversion", "version"), 1),
    (("metadata", "import_source", "reference_conversion", "version"), 2.0),
    (("metadata", "import_source", "reference_conversion", "gain"), "(1 + source_norm_weight).to(bfloat16)"),
    (("metadata", "import_source", "reference_conversion", "source_norm_cast"), "none"),
    (("metadata", "import_source", "reference_conversion", "tensors", "W_enc_original"), "W_enc * gain"),
    (("metadata", "import_source", "source_verification"), "profile_marker"),
    (("metadata", "import_source", "input_conversion"), "none"),
    (("metadata", "import_source", "output_conversion"), "none"),
    (("metadata", "import_source", "source", "revision"), "0" * 40),
    (("metadata", "import_source", "selection", "revision"), "0" * 40),
    (("metadata", "import_source", "reference", "revision"), "0" * 40),
    (("import_binding", "version"), 1),
    (("import_binding", "version"), 2.0),
    (("import_binding", "profile"), "arbitrary"),
    (("import_binding", "model_hash"), "sha256:" + "0" * 64),
    (("import_binding", "reference_conversion", "version"), 1),
    (("import_binding", "source", "revision"), "0" * 40),
    (("model_binding", "model_hash"), "sha256:" + "0" * 64),
    (("model_binding", "version"), 1),
    (("model_binding", "tensor_hash"), ""),
    (("model_binding", "forward_settings_hash"), "0" * 64),
    (("tensor_binding", "tensor_hash"), ""),
    (("tensor_binding", "version"), True),
    (("identity", "dictionary_hash"), "0" * 64),
    (("identity", "dictionary_fingerprint"), "0" * 64),
    (("identity", "index_hash"), "0" * 64),
    (("identity", "tokenizer_hash"), "0" * 64),
    (("identity", "model_hash"), "sha256:" + "0" * 64),
])
def test_changed_resource_evidence_cannot_qualify(path, value):
    response = published_reply()
    replace(response, path, value)
    result = qualify(response)
    assert not result["qualified"], (path, result)
    assert result["reason"]


@pytest.mark.parametrize("side", ["metadata", "import_binding"])
@pytest.mark.parametrize("layer", [0, 13, 25])
@pytest.mark.parametrize("field,value", [
    ("layer", 1), ("file", "layer_99.safetensors"), ("source_file", "wrong/params.npz"),
    ("source_sha256", "0" * 64), ("sha256", "0" * 64), ("original_dtype", "float64"),
    ("source_gain_dtype", "bfloat16"), ("runtime_norm_dtype", "float32"),
    ("input_gain_hash", "sha256:" + "0" * 64), ("source_gain_hash", "sha256:" + "0" * 64),
    ("float32_gain_hash", "sha256:" + "0" * 64), ("runtime_norm_hash", "sha256:" + "0" * 64),
    ("runtime_float32_gain_hash", "sha256:" + "0" * 64),
])
def test_changed_layer_source_file_and_gain_evidence_is_refused(side, layer, field, value):
    response = published_reply()
    parent = response["metadata"]["import_source"] if side == "metadata" else response[side]
    parent["layers"][layer][field] = value
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("side", ["metadata", "import_binding"])
@pytest.mark.parametrize("name", ["W_enc", "W_enc_original", "W_dec", "b_enc", "b_dec", "threshold"])
@pytest.mark.parametrize("fault", ["changed", "missing", "extra"])
def test_all_six_tensor_members_require_exact_import_evidence(side, name, fault):
    response = published_reply()
    parent = response["metadata"]["import_source"] if side == "metadata" else response[side]
    hashes = parent["layers"][0]["tensor_hashes"]
    if fault == "missing":
        hashes.pop(name)
    elif fault == "extra":
        hashes[name + "_inverse"] = hashes[name]
    else:
        hashes[name] = "sha256:" + "0" * 64
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("fault", ["missing_layer", "duplicate_layer", "both_missing_tensor", "both_wrong_gain_dtype",
                                  "both_wrong_runtime_dtype", "both_legacy_conversion", "both_inverse_conversion"])
def test_matching_but_incomplete_or_legacy_import_records_are_refused(fault):
    response = published_reply()
    imported, binding = response["metadata"]["import_source"], response["import_binding"]
    for record in (imported, binding):
        if fault == "missing_layer":
            record["layers"].pop()
        elif fault == "duplicate_layer":
            record["layers"][25] = deepcopy(record["layers"][24])
        elif fault == "both_missing_tensor":
            record["layers"][0]["tensor_hashes"].pop("W_enc_original")
        elif fault == "both_wrong_gain_dtype":
            record["layers"][0]["source_gain_dtype"] = "bfloat16"
        elif fault == "both_wrong_runtime_dtype":
            record["layers"][0]["runtime_norm_dtype"] = "float32"
        elif fault == "both_legacy_conversion":
            record["reference_conversion"]["version"] = 1
        else:
            record["reference_conversion"]["tensors"]["W_enc_original"] = "W_enc * gain"
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("path,value", [
    (("format_version",), 1), (("format_version",), 2.0), (("complete",), 1),
    (("tokens",), 0), (("tokens",), 2000001), (("tokens",), 2000000.0), (("token_budget",), 1999999),
    (("layers",), 25), (("features",), 8192), (("top_examples",), 8), (("window_tokens",), 32),
    (("tie_order",), "activation_desc"), (("vocab_size",), 10), (("documents",), True),
    (("encoder_policy",), "live_input_v1"), (("threshold_policy",), "cast_to_float32"),
    (("reference_input", "coordinate"), "post_gain_inverse"), (("reference_input", "dtype"), "float32"),
    (("dictionary_fingerprint",), "0" * 64), (("dictionary_source_hash",), None),
    (("context_length",), 255), (("context_policy", "prefix_ids"), []),
    (("context_policy", "content_context"), 256), (("context_policy", "version"), True),
    (("example_context_policy", "prefix_ids"), [True]), (("example_context_policy", "model_context"), 33),
    (("provenance", "source_id"), "other"), (("provenance", "partition"), "quality"),
    (("provenance", "partitions", "index"), [9400, 10000]),
    (("provenance", "dataset_revision"), "0" * 40), (("provenance", "publisher_training_overlap"), "known"),
    (("provenance", "source_kind"), "staged_jsonl"), (("provenance", "duplicate_documents"), "all occurrences"),
    (("files", "tokens.bin", "bytes"), 0), (("files", "tokens.bin", "sha256"), ""),
    (("model_encoding", "version"), True), (("model_encoding", "autocast_dtype"), "torch.bfloat16"),
    (("model_encoding", "float32_matmul_precision"), "high"),
    (("model_encoding", "forward_settings_hash"), "0" * 64),
    (("model_encoding", "tensors"), []),
    (("model_encoding", "tensors", 0, 1), "torch.float32"),
    (("model_encoding", "attention_backends"), [["", "sdpa"]]),
    (("model_encoding", "reference_profile"), "circuit_tracer_82996f14"),
    (("model_encoding", "reference_profile", "profile"), "arbitrary"),
    (("model_encoding", "reference_profile", "version"), True),
    (("model_encoding", "reference_profile", "dtype"), "float32"),
    (("model_encoding", "reference_profile", "reference_revision"), "0" * 40),
    (("model_encoding", "reference_profile", "encoder_coordinate"), "post_gain_inverse"),
    (("model_encoding", "reference_profile", "capture"), "model.layers.mlp.input"),
    (("model_encoding", "reference_profile", "gain"), "source_float32_gain"),
    (("model_encoding", "reference_profile", "runtime"), "default"),
    (("model_encoding", "reference_profile", "operation_sha256"), {}),
    (("model_encoding", "reference_profile", "operations"), {}),
    (("model_encoding", "reference_profile", "torch_version"), ""),
])
def test_semantically_wrong_index_refused_even_with_a_new_manifest_hash(path, value):
    response = published_reply()
    replace(response["index"], path, value)
    seal_index(response)
    result = qualify(response)
    assert not result["qualified"], (path, result)


@pytest.mark.parametrize("path", [
    ("dictionary_source_hash",), ("dictionary_hash",), ("dictionary_fingerprint",), ("tokenizer_hash",),
    ("files", "table.npy", "sha256"), ("files", "counts.npy", "sha256"),
    ("files", "tokens.bin", "sha256"), ("files", "documents.bin", "sha256"),
    ("model_encoding", "reference_profile", "operation_sha256", "caliper.interp._gemma_reference.normalized_input"),
    ("model_encoding", "reference_profile", "torch_version"),
    ("model_encoding", "tensors", 0, 2),
])
def test_current_manifest_hash_covers_all_resource_and_arithmetic_fields(path):
    response = published_reply()
    value = "2.11.0" if path[-1] == "torch_version" else "mps:0" if path[-1] == 2 else "0" * 64
    replace(response["index"], path, value)
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("parent,key", [("import_binding", "layers"), ("import_binding", "reference_conversion"),
    ("model_binding", "tensor_hash"), ("model_binding", "forward_settings_hash"), ("tensor_binding", "tensor_hash"),
    ("index", "model_encoding"), ("index", "dictionary_fingerprint"), ("index", "files"),
    ("index", "reference_input"), ("index", "context_policy"), ("index", "example_context_policy")])
def test_missing_current_members_cannot_be_forged_by_passed_markers(parent, key):
    response = published_reply()
    response[parent].pop(key)
    response.update(qualified=True, verified=True, profile="circuit_tracer_82996f14")
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("field", ["tensors", "attention_backends"])
@pytest.mark.parametrize("fault", ["missing", "duplicate", "renamed"])
def test_model_encoding_requires_every_current_member(field, fault):
    response = published_reply()
    rows = response["index"]["model_encoding"][field]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = deepcopy(rows[0])
    else:
        rows[-1][0] = "parameter:unrelated.weight" if field == "tensors" else "unrelated"
    seal_index(response)
    assert not qualify(response)["qualified"]


def test_wrapped_operations_retain_every_parent_hash():
    response = published_reply()
    hashes = response["index"]["model_encoding"]["reference_profile"]["operation_sha256"]
    name = "transformers.models.gemma2.modeling_gemma2.Gemma2Model.forward"
    for _ in range(3):
        name += ".__wrapped__"
        hashes[name] = digest(name)
    seal_index(response)
    assert qualify(response)["qualified"]
    hashes.pop(name.removesuffix(".__wrapped__"))
    seal_index(response)
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("tokens,documents", [(8, 1), (1999999, 1000), (2000001, 2000001)])
def test_index_size_is_a_measurement_with_exact_manifest_identity(tokens, documents):
    response = published_reply()
    response["index"].update(tokens=tokens, token_budget=tokens, documents=documents)
    seal_index(response)
    result = qualify(response)
    assert result["qualified"], result
    assert result["index_measurements"] == {"tokens": tokens, "token_budget": tokens, "documents": documents}
    response["index"]["tokens"] += 1
    assert not qualify(response)["qualified"]


@pytest.mark.parametrize("field,value", [("tokens", 0), ("tokens", True), ("token_budget", None),
    ("documents", 0), ("documents", True), ("documents", 2000001), ("complete", False)])
def test_index_measurements_require_complete_valid_counts(field, value):
    response = published_reply()
    response["index"][field] = value
    seal_index(response)
    assert not qualify(response)["qualified"]
