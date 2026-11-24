"""Index and example activations preserve the live model input precision."""
import json
import math
import hashlib

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from caliper.interp import capture, load_dictionary
from caliper.interp._index import FeatureIndex, build_index
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import fixture_manifest


def precision_fixture(tmp_path, activation):
    model = TinyLM(layers=1, width=4, heads=1, latents=4, k=1, dtype=torch.float32)
    target = 1 + 2**-12
    with torch.no_grad():
        model.embed_tokens.weight.fill_(1)
        model.layers[0].self_attn.o_proj.weight.zero_()
        norm = model.layers[0].post_attention_layernorm
        norm.weight.fill_(target * math.sqrt(1 + norm.eps))
    metadata = fixture_manifest(model)
    metadata.update(activation=activation, k=1 if activation == "topk" else None)
    weights = {"W_enc": torch.zeros(4, 4, dtype=torch.float16),
               "b_enc": torch.zeros(4, dtype=torch.float16),
               "W_dec": torch.ones(4, 4, dtype=torch.float16),
               "b_dec": torch.zeros(4, dtype=torch.float16)}
    weights["W_enc"][0, 0] = 1
    if activation == "jumprelu":
        weights["threshold"] = torch.ones(4, dtype=torch.float16)
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    save_file(weights, tmp_path / "layer_00.safetensors")
    dictionary = load_dictionary(tmp_path, model=model, dtype=torch.float16)
    return model, dictionary, weights


@pytest.mark.parametrize("activation", ["topk", "jumprelu"])
def test_half_dictionary_preserves_live_threshold_and_example_values(tmp_path, activation):
    model, dictionary, weights = precision_fixture(tmp_path, activation)
    metadata = dictionary.metadata
    point = metadata["read_points"][0]
    tokens = [65, 66, 67, 68]
    with torch.no_grad(), capture(model, [point]) as saved:
        model(torch.tensor([tokens]))
        # One encoder coefficient is one. The independent scalar oracle is x[0].
        expected = saved[point][0, :, 0].tolist()
        assert all(value > 1 for value in expected)
        assert saved[point][..., 0].half().eq(1).all()
    build_index(tmp_path, [("first", tokens)], model=model, tokenizer=TinyTokenizer(),
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    with FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer(), dictionary=dictionary) as index:
        assert int(index.counts[0, 0]) == 4
        assert not index.counts[0, 1:].any()
        np.testing.assert_array_equal(index.table[0, 0]["offset"][:4], list(range(4)))
        np.testing.assert_array_equal(index.table[0, 0]["activation"][:4], expected)
        rows = index.examples(0, 0)
        assert len(rows) == 4
        for row in rows:
            assert row["activation"] == expected[row["position"]]
            assert [value for value in row["activations"] if value is not None] == expected
    for name, value in dictionary.layers[0].state_dict().items():
        assert value.dtype == torch.float16
        assert torch.equal(value, weights[name])


def rewrite_manifest(folder, **changes):
    path = folder / "index" / "index.json"
    manifest = json.loads(path.read_text())
    manifest.pop("index_hash")
    for key, value in changes.items():
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["index_hash"] = hashlib.sha256(encoded.encode()).hexdigest()
    path.write_text(json.dumps(manifest))


def test_legacy_mixed_precision_threshold_index_requires_rebuild(tmp_path, monkeypatch):
    from caliper.interp import _index
    model, dictionary, _ = precision_fixture(tmp_path, "jumprelu")
    # Reproduce the actual version 1 encoder arithmetic, before its live-input fix.
    with monkeypatch.context() as patch:
        patch.setattr(_index, "_trace_features", lambda layer, x, **kw: layer.encode(x.to(layer.W_enc.dtype)))
        build_index(tmp_path, [("first", [65, 66, 67, 68])], model=model, tokenizer=TinyTokenizer(),
                    dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    assert not np.load(tmp_path / "index" / "counts.npy").any()
    rewrite_manifest(tmp_path, format_version=1, encoder_policy=None, model_encoding=None)
    with pytest.raises(ValueError, match="legacy index encoder precision differs; rebuild"):
        FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer(), dictionary=dictionary)


@pytest.mark.parametrize("policy", [None, "storage_precision", "live_input_v2"])
def test_new_index_requires_exact_encoder_policy(tmp_path, policy):
    model, dictionary, _ = precision_fixture(tmp_path, "topk")
    manifest = build_index(tmp_path, [("first", [65])], model=model, tokenizer=TinyTokenizer(),
                           dictionary=dictionary, provenance={"partition": "fixture"})
    assert manifest["format_version"] == 2 and manifest["encoder_policy"] == "live_input_v1"
    rewrite_manifest(tmp_path, encoder_policy=policy)
    with pytest.raises(ValueError, match="unsupported index encoder policy"):
        FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer(), dictionary=dictionary)


def test_legacy_uniform_precision_preserves_results(tmp_path):
    from .test_dictionary import write_fixture
    model, _ = write_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    build_index(tmp_path, [("first", [65, 66, 67, 68])], model=model, tokenizer=TinyTokenizer(),
                dictionary=dictionary, provenance={"partition": "fixture"})
    with FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer(), dictionary=dictionary) as index:
        layer, feature = map(int, np.argwhere(index.counts > 0)[0])
        expected = index.examples(layer, feature)
    rewrite_manifest(tmp_path, format_version=1, encoder_policy=None, model_encoding=None)
    with FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer(), dictionary=dictionary) as index:
        assert index.examples(layer, feature) == expected


def test_verified_checkpoint_precision_change_refuses_stale_threshold_counts(tmp_path, monkeypatch):
    from caliper.interp._dictionary import model_identity, verify_hf_model
    from caliper.interp._frozen import _trace_features
    from .test_gemma_runtime import gemma_model, local_gemma, runtime_dictionary
    model = gemma_model()
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(1)
        model.model.layers[0].self_attn.o_proj.weight.zero_()
        norm = model.model.layers[0].pre_feedforward_layernorm
        norm.weight.fill_((1 + 2**-12) * math.sqrt(16 + norm.eps) / 4 - 1)
    snapshot = local_gemma(tmp_path, model)
    verify_hf_model(model, snapshot)
    dictionary = runtime_dictionary(model).half()
    with torch.no_grad():
        for layer in dictionary.layers:
            layer.W_enc.zero_()
            layer.W_enc[0, 0] = 1
            layer.b_enc.zero_()
            layer.threshold.fill_(1)
    (tmp_path / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    for i, layer in enumerate(dictionary.layers):
        save_file(layer.state_dict(), tmp_path / f"layer_{i:02d}.safetensors")
    tokens, point = [3, 4, 5, 6], "model.layers.0.mlp_in"
    with torch.no_grad(), capture(model, [point]) as saved:
        model(torch.tensor([tokens]), use_cache=False)
    assert saved[point][..., 0].gt(1).all()
    build_index(tmp_path, [("first", tokens)], model=model, tokenizer=TinyTokenizer(),
                dictionary=dictionary, provenance={"partition": "fixture"})
    assert np.load(tmp_path / "index" / "counts.npy")[0, 0] == 4
    half = gemma_model().half()
    half.load_state_dict(model.state_dict())
    verify_hf_model(half, snapshot)
    assert model_identity(half) == model_identity(model)
    with torch.no_grad(), capture(half, [point]) as saved:
        half(torch.tensor([tokens]), use_cache=False)
    assert saved[point][..., 0].eq(1).all()
    assert not _trace_features(dictionary.layers[0], saved[point])[..., 0].any()
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("stale index maps opened"))
    with pytest.raises(ValueError, match="index model encoding differs; rebuild"):
        FeatureIndex(tmp_path, model=half, tokenizer=TinyTokenizer(), dictionary=dictionary)


@pytest.mark.parametrize("change", ["missing", "matmul", "autocast"])
def test_index_requires_matching_execution_evidence(tmp_path, monkeypatch, change):
    from contextlib import nullcontext
    model, dictionary, _ = precision_fixture(tmp_path, "topk")
    build_index(tmp_path, [("first", [65])], model=model, tokenizer=TinyTokenizer(),
                dictionary=dictionary, provenance={"partition": "fixture"})
    if change == "missing":
        rewrite_manifest(tmp_path, model_encoding=None)
    elif change == "matmul":
        original = torch.get_float32_matmul_precision()
        monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high" if original != "high" else "highest")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("stale index maps opened"))
    context = torch.autocast("cpu", dtype=torch.bfloat16) if change == "autocast" else nullcontext()
    with context, pytest.raises(ValueError, match="index model encoding differs; rebuild"):
        FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer(), dictionary=dictionary)
