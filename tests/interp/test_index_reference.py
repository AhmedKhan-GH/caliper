"""Original-coordinate indexes with tiny BF16 CPU fixtures."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from caliper.interp import _index
from caliper.interp._dictionary import (Dictionary, JumpReLUTranscoder, dictionary_fingerprint,
                                       load_dictionary, model_identity)
from caliper.interp._gemma_reference import PROFILE, install_reference_profile, profile_metadata
from caliper.interp._memory import CleanStop
from tools.interp.model_load import hidden_forward
from .test_gemma_reference import reference_fixture
from .test_gemma_runtime import memory
from .test_index_precision import rewrite_manifest


REFERENCE_INPUT = {"coordinate": "gemma_scope_pre_gain", "encoder": "W_enc_original", "dtype": "bfloat16"}


@pytest.fixture
def original_index(tmp_path):
    model = reference_fixture()
    model.config.bos_token_id, model.config.pad_token_id = 1, 0
    install_reference_profile(model, PROFILE)
    tokenizer = SimpleNamespace(bos_token_id=1, pad_token_id=0, eos_token_id=2,
        convert_ids_to_tokens=lambda i: str(i), decode=lambda ids, **kwargs: " ".join(map(str, ids)))
    layers = []
    generator = torch.Generator().manual_seed(8123)
    for block in model.model.layers:
        source = torch.randn(7, 24, generator=generator) * .5
        gain = (block.pre_feedforward_layernorm.weight.float() + 1).bfloat16()
        layers.append(JumpReLUTranscoder(
            (source / gain.float()).bfloat16(), torch.full((7,), .5, dtype=torch.bfloat16),
            torch.randn(7, 24, generator=generator).bfloat16(), torch.zeros(24, dtype=torch.bfloat16),
            torch.full((7,), .25, dtype=torch.bfloat16), W_enc_original=source.bfloat16()))
    metadata = {"format_version": 1, "model_id": model.config._name_or_path,
        "model_hash": model_identity(model), "revision": model.config._commit_hash,
        "layers": 2, "width": 24, "latents": 7, "activation": "jumprelu", "k": None,
        "read_points": [f"model.layers.{i}.mlp_in" for i in range(2)],
        "write_points": [f"model.layers.{i}.mlp_out" for i in range(2)],
        "reference_input": dict(REFERENCE_INPUT),
        "training": {"dataset_id": "fixture", "dataset_config": "random-inputs",
            "dataset_revision": "seed-8123", "split": "train", "partitions": {},
            "tokens": 0, "elapsed_seconds": 0, "peak_memory_bytes": 0},
        "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None, "evaluated_tokens": 0}}
    dictionary = Dictionary(metadata, layers)
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    for i, layer in enumerate(layers):
        save_file(layer.state_dict(), tmp_path / f"layer_{i:02d}.safetensors")
    return SimpleNamespace(root=tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary)


def oracle(case, ids, mask):
    """Read residuals independently of the reference capture and index encoder."""
    raw, post = {}, {}
    with ExitStack() as stack, torch.inference_mode():
        for i, block in enumerate(case.model.model.layers):
            def read(module, args, i=i):
                x = args[0].float()
                raw[i] = (x / (x.square().mean(-1, keepdim=True) + module.eps).sqrt()).bfloat16()
            def output(module, args, value, i=i):
                post[i] = value.clone()
            stack.callback(block.pre_feedforward_layernorm.register_forward_pre_hook(read).remove)
            stack.callback(block.pre_feedforward_layernorm.register_forward_hook(output).remove)
        hidden = case.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    values = []
    for i, layer in enumerate(case.dictionary.layers):
        pre = F.linear(raw[i], layer.W_enc_original, layer.b_enc)
        values.append(pre * (pre > layer.threshold))
    return values, raw, post, hidden


def build(case, documents=None, **kwargs):
    return _index.build_index(case.root, documents or [("a", [3, 4, 5, 6, 7]), ("b", [8]), ("c", [9, 10, 11])],
        model=case.model, tokenizer=case.tokenizer, dictionary=case.dictionary,
        provenance={"partition": "fixture"}, context_length=4, **kwargs)


def open_index(case):
    return _index.FeatureIndex(case.root, model=case.model, tokenizer=case.tokenizer, dictionary=case.dictionary)


def test_reference_top_examples_use_exact_source_encoder_and_source_offsets(original_index):
    case = original_index
    expected, inverted, compatibility = [[], []], [[], []], [[], []]
    for tokens in ([3, 4, 5, 6, 7], [8], [9, 10]):
        for begin in range(0, len(tokens), 3):
            ids = torch.tensor([[1, *tokens[begin:begin + 3]]])
            values, raw, post, _ = oracle(case, ids, torch.ones_like(ids))
            for i, layer in enumerate(case.dictionary.layers):
                gain = (case.model.model.layers[i].pre_feedforward_layernorm.weight.float() + 1).bfloat16()
                expected[i].append(values[i][0, 1:])
                wrong = F.linear(post[i] / gain, layer.W_enc_original, layer.b_enc)
                inverted[i].append((wrong * (wrong > layer.threshold))[0, 1:])
                compatibility[i].append(layer.encode(post[i])[0, 1:])
    assert any(not torch.equal(torch.cat(expected[i]), torch.cat(inverted[i])) for i in range(2))
    assert any(not torch.equal(torch.cat(expected[i]), torch.cat(compatibility[i])) for i in range(2))
    manifest = build(case, max_tokens=8)
    assert manifest["encoder_policy"] == _index.REFERENCE_ENCODER_POLICY
    assert manifest["reference_input"] == REFERENCE_INPUT
    assert manifest["model_encoding"]["reference_profile"] == profile_metadata(case.model)
    assert manifest["dictionary_fingerprint"] == dictionary_fingerprint(case.root, case.dictionary.metadata)
    assert manifest["threshold_policy"] == _index.ENCODER_POLICY
    assert manifest["tokens"] == 8 and manifest["context_policy"]["prefix_ids"] == [1]
    with open_index(case) as index:
        assert index.tokens.tolist() == [3, 4, 5, 6, 7, 8, 9, 10]
        assert index.documents["length"].tolist() == [5, 1, 2]
        assert index.documents["truncated"].tolist() == [False, False, True]
        for i in range(2):
            values = torch.cat(expected[i]).float().numpy()
            np.testing.assert_array_equal(index.counts[i], (values > 0).sum(0))
            for feature in range(7):
                ranked = sorted((offset for offset in range(8) if values[offset, feature] > 0),
                                key=lambda offset: (-values[offset, feature], offset))
                np.testing.assert_array_equal(index.table[i, feature]["offset"][:len(ranked)], ranked)
                np.testing.assert_array_equal(index.table[i, feature]["activation"][:len(ranked)],
                                              values[ranked, feature])
    loaded = load_dictionary(case.root, model=case.model)
    assert all(layer.W_enc_original.dtype == torch.bfloat16 for layer in loaded.layers)
    with _index.FeatureIndex(case.root, model=case.model, tokenizer=case.tokenizer) as index:
        assert index.metadata["dictionary_hash"] == manifest["dictionary_hash"]


@pytest.mark.parametrize("prefixed", [True, False])
def test_both_query_branches_use_original_coordinates_and_cache(original_index, monkeypatch, prefixed):
    case = original_index
    build(case)
    with open_index(case) as index:
        # Gemma selects the prefixed branch. Exercise the shared fallback branch
        # directly to prevent a future caller from reverting its encoder policy.
        if not prefixed:
            index.example_context_policy = None
        feature = int(np.argmax(index.counts[0]))
        rows = index._windows(0, feature)
        assert rows and {row["document_id"] for row in rows} >= {"a", "c"}
        width = 34 if prefixed else 33
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            content = [token for token in row["token_ids"] if token is not None]
            tokens = ([1] if prefixed else []) + content
            ids[i, :len(tokens)] = torch.tensor(tokens)
            mask[i, :len(tokens)] = 1
        expected, _, _, _ = oracle(case, ids, mask)
        calls = []
        forward = _index._forward
        def observed(model, ids, mask):
            calls.append(tuple(ids.shape))
            return forward(model, ids, mask)
        monkeypatch.setattr(_index, "_forward", observed)
        actual = index.examples(0, feature)
        assert calls == [(len(rows), width)]
        for i, row in enumerate(actual):
            values = [value for value in row["activations"] if value is not None]
            start = int(prefixed)
            assert values == expected[0][i, start:start + len(values), feature].tolist()
            assert len(row["activations"]) == 33 and row["center"] == 16
            assert all(value is None for value, valid in zip(row["activations"], row["valid"]) if not valid)
        assert index.examples(0, feature, 1) == actual[:1]
        assert len(calls) == 1
        actual[0]["activations"][16] = -100
        assert index.examples(0, feature)[0]["activations"][16] != -100


@pytest.mark.parametrize("change", ["legacy", "policy", "coordinate", "profile", "dtype", "operations", "fingerprint"])
def test_reference_index_rejects_stale_identity_before_maps(original_index, monkeypatch, change):
    case = original_index
    manifest = build(case)
    if change == "legacy":
        changes = {"format_version": 1, "encoder_policy": None, "model_encoding": None}
    elif change == "policy":
        changes = {"encoder_policy": _index.ENCODER_POLICY}
    elif change == "coordinate":
        changes = {"reference_input": None}
    elif change == "fingerprint":
        changes = {"dictionary_fingerprint": "old-float32-artifact"}
    else:
        encoding = deepcopy(manifest["model_encoding"])
        field = {"profile": "profile", "dtype": "dtype", "operations": "operation_sha256"}[change]
        encoding["reference_profile"][field] = "changed"
        changes = {"model_encoding": encoding}
    rewrite_manifest(case.root, **changes)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("stale maps opened"))
    with pytest.raises(ValueError, match="rebuild"):
        open_index(case)


@pytest.mark.parametrize("change", ["source_encoder", "compatibility_encoder", "source_file", "profile"])
def test_reference_index_binds_both_encoders_and_runtime_profile(original_index, monkeypatch, change):
    case = original_index
    manifest = build(case)
    if change == "profile":
        case.model._caliper_reference_profile = "other"
    elif change == "source_file":
        weights = case.dictionary.layers[0].state_dict()
        weights["W_enc_original"] = weights["W_enc_original"].clone() + 1
        save_file(weights, case.root / "layer_00.safetensors")
        assert dictionary_fingerprint(case.root, case.dictionary.metadata) != manifest["dictionary_fingerprint"]
    else:
        key = "W_enc_original" if change == "source_encoder" else "W_enc"
        getattr(case.dictionary.layers[0], key).data[0, 0] += 1
        assert _index._dictionary_hash(case.dictionary) != manifest["dictionary_hash"]
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("stale maps opened"))
    with pytest.raises(ValueError, match="identity|profile"):
        open_index(case)


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "declaration", "ordinary_model"])
def test_reference_artifact_mismatch_refuses_before_index_output(original_index, monkeypatch, damage):
    case = original_index
    if damage == "missing":
        del case.dictionary.layers[0].W_enc_original
    elif damage == "shape":
        case.dictionary.layers[0].W_enc_original = torch.nn.Parameter(torch.ones(1, 24, dtype=torch.bfloat16))
    elif damage == "dtype":
        case.dictionary.layers[0].threshold.data = case.dictionary.layers[0].threshold.float()
    elif damage == "declaration":
        del case.dictionary.metadata["reference_input"]
    else:
        ordinary = reference_fixture()
        ordinary.config.bos_token_id, ordinary.config.pad_token_id = 1, 0
        case.model = ordinary
        case.dictionary.metadata["model_hash"] = model_identity(ordinary)
    monkeypatch.setattr(_index, "_forward", lambda *args: pytest.fail("invalid forward ran"))
    with pytest.raises(ValueError, match="original encoder"):
        build(case)
    assert not (case.root / "index").exists()


@pytest.mark.parametrize("route", ["index", "hidden", "capture", "cached"])
@pytest.mark.parametrize("damage", ["autocast", "matmul", "adapter"])
def test_decoder_and_capture_require_reference_runtime(original_index, monkeypatch, route, damage):
    case = original_index
    with ExitStack() as stack:
        if route == "cached":
            build(case)
            index = stack.enter_context(open_index(case))
            feature = int(np.argmax(index.counts[0]))
            index.examples(0, feature)
        if damage == "matmul":
            monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")
        elif damage == "adapter":
            case.model.model.layers[0].pre_feedforward_layernorm.eps *= 2
        stack.enter_context(torch.autocast("cpu", dtype=torch.bfloat16) if damage == "autocast" else nullcontext())
        ids = torch.tensor([[1, 3, 5]])
        with pytest.raises(ValueError, match="reference"):
            if route == "hidden":
                hidden_forward(case.model, ids, torch.ones_like(ids))
            elif route == "index":
                _index._forward(case.model, ids, torch.ones_like(ids))
            elif route == "cached":
                index.examples(0, feature)
            else:
                with _index._capture_inputs(case.model, case.dictionary, case.dictionary.metadata["read_points"],
                                            ids, memory()):
                    pytest.fail("invalid capture entered")


def test_hidden_decoder_matches_reference_without_vocabulary_projection(original_index):
    case = original_index
    ids = torch.tensor([[1, 3, 5, 0], [1, 8, 9, 10]])
    mask = (ids != 0).long()
    _, raw, _, expected = oracle(case, ids, mask)
    def refuse(*args):
        pytest.fail("vocabulary projection ran")
    with ExitStack() as stack, torch.inference_mode():
        stack.callback(case.model.lm_head.register_forward_pre_hook(refuse).remove)
        with _index._capture_inputs(case.model, case.dictionary, case.dictionary.metadata["read_points"],
                                    ids, memory()) as captured:
            actual = hidden_forward(case.model, ids, mask)
        assert torch.equal(actual, expected)
        assert all(torch.equal(captured[f"model.layers.{i}.mlp_in"], value) for i, value in raw.items())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in case.model.modules())


def test_failed_decoder_clears_partial_capture_and_hooks(original_index):
    case = original_index
    ids = torch.tensor([[1, 3, 5]])
    def fail(*args):
        raise RuntimeError("fixture decoder failure")
    with ExitStack() as stack:
        stack.callback(case.model.model.layers[1].register_forward_pre_hook(fail).remove)
        with pytest.raises(RuntimeError, match="fixture decoder"):
            with _index._capture_inputs(case.model, case.dictionary, case.dictionary.metadata["read_points"],
                                        ids, memory()) as values:
                hidden_forward(case.model, ids)
        assert values == {}
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in case.model.modules())


def test_reference_capture_and_encoder_admit_before_allocation(original_index, monkeypatch):
    case = original_index
    ids = torch.tensor([[1, 3, 5]])
    guard = memory()
    retained = ids.numel() * 24 * 2 * 2
    guard.limit = retained - 1
    with pytest.raises(CleanStop, match="capture allocation"):
        with _index._capture_inputs(case.model, case.dictionary, case.dictionary.metadata["read_points"], ids, guard):
            pytest.fail("capture entered before admission")
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in case.model.modules())
    x = torch.ones(3, 24, dtype=torch.bfloat16)
    guard = memory()
    guard.limit = 3 * 7 * (4 * 2 + 1) - 1
    monkeypatch.setattr(F, "linear", lambda *args: pytest.fail("encoder allocated before admission"))
    with pytest.raises(CleanStop, match="encoder allocation"):
        _index._index_features(case.dictionary, 0, x, memory=guard)


def test_reference_index_preserves_prospective_host_reserve(original_index, monkeypatch):
    import psutil
    case = original_index
    monkeypatch.setattr(_index, "host_reserve_for_model", lambda model: 8_000_000_000)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8_001_000_000))
    monkeypatch.setattr(_index, "_forward", lambda *args: pytest.fail("forward ran before host admission"))
    with pytest.raises(CleanStop, match="8 GB host reserve"):
        build(case)
    assert not (case.root / "index/index.json").exists()


def test_reference_query_preserves_host_reserve_before_forward(original_index, monkeypatch):
    import psutil
    case = original_index
    build(case)
    with open_index(case) as index:
        index.memory._host_reserve_bytes = 8_000_000_000
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8_001_000_000))
        monkeypatch.setattr(_index, "_forward", lambda *args: pytest.fail("query ran before host admission"))
        feature = int(np.argmax(index.counts[0]))
        with pytest.raises(CleanStop, match="example allocation"):
            index.examples(0, feature)
