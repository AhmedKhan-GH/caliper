"""Exact JumpReLU boundaries and caller admission on CPU."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from caliper.interp import _dictionary, _frozen, _index
from caliper.interp._dictionary import JumpReLUTranscoder
from caliper.interp._memory import CleanStop


CASES = [(torch.float16, torch.float32, 2**-13),
         (torch.bfloat16, torch.float32, 2**-10),
         (torch.float32, torch.float64, 2**-26),
         (torch.float64, torch.float32, 0.)]


def boundary_layer(live, stored, gap, encoder_matches):
    dtype = live if encoder_matches else stored
    weights = torch.zeros(7, 4, dtype=dtype)
    weights[:, 0] = 1
    thresholds = torch.tensor([1 - gap, 1, 1 + gap, 1, 1, 1, 1], dtype=stored)
    layer = JumpReLUTranscoder(weights, torch.zeros(7, dtype=dtype),
                              weights.clone(), torch.zeros(4, dtype=dtype), thresholds)
    one = torch.tensor(1., dtype=live)
    x = torch.zeros(3, 4, dtype=live)
    x[:, 0] = torch.stack([torch.nextafter(one, one * 0), one, torch.nextafter(one, one * 2)])
    return layer, x


@pytest.mark.parametrize("live,stored,gap", CASES)
@pytest.mark.parametrize("encoder_matches", [False, True])
def test_live_boundary_rounding_and_bounded_threshold_copies(monkeypatch, live, stored, gap, encoder_matches):
    layer, x = boundary_layer(live, stored, gap, encoder_matches)
    before = {name: (p.data_ptr(), p.clone()) for name, p in layer.named_parameters()}
    pre = F.linear(x, layer.W_enc.to(live), layer.b_enc.to(live))
    expected = pre * (pre > layer.threshold.to(live))
    monkeypatch.setattr(_dictionary, "_THRESHOLD_BLOCK_ELEMENTS", 2)
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", 8)
    original_to = torch.Tensor.to
    copies, stages = [], []

    def bounded_to(value, *args, **kwargs):
        if value.untyped_storage().data_ptr() == layer.threshold.untyped_storage().data_ptr():
            assert value.numel() <= 2
            copies.append(value.numel())
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", bounded_to)
    with torch.no_grad():
        actual = _frozen._trace_features(layer, x, check=stages.append)
    assert torch.equal(actual, expected)
    assert not actual[:2].any() and torch.equal(actual[2], pre[2])
    assert copies == [2, 2, 2, 1]
    assert stages.count("dictionary threshold block") == 4
    for name, parameter in layer.named_parameters():
        pointer, value = before[name]
        assert parameter.data_ptr() == pointer and torch.equal(parameter, value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_uniform_precision_preserves_reference_numerics(dtype, monkeypatch):
    layer, x = boundary_layer(dtype, dtype, 0., True)
    pre = F.linear(x, layer.W_enc, layer.b_enc)
    expected = pre * (pre > layer.threshold)
    monkeypatch.setattr(torch, "empty_like", lambda *a, **kw: pytest.fail("uniform activation copied thresholds"))
    assert torch.equal(layer.activate(pre), expected)
    assert torch.equal(_frozen._trace_features(layer, x), expected)


@pytest.mark.parametrize("phase", ["encoder", "threshold_admission", "second_threshold_block"])
def test_threshold_only_mismatch_obeys_lower_caller_limit(monkeypatch, phase):
    layer, x = boundary_layer(torch.float32, torch.float64, 2**-26, True)
    memory = SimpleNamespace(check=lambda stage: None, limit=100_000,
                             current=0, peak=0, device=torch.device("cpu"))
    monkeypatch.setattr(_dictionary, "_THRESHOLD_BLOCK_ELEMENTS", 2)
    converted = []
    original_to = torch.Tensor.to

    def track(value, *args, **kwargs):
        if value.untyped_storage().data_ptr() == layer.threshold.untyped_storage().data_ptr():
            converted.append(value.numel())
        return original_to(value, *args, **kwargs)

    def check(stage):
        if ((phase == "threshold_admission" and stage == "dictionary threshold admission")
                or (phase == "second_threshold_block" and converted)):
            memory.limit = 1

    if phase == "encoder":
        memory.limit = 1
        monkeypatch.setattr(torch.Tensor, "new_empty", lambda *a, **kw: pytest.fail("encoder allocated before refusal"))
    monkeypatch.setattr(torch.Tensor, "to", track)
    with pytest.raises(CleanStop, match="encoder allocation"):
        _frozen._trace_features(layer, x, memory, check)
    assert converted == ([2] if phase == "second_threshold_block" else [])


def test_legacy_index_refuses_threshold_only_precision_mismatch():
    layer, _ = boundary_layer(torch.float32, torch.float64, 2**-26, True)
    model = SimpleNamespace(state_dict=lambda: {"weight": torch.ones(1)})
    dictionary = SimpleNamespace(metadata={}, layers=[layer])
    with pytest.raises(ValueError, match="legacy index encoder precision differs"):
        _index._encoding_policy({"format_version": 1}, model, dictionary, None)
    layer.threshold.data = layer.threshold.float()
    _index._encoding_policy({"format_version": 1}, model, dictionary, None)


@pytest.mark.parametrize("mixed", [False, True])
def test_version_two_index_requires_fixed_threshold_policy_only_for_mixed_precision(monkeypatch, mixed):
    layer, _ = boundary_layer(torch.float32, torch.float64 if mixed else torch.float32, 0., True)
    encoding = {"autocast_dtype": None, "tensors": [["weight", "torch.float32", "cpu"]]}
    monkeypatch.setattr(_index, "_model_encoding", lambda *a: encoding)
    manifest = {"format_version": 2, "encoder_policy": "live_input_v1", "model_encoding": encoding}
    dictionary = SimpleNamespace(metadata={}, layers=[layer])
    if mixed:
        with pytest.raises(ValueError, match="threshold precision differs"):
            _index._encoding_policy(manifest, None, dictionary, None)
    else:
        _index._encoding_policy(manifest, None, dictionary, None)
    manifest["threshold_policy"] = "live_input_v1"
    _index._encoding_policy(manifest, None, dictionary, None)


def test_index_build_admits_threshold_only_conversion_before_allocation(tmp_path, monkeypatch):
    import json
    from safetensors.torch import save_file
    from .fixture import TinyLM, TinyTokenizer
    from .test_dictionary import fixture_manifest
    from .test_research_evaluation import guard

    model = TinyLM(layers=1, width=4, heads=1, latents=7, k=1, dtype=torch.float32)
    layer, _ = boundary_layer(torch.float32, torch.float64, 2**-26, True)
    metadata = fixture_manifest(model)
    metadata.update(activation="jumprelu", k=None)
    dictionary = _dictionary.Dictionary(metadata, [layer])
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    save_file(layer.state_dict(), tmp_path / "layer_00.safetensors")
    memory = guard()

    def lower(stage):
        if stage == "model batch":
            memory.limit = 1

    monkeypatch.setattr(torch.Tensor, "new_empty", lambda *a, **kw: pytest.fail("encoder allocation preceded refusal"))
    with pytest.raises(CleanStop, match="encoder allocation"):
        _index.build_index(tmp_path, [("first", [65, 66])], model=model, tokenizer=TinyTokenizer(),
                           dictionary=dictionary, memory=memory, memory_guard=lower,
                           provenance={"partition": "fixture"}, context_length=2)
    assert not (tmp_path / "index" / "index.json").exists()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
