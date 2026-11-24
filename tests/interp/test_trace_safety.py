"""Small CPU checks for finite trace inputs and request recovery."""
import pytest
import torch

from caliper.interp import Tracer, _frozen, _publish, _trace
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


def resources():
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2)
    return model, TinyTokenizer(), exact_dictionary(model)


@pytest.mark.parametrize("name", ["W_enc", "b_enc", "W_dec", "b_dec"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_dictionary_refuses_before_forward_and_cache(monkeypatch, name, bad):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    monkeypatch.setattr(_publish, "_TENSORS", {})
    tracer.trace("A", max_targets=0)
    published = dict(_publish._TENSORS)
    # An alias write leaves the version counter unchanged.
    value = getattr(dictionary.layers[1], name)
    version = value._version
    value.data.view(-1)[-1] = bad
    assert value._version == version
    monkeypatch.setattr(model, "forward", lambda *a, **k: pytest.fail("nonfinite dictionary reached the model"))
    message = f"dictionary layer 1 {name} must be finite"
    with pytest.raises(ValueError, match=message):
        tracer.trace("A", max_targets=0)
    assert not tracer._cache and tracer._forward_cache is None
    with pytest.raises(ValueError, match=message):
        Tracer(model, tokenizer, dictionary)
    with pytest.raises(ValueError, match=message):
        _frozen.FrozenPath(model, torch.tensor([[1, 2]]), dictionary)
    assert all(_publish._TENSORS[key] is value for key, value in published.items())
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_finite_scan_bounds_strided_views_without_copies(monkeypatch):
    value = torch.arange(144, dtype=torch.float64).reshape(12, 12).T
    finite = torch.isfinite
    sizes = []
    def bounded(block):
        sizes.append(block.numel())
        assert block.numel() <= 16
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        return finite(block)
    monkeypatch.setattr(_frozen, "_FINITE_BLOCK_ELEMENTS", 16)
    monkeypatch.setattr(torch, "isfinite", bounded)
    _frozen._require_finite(value, "fixture")
    assert sum(sizes) == value.numel()
    value[-1, -1] = float("nan")
    with pytest.raises(ValueError, match="fixture must be finite"):
        _frozen._require_finite(value, "fixture")


@pytest.mark.parametrize("method", ["run", "replacement"])
@pytest.mark.parametrize("location", ["later_read", "logits"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_forward_residual_never_reaches_edges_or_cache(monkeypatch, method, location, bad):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    original = getattr(_frozen.FrozenPath, method)
    def corrupt(path, *args, **kwargs):
        reads, final = original(path, *args, **kwargs)
        assert torch.isfinite(reads[0]).all()
        if location == "later_read":
            reads[1] = torch.full_like(reads[1], bad)
        else:
            final = torch.full_like(final, bad)
        return reads, final
    monkeypatch.setattr(_frozen.FrozenPath, method, corrupt)
    monkeypatch.setattr(_trace, "_DirectEffects", lambda *a: pytest.fail("nonfinite forward reached edges"))
    with pytest.raises(ValueError, match="forward residual must be finite"):
        tracer.trace("A", max_targets=0)
    assert not tracer._cache and tracer._forward_cache is None
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_reconstruction_residual_refuses_before_edges(monkeypatch, bad):
    model, tokenizer, dictionary = resources()
    original = _frozen._trace_reconstruction
    def corrupt(layer, features):
        value = original(layer, features)
        return torch.full_like(value, bad) if layer is dictionary.layers[1] else value
    monkeypatch.setattr(_frozen, "_trace_reconstruction", corrupt)
    with pytest.raises(ValueError, match="layer 1 reconstruction residual must be finite"):
        Tracer(model, tokenizer, dictionary).trace("A", max_targets=0)


@pytest.mark.parametrize("field", ["absolute_residual", "relative_residual"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_target_residual_refuses_before_influence_and_cache(monkeypatch, field, bad):
    model, tokenizer, dictionary = resources()
    tracer = Tracer(model, tokenizer, dictionary)
    original = _trace._DirectEffects.backward
    def corrupt(effects, targets):
        incoming, rows = original(effects, targets)
        assert len(rows) > 1 and rows[0][field] < 1e-9
        rows[1][field] = bad
        return incoming, rows
    monkeypatch.setattr(_trace._DirectEffects, "backward", corrupt)
    monkeypatch.setattr(_trace, "influence", lambda *a: pytest.fail("nonfinite residual reached influence"))
    with pytest.raises(ValueError, match=f"direct-effect {field} must be finite"):
        tracer.trace("A", max_targets=0)
    assert not tracer._cache and tracer._forward_cache is None
