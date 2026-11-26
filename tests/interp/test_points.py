"""Probe identities, actual architecture paths, and capture lifecycle checks."""
import gc
import weakref
from contextlib import ExitStack

import pytest
import torch

from caliper import interp
from caliper._adapter import TORCH
from .fixture import TinyLM


def hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def semantic(model):
    return [p for p in interp.points(model) if "probe" in p]


def test_fixture_residual_identity_and_pattern():
    model = TinyLM()
    requested = semantic(model) + [{"point": f"layers.{i}.self_attn"} for i in range(4)]
    requested += [{"point": f"layers.{i}"} for i in range(4)]
    with interp.capture(model, requested) as values:
        model(torch.tensor([[1, 2, 3, 4]]))
    for i in range(4):
        base = f"layers.{i}"
        actual = values[f"{base}.resid_pre"] + values[f"{base}.self_attn"] + values[f"{base}.mlp_out"]
        torch.testing.assert_close(actual, values[base], atol=1e-6, rtol=0)
        if i < 3:
            torch.testing.assert_close(actual, values[f"layers.{i + 1}.resid_pre"], atol=1e-6, rtol=0)
        torch.testing.assert_close(values[f"{base}.attn_pattern"], model.layers[i].self_attn.last_pattern)
        torch.testing.assert_close(model.layers[i].mlp(values[f"{base}.mlp_in"]), values[f"{base}.mlp_out"])
    assert hooks(model) == 0


def test_capture_shares_storage_and_never_copies_to_cpu(monkeypatch):
    model = TinyLM(layers=1)
    x = torch.randn(1, 3, 64, dtype=torch.float64, requires_grad=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("capture must not copy tensors to CPU")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        with interp.capture(model, ["layers.0.resid_pre", "layers.0.mlp_out"]) as values:
            model(inputs_embeds=x)
    assert values["layers.0.resid_pre"].data_ptr() == x.data_ptr()
    assert values["layers.0.resid_pre"].device == x.device
    assert not any(v.requires_grad for v in values.values())
    values.close(clear=True)
    assert not values


def test_capture_uses_device_storage_without_device_transfer(monkeypatch):
    # Meta tensors expose device placement without a GPU or a data transfer.
    model = TinyLM(layers=1).to("meta")
    original = torch.Tensor.to
    def checked(self, *args, **kwargs):
        assert not args or str(args[0]) not in ("cpu", "cuda", "mps")
        assert "device" not in kwargs
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", checked)
    with interp.capture(model, semantic(model)) as values:
        model(torch.empty(1, 3, dtype=torch.long, device="meta"))
    assert all(t.device.type == "meta" for t in values.values())


def test_modules_heads_and_semantic_ablation():
    model = TinyLM(layers=1)
    names = {p["point"] for p in TORCH.points(model)}
    assert {"layers.0.mlp", "layers.0.self_attn#1", "layers.0.mlp_in"} <= names
    with interp.capture(model, ["layers.0.self_attn#1", "layers.0.mlp_out"]) as values:
        off = TORCH.ablate(model, "layers.0.mlp_out")
        try:
            model(torch.tensor([[2, 3]]))
            # The capture hook predates ablation and thus observes the original output.
            assert values["layers.0.self_attn#1"].shape == (1, 2, 16)
        finally:
            off()
    for point in ("layers.0.mlp_in", "layers.0.resid_pre", "layers.0.mlp_out"):
        off = TORCH.ablate(model, point)
        with interp.capture(model, [point]) as values:
            model(torch.tensor([[2, 3]]))
        off()
        assert values[point].count_nonzero() == 0
    with pytest.raises(ValueError, match="read-only"):
        TORCH.ablate(model, "layers.0.attn_pattern")


def test_capture_cleanup_after_error_and_partial_install(monkeypatch):
    model = TinyLM(layers=1)
    original = model.layers[0].mlp.register_forward_hook
    before = hooks(model)
    with pytest.raises(RuntimeError, match="failed"):
        with interp.capture(model, semantic(model)) as values:
            model(torch.tensor([[1, 2]]))
            raise RuntimeError("failed")
    assert not values and hooks(model) == before
    def failed(*args, **kwargs):
        raise RuntimeError("failed install")
    monkeypatch.setattr(model.layers[0].mlp, "register_forward_hook", failed)
    with pytest.raises(RuntimeError, match="failed install"):
        with interp.capture(model, ["layers.0.resid_pre", "layers.0.mlp_out"]):
            pass
    assert hooks(model) == before
    monkeypatch.setattr(model.layers[0].mlp, "register_forward_hook", original)
    with pytest.raises(IndexError):
        with interp.capture(model, semantic(model)) as failed_values:
            model(torch.tensor([[9999]]))
    assert not failed_values and hooks(model) == before


def test_capture_replaces_prior_forward_and_releases_results():
    model = TinyLM(layers=1)
    with interp.capture(model, ["layers.0.mlp_out"]) as values:
        model(torch.tensor([[1, 2]]))
        old = weakref.ref(values["layers.0.mlp_out"])
        model(torch.tensor([[3]]))
        assert old() is None
        assert values["layers.0.mlp_out"].shape[1] == 1
    current = weakref.ref(values["layers.0.mlp_out"])
    values.close(clear=True)
    gc.collect()
    assert current() is None
    with pytest.raises(RuntimeError, match="once"):
        values.__enter__()


def test_generic_decoder_refusal_preserves_existing_module_points():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert TORCH.points(model)[0]["point"] == "0"
    with pytest.raises(NotImplementedError, match="do not support decoder"):
        interp.points(model)
    with pytest.raises(NotImplementedError, match="do not support decoder"):
        interp.capture(model, ["0"])
    with pytest.raises(KeyError):
        interp.capture(TinyLM(), ["layers.0.missing"])


def test_overlapping_capture_cleanup_preserves_existing_hooks():
    from caliper.interp._points import clear_captures
    from .test_gemma_family import gemma_model
    model = gemma_model(4)
    implementation = model.config._attn_implementation
    existing = model.model.layers[0].register_forward_hook(lambda *args: None)
    one = interp.capture(model, ["model.layers.3.attn_pattern"])
    two = interp.capture(model, ["model.layers.3.attn_pattern"])
    try:
        one.__enter__()
        two.__enter__()
        model(torch.tensor([[1, 2]]), use_cache=False)
        one.close()
        assert model.config._attn_implementation == "eager"
        clear_captures(model)
        assert not two
        assert model.config._attn_implementation == implementation
        assert hooks(model) == 1
    finally:
        one.close(clear=True)
        two.close(clear=True)
        existing.remove()


def test_semantic_points_round_trip_through_real_controls_codec():
    from caliper import _wire
    from caliper._controls import Controls
    from caliper._watch import watch
    model = TinyLM(layers=1)
    controls = Controls()
    handle = watch(model, activations=False, groups=False, scene=False)
    controls.interventions_on(model, handle)
    try:
        model(torch.tensor([[1, 2]]))
        controls.checkpoint()
        message = _wire.Controls.from_frame(_wire.Frame(_wire.MsgType.CONTROLS, controls.controls_meta()))
        decoded = _wire.decode(_wire.encode(message))
        rows = {p["point"]: p for p in decoded.to_meta()["interventions"]["points"]}
        assert rows["layers.0.mlp_out"]["kind"] == "module"
        assert rows["layers.0.mlp_out"]["probe"] == "mlp_out"
        assert rows["layers.0.attn_pattern"]["pattern"] == "softmax"
        assert rows["layers.0.self_attn#0"]["kind"] == "head"
    finally:
        handle.remove()
    assert hooks(model) == 0
