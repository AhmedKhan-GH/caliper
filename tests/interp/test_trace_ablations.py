"""Supported frozen ablations and refusal before model execution."""
from types import SimpleNamespace

import pytest
import torch

from caliper._ablate import ablate
from caliper.interp._frozen import FrozenPath, _validate_ablation_points
from caliper.interp._trace import Tracer, _DirectEffects
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


@pytest.mark.parametrize("suffix", ["mlp", "mlp_out", "resid_pre", "mlp_in",
                                   "self_attn", "self_attn.o_proj", "self_attn#0"])
def test_supported_frozen_ablation_preserves_completeness(suffix):
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2)
    dictionary = exact_dictionary(model)
    point = "layers.0." + suffix
    handle = ablate(model, point)
    try:
        path = FrozenPath(model, torch.tensor([[1, 2]]), dictionary,
                          interventions=[{"point": point, "op": "ablate"}])
        assert path.forward_residual < 1e-9
        assert not handle.error
        effects = _DirectEffects(path, dictionary, [1, 2], TinyTokenizer())
        _, diagnostics = effects.backward(effects.logit_nodes[:2] + effects.feature_nodes[:2])
        assert max(row["absolute_residual"] for row in diagnostics) < 1e-9
        if suffix in ("mlp", "mlp_out"):
            assert not path.features[0].count_nonzero()
            assert not path.errors[0].count_nonzero()
            assert not any(node["kind"] == "feature" and node["layer"] == 0 for node in effects.nodes)
    finally:
        handle()


@pytest.mark.parametrize("point", ["layers.0.mlp.features", "layers.0.self_attn.v_proj",
                                  "layers.0.input_layernorm", "layers.0.post_attention_layernorm",
                                  "layers.0", "norm", "lm_head"])
def test_valid_untraced_ablation_refuses_before_model_work(point, monkeypatch):
    from caliper.interp import _frozen, _trace
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    dictionary = exact_dictionary(model)
    tracer = Tracer(model, TinyTokenizer(), dictionary)
    spec = {"point": point, "op": "ablate", "state": "active"}
    monkeypatch.setattr(tracer, "_intervention_state", lambda *_: [spec])

    def refuse(*args, **kwargs):
        pytest.fail("unsupported ablations must fail before identity checks or model work")

    monkeypatch.setattr(_frozen, "_validate_dictionary", refuse)
    monkeypatch.setattr(_trace, "_validate_dictionary", refuse)
    monkeypatch.setattr(model, "forward", refuse)
    handle = ablate(model, point)
    try:
        with pytest.raises(ValueError, match="trace does not support ablation point"):
            tracer.trace("A", 0)
        with pytest.raises(ValueError, match="trace does not support ablation point"):
            FrozenPath(model, torch.tensor([[1]]), dictionary, interventions=[spec])
        assert not tracer._cache and tracer._forward_cache is None
        assert handle.calls == 0
    finally:
        handle()
