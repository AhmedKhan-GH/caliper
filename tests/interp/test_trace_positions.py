"""Position refusals preserve live hooks, caches, and publication state."""
from contextlib import contextmanager

import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer
from caliper.interp._frozen import FrozenPath
from caliper.interp._graph import Graph
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


POSITION_FORMS = [(point, kind) for point in (
    "resid_pre", "mlp_in", "mlp", "mlp_out", "self_attn", "self_attn.o_proj"
) for kind in ("add", "patch")] + [("mlp_out", kind) for kind in ("scale", "clamp")]


@contextmanager
def controlled_position(point, kind):
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2).cpu()
    tokenizer = TinyTokenizer()
    dictionary = exact_dictionary(model)
    owner = Controls()
    service = owner.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    with torch.no_grad():
        model(tokenizer("ABCDEFGHI")["input_ids"])
    spec = {"kind": kind, "point": "layers.0." + point, "positions": [8]}
    if kind == "add":
        spec.update(vector=[.1] * 8, coefficient=.5)
    elif kind == "patch":
        spec.update(source_prompt="ABCDEFGHI", source_position=1)
    elif kind == "scale":
        spec.update(feature=0, multiplier=.5)
    else:
        spec.update(feature=0, value=.7)
    try:
        yield model, tokenizer, dictionary, owner, service, spec
    finally:
        service.remove()


@pytest.mark.parametrize("point,kind", POSITION_FORMS)
@pytest.mark.parametrize("entry", ["trace", "direct"])
def test_short_prompt_refuses_before_forward_and_preserves_live_state(point, kind, entry, monkeypatch):
    with controlled_position(point, kind) as (model, tokenizer, dictionary, owner, service, spec):
        published = []
        monkeypatch.setattr(Graph, "publish", lambda graph: published.append(graph) or graph)
        tracer = Tracer(model, tokenizer, dictionary, controls=owner)
        service.set(spec)
        owner.step()
        # The same position must remain valid for the original prompt.
        graph = tracer.trace("ABCDEFGHI", max_targets=4)
        assert graph.completeness["max_absolute_residual"] < 1e-9
        state = service.state_dict()
        hooks = [iv.hook for iv in owner._ivs.values()]
        calls = [hook.calls for hook in hooks]
        cached = list(tracer._cache.items())
        forward_cache = tracer._forward_cache
        sources = dict(service.executor._cache)
        published.clear()

        def refuse_forward(*args, **kwargs):
            pytest.fail("position preflight must precede every model forward")

        monkeypatch.setattr(model, "forward", refuse_forward)
        with pytest.raises(ValueError, match="intervention position.*outside.*prompt"):
            if entry == "trace":
                tracer.trace("AB", max_targets=4)
            else:
                FrozenPath(model, tokenizer("AB")["input_ids"], dictionary,
                           interventions=service.specifications())
        assert service.state_dict() == state
        assert [hook.calls for hook in hooks] == calls
        assert all(not hook.removed and hook.error is None for hook in hooks)
        assert list(tracer._cache.items()) == cached
        assert tracer._forward_cache is forward_cache and tracer._prompt == "ABCDEFGHI"
        assert service.executor._cache == sources
        assert published == []


@pytest.mark.parametrize("position", [3, 8, -1])
def test_direct_add_refuses_invalid_position_without_a_live_hook(position, monkeypatch):
    with controlled_position("mlp_out", "add") as (model, tokenizer, dictionary, _, _, spec):
        spec["positions"] = [position]
        monkeypatch.setattr(model, "forward", lambda *a, **kw: pytest.fail("unexpected forward"))
        with pytest.raises(ValueError, match="intervention position.*outside.*prompt"):
            FrozenPath(model, tokenizer("AB")["input_ids"], dictionary, interventions=[spec])


@pytest.mark.parametrize("positions", [[], [0], [2], [0, 2]])
def test_direct_add_preserves_valid_positions(positions):
    with controlled_position("mlp_out", "add") as (model, tokenizer, dictionary, _, _, spec):
        spec["positions"] = positions
        path = FrozenPath(model, tokenizer("AB")["input_ids"], dictionary, interventions=[spec])
        assert path.forward_residual < 1e-9
        assert path.tokens == 3
