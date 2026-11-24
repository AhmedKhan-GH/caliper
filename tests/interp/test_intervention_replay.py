"""CPU checks for snapshot replay without live lifecycle changes."""
from contextvars import Context

import pytest
import torch

from caliper.interp._intervene import InterventionService, _mute_hooks
from tests.interp.test_intervene import add_spec, controlled


@pytest.mark.parametrize("spec", [
    add_spec(point="layers.0.resid_pre"),
    add_spec(),
    {"kind": "clamp", "point": "layers.1.mlp_out", "feature": 0, "value": 10},
    {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "xyz"},
    {"kind": "ablate", "point": "layers.0.mlp"},
    {"kind": "ablate", "point": "layers.0.self_attn#0"},
    {"kind": "ablate", "point": "layers.0.mlp_in"},
])
def test_mute_scope_selects_objects_and_preserves_other_contexts(controlled, spec):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    baseline = model(**inputs).logits
    service.set(spec)
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook
    active = model(**inputs).logits
    assert not torch.equal(active, baseline)
    calls = hook.calls
    cache = dict(service.executor._cache)
    with _mute_hooks([hook]):
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
        assert hook.calls == calls
        torch.testing.assert_close(Context().run(model, **inputs).logits, active, rtol=0, atol=0)
        private = InterventionService(model)
        private.configure(dictionary=service.dictionary, tokenizer=tokenizer, dictionary_id="fixture")
        try:
            duplicate = private.install(spec)
            torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)
            with service.suspend():
                torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
            assert duplicate.calls == 1
        finally:
            private.close()
    assert not hook.removed and hook.calls == calls + 1
    assert all(service.executor._cache[key] is value for key, value in cache.items())
    torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)


def test_nested_mute_scope_resets_after_exception(controlled):
    model, tokenizer, ctl, service = controlled
    inputs = tokenizer("abc")
    baseline = model(**inputs).logits
    service.set(add_spec())
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook
    active = model(**inputs).logits
    with _mute_hooks([hook]):
        with pytest.raises(RuntimeError, match="scope exit"):
            with _mute_hooks([]):
                raise RuntimeError("scope exit")
        torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)
    torch.testing.assert_close(model(**inputs).logits, active, rtol=0, atol=0)
