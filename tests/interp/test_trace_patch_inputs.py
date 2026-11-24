"""Both trace caches bind current clean-source tokens before reuse."""
import pytest
import torch

from caliper._controls import Controls
from caliper.interp import Tracer
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary


@pytest.mark.parametrize("cache", ["graph", "forward"])
def test_source_only_token_change_refreshes_trace_and_matches_fresh_trace(cache):
    controls = Controls()

    class SourceTokenizer(TinyTokenizer):
        changed = False
        check_lock = False

        def __call__(self, text, **kwargs):
            if self.check_lock:
                assert not controls.lock.locked()
            return super().__call__("EF" if text == "CD" and self.changed else text, **kwargs)

    tokenizer = SourceTokenizer()
    model = TinyLM(layers=2, width=8, heads=2, latents=4, k=1).eval()
    dictionary = exact_dictionary(model)
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    tracer = Tracer(model, tokenizer, dictionary, controls=controls)
    try:
        service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "CD"})
        controls.step()
        tokenizer.check_lock = True
        first = tracer.trace("AB", max_targets=8, max_nodes=64)
        path = tracer._forward_cache[1]
        logits = path.logits.clone()
        target_tokens = tokenizer("AB")["input_ids"].clone()
        assert tracer.trace("AB", max_targets=8, max_nodes=64) is first
        tokenizer.changed = True
        assert torch.equal(tokenizer("AB")["input_ids"], target_tokens)
        nodes = 63 if cache == "forward" else 64
        changed = tracer.trace("AB", max_targets=8, max_nodes=nodes)
        assert changed is not first and tracer._forward_cache[1] is not path
        assert not torch.equal(tracer._forward_cache[1].logits, logits)
        oracle = Tracer(model, tokenizer, dictionary, controls=controls)
        fresh = oracle.trace("AB", max_targets=8, max_nodes=nodes)
        torch.testing.assert_close(tracer._forward_cache[1].logits, oracle._forward_cache[1].logits, rtol=0, atol=0)
        for name in fresh.tensors:
            torch.testing.assert_close(changed.tensors[name], fresh.tensors[name], rtol=0, atol=0)
        assert tracer.trace("AB", max_targets=8, max_nodes=nodes) is changed
    finally:
        service.remove()
