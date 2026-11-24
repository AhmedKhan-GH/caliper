"""Small CPU cases for boundary completion and fatal hook errors."""
import pytest
import torch

from caliper.interp._memory import CleanStop
from tests.interp.test_intervene import controlled


@pytest.mark.parametrize("kind", ["clamp", "scale"])
@pytest.mark.parametrize("fatal", [False, True])
def test_encoder_stop_cleans_hook_before_forward_continues(controlled, kind, fatal):
    model, tokenizer, ctl, service = controlled
    dictionary = service.dictionary
    dictionary.layers[0].float()
    error = CleanStop("encoder resource stop") if fatal else RuntimeError("encoder hook error")

    class Guard:
        def check(self, stage):
            if stage == "dictionary encoder block":
                raise error

    service.configure(dictionary=dictionary, tokenizer=tokenizer,
                      dictionary_id="fixture", resource_guard=Guard())
    value = {"value": 0} if kind == "clamp" else {"multiplier": 0}
    service.set({"kind": kind, "point": "layers.0.mlp_out", "feature": 0, **value})
    ctl.step()
    record = ctl._ivs[("layers.0.mlp_out", kind)]
    hook = record.hook
    downstream = []
    observer = model.layers[1].register_forward_pre_hook(lambda *args: downstream.append(True))
    try:
        if fatal:
            with pytest.raises(CleanStop) as caught:
                model(**tokenizer("abc"))
            assert caught.value is error
            assert downstream == []
        else:
            output = model(**tokenizer("abc")).logits
            assert downstream == [True]
            with service.suspend():
                expected = model(**tokenizer("abc")).logits
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert hook.removed and hook.errors == 1
        assert record.state == "ended" and record.hook is None
        assert record.error_code == "intervene.hook_failed"
        assert service.specifications() == []
    finally:
        observer.remove()
