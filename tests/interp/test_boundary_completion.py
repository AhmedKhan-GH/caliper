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


@pytest.mark.parametrize("started", [False, True])
def test_prospective_step_overflow_preserves_all_control_state(started):
    from caliper._controls import Controls
    from python.tests.test_controls_budget import control, encoded, fill_budget

    ctl = Controls()
    param = ctl.param("text", "original")
    calls = []
    action = ctl.action("queued", lambda: calls.append(True))
    if started:
        ctl.step()
        assert control(ctl, "set", name=param.name, value="pending")["status"] == "ok"
    assert control(ctl, "run", name=action.name)["status"] == "queued"
    fill_budget(ctl)
    before = ctl.controls_meta()
    pending = (ctl.n, ctl.busy, ctl._npending, param._version, list(ctl.queue))
    with pytest.raises(ValueError, match="wire limit"):
        ctl.step(10**1000)
    assert ctl.controls_meta() == before
    assert (ctl.n, ctl.busy, ctl._npending, param._version, list(ctl.queue)) == pending
    assert calls == [] and action.queued
    encoded(ctl)
    ctl.step()
    assert calls == [True]
    assert param.value == ("pending" if started else "original")
    encoded(ctl)


def test_large_step_and_implicit_increment_remain_supported():
    from caliper._controls import Controls
    from python.tests.test_controls_budget import encoded

    ctl = Controls()
    ctl.step(10**100)
    ctl.param("text", "ordinary value")
    ctl.step()
    assert ctl.n == 10**100 + 1
    encoded(ctl)


def test_implicit_step_growth_is_admitted_before_counter_change():
    from caliper._controls import Controls
    from python.tests.test_controls_budget import encoded, fill_budget

    ctl = Controls()
    ctl.step(10**100 - 1)
    pad = fill_budget(ctl)
    # Escaped padding leaves at most five bytes. Consume each remaining byte.
    for _ in range(6):
        try:
            pad.set(pad.value + "x")
        except ValueError:
            break
    else:
        pytest.fail("metadata did not reach its admission bound")
    before = ctl.controls_meta()
    with pytest.raises(ValueError, match="wire limit"):
        ctl.step()
    assert ctl.n == 10**100 - 1 and ctl.controls_meta() == before
    encoded(ctl)
