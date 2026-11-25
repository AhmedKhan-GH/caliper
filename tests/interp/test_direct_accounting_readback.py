"""CPU regressions for bounded direct-target readback and refusal paths."""
import pytest
import torch

from caliper.interp._trace import TraceTimeout
from .test_direct_affine_accounting import BiasPath, effects_for


@pytest.fixture
def readbacks(monkeypatch):
    events = []
    item, tolist = torch.Tensor.item, torch.Tensor.tolist

    def scalar(value, *args, **kwargs):
        events.append(("item", tuple(value.shape), value.dtype))
        return item(value, *args, **kwargs)

    def matrix(value, *args, **kwargs):
        events.append(("tolist", tuple(value.shape), value.dtype))
        return tolist(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", scalar)
    monkeypatch.setattr(torch.Tensor, "tolist", matrix)
    return events


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("reference", [False, True])
def test_direct_accounting_uses_one_readback(dtype, reference, readbacks):
    path = BiasPath(dtype)
    effects = effects_for(path)
    effects.reference = reference
    effects.prepare_decoders()
    stages = []
    effects.check = stages.append
    readbacks.clear()

    weights, rows = effects.backward(effects.feature_nodes + effects.logit_nodes)

    account_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert readbacks == [("tolist", (3, 9), account_dtype)]
    assert [stage for stage in stages if stage.startswith("finite block")] == [
        "finite block", "finite block complete", "finite block readback"]
    assert stages[-1] == "direct edge products"
    assert weights.dtype == dtype
    assert not torch.count_nonzero(weights)
    assert [row["biases"]["encoder"] for row in rows] == [.5, .5, 0.]
    if reference:
        assert [row["biases"]["decoder"] for row in rows] == [0., 32., 32.125]
        assert [row["expected"] for row in rows] == [
            .5, 32.5, 32. if dtype == torch.bfloat16 else 32.125]
        assert rows[-1]["absolute_residual"] == (.125 if dtype == torch.bfloat16 else 0.)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_last_target_refuses_the_entire_batch(bad, readbacks):
    path = BiasPath(torch.float32)
    effects = effects_for(path)
    effects.prepare_decoders()
    # The earlier feature targets stay finite. A later NaN must not escape.
    path.reference_final.fill_(bad)
    stages = []
    effects.check = stages.append
    readbacks.clear()

    with pytest.raises(ValueError, match="^direct-effect completeness residual must be finite$"):
        effects.backward(effects.feature_nodes + effects.logit_nodes)

    assert readbacks == [("tolist", (3, 9), torch.float32)]
    assert stages[-3:] == ["finite block", "finite block complete", "finite block readback"]
    assert "direct edge products" not in stages


@pytest.mark.parametrize("stop", ["finite block", "finite block complete", "finite block readback"])
def test_accounting_deadline_checks_surround_readback(stop, readbacks):
    path = BiasPath(torch.float64)
    effects = effects_for(path)
    effects.prepare_decoders()
    stages = []

    def check(stage):
        stages.append(stage)
        if stage == stop:
            raise TraceTimeout(stage)

    effects.check = check
    readbacks.clear()
    with pytest.raises(TraceTimeout, match=stop):
        effects.backward(effects.logit_nodes)

    assert stages[-1] == stop
    assert "direct edge products" not in stages
    assert readbacks == ([("tolist", (1, 9), torch.float64)] if stop == "finite block readback" else [])
