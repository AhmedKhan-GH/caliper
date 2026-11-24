"""Source refusals preserve intervention transactions and managed fork boundaries."""
import builtins
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from caliper._controls import Controls
from caliper._managed import ManagedRun
from caliper.interp import _intervene
from caliper.interp._intervene import InterventionError, InterventionService
from caliper.interp._memory import CleanStop
from .test_generic_patch_sources import SPEC, generic_model, tokenizer
from .test_intervention_replay import snapshot


class UnknownActivation:
    def __call__(self, value):
        pytest.fail("unknown activation ran")

    def __repr__(self):
        pytest.fail("unknown activation was serialized")


class CustomGELU(nn.GELU):
    pass


def refuse_activation(model, change):
    if change == "object":
        model.layer.activation = UnknownActivation()
    elif change == "closure":
        state = object()
        model.layer.activation = lambda value: state
    elif change == "partial_args":
        model.layer.activation = partial(F.gelu, torch.ones(1))
    elif change == "partial_option":
        model.layer.activation = partial(F.gelu, approximate=UnknownActivation())
    elif change == "module":
        model.layer.activation = CustomGELU()
    elif change == "forward":
        activation = nn.GELU()
        activation.forward = UnknownActivation()
        model.layer.activation = activation
    else:
        model.layer.activation = object()


@pytest.mark.parametrize("change", ["object", "closure", "partial_args", "partial_option", "module", "forward", "value"])
def test_unknown_source_state_refuses_before_weight_hash_or_forward(monkeypatch, change):
    model = generic_model("encoder", F.relu)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    try:
        original = service._source(SPEC)
        identity = service._cache_identity
        refuse_activation(model, change)
        monkeypatch.setattr(_intervene, "_content_identity",
                            lambda *a: pytest.fail("unsupported state reached the weight hash"))
        with pytest.raises(InterventionError, match="patch source refused") as refused:
            service._source(SPEC)
        assert refused.value.code == "intervene.not_supported"
        assert isinstance(refused.value.__cause__, (TypeError, ValueError))
        assert len(str(refused.value)) < 300
        assert service._cache["source", "head"] is original and service._cache_identity is identity
        assert not model.head._forward_hooks and not service.preparing_source
    finally:
        service.close()


@pytest.mark.parametrize("activation", [F.relu, nn.GELU()])
def test_native_source_does_not_require_transformers(monkeypatch, activation):
    original_import = builtins.__import__

    def without_transformers(name, *args, **kwargs):
        assert name != "transformers" and not name.startswith("transformers.")
        return original_import(name, *args, **kwargs)

    model = generic_model("encoder", activation)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    monkeypatch.setattr(builtins, "__import__", without_transformers)
    try:
        hook = service.install(SPEC)
        with torch.no_grad():
            actual = model(**tokenizer("target"))
        assert hook.error is None
        torch.testing.assert_close(actual, service._cache["source", "head"], rtol=0, atol=0)
    finally:
        service.close()


@pytest.mark.parametrize("mode", ["override", "restore"])
@pytest.mark.parametrize("change", ["object", "value"])
def test_managed_fork_source_refusal_blocks_forward_and_retains_state(tmp_path, capsys, mode, change):
    model = generic_model("encoder", F.relu)
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(tokenizer=tokenizer)
    run = ManagedRun("source-refusal", str(tmp_path), interventions=[SPEC] if mode == "override" else [])
    run._pkg = SimpleNamespace(_CTL=controls)
    if mode == "restore":
        run._restored_interventions = snapshot(SPEC)
    try:
        service.apply_overrides([SPEC])
        original = service.executor._cache["source", "head"]
        identity = service.executor._cache_identity
        state = service.state_dict()
        hook = controls._ivs["head", "patch"].hook
        refuse_activation(model, change)
        run._first = model.register_forward_pre_hook(lambda *args: run._intervene())
        first = run._first
        with pytest.raises(RuntimeError, match="fork's .*interventions cannot") as refused:
            model(**tokenizer("target"))
        cause = refused.value.__cause__
        assert isinstance(cause, InterventionError) and cause.code == "intervene.not_supported"
        assert service.state_dict() == state and controls._ivs["head", "patch"].hook is hook
        assert service.executor._cache["source", "head"] is original
        assert service.executor._cache_identity is identity and not hook.removed
        assert run._first is first and not run._intervening
        # A caught startup error must still block the next script forward.
        with pytest.raises(RuntimeError, match="fork's .*interventions cannot"):
            model(**tokenizer("target"))
        model.layer.activation = F.relu
        with torch.no_grad():
            actual = model(**tokenizer("target"))
        assert run._first is None
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        if mode == "override":
            assert "cannot apply its interventions" in capsys.readouterr().out
    finally:
        if run._first is not None:
            run._first.remove()
        service.remove()


@pytest.mark.parametrize("failure", ["forward", "missing", "output"])
def test_generic_source_failure_removes_temporary_capture_hooks(monkeypatch, failure):
    model = generic_model("encoder", F.relu)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    if failure == "forward":
        def broken(*args, **kwargs):
            raise ValueError("source forward refused")
        monkeypatch.setattr(model, "forward", broken)
    elif failure == "missing":
        monkeypatch.setattr(model, "forward", lambda input_ids, **kw: model.embed(input_ids))
    else:
        monkeypatch.setattr(model.head, "forward", lambda value: (value,))
    try:
        with pytest.raises(InterventionError, match="source"):
            service.install(SPEC)
        assert not service._cache and not service.preparing_source
        assert not model.head._forward_hooks
    finally:
        service.close()


def test_source_memory_stop_retains_its_type(monkeypatch):
    model = generic_model("encoder", F.relu)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    stop = CleanStop("source memory limit")

    def stopped(*args):
        raise stop

    monkeypatch.setattr(_intervene, "_content_identity", stopped)
    try:
        with pytest.raises(CleanStop) as refused:
            service.install(SPEC)
        assert refused.value is stop and not service._cache
    finally:
        service.close()


def test_gemma_fork_backend_refusal_retains_structured_cause(tmp_path):
    from .test_gemma_runtime import gemma_model

    model = gemma_model()
    model.config._attn_implementation = "sdpa"
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(tokenizer=tokenizer)
    spec = dict(SPEC, point="model.layers.0.mlp_out")
    run = ManagedRun("backend-refusal", str(tmp_path), interventions=[spec])
    run._pkg = SimpleNamespace(_CTL=controls)
    run._first = model.register_forward_pre_hook(lambda *args: run._intervene())
    try:
        with pytest.raises(RuntimeError, match="Gemma.*eager.*softcap") as refused:
            model(**tokenizer("target"))
        cause = refused.value.__cause__
        assert isinstance(cause, InterventionError) and cause.code == "intervene.not_supported"
        assert not service.specifications() and not service.executor._cache
        assert model.config._attn_implementation == "sdpa"
    finally:
        run._first.remove()
        service.remove()
