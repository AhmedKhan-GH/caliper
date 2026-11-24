"""Bounded CPU content checks for clean patch sources."""
import weakref

import pytest
import torch

from caliper.interp import _dictionary
from caliper.interp._intervene import InterventionService
from caliper.interp._memory import CleanStop
from .fixture import make_fixture
from .test_intervene import controlled
from .test_intervention_audit import SourceGuard, patch_spec


@pytest.mark.parametrize("write", ["normal", "data", "inference"])
def test_patch_refreshes_source_embedding_after_weight_write(write):
    with torch.inference_mode(write == "inference"):
        model, tokenizer = make_fixture()
        service = InterventionService(model)
        service.configure(tokenizer=tokenizer)
        spec = patch_spec()
        hook = service.install(spec)
        first = service._cache["AB", spec["point"]].clone()
        prior = weakref.ref(service._cache["AB", spec["point"]])
        parameter = model.embed_tokens.weight
        version = None if write == "inference" else parameter._version
        if write == "inference":
            with pytest.raises(RuntimeError, match="version counter"):
                _ = parameter._version
        with torch.no_grad():
            target = parameter.data if write == "data" else parameter
            target[tokenizer("AB")["input_ids"][0, 1]].add_(0.125)
        if write == "data":
            assert parameter._version == version
        elif write == "normal":
            assert parameter._version > version
        with service.suspend():
            expected = model(**tokenizer("AB")).logits
        try:
            torch.testing.assert_close(model(**tokenizer("XY")).logits, expected, rtol=0, atol=0)
            source = service._cache["AB", spec["point"]]
            torch.testing.assert_close(source, model.embed_tokens(tokenizer("AB")["input_ids"]), rtol=0, atol=0)
            assert not torch.equal(source, first) and prior() is None
            assert hook.errors == 0
        finally:
            service.close()
        assert hook.removed and not service._cache and service._cache_identity is None


@pytest.mark.parametrize("persistent", [False, True])
def test_patch_refreshes_source_after_buffer_alias_write(controlled, persistent):
    model, tokenizer, _, service = controlled
    service.configure(tokenizer=tokenizer)
    model.embed_tokens.register_buffer("offset", torch.zeros(64, dtype=torch.float64), persistent=persistent)
    handle = model.embed_tokens.register_forward_hook(lambda module, args, output: output + module.offset)
    spec = patch_spec()
    hook = service.executor.install(spec)
    first = service.executor._cache["AB", spec["point"]].clone()
    prior = weakref.ref(service.executor._cache["AB", spec["point"]])
    version = model.embed_tokens.offset._version
    model.embed_tokens.offset.data.add_(0.125)
    assert model.embed_tokens.offset._version == version
    try:
        with service.suspend():
            expected = model(**tokenizer("AB")).logits
        torch.testing.assert_close(model(**tokenizer("XY")).logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(service.executor._cache["AB", spec["point"]], first + 0.125, rtol=0, atol=0)
        assert prior() is None and hook.errors == 0
    finally:
        handle.remove()
        hook()
    assert not service.executor._cache


class IdentityGuard(SourceGuard):
    def __init__(self, limit=20_000_000, peak=0):
        super().__init__()
        self.device = torch.device("cpu")
        self.limit, self.peak = limit, peak


def test_content_readbacks_obey_chunk_bound_and_supplied_guard(controlled, monkeypatch):
    model, tokenizer, _, service = controlled
    guard = IdentityGuard()
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    monkeypatch.setattr(_dictionary, "_HASH_CHUNK_BYTES", 16_384)
    original = torch.Tensor.to
    copies = []
    checks = 0

    def to(tensor, *args, **kwargs):
        nonlocal checks
        if kwargs.get("device") == "cpu":
            assert len(guard.stages) > checks
            assert guard.stages[-1] == "model_identity"
            checks = len(guard.stages)
            copies.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    service.executor._identity()
    assert copies and max(copies) <= 16_384
    assert sum(copies) == sum(value.numel() * value.element_size()
                              for value in (*model.parameters(), *model.buffers()))


@pytest.mark.parametrize("lower", ["outer", "inner"])
def test_identity_respects_lower_guard_limit_before_readback(controlled, monkeypatch, lower):
    _, tokenizer, _, service = controlled
    guard = IdentityGuard()
    guard._guard = IdentityGuard()
    selected = guard if lower == "outer" else guard._guard
    selected.limit = 4 * _dictionary._HASH_CHUNK_BYTES - 1
    service.configure(tokenizer=tokenizer, resource_guard=guard)

    def forbidden(*args, **kwargs):
        pytest.fail("memory admission must precede a tensor readback")

    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        service.executor._identity()
    assert guard.stages == ["model_identity"]


@pytest.mark.parametrize("lower", ["primary", "extra"])
def test_hosted_guard_preserves_identity_memory_admission(controlled, monkeypatch, lower):
    from caliper.interp._hosted import _RunGuard
    _, tokenizer, _, service = controlled
    primary, extra = IdentityGuard(), IdentityGuard()
    guard = _RunGuard(primary, extra=extra)
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    selected = primary if lower == "primary" else extra
    selected.limit = 4 * _dictionary._HASH_CHUNK_BYTES + 100
    selected.peak = 101

    def forbidden(*args, **kwargs):
        pytest.fail("hosted memory admission must precede a tensor readback")

    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        service.executor._identity()
    assert primary.stages == extra.stages == ["model_identity"]


@pytest.mark.parametrize("display_only", [False, True])
def test_hosted_guard_exposes_device_budget_to_content_identity(monkeypatch, display_only):
    from caliper.interp._hosted import _RunGuard
    from caliper.interp._intervene import _content_identity
    guard = IdentityGuard()
    guard.device = torch.device("mps")
    hosted = _RunGuard(None, extra=guard) if display_only else _RunGuard(guard)

    class DeviceLabel(torch.Tensor):
        @property
        def device(self):
            return torch.device("mps")

    state = {"weight": torch.zeros(1).as_subclass(DeviceLabel)}

    def admitted(values, *, memory):
        assert values is state
        assert memory.device == guard.device
        assert memory.limit == guard.limit
        memory.check("model_identity")
        guard.peak = 10
        assert memory.peak == 10
        return "bounded identity"

    monkeypatch.setattr(_dictionary, "_state_hash", admitted)
    assert _content_identity(state, hosted) == "bounded identity"
    assert guard.stages == ["model_identity"]


def test_identity_failure_ends_only_patch_and_releases_cache(controlled):
    model, tokenizer, ctl, service = controlled
    guard = IdentityGuard()
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    service.set(patch_spec())
    service.set({"kind": "add", "point": "layers.2.mlp_out", "vector": [0.1] * 64, "coefficient": 1})
    ctl.step()
    patch = ctl._ivs["layers.0.resid_pre", "patch"].hook
    other = ctl._ivs["layers.2.mlp_out", "add"].hook
    prior = weakref.ref(service.executor._cache["AB", "layers.0.resid_pre"])
    guard.stop = "model_identity"
    model(**tokenizer("XY"))
    assert patch.removed and patch.errors == 1 and not other.removed and other.calls == 1
    assert prior() is None and not service.executor._cache
    assert ctl._ivs["layers.0.resid_pre", "patch"].state == "ended"
    model(**tokenizer("XY"))
    assert patch.calls == 1 and other.calls == 2


def test_noncontiguous_state_refuses_before_capture(controlled):
    model, tokenizer, _, service = controlled
    service.configure(tokenizer=tokenizer)
    model.register_buffer("strided_state", torch.zeros((2, 3)).t())
    with pytest.raises(ValueError, match="materialized contiguous dense"):
        service.executor.install(patch_spec())
    assert not service.executor._cache


def test_memory_admission_rechecks_before_each_readback(controlled, monkeypatch):
    _, tokenizer, _, service = controlled
    guard = IdentityGuard()
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    original = torch.Tensor.to
    copies = []

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            copies.append(tensor.numel())
            guard.limit = 1
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        service.executor._identity()
    assert len(copies) == 1 and len(guard.stages) == 3


def test_alias_write_between_patch_hooks_requires_new_source_identity(controlled):
    model, tokenizer, _, service = controlled
    service.configure(tokenizer=tokenizer)
    executor = service.executor
    first = executor.install(patch_spec())
    writes = []

    def update(module, args):
        if not executor.preparing_source and not writes:
            model.embed_tokens.weight.data.add_(0.125)
            writes.append(1)

    mutation = model.layers[1].register_forward_pre_hook(update)
    second = executor.install(patch_spec(point="layers.1.resid_pre"))
    old = weakref.ref(executor._cache["AB", "layers.1.resid_pre"])
    try:
        actual = model(**tokenizer("XY")).logits
        with service.suspend():
            expected = model(**tokenizer("AB")).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert writes == [1] and old() is None
        assert first.errors == second.errors == 0
    finally:
        mutation.remove()


@pytest.mark.parametrize("guard", [None, SourceGuard()])
def test_device_identity_requires_finite_guard_without_device_access(controlled, guard):
    model, tokenizer, _, service = controlled

    class DeviceLabel(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:0")

    model.register_buffer("device_label", torch.zeros(1).as_subclass(DeviceLabel))
    service.configure(tokenizer=tokenizer, resource_guard=guard)
    with pytest.raises(ValueError, match="bounded process memory guard"):
        service.executor._identity()
