"""Decoder source copies obey process limits before cache publication."""
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from caliper.interp import _intervene, _points
from caliper.interp._intervene import InterventionService
from caliper.interp._memory import CleanStop
from .fixture import make_fixture
from .test_intervention_replay import hook_objects


POINTS = ("layers.0.resid_pre", "layers.0.mlp_in")
SOURCE_BYTES = 1 * 3 * 4 * 4


def specification(point, prompt="AB"):
    return {"kind": "patch", "point": point, "source_prompt": prompt}


@pytest.fixture
def decoder_source():
    model, tokenizer = make_fixture(layers=1, width=4, heads=1, latents=8, k=2, dtype=torch.float32)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    hooks = hook_objects(model)
    try:
        yield model, tokenizer, service
    finally:
        service.close()
    assert hook_objects(model) == hooks and model not in _points._ACTIVE
    assert not service._cache and service._cache_identity is None
    assert not service.preparing_source


class DecoderBudget:
    """Report logical bytes only after source identity and the forward pass."""

    def __init__(self, lower, capacity, *, live=False, floor_only=False):
        high = 8 * 1024 * 1024
        self.limit = self._limit = high
        self._guard = SimpleNamespace(limit=high, peak=0, current=0)
        self.selected = self if lower == "outer" else self._guard
        self.selected.limit = 2 * high
        self.target = capacity if floor_only else high
        self.baseline = 0 if floor_only else high - capacity
        self.live = live
        self.device = torch.device("cpu")
        self.peak = self.current = self.copied = self.growth = 0
        self.complete = False
        self.stages = []

    def check(self, stage):
        self.stages.append(stage)
        if stage == "patch source forward complete":
            self.complete = True
            self.selected.limit = self.target
            other = self._guard if self.selected is self else self
            other.limit = 2 * self._limit
        if self.complete:
            self.current = self.baseline + (self.copied if self.live else 0) + self.growth
            self.peak = max(self.peak, self.current)
        return self.current


def prepare_sources(service, monkeypatch, *, eviction, points=POINTS):
    for point in points:
        # Capture the shape for ordinary module points before hook validation.
        service._source(specification(point))
        service.install(specification(point))
    service.invalidate_cache()
    if eviction:
        for point in points:
            service._source(specification(point, "CD"))
    monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", len(points))


def assert_retained(service, cache, identity):
    assert list(service._cache) == [key for key, _ in cache]
    assert all(service._cache[key] is tensor for key, tensor in cache)
    if identity is not None:
        assert service._cache_identity is identity
    else:
        assert not service._cache_identity[1]


def use_capture_views(model, monkeypatch):
    for module in (model.embed_tokens, model.layers[0].post_attention_layernorm):
        def forward(*args, original=module.forward, **kwargs):
            output = original(*args, **kwargs)
            backing = output.new_empty((*output.shape[:-1], 4 * output.shape[-1]))
            view = backing[..., 1::4]
            view.copy_(output)
            return view

        monkeypatch.setattr(module, "forward", forward)


@pytest.mark.parametrize("views", [False, True])
@pytest.mark.parametrize("failure", ["admission", "guard"])
@pytest.mark.parametrize("completed", [0, 1])
def test_retained_refusal_releases_capture_tensors_and_storage(
        decoder_source, monkeypatch, views, failure, completed):
    model, _, service = decoder_source
    initial_hooks = hook_objects(model)
    if views:
        use_capture_views(model, monkeypatch)
    prepare_sources(service, monkeypatch, eviction=True)
    cache, identity, hooks = list(service._cache.items()), service._cache_identity, hook_objects(model)
    capacity = 2 * SOURCE_BYTES - (failure == "admission" and completed == 0)
    guard = DecoderBudget("outer", capacity, live=True)
    service.resource_guard = guard
    save, clone, check = _points.Capture._save, torch.Tensor.clone, guard.check
    contexts, captures, copies = {}, [], []

    def record_capture(values, name, value):
        save(values, name, value)
        contexts[id(values)] = values
        tensor = values[name]
        storage = tensor.untyped_storage()
        captures.append((weakref.ref(tensor), StorageWeakRef(storage), storage.nbytes()))

    def record_copy(tensor, *args, **kwargs):
        result = clone(tensor, *args, **kwargs)
        if tensor.ndim == 3:
            copies.append((weakref.ref(result), StorageWeakRef(result.untyped_storage())))
            guard.copied += tensor.numel() * tensor.element_size()
            if failure == "admission":
                guard.growth = 1
        return result

    def stop(stage):
        current = check(stage)
        if failure == "guard" and stage == "patch source copy" and len(copies) == completed:
            raise CleanStop("decoder guard stop")
        return current

    monkeypatch.setattr(_points.Capture, "_save", record_capture)
    monkeypatch.setattr(torch.Tensor, "clone", record_copy)
    monkeypatch.setattr(guard, "check", stop)
    match = "patch source allocation exceeds" if failure == "admission" else "decoder guard stop"
    with pytest.raises(CleanStop, match=match) as refused:
        service.install(specification(POINTS[0]))
    error, traceback = refused.value, refused.value.__traceback__
    assert traceback is not None
    assert len(captures) == 2 and len(copies) == completed
    assert [size for _, _, size in captures] == [SOURCE_BYTES * (4 if views else 1)] * 2
    assert len(contexts) == 1
    assert all(values._closed and not values._handles and not values for values in contexts.values())
    assert_retained(service, cache, identity)
    assert hook_objects(model) == hooks and model not in _points._ACTIVE
    assert not service.preparing_source

    def assert_released():
        assert refused.value is error and error.__traceback__ is traceback
        assert all(reference() is None for reference, _, _ in captures)
        # Python storage wrappers can die before their native allocation.
        assert all(storage.expired() for _, storage, _ in captures)
        assert all(reference() is None and storage.expired() for reference, storage in copies)

    assert_released()
    service.close()
    assert_released()
    assert not service._cache and service._cache_identity is None
    assert hook_objects(model) == initial_hooks and model not in _points._ACTIVE


@pytest.mark.parametrize("eviction", [False, True])
@pytest.mark.parametrize("lower", ["outer", "inner"])
@pytest.mark.parametrize("live", [False, True])
def test_pending_copies_refuse_one_byte_below_and_accept_exact_capacity(
        decoder_source, monkeypatch, eviction, lower, live):
    model, tokenizer, service = decoder_source
    with torch.no_grad():
        embedded = model.embed_tokens(tokenizer("AB")["input_ids"])
        block = model.layers[0]
        mlp_input = block.post_attention_layernorm(embedded + block.self_attn(block.input_layernorm(embedded)))
    expected = dict(zip(POINTS, (embedded, mlp_input)))
    prepare_sources(service, monkeypatch, eviction=eviction)
    cache, identity, hooks = list(service._cache.items()), service._cache_identity, hook_objects(model)
    guard = DecoderBudget(lower, 2 * SOURCE_BYTES - 1, live=live)
    service.resource_guard = guard
    clone = torch.Tensor.clone

    def forbid_copy(tensor, *args, **kwargs):
        assert tensor.ndim != 3, "decoder source allocation preceded admission"
        return clone(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", forbid_copy)
        with pytest.raises(CleanStop, match="patch source allocation exceeds"):
            service._source(specification(POINTS[0]))
    assert "patch source forward complete" in guard.stages
    assert_retained(service, cache, identity)
    assert hook_objects(model) == hooks and model not in _points._ACTIVE
    assert not service.preparing_source

    guard = DecoderBudget(lower, 2 * SOURCE_BYTES, live=live)
    service.resource_guard = guard
    copies = []

    def record_copy(tensor, *args, **kwargs):
        if tensor.ndim == 3:
            assert_retained(service, cache, identity)
            assert guard.stages[-1] == "patch source copy"
        result = clone(tensor, *args, **kwargs)
        if tensor.ndim == 3:
            guard.copied += tensor.numel() * tensor.element_size()
            copies.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", record_copy)
    source = service._source(specification(POINTS[0]))
    assert source is service._cache["AB", POINTS[0]]
    assert len(copies) == 2 and guard.stages.count("patch source copy") == 2
    assert set(service._cache) == {("AB", point) for point in POINTS}
    assert {prompt for prompt, _ in service._cache_identity[1]} == {"AB"}
    for point, value in expected.items():
        cached = service._cache["AB", point]
        torch.testing.assert_close(cached, value, rtol=0, atol=0)
        assert cached.data_ptr() != value.data_ptr()
        assert any(reference() is cached for reference in copies)
    assert hook_objects(model) == hooks and model not in _points._ACTIVE
    assert not service.preparing_source


@pytest.mark.parametrize("eviction", [False, True])
@pytest.mark.parametrize("aliases", [False, True])
@pytest.mark.parametrize("views", [False, True])
def test_retained_storage_floor_counts_aliases_once(decoder_source, monkeypatch, eviction, aliases, views):
    model, _, service = decoder_source
    if views:
        use_capture_views(model, monkeypatch)
    points = ("layers.0.resid_pre", "embed_tokens") if aliases else POINTS
    prepare_sources(service, monkeypatch, eviction=eviction, points=points)
    cache, identity, hooks = list(service._cache.items()), service._cache_identity, hook_objects(model)
    # Captures can share storage. Each cache destination owns a separate copy.
    retained = (1 if aliases else 2) * SOURCE_BYTES * (4 if views else 1)
    retained += 2 * SOURCE_BYTES if eviction else 0
    capacity = retained + 2 * SOURCE_BYTES
    clone = torch.Tensor.clone

    def forbid_copy(tensor, *args, **kwargs):
        assert tensor.ndim != 3, "decoder source allocation preceded the retained storage floor"
        return clone(tensor, *args, **kwargs)

    service.resource_guard = DecoderBudget("inner", capacity - 1, floor_only=True)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", forbid_copy)
        with pytest.raises(CleanStop, match="patch source allocation exceeds"):
            service._source(specification(points[0]))
    assert_retained(service, cache, identity)
    assert hook_objects(model) == hooks and model not in _points._ACTIVE
    service.resource_guard = DecoderBudget("inner", capacity, floor_only=True)
    service._source(specification(points[0]))
    assert set(service._cache) == {("AB", point) for point in points}
    left, right = (service._cache["AB", point] for point in points)
    assert left.data_ptr() != right.data_ptr()
    if aliases:
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert hook_objects(model) == hooks and model not in _points._ACTIVE


@pytest.mark.parametrize("failure", ["forward", "copy", "growth", "guard", "nonfinite"])
def test_failure_preserves_cache_and_removes_only_temporary_hooks(decoder_source, monkeypatch, failure):
    model, tokenizer, service = decoder_source
    prepare_sources(service, monkeypatch, eviction=True)
    cache, identity = list(service._cache.items()), service._cache_identity
    guard = DecoderBudget("outer", 2 * SOURCE_BYTES, live=True)
    service.resource_guard = guard
    clone, capture = torch.Tensor.clone, _intervene.capture
    copies, captures = [], []

    def record_capture(*args, **kwargs):
        values = capture(*args, **kwargs)
        captures.append(values)
        return values

    def record_copy(tensor, *args, **kwargs):
        if tensor.ndim == 3:
            assert_retained(service, cache, identity)
            if failure == "copy" and copies:
                raise RuntimeError("decoder copy failed")
        result = clone(tensor, *args, **kwargs)
        if tensor.ndim == 3:
            copies.append(weakref.ref(result))
            guard.copied += tensor.numel() * tensor.element_size()
            if failure == "growth":
                guard.growth = 1
        return result

    with _points.capture(model, [POINTS[0]]) as existing:
        with service.suspend(), torch.no_grad():
            model(**tokenizer("CD"))
        observed = existing[POINTS[0]]
        hooks = hook_objects(model)
        monkeypatch.setattr(_intervene, "capture", record_capture)
        monkeypatch.setattr(torch.Tensor, "clone", record_copy)
        if failure == "forward":
            def fail(*args, **kwargs):
                raise RuntimeError("decoder forward failed")
            monkeypatch.setattr(model.norm, "forward", fail)
        elif failure == "nonfinite":
            monkeypatch.setattr(model.embed_tokens, "forward", lambda ids: torch.full((*ids.shape, 4), float("nan")))
        elif failure == "guard":
            check = guard.check

            def stop(stage):
                current = check(stage)
                if stage == "patch source copy" and copies:
                    raise CleanStop("decoder guard stop")
                return current

            monkeypatch.setattr(guard, "check", stop)
        error, match = {
            "growth": (CleanStop, "patch source allocation exceeds"),
            "guard": (CleanStop, "decoder guard stop"),
            "copy": (RuntimeError, "decoder copy failed"),
            "forward": (RuntimeError, "decoder forward failed"),
            "nonfinite": (_intervene.InterventionError, "finite batch-token-width"),
        }[failure]
        with pytest.raises(error, match=match) as refused:
            service._source(specification(POINTS[0]))
        assert_retained(service, cache, identity)
        assert len(copies) == (1 if failure in ("copy", "growth", "guard") else 0)
        # The exception retains its frames while the partial copies must disappear.
        assert refused.value.__traceback__ is not None
        assert all(reference() is None for reference in copies)
        assert len(captures) == 1 and captures[0]._closed and not captures[0]._handles and not captures[0]
        assert list(_points._ACTIVE[model].values()) == [existing]
        assert existing[POINTS[0]] is observed and hook_objects(model) == hooks
        assert not service.preparing_source


@pytest.mark.parametrize("lower", ["outer", "inner"])
@pytest.mark.parametrize("retiring", [False, True])
def test_returned_readings_admit_single_copy_before_source_retirement(
        decoder_source, monkeypatch, lower, retiring):
    model, tokenizer, service = decoder_source
    old = service.install(specification(POINTS[0], "CD")) if retiring else None
    cache, identity, hooks = list(service._cache.items()), service._cache_identity, hook_objects(model)
    monkeypatch.setattr(_intervene, "MAX_SOURCE_ENTRIES", 1)
    clone = torch.Tensor.clone

    class ReturnedBudget:
        def __init__(self, capacity):
            low = 8 * 1024 * 1024
            self.limit = low if lower == "outer" else 2 * low
            self._guard = SimpleNamespace(limit=low if lower == "inner" else 2 * low)
            self.baseline = low - capacity
            self.complete = False
            self.stages = []

        def check(self, stage):
            self.stages.append(stage)
            if stage == "patch source forward complete":
                self.complete = True
            return self.baseline if self.complete else 0

    def forbid_copy(tensor, *args, **kwargs):
        assert tensor.ndim != 3, "decoder source allocation preceded the returned memory check"
        return clone(tensor, *args, **kwargs)

    with service._retire_sources([old]):
        service.resource_guard = ReturnedBudget(SOURCE_BYTES - 1)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "clone", forbid_copy)
            with pytest.raises(CleanStop, match="patch source allocation exceeds"):
                service.install(specification(POINTS[0]))
        assert_retained(service, cache, identity)
        assert hook_objects(model) == hooks and model not in _points._ACTIVE
        assert not service.preparing_source and (old is None or not old.removed)
        service.resource_guard = ReturnedBudget(SOURCE_BYTES)
        copies = []

        def record_copy(tensor, *args, **kwargs):
            if tensor.ndim == 3:
                assert_retained(service, cache, identity)
                assert old is None or not old.removed
            result = clone(tensor, *args, **kwargs)
            if tensor.ndim == 3:
                copies.append(weakref.ref(result))
            return result

        monkeypatch.setattr(torch.Tensor, "clone", record_copy)
        new = service.install(specification(POINTS[0]))
        assert len(copies) == 1 and service.resource_guard.stages.count("patch source copy") == 1
        assert list(service._cache) == [("AB", POINTS[0])]
        source = service._cache["AB", POINTS[0]]
        assert copies[0]() is source
        with torch.no_grad():
            expected = model.embed_tokens(tokenizer("AB")["input_ids"])
        torch.testing.assert_close(source, expected, rtol=0, atol=0)
        if old is not None:
            assert not old.removed
            old()
    assert not new.removed and not service._retiring_sources and not service.preparing_source
    assert model not in _points._ACTIVE
