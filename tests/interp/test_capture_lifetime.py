"""Bounded CPU checks for capture ownership and late hook callbacks."""
import threading

import pytest
import torch

from caliper import interp
from caliper.interp import _points
from .test_points import hooks, qwen


def overlap(monkeypatch, first, second, paused, resume):
    """Release the first operation after the second blocks or completes."""
    reached = threading.Event()
    errors = []
    lock = getattr(_points, "_LIFECYCLE_LOCK", threading.RLock())

    class ObservedLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                if threading.current_thread().name == "capture-contender":
                    reached.set()
                lock.acquire()
            return self

        def __exit__(self, *args):
            lock.release()

    monkeypatch.setattr(_points, "_LIFECYCLE_LOCK", ObservedLock(), raising=False)

    def run(action, contender=False):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if contender:
                reached.set()

    owner = threading.Thread(target=run, args=(first,), name="capture-owner")
    contender = threading.Thread(target=run, args=(second, True), name="capture-contender")
    owner.start()
    try:
        assert paused.wait(5), errors
        contender.start()
        assert reached.wait(5), errors
    finally:
        resume.set()
        owner.join(5)
        if contender.ident is not None:
            contender.join(5)
    assert not owner.is_alive() and not contender.is_alive()
    assert not errors


def test_last_restore_and_new_capture_share_one_owner(monkeypatch):
    model = qwen()
    config = model.config
    config._attn_implementation = "sdpa"
    point = "model.layers.3.attn_pattern"
    first = interp.capture(model, [point]).__enter__()
    second = interp.capture(model, [point])
    paused, resume = threading.Event(), threading.Event()
    original = type(config).__setattr__

    def restore(self, name, value):
        if (self is config and name == "_attn_implementation" and value == "sdpa"
                and threading.current_thread().name == "capture-owner"):
            paused.set()
            assert resume.wait(5)
        original(self, name, value)

    monkeypatch.setattr(type(config), "__setattr__", restore)
    try:
        overlap(monkeypatch, first.close, second.__enter__, paused, resume)
        assert config._attn_implementation == "eager"
        assert _points._EAGER[id(config)][2] == 1
        model(torch.tensor([[1, 2]]), use_cache=False)
        assert second[point].shape == (1, 2, 2, 2)
    finally:
        second.close(clear=True)
        first.close(clear=True)
    assert config._attn_implementation == "sdpa"
    assert id(config) not in _points._EAGER
    assert hooks(model) == 0


@pytest.mark.parametrize("teardown", ["close", "watch"])
def test_teardown_waits_for_complete_hook_install(monkeypatch, teardown):
    from caliper import watch

    model = qwen()
    model.config._attn_implementation = "sdpa"
    watched = watch(model, activations=False, groups=False, scene=False) if teardown == "watch" else None
    existing = model.model.layers[0].register_forward_hook(lambda *args: None)
    values = interp.capture(model, ["model.layers.3.attn_pattern"])
    attention = model.model.layers[3].self_attn
    register = attention.register_forward_hook
    paused, resume = threading.Event(), threading.Event()

    def install(*args, **kwargs):
        paused.set()
        assert resume.wait(5)
        return register(*args, **kwargs)

    monkeypatch.setattr(attention, "register_forward_hook", install)
    try:
        remove = watched.remove if watched is not None else lambda: values.close(clear=True)
        overlap(monkeypatch, values.__enter__, remove, paused, resume)
        assert hooks(model) == 1
        assert not values._handles and not values._configs and not values._pending
        assert model.config._attn_implementation == "sdpa"
        assert id(model.config) not in _points._EAGER
        assert id(values) not in _points._ACTIVE.get(model, {})
        with pytest.raises(RuntimeError, match="once"):
            values.__enter__()
    finally:
        values.close(clear=True)
        if watched is not None:
            watched.remove()
        existing.remove()


@pytest.mark.parametrize("clear", [False, True])
def test_late_hooks_cannot_change_closed_results(clear):
    model = qwen()
    point = "model.layers.3.attn_pattern"
    values = interp.capture(model, [point]).__enter__()
    attention = model.model.layers[3].self_attn
    start = next(iter(model._forward_pre_hooks.values()))
    callback = next(iter(attention._forward_hooks.values()))
    callback(attention, (), (None, torch.ones(1, 2, 1, 1)))
    retained = values[point]
    values.close(clear=clear)
    start(model, ())
    callback(attention, (), (None, torch.zeros(1, 2, 1, 1)))
    if clear:
        assert not values
    else:
        assert values[point] is retained
    assert hooks(model) == 0


def test_late_delta_hooks_cannot_retain_projection_tensors():
    model = qwen()
    point = "model.layers.0.attn_pattern"
    values = interp.capture(model, [point]).__enter__()
    attention = model.model.layers[0].linear_attn
    start = next(iter(attention._forward_pre_hooks.values()))
    keep = next(iter(attention.in_proj_qkv._forward_hooks.values()))
    finish = next(iter(attention._forward_hooks.values()))
    pending = values._pending[point]
    keep(attention.in_proj_qkv, (), torch.ones(1, 2, attention.conv_dim))
    assert pending
    values.close(clear=True)
    assert not pending
    start(attention, (), {})
    keep(attention.in_proj_qkv, (), torch.ones(1, 2, attention.conv_dim))
    finish(attention, (), None)
    assert not values and not values._pending
    assert hooks(model) == 0


@pytest.mark.parametrize("suffix,module", [
    ("resid_pre", "model.layers.0"),
    ("mlp_in", "model.layers.0.mlp"),
    ("attn_pattern", "model.layers.0.linear_attn"),
])
def test_removed_pre_hook_tolerates_pytorch_dispatch_without_kwargs(suffix, module):
    model = qwen(1)
    values = interp.capture(model, ["model.layers.0." + suffix]).__enter__()
    # PyTorch snapshots hooks, but checks with_kwargs membership just before each call.
    stop = model.get_submodule(module).register_forward_pre_hook(
        lambda *_: values.close(clear=True), prepend=True)
    try:
        model(torch.tensor([[1, 2]]), use_cache=False)
        assert not values and not values._pending
        assert hooks(model) == 1
    finally:
        stop.remove()
        values.close(clear=True)


def test_close_during_delta_computation_discards_late_result(monkeypatch):
    model = qwen(1)
    values = interp.capture(model, ["model.layers.0.attn_pattern"]).__enter__()
    paused, resume = threading.Event(), threading.Event()

    def pattern(module, observed):
        assert set(observed) == {"qkv", "a", "b"}
        paused.set()
        assert resume.wait(5)
        return observed["qkv"].new_ones(1, 2, 2, 2)

    def close():
        values.close(clear=True)
        assert not resume.is_set(), "Close must not wait for the tensor computation."

    monkeypatch.setattr(_points, "_delta_pattern", pattern)
    try:
        overlap(monkeypatch, lambda: model(torch.tensor([[1, 2]]), use_cache=False),
                close, paused, resume)
        assert not values and not values._pending
        assert hooks(model) == 0
    finally:
        values.close(clear=True)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_nested_patterns_preserve_weights_masks_and_backend(backend):
    model = qwen()
    model.config._attn_implementation = backend
    point = "model.layers.3.attn_pattern"
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        baseline = model(ids, attention_mask=mask, use_cache=False).logits
        with interp.capture(model, [point]) as outer:
            with interp.capture(model, [point]) as inner:
                assert model.config._attn_implementation == "eager"
                actual = model(ids, attention_mask=mask, use_cache=False).logits
                torch.testing.assert_close(inner[point], outer[point], atol=0, rtol=0)
            assert model.config._attn_implementation == "eager"
            assert id(inner) not in _points._ACTIVE[model]
            assert _points._EAGER[id(model.config)][2] == 1
        assert model.config._attn_implementation == backend
        torch.testing.assert_close(actual, baseline, atol=1e-6, rtol=1e-5)
        assert not outer[point].triu(1).count_nonzero()
        assert not outer[point][1, :, :, 2].count_nonzero()
        torch.testing.assert_close(outer[point].sum(-1), torch.ones(2, 2, 3))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], atol=0, rtol=0)
    assert model not in _points._ACTIVE
    assert id(model.config) not in _points._EAGER
    assert hooks(model) == 0


def test_pattern_backend_remains_visible_to_evaluation_binding():
    from tools.interp.evaluation_binding import EvaluationBinding, _evaluation_settings

    model = qwen()
    model.config._attn_implementation = "sdpa"
    # Exercise the actual runtime guard before artifact checks, without an artifact.
    binding = EvaluationBinding.__new__(EvaluationBinding)
    binding.model, binding.memory = model, None
    binding.model_stamps = binding._model_stamps()
    binding.model_settings = _evaluation_settings(model, None)
    before = dict(binding.model_settings)
    with interp.capture(model, ["model.layers.3.attn_pattern"]):
        current = _evaluation_settings(model, None)
        assert current["config._attn_implementation"] == '"eager"'
        with pytest.raises(ValueError, match="forward setting changed: .*_attn_implementation"):
            binding.check()
    assert _evaluation_settings(model, None) == before
    # A deliberate backend change must produce the same refusal after capture ends.
    model.config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="forward setting changed: .*_attn_implementation"):
        binding.check()


def test_failed_nested_install_preserves_other_owner(monkeypatch):
    model = qwen()
    model.config._attn_implementation = "sdpa"
    point = "model.layers.3.attn_pattern"
    with interp.capture(model, [point]) as outer:
        def failed(*args, **kwargs):
            raise RuntimeError("synthetic hook install failure")

        monkeypatch.setattr(model.model.layers[3].self_attn, "register_forward_hook", failed)
        with pytest.raises(RuntimeError, match="synthetic hook install failure"):
            with interp.capture(model, [point]):
                pass
        assert model.config._attn_implementation == "eager"
        assert _points._EAGER[id(model.config)][2] == 1
        assert list(_points._ACTIVE[model].values()) == [outer]
        model(torch.tensor([[1, 2]]), use_cache=False)
        assert outer[point].shape == (1, 2, 2, 2)
    assert model.config._attn_implementation == "sdpa"
    assert hooks(model) == 0
