"""Clean patch sources follow scalar settings and current tokenized inputs."""
import pytest
import torch

from caliper._controls import Controls
from caliper.interp._intervene import InterventionError, InterventionService, _source_input_identity
from caliper.interp._memory import CleanStop
from caliper.interp._points import capture
from .fixture import TinyLM


class MutableTokenizer:
    add_bos_token = True

    def __call__(self, text, **kwargs):
        tokens = [65, 66] if text == "source" else [67, 68]
        if self.add_bos_token:
            tokens.insert(0, 256)
        ids = torch.tensor([tokens])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.parametrize("change", ["epsilon", "tokens"])
def test_patch_refreshes_after_scalar_or_tokenizer_change(change):
    model = TinyLM(layers=2, width=8, heads=2, latents=4, k=1).eval()
    tokenizer = MutableTokenizer()
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    point = "layers.1.resid_pre" if change == "epsilon" else "layers.0.resid_pre"
    spec = {"kind": "patch", "point": point, "source_prompt": "source", "source_position": 0}
    hook = service.install(spec)
    previous = service._cache["source", point].clone()
    if change == "epsilon":
        model.layers[0].input_layernorm.eps = 2.0
    else:
        tokenizer.add_bos_token = False
    try:
        with service.suspend(), capture(model, [point]) as values, torch.no_grad():
            model(**tokenizer("source"))
            expected = values[point].clone()
        with capture(model, [point]) as values, torch.no_grad():
            model(**tokenizer("target"))
            actual = values[point].clone()
        assert hook.error is None
        torch.testing.assert_close(actual, expected[:, :1].expand_as(actual), rtol=0, atol=0)
        torch.testing.assert_close(service._cache["source", point], expected, rtol=0, atol=0)
        assert previous.shape != expected.shape or not torch.equal(previous, expected)
        assert service._cache_identity[1]
        hook()
        assert not service._cache and not service._cache_identity[1]
    finally:
        service.close()
    assert service._cache_identity is None and not service._cache


def test_unchanged_tokenized_source_reuses_cached_tensor():
    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).eval()
    service = InterventionService(model)
    service.configure(tokenizer=MutableTokenizer())
    spec = {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "source"}
    try:
        first = service._source(spec)
        assert service._source(spec) is first
    finally:
        service.close()


class ViewTokenizer(MutableTokenizer):
    def __init__(self, layout):
        self.layout = layout
        self.ids = torch.tensor([[256, 0, 65, 0, 66, 0]])[:, ::2]

    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        if self.layout in ("ids", "both"):
            encoded["input_ids"] = self.ids
        if self.layout in ("mask", "both"):
            encoded["attention_mask"] = torch.ones(1, 1, dtype=torch.int64).expand(1, 3)
        return encoded


@pytest.mark.parametrize("layout", ["ids", "mask", "both"])
def test_patch_accepts_input_views_and_reuses_equal_content(layout):
    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).eval()
    tokenizer = ViewTokenizer(layout)
    service = InterventionService(model)
    service.configure(tokenizer=tokenizer)
    point = "layers.0.resid_pre"
    spec = {"kind": "patch", "point": point, "source_prompt": "source", "source_position": 0}
    try:
        hook = service.install(spec)
        original = service._cache["source", point]
        expected = model.embed_tokens(tokenizer("source")["input_ids"])
        torch.testing.assert_close(original, expected, rtol=0, atol=0)
        tokenizer.layout = "dense"
        assert service._source(spec) is original
        tokenizer.layout = "both"
        tokenizer.ids[0, 0] = 67
        with capture(model, [point]) as values, torch.no_grad():
            model(**MutableTokenizer()("target"))
            actual = values[point].clone()
        expected = model.embed_tokens(tokenizer.ids)
        torch.testing.assert_close(actual, expected[:, :1].expand_as(actual), rtol=0, atol=0)
        assert service._cache["source", point] is not original
        assert hook.error is None
    finally:
        service.close()


@pytest.mark.parametrize("lower", ["outer", "inner"])
def test_input_copy_and_hash_reserve_precede_allocation(monkeypatch, lower):
    from caliper.interp._dictionary import _HASH_CHUNK_BYTES
    from .test_intervention_identity import IdentityGuard

    inputs = ViewTokenizer("both")("source")
    reserve = sum(value.numel() * value.element_size() for value in inputs.values())
    guard = IdentityGuard()
    guard._guard = IdentityGuard()
    selected = guard if lower == "outer" else guard._guard
    selected.limit = 4 * _HASH_CHUNK_BYTES + reserve - 1

    def forbidden(*args, **kwargs):
        pytest.fail("input canonicalization preceded memory admission")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", forbidden)
        with pytest.raises(CleanStop, match="allocation exceeds"):
            _source_input_identity(inputs, guard)
    assert guard.stages[:2] == ["patch source input copy", "model_identity"]
    selected.limit += 1
    canonical, digest = _source_input_identity(inputs, guard)
    assert all(value.is_contiguous() for value in canonical.values())
    assert digest == _source_input_identity(canonical, None)[1]


def test_input_canonicalization_preserves_weight_layout_refusal():
    from caliper.interp._dictionary import _state_hash

    inputs = ViewTokenizer("both")("source")
    _source_input_identity(inputs, None)
    with pytest.raises(ValueError, match="contiguous dense tensors"):
        _state_hash(inputs)


@pytest.mark.parametrize("operation", ["boundary", "restore", "override"])
def test_source_evidence_restores_with_the_control_transaction(operation):
    from .test_intervention_replay import snapshot

    class ArmedTokenizer(MutableTokenizer):
        change_on_call = False

        def __call__(self, text, **kwargs):
            if self.change_on_call:
                self.add_bos_token = False
                self.change_on_call = False
            return super().__call__(text, **kwargs)

    model = TinyLM(layers=1, width=8, heads=2, latents=4, k=1).eval()
    tokenizer = ArmedTokenizer()
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(tokenizer=tokenizer)
    executor = service.executor
    spec = {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "source", "source_position": 0}
    try:
        service.set(spec)
        controls.step()
        original = executor._cache["source", spec["point"]]
        cache, identity = executor._cache.copy(), executor._cache_identity
        old_hook = controls._ivs[spec["point"], "patch"].hook
        replacement = dict(spec, source_position=1)
        if operation == "boundary":
            service.set(replacement)
            transaction = controls.step
        elif operation == "restore":
            transaction = lambda: service.load_state_dict(snapshot(replacement), restore_active=True)
        else:
            transaction = lambda: service.apply_overrides([replacement])
        tokenizer.change_on_call = True
        observed = []

        def cancel_preparation(module, args, kwargs):
            assert executor.preparing_source
            observed.append(kwargs["input_ids"].shape[1])
            # A real request changes the transaction revision during its forward.
            service.set({"kind": "add", "point": "layers.0.mlp_out", "vector": [.1] * 8, "coefficient": 1.})

        observer = model.register_forward_pre_hook(cancel_preparation, with_kwargs=True)
        try:
            if operation == "boundary":
                transaction()
            else:
                with pytest.raises(InterventionError, match="state changed"):
                    transaction()
        finally:
            observer.remove()
        assert observed == [2] and not tokenizer.add_bos_token
        assert controls._ivs[spec["point"], "patch"].hook is old_hook and not old_hook.removed
        assert executor._cache_identity is identity
        assert executor._cache.keys() == cache.keys()
        assert all(executor._cache[key] is value for key, value in cache.items())
        assert ("layers.0.mlp_out", "add") in controls._ivs
        refreshed = executor._source(spec)
        assert refreshed is not original
        assert refreshed.shape[1] == 2
        expected = model.embed_tokens(tokenizer("source")["input_ids"])
        torch.testing.assert_close(refreshed, expected, rtol=0, atol=0)
        assert executor._cache_identity[1] != identity[1]
    finally:
        service.remove()
