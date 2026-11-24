"""Exact dictionary bytes with bounded snapshots and unchanged restore identities."""
import weakref

import pytest
import torch

from caliper.interp import _intervene
from caliper.interp._intervene import InterventionError
from caliper.interp._memory import CleanStop
from .test_intervene import controlled, dictionary_for
from .test_intervention_dictionary_identity import feature_spec
from .test_intervention_identity import IdentityGuard


def prepare(controlled, *, guard=None):
    model, tokenizer, _, service = controlled
    dictionary = dictionary_for(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer, resource_guard=guard)
    identifier = service.executor.dictionary_id
    service.executor._prepare_dictionary_identity()
    assert service.executor._dictionary_snapshot is not None
    assert service.executor.dictionary_id == identifier
    return dictionary, service


def test_snapshot_checks_every_byte_without_hash_or_cpu_copy(controlled, monkeypatch):
    dictionary, service = prepare(controlled)
    sizes = []
    equal = torch.equal

    def compare(left, right):
        assert left.dtype == right.dtype == torch.uint8
        assert left.data_ptr() != right.data_ptr()
        sizes.append(left.numel())
        return equal(left, right)

    def forbidden(*args, **kwargs):
        pytest.fail("unchanged dictionary validation must use the byte snapshot")

    monkeypatch.setattr(torch, "equal", compare)
    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    monkeypatch.setattr(_intervene, "_content_identity", forbidden)
    service.executor.validate(feature_spec())
    expected = [t.numel() * t.element_size() for t in dictionary.state_dict().values()]
    assert sizes == expected


@pytest.mark.parametrize("write", ["normal", "data", "numpy", "replacement", "topk", "metadata", "signed_zero"])
def test_snapshot_refuses_mutation_before_installation(controlled, write):
    dictionary, service = prepare(controlled)
    executor = service.executor
    parameter = dictionary.layers[-1].b_dec
    version = parameter._version
    if write == "normal":
        with torch.no_grad():
            parameter[-1].add_(.125)
    elif write == "data":
        parameter.data[-1].add_(.125)
        assert parameter._version == version
    elif write == "numpy":
        parameter.detach().numpy()[-1] += .125
        assert parameter._version == version
    elif write == "replacement":
        dictionary.layers[-1].b_dec = torch.nn.Parameter(parameter.detach().clone() + .125)
    elif write == "topk":
        dictionary.layers[-1].k -= 1
    elif write == "metadata":
        dictionary.metadata["revision"] = "changed"
    else:
        assert parameter[-1] == 0
        parameter.data[-1] = -0.
    with pytest.raises(InterventionError, match="dictionary content changed"):
        executor.install(feature_spec())
    assert not executor._hooks


def test_snapshot_preserves_nan_bits_and_ignores_quality(controlled):
    model, tokenizer, _, service = controlled
    dictionary = dictionary_for(model)
    dictionary.layers[-1].b_dec.data[-1] = float("nan")
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    service.executor._prepare_dictionary_identity()
    dictionary.metadata["quality"] = {"fvu": .25}
    assert service.executor.validate(feature_spec())["dictionary"] == service.executor.dictionary_id
    dictionary.layers[-1].b_dec.data[-1] = 0.
    with pytest.raises(InterventionError, match="dictionary content changed"):
        service.executor.validate(feature_spec())


def test_snapshot_supports_inference_tensors_without_version_counters(controlled):
    with torch.inference_mode():
        dictionary, service = prepare(controlled)
        with pytest.raises(RuntimeError, match="version counter"):
            _ = dictionary.layers[0].W_dec._version
        service.executor.validate(feature_spec())
        dictionary.layers[0].W_dec[0, 0].add_(.125)
        with pytest.raises(InterventionError, match="dictionary content changed"):
            service.executor.validate(feature_spec())


def test_snapshot_restore_and_replay_preserve_exact_logits(controlled):
    model, tokenizer, controls, _ = controlled
    dictionary, service = prepare(controlled)
    service.set(feature_spec())
    controls.step()
    state = service.state_dict()
    expected = model(**tokenizer("abc")).logits
    with service.temporary_state(state):
        torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)
    # A replay checks bytes again even when it borrows the original snapshot.
    dictionary.layers[-1].W_dec.data[-1, -1].add_(.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        with service.temporary_state(state):
            pytest.fail("a replay must refuse changed bytes")
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
    service.load_state_dict(state, restore_active=True)
    torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)


def test_snapshot_is_released_on_reconfiguration_and_close(controlled):
    model, tokenizer, _, _ = controlled
    _, service = prepare(controlled)
    saved = weakref.ref(next(iter(service.executor._dictionary_snapshot.state.values())))
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
    assert saved() is None and service.executor._dictionary_snapshot is None
    service.executor._prepare_dictionary_identity()
    saved = weakref.ref(next(iter(service.executor._dictionary_snapshot.state.values())))
    service.executor.close()
    assert saved() is None


def test_snapshot_falls_back_to_full_hash_when_reserve_does_not_fit(controlled, monkeypatch):
    model, tokenizer, _, service = controlled
    guard = IdentityGuard()
    dictionary = dictionary_for(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer, resource_guard=guard)
    guard.limit = 4 * 1024 * 1024 + 1
    def forbidden(*args, **kwargs):
        pytest.fail("snapshot allocation must follow admission")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "clone", forbidden)
        service.executor._prepare_dictionary_identity()
    assert service.executor._dictionary_snapshot is None
    service.executor.validate(feature_spec())
    dictionary.layers[-1].W_dec.data[-1, -1].add_(.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        service.executor.validate(feature_spec())


def test_snapshot_checks_lower_limits_between_copies(controlled, monkeypatch):
    model, tokenizer, _, service = controlled
    guard = IdentityGuard()
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer, resource_guard=guard)
    clone, copies = torch.Tensor.clone, []
    def copied(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        guard.limit = 1
        return clone(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "clone", copied)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        service.executor._prepare_dictionary_identity()
    assert len(copies) == 1 and service.executor._dictionary_snapshot is None


def test_snapshot_checks_lower_limits_between_comparisons(controlled, monkeypatch):
    guard = IdentityGuard()
    _, service = prepare(controlled, guard=guard)
    equal, comparisons = torch.equal, []
    def compare(left, right):
        comparisons.append(left.numel())
        guard.limit = 1
        return equal(left, right)
    monkeypatch.setattr(torch, "equal", compare)
    with pytest.raises(CleanStop, match="allocation exceeds"):
        service.executor.validate(feature_spec())
    assert len(comparisons) == 1


def test_snapshot_refuses_mutation_between_initial_hash_and_copy(controlled, monkeypatch):
    model, tokenizer, _, service = controlled
    dictionary = dictionary_for(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    clone, copies = torch.Tensor.clone, []
    def copied(tensor, *args, **kwargs):
        if not copies:
            dictionary.layers[-1].W_dec.data[-1, -1].add_(.125)
        copies.append(tensor.numel())
        return clone(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "clone", copied)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        service.executor._prepare_dictionary_identity()
    assert service.executor._dictionary_snapshot is None


@pytest.mark.parametrize("phase", ["create", "compare"])
@pytest.mark.parametrize("guard", [None, IdentityGuard(), IdentityGuard(limit=float("inf"))])
def test_snapshot_requires_the_device_guard_before_any_device_operation(controlled, monkeypatch, phase, guard):
    dictionary, service = prepare(controlled)
    executor = service.executor
    class DeviceLabel(torch.Tensor):
        @property
        def device(self):
            return torch.device("mps")
    state = {name: tensor.as_subclass(DeviceLabel) for name, tensor in dictionary.state_dict().items()}
    monkeypatch.setattr(dictionary, "state_dict", lambda: state)
    def forbidden(*args, **kwargs):
        pytest.fail("a matching bounded guard must precede device work")
    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    monkeypatch.setattr(torch, "equal", forbidden)
    executor.resource_guard = guard
    with pytest.raises(ValueError, match="guard"):
        if phase == "create":
            executor._prepare_dictionary_identity()
        else:
            executor.validate(feature_spec())
