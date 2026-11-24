"""Exact dictionary bytes with bounded snapshots and unchanged restore identities."""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import pytest
import torch

from caliper.interp import _dictionary, _intervene, _memory
from caliper.interp._admission import GEMMA_ID
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


def test_snapshot_bounds_comparisons_and_checks_the_last_byte(controlled, monkeypatch):
    monkeypatch.setattr(_intervene, "_SNAPSHOT_COMPARE_BYTES", 128)
    dictionary, service = prepare(controlled)
    equal, sizes = torch.equal, []
    def compare(left, right):
        sizes.append(left.numel())
        return equal(left, right)
    monkeypatch.setattr(torch, "equal", compare)
    service.executor.validate(feature_spec())
    assert max(sizes) <= 128
    assert sum(sizes) == sum(t.numel() * t.element_size() for t in dictionary.state_dict().values())
    dictionary.layers[-1].W_dec.data[-1, -1].add_(.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        service.executor.validate(feature_spec())


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


@pytest.fixture
def device_snapshot(controlled, monkeypatch):
    """Label CPU views as MPS tensors and replace transfers with private CPU copies."""
    def make(*, model_id=GEMMA_ID, device="mps", guard=None):
        model, tokenizer, _, service = controlled
        dictionary = dictionary_for(model)
        dictionary.metadata["model_id"] = model_id
        dictionary.layers = torch.nn.ModuleList([
            _dictionary.JumpReLUTranscoder(layer.W_enc, layer.b_enc, layer.W_dec, layer.b_dec,
                                          torch.full_like(layer.b_enc, .05))
            for layer in dictionary.layers])
        guard = IdentityGuard() if guard is None else guard
        guard.device = torch.device(device)
        service.configure(dictionary=dictionary, tokenizer=tokenizer, resource_guard=guard)
        case = SimpleNamespace(dictionary=dictionary, service=service, executor=service.executor,
                               guard=guard, transfers=[], scopes=[], readback=None)

        @contextmanager
        def scope(device):
            case.scopes.append(torch.device(device))
            try:
                yield
            finally:
                case.scopes.pop()

        class DeviceLabel(torch.Tensor):
            @property
            def device(self):
                return torch.device(device)

            def to(self, *, device):
                assert device == "cpu"
                assert case.scopes[-1] == self.device
                assert case.readback is None or case.readback() is None
                case.transfers.append((self.numel() * self.element_size(), len(guard.stages)))
                result = self.as_subclass(torch.Tensor).clone()
                case.readback = weakref.ref(result)
                return result

        state_dict = dictionary.state_dict
        if device != "cpu":
            monkeypatch.setattr(dictionary, "state_dict", lambda: {
                name: tensor.as_subclass(DeviceLabel) for name, tensor in state_dict().items()})
        monkeypatch.setattr(_memory, "autorelease_scope", scope)
        monkeypatch.setattr(_dictionary, "autorelease_scope", scope)
        monkeypatch.setattr(_intervene, "_SNAPSHOT_COMPARE_BYTES", 4096)
        return case
    return make


@pytest.mark.parametrize("model_id,device,retained", [
    (GEMMA_ID, "mps", "cpu"), (GEMMA_ID, "mps:0", "cpu"),
    (GEMMA_ID, "cpu", "cpu"), ("Qwen/Qwen3.5-0.8B", "mps", "mps"),
    ("fixture", "cpu", "cpu"),
])
def test_snapshot_selects_cpu_retention_only_for_gemma_mps(device_snapshot, model_id, device, retained):
    case = device_snapshot(model_id=model_id, device=device)
    identifier = case.executor.dictionary_id
    case.executor._prepare_dictionary_identity()
    snapshot = case.executor._dictionary_snapshot
    assert snapshot is not None
    assert {str(t.device) for t in snapshot.state.values()} == {retained}
    assert set(snapshot.source_devices.values()) == {torch.device(device)}
    assert case.executor.dictionary_id == identifier
    assert case.executor._dictionary_status()["mode"] == "snapshot"
    assert snapshot.matches(case.dictionary, identifier, case.guard)
    assert not case.scopes


def test_cpu_snapshot_bounds_copy_and_reads_every_live_block(device_snapshot, monkeypatch):
    case = device_snapshot()
    empty, clone, equal = torch.empty, torch.Tensor.clone, torch.equal
    allocations, comparisons = [], []

    def allocate(*args, **kwargs):
        assert kwargs["device"] == "cpu"
        assert case.guard.stages[-1] == "dictionary identity"
        tensor = empty(*args, **kwargs)
        allocations.append(tensor.numel() * tensor.element_size())
        return tensor

    def copy(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return clone(tensor, *args, **kwargs)

    def compare(left, right):
        assert left.device.type == right.device.type == "cpu"
        assert left.dtype == right.dtype == torch.uint8
        assert left.data_ptr() != right.data_ptr()
        comparisons.append(left.numel())
        return equal(left, right)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.Tensor, "clone", copy)
    monkeypatch.setattr(torch, "equal", compare)
    monkeypatch.setattr(_intervene, "_cpu_byte_equal", compare)
    case.executor._prepare_dictionary_identity()
    total = sum(t.numel() * t.element_size() for t in case.dictionary.state_dict().values())
    assert sum(allocations) == total
    assert max(size for size, _ in case.transfers) <= 4096
    assert sum(size for size, _ in case.transfers) == 2 * total
    assert sum(comparisons) == total
    case.transfers.clear()
    comparisons.clear()
    def forbidden(*args, **kwargs):
        pytest.fail("CPU snapshot validation must compare the complete live bytes")
    monkeypatch.setattr(_intervene, "_content_identity", forbidden)
    case.executor.validate(feature_spec())
    assert sum(size for size, _ in case.transfers) == sum(comparisons) == total
    assert max(comparisons) <= 4096
    checks = [count for _, count in case.transfers]
    assert all(right > left for left, right in zip(checks, checks[1:]))
    case.dictionary.layers[-1].threshold.data[-1].add_(.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        case.executor.validate(feature_spec())


def test_cpu_snapshot_preserves_source_device_identity(device_snapshot, monkeypatch):
    case = device_snapshot()
    case.executor._prepare_dictionary_identity()
    snapshot = case.executor._dictionary_snapshot
    # The retained CPU bytes cannot make a later source move appear unchanged.
    state = torch.nn.Module.state_dict(case.dictionary)
    monkeypatch.setattr(case.dictionary, "state_dict", lambda: state)
    case.transfers.clear()
    assert not snapshot.matches(case.dictionary, case.executor.dictionary_id, case.guard)
    assert not case.transfers


@pytest.mark.parametrize("write", [
    "normal", "data", "numpy", "storage", "replacement", "late_block",
    "signed_zero", "shape", "dtype", "metadata", "activation", "threshold",
])
def test_cpu_snapshot_refuses_alias_and_identity_changes(device_snapshot, write):
    case = device_snapshot()
    case.executor._prepare_dictionary_identity()
    layer = case.dictionary.layers[-1]
    parameter = layer.b_dec
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
    elif write == "storage":
        alias = torch.empty(0, dtype=parameter.dtype).set_(parameter.untyped_storage())
        alias[-1].add_(.125)
        assert parameter._version == version
    elif write == "replacement":
        layer.b_dec = torch.nn.Parameter(parameter.detach().clone() + .125)
    elif write == "late_block":
        layer.W_dec.data[-1, -1].add_(.125)
    elif write == "signed_zero":
        assert parameter[-1] == 0
        parameter.data[-1] = -0.
    elif write == "shape":
        layer.W_dec.data = layer.W_dec.data.reshape(-1)
    elif write == "dtype":
        parameter.data = parameter.data.view(torch.int64)
    elif write == "metadata":
        case.dictionary.metadata["revision"] = "changed"
    elif write == "activation":
        layer.activation = "topk"
    else:
        layer.threshold.data[-1].add_(.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        case.executor.install(feature_spec())
    assert not case.executor._hooks


def test_cpu_snapshot_preserves_nan_payloads_without_version_counters(device_snapshot):
    with torch.inference_mode():
        case = device_snapshot()
        parameter = case.dictionary.layers[-1].b_dec
        with pytest.raises(RuntimeError, match="version counter"):
            _ = parameter._version
        bits = parameter.view(torch.int64)
        bits[-1] = 0x7FF8000000000001
        case.executor.dictionary_id = _intervene._dictionary_identifier(case.dictionary, case.guard)
        case.executor._prepare_dictionary_identity()
        case.dictionary.metadata["quality"] = {"fvu": .25}
        case.executor.validate(feature_spec())
        bits[-1] = 0x7FF8000000000002
        assert torch.isnan(parameter[-1])
        with pytest.raises(InterventionError, match="dictionary content changed"):
            case.executor.validate(feature_spec())


@pytest.mark.parametrize("lower", ["outer", "inner"])
@pytest.mark.parametrize("margin", [-1, 0])
def test_cpu_snapshot_admits_the_complete_reserve_before_allocation(device_snapshot, monkeypatch, lower, margin):
    guard = IdentityGuard()
    guard._guard = IdentityGuard()
    case = device_snapshot(guard=guard)
    state = case.dictionary.state_dict()
    sizes = [t.numel() * t.element_size() for t in state.values()]
    scratch = max(4 * min(max(sizes), _intervene._SNAPSHOT_COMPARE_BYTES),
                  4 * _dictionary._HASH_CHUNK_BYTES)
    selected = guard if lower == "outer" else guard._guard
    selected.peak = 321
    selected.limit = selected.peak + sum(sizes) + scratch + margin
    def forbidden(*args, **kwargs):
        pytest.fail("the complete snapshot must fit before its first allocation")
    with monkeypatch.context() as patch:
        if margin < 0:
            patch.setattr(torch, "empty", forbidden)
        case.executor._prepare_dictionary_identity()
    if margin == 0:
        assert case.executor._dictionary_snapshot is not None
        return
    assert case.executor._dictionary_snapshot is None and not case.transfers
    assert case.executor._dictionary_status()["mode"] == "hash"
    case.executor.validate(feature_spec())
    assert sum(size for size, _ in case.transfers) == sum(sizes)
    assert max(size for size, _ in case.transfers) <= _dictionary._HASH_CHUNK_BYTES
    case.dictionary.layers[-1].threshold.data[-1].add_(.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        case.executor.validate(feature_spec())


@pytest.mark.parametrize("phase", ["create", "compare"])
@pytest.mark.parametrize("failure", ["outer_limit", "inner_limit", "stop"])
def test_cpu_snapshot_stops_between_blocks(device_snapshot, monkeypatch, phase, failure):
    guard = IdentityGuard()
    guard._guard = IdentityGuard()
    case = device_snapshot(guard=guard)
    if phase == "compare":
        case.executor._prepare_dictionary_identity()
        case.transfers.clear()
    check = guard.check

    def stopped(stage):
        check(stage)
        if case.transfers:
            if failure == "stop":
                raise CleanStop("snapshot stop")
            selected = guard if failure == "outer_limit" else guard._guard
            selected.limit = 1

    monkeypatch.setattr(guard, "check", stopped)
    with pytest.raises(CleanStop, match="snapshot stop|allocation exceeds"):
        if phase == "create":
            case.executor._prepare_dictionary_identity()
        else:
            case.executor.validate(feature_spec())
    assert len(case.transfers) == 1 and not case.scopes and case.readback() is None
    if phase == "create":
        assert case.executor._dictionary_snapshot is None


@pytest.mark.parametrize("target", ["copied", "pending"])
def test_cpu_snapshot_refuses_changes_during_copy(device_snapshot, monkeypatch, target):
    case = device_snapshot()
    copy, copies = torch.Tensor.copy_, []

    def changed(tensor, source, *args, **kwargs):
        result = copy(tensor, source, *args, **kwargs)
        if not copies:
            parameter = (case.dictionary.layers[0].W_enc if target == "copied" else
                         case.dictionary.layers[-1].threshold)
            parameter.data.view(-1)[0].add_(.125)
        copies.append(tensor.numel())
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", changed)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        case.executor._prepare_dictionary_identity()
    assert case.executor._dictionary_snapshot is None and not case.scopes


def test_cpu_snapshot_replay_release_and_live_weights(device_snapshot, controlled):
    case = device_snapshot()
    model, tokenizer, controls, _ = controlled
    case.executor._prepare_dictionary_identity()
    case.service.set(feature_spec())
    controls.step()
    state = case.service.state_dict()
    expected = model(**tokenizer("abc")).logits
    with case.service.temporary_state(state):
        torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)
    saved = weakref.ref(next(iter(case.executor._dictionary_snapshot.state.values())))
    assert saved() is not None
    case.dictionary.layers[-1].threshold.data[-1].add_(.125)
    with pytest.raises(InterventionError, match="dictionary content changed"):
        with case.service.temporary_state(state):
            pytest.fail("a replay must refuse changed live bytes")
    case.service.configure(dictionary=case.dictionary, tokenizer=tokenizer, resource_guard=case.guard)
    assert saved() is None and case.executor._dictionary_snapshot is None
    case.executor._prepare_dictionary_identity()
    case.service.set(feature_spec())
    controls.step()
    expected = model(**tokenizer("abc")).logits
    # Private snapshot damage must not supply a different decoder to an installed hook.
    case.executor._dictionary_snapshot.state["layers.1.W_dec"].fill_(100)
    torch.testing.assert_close(model(**tokenizer("abc")).logits, expected, rtol=0, atol=0)
    saved = weakref.ref(next(iter(case.executor._dictionary_snapshot.state.values())))
    case.executor.close()
    assert saved() is None


@pytest.mark.parametrize("fallback", [False, True])
def test_cpu_byte_comparison_checks_complete_views_and_offsets(monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(_intervene, "_native_byte_compare", lambda: None)
    left = torch.arange(255, dtype=torch.uint8)
    right = left.clone()
    assert _intervene._cpu_byte_equal(left[7:250], right[7:250])
    assert _intervene._cpu_byte_equal(left[:0], right[:0])
    right[249] ^= 1
    assert not _intervene._cpu_byte_equal(left[7:250], right[7:250])
    assert _intervene._cpu_byte_equal(left[7:249], right[7:249])


@pytest.mark.parametrize("fault", ["dtype", "shape", "stride"])
def test_cpu_byte_comparison_refuses_invalid_views_before_native_call(monkeypatch, fault):
    left = torch.arange(16, dtype=torch.uint8)
    right = left.clone()
    if fault == "dtype":
        left = left.float()
    elif fault == "shape":
        right = right[:4]
    else:
        left, right = left[::2], right[::2]
    monkeypatch.setattr(_intervene, "_native_byte_compare", lambda: pytest.fail("invalid native comparison"))
    with pytest.raises(ValueError, match="contiguous CPU byte views"):
        _intervene._cpu_byte_equal(left, right)
