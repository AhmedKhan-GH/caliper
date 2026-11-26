"""Tiny CPU evidence for prepared published model byte comparisons."""
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest
import torch

from caliper.interp import _intervene, _published_identity as identity
from caliper.interp._memory import CleanStop
from tests.interp.test_intervention_identity import IdentityGuard
from tests.interp.test_replication_groups import grouped, published


def snapshot_for(service, guard=None):
    original = service.executor._identity()
    return identity._PublishedModelSnapshot.create(service.executor, original, guard)


def matches(snapshot, service, guard=None):
    executor = service.executor
    return snapshot.matches(executor, executor.model, executor.tokenizer, guard)


def test_preparation_preserves_full_hash_and_actions_scan_every_byte(published, monkeypatch):
    from caliper.interp import _replication_actions as actions
    model, _, _, service, runtime = published
    original = service.executor._identity()
    assert runtime.preparation['model_content_hash'] == original[4]
    assert runtime._identity == original
    calls, sizes, forwards = [], [], []
    full_hash = _intervene._content_identity
    compare = _intervene._cpu_byte_equal
    def hashed(state, guard):
        if any(name.startswith('parameter:') for name in state):
            calls.append(tuple(state))
        return full_hash(state, guard)
    def compared(left, right):
        sizes.append(left.numel())
        assert left.data_ptr() != right.data_ptr()
        return compare(left, right)
    monkeypatch.setattr(_intervene, '_content_identity', hashed)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', compared)
    monkeypatch.setattr(_intervene, '_SNAPSHOT_COMPARE_BYTES', 1024)
    forward = actions.forward_logits
    def counted(*args, **kwargs):
        forwards.append(1)
        return forward(*args, **kwargs)
    monkeypatch.setattr(actions, 'forward_logits', counted)
    for _ in range(2):
        runtime.run(26)
    assert forwards == [1, 1]
    assert calls == []
    assert sum(sizes) == 4 * runtime._model_snapshot.nbytes
    assert max(sizes) <= 1024
    assert runtime.status('AB', service.specifications())['model_identity']['mode'] == 'snapshot'
    service.executor._identity()
    assert len(calls) == 1, 'Ordinary source identity must retain the full hash boundary'


@pytest.mark.parametrize('change', ['data', 'numpy', 'signed_zero', 'replacement', 'storage', 'shape', 'dtype', 'stride'])
def test_snapshot_detects_parameter_bytes_and_structure(grouped, change):
    model, _, _, service = grouped
    snapshot = snapshot_for(service)
    parameter = model.layers[0].mlp.b_dec
    version = parameter._version
    if change == 'data':
        parameter.data[-1] += .125
    elif change == 'numpy':
        parameter.detach().numpy()[-1] = .125
    elif change == 'signed_zero':
        parameter.data[-1] = -0.
        assert torch.equal(parameter, snapshot.state['parameter:layers.0.mlp.b_dec'])
    elif change == 'replacement':
        model.layers[0].mlp.b_dec = torch.nn.Parameter(parameter.detach().clone())
    elif change == 'storage':
        parameter.data = parameter.detach().clone()
    elif change == 'shape':
        parameter.data = parameter.data.view(8, 8)
    elif change == 'dtype':
        parameter.data = parameter.data.float()
    else:
        parameter.data = torch.zeros(parameter.numel() * 2, dtype=parameter.dtype)[::2]
    assert parameter._version == version
    if change == 'stride':
        with pytest.raises(ValueError, match='contiguous'):
            matches(snapshot, service)
    else:
        assert not matches(snapshot, service)


@pytest.mark.parametrize('persistent', [False, True])
def test_snapshot_checks_buffer_aliases_and_signed_zero(grouped, persistent):
    model, _, _, service = grouped
    model.register_buffer('inv_freq', torch.zeros(4), persistent=persistent)
    model.register_buffer('original_inv_freq', model.inv_freq, persistent=False)
    snapshot = snapshot_for(service)
    version = model.inv_freq._version
    model.original_inv_freq.data[-1] = -0.
    assert model.inv_freq._version == version
    assert not matches(snapshot, service)


def test_snapshot_detects_alias_registration_and_empty_storage_offset(grouped):
    model, _, _, service = grouped
    storage = torch.empty(2)
    model.register_buffer('empty', storage[:0], persistent=False)
    snapshot = snapshot_for(service)
    model.empty.data = storage[1:1]
    assert not matches(snapshot, service)
    model.register_parameter('alias', model.embed_tokens.weight)
    assert matches(snapshot_for(service), service)
    snapshot = snapshot_for(service)
    del model._parameters['alias']
    assert not matches(snapshot, service)


def test_inference_tensors_need_no_version_counter(grouped):
    model, _, _, service = grouped
    with torch.inference_mode():
        model.register_buffer('inv_freq', torch.zeros(4), persistent=False)
        model.embed_tokens.weight = torch.nn.Parameter(model.embed_tokens.weight.clone())
        with pytest.raises(RuntimeError, match='version counter'):
            _ = model.embed_tokens.weight._version
        snapshot = snapshot_for(service)
        assert matches(snapshot, service)
        model.embed_tokens.weight.data[-1, -1] += .125
        assert not matches(snapshot, service)


@pytest.mark.parametrize('change', ['training', 'epsilon', 'autocast', 'matmul', 'module', 'forward', 'tensor_setting'])
def test_snapshot_rechecks_execution_settings(grouped, monkeypatch, change):
    model, _, _, service = grouped
    if change == 'tensor_setting':
        model.norm.frozen_scale = torch.ones(1, 3, 1, dtype=torch.float64)
    snapshot = snapshot_for(service)
    if change == 'training':
        model.train()
    elif change == 'epsilon':
        model.norm.eps *= 2
    elif change == 'autocast':
        with torch.autocast('cpu', dtype=torch.bfloat16):
            assert not matches(snapshot, service)
        return
    elif change == 'matmul':
        original = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('medium')
            assert not matches(snapshot, service)
        finally:
            torch.set_float32_matmul_precision(original)
        return
    elif change == 'module':
        model.norm = deepcopy(model.norm)
    elif change == 'forward':
        monkeypatch.setattr(model.norm, 'forward', lambda value: value)
    else:
        model.norm.frozen_scale.data[-1] = -0.
    assert not matches(snapshot, service)


@pytest.mark.parametrize('change', ['dictionary', 'dictionary_snapshot', 'identifier', 'executor_model', 'runtime_model', 'tokenizer'])
def test_stale_preparation_refuses_without_publication(published, monkeypatch, change):
    from tests.interp.fixture import make_fixture
    model, tokenizer, _, service, runtime = published
    first = runtime.run(26)
    records, files = service.state_dict(), set(runtime.directory.iterdir())
    if change == 'dictionary':
        monkeypatch.setattr(service.executor, 'dictionary', deepcopy(service.dictionary))
    elif change == 'dictionary_snapshot':
        service.executor._prepare_dictionary_identity()
    elif change == 'identifier':
        monkeypatch.setattr(service.executor, 'dictionary_id', 'sha256:stale')
    elif change == 'executor_model':
        monkeypatch.setattr(service.executor, 'model', make_fixture()[0])
    elif change == 'runtime_model':
        monkeypatch.setattr(runtime, 'model', make_fixture()[0])
    else:
        monkeypatch.setattr(runtime, 'tokenizer', deepcopy(tokenizer))
    published_outputs = []
    with pytest.raises(ValueError):
        runtime.run(10, prepare=lambda value: published_outputs.append(value))
    assert not published_outputs
    assert service.state_dict() == records and runtime.last_result == first
    assert set(runtime.directory.iterdir()) == files
    assert runtime._model_snapshot.nbytes == 0


@pytest.mark.parametrize('change', ['alias', 'buffer', 'setting', 'hook'])
def test_mutation_between_scans_rolls_back_and_releases_copies(published, monkeypatch, change):
    from caliper.interp import _replication_actions as actions
    model, _, ctl, service, runtime = published
    first = runtime.run(26)
    records, state = dict(ctl._ivs), service.state_dict()
    files = set(runtime.directory.iterdir())
    copies = [weakref.ref(value) for value in runtime._model_snapshot.state.values()]
    forward = actions.forward_logits
    handles = []
    def changed(*args, **kwargs):
        output = forward(*args, **kwargs)
        if change == 'alias':
            model.layers[0].mlp.b_dec.data[-1] = -0.
        elif change == 'buffer':
            model.register_buffer('inv_freq', torch.ones(1), persistent=False)
        elif change == 'setting':
            model.norm.eps *= 2
        else:
            handles.append(model.register_forward_hook(lambda *args: None))
        return output
    monkeypatch.setattr(actions, 'forward_logits', changed)
    outputs = []
    try:
        with pytest.raises(ValueError, match='changed during'):
            runtime.run(10, prepare=lambda value: outputs.append(value))
    finally:
        for handle in handles:
            handle.remove()
    assert not outputs and service.state_dict() == state
    assert all(ctl._ivs[key] is record for key, record in records.items())
    assert runtime.last_result == first and set(runtime.directory.iterdir()) == files
    assert all(ref() is None for ref in copies)
    assert runtime.status('AB', service.specifications())['model_identity']['mode'] == 'hash'


def test_failure_fallback_retains_full_hash_and_previous_replay(published, monkeypatch):
    model, tokenizer, _, service, runtime = published
    first = runtime.run(26)
    expected = model(**tokenizer('AB')).logits
    def refuse(_):
        raise RuntimeError('publication refusal')
    with pytest.raises(RuntimeError, match='publication refusal'):
        runtime.run(10, prepare=refuse)
    assert runtime.last_result == first
    torch.testing.assert_close(model(**tokenizer('AB')).logits, expected, rtol=0, atol=0)
    calls = []
    full = runtime._model_identity
    def hashed():
        calls.append(1)
        return full()
    monkeypatch.setattr(runtime, '_model_identity', hashed)
    runtime.run(10)
    assert calls == [1, 1]
    assert runtime._model_snapshot.nbytes == 0


@pytest.mark.parametrize('limit', ['outer', 'inner', 'executor', 'cap', 'historical'])
def test_preparation_refuses_before_private_allocation(grouped, monkeypatch, limit):
    _, _, _, service = grouped
    original = service.executor._identity()
    guard = IdentityGuard(limit=50_000_000_000)
    guard._guard = IdentityGuard(limit=50_000_000_000)
    if limit == 'outer':
        guard.limit = 1
    elif limit == 'inner':
        guard._guard.limit = 1
    elif limit == 'executor':
        service.executor.resource_guard = IdentityGuard(limit=1)
    elif limit == 'cap':
        guard.current = 39_999_999_999
    else:
        guard.peak = 40_000_000_001
    monkeypatch.setattr(torch.Tensor, 'clone', lambda *a, **k: pytest.fail('copy before admission'))
    with pytest.raises(CleanStop, match='process memory limit'):
        identity._PublishedModelSnapshot.create(service.executor, original, guard)


@pytest.mark.parametrize('phase', ['copy', 'compare'])
def test_late_guard_stop_releases_private_views(grouped, monkeypatch, phase):
    _, _, _, service = grouped
    guard = IdentityGuard()
    original = service.executor._identity()
    clone = torch.Tensor.clone
    copies = []
    def cloned(value, *args, **kwargs):
        result = clone(value, *args, **kwargs)
        copies.append(weakref.ref(result))
        if phase == 'copy':
            guard.limit = 1
        return result
    monkeypatch.setattr(torch.Tensor, 'clone', cloned)
    if phase == 'compare':
        compare = _intervene._cpu_byte_equal
        def compared(left, right):
            guard.limit = 1
            return compare(left, right)
        monkeypatch.setattr(_intervene, '_cpu_byte_equal', compared)
    with pytest.raises(CleanStop):
        identity._PublishedModelSnapshot.create(service.executor, original, guard)
    assert copies and all(ref() is None for ref in copies)


def test_initial_copy_must_reproduce_original_hash(grouped, monkeypatch):
    model, _, _, service = grouped
    original = service.executor._identity()
    model.layers[0].mlp.b_dec.data[-1] = -0.
    with pytest.raises(ValueError, match='preparation hash'):
        identity._PublishedModelSnapshot.create(service.executor, original, None)


def test_close_releases_all_private_copies_and_hooks(published):
    _, _, _, _, runtime = published
    copies = [weakref.ref(value) for value in runtime._model_snapshot.state.values()]
    runtime.close()
    assert all(ref() is None for ref in copies)
    assert runtime._model_snapshot is None and runtime.preparation is None
    assert not runtime._handles and not runtime._forward_scopes
    with pytest.raises(ValueError, match='incomplete'):
        runtime.run(10)


@pytest.fixture
def reference_snapshot(monkeypatch):
    from caliper.interp import _gemma_reference as reference
    from tests.interp.test_gemma_reference import reference_fixture
    model = reference_fixture()
    reference.install_reference_profile(model, reference.PROFILE)
    executor = _intervene.InterventionService(model)
    executor.dictionary = torch.nn.Module()
    executor.dictionary_id, executor._default_dictionary_id = 'fixture', True
    guard = IdentityGuard()
    executor.resource_guard = guard
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=64_000_000_000))
    original = executor._identity()
    snapshot = identity._PublishedModelSnapshot.create(executor, original, guard)
    return model, executor, guard, snapshot


@pytest.mark.parametrize('change', ['rotary', 'code', 'operation', 'reference_hook', 'training'])
def test_reference_profile_checks_remain_live(reference_snapshot, monkeypatch, change):
    from caliper.interp import _gemma_reference as reference
    model, executor, guard, snapshot = reference_snapshot
    def changed(module, value):
        return value
    if change == 'rotary':
        model.model.rotary_emb.inv_freq.data[-1] += .125
        assert not snapshot.matches(executor, model, executor.tokenizer, guard)
    else:
        if change == 'code':
            monkeypatch.setattr(reference.normalized_input, '__code__', changed.__code__)
        elif change == 'operation':
            monkeypatch.setattr(reference, 'normalized_input', changed)
        elif change == 'reference_hook':
            model.model.layers[0].pre_feedforward_layernorm.reference_input = torch.nn.Identity()
        else:
            model.train()
        with pytest.raises(ValueError):
            snapshot.matches(executor, model, executor.tokenizer, guard)


@pytest.mark.parametrize('phase', ['create', 'compare'])
@pytest.mark.parametrize('owner', ['runtime', 'executor'])
def test_snapshot_obeys_each_supplied_guard_latch(grouped, monkeypatch, phase, owner):
    _, _, _, service = grouped
    guard, extra = IdentityGuard(), IdentityGuard()
    service.executor.resource_guard = extra
    original = service.executor._identity()
    snapshot = None
    if phase == 'compare':
        snapshot = identity._PublishedModelSnapshot.create(service.executor, original, guard)
    selected = guard if owner == 'runtime' else extra
    error = CleanStop('retained stop')
    def stopped(stage):
        raise error
    monkeypatch.setattr(selected, 'check', stopped)
    monkeypatch.setattr(torch.Tensor, 'clone', lambda *a, **k: pytest.fail('copy after a guard stop'))
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', lambda *a, **k: pytest.fail('compare after a guard stop'))
    with pytest.raises(CleanStop, match='retained stop') as caught:
        if phase == 'create':
            identity._PublishedModelSnapshot.create(service.executor, original, guard)
        else:
            matches(snapshot, service, guard)
    assert caught.value is error


@pytest.mark.parametrize('phase', ['create', 'compare'])
@pytest.mark.parametrize('reserve', [8_000_000_000, 9_000_000_000])
def test_real_model_policy_refuses_before_allocation_or_comparison(reference_snapshot, monkeypatch, phase, reserve):
    model, executor, guard, snapshot = reference_snapshot
    guard.host_reserve_bytes = reserve
    # Real-model policy with a tiny randomly initialized CPU Gemma fixture.
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=reserve + 1))
    monkeypatch.setattr(torch.Tensor, 'clone', lambda *a, **k: pytest.fail('copy before host admission'))
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', lambda *a, **k: pytest.fail('compare before host admission'))
    with pytest.raises(CleanStop, match='host reserve'):
        if phase == 'create':
            identity._PublishedModelSnapshot.create(executor, snapshot.original, guard)
        else:
            snapshot.matches(executor, model, executor.tokenizer, guard)


def test_prospective_retained_floor_survives_stale_process_reader(reference_snapshot, monkeypatch):
    model, executor, guard, snapshot = reference_snapshot
    assert getattr(guard, 'current', 0) == 0
    assert snapshot.floor >= 2 * snapshot.nbytes
    available = 8_000_000_000 + snapshot.floor - 1
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', lambda *a: pytest.fail('comparison consumed pending reserve'))
    with pytest.raises(CleanStop, match='host reserve'):
        snapshot.matches(executor, model, executor.tokenizer, guard)


def test_preparation_failure_releases_snapshot_and_owned_directory(published, monkeypatch):
    model, tokenizer, _, service, ready = published
    from caliper.interp import _replication_actions as actions
    runtime = actions.PublishedCases(model, tokenizer, service, output_directory=ready.output_directory)
    create = identity._PublishedModelSnapshot.create
    copies = []
    def created(*args):
        snapshot = create(*args)
        copies.extend(weakref.ref(value) for value in snapshot.state.values())
        return snapshot
    def fail_after_write(name, binding):
        actions.PublishedCases._write_json(runtime, name, binding)
        raise RuntimeError('preparation export refusal')
    monkeypatch.setattr(identity._PublishedModelSnapshot, 'create', created)
    monkeypatch.setattr(runtime, '_write_json', fail_after_write)
    directories = set(ready.output_directory.iterdir())
    with pytest.raises(RuntimeError, match='preparation export refusal'):
        runtime.prepare()
    assert copies and all(ref() is None for ref in copies)
    assert set(ready.output_directory.iterdir()) == directories
    assert runtime._model_snapshot is None and not runtime._handles
    assert runtime.preparation is None


def test_failed_reprepare_preserves_prior_evidence(published, monkeypatch):
    _, _, _, _, runtime = published
    runtime.run(26)
    directory = runtime.directory
    files = {path: path.read_bytes() for path in directory.iterdir()}
    runtime.close()
    monkeypatch.setattr(runtime, '_model_identity', lambda: (_ for _ in ()).throw(RuntimeError('identity refusal')))
    with pytest.raises(RuntimeError, match='identity refusal'):
        runtime.prepare()
    assert {path: path.read_bytes() for path in directory.iterdir()} == files


def test_complete_comparison_continues_after_first_byte_mismatch(grouped, monkeypatch):
    model, _, _, service = grouped
    snapshot = snapshot_for(service)
    model.embed_tokens.weight.data[0, 0] += .125
    sizes = []
    compare = _intervene._cpu_byte_equal
    def compared(left, right):
        sizes.append(left.numel())
        return compare(left, right)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', compared)
    assert not matches(snapshot, service)
    assert sum(sizes) == snapshot.nbytes


# The existing device double executes tensor operations only on CPU storage.
from tests.interp.test_dictionary_snapshot import device_snapshot
from tests.interp.test_intervene import controlled


@pytest.mark.parametrize('capacity', [1, 3])
def test_published_device_batches_reuse_bounded_byte_checks(device_snapshot, monkeypatch, capacity):
    from caliper.interp import _memory
    case = device_snapshot()
    label = type(next(iter(case.dictionary.state_dict().values())))
    state = {'parameter:weight': torch.arange(71, dtype=torch.uint8).as_subclass(label)}
    monkeypatch.setattr(_intervene, '_model_state', lambda model: state.copy())
    monkeypatch.setattr(_intervene, '_SNAPSHOT_COMPARE_BYTES', 16)
    monkeypatch.setattr(_intervene, '_SNAPSHOT_BATCH_BLOCKS', capacity)
    monkeypatch.setattr(identity, 'autorelease_scope', _memory.autorelease_scope)
    original = case.executor._identity()
    case.copies.clear()
    snapshot = identity._PublishedModelSnapshot.create(case.executor, original, case.guard)
    assert sum(size for size, _ in case.copies) == 71
    case.comparisons.clear()
    case.transfers.clear()
    assert snapshot.matches(case.executor, case.executor.model, case.executor.tokenizer, case.guard)
    assert sum(size for size, _ in case.comparisons) == 71
    assert [size for size, _ in case.transfers] == ([1] * 5 if capacity == 1 else [3, 2])
    assert not case.scopes
    state['parameter:weight'][-1] = 0
    assert not snapshot.matches(case.executor, case.executor.model, case.executor.tokenizer, case.guard)
    snapshot.release()
    assert snapshot.nbytes == 0


def test_close_failure_still_releases_model_copies(published):
    _, _, _, _, runtime = published
    copies = [weakref.ref(value) for value in runtime._model_snapshot.state.values()]
    class FailedScope:
        def __exit__(self, *args):
            raise RuntimeError('scope cleanup refusal')
    runtime._forward_scopes.append(FailedScope())
    with pytest.raises(RuntimeError, match='scope cleanup refusal'):
        runtime.close()
    assert all(ref() is None for ref in copies)
    assert runtime._model_snapshot is None


def test_dictionary_content_failure_releases_snapshot_before_forward(published, monkeypatch):
    from caliper.interp import _replication_actions as actions
    _, _, _, service, runtime = published
    service.dictionary.layers[-1].W_dec.data[-1, -1] += .125
    monkeypatch.setattr(actions, 'forward_logits', lambda *a, **k: pytest.fail('forward with stale dictionary'))
    with pytest.raises(ValueError, match='dictionary content changed'):
        runtime.run(26)
    assert not service.specifications()
    assert runtime._model_snapshot.nbytes == 0
    assert list(runtime.directory.iterdir()) == [runtime.directory / 'preparation.json']


def test_stale_dictionary_snapshot_does_not_retain_its_private_copies(grouped):
    _, _, _, service = grouped
    service.executor._prepare_dictionary_identity()
    previous = weakref.ref(service.executor._dictionary_snapshot)
    copies = [weakref.ref(value) for value in service.executor._dictionary_snapshot.state.values()]
    snapshot = snapshot_for(service)
    service.executor._prepare_dictionary_identity()
    assert previous() is None and all(ref() is None for ref in copies)
    assert not matches(snapshot, service)


def test_unguarded_snapshot_bound_refuses_before_clone(grouped, monkeypatch):
    _, _, _, service = grouped
    original = service.executor._identity()
    monkeypatch.setattr(identity._dictionary, '_UNVERIFIED_CPU_BYTES', 1)
    monkeypatch.setattr(torch.Tensor, 'clone', lambda *a, **k: pytest.fail('copy before unguarded admission'))
    with pytest.raises(CleanStop, match='unguarded CPU limit'):
        identity._PublishedModelSnapshot.create(service.executor, original, None)


def test_snapshot_admits_the_current_hash_workspace_before_copies(grouped, monkeypatch):
    _, _, _, service = grouped
    original = service.executor._identity()
    monkeypatch.setattr(identity._dictionary, '_HASH_CHUNK_BYTES', 8_000_000)
    monkeypatch.setattr(torch.Tensor, 'clone', lambda *a, **k: pytest.fail('copy before hash workspace admission'))
    with pytest.raises(CleanStop, match='process memory limit'):
        identity._PublishedModelSnapshot.create(service.executor, original, IdentityGuard())
