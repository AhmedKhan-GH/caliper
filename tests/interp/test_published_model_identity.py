"""Independent CPU byte evidence for published model identity reuse."""
from copy import deepcopy
import gc
import json
from types import SimpleNamespace
import weakref

import pytest
import torch

from caliper.interp import _dictionary, _intervene, _published_identity
from caliper.interp._frozen import _validate_dictionary
from caliper.interp._memory import CleanStop
from tests.interp.test_intervention_identity import IdentityGuard
from tests.interp.test_published_identity import reference_snapshot, snapshot_for
from tests.interp.test_replication_groups import grouped, published


class Owner:
    pass


def bind(snapshot, guard=None):
    owner, service = Owner(), Owner()
    service.executor = snapshot.executor
    owner.model, owner.tokenizer = snapshot.model, snapshot.tokenizer
    owner.interventions, owner.guard = service, guard
    owner._model_snapshot, owner.preparation = snapshot, {}
    snapshot.bind_public_identity(owner)
    snapshot.register_public_identity()
    return owner


def raw(tensor):
    """Read source storage independently of the production comparison helper."""
    return tensor.detach().reshape(-1).view(torch.uint8).numpy().tobytes()


def hashes(monkeypatch):
    calls = []
    full = _dictionary._state_hash
    def recorded(state, **kwargs):
        calls.append(tuple(state))
        return full(state, **kwargs)
    monkeypatch.setattr(_dictionary, '_state_hash', recorded)
    return calls


def test_trace_dictionary_admission_reads_all_bytes_without_full_hash(published, monkeypatch):
    model, _, _, service, runtime = published
    snapshot = runtime._model_snapshot
    expected = _dictionary._full_model_identity(model)
    trace_dictionary = deepcopy(service.dictionary)
    trace_dictionary.metadata.update(layers=len(model.layers), latents=256, k=8, activation='topk')
    source = b''.join(raw(value) for value in model.state_dict().values())
    calls, sizes, stages, observed = hashes(monkeypatch), [], [], []
    def compared(left, right):
        sizes.append(left.numel())
        observed.append(raw(left))
        return observed[-1] == raw(right)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', compared)
    monkeypatch.setattr(torch.Tensor, 'clone', lambda *a, **k: pytest.fail('additional model copy'))
    monkeypatch.setattr(_intervene, '_SNAPSHOT_COMPARE_BYTES', 8192)
    for _ in range(2):
        _validate_dictionary(model, trace_dictionary, stages.append, memory=IdentityGuard())
    assert calls == []
    assert b''.join(observed) == source * 2
    assert sum(sizes) == 2 * len(source)
    assert max(sizes) <= 8192 and stages.count('model_identity') >= 2 * len(sizes)
    assert _dictionary.model_identity(model) == expected
    # Finite validation still runs on each admission.
    assert stages.count('dictionary identity') == 2
    assert stages.count('finite block') > 0


@pytest.mark.parametrize('change', ['data', 'numpy', 'negative_zero', 'byte', 'replacement', 'storage', 'alias'])
def test_changed_weights_use_full_current_hash_and_cannot_rearm(published, monkeypatch, change):
    model, _, _, _, runtime = published
    snapshot = runtime._model_snapshot
    parameter = model.layers[0].mlp.b_dec
    before, version = raw(parameter), parameter._version
    saved = parameter.detach().clone()
    if change == 'data':
        parameter.data[-1] += .125
    elif change == 'numpy':
        parameter.detach().numpy()[-1] = .125
    elif change == 'negative_zero':
        parameter.data[-1] = -0.
        assert torch.equal(parameter, saved)
    elif change == 'byte':
        parameter.detach().view(torch.uint8).numpy()[-1] ^= 1
    elif change == 'replacement':
        model.layers[0].mlp.b_dec = torch.nn.Parameter(saved)
    elif change == 'storage':
        parameter.data = saved
    else:
        model.register_parameter('tied', parameter)
    assert parameter._version == version
    if change in ('data', 'numpy', 'negative_zero', 'byte'):
        assert raw(parameter) != before
    expected = _dictionary._full_model_identity(model)
    calls = hashes(monkeypatch)
    assert _dictionary.model_identity(model) == expected and calls
    assert snapshot._public is None
    parameter.data.copy_(saved)
    calls.clear()
    _dictionary.model_identity(model)
    assert calls
    with pytest.raises(ValueError, match='cannot be rebound'):
        snapshot.bind_public_identity(runtime)


@pytest.mark.parametrize('persistent', [False, True])
def test_buffer_aliases_and_transient_bytes_have_no_version_shortcut(grouped, monkeypatch, persistent):
    model, _, _, service = grouped
    model.register_buffer('inv_freq', torch.zeros(4), persistent=persistent)
    model.register_buffer('original_inv_freq', model.inv_freq, persistent=False)
    snapshot = snapshot_for(service)
    owner = bind(snapshot)
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert not calls
    before, version = raw(model.inv_freq), model.inv_freq._version
    model.original_inv_freq.data[-1] = -0.
    assert raw(model.inv_freq) != before and model.inv_freq._version == version
    _dictionary.model_identity(model)
    assert calls and owner._model_snapshot._public is None


@pytest.mark.parametrize('change', ['setting_bytes', 'config', 'training', 'module', 'registration', 'empty_offset'])
def test_settings_and_registration_changes_reject_reuse(grouped, monkeypatch, change):
    model, _, _, service = grouped
    model.norm.frozen_scale = torch.ones(1, 3, 1, dtype=torch.float64)
    storage = torch.empty(2)
    model.register_buffer('empty', storage[:0], persistent=False)
    snapshot = snapshot_for(service)
    owner = bind(snapshot)
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls == []
    if change == 'setting_bytes':
        before = raw(model.norm.frozen_scale)
        model.norm.frozen_scale.data[-1] = -0.
        assert raw(model.norm.frozen_scale) != before
    elif change == 'config':
        model.config.hidden_size += 1
    elif change == 'training':
        model.train()
    elif change == 'module':
        model.norm = deepcopy(model.norm)
    elif change == 'registration':
        model._non_persistent_buffers_set.remove('empty')
    else:
        model.empty.data = storage[1:1]
    _dictionary.model_identity(model)
    assert calls and owner._model_snapshot._public is None


@pytest.mark.parametrize('phase', ['before', 'after'])
@pytest.mark.parametrize('serializer', ['state_dict', 'save', 'extra', 'pre_hook', 'post_hook'])
def test_custom_serializers_always_use_full_hash(grouped, monkeypatch, phase, serializer):
    model, _, _, service = grouped
    snapshot = snapshot_for(service)
    owner = bind(snapshot) if phase == 'after' else None
    if serializer == 'state_dict':
        original = model.state_dict
        monkeypatch.setattr(model, 'state_dict', lambda **kwargs: original(**kwargs))
    elif serializer == 'save':
        original = model._save_to_state_dict
        monkeypatch.setattr(model, '_save_to_state_dict', lambda *a: original(*a))
    elif serializer == 'extra':
        monkeypatch.setattr(model, 'get_extra_state', lambda: None)
    elif serializer == 'pre_hook':
        model.register_state_dict_pre_hook(lambda *a: None)
    else:
        model.register_state_dict_post_hook(lambda *a: None)
    if owner is None:
        owner = bind(snapshot)
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls and owner._model_snapshot._public is None


@pytest.mark.parametrize('replacement', ['snapshot', 'service', 'executor', 'model', 'tokenizer', 'guard',
                                          'executor_guard', 'dictionary', 'dictionary_snapshot', 'config'])
def test_owner_replacement_in_callback_rejects_proof(published, monkeypatch, replacement):
    model, _, _, service, runtime = published
    snapshot = runtime._model_snapshot
    compare = _intervene._cpu_byte_equal
    called = False
    def changed(left, right):
        nonlocal called
        if not called:
            called = True
            if replacement == 'snapshot':
                runtime._model_snapshot = None
            elif replacement == 'service':
                runtime.interventions = Owner()
            elif replacement == 'executor':
                service.executor = Owner()
            elif replacement == 'model':
                runtime.model = Owner()
            elif replacement == 'tokenizer':
                runtime.tokenizer = deepcopy(runtime.tokenizer)
            elif replacement == 'guard':
                runtime.guard = IdentityGuard()
            elif replacement == 'executor_guard':
                service.executor.resource_guard = IdentityGuard()
            elif replacement == 'dictionary':
                service.executor.dictionary = deepcopy(service.dictionary)
            elif replacement == 'dictionary_snapshot':
                service.executor._dictionary_snapshot = Owner()
            else:
                model.config = deepcopy(model.config)
        return compare(left, right)
    calls = hashes(monkeypatch)
    # Restore test owners before the shared fixture cleanup.
    with monkeypatch.context() as patch:
        for owner, field in ((runtime, '_model_snapshot'), (runtime, 'interventions'), (runtime, 'model'),
                             (runtime, 'tokenizer'), (runtime, 'guard'), (service, 'executor'),
                             (service.executor, 'resource_guard'), (service.executor, 'dictionary'),
                             (service.executor, '_dictionary_snapshot'), (model, 'config')):
            patch.setattr(owner, field, getattr(owner, field))
        patch.setattr(_intervene, '_cpu_byte_equal', changed)
        _dictionary.model_identity(model)
        assert called and calls and snapshot._public is None
    calls.clear()
    _dictionary.model_identity(model)
    assert calls


@pytest.mark.parametrize('phase', ['check', 'comparison', 'last_check'])
def test_fresh_settings_surround_callbacks(published, monkeypatch, phase):
    model, _, _, _, runtime = published
    snapshot = runtime._model_snapshot
    compared = 0
    def compare(left, right):
        nonlocal compared
        compared += left.numel()
        if phase == 'comparison':
            model.norm.eps *= 2
        return raw(left) == raw(right)
    def check(stage):
        if phase == 'check' or (phase == 'last_check' and compared == snapshot.nbytes):
            model.norm.eps *= 2
    calls = hashes(monkeypatch)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', compare)
    _dictionary.model_identity(model, check=check)
    assert calls and snapshot._public is None


@pytest.mark.parametrize('exception', [CleanStop, ValueError])
def test_caller_cancellation_stops_between_bounded_blocks(published, monkeypatch, exception):
    model, _, _, _, runtime = published
    snapshot = runtime._model_snapshot
    compared = []
    error = exception('caller cancellation')
    def compare(left, right):
        compared.append(left.numel())
        return raw(left) == raw(right)
    def check(stage):
        assert stage == 'model_identity'
        if compared:
            raise error
    monkeypatch.setattr(_intervene, '_SNAPSHOT_COMPARE_BYTES', 1024)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', compare)
    with pytest.raises(exception) as caught:
        _dictionary.model_identity(model, check=check)
    assert caught.value is error and compared == [1024] and snapshot._public is None
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls


@pytest.fixture
def trusted(reference_snapshot):
    model, executor, guard, snapshot = reference_snapshot
    loader = IdentityGuard()
    _dictionary._record_verified_hf_load(model, memory=loader)
    runtime_guard = IdentityGuard()
    owner = bind(snapshot, runtime_guard)
    return model, executor, runtime_guard, loader, snapshot, owner


def test_trusted_snapshot_reuses_exact_loader_identity(trusted, monkeypatch):
    model, _, _, _, snapshot, _ = trusted
    expected = _dictionary._full_model_identity(model)
    calls = hashes(monkeypatch)
    assert _dictionary.model_identity(model) == expected
    assert _dictionary.model_identity(model) == expected and not calls
    assert snapshot._public[0] == expected


@pytest.mark.parametrize('change', ['data', 'transient', 'setting', 'name', 'revision', 'verification'])
def test_trusted_identity_rejects_live_changes(trusted, monkeypatch, change):
    model, _, _, _, snapshot, _ = trusted
    if change == 'data':
        parameter = next(model.parameters())
        before, version = raw(parameter), parameter._version
        parameter.detach().view(torch.uint8).numpy().reshape(-1)[-1] ^= 1
        assert raw(parameter) != before and parameter._version == version
    elif change == 'transient':
        model.model.rotary_emb.inv_freq.data[-1] += .125
    elif change == 'setting':
        model.config.final_logit_softcapping *= 2
    elif change == 'name':
        model.config._name_or_path = 'fixture/changed'
    elif change == 'revision':
        model.config._commit_hash = 'c' * 40
    else:
        original = _dictionary._VERIFIED_MODELS[model]
        monkeypatch.setitem(_dictionary._VERIFIED_MODELS, model, deepcopy(original))
    if change == 'verification':
        calls = hashes(monkeypatch)
        _dictionary.model_identity(model)
        assert calls
    else:
        with pytest.raises(ValueError):
            _dictionary.model_identity(model)
    assert snapshot._public is None


@pytest.mark.parametrize('owner', ['loader', 'runtime', 'executor', 'caller'])
@pytest.mark.parametrize('refusal', ['limit', 'history', 'latch', 'inner'])
def test_every_guard_remains_authoritative(trusted, monkeypatch, owner, refusal):
    model, executor, runtime, loader, snapshot, proof_owner = trusted
    caller = IdentityGuard()
    # The executor and runtime can supply distinct guards at preparation.
    if owner == 'executor':
        selected = executor.resource_guard
    else:
        selected = {'loader': loader, 'runtime': runtime, 'caller': caller}[owner]
    error = CleanStop('retained guard refusal')
    if refusal == 'limit':
        selected.limit = 1
    elif refusal == 'history':
        selected.peak = selected.limit + 1
    elif refusal == 'inner':
        selected._guard = IdentityGuard(limit=1)
    else:
        monkeypatch.setattr(selected, 'check', lambda *a, **k: (_ for _ in ()).throw(error))
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', lambda *a: pytest.fail('comparison after refusal'))
    with pytest.raises(CleanStop):
        _dictionary.model_identity(model, memory=caller)
    assert snapshot._public is None


@pytest.mark.parametrize('with_guard', [False, True])
def test_lower_explicit_budget_refuses_before_comparison(published, monkeypatch, with_guard):
    model, _, _, _, runtime = published
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', lambda *a: pytest.fail('comparison before admission'))
    with pytest.raises(CleanStop):
        _dictionary.model_identity(model, memory=IdentityGuard() if with_guard else None, memory_limit_gb=1e-9)
    assert runtime._model_snapshot._public is None


@pytest.mark.parametrize('owner', ['loader', 'runtime', 'caller'])
def test_host_reserve_includes_prospective_workspace(trusted, monkeypatch, owner):
    model, _, runtime, loader, snapshot, _ = trusted
    caller = IdentityGuard()
    selected = {'loader': loader, 'runtime': runtime, 'caller': caller}[owner]
    selected.host_reserve_bytes = 9_000_000_000
    available = selected.host_reserve_bytes + snapshot.floor + 1
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', lambda *a: pytest.fail('comparison before host admission'))
    with pytest.raises(CleanStop, match='host reserve'):
        _dictionary.model_identity(model, memory=caller)


def test_release_unregisters_proof_and_frees_snapshot_storage(published, monkeypatch):
    model, _, _, _, runtime = published
    snapshot = runtime._model_snapshot
    copies = [weakref.ref(value) for value in snapshot.state.values()]
    runtime.close()
    assert snapshot._public is None and model not in _dictionary._PUBLISHED_MODELS
    assert all(ref() is None for ref in copies)
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls
    with pytest.raises(ValueError, match='cannot be rebound'):
        snapshot.bind_public_identity(runtime)


def test_registry_does_not_retain_owner_or_snapshot(grouped, monkeypatch):
    model, _, _, service = grouped
    snapshot = snapshot_for(service)
    owner = bind(snapshot)
    snapshot_ref, owner_ref = weakref.ref(snapshot), weakref.ref(owner)
    del snapshot, owner
    gc.collect()
    assert owner_ref() is None and snapshot_ref() is None
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls


@pytest.mark.parametrize('failure', ['hash', 'comparison', 'registration'])
def test_preparation_failure_cleans_proof_and_private_storage(published, monkeypatch, failure):
    model, _, _, _, runtime = published
    runtime.close()
    copies = []
    create = _published_identity._PublishedModelSnapshot.create
    def created(*args):
        snapshot = create(*args)
        copies.extend(weakref.ref(value) for value in snapshot.state.values())
        return snapshot
    def refused(*a, **k):
        raise ValueError('proof preparation refusal')
    monkeypatch.setattr(_published_identity._PublishedModelSnapshot, 'create', created)
    if failure == 'hash':
        monkeypatch.setattr(_dictionary, '_full_model_identity', refused)
    elif failure == 'comparison':
        original = _dictionary._full_model_identity
        def hashed(*a, **k):
            result = original(*a, **k)
            monkeypatch.setattr(_intervene, '_cpu_byte_equal', refused)
            return result
        monkeypatch.setattr(_dictionary, '_full_model_identity', hashed)
    else:
        monkeypatch.setattr(_published_identity._PublishedModelSnapshot, 'register_public_identity', refused)
    with pytest.raises(ValueError, match='proof preparation refusal'):
        runtime.prepare()
    assert copies and all(ref() is None for ref in copies)
    assert runtime._model_snapshot is None and runtime.preparation is None
    assert model not in _dictionary._PUBLISHED_MODELS


def test_bind_requires_full_current_hash_and_matching_snapshot(grouped, monkeypatch):
    model, _, _, service = grouped
    snapshot = snapshot_for(service)
    model.layers[0].mlp.b_dec.data[-1] = -0.
    calls = hashes(monkeypatch)
    with pytest.raises(ValueError, match='differs from its snapshot'):
        bind(snapshot)
    assert calls and snapshot._public is None and model not in _dictionary._PUBLISHED_MODELS


def test_ordinary_model_attributes_cannot_authorize_reuse(grouped, monkeypatch):
    model, _, _, _ = grouped
    model._public_identity = 'forged'
    model._checkpoint = 'forged'
    expected = _dictionary._full_model_identity(model)
    calls = hashes(monkeypatch)
    assert _dictionary.model_identity(model) == expected and calls


def test_proof_cost_remains_in_preparation(published, monkeypatch):
    from caliper.interp import _replication_actions as actions
    _, _, _, _, runtime = published
    runtime.close()
    clock = [10.]
    monkeypatch.setattr(actions.time, 'perf_counter', lambda: clock[0])
    full = _dictionary._full_model_identity
    def hashed(*a, **k):
        result = full(*a, **k)
        clock[0] += 7.
        return result
    monkeypatch.setattr(_dictionary, '_full_model_identity', hashed)
    result = runtime.prepare()
    assert result['seconds'] == 7.
    assert json.loads((runtime.directory / 'preparation.json').read_text())['seconds'] == 7.


def test_inference_tensors_and_tied_registrations_use_all_source_bytes(grouped, monkeypatch):
    model, _, _, service = grouped
    with torch.inference_mode():
        model.embed_tokens.weight = torch.nn.Parameter(model.embed_tokens.weight.clone())
        model.register_parameter('tied_embedding', model.embed_tokens.weight)
        model.register_buffer('transient', torch.zeros(3), persistent=False)
        model.register_buffer('transient_alias', model.transient, persistent=False)
        model.norm.frozen_scale = torch.ones(1, 3, 1, dtype=torch.float64)
        source_size = sum(len(raw(value)) for value in
                          [*model.parameters(), *model.buffers(), model.norm.frozen_scale])
        snapshot = snapshot_for(service)
        owner = bind(snapshot)
        expected = _dictionary._full_model_identity(model)
        with pytest.raises(RuntimeError, match='version counter'):
            _ = model.embed_tokens.weight._version
        calls, compared = hashes(monkeypatch), []
        def compare(left, right):
            compared.append(left.numel())
            return raw(left) == raw(right)
        monkeypatch.setattr(_intervene, '_cpu_byte_equal', compare)
        assert _dictionary.model_identity(model) == expected and not calls
        assert sum(compared) == source_size
        model.tied_embedding.data[-1, -1] += .125
        compared.clear()
        assert _dictionary.model_identity(model) != expected and calls
        assert sum(compared) == snapshot.nbytes
        assert owner._model_snapshot._public is None


def test_changed_first_block_still_compares_remaining_bytes(published, monkeypatch):
    model, _, _, _, runtime = published
    model.embed_tokens.weight.data[0, 0] += .125
    sizes = []
    def compare(left, right):
        sizes.append(left.numel())
        return raw(left) == raw(right)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', compare)
    _dictionary.model_identity(model)
    assert sum(sizes) == runtime._model_snapshot.nbytes
    assert runtime._model_snapshot._public is None


@pytest.mark.parametrize('phase', ['hash', 'comparison'])
def test_bind_rechecks_owners_after_callbacks(grouped, monkeypatch, phase):
    _, _, _, service = grouped
    snapshot = snapshot_for(service)
    owner, proxy = Owner(), Owner()
    proxy.executor = service.executor
    owner.model, owner.tokenizer, owner.interventions = snapshot.model, snapshot.tokenizer, proxy
    owner._model_snapshot, owner.guard, owner.preparation = snapshot, None, None
    if phase == 'hash':
        full = _dictionary._full_model_identity
        def changed(*a, **k):
            result = full(*a, **k)
            owner._model_snapshot = None
            return result
        monkeypatch.setattr(_dictionary, '_full_model_identity', changed)
    else:
        def changed(left, right):
            owner._model_snapshot = None
            return raw(left) == raw(right)
        monkeypatch.setattr(_intervene, '_cpu_byte_equal', changed)
    with pytest.raises(ValueError, match='ownership changed'):
        snapshot.bind_public_identity(owner)
    assert snapshot._public is None
    assert _dictionary._published_snapshot(snapshot.model) is None


def test_real_trace_deadline_cancels_snapshot_comparison(published, monkeypatch):
    from caliper.interp import _trace
    model, _, _, _, runtime = published
    clock, sizes = [10.], []
    monkeypatch.setattr(_trace.time, 'monotonic', lambda: clock[0])
    deadline = _trace._Deadline()
    def compare(left, right):
        sizes.append(left.numel())
        clock[0] = deadline.expires
        return raw(left) == raw(right)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', compare)
    monkeypatch.setattr(_intervene, '_SNAPSHOT_COMPARE_BYTES', 1024)
    with pytest.raises(_trace.TraceTimeout, match='600 second safety timeout'):
        _dictionary.model_identity(model, memory=IdentityGuard(), check=deadline)
    assert sizes == [1024] and runtime._model_snapshot._public is None


@pytest.mark.parametrize('change', ['owner', 'lower_limit'])
def test_guard_callback_cannot_leave_stale_authority(published, monkeypatch, change):
    model, _, _, _, runtime = published
    snapshot = runtime._model_snapshot
    guard, stages = IdentityGuard(), []
    def check(stage):
        stages.append(stage)
        if change == 'owner':
            runtime._model_snapshot = None
        else:
            guard.limit = 1
    monkeypatch.setattr(guard, 'check', check)
    monkeypatch.setattr(_intervene, '_cpu_byte_equal', lambda *a: pytest.fail('bytes after guard mutation'))
    try:
        if change == 'owner':
            calls = hashes(monkeypatch)
            _dictionary.model_identity(model, memory=guard)
            assert calls
        else:
            with pytest.raises(CleanStop):
                _dictionary.model_identity(model, memory=guard)
        assert stages and snapshot._public is None
    finally:
        runtime._model_snapshot = snapshot


def test_unverified_hf_snapshot_retains_full_hash(reference_snapshot, monkeypatch):
    model, _, guard, snapshot = reference_snapshot
    expected = _dictionary._full_model_identity(model, memory=guard)
    owner = bind(snapshot, guard)
    calls = hashes(monkeypatch)
    assert _dictionary.model_identity(model, memory=guard) == expected and calls
    assert owner._model_snapshot._public is None
    assert model not in _dictionary._PUBLISHED_MODELS


def test_missing_registry_entry_uses_full_hash(published, monkeypatch):
    model, _, _, _, _ = published
    monkeypatch.delitem(_dictionary._PUBLISHED_MODELS, model)
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls


def test_ordinary_unhashable_cpu_module_keeps_content_identity(monkeypatch):
    class Unhashable(torch.nn.Linear):
        __hash__ = None
    model = Unhashable(3, 2)
    expected = _dictionary._full_model_identity(model)
    calls = hashes(monkeypatch)
    assert _dictionary.model_identity(model) == expected and calls


@pytest.mark.parametrize('mode', ['method', 'code'])
def test_global_serializer_changes_cannot_authorize_reuse(published, monkeypatch, mode):
    model, _, _, _, runtime = published
    original = torch.nn.Module._save_to_state_dict
    if mode == 'method':
        def wrapped(*args):
            return original(*args)
        monkeypatch.setattr(torch.nn.Module, '_save_to_state_dict', wrapped)
    else:
        def changed(self, destination, prefix, keep_vars):
            for name, value in self._parameters.items():
                if value is not None:
                    destination[prefix + name] = value if keep_vars else value.detach()
            for name, value in self._buffers.items():
                if value is not None and name not in self._non_persistent_buffers_set:
                    destination[prefix + name] = value if keep_vars else value.detach()
        monkeypatch.setattr(original, '__code__', changed.__code__)
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls and runtime._model_snapshot._public is None


@pytest.mark.parametrize('phase', ['binding', 'lookup'])
def test_tensor_subclasses_keep_full_hash(grouped, monkeypatch, phase):
    class TensorSubclass(torch.Tensor):
        pass
    model, _, _, service = grouped
    if phase == 'binding':
        model.register_buffer('custom', torch.ones(3).as_subclass(TensorSubclass))
    snapshot = snapshot_for(service)
    owner = bind(snapshot)
    if phase == 'lookup':
        model.register_buffer('custom', torch.ones(3).as_subclass(TensorSubclass))
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls and owner._model_snapshot._public is None


def test_rejected_proof_keeps_preparation_guard_on_full_hash(trusted, monkeypatch):
    model, _, runtime, _, snapshot, _ = trusted
    original = model.state_dict
    with monkeypatch.context() as patch:
        patch.setattr(model, 'state_dict', lambda **kwargs: original(**kwargs))
        _dictionary.model_identity(model)
    assert snapshot._public is None
    runtime.limit = 1
    with pytest.raises(CleanStop):
        _dictionary.model_identity(model)


def test_new_proof_replaces_old_authority_without_old_release_damage(grouped, monkeypatch):
    model, _, _, service = grouped
    first, second = snapshot_for(service), snapshot_for(service)
    owner1, owner2 = bind(first), bind(second)
    assert first._public is None and _dictionary._published_snapshot(model) is second
    first.release()
    assert owner1._model_snapshot is first and owner2._model_snapshot is second
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls == []


@pytest.mark.parametrize('exception', [ValueError, KeyError, TypeError])
def test_guard_exceptions_never_select_fallback(published, monkeypatch, exception):
    model, _, _, _, runtime = published
    error = exception('guard refusal')
    guard, stages = IdentityGuard(), []
    def stopped(stage):
        stages.append(stage)
        if len(stages) == 1:
            raise error
    monkeypatch.setattr(guard, 'check', stopped)
    calls = hashes(monkeypatch)
    with pytest.raises(exception) as caught:
        _dictionary.model_identity(model, memory=guard)
    assert caught.value is error and len(stages) == 1
    assert not calls and runtime._model_snapshot._public is None


def test_tensor_setting_subclass_uses_full_hash(grouped, monkeypatch):
    class TensorSubclass(torch.Tensor):
        pass
    model, _, _, service = grouped
    model.norm.frozen_scale = torch.ones(1, 3, 1, dtype=torch.float64).as_subclass(TensorSubclass)
    snapshot = snapshot_for(service)
    owner = bind(snapshot)
    calls = hashes(monkeypatch)
    _dictionary.model_identity(model)
    assert calls and owner._model_snapshot._public is None
