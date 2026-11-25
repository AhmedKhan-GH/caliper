"""CPU checks for dictionary identity at the atomic commit boundary."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _publish
from caliper.interp._intervene import InterventionError
from caliper.interp._memory import CleanStop
from caliper.interp._published_identity import forward_hooks
from tests.interp.test_replication_groups import clamp, grouped, published


def patch(prompt):
    return {'kind': 'patch', 'point': 'layers.2.mlp_out', 'positions': [1],
            'source_prompt': prompt}


@pytest.fixture(params=['generic', 'published'])
def transaction(request, monkeypatch):
    grouped_values = request.getfixturevalue('grouped')
    if request.node.callspec.params.get('mode', 'snapshot') == 'snapshot':
        # The published preparation pins this snapshot object as part of ownership.
        grouped_values[3].executor._prepare_dictionary_identity()
    values = request.getfixturevalue('published' if request.param == 'published' else 'grouped')
    model, tokenizer, controls, service = values[:4]
    runtime = values[4] if request.param == 'published' else None
    monkeypatch.setattr(_publish, '_TENSORS', {})
    monkeypatch.setattr(_publish, '_VERSIONS', {})
    monkeypatch.setattr(_publish, '_VERSION', 0)

    def prepare(logits):
        return _publish._prepare_many({'output.logits': logits})

    def evaluate():
        return model(**tokenizer('AB')).logits

    if runtime is not None:
        runtime.run(26, prepare=prepare)
    # Retain an actual patch source and its hook before the candidate transaction.
    service.replace_atomic([*service.specifications(), clamp(7, 1), patch('CD')], evaluate, prepare=prepare)

    def run(*, prepare=prepare, check=None):
        if runtime is not None:
            return runtime.run(26, prepare=prepare, check=check)
        return service.replace_atomic([clamp(1, 1), clamp(2, 1, layer=1), patch('EF')],
                                      evaluate, prepare=prepare, check=check)

    return SimpleNamespace(model=model, service=service, executor=service.executor,
                           controls=controls, runtime=runtime, run=run, prepare=prepare)


def saved_state(case):
    executor = case.executor
    return SimpleNamespace(
        records=dict(case.controls._ivs), state=case.service.state_dict(),
        revision=case.controls._iv_revision, hooks=forward_hooks(case.model),
        groups={point: tuple(group.members) for point, group in executor._feature_groups.items()},
        cache=list(executor._cache.items()), cache_identity=executor._cache_identity,
        tensors=dict(_publish._TENSORS), versions=dict(_publish._VERSIONS), version=_publish._VERSION,
        values={name: tensor.tolist() for name, tensor in _publish._TENSORS.items()},
        result=None if case.runtime is None else deepcopy(case.runtime.last_result),
        files=None if case.runtime is None else set(case.runtime.directory.iterdir()))


def assert_preserved(case, saved):
    assert case.service.state_dict() == saved.state
    assert case.controls._iv_revision == saved.revision
    assert case.controls._ivs.keys() == saved.records.keys()
    assert all(case.controls._ivs[key] is record for key, record in saved.records.items())
    assert forward_hooks(case.model) == saved.hooks
    assert {point: tuple(group.members) for point, group in case.executor._feature_groups.items()} == saved.groups
    assert list(case.executor._cache) == [key for key, _ in saved.cache]
    assert all(case.executor._cache[key] is value for key, value in saved.cache)
    assert case.executor._cache_identity is saved.cache_identity
    assert _publish._TENSORS.keys() == saved.tensors.keys()
    assert _publish._VERSIONS == saved.versions and _publish._VERSION == saved.version
    for name, tensor in saved.tensors.items():
        assert _publish._resolve(name) is tensor
        assert tensor.tolist() == saved.values[name]
    if case.runtime is not None:
        assert case.runtime.last_result == saved.result
        assert case.runtime.last_attempt['status'] == 'error'
        assert set(case.runtime.directory.iterdir()) == saved.files


def change_dictionary(executor, change):
    if change == 'content':
        decoder = executor.dictionary.layers[0].W_dec
        version = decoder._version
        decoder.data[0, 0] += .125
        assert decoder._version == version
    elif change == 'metadata':
        executor.dictionary.metadata['revision'] = 'changed'
    elif change == 'topk':
        executor.dictionary.layers[0].k -= 1
    else:
        raise AssertionError(change)


@pytest.mark.parametrize('mode', ['hash', 'snapshot'])
@pytest.mark.parametrize('phase', ['before', 'forward', 'prepare', 'check'])
@pytest.mark.parametrize('change', ['content', 'metadata', 'topk'])
def test_dictionary_change_refuses_before_commit(transaction, mode, phase, change):
    case = transaction
    executor = case.executor
    assert executor._dictionary_status()['mode'] == mode
    calls, handle = [], None

    def mutate():
        change_dictionary(executor, change)
        calls.append(phase)

    def forward(_model, _args):
        if executor.preparing_source:
            return
        if phase == 'before':
            pytest.fail('candidate forward preceded dictionary admission')
        if phase == 'forward':
            mutate()

    handle = case.model.register_forward_pre_hook(forward)
    saved = saved_state(case)
    before = executor._dictionary_status()['checks']

    def prepare(logits):
        if phase == 'prepare':
            mutate()
        if case.runtime is not None:
            assert set(case.runtime.directory.iterdir()) > saved.files
        assert all(_publish._resolve(name) is tensor for name, tensor in saved.tensors.items())
        return case.prepare(logits)

    try:
        if phase == 'before':
            mutate()
        with pytest.raises(InterventionError, match='dictionary content changed') as error:
            case.run(prepare=prepare, check=mutate if phase == 'check' else None)
        assert error.value.code == 'intervene.unknown_dictionary'
        assert calls == [phase]
        assert executor._dictionary_status()['checks'] == before + (1 if phase == 'before' else 2)
        assert not executor._dictionary_admitted()
        assert_preserved(case, saved)
        # Refusal preserves external edits and the next admission still rejects them.
        assert not executor._dictionary_matches()
    finally:
        handle.remove()


@pytest.mark.parametrize('mode', ['hash', 'snapshot'])
@pytest.mark.parametrize('publication', [False, True])
def test_final_scan_follows_callbacks_and_commits_unchanged(transaction, monkeypatch, mode, publication):
    case = transaction
    assert case.executor._dictionary_status()['mode'] == mode
    events = []
    matches = case.executor._dictionary_matches

    def verify():
        events.append('scan')
        return matches()

    monkeypatch.setattr(case.executor, '_dictionary_matches', verify)

    def prepare(logits):
        events.append('prepare')
        return case.prepare(logits)

    def check():
        events.append('check')

    result = case.run(prepare=prepare if publication else None, check=check)
    assert events == ['scan', *(['prepare'] if publication else []), 'check', 'scan']
    assert not case.executor._dictionary_admitted()
    assert len(case.service.specifications()) == (5 if case.runtime is not None else 3)
    if case.runtime is not None:
        assert case.runtime.last_result == result
        assert result['action_seconds'] >= result['compute_export_seconds']


@pytest.mark.parametrize('publication', [False, True])
def test_final_scan_resource_refusal_preserves_prior_state(transaction, monkeypatch, publication):
    case = transaction
    assert case.executor._dictionary_status()['mode'] == 'snapshot'
    saved = saved_state(case)
    stages = []
    guard = SimpleNamespace(limit=2_000_000_000, current=0, peak=0, device=torch.device('cpu'))

    def guard_check(stage):
        stages.append(stage)
        return 0

    guard.check = guard_check
    case.executor.resource_guard = guard

    def check():
        stages.clear()
        guard.limit = 0
        # A comparison must refuse its workspace before it reads tensor bytes.
        monkeypatch.setattr(torch, 'equal', lambda *a, **k: pytest.fail('byte read after resource refusal'))

    with pytest.raises(CleanStop, match='process memory limit'):
        case.run(prepare=case.prepare if publication else None, check=check)
    assert stages and set(stages) == {'dictionary identity'}
    assert not case.executor._dictionary_admitted()
    assert_preserved(case, saved)


@pytest.mark.parametrize('change', ['content', 'metadata'])
@pytest.mark.parametrize('publication', [False, True])
def test_explicit_identifier_retains_caller_ownership(grouped, change, publication):
    model, tokenizer, _, service = grouped
    service.configure(dictionary=service.dictionary, tokenizer=tokenizer, dictionary_id='caller-owned')
    before = service.executor._dictionary_status()['checks']
    prepare = (lambda logits: _publish._prepare_many({})) if publication else None
    service.replace_atomic([clamp(1, 1)], lambda: model(**tokenizer('AB')).logits, prepare=prepare,
                           check=lambda: change_dictionary(service.executor, change))
    assert service.executor.dictionary_id == 'caller-owned'
    assert service.executor._dictionary_status()['checks'] == before
    assert len(service.specifications()) == 1


@pytest.mark.parametrize('change', [None, 'content', 'metadata'])
def test_nested_admission_keeps_final_commit_check(grouped, change):
    model, tokenizer, controls, service = grouped
    executor = service.executor
    executor._prepare_dictionary_identity()
    before = executor._dictionary_status()['checks']
    with executor.dictionary_admission():
        with executor.dictionary_admission():
            assert executor._dictionary_status()['checks'] == before + 1
            service.replace_atomic([clamp(1, 1)], lambda: model(**tokenizer('AB')))
            assert executor._dictionary_status()['checks'] == before + 2
            state, records = service.state_dict(), dict(controls._ivs)
            if change is not None:
                with pytest.raises(InterventionError, match='dictionary content changed'):
                    service.replace_atomic([clamp(2, 1)], lambda: model(**tokenizer('AB')),
                                           check=lambda: change_dictionary(executor, change))
                assert service.state_dict() == state
                assert all(controls._ivs[key] is record for key, record in records.items())
                assert executor._dictionary_status()['checks'] == before + 3
            assert executor._dictionary_admitted()
        assert executor._dictionary_admitted()
    assert not executor._dictionary_admitted()
    if change is not None:
        with pytest.raises(InterventionError, match='dictionary content changed'):
            with executor.dictionary_admission():
                pytest.fail('refused admission yielded')
        assert not executor._dictionary_admitted()


@pytest.mark.parametrize('attribute,value', [('_default_dictionary_id', False),
                                           ('dictionary_id', 'caller-owned'),
                                           ('_dictionary_snapshot', None)])
def test_commit_rejects_changed_admission_owner(transaction, attribute, value):
    case = transaction
    assert case.executor._dictionary_status()['mode'] == 'snapshot'
    saved = saved_state(case)
    with pytest.raises(InterventionError, match='dictionary content changed'):
        case.run(check=lambda: setattr(case.executor, attribute, value))
    assert_preserved(case, saved)


def test_atomic_quality_metadata_retains_identity_contract(grouped):
    model, tokenizer, _, service = grouped
    service.executor._prepare_dictionary_identity()
    service.replace_atomic([clamp(1, 1)], lambda: model(**tokenizer('AB')),
                           check=lambda: service.dictionary.metadata.update(quality={'fvu': .25}))
    assert len(service.specifications()) == 1


@pytest.mark.parametrize('change', ['content', 'metadata'])
def test_final_callback_without_publication_refuses(transaction, change):
    saved = saved_state(transaction)
    with pytest.raises(InterventionError, match='dictionary content changed'):
        transaction.run(prepare=None, check=lambda: change_dictionary(transaction.executor, change))
    assert_preserved(transaction, saved)


def test_final_snapshot_scan_reuses_all_private_bytes(transaction, monkeypatch):
    case = transaction
    snapshot = case.executor._dictionary_snapshot
    expected = sum(value.numel() * value.element_size() for value in snapshot.state.values())
    sizes, equal = [], torch.equal

    def compare(left, right):
        assert left.dtype == right.dtype == torch.uint8
        sizes.append(left.numel())
        return equal(left, right)

    def forbidden(*args, **kwargs):
        pytest.fail('final snapshot check copied or hashed tensor storage')

    def check():
        monkeypatch.setattr(torch, 'equal', compare)
        monkeypatch.setattr(torch.Tensor, 'clone', forbidden)
        monkeypatch.setattr(torch.Tensor, 'to', forbidden)
        monkeypatch.setattr(torch.Tensor, 'cpu', forbidden)
        monkeypatch.setattr('caliper.interp._intervene._content_identity', forbidden)

    case.run(check=check)
    assert sum(sizes) == expected
    assert case.executor._dictionary_snapshot is snapshot


def test_nested_admission_cannot_commit_a_change_before_inner_entry(grouped):
    model, tokenizer, controls, service = grouped
    executor = service.executor
    executor._prepare_dictionary_identity()
    with executor.dictionary_admission():
        change_dictionary(executor, 'content')
        with pytest.raises(InterventionError, match='dictionary content changed'):
            service.replace_atomic([clamp(1, 1)], lambda: model(**tokenizer('AB')))
        assert not controls._ivs and not executor._feature_groups
        assert executor._dictionary_admitted()
    assert not executor._dictionary_admitted()
