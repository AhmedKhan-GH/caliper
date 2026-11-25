"""Bounded CPU checks for atomic published feature groups."""
from copy import deepcopy

import pytest
import torch

from caliper._controls import Controls, intervention_key
from caliper.interp._intervene import InterventionError
from tests.interp.fixture import make_fixture
from tests.interp.test_intervene import dictionary_for


def clamp(feature, position, value=1., layer=0):
    return {"kind": "clamp", "point": f"layers.{layer}.mlp_out", "feature": feature,
            "positions": [position], "value": value}


@pytest.fixture
def grouped():
    model, tokenizer = make_fixture()
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
    yield model, tokenizer, controls, service
    service.remove()


def test_positioned_identity_preserves_legacy_scope():
    assert intervention_key(clamp(12274, 9)) != intervention_key(clamp(12274, 10))
    assert intervention_key(clamp(4257, 9)) != intervention_key(clamp(11570, 9))
    assert intervention_key(dict(clamp(1, 2), positions=[2, 1])) == intervention_key(dict(clamp(1, 2), positions=[1, 2]))
    spec = clamp(1, 2)
    del spec['positions']
    assert intervention_key(spec) == ('layers.0.mlp_out', 'clamp')


def test_atomic_collision_group_restore_replay_fork_and_clear(grouped):
    model, tokenizer, ctl, service = grouped
    specs = [clamp(3, 1), clamp(2, 1, -1), clamp(1, 1, 2), clamp(3, 2, -2)]
    ids = tokenizer('AB')['input_ids']
    baseline = model(ids).logits
    service.replace_atomic(specs, lambda: model(ids).logits)
    saved = service.state_dict()
    expected = model(ids).logits
    assert len(ctl._ivs) == 4 and len(service.executor._feature_groups) == 1
    assert not torch.equal(baseline, expected)
    with service.temporary_state(saved):
        torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)
    service.load_state_dict(saved, restore_active=True)
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)
    service.apply_overrides([clamp(3, 2, 5)])
    assert len(ctl._ivs) == 4
    assert service.clear_feature('layers.0.mlp_out', 3, positions=[1])
    ctl.step()
    assert len(ctl._ivs) == 3
    service.clear('layers.0.mlp_out')
    ctl.step()
    assert not service.specifications() and not service.executor._feature_groups
    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)


@pytest.mark.parametrize('phase', ['validate', 'install', 'forward', 'evaluate', 'conflict'])
def test_atomic_failure_retains_exact_prior_records(grouped, monkeypatch, phase):
    model, tokenizer, ctl, service = grouped
    ids = tokenizer('AB')['input_ids']
    service.replace_atomic([clamp(3, 1)], lambda: model(ids))
    saved = service.state_dict()
    original = dict(ctl._ivs)
    output = model(ids).logits
    specs = [clamp(1, 1), clamp(2, 2)]
    if phase == 'validate':
        specs[-1]['feature'] = 999999
    if phase == 'forward':
        specs[-1]['positions'] = [31]
    install = service.executor.install
    if phase == 'install':
        def failed(spec, **kwargs):
            if spec['feature'] == 2:
                raise RuntimeError('install refusal')
            return install(spec, **kwargs)
        monkeypatch.setattr(service.executor, 'install', failed)
    def evaluate():
        result = model(ids)
        if phase == 'evaluate':
            raise RuntimeError('output refusal')
        if phase == 'conflict':
            service.set({'kind': 'ablate', 'point': 'layers.1.mlp_out'})
        return result
    with pytest.raises((InterventionError, RuntimeError)):
        service.replace_atomic(specs, evaluate)
    for key, record in original.items():
        assert ctl._ivs[key] is record
    if phase != 'conflict':
        assert service.state_dict() == saved
    torch.testing.assert_close(model(ids).logits, output, rtol=0, atol=0)


def test_atomic_group_shares_one_complete_dictionary_check(grouped, monkeypatch):
    model, tokenizer, ctl, service = grouped
    executor = service.executor
    executor._prepare_dictionary_identity()
    before = executor._dictionary_status()['checks']
    specs = [clamp(1, 1), clamp(2, 1), clamp(3, 1)]
    service.replace_atomic(specs, lambda: model(**tokenizer('AB')))
    assert executor._dictionary_status()['checks'] == before + 1
    saved = service.state_dict()
    executor.dictionary.layers[0].W_dec.data[0, 0] += 1
    with pytest.raises(InterventionError, match='content changed'):
        service.replace_atomic(specs, lambda: pytest.fail('forward before byte verification'))
    assert service.state_dict() == saved


def test_bfloat16_group_uses_sorted_decoder_accumulation(grouped, monkeypatch):
    model, _, _, service = grouped
    specs = [clamp(3, 1), clamp(1, 1), clamp(2, 1), clamp(3, 2)]
    service.replace_atomic(specs, lambda: None)
    group = service.executor._feature_groups['layers.0.mlp_out']
    weights = service.dictionary.layers[0]
    weights.to(torch.bfloat16)
    weights.W_dec.zero_()
    weights.W_dec[1, 0], weights.W_dec[2, 0], weights.W_dec[3, 0] = 0.003, 0.003, 0.003
    activation = torch.zeros(1, 3, weights.W_dec.shape[0], dtype=torch.bfloat16)
    calls = []
    def encode(*args):
        calls.append(1)
        return activation
    monkeypatch.setattr('caliper.interp._intervene._trace_features', encode)
    value = torch.ones(1, 3, 64, dtype=torch.bfloat16)
    result = group._apply_members(value, value, group.members)
    delta = torch.zeros_like(value)
    delta[0].index_add_(0, torch.tensor([1, 1, 1, 2]), weights.W_dec[torch.tensor([1, 2, 3, 3])])
    expected = value + delta
    sequential = value.clone()
    sequential[0, 1] += weights.W_dec[3]
    sequential[0, 1] += weights.W_dec[1]
    sequential[0, 1] += weights.W_dec[2]
    assert not torch.equal(expected[0, 1], sequential[0, 1])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert calls == [1]


@pytest.fixture
def published(grouped, monkeypatch, tmp_path):
    from caliper.interp import _replication_actions as actions
    from tests.interp.test_intervene import explicit_forward
    model, tokenizer, ctl, service = grouped
    _, acts = explicit_forward(model, tokenizer('AB')['input_ids'])
    features = acts[0][0, 1].topk(3).indices.tolist()
    groups = {'Texas': [(0, features[0], 1), (0, features[1], 1)],
              'China': [(0, features[2], 1), (1, 3, 1), (1, 3, 2)]}
    prompts = {'dallas': 'AB', 'oakland': 'CD', 'shanghai': 'EF', 'vancouver': 'GH'}
    cases = {10: {'label': 'Baseline', 'operations': []},
             26: {'label': 'Replace Texas With China', 'operations': [
                 {'group': 'Texas', 'factor': -2, 'activation_prompt': 'dallas'},
                 {'group': 'China', 'factor': 2, 'activation_prompt': 'shanghai'}]}}
    monkeypatch.setattr(actions, 'GROUPS', groups)
    monkeypatch.setattr(actions, 'PROMPTS', prompts)
    monkeypatch.setattr(actions, 'CASES', cases)
    runtime = actions.PublishedCases(model, tokenizer, service, output_directory=tmp_path)
    runtime.prepare()
    yield model, tokenizer, ctl, service, runtime
    runtime.close()


def test_published_values_come_from_independent_caliper_prompts(published):
    from tests.interp.test_intervene import explicit_forward
    from caliper.interp._replication_actions import GROUPS, PROMPTS
    model, tokenizer, _, service, runtime = published
    service.replace_atomic([clamp(7, 1, 12)], lambda: None)
    with service.suspend():
        _, baseline = explicit_forward(model, tokenizer(PROMPTS['dallas'])['input_ids'])
        _, donor = explicit_forward(model, tokenizer(PROMPTS['shanghai'])['input_ids'])
    specs, members = runtime.case(26)
    for spec, member in zip(specs, members):
        acts = baseline if member['activation_prompt'] == 'dallas' else donor
        expected = float(acts[member['layer']][0, member['position'], member['feature']].detach())
        assert member['activation'] == pytest.approx(expected, abs=1e-12)
        assert spec['value'] == member['factor'] * member['activation']
    assert len({intervention_key(s) for s in specs}) == 5
    assert runtime.preparation['producer'] == 'caliper'
    assert runtime.preparation['freeze_norms'] is False


def test_published_actions_export_all_probabilities_and_repeat_real_forward(published):
    import json
    import numpy as np
    from caliper.interp._output import forward_logits
    model, tokenizer, _, service, runtime = published
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    try:
        first = runtime.run(26)
        second = runtime.run(26)
        assert calls == [1, 1]
        assert first['evidence']['file'] != second['evidence']['file']
        assert first['argmax_token_ids'] == second['argmax_token_ids']
        archive = runtime.output_directory / second['evidence']['file']
        with np.load(archive, allow_pickle=False) as arrays:
            probabilities = arrays['probabilities']
            logits = arrays['logits']
        assert probabilities.shape == (2, model.config.vocab_size)
        assert second['argmax_token_ids'] == probabilities.argmax(-1).tolist()
        assert len(json.dumps(runtime.status('AB', service.specifications()))) < 32768
        assert 'probabilities' not in second
        saved = service.state_dict()
        direct = forward_logits(model, tokenizer('AB'))[0, -1]
        torch.testing.assert_close(direct.float(), torch.from_numpy(logits[1]), rtol=0, atol=0)
        service.load_state_dict(saved, restore_active=True)
        with service.temporary_state(saved):
            replay = forward_logits(model, tokenizer('AB'))[0, -1]
        torch.testing.assert_close(replay, direct, rtol=0, atol=0)
        runtime.run(10)
        assert not service.specifications()
    finally:
        handle.remove()


@pytest.mark.parametrize('phase', ['forward', 'export', 'publication', 'metadata', 'identity'])
def test_published_failure_has_no_success_or_partial_case(published, monkeypatch, phase):
    from caliper.interp import _replication_actions as actions
    model, _, ctl, service, runtime = published
    first = runtime.run(26)
    saved, records = service.state_dict(), dict(ctl._ivs)
    files = set(runtime.directory.iterdir())
    if phase == 'forward':
        monkeypatch.setattr(actions, 'forward_logits', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('forward refusal')))
    if phase == 'export':
        monkeypatch.setattr('numpy.savez', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('export refusal')))
    if phase == 'metadata':
        monkeypatch.setattr(runtime, '_write_json', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('metadata refusal')))
    if phase == 'identity':
        model.layers[0].mlp.W_enc.data[0, 0] += .1
    def publish(logits):
        if phase == 'publication':
            raise RuntimeError('publication refusal')
    with pytest.raises((ValueError, RuntimeError)):
        runtime.run(10, publish=publish)
    assert service.state_dict() == saved
    assert all(ctl._ivs[key] is record for key, record in records.items())
    assert runtime.last_result == first
    assert runtime.last_attempt['status'] == 'error'
    assert set(runtime.directory.iterdir()) == files


def test_published_china_contract_has_thirteen_distinct_members():
    from types import SimpleNamespace
    from caliper.interp._replication_actions import PublishedCases, GROUPS, CASES
    runtime = PublishedCases.__new__(PublishedCases)
    runtime.preparation = {'id': 'fixture'}
    runtime.interventions = SimpleNamespace(executor=SimpleNamespace(dictionary_id='fixture', dictionary=SimpleNamespace(
        metadata={'write_points': [f'model.layers.{i}.mlp_out' for i in range(26)]})))
    runtime.values = {source: {node: 1. for group in GROUPS.values() for node in group}
                      for source in ('dallas', 'shanghai')}
    specs, members = runtime.case(26)
    assert len(specs) == len({intervention_key(s) for s in specs}) == 13
    assert {(m['feature'], m['position']) for m in members if m['layer'] == 4} == {(4257, 9), (11570, 9), (13154, 9)}
    assert {(m['feature'], m['position']) for m in members if m['layer'] == 19} == {(7477, 9), (12274, 9), (12274, 10)}
