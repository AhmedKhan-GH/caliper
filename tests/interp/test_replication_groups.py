"""Bounded CPU checks for atomic published feature groups."""
from copy import deepcopy
import warnings

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


def test_ordinary_group_position_failure_ends_members_and_reports_once(grouped):
    model, tokenizer, ctl, service = grouped
    ids = tokenizer('A')['input_ids']
    longer = tokenizer('AB')['input_ids']
    observer = model.layers[0].mlp.register_forward_hook(lambda _module, _args, output: output + .01)
    try:
        baseline = model(ids).logits
        for feature in (1, 2):
            service.set(clamp(feature, 2, 0))
        ctl.step()
        hooks = [record.hook for record in ctl._ivs.values()]
        group = service.executor._feature_groups['layers.0.mlp_out']
        assert len(group.members) == 2
        model(longer)
        torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
        entries = ctl.controls_meta()['interventions']['active']
        assert len(entries) == 2
        assert all(entry['state'] == 'ended' and entry['error_code'] == 'intervene.hook_failed'
                   and 'selected position is outside the live tensor' in entry['error'] for entry in entries)
        assert all(hook.calls == 2 and hook.errors == 1 and hook.removed for hook in hooks)
        assert not group.handles and not group.members and not service.executor._feature_groups
        assert not service.specifications()
        assert observer.id in model.layers[0].mlp._forward_hooks
        torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
        assert all(hook.calls == 2 for hook in hooks)
        with warnings.catch_warnings(record=True) as reports:
            warnings.simplefilter('always')
            ctl.step()
            assert len(reports) == 2  # Each failed intervention reports once at this boundary.
            assert all(item.category is RuntimeWarning and 'failed and ended' in str(item.message)
                       for item in reports)
            ctl.step()
            assert len(reports) == 2
        assert all(not record.report for record in ctl._ivs.values())
    finally:
        observer.remove()


@pytest.mark.parametrize('member_count', [1, 2])
def test_group_clean_stop_propagates_after_member_cleanup(grouped, monkeypatch, member_count):
    from caliper.interp._memory import CleanStop
    model, tokenizer, ctl, service = grouped
    for feature in range(member_count):
        service.set(clamp(feature, 1))
    ctl.step()
    hooks = [record.hook for record in ctl._ivs.values()]
    stop = CleanStop('group resource limit')

    def refuse(*args):
        raise stop

    monkeypatch.setattr('caliper.interp._intervene._trace_features', refuse)
    with pytest.raises(CleanStop) as error:
        model(**tokenizer('AB'))
    assert error.value is stop
    assert all(hook.errors == 1 and hook.removed for hook in hooks)
    assert not service.specifications() and not service.executor._feature_groups
    assert all(record.state == 'ended' and record.error_code == 'intervene.hook_failed'
               for record in ctl._ivs.values())


def test_group_failure_preserves_muted_member_at_the_same_point(grouped):
    from caliper.interp._intervene import _mute_hooks
    model, tokenizer, _, service = grouped
    ids = tokenizer('AB')['input_ids']
    baseline = model(ids).logits
    prior = service.executor.install(clamp(3, 1, 2))
    expected = model(ids).logits
    assert not torch.equal(expected, baseline)
    group = prior.group
    failed = [service.executor.install(clamp(feature, 31)) for feature in (1, 2)]
    with _mute_hooks([prior]):
        torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
    assert all(hook.removed and hook.errors == 1 for hook in failed)
    assert not prior.removed and prior.errors == 0 and prior.calls == 1
    assert group.members == [prior]
    assert service.executor._feature_groups['layers.0.mlp_out'] is group
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)


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


def test_atomic_group_shares_admission_and_final_dictionary_checks(grouped, monkeypatch):
    model, tokenizer, ctl, service = grouped
    executor = service.executor
    executor._prepare_dictionary_identity()
    before = executor._dictionary_status()['checks']
    specs = [clamp(1, 1), clamp(2, 1), clamp(3, 1), clamp(4, 1, layer=1)]
    service.replace_atomic(specs, lambda: model(**tokenizer('AB')))
    assert executor._dictionary_status()['checks'] == before + 2
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


def test_published_dictionary_mutation_in_persistent_pre_hook_refuses(published):
    from caliper.interp._replication_actions import GROUPS
    model, _, ctl, service, runtime = published
    first = runtime.run(26)
    saved, records = service.state_dict(), dict(ctl._ivs)
    files = set(runtime.directory.iterdir())
    layer, feature, position = GROUPS['Texas'][0]
    assert runtime.values['dallas'][(layer, feature, position)] > 0
    decoder = service.dictionary.layers[layer].W_dec
    version = decoder._version
    calls = []

    def mutate(_model, _args):
        decoder.data[feature, 0] += .1
        calls.append(1)
        assert decoder._version == version
        assert runtime._model_matches()

    handle = model.register_forward_pre_hook(mutate)
    try:
        with pytest.raises(InterventionError, match='dictionary content changed'):
            runtime.run(26)
        assert calls == [1]
        assert service.state_dict() == saved
        assert all(ctl._ivs[key] is record for key, record in records.items())
        assert runtime.last_result == first
        assert runtime.last_attempt['status'] == 'error'
        assert set(runtime.directory.iterdir()) == files
    finally:
        handle.remove()


@pytest.mark.parametrize('phase', ['forward', 'export', 'publication', 'metadata', 'written_metadata', 'identity'])
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
    if phase == 'written_metadata':
        write_json = runtime._write_json
        def fail_after_write(*args):
            write_json(*args)
            raise RuntimeError('metadata completion refusal')
        monkeypatch.setattr(runtime, '_write_json', fail_after_write)
    if phase == 'identity':
        model.layers[0].mlp.W_enc.data[0, 0] += .1
    def prepare(logits):
        if phase == 'publication':
            raise RuntimeError('publication refusal')
    with pytest.raises((ValueError, RuntimeError)):
        runtime.run(10, prepare=prepare)
    assert service.state_dict() == saved
    assert all(ctl._ivs[key] is record for key, record in records.items())
    assert runtime.last_result == first
    assert runtime.last_attempt['status'] == 'error'
    assert set(runtime.directory.iterdir()) == files


def test_published_group_error_aborts_forward_and_restores_prior_case(published, monkeypatch):
    from caliper.interp import _replication_actions as actions
    _, _, ctl, service, runtime = published
    first = runtime.run(26)
    saved, records = service.state_dict(), dict(ctl._ivs)
    groups = dict(service.executor._feature_groups)
    files = set(runtime.directory.iterdir())
    completed = []
    forward_logits = actions.forward_logits

    def forward(*args, **kwargs):
        result = forward_logits(*args, **kwargs)
        completed.append(True)
        return result

    def refuse(*args):
        assert service.executor._published_forward_depth > 0
        raise InterventionError('group encoder refusal')

    monkeypatch.setattr(actions, 'forward_logits', forward)
    monkeypatch.setattr('caliper.interp._intervene._trace_features', refuse)
    with pytest.raises(InterventionError, match='group encoder refusal'):
        runtime.run(26, prepare=lambda _: pytest.fail('publication after a failed group'))
    assert not completed
    assert service.executor._published_forward_depth == 0
    assert service.state_dict() == saved
    assert service.executor._feature_groups == groups
    assert all(ctl._ivs[key] is record and not record.hook.removed for key, record in records.items())
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


@pytest.fixture
def replicated_hosted(tmp_path, monkeypatch, request):
    import caliper
    from types import SimpleNamespace
    from caliper.interp import _replication_actions as actions
    from tests.interp import test_hosted
    from tests.interp.cpu_pressure import cpu_fixture_pressure
    hosted_type = test_hosted.HostedInference
    monkeypatch.setattr(actions, 'PROMPTS', {'dallas': 'AB', 'oakland': 'CD', 'shanghai': 'EF', 'vancouver': 'GH'})
    monkeypatch.setattr(actions, 'GROUPS', {name: [(layer % 4, feature % 256, position % 3)
                                                 for layer, feature, position in nodes]
                                          for name, nodes in actions.GROUPS.items()})
    monkeypatch.setattr(caliper, 'serve', lambda: SimpleNamespace(url='in-process'))
    monkeypatch.setattr(test_hosted, 'HostedInference', lambda *a, **k: hosted_type(
        *a, **k, replication_profile='published_dallas_v1', private_output_directory=tmp_path / 'private'))
    with cpu_fixture_pressure():
        yield from test_hosted._hosted_run(tmp_path, monkeypatch, request)


def test_named_published_action_completes_once_and_exposes_bounded_status(replicated_hosted):
    from caliper import _wire
    run, route = replicated_hosted
    names = [f'published_dallas_v1.cell{cell}' for cell in (10, 14, 16, 18, 20, 24, 26, 29)]
    assert all(name in run.controls.actions for name in names)
    queued = route.on_control(None, _wire.Control(70, 'run', name=names[-2]).to_meta())
    assert queued['status'] == 'queued'
    run.boundary()
    replies = [reply for reply in route.transport.replies if reply.request_id == 70]
    assert len(replies) == 1 and replies[0].status == 'ok'
    status, result = run.service.submit({'version': 1, 'method': 'trace.status', 'params': {}}, lambda *a: None)
    assert status == 'ok'
    evidence = result['inference']['replication']['result']
    assert evidence['case_id'] == names[-2]
    assert len(evidence['group_values']) == len(run.interventions.specifications()) == 13
    assert evidence['argmax_token_ids'] == result['inference']['comparison']['argmax_token_ids']
    assert (run.replication.output_directory / evidence['evidence']['file']).is_file()
    import json
    assert len(json.dumps(result)) < 32768
    run.boundary()
    assert len([reply for reply in route.transport.replies if reply.request_id == 70]) == 1


def test_named_published_action_failure_keeps_prior_case_and_one_error(replicated_hosted, monkeypatch):
    from caliper import _wire
    run, route = replicated_hosted
    run.apply_published_case(18)
    before, result = run.interventions.state_dict(), deepcopy(run.replication.last_result)
    def fail(*args, **kwargs):
        raise RuntimeError('publication refusal')
    monkeypatch.setattr(run.output, 'prepare_logits', fail)
    queued = route.on_control(None, _wire.Control(71, 'run', name='published_dallas_v1.cell26').to_meta())
    assert queued['status'] == 'queued'
    with pytest.warns(RuntimeWarning, match='publication refusal'):
        run.boundary()
    replies = [reply for reply in route.transport.replies if reply.request_id == 71]
    assert len(replies) == 1 and replies[0].status == 'error'
    assert run.interventions.state_dict() == before
    assert run.replication.last_result == result and run.replication.last_attempt['status'] == 'error'
    assert len(run.interventions.specifications()) == 6


def test_donor_token_change_refuses_before_forward(published, monkeypatch):
    from caliper.interp import _replication_actions as actions
    _, _, _, service, runtime = published
    inputs = actions.prompt_inputs
    def changed(model, tokenizer, prompt):
        return inputs(model, tokenizer, 'ZZ' if prompt == 'EF' else prompt)
    monkeypatch.setattr(actions, 'prompt_inputs', changed)
    monkeypatch.setattr(actions, 'forward_logits', lambda *a, **k: pytest.fail('forward after donor change'))
    with pytest.raises(ValueError, match='tokenizer'):
        runtime.run(26)
    assert not service.specifications() and runtime.last_attempt['status'] == 'error'


def test_atomic_schedules_refuse_before_any_effect(grouped):
    _, _, _, service = grouped
    with pytest.raises(InterventionError, match='unscheduled'):
        service.replace_atomic([dict(clamp(1, 1), from_step=99)], lambda: pytest.fail('scheduled candidate ran'))
    assert not service.specifications()


def test_published_model_scan_reserves_host_memory_before_readback(published, monkeypatch):
    from types import SimpleNamespace
    from caliper.interp._memory import CleanStop
    _, _, _, service, runtime = published
    # Exercise the real-model reserve policy without a real Gemma allocation.
    monkeypatch.setattr(runtime.model.config, 'model_type', 'gemma2')
    runtime.guard = SimpleNamespace(limit=40_000_000_000, current=0, peak=0, device=torch.device('cpu'),
                                    check=lambda *a, **k: 0)
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=8_000_000_100))
    monkeypatch.setattr(service.executor, '_identity', lambda: pytest.fail('model scan before host admission'))
    with pytest.raises(CleanStop, match='host reserve'):
        runtime._model_identity()
    assert not service.specifications()


def test_generic_cpu_group_keeps_caller_budget_without_gemma_reserve(grouped, monkeypatch):
    from types import SimpleNamespace
    from caliper.interp._memory import CleanStop
    _, _, _, service = grouped
    service.replace_atomic([clamp(1, 1)], lambda: None)
    service.executor.resource_guard = SimpleNamespace(limit=1_000_000, current=0, peak=0,
                                                      device=torch.device('cpu'), check=lambda *a, **k: 0)
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=1_000_000))
    group = service.executor._feature_groups['layers.0.mlp_out']
    value = torch.ones(1, 3, 64, dtype=torch.float64)
    group._apply_members(value, value, group.members)
    service.executor.resource_guard.limit = 1
    with pytest.raises(CleanStop, match='process memory limit'):
        group._apply_members(value, value, group.members)


@pytest.mark.parametrize('phase', ['revision', 'metadata', 'guard', 'prompt', 'watch', 'publication'])
def test_prepared_candidate_refusal_preserves_public_success(replicated_hosted, monkeypatch, phase):
    import threading
    from caliper import _programs
    from caliper.interp import _publish
    run, _ = replicated_hosted
    run.apply_published_case(18)
    records = dict(run.controls._ivs)
    state = run.interventions.state_dict()
    successful = deepcopy(run.replication.last_result)
    comparison = deepcopy(run._comparison)
    files = set(run.replication.directory.iterdir())
    public = {name: (tensor, tensor.clone(), _publish._VERSIONS[name])
              for name, tensor in _publish._TENSORS.items()}
    original = run.output.prepare_logits
    forwards, candidates = [], []
    handle = run.model.register_forward_hook(lambda *args: forwards.append(1))

    def prepare(*args, **kwargs):
        publication, result = original(*args, **kwargs)
        candidates.append(publication)
        assert set(run.replication.directory.iterdir()) > files
        assert all(_publish._resolve(name) is tensor for name, (tensor, _, _) in public.items())
        # A separate owner must acquire both locks while tensor work is private.
        finished = threading.Event()
        def edit():
            with _programs._LOCK, run.controls.lock:
                if phase == 'watch':
                    monkeypatch.setattr(_programs, '_WATCH_EPOCH', _programs._WATCH_EPOCH + 1)
            if phase == 'revision':
                run.interventions.set({'kind': 'ablate', 'point': 'layers.3.mlp_out'})
            if phase == 'prompt':
                run.prompt.set('CD')
            finished.set()
        worker = threading.Thread(target=edit)
        worker.start()
        assert finished.wait(5), 'Candidate preparation retained a publication or Controls lock'
        worker.join(5)
        assert not worker.is_alive()
        if phase == 'metadata':
            monkeypatch.setattr(run.controls, '_admit_interventions',
                                lambda *a, **k: (_ for _ in ()).throw(ValueError('late metadata refusal')))
        if phase == 'guard':
            raise RuntimeError('late resource refusal')
        if phase == 'publication':
            monkeypatch.setattr(_publish, '_model_name', lambda model, name: name == 'output.label_text')
        return publication, result

    monkeypatch.setattr(run.output, 'prepare_logits', prepare)
    try:
        with pytest.raises((InterventionError, ValueError, RuntimeError)):
            run.apply_published_case(26)
    finally:
        handle.remove()
    assert forwards == [1] and len(candidates) == 1
    assert all(run.controls._ivs[key] is record for key, record in records.items())
    if phase != 'revision':
        assert run.interventions.state_dict() == state
    else:
        assert ('layers.3.mlp_out', 'ablate') in run.controls._ivs
    assert run.replication.last_result == successful and run._comparison == comparison
    assert run.replication.last_attempt['status'] == 'error'
    assert set(run.replication.directory.iterdir()) == files
    assert set(_publish._TENSORS) == set(public)
    for name, (tensor, value, version) in public.items():
        assert _publish._resolve(name) is tensor and _publish._VERSIONS[name] == version
        torch.testing.assert_close(tensor, value, rtol=0, atol=0)
