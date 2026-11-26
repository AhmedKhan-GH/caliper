"""CPU checks for hook identity across published preparation and execution."""
import pytest
import torch

from tests.interp.test_replication_groups import grouped, published


def test_output_hook_after_preparation_refuses_before_forward_or_commit(published, monkeypatch):
    from caliper.interp import _replication_actions as actions
    model, _, ctl, service, runtime = published
    prior = runtime.run(26)
    records, state = dict(ctl._ivs), service.state_dict()
    files = {path: path.read_bytes() for path in runtime.directory.iterdir()}
    preparation_id = runtime.preparation['id']
    baseline = runtime.baseline_logits
    assert torch.isfinite(baseline).all() and baseline.max() != baseline.min()
    outputs, forwards = [], []
    forward = actions.forward_logits

    def counted(*args, **kwargs):
        forwards.append(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(actions, 'forward_logits', counted)
    handle = model.lm_head.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
    try:
        with pytest.raises(ValueError, match='hook'):
            runtime.run(10, prepare=lambda pair: outputs.append(pair))
    finally:
        handle.remove()
    assert not forwards and not outputs
    assert runtime.preparation['id'] == preparation_id
    assert runtime.last_result == prior and runtime.last_attempt['status'] == 'error'
    assert service.state_dict() == state
    assert all(ctl._ivs[key] is record for key, record in records.items())
    assert {path: path.read_bytes() for path in runtime.directory.iterdir()} == files


@pytest.mark.parametrize('location', ['head_pre', 'layer', 'root', 'global_pre', 'global_post'])
def test_added_user_hooks_refuse_without_trial_calls(published, location):
    from torch.nn.modules import module as global_hooks
    model, _, _, _, runtime = published
    calls = []

    def delayed_change(module, args, output=None):
        calls.append(1)
        if len(calls) > 1 and isinstance(output, torch.Tensor):
            return torch.zeros_like(output)

    register = {
        'head_pre': model.lm_head.register_forward_pre_hook,
        'layer': model.layers[0].register_forward_hook,
        'root': model.register_forward_hook,
        'global_pre': global_hooks.register_module_forward_pre_hook,
        'global_post': global_hooks.register_module_forward_hook,
    }[location]
    handle = register(delayed_change)
    try:
        with pytest.raises(ValueError, match='hooks changed'):
            runtime.run(10)
        assert not calls
        assert list(runtime.directory.iterdir()) == [runtime.directory / 'preparation.json']
    finally:
        handle.remove()


@pytest.mark.parametrize('change', ['unchanged', 'remove', 'replace', 'code', 'order', 'kwargs', 'always'])
def test_prepared_user_hook_registrations_and_flags(published, monkeypatch, change):
    model, _, _, _, runtime = published
    runtime.close()

    def first(*args):
        return None

    def second(*args):
        return None

    handles = [model.lm_head.register_forward_hook(first), model.lm_head.register_forward_hook(second)]
    try:
        runtime.prepare()
        key = handles[0].id
        if change == 'remove':
            handles[0].remove()
        elif change == 'replace':
            model.lm_head._forward_hooks[key] = second
        elif change == 'code':
            monkeypatch.setattr(first, '__code__', second.__code__)
        elif change == 'order':
            model.lm_head._forward_hooks.move_to_end(key)
        elif change == 'kwargs':
            model.lm_head._forward_hooks_with_kwargs[key] = True
        elif change == 'always':
            model.lm_head._forward_hooks_always_called[key] = True
        if change == 'unchanged':
            result = runtime.run(10)
            native = torch.load(runtime.output_directory / result['native_evidence']['file'], weights_only=True)
            torch.testing.assert_close(native['logits'][0], native['logits'][1], rtol=0, atol=0)
        else:
            with pytest.raises(ValueError, match='hooks changed'):
                runtime.run(10)
            assert list(runtime.directory.iterdir()) == [runtime.directory / 'preparation.json']
    finally:
        for handle in handles:
            handle.remove()


def test_owned_capture_and_program_observation_can_enter_after_preparation(published, monkeypatch):
    from caliper import _programs
    from caliper.interp._points import capture
    model, _, _, _, runtime = published
    monkeypatch.setitem(_programs._STATE, 'model', model)
    monkeypatch.setattr(_programs, '_ACT_HOOKS', {})
    monkeypatch.setattr(_programs, '_ACTS', {})
    monkeypatch.setattr(_programs, '_PROGS', {})
    with pytest.raises(_programs._Waiting):
        _programs._resolve('lm_head.act')
    try:
        with capture(model, ['layers.0.mlp_in', 'lm_head']) as observed:
            result = runtime.run(26)
            assert set(observed) == {'layers.0.mlp_in', 'lm_head'}
            torch.testing.assert_close(observed['lm_head'], _programs._resolve('lm_head.act'), rtol=0, atol=0)
            assert result['cell'] == 26
    finally:
        _programs._ACT_HOOKS['lm_head'].remove()
        _programs._ACT_HOOKS.clear()
    assert runtime.run(10)['cell'] == 10


@pytest.mark.parametrize('owner', ['capture', 'intervention', 'runtime'])
@pytest.mark.parametrize('change', ['callback', 'flags'])
def test_owned_handle_does_not_admit_a_replacement_callback(published, owner, change):
    from caliper.interp._points import capture
    model, _, _, service, runtime = published
    runtime.run(26)
    with capture(model, ['lm_head']) as observed:
        if owner == 'capture':
            handle = observed._handles[-1]
        elif owner == 'intervention':
            handle = next(iter(service.executor._feature_groups.values())).handles[-1]
        else:
            handle = runtime._handles[-1]
        registry = handle.hooks_dict_ref()
        callback = registry[handle.id]
        flags = handle.extra_dict_ref[0]()
        original = dict(flags)
        try:
            if change == 'callback':
                registry[handle.id] = lambda *args: None
            elif handle.id in flags:
                del flags[handle.id]
            else:
                flags[handle.id] = True
            with pytest.raises(ValueError, match='hooks changed'):
                runtime.run(10)
        finally:
            registry[handle.id] = callback
            flags.clear()
            flags.update(original)


@pytest.mark.parametrize('phase', ['tokenizer', 'export', 'publication', 'commit'])
def test_late_hook_changes_roll_back_prior_evidence(published, monkeypatch, phase):
    from caliper.interp import _replication_actions as actions
    model, _, ctl, service, runtime = published
    prior = runtime.run(26)
    files = {path: path.read_bytes() for path in runtime.directory.iterdir()}
    records, state = dict(ctl._ivs), service.state_dict()
    handles, outputs = [], []

    def change():
        handles.append(model.lm_head.register_forward_hook(lambda module, args, output: torch.zeros_like(output)))

    if phase == 'tokenizer':
        prompt_inputs = actions.prompt_inputs
        def inputs(*args, **kwargs):
            result = prompt_inputs(*args, **kwargs)
            if not handles:
                change()
            return result
        monkeypatch.setattr(actions, 'prompt_inputs', inputs)
        monkeypatch.setattr(actions, 'forward_logits', lambda *a, **k: pytest.fail('forward after hook change'))
    elif phase == 'export':
        write = runtime._write_json
        def changed(*args, **kwargs):
            result = write(*args, **kwargs)
            change()
            return result
        monkeypatch.setattr(runtime, '_write_json', changed)

    def prepare(pair):
        outputs.append(pair)
        change()

    try:
        with pytest.raises(ValueError, match='hooks changed'):
            runtime.run(10, prepare=prepare if phase == 'publication' else None,
                        check=change if phase == 'commit' else None)
    finally:
        for handle in handles:
            handle.remove()
    assert bool(outputs) == (phase == 'publication')
    assert runtime.last_result == prior and runtime.last_attempt['status'] == 'error'
    assert service.state_dict() == state
    assert all(ctl._ivs[key] is record for key, record in records.items())
    assert {path: path.read_bytes() for path in runtime.directory.iterdir()} == files


def test_hook_mutation_during_preparation_refuses_and_cleans_owned_state(published, monkeypatch):
    from caliper.interp import _replication_actions as actions
    model, _, _, _, runtime = published
    runtime.close()
    directories = set(runtime.output_directory.iterdir())
    forward, handles = actions.forward_logits, []

    def changed(*args, **kwargs):
        result = forward(*args, **kwargs)
        handles.append(model.lm_head.register_forward_hook(lambda *args: None))
        return result

    monkeypatch.setattr(actions, 'forward_logits', changed)
    try:
        with pytest.raises(ValueError, match='hooks changed'):
            runtime.prepare()
    finally:
        for handle in handles:
            handle.remove()
    assert runtime.preparation is None and runtime._hook_snapshot is None
    assert runtime._model_snapshot is None and not runtime._handles
    assert set(runtime.output_directory.iterdir()) == directories


def test_unchanged_output_hook_is_part_of_both_preparation_and_execution(published):
    model, _, _, _, runtime = published
    runtime.close()
    handle = model.lm_head.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
    try:
        runtime.prepare()
        result = runtime.run(10)
        native = torch.load(runtime.output_directory / result['native_evidence']['file'], weights_only=True)
        assert torch.count_nonzero(native['logits']) == 0
        torch.testing.assert_close(native['probabilities'][0], native['probabilities'][1], rtol=0, atol=0)
    finally:
        handle.remove()


def test_removal_of_rejected_hook_restores_the_prepared_registration(published):
    model, _, _, _, runtime = published
    preparation = runtime.preparation['id']
    handle = model.lm_head.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
    try:
        with pytest.raises(ValueError, match='hooks changed'):
            runtime.run(10)
    finally:
        handle.remove()
    result = runtime.run(10)
    assert result['preparation_id'] == preparation and runtime.last_attempt['status'] == 'ok'
    assert runtime._model_snapshot.nbytes == 0


@pytest.mark.parametrize('point', ['layers.0.mlp_out', 'layers.0.resid_pre', 'layers.0.self_attn#0'])
def test_owned_ablation_can_retire_after_clean_preparation(published, point):
    _, _, _, service, runtime = published
    service.replace_atomic([{'kind': 'ablate', 'point': point}], lambda: None)
    runtime.close()
    runtime.prepare()
    assert runtime.run(26)['cell'] == 26
    assert runtime.run(10)['cell'] == 10
    assert not service.specifications()
