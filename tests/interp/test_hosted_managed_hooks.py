"""CPU checks for published preparation during managed fork startup."""
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
import socket

import pytest
import torch

from caliper import _managed, _wire
from caliper.interp import _replication_actions as actions
from caliper.interp._output import forward_logits, prompt_inputs
from .test_replication_groups import clamp, replicated_hosted


@contextmanager
def changed_hook(runtime, managed, first, callback, patch, change):
    model, hosted = runtime.model, managed.inference
    added = None
    if change == 'callback':
        patch.setitem(model._forward_pre_hooks, first.id, lambda *args: None)
    elif change == 'owner':
        patch.setitem(model._forward_pre_hooks, first.id, MethodType(callback.__func__, object()))
    elif change == 'bound_identity':
        patch.setitem(model._forward_pre_hooks, first.id, MethodType(callback.__func__, managed))
    elif change == 'method':
        patch.setitem(model._forward_pre_hooks, first.id, managed._forward)
    elif change == 'code':
        patch.setattr(callback.__func__, '__code__', (lambda *args: None).__code__)
    elif change == 'flags':
        patch.setitem(model._forward_pre_hooks_with_kwargs, first.id, True)
    elif change == 'registration':
        patch.delitem(model._forward_pre_hooks, first.id)
    elif change == 'key':
        patch.setattr(first, 'id', first.id + 1000000)
    elif change == 'registry':
        patch.setattr(model, '_forward_pre_hooks', model._forward_pre_hooks.copy())
    elif change == 'handle':
        patch.setattr(managed, '_first', SimpleNamespace(id=first.id))
    elif change == 'watch':
        patch.setattr(hosted, 'handle', SimpleNamespace(_handles=[first]))
    elif change == 'hosted':
        patch.setattr(managed, 'inference', SimpleNamespace(model=model))
    elif change == 'managed':
        patch.setattr(_managed, 'current', lambda: SimpleNamespace(inference=hosted))
    elif change == 'extra_callback':
        added = model.register_forward_pre_hook(callback)
    elif change == 'resurrect':
        patch.setitem(model._forward_pre_hooks, first.id, callback)
        patch.setattr(managed, '_first', first)
    else:
        raise AssertionError('Unknown hook mutation')
    try:
        yield
    finally:
        if added is not None:
            added.remove()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import caliper

    def refuse(*args, **kwargs):
        pytest.fail("CPU hook checks must not use sockets")

    monkeypatch.setattr(caliper, 'serve', lambda: SimpleNamespace(url='in-process'))
    monkeypatch.setattr(socket.socket, 'bind', refuse)
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket.socket, 'connect_ex', refuse)


@pytest.fixture
def managed_hosted(monkeypatch, request):
    initialize = _managed.ManagedRun.__init__
    prepare = actions.PublishedCases.prepare
    observed = {}
    override = clamp(1, 1, 8.)

    def initialize_fork(self, *args, **kwargs):
        initialize(self, *args, **kwargs, interventions=[override])

    def prepare_clean(self):
        managed = _managed.current()
        first = managed._first
        assert first is not None
        assert first.hooks_dict_ref() is self.model._forward_pre_hooks
        callback = self.model._forward_pre_hooks[first.id]
        assert callback.__self__ is managed
        assert callback.__func__ is _managed.ManagedRun._intervene_on_forward
        assert not self.interventions.specifications()
        expected = forward_logits(self.model, prompt_inputs(self.model, self.tokenizer, 'AB'))[0, -1]
        result = prepare(self)
        assert managed._first is first
        assert self.model._forward_pre_hooks[first.id] is callback
        assert not self.interventions.specifications()
        torch.testing.assert_close(self.baseline_logits, expected, rtol=0, atol=0)
        observed.update(first=first, callback=callback, baseline=expected.clone())
        if getattr(request, 'param', None) is not None:
            with monkeypatch.context() as patch:
                with changed_hook(self, managed, first, callback, patch, request.param):
                    with pytest.raises(ValueError, match='hooks changed'):
                        self._check_hooks()
            self._check_hooks()
        return result

    monkeypatch.setattr(_managed.ManagedRun, '__init__', initialize_fork)
    monkeypatch.setattr(actions.PublishedCases, 'prepare', prepare_clean)
    run, route = request.getfixturevalue('replicated_hosted')
    assert run.managed._first is None
    assert observed['first'].id not in run.model._forward_pre_hooks
    installed, = run.interventions.specifications()
    assert all(installed[key] == value for key, value in override.items())
    changed = forward_logits(run.model, prompt_inputs(run.model, run.tokenizer, 'AB'))[0, -1]
    assert not torch.equal(changed, observed['baseline'])
    return run, route, observed


@pytest.mark.parametrize('cell', [10, 26])
def test_published_callback_after_managed_fork_startup(managed_hosted, cell):
    run, route, observed = managed_hosted
    name = f'published_dallas_v1.cell{cell}'
    reply = route.on_control(None, _wire.Control(70, 'run', name=name).to_meta())
    assert reply['status'] == 'queued'
    run.boundary()
    replies = [reply for reply in route.transport.replies if reply.request_id == 70]
    assert len(replies) == 1
    assert replies[0].status == 'ok', run.replication.last_attempt
    result = run.replication.last_result
    assert result['case_id'] == name
    assert run._comparison['case_id'] == name
    assert run.replication.last_attempt['status'] == 'ok'
    native = torch.load(run.replication.output_directory / result['native_evidence']['file'], weights_only=True)
    torch.testing.assert_close(native['logits'][0], observed['baseline'], rtol=0, atol=0)
    if cell == 10:
        torch.testing.assert_close(native['logits'][0], native['logits'][1], rtol=0, atol=0)
        assert not run.interventions.specifications()
    else:
        assert len(run.interventions.specifications()) == 13
        assert torch.isfinite(native['logits']).all()


@pytest.mark.parametrize('managed_hosted', [
    'callback', 'owner', 'bound_identity', 'method', 'code', 'flags', 'registration', 'key', 'registry',
    'handle', 'watch', 'hosted', 'managed', 'extra_callback',
], indirect=True)
def test_pending_managed_hook_mutation_refuses(managed_hosted):
    run, _, _ = managed_hosted
    assert run.replication.run(10)['cell'] == 10


@pytest.mark.parametrize('change', ['resurrect', 'extra_callback', 'flags', 'handle', 'watch', 'hosted', 'managed'])
def test_retired_managed_hook_mutation_refuses_before_forward(managed_hosted, monkeypatch, change):
    run, _, observed = managed_hosted
    runtime = run.replication
    files = set(runtime.directory.iterdir())
    state = run.interventions.state_dict()

    def refuse(*args, **kwargs):
        pytest.fail('A changed hook reached the model forward')

    with monkeypatch.context() as patch:
        patch.setattr(actions, 'forward_logits', refuse)
        with changed_hook(runtime, run.managed, observed['first'], observed['callback'], patch, change):
            with pytest.raises(ValueError, match='hooks changed'):
                runtime.run(10)
    assert set(runtime.directory.iterdir()) == files
    assert run.interventions.state_dict() == state
    assert runtime.last_result is None and runtime.last_attempt['status'] == 'error'
    assert runtime.run(10)['cell'] == 10
