"""CPU admission checks for hosted compute and blocked resource delivery."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from caliper import _programs as P, _wire
from caliper.interp import publish
from .test_publication import registry  # noqa: F401


@pytest.fixture
def blocked_outbox(registry, monkeypatch):
    program = P._Program('delivery', {}, '', None, '')
    program.kind, program.state, program.handle = 'piece', 'running', 1
    program.params_dirty = False
    link = SimpleNamespace(caps={_wire.CAP_RESOURCES}, attached=True, closed=False,
                           queued=P.RESOURCE_BACKLOG_BYTES)
    link.backlog = lambda: link.queued
    posts = []
    ctl = SimpleNamespace(server=SimpleNamespace(producer=SimpleNamespace(links=[link])),
                          post=lambda ln, frame, cap: posts.append(frame))
    program.relayed.add(link)
    P._PROGS[program.id] = program
    monkeypatch.setattr(P, '_controls_ports', lambda: [ctl])
    monkeypatch.setattr(P, '_OUTBOX', {id(link): [link, ctl, program, deque([(8, [b'resource'])])]})
    yield SimpleNamespace(program=program, link=link, posts=posts)
    program.handle = None
    P._OUTBOX.clear()


def test_outbox_admission_uses_the_next_chunk_size(blocked_outbox):
    case = blocked_outbox
    assert P.inference_pending() and P.inference_blocked()
    case.link.queued = P.RESOURCE_BACKLOG_BYTES - 7
    assert P.inference_blocked()
    case.link.queued -= 1
    assert not P.inference_blocked()
    P._pump()
    assert case.posts == [b'resource']
    assert not P._OUTBOX and not P.inference_pending() and not P.inference_blocked()


def test_ready_compute_remains_admissible_beside_blocked_delivery(blocked_outbox):
    program = P._Program('compute', {
        'inputs': [{'var': 'probs', 'source': 'live:trace.probs'}],
        'output': {'kind': 'tensor', 'shape': [1]},
    }, '', None, '')
    P._PROGS[program.id] = program
    assert P.inference_blocked()  # This compute input is absent.
    publish('trace.probs', torch.ones(1))
    assert P.inference_pending() and not P.inference_blocked()
    program.state = 'stopped'
    program.error = {'code': 'program.over_budget'}
    assert P.inference_pending() and P.inference_blocked()
    assert program.state == 'stopped' and program.error['code'] == 'program.over_budget'


def test_readiness_keeps_budget_history_and_does_not_load_a_runtime(blocked_outbox, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('readiness called the runtime or reset the cadence')

    P._CAD.base.append(.25)
    P._CAD.cost.append(.02)
    P._CAD.over = 2
    monkeypatch.setattr(P, '_runtime', forbidden)
    monkeypatch.setattr(P, '_import', forbidden)
    monkeypatch.setattr(P, '_torch_adapter', forbidden)
    with monkeypatch.context() as patch:
        patch.setattr(P._Cadence, 'reset', forbidden)
        patch.setattr(P._Cadence, 'restart', forbidden)
        assert P.inference_pending() and P.inference_blocked()
        blocked_outbox.link.queued = 0
        assert not P.inference_blocked()
        assert list(P._CAD.base) == [.25] and list(P._CAD.cost) == [.02]
        assert P._CAD.over == 2


def test_input_refusal_does_not_wait_for_transport(blocked_outbox):
    program = blocked_outbox.program
    program.inputs = [('unknown', 'live:unknown')]
    program.params_dirty = True
    assert P.inference_pending() and not P.inference_blocked()


def test_caught_up_viewer_does_not_admit_a_blocked_viewers_retry(blocked_outbox):
    case = blocked_outbox
    case.program.resources = [{'publish': True}]
    case.program.behind.add(case.link)
    ready = SimpleNamespace(caps={_wire.CAP_RESOURCES}, attached=True, closed=False,
                            backlog=lambda: 0)
    case.program.relayed.add(ready)
    ctl = P._controls_ports()[0]
    ctl.server.producer.links.append(ready)
    assert P.inference_pending() and P.inference_blocked()
    # A new parameter value needs a pass for the available viewer.
    case.program.params_dirty = True
    assert not P.inference_blocked()
