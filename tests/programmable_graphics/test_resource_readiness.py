"""Bounded CPU checks for resource delivery after a viewer attaches."""
import struct
from types import SimpleNamespace

import pytest
import torch

from caliper import _programs as P, _wire as w
from caliper.interp import _publish, publish
from test_inference_readiness import producer, warmup  # noqa: F401


LARGE_COUNT = ((8 << 20) + (128 << 10)) // 4


class Viewer:
    def __init__(self):
        self.caps = {w.CAP_RESOURCES}
        self.attached, self.closed = True, False
        self.queued = self.unsent = 0
        self.frames, self.whole = [], []
        self.parts = w.SplitResources()

    def backlog(self):
        return self.queued


class Controls:
    def __init__(self, *links):
        self.server = SimpleNamespace(producer=SimpleNamespace(links=list(links)))

    def unsent(self, link):
        return link.unsent

    def post(self, link, data, cap):
        assert cap == w.CAP_RESOURCES
        if link is None:
            for viewer in P._resource_links(self):
                self.post(viewer, data, cap)
            return
        link.queued += len(data)
        assert link.queued + link.unsent <= P.RESOURCE_BACKLOG_BYTES
        frame = w.decode_frame(bytes(data))
        link.frames.append(frame)
        if frame.type == w.MsgType.RESOURCE:
            assert any(f.type == w.MsgType.PROGRAM and f.meta['id'] == frame.meta['program']
                       for f in link.frames[:-1])
            resource = link.parts.add(w.Resource.from_frame(frame))
            if resource is not None:
                link.whole.append(resource)


def install(producer, monkeypatch, ctl, *, pid='delivery', count=LARGE_COUNT,
            static_count=0, every=1, published=True):
    monkeypatch.setattr(P, '_controls_ports', lambda: [ctl])
    monkeypatch.setattr(P, '_warn_once', lambda key, message: pytest.fail(message))
    runtime = producer.runtime
    if not hasattr(runtime, 'resource_specs'):
        runtime.resource_specs, runtime.resource_calls = {}, []

    def resource(handle, name):
        runtime.resource_calls.append((handle, name))
        size, value = runtime.resource_specs[handle][name]
        data = struct.pack('<f', runtime.values[handle] if value is None else value) * size
        return data, size, 4, 0, 0, 0

    monkeypatch.setattr(runtime, 'resource', resource)
    publish('trace.node_f', torch.tensor([7.]))
    specs = {}
    if count:
        specs['result'] = (count, None)
    if static_count:
        specs['static'] = (static_count, 3.)
    manifest = {
        'name': 'delivery', 'version': 1,
        'inputs': [{'name': 'payload', 'kind': 'tensor', 'dtype': 'f32'}],
        'resources': [{'id': name, 'kind': 'buffer', 'element': 'f32',
                       'count': size, 'publish': published} for name, (size, _) in specs.items()],
        'derive': {'every': every, 'passes': [
            {'entry': 'derive_values', 'writes': ['result'] if count else []}]},
    }
    item = P.install_piece(
        pid, piece={'name': 'delivery', 'version': 1,
                    'hash': 'sha256:' + '0' * 64, 'from': 'workspace'},
        manifest=manifest, bind={'payload': 'live:trace.node_f'}, derive='CPU fixture')
    assert item['state'] == 'compiled'
    program = P._PROGS[pid]
    runtime.resource_specs[program.handle] = specs
    expected = {name: struct.pack('<f', 7. if value is None else value) * size
                for name, (size, value) in specs.items() if published}
    return program, expected


def drain(producer, links, *, start=13, steps=24):
    for link in links:
        link.queued = link.unsent = 0
    for number in range(start, start + steps):
        if not P.inference_pending():
            break
        producer.step(number)
        for link in links:
            link.queued = link.unsent = 0
    assert not P.inference_pending() and not P._OUTBOX
    for link in links:
        if link.attached and not link.closed:
            assert not link.parts.open
            assert (link.parts.dropped, link.parts.discarded, link.parts.expired) == (0, 0, 0)


def assert_resources(link, pid, expected):
    actual = {r.resource: r for r in link.whole if r.program == pid}
    assert set(actual) == set(expected)
    for name, data in expected.items():
        assert actual[name].count == len(data) // 4 and actual[name].stride == 4
        assert bytes(actual[name].data) == data


@pytest.mark.parametrize('arrival', ['before_snapshot', 'after_snapshot'])
def test_late_viewer_receives_current_resources_after_busy_viewers_drain(producer, monkeypatch, arrival):
    first, late = Viewer(), Viewer()
    ctl = Controls(first)
    program, expected = install(producer, monkeypatch, ctl)
    warmup(producer)
    producer.step(9)
    assert id(first) in P._OUTBOX and not first.whole
    assert program.input_version == _publish._VERSION
    version = _publish._VERSION
    calls = len(producer.runtime.resource_calls)
    if arrival == 'before_snapshot':
        ctl.server.producer.links.append(late)
    else:
        resource_frames = P._resource_frames

        def attach_after_snapshot(runtime, program):
            frames = resource_frames(runtime, program)
            if late not in ctl.server.producer.links:
                assert frames == [] and len(runtime.resource_calls) == calls
                ctl.server.producer.links.append(late)
            return frames

        monkeypatch.setattr(P, '_resource_frames', attach_after_snapshot)
    for number in (10, 11, 12):
        assert P.inference_pending()
        producer.step(number)
    assert producer.runtime.runs == [9, 12]
    # Drain transport queues without a new input version or an unconditional pass.
    drain(producer, [first, late])
    assert _publish._VERSION == version
    assert_resources(first, program.id, expected)
    assert_resources(late, program.id, expected)
    assert late.whole[-1].step >= 12


@pytest.mark.parametrize('static_count', [1, LARGE_COUNT])
@pytest.mark.parametrize('every', [1, 5])
def test_late_viewer_receives_static_and_written_resources(producer, monkeypatch, static_count, every):
    first, second, late = Viewer(), Viewer(), Viewer()
    ctl = Controls(first, second)
    program, expected = install(producer, monkeypatch, ctl, static_count=static_count, every=every)
    warmup(producer)
    producer.step(9)
    assert all(id(link) in P._OUTBOX for link in (first, second))
    assert not first.whole and not second.whole
    version = _publish._VERSION
    ctl.server.producer.links.append(late)
    for number in range(10, 16):
        producer.step(number)
    assert late in program.relayed
    drain(producer, [first, second, late], start=16)
    assert _publish._VERSION == version
    assert all(b - a >= max(P.N_MIN, every)
               for a, b in zip(producer.runtime.runs, producer.runtime.runs[1:]))
    for link in (first, second, late):
        assert_resources(link, program.id, expected)
        assert len([r for r in link.whole if r.resource == 'static']) == 1


@pytest.mark.parametrize('count', [0, 1])
@pytest.mark.parametrize('static_count', [1, LARGE_COUNT])
@pytest.mark.parametrize('queue', ['queued', 'unsent'])
def test_static_resources_retry_after_first_pass_rejection(
        producer, monkeypatch, count, static_count, queue):
    viewer = Viewer()
    program, expected = install(producer, monkeypatch, Controls(viewer),
                                count=count, static_count=static_count)
    warmup(producer)
    viewer.queued = 0
    setattr(viewer, queue, P.RESOURCE_BACKLOG_BYTES)
    producer.step(9)
    assert not viewer.whole and not viewer.parts.open
    assert viewer in program.behind and program.skipped == 1
    version = _publish._VERSION
    drain(producer, [viewer], start=10)
    assert _publish._VERSION == version
    assert_resources(viewer, program.id, expected)


@pytest.mark.parametrize('state', ['no_capability', 'detached', 'closed'])
def test_ineligible_viewers_do_not_request_resource_passes(producer, monkeypatch, state):
    first, late = Viewer(), Viewer()
    ctl = Controls(first)
    program, expected = install(producer, monkeypatch, ctl, count=1)
    warmup(producer)
    producer.step(9)
    assert_resources(first, program.id, expected)
    if state == 'no_capability':
        late.caps = {w.CAP_PROGRAMS}
    elif state == 'detached':
        late.attached = False
    else:
        late.closed = True
    ctl.server.producer.links.append(late)
    assert not P.inference_pending() and not late.frames
    assert producer.runtime.runs == [9]


@pytest.mark.parametrize('count,published', [(0, True), (1, False)])
def test_pieces_without_published_resources_finish_after_draw_relay(producer, monkeypatch, count, published):
    late = Viewer()
    ctl = Controls()
    program, expected = install(producer, monkeypatch, ctl, count=count, published=published)
    warmup(producer)
    producer.step(9)
    assert not P.inference_pending()
    ctl.server.producer.links.append(late)
    drain(producer, [late], start=10)
    assert late in program.relayed and late not in program.behind
    assert [frame.type for frame in late.frames] == [w.MsgType.PROGRAM]
    assert_resources(late, program.id, expected)


@pytest.mark.parametrize('detach_all', [False, True])
def test_attach_and_detach_during_busy_delivery(producer, monkeypatch, detach_all):
    first, second, late = Viewer(), Viewer(), Viewer()
    ctl = Controls(first, second)
    program, expected = install(producer, monkeypatch, ctl, static_count=1)
    warmup(producer)
    producer.step(9)
    assert all(id(link) in P._OUTBOX for link in (first, second))
    first.attached = False
    second.closed = detach_all
    ctl.server.producer.links.append(late)
    for number in (10, 11, 12):
        producer.step(number)
    assert id(first) not in P._OUTBOX
    assert not first.whole
    if detach_all:
        assert id(second) not in P._OUTBOX and not second.whole
    drain(producer, [first, second, late])
    assert_resources(late, program.id, expected)
    if not detach_all:
        assert_resources(second, program.id, expected)


def test_static_relay_rejection_keeps_the_late_viewer_pending(producer, monkeypatch):
    first, late = Viewer(), Viewer()
    ctl = Controls(first)
    program, expected = install(producer, monkeypatch, ctl, static_count=LARGE_COUNT)
    warmup(producer)
    producer.step(9)
    ctl.server.producer.links.append(late)
    late.unsent = P.RESOURCE_BACKLOG_BYTES
    for number in (10, 11, 12):
        producer.step(number)
    assert late not in program.relayed and not late.frames
    assert P.inference_pending()
    drain(producer, [first, late])
    assert_resources(late, program.id, expected)


@pytest.mark.parametrize('second_count', [0, LARGE_COUNT])
def test_multiple_programs_deliver_resources_after_shared_outboxes_drain(producer, monkeypatch, second_count):
    first, late = Viewer(), Viewer()
    ctl = Controls(first)
    programs = [install(producer, monkeypatch, ctl, pid='one', static_count=1),
                install(producer, monkeypatch, ctl, pid='two', count=second_count, static_count=1)]
    warmup(producer)
    producer.step(9)
    assert id(first) in P._OUTBOX
    assert first in programs[1][0].behind
    ctl.server.producer.links.append(late)
    version = _publish._VERSION
    for number in (10, 11, 12):
        producer.step(number)
    if not second_count:
        drain(producer, [first, late])
    else:
        # Two large programs cannot share a queue. Each must deliver a whole pass
        # within these finite admissions, even when the other program skips it.
        for number in range(13, 25):
            for link in (first, late):
                link.queued = 0
            assert P.inference_pending()
            producer.step(number)
            if all(set(expected) <= {r.resource for r in link.whole if r.program == program.id}
                   for program, expected in programs for link in (first, late)):
                break
    assert _publish._VERSION == version
    for program, expected in programs:
        for link in (first, late):
            assert_resources(link, program.id, expected)
            assert len([r for r in link.whole if r.program == program.id and r.resource == 'static']) == 1
            assert (link.parts.dropped, link.parts.discarded, link.parts.expired) == (0, 0, 0)
    for program, _ in programs:
        P.remove(program.id)
    assert not P._OUTBOX and not P.inference_pending()


@pytest.mark.parametrize('static_count', [1, LARGE_COUNT])
def test_static_only_late_viewer_receives_one_complete_version(producer, monkeypatch, static_count):
    late = Viewer()
    ctl = Controls()
    program, expected = install(producer, monkeypatch, ctl, count=0, static_count=static_count)
    warmup(producer)
    producer.step(9)
    assert not P.inference_pending()
    ctl.server.producer.links.append(late)
    drain(producer, [late], start=10)
    assert_resources(late, program.id, expected)
    assert len(late.whole) == 1 and late.whole[0].step == 9


def test_recompile_invalidates_each_viewers_static_admission(producer, monkeypatch):
    viewer = Viewer()
    program, expected = install(producer, monkeypatch, Controls(viewer), count=1, static_count=1)
    warmup(producer)
    producer.step(9)
    assert_resources(viewer, program.id, expected)
    old_handle = program.handle
    derive = producer.runtime.derive

    def recompile(*args, **kwargs):
        handle = derive(*args, **kwargs)
        producer.runtime.resource_specs[handle] = {'result': (1, None), 'static': (1, 4.)}
        return handle

    monkeypatch.setattr(producer.runtime, 'derive', recompile)
    publish('trace.node_f', torch.tensor([7., 7.]))
    drain(producer, [viewer], start=10)
    assert program.handle != old_handle
    assert_resources(viewer, program.id, {'result': struct.pack('<f', 14.), 'static': struct.pack('<f', 4.)})
    assert [r.step for r in viewer.whole if r.resource == 'static'] == [9, 12]
