"""CPU checks for event-driven hosted resource admission."""
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
import threading

import pytest

from caliper import _controls, _programs as P, _wire
from caliper._net import Link
from caliper._net_controls import ControlsPort
from caliper._net_native import _NativeLink, NativeServer
from caliper.interp import _hosted


class BoundaryReached(BaseException):
    pass


class NativeQueue:
    """Logical queue with the native atomic observation contract."""

    def __init__(self):
        self.queued = {}
        self.watches = {}
        self.sent = []
        self.events = []
        self.transfer = None

    def control_bytes(self, conn):
        return self.queued.get(conn, 0)

    def watch_capacity(self, conn, limit, token):
        self.watches.pop(conn, None)
        if limit < 0 or self.control_bytes(conn) <= limit:
            return True
        self.watches[conn] = (limit, token)
        return False

    def send(self, conn, frame):
        self.sent.append((conn, frame))
        self.queued[conn] = self.control_bytes(conn) + len(frame)
        if self.transfer:
            self.transfer(conn)
        return True

    def wake(self):
        pass

    def drain(self, conn, left=0):
        self.queued[conn] = left
        watch = self.watches.get(conn)
        if watch is not None and left <= watch[0]:
            del self.watches[conn]
            self.events.append(("capacity", conn, str(watch[1]).encode()))


@pytest.fixture
def capacity_run(monkeypatch):
    controls = _controls.Controls()
    controls.state = _controls.RUN_RUNNING
    monkeypatch.setattr(_controls, "CONTROLS", controls)
    run = _hosted.HostedInference.__new__(_hosted.HostedInference)
    run.controls, run.service = controls, SimpleNamespace(pending=False)
    run.guard = run.managed = run._failure = None
    run.status = SimpleNamespace(set=pytest.fail)
    native = NativeQueue()
    producer = SimpleNamespace(nio=native, links=set(), by_conn={},
                               attached_changed=lambda delta: None)
    server = SimpleNamespace(nio=native, producer=producer, closed=False,
                             ring=SimpleNamespace(sender_cost=nullcontext))
    port = ControlsPort(server, True, native=True)
    producer.controls = port
    program = P._Program("delivery", {}, "", None, "")
    program.kind, program.state, program.handle = "piece", "running", 1
    program.params_dirty = False
    monkeypatch.setattr(P, "_PROGS", {program.id: program})
    monkeypatch.setattr(P, "_OUTBOX", {})
    monkeypatch.setattr(P, "_CAD", P._Cadence())
    monkeypatch.setattr(P, "_VIEWS", {"ring": None})
    monkeypatch.setattr(P, "_STATE", {**P._STATE, "rt": None})
    monkeypatch.setattr(P, "_controls_ports", lambda: [port])

    def viewer(conn, queued=P.RESOURCE_BACKLOG_BYTES):
        link = _NativeLink(producer, conn)
        link.caps, link.attached = {_wire.CAP_RESOURCES}, True
        native.queued[conn] = queued
        producer.links.add(link)
        producer.by_conn[conn] = link
        program.relayed.add(link)
        P._OUTBOX[id(link)] = [link, port, program, deque([(8, [b"resource"])])]
        return link

    def dispatch():
        events, native.events = native.events, []
        for event in events:
            NativeServer._on_control(server, *event)

    def drain(link, left=0):
        native.drain(link.conn, left)
        dispatch()

    def boundary():
        P._pump()
        raise BoundaryReached

    monkeypatch.setattr(run, "boundary", boundary)
    for name in ("inference_pending", "inference_blocked"):
        query = getattr(P, name)

        def outside_controls(query=query):
            assert not controls.lock.locked(), "readiness must precede the controls lock"
            return query()

        monkeypatch.setattr(P, name, outside_controls)
    link = viewer(1)
    yield SimpleNamespace(run=run, native=native, port=port, link=link, program=program,
                          viewer=viewer, drain=drain, dispatch=dispatch, server=server)
    program.handle = None
    P._OUTBOX.clear()
    port.detach()


def posted(case):
    return [(ln.conn, frame) for ln, frame, _ in case.port._posts]


def test_capacity_notification_ends_the_condition_wait(capacity_run, monkeypatch):
    case = capacity_run
    controls = case.run.controls
    entered = threading.Event()
    waits, wakes, failures = [], [], []
    wait = controls.cond.wait

    def observed_wait(timeout=None):
        waits.append(timeout)
        assert len(waits) == 1, "capacity must end the first condition wait"
        entered.set()
        result = wait(timeout)
        wakes.append(result)
        return result

    def hosted():
        try:
            case.run.run()
        except BoundaryReached:
            pass
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(controls.cond, "wait", observed_wait)
    thread = threading.Thread(target=hosted, daemon=True)
    thread.start()
    try:
        assert entered.wait(3), "host did not enter its resource wait"
        case.drain(case.link)
    finally:
        thread.join(3)
    assert not thread.is_alive()
    assert not failures
    assert wakes == [True]
    assert waits == [1.]
    assert posted(case) == [(1, b"resource")]


def drive_capacity_event(case, monkeypatch, *, action=False):
    """Use the actual Condition.wait_for algorithm with a logical clock."""
    controls = case.run.controls
    clock = SimpleNamespace(now=0., waits=[])
    monkeypatch.setattr(threading, "_time", lambda: clock.now)

    def wait(timeout=None):
        clock.waits.append(timeout)
        assert len(clock.waits) == 1, "publication must not repeat the timeout"
        controls.cond.release()
        try:
            case.drain(case.link)
            if action:
                case.run.service.pending = True
        finally:
            controls.cond.acquire()
        return True

    monkeypatch.setattr(controls.cond, "wait", wait)
    with pytest.raises(BoundaryReached):
        case.run.run()
    assert posted(case) == [(1, b"resource")]
    assert not P._OUTBOX and not P.inference_pending()
    assert getattr(P._INFERENCE, "clock", None) is None
    return clock


def test_drained_delivery_needs_no_condition_timeout(capacity_run, monkeypatch):
    clock = drive_capacity_event(capacity_run, monkeypatch)
    assert clock.now == 0.
    assert clock.waits == [1.]


def test_pending_action_admits_drained_delivery(capacity_run, monkeypatch):
    clock = drive_capacity_event(capacity_run, monkeypatch, action=True)
    assert clock.now == 0.


def test_notification_before_wait_retains_its_identity(capacity_run, monkeypatch):
    case = capacity_run
    query = P.inference_blocked
    calls = []

    def before_wait():
        blocked = query()
        calls.append(blocked)
        if len(calls) == 1:
            assert blocked
            case.drain(case.link)
        return blocked

    monkeypatch.setattr(P, "inference_blocked", before_wait)
    monkeypatch.setattr(case.run.controls.cond, "wait", lambda timeout: pytest.fail("lost event"))
    with pytest.raises(BoundaryReached):
        case.run.run()
    assert calls == [True, False]
    assert posted(case) == [(1, b"resource")]


def test_capacity_check_arms_after_a_concurrent_drain(capacity_run, monkeypatch):
    case = capacity_run
    watch = case.native.watch_capacity

    def drain_before_arm(conn, limit, token):
        case.native.queued[conn] = 0
        return watch(conn, limit, token)

    monkeypatch.setattr(case.native, "watch_capacity", drain_before_arm)
    assert not P.inference_blocked()
    assert not case.native.watches and not case.port._capacity_waits


@pytest.mark.parametrize("during_transfer", [False, True])
def test_posts_rearm_after_native_transfer(capacity_run, during_transfer):
    case = capacity_run
    case.native.queued[1] = 0
    case.port.post(case.link, bytes(P.RESOURCE_BACKLOG_BYTES), _wire.CAP_RESOURCES)
    assert P.inference_blocked()
    assert case.native.watches == {}  # Native capacity alone cannot admit the next chunk.
    if during_transfer:
        case.native.transfer = lambda conn: case.native.drain(conn)
    case.port.flush()
    assert case.port.unsent(case.link) == 0
    if not during_transfer:
        assert case.run.controls.publication_event == 0
        assert case.native.watches[1][0] == P.RESOURCE_BACKLOG_BYTES - 8
        case.drain(case.link)
    assert case.run.controls.publication_event == 1
    assert not P.inference_blocked()
    assert not case.port._capacity_waits


def test_partial_capacity_admits_only_the_next_chunk(capacity_run):
    case = capacity_run
    assert P.inference_blocked()
    case.drain(case.link, P.RESOURCE_BACKLOG_BYTES - 7)
    assert case.run.controls.publication_event == 0
    case.drain(case.link, P.RESOURCE_BACKLOG_BYTES - 8)
    assert case.run.controls.publication_event == 1
    assert not P.inference_blocked()
    P._pump()
    assert posted(case) == [(1, b"resource")]


def test_new_snapshot_waits_for_an_empty_queue(capacity_run):
    case = capacity_run
    P._OUTBOX.clear()
    case.program.behind.add(case.link)
    assert P.inference_blocked()
    assert case.native.watches[1][0] == 0
    case.drain(case.link, 1)
    assert case.run.controls.publication_event == 0
    case.drain(case.link)
    assert case.run.controls.publication_event == 1
    assert not P.inference_blocked()


def test_viewers_have_independent_observations(capacity_run):
    case = capacity_run
    second = case.viewer(2)
    assert P.inference_blocked()
    case.drain(second)
    assert case.run.controls.publication_event == 1
    assert not P.inference_blocked()
    P._pump()
    assert posted(case) == [(2, b"resource")]
    assert P.inference_blocked()
    assert set(case.native.watches) == {1}
    case.drain(second)
    assert case.run.controls.publication_event == 1
    case.drain(case.link)
    assert case.run.controls.publication_event == 2
    assert not P.inference_blocked()


def test_replaced_observation_ignores_an_inflight_event(capacity_run):
    case = capacity_run
    assert P.inference_blocked()
    token = case.native.watches[1][1]
    assert P.inference_blocked()
    NativeServer._on_control(case.server, "capacity", 1, str(token).encode())
    assert case.run.controls.publication_event == 0
    assert len(case.native.watches) == len(case.port._capacity_waits) == 1
    case.drain(case.link)
    assert case.run.controls.publication_event == 1


@pytest.mark.parametrize("terminal", ["detached", "closed", "port", "program"])
def test_terminal_cleanup_removes_observations(capacity_run, terminal):
    case = capacity_run
    assert P.inference_blocked()
    if terminal in ("detached", "closed"):
        NativeServer._on_control(case.server, terminal, 1, b"")
    elif terminal == "port":
        case.port.detach()
    else:
        P.remove(case.program.id)
    assert not case.native.watches and not case.port._capacity_waits
    event = case.run.controls.publication_event
    case.drain(case.link)
    assert case.run.controls.publication_event == event


def test_closed_port_outbox_needs_one_cleanup_boundary(capacity_run, monkeypatch):
    case = capacity_run
    assert P.inference_blocked()
    case.port.detach()
    # A closed native server can end its event pump before it marks each Python link closed.
    assert case.link.attached and not case.link.closed
    monkeypatch.setattr(P, "_controls_ports", lambda: [])
    assert P.inference_pending() and not P.inference_blocked()
    P._pump()
    assert not P._OUTBOX and not P.inference_pending()
    assert posted(case) == []


@pytest.mark.parametrize("work", ["service", "parameters", "actions"])
def test_controls_remain_ready_during_capacity_blockage(capacity_run, monkeypatch, work):
    case = capacity_run
    if work == "service":
        case.run.service.pending = True
    elif work == "parameters":
        case.run.controls._npending = 1
    else:
        case.run.controls.queue.append(object())
    monkeypatch.setattr(case.run.controls.cond, "wait", lambda timeout: pytest.fail("control wait"))
    with pytest.raises(BoundaryReached):
        case.run.run()
    assert posted(case) == []
    assert P.inference_pending() and P.inference_blocked()


def test_spurious_notification_does_not_create_a_busy_loop(capacity_run, monkeypatch):
    case = capacity_run
    controls = case.run.controls
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        assert len(waits) <= 2, "bounded wait count"
        if len(waits) == 2:
            controls.cond.release()
            try:
                case.drain(case.link)
            finally:
                controls.cond.acquire()
        return True

    monkeypatch.setattr(controls.cond, "wait", wait)
    with pytest.raises(BoundaryReached):
        case.run.run()
    assert len(waits) == 2
    assert posted(case) == [(1, b"resource")]


def test_older_native_module_retains_timeout_fallback(capacity_run, monkeypatch):
    case = capacity_run
    legacy = SimpleNamespace(control_bytes=case.native.control_bytes)
    monkeypatch.setattr(case.link.p, "nio", legacy)
    assert P.inference_blocked()
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(threading, "_time", lambda: clock.now)

    def timeout(wait):
        clock.now += wait
        case.native.queued[1] = 0
        return False

    monkeypatch.setattr(case.run.controls.cond, "wait", timeout)
    with pytest.raises(BoundaryReached):
        case.run.run()
    assert clock.now == 1.
    assert posted(case) == [(1, b"resource")]


def test_python_sender_emits_one_capacity_event(capacity_run):
    case = capacity_run
    link = Link(case.server.producer)
    link.attached, link.caps = True, {_wire.CAP_RESOURCES}
    link.push(b"first")
    link.push(b"last")
    assert not case.port.watch_capacity({link: (4, {case.program})})
    assert link.next_frame(0)[0] == b"first"
    assert case.run.controls.publication_event == 1
    assert link.next_frame(0)[0] == b"last"
    assert case.run.controls.publication_event == 1
    assert not case.port._capacity_waits


def test_unrelated_post_transfer_does_not_rearm_a_viewer(capacity_run):
    case = capacity_run
    second = case.viewer(2)
    assert P.inference_blocked()
    first_watch = case.native.watches[1]
    case.port.post(second, b"other", _wire.CAP_RESOURCES)
    case.port.flush()
    assert case.native.watches[1] == first_watch
    assert case.run.controls.publication_event == 0


def test_broadcast_posts_count_for_each_matching_viewer(capacity_run):
    case = capacity_run
    second = case.viewer(2, queued=0)
    case.native.queued[1] = 0
    case.port.post(None, bytes(P.RESOURCE_BACKLOG_BYTES), _wire.CAP_RESOURCES)
    assert P.inference_blocked()
    assert not case.native.watches
    case.port.flush()
    assert set(case.native.watches) == {1, 2}
    case.drain(second)
    assert case.run.controls.publication_event == 1
    assert set(case.native.watches) == {1}


def test_program_removal_replaces_a_shared_observation(capacity_run, monkeypatch):
    case = capacity_run
    notify = case.run.controls.publication_changed

    def outside_program_lock():
        assert not P._LOCK._is_owned(), "program removal must not acquire Controls.lock"
        notify()

    monkeypatch.setattr(case.run.controls, "publication_changed", outside_program_lock)
    other = P._Program("other", {}, "", None, "")
    other.kind, other.state, other.handle = "piece", "running", 2
    other.behind.add(case.link)
    other.params_dirty = False
    P._PROGS[other.id] = other
    P._OUTBOX.clear()
    case.program.behind.add(case.link)
    try:
        assert P.inference_blocked()
        assert case.port._capacity_waits[case.link][1] == {case.program, other}
        P.remove(case.program.id)
        assert not case.port._capacity_waits and not case.native.watches
        assert case.run.controls.publication_event == 0
        case.port.flush()
        assert case.run.controls.publication_event == 1
        assert P.inference_blocked()
        assert case.port._capacity_waits[case.link][1] == {other}
    finally:
        P.remove(other.id)
