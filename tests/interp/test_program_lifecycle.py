"""Deterministic watch replacement with external listeners and service locks."""
import threading
import weakref
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import caliper
from caliper import _controls, _programs as P
from caliper._watch import WatchHandle
from caliper.interp import publish
from caliper.interp._service import InterpretationService, register_service, unregister_service
from .test_publication import install_piece, registry


class Model:
    def named_parameters(self):
        return []

    def named_buffers(self):
        return []

    def named_modules(self):
        return []


@pytest.fixture(autouse=True)
def output_events(registry, monkeypatch):
    monkeypatch.setattr(P, "_OUTPUT_EVENTS", deque())
    monkeypatch.setattr(P, "_OUTPUT_DRAINING", False)


@pytest.mark.parametrize("replace", [True, False])
def test_watch_cleanup_releases_program_lock_before_listener_service_lock(registry, monkeypatch, replace):
    first, second, third = Model(), Model(), Model()
    first_handle, second_handle, third_handle = WatchHandle(), WatchHandle(), WatchHandle()
    controls = _controls.Controls()
    service = register_service(InterpretationService(model=first), controls=controls)
    replacement = InterpretationService(model=third)
    P.attach(first, object(), first_handle)
    publish("custom.old", torch.ones(1))
    program = P._Program("old", {"output": {"kind": "tensor", "shape": [1]}}, "", None, "")
    program.runs, program.handle = 1, 17
    imported = P._Import("custom.old", 23, ())
    program.bound = {"input": imported}
    P._PROGS[program.id] = program
    P._IMPORTS[imported.name] = P._LIVE[23] = imported
    removed, released, failures, observed = [], [], [], []
    P._STATE["rt"] = SimpleNamespace(remove=removed.append, release=released.append)
    entered = threading.Event()

    def listener(item):
        if item["id"] != "old":
            return
        entered.set()
        acquired = service.lock.acquire(timeout=5)
        if not acquired:
            failures.append("The listener could not acquire the service lock.")
            return
        try:
            observed.append((P._STATE["model"], controls.interpretation_service))
        finally:
            service.lock.release()

    def remove_output(name):
        assert name == "program.old"
        # The output registry can call a listener which requests program status.
        P._notify([{"id": "old", "state": "removed"}])

    def change_watch():
        try:
            if replace:
                P.attach(second, None, second_handle)
            else:
                P.detach(first)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(caliper, "remove", remove_output)
    P.listeners.append(listener)
    worker = threading.Thread(target=change_watch)
    service.lock.acquire()
    worker.start()
    try:
        assert entered.wait(5)
        # This order deadlocks if the callback still owns the program lock.
        acquired = P._LOCK.acquire(timeout=2)
        assert acquired, "The output listener retained the program lock."
        try:
            assert P._STATE["model"] is (second if replace else None)
            assert controls.interpretation_service is service
            assert second_handle.active is replace
            assert not P._PROGS and not P._IMPORTS and not P._LIVE
            assert P._resolve("custom.old") is None
            assert removed == [17] and released == [23]
        finally:
            P._LOCK.release()
        # A new watch and service must survive completion of the old callback.
        P.attach(third, None, third_handle)
        register_service(replacement, controls=controls)
        publish("custom.keep", torch.ones(1))
        assert install_piece("logit_bars")["state"] == "waiting"
        first_handle.remove()
        second_handle.remove()
        assert P._STATE["model"] is third
    finally:
        service.lock.release()
        worker.join(6)
    try:
        assert not worker.is_alive()
        assert not failures
        assert observed == [(third, replacement)]
        assert P._STATE["model"] is third
        assert controls.interpretation_service is replacement
        assert P._resolve("custom.keep") is not None
        assert P.status()[0]["id"] == "logit_bars"
        assert third_handle.active
    finally:
        third_handle.remove()
        unregister_service(replacement, controls=controls)


def test_conflicting_watch_cleanup_callback_runs_outside_program_lock(registry):
    model = torch.nn.Linear(2, 2)
    publish("weight", torch.ones(1))
    completed, failures = threading.Event(), []

    def inspect():
        acquired = P._LOCK.acquire(timeout=2)
        if not acquired:
            failures.append("The rejected handle retained the program lock.")
            return
        try:
            assert P._STATE["model"] is None
            completed.set()
        finally:
            P._LOCK.release()

    class Handle:
        def remove(self):
            worker = threading.Thread(target=inspect)
            worker.start()
            worker.join(3)
            assert not worker.is_alive()

    with pytest.raises(ValueError, match="conflicts"):
        P.attach(model, None, Handle())
    assert completed.is_set() and not failures


_REAL_RELAY = P._relay
_REAL_RELAY_REMOVAL = P._relay_removal


@pytest.fixture
def output_sink(registry, monkeypatch):
    """Exercise real output calls and wire frames with CPU-only endpoints."""
    from caliper import _wire

    views, pieces, events = {}, {}, []
    link = SimpleNamespace(attached=True, closed=False, caps={_wire.CAP_RESOURCES})

    def post(target, frame, cap):
        assert target is None or target is link
        assert cap == _wire.CAP_RESOURCES
        meta = _wire.decode_frame(frame).meta
        events.append((meta["op"], meta["id"]))
        if meta["op"] == "remove":
            pieces.pop(meta["id"], None)
        else:
            pieces[meta["id"]] = meta["piece"]["hash"]

    def show(tensor, name):
        events.append(("show", name))
        views[name] = tensor

    def remove(name):
        events.append(("remove", name))
        views.pop(name, None)

    ctl = SimpleNamespace(post=post, server=SimpleNamespace(producer=SimpleNamespace(links=[link])))
    monkeypatch.setattr(P, "_controls_ports", lambda: [ctl])
    monkeypatch.setattr(P, "_relay", _REAL_RELAY)
    monkeypatch.setattr(P, "_relay_removal", _REAL_RELAY_REMOVAL)
    monkeypatch.setattr(P, "_OUTBOX", {})
    monkeypatch.setattr(P, "_prepare", lambda *args: None)
    monkeypatch.setattr(P, "_compile", lambda p: setattr(p, "state", "compiled"))
    monkeypatch.setattr(caliper, "show", show)
    monkeypatch.setattr(caliper, "remove", remove)
    return SimpleNamespace(views=views, pieces=pieces, events=events, link=link, ctl=ctl)


def install_output(kind, version, pid="shared"):
    if kind == "piece":
        result = P.install_piece(
            pid, piece={"name": "fixture", "version": 1, "hash": "sha256:" + version * 64},
            manifest={"name": "fixture", "version": 1}, draw="fixture")
        program = P._DRAWN[pid]
    else:
        result = P.install(pid, {"output": {"kind": "tensor", "shape": [1]}}, "fixture")
        program = P._PROGS[pid]
        program.runs = 1
        P._publish(program, {"out": torch.tensor([float(version)])})
    assert result["state"] == "compiled", result
    return program


@pytest.mark.parametrize("kind", ["tensor", "piece"])
@pytest.mark.parametrize("change", ["replace", "detach", "remove"])
def test_deferred_cleanup_preserves_same_id_replacement(output_sink, monkeypatch, kind, change):
    first, second = Model(), Model()
    old_handle, new_handle = WatchHandle(), WatchHandle()
    P.attach(first, None, old_handle)
    old = install_output(kind, "1")
    entered, release = threading.Event(), threading.Event()
    failures = []
    clear = P._remove_output

    def paused(program):
        if program.id == old.id:
            entered.set()
            assert release.wait(5), "The cleanup event was not released."
        clear(program)

    def change_watch():
        try:
            if change == "replace":
                P.attach(second, None, new_handle)
            elif change == "detach":
                P.detach(first)
            else:
                P.remove(old.id)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(P, "_remove_output", paused)
    worker = threading.Thread(target=change_watch)
    worker.start()
    try:
        assert entered.wait(5)
        if change != "replace":
            P.attach(second, None, new_handle)
        new = install_output(kind, "2")
        published = torch.ones(1, requires_grad=True)
        publish("custom.keep", published)
        old_handle.remove()
        assert P._STATE["model"] is second and new_handle.active
        # A same-ID outbox already uses instance identity and must survive.
        output_sink_outbox = [output_sink.link, output_sink.ctl, new, []]
        P._OUTBOX[id(output_sink.link)] = output_sink_outbox
    finally:
        release.set()
        worker.join(6)
    assert not worker.is_alive() and not failures
    assert (P._PROGS if kind == "tensor" else P._DRAWN)["shared"] is new
    assert P._OUTBOX[id(output_sink.link)] is output_sink_outbox
    resolved = P._resolve("custom.keep")
    assert resolved.data_ptr() == published.data_ptr() and not resolved.requires_grad
    if kind == "tensor":
        assert output_sink.views["program.shared"].item() == 2
    else:
        assert output_sink.pieces["shared"] == "sha256:" + "2" * 64
    new_handle.remove()
    assert not output_sink.views and not output_sink.pieces


@pytest.mark.parametrize("kind", ["tensor", "piece"])
def test_replacement_inside_removal_callback_has_ordered_outputs(output_sink, monkeypatch, kind):
    first, second = Model(), Model()
    P.attach(first, None)
    old = install_output(kind, "1")
    remove = caliper.remove if kind == "tensor" else output_sink.ctl.post
    inspected, failures = [], []

    def inspect():
        acquired = P._LOCK.acquire(timeout=2)
        if not acquired:
            failures.append("An output callback retained the program lock.")
            return
        try:
            inspected.append(P._STATE["model"])
        finally:
            P._LOCK.release()

    def replace_then_remove(*args):
        if kind == "piece":
            from caliper import _wire
            if _wire.decode_frame(args[1]).meta["op"] != "remove":
                return remove(*args)
        worker = threading.Thread(target=inspect)
        worker.start()
        worker.join(3)
        assert not worker.is_alive()
        # Reentry must return without a wait on the callback that owns this thread.
        install_output(kind, "2")
        return remove(*args)

    with monkeypatch.context() as patch:
        if kind == "tensor":
            patch.setattr(caliper, "remove", replace_then_remove)
        else:
            patch.setattr(output_sink.ctl, "post", replace_then_remove)
        P.attach(second, None)
    assert not failures and inspected == [second]
    if kind == "tensor":
        assert output_sink.views["program.shared"].item() == 2
        assert output_sink.events == [
            ("show", "program.shared"), ("remove", "program.shared"), ("show", "program.shared")]
    else:
        assert output_sink.pieces["shared"] == "sha256:" + "2" * 64
        assert output_sink.events == [("set", old.id), ("remove", old.id), ("set", old.id)]
    assert not P._OUTPUT_EVENTS and not P._OUTPUT_DRAINING
    P.detach(second)
    assert not output_sink.views and not output_sink.pieces


def test_removed_pass_cannot_publish_after_same_id_replacement(output_sink):
    P.attach(Model(), None)
    old = install_output("tensor", "1")
    P.detach()
    P.attach(Model(), None)
    new = install_output("tensor", "2")
    events = list(output_sink.events)
    P._publish_pass([(old, {"out": torch.tensor([3.0])})])
    assert output_sink.events == events
    assert output_sink.views["program.shared"].item() == 2
    assert P._PROGS["shared"] is new


@contextmanager
def stalled_cleanup(monkeypatch):
    """Hold one removal callback while the caller changes program state."""
    if P._STATE["model"] is None:
        P.attach(Model(), None)
    install_output("tensor", "1", "barrier")
    entered, release = threading.Event(), threading.Event()
    failures = []
    remove = P._remove_output

    def pause(p):
        if p.id == "barrier":
            entered.set()
            assert release.wait(5)
        remove(p)

    def worker_main():
        try:
            P.remove("barrier")
        except BaseException as exc:
            failures.append(exc)

    with monkeypatch.context() as patch:
        patch.setattr(P, "_remove_output", pause)
        worker = threading.Thread(target=worker_main)
        worker.start()
        try:
            assert entered.wait(5)
            yield
        finally:
            release.set()
            worker.join(6)
        assert not worker.is_alive() and not failures


def pending_passes():
    return [args for callback, args in P._OUTPUT_EVENTS if callback is P._publish_output]


def test_pending_passes_coalesce_and_release_retired_storage(output_sink, monkeypatch):
    with stalled_cleanup(monkeypatch):
        p = install_output("tensor", "1")
        first = torch.ones(1)
        reference = weakref.ref(first)
        P._publish(p, {"out": first})
        del first
        latest = torch.full((1,), 2.0)
        P._publish(p, {"out": latest})
        assert reference() is None
        assert pending_passes() == [(p, {"out": latest})]
        assert pending_passes()[0][1]["out"].data_ptr() == latest.data_ptr()
        # The compact removal must not retain program-owned static frames either.
        marker = torch.ones(1)
        marker_ref = weakref.ref(marker)
        p.static_frames["fixture"] = [marker]
        reference = weakref.ref(latest)
        P.remove(p.id)
        del p, marker, latest
        assert reference() is None and marker_ref() is None
        assert not pending_passes()
        assert all(isinstance(args[0], P._OutputRemoval) for _, args in P._OUTPUT_EVENTS)
        install_output("tensor", "3")
        P.remove("shared")
        assert len(P._OUTPUT_EVENTS) == 1
        install_output("tensor", "4")
    assert output_sink.views["program.shared"].item() == 4
    assert not P._OUTPUT_EVENTS and not P._OUTPUT_DRAINING


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_pending_pass_bounds_evict_oldest_payload(output_sink, monkeypatch, limit):
    assert P._OUTPUT_PENDING_MAX == 16 and P._OUTPUT_BYTES_MAX == 64 << 20
    monkeypatch.setattr(P, "_OUTPUT_PENDING_MAX", 2 if limit == "count" else 16)
    monkeypatch.setattr(P, "_OUTPUT_BYTES_MAX", 8 if limit == "bytes" else 64)
    with stalled_cleanup(monkeypatch):
        first = install_output("tensor", "1", "first")
        second = install_output("tensor", "2", "second")
        third = install_output("tensor", "3", "third")
        pending = pending_passes()
        assert [p for p, _ in pending] == [second, third]
        assert sum(P._output_bytes(outs) for _, outs in pending) == 8
        assert first.output_retry and first.skipped == 1
    assert "program.first" not in output_sink.views
    assert output_sink.views["program.second"].item() == 2
    assert output_sink.views["program.third"].item() == 3
    P._publish(first, {"out": torch.tensor([4.0])})
    assert output_sink.views["program.first"].item() == 4 and not first.output_retry


def test_pending_tensor_bound_counts_underlying_storage(output_sink, monkeypatch):
    monkeypatch.setattr(P, "_OUTPUT_BYTES_MAX", 8)
    with stalled_cleanup(monkeypatch):
        p = install_output("tensor", "1")
        storage = torch.arange(4, dtype=torch.float32)
        assert P._output_bytes({"out": storage[:1], "alias": storage[1:2]}) == 16
        reference = weakref.ref(storage)
        P._publish(p, {"out": storage[:1]})
        del storage
        assert reference() is None and not pending_passes()
        assert p.output_retry and p.skipped == 1
    P._publish(p, {"out": torch.tensor([2.0])})
    assert output_sink.views["program.shared"].item() == 2


def test_coalesced_piece_pass_replays_static_resources(output_sink, monkeypatch):
    received = []
    monkeypatch.setattr(P, "_relay", lambda p: None)
    monkeypatch.setattr(P, "_send_resources", lambda p, frames: received.append(frames))
    with stalled_cleanup(monkeypatch):
        P.install_piece("derived", piece={"name": "fixture", "hash": "sha256:" + "1" * 64},
                        manifest={"derive": {"passes": [{"entry": "main"}]}}, derive="fixture")
        p = P._PROGS["derived"]
        static, old, latest = b"static", b"old", b"latest"
        p.static_frames["static"] = [static]
        P._publish(p, [static, old])
        P._publish(p, [latest])
        assert pending_passes() == [(p, [static, latest])]
    assert received == [[static, latest]]


def test_retired_piece_releases_pending_relay_and_outbox(output_sink, monkeypatch):
    with stalled_cleanup(monkeypatch):
        p = install_output("piece", "1")
        marker = torch.ones(1)
        reference = weakref.ref(marker)
        p.static_frames["fixture"] = [marker]
        P._OUTBOX[id(output_sink.link)] = [output_sink.link, output_sink.ctl, p, [marker]]
        P.remove(p.id)
        del p, marker
        assert reference() is None and not P._OUTBOX
        assert all(isinstance(args[0], P._OutputRemoval) for _, args in P._OUTPUT_EVENTS)
        install_output("piece", "2")
    assert output_sink.pieces["shared"] == "sha256:" + "2" * 64
    assert ("set", "shared") in output_sink.events
    assert output_sink.events.count(("set", "shared")) == 1


def test_callback_exception_does_not_strand_later_outputs(output_sink, monkeypatch):
    failed = []
    relay = P._relay

    def raise_once(p):
        if not failed:
            failed.append(p.id)
            install_output("tensor", "2")
            raise RuntimeError("fixture callback failure")
        relay(p)

    monkeypatch.setattr(P, "_relay", raise_once)
    with pytest.warns(RuntimeWarning, match="fixture callback failure"):
        install_output("piece", "1", "broken")
    assert failed == ["broken"]
    assert output_sink.views["program.shared"].item() == 2
    assert not P._OUTPUT_EVENTS and not P._OUTPUT_DRAINING
    P.remove("shared")
    assert not output_sink.views


def test_base_exception_releases_drain_for_next_call(output_sink, monkeypatch):
    class Cancelled(BaseException):
        pass

    def cancel(p):
        install_output("tensor", "2")
        raise Cancelled()

    with monkeypatch.context() as patch:
        patch.setattr(P, "_relay", cancel)
        with pytest.raises(Cancelled):
            install_output("piece", "1", "cancelled")
    assert not P._OUTPUT_DRAINING and len(pending_passes()) == 1
    P.remove("unknown")
    assert output_sink.views["program.shared"].item() == 2
    assert not P._OUTPUT_EVENTS and not P._OUTPUT_DRAINING


@pytest.mark.parametrize("kind", ["tensor", "piece"])
def test_control_capacity_preserves_published_cleanup_and_refuses_new_ids(output_sink, monkeypatch, kind):
    assert P._OUTPUT_CONTROL_MAX == 16
    monkeypatch.setattr(P, "_OUTPUT_CONTROL_MAX", 2)
    P.attach(Model(), None)
    published = install_output(kind, "1", "published")
    with stalled_cleanup(monkeypatch):
        # One queued removal and one live instance consume both control slots.
        P.remove(published.id)
        install_output(kind, "2", "pending")
        # Replacement can reuse an existing slot even when admission is full.
        replacement = install_output(kind, "3", "pending")
        with pytest.raises(RuntimeError, match="program.over_limits.*all 2 slots"):
            install_output(kind, "4", "refused")
        assert (P._PROGS if kind == "tensor" else P._DRAWN)["pending"] is replacement
        assert "refused" not in P._PROGS and "refused" not in P._DRAWN
        P.remove("pending")
        assert len(P._OUTPUT_EVENTS) == 2
        with pytest.raises(RuntimeError, match="program.over_limits"):
            install_output(kind, "5", "next")
    assert not output_sink.views and not output_sink.pieces
    assert not P._OUTPUT_EVENTS and not P._OUTPUT_DRAINING
    install_output(kind, "6", "next")
    if kind == "tensor":
        assert output_sink.views["program.next"].item() == 6
    else:
        assert output_sink.pieces["next"] == "sha256:" + "6" * 64
