"""Deterministic watch replacement with external listeners and service locks."""
import threading
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
    removed, released, captures, failures, observed = [], [], [], [], []
    P._STATE["rt"] = SimpleNamespace(remove=removed.append, release=released.append)
    monkeypatch.setattr("caliper.interp._points.clear_captures", captures.append)
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
            assert captures == [first]
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
