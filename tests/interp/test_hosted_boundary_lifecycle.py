"""CPU checks for hosted control delivery and boundary cleanup."""
from types import SimpleNamespace

import pytest

from caliper import _controls, _wire
from .hosted_publication_fixture import publication_run


@pytest.mark.parametrize("arrival", ["checkpoint", "callback"])
def test_late_action_reaches_publication(publication_run, monkeypatch, arrival):
    case = publication_run
    controls = case.run.controls
    called, replies = [], []

    def late_action():
        called.append(controls.n)

    controls.action("late", late_action)
    route = _controls.Route(controls, SimpleNamespace(
        may_control=lambda conn: True, reply=lambda conn, meta: replies.append(meta)))
    queued = False

    def enqueue():
        nonlocal queued
        if not queued:
            queued = True
            reply = route.on_control(None, _wire.Control(2, "run", name="late").to_meta())
            assert reply["status"] == "queued"

    if arrival == "checkpoint":
        original = controls.checkpoint

        def checkpoint():
            # The hosted loop selected the boundary before this action arrived.
            enqueue()
            original()

        monkeypatch.setattr(controls, "checkpoint", checkpoint)
    else:
        action = controls.actions["compare"]
        original = action.fn

        def callback():
            # The service captured its admitted actions before this callback.
            enqueue()
            return original()

        monkeypatch.setattr(action, "fn", callback)
    case.drive()
    assert called == [1 if arrival == "checkpoint" else 2]
    assert len(replies) == 1 and replies[0]["status"] == "ok"
    assert case.shown == list(range(8))


@pytest.mark.parametrize("failure", [False, True])
def test_callback_failure_returns_a_refusal(publication_run, failure):
    case = publication_run

    def callback():
        if failure:
            raise ValueError("callback failure")

    action = case.run.controls.action("scope", callback)
    if failure:
        with pytest.warns(RuntimeWarning, match="callback failure"):
            ok, _ = case.run.controls._run_action(action)
        assert not ok
    else:
        assert case.run.controls._run_action(action) == (True, "")


@pytest.mark.parametrize("failure", [False, True])
def test_pause_wait_resumes_after_interrupt(publication_run, monkeypatch, failure):
    case = publication_run
    controls = case.run.controls
    controls.state = _controls.RUN_PAUSING
    route = SimpleNamespace(_controllers=lambda: 1)
    monkeypatch.setattr(controls, "routes", [route])
    services = []

    def service(self):
        assert self is controls
        services.append(self.state)

    def wait(self, timeout):
        assert self is controls.cond
        assert timeout == _controls.WAIT_POLL_S
        if failure:
            raise KeyboardInterrupt
        controls.state = _controls.RUN_RUNNING

    monkeypatch.setattr(_controls.Controls, "_service", service)
    monkeypatch.setattr(_controls.Controls, "_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(type(controls.cond), "wait", wait)
    if failure:
        with pytest.raises(KeyboardInterrupt):
            controls._wait()
    else:
        controls._wait()
    assert controls.state == _controls.RUN_RUNNING


@pytest.mark.parametrize("failure", [False, True])
def test_interpretation_request_clears_running_state(publication_run, monkeypatch, failure):
    from collections import deque
    import threading
    from caliper.interp._service import InterpretationService

    service = InterpretationService.__new__(InterpretationService)
    service.lock = threading.RLock()
    service.idle = threading.Condition(service.lock)
    service.running, service.closed, service.failure = False, False, None
    replies = []
    service.queue = deque([("trace.run", {}, lambda *args: replies.append(args))])

    def request(self, *args):
        if failure:
            raise ValueError("request failure")
        return "ok", {}, None

    monkeypatch.setattr(InterpretationService, "_run_request", request)
    if failure:
        with pytest.raises(ValueError, match="request failure"):
            service.step(1)
    else:
        assert service.step(1) == 1
    assert not service.running and service.running_thread is None
    assert len(replies) == (0 if failure else 1)


def test_intervention_install_keeps_the_applied_specification(publication_run):
    case = publication_run
    controls = case.run.controls

    def install(spec, **kwargs):
        return object()

    record = SimpleNamespace(hook=None, applied_spec=None)
    service = SimpleNamespace(executor=SimpleNamespace(install=install))
    controls._install_intervention(service, record, {"op": "patch"})
    assert record.hook is not None and record.applied_spec == {"op": "patch"}


def test_output_comparison_retains_completion_and_boundary_checks(monkeypatch):
    import threading
    from caliper.interp import _hosted

    calls = []
    clock = iter([10., 10.5])
    monkeypatch.setattr(_hosted, "time", SimpleNamespace(perf_counter=lambda: next(clock)))

    def compare(prompt):
        calls.append(("compare", prompt))
        return {"argmax_token_ids": [1, 1], "baseline_top_text": "token", "baseline_top_probability": .5}

    def retrace(params, step):
        calls.append(("retrace", params, step))
        return "ok", {}, None

    class Owner(_hosted.HostedInference):
        def _input_state(self):
            return "prompt", []

        def _input_hashes(self, key):
            return {}

        def _sync_clamp_control(self, **kwargs):
            calls.append("sync")

    owner = Owner.__new__(Owner)
    owner.controls = SimpleNamespace(lock=threading.RLock(), n=1,
                                    actions={"apply_clamp": SimpleNamespace(run_id=1)})
    owner._inference_lock = threading.RLock()
    owner.output = SimpleNamespace(compare=compare)
    owner.replication = owner.replication_profile = None
    owner.service = SimpleNamespace(_retrace=retrace)
    owner.status = SimpleNamespace(set=lambda text: calls.append(("status", text)))
    owner.guard = SimpleNamespace(check=lambda *a, **k: calls.append("guard"))
    owner.retrace = SimpleNamespace(value=True)
    owner.traced_prompt, owner.traced_specifications = None, []
    owner._output_key, owner._retrace = None, False
    owner._update_output()
    owner._update_output()
    assert calls == [("compare", "prompt"), ("status", "ready"), ("retrace", {"prompt": "prompt"}, 1),
                     "sync", "guard", "sync", "guard"]
    assert owner._comparison["compute_publish_seconds"] == .5
    assert owner._comparison["completed_step"] == owner._output_action_id == 1


class OrderedProgramLock:
    """Reject program lock acquisition while this test owns the controls lock."""

    def __init__(self, controls):
        import threading
        self.controls = controls
        self.lock = threading.RLock()

    def __enter__(self):
        assert not self.controls.lock.locked(), "program lock follows the controls lock"
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


@pytest.mark.parametrize("wait_path", ["idle", "paused"])
def test_boundary_wait_keeps_program_lock_order(publication_run, monkeypatch, wait_path):
    from caliper import _programs as P

    run = publication_run.run
    controls = run.controls
    waits = []
    monkeypatch.setattr(P, "_LOCK", OrderedProgramLock(controls))

    class Finished(BaseException):
        pass

    if wait_path == "idle":
        P._PROGS.clear()
        P._set_active()

        def wait_for(predicate, **kwargs):
            assert controls.lock.locked()
            assert not predicate()
            waits.append(controls.n)
            raise Finished

        monkeypatch.setattr(controls.cond, "wait_for", wait_for)
        with pytest.raises(Finished):
            run.run()
        assert waits == [0] and controls.n == 0
    else:
        controls.state = _controls.RUN_PAUSING
        monkeypatch.setattr(controls, "routes", [SimpleNamespace(_controllers=lambda: 1)])
        monkeypatch.setattr(controls, "_publish", lambda *args, **kwargs: None)

        def wait(timeout):
            assert controls.lock.locked()
            waits.append(controls.n)
            controls.state = _controls.RUN_RUNNING

        monkeypatch.setattr(controls.cond, "wait", wait)
        run.boundary()
        assert waits == [1] and controls.n == 1
        assert controls.state == _controls.RUN_RUNNING
    assert not controls.lock.locked()
