"""CPU checks for complete hosted boundary time and late control work."""
from types import SimpleNamespace

import pytest
import torch

from caliper import _controls, _programs as P, _wire
from caliper.interp import publish
from .test_hosted_cadence import action_run


@pytest.mark.parametrize("phase", ["checkpoint", "guard", "keep"])
def test_hosted_clock_includes_non_action_boundary_work(action_run, monkeypatch, phase):
    case = action_run
    case.clock.idle_s = case.clock.action_s = .55
    calls = []

    def work(*args, **kwargs):
        calls.append(case.run.controls.n)
        case.clock.now += .0004

    if phase == "checkpoint":
        checkpoint = case.run.controls.checkpoint

        def checkpoint_work():
            work()
            checkpoint()

        monkeypatch.setattr(case.run.controls, "checkpoint", checkpoint_work)
    elif phase == "guard":
        case.run.guard = SimpleNamespace(failure=None, check=work)
    else:
        case.run.model = object()
        case.run.managed = SimpleNamespace(stopping=False, _keep=work)
    case.drive()
    assert case.shown == list(range(8))
    assert len(calls) == case.run.controls.n
    assert all(reply["status"] == "ok" for reply in case.replies)
    useful = case.clock.now - 8 * case.clock.action_s - 7 * case.clock.idle_s
    assert P._CAD.inference_s == pytest.approx(useful)
    assert P._CAD.step_s() == pytest.approx(.001)
    assert P._CAD.cost_s() == pytest.approx(.00002)
    assert len(P._CAD.cost) == 7


def test_hosted_clock_includes_readiness_scan(action_run, monkeypatch):
    case = action_run
    publish("output.probs", torch.ones(1))
    pending = P.inference_pending
    boundary = case.run.boundary
    calls = []

    def readiness():
        calls.append(case.run.controls.n)
        case.clock.now += .0004
        return pending()

    class Finished(BaseException):
        pass

    def one_boundary():
        boundary()
        raise Finished

    monkeypatch.setattr(P, "inference_pending", readiness)
    monkeypatch.setattr(case.run, "boundary", one_boundary)
    with pytest.raises(Finished):
        case.run.run()
    assert calls == [0]
    assert P._CAD.inference_s == pytest.approx(case.clock.now)


@pytest.mark.parametrize("arrival", ["checkpoint", "callback"])
def test_late_action_does_not_enter_cadence(action_run, monkeypatch, arrival):
    case = action_run
    controls = case.run.controls
    called, replies = [], []

    def late_action():
        case.clock.now += .7
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
    assert P._CAD.cost_s() == pytest.approx(.00002)
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert len(P._CAD.cost) == 7


def snapshot_ring(clock):
    """Use the real gate lifecycle with controlled snapshot and sender durations."""
    from caliper._net_ring import NetRing

    ring = NetRing(engine=object())
    ring.pass_t = 0.
    return ring


@pytest.mark.parametrize("failure", [False, True])
def test_nested_scopes_restore_time_and_snapshot_ownership(action_run, monkeypatch, failure):
    case = action_run
    ring = snapshot_ring(case.clock)
    # Gate timestamps use the same controlled clock as their snapshot costs.
    from caliper import _net_ring
    monkeypatch.setattr(_net_ring.time, "perf_counter", lambda: case.clock.now)
    P.views(ring)
    ring.open_pass(1)

    def snapshot():
        case.clock.now += .003
        ring.gate_cost += .003
        ring.gate_taken += 1

    snapshot()                                    # outside the measured scope
    with P.inference_boundary():
        case.clock.now += .001
        snapshot()                                # already in elapsed measured time
        with P.inference_boundary():
            case.clock.now += .002
        try:
            with P.inference_excluded():
                case.clock.now += .7
                snapshot()                        # charged despite the action exclusion
                with P.inference_boundary(), P.inference_excluded():
                    case.clock.now += .8
                    snapshot()
                if failure:
                    raise ValueError("scope failure")
        except ValueError:
            pass
        case.clock.now += .004
    assert P._CAD.inference_s == pytest.approx(.019)
    assert getattr(P._INFERENCE, "clock", None) is None
    # Reentry into the same open gate must not charge its old snapshots again.
    with P.inference_boundary():
        case.clock.now += .001
    assert P._CAD.inference_s == pytest.approx(.020)


@pytest.mark.parametrize("change", ["reset", "replace", "close"])
def test_snapshot_gate_identity_survives_changes(action_run, monkeypatch, change):
    case = action_run
    from caliper import _net_ring
    monkeypatch.setattr(_net_ring.time, "perf_counter", lambda: case.clock.now)
    ring = snapshot_ring(case.clock)
    P.views(ring)
    ring.open_pass(1)
    with P.inference_boundary():
        with P.inference_excluded():
            case.clock.now += .003
            ring.gate_cost = .003
        if change == "reset":
            # Repeated step numbers still have a distinct gate timestamp.
            ring.open_pass(1)
        elif change == "replace":
            ring = snapshot_ring(case.clock)
            P.views(ring)
            ring.open_pass(1)
        else:
            P.views_closed(ring)
            P.views(ring)
            ring.open_pass(1)
        with P.inference_excluded():
            case.clock.now += .002
            ring.gate_cost = .002
    assert P._CAD.inference_s == pytest.approx(.005)
    assert P._CAD.inference_gate[0][0] is ring
    assert P._CAD.inference_gate[1] == .002


@pytest.mark.parametrize("failure", [False, True])
def test_callback_failure_restores_measured_scope(action_run, failure):
    case = action_run

    def callback():
        case.clock.now += .7
        if failure:
            raise ValueError("callback failure")

    action = case.run.controls.action("scope", callback)
    with P.inference_boundary():
        case.clock.now += .001
        if failure:
            with pytest.warns(RuntimeWarning, match="callback failure"):
                ok, _ = case.run.controls._run_action(action)
            assert not ok
        else:
            assert case.run.controls._run_action(action) == (True, "")
        case.clock.now += .002
    assert P._CAD.inference_s == pytest.approx(.003)


@pytest.mark.parametrize("failure", [False, True])
def test_pause_wait_excludes_only_the_wait(action_run, monkeypatch, failure):
    case = action_run
    controls = case.run.controls
    controls.state = _controls.RUN_PAUSING
    route = SimpleNamespace(_controllers=lambda: 1)
    monkeypatch.setattr(controls, "routes", [route])
    services = []

    def service(self):
        assert self is controls
        services.append(self.state)
        case.clock.now += .001

    def wait(self, timeout):
        assert self is controls.cond
        assert timeout == _controls.WAIT_POLL_S
        case.clock.now += .7
        if failure:
            raise KeyboardInterrupt
        controls.state = _controls.RUN_RUNNING

    monkeypatch.setattr(_controls.Controls, "_service", service)
    monkeypatch.setattr(_controls.Controls, "_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(type(controls.cond), "wait", wait)
    with P.inference_boundary():
        if failure:
            with pytest.raises(KeyboardInterrupt):
                controls._wait()
        else:
            controls._wait()
        case.clock.now += .002
    assert P._CAD.inference_s == pytest.approx(.002 + .001 * len(services))
    assert controls.state == _controls.RUN_RUNNING


@pytest.mark.parametrize("failure", [False, True])
def test_interpretation_request_exclusion_and_cleanup(action_run, monkeypatch, failure):
    from collections import deque
    import threading
    from caliper.interp._service import InterpretationService

    case = action_run
    service = InterpretationService.__new__(InterpretationService)
    service.lock = threading.RLock()
    service.idle = threading.Condition(service.lock)
    service.running, service.closed, service.failure = False, False, None
    replies = []
    service.queue = deque([("trace.run", {}, lambda *args: replies.append(args))])

    def request(self, *args):
        case.clock.now += .7
        if failure:
            raise ValueError("request failure")
        return "ok", {}, None

    monkeypatch.setattr(InterpretationService, "_run_request", request)
    with P.inference_boundary():
        if failure:
            with pytest.raises(ValueError, match="request failure"):
                service.step(1)
        else:
            assert service.step(1) == 1
        case.clock.now += .002
    assert P._CAD.inference_s == pytest.approx(.002)
    assert not service.running and service.running_thread is None
    assert len(replies) == (0 if failure else 1)


def test_intervention_install_excludes_patch_preparation(action_run):
    case = action_run
    controls = case.run.controls

    def install(spec, **kwargs):
        case.clock.now += .7
        return object()

    record = SimpleNamespace(hook=None, applied_spec=None)
    service = SimpleNamespace(executor=SimpleNamespace(install=install))
    with P.inference_boundary():
        controls._install_intervention(service, record, {"op": "patch"})
        case.clock.now += .002
    assert P._CAD.inference_s == pytest.approx(.002)
    assert record.hook is not None and record.applied_spec == {"op": "patch"}


def test_sender_credit_survives_intermediate_pass_without_snapshot(action_run, monkeypatch):
    case = action_run
    from caliper import _net_ring
    monkeypatch.setattr(_net_ring.time, "perf_counter", lambda: case.clock.now)
    ring = snapshot_ring(case.clock)
    P.views(ring)
    publish("output.probs", torch.ones(1))
    credited = False
    for step in range(1, 33):
        with P.inference_boundary():
            if P._STATE["pass"] == step and case.program.runs in (0, 2):
                assert ring.gate_open
                ring.gate_taken = 1
                ring.gate_cost = .00003
                case.clock.now += .00003
            if case.program.runs == 2 and not credited:
                # The second derive pass discards the first snapshot's view_cost.
                assert 9 not in P._CAD.view_cost
                ring.publish_s += .00001
                credited = True
            P.at_step(step)
        if case.program.runs == 3:
            break
    assert credited and case.program.runs == 3
    assert len(P._CAD.cost) == 2
    assert sum(P._CAD.cost) == pytest.approx(.00005 + .00002 + .00001)


@pytest.mark.parametrize("send_step", [10, 11])
def test_outbox_interval_has_one_hosted_cost_owner(action_run, monkeypatch, send_step):
    from collections import deque

    case = action_run
    publish("output.probs", torch.ones(1))
    link = SimpleNamespace(closed=False, attached=True)
    sent = []

    def post(control, target, chunk):
        assert target is link
        sent.append(chunk)
        case.clock.now += .00003

    monkeypatch.setattr(P, "_post", post)
    monkeypatch.setattr(P, "_backlog", lambda *args: 0)
    for step in range(1, 13):
        if step == send_step:
            P._OUTBOX[id(link)] = [link, None, case.program, deque([(4, [b"part"])])]
        with P.inference_boundary():
            P.at_step(step)
    assert sent == [(4, [b"part"])] and not P._OUTBOX
    assert P._CAD.step_s() == pytest.approx(.0006)
    assert P._CAD.cost_s() == pytest.approx(.00002 + .00003)
    assert P._CAD.inference_s == pytest.approx(12 * .0006 + 2 * .00002)
    assert P._CAD.extra == 0.


def controlled_ratio(patch, *, measured):
    """Compare one controlled omission with the complete scope, without elapsed waits."""
    from contextlib import nullcontext

    case = action_run.__wrapped__(patch)
    case.clock.scheduler_s, case.clock.publish_s = .0000055, 0.
    case.clock.pass_s, case.clock.waves = .00058, 12
    case.clock.action_s = case.clock.idle_s = .55

    class RoutineGuard:
        failure = None

        def check(self, *args, **kwargs):
            # This interval is a controlled input, not measured model performance.
            with nullcontext() if measured else P.inference_excluded():
                case.clock.now += .0004

    case.run.guard = RoutineGuard()
    case.drive()
    assert case.shown == list(range(12))
    assert all(reply["status"] == "ok" for reply in case.replies)
    assert case.program.state == "running" and len(P._CAD.cost) == P.COST_SAMPLES
    assert P._CAD.cost_s() == pytest.approx(.00058)
    return {"base_seconds": P._CAD.step_s(), "cost_seconds": P._CAD.cost_s(),
            "cadence_steps": P._CAD.n, "wanted_steps": P._CAD.want,
            "boundaries": case.run.controls.n,
            "total_virtual_boundary_seconds": case.clock.now - 12 * .55 - 11 * .55,
            "total_virtual_publication_delay_seconds": sum(case.delays),
            "last_virtual_publication_delay_seconds": case.delays[-1]}


def test_representative_ratio_changes_steps_and_virtual_delay(monkeypatch):
    with monkeypatch.context() as patch:
        omitted = controlled_ratio(patch, measured=False)
    with monkeypatch.context() as patch:
        measured = controlled_ratio(patch, measured=True)
    assert omitted["base_seconds"] == pytest.approx(.0000055)
    assert measured["base_seconds"] == pytest.approx(.0004055)
    assert omitted["wanted_steps"] == 2637
    assert measured["wanted_steps"] == 36
    assert measured["boundaries"] < omitted["boundaries"]
    assert measured["total_virtual_publication_delay_seconds"] < omitted["total_virtual_publication_delay_seconds"]
    assert measured["last_virtual_publication_delay_seconds"] < omitted["last_virtual_publication_delay_seconds"]


@pytest.mark.parametrize("change", ["replace", "close", "unsubscribe"])
def test_subscription_thread_retains_excluded_snapshot_cost(action_run, monkeypatch, change):
    import threading
    from caliper import _net_ring

    case = action_run
    monkeypatch.setattr(_net_ring.time, "perf_counter", lambda: case.clock.now)
    ring = snapshot_ring(case.clock)
    P.views(ring)
    ring.open_pass(1)
    replacement = snapshot_ring(case.clock) if change == "replace" else None
    with P.inference_boundary(), P.inference_excluded():
        case.clock.now += .003
        ring.gate_cost = .003
        ring.registered = True
        if replacement is not None:
            replacement.views["view"] = SimpleNamespace(groups={"subscription": None})
            transition = replacement.changed
        else:
            transition = ring.close if change == "close" else ring.changed
        worker = threading.Thread(target=transition)
        worker.start()
        worker.join(timeout=1.)
        assert not worker.is_alive()
        if replacement is not None:
            replacement.open_pass(1)
            case.clock.now += .002
            replacement.gate_cost = .002
    assert P._CAD.inference_s == pytest.approx(.005 if replacement else .003)


def test_sender_replacement_credits_only_each_rings_new_work(action_run):
    case = action_run
    first, second = snapshot_ring(case.clock), snapshot_ring(case.clock)
    first.publish_s, second.publish_s = 2., 8.
    P.views(first)
    with P.inference_boundary():
        P.at_step(1)
    first.publish_s += .003
    P.views(second)
    assert P._CAD.extra == pytest.approx(.003)
    second.publish_s += .002
    P.views_closed(second)
    assert P._CAD.extra == pytest.approx(.005)
    assert P._CAD.inference_sender is None


def test_clock_mode_changes_retain_duration_history_without_mixed_marks(action_run):
    case = action_run
    case.drive()
    base, cost = list(P._CAD.base), list(P._CAD.cost)
    period, wanted, ceiling, over = P._CAD.n, P._CAD.want, P._CAD.ceiling, P._CAD.over
    P._CAD.extra = .000007
    case.clock.now += 50.
    P.at_step(case.run.controls.n + 1)              # ordinary training uses wall time
    assert not P._CAD.inference
    assert min(P._CAD.done.values()) >= 50.
    with P.inference_boundary():
        P.at_step(case.run.controls.n + 2)          # hosted marks use accumulated durations
    assert P._CAD.inference
    assert max(P._CAD.done.values()) < .1
    assert list(P._CAD.base) == base and list(P._CAD.cost) == cost
    assert (P._CAD.n, P._CAD.want, P._CAD.ceiling, P._CAD.over) == (period, wanted, ceiling, over)
    assert P._CAD.extra == .000007


@pytest.mark.parametrize("snapshot_s,allowed", [(.002, True), (.02, False)])
def test_floor_uses_elapsed_view_age_and_its_existing_share(action_run, snapshot_s, allowed):
    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    with P.inference_boundary():
        P.at_step(1)
    P._STATE["view_s"].append(snapshot_s)
    assert not P._floor_due(ring, P.FLOOR_S - .001)
    assert P._floor_due(ring, P.FLOOR_S) is allowed
    # The floor query neither changes program admission nor consumes budget history.
    assert case.program.runs == 0 and not P._CAD.cost


def test_hosted_output_actions_keep_routine_checks_and_elapsed_timer(action_run, monkeypatch):
    import threading
    from caliper.interp import _hosted

    case = action_run
    monkeypatch.setattr(_hosted.time, "perf_counter", lambda: case.clock.now)

    def compare(prompt):
        case.clock.now += .55
        return {"argmax_token_ids": [1, 1], "baseline_top_text": "token", "baseline_top_probability": .5}

    def retrace(params, step):
        case.clock.now += .7
        return "ok", {}, None

    def guard(*args, **kwargs):
        case.clock.now += .0003

    class Owner(_hosted.HostedInference):
        def _input_state(self):
            case.clock.now += .0001
            return "prompt", []

        def _input_hashes(self, key):
            return {}

        def _sync_clamp_control(self, **kwargs):
            case.clock.now += .0002

    owner = Owner.__new__(Owner)
    owner.controls = SimpleNamespace(lock=threading.RLock(), n=1,
                                    actions={"apply_clamp": SimpleNamespace(run_id=1)})
    owner._inference_lock = threading.RLock()
    owner.output = SimpleNamespace(compare=compare)
    owner.service = SimpleNamespace(_retrace=retrace)
    owner.status = SimpleNamespace(set=lambda text: None)
    owner.guard = SimpleNamespace(check=guard)
    owner.retrace = SimpleNamespace(value=True)
    owner.traced_prompt, owner.traced_specifications = None, []
    owner._output_key, owner._retrace = None, False
    with P.inference_boundary():
        owner._update_output()
        owner._update_output()
    assert P._CAD.inference_s == pytest.approx(.0012)
    assert owner._comparison["compute_publish_seconds"] == pytest.approx(.55)
    assert owner._output_action_id == 1


def test_reentrant_checkpoint_measures_publication_and_restores_exclusion(action_run):
    case = action_run
    with P.inference_boundary():
        case.clock.now += .001
        with P.inference_excluded():
            case.clock.now += .7
            P.at_step(1)
            case.clock.now += .8
        case.clock.now += .002
    assert P._CAD.inference_s == pytest.approx(.0036)
    P.at_step(2)
    assert P._CAD.inference_s == pytest.approx(.0036)
    assert not P._CAD.inference


def test_sender_charge_and_baseline_use_the_same_counter_sample(action_run):
    class Sender:
        values = iter([0., .003, .005, .005])

        @property
        def publish_s(self):
            return next(self.values)

    ring = Sender()
    with P._LOCK:
        P._inference_sender(ring)
        P._inference_sender(ring)
        P._inference_sender(ring)
    assert P._CAD.extra == pytest.approx(.005)
    assert P._CAD.inference_sender == (ring, .005)


def test_sender_credit_survives_the_return_to_training(action_run):
    case = action_run
    ring = snapshot_ring(case.clock)
    P.views(ring)
    with P.inference_boundary():
        P.at_step(1)
    ring.publish_s += .003
    P.at_step(2)
    assert not P._CAD.inference and P._CAD.inference_sender is None
    assert P._CAD.extra == pytest.approx(.003)
