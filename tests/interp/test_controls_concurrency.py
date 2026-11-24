"""Bounded CPU checks for intervention transactions and control traffic."""
from copy import deepcopy
from contextlib import contextmanager
import os
import signal
from threading import Event, Thread

import pytest
import torch

from caliper.interp._intervene import InterventionError
from tests.interp.concurrency_isolation import isolated_test, run_isolated_test
from tests.interp.test_intervene import add_spec, controlled
from tests.interp.test_intervention_replay import ReplayTransport, hook_objects, snapshot


PATCH = {"kind": "patch", "point": "layers.1.resid_pre", "source_prompt": "xyz"}


class Job:
    """Own one thread and retain its result or exception."""

    def __init__(self, function):
        self.done = Event()
        self.result = self.error = None

        def run():
            try:
                self.result = function()
            except BaseException as error:
                self.error = error
            finally:
                self.done.set()

        self.thread = Thread(target=run)
        self.thread.start()

    def join(self):
        self.thread.join(10)
        assert not self.thread.is_alive(), "fixture thread did not finish"

    def success(self):
        self.join()
        if self.error is not None:
            raise self.error
        return self.result


@contextmanager
def source_gate(model, function):
    entered, release = Event(), Event()

    def wait(*args):
        entered.set()
        assert release.wait(5), "source fixture exceeded its finite wait"

    observer = model.register_forward_pre_hook(wait)
    job = Job(function)
    try:
        assert entered.wait(5), "source fixture did not reach its barrier"
        yield job
    finally:
        release.set()
        try:
            job.join()
        finally:
            observer.remove()


def other_thread(function):
    job = Job(function)
    try:
        assert job.done.wait(2), "control traffic waited for source preparation"
        return job.success()
    finally:
        job.join()


def operation_call(ctl, service, operation, specs):
    if operation == "boundary":
        for spec in specs:
            service.set(spec)
        return ctl.step
    if operation == "restore":
        return lambda: service.load_state_dict(snapshot(*deepcopy(specs)), restore_active=True)
    return lambda: service.apply_overrides(deepcopy(specs))


@pytest.mark.parametrize("blocked_teardown", [False, True])
def test_isolation_retains_failure_and_stops_child(tmp_path, blocked_teardown):
    probe = tmp_path / "test_isolation_probe.py"
    probe.write_text('''from threading import Event, Lock, Thread
import os
import signal
import pytest

@pytest.fixture
def held_lock():
    if BLOCKED_TEARDOWN and os.name != "nt":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    lock, entered, release = Lock(), Event(), Event()

    def hold():
        with lock:
            entered.set()
            release.wait(30)

    thread = Thread(target=hold)
    thread.start()
    assert entered.wait(2), "injected worker did not start"
    yield lock
    print("child fixture teardown entered", flush=True)
    if BLOCKED_TEARDOWN:
        assert lock.acquire(timeout=30), "injected cleanup wait expired"
        lock.release()
    print("child fixture teardown finished", flush=True)

def test_failure(held_lock):
    assert False, "injected child assertion survives"
'''.replace("BLOCKED_TEARDOWN", repr(blocked_teardown)), encoding="utf-8")
    result = run_isolated_test(f"{probe}::test_failure", tmp_path / "probe", timeout=10)
    assert result.timed_out, result.output
    assert result.returncode != 0, result.output
    if os.name != "nt":
        assert result.returncode == -(signal.SIGKILL if blocked_teardown else signal.SIGTERM)
    assert "injected child assertion survives" in result.output
    assert "child fixture teardown entered" in result.output
    assert "Timeout (" in result.output and "in hold" in result.output
    if blocked_teardown:
        assert "child fixture teardown finished" not in result.output
    else:
        assert "child fixture teardown finished" in result.output
        assert "1 failed" in result.output


@isolated_test("controlled", "monkeypatch")
def test_source_gate_removes_observer_when_join_fails(controlled, monkeypatch):
    model, tokenizer, _, _ = controlled
    hooks = hook_objects(model)
    with pytest.raises(AssertionError, match="injected join failure"):
        with source_gate(model, lambda: model(**tokenizer("ABC"))) as job:
            join = job.join

            def fail_join():
                join()
                raise AssertionError("injected join failure")

            monkeypatch.setattr(job, "join", fail_join)
    assert hook_objects(model) == hooks
    assert job.done.is_set() and job.error is None


@pytest.mark.parametrize("operation", ["boundary", "restore", "override"])
def test_source_forward_can_inspect_controls(controlled, operation):
    model, _, ctl, service = controlled
    observed = []

    def inspect(*args):
        # Refuse the old path before a callback can enter an unbounded deadlock.
        assert ctl.lock.acquire(blocking=False), "source forward holds Controls.lock"
        ctl.lock.release()
        observed.append((service.specifications(), service.state_dict()))

    observer = model.register_forward_pre_hook(inspect)
    try:
        if operation == "boundary":
            service.set(PATCH)
            ctl.step()
            assert service.specifications(), ctl.summary()
        elif operation == "restore":
            service.load_state_dict(snapshot(deepcopy(PATCH)), restore_active=True)
        else:
            service.apply_overrides([deepcopy(PATCH)])
        assert len(observed) == 1
    finally:
        observer.remove()


@pytest.mark.parametrize("operation", ["boundary", "restore", "override"])
@isolated_test("controlled")
def test_reads_and_harmless_network_control_complete_during_source(controlled, operation):
    model, _, ctl, service = controlled
    ctl.step()
    param = ctl.param("coefficient", 1.0)
    route = ctl.attach(ReplayTransport())
    call = operation_call(ctl, service, operation, [PATCH])
    prior = service.state_dict()

    def controls():
        assert service.state_dict() == prior
        assert service.specifications() == []
        reply = route.on_control(None, {"request_id": 1, "op": "resume"})
        assert reply["status"] == "ok"
        reply = route.on_control(None, {"request_id": 2, "op": "set", "name": param.name, "value": 2.0})
        assert reply["status"] == "ok"

    try:
        with source_gate(model, call) as job:
            other_thread(controls)
            assert not job.done.is_set()
        job.success()
        assert service.specifications()[0]["source_prompt"] == "xyz"
        assert param.pending == 2.0
    finally:
        route.detach()


@pytest.mark.parametrize("edit", ["replace", "clear", "clear_all", "new"])
@isolated_test("controlled")
def test_boundary_conflict_keeps_old_effect_and_queues_next_boundary(controlled, edit):
    model, tokenizer, ctl, service = controlled
    old = dict(PATCH, source_prompt="abc")
    service.set(old)
    ctl.step()
    before = model(**tokenizer("ABC")).logits
    record = ctl._ivs[(PATCH["point"], "patch")]
    hook = record.hook
    sources = dict(service.executor._cache)
    service.set(PATCH)
    route = ctl.attach(ReplayTransport())

    def queue():
        request = ({"set": dict(PATCH, source_prompt="ABC")} if edit == "replace" else
                   {"clear": PATCH["point"]} if edit == "clear" else
                   {"clear_all": True} if edit == "clear_all" else {"set": add_spec()})
        reply = route.on_control(None, {"request_id": 1, "op": "intervene", "intervene": request})
        assert reply["status"] == "ok", reply
        return service.state_dict()

    try:
        with source_gate(model, ctl.step) as job:
            queued = other_thread(queue)
        job.success()
        assert service.state_dict() == queued
        assert record.hook is hook and not hook.removed
        assert all(service.executor._cache[key] is value for key, value in sources.items())
        assert len(service.executor._cache) == len(sources)
        torch.testing.assert_close(model(**tokenizer("ABC")).logits, before, rtol=0, atol=0)
        ctl.step()
        assert hook.removed
        specs = service.specifications()
        if edit in ("clear", "clear_all"):
            assert specs == [] and not service.executor._cache
        else:
            assert specs[0]["source_prompt"] == ("ABC" if edit == "replace" else "xyz")
            assert len(specs) == (1 if edit == "replace" else 2)
    finally:
        route.detach()


@pytest.mark.parametrize("operation", ["restore", "override"])
@pytest.mark.parametrize("edit", ["replace", "clear", "new"])
@isolated_test("controlled")
def test_restore_conflict_rolls_back_and_retains_concurrent_requests(controlled, operation, edit):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    service.set(dict(PATCH, source_prompt="abc"))
    ctl.step()
    before = model(**tokenizer("ABC")).logits
    originals = list(ctl._ivs.items())
    hooks = hook_objects(model)
    cache, identity = dict(service.executor._cache), service.executor._cache_identity
    call = operation_call(ctl, service, operation, [add_spec(coefficient=2), PATCH])

    def queue():
        if edit == "clear":
            service.clear()
        else:
            service.set(add_spec(**({"coefficient": 3} if edit == "replace" else
                                   {"point": "layers.3.mlp_out"})))
        return service.state_dict()

    with source_gate(model, call) as job:
        queued = other_thread(queue)
    assert isinstance(job.error, InterventionError) and "state changed" in str(job.error)
    assert service.state_dict() == queued
    assert all(ctl._ivs[key] is record for key, record in originals)
    assert hook_objects(model) == hooks
    assert service.executor._cache_identity is identity
    assert list(service.executor._cache) == list(cache)
    assert all(service.executor._cache[key] is value for key, value in cache.items())
    torch.testing.assert_close(model(**tokenizer("ABC")).logits, before, rtol=0, atol=0)
    ctl.step()
    specs = service.specifications()
    if edit == "clear":
        assert not specs
    elif edit == "replace":
        assert specs[0]["coefficient"] == 3
    else:
        assert specs[-1]["point"] == "layers.3.mlp_out"


@pytest.mark.parametrize("operation", ["boundary", "restore", "override"])
@isolated_test("controlled")
def test_detach_cancels_preparation_and_owns_deferred_cleanup(controlled, operation):
    model, _, ctl, service = controlled
    ctl.step()
    no_effects = hook_objects(model)
    service.set(add_spec())
    ctl.step()
    old = next(iter(ctl._ivs.values())).hook
    call = operation_call(ctl, service, operation, [PATCH])
    with source_gate(model, call) as job:
        other_thread(service.remove)
        assert ctl.intervention_service is None and not ctl._ivs
        assert not old.removed
        with pytest.raises(InterventionError, match="operation is in progress"):
            ctl.interventions_on(model)
    if operation == "boundary":
        job.success()
    else:
        assert isinstance(job.error, InterventionError) and "state changed" in str(job.error)
    assert old.removed and not service.executor._cache
    assert all(hook.removed for hook in service.executor._hooks)
    assert service.executor.dictionary is None and service.executor.tokenizer is None
    assert hook_objects(model) == no_effects and not ctl._iv_retiring
    replacement = ctl.interventions_on(model)
    replacement.remove()


@pytest.mark.parametrize("operation", ["restore", "override"])
def test_source_failure_restores_exact_hooks_records_and_cache(controlled, operation):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    service.set(dict(PATCH, source_prompt="abc"))
    ctl.step()
    before = model(**tokenizer("ABC")).logits
    prior = service.state_dict()
    records, original = ctl._ivs, list(ctl._ivs.items())
    hooks = hook_objects(model)
    cache, identity = dict(service.executor._cache), service.executor._cache_identity

    def fail(*args):
        assert ctl.lock.acquire(blocking=False)
        ctl.lock.release()
        raise RuntimeError("fixture source failure")

    observer = model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="fixture source failure"):
            operation_call(ctl, service, operation, [add_spec(coefficient=2), PATCH])()
    finally:
        observer.remove()
    assert ctl._ivs is records and list(records.items()) == original
    assert service.state_dict() == prior and hook_objects(model) == hooks
    assert service.executor._cache_identity is identity
    assert list(service.executor._cache) == list(cache)
    assert all(service.executor._cache[key] is value for key, value in cache.items())
    torch.testing.assert_close(model(**tokenizer("ABC")).logits, before, rtol=0, atol=0)


@pytest.mark.parametrize("action", ["configure", "restore", "override", "replay", "watch", "invalidate"])
@isolated_test("controlled")
def test_conflicting_lifecycle_operations_refuse_without_wait(controlled, action):
    model, _, ctl, service = controlled
    ctl.step()
    service.set(PATCH)

    def attempt():
        with pytest.raises(InterventionError, match="operation is in progress"):
            if action == "configure":
                service.configure()
            elif action == "restore":
                service.load_state_dict(snapshot())
            elif action == "override":
                service.apply_overrides([add_spec()])
            elif action == "replay":
                with service.temporary_state([]):
                    pytest.fail("concurrent replay entered")
            elif action == "watch":
                ctl.interventions_on(model)
            else:
                service.invalidate_source_cache()

    with source_gate(model, ctl.step) as job:
        other_thread(attempt)
    job.success()
    assert service.specifications()[0]["source_prompt"] == "xyz"


@pytest.mark.parametrize("detach", [False, True])
@isolated_test("controlled", "monkeypatch")
def test_network_dictionary_validation_releases_lock_and_pins_resources(controlled, monkeypatch, detach):
    _, tokenizer, ctl, service = controlled
    service.configure(dictionary=service.dictionary, tokenizer=tokenizer)
    ctl.step()
    entered, release = Event(), Event()
    original = service.executor._dictionary_matches

    def validate():
        assert ctl.lock.acquire(blocking=False), "dictionary validation holds Controls.lock"
        ctl.lock.release()
        entered.set()
        assert release.wait(5), "dictionary fixture exceeded its finite wait"
        return original()

    monkeypatch.setattr(service.executor, "_dictionary_matches", validate)
    route = ctl.attach(ReplayTransport())
    spec = {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 1}
    job = Job(lambda: route.on_control(None, {"request_id": 1, "op": "intervene", "intervene": {"set": spec}}))
    try:
        assert entered.wait(5)

        def controls():
            assert service.state_dict() == snapshot()
            assert route.on_control(None, {"request_id": 2, "op": "resume"})["status"] == "ok"
            with pytest.raises(InterventionError, match="operation is in progress"):
                service.configure()
            if detach:
                service.remove()
                assert service.executor.dictionary is not None

        other_thread(controls)
    finally:
        release.set()
        try:
            job.join()
        finally:
            route.detach()
    reply = job.success()
    if detach:
        assert reply["status"] == "error" and "no longer attached" in reply["error"]["message"]
        assert service.executor.dictionary is None and not ctl._iv_retiring
    else:
        assert reply["status"] == "ok"
        ctl.step()
        assert service.specifications()[0]["feature"] == 0


@pytest.mark.parametrize("operation", ["restore", "override"])
@isolated_test("controlled", "monkeypatch")
def test_restore_rechecks_metadata_after_concurrent_parameter_admission(controlled, monkeypatch, operation):
    from caliper import _controls
    from python.tests.test_controls_budget import fill_budget

    model, _, ctl, service = controlled
    ctl.step()
    monkeypatch.setattr(_controls, "MAX_META_BYTES", 40000)
    service.set(add_spec())
    ctl.step()
    prior, hooks = service.state_dict(), hook_objects(model)
    with source_gate(model, operation_call(ctl, service, operation, [add_spec(), PATCH])) as job:
        other_thread(lambda: fill_budget(ctl))
    assert isinstance(job.error, InterventionError) and job.error.code == "limit"
    assert service.state_dict() == prior and hook_objects(model) == hooks
    assert not service.executor._cache
    with ctl.lock:
        ctl._check_metadata_budget()


@isolated_test("controlled")
def test_due_activation_retains_overlap_order_before_conflicting_replacement(controlled):
    model, tokenizer, ctl, service = controlled
    ablation = {"kind": "ablate", "point": "layers.0.mlp_out", "from_step": 10}
    service.set(ablation)
    service.set(add_spec())
    ctl.step()
    old = ctl._ivs[("layers.0.mlp_out", "add")].hook
    service.set(PATCH)
    with source_gate(model, lambda: ctl.step(9)) as job:
        other_thread(lambda: service.set(add_spec(coefficient=2)))
    job.success()
    assert old.removed
    ablation = ctl._ivs[("layers.0.mlp_out", "ablate")]
    addition = ctl._ivs[("layers.0.mlp_out", "add")]
    assert ablation.state == "active" and ablation.since_step == 10
    assert addition.applied_spec["coefficient"] == 1 and addition.spec["coefficient"] == 2

    def observed_output():
        values = []
        observer = model.layers[0].mlp.register_forward_hook(lambda m, a, y: values.append(y.detach().clone()))
        try:
            model(**tokenizer("ABC"))
        finally:
            observer.remove()
        assert len(values) == 1
        return values[0]

    current = observed_output()
    torch.testing.assert_close(current, torch.full_like(current, 0.1), rtol=0, atol=0)
    ctl.step(9)
    current = observed_output()
    torch.testing.assert_close(current, torch.full_like(current, 0.2), rtol=0, atol=0)


def test_boundary_source_failure_ends_only_failed_record(controlled):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    old = ctl._ivs[("layers.0.mlp_out", "add")].hook
    before = model(**tokenizer("ABC")).logits
    service.set(PATCH)

    def fail(*args):
        raise RuntimeError("fixture source failure")

    observer = model.register_forward_pre_hook(fail)
    try:
        with pytest.warns(RuntimeWarning, match="fixture source failure"):
            ctl.step()
    finally:
        observer.remove()
    assert ctl._ivs[("layers.0.mlp_out", "add")].hook is old and not old.removed
    failed = ctl._ivs[(PATCH["point"], "patch")]
    assert failed.state == "ended" and failed.hook is None and not service.executor._cache
    torch.testing.assert_close(model(**tokenizer("ABC")).logits, before, rtol=0, atol=0)


@isolated_test("controlled", "monkeypatch")
def test_cancelled_boundary_does_not_report_or_reset_linger(controlled, monkeypatch):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    old = next(iter(ctl._ivs.values())).hook
    ctl._alone_since, ctl._linger_next = float("-inf"), 0
    entered, release = Event(), Event()
    original = ctl._prepare_interventions

    def prepare(*args):
        entered.set()
        assert release.wait(5), "boundary fixture exceeded its finite wait"
        return original(*args)

    monkeypatch.setattr(ctl, "_prepare_interventions", prepare)
    job = Job(ctl.step)
    try:
        assert entered.wait(5)
        other_thread(lambda: service.set(add_spec(coefficient=2)))
    finally:
        release.set()
        job.join()
    job.success()
    assert not old.removed and service.specifications()[0]["coefficient"] == 1
    assert ctl.counters["interventions_lingered"] == 0
    assert ctl._alone_since == float("-inf")
    ctl._linger_next = 0
    with pytest.warns(RuntimeWarning, match="no connection"):
        ctl.step()
    assert not service.specifications() and old.removed
    assert ctl.counters["interventions_lingered"] == 1


def test_request_during_shape_refresh_retains_admitted_point_metadata(controlled, monkeypatch):
    model, tokenizer, ctl, service = controlled
    model(**tokenizer("ABC"))
    original = service.adapter.points
    expected = list(original(model))
    assert any(point["shape"] is not None for point in expected)

    def points(model):
        service.set(add_spec(from_step=100))
        return original(model)

    monkeypatch.setattr(service.adapter, "points", points)
    ctl.step()
    assert service.points == expected
    assert service.specifications() == []
    assert service.specifications(active_only=False)[0]["from_step"] == 100
    assert service._shapes_off is None
