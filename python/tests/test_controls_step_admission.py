"""CPU checks for step admission and intervention lifecycle timestamps."""
from types import SimpleNamespace

import pytest

import caliper
from caliper import _controls as controls
from caliper import _programs, _wire as wire
from python.tests.test_controls_budget import Transport, control, encoded, fill_budget


@pytest.fixture
def ctl(monkeypatch):
    value = controls.Controls()
    monkeypatch.setattr(caliper, "_CTL", value)
    monkeypatch.setattr(controls, "CONTROLS", value)
    monkeypatch.setattr(value._timers, "at", lambda *args: None)
    yield value
    for route in list(value.routes):
        route.detach()
    if value.intervention_service is not None:
        value.intervention_service.remove()


@pytest.mark.parametrize("public", [False, True])
def test_step_budget_refusal_preserves_pending_work(ctl, monkeypatch, public):
    step = caliper.step if public else ctl.step
    step()
    param = ctl.param("value", 0)
    observed = []
    action = ctl.action("inspect", lambda: observed.append(param.value))
    assert control(ctl, "set", name=param.name, value=1)["status"] == "ok"
    assert control(ctl, "run", name=action.name)["status"] == "queued"
    fill_budget(ctl)
    transport = Transport()
    ctl.attach(transport)
    program_steps = []
    monkeypatch.setattr(_programs, "ACTIVE", True)
    monkeypatch.setattr(_programs, "at_step", program_steps.append)
    before = ctl.summary()
    revision, pending, queue = param._version, ctl._npending, list(ctl.queue)
    with pytest.raises(ValueError, match="wire limit"):
        step(10**200)
    assert ctl.summary() == before and ctl.n == 1
    assert (param._version, ctl._npending, list(ctl.queue)) == (revision, pending, queue)
    assert not observed and not program_steps and not transport.frames
    encoded(ctl)
    step()
    assert observed == [1] and ctl.n == 2 and not action.queued
    assert program_steps == ([2] if public else [])
    assert transport.frames
    encoded(ctl)


@pytest.mark.parametrize("public", [False, True])
def test_large_run_counter_remains_supported_with_metadata_space(ctl, public):
    step = caliper.step if public else ctl.step
    step(10**100)
    step()
    assert ctl.n == 10**100 + 1
    assert ctl.controls_meta()["run"]["step"] == ctl.n
    encoded(ctl)


def test_public_step_runs_programs_once_after_control_work(ctl, monkeypatch):
    observed = []
    monkeypatch.setattr(_programs, "ACTIVE", True)
    monkeypatch.setattr(_programs, "at_step", lambda step: observed.append(("program", step)))
    ctl.action("inspect", lambda: observed.append(("action", ctl.n)))
    control(ctl, "run", name="inspect")
    caliper.step(12)
    assert observed == [("action", 12), ("program", 12)]
    assert not ctl.busy
    caliper.step()
    assert observed[-1] == ("program", 13) and ctl.n == 13
    ctl.step()
    assert observed[-1] == ("program", 13) and ctl.n == 14


def test_internal_source_forward_skips_both_step_and_programs(ctl, monkeypatch):
    monkeypatch.setattr(ctl, "_iv_model", SimpleNamespace(
        executor=SimpleNamespace(preparing_source=True)))
    monkeypatch.setattr(_programs, "ACTIVE", True)
    monkeypatch.setattr(_programs, "at_step", lambda step: pytest.fail("internal forward ran a program"))
    monkeypatch.setattr(ctl, "checkpoint", lambda: pytest.fail("internal forward ran a checkpoint"))
    try:
        caliper.step(10**200)
        ctl.step()
        assert ctl.n == 0
    finally:
        ctl._iv_model = None


MAX_STEP = 2**53 - 1
ABLATE = {"point": "norm", "op": "ablate"}


@pytest.fixture
def live(ctl):
    from tests.interp.fixture import make_fixture

    model, _ = make_fixture(layers=1, width=4, heads=1, latents=8, k=2)
    service = ctl.interventions_on(model)

    class CheckedTransport(Transport):
        def __init__(self):
            super().__init__()
            self.errors = []

        def broadcast(self, kind, meta, **kwargs):
            try:
                super().broadcast(kind, meta, **kwargs)
            except (wire.WireError, wire.MessageError) as error:
                self.errors.append(str(error))
                raise

    transport = CheckedTransport()
    route = ctl.attach(transport)
    try:
        yield model, service
        assert not transport.errors
    finally:
        route.detach()


def hooks(model):
    return [(dict(module._forward_pre_hooks), dict(module._forward_hooks))
            for module in model.modules()]


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("target", [-2, MAX_STEP, MAX_STEP + 1, 10**100])
def test_pending_activation_refuses_before_counter_or_hooks(ctl, live, monkeypatch, public, target):
    model, service = live
    step = caliper.step if public else ctl.step
    step(0)
    param = ctl.param("value", 0)
    observed = []
    ctl.action("inspect", lambda: observed.append(param.value))
    control(ctl, "set", name=param.name, value=1)
    control(ctl, "run", name="inspect")
    service.set(ABLATE)
    before, saved, original = ctl.summary(), service.state_dict(), hooks(model)
    program_steps = []
    monkeypatch.setattr(_programs, "ACTIVE", True)
    monkeypatch.setattr(_programs, "at_step", program_steps.append)
    monkeypatch.setattr(service.adapter, "ablate", lambda *args: pytest.fail("refusal installed a hook"))
    with pytest.raises(ValueError, match="since_step"):
        step(target)
    assert ctl.n == 0 and ctl.summary() == before
    assert service.state_dict() == saved and hooks(model) == original
    assert not observed and not program_steps and param._version == 0
    encoded(ctl)


@pytest.mark.parametrize("entry", ["step", "checkpoint"])
def test_scheduled_activation_checks_actual_boundary_timestamp(ctl, live, entry):
    model, service = live
    ctl.step(MAX_STEP - 1)
    service.set(dict(ABLATE, from_step=MAX_STEP))
    if entry == "checkpoint":
        # A checkpoint owner can restore its counter independently of interventions.
        ctl.n = MAX_STEP
    before, saved, original = ctl.summary(), service.state_dict(), hooks(model)
    with pytest.raises(ValueError, match="since_step"):
        caliper.step() if entry == "step" else ctl.checkpoint()
    assert ctl.summary() == before and hooks(model) == original
    assert service.state_dict() == saved
    caliper.step(MAX_STEP - 1)
    assert ctl.controls_meta()["interventions"]["active"][0]["since_step"] == MAX_STEP
    encoded(ctl)


@pytest.mark.parametrize("start", [-1, MAX_STEP - 1])
@pytest.mark.parametrize("restore_active", [False, True])
def test_exact_lifecycle_endpoints_round_trip_and_survive_large_counters(ctl, live, start, restore_active):
    model, service = live
    service.set(ABLATE)
    caliper.step(start)
    record, = service.state_dict()["interventions"]
    assert record["since_step"] == start + 1 and record["state"] == "active"
    assert model.norm._forward_hooks
    saved = service.state_dict()
    service.load_state_dict(saved, restore_active=restore_active)
    ctl.checkpoint()
    assert service.state_dict() == saved
    caliper.step(10**100)
    caliper.step()
    assert service.state_dict() == saved and ctl.n == 10**100 + 1
    encoded(ctl)
    service.configure()
    ended = service.state_dict()
    assert ended["interventions"][0]["since_step"] == start + 1
    assert ended["interventions"][0]["state"] == "ended"
    service.load_state_dict(ended, restore_active=True)
    assert service.state_dict() == ended
    encoded(ctl)


@pytest.mark.parametrize("entry", ["script", "viewer", "restore"])
@pytest.mark.parametrize("counter", [-2, MAX_STEP])
def test_new_activation_admission_refuses_at_invalid_current_step(ctl, live, entry, counter):
    from caliper.interp._intervene import InterventionError

    model, service = live
    ctl.step(counter)
    before, saved, original = ctl.summary(), service.state_dict(), hooks(model)
    if entry == "viewer":
        reply = control(ctl, "intervene", intervene={"set": ABLATE})
        assert reply["error"]["code"] == controls.ERR_BAD_RANGE
        # The request counter counts refused viewer traffic.
        before["controls_in"] = before.get("controls_in", 0) + 1
    else:
        with pytest.raises(InterventionError, match="since_step") as caught:
            if entry == "script":
                service.set(ABLATE)
            else:
                service.load_state_dict({"version": 1, "interventions": [{
                    "specification": ABLATE, "state": "pending", "since_step": None,
                    "clear": False, "error": None, "error_code": None}]})
        assert caught.value.code == controls.ERR_BAD_RANGE
    assert ctl.summary() == before and service.state_dict() == saved
    assert hooks(model) == original
    encoded(ctl)


@pytest.mark.parametrize("counter", [-2, MAX_STEP, 10**100])
def test_invalid_override_preserves_applied_effect_and_counter(ctl, live, counter):
    from caliper.interp._intervene import InterventionError

    model, service = live
    service.set(ABLATE)
    ctl.step(1)
    ctl.step(counter)
    before, saved, original = ctl.summary(), service.state_dict(), hooks(model)
    with pytest.raises(InterventionError):
        service.apply_overrides([ABLATE])
    assert ctl.summary() == before and service.state_dict() == saved
    assert hooks(model) == original
    service.apply_overrides([])
    assert ctl.summary() == before and service.state_dict() == saved
    encoded(ctl)


@pytest.mark.parametrize("counter", [-1, MAX_STEP - 1])
def test_override_uses_exact_endpoint_without_advancing_counter(ctl, live, counter):
    _, service = live
    ctl.step(counter)
    before = ctl.n
    service.apply_overrides([ABLATE])
    assert ctl.n == before
    record, = service.state_dict()["interventions"]
    assert record["since_step"] == counter + 1 and record["state"] == "active"
    saved = service.state_dict()
    service.load_state_dict(saved, restore_active=True)
    assert service.state_dict() == saved
    encoded(ctl)


def test_negative_counter_can_queue_a_future_override(ctl, live):
    _, service = live
    caliper.step(-2)
    service.apply_overrides([dict(ABLATE, from_step=0)])
    record, = service.state_dict()["interventions"]
    assert record["state"] == "pending" and record["since_step"] is None
    caliper.step(-1)
    record, = service.state_dict()["interventions"]
    assert record["state"] == "active" and record["since_step"] == 0
    encoded(ctl)


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("operation", ["clear", "expire"])
def test_large_boundary_allows_clear_and_expiration(ctl, live, pending, operation):
    _, service = live
    service.set(dict(ABLATE, from_step=10 if pending else None, until_step=20))
    caliper.step()
    if operation == "clear":
        service.clear()
    caliper.step(10**100)
    assert ctl.n == 10**100 and not service.specifications()
    saved = service.state_dict()
    if operation == "expire":
        record, = saved["interventions"]
        assert record["state"] == "ended"
        assert record["since_step"] == (None if pending else 2)
    else:
        assert saved["interventions"] == []
    service.load_state_dict(saved)
    assert service.state_dict() == saved
    encoded(ctl)


def test_applied_expiration_requires_a_new_timestamp_for_replacement(ctl, live):
    model, service = live
    service.set(dict(ABLATE, until_step=5))
    caliper.step(1)
    service.set(ABLATE)
    before, saved, original = ctl.summary(), service.state_dict(), hooks(model)
    with pytest.raises(ValueError, match="since_step"):
        caliper.step(MAX_STEP)
    assert ctl.summary() == before and service.state_dict() == saved
    assert hooks(model) == original
    caliper.step(4)
    record, = service.state_dict()["interventions"]
    assert record["state"] == "active" and record["since_step"] == 5
    encoded(ctl)


def test_active_replacement_preserves_valid_timestamp_at_large_counter(ctl, live):
    _, service = live
    addition = {"point": "layers.0.mlp_out", "kind": "add", "vector": [1.] * 4, "coefficient": 1}
    service.set(addition)
    caliper.step(1)
    record = next(iter(ctl._ivs.values()))
    old = record.hook
    caliper.step(10**100)
    service.set(dict(addition, coefficient=2))
    caliper.step()
    assert old.removed and record.applied_spec["coefficient"] == 2
    assert record.since_step == 2 and ctl.n == 10**100 + 1
    saved = service.state_dict()
    service.load_state_dict(saved, restore_active=True)
    assert service.state_dict() == saved
    encoded(ctl)


def test_backward_counter_preserves_future_schedule(ctl, live):
    _, service = live
    service.set(dict(ABLATE, from_step=0))
    caliper.step(1)
    caliper.step(-2)
    record, = service.state_dict()["interventions"]
    assert record["state"] == "pending" and record["since_step"] is None
    caliper.step(-1)
    record, = service.state_dict()["interventions"]
    assert record["state"] == "active" and record["since_step"] == 0
    encoded(ctl)


def test_large_counter_reserves_only_valid_lifecycle_timestamps(ctl, live):
    _, service = live
    service.set(ABLATE)
    caliper.step(1)
    saved = service.state_dict()
    pad = fill_budget(ctl)
    # This releases 192 JSON bytes. The run counter grows by 185 reserved bytes.
    assert len(pad.value) > 32
    pad.set(pad.value[:-32])
    caliper.step(10**200)
    assert ctl.n == 10**200 and service.state_dict() == saved
    encoded(ctl)


@pytest.mark.parametrize("counter", [-5, True, 1.5])
def test_legacy_run_counter_values_keep_their_wire_representation(ctl, counter):
    caliper.step(counter)
    assert ctl.n == counter and "step" not in ctl.controls_meta()["run"]
    encoded(ctl)


def test_future_schedule_admission_remains_valid_at_negative_counter(ctl, live):
    _, service = live
    caliper.step(-5)
    service.set(dict(ABLATE, from_step=0))
    saved = service.state_dict()
    service.load_state_dict(saved)
    assert service.state_dict() == saved
    caliper.step()
    assert service.state_dict() == saved and ctl.n == -4
    caliper.step(-1)
    record, = service.state_dict()["interventions"]
    assert record["since_step"] == 0 and record["state"] == "active"
    encoded(ctl)
