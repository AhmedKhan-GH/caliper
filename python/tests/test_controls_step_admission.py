"""CPU checks for step admission and intervention lifecycle timestamps."""
from types import SimpleNamespace

import pytest

import caliper
from caliper import _controls as controls
from caliper import _programs
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
