"""CPU checks for program admission after an exhausted shared cadence."""
from contextlib import nullcontext

import pytest
import torch

from caliper import _programs as P
from caliper._net_ring import NetRing
from caliper.interp import publish
from .test_hosted_cadence import action_run


@pytest.fixture
def exhausted_run(action_run, monkeypatch):
    case = action_run
    ring = NetRing(engine=object())
    P.views(ring)
    case.clock.scheduler_s, case.clock.publish_s = .1, .01
    case.clock.export_s = .3
    publish("output.probs", torch.ones(1))
    step = 0

    def tick(*, hosted=True):
        nonlocal step
        step += 1
        with P.inference_boundary() if hosted else nullcontext():
            P.at_step(step)

    for _ in range(1500):
        tick()
        if case.program.state == "stopped":
            break
    assert case.program.state == "stopped"
    assert case.program.error["code"] == "program.over_budget"
    assert P.ACTIVE and P._VIEWS["ring"] is ring
    assert P._CAD.over_budget(1)
    assert P._CAD.cost

    def compile_program(program):
        program.state, program.handle = "compiled", 2

    monkeypatch.setattr(P, "_prepare", lambda *args: None)
    monkeypatch.setattr(P, "_compile", compile_program)
    monkeypatch.setattr(P, "_relay", lambda *args: None)

    def install(*, piece=False, name="replacement"):
        if piece:
            P.install_piece(name, piece={}, manifest={
                "inputs": [{"name": "probs", "dtype": "f32"}],
                "resources": [], "params": [],
                "derive": {"passes": [{"entry": "main", "workgroups": [1, 1, 1]}]},
                "draw": {},
            }, bind={"probs": "live:output.probs"}, derive="fixture")
        else:
            P.install(name, case.program.decl, "fixture")
        program = P._PROGS[name]
        assert program.state == "compiled", program.error
        return program

    case.tick, case.install = tick, install
    return case


@pytest.mark.parametrize("piece", [False, True])
@pytest.mark.parametrize("hosted", [False, True])
def test_new_program_gets_a_measured_trial_with_viewer(exhausted_run, piece, hosted):
    case = exhausted_run
    history = (list(P._CAD.base), list(P._CAD.cost), P._CAD.n, P._CAD.want,
               P._CAD.ceiling, P._CAD.over)
    program = case.install(piece=piece)
    assert (list(P._CAD.base), list(P._CAD.cost), P._CAD.n, P._CAD.want,
            P._CAD.ceiling, P._CAD.over) == history
    case.clock.export_s = 0.
    case.tick(hosted=hosted)
    assert program.state == "compiled"
    assert program.warm == P.WARMUP_STEPS - 1
    for _ in range(1500):
        case.tick(hosted=hosted)
        if program.state == "stopped" or program.runs >= P.COST_SAMPLES + 3:
            break
    assert program.state == "running"
    assert program.runs >= P.COST_SAMPLES + 3
    assert P._CAD.cost_s() == pytest.approx(case.clock.pass_s)
    assert P._CAD.target == P.VIEWS_SHARE
    assert not P._CAD.over_budget(1)
    assert P._CAD.share() <= P.VIEWS_SHARE


@pytest.mark.parametrize("piece", [False, True])
def test_expensive_replacements_cannot_extend_the_trial(exhausted_run, piece):
    case = exhausted_run
    program = case.install(piece=piece)
    allowance = P._CAD.admission_left
    assert allowance == P.COST_SAMPLES + 1
    total_runs = 0
    for _ in range(2000):
        runs = program.runs
        case.tick()
        total_runs += program.runs - runs
        assert P._CAD.admission_left <= allowance
        allowance = P._CAD.admission_left
        if program.state == "stopped":
            break
        if program.runs:
            program = case.install(piece=piece)
            assert P._CAD.admission_left == allowance
    assert program.state == "stopped"
    assert program.error["code"] == "program.over_budget"
    assert 0 < total_runs <= P.COST_SAMPLES + 2
    assert P._CAD.admission_left == 0
    assert P._CAD.admission_used
    assert P._CAD.n == P._CAD.ceiling
    assert P._CAD.target == P.VIEWS_SHARE
    refused = case.install(piece=piece, name="another")
    case.tick()
    assert refused.state == "stopped" and refused.runs == 0
    assert refused.error["code"] == "program.over_budget"


def test_live_program_and_clock_switch_do_not_renew_trial(exhausted_run):
    case = exhausted_run
    first = case.install()
    for _ in range(1000):
        case.tick()
        if P._CAD.admission_left < P.COST_SAMPLES + 1:
            break
    assert first.runs and 0 < P._CAD.admission_left < P.COST_SAMPLES + 1
    allowance = P._CAD.admission_left
    second = case.install(name="second")
    assert second.state == "compiled" and P._CAD.admission_left == allowance
    history = (list(P._CAD.base), list(P._CAD.cost), P._CAD.n, P._CAD.want, P._CAD.over)
    case.tick(hosted=False)
    assert not P._CAD.inference
    assert P._CAD.admission_left == allowance and P._CAD.admission_used
    assert (list(P._CAD.base), list(P._CAD.cost), P._CAD.n, P._CAD.want, P._CAD.over) == history


def test_rejected_install_does_not_consume_trial(exhausted_run):
    case = exhausted_run
    invalid = {**case.program.decl, "output": {"kind": "invalid"}}
    result = P.install("invalid", invalid, "fixture")
    assert result["state"] == "refused"
    assert not P._CAD.admission_used and P._CAD.admission_left == 0
    assert case.install().state == "compiled"
    assert P._CAD.admission_used and P._CAD.admission_left == P.COST_SAMPLES + 1
