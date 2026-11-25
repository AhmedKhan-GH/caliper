"""Driver completion with CPU data and file resources, without child processes."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_circuit_trace import driver, inference_fixture, qualification_reply, steering_case
from test_gemma_acceptance import gemma_reply
from _checks import check, finish
from _mcp import Failure


@pytest.fixture
def completion_case(tmp_path, monkeypatch, steering_case):
    import parity

    prompt, prior, status, params = inference_fixture()
    graph = steering_case["graph"]
    graph["nodes"][1]["effect"] = 4.64
    graph["nodes"].extend({"kind": "token", "index": i} for i in range(4, 20))
    examples = {"examples": [dict(row, valid=[False] * 4 + [True] + [False] * 24,
                                  token_ids=[1] * 29, activations=[row["activation"]] * 29)
                              for row in steering_case["examples"]["examples"]]}
    state = SimpleNamespace(out=tmp_path / "result", sessions=[], calls=[], publications=[],
                            family="gemma", show=False, fault=None, stage=None,
                            cleanup_fault=None, cleanup_report=None, result=None,
                            parity_checks=[check(name, True, True, True)
                                           for name in ("fixture_parity", "parity_process_memory")])
    real_close, real_acceptance, real_save = driver.Session.close, driver.acceptance, driver.save

    def checkpoint(stage):
        state.calls.append(stage)
        if state.stage == stage:
            raise state.fault

    class FakeSession:
        def __init__(self, args, out, *, show=False):
            self.out, self.show = out, show
            out.mkdir()
            self.app = None
            self.started = False
            self.run_id = "circuit-trace"
            self.cleanup_error = None
            self.app_log = (out / "app.log").open("w")
            self.mcp_closed = False
            self.mcp = SimpleNamespace(body=self.body, close=self.close_mcp)
            self.events = SimpleNamespace(complete=self.complete, until=lambda name, read: read(), history=[])
            self.program_evidence = {"before": {}, "after": {}}
            self.steered = self.generated = False
            self.dictionary = gemma_reply() if args.family == "gemma" else qualification_reply()
            state.sessions.append(self)

        def complete(self, method, **kwargs):
            if method == "trace.run":
                checkpoint(self.out.name + ".trace")
                return {"step": 1, "seconds": 0}
            if method == "trace.status":
                inference = deepcopy(status["inference"] if self.generated else prior)
                inference["comparison"] = {key: inference[key] for key in ("prompt_hash", "interventions_hash")}
                inference["comparison"].update(argmax_token_ids=[1, 1], completed_step=14 if self.steered else 9)
                return {"state": "done", "counts": {"nodes": len(graph["nodes"])}, "inference": inference}
            if method == "feature.dictionary":
                return self.dictionary
            if method == "feature.examples":
                return examples
            if method == "trace.validate":
                checkpoint("validation")
                return steering_case["validation"]
            raise AssertionError(method)

        def body(self, method, **kwargs):
            if method == "control.set":
                return {}
            if method == "run.action":
                checkpoint(kwargs["name"])
                if kwargs["name"] == "generate":
                    self.generated = True
                elif kwargs["name"] == "reset":
                    self.steered = False
                return {}
            if method == "run.state":
                return {"params": [{"name": name, "value": value} for name, value in params.items()]}
            if method == "applet.save":
                checkpoint("applet.save")
                return {}
            if method == "spec.get":
                return {"text": "# CPU Fixture\n"}
            raise AssertionError(method)

        def read(self, program, resource, **kwargs):
            return {"step": 14 if self.steered else 9, "elements": [
                [0, 0, 0, 0, .5 if token == 1 else .05,
                 (.3 if self.steered else .5) if token == 1 else .05, token, token - 1]
                for token in range(1, 11)]}

        def example_page(self, page, **kwargs):
            rows, cells = [], []
            for i, row in enumerate(examples["examples"][page * 8:page * 8 + 8]):
                rows.append([row["example_id"], page * 8 + i, i, 1])
                cells.extend([0, 0, 0, 0, row["activation"], 1, i, col == 0] for col in range(25))
            return {"step": 9, "elements": rows}, {"step": 9, "elements": cells}

        def ablate_selected(self, prompt, **kwargs):
            self.steered = True
            return self.read("probabilities", "cells"), self.complete("trace.status")["inference"]["comparison"]

        def ready_programs(self, identifiers):
            pass

        def capture(self, name, steps):
            return self.out / (name + ".png"), None if self.show else {"name": name}

        def live(self):
            return not self.app_log.closed

        def close_mcp(self):
            self.mcp_closed = True

        def close(self):
            state.calls.append(self.out.name + ".close")
            if self.out.name == "acceptance" and state.cleanup_fault is not None:
                raise state.cleanup_fault
            real_close(self)
            if self.out.name == "acceptance":
                self.cleanup_error = state.cleanup_report

    def acceptance(session, args, result):
        state.result = result
        return real_acceptance(session, args, result)

    def capture_checks(folder, metadata, png, graph, bars, **kwargs):
        if metadata["name"] == "before":
            assert state.result["pixel_evidence_collected"] is True
            assert "validation" not in state.result
            assert not state.result["acceptance_complete"]
            checkpoint("capture")
        return [check("synthetic_pixels", True, True, True)], {
            "row_ids": list(range(8)), "height": 30 if metadata["name"] == "after" else 50}

    def save(path, value):
        real_save(path, value)
        if path == state.out / "result.json":
            state.publications.append(deepcopy(value))

    def run_parity(out):
        assert state.result["acceptance_complete"] and not state.result["completed"]
        assert all(s.app_log.closed and s.mcp_closed for s in state.sessions)
        checkpoint("parity")
        return state.parity_checks

    def keep_alive(bridge):
        assert not bridge.session.app_log.closed
        assert not state.publications[-1]["passed"]
        checkpoint("keep_alive")

    def run():
        model = {"gemma": tmp_path / "models--google--gemma-2-2b" / "snapshots" / driver.GEMMA_REVISION,
                 "qwen": tmp_path / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / driver.MODEL_REVISION,
                 "fixture": "tiny-lm"}[state.family]
        return driver.main(["--model", str(model), "--dictionary", str(tmp_path / "dictionary"),
                            "--prompt", prompt, "--out", str(state.out), "--show" if state.show else "--hidden"])

    state.run = run
    state.read = lambda: json.loads((state.out / "result.json").read_text())
    monkeypatch.setattr(driver, "Session", FakeSession)
    monkeypatch.setattr(driver, "acceptance", acceptance)
    monkeypatch.setattr(driver, "save", save)
    monkeypatch.setattr(driver, "instrument", lambda session: None)
    monkeypatch.setattr(driver, "programs", lambda session: None)
    monkeypatch.setattr(driver, "read_graph", lambda session: graph)
    monkeypatch.setattr(driver, "memory_bytes", lambda folder: 1024)
    monkeypatch.setattr(driver, "program_budget_checks", lambda snapshot: [check("synthetic_budget", True, 0, 1)])
    monkeypatch.setattr(driver, "capture_checks", capture_checks)
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: 0))
    monkeypatch.setattr(driver.Bridge, "select", lambda bridge, row: examples)
    monkeypatch.setattr(driver.Bridge, "details", lambda bridge, expanded: setattr(bridge, "details_expanded", expanded))
    monkeypatch.setattr(driver.Bridge, "keep_alive", keep_alive)
    monkeypatch.setattr(parity, "export_resources", lambda session, out: checkpoint("parity.export"))
    monkeypatch.setattr(parity, "run_parity", run_parity)
    yield state
    for session in state.sessions:
        real_close(session)


@pytest.mark.parametrize("family", ["fixture", "qwen", "gemma"])
@pytest.mark.parametrize("completion", [None, False, 1, "true"])
def test_finish_requires_explicit_completion(family, completion):
    result = {"mode": family, "checks": [check("partial", True, True, True)],
              "dictionary_evidence": gemma_reply() if family == "gemma" else qualification_reply(),
              "pixel_evidence_collected": True, "passed": True, "gemma_acceptance": True,
              "qwen_acceptance": True, "qwen_full_mlp_acceptance": True}
    if completion is not None:
        result["completed"] = completion
    assert finish(result) == 1
    assert not result["passed"] and not result["gemma_acceptance"] and not result["qwen_acceptance"]
    assert not result["qwen_full_mlp_acceptance"] and result["qwen_acceptance_scope"] is None


@pytest.mark.parametrize("family", ["fixture", "qwen", "gemma"])
def test_complete_driver_publishes_success_after_cleanup(completion_case, family):
    case = completion_case
    case.family = family
    assert case.run() == 0
    result = case.read()
    assert result["completed"] and result["acceptance_complete"] and result["passed"]
    assert result["gemma_acceptance"] is (family == "gemma")
    assert result["qwen_acceptance"] is (family == "qwen")
    gates = {gate["name"] for gate in result["checks"]}
    assert {"validation", "generation", "reset", "dictionary_identity_stable", "process_memory"} <= gates
    assert all(s.app_log.closed and s.mcp_closed for s in case.sessions)
    assert all(not report["passed"] for report in case.publications[:-1])
    if family == "fixture":
        assert result["parity_evidence_collected"] and "parity" in case.calls
    assert "error" not in result


@pytest.mark.parametrize("fault,detail", [(KeyboardInterrupt(), "KeyboardInterrupt"),
                                         (KeyboardInterrupt("capture stopped"), "KeyboardInterrupt: capture stopped"),
                                         (SystemExit(), "SystemExit"), (SystemExit(7), "SystemExit: 7")])
def test_capture_interruption_preserves_error_and_cleanup(completion_case, fault, detail):
    case = completion_case
    case.stage, case.fault = "capture", fault
    with pytest.raises(type(fault)) as raised:
        case.run()
    assert raised.value is fault
    result = case.read()
    assert result["pixel_evidence_collected"] and result["dictionary_qualification"]["qualified"]
    assert not result["completed"] and not result["acceptance_complete"] and not result["passed"]
    assert not result["gemma_acceptance"] and not result["qwen_acceptance"]
    assert result["error"] == detail
    assert "validation" not in case.calls and "generate" not in case.calls and "reset" not in case.calls
    assert all(s.app_log.closed and s.mcp_closed for s in case.sessions)
    assert all(not report["passed"] for report in case.publications)


@pytest.mark.parametrize("stage", ["generate", "reset", "applet.save", "parity.export", "parity"])
def test_late_failure_prevents_completion(completion_case, stage):
    case = completion_case
    case.family = "fixture" if stage.startswith("parity") else "gemma"
    case.stage, case.fault = stage, Failure("late stage failed")
    assert case.run() == 1
    result = case.read()
    assert not result["completed"] and not result["passed"] and not result["gemma_acceptance"]
    assert result["acceptance_complete"] is stage.startswith("parity")
    assert result["error"] == "Failure: late stage failed"
    assert all(s.app_log.closed and s.mcp_closed for s in case.sessions)


@pytest.mark.parametrize("fault,detail", [(RuntimeError("cleanup failed"), "RuntimeError: cleanup failed"),
                                         (KeyboardInterrupt(), "KeyboardInterrupt"), (SystemExit(), "SystemExit")])
def test_cleanup_exception_prevents_success(completion_case, fault, detail):
    case = completion_case
    case.cleanup_fault = fault
    if isinstance(fault, Exception):
        assert case.run() == 1
    else:
        with pytest.raises(type(fault)) as raised:
            case.run()
        assert raised.value is fault
    result = case.read()
    assert result["acceptance_complete"] and not result["completed"]
    assert not result["passed"] and not result["gemma_acceptance"]
    assert result["cleanup_error"] == result["error"] == detail
    assert not case.sessions[-1].app_log.closed


def test_cleanup_report_preserves_original_interruption(completion_case):
    case = completion_case
    case.stage, case.fault = "capture", KeyboardInterrupt()
    case.cleanup_report = "owned cleanup failed"
    with pytest.raises(KeyboardInterrupt):
        case.run()
    result = case.read()
    assert result["error"] == "KeyboardInterrupt"
    assert result["cleanup_error"] == "Failure: owned cleanup failed"
    assert not result["completed"] and not result["passed"] and not result["gemma_acceptance"]


def test_cleanup_report_blocks_an_otherwise_complete_driver(completion_case):
    case = completion_case
    case.cleanup_report = "owned cleanup failed"
    assert case.run() == 1
    result = case.read()
    assert result["acceptance_complete"] and not result["completed"]
    assert not result["passed"] and not result["gemma_acceptance"]
    assert result["cleanup_error"] == result["error"] == "Failure: owned cleanup failed"


def test_preflight_interruption_closes_owned_session(completion_case):
    case = completion_case
    case.stage, case.fault = "bounded.trace", KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        case.run()
    result = case.read()
    assert result["error"] == "KeyboardInterrupt"
    assert not result["completed"] and not result["passed"] and not result["gemma_acceptance"]
    assert len(case.sessions) == 1 and case.sessions[0].app_log.closed and case.sessions[0].mcp_closed


@pytest.mark.parametrize("missing", ["all", "fixture_parity", "parity_process_memory"])
def test_early_parity_return_cannot_claim_completion(completion_case, missing):
    case = completion_case
    case.family = "fixture"
    case.parity_checks = [gate for gate in case.parity_checks if missing != "all" and gate["name"] != missing]
    assert case.run() == 1
    result = case.read()
    assert result["acceptance_complete"] and not result["completed"] and not result["passed"]
    assert result["error"] == "Failure: The fixture parity checks did not complete."
    assert not result.get("parity_evidence_collected")


def test_failed_parity_gate_still_blocks_completed_driver(completion_case):
    case = completion_case
    case.family = "fixture"
    case.parity_checks[0]["passed"] = False
    assert case.run() == 1
    result = case.read()
    assert result["completed"] and not result["passed"]
    assert result["failed_checks"] == ["fixture_parity"]


def test_parity_interruption_preserves_error_after_session_cleanup(completion_case):
    case = completion_case
    case.family = "fixture"
    case.stage, case.fault = "parity", KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        case.run()
    result = case.read()
    assert result["error"] == "KeyboardInterrupt"
    assert result["acceptance_complete"] and not result["completed"] and not result["passed"]
    assert all(s.app_log.closed and s.mcp_closed for s in case.sessions)


def test_early_acceptance_return_cannot_claim_completion(completion_case, monkeypatch):
    def partial(session, args, result):
        result.update(dictionary_evidence=gemma_reply(), pixel_evidence_collected=True)
        result["checks"].append(check("partial", True, True, True))
        return driver.Bridge(session)
    monkeypatch.setattr(driver, "acceptance", partial)
    assert completion_case.run() == 1
    result = completion_case.read()
    assert not result["acceptance_complete"] and not result["completed"]
    assert not result["passed"] and not result["gemma_acceptance"]
    assert all(s.app_log.closed for s in completion_case.sessions)


@pytest.mark.parametrize("failed", [False, True])
def test_visible_lifetime_waits_for_owned_app(completion_case, failed):
    case = completion_case
    case.show = True
    if failed:
        case.stage, case.fault = "generate", Failure("generation failed")
    assert case.run() == int(failed)
    assert case.calls.index("keep_alive") < case.calls.index("acceptance.close")
    result = case.read()
    assert result["completed"] is (not failed) and result["passed"] is (not failed)
    assert not result["gemma_acceptance"]
    assert all(s.app_log.closed for s in case.sessions)


def test_visible_recovery_interruption_preserves_both_errors(completion_case, monkeypatch):
    case = completion_case
    case.show = True
    case.stage, case.fault = "generate", Failure("generation failed")
    def interrupted(bridge):
        raise KeyboardInterrupt()
    monkeypatch.setattr(driver.Bridge, "keep_alive", interrupted)
    with pytest.raises(KeyboardInterrupt):
        case.run()
    result = case.read()
    assert result["error"] == "Failure: generation failed" and result["bridge_error"] == "KeyboardInterrupt"
    assert not result["completed"] and not result["passed"] and not result["gemma_acceptance"]
    assert all(s.app_log.closed and s.mcp_closed for s in case.sessions)
