"""Driver completion with CPU data and file resources, without child processes."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_circuit_trace import driver, inference_fixture, qualification_reply, steering_case
from test_gemma_acceptance import gemma_reply
from _checks import MODEL_PROMPT, check, finish
from _mcp import Failure


@pytest.fixture
def completion_case(tmp_path, monkeypatch, steering_case):
    import parity

    prompt, prior, status, params = inference_fixture(MODEL_PROMPT)
    graph = steering_case["graph"]
    graph["nodes"][1]["effect"] = 4.64
    graph["nodes"].extend({"kind": "token", "index": i} for i in range(4, 20))
    examples = {"examples": [dict(row, valid=[False] * 4 + [True] + [False] * 24,
                                  token_ids=[1] * 29, activations=[row["activation"]] * 29)
                              for row in steering_case["examples"]["examples"]]}
    state = SimpleNamespace(out=tmp_path / "result", sessions=[], calls=[], publications=[],
                            family="gemma", prompt=prompt, show=False, fault=None, stage=None,
                            cleanup_fault=None, cleanup_report=None, result=None,
                            preflight_edit=lambda status: None, bounded_cleanup_report=None,
                            baseline_top_text=" Austin", baseline_top_probability=.409498900175,
                            parity_checks=[check(name, True, True, True)
                                           for name in ("fixture_parity", "parity_process_memory")])
    real_close, real_acceptance, real_save = driver.Session.close, driver.acceptance, driver.save

    def checkpoint(stage):
        state.calls.append(stage)
        if state.stage == stage:
            raise state.fault

    class FakeSession:
        def __init__(self, args, out, *, show=False, interactive=False):
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
                inference["state"] = "ready"
                inference["comparison"] = {key: inference[key] for key in ("prompt_hash", "interventions_hash")}
                inference["comparison"].update(argmax_token_ids=[1, 1], completed_step=14 if self.steered else 9,
                                               baseline_top_text=state.baseline_top_text,
                                               baseline_top_probability=state.baseline_top_probability)
                response = {"state": "done", "counts": {"nodes": len(graph["nodes"])}, "inference": inference}
                if self.out.name == "bounded":
                    checkpoint("bounded.status")
                    state.preflight_edit(response)
                return response
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
                return {"attached": True, "params": [{"name": name, "value": value} for name, value in params.items()]}
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
            else:
                self.cleanup_error = state.bounded_cleanup_report

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
        nonlocal prompt, prior, status, params
        prompt, prior, status, params = inference_fixture(state.prompt)
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
    monkeypatch.setattr(driver, "check_assets", lambda session: {"wasm": {"status": 200}})
    monkeypatch.setattr(driver, "hold_after_error", lambda session, error: driver.Bridge(session).keep_alive())
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
    assert case.run() == (1 if family == "gemma" else 0)
    result = case.read()
    assert result["completed"] and result["acceptance_complete"]
    assert result["passed"] is (family != "gemma")
    assert not result["gemma_acceptance"]
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
        case.stage, case.fault = "acceptance.trace", Failure("trace failed before the bridge")
    assert case.run() == int(failed)
    assert case.calls.index("keep_alive") < case.calls.index("acceptance.close")
    result = case.read()
    assert not result["completed"] and not result["passed"]
    assert result.get("demo_ready", False) is (not failed)
    assert "generate" not in case.calls and "validation" not in case.calls
    assert not result["gemma_acceptance"]
    assert all(s.app_log.closed for s in case.sessions)


def test_visible_recovery_interruption_preserves_both_errors(completion_case, monkeypatch):
    case = completion_case
    case.show = True
    case.stage, case.fault = "acceptance.trace", Failure("trace failed before the bridge")
    def interrupted(bridge):
        raise KeyboardInterrupt()
    monkeypatch.setattr(driver.Bridge, "keep_alive", interrupted)
    with pytest.raises(KeyboardInterrupt):
        case.run()
    result = case.read()
    assert result["error"] == "Failure: trace failed before the bridge" and result["bridge_error"] == "KeyboardInterrupt"
    assert not result["completed"] and not result["passed"] and not result["gemma_acceptance"]
    assert all(s.app_log.closed and s.mcp_closed for s in case.sessions)


def test_complete_gemma_custom_prompt_retains_confidence_failure(completion_case):
    case = completion_case
    case.prompt = "A neutral custom prompt"
    assert case.run() == 1
    result = case.read()
    assert result["completed"] and result["acceptance_complete"]
    assert not result["passed"] and not result["gemma_acceptance"]
    gate = next(gate for gate in result["checks"] if gate["name"] == "prompt_confidence")
    assert gate["required"] and not gate["passed"]
    assert gate["measured"]["baseline_probability"] == .5


def assert_preflight_stopped(case):
    result = case.read()
    assert len(case.sessions) == 1
    assert case.sessions[0].out.name == "bounded"
    assert case.sessions[0].app_log.closed and case.sessions[0].mcp_closed
    assert case.calls.count("bounded.trace") == 1
    assert "acceptance.trace" not in case.calls
    assert not result["completed"] and not result["acceptance_complete"]
    assert not result["passed"] and not result["gemma_acceptance"]
    assert all(not report["passed"] for report in case.publications)
    return result


@pytest.mark.parametrize("answer", [" Dallas", "Austin", " Austin ", "austin", "ĠAustin"])
@pytest.mark.parametrize("probability", [.409498900175, .95])
def test_dallas_mismatch_stops_before_full_session(completion_case, answer, probability):
    case = completion_case
    case.baseline_top_text = answer
    case.baseline_top_probability = probability
    assert case.run() == 1
    result = assert_preflight_stopped(case)
    gate = next(g for g in result["checks"] if g["name"] == "dallas_preflight")
    assert gate["required"] and not gate["passed"]
    assert gate["measured"]["mismatch"] is True
    assert gate["measured"]["baseline_top_text"] == answer
    assert gate["measured"]["baseline_top_probability"] == probability
    assert "not ' Austin'" in result["error"]


@pytest.mark.parametrize("top,probability", [(1, .409498900175), (73, .123456789)])
def test_dallas_preflight_records_bound_probability_below_confidence_threshold(completion_case, top, probability):
    from _inference_evidence import text_hash
    case = completion_case
    case.baseline_top_probability = probability
    case.preflight_edit = lambda status: status["inference"]["comparison"].update(argmax_token_ids=[top, top])
    assert case.run() == 1
    result = case.read()
    gate = next(g for g in result["checks"] if g["name"] == "dallas_preflight")
    assert gate["required"] and gate["passed"]
    measured = gate["measured"]
    assert measured["mismatch"] is False
    assert measured["baseline_top_text"] == " Austin"
    assert measured["baseline_top_probability"] == probability
    assert measured["argmax_token_ids"] == [top, top]
    assert measured["completed_step"] == 9
    assert measured["prompt_hash"] == text_hash(MODEL_PROMPT)
    assert measured["interventions_hash"] == text_hash("[]")
    assert len(case.sessions) == 2
    assert case.calls.count("bounded.trace") == case.calls.count("acceptance.trace") == 1
    assert case.calls.index("bounded.status") < case.calls.index("bounded.close") < case.calls.index("acceptance.trace")
    assert result["completed"] and not result["passed"] and not result["gemma_acceptance"]
    assert {"published_profile", "external_replication_review"} <= set(result["failed_checks"])


@pytest.mark.parametrize("probability", [None, True, "0.409", float("nan"), float("inf"),
                                       -float("inf"), -.01, 1.01])
def test_invalid_dallas_probability_stops_before_full_session(completion_case, probability):
    case = completion_case
    case.baseline_top_probability = probability
    assert case.run() == 1
    result = assert_preflight_stopped(case)
    assert "finite probability" in result["error"]


@pytest.mark.parametrize("fault", ["missing_state", "missing_comparison", "missing_text", "invalid_text", "empty_text",
                                   "missing_probability", "wrong_prompt", "stale_prompt", "stale_interventions",
                                   "invalid_hash", "missing_step", "boolean_step", "negative_step", "float_step",
                                   "missing_argmax", "boolean_argmax", "negative_argmax", "short_argmax",
                                   "trace_pending", "inference_starting"])
def test_stale_or_incomplete_dallas_evidence_stops_before_full_session(completion_case, fault):
    case = completion_case
    def corrupt(status):
        state = status["inference"]
        record = state["comparison"]
        if fault == "missing_state":
            status.pop("inference")
        elif fault == "missing_comparison":
            state.pop("comparison")
        elif fault in ("missing_text", "missing_probability", "missing_step", "missing_argmax"):
            record.pop({"missing_text": "baseline_top_text", "missing_probability": "baseline_top_probability",
                        "missing_step": "completed_step", "missing_argmax": "argmax_token_ids"}[fault])
        elif fault in ("invalid_text", "empty_text"):
            record["baseline_top_text"] = None if fault == "invalid_text" else ""
        elif fault == "wrong_prompt":
            state["prompt_hash"] = record["prompt_hash"] = "0" * 64
        elif fault in ("stale_prompt", "stale_interventions"):
            record["prompt_hash" if fault == "stale_prompt" else "interventions_hash"] = "0" * 64
        elif fault == "invalid_hash":
            state["interventions_hash"] = record["interventions_hash"] = "invalid"
        elif fault.endswith("_step"):
            record["completed_step"] = {"boolean_step": True, "negative_step": -1, "float_step": 9.}[fault]
        elif fault.endswith("_argmax"):
            record["argmax_token_ids"] = {"boolean_argmax": [True, 1], "negative_argmax": [-1, 1],
                                          "short_argmax": [1]}[fault]
        elif fault == "trace_pending":
            status["state"] = "pending"
        else:
            state["state"] = "starting"
    case.preflight_edit = corrupt
    assert case.run() == 1
    result = assert_preflight_stopped(case)
    assert result["error"].startswith("Failure:")


@pytest.mark.parametrize("family,prompt", [("fixture", MODEL_PROMPT), ("qwen", MODEL_PROMPT),
                                          ("gemma", "Custom fact"), ("gemma", MODEL_PROMPT + " ")])
def test_dallas_preflight_scope_preserves_existing_prompt_behavior(completion_case, family, prompt):
    case = completion_case
    case.family, case.prompt = family, prompt
    case.baseline_top_text = " Dallas"
    expected = 1 if family == "gemma" else 0
    assert case.run() == expected
    result = case.read()
    assert len(case.sessions) == 2 and "bounded.status" not in case.calls
    assert not any(g["name"] == "dallas_preflight" for g in result["checks"])
    assert result["completed"]
    if family == "gemma":
        assert result["failed_checks"] == ["prompt_confidence", "published_profile", "external_replication_review"]
    else:
        assert result["passed"]


@pytest.mark.parametrize("answer", [" Austin", " Dallas"])
def test_bounded_cleanup_failure_cannot_start_full_session(completion_case, answer):
    case = completion_case
    case.baseline_top_text = answer
    case.bounded_cleanup_report = "owned bounded cleanup failed"
    assert case.run() == 1
    result = assert_preflight_stopped(case)
    assert result["cleanup_error"] == "Failure: owned bounded cleanup failed"
    if answer == " Dallas":
        assert "not ' Austin'" in result["error"]


def test_preflight_status_interruption_preserves_cleanup(completion_case):
    case = completion_case
    case.stage, case.fault = "bounded.status", KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        case.run()
    result = assert_preflight_stopped(case)
    assert result["error"] == "KeyboardInterrupt"
