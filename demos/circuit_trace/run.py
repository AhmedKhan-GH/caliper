#!/usr/bin/env python3
"""Circuit trace instrument through the actual Caliper MCP process."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from _mcp import Events, Failure, MCP, attempt, stop, wait_file
from _checks import (FIXTURE_HASH, FIXTURE_PROMPT, MODEL_PROMPT, bar_direction_check, bar_rows, capture_checks,
                     check, dallas_preflight_check, finish, math_checks, program_budget_checks,
                     secondary_selection, selected_feature)
from _inference_evidence import (bound_completion, comparison_evidence, generation_baseline,
                                 generation_evidence, inference_state)
from _qualification import (GEMMA_REVISION, MODEL_REVISION, dictionary_qualification,
                             validate_quality_override)
from _spec import scene_programs
from _visible import StartupFailure, check_assets, configure_assets, hold_after_error, startup_instrument
import _published as published

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_RESOURCES = {
    "attribution": ("nodes", "curves", "labels", "strings"),
    "example_tokens": ("rows", "cells", "labels", "strings"),
    "probabilities": ("cells", "labels", "strings"),
}
METHOD_DISCLOSURES = {
    "dictionary": "The graph uses a separate transcoder dictionary for each layer.",
    "routing": "The graph fixes attention routing, normalization scales, and feature masks for this prompt.",
    "labels": "Feature labels use a heuristic from promoted output tokens and frequent example tokens.",
    "prompt_limit": "Prompts contain at most 32 tokens, including special tokens.",
    "analysis_scope": "The graph describes local prompt effects and excludes global weight analysis.",
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def error_detail(error):
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def executable(variable, relative):
    value = Path(os.environ.get(variable, ROOT / relative)).resolve()
    if not value.is_file() or not os.access(value, os.X_OK):
        raise Failure(f"{variable} must name a built executable.")
    return value


def memory_bytes(folder, *, platform=None):
    platform = platform or sys.platform
    peaks = []
    for path in folder.glob("*memory*.txt"):
        text = path.read_text()
        mac = re.search(r"^\s*(\d+)\s+peak memory footprint\s*$", text, re.MULTILINE)
        linux = re.search(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$", text, re.MULTILINE)
        if platform == "darwin":
            if mac is None:
                raise Failure(f"No peak memory footprint in {path.name}.")
            peaks.append(int(mac[1]))
        elif linux is not None:
            peaks.append(int(linux[1]) * 1024)
        else:
            raise Failure(f"No process high-water measurement in {path.name}.")
    if not peaks or min(peaks) <= 0:
        raise Failure("No completed process memory measurement exists.")
    return max(peaks)


class Session:
    """Own one app and one MCP process, plus the hosted run that MCP starts."""
    def __init__(self, args, out, *, show=False, interactive=False):
        self.out, self.show = out, show
        self.quality_override = getattr(args, "quality_override", None)
        self.replication_profile = getattr(args, "replication_profile", None)
        self.private_output_directory = getattr(args, "private_output_directory", None)
        self.family = getattr(args, "family", "fixture" if args.model == "tiny-lm" else "qwen")
        self.app = self.mcp = self.events = None
        self.app_process = None
        self.app_owner = None
        self.run_id = "circuit-trace"
        self.started = False
        self.capture_number = 0
        self.program_evidence = {}
        self.cleanup_error = None
        out.mkdir(parents=True)
        for name in ("data", "workspace", "user-applets", "captures", "temp"):
            (out / name).mkdir()
        env = os.environ.copy()
        for name in ("CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_TEST_SPEC_UI",
                     "CALIPER_TEST_APPROVALS", "CALIPER_API", "CALIPER_APPLET_PATH", "CALIPER_SESSION"):
            env.pop(name, None)
        env.update(CALIPER_AGENT_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   CALIPER_DATA_DIR=str(out / "data"), CALIPER_WORKSPACE=str(out / "workspace"),
                   CALIPER_USER_APPLETS_DIR=str(out / "user-applets"), CALIPER_WINDOW_SIZE="2048x1000",
                   CALIPER_HIDDEN="0" if show else "1", TMPDIR=str(out / "temp"),
                   PYTHONPYCACHEPREFIX=str(out / "pycache"))
        if args.model == "tiny-lm":
            env["CALIPER_FIXTURE_DISPLAY_MPS"] = "1"
        else:
            env.pop("CALIPER_FIXTURE_DISPLAY_MPS", None)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), str(ROOT), *[
            str(Path(entry).resolve()) for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]])
        # This mode applies only to the task's new data folder.
        permissions = out / "data" / "agent-permissions.json"
        save(permissions, {"version": 1, "modes": {"chat": "ask", "mcp": "allow_all"}, "rules": []})
        permissions.chmod(0o600)
        flags = ["-l"] if sys.platform == "darwin" else ["-v"]
        # The runner still executes the real installed Python module. time records
        # the child high-water mark and does not issue a model control operation.
        launcher = out / "run-python"
        launcher.write_text("#!/bin/sh\nexec /usr/bin/time " + " ".join(flags)
                            + " -o " + shlex.quote(str(out / "run-memory"))
                            + '-$$.txt ' + shlex.quote(sys.executable) + ' "$@"\n')
        launcher.chmod(0o700)
        env["CALIPER_RUN_PYTHON"] = str(launcher)
        configure_assets(env)
        if not show:
            env["CALIPER_TEST_CIRCUIT_CAPTURE_DIR"] = str(out / "captures")
        else:
            env.pop("CALIPER_TEST_CIRCUIT_CAPTURE_DIR", None)
        self.env = env
        app = executable("CALIPER_APP_EXE", "cmake-build-release/caliper")
        mcp = executable("CALIPER_MCP_EXE", "cmake-build-release/src/builder/mcp/caliper-mcp")
        self.app_log = (out / "app.log").open("w")
        command = ["/usr/bin/time", *flags, "-o", str(out / "app-memory.txt"), str(app),
                   *([] if show else ["--hidden"]), "--api", "--api-port", "0"]
        try:
            self.app = subprocess.Popen(command, env=env, stdout=self.app_log, stderr=subprocess.STDOUT)
            import psutil
            self.app_owner = psutil.Process(self.app.pid)
            wait_file(out / "data" / "api.json", process=self.app)
            children = self.app_owner.children()
            if len(children) != 1:
                raise Failure("The owned app process cannot be identified.")
            self.app_process = children[0]
            self.mcp = MCP(mcp, out / "data" / "api.json", out, env)
            self.events = Events(self.mcp)
            if show or interactive:
                startup_instrument(self)
            options = ({"replication_profile": self.replication_profile,
                        "private_output_directory": str(self.private_output_directory)}
                       if self.replication_profile else {})
            self.mcp.body("run.start", kind="inference", id=self.run_id, model=args.model,
                          dictionary=str(Path(args.dictionary).resolve()), prompt=args.prompt, **options)
            self.started = True
            def ready():
                runs = self.mcp.body("run.list")["runs"]
                run = next((r for r in runs if r["id"] == self.run_id), None)
                if run and run["state"] in ("failed", "exited"):
                    raise Failure(f"The hosted run failed: {run.get('error', '')}")
                if not run or run["state"] not in ("running", "paused"):
                    return False
                state = self.mcp.body("run.state", run=self.run_id)
                names = {p["name"] for p in state["params"]}
                return state if state["attached"] and {"prompt", "selected_node", "clamp", "retrace"} <= names else False
            # The explicit reference path verifies all source bytes before attachment.
            # Its startup allowance is separate from trace and action timing.
            ready_options = {"timeout": 300} if self.replication_profile else {}
            self.events.until("hosted inference readiness", ready, **ready_options)
        except BaseException as exc:
            if (show or interactive) and isinstance(exc, Exception) and self.live():
                raise StartupFailure(self, exc) from exc
            self.close()
            raise

    def live(self):
        return self.app is not None and self.app.poll() is None

    def parameter(self, name, value):
        self.mcp.body("run.set_param", run=self.run_id, name=name, value=value)
        def applied():
            state = self.mcp.body("run.state", run=self.run_id)
            param = next(p for p in state["params"] if p["name"] == name)
            return state if param.get("pending") is None and param["value"] == value else False
        return self.events.until(f"parameter {name}", applied)

    def action(self, name):
        self.events.wait(0)
        first = len(self.events.pending)
        self.mcp.body("run.action", run=self.run_id, name=name)
        def completed():
            for event in self.events.pending[first:]:
                if (event.get("kind") == "run" and event.get("change") == "action"
                        and event.get("name") == name):
                    if event.get("status") != "ok":
                        raise Failure(f"The {name} action failed: {event.get('text', '')}")
                    return event
            return False
        return self.events.until(f"completed action {name}", completed)

    def ablate_selected(self, prompt, *, minimum_step):
        self.parameter("clamp", 0.)
        def comparison(*, pending=False):
            status = self.events.complete("trace.status")
            state = inference_state(status, prompt)
            record = state.get("comparison")
            if not bound_completion(record, state):
                if pending:
                    return False
                raise Failure("Output comparison evidence is stale or incomplete.")
            return status, record["completed_step"]
        # Parameter metadata can arrive before its output comparison completes.
        _, prior_step = self.events.until("slider output comparison", lambda: comparison(pending=True))
        self.action("apply_clamp")
        def updated():
            status, step = comparison()
            return (status, step) if step > prior_step else False
        status, step = self.events.until("applied clamp output comparison", updated)
        reply = self.read("probabilities", "cells", minimum_step=max(minimum_step, step))
        evidence = comparison_evidence(self.events.complete("trace.status"), prompt, bar_rows(reply))
        if evidence != status["inference"]["comparison"]:
            raise Failure("The output comparison changed during resource publication.")
        return reply, evidence

    def read(self, program, resource, *, minimum_step=0, count=1024, first=0):
        def published():
            reply = self.mcp.body("program.read", id=program, resource=resource, first=first, count=count)
            if reply["step"] is not None and reply["step"] >= minimum_step:
                return reply
            # The hosted run advances pending publications through its scheduler.
            # Read the real resource step and retain program failure checks.
            self.ready_programs([program])
            return False
        return self.events.until(f"published {program}/{resource}", published)

    def example_page(self, page, *, minimum_step=0):
        def published():
            rows = self.read("example_tokens", "rows", minimum_step=minimum_step)
            records = rows.get("elements", [])
            if len(records) != 8 or any(len(row) != 4 or row[1:3] != [page * 8 + i, i]
                                        for i, row in enumerate(records)):
                self.ready_programs(["example_tokens"])
                return False
            cells = self.read("example_tokens", "cells", minimum_step=rows["step"])
            if cells["step"] != rows["step"]:
                return False
            return rows, cells
        return self.events.until(f"published example page {page + 1}", published)

    def ready_programs(self, identifiers):
        def compiled():
            statuses = self.mcp.body("program.status")["programs"]
            found = {p["id"]: p for p in statuses}
            for identifier in identifiers:
                program = found.get(identifier, {})
                if program.get("error") or program.get("state") in ("refused", "stopped"):
                    raise Failure(f"The {identifier} program cannot run: {program}")
                if program.get("state") not in ("compiled", "running"):
                    return False
            return True
        return self.events.until("compiled derive programs", compiled)

    def capture(self, name, expected_steps):
        """Hold resource history until a fresh, coherent frame reaches the capture."""
        if not expected_steps or set(expected_steps) - CAPTURE_RESOURCES.keys():
            raise Failure("Capture requires known circuit programs and their resource steps.")
        expected_steps = dict(expected_steps)
        selection_step = getattr(getattr(self, "bridge", None), "selection_step", None)
        if "attribution" in expected_steps and selection_step is not None:
            expected_steps["attribution"] = max(expected_steps["attribution"], selection_step)
        cursor = self.mcp.body("cursor.get")["cursor"]
        def current_cursor():
            self._capture_status(expected_steps)
            current = self.mcp.body("cursor.get")["cursor"]
            # A derive pass can finish after the selection acknowledgment step.
            # Do not hold a cursor that predates the resources required for selection.
            steps = self._capture_resources(expected_steps, current["latest"])
            return current if steps else False
        current = self.events.until("capture cursor step", current_cursor)
        try:
            held = self.mcp.body("cursor.set", step=current["latest"], live=False, playing=False)["cursor"]
            return self._capture_held(name, expected_steps, held["step"])
        finally:
            restore = {key: cursor[key] for key in ("live", "playing", "speed")}
            if not cursor["live"]:
                restore["step"] = cursor["step"]
            self.mcp.body("cursor.set", **restore)

    def _capture_status(self, expected_steps):
        status = self.mcp.body("program.status")
        for program in status["programs"]:
            if program["id"] not in expected_steps:
                continue
            draw = program.get("draw") or {}
            if (program.get("error") or draw.get("error")
                    or program.get("state") in ("refused", "stopped")
                    or draw.get("state") in ("refused", "stopped")):
                raise Failure(f"The {program['id']} program reports an error: {program}")
        return status

    def _capture_resources(self, expected_steps, cursor_step):
        # Each circuit piece writes all its listed resources on each derive run.
        resources = {}
        for identifier, minimum in expected_steps.items():
            steps = {resource: self.mcp.body("program.read", id=identifier, resource=resource,
                                            step=cursor_step, count=1)["step"]
                     for resource in CAPTURE_RESOURCES[identifier]}
            if any(step is None or step < minimum for step in steps.values()) or len(set(steps.values())) != 1:
                return None
            resources[identifier] = steps
        return resources

    def _capture_held(self, name, expected_steps, cursor_step):
        initial = self._capture_status(expected_steps)
        first_frames = {p["id"]: (p.get("draw") or {}).get("frames", 0) for p in initial["programs"]}
        resource_steps = {}
        def rendered():
            cursor = self.mcp.body("cursor.get")["cursor"]
            if cursor["step"] != cursor_step or cursor["live"] or cursor["playing"]:
                raise Failure("The capture cursor changed before the resource barrier completed.")
            status = self._capture_status(expected_steps)
            by_id = {p["id"]: p for p in status["programs"]}
            # Read the viewer's retained history at the held cursor. Producer status alone
            # cannot prove that the final resource chunk arrived or that a frame used it.
            resources = self._capture_resources(expected_steps, cursor_step)
            if resources is None:
                return False
            for identifier, steps in resources.items():
                program = by_id.get(identifier, {})
                draw = program.get("draw") or {}
                shown = draw.get("resource_steps") or {}
                if any(shown.get(resource) != step for resource, step in steps.items()):
                    return False
                if draw.get("frames", 0) <= first_frames.get(identifier, 0):
                    return False
                if draw.get("frame_ms") is None or program.get("last_pass_ms") is None:
                    return False
                if program.get("last_step") is None or program["last_step"] < max(steps.values()):
                    return False
                resource_steps[identifier] = steps
            return status
        status = self.events.until("rendered circuit resources", rendered)
        pieces = {}
        for program in status["programs"]:
            if program["id"] in expected_steps:
                pin = program["piece"]
                pieces[program["id"]] = self.mcp.body("piece.show", name=pin["name"], version=pin.get("version", 1),
                                                     **{"from": pin.get("from", "library")})["piece"]
        snapshot = {"status": status, "pieces": pieces, "expected_steps": expected_steps,
                    "resource_steps": resource_steps, "cursor_step": cursor_step,
                    "phase": "after the resource barrier and before capture.frame"}
        self.program_evidence[name] = snapshot
        save(self.out / (name + "-program-status.json"), snapshot)
        reply = self.mcp.call("capture.frame")
        image = next((b for b in reply.get("content", []) if b.get("type") == "image"), None)
        encoded = image["data"] if image else reply["structuredContent"]["png_base64"]
        png = self.out / (name + ".png")
        png.write_bytes(base64.b64decode(encoded, validate=True))
        self.capture_number += 1
        if self.show:
            return png, None
        metadata = wait_file(self.out / "captures" / f"capture-{self.capture_number}.json", process=self.app, timeout=30)
        metadata["expected_steps"] = expected_steps
        save(self.out / (name + ".json"), metadata)
        return png, metadata

    def close(self):
        errors = []
        if self.mcp is not None and self.live() and self.started:
            attempt(errors, "run logs", lambda: save(self.out / "run-logs.json",
                    self.mcp.body("run.logs", run=self.run_id, lines=200)))
            def stop_run():
                self.mcp.body("run.stop", run=self.run_id, kill_after_s=10)
                self.events.until("hosted run exit", lambda: all(
                    r["state"] in ("failed", "exited") for r in self.mcp.body("run.list")["runs"]), timeout=30)
            attempt(errors, "run stop", stop_run)
        if self.events is not None:
            attempt(errors, "event save", lambda: save(self.out / "events.json", self.events.history))
        if self.mcp is not None:
            attempt(errors, "MCP close", self.mcp.close)
            self.mcp = None
        if self.app is not None:
            import psutil
            children = []
            for owner in {self.app_owner, self.app_process} - {None}:
                if owner.is_running():
                    found = attempt(errors, "owned child list", lambda owner=owner: owner.children(recursive=True)) or []
                    children.extend(child for child in found if child not in children)
            if self.app_process is not None and self.app_process not in children:
                children.append(self.app_process)
            def stop_child(owned):
                try:
                    owned.terminate()
                    try:
                        owned.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        owned.kill()
                        owned.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass
            for child in reversed(children):
                attempt(errors, "owned child stop", lambda child=child: stop_child(child))
            def reap():
                try:
                    self.app.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop(self.app)
            attempt(errors, "app wrapper stop", reap)
            self.app = None
        attempt(errors, "app log close", self.app_log.close)
        if errors:
            self.cleanup_error = "; ".join(errors)
            attempt(errors, "cleanup report", lambda: save(self.out / "cleanup-error.json", {"error": self.cleanup_error}))


def read_graph(session):
    first = session.events.complete("trace.read", node_limit=256, edge_limit=256)
    graph = {**first, "nodes": list(first["nodes"]), "edges": list(first["edges"])}
    for key, target, limit in (("next_node_offset", "nodes", "node"), ("next_edge_offset", "edges", "edge")):
        offset = first[key]
        while offset is not None:
            page = session.events.complete("trace.read", **{limit + "_offset": offset, limit + "_limit": 256,
                                                            ("edge" if limit == "node" else "node") + "_limit": 0})
            if page["trace_id"] != graph["trace_id"]:
                raise Failure("The trace changed during graph pagination.")
            graph[target].extend(page[target])
            offset = page[key]
        graph[key] = None
    return graph


def instrument(session):
    if getattr(session, "replication_profile", None) == published.PROFILE:
        from _published_ui import instrument as published_instrument
        return published_instrument(session)
    m = session.mcp
    disclosures = dict(METHOD_DISCLOSURES)
    if getattr(session, "family", None) == "gemma":
        evidence = session.events.complete("feature.dictionary")
        research = evidence.get("research_quality", {})
        if not dictionary_qualification(evidence, family="gemma")["qualified"]:
            raise Failure("The Gemma research comparison sample or artifact identity is unavailable.")
        disclosures["quality_measurements"] = (
            f"Research comparison uses {research['source_tokens']:,} source positions ({research['status']}). "
            f"FVU {research['fvu']:.6f}. Dead fraction {research['dead_fraction']:.6f}. "
            f"All-layer loss rise {research['cross_entropy_delta']:.6f} nats. Quality bars await calibration.")
    override = getattr(session, "quality_override", None)
    if override is not None:
        override = validate_quality_override(override)
        disclosures["dictionary_quality_override"] = (
            f"Founder override. Quality remains failed. FVU {override['fvu']:.9f}. "
            f"Dead fraction {override['dead_fraction']:.9f}. Cross-entropy rise {override['cross_entropy_delta']:.9f} nats.")
        disclosures["dictionary_identity"] = "Dictionary SHA256: " + override["dictionary_fingerprint"]
    m.body("spec.new", id="circuit-trace", title="Circuit Trace", description=" ".join(disclosures.values()))
    for identifier, default, maximum in (("selected_node", -1, 511), ("example_page", 0, 1)):
        m.body("state.create", id=identifier, type="integer", default=default, min=default, max=maximum)
    m.body("state.create", id="activation_max", type="number", default=1., min=1e-6, max=1e20)
    for identifier, title in (("graph", "Attribution Graph"), ("examples", "Feature Examples"), ("output", "Output Probabilities")):
        m.body("panel.create", id=identifier, primitive="text", title=title,
               options={"text": "No feature is selected." if identifier == "examples" else "The trace is not available."})
    controls = [("prompt", "text", "param:prompt", "Prompt", {}),
                ("clamp", "slider", "param:clamp", "Feature Clamp", {"min": -4, "max": 4}),
                ("apply_clamp", "button", "action:apply_clamp", "Apply Clamp", {}),
                ("retrace", "toggle", "param:retrace", "Automatic Trace", {}),
                ("reset", "button", "action:reset", "Feature Reset", {}),
                ("generate", "button", "action:generate", "Token Generation", {}),
                ("page", "segmented", "state:example_page", "Example Page",
                 {"choices": [{"value": 0, "label": "1–8"}, {"value": 1, "label": "9–16"}]}),
                ("baseline_text", "text", "param:output.baseline", "Baseline Tokens", {}),
                ("steered_text", "text", "param:output.steered", "Steered Tokens", {})]
    for identifier, kind, bind, title, options in controls:
        m.body("control.create", id=identifier, kind=kind, bind=bind, title=title, options=options)
    # The unplaced text control supplies the parameter to the non-editable readout.
    m.body("control.create", id="clamp_status_value", kind="text", bind="param:clamp.status", title="Clamp Status")
    m.body("control.create", id="clamp_status", kind="readout", title="Clamp Status",
           options={"format": "{status:s}", "fields": {"status": {"control": "clamp_status_value"}}})
    for name, text in disclosures.items():
        m.body("control.create", id="method_" + name, kind="readout", title=name.replace("_", " ").title(),
               options={"format": text})
    m.body("state.create", id="method_details", type="boolean", default=False)
    m.body("control.create", id="method_details", kind="toggle", bind="state:method_details", title="Method Details")
    instrument_layout(session, expanded=False)


def instrument_layout(session, *, expanded):
    override = getattr(session, "quality_override", None)
    research = getattr(session, "family", None) == "gemma"
    # Flat control groups stay within the spec's eight-level layout limit.
    ratio = .74 if expanded else (.90 if override or research else .95)
    nodes = [{"id": "instrument", "split": "vertical", "ratio": ratio, "children": ["main", "methods"]},
             {"id": "main", "split": "vertical", "ratio": .10225, "children": ["header", "body"]},
             {"id": "header", "split": "vertical", "children": ["prompt", "clamp_status"]},
             {"id": "body", "split": "vertical", "ratio": .64 if expanded else .70, "children": ["plots", "examples"]},
             {"id": "plots", "split": "horizontal", "ratio": .67, "children": ["graph", "right"]},
             {"id": "right", "split": "vertical", "ratio": .55, "children": ["output", "controls"]}]
    control_ids = ["clamp", "apply_clamp", "reset", "retrace", "generate", "page", "baseline_text", "steered_text"]
    if expanded:
        # Two columns preserve bar-label height beside the expanded disclosures.
        nodes.extend([{"id": "controls", "split": "horizontal", "children": ["primary_controls", "secondary_controls"]},
                      {"id": "primary_controls", "split": "vertical", "children": control_ids[:4]},
                      {"id": "secondary_controls", "split": "vertical", "children": control_ids[4:]}])
    else:
        nodes.append({"id": "controls", "split": "vertical", "children": control_ids})
    method_ids = ["method_details"]
    if expanded:
        method_ids.extend("method_" + name for name in METHOD_DISCLOSURES)
    if override is not None:
        method_ids.extend(["method_dictionary_quality_override", "method_dictionary_identity"])
    if research:
        method_ids.append("method_quality_measurements")
    if len(method_ids) == 1:
        nodes[0]["children"][1] = method_ids[0]
    else:
        nodes.append({"id": "methods", "split": "vertical", "children": method_ids})
    session.mcp.body("layout.set", layout={"root": "instrument", "nodes": nodes})


def programs(session, *, examples=False):
    if examples:
        scene_programs(session, [dict(id="example_tokens", panel="examples", piece={"name": "token_strip", "version": 1},
                         bind={key: "live:examples." + key for key in ("tokens", "activations", "example_ids", "valid_tokens", "label_text")},
                         params={"page": {"state": "example_page"}, "activation_max": {"state": "activation_max"}})])
        return
    scene_programs(session, [dict(id="attribution", panel="graph", piece={"name": "attribution_graph", "version": 1},
                     bind={"graph_counts": "live:trace.counts", **{k: "live:trace." + k for k in
                           ("node_i", "node_f", "edge_i", "edge_f", "label_text")}},
                     params={"selected_node": {"state": "selected_node"}}),
                  dict(id="probabilities", panel="output", piece={"name": "logit_bars", "version": 1},
                     bind={key: "live:output." + key for key in ("probs", "tokens", "label_text")})])


class Bridge:
    def __init__(self, session):
        self.session = session
        self.last_examples = None
        self.last_selection = None
        self.selection_step = None
        self.selection_generation = 0
        self.examples_present = False
        self.details_expanded = False

    def clear_examples(self):
        if self.examples_present:
            scene_programs(self.session, [], remove_examples=True)
            self.examples_present = False
        self.last_selection = self.last_examples = None
        self.selection_step = None

    def fill_examples(self, index):
        s = self.session
        feature = s.events.complete("feature.show", node=index)
        self.last_examples = s.events.complete("feature.examples", node=index, limit=16)
        s.mcp.body("state.set", id="activation_max", value=max(1e-6, feature["maximum"]))
        if self.examples_present is False:
            programs(s, examples=True)
            self.examples_present = True
            s.ready_programs(["example_tokens"])
            s.events.complete("feature.examples", node=index, limit=16)
        self.last_selection = index
        self.selection_generation += 1
        save(s.out / "selected-examples.json", self.last_examples)

    def handle(self, event):
        s = self.session
        if getattr(s, "replication_profile", None) == published.PROFILE:
            from _published_ui import display_result, group_text
            if event["kind"] == "control" and event.get("control") == "published_group":
                s.mcp.body("state.set", id="published_members", value=group_text(int(event["value"])))
                return
            if event["kind"] == "run" and event.get("change") == "action" and event.get("name", "").startswith(published.PROFILE + "."):
                handled = getattr(s, "published_handled_events", [])
                if any(event is item for item in handled):
                    s.published_handled_events = [item for item in handled if item is not event]
                    return
                status = s.events.complete("trace.status")
                _, replication = published.replication_state(status)
                sequence = getattr(s, "published_last_sequence", 0)
                if event.get("status") != "ok":
                    published.require(replication.get("last_attempt") == {
                        "case_id": event["name"], "sequence": sequence + 1, "status": "error"},
                        "The live action failure belongs to another attempt.")
                    s.published_last_sequence = sequence + 1
                    s.mcp.body("state.set", id="published_status", value=f"{event['name']}: {event.get('text', 'Action failed.')}")
                    return
                record = replication.get("result") or {}
                published.require(record.get("case_id") == event["name"], "The live case publication is stale.")
                evidence = published.validate_case(s.private_output_directory, status, record["cell"],
                                                   previous_sequence=sequence,
                                                   preparation_id=getattr(s, "published_preparation_id", None))
                published.require(evidence["baseline_array_sha256"] == s.published_baseline_hash,
                                  "The live case changed baseline arrays.")
                s.published_last_sequence = record["sequence"]
                display_result(s, evidence)
                save(s.out / "live-case.json", evidence)
                return
        if event["kind"] == "pick" and event.get("panel") == "graph":
            fields = event.get("fields") or {}
            index = fields.get("index", event.get("index"))
            if fields.get("kind") != 1:
                s.mcp.body("state.set", id="selected_node", value=-1)
                s.parameter("selected_node", -1)
                self.clear_examples()
                return
            self.selection_step = s.parameter("selected_node", int(index))["step"]
            self.fill_examples(int(index))
        elif event["kind"] == "run" and event.get("change") == "param" and event.get("name") == "selected_node":
            index = int(event["value"])
            if (index if index >= 0 else None) != self.last_selection:
                self.selection_step = event.get("step")
                s.mcp.body("state.set", id="selected_node", value=index)
                if index < 0:
                    self.clear_examples()
                else:
                    self.fill_examples(index)
        elif event["kind"] == "control" and event.get("control") == "page" and self.last_selection is not None:
            s.events.complete("feature.examples", node=self.last_selection, limit=16)
        elif event["kind"] == "control" and event.get("control") == "method_details":
            instrument_layout(s, expanded=bool(event["value"]))
            self.details_expanded = bool(event["value"])

    def consume(self, *, tolerate_refusals=False):
        s = self.session
        queued, s.events.pending = s.events.pending, []
        for event in queued:
            try:
                self.handle(event)
            except Failure as exc:
                if not tolerate_refusals:
                    raise
                if getattr(s, "replication_profile", None) == published.PROFILE:
                    s.mcp.body("state.set", id="published_status", value="Case evidence error: " + str(exc))
                with (s.out / "bridge-errors.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"event": event, "error": str(exc)}) + "\n")

    def select(self, row):
        generation = self.selection_generation
        self.session.mcp.body("selection.set", panel="graph", program="attribution", item=row)
        def selected():
            self.consume()
            return self.last_examples if self.last_selection == row and self.selection_generation > generation else False
        return self.session.events.until("feature pick bridge", selected)

    def details(self, expanded):
        self.session.mcp.body("control.set", id="method_details", value=expanded)
        def applied():
            self.consume()
            return self.details_expanded == expanded
        self.session.events.until("method disclosure layout", applied)

    def keep_alive(self):
        while self.session.live():
            try:
                self.session.events.wait(1)
                self.consume(tolerate_refusals=True)
            except Failure as exc:
                if not self.session.live():
                    return
                if "mcp.app_busy" in str(exc) and "app is closing" in str(exc):
                    try:
                        self.session.app.wait(timeout=5)
                        return
                    except subprocess.TimeoutExpired:
                        pass
                raise


def acceptance(session, args, result):
    if getattr(session, "replication_profile", None) == published.PROFILE:
        return published_demo(session, args, result, collect=True)
    s = session
    if result.get("dictionary_quality_override") is not None:
        qualification = dictionary_qualification(s.events.complete("feature.dictionary"),
                                                  quality_override=result["dictionary_quality_override"])
        if not qualification["qualified"]:
            raise Failure("The quality override does not qualify this dictionary: " + qualification.get("reason", "unknown"))
    instrument(s)
    trace = s.events.complete("trace.run", prompt=args.prompt)
    status = s.events.complete("trace.status")
    if status["state"] != "done":
        raise Failure("The actual trace did not complete.")
    graph = read_graph(s)
    save(s.out / "graph.json", graph)
    result["trace"] = status
    result["checks"].append(check("trace_count", status["counts"]["nodes"] == len(graph["nodes"]),
                                   status["counts"]["nodes"], len(graph["nodes"])))
    selected = selected_feature(graph)
    # A zero clamp removes the feature's direct contribution to the original top logit.
    predicted_ablation_effect = -selected["effect"]
    result["selected_feature"] = selected
    result["selection_rule"] = "largest absolute top-logit effect before the last layer"
    result["secondary_selection"] = secondary_selection(graph)
    programs(s)
    s.ready_programs(["attribution", "probabilities"])
    result["dictionary_evidence"] = s.events.complete("feature.dictionary")
    s.read("attribution", "nodes", minimum_step=trace["step"])
    bridge = Bridge(s)
    s.bridge = bridge
    examples = bridge.select(selected["index"])
    s.events.complete("feature.examples", node=selected["index"], limit=16)
    first_page, first_cells = s.example_page(0)
    before_reply = s.read("probabilities", "cells")
    before = bar_rows(before_reply)
    before_comparison = comparison_evidence(s.events.complete("trace.status"), args.prompt, before)
    baseline_top_token = before_comparison["argmax_token_ids"][0]
    if result.get("mode") == "gemma":
        from _checks import prompt_confidence_check
        result["checks"].append(prompt_confidence_check(args.prompt, before, baseline_top_token))
    expected_steps = {"attribution": trace["step"], "example_tokens": first_page["step"],
                      "probabilities": before_reply["step"]}
    result["checks"].append(check("selection_only", all(abs(r["probability"] - r["baseline"]) <= 1e-7 for r in before),
                                   before, "selection preserves the baseline"))
    before_png, before_meta = s.capture("before", dict(expected_steps))
    result["checks"].extend(dict(gate, name="before." + gate["name"])
                            for gate in program_budget_checks(s.program_evidence["before"]))
    s.mcp.body("control.set", id="page", value=1)
    s.events.complete("feature.examples", node=selected["index"], limit=16)
    page, page_cells = s.example_page(1, minimum_step=first_page["step"] + 1)
    expected_steps["example_tokens"] = page["step"]
    from _checks import example_page_check
    result["checks"].extend([example_page_check(examples, first_page, first_cells, 0),
                             example_page_check(examples, page, page_cells, 1)])
    save(s.out / "example-pages.json", {"first": first_page, "second": page,
                                        "first_cells": first_cells, "second_cells": page_cells})
    page_png, page_meta = s.capture("examples-page-2", dict(expected_steps))
    s.mcp.body("control.set", id="page", value=0)
    s.events.complete("feature.examples", node=selected["index"], limit=16)
    restored_page, _ = s.example_page(0, minimum_step=page["step"] + 1)
    expected_steps["example_tokens"] = restored_page["step"]
    prior_output = s.read("probabilities", "cells")
    started = time.monotonic()
    after_reply, after_comparison = s.ablate_selected(args.prompt, minimum_step=prior_output["step"] + 1)
    steering_seconds = time.monotonic() - started
    after = bar_rows(after_reply)
    if (after_comparison["completed_step"] <= before_comparison["completed_step"]
            or after_comparison["argmax_token_ids"][0] != before_comparison["argmax_token_ids"][0]):
        raise Failure("The steered comparison is stale or changed its baseline identity.")
    result["comparison"] = {"before": before_comparison, "after": after_comparison}
    expected_steps["probabilities"] = after_reply["step"]
    after_png, after_meta = s.capture("after", dict(expected_steps))
    result["checks"].extend(dict(gate, name="after." + gate["name"])
                            for gate in program_budget_checks(s.program_evidence["after"]))
    result.update(before=before, after=after, steering_seconds=steering_seconds)
    result["pixel_evidence_collected"] = before_meta is not None
    if before_meta is not None:
        reports = []
        for name, meta, png, bars in (("before", before_meta, before_png, before),
                                      ("page-2", page_meta, page_png, before), ("after", after_meta, after_png, after)):
            gates, measured = capture_checks(s.out / "captures", meta, png, graph, bars, selected=selected,
                                              baseline_top_token=baseline_top_token)
            for gate in gates:
                gate["name"] = name + "." + gate["name"]
            result["checks"].extend(gates)
            reports.append(measured)
        rows = reports[0]["row_ids"] + reports[1]["row_ids"]
        # ID items address eight row records. The real records supply stable global IDs.
        result["checks"].append(check("example_ids", rows == list(range(8)) * 2,
                                       {"id_items": rows, "second_page_records": page["elements"]}, "two actual pages of eight rows"))
        result["checks"].append(bar_direction_check(reports[0]["height"], reports[2]["height"],
                                                    predicted_ablation_effect=predicted_ablation_effect,
                                                    fixture=args.model == "tiny-lm"))
    else:
        result["checks"].append(check("hidden_capture_evidence", False, "not collected in the visible invocation",
                                       "required for hidden acceptance", required=not s.show))
    try:
        validation = s.events.complete("trace.validate")
    except Failure as error:
        validation = {"error": str(error), "passed": False}
    save(s.out / "validation.json", validation)
    result["validation"] = validation
    result["secondary_selection"] = secondary_selection(graph, validation)
    result["checks"] += math_checks(graph, examples, before, after, validation, trace["seconds"], steering_seconds,
                                     fixture=args.model == "tiny-lm",
                                     argmax_token_ids=after_comparison["argmax_token_ids"],
                                     predicted_ablation_effect=predicted_ablation_effect)
    save(s.out.parent / "result.json", result)
    if not s.show:
        bridge.details(True)
        expanded_png, expanded_meta = s.capture("methods-expanded", dict(expected_steps))
        gates, _ = capture_checks(s.out / "captures", expanded_meta, expanded_png, graph, after,
                                  selected=selected, baseline_top_token=baseline_top_token, expanded=True)
        for gate in gates:
            gate["name"] = "methods-expanded." + gate["name"]
        result["checks"].extend(gates)
        result["method_details_capture"] = expanded_png.name
        bridge.details(False)
        result["checks"].append(check("method_details_toggle", not bridge.details_expanded,
                                       {"expanded_capture": expanded_png.name, "restored": True},
                                       "expanded capture and restored compact layout"))
    # Generate while the selected feature clamp remains active.
    prior_generation = generation_baseline(s.events.complete("trace.status"), args.prompt)
    s.mcp.body("run.action", run=s.run_id, name="generate")
    def generated():
        status = s.events.complete("trace.status")
        state = s.mcp.body("run.state", run=s.run_id)
        params = {p["name"]: p["value"] for p in state["params"]}
        return generation_evidence(prior_generation, status, params, args.prompt)
    generation = s.events.until("fresh 24-token generation", generated)
    save(s.out / "generation.json", generation)
    result["checks"].append(check("generation", True, generation["completion"], "fresh completion, 2x24 token IDs, matching texts and state"))
    # Reset tests the real control and preserves unrelated intervention kinds in the runtime.
    previous = s.read("probabilities", "cells")["step"]
    s.mcp.body("run.action", run=s.run_id, name="reset")
    reset = bar_rows(s.read("probabilities", "cells", minimum_step=previous + 1))
    save(s.out / "reset.json", reset)
    baseline = {r["token"]: r["baseline"] for r in before}
    restored = all(r["token"] in baseline and abs(r["probability"] - baseline[r["token"]]) <= 1e-7 for r in reset)
    result["checks"].append(check("reset", restored, reset, "baseline restored within 1e-7"))
    current_dictionary = s.events.complete("feature.dictionary")
    result["checks"].append(check("dictionary_identity_stable", current_dictionary == result["dictionary_evidence"],
                                   current_dictionary.get("identity"), "the qualification evidence remains unchanged"))
    s.mcp.body("applet.save")
    spec = s.mcp.body("spec.get")
    (s.out / "instrument.toml").write_text(spec["text"], encoding="utf-8")
    result["acceptance_complete"] = True
    return bridge


def live_demo(session, args, result):
    """Publish a baseline instrument, then let its controls drive the model."""
    if getattr(session, "replication_profile", None) == published.PROFILE:
        return published_demo(session, args, result, collect=False)
    result["browser_assets"] = check_assets(session)
    instrument(session)
    trace = session.events.complete("trace.run", prompt=args.prompt)
    status = session.events.complete("trace.status")
    if status["state"] != "done":
        raise Failure("The actual trace did not complete.")
    graph = read_graph(session)
    save(session.out / "graph.json", graph)
    result["trace"] = status
    programs(session)
    session.ready_programs(["attribution", "probabilities"])
    session.read("attribution", "nodes", minimum_step=trace["step"])
    bridge = Bridge(session)
    session.bridge = bridge
    # Initial selection only supplies examples. It does not apply an intervention.
    selected = selected_feature(graph)
    bridge.select(selected["index"])
    rows, _ = session.example_page(0)
    bars = session.read("probabilities", "cells")
    session.capture("demo", {"attribution": trace["step"], "probabilities": bars["step"],
                             "example_tokens": rows["step"]})
    session.mcp.body("applet.save")
    spec = session.mcp.body("spec.get")
    (session.out / "instrument.toml").write_text(spec["text"], encoding="utf-8")
    state = session.mcp.body("run.state", run=session.run_id)
    if not state["attached"]:
        raise Failure("The hosted model detached before the live demo was ready.")
    result["demo_ready"] = True
    save(session.out / "demo-ready.json", {"attached": True, "browser_assets": result["browser_assets"],
                                          "capture": "demo.png", "acceptance": False})
    return bridge



def published_action(session, cell, preparation_id, *, display=True):
    """Measure a real callback through terminal reply and current publication."""
    s = session
    before = s.events.complete("trace.status")
    _, prior = published.replication_state(before)
    attempt_record = prior.get("last_attempt")
    sequence = 0 if attempt_record is None else attempt_record["sequence"]
    attachment = s.mcp.body("run.state", run=s.run_id)
    entry = {"case_id": published.action_name(cell), "prior_sequence": sequence,
             "before_status": before, "attachment_before": attachment, "completed": False}
    started = time.monotonic()
    try:
        published.require(attachment.get("attached") is True, "The producer detached before the action.")
        entry["event"] = s.action(published.action_name(cell))
        s.published_handled_events = [*getattr(s, "published_handled_events", []), entry["event"]]
        entry["callback_seconds"] = time.monotonic() - started
        status = s.events.complete("trace.status")
        entry["status"] = status
        evidence = published.validate_case(s.private_output_directory, status, cell,
                                            previous_sequence=sequence, preparation_id=preparation_id)
        prior_step = (before["inference"].get("comparison") or {}).get("completed_step")
        step = status["inference"]["comparison"]["completed_step"]
        published.require(prior_step is None or step > prior_step, "The published output step did not advance.")
        baseline_hash = getattr(s, "published_baseline_hash", None)
        published.require(baseline_hash is None or baseline_hash == evidence["baseline_array_sha256"],
                          "The published baseline arrays changed between cases.")
        s.published_baseline_hash = evidence["baseline_array_sha256"]
        s.published_last_sequence = sequence + 1
        s.published_preparation_id = evidence["preparation"]["id"]
        # The array publication and its case metadata must refer to one output step.
        if getattr(s, "published_programs_ready", False):
            reply = s.read("probabilities", "cells", minimum_step=step)
            bars = bar_rows(reply)
            comparison_evidence(status, published.PROMPT, bars)
            import numpy as np
            with np.load(published.private_path(s.private_output_directory, evidence["result"]["evidence"]["file"]),
                         allow_pickle=False) as archive:
                probabilities = archive["probabilities"]
                published.require(all(0 <= row["token"] < probabilities.shape[1]
                                      and abs(row["baseline"] - float(probabilities[0, row["token"]])) <= 1e-7
                                      and abs(row["probability"] - float(probabilities[1, row["token"]])) <= 1e-7
                                      for row in bars), "Displayed output differs from the private case arrays.")
            entry.update(bars=bars, publication_step=reply["step"])
        current = s.events.complete("trace.status")
        published.require(current["inference"] == status["inference"], "The output changed during case publication.")
        entry["attachment_after"] = s.mcp.body("run.state", run=s.run_id)
        published.require(entry["attachment_after"].get("attached") is True, "The producer detached after the action.")
        entry.update(evidence=evidence, completed=True)
        if display:
            from _published_ui import display_result
            display_result(s, evidence)
    except Exception as error:
        entry["completed"] = False
        entry["error"] = error_detail(error)
    finally:
        entry["full_action_seconds"] = time.monotonic() - started
        save(s.out / f"action-{sequence + 1}.json", entry)
    return entry


def published_cases(session, preparation_id):
    """Execute each intervention after a new baseline callback."""
    for cell in published.CASES:
        if cell == 10:
            continue
        fresh = published_action(session, 10, preparation_id)
        case = (published_action(session, cell, preparation_id) if fresh["completed"] else
                {"case_id": published.action_name(cell), "completed": False, "error": "The fresh baseline failed."})
        yield cell, fresh, case


def published_demo(session, args, result, *, collect):
    """Collect published cases or expose the same baseline through live controls."""
    s = session
    result.update(replication_profile=published.PROFILE, source_manifest_sha256=published.MANIFEST_SHA256,
                  evidence_collection_complete=False, accepted_replication=False, published_actions=[])
    result["browser_assets"] = check_assets(s)
    result["dictionary_evidence"] = s.events.complete("feature.dictionary")
    state, catalog = published.replication_state(s.events.complete("trace.status"))
    prepared = published.preparation(s.private_output_directory, catalog)
    result["preparation_id"] = prepared["id"]
    instrument(s)
    baseline = published_action(s, 10, prepared["id"])
    result["published_actions"].append(baseline)
    published.require(baseline["completed"], baseline.get("error", "The published baseline failed."))
    trace = s.events.complete("trace.run", prompt=published.PROMPT, **published.TRACE_REQUEST)
    graph, pages = published.read_graph(s, trace, token_ids=prepared["prompts"]["dallas"]["token_ids"])
    save(s.out / "graph-pages.json", pages)
    save(s.out / "graph.json", graph)
    result["trace"] = trace
    result["checks"].extend([
        check("trace_time", published.finite(trace.get("seconds")) and trace["seconds"] <= 20, trace.get("seconds"), 20),
        check("completeness", published.finite(graph["completeness"].get("max_absolute_residual"))
              and graph["completeness"]["max_absolute_residual"] <= .02, graph["completeness"], .02),
        check("complete_graph", True, graph["counts"], "all node and edge pages with one trace identity"),
    ])
    programs(s)
    s.ready_programs(["attribution", "probabilities"])
    s.published_programs_ready = True
    s.read("attribution", "nodes", minimum_step=trace["step"])
    bridge = Bridge(s)
    s.bridge = bridge
    selected = published.published_selection(graph)
    result["example_feature"] = selected
    examples = bridge.select(selected["index"])
    rows, cells = s.example_page(0)
    bars_reply = s.read("probabilities", "cells")
    bars = bar_rows(bars_reply)
    baseline_top = baseline["evidence"]["result"]["argmax_token_ids"][0]
    steps = {"attribution": trace["step"], "probabilities": bars_reply["step"], "example_tokens": rows["step"]}

    def capture_case(name, bars):
        png, metadata = s.capture(name, dict(steps))
        result["checks"].extend(dict(gate, name=name + "." + gate["name"])
                                for gate in program_budget_checks(s.program_evidence[name]))
        if metadata is None:
            result["checks"].append(check(name + ".native_pixels", False, None, "native diagnostic capture"))
            return
        gates, measured = capture_checks(s.out / "captures", metadata, png, graph, bars,
                                         selected=selected, baseline_top_token=baseline_top)
        result["checks"].extend(dict(gate, name=name + "." + gate["name"]) for gate in gates)
        result["pixel_evidence_collected"] = True

    capture_case("cell10", bars)
    if collect:
        from _checks import example_page_check
        result["checks"].append(example_page_check(examples, rows, cells, 0))
        s.mcp.body("control.set", id="page", value=1)
        s.events.complete("feature.examples", node=selected["index"], limit=16)
        page, page_cells = s.example_page(1, minimum_step=rows["step"] + 1)
        result["checks"].append(example_page_check(examples, page, page_cells, 1))
        steps["example_tokens"] = page["step"]
        capture_case("examples-page-2", bars)
        save(s.out / "example-pages.json", {"first": rows, "second": page, "first_cells": cells, "second_cells": page_cells})
        s.mcp.body("control.set", id="page", value=0)
        s.events.complete("feature.examples", node=selected["index"], limit=16)
        rows, _ = s.example_page(0, minimum_step=page["step"] + 1)
        steps["example_tokens"] = rows["step"]
        for cell, fresh, case in published_cases(s, prepared["id"]):
            result["published_actions"].extend((fresh, case))
            result["checks"].append(check(f"cell{cell}.current_evidence", fresh["completed"] and case["completed"],
                                           case.get("error"), "fresh baseline, callback, publication, private evidence, attachment"))
            if case["completed"]:
                record = case["evidence"]["result"]
                duration = max(case["full_action_seconds"], record["action_seconds"])
                result["checks"].append(check(f"cell{cell}.steering_time", duration <= 1, duration, 1))
                steps["probabilities"] = case["publication_step"]
                capture_case(f"cell{cell}", case["bars"])
            save(s.out.parent / "result.json", result)
        from parity import export_resources
        export_resources(s, s.out / "parity")
        result["evidence_collection_complete"] = all(item["completed"] for item in result["published_actions"])
        current_dictionary = s.events.complete("feature.dictionary")
        result["checks"].append(check("dictionary_identity_stable", current_dictionary == result["dictionary_evidence"],
                                       current_dictionary.get("identity"), "the qualification evidence remains unchanged"))
    s.mcp.body("applet.save")
    (s.out / "instrument.toml").write_text(s.mcp.body("spec.get")["text"], encoding="utf-8")
    attachment = s.mcp.body("run.state", run=s.run_id)
    save(s.out / "attachment.json", attachment)
    published.require(attachment.get("attached") is True, "The hosted model detached before completion.")
    if not collect:
        result["demo_ready"] = True
        save(s.out / "demo-ready.json", {"attached": True, "browser_assets": result["browser_assets"],
                                         "capture": "cell10.png", "acceptance": False, "profile": published.PROFILE})
    return bridge

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dictionary", required=True)
    p.add_argument("--prompt")
    p.add_argument("--quality-override", type=Path, help="Explicit founder decision for one dictionary and its measured quality")
    p.add_argument("--replication-profile", choices=[published.PROFILE])
    p.add_argument("--private-output-directory", type=Path, help="Existing private evidence root; --out must be beneath it")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--hidden", action="store_true", default=True)
    group.add_argument("--show", action="store_true", help="Founder-only live window")
    p.add_argument("--demo", action="store_true", help="Live instrument without the acceptance batch; hidden unless --show is set")
    p.add_argument("--out", type=Path, required=True)
    return p


def acceptance_profile(model):
    """Select a budget only for an exact supported local snapshot path."""
    if model == "tiny-lm":
        return "fixture", 12_000_000_000
    snapshot = Path(model).resolve()
    for revision, repository, family, limit in (
        (MODEL_REVISION, "models--Qwen--Qwen3.5-0.8B", "qwen", 12_000_000_000),
        (GEMMA_REVISION, "models--google--gemma-2-2b", "gemma", 40_000_000_000),
    ):
        if (snapshot.name == revision and snapshot.parent.name == "snapshots"
                and snapshot.parent.parent.name == repository):
            return family, limit
    raise Failure("Acceptance requires tiny-lm or an exact supported local snapshot path.")


def main(argv=None):
    args = parser().parse_args(argv)
    family, memory_limit = acceptance_profile(args.model)
    args.family = family
    if args.quality_override is not None:
        if family != "qwen" or args.quality_override.stat().st_size > 4096:
            raise Failure("A quality override requires Qwen and at most 4096 JSON bytes.")
        args.quality_override = validate_quality_override(json.loads(args.quality_override.read_text(encoding="utf-8")))
    args.prompt = args.prompt or (FIXTURE_PROMPT if args.model == "tiny-lm" else MODEL_PROMPT)
    if bool(args.replication_profile) != bool(args.private_output_directory):
        raise Failure("The replication profile and private output directory must appear together.")
    if args.replication_profile:
        if family != "gemma" or args.prompt != published.PROMPT:
            raise Failure("Published Dallas requires the exact Gemma snapshot and lowercase source prompt.")
        args.private_output_directory = args.private_output_directory.resolve(strict=True)
        if not args.private_output_directory.is_dir() or not args.out.resolve().is_relative_to(args.private_output_directory):
            raise Failure("The output folder must be beneath the existing private evidence root.")
    if args.model != "tiny-lm":
        args.model = str(Path(args.model).resolve())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise Failure("The output folder must be empty to preserve earlier evidence.")
    result = {"mode": family, "prompt": args.prompt, "memory_limit_bytes": memory_limit,
              "replication_profile": args.replication_profile,
              "method_disclosures": dict(METHOD_DISCLOSURES),
              "fixture_forward_hash": FIXTURE_HASH if args.model == "tiny-lm" else None,
              "checks": [], "acceptance_complete": False, "completed": False,
              "passed": False, "qwen_acceptance": False, "gemma_acceptance": False}
    if args.quality_override is not None:
        result["dictionary_quality_override"] = args.quality_override
    session = None
    completed = False
    try:
        if args.replication_profile and not (args.show or args.demo):
            # Bound the explicit model preparation and baseline before the full trace.
            session = Session(args, out / "bounded")
            _, catalog = published.replication_state(session.events.complete("trace.status"))
            prepared = published.preparation(args.private_output_directory, catalog)
            baseline = published_action(session, 10, prepared["id"], display=False)
            result["bounded_evidence"] = baseline
            save(out / "bounded-baseline.json", baseline)
            published.require(baseline["completed"], baseline.get("error", "The bounded published baseline failed."))
            session.close()
            if session.cleanup_error:
                raise Failure(session.cleanup_error)
            session = None
            peak = memory_bytes(out / "bounded")
            result["checks"].append(check("bounded_memory", peak <= memory_limit, peak, memory_limit))
            published.require(peak <= memory_limit, "The bounded published baseline exceeds the process memory limit.")
        elif not args.replication_profile:
            # Generic acceptance retains its bounded precursor.
            session = Session(args, out / "bounded")
            measured = session.events.complete("trace.run", prompt=args.prompt, max_targets=32, max_nodes=64, max_edges=256)
            save(out / "bounded-trace.json", measured)
            if family == "gemma" and args.prompt == MODEL_PROMPT:
                preflight = dallas_preflight_check(session.events.complete("trace.status"), args.prompt)
                result["checks"].append(preflight)
                if not preflight["passed"]:
                    raise Failure("The default Dallas prompt baseline top token is not ' Austin'.")
            session.close()
            if session.cleanup_error:
                raise Failure(session.cleanup_error)
            session = None
            peak = memory_bytes(out / "bounded")
            result["checks"].append(check("bounded_memory", peak <= memory_limit, peak, memory_limit))
            if peak > memory_limit:
                raise Failure("The bounded app case exceeds the process memory limit.")
        session = Session(args, out / "acceptance", show=args.show, interactive=args.demo)
        if args.show or args.demo:
            bridge = live_demo(session, args, result)
        else:
            bridge = acceptance(session, args, result)
        if args.model == "tiny-lm" and not (args.show or args.demo):
            from parity import export_resources, run_parity
            export_resources(session, out / "parity")
            session.close()
            if session.cleanup_error:
                raise Failure(session.cleanup_error)
            session = None
            parity_checks = run_parity(out / "parity")
            result["checks"].extend(parity_checks)
            if not {"fixture_parity", "parity_process_memory"} <= {gate["name"] for gate in parity_checks}:
                raise Failure("The fixture parity checks did not complete.")
            result["parity_evidence_collected"] = True
        elif args.model == "tiny-lm":
            result["parity_evidence_collected"] = False
        if args.show or args.demo:
            result["lifetime"] = "live bridge until the owned app closes"
            finish(result)
            save(out / "result.json", result)
            bridge.keep_alive()
        completed = result["acceptance_complete"] is True
    except BaseException as exc:
        if isinstance(exc, StartupFailure):
            session = exc.session
        result["error"] = error_detail(exc)
        if not isinstance(exc, Exception):
            raise
        if (args.show or args.demo) and session is not None and session.live():
            result["lifetime"] = "owned app remains open after a driver error"
            finish(result)
            save(out / "result.json", result)
            try:
                hold_after_error(session, exc)
            except BaseException as bridge_error:
                result["bridge_error"] = error_detail(bridge_error)
                if not isinstance(bridge_error, Exception):
                    raise
    finally:
        try:
            if session is not None:
                session.close()
                if session.cleanup_error:
                    raise Failure(session.cleanup_error)
            try:
                peak = memory_bytes(out / "acceptance")
                result["checks"].append(check("process_memory", peak <= memory_limit, peak, memory_limit))
            except Failure as exc:
                result["checks"].append(check("process_memory", False, str(exc), memory_limit))
        except BaseException as exc:
            result["cleanup_error"] = error_detail(exc)
            result.setdefault("error", result["cleanup_error"])
            if not isinstance(exc, Exception):
                raise
        finally:
            # Publication requires the full driver, including parity and cleanup.
            result["completed"] = completed and not result.get("error")
            code = finish(result)
            save(out / "result.json", result)
            if args.replication_profile and not (args.show or args.demo):
                from _replication_review import write_collection
                try:
                    write_collection(out, args.private_output_directory, result)
                except Exception as error:
                    result.update(error=error_detail(error), evidence_collection_complete=False)
                    code = finish(result)
                    save(out / "result.json", result)
    if args.show or args.demo:
        return 0 if result.get("demo_ready") and not result.get("error") and not result.get("cleanup_error") else 1
    if args.replication_profile:
        return 0 if result.get("evidence_collection_complete") and not result.get("error") else 1
    return code


if __name__ == "__main__":
    def shutdown(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, shutdown)
    raise SystemExit(main())
