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
from _checks import (FIXTURE_HASH, FIXTURE_PROMPT, QWEN_PROMPT, bar_rows, capture_checks,
                     check, finish, math_checks, selected_feature)

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    def __init__(self, args, out, *, show=False):
        self.out, self.show = out, show
        self.app = self.mcp = self.events = None
        self.app_process = None
        self.app_owner = None
        self.run_id = "circuit-trace"
        self.started = False
        self.capture_number = 0
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
                   CALIPER_USER_APPLETS_DIR=str(out / "user-applets"), CALIPER_WINDOW_SIZE="1600x1000",
                   CALIPER_HIDDEN="0" if show else "1", TMPDIR=str(out / "temp"),
                   PYTHONPYCACHEPREFIX=str(out / "pycache"))
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
            self.mcp.body("run.start", kind="inference", id=self.run_id, model=args.model,
                          dictionary=str(Path(args.dictionary).resolve()), prompt=args.prompt)
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
            self.events.until("hosted inference readiness", ready)
        except BaseException:
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

    def read(self, program, resource, *, minimum_step=0, count=1024, first=0):
        advances = 0
        def published():
            nonlocal advances
            reply = self.mcp.body("program.read", id=program, resource=resource, first=first, count=count)
            if reply["step"] is not None and reply["step"] >= minimum_step:
                return reply
            # An idle hosted run has no automatic step stream. This existing
            # metadata request advances a boundary without another model forward.
            # Its real publication remains the completion condition.
            advances += 1
            if advances > 64:
                raise Failure(f"No {program}/{resource} publication after 64 requested boundaries.")
            self.ready_programs([program])
            self.events.complete("feature.dictionary")
            return False
        return self.events.until(f"published {program}/{resource}", published)

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
        def rendered():
            statuses = self.mcp.body("program.status")["programs"]
            by_id = {p["id"]: p for p in statuses}
            for identifier, minimum in expected_steps.items():
                program = by_id.get(identifier, {})
                draw = program.get("draw", {})
                if program.get("error") or draw.get("error"):
                    raise Failure(f"The {identifier} program reports an error: {program}")
                if draw.get("resource_step") is None or draw["resource_step"] < minimum:
                    return False
            return True
        self.events.until("rendered resource steps", rendered)
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
    m = session.mcp
    m.body("spec.new", id="circuit-trace", title="Circuit Trace")
    for identifier, default, maximum in (("selected_node", -1, 511), ("example_page", 0, 1)):
        m.body("state.create", id=identifier, type="integer", default=default, min=default, max=maximum)
    m.body("state.create", id="activation_max", type="number", default=1., min=1e-6, max=1e20)
    for identifier, title in (("graph", "Attribution Graph"), ("examples", "Feature Examples"), ("output", "Output Probabilities")):
        options = {"axes": False, "labels": True, "scale_bar": False}
        if identifier == "graph":
            options["select"] = {"state": "selected_node", "field": "index"}
        m.body("panel.create", id=identifier, primitive="scene", title=title, options=options)
    controls = [("prompt", "text", "param:prompt", "Prompt", {}),
                ("clamp", "slider", "param:clamp", "Feature Clamp", {"min": -4, "max": 4}),
                ("retrace", "toggle", "param:retrace", "Automatic Trace", {}),
                ("reset", "button", "action:reset", "Feature Reset", {}),
                ("generate", "button", "action:generate", "Token Generation", {}),
                ("page", "segmented", "state:example_page", "Example Page",
                 {"choices": [{"value": 0, "label": "1–8"}, {"value": 1, "label": "9–16"}]}),
                ("baseline_text", "text", "param:output.baseline", "Baseline Tokens", {}),
                ("steered_text", "text", "param:output.steered", "Steered Tokens", {})]
    for identifier, kind, bind, title, options in controls:
        m.body("control.create", id=identifier, kind=kind, bind=bind, title=title, options=options)
    # Controls occupy one strip beside the three acceptance panels.
    nodes = [{"id": "main", "split": "vertical", "ratio": .10, "children": ["prompt", "body"]},
             {"id": "body", "split": "horizontal", "ratio": .66, "children": ["graph", "right"]},
             {"id": "right", "split": "vertical", "ratio": .37, "children": ["examples", "lower"]},
             {"id": "lower", "split": "vertical", "ratio": .48, "children": ["output", "controls"]}]
    control_ids = [row[0] for row in controls if row[0] != "prompt"]
    for index in range(len(control_ids) - 1):
        nodes.append({"id": "controls" if index == 0 else f"controls-{index}", "split": "vertical",
                      "ratio": 1 / (len(control_ids) - index),
                      "children": [control_ids[index], control_ids[index + 1] if index + 2 == len(control_ids)
                                   else f"controls-{index + 1}"]})
    m.body("layout.set", layout={"root": "main", "nodes": nodes})


def programs(session, *, examples=False):
    if examples:
        session.mcp.body("program.create", id="example_tokens", panel="examples", piece={"name": "token_strip", "version": 1},
                         bind={key: "live:examples." + key for key in ("tokens", "activations", "example_ids", "valid_tokens", "label_text")},
                         params={"page": {"state": "example_page"}, "activation_max": {"state": "activation_max"}})
        return
    session.mcp.body("program.create", id="attribution", panel="graph", piece={"name": "attribution_graph", "version": 1},
                     bind={"graph_counts": "live:trace.counts", **{k: "live:trace." + k for k in
                           ("node_i", "node_f", "edge_i", "edge_f", "label_text")}},
                     params={"selected_node": {"state": "selected_node"}})
    session.mcp.body("program.create", id="probabilities", panel="output", piece={"name": "logit_bars", "version": 1},
                     bind={key: "live:output." + key for key in ("probs", "tokens", "label_text")})


class Bridge:
    def __init__(self, session):
        self.session = session
        self.last_examples = None
        self.last_selection = None

    def consume(self):
        s = self.session
        queued, s.events.pending = s.events.pending, []
        for event in queued:
            if event["kind"] == "pick" and event.get("panel") == "graph":
                fields = event.get("fields") or {}
                index = fields.get("index", event.get("index"))
                if fields.get("kind") != 1:
                    s.parameter("selected_node", -1)
                    self.last_selection = None
                    continue
                s.parameter("selected_node", int(index))
                feature = s.events.complete("feature.show", node=int(index))
                self.last_examples = s.events.complete("feature.examples", node=int(index), limit=16)
                s.mcp.body("state.set", id="activation_max", value=max(1e-6, feature["maximum"]))
                self.last_selection = int(index)
                save(s.out / "selected-examples.json", self.last_examples)
            elif event["kind"] == "control" and event.get("control") == "page":
                # A boundary republishes the derive resources after the app's page state changes.
                s.events.complete("feature.examples", node=self.last_selection, limit=16) if self.last_selection is not None else None

    def select(self, row):
        self.session.mcp.body("selection.set", panel="graph", program="attribution", item=row)
        def selected():
            self.consume()
            return self.last_examples if self.last_selection == row else False
        return self.session.events.until("feature pick bridge", selected)

    def keep_alive(self):
        while self.session.live():
            try:
                self.session.events.wait(1)
                self.consume()
            except Failure:
                if not self.session.live():
                    return
                raise


def acceptance(session, args, result):
    s = session
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
    result["selected_feature"] = selected
    programs(s)
    s.ready_programs(["attribution", "probabilities"])
    s.events.complete("feature.dictionary")
    s.read("attribution", "nodes", minimum_step=trace["step"])
    bridge = Bridge(s)
    s.bridge = bridge
    examples = bridge.select(selected["index"])
    programs(s, examples=True)
    s.ready_programs(["example_tokens"])
    s.events.complete("feature.examples", node=selected["index"], limit=16)
    first_page = s.read("example_tokens", "rows")
    first_cells = s.read("example_tokens", "cells", minimum_step=first_page["step"])
    before_reply = s.read("probabilities", "cells")
    before = bar_rows(before_reply)
    # A resource barrier confirms selected-label derivation before the capture.
    labels = s.read("attribution", "labels", minimum_step=before_reply["step"])
    expected_steps = {"attribution": labels["step"], "example_tokens": first_page["step"],
                      "probabilities": before_reply["step"]}
    result["checks"].append(check("selection_only", all(abs(r["probability"] - r["baseline"]) <= 1e-7 for r in before),
                                   before, "selection preserves the baseline"))
    before_png, before_meta = s.capture("before", dict(expected_steps))
    s.mcp.body("control.set", id="page", value=1)
    s.events.complete("feature.examples", node=selected["index"], limit=16)
    page = s.read("example_tokens", "rows", minimum_step=first_page["step"] + 1)
    page_cells = s.read("example_tokens", "cells", minimum_step=page["step"])
    expected_steps["example_tokens"] = page["step"]
    from _checks import example_page_check
    result["checks"].extend([example_page_check(examples, first_page, first_cells, 0),
                             example_page_check(examples, page, page_cells, 1)])
    save(s.out / "example-pages.json", {"first": first_page, "second": page,
                                        "first_cells": first_cells, "second_cells": page_cells})
    page_png, page_meta = s.capture("examples-page-2", dict(expected_steps))
    s.mcp.body("control.set", id="page", value=0)
    s.events.complete("feature.examples", node=selected["index"], limit=16)
    restored_page = s.read("example_tokens", "rows", minimum_step=page["step"] + 1)
    expected_steps["example_tokens"] = restored_page["step"]
    prior_output = s.read("probabilities", "cells")
    started = time.monotonic()
    s.parameter("clamp", 0.)
    after_reply = s.read("probabilities", "cells", minimum_step=prior_output["step"] + 1)
    steering_seconds = time.monotonic() - started
    after = bar_rows(after_reply)
    expected_steps["probabilities"] = after_reply["step"]
    after_png, after_meta = s.capture("after", dict(expected_steps))
    validation = s.events.complete("trace.validate")
    save(s.out / "validation.json", validation)
    result["checks"] += math_checks(graph, examples, before, after, validation, trace["seconds"], steering_seconds,
                                     fixture=args.model == "tiny-lm")
    result.update(before=before, after=after, validation=validation)
    result["pixel_evidence_collected"] = before_meta is not None
    if before_meta is not None:
        reports = []
        for name, meta, png, bars in (("before", before_meta, before_png, before),
                                      ("page-2", page_meta, page_png, before), ("after", after_meta, after_png, after)):
            gates, measured = capture_checks(s.out / "captures", meta, png, graph, bars, selected=selected)
            for gate in gates:
                gate["name"] = name + "." + gate["name"]
            result["checks"].extend(gates)
            reports.append(measured)
        rows = reports[0]["row_ids"] + reports[1]["row_ids"]
        # ID items address eight row records. The real records supply stable global IDs.
        result["checks"].append(check("example_ids", rows == list(range(8)) * 2,
                                       {"id_items": rows, "second_page_records": page["elements"]}, "two actual pages of eight rows"))
        result["checks"].append(check("shorter_bar", reports[2]["height"] < reports[0]["height"],
                                       [reports[0]["height"], reports[2]["height"]], "after < before", required=args.model != "tiny-lm"))
    else:
        result["checks"].append(check("hidden_capture_evidence", False, "not collected in the visible invocation",
                                       "required for hidden acceptance", required=not s.show))
    # Generate while the selected feature clamp remains active.
    s.mcp.body("run.action", run=s.run_id, name="generate")
    def generated():
        state = s.mcp.body("run.state", run=s.run_id)
        params = {p["name"]: p["value"] for p in state["params"]}
        return params if params.get("output.baseline") and params.get("output.steered") else None
    save(s.out / "generation.json", s.events.until("24-token generation", generated))
    # Reset tests the real control and preserves unrelated intervention kinds in the runtime.
    previous = s.read("probabilities", "cells")["step"]
    s.mcp.body("run.action", run=s.run_id, name="reset")
    reset = bar_rows(s.read("probabilities", "cells", minimum_step=previous + 1))
    save(s.out / "reset.json", reset)
    baseline = {r["token"]: r["baseline"] for r in before}
    restored = all(r["token"] in baseline and abs(r["probability"] - baseline[r["token"]]) <= 1e-7 for r in reset)
    result["checks"].append(check("reset", restored, reset, "baseline restored within 1e-7"))
    s.mcp.body("applet.save")
    spec = s.mcp.body("spec.get")
    (s.out / "instrument.toml").write_text(spec["text"], encoding="utf-8")
    return bridge


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dictionary", required=True)
    p.add_argument("--prompt")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--hidden", action="store_true", default=True)
    group.add_argument("--show", action="store_true", help="Founder-only live window")
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.prompt = args.prompt or (FIXTURE_PROMPT if args.model == "tiny-lm" else QWEN_PROMPT)
    if args.model != "tiny-lm":
        args.model = str(Path(args.model).resolve())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise Failure("The output folder must be empty to preserve earlier evidence.")
    result = {"mode": "fixture" if args.model == "tiny-lm" else "qwen", "prompt": args.prompt,
              "fixture_forward_hash": FIXTURE_HASH if args.model == "tiny-lm" else None,
              "checks": [], "passed": False, "qwen_acceptance": False}
    session = None
    try:
        # One bounded app case establishes measured memory before the full-cap case.
        preflight = Session(args, out / "bounded")
        try:
            measured = preflight.events.complete("trace.run", prompt=args.prompt, max_targets=32, max_nodes=64, max_edges=256)
            save(out / "bounded-trace.json", measured)
        finally:
            preflight.close()
        if preflight.cleanup_error:
            raise Failure(preflight.cleanup_error)
        peak = memory_bytes(out / "bounded")
        result["checks"].append(check("bounded_memory", peak <= 12_000_000_000, peak, 12_000_000_000))
        if peak > 12_000_000_000:
            raise Failure("The bounded app case exceeds the process memory limit.")
        session = Session(args, out / "acceptance", show=args.show)
        bridge = acceptance(session, args, result)
        if args.model == "tiny-lm" and not args.show:
            from parity import export_resources, run_parity
            export_resources(session, out / "parity")
            session.close()
            if session.cleanup_error:
                raise Failure(session.cleanup_error)
            session = None
            result["checks"].extend(run_parity(out / "parity"))
            result["parity_evidence_collected"] = True
        elif args.model == "tiny-lm":
            result["parity_evidence_collected"] = False
        if args.show:
            result["lifetime"] = "live bridge until the owned app closes"
            finish(result)
            save(out / "result.json", result)
            bridge.keep_alive()
    except Exception as exc:
        result["error"] = str(exc)
        if args.show and session is not None and hasattr(session, "bridge") and session.live():
            result["lifetime"] = "live bridge until the owned app closes"
            finish(result)
            save(out / "result.json", result)
            try:
                session.bridge.keep_alive()
            except Exception as bridge_error:
                result["bridge_error"] = str(bridge_error)
    finally:
        if session is not None:
            session.close()
            if session.cleanup_error:
                result["error"] = session.cleanup_error
        try:
            peak = memory_bytes(out / "acceptance")
            result["checks"].append(check("process_memory", peak <= 12_000_000_000, peak, 12_000_000_000))
        except Failure as exc:
            result["checks"].append(check("process_memory", False, str(exc), 12_000_000_000))
        code = finish(result)
        save(out / "result.json", result)
    return code


if __name__ == "__main__":
    def shutdown(signum, frame):
        raise Failure("The driver received a shutdown signal.")
    signal.signal(signal.SIGTERM, shutdown)
    raise SystemExit(main())
