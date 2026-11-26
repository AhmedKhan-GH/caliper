#!/usr/bin/env python3
"""Live Circuit Trace demo and Gemma 2 2B Dallas acceptance driver via Caliper MCP."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _mcp import Events, Failure, MCP, attempt, stop, wait_file  # noqa: E402
from _published_ui import (  # noqa: E402
    DALLAS_PROMPT,
    DEFAULT_PROMPT,
    PROMPT_PRESETS,
    build_instrument,
    group_text,
    load_dallas_manifest,
    preset_index_for_prompt,
    preset_prompt,
    preset_prompt_by_name,
    published_trace_options,
)
from _spec import scene_programs  # noqa: E402

FIXTURE_PROMPT = "ABCD"
BF16_ULP = 2**-10


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_startup_prompt(
    model: str,
    *,
    prompt: str | None = None,
    preset: str | None = None,
    acceptance: bool = False,
) -> str:
    """Resolve startup prompt: non-city default, named preset, or Dallas acceptance."""
    if prompt is not None and preset is not None:
        raise ValueError("--prompt and --preset cannot be used together.")
    if prompt is not None:
        return prompt
    if preset is not None:
        return preset_prompt_by_name(preset)
    if acceptance:
        return DALLAS_PROMPT
    if model == "tiny-lm":
        return FIXTURE_PROMPT
    return DEFAULT_PROMPT


def resolve_feature_index(dictionary: str, explicit_index: str = "") -> Path | None:
    """Locate an explicit FeatureIndex directory or auto-discover <dictionary>/index or sibling index."""
    if explicit_index:
        candidate = Path(explicit_index).expanduser()
        return candidate.resolve() if candidate.exists() else candidate
    if not dictionary:
        return None
    dict_path = Path(dictionary).expanduser()
    if not dict_path.exists():
        return None
    dict_path = dict_path.resolve()
    candidates = (
        [dict_path, dict_path / "index", dict_path.parent / "index"]
        if dict_path.is_dir()
        else [dict_path.parent / "index"]
    )
    for candidate in candidates:
        if (candidate / "index.json").is_file():
            return candidate
    return None


def find_executable(env_var: str, relative: str) -> Path:
    candidate = Path(os.environ.get(env_var, ROOT / relative)).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise Failure(f"{env_var} ({candidate}) is not a built executable.")
    return candidate


def peak_memory_bytes(folder: Path) -> int:
    peaks: list[int] = []
    for path in folder.glob("*memory*.txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        mac = re.search(r"^\s*(\d+)\s+peak memory footprint\s*$", text, re.MULTILINE)
        linux = re.search(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$", text, re.MULTILINE)
        if mac is not None:
            peaks.append(int(mac.group(1)))
        elif linux is not None:
            peaks.append(int(linux.group(1)) * 1024)
    return max(peaks) if peaks else 0


def install_programs(session, *, examples: bool = False, state_defaults: dict | None = None) -> None:
    if examples:
        scene_programs(
            session,
            [
                {
                    "id": "example_tokens",
                    "panel": "examples",
                    "piece": {"name": "token_strip", "version": 1},
                    "bind": {
                        k: f"live:examples.{k}"
                        for k in ("tokens", "activations", "example_ids", "valid_tokens", "label_text")
                    },
                    "params": {
                        "page": {"state": "example_page"},
                        "activation_max": {"state": "activation_max"},
                    },
                }
            ],
            state_defaults=state_defaults,
        )
        return
    scene_programs(
        session,
        [
            {
                "id": "attribution",
                "panel": "graph",
                "piece": {"name": "attribution_graph", "version": 1},
                "bind": {
                    "graph_counts": "live:trace.counts",
                    **{k: f"live:trace.{k}" for k in ("node_i", "node_f", "edge_i", "edge_f", "label_text")},
                },
                "params": {"selected_node": {"state": "selected_node"}},
            },
            {
                "id": "probabilities",
                "panel": "output",
                "piece": {"name": "logit_bars", "version": 1},
                "bind": {k: f"live:output.{k}" for k in ("probs", "tokens", "label_text")},
            },
        ],
        state_defaults=state_defaults,
    )


class Session:
    """Own one hidden (or interactive) Caliper app, one MCP client, and one hosted inference run."""

    def __init__(self, args, out: Path, *, show: bool = False):
        self.args = args
        self.out = out
        self.show = show
        self.run_id = "circuit-trace"
        self.app = self.mcp = self.events = None
        self.started = False
        self.startup_error: str | None = None
        self.examples_installed = False
        self.applet_saved = False
        self.selected_node = -1
        self.current_prompt = args.prompt
        self.show_more = False
        self.progress = "ready"
        self.hosted_params: set[str] = set()
        self.hosted_actions: set[str] = set()
        self.feature_index = resolve_feature_index(str(args.dictionary), str(getattr(args, "index", "") or ""))
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "workspace", "user-applets", "temp"):
            (out / sub).mkdir(exist_ok=True)

        env = os.environ.copy()
        for key in ("CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_TEST_SPEC_UI",
                    "CALIPER_TEST_APPROVALS", "CALIPER_API", "CALIPER_APPLET_PATH", "CALIPER_SESSION"):
            env.pop(key, None)
        env.update(
            CALIPER_AGENT_OFFLINE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            CALIPER_DATA_DIR=str(out / "data"),
            CALIPER_WORKSPACE=str(out / "workspace"),
            CALIPER_USER_APPLETS_DIR=str(out / "user-applets"),
            CALIPER_WINDOW_SIZE="2048x1000",
            CALIPER_HIDDEN="0" if show else "1",
            TMPDIR=str(out / "temp"),
            PYTHONPYCACHEPREFIX=str(out / "pycache"),
        )
        if self.feature_index is not None:
            env["CALIPER_FEATURE_INDEX"] = str(self.feature_index)
        else:
            env.pop("CALIPER_FEATURE_INDEX", None)
        if args.model == "tiny-lm":
            env["CALIPER_FIXTURE_DISPLAY_MPS"] = "1"
        else:
            env.pop("CALIPER_FIXTURE_DISPLAY_MPS", None)
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT / "python"),
            str(ROOT),
            *[str(Path(p).resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p],
        ])

        perms = out / "data" / "agent-permissions.json"
        save_json(perms, {"version": 1, "modes": {"chat": "ask", "mcp": "allow_all"}, "rules": []})
        perms.chmod(0o600)

        time_bin = Path("/usr/bin/time")
        if time_bin.is_file():
            flags = ["-l"] if sys.platform == "darwin" else ["-v"]
            launcher = out / "run-python"
            launcher.write_text(
                "#!/bin/sh\nexec /usr/bin/time " + " ".join(flags)
                + " -o " + shlex.quote(str(out / "run-memory"))
                + '-$$.txt ' + shlex.quote(sys.executable) + ' "$@"\n',
                encoding="utf-8",
            )
            launcher.chmod(0o700)
            env["CALIPER_RUN_PYTHON"] = str(launcher)
        else:
            env["CALIPER_RUN_PYTHON"] = sys.executable

        self.env = env
        app_exe = find_executable("CALIPER_APP_EXE", "cmake-build-release/caliper")
        mcp_exe = find_executable("CALIPER_MCP_EXE", "cmake-build-release/src/builder/mcp/caliper-mcp")
        self.app_log = (out / "app.log").open("w", encoding="utf-8")
        app_cmd = [str(app_exe), *([] if show else ["--hidden"]), "--api", "--api-port", "0"]
        if time_bin.is_file():
            flags = ["-l"] if sys.platform == "darwin" else ["-v"]
            app_cmd = ["/usr/bin/time", *flags, "-o", str(out / "app-memory.txt"), *app_cmd]

        t0 = time.perf_counter()
        try:
            self.app = subprocess.Popen(
                app_cmd,
                env=env,
                stdout=self.app_log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
            self.api_info = wait_file(out / "data" / "api.json", process=self.app)
            self.mcp = MCP(mcp_exe, out / "data" / "api.json", out, env)
            self.events = Events(self.mcp)
            self.mcp.body(
                "run.start",
                kind="inference",
                id=self.run_id,
                model=str(args.model),
                dictionary=(
                    str(Path(args.dictionary).resolve())
                    if Path(str(args.dictionary)).exists()
                    else str(args.dictionary)
                ),
                prompt=args.prompt,
            )
            self.started = True

            def ready():
                runs = self.mcp.body("run.list")["runs"]
                run = next((r for r in runs if r["id"] == self.run_id), None)
                if run and run["state"] in ("failed", "exited"):
                    logs = self.mcp.body("run.logs", run=self.run_id, lines=80)
                    raise Failure(f"Hosted run failed: {run.get('error', '')} | logs={logs}")
                if not run or run["state"] not in ("running", "paused"):
                    return False
                state = self.mcp.body("run.state", run=self.run_id)
                names = {p["name"] for p in state["params"]}
                return state if state["attached"] and {"prompt", "selected_node", "clamp", "retrace"} <= names else False

            initial_state = self.events.until("hosted inference readiness", ready, timeout=300.0)
            self.hosted_params = {p["name"] for p in initial_state.get("params", [])}
            self.hosted_actions = {
                a["name"] for a in (initial_state.get("actions") or []) if isinstance(a, dict) and "name" in a
            }
            self.load_seconds = round(time.perf_counter() - t0, 4)
        except BaseException as exc:
            if not show or isinstance(exc, KeyboardInterrupt):
                self.close()
                raise
            self.startup_error = str(exc)

    def set_param(self, name: str, value) -> dict:
        self.mcp.body("run.set_param", run=self.run_id, name=name, value=value)

        def applied():
            state = self.mcp.body("run.state", run=self.run_id)
            param = next(p for p in state["params"] if p["name"] == name)
            return state if param.get("pending") is None and param["value"] == value else False

        return self.events.until(f"param {name}", applied)

    def run_action(self, name: str) -> dict:
        self.events.wait(0)
        first = len(self.events.pending)
        self.mcp.body("run.action", run=self.run_id, name=name)

        def completed():
            for event in self.events.pending[first:]:
                if event.get("kind") == "run" and event.get("change") == "action" and event.get("name") == name:
                    if event.get("status") != "ok":
                        raise Failure(f"Action {name} failed: {event.get('text', '')}")
                    return event
            return False

        return self.events.until(f"action {name}", completed, timeout=180.0)

    def set_progress(self, status: str) -> None:
        """Update the status row progress readout (`tracing` -> `ready`)."""
        self.progress = status
        if self.mcp is None or "output.status" in self.hosted_params:
            return
        try:
            spec_info = self.mcp.body("spec.get")
            if not spec_info.get("open"):
                return
            spec_json = spec_info.get("json") or {}
            states = {s.get("id") for s in (spec_json.get("state") or []) if isinstance(s, dict)}
            if "trace_progress" not in states:
                self.mcp.body("state.create", id="trace_progress", type="text", default="ready")
                controls = {c.get("id") for c in (spec_json.get("control") or []) if isinstance(c, dict)}
                if "trace_status" in controls:
                    self.mcp.body(
                        "control.patch",
                        id="trace_status",
                        patch={
                            "title": "Status",
                            "options": {"format": "{status:s}", "fields": {"status": {"state": "trace_progress"}}},
                        },
                    )
            self.mcp.body("state.set", id="trace_progress", value=status)
        except Exception:
            pass

    def wait_programs_rendered(self, ids: tuple[str, ...], *, min_steps: dict[str, int] | None = None) -> dict:
        required = min_steps or {}

        def drawn():
            status = self.mcp.body("program.status")
            by_id = {p["id"]: p for p in status["programs"]}
            for pid in ids:
                prog = by_id.get(pid)
                if not prog:
                    return False
                draw = prog.get("draw") or {}
                if (prog.get("error") or draw.get("error")
                        or prog.get("state") in ("refused", "stopped")
                        or draw.get("state") in ("refused", "stopped")):
                    raise Failure(f"Program {pid} failed: {prog}")
                last_step = prog.get("last_step")
                res_step = draw.get("resource_step")
                if prog.get("state") not in ("compiled", "running") or last_step is None:
                    return False
                if last_step < required.get(pid, 0):
                    return False
                if draw.get("state") != "running" or draw.get("frame_ms") is None:
                    return False
                if res_step is None or res_step < last_step:
                    return False
            return status

        return self.events.until("rendered programs", drawn, timeout=90.0)

    def finalize_applet(self) -> None:
        if not self.applet_saved:
            self.mcp.body("applet.save")
            self.mcp.body("applet.open", id="circuit-trace")
            self.applet_saved = True

    def select_feature_node(self, node_index: int) -> dict:
        self.set_param("selected_node", node_index)
        feat = self.events.complete("feature.show", node=node_index)
        examples = self.events.complete("feature.examples", node=node_index, limit=16)
        act_max = max(1e-6, float(feat["maximum"]))
        if not self.examples_installed:
            install_programs(
                self,
                examples=True,
                state_defaults={"selected_node": node_index, "activation_max": act_max},
            )
            self.examples_installed = True
            self.finalize_applet()
            self.events.complete("feature.examples", node=node_index, limit=16)
        else:
            self.mcp.body("state.set", id="selected_node", value=node_index)
            self.mcp.body("state.set", id="activation_max", value=act_max)
        self.selected_node = node_index
        self.wait_programs_rendered(("attribution", "probabilities", "example_tokens"))
        return {"feature": feat, "examples": examples}

    def preferred_feature_node(self, graph: dict) -> int:
        """Return the hosted important feature node if valid, else the highest-influence feature node."""
        features = [n for n in graph.get("nodes", []) if n.get("kind") == "feature"]
        if not features:
            return -1
        by_idx = {int(n["index"]): n for n in features}
        if self.mcp is not None and self.started:
            try:
                status = self.events.complete("trace.status")
                hosted_sel = (status.get("inference") or {}).get("selected_node")
                if isinstance(hosted_sel, int) and hosted_sel in by_idx:
                    return hosted_sel
            except Exception:
                pass
        best = max(
            features,
            key=lambda n: (
                abs(float(n.get("influence", 0.0) or 0.0)),
                abs(float(n.get("activation", 0.0) or 0.0)),
                -int(n["index"]),
            ),
        )
        return int(best["index"])

    def refresh_selected_feature(self, graph: dict | None = None) -> tuple[int, dict | None]:
        """Select a valid feature node for the active graph and refresh its examples."""
        if graph is None:
            graph = self.read_full_graph()
        feat_node = self.preferred_feature_node(graph)
        if feat_node >= 0:
            return feat_node, self.select_feature_node(feat_node)
        self.selected_node = -1
        if self.mcp is not None:
            self.mcp.body("state.set", id="selected_node", value=-1)
        if self.examples_installed:
            scene_programs(self, [], remove_examples=True)
            self.examples_installed = False
        return -1, None

    def recompute_prompt(
        self,
        prompt: str,
        *,
        show_more: bool | None = None,
        sync_prompt_param: bool = True,
        apply_current_clamp: bool = False,
    ) -> dict:
        """Retrace with published pruning, recompute output bars, regenerate text, and refresh examples."""
        expanded = self.show_more if show_more is None else bool(show_more)
        self.set_progress("tracing")
        try:
            if sync_prompt_param:
                self.set_param("prompt", prompt)
            if "show_more" in self.hosted_params:
                self.set_param("show_more", bool(expanded))
            if self.mcp is not None:
                self.mcp.body("state.set", id="prompt_preset", value=preset_index_for_prompt(prompt))
            already_traced = not apply_current_clamp and self._traced_matches(prompt, expanded)
            if already_traced:
                graph = self.read_full_graph()
                feat_node, selected_info = self.refresh_selected_feature(graph)
            elif "run" in self.hosted_actions:
                self.run_action("run")
                graph = self.read_full_graph()
                feat_node, selected_info = self.refresh_selected_feature(graph)
            else:
                self.events.complete("trace.run", prompt=prompt, **published_trace_options(show_more=expanded))
                graph = self.read_full_graph()
                feat_node, selected_info = self.refresh_selected_feature(graph)
                if apply_current_clamp and feat_node >= 0:
                    self.run_action("apply_clamp")
                else:
                    self.run_action("reset")
                self.run_action("generate")
            self.current_prompt = prompt
            self.show_more = expanded
            self.set_progress("ready")
            return {"graph": graph, "selected_node": feat_node, "selected_info": selected_info}
        except Exception:
            self.set_progress("ready")
            raise

    def capture_native(self, name: str, *, min_steps: dict[str, int] | None = None) -> Path:
        prog_ids = ("attribution", "probabilities", *(["example_tokens"] if self.examples_installed else []))
        self.wait_programs_rendered(prog_ids, min_steps=min_steps)
        reply = self.mcp.call("capture.frame")
        image = next((b for b in reply.get("content", []) if b.get("type") == "image"), None)
        encoded = image["data"] if image else reply["structuredContent"]["png_base64"]
        png = self.out / f"{name}.png"
        png.write_bytes(base64.b64decode(encoded, validate=True))
        return png

    def capture_browser(self, name: str) -> Path:
        spec_url = self.api_info.get("spec_url")
        if not spec_url:
            raise Failure("The app did not publish a spec_url in api.json.")
        target_url = f"{spec_url}&applet=circuit-trace"
        png = self.out / f"{name}.png"
        node_script = (
            "const { chromium } = require('@playwright/test');"
            "(async () => {"
            "  const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });"
            "  try {"
            "    const page = await browser.newPage({ viewport: { width: 1600, height: 960 }, deviceScaleFactor: 1 });"
            "    page.on('pageerror', (e) => console.error('PAGEERROR:', e));"
            "    page.on('console', (m) => { if (m.type() === 'error') console.error('CONSOLE:', m.text()); });"
            "    await page.goto(process.argv[1], { waitUntil: 'domcontentloaded', timeout: 30000 });"
            "    try {"
            "      await page.waitForFunction((wantCount) => {"
            "        const s = window.caliperSpec?.debugState?.();"
            "        if (!s) return false;"
            "        const mounts = Object.values(s.programs || {});"
            "        return mounts.length >= wantCount && mounts.every((m) => m.mounted && m.programs.every((p) => p.module === 'running' && p.frames >= 1));"
            "      }, Number(process.argv[3]), { timeout: 20000, polling: 100 });"
            "    } catch (err) {"
            "      const st = await page.evaluate(() => window.caliperSpec?.debugState?.());"
            "      console.error('DEBUG_STATE:', JSON.stringify(st));"
            "      throw err;"
            "    }"
            "    await page.evaluate(() => new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok))));"
            "    await page.screenshot({ path: process.argv[2] });"
            "  } finally {"
            "    await browser.close();"
            "  }"
            "})().catch((e) => { console.error(e); process.exit(1); });"
        )
        want = "3" if self.examples_installed else "2"
        subprocess.run(
            ["node", "-e", node_script, target_url, str(png), want],
            cwd=str(ROOT / "web"),
            check=True,
            timeout=120,
        )
        return png

    def read_full_graph(self) -> dict:
        first = self.events.complete("trace.read", node_limit=256, edge_limit=256)
        graph = {**first, "nodes": list(first["nodes"]), "edges": list(first["edges"])}
        for key, target, kind in (("next_node_offset", "nodes", "node"), ("next_edge_offset", "edges", "edge")):
            offset = first[key]
            while offset is not None:
                other = "edge" if kind == "node" else "node"
                page = self.events.complete(
                    "trace.read",
                    **{f"{kind}_offset": offset, f"{kind}_limit": 256, f"{other}_limit": 0},
                )
                graph[target].extend(page[target])
                offset = page[key]
            graph[key] = None
        return graph

    def show_error(self, message: str) -> None:
        text = str(message).strip() or "Circuit Trace encountered an error."
        print(f"error: {text}", file=sys.stderr, flush=True)
        if self.mcp is None or self.app is None or self.app.poll() is not None:
            return
        clipped = text[:1000]
        try:
            spec_info = self.mcp.body("spec.get")
            if not spec_info.get("open"):
                self.mcp.body("spec.new", id="circuit-trace-error", title="Circuit Trace Error")
                self.mcp.body("panel.create", id="error", primitive="text", title="Error", options={"text": clipped})
                self.mcp.body("layout.set", layout={"root": "error", "nodes": []})
                return
            spec_json = spec_info.get("json") or {}
            states = {s.get("id") for s in (spec_json.get("state") or []) if isinstance(s, dict)}
            if "error_status" not in states:
                self.mcp.body("state.create", id="error_status", type="text", default=clipped)
            else:
                self.mcp.body("state.set", id="error_status", value=clipped)
            controls = {c.get("id") for c in (spec_json.get("control") or []) if isinstance(c, dict)}
            if "clamp_status" in controls:
                self.mcp.body(
                    "control.patch",
                    id="clamp_status",
                    patch={
                        "title": "Error",
                        "options": {"format": "{status:s}", "fields": {"status": {"state": "error_status"}}},
                    },
                )
        except Exception:
            pass

    def _hosted_prompt_and_detail(self) -> tuple[str, bool]:
        if self.mcp is None or not self.started:
            return self.current_prompt, self.show_more
        state = self.mcp.body("run.state", run=self.run_id)
        by_name = {p["name"]: p for p in state.get("params", [])}
        prompt_param = by_name.get("prompt")
        prompt = (
            str(prompt_param["value"])
            if prompt_param and prompt_param.get("pending") is None and prompt_param.get("value") is not None
            else self.current_prompt
        )
        detail_param = by_name.get("show_more")
        show_more = (
            bool(detail_param["value"])
            if detail_param and detail_param.get("pending") is None and detail_param.get("value") is not None
            else self.show_more
        )
        return prompt, show_more

    def _traced_matches(self, prompt: str, show_more: bool) -> bool:
        try:
            status = self.events.complete("trace.status")
        except Exception:
            return False
        inf = status.get("inference")
        if isinstance(inf, dict) and "traced_prompt" in inf:
            if inf.get("state") != "ready":
                return False
            if inf.get("traced_prompt") != prompt:
                return False
            traced_more = inf.get("traced_show_more")
            if traced_more is None:
                return bool(show_more) == bool(self.show_more)
            return bool(traced_more) == bool(show_more)
        traced_prompt = status.get("prompt") or (status.get("trace") or {}).get("prompt")
        traced_more = status.get("show_more")
        if traced_more is None and isinstance(status.get("trace"), dict):
            traced_more = status["trace"].get("show_more")
        if traced_prompt != prompt:
            return False
        if traced_more is None:
            return bool(show_more) == bool(self.show_more)
        return bool(traced_more) == bool(show_more)

    def _wait_hosted_trace(self, prompt: str, show_more: bool) -> bool:
        """Wait for HostedInference's automatic refresh to finish for `(prompt, show_more)`."""
        def settled():
            status = self.events.complete("trace.status")
            inf = status.get("inference")
            if not isinstance(inf, dict):
                return False
            state = inf.get("state")
            if state == "error":
                return "error"
            if state != "ready":
                return False
            if inf.get("traced_prompt") == prompt and bool(inf.get("traced_show_more")) == bool(show_more):
                return True
            cur_prompt, cur_more = self._hosted_prompt_and_detail()
            if (cur_prompt, cur_more) != (prompt, bool(show_more)):
                return True
            return False

        outcome = self.events.until("hosted prompt refresh", settled, timeout=180.0)
        return outcome is True

    def handle_event(self, ev: dict, manifest: dict | None) -> None:
        if self.mcp is None:
            return
        kind = ev.get("kind")
        if manifest and kind == "control" and ev.get("control") == "published_group":
            self.mcp.body("state.set", id="published_members", value=group_text(manifest, int(ev["value"])))
            return
        if kind == "control" and ev.get("control") == "preset":
            target = preset_prompt(int(ev["value"]))
            if target is not None and target != self.current_prompt:
                if "run" in self.hosted_actions:
                    self.set_param("prompt", target)
                    if self._wait_hosted_trace(target, self.show_more):
                        self.current_prompt = target
                        self.mcp.body("state.set", id="prompt_preset", value=preset_index_for_prompt(target))
                        self.refresh_selected_feature()
                else:
                    self.recompute_prompt(target, sync_prompt_param=True, apply_current_clamp=False)
            return
        if kind == "control" and ev.get("control") == "show_more":
            expanded = bool(ev["value"])
            if expanded != self.show_more:
                if "show_more" in self.hosted_params and "run" in self.hosted_actions:
                    if self._wait_hosted_trace(self.current_prompt, expanded):
                        self.show_more = expanded
                        self.refresh_selected_feature()
                else:
                    self.recompute_prompt(
                        self.current_prompt,
                        show_more=expanded,
                        sync_prompt_param=False,
                        apply_current_clamp=False,
                    )
            return
        if (kind == "control" and ev.get("control") == "prompt") or (
            kind == "run" and ev.get("change") == "param" and ev.get("name") == "prompt"
        ):
            new_prompt = str(ev.get("value") or "")
            if new_prompt and new_prompt != self.current_prompt:
                if "run" in self.hosted_actions:
                    if self._wait_hosted_trace(new_prompt, self.show_more):
                        self.current_prompt = new_prompt
                        self.mcp.body("state.set", id="prompt_preset", value=preset_index_for_prompt(new_prompt))
                        self.refresh_selected_feature()
                elif self._traced_matches(new_prompt, self.show_more):
                    self.current_prompt = new_prompt
                    self.mcp.body("state.set", id="prompt_preset", value=preset_index_for_prompt(new_prompt))
                    self.refresh_selected_feature()
                else:
                    self.recompute_prompt(new_prompt, sync_prompt_param=False, apply_current_clamp=False)
            return
        if kind == "run" and ev.get("change") == "param" and ev.get("name") == "show_more":
            expanded = bool(ev.get("value"))
            if expanded != self.show_more:
                if "show_more" in self.hosted_params and "run" in self.hosted_actions:
                    if self._wait_hosted_trace(self.current_prompt, expanded):
                        self.show_more = expanded
                        self.refresh_selected_feature()
                elif self._traced_matches(self.current_prompt, expanded):
                    self.show_more = expanded
                    self.refresh_selected_feature()
                else:
                    self.recompute_prompt(
                        self.current_prompt,
                        show_more=expanded,
                        sync_prompt_param=False,
                        apply_current_clamp=False,
                    )
            return
        if kind == "run" and ev.get("change") == "action" and ev.get("name") == "run":
            prompt, show_more = self._hosted_prompt_and_detail()
            if ev.get("status") == "ok":
                self.current_prompt = prompt
                self.show_more = show_more
                self.mcp.body("state.set", id="prompt_preset", value=preset_index_for_prompt(prompt))
                self.refresh_selected_feature()
            elif "run" not in self.hosted_actions:
                self.recompute_prompt(
                    prompt,
                    show_more=show_more,
                    sync_prompt_param=False,
                    apply_current_clamp=True,
                )
            else:
                self.show_error(str(ev.get("text") or "Run action failed."))
            return
        if kind == "pick" and ev.get("panel") == "graph":
            fields = ev.get("fields") or {}
            idx = fields.get("index", ev.get("index"))
            if fields.get("kind") == 1 and isinstance(idx, int) and idx >= 0:
                self.select_feature_node(idx)

    def sync_from_hosted_state(self) -> None:
        """Detect prompt or show_more updates submitted directly through the host UI or agent."""
        prompt, show_more = self._hosted_prompt_and_detail()
        if prompt == self.current_prompt and show_more == self.show_more:
            return
        if "run" in self.hosted_actions:
            if self._wait_hosted_trace(prompt, show_more):
                self.current_prompt = prompt
                self.show_more = show_more
                self.mcp.body("state.set", id="prompt_preset", value=preset_index_for_prompt(prompt))
                self.refresh_selected_feature()
            return
        if self._traced_matches(prompt, show_more):
            self.current_prompt = prompt
            self.show_more = show_more
            self.mcp.body("state.set", id="prompt_preset", value=preset_index_for_prompt(prompt))
            self.refresh_selected_feature()
            return
        self.recompute_prompt(
            prompt,
            show_more=show_more,
            sync_prompt_param=False,
            apply_current_clamp=False,
        )

    def keep_alive(self, manifest: dict | None) -> None:
        while self.app is not None and self.app.poll() is None:
            try:
                self.events.wait(0.5)
            except Exception as exc:
                if self.app.poll() is not None:
                    break
                self.show_error(str(exc))
                time.sleep(0.5)
                continue
            queued, self.events.pending = self.events.pending, []
            for ev in queued:
                try:
                    self.handle_event(ev, manifest)
                except Exception as exc:
                    if self.app.poll() is not None:
                        break
                    self.show_error(str(exc))
            try:
                self.sync_from_hosted_state()
            except Exception as exc:
                if self.app.poll() is not None:
                    break
                self.show_error(str(exc))

    def close(self, *, stop_app: bool = True) -> None:
        errors: list[str] = []
        if stop_app and self.mcp is not None and self.app is not None and self.app.poll() is None and self.started:
            attempt(
                errors,
                "run stop",
                lambda: (
                    self.mcp.body("run.stop", run=self.run_id, kill_after_s=10),
                    self.events.until(
                        "hosted run exit",
                        lambda: all(r["state"] in ("failed", "exited") for r in self.mcp.body("run.list")["runs"]),
                        timeout=30.0,
                    ),
                ),
            )
        if self.mcp is not None and (stop_app or self.app is None or self.app.poll() is not None):
            attempt(errors, "MCP close", self.mcp.close)
            self.mcp = None
        if stop_app and self.app is not None:
            attempt(errors, "app stop", lambda: stop(self.app, group=True))
            self.app = None
        if (stop_app or self.app is None or self.app.poll() is not None) and hasattr(self, "app_log") and self.app_log is not None:
            attempt(errors, "app log close", self.app_log.close)


def run_dallas_acceptance(session: Session, reference_path: Path | None) -> dict:
    manifest = build_instrument(session, dallas=True, prompt=DALLAS_PROMPT)

    session.set_progress("tracing")
    t0 = time.perf_counter()
    trace_reply = session.events.complete(
        "trace.run",
        prompt=DALLAS_PROMPT,
        **published_trace_options(show_more=False),
    )
    trace_wall = round(time.perf_counter() - t0, 4)
    graph = session.read_full_graph()
    save_json(session.out / "graph-summary.json", {k: v for k, v in graph.items() if k not in ("nodes", "edges")})

    install_programs(session, examples=False)
    feat_node = next(
        (n["index"] for n in graph["nodes"] if n["kind"] == "feature" and (n["layer"], n["feature"]) == (20, 15589)),
        next(n["index"] for n in graph["nodes"] if n["kind"] == "feature"),
    )
    selected_info = session.select_feature_node(feat_node)
    session.set_progress("ready")
    baseline_png = session.capture_native("dallas-baseline-app")
    browser_baseline_png = session.capture_browser("dallas-baseline-browser") if session.args.browser_capture else None

    cell_results: list[dict] = []
    cell24_png = None
    cell24_browser_png = None
    gen_result = None
    for case in manifest["interventions"]:
        cell = int(case["cell"])
        t_cell = time.perf_counter()
        session.run_action(f"dallas.cell{cell}")
        wall_s = round(time.perf_counter() - t_cell, 4)
        status = session.events.complete("trace.status")
        comp = status["inference"]["comparison"]
        row = {
            "cell": cell,
            "label": case["label"],
            "baseline_top": comp["baseline_top"],
            "steered_top": comp["steered_top"],
            "compute_seconds": round(comp["compute_publish_seconds"], 4),
            "wall_seconds": wall_s,
        }
        if cell == 29:
            row["say_victoria_activation"] = comp["say_victoria_activation"]
        cell_results.append(row)
        if cell == 24:
            t_gen = time.perf_counter()
            session.run_action("generate")
            gen_wall = round(time.perf_counter() - t_gen, 4)
            run_state = session.mcp.body("run.state", run=session.run_id)
            by_name = {p["name"]: p["value"] for p in run_state["params"]}
            gen_result = {
                "baseline": by_name["output.baseline"],
                "steered": by_name["output.steered"],
                "seconds": gen_wall,
            }
            cell24_png = session.capture_native(
                "dallas-cell24-sacramento-app",
                min_steps={"probabilities": int(comp["completed_step"])},
            )
            if session.args.browser_capture:
                cell24_browser_png = session.capture_browser("dallas-cell24-sacramento-browser")

    t_sweep = time.perf_counter()
    sweep = session.events.complete(
        "patch.sweep",
        clean_prompt=manifest["prompts"]["oakland"],
        corrupted_prompt=manifest["prompts"]["dallas"],
        metric={"positive_token": 43939, "negative_token": 22605},
    )
    sweep_seconds = round(time.perf_counter() - t_sweep, 4)

    session.run_action("reset")
    t_val = time.perf_counter()
    validation = session.events.complete("trace.validate")
    validate_seconds = round(time.perf_counter() - t_val, 4)

    comparison_checks = {}
    if reference_path is not None:
        ref = json.loads(reference_path.read_text(encoding="utf-8"))
        ref_cases = {int(c["cell"]): c["observed_top"] for c in ref["cases"]}
        ref_graph = ref["graph"]
        comp_meta = graph["completeness"]
        graph_match = (
            comp_meta["full_nodes"] == ref_graph["full_nodes"]
            and comp_meta["kept_nodes"] == ref_graph["kept_nodes"]
            and comp_meta["kept_features"] == ref_graph["kept_features"]
        )
        base_obs = cell_results[0]["baseline_top"]
        base_exp = ref_cases[10]
        baseline_exact = (
            base_obs["id"] == base_exp["id"]
            and base_obs["text"] == base_exp["text"]
            and base_obs["probability"] == base_exp["probability"]
            and base_obs["logit"] == base_exp["logit"]
        )
        bitwise_count = 0
        all_tokens_match = True
        all_probs_within_ulp = True
        for row in cell_results:
            exp = ref_cases[row["cell"]]
            obs = row["steered_top"]
            tok_ok = obs["id"] == exp["id"] and obs["text"] == exp["text"]
            diff = abs(obs["probability"] - exp["probability"])
            exact = tok_ok and diff == 0.0 and obs["logit"] == exp["logit"]
            within_ulp = tok_ok and (diff == 0.0 or (row["cell"] == 26 and diff <= BF16_ULP))
            row["reference"] = exp
            row["abs_probability_diff"] = diff
            row["top_token_exact"] = tok_ok
            row["bitwise_equal"] = exact
            row["within_bf16_ulp"] = within_ulp
            if exact:
                bitwise_count += 1
            all_tokens_match = all_tokens_match and tok_ok
            all_probs_within_ulp = all_probs_within_ulp and within_ulp
        cell29_vic = next(r["say_victoria_activation"] for r in cell_results if r["cell"] == 29)
        comparison_checks = {
            "reference_file": str(reference_path),
            "graph_counts_exact": graph_match,
            "baseline_cell10_exact": baseline_exact,
            "all_seven_top_tokens_exact": all_tokens_match,
            "bitwise_equal_cells": bitwise_count,
            "all_probabilities_equal_or_1ulp_cell26": all_probs_within_ulp,
            "cell29_say_victoria_zero": cell29_vic == 0.0,
        }
        if not all(
            (graph_match, baseline_exact, all_tokens_match, all_probs_within_ulp, cell29_vic == 0.0)
        ):
            raise Failure(f"Reference comparison mismatch: {comparison_checks}")

    return {
        "mode": "dallas_acceptance",
        "prompt": DALLAS_PROMPT,
        "timings_seconds": {
            "hosted_load": session.load_seconds,
            "trace_service": round(trace_reply["seconds"], 4),
            "trace_wall": trace_wall,
            "interventions_compute_total": round(sum(r["compute_seconds"] for r in cell_results), 4),
            "cell24_generate": gen_result["seconds"] if gen_result else None,
            "patch_sweep": sweep_seconds,
            "trace_validate": validate_seconds,
        },
        "graph": {
            "counts": graph["counts"],
            "completeness": graph["completeness"],
            "feature_share": graph["feature_share"],
            "error_share": graph["error_share"],
        },
        "selected_feature": {
            "node": feat_node,
            "show": selected_info["feature"],
            "example_count": len(selected_info["examples"]["examples"]),
        },
        "published_interventions": cell_results,
        "cell24_generation": gen_result,
        "patch_sweep": {
            "layers": sweep["layers"],
            "positions": sweep["positions"],
            "clean_metric": sweep["clean_metric"],
            "corrupted_metric": sweep["corrupted_metric"],
        },
        "validation": {
            "spearman": validation["spearman"],
            "correct_signs": validation["correct_signs"],
            "count": validation["count"],
        },
        "captures": {
            "baseline_app": str(baseline_png),
            "baseline_browser": str(browser_baseline_png) if browser_baseline_png else None,
            "cell24_app": str(cell24_png) if cell24_png else None,
            "cell24_browser": str(cell24_browser_png) if cell24_browser_png else None,
        },
        "program_draws": {
            p["id"]: p["draw"] for p in session.mcp.body("program.status")["programs"] if p.get("draw")
        },
        "reference_comparison": comparison_checks,
    }


def run_standard(session: Session) -> dict:
    manifest = None
    try:
        manifest = build_instrument(session, dallas=False, prompt=session.args.prompt)
        session.set_progress("tracing")
        t0 = time.perf_counter()
        trace_reply = session.events.complete(
            "trace.run",
            prompt=session.args.prompt,
            **published_trace_options(show_more=False),
        )
        trace_wall = round(time.perf_counter() - t0, 4)
        graph = session.read_full_graph()
        install_programs(session, examples=False)
        feat_node = session.preferred_feature_node(graph)
        if feat_node >= 0:
            selected_info = session.select_feature_node(feat_node)
        else:
            selected_info = None
            session.finalize_applet()
        if "run" not in session.hosted_actions:
            session.run_action("generate")
        session.set_progress("ready")
        before_png = session.capture_native("before")
        before_browser = session.capture_browser("before-browser") if session.args.browser_capture else None
        if not session.show:
            if feat_node >= 0:
                session.set_param("clamp", 0.0)
                session.run_action("apply_clamp")
            comp_step = session.events.complete("trace.status")["inference"]["comparison"]["completed_step"]
            session.run_action("generate")
            validation = session.events.complete("trace.validate")
            after_png = session.capture_native("after", min_steps={"probabilities": int(comp_step)})
        else:
            validation = {"count": 0, "correct_signs": 0}
            after_png = before_png
    except Exception as exc:
        if not session.show:
            raise
        session.show_error(str(exc))
        session.keep_alive(manifest)
        return {
            "mode": "standard",
            "model": str(session.args.model),
            "prompt": session.args.prompt,
            "error": str(exc),
        }
    if session.show:
        session.keep_alive(manifest)
    return {
        "mode": "standard",
        "model": str(session.args.model),
        "prompt": session.args.prompt,
        "timings_seconds": {
            "hosted_load": session.load_seconds,
            "trace_service": round(trace_reply["seconds"], 4),
            "trace_wall": trace_wall,
        },
        "counts": graph["counts"],
        "selected_node": feat_node,
        "example_count": len(selected_info["examples"]["examples"]) if selected_info else 0,
        "validation": {"count": validation["count"], "correct_signs": validation["correct_signs"]},
        "captures": {
            "before": str(before_png),
            "before_browser": str(before_browser) if before_browser else None,
            "after": str(after_png),
        },
        "program_draws": (
            {p["id"]: p["draw"] for p in session.mcp.body("program.status")["programs"] if p.get("draw")}
            if session.mcp is not None
            else {}
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Circuit Trace MCP Instrument")
    parser.add_argument("--model", default="tiny-lm")
    parser.add_argument("--dictionary", default="")
    parser.add_argument("--index", default="", help="Optional FeatureIndex directory for feature examples.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--preset",
        choices=[key for key, _label, _text in PROMPT_PRESETS],
        default=None,
        help="Named prompt preset (default or dallas).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--browser-capture", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.show and os.environ.get("CALIPER_TEST_NO_WINDOW") == "1":
        raise SystemExit("--show is forbidden when CALIPER_TEST_NO_WINDOW=1")
    try:
        args.prompt = resolve_startup_prompt(
            args.model,
            prompt=args.prompt,
            preset=args.preset,
            acceptance=args.acceptance,
        )
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(args.output).resolve()
    if args.model == "tiny-lm" and not args.dictionary:
        from caliper.interp._hosted import fixture_resources
        args.dictionary = str(fixture_resources(out / "fixture-dict"))

    session = Session(args, out, show=args.show)
    try:
        if session.startup_error is not None:
            session.show_error(session.startup_error)
            session.keep_alive(None)
            summary = {"mode": "show", "model": str(args.model), "prompt": args.prompt, "error": session.startup_error}
        elif args.acceptance:
            summary = run_dallas_acceptance(session, args.reference)
            if args.show:
                session.keep_alive(load_dallas_manifest())
        else:
            summary = run_standard(session)
    except Exception as exc:
        if not args.show:
            raise
        session.show_error(str(exc))
        session.keep_alive(load_dallas_manifest() if args.acceptance else None)
        summary = {"mode": "dallas_acceptance" if args.acceptance else "standard", "error": str(exc)}
    finally:
        session.close(stop_app=not args.show)
    summary["peak_process_bytes"] = peak_memory_bytes(out)
    save_json(out / "result.json", summary)
    return 1 if "error" in summary else 0


if __name__ == "__main__":
    raise SystemExit(main())
