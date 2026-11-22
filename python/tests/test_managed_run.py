"""A managed run without the app: each test starts a script as the model runner of the Caliper app
does (the CALIPER_RUN_* variables, output into output.log) and acts as the runner over the stream
link that the script writes into its link file. The link file is written, the parameter values
apply, the action caliper.checkpoint writes a loadable file, a fork resumes from it at the
checkpoint's step with a changed parameter and its first loss continues the parent's,
caliper.stop ends the process with exit code 0, watch(model) alone serves an inference loop, and a
run ends when the process that started it ends without ending it.

The scripts pace their loops with a short sleep; the tests wait only on files and messages."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import stat
import subprocess
import sys
import time
import types
import urllib.parse

import pytest
import torch

import caliper
from caliper import _controls as C
from caliper import _managed as M
from caliper import _wire as w

pytest.importorskip("websockets")
from test_net_controls import MODES, Viewer, run  # noqa: E402

PYTHONPATH = str(pathlib.Path(caliper.__file__).parent.parent)

TRAIN = """
import json, sys, time
import torch
import caliper

torch.manual_seed(0)
torch.set_num_threads(1)
model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
x = torch.randn(64, 8)
y = x.sum(1, keepdim=True).sin()
noise = caliper.param("noise", 0.0, min=0.0, max=1.0)    # before watch()
caliper.watch(model, opt)
depth = caliper.param("depth", 1, min=1, max=8)          # after watch()
out = open(sys.argv[1], "a", buffering=1)
for _ in range(20000):
    loss = (model(x) - y).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    out.write(json.dumps({"step": caliper._CTL.n, "loss": loss.item(),
                          "lr": opt.param_groups[0]["lr"], "noise": noise.value,
                          "depth": depth.value}) + "\\n")
    time.sleep(0.005)
"""

INFER = """
import time
import torch
import caliper

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU())
caliper.watch(model)
x = torch.randn(16, 8)
with torch.no_grad():
    for _ in range(20000):
        model(x)
        time.sleep(0.005)
"""


class Run:
    """A script started as a managed run, ended by its pid at the end of the test."""

    def __init__(self, tmp, script, run_id, mode, params=None, resume=None, parent=None):
        self.dir = tmp / "runs" / run_id
        self.dir.mkdir(parents=True)
        self.link_file = self.dir / "link"
        self.steps = self.dir / "steps.jsonl"
        src = tmp / "script.py"
        if not src.exists():
            src.write_text(script)
        env = dict(os.environ, PYTHONPATH=PYTHONPATH, CALIPER_SESSION="0",
                   CALIPER_NETIO="1" if mode == "native" else "0",
                   CALIPER_RUN_ID=run_id, CALIPER_RUN_DIR=str(self.dir),
                   CALIPER_RUN_LINK_FILE=str(self.link_file))
        for k in ("CALIPER_RUN_PARAMS", "CALIPER_RUN_RESUME", "CALIPER_RUN_PARENT"):
            env.pop(k, None)
        if params is not None:
            env["CALIPER_RUN_PARAMS"] = json.dumps(params)
        if resume is not None:
            env["CALIPER_RUN_RESUME"] = str(resume)
        if parent is not None:
            env["CALIPER_RUN_PARENT"] = parent
        self.log = open(self.dir / "output.log", "wb")
        self.proc = subprocess.Popen([sys.executable, str(src), str(self.steps)], env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)

    def output(self) -> str:
        return (self.dir / "output.log").read_text(errors="replace")

    def wait_for(self, fn, what, timeout=60.0):
        end = time.monotonic() + timeout
        while True:
            got = fn()
            if got:
                return got
            assert self.proc.poll() is None, f"the script ended before {what}:\n{self.output()}"
            assert time.monotonic() < end, f"no {what} after {timeout} s:\n{self.output()}"
            time.sleep(0.02)

    def link(self) -> str:
        return self.wait_for(lambda: self.link_file.exists() and self.link_file.read_text(),
                             "link file")

    def lines(self) -> list:
        if not self.steps.exists():
            return []
        return [json.loads(s) for s in self.steps.read_text().splitlines() if s.endswith("}")]

    def line_at(self, step: int) -> dict:
        return self.wait_for(lambda: next((ln for ln in self.lines() if ln["step"] == step),
                                          None), f"step {step}")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()                      # by its own pid
        self.proc.wait(10)
        self.log.close()


@pytest.fixture
def runs():
    made = []
    yield made
    for r in made:
        r.close()


def _stream(link: str) -> types.SimpleNamespace:
    u = urllib.parse.urlsplit(link)
    token = urllib.parse.parse_qs(u.query)["token"][0]
    return types.SimpleNamespace(stream_url=f"ws://{u.netloc}{w.PATH}?{w.TOKEN_PARAM}={token}")


def _act(link: str, name: str):
    """Run the action ``name`` as the runner does; returns (CONTROLS before it, its last reply)."""
    async def go():
        async with Viewer(_stream(link)) as v:
            state = await v.state(lambda m: m.run["state"] == "running" and {
                a["name"] for a in m.actions} >= {M.CHECKPOINT, M.STOP}, timeout=30)
            r = await v.control("run", name=name)
            assert r.status == "queued", r
            done = await v.expect(w.ControlReply, lambda m: m.request_id == r.request_id,
                                  timeout=30)
            return state, done
    return run(go(), timeout=90)


@pytest.mark.parametrize("mode", MODES)
def test_fork_resumes_with_a_changed_parameter(tmp_path, runs, mode):
    parent = Run(tmp_path, TRAIN, "parent", mode, params={"lr": 0.02, "noise": 0.5, "depth": 3})
    runs.append(parent)
    link = parent.link()
    assert link.startswith("http://127.0.0.1:") and "token=" in link and "\n" not in link
    assert stat.S_IMODE(parent.link_file.stat().st_mode) == 0o600
    assert link in parent.output()                            # the link that serve() printed

    state, done = _act(link, M.CHECKPOINT)
    params = {p["name"]: p["value"] for p in state.params}
    assert params == {"noise": 0.5, "lr": 0.02, "depth": 3}   # before, inside and after watch()
    assert done.status == "ok", done
    path = pathlib.Path(done.message)
    assert path.parent == parent.dir and path.name.startswith("checkpoint-")

    ckpt = torch.load(path, weights_only=True)                # a loadable torch file
    step = ckpt["step"]
    assert path.name == f"checkpoint-{step}.pt" and step >= 1
    assert ckpt["format"] == M.FORMAT and ckpt["run"] == "parent" and ckpt["parent"] is None
    assert ckpt["params"] == {"noise": 0.5, "lr": 0.02, "depth": 3}
    assert ckpt["optimizer"]["state"]                         # the momentum buffers
    torch.manual_seed(0)
    torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(),
                        torch.nn.Linear(16, 1)).load_state_dict(ckpt["model"])
    first = parent.line_at(1)
    assert first["lr"] == 0.02 and first["noise"] == 0.5 and first["depth"] == 3
    at = parent.line_at(step)
    nxt = parent.line_at(step + 1)

    child = Run(tmp_path, TRAIN, "child", mode, params={"lr": 0.01}, resume=path,
                parent="parent")
    runs.append(child)
    c0 = child.wait_for(lambda: child.lines()[:2] if len(child.lines()) >= 2 else None,
                        "two steps of the fork")
    assert c0[0]["step"] == step and c0[1]["step"] == step + 1
    assert c0[0]["loss"] == pytest.approx(at["loss"], rel=1e-6, abs=1e-9)
    assert c0[0]["lr"] == 0.01 and c0[0]["noise"] == 0.5 and c0[0]["depth"] == 3
    assert c0[1]["loss"] != nxt["loss"]                       # the new rate changed the weights
    assert f"resumes from {path} at step {step}" in child.output()

    for r in (parent, child):
        _, done = _act(r.link(), M.STOP)
        assert done.status == "ok" and done.message.startswith("stopping at step"), done
        assert r.proc.wait(30) == 0, r.output()
        assert "stops at step" in r.output()


@pytest.mark.parametrize("mode", MODES)
def test_watch_model_alone_serves_an_inference_loop(tmp_path, runs, mode):
    r = Run(tmp_path, INFER, "infer", mode)
    runs.append(r)
    state, done = _act(r.link(), M.CHECKPOINT)
    assert state.params == [] and done.status == "ok", done
    ckpt = torch.load(done.message, weights_only=True)
    assert ckpt["optimizer"] is None and ckpt["step"] >= 1        # each forward pass is a step
    assert set(ckpt["model"]) == {"0.weight", "0.bias"}
    _, done = _act(r.link(), M.STOP)
    assert done.status == "ok", done
    assert r.proc.wait(30) == 0, r.output()


def test_values_that_do_not_fit_warn_or_clamp(tmp_path, monkeypatch):
    """A value of the wrong type warns and keeps the script's value; a number out of range is
    clamped like a value from a viewer; a parameter the runner names nothing for is untouched."""
    monkeypatch.setenv("CALIPER_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("CALIPER_RUN_PARAMS", json.dumps({"a": "high", "b": 99, "c": 0.25}))
    ctl = C.Controls()
    with pytest.warns(RuntimeWarning, match="'a'"):
        assert ctl.param("a", 1.0).value == 1.0
    assert ctl.param("b", 2, min=0, max=10).value == 10
    assert ctl.param("c", 0.5).value == 0.25
    assert ctl.param("d", True).value is True
    monkeypatch.setenv("CALIPER_RUN_PARAMS", "[1, 2]")
    with pytest.warns(RuntimeWarning, match="not a JSON object"):
        assert M.current().values == {}
    monkeypatch.delenv("CALIPER_RUN_DIR")
    assert M.current() is None and ctl.param("c", 0.5).value == 0.5


LAUNCH = """
import os, subprocess, sys, time
script, steps, pid_file, log, link = sys.argv[1:6]
child = subprocess.Popen([sys.executable, script, steps], stdout=open(log, "wb"),
                         stderr=subprocess.STDOUT)
with open(pid_file, "w") as f:
    f.write(str(child.pid))
end = time.monotonic() + 60
while not os.path.exists(link) and child.poll() is None and time.monotonic() < end:
    time.sleep(0.02)
os._exit(0)                       # as in a crash of the app: nothing ends the child
"""


@pytest.mark.skipif(os.name != "posix", reason="the job object of the runner covers Windows")
def test_a_run_ends_when_the_process_that_started_it_ends(tmp_path):
    """The parent of a serving managed run exits without ending it; the run sees its new parent
    and exits within a few seconds, and says why in its output."""
    run_dir = tmp_path / "runs" / "orphan"
    run_dir.mkdir(parents=True)
    script, launch = tmp_path / "script.py", tmp_path / "launch.py"
    script.write_text(TRAIN)
    launch.write_text(LAUNCH)
    pid_file, log, link = tmp_path / "pid", run_dir / "output.log", run_dir / "link"
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, CALIPER_SESSION="0", CALIPER_NETIO="0",
               CALIPER_RUN_ID="orphan", CALIPER_RUN_DIR=str(run_dir),
               CALIPER_RUN_LINK_FILE=str(link))
    for k in ("CALIPER_RUN_PARAMS", "CALIPER_RUN_RESUME", "CALIPER_RUN_PARENT"):
        env.pop(k, None)
    subprocess.run([sys.executable, str(launch), str(script), str(run_dir / "steps.jsonl"),
                    str(pid_file), str(log), str(link)], env=env, timeout=120, check=True)
    pid = int(pid_file.read_text())
    alive = True
    end = time.monotonic() + 15
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.05)
    if alive:
        os.kill(pid, signal.SIGKILL)      # by its own pid, so that no process is left
    text = log.read_text(errors="replace")
    assert link.exists(), text
    assert not alive, f"the run {pid} outlived its parent:\n{text}"
    assert "that started this managed run ended" in text
