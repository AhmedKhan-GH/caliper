"""A managed run without the app: each test starts a script as the model runner of the Caliper app
does (the CALIPER_RUN_* variables, output into output.log) and acts as the runner over the stream
link that the script writes into its link file. The link file is written, the parameter values
apply, the action caliper.checkpoint writes a loadable file, a fork resumes from it at the
checkpoint's step with a changed parameter and its first loss continues the parent's,
caliper.stop ends the process with exit code 0, watch(model) alone serves an inference loop, and a
run ends when the process that started it ends without ending it, also when the app ended
before the script imported the managed-run code, and the run's files are written through new
temporary files that never follow a link.

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


# Put before a script: the server's thread hands the final replies of actions to the connections
# LATE_S seconds late, as a slow runner can schedule it, so that the script reaches its exit first.
LATE = """
import time
from caliper import _net_controls
LATE_S = 1.5
_flush = _net_controls.ControlsPort.flush

def _late(self, *args, **kwargs):
    if self._replies:
        time.sleep(LATE_S)
    return _flush(self, *args, **kwargs)

_net_controls.ControlsPort.flush = _late
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
def test_a_stop_sends_its_replies_before_the_close(tmp_path, runs, mode):
    """A checkpoint and at once a stop, as a fork followed by a stop of its parent sends them,
    while the server's thread hands the final replies over late: both final replies arrive, then
    BYE shutdown and the close code 1001, and the script exits with 0. A second connection that
    reads nothing after its HELLO does not keep the script alive."""
    from websockets.asyncio.client import connect

    r = Run(tmp_path, LATE + TRAIN, "late", mode)
    runs.append(r)
    stream = _stream(r.link())

    async def go():
        idle = await connect(stream.stream_url, max_queue=1, ping_interval=None, close_timeout=1)
        try:
            await idle.send(w.encode(w.Hello("viewer", [w.CAP_CONTROLS], {"name": "idle",
                                                                         "version": "0"})))
            async with Viewer(stream) as v:
                await v.state(lambda m: m.run["state"] == "running" and {
                    a["name"] for a in m.actions} >= {M.CHECKPOINT, M.STOP}, timeout=30)
                ck = await v.control("run", name=M.CHECKPOINT)
                st = await v.control("run", name=M.STOP)
                assert ck.status == "queued" and st.status == "queued", (ck, st)
                got = [await v.expect(w.ControlReply, lambda m, rid=q.request_id:
                                      m.request_id == rid, timeout=30) for q in (ck, st)]
                bye = await v.expect(w.Bye, timeout=30)
                await v.ws.wait_closed()
                return got, bye, v.ws.close_code, v.seen
        finally:
            idle.transport.abort()

    (ck, st), bye, code, seen = run(go(), timeout=90)
    assert ck.status == "ok" and pathlib.Path(ck.message).is_file(), ck
    assert st.status == "ok" and st.message.startswith("stopping at step"), st
    assert bye.reason == w.BYE_SHUTDOWN and code == w.CLOSE_GOING_AWAY, (bye, code)
    assert seen.index(st) < seen.index(bye)
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


@pytest.mark.skipif(os.name != "posix", reason="the job object of the runner covers Windows")
def test_a_run_ends_when_its_app_ended_before_the_import(tmp_path):
    """The app named by CALIPER_RUN_APP_PID ended before the script imported the managed-run code
    and before any watch(); the run exits at the import, with the reason in its output."""
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=60)                 # an app that ended: its id names no process now
    run_dir = tmp_path / "runs" / "early"
    run_dir.mkdir(parents=True)
    script = tmp_path / "script.py"
    script.write_text("import time\nimport caliper\ncaliper.param('lr', 0.1)\ntime.sleep(60)\n")
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, CALIPER_SESSION="0", CALIPER_NETIO="0",
               CALIPER_RUN_ID="early", CALIPER_RUN_DIR=str(run_dir),
               CALIPER_RUN_LINK_FILE=str(run_dir / "link"), CALIPER_RUN_APP_PID=str(gone.pid))
    for k in ("CALIPER_RUN_PARAMS", "CALIPER_RUN_RESUME", "CALIPER_RUN_PARENT"):
        env.pop(k, None)
    proc = subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    try:
        out, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()                       # by its own pid, so that no process is left
        out, _ = proc.communicate()
    text = out.decode(errors="replace")
    assert proc.returncode == M.PARENT_GONE_EXIT, text
    assert f"the process {gone.pid} that started this managed run ended" in text


@pytest.mark.skipif(os.name != "posix", reason="the job object of the runner covers Windows")
def test_import_caliper_alone_ends_a_run_whose_app_ended(tmp_path):
    """A script that imports caliper and calls neither watch() nor param() still ends when the app
    named by CALIPER_RUN_APP_PID ended, with the reason in its output."""
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=60)                 # an app that ended: its id names no process now
    run_dir = tmp_path / "runs" / "bare"
    run_dir.mkdir(parents=True)
    script = tmp_path / "script.py"
    script.write_text("import time\nimport caliper\ntime.sleep(60)\n")
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, CALIPER_SESSION="0", CALIPER_NETIO="0",
               CALIPER_RUN_ID="bare", CALIPER_RUN_DIR=str(run_dir),
               CALIPER_RUN_LINK_FILE=str(run_dir / "link"), CALIPER_RUN_APP_PID=str(gone.pid))
    for k in ("CALIPER_RUN_PARAMS", "CALIPER_RUN_RESUME", "CALIPER_RUN_PARENT"):
        env.pop(k, None)
    proc = subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    try:
        out, _ = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()                       # by its own pid, so that no process is left
        out, _ = proc.communicate()
    text = out.decode(errors="replace")
    assert proc.returncode == M.PARENT_GONE_EXIT, text
    assert f"the process {gone.pid} that started this managed run ended" in text


IMPORT_PROBE = """
import json, sys, threading
import caliper
print(json.dumps({"managed": "caliper._managed" in sys.modules, "torch": "torch" in sys.modules,
                  "watch": any(t.name == "caliper-parent-watch" for t in threading.enumerate())}))
"""


def test_import_caliper_watches_the_app_only_when_the_app_is_named(tmp_path):
    """Without CALIPER_RUN_APP_PID, import caliper imports neither caliper._managed nor torch and
    starts no thread; with it, the import starts the watch of the app and still imports no torch."""
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, CALIPER_SESSION="0", CALIPER_NETIO="0")
    for k in M._VARS + (M.ENV_APP_PID,):
        env.pop(k, None)

    def probe(env):
        done = subprocess.run([sys.executable, "-c", IMPORT_PROBE], env=env, cwd=str(tmp_path),
                              capture_output=True, timeout=60)
        assert done.returncode == 0, done.stderr.decode(errors="replace")
        return json.loads(done.stdout.decode().strip().splitlines()[-1])

    assert probe(dict(env, CALIPER_RUN_ID="plain", CALIPER_RUN_DIR=str(tmp_path))) == \
        {"managed": False, "torch": False, "watch": False}
    assert probe(env) == {"managed": False, "torch": False, "watch": False}
    # This process is the app: it runs while the probe looks.
    named = probe(dict(env, CALIPER_RUN_APP_PID=str(os.getpid())))
    assert named == {"managed": True, "torch": False, "watch": os.name == "posix"}


def test_write_never_writes_through_a_link(tmp_path, monkeypatch):
    """The temporary file of _write is new (O_EXCL | O_NOFOLLOW): a link at its name is refused and
    the file that the link names is not changed; a normal write gives mode 0600."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    trap = tmp_path / ".link.trap.tmp"
    trap.symlink_to(victim)
    monkeypatch.setattr(M, "_temp_name", lambda folder, base: str(trap))
    with pytest.raises(FileExistsError):
        M._write(str(tmp_path / "link"), b"http://127.0.0.1:1/")
    assert victim.read_text() == "keep"
    assert not (tmp_path / "link").exists()
    monkeypatch.undo()
    M._write(str(tmp_path / "link"), b"http://127.0.0.1:1/")
    assert (tmp_path / "link").read_bytes() == b"http://127.0.0.1:1/"
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "link").stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == [trap.name]
