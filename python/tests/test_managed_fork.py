"""The parts of a managed run that make a fork: the kept checkpoints, the interventions of
CALIPER_RUN_INTERVENTIONS, and the resume from a kept checkpoint. The kept checkpoints run with a
fake clock, the parser of the interventions runs on its own, and the scripts run as subprocesses
the way the model runner of the Caliper app starts them (output into a pipe).

The tests wait only on the exit of a process, with a bounded time."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import types

import pytest

pytest.importorskip("torch")

import caliper  # noqa: E402
from caliper import _managed as M  # noqa: E402

PYTHONPATH = str(pathlib.Path(caliper.__file__).parent.parent)


class Clock:
    """A fake clock: a list that holds the time, and a callable that reads it."""

    def __init__(self, t=0.0):
        self.t = [t]

    def __call__(self):
        return self.t[0]

    def move(self, dt):
        self.t[0] += dt


def _folder(tmp_path):
    return str(tmp_path / "kept")


def test_the_first_step_is_kept_and_a_step_inside_the_stride_is_not(tmp_path):
    """The first step is always kept as checkpoint-<step>.pt, and a step less than stride past the
    newest kept one is not kept."""
    ks = M.KeptCheckpoints(_folder(tmp_path), clock=Clock(), share=1e9)
    first = ks.keep(5, lambda: b"abc")
    assert first == os.path.join(_folder(tmp_path), "checkpoint-5.pt")
    assert open(first, "rb").read() == b"abc"
    ks.stride = 4
    called = []
    assert ks.keep(8, lambda: called.append(1) or b"abc") is None
    assert called == [] and [s for s, _ in ks.kept] == [5]
    assert ks.keep(9, lambda: b"abc") is not None
    assert [s for s, _ in ks.kept] == [5, 9]


def test_a_step_waits_while_the_kept_checkpoints_take_too_much_time(tmp_path):
    """A due step is not kept while the time spent on keeping is above the share of the clock, and
    it is kept once the clock has moved on enough."""
    clock = Clock()

    def slow():
        clock.move(1.0)                       # one checkpoint takes one second
        return b"abc"

    ks = M.KeptCheckpoints(_folder(tmp_path), clock=clock, share=0.1)
    assert ks.keep(1, slow) is not None
    assert ks.keep(2, slow) is None           # 1 s spent, and 0.1 of 1 s is 0.1 s
    assert not ks.due(2)
    clock.move(20.0)                          # 0.1 of 21 s is 2.1 s
    assert ks.due(2)
    assert ks.keep(2, slow) is not None
    assert [s for s, _ in ks.kept] == [1, 2]


def test_the_fifth_keep_thins_the_files_and_doubles_the_stride(tmp_path):
    """With limit 4, the fifth keep removes the second and fourth files, keeps the oldest and the
    newest, and doubles the stride."""
    ks = M.KeptCheckpoints(_folder(tmp_path), clock=Clock(), share=1e9, limit=4)
    for step in range(1, 5):
        assert ks.keep(step, lambda: b"abc") is not None
    assert ks.stride == 1 and len(ks.kept) == 4
    ks.keep(5, lambda: b"abc")
    assert [s for s, _ in ks.kept] == [1, 3, 5]
    assert ks.stride == 2
    names = sorted(os.listdir(_folder(tmp_path)))
    assert names == ["checkpoint-1.pt", "checkpoint-3.pt", "checkpoint-5.pt"]


def test_a_thousand_steps_stay_within_the_limit_and_span_the_run(tmp_path):
    """Over 1,000 steps with limit 8, the kept steps start at the first step, end near the last,
    and are at most 9 entries at any time."""
    ks = M.KeptCheckpoints(_folder(tmp_path), clock=Clock(), share=1e9, limit=8)
    most = 0
    for step in range(1, 1001):
        ks.keep(step, lambda: b"abc")
        most = max(most, len(ks.kept))
    steps = [s for s, _ in ks.kept]
    assert most <= 9 and len(steps) <= 9
    assert steps[0] == 1
    assert steps[-1] > 1000 - ks.stride
    assert steps == sorted(steps)
    assert len(os.listdir(_folder(tmp_path))) == len(steps)


def test_a_checkpoint_too_large_for_two_turns_the_keeping_off(tmp_path, capsys):
    """When two checkpoints pass bytes_max, the run keeps its first only, says so in one line, and
    writes nothing later."""
    ks = M.KeptCheckpoints(_folder(tmp_path), clock=Clock(), share=1e9, bytes_max=150)
    assert ks.keep(1, lambda: b"x" * 100) is not None
    assert ks.off is True
    out = capsys.readouterr().out
    assert any(line.startswith("caliper: the run keeps no checkpoint after step")
               for line in out.splitlines()), out
    assert ks.keep(2, lambda: b"x" * 100) is None
    assert ks.keep(3, lambda: b"x" * 100) is None
    assert os.listdir(_folder(tmp_path)) == ["checkpoint-1.pt"]


def _linked_run(tmp_path):
    """A managed run whose kept folder is a symlink, with a stand-in package."""
    real = tmp_path / "elsewhere"
    real.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / M.KEEP_DIR).symlink_to(real, target_is_directory=True)
    run = M.ManagedRun("linked", str(run_dir))
    ctl = types.SimpleNamespace(n=3, lock=threading.Lock(), params={})
    run._pkg = types.SimpleNamespace(_CTL=ctl)
    return run, real


@pytest.mark.skipif(os.name != "posix", reason="symbolic links need POSIX here")
def test_a_kept_folder_that_is_a_link_raises(tmp_path):
    """The keep of a folder that is a symbolic link raises OSError and writes nothing."""
    _, real = _linked_run(tmp_path)
    ks = M.KeptCheckpoints(str(tmp_path / "run" / M.KEEP_DIR), clock=Clock())
    with pytest.raises(OSError):
        ks.keep(1, lambda: b"abc")
    assert list(real.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="symbolic links need POSIX here")
def test_the_run_turns_a_keep_error_into_a_warning(tmp_path, monkeypatch):
    """ManagedRun._keep sets kept.off and warns with a RuntimeWarning when the keep raises."""
    run, real = _linked_run(tmp_path)
    monkeypatch.setattr(M.TORCH, "state", lambda *a, **k: b"abc")
    with pytest.warns(RuntimeWarning, match="keeps no more checkpoints"):
        run._keep(object(), object())
    assert run.kept.off is True
    assert list(real.iterdir()) == []


def test_no_text_and_none_give_no_interventions():
    """An empty text and None give an empty list."""
    assert M._interventions("") == []
    assert M._interventions(None) == []


def test_text_that_is_not_a_list_of_objects_raises():
    """Text that is not JSON, a JSON object, or 17 objects raises ValueError."""
    with pytest.raises(ValueError):
        M._interventions("not json")
    with pytest.raises(ValueError):
        M._interventions('{"point": "a", "op": "ablate"}')
    with pytest.raises(ValueError):
        M._interventions(json.dumps([{"point": "a", "op": "ablate"}] * 17))
    with pytest.raises(ValueError):
        M._interventions("[1, 2]")


def test_a_list_of_objects_is_returned_as_it_is(tmp_path):
    """A list of two objects comes back unchanged, also through ManagedRun.from_env."""
    ivs = [{"point": "a", "op": "ablate"}, {"point": "b#0", "op": "ablate"}]
    assert M._interventions(json.dumps(ivs)) == ivs
    run = M.ManagedRun.from_env({"CALIPER_RUN_DIR": str(tmp_path),
                                 "CALIPER_RUN_INTERVENTIONS": json.dumps(ivs)})
    assert run.interventions == ivs


SCRIPT = """
import json, sys
import torch
import caliper

torch.manual_seed(0)
torch.set_num_threads(1)


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 2
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(x)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.attn(x))


model = Net()
opt = torch.optim.SGD(model.parameters(), lr=0.05)
caliper.watch(model, opt)
seen = []
# A forward hook sees the input after every pre-hook, also the ablation, which the first forward
# pass registers on proj after this line.
model.attn.proj.register_forward_hook(
    lambda mod, args, out: seen.append(args[0].detach().clone()) if not seen else None)
x = torch.randn(4, 8)
y = torch.randn(4, 1)
for i in range(3):
    loss = (model(x) - y).square().mean()
    if i == 0:
        first = seen[0]
        print(json.dumps({"zero_high": bool((first[:, 4:] == 0).all()),
                          "low_nonzero": bool((first[:, :4] != 0).any())}), flush=True)
    opt.zero_grad()
    loss.backward()
    opt.step()
print("done", flush=True)
"""


def _start(tmp_path, name, interventions=None, resume=None, parent=None):
    """Run SCRIPT as a managed run to its end; returns (run dir, exit code, output)."""
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    src = tmp_path / f"{name}.py"
    src.write_text(SCRIPT)
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, CALIPER_SESSION="0", CALIPER_NETIO="0",
               CALIPER_RUN_ID=name, CALIPER_RUN_DIR=str(run_dir),
               CALIPER_RUN_LINK_FILE=str(run_dir / "link"))
    for k in ("CALIPER_RUN_PARAMS", "CALIPER_RUN_RESUME", "CALIPER_RUN_PARENT",
              "CALIPER_RUN_INTERVENTIONS", "CALIPER_RUN_APP_PID"):
        env.pop(k, None)
    if interventions is not None:
        env["CALIPER_RUN_INTERVENTIONS"] = interventions
    if resume is not None:
        env["CALIPER_RUN_RESUME"] = str(resume)
    if parent is not None:
        env["CALIPER_RUN_PARENT"] = parent
    proc = subprocess.Popen([sys.executable, str(src)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    try:
        out, _ = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()                           # by its own pid
        out, _ = proc.communicate()
        pytest.fail("the script did not end in 90 s:\n" + out.decode(errors="replace"))
    return run_dir, proc.returncode, out.decode(errors="replace")


def test_a_fork_ablates_a_head_before_its_first_step(tmp_path):
    """With an ablate intervention on a head, the first forward pass already sees that head's
    slice of the projection input as zeros, and the run applies it and ends with exit code 0."""
    _, code, out = _start(tmp_path, "ablated", '[{"point": "attn#1", "op": "ablate"}]')
    assert code == 0, out
    assert '{"zero_high": true, "low_nonzero": true}' in out, out
    assert "applies ablate attn#1 from its first step" in out, out
    assert "done" in out


def test_a_fork_with_an_unknown_point_fails(tmp_path):
    """An intervention on a head that does not exist ends the script with a non-zero code and says
    that the interventions cannot apply."""
    _, code, out = _start(tmp_path, "refused", '[{"point": "attn#7", "op": "ablate"}]')
    assert code != 0, out
    assert "cannot apply its interventions" in out, out
    assert "done" not in out


def test_a_managed_run_keeps_its_first_checkpoint_and_a_fork_resumes_from_it(tmp_path):
    """A run with an optimizer writes kept/checkpoint-1.pt at its first step, and a second run
    with CALIPER_RUN_RESUME set to that file says that it resumes at step 1."""
    run_dir, code, out = _start(tmp_path, "parent")
    assert code == 0, out
    kept = run_dir / M.KEEP_DIR / "checkpoint-1.pt"
    assert kept.is_file(), out
    _, code, out = _start(tmp_path, "child", resume=kept, parent="parent")
    assert code == 0, out
    assert "resumes from" in out and "at step 1" in out, out
