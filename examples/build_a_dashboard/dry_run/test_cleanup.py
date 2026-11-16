"""The dry runs of the build-a-dashboard walk-through delete only what they made.

    python examples/build_a_dashboard/dry_run/test_cleanup.py

No app, no model and no network: a small stand-in program takes the app's
place, and it either fails at once or waits until it is stopped. With it,
steps 1, 2 and 6 and run_all.py run their failure path and their interrupt
path (SIGTERM, and SIGINT for the runner), with a --work and an --out that the
caller named and that hold a marker file. Each marker must survive, the
stand-in app must be stopped, and a temporary folder a step made for itself
must be gone. The folder helper of hidden_app.py must refuse every folder it
did not make: one the caller named, one that became a symbolic link, and one
outside the temporary root. It runs with the standard library only, under
ctest as caliper_example_dry_run_cleanup.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import hidden_app  # noqa: E402

FAKE_APP = """#!{python}
import os, sys, time
mode = os.environ.get("DRY_RUN_TEST_APP", "fail")
if mode == "fail":
    print("the stand-in app fails at once", flush=True)
    sys.exit(1)
started = os.environ["DRY_RUN_TEST_STARTED"]
with open(started + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(started + ".tmp", started)
time.sleep(120)
"""


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A process that ended and waits for its parent to collect it counts as ended.
    r = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")


class Folders(unittest.TestCase):
    """The helper removes only the temporary folders this process made, inside the temporary root."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="dry-run-test-")).resolve()
        self.root = self.base / "tmproot"
        self.root.mkdir()
        self.saved = tempfile.tempdir
        tempfile.tempdir = str(self.root)
        self.caller = self.base / "caller"
        self.caller.mkdir()
        (self.caller / "marker").write_text("keep")

    def tearDown(self):
        tempfile.tempdir = self.saved
        hidden_app._TEMPORARY.clear()
        shutil.rmtree(self.base)

    def test_removes_its_own_temporary_folder(self):
        made = hidden_app.make_temporary("t-")
        (made / "file").write_text("x")
        self.assertEqual(hidden_app.remove_temporary(made), "")
        self.assertFalse(made.exists())

    def test_refuses_a_folder_it_did_not_make(self):
        self.assertIn("did not make", hidden_app.remove_temporary(self.caller))
        self.assertTrue((self.caller / "marker").is_file())

    def test_refuses_a_folder_outside_the_temporary_root(self):
        hidden_app._TEMPORARY[self.caller] = self.caller.resolve()     # as if it had been recorded
        self.assertIn("not inside the temporary root", hidden_app.remove_temporary(self.caller))
        self.assertTrue((self.caller / "marker").is_file())

    def test_refuses_a_temporary_folder_that_became_a_link(self):
        made = hidden_app.make_temporary("t-")
        made.rmdir()
        made.symlink_to(self.caller, target_is_directory=True)
        self.assertIn("symbolic link", hidden_app.remove_temporary(made))
        self.assertTrue((self.caller / "marker").is_file())

    def test_a_fresh_folder_leaves_an_earlier_one_alone(self):
        (self.caller / "step2").mkdir()
        (self.caller / "step2" / "stale").write_text("from before")
        fresh = hidden_app.fresh_folder(self.caller, "step2")
        self.assertEqual(fresh.name, "step2-2")
        self.assertEqual(list(fresh.iterdir()), [])
        self.assertTrue((self.caller / "step2" / "stale").is_file())


class Scripts(unittest.TestCase):
    """Each script's failure and interrupt paths keep the caller's folders and stop the app."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="dry-run-test-")).resolve()
        self.root = self.base / "tmproot"          # the scripts' temporary root, to see what they leave there
        self.root.mkdir()
        self.app = self.base / "fake-caliper"
        self.app.write_text(FAKE_APP.format(python=sys.executable))
        self.app.chmod(0o755)
        self.work, self.out = self.base / "work", self.base / "out"
        # Markers at the top of the caller's folders, and in the subfolders a run would write into.
        self.markers = [self.work / "marker", self.out / "marker", self.out / "640x400" / "marker"]
        self.markers += [self.work / f"step{n}" / "marker" for n in (1, 2, 6)]
        for m in self.markers:
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text("keep")
        self.started = self.base / "started"

    def tearDown(self):
        shutil.rmtree(self.base)

    def env(self, mode):
        env = dict(os.environ, DRY_RUN_TEST_APP=mode, DRY_RUN_TEST_STARTED=str(self.started), TMPDIR=str(self.root),
                   CALIPER_AGENT_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("CALIPER_DEMO_HEAVY", None)
        env.pop("CALIPER_DEMO_MODEL", None)
        return env

    def kept(self):
        for m in self.markers:
            self.assertEqual(m.read_text(), "keep", f"the marker {m.relative_to(self.base)}")

    def run_script(self, argv, mode, sig=None, timeout=120):
        """Runs a script with the stand-in app; with sig, sends it once the app has started."""
        p = subprocess.Popen([sys.executable, *argv], env=self.env(mode), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, cwd=str(self.base))
        app_pid = None
        if sig is not None:
            end = time.time() + 60
            while time.time() < end and not self.started.is_file() and p.poll() is None:
                time.sleep(0.1)
            self.assertTrue(self.started.is_file(), "the stand-in app started")
            app_pid = int(self.started.read_text())
            p.send_signal(sig)
        out, _ = p.communicate(timeout=timeout)
        if app_pid is not None:
            end = time.time() + 10
            while time.time() < end and alive(app_pid):
                time.sleep(0.1)
            self.assertFalse(alive(app_pid), f"the stand-in app (pid {app_pid}) was stopped:\n{out}")
        return p.returncode, out

    def step_args(self, name, work=True):
        argv = [str(HERE / name), "--caliper", str(self.app), "--out", str(self.out), "--size", "640x400"]
        if name.startswith("step1"):
            argv += ["--url", "http://127.0.0.1:9/?token=test"]
        return argv + (["--work", str(self.work)] if work else [])

    def check_step(self, name):
        code, out = self.run_script(self.step_args(name), "fail")
        self.assertEqual(code, 1, out)
        self.kept()
        code, out = self.run_script(self.step_args(name), "hang", signal.SIGTERM)
        self.assertEqual(code, 130, out)
        self.kept()
        # Without --work: a temporary folder of its own, removed at the end.
        self.started.unlink()
        code, out = self.run_script(self.step_args(name, work=False), "hang", signal.SIGTERM)
        self.assertEqual(code, 130, out)
        self.kept()
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [], f"left in the temporary root:\n{out}")
        # The caller's --work holds each run's folder side by side; none was cleared.
        self.assertGreaterEqual(len([p for p in self.work.iterdir() if p.is_dir()]), 3)

    def test_step1(self):
        self.check_step("step1_train_and_attach.py")

    def test_step2(self):
        self.check_step("step2_drop_and_build.py")

    @unittest.skipUnless(shutil.which("node"), "step 6 needs node")
    def test_step6(self):
        self.check_step("step6_save_reopen_browser.py")

    def test_runner(self):
        # A non-empty --out is refused, and kept; the runner's --out must be new or empty.
        code, out = self.run_script([str(HERE / "run_all.py"), "--out", str(self.out), "--steps", "2",
                                     "--caliper", str(self.app)], "fail")
        self.assertEqual(code, 2, out)
        self.kept()
        # The failure path and the interrupt path, into a new --out.
        code, out = self.run_script([str(HERE / "run_all.py"), "--out", str(self.base / "run-a"), "--steps", "2",
                                     "--caliper", str(self.app)], "fail")
        self.assertEqual(code, 1, out)
        self.assertIn("2     failed", (self.base / "run-a" / "summary.txt").read_text())
        code, out = self.run_script([str(HERE / "run_all.py"), "--out", str(self.base / "run-b"), "--steps", "2",
                                     "--caliper", str(self.app)], "hang", signal.SIGINT)
        self.assertEqual(code, 130, out)
        self.assertIn("2     interrupted", (self.base / "run-b" / "summary.txt").read_text())
        self.kept()


if __name__ == "__main__":
    unittest.main(verbosity=2)
