"""Process limits for tests whose threads can retain fixture locks."""
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, Signature, signature
import math
import os
from pathlib import Path
import subprocess
import sys


CHILD_ENV = "CALIPER_CONCURRENCY_TEST_CHILD"
TEST_TIMEOUT = 45
STOP_TIMEOUT = 5


@dataclass(frozen=True)
class ChildResult:
    pid: int
    returncode: int
    timed_out: bool
    output: str


def _stop(child):
    """Stop and reap only the process that this invocation owns."""
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=STOP_TIMEOUT)


def run_isolated_test(nodeid, directory, *, timeout=TEST_TIMEOUT):
    """Run one pytest node with its fixtures and interpreter exit under a limit."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("child timeout must be positive and finite")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["PYTHONUNBUFFERED"] = "1"
    # Keep child scratch beside its log, including on Windows.
    for name in ("TMPDIR", "TEMP", "TMP"):
        env[name] = str(directory)
    command = [sys.executable, str(Path(__file__).resolve()), nodeid,
               str(directory / "fixtures"), str(timeout)]
    log_path = directory / "child.log"
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _stop(child)
        log.write(f"\nParent result: child PID {child.pid} reaped, "
                  f"exit={child.returncode}, timeout={timed_out}\n")
    return ChildResult(child.pid, child.returncode, timed_out,
                       log_path.read_text(encoding="utf-8", errors="replace"))


def isolated_test(*child_fixtures):
    """Keep named fixtures inside the child and retain pytest parameter IDs."""
    def decorate(function):
        if os.environ.get(CHILD_ENV) == "1":
            return function

        @wraps(function)
        def run(request, tmp_path, **parameters):
            result = run_isolated_test(request.node.nodeid, tmp_path / "isolated")
            assert not result.timed_out and result.returncode == 0, (
                f"{request.node.nodeid}: child PID {result.pid}, "
                f"exit {result.returncode}, timeout={result.timed_out} "
                f"(limit {TEST_TIMEOUT}s)\n{result.output}"
            )

        # Pytest must not construct a lock-owning fixture in the parent.
        parameters = signature(function).parameters
        assert set(child_fixtures) <= parameters.keys()
        run.__signature__ = Signature([
            Parameter("request", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("tmp_path", Parameter.POSITIONAL_OR_KEYWORD),
            *[parameter for name, parameter in parameters.items()
              if name not in (*child_fixtures, "request", "tmp_path")],
        ])
        return run
    return decorate


class _FailureReports:
    def pytest_runtest_logreport(self, report):
        if report.failed:
            # Publish assertions before a later fixture finalizer can block.
            print(f"\nChild failure ({report.when}): {report.nodeid}", flush=True)
            print(report.longreprtext, flush=True)


if __name__ == "__main__":
    import faulthandler
    import pytest

    nodeid, basetemp, timeout = sys.argv[1:]
    faulthandler.enable()
    # Retain thread stacks if fixture cleanup or interpreter exit stalls.
    faulthandler.dump_traceback_later(float(timeout) / 2)
    raise SystemExit(pytest.main([
        "-q", "-s", "--tb=short", "--noconftest", "-p", "no:cacheprovider",
        # Pytest must not cancel the timer before interpreter shutdown.
        "-p", "no:faulthandler",
        "-o", "addopts=", "--basetemp", basetemp, nodeid,
    ], plugins=[_FailureReports()]))
