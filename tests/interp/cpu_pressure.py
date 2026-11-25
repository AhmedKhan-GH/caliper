"""Deterministic pressure readers for small CPU numerical fixtures."""
from contextlib import contextmanager
from pathlib import Path
import runpy
import sys

import pytest


@contextmanager
def cpu_fixture_pressure():
    """Keep real allocation checks and explicit pressure injections intact."""
    from caliper.interp import _frozen, _memory

    with pytest.MonkeyPatch.context() as patch:
        # The process guard binds its default reader when Python imports it.
        patch.setitem(_memory.MemoryGuard.__init__.__kwdefaults__, "pressure", lambda: 80.)
        # The trace guard has a separate reader and retains its own sample cache.
        patch.setattr(_frozen, "free_percent", lambda: 80.)
        yield


@pytest.fixture(autouse=True)
def deterministic_cpu_pressure():
    """Apply pressure isolation only in modules that import this fixture."""
    with cpu_fixture_pressure():
        yield


def hosted_fixture_command(*args):
    """Run the hosted module with the same CPU readers in a fresh process."""
    return [sys.executable, str(Path(__file__).resolve()), *args]


if __name__ == "__main__":
    if "--fixture-dir" not in sys.argv and "--model=tiny-lm" not in sys.argv:
        raise ValueError("pressure isolation requires the tiny CPU fixture")
    with cpu_fixture_pressure():
        runpy.run_module("caliper.interp._hosted", run_name="__main__", alter_sys=True)
