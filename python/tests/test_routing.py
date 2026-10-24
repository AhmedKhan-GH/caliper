"""Routing and hot-path hygiene: no host readback calls in the snapshot code,
the session guard, the implicit session on the first call."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import caliper

PKG = Path(caliper.__file__).resolve().parent
FORBIDDEN = {"item", "cpu", "numpy", "tolist", "synchronize"}


@pytest.mark.parametrize("module", ["_convert.py", "_ring.py"])
def test_snapshot_code_never_reads_back(module):
    """The snapshot path (device-side prep + ring) calls no .item(), .cpu(),
    .numpy(), .tolist() or synchronize(): an MPS snapshot never waits for the
    GPU and never copies to the host."""
    tree = ast.parse((PKG / module).read_text())
    bad = [(n.lineno, n.func.attr) for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr in FORBIDDEN]
    assert bad == [], f"{module}: {bad}"


def _session_available() -> bool:
    try:
        from caliper.live import SessionTransport  # noqa: F401
        return True
    except ImportError:
        return False


def test_session_guard_or_session():
    if _session_available():
        pytest.skip("caliper.live.SessionTransport present (covered by test_live_session.py)")
    with pytest.raises(RuntimeError, match="SessionTransport"):
        caliper.session()


def _run(code: str, env_extra=None) -> str:
    env = dict(os.environ)
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_first_call_without_transport_is_a_cheap_no_op_when_sessions_are_off():
    code = ("import torch, caliper\n"
            "r = [caliper.show(torch.ones(4, 4)), caliper.series(1.0), caliper.points(torch.ones(3, 2))]\n"
            "print(r, caliper._native_core() is None, len(caliper._routes))")
    assert _run(code, {"CALIPER_SESSION": "0"}) == "[False, False, False] True 0"


def test_first_call_opens_the_session_or_warns_once():
    code = ("import warnings, torch, caliper\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    a = caliper.show(torch.ones(4, 4)); b = caliper.show(torch.ones(4, 4))\n"
            "print(len(w), caliper._session_route is not None)\n"
            "caliper.close()")
    out = _run(code)
    if _session_available():
        assert out.endswith("True"), out
    else:
        assert out == "1 False", out          # one clear warning, then silent no-ops


def test_window_and_offscreen_need_the_main_thread():
    import threading
    errs = []

    def t():
        for fn in (caliper.window, caliper.offscreen):
            try:
                fn()
            except RuntimeError as e:
                errs.append(str(e))
    th = threading.Thread(target=t)
    th.start()
    th.join()
    assert len(errs) == 2 and all("main thread" in e for e in errs)
