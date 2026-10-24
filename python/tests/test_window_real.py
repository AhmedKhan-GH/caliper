"""A real GLFW/AppKit window (not activated, so the terminal keeps focus):
caliper.run() trains on a worker thread while the main thread runs the event
loop; the close button stops the views without stopping training.
Set CALIPER_TEST_NO_WINDOW=1 to skip (no window server)."""
import os
import time

import pytest
import torch

import caliper
from conftest import requires_mps

pytestmark = pytest.mark.skipif(os.environ.get("CALIPER_TEST_NO_WINDOW") == "1",
                                reason="CALIPER_TEST_NO_WINDOW=1")


@pytest.fixture
def real_window():
    w = caliper.window(320, 200, title="caliper-test", activate=False)
    yield w
    caliper.close()


def test_run_trains_on_a_worker_and_returns(real_window):
    w = real_window
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.rand(128, 128, device=dev)

    def train(n):
        import threading
        assert threading.current_thread() is not threading.main_thread()
        took = 0
        for i in range(n):
            x.add_(0.01)
            took += bool(caliper.show(x, "rw", rate=0))
            caliper.series(float(i), "rw_loss")
            time.sleep(0.002)
        return took

    t0 = time.perf_counter()
    took = caliper.run(train, 150)
    dt = time.perf_counter() - t0
    assert took > 10
    assert dt < 150 * 0.002 * 4 + 1.0                  # run() returns promptly after train
    caliper.flush()
    time.sleep(0.2)
    st = w.stats()
    assert st["frames"] > 5 and st["drawn"] > 0 and st["errors"] == 0, st["last_error"]
    assert caliper.view_info("rw")["updates"] >= 1


def test_close_button_stops_views_not_training(real_window):
    w = real_window
    shown = []

    def train():
        for i in range(200):
            if i == 20:
                w.core.post_probe(2)                   # -[NSWindow performClose:], main queue
            shown.append(caliper.show(torch.rand(16, 16), "rc", rate=0))
            time.sleep(0.003)
        return len(shown)

    assert caliper.run(train) == 200                   # training ran to completion
    assert w.closed and caliper._native_core() is None
    assert shown[0] is True and shown[-1] is False     # views stopped after the close
