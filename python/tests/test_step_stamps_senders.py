"""The step stamps of snapshots end to end, on both senders of caliper.serve(), with a websockets viewer.

See design/instrument-interface.md, section Step History, and test_step_stamps.py for the frames.
"""
from __future__ import annotations

import asyncio
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("websockets")

import caliper  # noqa: E402
from caliper import _wire as w  # noqa: E402
# The fixtures of the two senders and the controls, and the viewer and loop of the controls tests.
from test_net_controls import CAPS, Loop, Viewer, ctl, make_server, mode, run, server  # noqa: E402,F401


def test_a_kept_value_has_the_step_it_was_taken_at(server, ctl):
    """To a viewer that announces "timelines", both senders declare the timelines in VIEWS and
    stamp each SNAPSHOT with the run's step of the view call that took it. Under the shared
    cadence the snapshots come from pass steps and arrive steps later, so the step at arrival
    would be wrong: each heatmap holds the step it was taken at, and its stamp names that step."""
    from caliper import _programs

    def body(i):
        caliper.show(torch.full((4, 4), float(ctl.n)), name="h")

    async def go():
        async with Viewer(server, caps=CAPS + [w.CAP_TIMELINES]) as v:
            assert w.CAP_TIMELINES in v.welcome.capabilities
            with Loop(body=body, delay=0.0005):
                views = await v.expect(w.Views, lambda m: "h" in {x["name"] for x in m.views})
                assert views.timelines[0] == {"name": "step", "kind": "sequence"}
                assert views.timelines[1]["name"] == "time"
                await v.send(w.Subscribe(1, "h", 1))
                kept = []
                for _ in range(6):
                    m = await v.expect(w.Snapshot, lambda m: m.view == "h")
                    step, wall = m.stamp.array().tolist()
                    # A fast machine delivers a snapshot steps after it was taken, and a slow one can
                    # deliver it within its own step. The run goes on in both cases, so wait for it to
                    # pass the step: the kept value must still hold the step it was taken at.
                    deadline = time.monotonic() + 30
                    while ctl.n <= step and time.monotonic() < deadline:
                        await asyncio.sleep(0.002)
                    kept.append((step, ctl.n))
                    assert set(m.data.array().ravel().tolist()) == {float(step)}, (step, m.data.array())
                    assert wall > 0
                assert _programs._VIEWS["ring"] is server.ring    # the passes of the cadence took them
                assert all(now > step for step, now in kept), kept
    run(go())
