"""Picks, hovers and selections in the Caliper app's scene view reach the
script through caliper.serve() (design/scenes.md, "Picking, hover and
selection"), end to end, with nothing on screen.

A script serves a 3-D scene of three points on the x axis, logged static so
that the app receives them when it attaches late with ``caliper --hidden
--url``. The fitted orbit puts the middle point, which sits at the bounds'
centre, at the picture's centre. The app's UI driver (``CALIPER_TEST_UI``)
moves the pointer to the middle of the scene view and rests there, clicks
there, and drags a shift+alt rectangle around it. The script's
``server.events()`` then shows a hover and a pick of the middle point (its
entity, primitive, index, row id and a world position on its disc) and a
selection of that point alone.

Caliper LIB refuses ``items=`` on points, so the linked selections of a scene
are covered by the native test in tests/test_live_net.cpp only.

It needs the app (``CALIPER_TEST_CALIPER_EXE``) and Metal; it skips without
either."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

import caliper
from caliper import _scene

EXE = os.environ.get("CALIPER_TEST_CALIPER_EXE", "")
pytestmark = pytest.mark.skipif(sys.platform != "darwin" or not EXE or not os.path.exists(EXE),
                                reason="needs the Caliper app (CALIPER_TEST_CALIPER_EXE) on macOS")

RADIUS = 0.12
UI = ("hover world 0.5 0.5; wait 10; pick world 0.5 0.5; wait 10; "
      "rect world 0.45 0.45 0.55 0.55; wait 10; wait 400")


class App:
    """The app, hidden, attached to the script's stream, playing the UI driver's steps."""

    def __init__(self, url, data, ui):
        env = dict(os.environ, CALIPER_DATA_DIR=os.path.join(data, "data"),
                   CALIPER_USER_INSTRUMENTS_DIR=os.path.join(data, "user"), CALIPER_TEST_UI=ui)
        os.makedirs(env["CALIPER_DATA_DIR"])
        os.makedirs(env["CALIPER_USER_INSTRUMENTS_DIR"])
        self.p = subprocess.Popen([EXE, "--hidden", "--url", url], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.lines: list[str] = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.p.stderr:
            self.lines.append(line.rstrip("\n"))

    def wait_for(self, pred, timeout=60.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for line in list(self.lines):
                if pred(line):
                    return line
            if self.p.poll() is not None:
                break
            time.sleep(0.05)
        return None

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(20)
            except subprocess.TimeoutExpired:
                self.p.kill()
        return self.p.returncode


def test_picks_hovers_and_selections_in_the_apps_scene_view_reach_the_script(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "0")
    s = caliper.serve(port=0)
    tmp = tempfile.mkdtemp(prefix="caliper-scene-events-")
    app = None
    try:
        scene = caliper.scene("world", up="+z", units="m")
        positions = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
        assert scene.points("cloud", positions, radii=RADIUS, colors=(255, 80, 80), static=True)
        s.events()

        app = App(s.stream_url, tmp, UI)
        renderer = app.wait_for(lambda l: l.startswith("[renderer] "), 60)
        if renderer is None or "metal" not in renderer:
            pytest.skip(f"the app has no Metal renderer here ({renderer})")

        # The first event of each type from the scene view; the driver's log says where it is.
        got: dict[str, dict] = {}
        end = time.monotonic() + 60
        while time.monotonic() < end and not {"hover", "pick", "select"} <= got.keys():
            for ev in s.events():
                if ev["view"] == "world":
                    got.setdefault(ev["type"], ev)
            if app.p.poll() is not None:
                break
            time.sleep(0.05)
        log = "\n".join(l for l in app.lines if l.startswith("[ui]") or "error" in l.lower())
        assert {"hover", "pick", "select"} <= got.keys(), f"events {sorted(got)}; the app said:\n{log}"

        # The hover and the pick: the middle point, index 1, its row id the index (no payload axes),
        # and a position on its disc around the origin.
        for kind in ("hover", "pick"):
            d = got[kind]["detail"]
            assert d["entity"] == "cloud", (kind, d)
            assert d["primitive"] == "points", (kind, d)
            assert d["index"] == 1, (kind, d)
            assert d["row_id"] == 1, (kind, d)
            assert "position" in d, (kind, d)
            assert math.dist(d["position"], (0.0, 0.0, 0.0)) < RADIUS, (kind, d)

        # The rectangle around the picture's centre selects the middle point alone.
        d = got["select"]["detail"]
        assert len(d["entities"]) == 1, d
        e = d["entities"][0]
        assert e["entity"] == "cloud" and e["primitive"] == "points", d
        assert e["indices"] == [1], d
        assert e["row_ids"] == [1], d
        assert "space" not in d, d
    finally:
        if app:
            code = app.close()
            assert code == 0, "\n".join(app.lines[-40:])
        s.close()
        for name in list(_scene.SCENES):
            caliper.remove(name)
        shutil.rmtree(tmp, ignore_errors=True)
