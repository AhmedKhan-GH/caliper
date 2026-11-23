"""Hidden app captures compare scene resources with the saved-recording converter.
The test-only legacy encoder supplies the recording form on the second attachment.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pytest
import torch

import caliper
from caliper import _scene, _net_scene, _wire

EXE = os.environ.get("CALIPER_TEST_CALIPER_EXE", "")
pytestmark = pytest.mark.skipif(sys.platform != "darwin" or not EXE or not os.path.exists(EXE),
                                reason="needs the Caliper app (CALIPER_TEST_CALIPER_EXE) on macOS")
pytest.importorskip("websockets")
from test_scene_app import App, Subscriber, Where, keep, read_png  # noqa: E402

SCENE = "landscape"          # the scene that test_scene_app's Subscriber subscribes to


def log_scene(scene) -> int:
    """Logs the scene; returns its points and image batches."""
    g = torch.Generator().manual_seed(7)
    cloud = torch.randn(400, 3, generator=g) * 0.5
    assert scene.points("cloud", cloud, values=cloud[:, 2], cmap="magma")
    ring = torch.stack([torch.cos(torch.linspace(0, 6.2, 24)), torch.sin(torch.linspace(0, 6.2, 24)),
                        torch.zeros(24)], -1) * 1.5
    assert scene.points("ring", ring, colors=torch.rand(24, 3, generator=g), radii=torch.full((24,), 0.06))
    scene.classes({0: "floor", 1: "wall", 2: "door"})
    blocks = torch.rand(60, 3, generator=g) * 2 - 1 + torch.tensor([0.0, 0.0, 1.2])
    assert scene.points("blocks", blocks, class_ids=torch.randint(0, 3, (60,), generator=g))
    turn = torch.eye(4)
    turn[:3, 3] = torch.tensor([0.0, -3.0, 0.4])
    turn[:3, :3] = torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])     # looks along +y, y down
    scene.transform("rig", matrix=turn)
    scene.camera("rig", width=64, height=48, focal=60.0)
    img = torch.rand(48, 64, 3, generator=g)
    img[:, :32] = torch.tensor([0.9, 0.3, 0.1])
    assert scene.image("rig/rgb", img, opacity=0.8)
    assert scene.image("rig/grey", (torch.rand(12, 16, generator=g) * 255).to(torch.uint8))
    return 5


def attach(url, tmp, name, log):
    app = App(url, os.path.join(tmp, name), f"where {SCENE}; wait 60; where {SCENE}; wait 400")
    renderer = app.wait_for(lambda l: l.startswith("[renderer] "), 60)
    if renderer is None or "metal" not in renderer:
        app.close()
        pytest.skip(f"the app has no Metal renderer here ({renderer})")
    lines = []
    end = time.monotonic() + 60
    while time.monotonic() < end and len(lines) < 2:
        lines = [l for l in app.lines if l.startswith(f"[ui] where {SCENE} ")]
        time.sleep(0.05)
    assert len(lines) == 2, "\n".join(app.lines[-40:])
    return app, Where(lines[1])


def area_of(png, where):
    img = read_png(png)
    k = img.shape[1] / 1280
    ax, ay, aw, ah = (int(round(v * k)) for v in where.area)
    return img[ay:ay + ah, ax:ax + aw]


def stable(app, where, timeout=30.0):
    """The scene view's area once two captures in a row agree (the frames have every batch)."""
    end, last = time.monotonic() + timeout, None
    while time.monotonic() < end:
        png = app.capture()
        a = area_of(png, where)
        if last is not None and np.array_equal(a, last[1]):
            return png, a
        last = (png, a)
    raise AssertionError("the scene view did not settle")


def test_the_piece_form_draws_as_the_per_shape_parts(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "0")
    s = caliper.serve(port=0)
    tmp = tempfile.mkdtemp(prefix="caliper-scene-pieces-")
    sub = None
    try:
        scene = caliper.scene(SCENE, up="+z", units="")
        sub = Subscriber(s.stream_url)
        sub.start()
        assert sub.ready.wait(10), "the websockets viewer did not subscribe"
        log_scene(scene)
        end = time.monotonic() + 10
        while time.monotonic() < end and {"cloud", "ring", "blocks", "rig/rgb", "rig/grey"} - {p for p, _ in sub.updates}:
            time.sleep(0.05)

        frames = {}
        for name in ("form", "parts"):
            if name == "parts":
                import legacy_scene_encoder
                legacy_scene_encoder.install()
                def recorded_frames(self, batch, state):
                    return [_wire.SceneUpdate(self.sub, self.view, batch.seq, batch.stamp_ns,
                                              batch.path, batch.part, batch.arrays, batch.params,
                                              static=batch.static, state=state, axes=batch.axes)]
                monkeypatch.setattr(_net_scene.SubQueue, "frames", recorded_frames)
            log = os.path.join(tmp, f"{name}.log")
            app, where = attach(s.stream_url, tmp, name, log)
            try:
                png, area = stable(app, where)
                keep(f"scene-pieces-{name}.png", png)
                frames[name] = (area, where)
            finally:
                code = app.close()
            assert code == 0, "\n".join(app.lines[-40:])
        (a, wa), (b, wb) = frames["form"], frames["parts"]
        assert (wa.target_px, wa.distance, wa.azimuth, wa.elevation) == (wb.target_px, wb.distance, wb.azimuth,
                                                                         wb.elevation)
        assert a.shape == b.shape and len({tuple(c) for c in a[::5, ::5].reshape(-1, 3)}) > 20, "the scene is blank"
        differ = int(np.any(a != b, axis=-1).sum())
        print(f"[scene-pieces-app] {differ} of {a.shape[0] * a.shape[1]} pixels of the scene view differ")
        assert differ == 0
    finally:
        if sub:
            sub.stop.set()
            sub.join(5)
        s.close()
        for name in list(_scene.SCENES):
            caliper.remove(name)
        shutil.rmtree(tmp, ignore_errors=True)
