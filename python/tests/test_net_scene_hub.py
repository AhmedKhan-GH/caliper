"""caliper.scene through a real hub: `caliper serve` of the Caliper app
(``CALIPER_TEST_CALIPER_EXE``), Caliper LIB connected with caliper.connect()
(the native sender), and websockets viewers at the hub.

* every part, the class table, entity transforms and a camera reach a viewer
  through the hub, and a viewer that subscribes later gets the hub's own state
  in the draft's order;
* a clear reaches the viewer and leaves the hub's state without what it
  cleared;
* a batch over one frame travels through the hub as a split update that the
  codec puts back together;
* a viewer without the capability "scenes" (a 3.1 viewer) sees no scene and
  gets unknown_view for one;
* points travel through the hub in the piece form, the RESOURCE of their scene
  program before the update, to a viewer with "scene_pieces", and a viewer
  without it gets the scene's structure alone.

It skips without the app or the native sender."""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import pytest
import torch

import caliper
from caliper import _net_scene, _scene_pieces, _wire as w

from test_net_scene import NATIVE, Viewer, run
from test_net_scene import _forget_scenes  # noqa: F401  (fixture)

EXE = os.environ.get("CALIPER_TEST_CALIPER_EXE", "")
pytestmark = [
    pytest.mark.skipif(not EXE or not os.path.exists(EXE),
                       reason="needs the Caliper app for `caliper serve` (CALIPER_TEST_CALIPER_EXE)"),
    pytest.mark.skipif(not NATIVE, reason="caliper.connect() needs caliper._netio"),
]


class _Url:
    def __init__(self, url):
        self.stream_url = url


@pytest.fixture
def hub(tmp_path, monkeypatch):
    hub_file = tmp_path / "hub.json"
    env = dict(os.environ, CALIPER_HUB_FILE=str(hub_file), CALIPER_AGENT_OFFLINE="1")
    proc = subprocess.Popen([EXE, "serve", "--port", "0"], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    info = None
    deadline = time.monotonic() + 30
    while info is None:                       # the hub writes its file once it listens
        assert proc.poll() is None, "caliper serve exited"
        assert time.monotonic() < deadline, "caliper serve wrote no hub file"
        try:
            info = json.loads(hub_file.read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(0.02)
    monkeypatch.setenv("CALIPER_HUB_FILE", str(hub_file))
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1")
    url = f"ws://127.0.0.1:{info['port']}{w.PATH}?{w.TOKEN_PARAM}={info['token']}"
    yield _Url(url)
    caliper.close()
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(10)


def _kind(m) -> str:
    if isinstance(m, w.SceneUpdate):
        return f"{m.path}:{m.part}"
    if isinstance(m, w.SceneTransforms):
        return f"transforms {'static' if m.static else 'stamped'} {','.join(m.paths)}"
    if isinstance(m, w.SceneClasses):
        return "classes"
    return "reset" if m.reset else f"clear {m.path}"


SCENE_MSGS = (w.SceneUpdate, w.SceneTransforms, w.SceneClasses, w.SceneClear)
PIECES = (w.CAP_SCENES, w.CAP_SCENE_PIECES)


def test_scene_parts_through_the_hub(hub, monkeypatch):
    caliper.connect()
    sc = caliper.scene("world", up="+z", view_camera={"azimuth": 30.0})
    sc.classes({0: "floor", 1: ("obstacle", (230, 80, 60))})
    sc.transform("rig", translation=(0.0, -2.5, 1.2), static=True)
    sc.camera("rig/cam", width=8, height=6, focal=5.0, static=True)
    sc.points("room/cloud", torch.rand(6, 3), class_ids=torch.tensor([0, 1, 0, 1, 0, 1]),
              static=True)

    async def go():
        async with Viewer(hub, caps=PIECES) as v:
            assert v.welcome.role == "hub" and w.CAP_SCENES in v.welcome.capabilities
            assert w.CAP_SCENE_PIECES in v.welcome.capabilities
            r = await v.subscribe(1, "world")
            assert r.error is None
            assert v.views["world"]["scene"]["view_camera"] == {"azimuth": 30.0}
            state = [await v.expect(object, lambda m: isinstance(m, SCENE_MSGS))
                     for _ in range(5)]
            assert [_kind(m) for m in state] == ["reset", "classes", "transforms static rig",
                                                 "rig/cam:camera", "room/cloud:points"]
            assert all(m.state for m in state)
            cloud = next(m for m in state if isinstance(m, w.SceneUpdate) and m.part == "points")
            assert cloud.program is not None and cloud.names == ["positions", "class_ids"]
            res = await v.expect(w.Resource, lambda m: m.program == cloud.program)
            back = _scene_pieces.unpack_points(bytes(memoryview(res.data).cast("B")), cloud.names, 3)
            assert back["class_ids"].tolist() == [0, 1, 0, 1, 0, 1]
            # live messages: index-pair edges, an entity transform, a stamped batch and its clear
            pos = torch.rand(4, 3)
            sc.lines("graph", pos, indices=torch.tensor([[0, 1], [2, 3]]))
            g = await v.update("graph", "lines")
            assert g.params["mode"] == "edges" and g.arrays["indices"].array().tolist() == \
                [[0, 1], [2, 3]]
            sc.transform("a", scale=2.0)
            t = await v.expect(w.SceneTransforms, lambda m: m.paths == ["a"])
            assert t.matrices.array().reshape(4, 4)[0, 0] == 2.0
            sc.boxes("dyn", centers=[(0.0, 0.0, 0.0)], half_sizes=[(1.0, 1.0, 1.0)])
            await v.update("dyn", "boxes")
            sc.clear("dyn")
            c = await v.expect(w.SceneClear, lambda m: m.path == "dyn")
            assert not c.reset
            # a split update through the hub
            monkeypatch.setattr(_net_scene, "FRAME_ROOM", 16 * 1024)
            monkeypatch.setattr(_net_scene, "PIECE_BYTES", 24 * 1024)
            big = torch.arange(3 * 3000, dtype=torch.float32).reshape(3000, 3)
            sc.mesh("big", big)
            su = w.SplitUpdates()
            whole = None
            while whole is None:
                whole = su.add(await v.update("big", "mesh"))
            np.testing.assert_array_equal(whole.arrays["positions"].array(), big.numpy())
            # A viewer that subscribes while the first one watches gets the hub's own state.
            async with Viewer(hub, caps=PIECES) as late:                        # the hub's own state
                await late.subscribe(2, "world")
                want = ["reset", "classes", "transforms static rig", "transforms stamped a",
                        "big:mesh", "graph:lines", "rig/cam:camera", "room/cloud:points"]
                got = []
                while not got or _kind(got[-1]) != want[-1]:   # the split update's pieces: one each
                    got.append(await late.expect(object, lambda m: isinstance(m, SCENE_MSGS)))
                kinds = []
                for m in got:
                    k = _kind(m)
                    if not kinds or k != kinds[-1] or not k.startswith("big:"):
                        kinds.append(k)
                assert kinds == want
                assert all(m.state for m in got)
                pieces = [m for m in got if isinstance(m, w.SceneUpdate) and m.path == "big"]
                assert len(pieces) >= 2 and [m.split["piece"] for m in pieces] == list(range(len(pieces)))
                su = w.SplitUpdates()
                whole = [su.add(m) for m in pieces][-1]
                np.testing.assert_array_equal(whole.arrays["positions"].array(), big.numpy())

    run(go(), timeout=60)


def test_a_31_viewer_through_the_hub_sees_no_scene(hub):
    caliper.connect()
    caliper.scene("hidden").points("p", torch.zeros(1, 3), static=True)
    caliper.series(1.0, "loss")

    async def go():
        async with Viewer(hub, caps=()) as v:
            await v.expect(w.Views, lambda m: any(e["name"] == "loss" for e in m.views))
            assert "hidden" not in v.views
            await v.send(w.Subscribe(1, "hidden", 101))
            r = await v.expect(w.ViewReply, lambda m: m.request_id == 101)
            assert r.error["code"] == w.ERR_UNKNOWN_VIEW

    run(go(), timeout=60)


def test_a_viewer_without_scene_pieces_through_the_hub_gets_the_structure(hub):
    caliper.connect()
    sc = caliper.scene("world")
    sc.camera("cam", width=8, height=6, focal=5.0, static=True)
    sc.points("cloud", torch.rand(5, 3), static=True)
    sc.lines("edges", torch.rand(4, 3), static=True)

    async def go():
        async with Viewer(hub) as old:
            await old.subscribe(1, "world")
            state = [await old.expect(object, lambda m: isinstance(m, SCENE_MSGS)) for _ in range(3)]
            assert [_kind(m) for m in state] == ["reset", "cam:camera", "edges:lines"]
            assert not [m for m in old.inbox if isinstance(m, (w.Resource, w.SceneUpdate))]

    run(go(), timeout=60)
