"""The rest of caliper.scene on the network route (caliper/_scene.py,
_net_scene.py): every part of the protocol 3.2 draft's SCENE_UPDATE, entity
transforms, class tables, clears and the starting camera, over a real
loopback connection with a websockets viewer, on both senders (the native one
when caliper._netio is built) and on the CPU and, where there is one, MPS:

* arrows, boxes, images in four forms, cameras, index-pair edges, class ids,
  labels, normals and textures reach a viewer with the right arrays, cast and
  bounded on the device;
* entity transforms are composed as T R S on the device of their tensors, and
  a subscription's pending transforms merge into one SCENE_TRANSFORMS;
* a late subscriber receives the scene's state in the draft's order: the
  reset, the class table, the static and then the other entity transforms,
  then every camera and batch in the byte order of their paths;
* a clear reaches the viewer in order, drops what it clears, and leaves
  static parts unless it is static too;
* a batch over one frame travels as a split update that the codec puts back
  together;
* every refusal of the new calls raises, attached or not."""
from __future__ import annotations

import asyncio
import math
import struct
import zlib

import numpy as np
import pytest
import torch

import caliper
from caliper import _net_scene, _scene, _wire as w

from test_net_scene import DEVICES, MODES, Viewer, run  # noqa: F401  (MODES for the server fixture)
from test_net_scene import _forget_scenes, server  # noqa: F401  (fixtures)


@pytest.fixture(autouse=True)
def _no_session(monkeypatch):
    """No shared-memory session: a call without caliper.serve() has no route."""
    monkeypatch.setenv("CALIPER_SESSION", "0")


def _png(rgb=(255, 128, 0)) -> bytes:
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + bytes(rgb))) + chunk(b"IEND", b""))


def _arr(m, name):
    return m.arrays[name].array()


def _quat_matrix(q, t, s=(1.0, 1.0, 1.0)):
    """T R S as a row-major 4 x 4 numpy matrix, for checking."""
    x, y, z, wq = np.asarray(q, np.float64) / np.linalg.norm(q)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * wq), 2 * (x * z + y * wq)],
                  [2 * (x * y + z * wq), 1 - 2 * (x * x + z * z), 2 * (y * z - x * wq)],
                  [2 * (x * z - y * wq), 2 * (y * z + x * wq), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * np.asarray(s)
    m[:3, 3] = t
    return m


@pytest.mark.parametrize("device", DEVICES)
def test_every_part_reaches_a_viewer(server, device):
    view_camera = {"target": [0.0, 0.0, 0.5], "azimuth": 30.0, "elevation": 25.0}
    sc = caliper.scene("world", up="+z", view_camera=view_camera)
    dev = torch.device(device)
    gen = torch.Generator().manual_seed(4)
    pts = torch.rand(5, 3, generator=gen).to(dev)
    ids = torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64, device=dev)
    edges = torch.tensor([[0, 1], [1, 2], [2, 4]], dtype=torch.int64, device=dev)
    q = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    grid = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float32, device=dev)
    tri = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32, device=dev)
    nrm = torch.tensor([[0, 0, 1]] * 4, dtype=torch.float32, device=dev)
    uv = grid[:, :2].clone()
    tex = torch.rand(2, 2, 3, generator=gen).to(dev)                         # floats, RGB
    img = torch.randint(0, 256, (6, 8, 3), generator=gen, dtype=torch.uint8).to(dev)
    depth = torch.tensor([[0.5, 1.0, 0.0, 2.0]] * 6, dtype=torch.float32, device=dev)
    seg = torch.randint(0, 300, (6, 8), generator=gen, dtype=torch.int64).to(dev)
    png = _png()

    async def go():
        async with Viewer(server) as v:
            await v.subscribe(1, "world")
            assert v.views["world"]["scene"]["view_camera"] == view_camera
            await v.expect(w.SceneClear, lambda m: m.reset and m.state)
            assert sc.classes({0: "floor", 1: ("obstacle", (230, 80, 60))}) is True
            assert sc.transform("rig", translation=torch.tensor([1.0, 2.0, 3.0], device=dev),
                                rotation=q, static=True) is True
            assert sc.camera("rig/left", width=8, height=6, focal=5.25, static=True) is True
            assert sc.points("room/cloud", pts, class_ids=ids, labels="p")
            assert sc.lines("graph/edges", pts, indices=edges, values=torch.tensor(
                [0.25, 0.5, 1.0], device=dev), cmap="magma", width=1.5)
            assert sc.arrows("field", pts[:2], pts[:2] * 2, colors=(255, 0, 0))
            assert sc.boxes("arm/link0", centers=[(0.0, 0.0, 0.25)],
                            half_sizes=[(0.05, 0.05, 0.25)], rotations=[(0.0, 0.0, 0.0, 1.0)],
                            labels="link", fill=True)
            assert sc.mesh("floor", grid, tri, normals=nrm, uvs=uv, texture=tex)
            assert sc.image("rig/left/rgb", img, opacity=0.75)
            assert sc.depth("rig/left/depth", depth)
            assert sc.segmentation("rig/left/classes", seg)
            assert sc.image("rig/left/frame", png)

            k = await v.expect(w.SceneClasses)
            assert k.classes == [{"id": 0, "label": "floor", "color": [31, 119, 180, 255]},
                                 {"id": 1, "label": "obstacle", "color": [230, 80, 60, 255]}]
            assert k.static
            t = await v.expect(w.SceneTransforms)
            assert t.paths == ["rig"] and t.static and t.stamp is None
            got = t.matrices.array().reshape(4, 4).T                       # column-major
            np.testing.assert_allclose(got, _quat_matrix(q, (1, 2, 3)), atol=1e-6)
            cam = await v.update("rig/left", "camera")
            assert cam.arrays == {} and cam.static
            assert cam.params == {"fx": 5.25, "fy": 5.25, "cx": 4.0, "cy": 3.0, "width": 8,
                                  "height": 6, "convention": "rdf"}
            p = await v.update("room/cloud", "points")
            assert _arr(p, "class_ids").dtype == np.int32
            assert _arr(p, "class_ids").tolist() == ids.cpu().tolist()
            assert p.params["labels"] == ["p"]
            e = await v.update("graph/edges", "lines")
            assert (e.params["mode"], e.params["color_per"]) == ("edges", "edge")
            assert _arr(e, "indices").dtype == np.int32
            assert _arr(e, "indices").tolist() == edges.cpu().tolist()
            assert _arr(e, "values").tolist() == [0.25, 0.5, 1.0]
            assert _arr(e, "positions").shape == (5, 3)
            a = await v.update("field", "arrows")
            np.testing.assert_array_equal(_arr(a, "vectors"), (pts[:2] * 2).cpu().numpy())
            assert a.params["color"] == [255, 0, 0, 255] and a.params["width"] == 2.0
            b = await v.update("arm/link0", "boxes")
            assert b.params["labels"] == ["link"] and b.params["fill"] is True
            assert _arr(b, "rotations").tolist() == [[0, 0, 0, 1]]
            r = math.sqrt(2 * 0.05 ** 2 + 0.25 ** 2)                          # its bounding sphere
            np.testing.assert_allclose(b.params["stats"][2:], [-r, r, -r, r, 0.25 - r, 0.25 + r],
                                       rtol=0, atol=1e-6)
            m = await v.update("floor", "mesh")
            assert _arr(m, "normals").tolist() == nrm.cpu().tolist()
            tx = _arr(m, "texture")
            assert tx.shape == (2, 2, 4) and tx.dtype == np.uint8 and (tx[..., 3] == 255).all()
            want = (tex.cpu().clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()
            np.testing.assert_array_equal(tx[..., :3], want)
            im = await v.update("rig/left/rgb", "image")
            assert im.params["form"] == "color" and im.params["opacity"] == 0.75
            px = _arr(im, "pixels")
            assert px.shape == (6, 8, 4)
            np.testing.assert_array_equal(px[..., :3], img.cpu().numpy())
            d = await v.update("rig/left/depth", "image")
            assert d.params["form"] == "depth" and d.params["cloud"] is True     # under a camera
            assert d.params["scale"] == 1.0 and d.params["flags"] == 3
            assert d.params["stats"] == [0.5, 2.0]                            # zeros left out
            assert _arr(d, "pixels").dtype == np.float32
            s = await v.update("rig/left/classes", "image")
            assert _arr(s, "pixels").dtype == np.uint16
            assert _arr(s, "pixels").tolist() == seg.cpu().tolist()
            f = await v.update("rig/left/frame", "image")
            assert f.params["form"] == "color" and "format" not in f.params
            assert _arr(f, "pixels").tolist() == [[[255, 128, 0, 255]]]

    run(go())


def test_a_2d_scene_and_the_pixel_space_of_a_camera(server):
    flat = caliper.scene("flat", space="2d")
    world = caliper.scene("world3")

    async def go():
        async with Viewer(server) as v:
            await v.subscribe(1, "flat")
            await v.subscribe(2, "world3")
            await v.expect(w.SceneClear, lambda m: m.sub == 1 and m.reset)
            await v.expect(w.SceneClear, lambda m: m.sub == 2 and m.reset)
            flat.boxes("cars", corners=[(10.0, 20.0, 4.0, 2.0)], colors=(0, 255, 0))
            flat.transform("cars", translation=(5.0, 6.0), rotation=math.pi / 2, scale=2.0)
            flat.image("map", torch.zeros(3, 4, dtype=torch.uint8))          # grey, anywhere in 2-D
            b = await v.update("cars", "boxes")
            assert _arr(b, "centers").tolist() == [[7.0, 11.0]]
            assert _arr(b, "half_sizes").tolist() == [[3.0, 9.0]]
            t = await v.expect(w.SceneTransforms, lambda m: m.sub == 1)
            got = t.matrices.array().reshape(4, 4).T
            want = np.array([[0, -2, 0, 5], [2, 0, 0, 6], [0, 0, 1, 0], [0, 0, 0, 1]], float)
            np.testing.assert_allclose(got, want, atol=1e-6)
            g = await v.update("map", "image")
            assert _arr(g, "pixels").shape == (3, 4)
            # 3-D: a camera's children live in its 2-D pixel space
            world.camera("cam", width=4, height=3, fov=90.0)
            world.points("cam/marks", torch.zeros(2, 2))
            world.transform("cam/marks", translation=(1.0, 2.0))
            c = await v.update("cam", "camera")
            assert c.params["fx"] == pytest.approx(1.5) and c.params["fy"] == pytest.approx(1.5)
            mk = await v.update("cam/marks", "points")
            assert _arr(mk, "positions").shape == (2, 2)
            await v.expect(w.SceneTransforms, lambda m: m.sub == 2 and m.paths == ["cam/marks"])

    run(go())


def test_a_late_subscriber_gets_the_scene_state_in_order(server):
    sc = caliper.scene("late")
    sc.classes({3: "c"})
    sc.transform("z/b", translation=(0.0, 0.0, 1.0), static=True)
    sc.transform("a", scale=2.0)
    sc.camera("c/cam", width=2, height=2, focal=1.0)
    sc.points("b", torch.zeros(2, 3), static=True)
    sc.mesh("a", torch.zeros(3, 3), static=True)

    async def go():
        async with Viewer(server) as v:
            await v.subscribe(1, "late")
            seen = []
            while len(seen) < 7:
                m = await v.expect(object, lambda m: isinstance(
                    m, (w.SceneClear, w.SceneClasses, w.SceneTransforms, w.SceneUpdate)))
                seen.append(m)
            kinds = [type(m).__name__ + (":" + m.path + "/" + m.part if isinstance(
                m, w.SceneUpdate) else "") for m in seen]
            assert kinds == ["SceneClear", "SceneClasses", "SceneTransforms", "SceneTransforms",
                             "SceneUpdate:a/mesh", "SceneUpdate:b/points",
                             "SceneUpdate:c/cam/camera"]
            assert seen[0].reset and all(m.state for m in seen)
            assert seen[2].static and seen[2].paths == ["z/b"]
            assert not seen[3].static and seen[3].paths == ["a"]
            np.testing.assert_allclose(seen[3].matrices.array().reshape(4, 4).T,
                                       np.diag([2.0, 2.0, 2.0, 1.0]), atol=0)

    run(go())


def test_clears_reach_the_viewer_in_order_and_leave_static_parts(server):
    sc = caliper.scene("clr")
    sc.points("a/static", torch.zeros(1, 3), static=True)

    async def go():
        async with Viewer(server) as v:
            await v.subscribe(1, "clr")
            await v.update("a/static", "points")
            sc.points("a", torch.ones(2, 3))
            await v.update("a", "points")
            sc.transform("a", translation=(1.0, 0.0, 0.0))
            await v.expect(w.SceneTransforms)
            assert sc.clear("a", recursive=True) is True
            c = await v.expect(w.SceneClear, lambda m: not m.reset)
            assert (c.path, c.recursive, c.static) == ("a", True, False)
        async with Viewer(server) as late:                                   # a later viewer
            await late.subscribe(1, "clr")
            state = []
            while not state or not isinstance(state[-1], w.SceneUpdate):
                state.append(await late.expect(object, lambda m: isinstance(
                    m, (w.SceneClear, w.SceneTransforms, w.SceneUpdate))))
            assert [type(m).__name__ for m in state] == ["SceneClear", "SceneUpdate"]
            assert state[1].path == "a/static"                               # static stays
            sc.clear("", recursive=True, static=True)
            await late.expect(w.SceneClear, lambda m: m.static and m.path == "")
        assert _scene.SCENES["clr"].static == {}
        async with Viewer(server) as last:
            await last.subscribe(1, "clr")
            await last.expect(w.SceneClear, lambda m: m.reset)
            sc.points("z", torch.zeros(1, 3))                # the state goes first: nothing is left
            assert (await last.expect(w.SceneUpdate)).path == "z"

    run(go())


def test_large_resources_arrive_before_the_update(server, monkeypatch):
    from caliper import _programs
    monkeypatch.setattr(_programs, "PART_BYTES", 24 * 1024)
    sc = caliper.scene("big")
    pos = torch.arange(3 * 4000, dtype=torch.float32).reshape(4000, 3) / 7

    async def go():
        async with Viewer(server) as v:
            await v.subscribe(1, "big")
            sc.mesh("m", pos[:3999], values=pos[:3999, 0], vmin=0.0, vmax=1.0)
            whole = await v.update("m", "mesh")
            assert whole.split is None and whole.program and whole.params["vmax"] == 1.0
            resources = [m for m in v.inbox if isinstance(m, w.Resource) and m.split]
            assert len(resources) > 1
            np.testing.assert_array_equal(_arr(whole, "positions"), pos[:3999].numpy())
            np.testing.assert_array_equal(_arr(whole, "values"), pos[:3999, 0].numpy())
            async with Viewer(server) as late:
                await late.subscribe(1, "big")
                whole = await late.update("m", "mesh")
                assert whole.state and whole.program and whole.split is None
                np.testing.assert_array_equal(_arr(whole, "positions"), pos[:3999].numpy())
    run(go())


def test_view_camera_travels_in_views(server):
    async def go():
        async with Viewer(server) as v:
            caliper.scene("vc", view_camera="rig/left")
            await v.expect(w.Views, lambda m: any(
                e["name"] == "vc" and e["scene"].get("view_camera") == "rig/left" for e in m.views))
            caliper.scene("vc")                                              # None leaves it
            caliper.scene("vc", view_camera={"distance": 3})
            m = await v.expect(w.Views, lambda m: any(
                e["name"] == "vc" and e["scene"].get("view_camera") == {"distance": 3.0}
                for e in m.views))
            info = w.scene_view_info(next(e for e in m.views if e["name"] == "vc"))
            assert info["view_camera"] == {"distance": 3.0}
            caliper.scene("vc", view_camera={})                              # back to the fit
            await v.expect(w.Views, lambda m: any(
                e["name"] == "vc" and e["scene"].get("view_camera") == {} for e in m.views))

    run(go())


def test_view_camera_refusals():
    for vc, exc in ((3, TypeError), ([0, 0, 1], TypeError), ({"target": "abc"}, TypeError),
                    ({"azimuth": "x"}, TypeError), ({"fov": 45}, ValueError),
                    ({"distance": 0}, ValueError), ({"elevation": 90}, ValueError),
                    ({"azimuth": math.inf}, ValueError), ({"target": [0, 0, math.nan]}, ValueError),
                    ("", ValueError), ("a/../b", ValueError)):
        with pytest.raises(exc):
            caliper.scene("vcr", view_camera=vc)
    with pytest.raises(ValueError, match="2-D"):
        caliper.scene("vcr2", space="2d", view_camera={})
    sc = caliper.scene("vcr", view_camera={"elevation": -89, "azimuth": 400})
    assert sc.view_camera == {"elevation": -89.0, "azimuth": 400.0}


def test_refusals_of_the_new_calls():
    sc = caliper.scene("ref")
    flat = caliper.scene("ref2", space="2d")
    p3, p2 = torch.zeros(4, 3), torch.zeros(4, 2)
    sc.camera("cam", width=4, height=3, focal=2.0)
    bad = [
        (ValueError, lambda: sc.points("p", p3, class_ids=torch.zeros(3, dtype=torch.int32))),
        (ValueError, lambda: sc.points("p", p3, class_ids=torch.zeros(4))),        # floats
        (ValueError, lambda: sc.points("p", p3, labels=["a", "b"])),
        (ValueError, lambda: sc.points("p", p3, labels=["x" * 64] * 4)),
        (TypeError, lambda: sc.points("p", p3, labels=3)),
        (ValueError, lambda: sc.lines("l", p3, indices=torch.zeros(2, 3, dtype=torch.int64))),
        (ValueError, lambda: sc.lines("l", p3, indices=torch.zeros(2, 2), )),        # floats
        (ValueError, lambda: sc.lines("l", p3, indices=torch.zeros(3, 2, dtype=torch.int32),
                                      strip_lengths=[4])),
        (ValueError, lambda: sc.lines("l", p3, indices=torch.zeros(4, 2, dtype=torch.int32),
                                      values=torch.zeros(4))),                      # V == E
        (ValueError, lambda: sc.lines("l", p3, indices=torch.zeros(3, 2, dtype=torch.int32),
                                      color_per="strip")),
        (ValueError, lambda: sc.lines("l", p3, color_per="edge")),
        (ValueError, lambda: sc.arrows("a", p3, p3[:3])),
        (ValueError, lambda: sc.arrows("a", p3, p2)),
        (TypeError, lambda: sc.boxes("b")),
        (ValueError, lambda: sc.boxes("b", p3, p3[:3])),
        (ValueError, lambda: sc.boxes("b", p3, p3, rotations=torch.zeros(4, 3))),
        (ValueError, lambda: sc.boxes("b", corners=torch.zeros(1, 4))),              # 3-D
        (ValueError, lambda: flat.boxes("b", p2, p2, rotations=torch.zeros(4, 4))),
        (ValueError, lambda: flat.boxes("b", p2, p2, corners=torch.zeros(4, 4))),
        (ValueError, lambda: sc.mesh("m", p3[:3], uvs=torch.zeros(3, 2))),
        (ValueError, lambda: sc.mesh("m", p3[:3], uvs=torch.zeros(3, 2),
                                     texture=torch.zeros(2, 2, 2))),
        (ValueError, lambda: sc.mesh("m", p3[:3], colors=(1, 2, 3), uvs=torch.zeros(3, 2),
                                     texture=torch.zeros(2, 2, 3))),
        (ValueError, lambda: sc.mesh("m", p3[:3], normals=torch.zeros(3, 2))),
        (ValueError, lambda: sc.image("img", torch.zeros(2, 2, 3))),                # not under a camera
        (ValueError, lambda: sc.image("cam/i", torch.zeros(2, 2, 2))),
        (ValueError, lambda: sc.image("cam/i", torch.zeros(2, 2, 3), opacity=2)),
        (ValueError, lambda: sc.image("cam/i", b"not an image")),
        (ValueError, lambda: sc.image("cam/i", torch.zeros(8193, 1, dtype=torch.uint8))),
        (ValueError, lambda: sc.depth("cam/d", torch.zeros(2, 2, dtype=torch.int32))),
        (ValueError, lambda: sc.depth("cam/d", torch.zeros(2, 2), scale=0)),
        (ValueError, lambda: sc.segmentation("cam/s", torch.zeros(2, 2))),
        (ValueError, lambda: sc.points("cam", p3)),                                 # a camera entity
        (ValueError, lambda: sc.points("cam/p", p3)),                               # 3-D under it
        (ValueError, lambda: sc.camera("cam/inner", width=2, height=2, focal=1.0)),
        (ValueError, lambda: sc.camera("", width=2, height=2, focal=1.0)),
        (TypeError, lambda: sc.camera("c2", width=2.5, height=2, focal=1.0)),
        (ValueError, lambda: sc.camera("c2", width=0, height=2, focal=1.0)),
        (ValueError, lambda: sc.camera("c2", width=2, height=2)),
        (ValueError, lambda: sc.camera("c2", width=2, height=2, focal=1.0, fov=60)),
        (ValueError, lambda: sc.camera("c2", width=2, height=2, fov=180)),
        (ValueError, lambda: sc.camera("c2", width=2, height=2, focal=(1.0, -1.0))),
        (ValueError, lambda: sc.camera("c2", width=2, height=2, focal=1.0, convention="flu")),
        (ValueError, lambda: sc.transform("", translation=(1, 2, 3))),
        (ValueError, lambda: sc.transform("t", translation=(1, 2))),
        (ValueError, lambda: sc.transform("t", rotation=(0, 0, 1))),
        (ValueError, lambda: sc.transform("t", scale=(1, 2))),
        (ValueError, lambda: sc.transform("t", matrix=np.eye(3))),
        (ValueError, lambda: sc.transform("t", matrix=np.eye(4), scale=2.0)),
        (ValueError, lambda: flat.transform("t", rotation=(0, 0, 0, 1))),
        (TypeError, lambda: sc.classes([1, 2])),
        (TypeError, lambda: sc.classes({"a": "b"})),
        (ValueError, lambda: sc.classes({70000: "x"})),
        (ValueError, lambda: sc.classes({1: "x" * 64})),
        (TypeError, lambda: sc.classes({1: 3})),
        (ValueError, lambda: sc.classes({i: "c" for i in range(4097)})),
        (TypeError, lambda: sc.clear(3)),
        (TypeError, lambda: sc.clear("a", recursive=1)),
    ]
    for exc, call in bad:
        with pytest.raises(exc):
            call()
    # one colour for all, and a camera's children taking 2-D data are fine
    assert sc.points("cam/p", p2) is False                                  # nothing attached
    assert sc.transform("t", matrix=np.eye(4)) is True
    assert sc.clear("") is True


def test_a_subscriptions_pending_transforms_merge_and_a_clear_drops_them():
    q = _net_scene.SubQueue(1, "s", 0.0)
    eye = np.eye(4, dtype=np.float32).reshape(16)

    def tr(path, seq, static=False):
        t = _scene.Transform("s", path, seq, seq, static, None)
        t.values = eye * seq
        return t
    for item in (tr("a", 1), tr("b", 2), tr("a", 3), tr("s", 4, static=True)):
        q.offer(item)
    first, _ = q.next_item(0.0)
    assert isinstance(first, w.SceneTransforms) and first.static and first.paths == ["s"]
    second, _ = q.next_item(0.0)
    assert second.paths == ["a", "b"] and not second.static
    assert second.matrices.tolist()[0] == (eye * 3).tolist()                 # the newest of "a"
    assert q.next_item(0.0) == (None, None)
    q.offer(tr("a", 5))
    q.offer(_scene.Classes("s", 6, 6, True, []))
    q.offer(_scene.Classes("s", 7, 7, True, [{"id": 1, "label": "x", "color": [0, 0, 0, 255]}]))
    q.offer(_scene.Clear("s", 8, 8, "a", False, False))
    msgs = [q.next_item(0.0)[0] for _ in range(2)]
    assert isinstance(msgs[0], w.SceneClear) and msgs[0].path == "a"
    assert isinstance(msgs[1], w.SceneClasses) and msgs[1].seq == 7          # the newer table
    assert q.next_item(0.0) == (None, None)                                 # "a" was cleared
