"""The piece form of scene batches on the network route (caliper/_scene_pieces.py, _net_scene.py), wire
protocol 3.2 (draft), capability "scene_pieces", over a real loopback connection on both senders:

* to a viewer that announced "scene_pieces", a points batch goes as the RESOURCE version of its scene
  program, points@1's resource "points" written as the piece's derive half writes it, followed by the
  SCENE_UPDATE in the piece form that names it, and the arrays come back from the bytes exactly;
* colors, class ids, radii, one radius and a 2-D scene each give their bytes;
* a viewer without the capability receives scene structure only;
* a late subscriber gets each held batch as its RESOURCE and then its update, with ``state`` true;
* a version above 8 MiB goes as RESOURCE parts that put together give the same bytes;
* each entity part keeps its scene program."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
import torch

import caliper
from caliper import _scene, _scene_pieces, _wire as w

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

MPS = torch.backends.mps.is_available()
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not MPS, reason="no MPS device"))]
try:
    from caliper import _netio  # noqa: F401
    NATIVE = True
except ImportError:
    NATIVE = False
MODES = ["python", pytest.param("native", marks=pytest.mark.skipif(
    not NATIVE, reason="caliper._netio is not built (python scripts/build_netio.py)"))]
PIECES = (w.CAP_SCENES, w.CAP_SCENE_PIECES)


@pytest.fixture(autouse=True)
def _forget_scenes(monkeypatch):
    yield
    for name in list(_scene.SCENES):
        caliper.remove(name)


@pytest.fixture(params=MODES)
def server(request, monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1" if request.param == "native" else "0")
    s = caliper.serve(port=0)
    yield s
    s.close()


def run(coro, timeout=60.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


class Viewer:
    """A viewer that keeps every message in the order it arrived, RESOURCE parts put together."""

    def __init__(self, server, caps=PIECES):
        self.url = server.stream_url
        self.caps = list(caps)
        self.got: list = []
        self.views: dict = {}
        self.parts = w.SplitResources()

    async def __aenter__(self):
        self.ws = await connect(self.url, max_size=None, compression=None, ping_interval=None,
                                close_timeout=1).__aenter__()
        await self.ws.send(w.encode(w.Hello("viewer", self.caps)))
        self.welcome = await self.next(lambda m: isinstance(m, w.Welcome))
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def _recv(self, timeout):
        m = w.decode(await asyncio.wait_for(self.ws.recv(), timeout))
        if isinstance(m, w.Views):
            self.views = {v["name"]: v for v in m.views}
        if isinstance(m, w.Resource) and m.split is not None:
            whole = self.parts.add(m)
            if whole is None:
                return None
            m = whole
        return m

    async def next(self, pred, timeout=20.0):
        """The first message, kept or new, for which ``pred`` holds; it stays in ``got``."""
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            for m in self.got[seen:]:
                if pred(m):
                    return m
            seen = len(self.got)
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"not received; got {[type(m).__name__ for m in self.got]}")
            try:
                m = await self._recv(left)
            except asyncio.TimeoutError:
                continue
            if m is not None:
                self.got.append(m)

    async def subscribe(self, sub, view):
        if view not in self.views:
            await self.next(lambda m: isinstance(m, w.Views) and any(v["name"] == view for v in m.views))
        await self.ws.send(w.encode(w.Subscribe(sub, view, 100 + sub, None, None)))
        await self.next(lambda m: isinstance(m, w.ViewReply) and m.request_id == 100 + sub)

    async def update(self, path, part="points", state=None):
        return await self.next(lambda m: isinstance(m, w.SceneUpdate) and m.path == path and m.part == part
                               and (state is None or m.state == state))

    def resource_before(self, u):
        """The RESOURCE of ``u``'s program that came right before ``u``, with only other kinds between."""
        i = self.got.index(u)
        for m in reversed(self.got[:i]):
            if isinstance(m, w.Resource):
                assert m.program == u.program and m.resource == "points"
                return m
        raise AssertionError("no RESOURCE before the update")


def bytes_of(r) -> bytes:
    return bytes(memoryview(r.data).cast("B"))


def expect_piece(v, u, arrays: dict, dims: int = 3):
    """``u`` is a points update in the piece form whose RESOURCE holds ``arrays``."""
    assert u.program is not None and w.scene_program_id(u.program)
    assert u.piece == {"name": "points", "version": 1}
    assert u.arrays == {}
    r = v.resource_before(u)
    n = arrays["positions"].shape[0]
    assert u.resources == {"points": n} and (r.count, r.stride) == (n, 32)
    assert u.names == [k for k in w.PART_SECTIONS["points"] if k in arrays]
    back = _scene_pieces.unpack_points(bytes_of(r), u.names, dims)
    for k, a in arrays.items():
        np.testing.assert_array_equal(back[k], a, err_msg=k)
    rows = np.frombuffer(bytes_of(r), dtype="<u4").reshape(n, 8)
    np.testing.assert_array_equal(rows[:, 5], np.arange(n))           # the row index of each point
    np.testing.assert_array_equal(rows[:, 6:], 0)                     # the padding
    return r


@pytest.mark.parametrize("device", DEVICES)
def test_points_in_the_piece_form_reach_a_viewer(server, device):
    sc = caliper.scene("world")
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(97, 3, generator=g).to(device)
    vals = torch.rand(97, generator=g).to(device)

    async def body():
        async with Viewer(server) as v:
            assert w.CAP_SCENE_PIECES in v.welcome.capabilities
            await v.subscribe(1, "world")
            assert sc.points("cloud", pos, values=vals, cmap="magma", vmin=0.0, vmax=1.0)
            u = await v.update("cloud")
            r = expect_piece(v, u, {"positions": pos.cpu().numpy(), "values": vals.cpu().numpy()})
            f = np.frombuffer(bytes_of(r), dtype="<f4").reshape(97, 8)
            np.testing.assert_array_equal(f[:, 3], 1.0)                 # no radius: 1 in w
            assert (u.params["vmin"], u.params["vmax"]) == (0.0, 1.0)
            assert len(u.params["stats"]) == 8
            assert u.stamp is None and not u.state and not u.static
    run(body())


def test_each_color_source_radii_and_2d_give_their_bytes(server):
    sc = caliper.scene("world")
    flat = caliper.scene("flat", space="2d")
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(11, 3, generator=g)
    rgba = (torch.rand(11, 4, generator=g) * 255).to(torch.uint8)
    ids = torch.randint(0, 9, (11,), generator=g, dtype=torch.int32)
    radii = torch.rand(11, generator=g) + 0.1

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "world")
            await v.subscribe(2, "flat")
            assert sc.points("rgba", pos, colors=rgba, radii=radii)
            assert sc.points("ids", pos, class_ids=ids, radii=0.5)
            assert sc.points("plain", pos)
            assert flat.points("xy", pos[:, :2], values=pos[:, 0])
            u = await v.update("rgba")
            expect_piece(v, u, {"positions": pos.numpy(), "colors": rgba.numpy(), "radii": radii.numpy()})
            u = await v.update("ids")
            r = expect_piece(v, u, {"positions": pos.numpy(), "class_ids": ids.numpy()})
            np.testing.assert_array_equal(np.frombuffer(bytes_of(r), dtype="<f4").reshape(11, 8)[:, 3], 0.5)
            assert u.params["radius"] == 0.5
            u = await v.update("plain")
            r = expect_piece(v, u, {"positions": pos.numpy()})
            np.testing.assert_array_equal(np.frombuffer(bytes_of(r), dtype="<u4").reshape(11, 8)[:, 4],
                                          _scene_pieces.NAN_BITS)
            u = await v.update("xy")
            r = expect_piece(v, u, {"positions": pos[:, :2].numpy(), "values": pos[:, 0].numpy()}, dims=2)
            np.testing.assert_array_equal(np.frombuffer(bytes_of(r), dtype="<f4").reshape(11, 8)[:, 2], 0.0)
    run(body())


def test_older_viewers_get_structure_only(server):
    sc = caliper.scene("world")
    pos = torch.rand(5, 3)

    async def body():
        async with Viewer(server) as new, Viewer(server, caps=(w.CAP_SCENES,)) as old:
            await new.subscribe(1, "world")
            await old.subscribe(1, "world")
            assert sc.points("cloud", pos)
            assert sc.camera("camera", width=8, height=6, focal=5, static=True)
            assert (await new.update("cloud")).program is not None
            await old.update("camera", "camera")
            assert not [m for m in old.got if isinstance(m, w.Resource)]
            assert not [m for m in old.got if isinstance(m, w.SceneUpdate) and m.part != "camera"]
    run(body())


def test_a_late_subscriber_gets_each_resource_before_its_update(server):
    sc = caliper.scene("world")
    still = torch.rand(4, 3)
    moving = torch.rand(6, 3)
    assert sc.points("still", still, static=True) is True

    async def body():
        async with Viewer(server) as a:
            await a.subscribe(1, "world")
            assert sc.points("moving", moving, values=moving[:, 1])
            await a.update("moving")
            async with Viewer(server) as b:
                await b.subscribe(5, "world")
                s1 = await b.update("still", state=True)
                s2 = await b.update("moving", state=True)
                assert s1.static and not s2.static and s1.program != s2.program
                expect_piece(b, s1, {"positions": still.numpy()})
                expect_piece(b, s2, {"positions": moving.numpy(), "values": moving[:, 1].numpy()})
                assert b.got.index(s2) < b.got.index(s1)        # in the byte order of the paths
    run(body())


def test_a_version_above_8_mib_goes_in_parts(server):
    sc = caliper.scene("world")
    n = 300_000                                   # 9.6 MB of points: two parts of 8 MiB at most
    pos = torch.rand(n, 3, generator=torch.Generator().manual_seed(2))

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "world")
            # the default scene data view samples above 100,000 points: a data view of more is not
            # built, thus the batch arrives sampled by stride, in the piece form
            assert sc.points("big", pos)
            u = await v.update("big")
            r = v.resource_before(u)
            assert u.axes is not None and r.count == u.resources["points"]
    run(body())


def test_a_version_in_parts_is_put_together(monkeypatch):
    from caliper import _programs
    monkeypatch.setattr(_programs, "PART_BYTES", 4096)
    st = caliper.scene("parts")
    b = _scene.Batch("parts", "p", "points", 1, 0, False, {}, {}, None)
    b.arrays = {"positions": np.random.default_rng(3).random((1000, 3), dtype=np.float32)}
    frames = _scene_pieces.bundle(1, "parts", b, False)
    assert len(frames) == 1 + -(-1000 * 32 // 4096) and isinstance(frames[-1], w.SceneUpdate)
    sr = w.SplitResources()
    whole = None
    for f in frames[:-1]:
        assert f.split is not None
        whole = sr.add(w.decode(w.encode(f))) or whole
    assert bytes(memoryview(whole.data).cast("B")) == bytes(_scene_pieces.pack_points(b))
    del st


def test_each_entity_part_keeps_its_scene_program():
    a = _scene_pieces.program_id("s", "x", "points")
    assert _scene_pieces.program_id("s", "x", "points") == a
    assert _scene_pieces.program_id("s", "y", "points") != a
    assert _scene_pieces.program_id("t", "x", "points") != a
    assert w.scene_program_id(a)


def test_color_and_grey_images_in_the_piece_form(server):
    flat = caliper.scene("flat", space="2d")
    g = torch.Generator().manual_seed(4)
    rgb = torch.rand(5, 7, 3, generator=g)                       # floats: converted to RGBA u8
    grey = (torch.rand(4, 6, generator=g) * 255).to(torch.uint8)

    async def body():
        async with Viewer(server) as v:
            await v.subscribe(1, "flat")
            assert flat.image("rgb", rgb, opacity=0.5)
            assert flat.image("grey", grey)
            for path, h, wd, ch in (("rgb", 5, 7, 4), ("grey", 4, 6, 1)):
                u = await v.update(path, "image")
                assert u.piece == {"name": "image", "version": 1} and u.names == ["pixels"]
                assert u.resources == {"pixels": h * wd, "info": 1} and u.arrays == {}
                i = v.got.index(u)
                info, pixels = v.got[i - 1], v.got[i - 2]
                assert (pixels.resource, pixels.count, pixels.stride) == ("pixels", h * wd, 4)
                assert (pixels.width, pixels.height, pixels.format) == (wd, h, None)   # rgba8unorm texels
                assert (info.resource, info.count, info.stride) == ("info", 1, 80)
                assert pixels.program == info.program == u.program
                size = np.frombuffer(bytes_of(info), dtype="<u4")
                assert size[:4].tolist() == [wd, h, ch, 0]
                np.testing.assert_array_equal(np.frombuffer(bytes_of(info), dtype="<f4")[4:].reshape(4, 4), np.eye(4))
                px = np.frombuffer(bytes_of(pixels), dtype=np.uint8).reshape(h, wd, 4)
                if ch == 1:
                    np.testing.assert_array_equal(px[..., 0], grey.numpy())
                    np.testing.assert_array_equal(px[..., 1], grey.numpy())
                    assert (px[..., 3] == 255).all()
                else:
                    want = (rgb.clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()
                    np.testing.assert_array_equal(px[..., :3], want)
                    assert (px[..., 3] == 255).all() and u.params["opacity"] == 0.5
    run(body())


def test_every_library_layout_preserves_the_source_arrays():
    from pathlib import Path
    from scene_piece_reader import arrays
    folder = Path(__file__).resolve().parents[2] / "tests/fixtures/wire_v3/v3_2"
    checked = set()
    for path in folder.glob("scene_update_*_bundle.bin"):
        update = w.decode(path.read_bytes())
        base = path.stem.removesuffix("_bundle")
        source = w.decode((folder / (base + ".bin")).read_bytes())
        resources = {rid: w.decode((folder / (path.stem.replace("scene_update_", "resource_scene_", 1) + "_" + rid + ".bin")).read_bytes())
                     for rid in update.resources}
        first = next(iter(source.arrays.values()))
        dims = first.shape[1] if source.part != "image" else 3
        back = arrays(update, resources, dims)
        for name, section in source.arrays.items():
            np.testing.assert_array_equal(back[name], section.array(), err_msg=path.stem + ":" + name)
        checked.add(update.piece["name"])
    assert checked == set(_scene_pieces.PINS)


def test_lines_preserve_unused_vertices_and_singleton_strips():
    from types import SimpleNamespace
    from scene_piece_reader import arrays
    positions = np.arange(15, dtype="<f4").reshape(5, 3)
    for extra, params in (({"indices": np.array([[0, 1]], "<i4")}, {"mode": "edges"}),
                          ({"strip_lengths": np.array([1, 3, 1], "<i4")}, {"mode": "strips"})):
        batch = SimpleNamespace(part="lines", arrays={"positions": positions, **extra}, params=params, packed=None)
        names, packed = _scene_pieces.pack(batch, "lines")
        update = w.SceneUpdate(1, "s", 1, 0, "line", "lines", {}, params,
                               program="scene.1", piece=_scene_pieces.PINS["lines"], names=names,
                               resources={name: count for name, count, *_ in packed})
        resources = {name: w.Resource("scene.1", name, 0, count, stride, data, width, height)
                     for name, count, stride, data, width, height in packed}
        back = arrays(update, resources)
        for name, expected in batch.arrays.items():
            np.testing.assert_array_equal(back[name], expected)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_encoded_images_become_bounded_texture_resources(format):
    import io
    from types import SimpleNamespace
    from PIL import Image
    image = Image.new("RGB", (7, 5), (240, 40, 80))
    stream = io.BytesIO()
    image.save(stream, format=format)
    batch = SimpleNamespace(part="image", arrays={"pixels": np.frombuffer(stream.getvalue(), np.uint8)},
                            params={"form": "encoded", "format": format.lower()}, packed=None)
    assert _scene_pieces.route(batch) == "image"
    assert batch.arrays["pixels"].shape == (5, 7, 4)
    assert batch.params == {"form": "color"}
    _, resources = _scene_pieces.pack(batch, "image")
    assert resources[0][1:3] == (35, 4)


def test_encoded_images_refuse_malformed_and_oversize_headers():
    import io
    from types import SimpleNamespace
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (8193, 1)).save(stream, format="PNG")
    for data in (b"not an image", stream.getvalue()):
        batch = SimpleNamespace(part="image", arrays={"pixels": np.frombuffer(data, np.uint8)},
                                params={"form": "encoded", "format": "png"}, packed=None)
        with pytest.raises(ValueError):
            _scene_pieces.route(batch)


@pytest.mark.parametrize("device", DEVICES)
def test_point_colors_preserve_class_ids_for_labels(server, device):
    from test_net_scene import Viewer as ArrayViewer
    scene = caliper.scene("labels")
    positions = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32, device=device)
    colors = torch.tensor([[255, 10, 20, 255], [30, 40, 255, 255]], dtype=torch.uint8, device=device)
    ids = torch.tensor([7, 12], dtype=torch.int32, device=device)
    async def body():
        async with ArrayViewer(server) as viewer:
            await viewer.subscribe(1, "labels")
            assert scene.points("points", positions, colors=colors, class_ids=ids)
            update = await viewer.update("points")
            np.testing.assert_array_equal(update.arrays["class_ids"].array(), ids.cpu().numpy())
            np.testing.assert_array_equal(update.arrays["colors"].array(), colors.cpu().numpy())
    run(body())
