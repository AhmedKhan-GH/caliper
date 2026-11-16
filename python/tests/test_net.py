"""caliper.serve(): the network route over wire protocol v3 (caliper/_net.py,
_net_ring.py, _net_dataview.py), tested with a websockets client and the codec
caliper/_wire.py. MPS cases skip without an MPS device."""
from __future__ import annotations

import ast
import asyncio
import gc
import json
import os
import resource
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import caliper
from caliper import _net, _wire as w

PKG = Path(caliper.__file__).resolve().parent
websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

MPS = torch.backends.mps.is_available()
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not MPS, reason="no MPS device"))]

try:
    from caliper import _netio  # noqa: F401
    NATIVE = True
except ImportError:
    NATIVE = False
# Every test with a server runs on both senders: the native one (caliper._netio,
# when it is built) and the pure-Python one.
MODES = ["python", pytest.param("native", marks=pytest.mark.skipif(
    not NATIVE, reason="caliper._netio is not built (python scripts/build_netio.py)"))]


@pytest.fixture(params=MODES)
def mode(request, monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1" if request.param == "native" else "0")
    return request.param


@pytest.fixture
def server(mode):
    s = caliper.serve(port=0)
    assert (type(s).__name__ == "NativeServer") == (mode == "native")
    yield s
    s.close()


def run(coro, timeout=30.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


class Client:
    """A viewer: HELLO, then everything decoded; helpers to wait for a type."""

    def __init__(self, server, origin=None, url=None, max_queue=16):
        self.url = url or server.stream_url
        self.origin = origin
        self.max_queue = max_queue
        self.ws = None
        self.inbox: list = []
        self.views: set = set()               # names in the last VIEWS received

    async def __aenter__(self):
        kw = {"origin": self.origin} if self.origin else {}
        self.ws = await connect(self.url, max_size=None, compression=None, ping_interval=None, close_timeout=1,
                                max_queue=self.max_queue, **kw).__aenter__()
        await self.send(w.Hello("viewer", software={"name": "test", "version": "0"}))
        self.welcome = await self.expect(w.Welcome)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def send(self, msg):
        await self.ws.send(w.encode(msg))

    async def recv(self, timeout=10.0):
        m = w.decode(await asyncio.wait_for(self.ws.recv(), timeout))
        if isinstance(m, w.Views):
            self.views = {v["name"] for v in m.views}
        return m

    async def expect(self, cls, timeout=10.0, pred=lambda m: True, pump=None):
        """The next message of type ``cls`` matching ``pred``; ``pump()`` runs
        (synchronously, as a training step would) while waiting."""
        deadline = time.monotonic() + timeout
        while True:
            for i, m in enumerate(self.inbox):
                if isinstance(m, cls) and pred(m):
                    return self.inbox.pop(i)
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no {cls.__name__}; got {[type(m).__name__ for m in self.inbox]}")
            if pump is not None:
                pump()
            try:
                self.inbox.append(await self.recv(0.02 if pump is not None else left))
            except asyncio.TimeoutError:
                pass

    async def subscribe(self, sub, view, data_view=None, max_rate=None, pump=None):
        if view not in self.views:
            await self.expect(w.Views, pred=lambda m: any(v["name"] == view for v in m.views),
                              pump=pump)
        await self.send(w.Subscribe(sub, view, 100 + sub, max_rate, data_view))
        return await self.expect(w.ViewReply, pred=lambda m: m.request_id == 100 + sub, pump=pump)


def _http(port, path, headers, method="GET"):
    """A raw HTTP/1.1 request; returns (status, headers dict, body)."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        lines = [f"{method} {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in headers.items()]
        s.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data:
                head, _, body = data.partition(b"\r\n\r\n")
                hdrs = {}
                for line in head.decode().split("\r\n")[1:]:
                    k, _, v = line.partition(":")
                    hdrs[k.strip().lower()] = v.strip()
                n = int(hdrs.get("content-length", "-1"))
                if n < 0 or len(body) >= n:
                    break
    status = int(head.split()[1])
    return status, hdrs, body


UPGRADE = {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13",
           "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}


# ---------------------------------------------------------------------------
# import and hot-path hygiene
# ---------------------------------------------------------------------------
def test_import_caliper_loads_neither_torch_nor_websockets():
    code = ("import sys, caliper; "
            "bad = [m for m in ('torch', 'websockets', 'caliper._net') if m in sys.modules]; "
            "sys.exit(repr(bad) if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(PKG.parent)},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


FORBIDDEN = {"item", "cpu", "numpy", "tolist", "synchronize"}


@pytest.mark.parametrize("module", ["_net_ring.py", "_net_dataview.py"])
def test_training_thread_code_never_reads_back(module):
    """The training-thread side of serve() (the ring and the data view engine)
    calls no .item(), .cpu(), .numpy(), .tolist() or synchronize()."""
    tree = ast.parse((PKG / module).read_text())
    bad = [(n.lineno, n.func.attr) for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr in FORBIDDEN]
    assert bad == [], f"{module}: {bad}"


def test_no_viewer_means_no_work(server):
    assert not server.transport.attached()
    t = torch.randn(64, 64)
    assert caliper.show(t, name="idle") is False
    caliper.series(1.0, name="idle_s")                  # retained, not sent
    assert "idle" not in server.stats()["views"]         # never reached the ring


def test_serve_twice_and_close(server):
    assert caliper.serve() is server
    assert caliper._server is server and _net.current_server() is server
    assert server.url == f"http://127.0.0.1:{server.port}/?token={server.token}"
    assert server.stream_url == f"ws://127.0.0.1:{server.port}/v3/stream?token={server.token}"
    assert len(server.token) >= w.TOKEN_MIN
    server.close()
    assert server.closed and caliper._server is None and _net.current_server() is None
    assert caliper._series_tap is None
    s2 = caliper.serve()
    assert s2 is not server and s2.port != 0
    s2.close()


def test_context_manager(mode):
    with caliper.serve() as s:
        assert not s.closed
    assert s.closed and caliper._server is None


# ---------------------------------------------------------------------------
# the upgrade and the handshake
# ---------------------------------------------------------------------------
def _upgrade(server, path=None, host=None, origin=None):
    h = dict(UPGRADE, Host=host or f"127.0.0.1:{server.port}")
    if origin is not None:
        h["Origin"] = origin
    return _http(server.port, path or f"/v3/stream?token={server.token}", h)[0]


def test_upgrade_accepted_from_loopback_origins(server):
    assert _upgrade(server) == 101
    assert _upgrade(server, host=f"localhost:{server.port}") == 101
    assert _upgrade(server, host=f"LOCALHOST:{server.port}") == 101
    for origin in ("null", f"http://127.0.0.1:{server.port}", "http://127.0.0.1:8888",
                   "http://localhost:8888", "https://localhost", "http://[::1]:8888",
                   "vscode-webview://1abc2def"):
        assert _upgrade(server, origin=origin) == 101, origin


def test_jupyterlab_origin(server):
    """A WebSocket from a JupyterLab page carries its page's origin."""
    async def go():
        async with Client(server, origin="http://127.0.0.1:8888") as c:
            assert c.welcome.role == "producer"
    run(go())


@pytest.mark.parametrize("path", ["/v3/stream", "/v3/stream?token=", "/v3/stream?token=wrong",
                                  "/v3/stream?tok=x"])
def test_bad_token_403(server, path):
    assert _upgrade(server, path=path) == 403


def test_token_compared_whole(server):
    assert _upgrade(server, path=f"/v3/stream?token={server.token[:-1]}") == 403
    assert _upgrade(server, path=f"/v3/stream?token={server.token}x") == 403


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1", "127.0.0.1:1", "localhost",
                                  "attacker.test:{port}", "0.0.0.0:{port}"])
def test_bad_host_403(server, host):
    assert _upgrade(server, host=host.format(port=server.port)) == 403


@pytest.mark.parametrize("origin", ["https://evil.example", "http://127.0.0.1.evil.example",
                                    "http://localhost:8888/path", "http://user@localhost",
                                    "file://", "vscode-webview://", ""])
def test_bad_origin_403(server, origin):
    assert _upgrade(server, origin=origin) == 403


def test_other_paths_404(server):
    h = {"Host": f"127.0.0.1:{server.port}"}
    assert _http(server.port, "/v3/other", h)[0] == 404
    assert _http(server.port, "/../etc/passwd", h)[0] == 404


@pytest.mark.parametrize("method", ["POST", "HEAD", "PUT", "OPTIONS"])
def test_only_get(server, method):
    h = {"Host": f"127.0.0.1:{server.port}", "Content-Length": "0"}
    st, hdrs, _ = _http(server.port, "/", h, method)
    assert st == 405 and hdrs["allow"] == "GET"


def test_producer_refused_4002(server):
    async def go():
        async with connect(server.stream_url, max_size=None) as ws:
            await ws.send(w.encode(w.Hello("producer", session={
                "id": "clp-1-abc", "script": "x", "pid": 1, "start_unix_ms": 1})))
            m = w.decode(await ws.recv())
            assert isinstance(m, w.Error) and m.fatal
            with pytest.raises(ConnectionClosed):
                await ws.recv()
            return ws.close_code
    assert run(go()) == w.CLOSE_ROLE_REFUSED


def test_wrong_major_4001(server):
    async def go():
        async with connect(server.stream_url, max_size=None) as ws:
            b = bytearray(w.encode(w.Hello("viewer")))
            b[4] = 4
            await ws.send(bytes(b))
            with pytest.raises(ConnectionClosed):
                await ws.recv()
            return ws.close_code
    assert run(go()) == w.CLOSE_UNSUPPORTED_MAJOR


def test_text_message_1003(server):
    async def go():
        async with Client(server) as c:
            await c.ws.send("hello")
            with pytest.raises(ConnectionClosed):
                while True:
                    await c.ws.recv()
            return c.ws.close_code
    assert run(go()) == w.CLOSE_UNSUPPORTED_DATA


def test_first_message_must_be_hello(server):
    async def go():
        async with connect(server.stream_url, max_size=None) as ws:
            await ws.send(w.encode(w.Ping(1)))
            with pytest.raises(ConnectionClosed):
                while True:
                    await ws.recv()
            return ws.close_code
    assert run(go()) == w.CLOSE_PROTOCOL_ERROR


def test_handshake_timeout_4003(mode, monkeypatch):
    monkeypatch.setattr(w, "HANDSHAKE_TIMEOUT_MS", 300)
    server = caliper.serve()

    async def go():
        async with connect(server.stream_url, max_size=None) as ws:
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), 5)
            return ws.close_code
    try:
        assert run(go()) == w.CLOSE_HANDSHAKE_TIMEOUT
    finally:
        server.close()


def test_idle_ping_then_4004(mode, monkeypatch):
    monkeypatch.setattr(w, "PING_INTERVAL_MS", 200)
    monkeypatch.setattr(w, "IDLE_TIMEOUT_MS", 600)
    server = caliper.serve()

    async def go():
        async with Client(server) as c:
            ping = await c.expect(w.Ping, timeout=5)
            assert isinstance(ping.nonce, int)
            with pytest.raises(ConnectionClosed):
                while True:
                    await c.ws.recv()
            return c.ws.close_code
    try:
        assert run(go()) == w.CLOSE_IDLE_TIMEOUT
    finally:
        server.close()


def test_handshake_sessions_views_ping(server):
    async def go():
        async with Client(server) as c:
            assert c.welcome.session["id"] == server.session
            assert w.CAP_EVENTS in c.welcome.capabilities
            ss = await c.expect(w.Sessions)
            assert [s["id"] for s in ss.sessions] == [server.session]
            assert ss.sessions[0]["state"] == "live"
            v = await c.expect(w.Views)
            assert v.session == server.session
            await c.send(w.Ping(77))
            assert (await c.expect(w.Pong)).nonce == 77
            await c.send(w.Attach("clp-0-nope"))
            err = await c.expect(w.Error)
            assert err.code == w.ERR_UNKNOWN_SESSION
            await c.send(w.Attach(None))
            v = await c.expect(w.Views)
            assert v.session is None and v.views == []
            assert not server.transport.attached() or _wait(lambda: not server.transport.attached())
    run(go())


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ---------------------------------------------------------------------------
# static files
# ---------------------------------------------------------------------------
def test_viewer_page_without_assets(mode, monkeypatch, tmp_path):
    monkeypatch.setenv("CALIPER_WEB_DIR", str(tmp_path))            # an empty folder
    with caliper.serve() as server:
        st, h, body = _http(server.port, "/", {"Host": f"127.0.0.1:{server.port}"})
        assert st == 200 and h["content-type"].startswith("text/html")
        assert b"not built" in body and h["cache-control"] == "no-store"
        assert _http(server.port, "/caliper-view.js", {"Host": f"127.0.0.1:{server.port}"})[0] == 404


def test_static_files_from_assets(mode, monkeypatch, tmp_path):
    files = {"index.html": b"<!doctype html><p>page", "caliper-view.js": b"export {}",
             "views.wasm": b"\0asm\1\0\0\0", "fonts/a.woff2": b"wOF2"}
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(body)
    (tmp_path / ".hidden").write_bytes(b"secret")
    monkeypatch.setenv("CALIPER_WEB_DIR", str(tmp_path))
    types_ = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8",
              "wasm": "application/wasm", "woff2": "font/woff2"}
    with caliper.serve() as server:
        host = {"Host": f"127.0.0.1:{server.port}"}
        for path, name in [("/", "index.html"), ("/?token=x", "index.html"),
                           ("/caliper-view.js", "caliper-view.js"), ("/views.wasm", "views.wasm"),
                           ("/fonts/a.woff2", "fonts/a.woff2")]:
            st, h, body = _http(server.port, path, host)
            assert st == 200 and body == files[name], path
            assert h["content-type"] == types_[name.rsplit(".", 1)[1]]
            assert h["cache-control"] == "no-store"
            assert "access-control-allow-origin" not in h
        st, h, _ = _http(server.port, "/views.wasm", dict(host, Origin="http://127.0.0.1:8888"))
        assert st == 200 and h["access-control-allow-origin"] == "http://127.0.0.1:8888"
        assert h["vary"] == "Origin"
        st, h, _ = _http(server.port, "/views.wasm", dict(host, Origin="https://evil.example"))
        assert st == 200 and "access-control-allow-origin" not in h
        assert _http(server.port, "/missing.js", host)[0] == 404
        assert _http(server.port, "/.hidden", host)[0] == 404
        assert _http(server.port, "/../x", host)[0] == 404
        assert _http(server.port, "/", {"Host": "evil.example"})[0] == 403


# ---------------------------------------------------------------------------
# the message flow of each kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_heatmap_pooled_to_budget(server, device):
    W = torch.randn(700, 1000, device=device)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            r = await c.subscribe(1, "w", pump=show)
            assert r.data_view == {"source": "data", "index": {}, "reduce": {},
                                   "budget": {"max_extent": 512, "pool": "mean"}}
            s = await c.expect(w.Snapshot, pump=show)
            a = s.data.array()
            assert s.kind == w.Kind.HEATMAP and a.shape == (512, 512) and a.dtype == np.float32
            ref = F.adaptive_avg_pool2d(W.cpu()[None, None], (512, 512))[0, 0].numpy()
            assert np.allclose(a, ref, atol=1e-5)
            assert s.axes == [{"dim": "row", "start": 0, "step": 1, "count": 700, "size": 512},
                              {"dim": "col", "start": 0, "step": 1, "count": 1000, "size": 512}]
            lo, hi = s.stats                     # the whole tensor's range (docs/references/python-api.md)
            full = [pytest.approx(W.min().cpu().item()), pytest.approx(W.max().cpu().item())]
            assert [lo, hi] == full
            assert s.flags == w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX and s.data_view == r.data_view
            # a new budget through VIEW_REQUEST
            await c.send(w.ViewRequest(9, 1, "w", {"budget": {"max_extent": 100, "pool": "max"}}))
            rr = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 9, pump=show)
            assert rr.data_view["budget"] == {"max_extent": 100, "pool": "max"}
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.data.shape == (100, 100))
            ref = F.adaptive_max_pool2d(W.cpu()[None, None], (100, 100))[0, 0].numpy()
            assert np.array_equal(s.data.array(), ref)
            assert s.stats == full
            # a zoom keeps the colour range of the whole tensor, as the engine's slow path does
            await c.send(w.ViewRequest(10, 1, "w", {"index": {"row": [100, 200], "col": [0, 400, 4]}}))
            rr = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 10, pump=show)
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.data_view == rr.data_view)
            assert np.array_equal(s.data.array(), W.cpu().numpy()[100:200, 0:400:4])
            assert s.stats == full
    run(go())


@pytest.mark.parametrize("device", DEVICES)
def test_mosaic_range_leaves_out_the_padding(server, device):
    x = torch.rand(10, 6, 5, device=device) + 2.0          # 10 channels: a 4x3 mosaic, 2 cells NaN
    show = lambda: caliper.show(x, name="mosaic", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "mosaic", pump=show)
            s = await c.expect(w.Snapshot, pump=show)
            a = s.data.array()
            assert np.isnan(a).any() and a.shape == (18, 20)
            assert s.stats == pytest.approx([float(np.nanmin(a)), float(np.nanmax(a))])
    run(go())


@pytest.mark.parametrize("device", DEVICES)
def test_image(server, device):
    img = torch.rand(3, 600, 300, device=device)
    show = lambda: caliper.show(img, name="img", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            r = await c.subscribe(1, "img", pump=show)
            assert r.data_view["budget"] == {"max_extent": 512}
            s = await c.expect(w.Snapshot, pump=show)
            a = s.data.array()
            assert s.kind == w.Kind.IMAGE and a.shape == (512, 300, 4) and a.dtype == np.uint8
            assert (a[..., 3] == 255).all()
    run(go())


@pytest.mark.parametrize("device", DEVICES)
def test_points_sampled_with_their_labels(server, device):
    n = 250_000
    z = torch.randn(n, 3, device=device)
    labels = (torch.arange(n, device=device) % 7).float()
    pts = lambda: caliper.points(z, labels, name="emb", classes=7, rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            r = await c.subscribe(1, "emb", pump=pts)
            assert r.data_view["budget"] == {"max_points": 100_000}
            s = await c.expect(w.Snapshot, pump=pts)
            d, aux = s.data.array(), s.aux.array()
            assert d.shape == (100_000, 3) and aux.shape == (100_000,)
            rows = np.arange(100_000, dtype=np.int64) * n // 100_000
            assert np.array_equal(d, z.cpu().numpy()[rows])
            assert np.array_equal(aux, (rows % 7).astype(np.float32))      # same rows
            assert s.points_dims == 3 and s.label_count == 7
            assert s.stats[2] == pytest.approx(d[:, 0].min()) and s.stats[7] == pytest.approx(d[:, 2].max())
            # the stride sample is stable while N does not change
            s2 = await c.expect(w.Snapshot, pump=pts, pred=lambda m: m.seq > s.seq)
            assert np.array_equal(s2.data.array(), d)
            await c.send(w.ViewRequest(5, 1, "emb", {"budget": {"max_points": 1000}}))
            await c.expect(w.ViewReply, pred=lambda m: m.request_id == 5, pump=pts)
            s3 = await c.expect(w.Snapshot, pump=pts, pred=lambda m: m.data.shape[0] == 1000)
            assert s3.axes == [{"dim": "point", "start": 0, "step": 1, "count": n, "size": 1000}]
    run(go())


@pytest.mark.parametrize("device", DEVICES)
def test_points_2d_compact(server, device):
    z = torch.randn(500, 2, device=device)
    pts = lambda: caliper.points(z, name="p2", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "p2", pump=pts)
            s = await c.expect(w.Snapshot, pump=pts)
            assert s.data.array().shape == (500, 2) and s.aux is None and s.points_dims == 2
            assert np.array_equal(s.data.array(), z.cpu().numpy())
    run(go())


@pytest.mark.parametrize("device", DEVICES)
def test_histogram(server, device):
    x = torch.randn(10_000, device=device)
    hist = lambda: caliper.histogram(x, name="h", bins=32, rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "h", pump=hist)
            s = await c.expect(w.Snapshot, pump=hist)
            a = s.data.array()
            assert s.kind == w.Kind.HISTOGRAM and a.shape == (32,) and a.sum() == 10_000
            assert s.stats[0] == pytest.approx(x.min().cpu().item())
            assert s.axes == []
    run(go())


@pytest.mark.parametrize("device", DEVICES)
def test_series_backfill_for_a_late_subscriber(server, device):
    # values appended before any viewer connected are retained
    for i in range(3000):
        caliper.series(float(i), name="loss")
    vals = torch.arange(3000, 5000, dtype=torch.float32, device=device)
    for v in vals:
        caliper.series(v, name="loss")                    # 0-d device tensors
    caliper.series(vals[:5], name="loss")                  # one call, five values

    async def go():
        async with Client(server) as c:
            await c.expect(w.Views)
            caliper.flush()                                # the last gathered chunk
            r = await c.subscribe(1, "loss")
            assert r.data_view == {"budget": {"max_backfill": _net.SERIES_RETAIN}}
            got = []
            while sum(len(x) for x in got) < 5005:
                m = await c.expect(w.Series)
                assert m.start == sum(len(x) for x in got) and m.view == "loss"
                got.append(m.values.array().copy())
            allv = np.concatenate(got)
            expect = np.concatenate([np.arange(5000), np.arange(3000, 3005)]).astype(np.float32)
            assert np.array_equal(allv, expect) and m.total == 5005
            # then deltas: new values follow at the next index
            caliper.series(42.0, name="loss")
            m = await c.expect(w.Series)
            assert m.start == 5005 and m.values.array().tolist() == [42.0]
            # a second subscriber with a limited backfill
            await c.send(w.Subscribe(2, "loss", 2, None, {"budget": {"max_backfill": 10}}))
            m = await c.expect(w.Series, pred=lambda m: m.sub == 2)
            assert m.start == 5006 - 10 and len(m.values.array()) == 10
    run(go())


# ---------------------------------------------------------------------------
# requests that fail
# ---------------------------------------------------------------------------
def test_bad_requests(server):
    W = torch.randn(32, 32)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            await c.expect(w.Views)
            await c.send(w.Subscribe(1, "nope", 1))
            r = await c.expect(w.ViewReply)
            assert r.error["code"] == w.ERR_UNKNOWN_VIEW
            await c.subscribe(2, "w", pump=show)
            for req in ({"index": {"row": [4, 0]}}, {"reduce": {"row": "mean"}},
                        {"source": "other"}, {"budget": {"max_extent": 0}}, {"tile": "row"}):
                await c.send(w.ViewRequest(7, 2, "w", req))
                r = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 7)
                assert r.error["code"] == w.ERR_BAD_DATA_VIEW, req
            await c.send(w.ViewRequest(8, 99, "w", {}))
            r = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 8)
            assert r.error["code"] == w.ERR_UNKNOWN_SUBSCRIPTION
            await c.send(w.Subscribe(3, "w", 3, None, {"index": {"row": 1}}))
            r = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 3)
            assert r.error["code"] == w.ERR_BAD_DATA_VIEW
            # the subscription that failed does not exist; the good one still delivers
            s = await c.expect(w.Snapshot, pump=show)
            assert s.sub == 2
    run(go())


def test_remove_ends_subscriptions(server):
    W = torch.randn(16, 16)
    show = lambda: caliper.show(W, name="gone", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "gone", pump=show)
            await c.expect(w.Snapshot, pump=show)
            caliper.remove("gone")
            v = await c.expect(w.Views, pred=lambda m: all(x["name"] != "gone" for x in m.views))
            assert v is not None
            assert _wait(lambda: server.stats()["connections"][0]["subscriptions"] == 0)
    run(go())


# ---------------------------------------------------------------------------
# flow control
# ---------------------------------------------------------------------------
def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == "darwin" else r / 1024


@pytest.mark.parametrize("device", DEVICES)
def test_slow_viewer_latest_wins(server, device):
    """A viewer that stops reading: the producer's step rate does not drop,
    its memory does not grow, and the viewer gets a recent snapshot when it
    reads again."""
    W = torch.randn(1024, 1024, device=device)
    show = lambda: caliper.show(W, name="big", rate=0)  # noqa: E731

    def steps(seconds):
        n, end = 0, time.perf_counter() + seconds
        while time.perf_counter() < end:
            W.add_(1e-3)
            show()
            n += 1
        return n / seconds

    async def go():
        async with Client(server, max_queue=1) as c:
            await c.subscribe(1, "big", {"budget": {"max_extent": 1024}}, max_rate=240, pump=show)
            await c.expect(w.Snapshot, pump=show)
            loop = asyncio.get_running_loop()

            async def drain():
                while True:
                    await c.ws.recv()
            reader = asyncio.ensure_future(drain())
            base = await loop.run_in_executor(None, steps, 1.0)
            reader.cancel()
            await asyncio.sleep(0.1)
            sent0 = server.stats()["connections"][0]["snapshots_sent"]
            rss0 = _rss_mb()
            slow = await loop.run_in_executor(None, steps, 2.0)     # the viewer reads nothing
            rss1 = _rss_mb()
            st = server.stats()["connections"][0]
            seq_last = server.stats()["views"]["big"]["seq"]
            last, end = None, time.monotonic() + 5
            while time.monotonic() < end:
                try:
                    m = await c.recv(0.5)
                except asyncio.TimeoutError:
                    break
                if isinstance(m, w.Snapshot):
                    last = m
            return base, slow, rss1 - rss0, st, st["snapshots_sent"] - sent0, seq_last, last
    base, slow, grew, st, sent, seq_last, last = run(go(), 60)
    assert slow > 0.7 * base, (slow, base)
    assert st["superseded"] > 0 and sent < 20      # stalled sends; newer snapshots replaced older
    assert grew < 100, grew                          # MB; nothing queues up
    assert last is not None and last.seq >= seq_last - 3


def test_series_coalesced_never_dropped(server):
    async def go():
        async with Client(server) as c:
            caliper.series(0.0, name="s")
            await c.subscribe(1, "s", max_rate=5)       # at most 5 messages a second
            n = 20_000
            for i in range(1, n):
                caliper.series(float(i), name="s")
            got = []
            while sum(len(x) for x in got) < n:
                m = await c.expect(w.Series)
                got.append(m.values.array().copy())
            assert np.array_equal(np.concatenate(got), np.arange(n, dtype=np.float32))
            assert len(got) < 20                          # coalesced
    run(go())


def test_two_viewers_share_a_group(server):
    W = torch.randn(64, 64)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as a, Client(server) as b:
            await a.subscribe(1, "w", pump=show)
            await b.subscribe(5, "w", pump=show)
            sa = await a.expect(w.Snapshot, pump=show)
            sb = await b.expect(w.Snapshot, pump=show)
            assert (sa.sub, sb.sub) == (1, 5)
            assert server.stats()["views"]["w"]["groups"] == 1
    run(go())


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def test_events_reach_python(server):
    seen, threads = [], []

    @server.on_event
    def handler(ev):
        seen.append(ev)
        threads.append(threading.current_thread().name)

    async def go():
        async with Client(server) as c:
            await c.expect(w.Views)
            await c.send(w.Event("w", "select", {"rows": [1, 5], "cols": [0, 3]}, sub=1, seq=4))
            await c.send(w.Event("w", "pick", {"row": 2, "col": 1, "value": 0.5}))
            await c.send(w.Ping(1))
            await c.expect(w.Pong)
    run(go())
    assert _wait(lambda: len(seen) == 2)
    assert [e["type"] for e in seen] == ["select", "pick"]
    assert seen[0]["detail"] == {"rows": [1, 5], "cols": [0, 3]} and seen[0]["seq"] == 4
    assert set(threads) == {"caliper.events"}
    evs = server.events()
    assert [e["type"] for e in evs] == ["select", "pick"] and server.events() == []
    assert server.remove_event_handler(handler)


def test_event_handler_errors_do_not_stop_delivery(server):
    got = []
    server.on_event(lambda ev: 1 / 0)
    server.on_event(got.append)

    async def go():
        async with Client(server) as c:
            await c.expect(w.Views)
            for i in range(3):
                await c.send(w.Event("v", "hover", {"row": i, "col": 0}))
            await c.send(w.Ping(1))
            await c.expect(w.Pong)
    with pytest.warns(RuntimeWarning):
        run(go())
        assert _wait(lambda: len(got) == 3)


def test_event_queue_is_bounded(server):
    async def go():
        async with Client(server) as c:
            await c.expect(w.Views)
            for i in range(w.MAX_QUEUED_EVENTS + 50):
                await c.send(w.Event("v", "hover", {"step": i}))
            await c.send(w.Ping(1))
            await c.expect(w.Pong)
    run(go())
    evs = server.events()
    assert len(evs) == w.MAX_QUEUED_EVENTS and evs[-1]["detail"] == {"step": w.MAX_QUEUED_EVENTS + 49}


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------
def test_close_with_a_viewer_connected(mode):
    server = caliper.serve()
    port = server.port
    W = torch.randn(32, 32)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "w", pump=show)
            await asyncio.get_running_loop().run_in_executor(None, caliper.close)
            bye = await c.expect(w.Bye)
            assert bye.reason == w.BYE_SHUTDOWN
            with pytest.raises(ConnectionClosed):
                while True:
                    await c.ws.recv()
            return c.ws.close_code
    assert run(go()) == w.CLOSE_GOING_AWAY
    assert server.closed and caliper._server is None
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    assert _wait(lambda: not any(t.name == "caliper.net" and t.is_alive()
                                 for t in threading.enumerate()))
    gc.collect()
    assert caliper.show(W, name="w") is False          # no route left


def test_viewer_disconnect_detaches(server):
    async def go():
        async with Client(server) as c:
            await c.expect(w.Views)
            assert _wait(lambda: server.transport.attached())
    run(go())
    assert _wait(lambda: not server.transport.attached())


# ---------------------------------------------------------------------------
# stress: abrupt viewers, shutdown under load, floods, memory
# ---------------------------------------------------------------------------
DEV = "mps" if MPS else "cpu"


def _raw_viewer(server):
    """A WebSocket viewer on a bare socket (to reset it in the middle of a frame)."""
    import base64
    s = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /v3/stream?token={server.token} HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(1)
    assert b" 101 " in head.split(b"\r\n")[0]
    return s


def _raw_send(s, payload: bytes) -> None:
    import struct as st
    mask = os.urandom(4)
    n = len(payload)
    if n < 126:
        hdr = bytes((0x82, 0x80 | n))
    elif n < 65536:
        hdr = bytes((0x82, 0x80 | 126)) + st.pack("!H", n)
    else:
        hdr = bytes((0x82, 0x80 | 127)) + st.pack("!Q", n)
    m = np.frombuffer(mask * (n // 4 + 1), np.uint8)[:n]
    s.sendall(hdr + mask + (np.frombuffer(payload, np.uint8) ^ m).tobytes())


def test_viewer_resets_in_the_middle_of_a_frame(server):
    W = torch.randn(2048, 2048, device=DEV)
    show = lambda: caliper.show(W, name="big", rate=0)  # noqa: E731
    s = _raw_viewer(server)
    _raw_send(s, w.encode(w.Hello("viewer")))
    end = time.monotonic() + 10
    while "big" not in server.stats()["views"] and time.monotonic() < end:
        show()
        time.sleep(0.01)
    _raw_send(s, w.encode(w.Subscribe(1, "big", 1, 240.0, {"budget": {"max_extent": 2048}})))
    got = 0
    while got < 6 << 20:                    # into the middle of a 16 MB snapshot frame
        show()
        got += len(s.recv(1 << 20))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct_linger_zero())
    s.close()                               # RST, mid-frame
    for _ in range(50):
        show()
    assert _wait(lambda: not server.stats()["connections"])
    small = torch.randn(64, 64, device=DEV)

    async def go():                         # the server serves the next viewer as before
        async with Client(server) as c:
            await c.subscribe(1, "small", pump=lambda: caliper.show(small, name="small", rate=0))
            sn = await c.expect(w.Snapshot, pump=lambda: caliper.show(small, name="small", rate=0))
            assert sn.data.array().shape == (64, 64)
    run(go())


def struct_linger_zero() -> bytes:
    import struct as st
    return st.pack("ii", 1, 0)


def test_shutdown_with_frames_queued(mode):
    server = caliper.serve()
    port = server.port
    W = torch.randn(2048, 2048, device=DEV)
    show = lambda: caliper.show(W, name="big", rate=0)  # noqa: E731

    async def go():
        async with Client(server, max_queue=1) as c:
            await c.subscribe(1, "big", {"budget": {"max_extent": 2048}}, max_rate=240, pump=show)
            await c.expect(w.Snapshot, pump=show)
            loop = asyncio.get_running_loop()
            end = time.monotonic() + 1.0
            while time.monotonic() < end:     # the viewer reads nothing: its socket fills
                show()
                await asyncio.sleep(0)
            t0 = time.monotonic()
            await loop.run_in_executor(None, server.close)
            return time.monotonic() - t0
    took = run(go(), 60)
    assert took < 8, took
    assert server.closed and caliper._server is None
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    assert _wait(lambda: not any(t.name == "caliper.net" and t.is_alive()
                                 for t in threading.enumerate()))


def test_flood_of_control_messages(server):
    """A viewer flooding SUBSCRIBE and EVENT: the training thread's calls stay
    fast, the event queue stays bounded, and the connection still answers."""
    W = torch.randn(64, 64, device=DEV)
    caliper.show(W, name="w", rate=0)

    async def go():
        async with Client(server, max_queue=4096) as c:
            await c.subscribe(1, "w", pump=lambda: caliper.show(W, name="w", rate=0))
            frames = [w.encode(w.Subscribe(1 + i % 50, "w", i, 30.0)) for i in range(5_000)]
            frames += [w.encode(w.Event("w", "hover", {"row": i, "col": 0})) for i in range(5_000)]
            loop = asyncio.get_running_loop()
            worst = []

            def calls():
                end = time.monotonic() + 2.0
                while time.monotonic() < end:
                    t0 = time.perf_counter()
                    caliper.show(W, name="w", rate=0)
                    worst.append(time.perf_counter() - t0)
            fut = loop.run_in_executor(None, calls)
            for f in frames:
                await c.ws.send(f)
            await fut
            await c.send(w.Ping(9))
            pong = await c.expect(w.Pong, timeout=60)
            return pong, max(worst), float(np.median(worst))
    pong, worst, med = run(go(), 120)
    assert pong.nonce == 9
    assert med < 0.002, med                  # a call stays cheap during the flood
    assert len(server.events()) <= w.MAX_QUEUED_EVENTS


def _rss_now_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True).stdout
    return int(out.strip()) / 1024


def test_memory_flat_long_run(server):
    W = torch.randn(1024, 1024, device=DEV)
    Z = torch.randn(100_000, 3, device=DEV)
    L = torch.randint(0, 4, (100_000,), device=DEV).float()

    def step(i=[0]):
        i[0] += 1
        W.add_(1e-3)
        caliper.show(W, name="w")
        caliper.points(Z, L, name="z", classes=4)
        caliper.series(float(i[0]), name="loss")

    async def go():
        async with Client(server) as c:
            for v in ("w", "z", "loss"):
                await c.subscribe(len(c.views) + hash(v) % 1000, v, pump=step)
            loop = asyncio.get_running_loop()

            async def drain():
                while True:
                    await c.ws.recv()
            reader = asyncio.ensure_future(drain())

            def run_for(seconds):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    step()
            await loop.run_in_executor(None, run_for, 2.0)          # warm-up
            gc.collect()
            rss0 = _rss_now_mb()
            await loop.run_in_executor(None, run_for, 6.0)
            gc.collect()
            rss1 = _rss_now_mb()
            reader.cancel()
            return rss0, rss1
    rss0, rss1 = run(go(), 90)
    assert rss1 - rss0 < 40, (rss0, rss1)


# ---------------------------------------------------------------------------
# caliper.connect(): the producer's connection to a hub (a fake hub built from
# the codec; the real hub is tested at integration)
# ---------------------------------------------------------------------------
class FakeHub:
    """A hub that accepts producers: WELCOME with an assigned session id
    (the proposal plus "-hub" the first time, the proposal afterwards), then
    records what arrives; tests drive it through call()."""

    def __init__(self, token="hubtoken-0123456789abcdefghij"):
        self.token = token
        self.hellos, self.msgs, self.closes = [], [], []
        self.ws = None
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        assert self.ready.wait(10)

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}/v3/stream?token={self.token}"

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start())
        self.ready.set()
        self.loop.run_forever()

    async def _start(self):
        from websockets.asyncio.server import serve

        async def handler(ws):
            if ws.request.path != f"/v3/stream?token={self.token}":
                await ws.close(4000)
                return
            hello = w.decode(await ws.recv())
            self.hellos.append(hello)
            sid = hello.session["id"]
            if len(self.hellos) == 1:
                sid += "-hub"
            await ws.send(w.encode(w.Welcome("hub", [], {"name": "fake-hub", "version": "0"},
                                             dict(hello.session, id=sid))))
            self.ws = ws
            try:
                async for m in ws:
                    self.msgs.append(w.decode(m))
            except ConnectionClosed:
                pass
            self.closes.append(ws.close_code)

        self.server = await serve(handler, "127.0.0.1", 0, compression=None, max_size=None,
                                  ping_interval=None)
        self.port = self.server.sockets[0].getsockname()[1]

    def call(self, coro, timeout=10):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def send(self, msg):
        async def go():
            await self.ws.send(w.encode(msg))
        self.call(go())

    def drop(self, code=1011):
        async def go():
            await self.ws.close(code)
        self.call(go())

    def wait_for(self, pred, timeout=10.0, pump=None):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for m in list(self.msgs):
                if pred(m):
                    return m
            if pump is not None:
                pump()
            time.sleep(0.01)
        raise TimeoutError(f"got {[type(m).__name__ for m in self.msgs]}")

    def close(self):
        async def go():
            self.server.close()
            await self.server.wait_closed()
        try:
            self.call(go())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


needs_native = pytest.mark.skipif(not NATIVE, reason="caliper.connect() needs caliper._netio")


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1")
    h = FakeHub()
    yield h
    caliper.close()
    h.close()


@needs_native
def test_connect_streams_to_a_hub(hub):
    conn = caliper.connect(hub.url)
    assert conn.wait_connected()
    (hello,) = hub.hellos
    assert hello.role == "producer" and hello.session["pid"] == os.getpid()
    assert conn.session == hello.session["id"] + "-hub"              # the hub's assignment
    W = torch.randn(300, 200, device=DEV)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731
    views = hub.wait_for(lambda m: isinstance(m, w.Views) and any(v["name"] == "w" for v in m.views),
                         pump=show)
    assert views.session == conn.session
    hub.send(w.Subscribe(7, "w", 70))
    reply = hub.wait_for(lambda m: isinstance(m, w.ViewReply) and m.request_id == 70, pump=show)
    assert reply.data_view["budget"]["max_extent"] == 512
    snap = hub.wait_for(lambda m: isinstance(m, w.Snapshot) and m.sub == 7, pump=show)
    assert snap.data.array().shape == (300, 200)
    assert np.allclose(snap.data.array(), W.cpu().numpy())
    for i in range(100):
        caliper.series(float(i), name="loss")
    hub.wait_for(lambda m: isinstance(m, w.Views) and any(v["name"] == "loss" for v in m.views))
    hub.send(w.Subscribe(8, "loss", 80))
    vals = []
    end = time.monotonic() + 10
    while len(vals) < 100 and time.monotonic() < end:
        vals = [x for m in hub.msgs if isinstance(m, w.Series) and m.sub == 8
                for x in m.values.array().tolist()]
        time.sleep(0.02)
    assert vals == [float(i) for i in range(100)]                   # the backfill
    hub.send(w.Event("w", "pick", {"row": 3, "col": 4}, sub=7))
    s = _net.current_server() or _net._producer
    assert _wait(lambda: [e["detail"] for e in s.events()] == [{"row": 3, "col": 4}])
    assert caliper._server is None                                  # nothing listens for viewers


@needs_native
def test_connect_reconnects_and_resumes(hub):
    conn = caliper.connect(hub.url)
    assert conn.wait_connected()
    assigned = conn.session
    hub.drop()
    assert _wait(lambda: len(hub.hellos) == 2, timeout=10)
    assert hub.hellos[1].session["id"] == assigned                  # proposes the assigned id
    assert conn.wait_connected() and conn.stats()["connects"] == 2
    hub.send(w.Bye(w.BYE_REPLACED, "another connection of this session"))
    assert _wait(lambda: conn.state == "replaced", timeout=5)
    time.sleep(1.5)
    assert len(hub.hellos) == 2                                     # no reconnect after replaced


@needs_native
def test_connect_never_blocks_training(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    monkeypatch.setenv("CALIPER_NETIO", "1")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                                       # nothing listens there
    try:
        t0 = time.perf_counter()
        conn = caliper.connect(f"ws://127.0.0.1:{port}/v3/stream?token=" + "x" * 32)
        assert time.perf_counter() - t0 < 0.5
        W = torch.randn(64, 64, device=DEV)
        worst = 0.0
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            t0 = time.perf_counter()
            caliper.show(W, name="w")
            caliper.series(1.0, name="s")
            worst = max(worst, time.perf_counter() - t0)
        assert worst < 0.05
        assert conn.state in ("connecting", "waiting") and conn.stats()["attempts"] >= 2
        assert not conn.connected
    finally:
        caliper.close()


@needs_native
def test_connect_finds_the_hub(hub, monkeypatch, tmp_path):
    f = tmp_path / "hub.json"
    f.write_text(json.dumps({"pid": 1, "port": hub.port, "token": hub.token, "version": 1,
                             "stream_url": f"ws://127.0.0.1:{hub.port}/v3/stream",
                             "url": f"http://127.0.0.1:{hub.port}/?token={hub.token}"}))
    monkeypatch.setenv("CALIPER_HUB_FILE", str(f))
    monkeypatch.delenv("CALIPER_HUB_URL", raising=False)
    assert _net.hub_file() == str(f)
    assert _net.hub_stream_url() == hub.url
    monkeypatch.setenv("CALIPER_HUB_URL", f"http://localhost:{hub.port}/?token=abcdefghijklmnopqrstuvwx")
    assert _net.hub_stream_url() == f"ws://127.0.0.1:{hub.port}/v3/stream?token=abcdefghijklmnopqrstuvwx"
    assert _net.hub_stream_url(hub.url) == hub.url
    monkeypatch.delenv("CALIPER_HUB_URL")
    conn = caliper.connect()                                        # through the hub file
    assert conn.wait_connected()
    monkeypatch.setenv("CALIPER_HUB_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(RuntimeError, match="no hub runs"):
        _net.hub_stream_url()
    with pytest.raises(ValueError):
        _net.hub_stream_url("ws://example.com:7340/v3/stream?token=abc")


@needs_native
def test_serve_and_connect_share_one_producer(hub):
    conn = caliper.connect(hub.url)
    server = caliper.serve()                                        # starts listening on the same producer
    assert conn.wait_connected() and server is _net._producer
    W = torch.randn(64, 64, device=DEV)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731
    hub.wait_for(lambda m: isinstance(m, w.Views) and any(v["name"] == "w" for v in m.views), pump=show)
    hub.send(w.Subscribe(1, "w", 1))

    async def go():
        async with Client(server) as c:
            await c.subscribe(2, "w", pump=show)
            return await c.expect(w.Snapshot, pump=show)
    local = run(go())
    remote = hub.wait_for(lambda m: isinstance(m, w.Snapshot), pump=show)
    assert local.data.array().shape == remote.data.array().shape == (64, 64)
    assert server.stats()["views"]["w"]["groups"] == 1               # computed once for both
    caliper.close()
    assert _wait(lambda: w.CLOSE_GOING_AWAY in hub.closes)
    assert any(isinstance(m, w.Bye) and m.reason == w.BYE_SHUTDOWN for m in hub.msgs)


# ---------------------------------------------------------------------------
# ticks: the views of a step share one event
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_views_share_one_tick(server, device):
    W = torch.randn(256, 256, device=device)
    Z = torch.randn(2000, 2, device=device)
    loss = torch.zeros((), device=device)
    events = []
    if device == "mps":
        real = torch.mps.Event

        class Counted(real):
            def record(self, *a, **k):
                events.append(time.perf_counter())
                return super().record(*a, **k)

    def step():
        W.add_(1e-3)
        time.sleep(0.0005)                               # a training step's length, about
        caliper.show(W, name="w")
        caliper.series(loss + 1, name="loss")
        caliper.points(Z, name="z")

    async def go():
        async with Client(server) as c:
            for i, v in enumerate(("w", "z", "loss")):
                await c.subscribe(i + 1, v, pump=step)
            await c.expect(w.Snapshot, pump=step, pred=lambda m: m.view == "z")
            ring = server.ring
            t0, n0, k0 = time.perf_counter(), ring.counters["taken"], ring.counters["ticks"]
            if device == "mps":
                import caliper._net_ring as R
                R._torch().mps.Event = Counted
            try:
                end = t0 + 1.0
                while time.perf_counter() < end:
                    step()
            finally:
                if device == "mps":
                    R._torch().mps.Event = real
            el = time.perf_counter() - t0
            taken, ticks = ring.counters["taken"] - n0, ring.counters["ticks"] - k0
            assert ring.tick is None                    # the last step's tick closed itself
            seq = server.stats()["views"]["w"]["seq"]
            got = await c.expect(w.Snapshot, pred=lambda m: m.view == "w" and m.seq == seq)
            return taken, ticks, el, got
    taken, ticks, el, got = run(go())
    # w and z share the ticks (a tick lacks one only when the pump has not yet
    # freed its staging buffers, which a loaded machine can cause)
    assert 1.5 * ticks <= taken <= 2 * ticks
    assert ticks <= 60 * el + 2
    if device == "mps":
        assert len(events) <= ticks + 2                  # one event per tick (the series rides in it)


@pytest.mark.parametrize("device", DEVICES)
def test_slower_rates_hold_with_ticks(server, device):
    """A view called with rate=30 alone, and a subscription with max_rate 20
    beside a view at the session's rate, keep to their rates."""
    A = torch.randn(64, 64, device=device)
    B_ = torch.randn(64, 64, device=device)
    alone = [True]

    def step():
        time.sleep(0.0005)
        caliper.show(A, name="a", rate=30)
        if not alone[0]:
            caliper.show(B_, name="b")

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "a", pump=step)
            await c.expect(w.Snapshot, pump=step, pred=lambda m: m.view == "a")
            ring = server.ring
            n0, t0 = ring.views["a"].seq, time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                step()
            rate_a = (ring.views["a"].seq - n0) / (time.perf_counter() - t0)
            alone[0] = False
            await c.subscribe(2, "b", pump=step)
            await c.send(w.Unsubscribe(1))
            await c.subscribe(3, "a", pump=step, max_rate=20)
            await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 3)
            na, nb, t0 = ring.views["a"].seq, ring.views["b"].seq, time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                step()
            el = time.perf_counter() - t0
            return rate_a, (ring.views["a"].seq - na) / el, (ring.views["b"].seq - nb) / el
    rate_a, rate_20, rate_b = run(go())
    assert 22 <= rate_a <= 31, rate_a
    assert 14 <= rate_20 <= 21, rate_20
    assert rate_b >= 45, rate_b


@pytest.mark.parametrize("device", DEVICES)
def test_late_view_resumes_after_a_pause(server, device):
    """A view called after the step's tick has closed is served again after
    the calls pause for longer than a second (an evaluation, say)."""
    A = torch.randn(64, 64, device=device)
    Z = torch.randn(500, 2, device=device)

    def step():
        time.sleep(0.0005)
        caliper.show(A, name="first")
        caliper.points(Z, name="late")

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "first", pump=step)
            await c.subscribe(2, "late", pump=step)
            await c.expect(w.Snapshot, pump=step, pred=lambda m: m.sub == 2)
            time.sleep(1.2)                         # no view calls
            ring = server.ring
            n0, t0 = ring.views["late"].seq, time.perf_counter()
            while time.perf_counter() - t0 < 0.5:
                step()
            return (ring.views["late"].seq - n0) / (time.perf_counter() - t0)
    rate = run(go())
    assert rate >= 45, rate


@pytest.mark.parametrize("device", DEVICES)
def test_calls_spread_in_a_step_share_a_tick(server, device):
    """View calls made far apart in one step (longer than a tick), with a
    series call between them, still share that step's tick."""
    W = torch.randn(64, 64, device=device)
    Z = torch.randn(500, 2, device=device)
    loss = torch.zeros((), device=device)

    def step():
        caliper.show(W, name="w")
        time.sleep(0.02)                                 # the forward and backward pass
        caliper.series(loss + 1, name="loss")
        time.sleep(0.005)
        caliper.points(Z, name="z")

    async def go():
        async with Client(server) as c:
            for i, v in enumerate(("w", "z", "loss")):
                await c.subscribe(i + 1, v, pump=step)
            await c.expect(w.Snapshot, pump=step, pred=lambda m: m.view == "z")
            ring = server.ring
            n0, k0 = ring.counters["taken"], ring.counters["ticks"]
            for _ in range(30):
                step()
            return ring.counters["taken"] - n0, ring.counters["ticks"] - k0
    taken, ticks = run(go())
    assert ticks >= 25 and taken >= 2 * ticks - 2, (taken, ticks)


@pytest.mark.parametrize("device", DEVICES)
def test_view_subscribed_while_a_series_ticks_alone(server, device):
    """A view not called for a while is subscribed while a watched series, called
    first in each step, opens and closes ticks on its own; the view is then
    served from its next steps on. Each step is longer than a tick, so every
    series call opens one: a view called after the series used to be shut out
    of the ring by those ticks for good."""
    H = torch.randn(32, 32, device=device)
    loss = torch.zeros((), device=device)

    def series_only():
        time.sleep(0.02)
        caliper.series(loss + 1, name="loss")

    def step():
        series_only()
        caliper.show(H, name="h")

    async def go():
        async with Client(server) as c:
            step()                                       # both views exist
            await c.subscribe(1, "loss", pump=series_only)
            end = time.perf_counter() + 1.3              # h is not called: it goes stale
            while time.perf_counter() < end:
                series_only()
            await c.subscribe(2, "h", pump=series_only)
            n0 = server.ring.views["h"].seq
            for _ in range(15):
                step()
            return server.ring.views["h"].seq - n0
    served = run(go())
    assert served >= 10, served


# ---------------------------------------------------------------------------
# a viewer that sends and never reads
# ---------------------------------------------------------------------------
def test_control_queue_is_bounded_for_a_peer_that_never_reads(server):
    """A viewer floods requests that each get a reply (ATTACH to an unknown
    session) and PINGs, and reads nothing. The server's control queue for it
    stays within its limits and the connection is closed with 1008 once they
    are reached; training calls keep returning throughout."""
    limit = _net.MAX_CONTROL_FRAMES
    W = torch.randn(64, 64)
    attach = w.encode(w.Attach("no-such-session"))
    ping = w.encode(w.Ping(7))

    def conns():
        return {c["id"]: c for c in server.stats()["connections"]}

    async def go():
        ws = await connect(server.stream_url, max_size=None, compression=None, ping_interval=None,
                           close_timeout=1, max_queue=1)
        await ws.send(w.encode(w.Hello("viewer", software={"name": "flood", "version": "0"})))
        peak, closed = 0, False
        for i in range(4 * limit):
            caliper.show(W, name="w")                 # the training loop goes on
            try:
                await ws.send(ping if i % 100 == 0 else attach)
            except ConnectionClosed:
                closed = True
                break
            if i % 500 == 0:
                await asyncio.sleep(0)                # let the server's own loop run (Python sender)
                cs = conns()
                peak = max([peak] + [c.get("control_frames", 0) for c in cs.values()])
                if server.stats()["producer"].get("overflow_closed", 0):
                    closed = True
                    break
        deadline = time.monotonic() + 20
        while not server.stats()["producer"].get("overflow_closed", 0) and time.monotonic() < deadline:
            caliper.show(W, name="w")
            await asyncio.sleep(0.01)
        n = server.stats()["producer"].get("overflow_closed", 0)
        # the connection goes away on the server's side
        while conns() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        left = len(conns())
        ws.transport.abort()
        return closed, peak, n, left
    closed, peak, n, left = run(go(), timeout=60)
    assert n == 1 and left == 0
    assert peak <= limit
    assert caliper.show(W, name="w") is False         # still returns at once, nothing attached


# ---------------------------------------------------------------------------
# the frame limit
# ---------------------------------------------------------------------------
def test_largest_views_fit_one_frame():
    """What resolution accepts encodes in one frame with a long head: a heatmap
    at the largest payload accepted, points at LIMIT_MAX_POINTS with labels
    and a histogram at MAX_BINS; one cell more than the heatmap does not."""
    from caliper import _dataview as dv
    names = {"index": {"layer" * 12 + str(i): [0, 4096, 1] for i in range(6)}}  # metadata of some size
    rows = w.MAX_PAYLOAD_BYTES // 4 // 4096
    cases = [(w.Kind.HEATMAP, np.zeros((rows, 4096), np.float32), None),
             (w.Kind.POINTS, np.zeros((w.LIMIT_MAX_POINTS, 3), np.float32),
              np.zeros(w.LIMIT_MAX_POINTS, np.float32)),
             (w.Kind.HISTOGRAM, np.zeros(w.MAX_BINS, np.float32), None)]
    for kind, data, aux in cases:
        _, n = w.encode_parts(w.Snapshot(1, "v" * 63, 2**40, 2**60, kind, names, data, aux=aux,
                                         stats=[1e30] * w.MAX_STATS))
        assert n <= w.MAX_FRAME_BYTES, (kind, n)
    view = {"id": 1, "name": "w", "kind": 1, "default": {},
            "sources": [{"name": "data", "dtype": "f32",
                         "dims": [{"name": "row", "size": 4096}, {"name": "col", "size": 4096}]}]}
    with pytest.raises(dv.DataViewError, match="ask for a smaller max_extent, at most 4063 here"):
        dv.resolve(view, {"budget": {"max_extent": 4096}})
    assert dv.resolve(view, {"budget": {"max_extent": 4031}})["budget"]["max_extent"] == 4031


def test_a_view_too_large_for_a_frame_is_refused(server):
    """max_extent 4096 on a 4096 by 4096 weight is refused with bad_data_view,
    and the connection goes on: the same subscription with a smaller budget
    gets its snapshots."""
    W = torch.zeros(4096, 4096)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731

    async def go():
        async with Client(server) as c:
            r = await c.subscribe(1, "w", {"budget": {"max_extent": 4096}}, pump=show)
            assert r.error is not None and r.error["code"] == w.ERR_BAD_DATA_VIEW, r.error
            assert "ask for a smaller max_extent, at most 4063 here" in r.error["message"]
            r = await c.subscribe(1, "w", {"budget": {"max_extent": 1024}}, pump=show)
            assert r.error is None
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.sub == 1)
            return s.data.shape
    assert run(go()) == (1024, 1024)


def test_a_snapshot_over_the_frame_limit_is_not_sent(server, mode):
    """A snapshot that cannot be encoded (only possible past resolution, so it
    is handed to the native sender directly here) is counted and dropped, not
    sent as an empty message, and the connection goes on."""
    if mode != "native":
        pytest.skip("the native sender's publish() is the path past resolution")
    W = torch.randn(32, 32)
    show = lambda: caliper.show(W, name="w", rate=0)  # noqa: E731
    big = np.zeros(w.MAX_FRAME_BYTES // 4 + 1, np.float32)

    async def go():
        async with Client(server) as c:
            await c.subscribe(1, "w", pump=show)
            await c.expect(w.Snapshot, pump=show, pred=lambda m: m.sub == 1)
            v = server.ring.views["w"]
            gid = next(iter(v.groups.values())).__dict__["gid"]
            server.nio.publish(gid, "w", 10_000, 0, int(w.Kind.HEATMAP), "[]", "", 0, 0, 0.0, 1.0, 0, 0,
                               0, -1, int(w.DType.F32), [big.size, 1], big.ctypes.data, big.nbytes, 0, 0)
            # the native thread tries to send it and counts the failure
            deadline = time.monotonic() + 10
            while not server.stats()["producer"]["encode_failed"] and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert server.stats()["producer"]["encode_failed"] == 1
            # the connection goes on: an empty message would have failed to decode here
            await c.expect(w.Snapshot, pump=show, pred=lambda m: m.sub == 1 and m.seq < 10_000)
            assert not any(isinstance(m, w.Snapshot) and m.seq == 10_000 for m in c.inbox)
    run(go())


# ---------------------------------------------------------------------------
# the session token stays out of reprs, errors and logs
# ---------------------------------------------------------------------------
def test_the_token_is_not_in_a_repr_or_an_error(server):
    tok = server.token
    assert tok in server.url                          # serve() prints the link by design
    for text in (repr(server), str(server)):
        assert tok not in text and "token=..." in text
    with pytest.raises(ValueError) as e:              # a link connect() cannot use
        caliper.connect(f"ftp://127.0.0.1:1/v3/stream?token={tok}")
    assert tok not in str(e.value)

    import logging
    import caliper._net_native as NN

    class Stub:                                       # a hub connection whose native error quotes the link
        def upstream(self):
            return {"state": "waiting", "connected": False, "session": "", "url": "",
                    "error": f"not a ws:// URL: wss://127.0.0.1:1/v3/stream?token={tok}", "attempts": 1,
                    "connects": 0}
    hub = NN.HubConnection(types.SimpleNamespace(nio=Stub(), identity={"id": "x"}),
                           f"ws://127.0.0.1:1/v3/stream?token={tok}")
    assert tok not in repr(hub) and tok not in str(hub.stats())

    seen = []

    class Keep(logging.Handler):
        def emit(self, record):
            seen.append(self.format(record))
    h = Keep()
    log = logging.getLogger("caliper.net")
    log.addHandler(h)
    old = log.level
    log.setLevel(logging.DEBUG)
    try:
        log.info("GET %s", f"/v3/stream?token={tok}")
        try:
            raise RuntimeError(f"rejected ws://127.0.0.1:1/v3/stream?token={tok}")
        except RuntimeError:
            log.debug("connection failed", exc_info=True)
    finally:
        log.removeHandler(h)
        log.setLevel(old)
    assert len(seen) == 2 and not any(tok in s for s in seen), seen


# ---------------------------------------------------------------------------
# the native sender's lifetime
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not NATIVE, reason="caliper._netio is not built")
def test_native_close_from_two_threads_joins_once():
    """close() from two threads at once (the script's close() and caliper.close()
    at exit, say) joins the native threads once and returns in both."""
    from caliper import _netio
    for _ in range(20):
        nio = _netio.Server(0, "t" * 32, json.dumps({"id": "clp-1-aaaaaa", "script": "x", "pid": 1,
                                                       "start_unix_ms": 1}), "{}", [], 4, 4096,
                            10_000, 10_000, 30_000)
        barrier = threading.Barrier(3)
        done = []

        def close():
            barrier.wait()
            nio.close()
            done.append(1)
        ts = [threading.Thread(target=close) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        assert len(done) == 3 and not any(t.is_alive() for t in ts)
        del nio                                       # the destructor's close() after both


@pytest.mark.skipif(not NATIVE, reason="caliper._netio is not built")
def test_native_listen_after_close_fails_at_once():
    """listen() on a closed native server raises instead of waiting for a loop
    that has exited."""
    from caliper import _netio
    nio = _netio.Server(-1, "t" * 32, json.dumps({"id": "clp-1-aaaaaa", "script": "x", "pid": 1,
                                                   "start_unix_ms": 1}), "{}", [], 4, 4096,
                        10_000, 10_000, 30_000)
    nio.close()
    out = []

    def listen():
        try:
            nio.listen(0)
            out.append("listened")
        except RuntimeError as e:
            out.append(str(e))
    t = threading.Thread(target=listen, daemon=True)
    t.start()
    t.join(10)
    assert out == ["listen: the server is closed"]
