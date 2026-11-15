"""caliper.serve(): the network route over wire protocol v3 (caliper/_net.py,
_net_ring.py, _net_dataview.py), tested with a websockets client and the codec
caliper/_wire.py. MPS cases skip without an MPS device."""
from __future__ import annotations

import ast
import asyncio
import gc
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


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    s = caliper.serve(port=0)
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
        self.ws = await connect(self.url, max_size=None, compression=None, ping_interval=None,
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


def _http(port, path, headers):
    """A raw HTTP/1.1 GET; returns (status, headers dict, body)."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        lines = [f"GET {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in headers.items()]
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


def test_context_manager(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
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


def test_handshake_timeout_4003(server, monkeypatch):
    monkeypatch.setattr(w, "HANDSHAKE_TIMEOUT_MS", 300)

    async def go():
        async with connect(server.stream_url, max_size=None) as ws:
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), 5)
            return ws.close_code
    assert run(go()) == w.CLOSE_HANDSHAKE_TIMEOUT


def test_idle_ping_then_4004(server, monkeypatch):
    monkeypatch.setattr(w, "PING_INTERVAL_MS", 200)
    monkeypatch.setattr(w, "IDLE_TIMEOUT_MS", 600)

    async def go():
        async with Client(server) as c:
            ping = await c.expect(w.Ping, timeout=5)
            assert isinstance(ping.nonce, int)
            with pytest.raises(ConnectionClosed):
                while True:
                    await c.ws.recv()
            return c.ws.close_code
    assert run(go()) == w.CLOSE_IDLE_TIMEOUT


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
def test_viewer_page_without_assets(server, monkeypatch):
    monkeypatch.setitem(sys.modules, "caliper._assets", None)     # import fails
    st, h, body = _http(server.port, "/", {"Host": f"127.0.0.1:{server.port}"})
    assert st == 200 and h["content-type"].startswith("text/html")
    assert b"not built" in body and h["cache-control"] == "no-store"
    assert _http(server.port, "/caliper-view.js", {"Host": f"127.0.0.1:{server.port}"})[0] == 404


def test_static_files_from_assets(server, monkeypatch):
    files = {"index.html": b"<!doctype html><p>page", "caliper-view.js": b"export {}",
             "views.wasm": b"\0asm\1\0\0\0", "fonts/a.woff2": b"wOF2"}
    types_ = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8",
              "wasm": "application/wasm", "woff2": "font/woff2"}
    fake = types.ModuleType("caliper._assets")
    fake.asset_names = lambda: sorted(files)

    def name_for_url_path(p):
        p = p.split("?", 1)[0]
        if ".." in p:
            raise ValueError(p)
        return p[1:] or "index.html"

    def read_asset(n):
        if n not in files:
            raise FileNotFoundError(n)
        return files[n]
    fake.name_for_url_path = name_for_url_path
    fake.read_asset = read_asset
    fake.content_type = lambda n: types_[n.rsplit(".", 1)[1]]
    monkeypatch.setitem(sys.modules, "caliper._assets", fake)
    monkeypatch.setattr(caliper, "_assets", fake, raising=False)
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
            lo, hi = s.stats
            assert lo == pytest.approx(W.min().cpu().item()) and hi == pytest.approx(W.max().cpu().item())
            assert s.flags == w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX and s.data_view == r.data_view
            # a new budget through VIEW_REQUEST
            await c.send(w.ViewRequest(9, 1, "w", {"budget": {"max_extent": 100, "pool": "max"}}))
            rr = await c.expect(w.ViewReply, pred=lambda m: m.request_id == 9, pump=show)
            assert rr.data_view["budget"] == {"max_extent": 100, "pool": "max"}
            s = await c.expect(w.Snapshot, pump=show, pred=lambda m: m.data.shape == (100, 100))
            ref = F.adaptive_max_pool2d(W.cpu()[None, None], (100, 100))[0, 0].numpy()
            assert np.array_equal(s.data.array(), ref)
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
            for req in ({"index": {"row": [0, 4]}}, {"reduce": {"row": "mean"}},
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
def test_close_with_a_viewer_connected(monkeypatch):
    monkeypatch.setenv("CALIPER_SESSION", "0")
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
