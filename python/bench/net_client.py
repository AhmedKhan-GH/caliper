"""The viewer side of bench_net.py: a separate process that connects to a
snapshot stream, subscribes to named views, decodes every frame and records
when it arrived and how many bytes it carried. It never opens a window.

    .venv/bin/python python/bench/net_client.py --url ws://127.0.0.1:PORT/v3/stream?token=T \\
        --views W,loss [--codec wire|standin] [--out FILE] [--seconds 600] [--wire-py FILE]

It prints READY on stdout once every view is subscribed. On SIGTERM, SIGINT,
the end of the stream or after --seconds it writes its records as JSON to --out
(or stdout) and exits:

    {"views": [...], "recv_ns": [...], "view": [...], "type": [...], "nbytes": [...],
     "stamp_ns": [...], "seq": [...], "decode_ns": [...], "errors": n, ...}

One entry per message received after READY. recv_ns is time.monotonic_ns()
right after the frame arrived (mach_absolute_time on macOS, the clock of the
producer's stamp_ns), stamp_ns the producer's stamp carried in the message (-1
when it has none), view an index into "views" (-1 for messages without a view).

--codec wire speaks wire protocol v3 through caliper._wire (or, with --wire-py,
a copy of that module loaded from a file). --codec standin speaks the framing
of bench_net.py's stand-in transport, defined below: the same shape as v3 (a
little-endian header, UTF-8 JSON metadata, payload sections aligned to 8 bytes)
with its own magic and a two-message handshake. The two adapter functions at
the top of this file are the only code that knows either protocol's messages.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import struct
import sys
import time
from typing import NamedTuple


class Msg(NamedTuple):
    type: str           # "snapshot", "series", "views", "view_reply", "ping", ...
    meta: dict          # the message's fields, without its payload sections
    sections: list      # payload sections (memoryviews, or caliper._wire Sections)


# ---------------------------------------------------------------------------
# adapters: the only code that knows the protocols' messages
# ---------------------------------------------------------------------------
def wire_codec(name: str, wire_py: str | None = None):
    """(encode(meta) -> bytes, decode(frame) -> Msg). encode takes {"type": t,
    **fields} for t in hello, subscribe and pong."""
    if name == "standin":
        return standin_encode, standin_decode
    if wire_py:
        spec = importlib.util.spec_from_file_location("caliper_wire_copy", wire_py)
        W = sys.modules["caliper_wire_copy"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(W)
    else:
        from caliper import _wire as W
    out = {"hello": W.Hello, "subscribe": W.Subscribe, "pong": W.Pong}

    def encode(meta, sections=()):
        meta = dict(meta)
        return W.encode(out[meta.pop("type")](**meta))

    def decode(buf):
        m = W.decode(buf)
        t = m.TYPE.name.lower() if hasattr(m, "TYPE") else "unknown"
        fields = vars(m)
        return Msg(t, {k: v for k, v in fields.items() if not isinstance(v, W.Section)},
                   [v for v in fields.values() if isinstance(v, W.Section)])
    return encode, decode


async def handshake(ws, codec_name, codec, views) -> dict:
    """Subscribe to every view with its default data view. v3: HELLO, WELCOME,
    then SUBSCRIBE each view once VIEWS lists it, until VIEW_REPLY confirms all.
    Stand-in: hello, welcome, one subscribe per view. Returns the welcome."""
    encode, decode = codec
    await ws.send(encode({"type": "hello", "role": "viewer",
                          "software": {"name": "bench_net", "version": "1"}}))
    want = {v: i + 1 for i, v in enumerate(views)}         # view -> sub and request_id
    asked, done, welcome = set(), set(), None
    while len(done) < len(want):
        msg = decode(await ws.recv())
        if msg.type == "welcome":
            welcome = msg.meta
            if codec_name == "standin":
                for v in views:
                    await ws.send(encode({"type": "subscribe", "view": v}))
                done = set(want)
        elif msg.type == "views":
            listed = {e["name"] for e in msg.meta["views"]}
            for v, i in want.items():
                if v in listed and v not in asked:
                    asked.add(v)
                    await ws.send(encode({"type": "subscribe", "sub": i, "view": v, "request_id": i}))
        elif msg.type == "view_reply":
            err = msg.meta.get("error")
            if err is None:
                done.add(msg.meta["view"])
            elif err.get("code") in ("unknown_view", "not_attached"):
                asked.discard(msg.meta["view"])            # ask again when VIEWS lists it
            else:
                raise RuntimeError(f"subscribe {msg.meta['view']!r} refused: {err}")
        elif msg.type == "ping":
            await ws.send(encode({"type": "pong", "nonce": msg.meta["nonce"]}))
        elif msg.type in ("error", "bye"):
            raise RuntimeError(f"stream refused the viewer: {msg.meta}")
    return welcome


# ---------------------------------------------------------------------------
# the stand-in codec
# ---------------------------------------------------------------------------
STANDIN_MAGIC = b"CLSB"                     # stand-in bench framing, not the v3 magic
STANDIN_VERSION = 1
_HDR = struct.Struct("<4sHHII")             # magic, version, flags, metadata bytes, sections
_ZEROS = bytes(8)


def _pad8(n: int) -> int:
    return (n + 7) & ~7


def standin_encode(meta: dict, sections=()) -> bytes:
    """Header, then one u64 length per section, the JSON metadata padded to 8
    bytes, then each section padded to 8 bytes."""
    m = json.dumps(meta, separators=(",", ":")).encode()
    secs = [memoryview(s).cast("B") for s in sections]
    parts = [_HDR.pack(STANDIN_MAGIC, STANDIN_VERSION, 0, len(m), len(secs)),
             struct.pack(f"<{len(secs)}Q", *(s.nbytes for s in secs)), m, _ZEROS[:_pad8(len(m)) - len(m)]]
    for s in secs:
        parts += (s, _ZEROS[:_pad8(s.nbytes) - s.nbytes])
    return b"".join(parts)


def standin_decode(buf) -> Msg:
    mv = memoryview(buf)
    magic, ver, _flags, mlen, n = _HDR.unpack_from(mv, 0)
    if magic != STANDIN_MAGIC or ver != STANDIN_VERSION:
        raise ValueError(f"not a stand-in frame (magic {bytes(magic)!r}, version {ver})")
    lens = struct.unpack_from(f"<{n}Q", mv, _HDR.size)
    off = _HDR.size + 8 * n
    meta = json.loads(bytes(mv[off:off + mlen]))
    off += _pad8(mlen)
    sections = []
    for ln in lens:
        if off + ln > len(mv):
            raise ValueError("truncated section")
        sections.append(mv[off:off + ln])
        off += _pad8(ln)
    return Msg(meta.get("type", ""), meta, sections)


# ---------------------------------------------------------------------------
# decoding payloads (what a viewer does before drawing)
# ---------------------------------------------------------------------------
def payload_arrays(msg: Msg):
    """The message's payload sections as numpy arrays (views, no copy)."""
    import numpy as np
    if msg.sections and hasattr(msg.sections[0], "array"):    # caliper._wire Sections
        return [s.array() for s in msg.sections]
    meta = msg.meta
    out = [np.frombuffer(s, dtype="<f4") for s in msg.sections]
    if msg.type == "series":
        if len(out[0]) != meta["n"]:
            raise ValueError(f"series: {len(out[0])} values, metadata says {meta['n']}")
    elif out and meta.get("shape"):
        out[0] = out[0].reshape(meta["shape"])
    return out


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------
async def run(url: str, views: list, codec_name: str, seconds: float, wire_py=None) -> dict:
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    codec = wire_codec(codec_name, wire_py)
    encode, decode = codec
    vidx = {v: i for i, v in enumerate(views)}
    rec = {k: [] for k in ("recv_ns", "view", "type", "nbytes", "stamp_ns", "seq", "decode_ns")}
    res = {"views": views, "codec": codec_name, "errors": 0, "first_error": None, "closed": None,
           "pings": 0}
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    mono = time.monotonic_ns
    # compression=None: v3 servers decline permessage-deflate, and the stand-in must not deflate
    async with connect(url, compression=None, max_size=None, ping_interval=None) as ws:
        res["welcome"] = {k: v for k, v in (await handshake(ws, codec_name, codec, views)).items()
                          if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        print("READY", flush=True)

        async def reader():
            try:
                while True:
                    frame = await ws.recv()
                    t = mono()
                    try:
                        msg = decode(frame)
                        if msg.type in ("snapshot", "series"):
                            payload_arrays(msg)
                        elif msg.type == "ping":
                            res["pings"] += 1
                            await ws.send(encode({"type": "pong", "nonce": msg.meta["nonce"]}))
                    except Exception as e:      # count and keep reading
                        res["errors"] += 1
                        res["first_error"] = res["first_error"] or repr(e)
                        continue
                    meta = msg.meta
                    rec["recv_ns"].append(t)
                    rec["view"].append(vidx.get(meta.get("view"), -1))
                    rec["type"].append(msg.type)
                    rec["nbytes"].append(len(frame))
                    rec["stamp_ns"].append(int(meta.get("stamp_ns", meta.get("t_ns", -1))))
                    rec["seq"].append(int(meta.get("seq", meta.get("start", -1))))
                    rec["decode_ns"].append(mono() - t)
            except ConnectionClosed as e:
                res["closed"] = f"{e.rcvd.code if e.rcvd else None}"
            finally:
                stop.set()

        task = asyncio.create_task(reader())
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
        except TimeoutError:
            pass
        task.cancel()
    res.update(rec)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="ws://127.0.0.1:PORT/v3/stream?token=T")
    ap.add_argument("--views", required=True, help="comma-separated view names to subscribe to")
    ap.add_argument("--codec", choices=("wire", "standin"), default="wire")
    ap.add_argument("--wire-py", help="load the v3 codec from this copy of caliper/_wire.py")
    ap.add_argument("--out", help="write the JSON records here (default: stdout)")
    ap.add_argument("--seconds", type=float, default=600.0, help="exit after this long at most")
    args = ap.parse_args()
    views = [v for v in args.views.split(",") if v]
    res = asyncio.run(run(args.url, views, args.codec, args.seconds, args.wire_py))
    text = json.dumps(res, separators=(",", ":"))
    if args.out:
        tmp = args.out + ".part"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, args.out)
    else:
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
