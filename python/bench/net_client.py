"""The viewer side of bench_net.py: a separate process that connects to a
snapshot stream, subscribes to named views, decodes every frame and records
when it arrived and how many bytes it carried. It never opens a window.

    .venv/bin/python python/bench/net_client.py --url ws://127.0.0.1:PORT/v3/stream?token=T \\
        --views W,loss [--codec standin|wire] [--out FILE] [--seconds 600]

It prints READY on stdout once it has subscribed. On SIGTERM, SIGINT, the end
of the stream or after --seconds it writes its records as JSON to --out (or
stdout) and exits:

    {"views": [...], "recv_ns": [...], "view": [...], "type": [...], "nbytes": [...],
     "stamp_ns": [...], "seq": [...], "decode_ns": [...], "errors": n, ...}

One entry per message received. recv_ns is time.monotonic_ns() right after
the frame arrived (mach_absolute_time on macOS, the clock bench_net.py stamps
calls with), stamp_ns the producer's stamp carried in the message (-1 when the
message has none), view an index into "views" (-1 for messages without a view).

The stand-in codec below is the stand-in transport's framing, written to the
facts fixed for wire protocol v3 (one binary frame: a little-endian header,
UTF-8 JSON metadata, payload sections aligned to 8 bytes) but with its own
magic. It is not the v3 codec. The two adapter functions at the top of this
file are the only places that know the sprint's codec (caliper._wire) and its
handshake.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import struct
import sys
import time
from typing import NamedTuple


class Msg(NamedTuple):
    type: str
    meta: dict
    sections: list      # memoryviews into the frame


# ---------------------------------------------------------------------------
# adapters: the only code that knows the sprint's wire codec (caliper._wire)
# and its handshake. Fit these two functions when the codec is committed.
# ---------------------------------------------------------------------------
def wire_codec(name: str):
    """(encode(meta, sections=()) -> bytes, decode(frame) -> Msg) for a codec name:
    "standin" (the stand-in transport of bench_net.py) or "wire" (caliper._wire)."""
    if name == "standin":
        return standin_encode, standin_decode
    from caliper import _wire                  # assumed API: decode(buf) -> (meta, sections)

    def decode(buf):
        meta, sections = _wire.decode(buf)
        return Msg(meta.get("type", ""), meta, list(sections))

    def encode(meta, sections=()):
        return _wire.encode(meta, sections)
    return encode, decode


async def handshake(ws, codec, views) -> dict:
    """hello -> welcome, then one subscribe per view (the empty data view, which
    is the view's default). Returns the welcome metadata."""
    encode, decode = codec
    await ws.send(encode({"type": "hello", "role": "viewer", "client": "bench_net"}))
    while True:
        msg = decode(await ws.recv())
        if msg.type == "welcome":
            break
        if msg.type == "error":
            raise RuntimeError(f"stream refused the hello: {msg.meta}")
    for v in views:
        await ws.send(encode({"type": "subscribe", "view": v, "data_view": {}}))
    return msg.meta


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
_DTYPES = {"f32": "<f4", "f16": "<f2", "u8": "u1", "i32": "<i4", "u32": "<u4"}


def payload_arrays(msg: Msg):
    """The message's sections as numpy arrays, checked against its metadata."""
    import numpy as np
    meta = msg.meta
    if msg.type == "series":
        vals = np.frombuffer(msg.sections[0], dtype="<f4")
        if "n" in meta and len(vals) != meta["n"]:
            raise ValueError(f"series: {len(vals)} values, metadata says {meta['n']}")
        return [vals]
    dt = np.dtype(_DTYPES.get(meta.get("dtype", "f32"), "<f4"))
    shape = tuple(meta.get("shape", ()))
    out = []
    for i, s in enumerate(msg.sections):
        a = np.frombuffer(s, dtype=dt if i == 0 else "<f4")
        if i == 0 and shape:
            a = a.reshape(shape)
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------
STAMP_KEYS = ("t_ns", "stamp_ns")


async def run(url: str, views: list, codec_name: str, seconds: float) -> dict:
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    codec = wire_codec(codec_name)
    _, decode = codec
    vidx = {v: i for i, v in enumerate(views)}
    rec = {k: [] for k in ("recv_ns", "view", "type", "nbytes", "stamp_ns", "seq", "decode_ns")}
    res = {"views": views, "codec": codec_name, "errors": 0, "first_error": None, "closed": None}
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    mono = time.monotonic_ns
    # compression=None: the producer must not deflate megabyte float frames
    async with connect(url, compression=None, max_size=None, ping_interval=None) as ws:
        res["welcome"] = await handshake(ws, codec, views)
        print("READY", flush=True)

        async def reader():
            try:
                while True:
                    frame = await ws.recv()
                    t = mono()
                    try:
                        msg = decode(frame)
                        payload_arrays(msg) if msg.type in ("snapshot", "series") else None
                    except Exception as e:      # count and keep reading
                        res["errors"] += 1
                        res["first_error"] = res["first_error"] or repr(e)
                        continue
                    meta = msg.meta
                    stamp = next((meta[k] for k in STAMP_KEYS if k in meta), -1)
                    rec["recv_ns"].append(t)
                    rec["view"].append(vidx.get(meta.get("view"), -1))
                    rec["type"].append(msg.type)
                    rec["nbytes"].append(len(frame))
                    rec["stamp_ns"].append(int(stamp))
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
    ap.add_argument("--codec", choices=("standin", "wire"), default="wire")
    ap.add_argument("--out", help="write the JSON records here (default: stdout)")
    ap.add_argument("--seconds", type=float, default=600.0, help="exit after this long at most")
    args = ap.parse_args()
    views = [v for v in args.views.split(",") if v]
    res = asyncio.run(run(args.url, views, args.codec, args.seconds))
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
