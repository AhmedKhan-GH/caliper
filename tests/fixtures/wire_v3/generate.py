#!/usr/bin/env python3
"""Generate the golden frames of wire protocol v3 (docs/reference/wire-protocol.md).

Every implementation of the protocol decodes these files in its tests: the
Python codec (python/tests/test_wire.py also checks that this script reproduces
them byte for byte), the native NetSource, the hub and the TypeScript viewer.

  <name>.bin           the exact bytes of one frame (one WebSocket message) of
                       a 3.0 sender, with minor 0 in its header
  <name>.json          what a decoder must find in it: the header fields, the
                       section lengths and offsets, the metadata object and the
                       section values (flattened, row-major)
  v3_1/<name>.bin      the frames that 3.1 adds (minor 1): its new messages and
  v3_1/<name>.json     the keys it adds to 3.0 messages. A 3.0 decoder finds
                       them well formed; their new types are unknown to it.
  invalid/<name>.bin   a frame every decoder must reject
  invalid/expect.json  per invalid file: the reason and the close code
  v3_2/...             the protocol 3.2 draft's scenes subset (gen_v3_2.py)

Run it from anywhere: ``python tests/fixtures/wire_v3/generate.py`` rewrites
the files, ``--check`` only compares them. It needs numpy and no torch.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from caliper import _wire as w  # noqa: E402

import gen_v3_2  # noqa: E402

SESSION = {"id": "clp-4242-a1b2c3", "script": "train.py", "pid": 4242,
           "start_unix_ms": 1763200000000}
PRODUCER_SOFTWARE = {"name": "caliper-python", "version": "0.1.0"}
VIEWER_SOFTWARE = {"name": "caliper-web", "version": "0.1.0"}

# The views of the example session: a heatmap with two sources, a points view
# and a series.
ATTN = {"id": 1, "name": "attn", "kind": int(w.Kind.HEATMAP),
        "sources": [{"name": layer, "dtype": "f32",
                     "dims": [{"name": "head", "size": 4}, {"name": "query", "size": 2},
                              {"name": "key", "size": 3}]} for layer in ("layer0", "layer1")],
        "default": {"index": {"head": 0}}}
EMBEDDING = {"id": 2, "name": "embedding", "kind": int(w.Kind.POINTS),
             "sources": [{"name": "z", "dtype": "f32", "dims": [{"name": "point", "size": 1000}]}],
             "default": {"budget": {"max_points": 4}}}
LOSS = {"id": 3, "name": "loss", "kind": int(w.Kind.SERIES), "sources": []}

STAMP = 81234567890123

# 3.1: item spaces. The embedding's points and the logits' rows are samples of
# the test set (item space "test"); the loss is indexed by training step.
EMBEDDING_ITEMS = dict(EMBEDDING, sources=[
    {"name": "z", "dtype": "f32", "dims": [{"name": "point", "size": 1000, "items": "test"}]}])
LOGITS = {"id": 4, "name": "logits", "kind": int(w.Kind.HEATMAP),
          "sources": [{"name": "logits", "dtype": "f32",
                       "dims": [{"name": "sample", "size": 1000, "items": "test"},
                                {"name": "class", "size": 10}]}]}
LOSS_ITEMS = dict(LOSS, items="step")

# 3.1: the controls of the example script.
LR = {"name": "lr", "type": w.PARAM_NUMBER, "value": 0.001, "pending": 0.01, "min": 1e-05,
      "max": 0.1, "log": True, "help": "learning rate"}
AUGMENT = {"name": "augment", "type": w.PARAM_BOOLEAN, "value": True}
OPTIMIZER = {"name": "optimizer", "type": w.PARAM_CHOICE, "value": "adam",
             "choices": ["adam", "sgd"]}
SAVE = {"name": "save checkpoint", "running": False, "queued": True}


def valid_vectors() -> dict[str, bytes]:
    attn_default = w.resolve_data_view(ATTN, None)
    attn_request = {"source": "layer1", "index": {}, "reduce": {"head": "mean"},
                    "budget": {"max_extent": 256}}
    try:
        w.resolve_data_view(ATTN, {"index": {"head": 9}})
        raise AssertionError("head 9 resolved")
    except w.MessageError as e:
        bad_index = e.message
    emb = w.resolve_data_view(EMBEDDING, {})
    emb_layout = w.payload_layout(EMBEDDING, emb)
    attn_layout = w.payload_layout(ATTN, attn_default)

    heat = np.array([[-1.5, -0.5, 0.0], [0.25, 0.5, 1.0]], dtype=np.float32)
    pts = np.array([[0.0, 1.0], [-2.0, 0.5], [1.5, -1.0], [0.25, 2.0]], dtype=np.float32)
    labels = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float32)
    bbox = [float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()), 0.0, 0.0]

    msgs = {
        "hello": w.Hello(w.ROLE_VIEWER, [], VIEWER_SOFTWARE),
        "hello_producer": w.Hello(w.ROLE_PRODUCER, [w.CAP_EVENTS], PRODUCER_SOFTWARE, SESSION),
        "welcome": w.Welcome(w.ROLE_PRODUCER, [w.CAP_EVENTS], PRODUCER_SOFTWARE, SESSION),
        "bye": w.Bye(w.BYE_SHUTDOWN, "the producer is exiting"),
        "error": w.Error(w.ERR_UNKNOWN_VIEW, "no view 'attn2'", False,
                         in_reply_to=int(w.MsgType.SUBSCRIBE), request_id=5, sub=2, view="attn2"),
        "ping": w.Ping(17),
        "pong": w.Pong(17),
        "sessions": w.Sessions([dict(SESSION, state=w.SESSION_LIVE, views=3),
                                {"id": "clp-4100-0f9e8d", "script": "sweep.py", "pid": 4100,
                                 "start_unix_ms": 1763190000000, "state": w.SESSION_ENDED,
                                 "views": 1}]),
        "attach": w.Attach(SESSION["id"]),
        "views": w.Views([ATTN, EMBEDDING, LOSS], SESSION["id"]),
        "subscribe": w.Subscribe(1, "attn", 7, 30.0),
        "unsubscribe": w.Unsubscribe(1),
        "view_request": w.ViewRequest(8, 1, "attn", attn_request),
        "view_reply": w.ViewReply(8, 1, "attn", w.resolve_data_view(ATTN, attn_request)),
        "view_reply_error": w.ViewReply(9, 1, "attn", error={"code": w.ERR_BAD_DATA_VIEW,
                                                             "message": bad_index}),
        "snapshot": w.Snapshot(1, "attn", 42, STAMP, w.Kind.HEATMAP, attn_default, heat,
                               colormap=w.COLORMAP_RDBU,
                               flags=w.SNAP_AUTO_VMIN | w.SNAP_AUTO_VMAX,
                               stats=[-1.5, 1.0], axes=attn_layout["axes"]),
        "snapshot_points": w.Snapshot(2, "embedding", 7, STAMP + 500, w.Kind.POINTS, emb, pts,
                                      aux=labels, stats=[0.0, 0.0] + bbox,
                                      axes=emb_layout["axes"], points_dims=2, label_count=3),
        "series": w.Series(3, "loss", 0, 3, STAMP + 1000,
                           np.array([2.5, 1.25, 0.75], dtype=np.float32)),
        "event": w.Event("attn", w.EVENT_PICK, {"row": 1, "col": 2, "value": 1.0}, sub=1, seq=42),
        "event_select": w.Event("embedding", w.EVENT_SELECT,
                                {"indices": [0, 2], "row_ids": [0, 500]}, sub=2, seq=7),
    }
    # These are frames of a 3.0 sender, and stay byte for byte what 3.0 wrote
    # whatever the codec's own minor is: minor 0 in the header.
    out = {name: w.encode(m, minor=0) for name, m in msgs.items()}
    # Forward compatibility: a later minor, unknown flags, a nonzero reserved
    # field and an unknown metadata key, on a known and on an unknown type.
    ping = bytearray(w.encode_frame(w.MsgType.PING, {"nonce": 17, "future_key": [1, 2]},
                                    flags=0x8001, minor=9))
    struct.pack_into("<H", ping, 22, 5)
    out["ping_future"] = bytes(ping)
    out["unknown_type"] = w.encode_frame(0x0777, {"future": True}, minor=9)
    return out


def valid_vectors_31() -> dict[str, bytes]:
    """The frames 3.1 adds, written with the codec's minor (1)."""
    hub_software = {"name": "caliper-hub", "version": "0.1.0"}
    msgs = {
        "hello": w.Hello(w.ROLE_VIEWER, [w.CAP_CONTROLS, w.CAP_SELECTION], VIEWER_SOFTWARE),
        "welcome": w.Welcome(w.ROLE_PRODUCER, [w.CAP_CONTROLS, w.CAP_EVENTS, w.CAP_SELECTION],
                             PRODUCER_SOFTWARE, SESSION, control=True),
        "welcome_hub_read_only": w.Welcome(w.ROLE_HUB, [w.CAP_CONTROLS, w.CAP_SELECTION],
                                           hub_software, control=False),
        "views_items": w.Views([EMBEDDING_ITEMS, LOGITS, LOSS_ITEMS], SESSION["id"]),
        "controls": w.Controls({"state": w.RUN_PAUSED, "step": 1200}, [LR, AUGMENT, OPTIMIZER],
                               [SAVE]),
        "controls_none": w.Controls({"state": w.RUN_NONE}, [], []),
        "control_pause": w.Control(11, w.OP_PAUSE),
        "control_step": w.Control(12, w.OP_STEP, count=5),
        "control_set": w.Control(13, w.OP_SET, "lr", 0.5),
        "control_run": w.Control(14, w.OP_RUN, "save checkpoint"),
        "control_reply_set": w.ControlReply(13, w.OP_SET, w.STATUS_OK, "lr",
                                            w.param_value(LR, 0.5)),
        "control_reply_queued": w.ControlReply(14, w.OP_RUN, w.STATUS_QUEUED, "save checkpoint"),
        "control_reply_done": w.ControlReply(14, w.OP_RUN, w.STATUS_OK, "save checkpoint",
                                             message="saved ckpt-1200.pt"),
        "control_reply_error": w.ControlReply(
            15, w.OP_PAUSE, w.STATUS_ERROR,
            error={"code": w.ERR_FORBIDDEN, "message": "this hub does not allow control"}),
        "event_select_items": w.Event("embedding", w.EVENT_SELECT,
                                      {"indices": [0, 2], "row_ids": [0, 500], "space": "test",
                                       "items": [0, 500]}, sub=2, seq=7),
        "selection": w.Selection("test", [0, 500], "embedding",
                                 details={"fields": ["label", "loss"], "items": [0, 500],
                                          "values": [[7, 0.125], [2, 1.5]]}),
        "selection_clear": w.Selection("step", []),
    }
    return {f"v3_1/{name}": w.encode(m) for name, m in msgs.items()}


def _raw(type_: int, meta: bytes, sections: list[bytes] = (), *, major: int = w.MAJOR,
         minor: int = 0) -> bytearray:
    """A frame from literal metadata bytes, laid out by the rules (no checks).
    Minor 0: the invalid frames are those of a 3.0 sender."""
    pos = w.HEADER_BYTES + 4 * len(sections) + len(meta)
    offs = []
    for s in sections:
        pos = (pos + 7) & ~7
        offs.append(pos)
        pos += len(s)
    out = bytearray(pos)
    w.HEADER.pack_into(out, 0, w.MAGIC, major, minor, type_, 0, pos, len(meta), len(sections), 0)
    struct.pack_into(f"<{len(sections)}I", out, w.HEADER_BYTES, *[len(s) for s in sections])
    mo = w.HEADER_BYTES + 4 * len(sections)
    out[mo:mo + len(meta)] = meta
    for s, o in zip(sections, offs):
        out[o:o + len(s)] = s
    return out


def invalid_vectors(valid: dict[str, bytes]) -> dict[str, tuple[bytes, str, int]]:
    ping, series = valid["ping"], valid["series"]
    P, M, B = w.CLOSE_PROTOCOL_ERROR, w.CLOSE_UNSUPPORTED_MAJOR, w.CLOSE_TOO_BIG

    def patch(buf, fmt, off, value):
        b = bytearray(buf)
        struct.pack_into(fmt, b, off, value)
        return bytes(b)

    f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    deep = b'{"a":' + b"[" * 16 + b"]" * 16 + b"}"
    return {
        "short": (ping[:6], "truncated", P),
        "bad_magic": (patch(ping, "<I", 0, 0x584C4143), "bad_magic", P),
        "major_4": (patch(ping, "<H", 4, 4), "unsupported_major", M),
        "major_4_short": (patch(ping, "<H", 4, 4)[:8], "unsupported_major", M),
        "header_only": (ping[:20], "truncated", P),
        "frame_too_big": (patch(ping, "<I", 12, w.MAX_FRAME_BYTES + 8), "too_big", B),
        "meta_too_big": (patch(ping, "<I", 16, w.MAX_META_BYTES + 1), "too_big", B),
        "too_many_sections": (patch(ping, "<H", 20, w.MAX_SECTIONS + 1), "too_big", B),
        "truncated": (series[:-4], "truncated", P),
        "trailing_bytes": (ping + bytes(8), "malformed", P),
        "layout_mismatch": (patch(ping, "<I", 16, len(ping) - w.HEADER_BYTES - 1), "malformed", P),
        "meta_not_json": (bytes(_raw(w.MsgType.PING, b"{nonce:17}")), "malformed", P),
        "meta_not_object": (bytes(_raw(w.MsgType.PING, b"[17]")), "malformed", P),
        "meta_nan": (bytes(_raw(w.MsgType.PING, b'{"nonce":NaN}')), "malformed", P),
        "meta_too_deep": (bytes(_raw(w.MsgType.PING, deep)), "malformed", P),
        "sections_missing": (bytes(_raw(w.MsgType.SERIES, b'{"sub":3}', [f32])), "malformed", P),
        "section_length_mismatch": (bytes(_raw(
            w.MsgType.SERIES,
            b'{"sections":[{"dtype":0,"name":"values","shape":[4]}],"start":0,"stamp_ns":1,'
            b'"sub":3,"total":4,"view":"loss"}', [f32])), "malformed", P),
    }


def _values(sec) -> list:
    if sec.encoding != w.Encoding.RAW:
        return []
    return [x.item() for x in sec.array().reshape(-1)]


def describe(name: str, buf: bytes) -> dict:
    f = w.decode_frame(buf)
    h = w.HEADER.unpack_from(buf, 0)
    count = h[7]
    lens = list(struct.unpack_from(f"<{count}I", buf, w.HEADER_BYTES))
    secs = []
    for s in f.sections:
        secs.append({"name": s.name, "dtype": s.dtype, "shape": list(s.shape),
                     "encoding": s.encoding, "length": s.data.nbytes, "values": _values(s)})
    # Section offsets by the layout rule (decode_frame has checked it).
    pos = w.HEADER_BYTES + 4 * count + h[6]
    for s, ln in zip(secs, lens):
        pos = (pos + 7) & ~7
        s["offset"] = pos
        pos += ln
    try:
        type_name = w.MsgType(f.type).name.lower()
    except ValueError:
        type_name = None
    return {
        "name": name,
        "type_name": type_name,
        "header": {"magic": h[0], "major": h[1], "minor": h[2], "type": h[3], "flags": h[4],
                   "frame_len": h[5], "meta_len": h[6], "section_count": h[7], "reserved": h[8]},
        "section_lengths": lens,
        "meta_offset": w.HEADER_BYTES + 4 * count,
        "meta": f.meta,
        "sections": secs,
    }


def _json(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def intervention_cases():
    """Shared accepted and refused M8 intervention specifications."""
    return [
        {"name": "legacy", "accepted": True, "specification": {"point": "layers.0.mlp", "op": "ablate"}},
        {"name": "kind", "accepted": True, "specification": {"point": "layers.0.mlp", "kind": "ablate"}},
        {"name": "both", "accepted": True, "specification": {"point": "layers.0.mlp", "op": "ablate", "kind": "ablate"}},
        {"name": "add", "accepted": True, "specification": {"point": "layers.0.mlp_out", "kind": "add", "vector": [1.0, -1.0], "coefficient": 0.5, "positions": [0, 31]}},
        {"name": "scale", "accepted": True, "specification": {"point": "layers.0.mlp_out", "kind": "scale", "feature": 0, "multiplier": -2.0, "dictionary": "fixture"}},
        {"name": "clamp", "accepted": True, "specification": {"point": "layers.0.mlp_out", "op": "clamp", "feature": 1, "value": 0.0, "positions": []}},
        {"name": "patch", "accepted": True, "specification": {"point": "layers.0.mlp_out", "kind": "patch", "source_prompt": "A café", "source_position": 1}},
        {"name": "reserved", "accepted": True, "specification": {"point": "layers.0.mlp", "op": "steer"}},
        {"name": "missing discriminator", "accepted": False, "specification": {"point": "layers.0.mlp"}},
        {"name": "conflicting discriminators", "accepted": False, "specification": {"point": "layers.0.mlp", "op": "ablate", "kind": "add"}},
        {"name": "unknown kind", "accepted": False, "specification": {"point": "layers.0.mlp", "kind": "multiply"}},
        {"name": "extra field", "accepted": False, "specification": {"point": "layers.0.mlp", "kind": "ablate", "value": 1}},
        {"name": "bool feature", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "scale", "feature": True, "multiplier": 1}},
        {"name": "missing vector", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "add", "coefficient": 1}},
        {"name": "bool coefficient", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "add", "vector": [1], "coefficient": True}},
        {"name": "empty vector", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "add", "vector": [], "coefficient": 1}},
        {"name": "duplicate positions", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "patch", "source_prompt": "AB", "positions": [0, 0]}},
        {"name": "position out of range", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "patch", "source_prompt": "AB", "positions": [32]}},
        {"name": "bool source position", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "patch", "source_prompt": "AB", "source_position": True}},
        {"name": "empty prompt", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "patch", "source_prompt": ""}},
        {"name": "empty dictionary", "accepted": False, "specification": {"point": "layers.0.mlp_out", "kind": "clamp", "feature": 1, "value": 0, "dictionary": ""}},
        {"name": "bad schedule", "accepted": False, "specification": {"point": "layers.0.mlp", "kind": "ablate", "from_step": 10, "until_step": 10}},
        {"name": "bool schedule", "accepted": False, "specification": {"point": "layers.0.mlp", "kind": "ablate", "from_step": True}},
    ]


def trace_cases():
    """Shared accepted and refused trace.run and trace.read requests."""
    def req(method, **params):
        return {"version": 1, "method": method, "params": params}

    return [
        {"name": "full trace settings", "accepted": True, "request": req(
            "trace.run", prompt="Fact: the capital of the state containing Dallas is",
            max_targets=5000, max_nodes=8192, max_edges=262144, max_n_logits=10,
            desired_logit_prob=0.95, node_threshold=0.7, edge_threshold=0.9,
            batch_size=512, update_interval=4)},
        {"name": "small trace ceilings", "accepted": True, "request": req(
            "trace.run", prompt="AB", max_targets=512, max_nodes=512, max_edges=8192)},
        {"name": "default batch null", "accepted": True, "request": req(
            "trace.run", prompt="AB", batch_size=None)},
        {"name": "boundary thresholds zero and one", "accepted": True, "request": req(
            "trace.run", prompt="AB", desired_logit_prob=0, node_threshold=0.0, edge_threshold=1.0,
            max_n_logits=1, batch_size=1, update_interval=1)},
        {"name": "read full offsets", "accepted": True, "request": req(
            "trace.read", node_offset=8192, edge_offset=262144, node_limit=256, edge_limit=256)},
        {"name": "removed profile field", "accepted": False, "request": req(
            "trace.run", prompt="AB", profile="caliper")},
        {"name": "targets above ceiling", "accepted": False, "request": req(
            "trace.run", prompt="AB", max_targets=5001)},
        {"name": "nodes above ceiling", "accepted": False, "request": req(
            "trace.run", prompt="AB", max_nodes=8193)},
        {"name": "edges above ceiling", "accepted": False, "request": req(
            "trace.run", prompt="AB", max_edges=262145)},
        {"name": "batch above ceiling", "accepted": False, "request": req(
            "trace.run", prompt="AB", batch_size=513)},
        {"name": "batch zero", "accepted": False, "request": req(
            "trace.run", prompt="AB", batch_size=0)},
        {"name": "logits zero", "accepted": False, "request": req(
            "trace.run", prompt="AB", max_n_logits=0)},
        {"name": "logits above ten", "accepted": False, "request": req(
            "trace.run", prompt="AB", max_n_logits=11)},
        {"name": "update interval above limit", "accepted": False, "request": req(
            "trace.run", prompt="AB", update_interval=129)},
        {"name": "threshold below zero", "accepted": False, "request": req(
            "trace.run", prompt="AB", node_threshold=-0.01)},
        {"name": "threshold above one", "accepted": False, "request": req(
            "trace.run", prompt="AB", edge_threshold=1.01)},
        {"name": "bool threshold", "accepted": False, "request": req(
            "trace.run", prompt="AB", desired_logit_prob=True)},
        {"name": "empty prompt", "accepted": False, "request": req("trace.run", prompt="")},
        {"name": "read edge offset above ceiling", "accepted": False, "request": req(
            "trace.read", edge_offset=262145)},
    ]


def render() -> dict[str, bytes]:
    """Every file of the fixture folder, by path relative to it."""
    files = {}
    valid = valid_vectors()
    valid_32 = gen_v3_2.valid_vectors()
    for name, buf in [*valid.items(), *valid_vectors_31().items(), *valid_32.items()]:
        files[f"{name}.bin"] = buf
        description = _json(describe(name, buf))
        if "_bundle" in name:
            # Resource bytes are lookup data. A compact row keeps large zero palettes readable.
            import re
            description = re.sub(rb'"values": \[([^\[\]]*)\]',
                                 lambda match: b'"values": [' + b' '.join(match[1].split()) + b']',
                                 description)
        files[f"{name}.json"] = description
    expect = {}
    for name, (buf, reason, code) in invalid_vectors(valid).items():
        files[f"invalid/{name}.bin"] = buf
        expect[f"{name}.bin"] = {"reason": reason, "close_code": code}
    files["invalid/expect.json"] = _json(expect)
    expect_32 = {}
    for name, (buf, want) in gen_v3_2.invalid_vectors(valid_32).items():
        files[f"v3_2/invalid/{name}.bin"] = buf
        expect_32[f"{name}.bin"] = want
    files["v3_2/invalid/expect.json"] = _json(expect_32)
    files["m8_interventions.json"] = (json.dumps(intervention_cases(), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    files["replication_trace.json"] = (json.dumps(trace_cases(), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return files


def main(argv: list[str]) -> int:
    files = render()
    check = "--check" in argv
    bad = []
    for rel, data in sorted(files.items()):
        p = HERE / rel
        if check:
            if not p.exists() or p.read_bytes() != data:
                bad.append(rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    on_disk = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.suffix in (".bin", ".json")}
    extra = sorted(on_disk - set(files))
    if check and (bad or extra):
        print("differ: " + ", ".join(bad + extra), file=sys.stderr)
        return 1
    if extra:
        print("not generated (remove them): " + ", ".join(extra), file=sys.stderr)
    print(f"{len(files)} files {'match' if check else 'written'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
