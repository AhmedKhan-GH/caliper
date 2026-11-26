"""Programmable graphics pieces: manifest checks, derive compilation, resource chunking, and relay."""
from __future__ import annotations

from collections import deque
import itertools
import math

from ._programs_types import (
    PIECE_COUNT_MAX,
    PIECE_FIELDS_MAX,
    PIECE_PASSES_MAX,
    PIECE_RESOURCES_MAX,
    PIECE_SIDE_MAX,
    TEXTURE_FORMATS,
    WORKGROUPS_MAX,
    _DTYPES,
    _LinkSet,
    _Program,
    _Refused,
    _Waiting,
    _tensor_dtype_ok,
)

_VERSIONS = itertools.count()   # the counter of versions in parts ("split" "version"), for every link


def install_piece(program_id: str, *, piece: dict, manifest: dict, bind: dict | None = None,
                  params: dict | None = None, panel: str | None = None, segments=None,
                  common: str = "", derive: str = "", draw: str = "") -> dict:
    """Install an instance of a piece, or replace the program with the same id.

    The derive half runs at the pass steps of the shared cadence and publishes its resources as
    RESOURCE messages; the draw half goes to each viewer that announced ``resources``, in a PROGRAM that carries
    all three sections.

    Args:
        program_id: The program's id.
        piece: The pin: ``name``, ``version``, ``hash`` and ``from``.
        manifest: The piece's manifest as JSON (``design/programmable-graphics.md``, Piece
            Manifest).
        bind: Each tensor input's ``live:<name>`` (a buffer input's binding is the viewer's).
        params: A number for each parameter that does not keep its default.
        panel: The scene panel of the draw half.
        segments: The lines of the lib modules and ``common.wgsl`` in ``common``, as PROGRAM
            carries them.
        common: The section ``common``: the lib modules and ``common.wgsl``, imports resolved.
        derive: The section ``derive``: ``derive.wgsl``, imports resolved.
        draw: The section ``draw``: ``draw.wgsl``, imports resolved.

    Returns:
        dict: The program's status item, as :func:`install` returns it, with the kind ``piece``.

    Raises:
        RuntimeError: Installed programs and pending output removals occupy all 16 admission slots.
    """
    from . import _programs as _mod

    with _mod._LOCK:
        _mod._check_output_capacity(program_id)
        _mod._remove(program_id)
        if _mod._ABANDONED:
            _mod._reap()
        if not _mod.ACTIVE:
            _mod._CAD.reset()
            _mod._STATE.update(dense=True, drained=0, short=0, long=0, switches=0,
                               interp=_mod._is_interpretability_run())
            if _mod._STATE["marks"]:
                _mod._STATE["marks"].clear()
        p = _Program(str(program_id), {}, str(derive or ""), None, "")
        p.kind = "piece"
        p.piece = dict(piece or {})
        p.manifest = dict(manifest or {})
        p.bind = dict(bind or {})
        p.panel = panel
        p.segments = segments
        p.common, p.draw = str(common or ""), str(draw or "")
        p.sha256 = str(p.piece.get("hash") or "").removeprefix("sha256:")
        has_derive = isinstance(p.manifest.get("derive"), dict)
        (_mod._PROGS if has_derive else _mod._DRAWN)[p.id] = p
        try:
            _check_piece(p, params)
            if has_derive:
                _mod._compile(p)
            else:
                p.state = "compiled"
        except _Waiting:
            p.state = "waiting"
        except _Refused as r:
            p.state = "refused"
            p.error = r.error
        _mod._set_active()
        item = p.item()
        if p.state != "refused":
            _mod._OUTPUT_EVENTS.append((_mod._relay, (p,)))
    _mod._drain_outputs()
    _mod._notify([item])
    return item


def _check_piece(p: _Program, values) -> None:
    """The facade's part of a piece's manifest: the types that the native call takes. The runtime
    checks every count, size and name again before anything runs."""
    from . import _programs as _mod

    m = p.manifest

    def bad(text, code="piece.bad_manifest"):
        raise _Refused(code, text)

    def whole(v, lo, hi, what):
        if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
            bad(f"{what} is a whole number from {lo} to {hi}.")
        return v

    inputs = m.get("inputs") or []
    if not isinstance(inputs, list) or len(inputs) > _mod.INPUTS_MAX:
        bad(f"A piece has at most {_mod.INPUTS_MAX} inputs.")
    p.inputs, p.types = [], {}
    for x in inputs:
        if not isinstance(x, dict) or not isinstance(x.get("name"), str):
            bad("An input needs a name.")
        kind, dtype = x.get("kind", "tensor"), x.get("dtype", "f32")
        if kind not in ("tensor", "buffer") or not _tensor_dtype_ok(dtype):
            bad(f"The input {x['name']} is a tensor of f32, i32, u32, u8 or u16, or several of them joined "
                f"by |, or a buffer.")
        src = ""
        if kind == "tensor":
            src = p.bind.get(x["name"])
            if src is None and x.get("optional") is True:
                src = ""                             # an optional input left unbound
            elif not isinstance(src, str) or not src.startswith("live:") or len(src) == len("live:"):
                raise _Refused("piece.bad_binding",
                               f"The tensor input {x['name']} needs a binding live:<tensor>.")
        p.inputs.append((x["name"], src))
        p.types[x["name"]] = (kind, dtype)
    declared = m.get("params") or []
    if not isinstance(declared, list) or len(declared) > _mod.PARAMS_MAX:
        bad(f"A piece has at most {_mod.PARAMS_MAX} params.")
    given = dict(values or {})
    p.params, p.param_types = {}, {}
    for x in declared:
        if not isinstance(x, dict) or not isinstance(x.get("name"), str) or \
                x.get("type", "f32") not in _DTYPES:
            bad("A param needs a name and the type f32, i32 or u32.")
        v = given.get(x["name"], x.get("default", 0))
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            v = x.get("default", 0)                  # a binding reference: the app sends its number
        p.params[x["name"]] = float(v)
        p.param_types[x["name"]] = x.get("type", "f32")
    resources = m.get("resources") or []
    if not isinstance(resources, list) or len(resources) > PIECE_RESOURCES_MAX:
        bad(f"A piece has at most {PIECE_RESOURCES_MAX} resources.", "program.bad_resource")
    p.resources = []
    for x in resources:
        if not isinstance(x, dict) or not isinstance(x.get("id"), str) or \
                x.get("kind", "buffer") not in ("buffer", "texture"):
            bad("A resource needs an id and the kind buffer or texture.", "program.bad_resource")
        fields = x.get("fields") or []
        if not isinstance(fields, list) or len(fields) > PIECE_FIELDS_MAX or \
                not all(isinstance(f, str) for f in fields):
            bad(f"The resource {x['id']} has at most {PIECE_FIELDS_MAX} fields name: type.",
                "program.bad_layout")
        texture = x.get("kind") == "texture"
        element = str(x.get("element") or "")
        per_word = {"u8": 4, "u16": 2}.get(element, 1) if not fields else 1   # narrow elements in a word
        fmt = x.get("format")
        if texture and fmt is not None and fmt not in TEXTURE_FORMATS:
            bad(f"The texture {x['id']} has a format other than rgba8unorm, r8unorm or r32uint.",
                "program.bad_resource")
        r = {"id": x["id"], "kind": "texture" if texture else "buffer",
             "element": element, "struct": str(x.get("struct") or ""),
             "fields": list(fields),
             "count": 0 if texture else whole(x.get("count", 0), 1, PIECE_COUNT_MAX * per_word, f"{x['id']}.count"),
             "width": whole(x.get("width", 0), 1, PIECE_SIDE_MAX, f"{x['id']}.width") if texture else 0,
             "height": whole(x.get("height", 0), 1, PIECE_SIDE_MAX, f"{x['id']}.height") if texture else 0,
             "format": (fmt or "rgba8unorm") if texture else "",
             "texels": 0,
             "narrow": element if per_word > 1 else "",
             "counted": bool(x.get("counted", False)), "publish": bool(x.get("publish", True)),
             "clear": bool(x.get("clear", False))}
        if texture and x.get("texels") is not None:
            r["texels"] = whole(x.get("texels"), 1, r["width"] * r["height"], f"{x['id']}.texels")
        p.resources.append(r)
    ids = {r["id"] for r in p.resources}
    d = m.get("derive")
    p.passes, p.written = [], set()
    if not isinstance(d, dict):
        return
    passes = d.get("passes") or []
    if not isinstance(passes, list) or not 1 <= len(passes) <= PIECE_PASSES_MAX:
        bad(f"A derive half has 1 to {PIECE_PASSES_MAX} passes.", "program.bad_pass")
    for i, x in enumerate(passes):
        if not isinstance(x, dict) or not isinstance(x.get("entry"), str):
            bad(f"derive.passes[{i}] needs an entry point.", "program.bad_pass")
        wg = x.get("workgroups", [1, 1, 1])
        if not isinstance(wg, list) or len(wg) != 3:
            bad(f"derive.passes[{i}].workgroups holds three counts.", "program.bad_pass")
        wg = tuple(whole(n, 1, WORKGROUPS_MAX, f"derive.passes[{i}].workgroups") for n in wg)
        writes = x.get("writes") or []
        if not isinstance(writes, list) or not all(isinstance(w, str) and w in ids for w in writes):
            raise _Refused("program.unknown_resource",
                           f"derive.passes[{i}].writes names a resource that the piece lacks.")
        p.passes.append((x["entry"], wg, list(writes)))
        p.written.update(writes)
    if not p.wgsl:
        bad("A piece with a derive half needs the section derive.")
    p.every_min = whole(d.get("every", 1), 1, _mod.EVERY_MAX, "derive.every")
    t = d.get("time_ms", _mod.TIME_MS_DEFAULT)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t <= _mod.TIME_MS_MAX:
        bad(f"derive.time_ms is above 0 and at most {_mod.TIME_MS_MAX:g}.", "program.over_limits")
    p.time_ms = float(t)
    p.memory_mb = whole(d.get("memory_mb", _mod.MEMORY_MB_DEFAULT), 1, _mod.MEMORY_MB_MAX, "derive.memory_mb")


def _compile_piece(p: _Program) -> None:
    """Import the tensor inputs and compile the derive half; raises _Refused or _Waiting."""
    from . import _programs as _mod

    tensors = _mod._resolve_inputs([(name, src) for name, src in p.inputs
                                    if p.types[name][0] == "tensor" and src])
    bound, inputs, unbound = {}, [], []
    for name, src in p.inputs:
        kind, dtype = p.types[name]
        if kind != "tensor" or not src:
            inputs.append((name, 0, dtype))          # the place of a buffer input, or an unbound optional one
            if kind == "tensor":
                unbound.append(name)
            continue
        tname = src[len("live:"):]
        bound[name] = _mod._import(tname, tensors[name])
        inputs.append((name, bound[name].handle, dtype))
    rt = _mod._runtime()
    resources = [(r["id"], r["kind"], r["element"], r["struct"], r["fields"], r["count"], r["width"],
                  r["height"], r["counted"], r["publish"], r["clear"], r["format"], r["texels"])
                 for r in p.resources]
    params = [(k, p.param_types[k], v) for k, v in p.params.items()]
    old = p.handle
    try:
        h = rt.derive(p.id, p.wgsl, p.common, inputs, resources, p.passes, params, time_ms=p.time_ms,
                      memory_bytes=p.memory_mb << 20, piece_hash=p.piece.get("hash") or None, unbound=unbound)
    except RuntimeError as e:
        raise _mod._refusal(e) from e
    if old is not None:
        rt.remove(old)
    p.handle = h
    p.bound = bound
    p.sent_static = False                            # new resources: the static ones go again
    _mod._drop_outbox(p)                             # and no part of an earlier install
    p.static_frames = {}
    p.static_relayed = _LinkSet()
    st = rt.status(h)
    p.memory_bytes = int(st.get("memory_bytes") or 0)
    if p.state != "running":
        p.state = "compiled"
    p.error = None


def _resource_frames(rt, p: _Program) -> list:
    """The RESOURCE frames of each published resource that a pass of the last run wrote, and of
    each static one after the first run, with a copy of its valid bytes: a version of at most
    PART_BYTES as one encoded frame, a larger one as its parts, which :func:`_post` encodes when
    it sends them. No copy is made when every viewer that would get the pass is busy with the
    parts of an earlier pass, which it would skip anyway.

    Raises _Refused with ``program.resource_too_large`` when a version holds more than
    the texture or buffer byte limit, or the versions of the run more than PUBLISHED_BYTES_MAX together:
    the app's check refuses such a manifest, thus only a manifest that skipped it gets here."""
    from . import _programs as _mod, _wire

    first = not p.sent_static
    if not first and _mod._all_busy(p):
        p.skipped += 1
        return []
    frames, total = [], 0
    for r in p.resources:
        if not r["publish"] or (r["id"] not in p.written and not first):
            continue
        data, count, stride, vw, vh, elements = rt.resource(p.handle, r["id"])
        total += len(data)
        limit = _mod.PUBLISHED_TEXTURE_MAX if r["kind"] == "texture" else _mod.PUBLISHED_RESOURCE_MAX
        if len(data) > limit or total > _mod.PUBLISHED_BYTES_MAX:
            raise _Refused("program.resource_too_large",
                           f"The published resources of {p.id} hold more than {limit} bytes "
                           f"in one resource or {_mod.PUBLISHED_BYTES_MAX} bytes together at a pass, more than a "
                           "viewer takes, thus no version could be sent. The program stopped.")
        # A texture's version carries its size (none for a version without texels) and its format, absent
        # for rgba8unorm as before; a buffer of narrow elements carries the element and their count.
        texture = r["kind"] == "texture"
        width, height = (int(vw), int(vh)) if texture and count else (None, None)
        fmt = r.get("format") if texture and count and r.get("format", "rgba8unorm") != "rgba8unorm" else None
        narrow = r.get("narrow") or None
        msg = _wire.Resource(p.id, r["id"], int(p.last_step or 0), int(count), int(stride), memoryview(data),
                             width=width, height=height, format=fmt, element=narrow,
                             elements=int(elements) if narrow else None)
        parts = _wire.resource_parts(msg, next(_VERSIONS), _mod.PART_BYTES)
        out = [_wire.encode(msg)] if len(parts) == 1 else parts
        if r["id"] not in p.written:
            p.static_frames[r["id"]] = out
        frames.extend(out)
    p.sent_static = True
    return frames


def _size(frame) -> int:
    """The bytes of an encoded frame, or of the frame of a RESOURCE part that :func:`_post` encodes."""
    from . import _wire
    return len(frame) if isinstance(frame, (bytes, bytearray, memoryview)) else _wire.encode_parts(frame)[1]


def _chunks(frames: list) -> list:
    """The frames of a pass in chunks: in order, each chunk at most RESOURCE_BACKLOG_BYTES, a chunk
    as ``(bytes, frames)``. A frame holds at most PART_BYTES of a version and a header of a few hundred
    bytes, thus every frame fits a chunk, and a pass of at most PART_BYTES is one chunk."""
    from . import _programs as _mod

    out, cur, n = [], [], 0
    for f in frames:
        size = _size(f)
        if cur and n + size > _mod.RESOURCE_BACKLOG_BYTES:
            out.append((n, cur))
            cur, n = [], 0
        cur.append(f)
        n += size
    if cur:
        out.append((n, cur))
    return out


def _post(ctl, ln, chunk) -> None:
    from . import _wire
    for f in chunk[1]:
        ctl.post(ln, f if isinstance(f, (bytes, bytearray, memoryview)) else _wire.encode(f), _wire.CAP_RESOURCES)


def _all_busy(p: _Program) -> bool:
    """Whether some viewer would get the pass of ``p`` and every such viewer waits for the chunks of
    an earlier pass."""
    from . import _programs as _mod

    links = [ln for ctl in _mod._controls_ports() for ln in _mod._resource_links(ctl)]
    with _mod._OUTBOX_LOCK:
        return bool(links) and all(id(ln) in _mod._OUTBOX for ln in links)


def _offer(p: _Program, ctl, ln, frames: list) -> bool:
    """Give the frames of a pass of ``p`` to viewer ``ln``: the first chunk now, the others to its
    outbox. False, and nothing sent, while the viewer waits for an earlier pass or its queue would pass
    RESOURCE_BACKLOG_BYTES with the first chunk."""
    from . import _programs as _mod

    chunks = _chunks(frames)
    with _mod._OUTBOX_LOCK:
        if id(ln) in _mod._OUTBOX:
            return False
        if not chunks or _backlog(ctl, ln) + chunks[0][0] > _mod.RESOURCE_BACKLOG_BYTES:
            return False
        _post(ctl, ln, chunks[0])
        if len(chunks) > 1:
            _mod._OUTBOX[id(ln)] = [ln, ctl, p, deque(chunks[1:])]
    return True


def _drop_outbox(p: _Program) -> None:
    """Forget the chunks of ``p`` that wait for a viewer: the program is gone or installed again."""
    from . import _programs as _mod

    with _mod._OUTBOX_LOCK:
        for key in [k for k, ob in _mod._OUTBOX.items() if ob[2] is p]:
            del _mod._OUTBOX[key]


def _pump() -> None:
    """At each step: send each viewer the chunks of its outbox that now fit its queue, and forget an
    outbox whose viewer or program is gone. The time counts toward the overhead of the passes."""
    from . import _programs as _mod

    if not _mod._OUTBOX:
        return
    t0 = _mod._clock()
    with _mod._OUTBOX_LOCK:
        try:
            for key, (ln, ctl, p, chunks) in list(_mod._OUTBOX.items()):
                if getattr(ln, "closed", True) or not getattr(ln, "attached", False) or \
                        _mod._PROGS.get(p.id) is not p or p.handle is None:
                    del _mod._OUTBOX[key]
                    continue
                while chunks and _backlog(ctl, ln) + chunks[0][0] <= _mod.RESOURCE_BACKLOG_BYTES:
                    _post(ctl, ln, chunks.popleft())
                if not chunks:
                    del _mod._OUTBOX[key]
        except Exception as e:  # the publish path never stops the step
            _mod._OUTBOX.clear()
            _mod._warn_once(("pump",), f"caliper: sending the parts of a resource failed: {e!r}")
    _mod._CAD.extra += _mod._clock() - t0


def _controls_ports() -> list:
    """The controls port of each running server of either sender."""
    try:
        from . import _net_programs
        return [port.controls for port in list(_net_programs._ports)
                if not getattr(port, "closed", False) and getattr(port, "controls", None) is not None]
    except Exception:
        return []


def _resource_links(ctl) -> list:
    from . import _wire
    links = getattr(getattr(getattr(ctl, "server", None), "producer", None), "links", ())
    return [ln for ln in list(links) if _wire.CAP_RESOURCES in getattr(ln, "caps", ())
            and getattr(ln, "attached", False) and not getattr(ln, "closed", True)]


def _relay_frame(p: _Program, op: str = "set") -> bytes:
    """PROGRAM of a piece's instance with the sections ``common``, ``derive`` and ``draw``.

    The pin's hash covers ``derive.wgsl``, so a viewer can check the hash only with that text."""
    from . import _wire
    import numpy as np
    meta = {"request_id": 0, "op": op, "id": p.id}
    sections = []
    if op == "set":
        meta.update({k: v for k, v in (("piece", p.piece), ("manifest", p.manifest), ("bind", p.bind),
                                       ("params", p.params or None), ("panel", p.panel),
                                       ("segments", p.segments)) if v is not None})
        for name, text in ((_wire.SECTION_COMMON, p.common), (_wire.SECTION_DERIVE, p.wgsl),
                           (_wire.SECTION_DRAW, p.draw)):
            if text:
                sections.append(_wire.Section.from_array(
                    name, np.frombuffer(text.encode("utf-8"), dtype=np.uint8)))
    return _wire.encode_frame(_wire.MsgType.PROGRAM, meta, sections)


def _backlog(ctl, ln) -> int:
    """The bytes of the control frames queued for viewer ``ln`` and of the frames posted for it that
    the server's thread has not queued yet."""
    return getattr(ln, "backlog", lambda: 0)() + getattr(ctl, "unsent", lambda _ln: 0)(ln)


def _relay_to(p: _Program, ctl, links) -> None:
    """The draw half and static versions as one pass to each link that has not had them."""
    from . import _wire
    frame = None
    statics = [f for fs in p.static_frames.values() for f in fs]
    for ln in links:
        if ln in p.relayed:
            continue
        if frame is None:
            frame = _relay_frame(p)
        if statics:
            if not _offer(p, ctl, ln, [frame, *statics]):
                continue
            p.static_relayed.add(ln)
        else:
            ctl.post(ln, frame, _wire.CAP_RESOURCES)
        p.relayed.add(ln)
        if any(r["publish"] and r["id"] in p.written for r in p.resources):
            p.behind.add(ln)


def _relay(p: _Program) -> None:
    """Send the draw half to each viewer that announced ``resources``, after an install."""
    from . import _programs as _mod

    try:
        for ctl in _mod._controls_ports():
            _relay_to(p, ctl, _mod._resource_links(ctl))
    except Exception as e:  # the relay never stops the script
        _mod._warn_once(("relay", p.id), f"caliper: relaying the draw half of {p.id} failed: {e!r}")


def _relay_removal(p: _Program) -> None:
    from . import _programs as _mod, _wire

    try:
        for ctl in _mod._controls_ports():
            if _mod._resource_links(ctl):
                ctl.post(None, _relay_frame(p, "remove"), _wire.CAP_RESOURCES)
    except Exception as e:
        _mod._warn_once(("relay", p.id), f"caliper: relaying the removal of {p.id} failed: {e!r}")


def _send_resources(p: _Program, frames: list) -> None:
    """Give the RESOURCE frames of a pass to the viewers."""
    from . import _programs as _mod

    try:
        statics = [f for fs in p.static_frames.values() for f in fs]
        static_ids = {id(f) for f in statics}
        frame_ids = {id(f) for f in frames}
        written = [f for f in frames if id(f) not in static_ids]
        retry = [*frames, *(f for f in statics if id(f) not in frame_ids)]
        needs_written = any(r["publish"] and r["id"] in p.written for r in p.resources)
        for ctl in _mod._controls_ports():
            links = _mod._resource_links(ctl)
            if not links:
                continue
            _relay_to(p, ctl, links)
            for ln in links:
                if ln not in p.relayed:
                    continue                         # its draw half first, at a later pass
                if needs_written and not written:
                    continue
                pending = written if ln in p.static_relayed else retry
                if not pending:
                    continue
                if not _offer(p, ctl, ln, pending):
                    p.skipped += 1
                    p.behind.add(ln)
                    continue
                if statics:
                    p.static_relayed.add(ln)
                p.behind.discard(ln)
    except Exception as e:  # the publish path never stops the step
        _mod._warn_once(("publish", p.id), f"caliper: sending the resources of {p.id} failed: {e!r}")
