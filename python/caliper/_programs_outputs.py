"""Ordered output publishing, draining, and cleanup for GPU programs."""
from __future__ import annotations

import sys

from ._programs_types import _OutputRemoval, _Program


def _drain_outputs(first=None) -> None:
    """Run ordered output callbacks without _LOCK. Concurrent callers leave work for this drain."""
    from . import _programs as _mod

    if first is None:
        with _mod._LOCK:
            if _mod._OUTPUT_DRAINING:
                return
            _mod._OUTPUT_DRAINING = True
    try:
        while True:
            if first is None:
                with _mod._LOCK:
                    if not _mod._OUTPUT_EVENTS:
                        _mod._OUTPUT_DRAINING = False
                        return
                    first = _mod._OUTPUT_EVENTS.popleft()
            callback, args = first
            first = None
            try:
                callback(*args)
            except Exception as e:
                _mod._warn_once(("output", args[0].id),
                                f"caliper: an output callback for {args[0].id} failed: {e!r}")
    except BaseException:
        with _mod._LOCK:
            _mod._OUTPUT_DRAINING = False
        raise


def _check_output_capacity(program_id) -> None:
    """Reserve cleanup capacity before installation under _LOCK. Replacement reuses its slot."""
    from . import _programs as _mod

    if str(program_id) in _mod._PROGS or str(program_id) in _mod._DRAWN:
        return
    removals = sum(isinstance(args[0], _OutputRemoval) for _, args in _mod._OUTPUT_EVENTS)
    if len(_mod._PROGS) + len(_mod._DRAWN) + removals >= _mod._OUTPUT_CONTROL_MAX:
        raise RuntimeError("program.over_limits: installed programs and pending output removals "
                           f"occupy all {_mod._OUTPUT_CONTROL_MAX} slots; retry after removal completes")


def _retire_output(p, clear_output) -> None:
    """Drop queued instance data under _LOCK and retain only the necessary removal address."""
    from . import _programs as _mod

    kept = [event for event in _mod._OUTPUT_EVENTS if event[1][0] is not p]
    _mod._OUTPUT_EVENTS.clear()
    _mod._OUTPUT_EVENTS.extend(kept)
    if p.kind == "piece":
        _mod._drop_outbox(p)
    if clear_output and (p.kind == "piece" or p.runs):
        removal = _OutputRemoval(p)
        kept = [event for event in _mod._OUTPUT_EVENTS
                if not (isinstance(event[1][0], _OutputRemoval)
                        and event[1][0].target() == removal.target())]
        _mod._OUTPUT_EVENTS.clear()
        _mod._OUTPUT_EVENTS.extend(kept)
        _mod._OUTPUT_EVENTS.append((_mod._remove_output, (removal,)))


def _remove_output(p) -> None:
    """Clear an old output outside _LOCK because its listeners can call other services."""
    from . import _programs as _mod

    if p is not None and p.kind == "piece":
        _mod._relay_removal(p)
        return
    if p is None or p.runs == 0:
        return
    try:
        pkg = sys.modules[__package__]
        if p.kind in ("pixels", "tensor"):
            pkg.remove(f"program.{p.id}")
        elif p.scene:
            pkg.scene(p.scene).clear(_mod._entity(p))
    except Exception as e:  # clearing an output never stops the script
        _mod._warn_once(("clear", p.id), f"caliper: clearing the output of program {p.id} failed: {e!r}")


def _outputs(rt, p: _Program):
    """The outputs of the last pass as tensors over the runtime's buffers, without a copy; for a
    piece, the RESOURCE frames of its published resources."""
    from . import _programs as _mod

    if p.kind == "piece":
        return _mod._resource_frames(rt, p)
    import torch
    names = {"pixels": ["pixels"], "tensor": ["out"]}.get(p.kind) or list(p.output.get("arrays") or [])
    out = {}
    for n in names:
        cap, count = rt.output(p.handle, n)
        out[n] = torch.from_dlpack(cap)
        out["count"] = count
    return out


def _entity(p: _Program) -> str:
    return p.output.get("entity") or f"programs/{p.id}"


def _publish_pass(published: list) -> None:
    """Publishes the outputs of one pass, ``(program, outputs)`` pairs: first the programs that a
    viewer skipped at their last pass, so that programs that do not fit in one pass of a viewer
    reach it in turn."""
    from . import _programs as _mod

    for p, outs in sorted(published, key=lambda po: not po[0].behind):
        _mod._publish(p, outs)


def _publish(p: _Program, outs: dict) -> None:
    """Order this pass with removal before its output callbacks run outside _LOCK."""
    from . import _programs as _mod

    first = None
    with _mod._LOCK:
        if _mod._PROGS.get(p.id) is not p:
            return
        pending = [event for event in _mod._OUTPUT_EVENTS if event[0] is _mod._publish_output]
        for event in pending:
            if event[1][0] is p:
                _mod._OUTPUT_EVENTS.remove(event)
                p.output_retry = True
        deferred = _mod._OUTPUT_DRAINING or bool(_mod._OUTPUT_EVENTS)
        if p.kind == "piece" and (p.output_retry or deferred):
            # A coalesced pass can contain the first version of a static resource.
            statics = [frame for frames in p.static_frames.values() for frame in frames]
            identities = {id(frame) for frame in statics}
            outs = statics + [frame for frame in outs if id(frame) not in identities]
        if deferred:
            size = _output_bytes(outs)
            if size > _mod._OUTPUT_BYTES_MAX:
                p.output_retry = True
                p.skipped += 1
                return
            pending = [event for event in _mod._OUTPUT_EVENTS if event[0] is _mod._publish_output]
            total = sum(_output_bytes(event[1][1]) for event in pending)
            while pending and (len(pending) >= _mod._OUTPUT_PENDING_MAX or total + size > _mod._OUTPUT_BYTES_MAX):
                event = pending.pop(0)
                _mod._OUTPUT_EVENTS.remove(event)
                total -= _output_bytes(event[1][1])
                event[1][0].output_retry = True
                event[1][0].skipped += 1
        p.output_retry = False
        event = (_mod._publish_output, (p, outs))
        if deferred:
            _mod._OUTPUT_EVENTS.append(event)
        else:
            # Reserve the sole active callback before another thread can enqueue a pass.
            _mod._OUTPUT_DRAINING = True
            first = event
    _mod._drain_outputs(first)


def _output_bytes(outs) -> int:
    """Count retained storage without a device transfer. Shared tensor storage counts once."""
    from . import _programs as _mod

    if isinstance(outs, list):
        return sum(_mod._size(frame) for frame in outs)
    storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                for value in outs.values() if hasattr(value, "untyped_storage")}
    return sum(storages.values())


def _publish_output(p: _Program, outs: dict) -> None:
    """Give the outputs to the publish path of caliper.scene or caliper.show; a piece's
    resources go to the viewers that announced ``resources``."""
    from . import _programs as _mod

    if p.kind == "piece":
        _mod._send_resources(p, outs)
        return
    pkg = sys.modules[__package__]
    try:
        if p.kind == "pixels":
            pkg.show(outs["pixels"], f"program.{p.id}")
            return
        if p.kind == "tensor":
            pkg.show(outs["out"].reshape([int(n) for n in p.shape]), f"program.{p.id}")
            return
        sc = pkg.scene(p.scene)
        count = int(outs.get("count", 0))
        part = p.output.get("part", "points")
        if part == "lines":
            count -= count % 2
        pos = outs["positions"][:count]
        if sc.space == "2d":
            pos = pos[:, :2]
        style = p.output.get("style") or {}
        kw = {"cmap": style.get("colormap") or "viridis", "vmin": style.get("vmin"),
              "vmax": style.get("vmax")}
        if "colors" in outs:
            kw["colors"] = outs["colors"][:count]
        elif "values" in outs:
            kw["values"] = outs["values"][:count]
        if part == "lines":
            if "colors" in kw or "values" in kw:
                kw["color_per"] = "vertex"
            sc.lines(_mod._entity(p), pos, **kw)
        else:
            if "radii" in outs:
                kw["radii"] = outs["radii"][:count]
            sc.points(_mod._entity(p), pos, **kw)
    except Exception as e:  # the publish path never stops the step
        _mod._warn_once(("publish", p.id), f"caliper: publishing the output of program {p.id} failed: {e!r}")
