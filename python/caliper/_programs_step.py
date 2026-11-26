"""Per-step execution, inference readiness, and storage rebinding for GPU programs."""
from __future__ import annotations

from . import _controls
from ._programs_types import _STOP_TEXT, _Import, _Program, _Refused, _Waiting


def _is_interpretability_run() -> bool:
    """Whether the current run is an interpretability run rather than a training loop."""
    from . import _programs as _mod

    svc = _controls.CONTROLS.interpretation_service
    return svc is not None and (getattr(svc, "owner", None) is not None or _mod._STATE["optimizer"] is None)


def _is_builtin_library_piece(p: _Program) -> bool:
    """Whether ``p`` is an instance of a built-in library piece."""
    return p.kind == "piece" and isinstance(p.piece, dict) and p.piece.get("from") == "library"


def at_step(step: int, *, control_work: bool = False) -> None:
    """Run the programs that are due at ``step`` and publish their outputs."""
    from . import _programs as _mod

    if not _mod.ACTIVE:
        return
    try:
        _at_step(int(step), control_work=control_work)
    except Exception as e:  # a program never stops the training loop
        _mod._warn_once(("step",), f"caliper: the GPU programs failed at step {step}: {e!r}")


def inference_pending() -> bool:
    """Keep hosted boundaries active until changed inputs reach a scheduled pass."""
    from . import _programs as _mod

    with _mod._LOCK:
        links = [(ctl, ln) for ctl in _mod._controls_ports() for ln in _mod._resource_links(ctl)]
        if any(_inference_work(p, links) for p in _mod._PROGS.values()):
            return True
    with _mod._OUTBOX_LOCK:
        return bool(_mod._OUTBOX)


def _inference_work(p: _Program, links, *, ready=False) -> bool:
    from . import _programs as _mod
    from .interp import _publish

    if p.state not in ("waiting", "compiled", "running"):
        return False
    changed = p.output_retry or p.params_dirty or p.act_dirty or any(
        _publish._VERSIONS.get(src[len("live:"):], -1) > p.input_version
        for _, src in p.inputs)
    delivery = [(ctl, ln) for ctl, ln in links
                if p.kind == "piece" and (ln not in p.relayed or ln in p.behind)]
    if not changed and not delivery:
        return False
    try:
        inputs = [(var, src) for var, src in p.inputs if src] if p.kind == "piece" else p.inputs
        _mod._resolve_inputs(inputs)
    except _Waiting:
        return False
    except _Refused:
        return True
    if ready and p.kind == "piece":
        candidates = links if changed else delivery
        if candidates and (any(r["publish"] for r in p.resources) or delivery):
            with _mod._OUTBOX_LOCK:
                return any(id(ln) not in _mod._OUTBOX and _mod._backlog(ctl, ln) <= 0
                           for ctl, ln in candidates)
    return True


def inference_blocked() -> bool:
    """Whether all known publication work must wait for transport capacity."""
    from . import _programs as _mod

    pending = False
    ready = False
    with _mod._LOCK:
        links = [(ctl, ln) for ctl in _mod._controls_ports() for ln in _mod._resource_links(ctl)]
        for p in _mod._PROGS.values():
            if _inference_work(p, links):
                pending = True
                if _inference_work(p, links, ready=True):
                    ready = True
        with _mod._OUTBOX_LOCK:
            for ln, ctl, p, chunks in _mod._OUTBOX.values():
                pending = True
                if getattr(ln, "closed", True) or not getattr(ln, "attached", False) or \
                        _mod._PROGS.get(p.id) is not p or p.handle is None or not chunks or \
                        _mod._backlog(ctl, ln) + chunks[0][0] <= _mod.RESOURCE_BACKLOG_BYTES:
                    ready = True
    return pending and not ready


def _at_step(step: int, *, control_work: bool = False) -> None:
    from . import _programs as _mod

    _mod._pump()
    changed = []
    published = []
    with _mod._LOCK:
        cad = _mod._CAD
        state = _mod._STATE
        interp = _is_interpretability_run()
        inference = getattr(_controls.CONTROLS.interpretation_service, "owner", None) is not None
        if state["interp"] != interp:
            state["interp"] = interp
            cad.reset_mode()
            if state["marks"]:
                state["marks"].clear()
            if interp:
                changed.extend(_mod._clear_over_budget_warnings())
        if _mod._ABANDONED:
            _mod._reap()
        now = _mod._clock()
        marks = None if interp else _mod._marks()
        ring = _mod._VIEWS["ring"]
        last = state["step"]
        state["step"] = step
        if last is not None and step <= last:        # the script numbered its steps again
            cad.restart()
            if marks is not None:
                marks.clear()
            if state["pass"] is not None and state["pass"] != step:
                state["pass"] = None
        elif state["pass"] is not None and step > state["pass"]:
            cad.repass(state["pass"], step)
            state["pass"] = step
        is_pass = state["pass"] == step              # decided at the checkpoint before
        if control_work and not interp:
            # Pauses, actions and interpretation requests inside c.step() are not program work:
            # drop the open cadence cycle so their wall time is not attributed to pass cost D.
            cad.restart()
            if marks is not None:
                marks.clear()
        cad.target = _mod.VIEWS_SHARE if ring is not None else _mod.SHARE
        budget = state.get("pass_budget", True)
        live = [p for p in _mod._PROGS.values() if p.state in ("waiting", "compiled", "running")]
        due = [p for p in live if p.warm <= 0 and _mod._every_ok(p, step)] if is_pass and budget else []
        links = [(ctl, ln) for ctl in _mod._controls_ports() for ln in _mod._resource_links(ctl)] if inference else []
        if inference:
            due = [p for p in due if p.kind != "piece" or _inference_work(p, links, ready=True)]
        for p in live:
            if p.warm > 0:
                p.warm -= 1
        work = ring.gate_cost if ring is not None and is_pass and ring.gate_open else 0.0
        done = [] if interp else _mod._mark(marks, step, now, is_pass)
        if work > 0.0 and done:
            done = [(k, lo, max(lo, hi - work)) for k, lo, hi in done]
        if ring is not None:
            ring.step_tick(step, now)
        ran = []
        prog_t0 = None
        if due:
            if marks is not None:                    # sync point 1: torch's work is complete
                done += marks.wait(step, _mod._clock)
                empty = _mod._clock() - now < _mod.WAIT_EMPTY
                state["short"] = state["short"] + 1 if empty else 0
                state["long"] = 0 if empty else state["long"] + 1
            else:
                _mod._torch_adapter().framework_done()
            if interp:
                prog_t0 = _mod._clock()
            try:
                for p in due:
                    before = p.state
                    try:
                        _refresh(p)
                    except _Waiting:
                        continue
                    except _Refused as r:
                        p.state, p.error, p.warning = "refused", r.error, None
                        changed.append(p)
                        continue
                    if before != p.state:
                        changed.append(p)
                    every = cad.period(p.every_min)
                    res = state["rt"].run([p.handle], int(step), _mod._clock() - p.t0, every)
                    if res.get("ran"):
                        p.runs += 1                  # a compile again starts a new native count
                        p.pass_times.append(now)
                        ran.append(p)
            except RuntimeError as e:                # a timeout or a lost device
                r = _mod._refusal(e)
                changed.extend(_mod._stop_all(r.error["code"], r.error["message"]))
                ran = []
            rt = state["rt"]
            for p in ran:
                st = rt.status(p.handle)
                p.last_step = int(step)
                p.last_pass_ms = st.get("last_pass_ms")
                p.every = cad.period(p.every_min)
                if st.get("state") == "stopped":
                    code = st.get("stop_code") or "program.over_time"
                    p.state = "stopped"
                    p.error = {"code": code, "message": _STOP_TEXT.get(code, code), "line": None,
                               "column": None}
                    p.warning = None
                    changed.append(p)
                else:
                    skipped = p.skipped
                    try:
                        outs = _mod._outputs(rt, p)
                    except _Refused as r:            # versions that no viewer's queue takes
                        p.state, p.error, p.warning = "stopped", r.error, None
                        rt.remove(p.handle)
                        p.handle = None
                        p.bound = {}
                        _mod._release_unused()
                        changed.append(p)
                        continue
                    if p.state != "running":
                        p.state = "running"
                        changed.append(p)
                    published.append((p, outs))
                    from .interp import _publish
                    if p.skipped == skipped:
                        p.input_version = _publish._VERSION
                        p.params_dirty = False
                        p.act_dirty = False
    _mod._publish_pass(published)
    with _mod._LOCK:
        cad = _mod._CAD
        state = _mod._STATE
        taken = 0
        if ring is not None and ring is _mod._VIEWS["ring"] and (is_pass or ring.gate_open):
            taken = ring.close_pass(step, _mod._clock())
            if taken and not interp:
                pub = ring.publish_s
                last = state.get("pub_pass")
                if last is not None and last[0] in cad.view_cost:
                    cad.view_cost[last[0]] += max(0.0, pub - last[1])
                state["pub_pass"] = (step, pub)
                state["view_s"].append(ring.gate_cost)
                if is_pass:
                    cad.view_cost[step] = ring.gate_cost
        if interp and (ran or taken):
            prog_s = (max(0.0, _mod._clock() - prog_t0) if prog_t0 is not None else 0.0) + work + cad.extra
            cad.extra = 0.0
            cad.record_pass_cost(prog_s)
        if is_pass and budget and not ran and not taken:        # nothing was due after all
            cad.unpass(step)
        ring = _mod._VIEWS["ring"]
        state["pass"] = None
        nxt = step + 1
        if cad.due(nxt):                             # the decision for the next step
            if any(p.warm <= 0 and _mod._every_ok(p, nxt) and
                   (not inference or p.kind != "piece" or _inference_work(p, links, ready=True))
                   for p in live) or \
                    (ring is not None and ring.wants_pass(now)):
                cad.passed(nxt)
                state["pass"] = nxt
                state["pass_budget"] = True
                if ring is not None:
                    ring.cadence = _mod._views_status(ring, False)
                    ring.open_pass(nxt)
        elif ring is not None and (_mod._floor_due(ring, now) or _mod._last_single_step()) and \
                ring.wants_pass(now):
            # The floor: a pass of the views alone, which the cadence does not count, because
            # their own work is cheap, or because the run pauses after the next step; the
            # programs keep to N.
            state["pass"] = nxt
            state["pass_budget"] = False
            state["floor"] += 1
            ring.cadence = _mod._views_status(ring, True)
            ring.open_pass(nxt)
        if not interp:
            if not state["dense"]:                   # the loop waits by itself: the clock times it
                done = [(step, now, now)]
            for k, _lo, hi in done:
                cad.completed(k, hi)
            changed.extend(_mod._update_over_budget())
        else:
            changed.extend(_mod._clear_over_budget_warnings())
        _mod._set_active()
        items = [p.item() for p in dict.fromkeys(changed)]
    _mod._notify(items)


def _refresh(p: _Program) -> None:
    """Compile a waiting program. When an input's storage changed, bind the new import without a
    compile if its element count and type are the same, else compile the program again."""
    from . import _programs as _mod

    if p.handle is None:
        _mod._compile(p)
        return
    moved = False
    for var, src in p.inputs:
        im = p.bound.get(var)
        if im is None:
            continue
        t = _mod._resolve(src[len("live:"):])
        if t is None:
            raise _Refused("program.unknown_input", f"The script no longer has the tensor {im.name}.")
        if _mod._storage_key(t) != im.key:
            moved = True
            if not _rebind(p, var, t, im):
                _mod._compile(p)
                break
    if moved:
        _mod._release_unused()


def _rebind(p: _Program, var: str, t, im: _Import) -> bool:
    """Bind the new storage of input ``var`` in the program without a compile; False when the
    layout changed or the runtime refuses, and the program must compile again."""
    from . import _programs as _mod

    key = _mod._storage_key(t)
    if key[2:4] != im.key[2:4]:                      # another element count or type
        return False
    new = _mod._import(im.name, t)
    try:
        _mod._STATE["rt"].rebind(p.handle, im.handle, new.handle)
    except (RuntimeError, AttributeError):
        return False
    p.bound[var] = new
    return True
