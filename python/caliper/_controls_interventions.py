"""Intervention records, lifecycle service, and boundary application for controls."""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
import numbers
import warnings

from ._controls_spec import (
    ERR_BAD_RANGE,
    ERR_NOT_SUPPORTED,
    ERR_UNKNOWN_POINT,
    IV_ACTIVE,
    IV_ENDED,
    IV_PENDING,
    MAX_INTERVENTIONS,
    MAX_POINTS_LISTED,
    RUN_NONE,
    _INT_EXACT,
    _log,
    _w,
    intervention_key,
)


class _Intervention:
    """One intervention of the watched model, with a stable specification key."""

    __slots__ = ("point", "op", "from_step", "until_step", "since_step", "state", "hook",
                 "clear", "error", "error_code", "spec", "applied_spec", "replace", "report")

    def __init__(self, point: str, op: str, from_step, until_step, spec=None):
        self.point, self.op = point, op
        self.from_step, self.until_step = from_step, until_step
        self.since_step = None
        self.state = IV_PENDING
        self.hook = None
        self.clear = False
        self.error = None
        self.error_code = None
        self.spec = spec or {"point": point, "op": op, "kind": op,
                             "from_step": from_step, "until_step": until_step}
        self.applied_spec = None
        self.replace = False
        self.report = False

    def entry(self) -> dict:
        entry = dict(deepcopy(self.spec), from_step=self.from_step, until_step=self.until_step,
                     since_step=self.since_step, state=self.state)
        if self.op == "ablate":
            entry.pop("kind", None)
        if self.error is not None:
            entry["error"] = self.error
        if self.error_code is not None:
            entry["error_code"] = self.error_code
        return entry


class Interventions:
    """The interventions of ``caliper.watch(model)``."""

    def __init__(self, ctl, model, adapter):
        from ._ablate import InterventionExecutor
        self._ctl = ctl
        self.model = model
        self.adapter = adapter
        self.points: list = []
        self.names: frozenset = frozenset()
        self._shapes_off = None
        self.executor = InterventionExecutor(model, adapter=adapter)

    @property
    def dictionary(self):
        return self.executor.dictionary

    @property
    def suspended(self) -> bool:
        return self.executor.suspended

    def suspend(self):
        return self.executor.suspend()

    def rows(self, indices):
        return self.executor.rows(indices)

    def invalidate_source_cache(self) -> None:
        with self._ctl.lock:
            with self._ctl._outside_lock():
                self.executor.invalidate_cache()

    def configure(self, *, dictionary=None, tokenizer=None, dictionary_id=None,
                  resource_guard=None, backend=None):
        """Bind resources and end prior interventions after validation succeeds."""
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise ValueError("the intervention service is no longer attached")
            records = list(self._ctl._ivs.values())
            with self._ctl._outside_lock():
                self.executor.configure(dictionary=dictionary, tokenizer=tokenizer,
                                        dictionary_id=dictionary_id, resource_guard=resource_guard,
                                        backend=backend)
                for iv in records:
                    if iv.hook is not None:
                        iv.hook()
            if self._ctl._iv_model is not self:
                raise ValueError("the intervention service is no longer attached")
            for iv in records:
                iv.hook, iv.applied_spec = None, None
                if iv.state != IV_ENDED:
                    iv.state = IV_ENDED
                    iv.error = "intervention resources changed"
                    iv.error_code = "intervene.configuration_changed"
            self._ctl._iv_revision += 1
            self._ctl._iv_due = True
            self._ctl._update_busy()
        self._ctl._publish()
        return self

    def set(self, specification) -> None:
        self._request({"set": specification})

    def clear(self, point=None) -> None:
        self._request({"clear_all": True} if point is None else {"clear": point})

    def _request(self, request) -> None:
        from ._intervention_protocol import InterventionError
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise InterventionError("the intervention service is no longer attached")
            refused = self._ctl._intervene(request)
            self._ctl._update_busy()
        if refused is not None:
            raise InterventionError(refused[1], refused[0])

    def specifications(self, *, active_only: bool = True) -> list[dict]:
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                return []
            if active_only:
                return [deepcopy(iv.applied_spec) for iv in self._ordered_records()
                        if iv.state == IV_ACTIVE and iv.hook is not None and iv.applied_spec is not None]
            return [deepcopy(iv.spec) for iv in self._ctl._ivs.values()
                    if iv.state != IV_ENDED and not iv.clear]

    def state_dict(self) -> dict:
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise ValueError("the intervention service is no longer attached")
            return self._state_dict()

    def _state_dict(self) -> dict:
        return {"version": 1, "interventions": [
            {"specification": deepcopy(iv.spec), "state": iv.state,
             "applied_specification": deepcopy(iv.applied_spec),
             "since_step": iv.since_step, "clear": iv.clear,
             "error": iv.error, "error_code": iv.error_code}
            for iv in self._ordered_records()]}

    def _ordered_records(self, records=None) -> list[_Intervention]:
        items = list(self._ctl._ivs.values() if records is None else records.values())
        active = iter(sorted((iv for iv in items if iv.hook is not None),
                             key=lambda iv: self.executor.execution_order(iv.hook)))
        return [next(active) if iv.hook is not None else iv for iv in items]

    def _order_live_records(self) -> None:
        ordered = {intervention_key(iv.spec): iv for iv in self._ordered_records()}
        self._ctl._ivs.clear()
        self._ctl._ivs.update(ordered)

    @contextmanager
    def temporary_state(self, state):
        """Replay active snapshot effects on the run thread without lifecycle changes."""
        from ._ablate import InterventionReplay
        from ._intervention_protocol import InterventionError
        if isinstance(state, list):
            if len(state) > MAX_INTERVENTIONS:
                raise InterventionError("invalid intervention records")
            state = {"version": 1, "interventions": [
                {"specification": spec, "state": IV_ACTIVE, "since_step": 0,
                 "clear": False, "error": None, "error_code": None} for spec in state]}
        if (not isinstance(state, dict) or type(state.get("version")) is not int
                or state["version"] != 1 or set(state) != {"version", "interventions"}):
            raise InterventionError("unsupported intervention state")
        records = state["interventions"]
        if not isinstance(records, list) or len(records) > MAX_INTERVENTIONS:
            raise InterventionError("invalid intervention records")
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise InterventionError("the intervention service is no longer attached")
            hooks = [iv.hook for iv in self._ctl._ivs.values() if iv.hook is not None]
            with self._ctl._outside_lock():
                replay = InterventionReplay(self.executor)
                specifications, keys = [], set()
                for record in records:
                    spec, applied = _record_specifications(record, replay.executor)
                    key = intervention_key(spec)
                    if key in keys:
                        raise InterventionError("duplicate intervention record")
                    keys.add(key)
                    if record["state"] == IV_ACTIVE:
                        spec = dict(applied)
                        spec.pop("from_step", None)
                        spec.pop("until_step", None)
                        specifications.append(spec)
                with replay._scope(hooks, specifications):
                    yield replay

    def load_state_dict(self, state, *, restore_active: bool = False) -> None:
        """Restore queued records, optionally installing saved active hooks."""
        from ._intervention_protocol import InterventionError
        if (not isinstance(state, dict) or type(state.get("version")) is not int
                or state["version"] != 1 or set(state) != {"version", "interventions"}):
            raise InterventionError("unsupported intervention state")
        records = state["interventions"]
        if not isinstance(records, list) or len(records) > MAX_INTERVENTIONS:
            raise InterventionError("invalid intervention records")
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise InterventionError("the intervention service is no longer attached")
            self._restore_records(records, restore_active=restore_active)

    def apply_overrides(self, specifications) -> None:
        """Apply explicit fork overrides before the first forward pass."""
        from ._intervention_protocol import InterventionError
        if not isinstance(specifications, list) or len(specifications) > MAX_INTERVENTIONS:
            raise InterventionError("invalid intervention overrides")
        with self._ctl.lock:
            if self._ctl._iv_model is not self:
                raise InterventionError("the intervention service is no longer attached")
            if not specifications:
                return
            records = {intervention_key(record["specification"]): record
                       for record in self._state_dict()["interventions"]}
            keys = set()
            step = self._ctl._next_step()
            for request in specifications:
                with self._ctl._outside_lock():
                    spec = self.executor.validate(request)
                key = intervention_key(spec)
                if key in keys:
                    raise InterventionError("duplicate intervention override")
                keys.add(key)
                if spec.get("until_step") is not None and spec["until_step"] <= step:
                    raise InterventionError("override until_step is in the past", ERR_BAD_RANGE)
                active = spec.get("from_step") is None or spec["from_step"] <= step
                records[key] = {
                    "specification": spec,
                    "state": IV_ACTIVE if active else IV_PENDING,
                    "applied_specification": deepcopy(spec) if active else None,
                    "since_step": step if active else None,
                    "clear": False,
                    "error": None,
                    "error_code": None,
                }
            while len(records) > MAX_INTERVENTIONS:
                old = next((k for k, rec in records.items() if rec["state"] == IV_ENDED), None)
                if old is None:
                    raise InterventionError("too many live intervention records", _w().ERR_LIMIT)
                del records[old]
            self._restore_records(list(records.values()), restore_active=True, evict_ended=True)
        self._ctl._publish()

    def _restore_records(self, records, *, restore_active: bool, evict_ended: bool = False) -> None:
        from ._intervention_protocol import InterventionError
        restored: dict = {}
        with self._ctl._outside_lock():
            for record in records:
                spec, applied = _record_specifications(record, self.executor)
                key = intervention_key(spec)
                if key in restored:
                    raise InterventionError("duplicate intervention record")
                iv = _Intervention(spec["point"], spec["op"], spec.get("from_step"), spec.get("until_step"), spec)
                iv.state, iv.since_step = record["state"], record["since_step"]
                iv.clear, iv.error, iv.error_code = record["clear"], record["error"], record["error_code"]
                iv.applied_spec = applied
                iv.replace = applied is not None and spec != applied
                restored[key] = iv
        try:
            self._ctl._check_intervention_step(restored, self._ctl._next_step())
        except ValueError as error:
            raise InterventionError(str(error), ERR_BAD_RANGE) from None
        try:
            restored = self._ctl._admit_interventions(restored, evict_ended=evict_ended)
        except ValueError as error:
            raise InterventionError(str(error), _w().ERR_LIMIT) from None
        cache = self.executor._cache.copy()
        try:
            if restore_active:
                with self._ctl._outside_lock():
                    for iv in restored.values():
                        if iv.applied_spec is not None:
                            self._ctl._install_intervention(self, iv, iv.applied_spec)
            old = list(self._ctl._ivs.values())
            self._ctl._ivs = {intervention_key(iv.spec): iv for iv in self._ordered_records(restored)}
            self._ctl._iv_revision += 1
            self._ctl._iv_due = True
            self._ctl.busy = True
        except BaseException:
            with self._ctl._outside_lock():
                for iv in restored.values():
                    self._ctl._unhook(iv, retain_source=True)
                self.executor._cache.clear()
                self.executor._cache.update(cache)
            raise
        with self._ctl._outside_lock():
            for iv in old:
                self._ctl._unhook(iv, retain_source=True)
            self.executor._discard_unused_sources()

    def remove(self) -> None:
        self._ctl._interventions_off(self)


def _record_specifications(record, executor):
    from . import _intervention_protocol as metadata
    from ._intervention_protocol import InterventionError
    fields = {"specification", "state", "since_step", "clear", "error", "error_code"}
    if not isinstance(record, dict) or set(record) not in (fields, fields | {"applied_specification"}):
        raise InterventionError("invalid intervention record")
    state, since = record["state"], record["since_step"]
    if (state not in (IV_PENDING, IV_ACTIVE, IV_ENDED) or type(record["clear"]) is not bool
            or (state == IV_ACTIVE and since is None)
            or (state == IV_PENDING and since is not None)):
        raise InterventionError("invalid intervention lifecycle state")
    spec = executor.validate(record["specification"], resources=False)
    entry = dict(spec, state=state, since_step=since)
    entry.update({key: record[key] for key in ("error", "error_code") if record[key] is not None})
    if not metadata.specification(entry, lifecycle=True):
        raise InterventionError("invalid intervention lifecycle metadata")
    if state != IV_ENDED:
        spec = executor.validate(spec)
    applied = record.get("applied_specification", spec if state == IV_ACTIVE else None)
    if state == IV_ACTIVE:
        applied = executor.validate(applied)
        if intervention_key(applied) != intervention_key(spec):
            raise InterventionError("applied intervention must match the record identity")
    elif applied is not None:
        raise InterventionError("inactive intervention cannot have an applied specification")
    return spec, applied


class _InterventionsMixin:
    """Intervention attachment, boundary application, and linger management for Controls."""

    @property
    def intervention_service(self):
        return self._iv_model

    def interventions_on(self, model, handle=None, adapter=None) -> Interventions:
        if adapter is None:
            from ._adapter import TORCH as adapter
        from . import _ablate
        m = Interventions(self, model, adapter)
        pts = list(adapter.points(model))
        m.points = pts[:MAX_POINTS_LISTED]
        m.names = frozenset(p["point"] for p in pts)
        m.executor._ablation_points = m.names
        old = self._iv_model
        if old is not None:
            self._interventions_off(old)
        try:
            m._shapes_off = _ablate.record_shapes(model)
        except Exception:
            _log.debug("the shape hooks failed", exc_info=True)
        with self.lock:
            self._check_metadata_budget(points=m.points, ivs={})
            self._iv_model = m
            self._iv_revision += 1
            self._iv_due = m._shapes_off is not None
            self._update_busy()
        if handle is not None and hasattr(handle, "_handles"):
            handle._handles.append(m)
        self._publish()
        return m

    def _interventions_off(self, m: Interventions) -> None:
        with self.lock:
            if self._iv_model is not m:
                return
            self._iv_model = None
            ivs, self._ivs = list(self._ivs.values()), {}
            shapes_off, m._shapes_off = m._shapes_off, None
            self._iv_revision += 1
            self._iv_due = False
            self._update_busy()
        for iv in ivs:
            self._unhook(iv)
        m.executor.close()
        m.executor.dictionary = m.executor.tokenizer = m.executor.backend = None
        if shapes_off is not None:
            shapes_off()
        self._publish()

    @staticmethod
    def _unhook(iv: _Intervention, *, retain_source: bool = False) -> None:
        hook, iv.hook = iv.hook, None
        iv.applied_spec = None
        if hook is not None:
            try:
                if retain_source and hasattr(hook, "_remove"):
                    hook._remove(retain_source=True)
                else:
                    hook()
            except Exception:
                _log.debug("removing the hook of %s failed", iv.point, exc_info=True)

    def _install_intervention(self, service, iv, spec, *, owner=None) -> None:
        target = iv if owner is None else owner
        callback = lambda message, iv=target: self._intervention_failed(iv, message)
        iv.hook = service.executor.install(spec, on_error=callback)
        iv.applied_spec = deepcopy(spec)

    def _intervene(self, req):
        m = self._iv_model
        if m is None:
            return (ERR_NOT_SUPPORTED, "the script takes no interventions: it watches no model "
                                       "with caliper.watch(model)")
        if (not isinstance(req, dict) or set(req) - {"set", "clear", "clear_all"}
                or sum(req.get(k) is not None for k in ("set", "clear", "clear_all")) != 1):
            return (_w().ERR_BAD_REQUEST, "intervene holds exactly one of set, clear or clear_all")
        if req.get("set") is not None:
            from ._intervention_protocol import InterventionError
            try:
                with self._outside_lock():
                    spec = m.executor.validate(req["set"])
            except InterventionError as error:
                return error.code, str(error)
            if self._iv_model is not m:
                return (ERR_NOT_SUPPORTED, "the intervention service is no longer attached")
            if getattr(self.interpretation_service, "failure", None) is not None:
                return ("interpret.stopped", "The interpretation run stopped.")
            nxt = self._next_step()
            point, op = spec["point"], spec["op"]
            frm, until = spec.get("from_step"), spec.get("until_step")
            if until is not None and until <= nxt:
                return (ERR_BAD_RANGE, f"until_step {until} is in the past: the next step is {nxt}")
            key = intervention_key(spec)
            iv = self._ivs.get(key)
            candidate = _Intervention(point, op, frm, until, spec)
            if iv is not None and iv.state != IV_ENDED:
                candidate.state, candidate.since_step = iv.state, iv.since_step
                candidate.applied_spec = iv.applied_spec
            try:
                self._check_intervention_step({key: candidate}, nxt)
            except ValueError as error:
                return ERR_BAD_RANGE, str(error)
            try:
                prospective = self._admit_interventions({**self._ivs, key: candidate}, evict_ended=True)
            except ValueError as error:
                return _w().ERR_LIMIT, str(error)
            if iv is not None:
                iv.replace = iv.applied_spec is not None and spec != iv.applied_spec
                iv.spec = spec
                iv.from_step, iv.until_step = frm, until
                iv.clear, iv.error, iv.error_code = False, None, None
                if iv.state == IV_ENDED:
                    iv.state, iv.since_step = IV_PENDING, None
                prospective[key] = iv
            self._ivs.clear()
            self._ivs.update(prospective)
        elif req.get("clear") is not None:
            point = req["clear"]
            if not isinstance(point, str) or point not in m.names:
                return (ERR_UNKNOWN_POINT, f"the model has no point {point!r}")
            for iv in self._ivs.values():
                if iv.point == point:
                    iv.clear = True
        elif req.get("clear_all") is True:
            for iv in self._ivs.values():
                iv.clear = True
        else:
            return (_w().ERR_BAD_REQUEST, "intervene holds set, clear or clear_all")
        self._iv_due = True
        self._iv_revision += 1
        self.counters["interventions_in"] += 1
        return None

    @staticmethod
    def _check_intervention_step(records, step):
        if 0 <= step < _INT_EXACT:
            return
        for iv in records.values():
            if (iv.clear or iv.state == IV_ENDED
                    or (iv.until_step is not None and iv.until_step <= step)
                    or (iv.from_step is not None and iv.from_step > step)):
                continue
            applied_until = (iv.applied_spec or {}).get("until_step")
            if iv.state != IV_ACTIVE or (applied_until is not None and applied_until <= step):
                raise ValueError("intervention since_step must be in 0 .. 2**53 - 1")

    def _next_step(self) -> int:
        n = self.n
        return int(n) + 1 if isinstance(n, numbers.Integral) and not isinstance(n, bool) else 1

    def _controllers_due(self):
        from . import _controls as _mod
        now = _mod.time.monotonic()
        with self.lock:
            if not any(iv.state != IV_ENDED and not iv.clear for iv in self._ivs.values()):
                self._alone_since = None
                return None
            if now < self._linger_next:
                return None
            self._linger_next = now + _mod.LINGER_CHECK_S
            routes = list(self.routes)
        return sum(r._controllers() for r in routes) if routes else 0

    def _linger(self, controllers: int, records=None) -> bool:
        from . import _controls as _mod
        from ._net_programs import LINGER_S
        now = _mod.time.monotonic()
        if controllers > 0:
            self._alone_since = None
            return False
        if self._alone_since is None:
            self._alone_since = now
        if now - self._alone_since < LINGER_S:
            return False
        if records is None:
            self._alone_since = None
        ended, removed = [], []
        for iv in (self._ivs if records is None else records).values():
            if iv.state == IV_ENDED or iv.clear:
                continue
            removed.append(copy(iv))
            iv.hook, iv.applied_spec = None, None
            iv.state = IV_ENDED
            iv.error = self._stop_message(
                f"ended: no connection that may control the run stayed for {LINGER_S:g} s")
            iv.error_code = "intervene.no_controller"
            ended.append(iv.point)
        if ended and records is None:
            self._iv_revision += 1
            with self._outside_lock():
                for iv in removed:
                    self._unhook(iv)
                self._report_linger(ended)
        return bool(ended)

    def _report_linger(self, ended):
        from ._net_programs import LINGER_S
        self.counters["interventions_lingered"] += len(ended)
        warnings.warn(f"caliper: no connection that may control the run stayed for {LINGER_S:g} s, "
                      f"so the interventions on {', '.join(ended)} ended", RuntimeWarning,
                      stacklevel=2)

    def _intervention_failed(self, iv, message):
        with self.lock:
            if self._ivs.get(intervention_key(iv.spec)) is not iv:
                return
            removed = copy(iv)
            iv.hook, iv.applied_spec = None, None
            iv.state, iv.error = IV_ENDED, self._stop_message(message)
            iv.error_code, iv.report = "intervene.hook_failed", True
            self._iv_revision += 1
            self._iv_due = True
            self._update_busy()
            with self._outside_lock():
                self._unhook(removed)

    def _apply_interventions(self, controllers=None) -> bool:
        changed, ready = self._intervention_boundary(controllers)
        if not ready:
            self._iv_boundary_failures += 1
        return changed

    def _intervention_boundary(self, controllers=None) -> tuple[bool, bool]:
        """Apply expirations, clears, and hook installations in one clean pass."""
        if not self._iv_due:
            return False, True
        m = self._iv_model
        if m is None:
            self._iv_due = False
            return False, True
        revision = self._iv_revision
        originals = {intervention_key(iv.spec): iv for iv in m._ordered_records()}
        records = {key: copy(iv) for key, iv in originals.items()}
        previous = {key: copy(iv) for key, iv in originals.items()}
        changed = controllers is not None and self._linger(controllers, records)
        lingered = [iv.point for key, iv in records.items()
                    if changed and iv.state == IV_ENDED and previous[key].state != IV_ENDED]
        s = self._next_step()
        self._check_intervention_step(records, s)
        cache = m.executor._cache.copy()
        committed = False
        reports: list[str] = []
        points_changed = False
        try:
            if m._shapes_off is not None and self.state != RUN_NONE:
                shapes_off, m._shapes_off = m._shapes_off, None
                try:
                    with self._outside_lock():
                        shapes_off()
                        pts = list(m.adapter.points(m.model))[:MAX_POINTS_LISTED]
                    if self._iv_model is m:
                        self._check_metadata_budget(points=pts)
                        m.points = pts
                        points_changed = True
                except Exception:
                    _log.debug("the shapes of the points failed", exc_info=True)
                changed = True
            with self._outside_lock():
                reinstall = False
                for key in list(records):
                    iv = records[key]
                    if iv.report:
                        changed = True
                    if iv.replace:
                        iv.hook, iv.applied_spec = None, None
                        iv.replace = False
                        changed = True
                    if iv.clear:
                        del records[key]
                        changed = True
                        continue
                    if iv.state == IV_ENDED:
                        continue
                    failed = iv.hook is not None and getattr(iv.hook, "errors", 0)
                    if failed:
                        iv.error = self._stop_message(getattr(iv.hook, "error", None) or "the hook failed")
                        iv.error_code = "intervene.hook_failed"
                    ended = failed or (iv.until_step is not None and iv.until_step <= s)
                    if ended and not failed:
                        iv.error, iv.error_code = None, None
                    on = not ended and (iv.from_step is None or iv.from_step <= s)
                    if on and (iv.hook is None or reinstall):
                        iv.hook, iv.applied_spec = None, None
                        try:
                            self._install_intervention(m, iv, iv.spec, owner=originals[key])
                            reinstall = True
                        except Exception as e:
                            from .interp._memory import CleanStop
                            if isinstance(e, CleanStop) or self._resource_stop(e):
                                raise
                            iv.error = self._stop_message(f"{type(e).__name__}: {e}")
                            iv.error_code = "intervene.hook_failed"
                            ended, on = True, False
                            reports.append(f"caliper: the intervention at {iv.point!r} failed: {iv.error}")
                    if not on:
                        iv.hook, iv.applied_spec = None, None
                    state = IV_ACTIVE if on else IV_ENDED if ended else IV_PENDING
                    if state != iv.state:
                        iv.state = state
                        iv.since_step = s if on else iv.since_step if ended else None
                        changed = True
                        if failed:
                            reports.append(f"caliper: the intervention at {iv.point!r} failed and ended: {iv.error}")
            if self._iv_model is not m or self._iv_revision != revision:
                return points_changed, False
            self._check_metadata_budget(ivs=records)
            for key, iv in originals.items():
                if key not in records:
                    del self._ivs[key]
                else:
                    prepared = records[key]
                    for field in _Intervention.__slots__:
                        setattr(iv, field, getattr(prepared, field))
            m._order_live_records()
            self._iv_revision += 1
            self._iv_due = m._shapes_off is not None or any(
                iv.clear or iv.state != IV_ENDED for iv in self._ivs.values())
            if lingered:
                self._alone_since = None
            committed = True
            return changed, True
        finally:
            with self._outside_lock():
                if committed:
                    effects_changed = False
                    for key, iv in previous.items():
                        if key not in records or iv.hook is not records[key].hook:
                            effects_changed = True
                            self._unhook(iv, retain_source=True)
                    if changed or effects_changed:
                        m.executor._discard_unused_sources()
                else:
                    for key, iv in records.items():
                        if iv.hook is not previous[key].hook:
                            self._unhook(iv, retain_source=True)
                    m.executor._cache.clear()
                    m.executor._cache.update(cache)
            if committed:
                for iv in self._ivs.values():
                    if iv.report:
                        iv.report = False
                        reports.append(f"caliper: the intervention at {iv.point!r} failed and ended: {iv.error}")
                with self._outside_lock():
                    if lingered:
                        self._report_linger(lingered)
                    for message in reports:
                        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _admit_interventions(self, ivs, *, evict_ended: bool = False):
        ivs = dict(ivs)
        while True:
            try:
                if len(ivs) > MAX_INTERVENTIONS:
                    raise ValueError(f"the script holds {MAX_INTERVENTIONS} interventions")
                self._check_metadata_budget(ivs=ivs)
                checkpoint_budget = getattr(self.interpretation_service, "_check_intervention_budget", None)
                if checkpoint_budget is not None:
                    checkpoint_budget(ivs)
                return ivs
            except ValueError:
                old = next((key for key, iv in ivs.items() if iv.state == IV_ENDED), None)
                if not evict_ended or old is None:
                    raise
                del ivs[old]

    def _interventions_obj(self) -> dict:
        return {"points": list(self._iv_model.points),
                "active": [iv.entry() for iv in self._ivs.values()]}
