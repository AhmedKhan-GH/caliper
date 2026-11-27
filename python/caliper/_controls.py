"""Controls of the running script and linked selection (wire protocol 3.1).

A viewer can pause, resume and single-step the training loop, change the
parameters the script registered, run the actions it registered, and select
items that every view of the session then highlights
(docs/reference/wire-protocol.md, "Controls" and "Linked selection"). This
module holds all of that state for the process, one :class:`Controls` object,
``CONTROLS``, behind the public calls of caliper/__init__.py:

    caliper.step()                 the checkpoint, once per training step
    caliper.pause()                pause at the next checkpoint
    caliper.param(name, value)     a Param: .value, .set(v), .remove()
    caliper.action(name, fn)       an Action: called by a viewer at a checkpoint
    caliper.items(space, details)  an ItemSpace: .selection, .select(ids)
    items= on show, points, series and offer: which axis indexes which space

It imports neither torch nor numpy, nor caliper._wire (which imports numpy)
until a transport attaches or a value is validated, and it works without any
transport: parameters keep their values, ``step()`` returns at once, and an
action can be called from Python.
"""
from __future__ import annotations

import collections
from contextlib import contextmanager
import heapq
import itertools
import numbers
import threading
import time
import warnings

from ._controls_interventions import (
    Interventions,
    _Intervention,
    _InterventionsMixin,
    _record_specifications,
)
from ._controls_selection import (
    ItemSpace,
    _DetailsWorker,
    _SelectionMixin,
    _axis_rows,
    _cell,
    _column,
    _details_from,
    _event_items,
    _fields,
    _ids_list,
    _normalise,
    _range_ids,
    _view_dims,
)
from ._controls_spec import (
    CAPABILITIES,
    CAP_CONTROLS,
    CAP_SELECTION,
    DETAIL_TEXT_MAX,
    ERR_BAD_RANGE,
    ERR_NOT_SUPPORTED,
    ERR_OP_RESERVED,
    ERR_UNKNOWN_POINT,
    INTERVENE_OPS,
    INTERVENE_RESERVED,
    IV_ACTIVE,
    IV_ENDED,
    IV_PENDING,
    MAX_ACTIONS,
    MAX_CHOICES,
    MAX_DETAIL_FIELDS,
    MAX_DETAIL_ITEMS,
    MAX_INTERVENTIONS,
    MAX_ITEM_SPACES,
    MAX_META_BYTES,
    MAX_PARAMS,
    MAX_POINTS_LISTED,
    MAX_SELECTION,
    NAME_MAX,
    OP_INTERVENE,
    PARAM_TEXT_MAX,
    POINT_MAX,
    RUN_NONE,
    RUN_PAUSED,
    RUN_PAUSING,
    RUN_RUNNING,
    RUN_STEPPING,
    TEXT_MAX,
    Action,
    Param,
    Route,
    _FIRST,
    _INT_EXACT,
    _ITEM_KINDS,
    _LAST,
    _LIFECYCLE_HEADROOM,
    _NO,
    _check_controls_budget,
    _check_help,
    _check_name,
    _cut,
    _json_bytes,
    _log,
    _name_ok,
    _param_spec,
    _run_value,
    _scalar,
    _step_ok,
    _w,
    intervention_key,
)

__all__ = [
    "CONTROLS",
    "Controls",
    "Param",
    "Action",
    "ItemSpace",
    "Route",
    "attach",
    "CAPABILITIES",
    "Interventions",
    "intervention_key",
]

ACTION_WAIT_S = 600.0                  # CALIPER_NET_ACTION_WAIT_MS
DETAILS_TIMEOUT_S = 2.0                # CALIPER_NET_DETAILS_TIMEOUT_MS
RELEASE_GRACE_S = 10.0
WAIT_POLL_S = 0.1
STEPS_REPORT_S = 0.1
RUN_REPORT_S = 1.0
LINGER_CHECK_S = 0.25


# ===========================================================================
# time-outs off the training thread
# ===========================================================================
class _Timers:
    def __init__(self):
        self._heap: list = []
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._thread = None

    def at(self, when: float, fn, *args) -> None:
        with self._cond:
            heapq.heappush(self._heap, (when, next(self._seq), fn, args))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="caliper.controls",
                                                daemon=True)
                self._thread.start()
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._heap or self._heap[0][0] > time.monotonic():
                    wait = None if not self._heap else self._heap[0][0] - time.monotonic()
                    self._cond.wait(wait)
                _, _, fn, args = heapq.heappop(self._heap)
            try:
                fn(*args)
            except Exception:
                _log.debug("a timer failed", exc_info=True)


# ===========================================================================
# the state
# ===========================================================================
class Controls(_SelectionMixin, _InterventionsMixin):
    """The controls and selections of this process (``CONTROLS``)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self._send = threading.RLock()
        self.busy = True
        self.n = 0
        self.state = RUN_NONE
        self.steps_left = 0
        self._report_due = False
        self._report_armed = False
        self._steps_reported = 0.0
        self.params: dict[str, Param] = {}
        self.actions: dict[str, Action] = {}
        self.interpretation_service = None
        self.interpretation_registration_lock = threading.Lock()
        self.queue: collections.deque = collections.deque()
        self.spaces: dict[str, ItemSpace] = {}
        self.view_items: dict[str, dict] = {}
        self._view_args: dict[str, tuple] = {}
        self._resolved: dict[str, dict] = {}
        self.routes: list[Route] = []
        self._npending = 0
        self.wait_hooks: list = []
        self._timers = _Timers()
        self._details_worker = _DetailsWorker()
        self.counters = collections.Counter()
        self._iv_model: Interventions | None = None
        self._ivs: dict = {}
        self._iv_due = False
        self._iv_boundary_failures = 0
        self._iv_revision = 0
        self._alone_since = None
        self._linger_next = 0.0

    @contextmanager
    def _outside_lock(self):
        self.lock.release()
        try:
            yield
        finally:
            self.lock.acquire()

    def step(self, step=None) -> None:
        if self._iv_model is not None and self._iv_model.executor.preparing_source:
            return
        prospective = self.n + 1 if step is None else step
        if (isinstance(prospective, numbers.Integral) and not isinstance(prospective, bool)
                and (prospective < -1 or prospective >= _INT_EXACT - 1)):
            with self.lock:
                self._check_intervention_step(self._ivs, int(prospective) + 1)
                if prospective >= _INT_EXACT:
                    meta = self._controls_meta()
                    meta["run"]["step"] = int(prospective)
                    _check_controls_budget(meta, ivs=self._ivs)
                self.n = prospective
        else:
            self.n = prospective
        if self.busy:
            self.checkpoint()

    def checkpoint(self) -> None:
        publish = False
        with self.lock:
            self._check_intervention_step(self._ivs, self._next_step())
            if self.state == RUN_NONE:
                self.state = RUN_RUNNING
                publish = True
            elif self.state == RUN_STEPPING:
                self.steps_left -= 1
                if self.steps_left <= 0:
                    self.steps_left = 0
                    self.state = RUN_PAUSING
                elif time.monotonic() - self._steps_reported >= STEPS_REPORT_S:
                    publish = True
            if self._report_due:
                self._report_due = False
                publish = True
            self._update_busy()
        if publish:
            self._publish()
        self._service()
        with self.lock:
            pause = self.state == RUN_PAUSING
        if pause:
            self._wait()
        with self.lock:
            self._update_busy()

    def _update_busy(self) -> None:
        self.busy = (self.state in (RUN_NONE, RUN_PAUSING, RUN_STEPPING)
                     or self._npending > 0 or bool(self.queue) or self._report_due
                     or self._iv_due or (self.interpretation_service is not None
                                         and self.interpretation_service.pending))

    def _arm_report(self) -> None:
        if self.state == RUN_RUNNING and not self._report_armed:
            self._report_armed = True
            self._timers.at(time.monotonic() + RUN_REPORT_S, self._report)

    def _report(self) -> None:
        with self.lock:
            self._report_armed = False
            if self.state == RUN_RUNNING and self.routes:
                self._report_due = True
                self.busy = True

    def _apply_pending(self) -> bool:
        if not self._npending:
            return False
        for p in self.params.values():
            if p._pending is not _NO:
                p._value, p._pending = p._pending, _NO
                p._version += 1
        self._npending = 0
        return True

    def _resource_stop(self, error) -> bool:
        service = self.interpretation_service
        return service is not None and isinstance(error, getattr(service, "stop_exception", ()))

    def _raise_resource_stop(self) -> None:
        failure = getattr(self.interpretation_service, "failure", None)
        if failure is not None:
            raise failure

    def _service(self) -> None:
        try:
            self._raise_resource_stop()
            self._service_work()
        except Exception as error:
            if self._resource_stop(error):
                self.interpretation_service.stop(error, self.n)
                self._terminate_actions(error)
            raise

    @staticmethod
    def _stop_message(error):
        text = str(error).replace("\0", "").encode("utf-8", errors="replace").decode("utf-8")
        return _cut(text, TEXT_MAX)

    def _terminate_actions(self, error):
        with self.lock:
            pending = []
            while self.queue:
                action = self.queue.popleft()
                pending.append((action, action.waiters))
                action.queued, action.waiters = False, []
            self._update_busy()
        self._publish()
        for action, waiters in pending:
            self._action_replies(action, waiters, False, self._stop_message(error))

    def _service_work(self) -> None:
        controllers = self._controllers_due()
        with self.lock:
            actions = {action: (action.run_id, len(action.waiters)) for action in self.queue}
            intervention_failures = self._iv_boundary_failures
            changed = self._apply_pending()
            changed = self._apply_interventions(controllers) or changed
        if changed:
            self._publish()
        self._raise_resource_stop()
        if self.interpretation_service is not None:
            self.interpretation_service.step(self.n)
        while True:
            self._raise_resource_stop()
            with self.lock:
                if self._iv_boundary_failures != intervention_failures or not self.queue:
                    return
                a = self.queue[0]
                if actions.get(a) != (a.run_id, len(a.waiters)):
                    return
                self.queue.popleft()
                a.queued, a.running = False, True
                waiters, a.waiters = a.waiters, []
                self._update_busy()
            self._publish()
            try:
                ok, text = self._run_action(a)
                self._raise_resource_stop()
            except BaseException as error:
                if self._resource_stop(error):
                    self.interpretation_service.stop(error, self.n)
                    self._action_replies(a, waiters, False, self._stop_message(error))
                raise
            finally:
                with self.lock:
                    a.running = False
                self._publish()
            self._action_replies(a, waiters, ok, text)

    def _action_replies(self, action, waiters, ok, text):
        w = _w()
        for route, conn, rid in waiters:
            meta = w.ControlReply(
                rid, w.OP_RUN, w.STATUS_OK if ok else w.STATUS_ERROR, action.name,
                message=(text or None) if ok else None,
                error=None if ok else {"code": w.ERR_ACTION_FAILED, "message": text},
            ).to_meta()
            route._reply(conn, meta)

    def _run_action(self, a: Action):
        self.counters["actions_run"] += 1
        try:
            r = a.fn()
        except Exception as e:
            if self._resource_stop(e):
                raise
            text = _cut(f"{type(e).__name__}: {e}", TEXT_MAX)
            warnings.warn(f"caliper: action {a.name!r} raised {text}", RuntimeWarning,
                          stacklevel=2)
            return False, text
        return True, "" if r is None else _cut(str(r), TEXT_MAX)

    def _wait(self) -> None:
        with self.lock:
            if self.state != RUN_PAUSING:
                return
            self.state = RUN_PAUSED
        for hook in list(self.wait_hooks):
            try:
                hook()
            except Exception:
                _log.debug("a wait hook failed", exc_info=True)
        self._publish(after_data=True)
        absent_since = None
        try:
            while True:
                self._service()
                with self.lock:
                    if self.state != RUN_PAUSED:
                        return
                    routes = list(self.routes)
                count = sum(r._controllers() for r in routes) if routes else 0
                now = time.monotonic()
                if count > 0:
                    absent_since = None
                elif not routes or (absent_since is not None
                                    and now - absent_since >= RELEASE_GRACE_S):
                    self._release()
                    return
                elif absent_since is None:
                    absent_since = now
                with self.lock:
                    if self.state != RUN_PAUSED:
                        return
                    if not self.queue and not self._npending:
                        self.cond.wait(WAIT_POLL_S)
        except BaseException:
            self._release()
            raise

    def _release(self) -> None:
        with self.lock:
            if self.state not in (RUN_PAUSED, RUN_PAUSING, RUN_STEPPING):
                return
            self.state = RUN_RUNNING
            self.steps_left = 0
            self._update_busy()
            self.cond.notify_all()
        self._publish()

    # -- the script's calls ------------------------------------------------------
    def pause(self) -> bool:
        with self.lock:
            routes = list(self.routes)
        if not routes:
            warnings.warn("caliper.pause(): no stream server that carries controls runs, so "
                          "no viewer could resume the run; it goes on", RuntimeWarning,
                          stacklevel=3)
            return False
        if sum(r._controllers() for r in routes) == 0:
            warnings.warn(f"caliper.pause(): no connection may control the run now; it "
                          f"resumes after {RELEASE_GRACE_S:g} s unless one connects",
                          RuntimeWarning, stacklevel=3)
        with self.lock:
            if self.state in (RUN_NONE, RUN_RUNNING, RUN_STEPPING):
                self.state = RUN_PAUSING
                self.steps_left = 0
                self._update_busy()
                changed = True
            else:
                changed = False
        if changed:
            self._publish()
        return True

    def param(self, name, value, *, min=None, max=None, log=False, choices=None,
              help=None) -> Param:
        spec = _param_spec(name, value, min, max, log, choices, help)
        with self.lock:
            old = self.params.get(name)
            if old is None and len(self.params) >= MAX_PARAMS:
                raise ValueError(f"at most {MAX_PARAMS} parameters")
            p = Param(self, name, *spec)
            _run_value(p)
            self._check_metadata_budget(params={**self.params, name: p})
            if old is not None:
                old.removed = True
                if old._pending is not _NO:
                    self._npending -= 1
            self.params[name] = p
            self._update_busy()
        self._publish()
        return p

    def _script_set(self, p: Param, value):
        w = _w()
        try:
            v = w.param_value(p.entry(), _scalar(value))
        except w.MessageError as e:
            raise ValueError(e.message) from None
        with self.lock:
            if self.params.get(p.name) is p:
                entry = p.entry()
                entry["value"] = v
                entry.pop("pending", None)
                self._check_param_budget(p, entry)
            p._value = v
            p._version += 1
            if p._pending is not _NO:
                p._pending = _NO
                self._npending -= 1
            self._update_busy()
        if not p.removed:
            self._publish()
        return v

    def _remove_param(self, p: Param) -> bool:
        with self.lock:
            if self.params.get(p.name) is not p:
                return False
            del self.params[p.name]
            p.removed = True
            if p._pending is not _NO:
                p._value, p._pending = p._pending, _NO
                p._version += 1
                self._npending -= 1
            self._update_busy()
        self._publish()
        return True

    def action(self, name, fn, *, help=None) -> Action:
        _check_name(name, "action")
        if not callable(fn):
            raise TypeError("caliper.action() takes a callable")
        help = _check_help(help)
        with self.lock:
            old = self.actions.get(name)
            if old is None and len(self.actions) >= MAX_ACTIONS:
                raise ValueError(f"at most {MAX_ACTIONS} actions")
            a = Action(self, name, fn, help)
            self._check_metadata_budget(actions={**self.actions, name: a})
            if old is not None:
                old.removed = True
                if old.queued:
                    a.queued, a.waiters, a.run_id = True, old.waiters, old.run_id
                    self.queue = collections.deque(a if x is old else x for x in self.queue)
                    old.queued, old.waiters = False, []
            self.actions[name] = a
        self._publish()
        return a

    def _remove_action(self, a: Action) -> bool:
        with self.lock:
            if self.actions.get(a.name) is not a:
                return False
            del self.actions[a.name]
            a.removed = True
            waiters = a.waiters if a.queued else []
            if a.queued:
                self.queue.remove(a)
                a.queued, a.waiters = False, []
            self._update_busy()
        self._publish()
        self._answer(waiters, a.name, "unknown_control", f"the action {a.name!r} was removed")
        return True

    # -- requests (the transport's thread) -------------------------------------------
    def _on_control(self, route: Route, conn, meta, *, acknowledge=None):
        w = _w()
        if not isinstance(meta, dict):
            raise w.MessageError(w.ERR_BAD_REQUEST, "a control's metadata is an object")
        msg = w.Control.from_frame(w.Frame(w.MsgType.CONTROL, meta))
        rid, op, name = msg.request_id, msg.op, msg.name
        self.counters["controls_in"] += 1

        def error(code, text):
            return w.ControlReply(rid, op, w.STATUS_ERROR, name,
                                  error={"code": code, "message": text}).to_meta()

        if op == "interpret":
            from . import _interpret_protocol as protocol
            from .interp._service import Refused
            if msg.interpretation["method"] not in protocol.READ_METHODS:
                if not route.transport.may_control(conn):
                    return error(w.ERR_FORBIDDEN, "This connection may not change the run.")
            service = self.interpretation_service
            if service is None:
                return error("interpret.unsupported", "The run has no interpretation service.")

            def completed(status, result, refused):
                route._reply(conn, w.ControlReply(rid, op, status, result=result, error=refused).to_meta())

            try:
                queued = (lambda: acknowledge(w.ControlReply(rid, op, w.STATUS_QUEUED).to_meta())) \
                    if acknowledge is not None else None
                status, result = service.submit(msg.interpretation, completed, acknowledge=queued)
            except Refused as exc:
                return error(exc.code, str(exc))
            with self.lock:
                self._update_busy()
                if status == w.STATUS_QUEUED:
                    self.cond.notify_all()
            if status == w.STATUS_QUEUED and acknowledge is not None:
                return None
            return w.ControlReply(rid, op, status, result=result).to_meta()

        try:
            allowed = bool(route.transport.may_control(conn))
        except Exception:
            allowed = False
        if not allowed:
            return error(w.ERR_FORBIDDEN, "this connection may not control the run")
        reply = None
        timer = None
        with self.lock:
            if getattr(self.interpretation_service, "failure", None) is not None:
                return error("interpret.stopped", "The interpretation run stopped.")
            if op in (w.OP_PAUSE, w.OP_STEP) and self.state == RUN_NONE:
                return error(w.ERR_UNSUPPORTED, "the script has called no checkpoint "
                                                "(caliper.step()) yet")
            if op == w.OP_PAUSE:
                if self.state in (RUN_RUNNING, RUN_STEPPING):
                    self.state = RUN_PAUSING
                    self.steps_left = 0
            elif op == w.OP_RESUME:
                if self.state in (RUN_PAUSING, RUN_PAUSED, RUN_STEPPING):
                    self.state = RUN_RUNNING
                    self.steps_left = 0
            elif op == w.OP_STEP:
                self.state = RUN_STEPPING
                self.steps_left = msg.count or 1
                self._steps_reported = time.monotonic()
            elif op == w.OP_SET:
                p = self.params.get(name)
                if p is None:
                    return error(w.ERR_UNKNOWN_CONTROL, f"no parameter {name!r}")
                try:
                    v = w.param_value(p.entry(), msg.value)
                except w.MessageError as e:
                    return error(e.code, e.message)
                entry = p.entry()
                if self.state == RUN_NONE:
                    entry["value"] = v
                    entry.pop("pending", None)
                else:
                    entry["pending"] = v
                try:
                    self._check_param_budget(p, entry)
                except ValueError as e:
                    return error(w.ERR_LIMIT, str(e))
                if self.state == RUN_NONE:
                    p._value = v
                    p._version += 1
                    if p._pending is not _NO:
                        p._pending = _NO
                        self._npending -= 1
                else:
                    if p._pending is _NO:
                        self._npending += 1
                    p._pending = v
                reply = w.ControlReply(rid, op, w.STATUS_OK, name, v).to_meta()
            elif op == w.OP_RUN:
                a = self.actions.get(name)
                if a is None:
                    return error(w.ERR_UNKNOWN_CONTROL, f"no action {name!r}")
                if not a.queued:
                    a.queued = True
                    a.run_id += 1
                    self.queue.append(a)
                    timer = (a, a.run_id)
                a.waiters.append((route, conn, rid))
                reply = w.ControlReply(rid, op, w.STATUS_QUEUED, name).to_meta()
            elif op == OP_INTERVENE:
                refused = self._intervene(getattr(msg, "intervene", None))
                if refused is not None:
                    return error(*refused)
            self._update_busy()
            self.cond.notify_all()
        if timer is not None:
            self._timers.at(time.monotonic() + ACTION_WAIT_S, self._expire, *timer)
        self._publish()
        return reply or w.ControlReply(rid, op, w.STATUS_OK, name).to_meta()

    def _expire(self, a: Action, run_id: int) -> None:
        with self.lock:
            if not a.queued or a.run_id != run_id:
                return
            self.queue.remove(a)
            a.queued = False
            waiters, a.waiters = a.waiters, []
            self._update_busy()
        self._publish()
        self._answer(waiters, a.name, "timeout",
                     f"no checkpoint ran the action within {ACTION_WAIT_S:g} s")

    def _answer(self, waiters, name, code, text) -> None:
        if not waiters:
            return
        w = _w()
        for route, conn, rid in waiters:
            route._reply(conn, w.ControlReply(rid, w.OP_RUN, w.STATUS_ERROR, name, error={
                "code": code, "message": text}).to_meta())

    # -- messages ------------------------------------------------------------------
    def _run_obj(self) -> dict:
        r = {"state": self.state}
        s = self.n if self.state != RUN_NONE else None
        if isinstance(s, numbers.Integral) and not isinstance(s, bool) and s >= 0:
            r["step"] = int(s)
        if self.state == RUN_STEPPING:
            r["steps_left"] = self.steps_left
        return r

    def controls_meta(self) -> dict:
        with self.lock:
            return self._controls_meta()

    def _controls_meta(self) -> dict:
        meta = {"run": self._run_obj(), "params": [p.entry() for p in self.params.values()],
                "actions": [a.entry() for a in self.actions.values()]}
        if self._iv_model is not None:
            meta["interventions"] = self._interventions_obj()
        return meta

    def _check_metadata_budget(self, *, params=None, actions=None, ivs=None, points=None):
        meta = self._controls_meta()
        if params is not None:
            meta["params"] = [p.entry() for p in params.values()]
        if actions is not None:
            meta["actions"] = [a.entry() for a in actions.values()]
        if points is not None:
            meta.setdefault("interventions", {"active": []})["points"] = points
        if ivs is not None:
            meta.setdefault("interventions", {"points": []})["active"] = [iv.entry() for iv in ivs.values()]
        _check_controls_budget(meta, ivs=self._ivs if ivs is None else ivs)

    def _check_param_budget(self, param, entry):
        meta = self._controls_meta()
        meta["params"] = [entry if p is param else p.entry() for p in self.params.values()]
        _check_controls_budget(meta, ivs=self._ivs)

    def snapshot_state(self) -> list:
        w = _w()
        with self.lock:
            out = [(w.MsgType.CONTROLS, self._controls_meta())]
            out += [(w.MsgType.SELECTION, self._selection_meta(sp))
                    for sp in self.spaces.values() if sp._selection]
        return out

    def _publish(self, after_data: bool = False) -> None:
        if not self.routes:
            return
        w = _w()
        with self._send:
            with self.lock:
                meta = self._controls_meta()
                routes = list(self.routes)
                if self.state == RUN_STEPPING:
                    self._steps_reported = time.monotonic()
                self._arm_report()
            self.counters["controls_out"] += 1
            for r in routes:
                r._broadcast(w.MsgType.CONTROLS, meta, after_data)

    def attach(self, transport) -> Route:
        r = Route(self, transport)
        with self.lock:
            self.routes.append(r)
            self._arm_report()
        return r

    def _detach(self, route: Route) -> None:
        with self.lock:
            route.attached = False
            if route in self.routes:
                self.routes.remove(route)
            self.cond.notify_all()

    def summary(self) -> dict:
        with self.lock:
            return {"run": self._run_obj(),
                    "params": {n: p._value for n, p in self.params.items()},
                    "pending": {n: p._pending for n, p in self.params.items()
                                if p._pending is not _NO},
                    "actions": {n: a.entry() for n, a in self.actions.items()},
                    "selections": {n: list(s._selection) for n, s in self.spaces.items()},
                    "interventions": [dict(iv.entry(), error=iv.error)
                                      for iv in self._ivs.values()],
                    "routes": len(self.routes), **dict(self.counters)}


CONTROLS = Controls()


def attach(transport) -> Route:
    return CONTROLS.attach(transport)
