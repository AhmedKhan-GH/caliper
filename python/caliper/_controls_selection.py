"""Linked item spaces, selections, and background details lookup for controls."""
from __future__ import annotations

import collections
import math
import numbers
import threading

from ._controls_spec import (
    DETAIL_TEXT_MAX,
    MAX_DETAIL_FIELDS,
    MAX_DETAIL_ITEMS,
    MAX_ITEM_SPACES,
    MAX_SELECTION,
    TEXT_MAX,
    _FIRST,
    _INT_EXACT,
    _ITEM_KINDS,
    _LAST,
    _check_name,
    _cut,
    _log,
    _scalar,
    _w,
)


def _column(seq) -> list:
    tolist = getattr(seq, "tolist", None)
    if tolist is not None and not isinstance(seq, (list, tuple)):
        detach = getattr(seq, "detach", None)
        if detach is not None:
            seq = detach()
            tolist = seq.tolist
        out = tolist()
        return out if isinstance(out, list) else [out]
    return list(seq)


def _cell(v):
    v = _scalar(v)
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, numbers.Integral):
        return int(v) if abs(int(v)) <= _INT_EXACT else float(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return f if math.isfinite(f) else None
    return _cut(v if isinstance(v, str) else str(v), DETAIL_TEXT_MAX)


def _fields(names) -> list:
    names = [n for n in names]
    if len(names) > MAX_DETAIL_FIELDS:
        raise ValueError(f"details have at most {MAX_DETAIL_FIELDS} fields, not {len(names)}")
    for n in names:
        _check_name(n, "details field")
    return names


class ItemSpace:
    """A set of items numbered from 0 that several views index, made by [`items`][caliper.items].

    An example is the samples of an evaluation set. A selection in one view highlights the same
    items in every view that names the space.

    Attributes:
        name (str): The name of the item space.
    """
    __slots__ = ("name", "_ctl", "_table", "_fields", "_fn", "_selection", "_truncated",
                 "_view", "_version", "_details", "_details_error", "_settled")

    def __init__(self, ctl, name):
        self.name = name
        self._ctl = ctl
        self._table = None
        self._fields: list = []
        self._fn = None
        self._selection: list = []
        self._truncated = False
        self._view = None
        self._version = 0
        self._details = None
        self._details_error = None
        self._settled = True

    @property
    def selection(self) -> list:
        """The list of selected item ids, ascending.

        A viewer or `select` set it. It is current even while a notebook cell runs.
        """
        return list(self._selection)

    @property
    def truncated(self) -> bool:
        return self._truncated

    def select(self, items) -> list:
        """Set the selection from the script, for every viewer.

        An empty sequence clears it.

        Args:
            items (Iterable[int]): The item ids to select.

        Returns:
            list[int]: The item ids as viewers get them: ascending, without repeats, at most 32,768.

        Raises:
            ValueError: When an id is negative.
        """
        ids = [int(i) for i in items]
        if any(i < 0 for i in ids):
            raise ValueError("item ids are non-negative integers")
        self._ctl._set_selection(self.name, ids, None, False)
        return self.selection

    def clear(self) -> None:
        """Clear the selection for every viewer."""
        self.select(())

    @staticmethod
    def _details_source(details) -> tuple:
        if details is None:
            return None, [], None
        if callable(details):
            return None, [], details
        if not (hasattr(details, "items") and callable(details.items)):
            raise TypeError("details must be a mapping {field: sequence} or a function")
        pairs = list(details.items())
        fields = _fields([k for k, _ in pairs])
        return {k: _column(v) for k, v in pairs}, fields, None

    def _lookup(self, ids) -> dict:
        values = []
        for i in ids:
            row = []
            for f in self._fields:
                col = self._table[f]
                row.append(_cell(col[i]) if 0 <= i < len(col) else None)
            values.append(row)
        return {"fields": list(self._fields), "items": list(ids), "values": values}

    def __repr__(self) -> str:
        n = len(self._selection)
        return f"<caliper.ItemSpace {self.name!r}, {n} selected>"


def _details_from(out, ids) -> dict:
    if isinstance(out, dict) or (hasattr(out, "items") and callable(out.items)
                                 and not isinstance(out, (list, tuple))):
        pairs = list(out.items())
        fields = _fields([k for k, _ in pairs])
        cols = [_column(v) for _, v in pairs]
        values = [[_cell(c[r]) if r < len(c) else None for c in cols] for r in range(len(ids))]
        return {"fields": fields, "items": list(ids), "values": values}
    rows = list(out)
    fields: list = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    fields = _fields(fields)
    values = [[_cell(rows[r].get(f)) if r < len(rows) else None for f in fields]
              for r in range(len(ids))]
    return {"fields": fields, "items": list(ids), "values": values}


def _view_dims(kind: str, items) -> dict:
    if kind not in _ITEM_KINDS:
        raise ValueError(f"items= is for heatmaps, images, points and series, not {kind!r}")
    if isinstance(items, str):
        _check_name(items, "item space")
        if kind == "series":
            return {None: items}
        return {(_LAST if kind == "points" else _FIRST): items}
    if isinstance(items, dict):
        if kind == "series":
            raise ValueError("a series has no dims: items= takes the space's name as a str")
        out = {}
        for d, s in items.items():
            _check_name(d, "dim")
            out[d] = _check_name(s, "item space")
        return out
    raise TypeError("items= takes an item space name (str) or a dict {dim: space}")


class _DetailsWorker:
    def __init__(self):
        self._jobs: collections.OrderedDict = collections.OrderedDict()
        self._cond = threading.Condition()
        self._thread = None

    def submit(self, key, fn) -> None:
        with self._cond:
            self._jobs.pop(key, None)
            self._jobs[key] = fn
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="caliper.details",
                                                daemon=True)
                self._thread.start()
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._jobs:
                    self._cond.wait()
                _, fn = self._jobs.popitem(last=False)
            try:
                fn()
            except Exception:
                _log.debug("a details job failed", exc_info=True)


def _normalise(ids):
    out = sorted({int(i) for i in ids if isinstance(i, numbers.Integral)
                  and not isinstance(i, bool) and i >= 0})
    return out[:MAX_SELECTION], len(out) > MAX_SELECTION


def _ids_list(v):
    if not isinstance(v, list):
        return None, False
    cut = len(v) > MAX_SELECTION
    return v[:MAX_SELECTION], cut


def _range_ids(r):
    if not (isinstance(r, list) and len(r) == 2 and all(
            isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in r)):
        return None, False
    a, b = max(int(r[0]), 0), int(r[1])
    if b <= a:
        return [], False
    return range(a, min(b, a + MAX_SELECTION)), b - a > MAX_SELECTION


def _axis_rows(axis: dict, positions) -> list:
    start, step, count, size = (int(axis["start"]), int(axis["step"]), int(axis["count"]),
                                int(axis["size"]))
    return [start + (int(i) * count // size) * step for i in positions
            if isinstance(i, numbers.Integral) and 0 <= i < size]


def _event_items(m: dict, detail: dict, axes=None):
    spaces = set(m.values())
    if not detail:
        return [(s, [], False) for s in sorted(spaces)]
    space = detail.get("space")
    if space is not None:
        ids, cut = _ids_list(detail.get("items"))
        if space not in spaces or ids is None:
            return []
        return [(space, ids, cut or detail.get("truncated") is True)]
    cut = detail.get("truncated") is True
    if "row_ids" in detail or "indices" in detail:
        dims = [k for k in m if k is not None]
        dim = axes[-1]["dim"] if axes else (dims[0] if len(dims) == 1 else "point")
        space = m.get(dim)
        if space is None:
            return []
        ids, c = _ids_list(detail.get("row_ids"))
        if ids is None:
            pos, c = _ids_list(detail.get("indices"))
            if pos is None or not axes:
                return []
            ids = _axis_rows(axes[-1], pos)
        return [(space, ids, cut or c)]
    if "steps" in detail:
        space = m.get(None)
        ids, c = _range_ids(detail.get("steps"))
        return [] if space is None or ids is None else [(space, ids, cut or c)]
    if "rows" in detail or "cols" in detail:
        row_dim = axes[-2]["dim"] if axes and len(axes) >= 2 else "row"
        col_dim = axes[-1]["dim"] if axes else "col"
        for dim, key in ((row_dim, "rows"), (col_dim, "cols")):
            if dim in m:
                ids, c = _range_ids(detail.get(key))
                if ids is not None:
                    return [(m[dim], ids, cut or c)]
    return []


class _SelectionMixin:
    """Item spaces, linked selection routing, and details lookup for Controls."""

    def items(self, space, details=None) -> ItemSpace:
        _check_name(space, "item space")
        source = ItemSpace._details_source(details)
        with self.lock:
            sp = self.spaces.get(space)
            if sp is None:
                if len(self.spaces) >= MAX_ITEM_SPACES:
                    raise ValueError(f"at most {MAX_ITEM_SPACES} item spaces")
                sp = self.spaces[space] = ItemSpace(self, space)
            sp._table, sp._fields, sp._fn = source
        return sp

    def view_items_call(self, name: str, kind: str, items) -> None:
        args = (kind, items)
        if self._view_args.get(name) == args:
            return
        mapping = _view_dims(kind, items)
        with self.lock:
            spaces = set(self.spaces) | {s for m in self.view_items.values() for s in m.values()}
            if len(spaces | set(mapping.values())) > MAX_ITEM_SPACES:
                raise ValueError(f"a session uses at most {MAX_ITEM_SPACES} item spaces")
            changed = self.view_items.get(name) != mapping
            self.view_items[name] = mapping
            self._view_args[name] = args
            self._resolved.pop(name, None)
            routes = list(self.routes)
        if changed:
            for r in routes:
                fn = getattr(r.transport, "views_changed", None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        _log.debug("views_changed failed", exc_info=True)

    def forget_view(self, name: str) -> None:
        with self.lock:
            self.view_items.pop(name, None)
            self._view_args.pop(name, None)
            self._resolved.pop(name, None)

    def annotate_view(self, entry: dict) -> dict:
        name = entry.get("name")
        m = self.view_items.get(name)
        if not m:
            return entry
        out = dict(entry)
        resolved = {}
        if None in m and entry.get("kind") == 4:
            out["items"] = resolved[None] = m[None]
        sources = []
        for s in entry.get("sources") or []:
            dims = list(s.get("dims", []))
            new = []
            for i, d in enumerate(dims):
                space = m.get(d.get("name"))
                if space is None and i == 0:
                    space = m.get(_FIRST)
                if space is None and i == len(dims) - 1:
                    space = m.get(_LAST)
                if space is not None:
                    resolved[d["name"]] = space
                    d = dict(d, items=space)
                new.append(d)
            sources.append(dict(s, dims=new))
        out["sources"] = sources
        with self.lock:
            if self.view_items.get(name) is m:
                self._resolved[name] = resolved
        return out

    def _dims_of(self, view: str) -> dict:
        r = self._resolved.get(view)
        if r is not None:
            return r
        m = self.view_items.get(view) or {}
        out = {}
        for k, space in m.items():
            if k == _FIRST:
                out.setdefault("row", space)
            elif k == _LAST:
                out.setdefault("point", space)
            else:
                out[k] = space
        return out

    def _on_select(self, meta, axes=None) -> None:
        if not isinstance(meta, dict) or meta.get("type") != "select":
            return
        view, detail = meta.get("view"), meta.get("detail")
        m = self._dims_of(view) if isinstance(view, str) else None
        if not m or not isinstance(detail, dict):
            return
        self.counters["selects_in"] += 1
        for space, ids, cut in _event_items(m, detail, axes):
            self._set_selection(space, ids, view, cut)

    def _set_selection(self, space: str, ids, view, cut: bool) -> None:
        from . import _controls as _mod
        ids, cut2 = _normalise(ids)
        submit = None
        with self.lock:
            sp = self.spaces.get(space)
            if sp is None:
                if len(self.spaces) >= MAX_ITEM_SPACES:
                    return
                sp = self.spaces[space] = ItemSpace(self, space)
            sp._selection, sp._truncated, sp._view = ids, cut or cut2, view
            sp._version += 1
            sp._details, sp._details_error, sp._settled = None, None, True
            if ids and sp._table is not None:
                sp._details = sp._lookup(ids[:MAX_DETAIL_ITEMS])
            elif ids and sp._fn is not None:
                sp._settled = False
                submit = (sp, sp._version)
        self._publish_selection(sp)
        if submit is not None:
            sp_, version = submit
            self._details_worker.submit(sp_.name, lambda: self._run_details(sp_, version))
            self._timers.at(_mod.time.monotonic() + _mod.DETAILS_TIMEOUT_S, self._details_late, sp_,
                            version)

    def _run_details(self, sp: ItemSpace, version: int) -> None:
        with self.lock:
            if sp._version != version or sp._settled:
                return
            ids, fn = sp._selection[:MAX_DETAIL_ITEMS], sp._fn
        details = error = None
        try:
            details = _details_from(fn(list(ids)), ids)
        except Exception as e:
            error = _cut(f"{type(e).__name__}: {e}", TEXT_MAX)
        with self.lock:
            if sp._version != version or sp._settled:
                return
            sp._settled = True
            sp._details, sp._details_error = details, error
        self._publish_selection(sp)

    def _details_late(self, sp: ItemSpace, version: int) -> None:
        from . import _controls as _mod
        with self.lock:
            if sp._version != version or sp._settled:
                return
            sp._settled = True
            sp._details_error = (f"timeout: the details function took more than "
                                 f"{_mod.DETAILS_TIMEOUT_S:g} s")
        self._publish_selection(sp)

    def _selection_meta(self, sp: ItemSpace) -> dict:
        w = _w()
        return w.Selection(sp.name, list(sp._selection), sp._view, sp._truncated, sp._details,
                           sp._details_error).to_meta()

    def _publish_selection(self, sp: ItemSpace) -> None:
        if not self.routes:
            return
        w = _w()
        with self._send:
            with self.lock:
                meta = self._selection_meta(sp)
                routes = list(self.routes)
            self.counters["selections_out"] += 1
            for r in routes:
                r._broadcast(w.MsgType.SELECTION, meta)
