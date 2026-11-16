"""caliper.watch(model): views of a module's activations and gradients.

Forward hooks show each selected module's output as ``<module>.act`` (a channel
mosaic for (N, C, H, W) outputs, a heatmap for (N, F)); gradient hooks show
each selected parameter's gradient as ``<module>.<param>.grad`` (a heatmap,
higher-rank weights flattened to (out, -1)). Every view goes through
caliper.show, so it is rate limited per view (default 60 Hz) and near free
while nothing is attached; ``every=N`` also skips N-1 of N hook calls.

Grouped views (``groups=True``): the same hooks also feed three heatmaps whose
data view picks the source, so that one widget can switch between layers.

* ``weights``: the parameters with 2 or more dimensions of every selected
  module, sources ``<module>.<param>``, each a heatmap with dims (out, in)
  (higher-rank weights flattened to (out, -1) as the gradient views are), so
  ``{"source": name}`` alone switches between them. A snapshot is taken when
  the owning module runs forward (with gradients only, after its gradient
  accumulates) and on caliper.update().
* ``activations``: the output of every selected module, source ``<module>``,
  with dims named from its rank: (batch, feature), (batch, position, feature),
  (batch, channel, height, width), (batch, channel, length) for a 1-d conv.
  The default data view is that of the first module by the rule of
  caliper._dataview.auto_default (leading dims at 0: a 4-d output shows
  channel 0 of item 0); because a request's index replaces the default's, a
  data view of a module of another rank names its own index (``{}`` for a
  2-d output), and ``"tile": "channel"`` shows the channel mosaic.
* ``gradients``: the gradients that ``<module>.<param>.grad`` shows, with the
  sources and dims of ``weights``.

A view has at most 64 sources (the wire protocol's limit); a group keeps the
first 64 with a warning. A source does tensor work only when a route needs it:
the window and the session need the default data view's source, a server the
sources its subscriptions select. The hook of any other source records its
shape (metadata) while something is attached, and returns.
"""
from __future__ import annotations

import fnmatch
import importlib
import math
import warnings
import weakref

from . import _dataview
from ._convert import CMAPS

__all__ = ["watch", "WatchHandle"]


class WatchHandle:
    """The hooks of one watch() call. ``remove()`` takes them all off."""

    def __init__(self):
        self._handles: list = []
        self._entries: list = []            # the grouped views' registry entries
        self.views: list[str] = []          # view names the hooks show into
        self.modules: list[str] = []        # module names that got a hook
        self.groups: list[str] = []         # grouped view names (weights, activations, gradients)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for e in self._entries:
            _dataview.REGISTRY.remove(e.name, e)
        self._entries.clear()

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def __len__(self) -> int:
        return len(self._handles)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __repr__(self) -> str:
        return f"WatchHandle({len(self._handles)} hooks, views={self.views})"


def _selector(include):
    if include is None:
        return lambda name, mod: not any(True for _ in mod.children())   # leaf modules
    if callable(include):
        return include
    pats = [include] if isinstance(include, str) else list(include)
    return lambda name, mod: any(fnmatch.fnmatchcase(name, p) for p in pats)


def _flat_weight(ref):
    def get():
        p = ref()
        return None if p is None else p.reshape(p.shape[0], -1)
    return get


def _activation_namer(mod):
    import torch
    conv = isinstance(mod, torch.nn.modules.conv._ConvNd)

    def namer(n: int) -> tuple:
        if n <= 2:
            return ("batch", "feature")[:n]
        spatial = {1: ("length",), 2: ("height", "width"), 3: ("depth", "height", "width")}
        if (conv or n >= 4) and n - 2 in spatial:
            return ("batch", "channel") + spatial[n - 2]
        if n == 3:
            return ("batch", "position", "feature")
        return ("batch",) + tuple(f"dim{i}" for i in range(1, n))
    return namer


def _register_groups(h, picked, activations, gradients, rate, prefix):
    """Register the grouped views of one watch() call; returns their entries
    (weights, activations, gradients; None where there is no source)."""
    ws, acts, grads, long = [], [], [], []
    for label, mod in picked:
        for pname, p in mod.named_parameters(recurse=False):
            if p.ndim < 2:
                continue
            meta = (f"{label}.{pname}", ("out", "in"), (p.shape[0], math.prod(p.shape[1:])),
                    _dataview.dtype_name(p.dtype))
            try:
                ws.append(_dataview.Source(*meta, get=_flat_weight(weakref.ref(p))))
            except ValueError:
                long.append(meta[0])
                continue
            if gradients and p.requires_grad:
                grads.append(_dataview.Source(*meta))
        if activations:
            try:
                acts.append(_dataview.Source(label, namer=_activation_namer(mod)))
            except ValueError:
                long.append(label)
    if long:
        warnings.warn(f"caliper.watch: {len(long)} sources have names longer than "
                      f"{_dataview.NAME_MAX} bytes and are left out of the grouped views "
                      f"(first: {long[0]!r})", RuntimeWarning, stacklevel=3)
    out = []
    for name, sources, cmap in (("weights", ws, "viridis"), ("activations", acts, "viridis"),
                                ("gradients", grads, "rdbu")):
        e = None
        if len(sources) > _dataview.MAX_SOURCES:
            warnings.warn(f"caliper.watch: the {prefix + name!r} view keeps the first "
                          f"{_dataview.MAX_SOURCES} of {len(sources)} sources; pass include= to "
                          "choose them", RuntimeWarning, stacklevel=3)
            sources = sources[:_dataview.MAX_SOURCES]
        if sources:
            e = _dataview.REGISTRY.register(prefix + name, "heatmap", sources, rate=rate,
                                            options={"cmap": CMAPS[cmap]})
            h._entries.append(e)
            h.groups.append(e.name)
        out.append(e)
    return out


def watch(model, activations: bool = True, gradients: bool = False, every: int = 1,
          include=None, rate=None, groups=True) -> WatchHandle:
    import torch
    if not isinstance(model, torch.nn.Module):
        raise TypeError("caliper.watch expects a torch.nn.Module")
    every = int(every)
    if every < 1:
        raise ValueError("every must be >= 1")
    pkg = importlib.import_module(__package__)       # call caliper.show late (patchable)
    pick = _selector(include)
    h = WatchHandle()
    picked = [(mname or type(model).__name__.lower(), mod) for mname, mod in model.named_modules()
              if pick(mname, mod)]
    gw = ga = gg = None
    if groups:
        prefix = f"{groups}." if isinstance(groups, str) else ""
        gw, ga, gg = _register_groups(h, picked, activations, gradients, rate, prefix)
    for label, mod in picked:
        hooked = False
        wsrc = frozenset(n for n in (f"{label}.{pname}" for pname, p in
                                     mod.named_parameters(recurse=False) if p.ndim >= 2)
                         if n in gw.sources) if gw is not None else frozenset()
        if activations:
            view = f"{label}.act"

            def fwd(m, inp, out, view=view, label=label, wsrc=wsrc, count=[0]):
                count[0] += 1
                if every > 1 and count[0] % every:
                    return
                t = out
                if isinstance(t, (tuple, list)):
                    t = next((x for x in t if isinstance(x, torch.Tensor)), None)
                if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
                    pkg.show(t, view, rate=rate)
                    if ga is not None:
                        pkg._dv_hook(ga, label, t)
                for src in wsrc:              # usually one: the module's weight
                    pkg._dv_hook(gw, src, None)
            h._handles.append(mod.register_forward_hook(fwd))
            h.views.append(view)
            hooked = True
        if gradients:
            for pname, p in mod.named_parameters(recurse=False):
                if p.ndim < 2 or not p.requires_grad:
                    continue                  # biases / norms: not heatmap material
                view = f"{label}.{pname}.grad"
                src = f"{label}.{pname}"
                wtrigger = gw is not None and not activations

                def on_grad(g, view=view, src=src, wtrigger=wtrigger, count=[0]):
                    count[0] += 1
                    if every > 1 and count[0] % every:
                        return
                    if g is not None:
                        g2 = g.reshape(g.shape[0], -1) if g.ndim > 2 else g
                        pkg.show(g2, view, cmap="rdbu", rate=rate)
                        if gg is not None:
                            pkg._dv_hook(gg, src, g2)
                    if wtrigger:
                        pkg._dv_hook(gw, src, None)

                if hasattr(p, "register_post_accumulate_grad_hook"):
                    hd = p.register_post_accumulate_grad_hook(lambda q, f=on_grad: f(q.grad))
                else:
                    hd = p.register_hook(lambda g, f=on_grad: f(g))
                h._handles.append(hd)
                h.views.append(view)
                hooked = True
        if hooked:
            h.modules.append(label)
    return h
