"""caliper.watch(model): views of a module's activations and gradients.

Forward hooks show each selected module's output as ``<module>.act`` (a channel
mosaic for (N, C, H, W) outputs, a heatmap for (N, F)); gradient hooks show
each selected parameter's gradient as ``<module>.<param>.grad`` (a heatmap,
higher-rank weights flattened to (out, -1)). Every view goes through
caliper.show, so it is rate limited per view (default 60 Hz) and near free
while nothing is attached; ``every=N`` also skips N-1 of N hook calls.
"""
from __future__ import annotations

import fnmatch
import importlib

__all__ = ["watch", "WatchHandle"]


class WatchHandle:
    """The hooks of one watch() call. ``remove()`` takes them all off."""

    def __init__(self):
        self._handles: list = []
        self.views: list[str] = []          # view names the hooks show into
        self.modules: list[str] = []        # module names that got a hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

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


def watch(model, activations: bool = True, gradients: bool = False, every: int = 1,
          include=None, rate=None) -> WatchHandle:
    import torch
    if not isinstance(model, torch.nn.Module):
        raise TypeError("caliper.watch expects a torch.nn.Module")
    every = int(every)
    if every < 1:
        raise ValueError("every must be >= 1")
    pkg = importlib.import_module(__package__)       # call caliper.show late (patchable)
    pick = _selector(include)
    h = WatchHandle()
    for mname, mod in model.named_modules():
        label = mname or type(model).__name__.lower()
        if not pick(mname, mod):
            continue
        hooked = False
        if activations:
            view = f"{label}.act"

            def fwd(m, inp, out, view=view, count=[0]):
                count[0] += 1
                if every > 1 and count[0] % every:
                    return
                t = out
                if isinstance(t, (tuple, list)):
                    t = next((x for x in t if isinstance(x, torch.Tensor)), None)
                if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
                    pkg.show(t, view, rate=rate)
            h._handles.append(mod.register_forward_hook(fwd))
            h.views.append(view)
            hooked = True
        if gradients:
            for pname, p in mod.named_parameters(recurse=False):
                if p.ndim < 2 or not p.requires_grad:
                    continue                  # biases / norms: not heatmap material
                view = f"{label}.{pname}.grad"

                def on_grad(g, view=view, count=[0]):
                    count[0] += 1
                    if every > 1 and count[0] % every:
                        return
                    if g is not None:
                        pkg.show(g.reshape(g.shape[0], -1) if g.ndim > 2 else g, view,
                                 cmap="rdbu", rate=rate)

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
