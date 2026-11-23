"""The ablation of intervention points through module hooks (``design/gpu-programs.md``, Interventions).

``TorchAdapter.points`` and ``TorchAdapter.ablate`` in ``caliper._adapter`` call these two
functions, which are the only torch code of the interventions. ``caliper._controls`` applies a
CONTROL of op ``intervene`` at the next step boundary through them, and sends the points and the
interventions in the key ``interventions`` of CONTROLS.

A POINT is a module of ``model.named_modules()`` by its dotted name, and a head of an attention
module as ``<module>#<h>``. The root module has no name and is not a point. A module gives heads
when it has an integer ``num_heads``, ``n_heads`` or ``n_head`` and exactly one direct child
``nn.Linear`` named ``out_proj``, ``o_proj``, ``c_proj`` or ``proj`` whose ``in_features`` divides
by the head count. ``nn.MultiheadAttention`` gives no heads: it hands ``out_proj.weight`` to its
functional forward and never calls ``out_proj``, so a hook on that child never runs and a head's
ablation would do nothing.

AN ABLATION is one hook. A module's ablation is a forward hook that gives every floating-point
tensor of the output as zeros. A head's ablation is a forward pre-hook on the output projection
that zeroes that head's slice of the last dimension of its input. Both use ``masked_fill``, which
keeps the autograd graph (``backward()`` still works when the ablated output is all that the loss
sees, and an in-place operation after it is still allowed) and gives exact zeros where the input
holds NaN or infinity. A hook never raises: an error leaves the tensor as it was, counts in
:attr:`Ablation.errors` with its text, and the caller ends the ablation at its next step.
"""
from __future__ import annotations

import weakref

__all__ = ["points", "ablate", "record_shapes", "Ablation", "HEAD_SEPARATOR"]

#: The separator of a head's point: ``<module>#<h>``.
HEAD_SEPARATOR = "#"

_HEAD_COUNTS = ("num_heads", "n_heads", "n_head")
_PROJECTIONS = ("out_proj", "o_proj", "c_proj", "proj")

# model -> {module name: the shape of its last output}, filled by record_shapes()
_shapes: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _head_count(module) -> int:
    for attr in _HEAD_COUNTS:
        v = getattr(module, attr, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return 0


def _projection(module, heads: int):
    """The one output projection of ``module`` that splits into ``heads`` slices, or None."""
    import torch
    if heads <= 0 or isinstance(module, torch.nn.MultiheadAttention):
        return None
    children = getattr(module, "_modules", {}) or {}
    found = [children[n] for n in _PROJECTIONS
             if isinstance(children.get(n), torch.nn.Linear)]
    if len(found) != 1 or found[0].in_features % heads:
        return None
    return found[0]


def _type_name(module) -> str:
    return type(module).__name__[:63] or "Module"


def points(model) -> list:
    """List the intervention points of ``model``; see ``TorchAdapter.points``."""
    shapes = _shapes.get(model, {})
    out = []
    for name, mod in model.named_modules():
        if not name:
            continue
        shape = shapes.get(name)
        out.append({"point": name, "kind": "module", "module": name, "head": None,
                    "type": _type_name(mod), "shape": list(shape) if shape is not None else None})
        heads = _head_count(mod)
        if _projection(mod, heads) is not None:
            out += [{"point": f"{name}{HEAD_SEPARATOR}{h}", "kind": "head", "module": name, "head": h,
                     "type": _type_name(mod), "shape": None} for h in range(heads)]
    from .interp._points import semantic_points
    return out + semantic_points(model, required=False)


def _find(model, point: str):
    """``(module, head)`` of ``point``, with head ``None`` for a module, or KeyError."""
    if not isinstance(point, str) or not point:
        raise KeyError(point)
    name, sep, head = point.rpartition(HEAD_SEPARATOR)
    if not sep:
        name, head = point, None
    mod = dict(model.named_modules()).get(name) if name else None
    if mod is None:
        raise KeyError(point)
    if head is None:
        return mod, None
    if not head.isdigit() or (len(head) > 1 and head[0] == "0"):
        raise KeyError(point)
    h, heads = int(head), _head_count(mod)
    proj = _projection(mod, heads)
    if proj is None or h >= heads:
        raise KeyError(point)
    return proj, (h, proj.in_features // heads)


class Ablation:
    """One ablation's hook, as :func:`ablate` returns it.

    Calling it removes the hook; a second call does nothing.

    Attributes:
        point (str): The point it ablates.
        calls (int): How many times the hook ran.
        errors (int): How many times the hook failed and left the tensor as it was.
        error (str | None): The text of the last failure, or None.
    """

    def __init__(self, point: str):
        self.point = point
        self.calls = 0
        self.errors = 0
        self.error = None
        self._handle = None
        self._masks: dict = {}

    def __call__(self) -> None:
        h, self._handle = self._handle, None
        if h is not None:
            h.remove()
        self._masks.clear()

    @property
    def removed(self) -> bool:
        """Whether the hook is off the model."""
        return self._handle is None

    def _failed(self, e: BaseException) -> None:
        self.errors += 1
        self.error = f"{type(e).__name__}: {e}"[:300]

    def _zero(self, t, mask_key):
        import torch
        if isinstance(t, torch.Tensor):
            if not (t.is_floating_point() or t.is_complex()):
                return t
            mask = self._masks.get((t.device, mask_key))
            if mask is None:
                mask = self._mask(t.device, mask_key)
            return t.masked_fill(mask, 0)
        if isinstance(t, tuple) and hasattr(t, "_fields"):          # a named tuple
            return type(t)(*(self._zero(x, mask_key) for x in t))
        if isinstance(t, (tuple, list)):
            return type(t)(self._zero(x, mask_key) for x in t)
        if isinstance(t, dict):
            return {k: self._zero(v, mask_key) for k, v in t.items()}
        return t

    def _mask(self, device, mask_key):
        import torch
        if mask_key is None:
            mask = torch.ones((), dtype=torch.bool, device=device)
        else:
            (h, width), features = mask_key
            mask = torch.zeros(features, dtype=torch.bool, device=device)
            mask[h * width:(h + 1) * width] = True
        self._masks[(device, mask_key)] = mask
        return mask

    def _forward_hook(self, module, args, output):
        self.calls += 1
        try:
            return self._zero(output, None)
        except Exception as e:                    # never stop the forward pass
            self._failed(e)
            return None

    def _pre_hook(self, head, features):
        key = (head, features)

        def hook(module, args):
            self.calls += 1
            try:
                if not args:
                    return None
                x = args[0]
                if x.shape[-1] != features:
                    raise ValueError(f"the projection's input has {x.shape[-1]} features, "
                                     f"not {features}")
                return (self._zero(x, key),) + tuple(args[1:])
            except Exception as e:                # never stop the forward pass
                self._failed(e)
                return None
        return hook


def ablate(model, point: str):
    """Ablate one point of ``model`` and return the function that removes it; see
    ``TorchAdapter.ablate``. The returned :class:`Ablation` also counts the hook's calls and
    failures."""
    try:
        target, head = _find(model, point)
    except KeyError:
        from .interp._points import _architecture, _input
        try:
            probe = _architecture(model).get(point)
        except NotImplementedError:
            raise KeyError(point) from None
        if probe is None:
            raise KeyError(point)
        if probe.event == "pattern":
            raise ValueError("attention patterns are read-only probe points")
        target = dict(model.named_modules())[probe.module]
        a = Ablation(point)
        if probe.event == "output":
            a._handle = target.register_forward_hook(a._forward_hook)
        else:
            def hook(module, args, kwargs):
                a.calls += 1
                try:
                    value = a._zero(_input(args, kwargs), None)
                    if args:
                        return (value,) + args[1:], kwargs
                    key = "hidden_states" if "hidden_states" in kwargs else "x"
                    return args, dict(kwargs, **{key: value})
                except Exception as e:
                    a._failed(e)
                    return None
            a._handle = target.register_forward_pre_hook(hook, with_kwargs=True)
        return a
    a = Ablation(point)
    if head is None:
        a._handle = target.register_forward_hook(a._forward_hook)
    else:
        a._handle = target.register_forward_pre_hook(a._pre_hook(head, target.in_features))
    return a


def _shape_of(output):
    import torch
    t = output
    if isinstance(t, (tuple, list)):
        t = next((x for x in t if isinstance(x, torch.Tensor)), None)
    if isinstance(t, torch.Tensor):
        return tuple(int(d) for d in t.shape)
    return None


def record_shapes(model):
    """Record the output shape of each module of ``model`` at its next forward pass, for the
    ``shape`` of :func:`points`. Returns the function that removes the hooks; the caller calls it
    at the next step boundary. A hook only stores a shape and never raises."""
    store = _shapes.setdefault(model, {})
    handles = []

    def make(name):
        def hook(module, args, output):
            try:
                s = _shape_of(output)
                if s is not None:
                    store[name] = s
            except Exception:
                pass
        return hook

    try:
        for name, mod in model.named_modules():
            if name:
                handles.append(mod.register_forward_hook(make(name)))
    except BaseException:
        for h in handles:
            h.remove()
        raise

    def remove() -> None:
        while handles:
            handles.pop().remove()
    return remove
