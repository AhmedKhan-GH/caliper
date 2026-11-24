"""Named device tensors for the existing producer runtime."""
from __future__ import annotations

import re

_TENSORS = {}
_VERSIONS = {}
_VERSION = 0
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
# These documented inputs can await their first trace or example query.
INTERPRETATION_TENSORS = frozenset({
    "trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f",
    "trace.label_text", "trace.probs", "trace.logit_tokens", "trace.logit_label_text",
    "examples.tokens", "examples.activations", "examples.example_ids",
    "examples.valid_tokens", "examples.label_text",
})


def publish(name, tensor):
    """Bind ``name`` to tensor storage for read-only ``live:<name>`` imports.

    Publication preserves the name, dtype, device, and storage. It detaches autograd.
    A later call with the same name replaces the binding at the next producer pass.
    The current watch owns the binding. Before a watch, the next watch owns it.
    Watch removal or replacement clears bindings and releases producer imports.
    This function returns ``None`` and does not load the native runtime.
    """
    global _VERSION
    import torch
    from .. import _programs

    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("a publication name must contain letters, digits, dots, hyphens, or underscores "
                         "and start with a letter, digit, or underscore")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("publish expects a torch.Tensor")
    with _programs._LOCK:
        model = _programs._STATE["model"]
        if model is not None and _model_name(model, name):
            raise ValueError(f"publication {name!r} conflicts with a model tensor")
        _TENSORS[name] = tensor.detach()
        _VERSION += 1
        _VERSIONS[name] = _VERSION


def _model_name(model, name):
    """Whether the name belongs to a parameter, buffer, gradient, or module output."""
    parameters = dict(model.named_parameters())
    return (name in parameters or name in dict(model.named_buffers()) or
            (name.endswith(".grad") and name[:-5] in parameters) or
            (name.endswith(".act") and name[:-4] in dict(model.named_modules())))


def _resolve(name):
    return _TENSORS.get(name)


def _clear():
    _TENSORS.clear()
    _VERSIONS.clear()
