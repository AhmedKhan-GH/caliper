"""Named device tensors for the existing producer runtime."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import re

_TENSORS = {}
_VERSIONS = {}
_VERSION = 0
_LEASE = ContextVar("caliper_publication_lease", default=None)
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
# These documented inputs can await their first trace or example query.
INTERPRETATION_TENSORS = frozenset({
    "trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f",
    "trace.label_text", "trace.probs", "trace.logit_tokens", "trace.logit_label_text",
    "examples.tokens", "examples.activations", "examples.example_ids",
    "examples.valid_tokens", "examples.label_text",
})


class _WatchChanged(RuntimeError):
    code = "interpret.watch_changed"


@dataclass(frozen=True)
class _WatchLease:
    epoch: int
    model: object


def _check_lease(lease):
    """Check ownership under the program lock without model work or service waits."""
    from .. import _programs
    current = _programs._STATE["model"]
    if (lease.epoch != _programs._WATCH_EPOCH or
            (current is not None and lease.model is not current)):
        raise _WatchChanged("The watch changed before interpretation publication.")


@contextmanager
def _publication_scope(model=None):
    """Carry the request's watch lifetime through lazy resources and nested traces."""
    from .. import _programs
    with _programs._LOCK:
        parent = _LEASE.get()
        if parent is None:
            lease = _WatchLease(_programs._WATCH_EPOCH,
                                _programs._STATE["model"] if model is None else model)
        else:
            if model is not None and parent.model is not None and model is not parent.model:
                raise _WatchChanged("The interpretation model does not own this publication request.")
            lease = _WatchLease(parent.epoch, parent.model if model is None else model)
        _check_lease(lease)
    token = _LEASE.set(lease)
    try:
        yield
    finally:
        _LEASE.reset(token)


def publish(name, tensor):
    """Bind ``name`` to tensor storage for read-only ``live:<name>`` imports.

    Publication preserves the name, dtype, device, and storage. It detaches autograd.
    A later call with the same name replaces the binding at the next producer pass.
    The current watch owns the binding. Before a watch, the next watch owns it.
    Watch removal or replacement clears bindings and releases producer imports.
    This function returns ``None`` and does not load the native runtime.
    """
    _validate_name(name)
    _publish_many({name: tensor})


def _validate_name(name):
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("a publication name must contain letters, digits, dots, hyphens, or underscores "
                         "and start with a letter, digit, or underscore")


def _publish_many(values):
    """Admit and replace a complete tensor group under the program lock."""
    global _VERSION
    from .. import _programs
    prepared = _prepare_many(values)
    with _programs._LOCK:
        _check_lease(prepared.lease)
        model = _programs._STATE["model"]
        for name in prepared.values:
            if model is not None and _model_name(model, name):
                raise ValueError(f"publication {name!r} conflicts with a model tensor")
        _TENSORS.update(prepared.values)
        for name in prepared.values:
            _VERSION += 1
            _VERSIONS[name] = _VERSION


def _prepare_many(values):
    """Detach a private tensor group and retain its original watch lease."""
    import torch
    from .. import _programs

    detached = {}
    for name, tensor in values.items():
        _validate_name(name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("publish expects a torch.Tensor")
        detached[name] = tensor.detach()
    with _programs._LOCK:
        lease = _LEASE.get() or _WatchLease(_programs._WATCH_EPOCH, _programs._STATE["model"])
        _check_lease(lease)
    return _PreparedPublication(detached, lease)


@dataclass
class _PreparedPublication:
    values: dict
    lease: _WatchLease

    @contextmanager
    def transaction(self):
        """Admit all names before a caller commits under its own inner lock."""
        global _VERSION
        from .. import _programs
        with _programs._LOCK:
            _check_lease(self.lease)
            model = _programs._STATE["model"]
            for name in self.values:
                if model is not None and _model_name(model, name):
                    raise ValueError(f"publication {name!r} conflicts with a model tensor")
            # Allocate maps before either public state can change.
            tensors = dict(_TENSORS, **self.values)
            versions = dict(_VERSIONS)
            version = _VERSION
            for name in self.values:
                version += 1
                versions[name] = version

            def commit():
                global _TENSORS, _VERSIONS, _VERSION
                _TENSORS, _VERSIONS, _VERSION = tensors, versions, version

            yield commit

    def commit(self):
        with self.transaction() as commit:
            commit()


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
