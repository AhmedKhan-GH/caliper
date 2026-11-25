"""Validated interventions on live model tensors, with scoped batch selection."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import lru_cache
from itertools import islice
import hashlib
import json
import math
import sys
import threading
import time
import traceback
import weakref

import torch

from .. import _intervention_protocol as metadata
from ._frozen import _trace_features
from ._memory import CleanStop, allocation_bytes, family_memory_limit
from ._points import _architecture, _input, capture

MAX_POSITIONS = 32
MAX_VECTOR = 16384
MAX_PROMPT_BYTES = 4096
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_ENTRIES = 64
_SNAPSHOT_COMPARE_BYTES = 16 * 1024 * 1024
_SNAPSHOT_BATCH_BLOCKS = 8
_SNAPSHOT_FLAG_RESERVE = 64 * 1024
KINDS = ("ablate", "add", "scale", "clamp", "patch")
_SUSPENDED = ContextVar("caliper_interventions_suspended", default=frozenset())
_ROWS = ContextVar("caliper_intervention_rows", default={})
_MUTED_HOOKS = ContextVar("caliper_muted_intervention_hooks", default=frozenset())
_ACTIVE_REPLAYS = ContextVar("caliper_intervention_replays", default=())


class InterventionError(ValueError):
    """A request refusal with a stable wire error code."""

    def __init__(self, message, code="intervene.bad_request", *, resource=None):
        super().__init__(message)
        self.code = code
        self.resource = resource


class _IdentityMemory:
    """Retain the supplied guard and its lower limits for bounded readbacks."""

    def __init__(self, guard):
        self.guard = guard
        self.inner = getattr(guard, "_guard", guard)
        self.device = getattr(guard, "device", getattr(self.inner, "device", torch.device("cpu")))
        self._checked_current = None
        self._checked_peak = 0
        self.floor = 0

    @property
    def limit(self):
        return min(getattr(self.guard, "limit", math.inf), getattr(self.inner, "limit", math.inf))

    @property
    def peak(self):
        return max(self._checked_peak, getattr(self.guard, "peak", 0), getattr(self.inner, "peak", 0))

    @property
    def current(self):
        values = [getattr(owner, "current", None) for owner in (self.guard, self.inner)]
        values.append(self._checked_current)
        readings = [value for value in values if value is not None]
        return max(self.floor, max(readings) if readings else self.peak)

    def check(self, stage, *, force=True):
        # Some supplied guards accept only the stage. Do not bypass their latch.
        check = self.guard.check if force else getattr(self.guard, "_check_interval", self.guard.check)
        current = check(stage)
        if isinstance(current, (int, float)):
            self._checked_current = current
            self._checked_peak = max(self._checked_peak, current)


def _identity_memory(state, guard):
    memory = _IdentityMemory(guard) if guard is not None else None
    if any(value.device.type != "cpu" for value in state.values()):
        if memory is None or not math.isfinite(memory.limit):
            raise ValueError("device content identity requires a bounded process memory guard")
    return memory


def _content_identity(state, guard):
    from ._dictionary import _state_hash
    return _state_hash(state, memory=_identity_memory(state, guard))


def _source_input_identity(inputs, guard):
    """Canonicalize admitted prompt views without a change to weight admission."""
    from ._dictionary import _hash_admission, _state_hash
    # The shared prompt contract permits two tensors with at most 32 elements.
    if (set(inputs) - {"input_ids", "attention_mask"}
            or any(value.layout != torch.strided or value.ndim != 2
                   or value.shape[0] != 1 or not 1 <= value.shape[1] <= MAX_POSITIONS
                   for value in inputs.values())):
        raise InterventionError("patch source inputs exceed the canonicalization bounds")
    copies = {key for key, value in inputs.items()
              if not value.is_contiguous() or value.is_conj() or value.is_neg()}
    reserve = sum(inputs[key].numel() * inputs[key].element_size() for key in copies)
    memory = _identity_memory(inputs, guard)
    if memory is not None and reserve:
        memory.check("patch source input copy")
        # Retain the copy reserve even if the supplied guard reports stale usage.
        memory.floor = allocation_bytes(memory) + reserve
        _hash_admission(memory)
    canonical = {key: value.detach().clone(memory_format=torch.contiguous_format)
                 if key in copies else value for key, value in inputs.items()}
    return canonical, _state_hash(canonical, memory=memory)


def _dictionary_identifier(dictionary, guard):
    """Bind default restore compatibility to content, metadata, and actual TopK values."""
    from ._admission import GEMMA_ID
    from ._dictionary import _HASH_CHUNK_BYTES
    state = dictionary.state_dict()
    if (dictionary.metadata.get("model_id") == GEMMA_ID
            and any(tensor.device.type == "mps" for tensor in state.values())):
        # Full-hash fallback must preserve the same process cap and host reserve.
        guard = _DictionarySnapshot._memory(state, guard, device_compare=True)
        _DictionarySnapshot._admit(guard, 4 * _HASH_CHUNK_BYTES, force=True)
    return _dictionary_digest(dictionary, _content_identity(state, guard))


def _dictionary_digest(dictionary, content):
    identity = {"version": 1, "content": content,
                "metadata": {key: value for key, value in dictionary.metadata.items() if key != "quality"},
                "topk": [layer.k for layer in dictionary.layers]}
    activations = [getattr(layer, "activation", "topk") for layer in dictionary.layers]
    if any(activation != "topk" for activation in activations):
        identity["activations"] = activations
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _native_byte_compare():
    if sys.platform != "darwin":
        return None
    import ctypes
    compare = ctypes.CDLL("/usr/lib/libSystem.B.dylib").memcmp
    compare.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    compare.restype = ctypes.c_int
    return compare


def _cpu_byte_equal(left, right):
    """Compare complete bounded CPU byte views without a tensor reduction."""
    if (left.device.type != "cpu" or right.device.type != "cpu"
            or left.dtype != torch.uint8 or right.dtype != torch.uint8
            or left.shape != right.shape or not left.is_contiguous() or not right.is_contiguous()):
        raise ValueError("snapshot comparison requires equal contiguous CPU byte views")
    if left.numel() == 0:
        return True
    compare = _native_byte_compare()
    if compare is None:
        return torch.equal(left, right)
    # Both tensors retain their storage until the bounded native call returns.
    return compare(left.data_ptr(), right.data_ptr(), left.numel()) == 0


class _DeviceSnapshotMemory(_IdentityMemory):
    """Reserve host headroom for private MPS bytes and bounded device work."""

    def __init__(self, guard):
        super().__init__(guard)
        self.reserve = self.reserve_floor = 0

    @property
    def limit(self):
        return min(super().limit, 40_000_000_000)

    def check(self, stage, *, force=True):
        import psutil
        super().check(stage, force=force)
        # A stale process reader must not hide admitted but untouched pages.
        observed = max(getattr(self.guard, "current", 0) or 0,
                       getattr(self.inner, "current", 0) or 0,
                       self._checked_current or 0)
        pending = max(0, max(self.floor, self.reserve_floor) - observed)
        if psutil.virtual_memory().available - pending - self.reserve < 8_000_000_000:
            raise CleanStop("dictionary identity allocation exceeds the host reserve")


def _snapshot_workspace(block_bytes, blocks):
    # Four byte workspaces per queued block cover masks and reduction scratch.
    # Each flag also reserves allocator pages, its host copy, and dispatch data.
    return blocks * (4 * block_bytes + _SNAPSHOT_FLAG_RESERVE)


@contextmanager
def _snapshot_device_batch(device):
    """Drain a bounded queue before its autorelease scope ends, also on failure."""
    from ._memory import autorelease_scope
    completed = []
    with autorelease_scope(device):
        try:
            yield completed
        except BaseException as error:
            try:
                torch.mps.synchronize()
            except BaseException as cleanup:
                # Preserve a memory stop or device failure if cleanup also fails.
                raise error from cleanup
            raise
        else:
            if not completed:
                torch.mps.synchronize()


class _DictionarySnapshot:
    """Prove unchanged content with private bytes, without version-counter assumptions.

    Every lookup compares all live bytes. Gemma MPS snapshots retain MPS bytes
    and read back bounded reduction flags. Data aliases, NumPy writes, and
    inference tensors need no mutation notification. The snapshot never supplies
    model weights, and a mismatch requires reconfiguration.
    """

    def __init__(self, state, content, source_devices, *, device_compare=False):
        self.state, self.content = state, content
        self.source_devices = source_devices
        self.device_compare = device_compare

    @staticmethod
    def _memory(state, guard, *, device_compare=False):
        from ._dictionary import _canonical_device
        memory = _IdentityMemory(guard) if guard is not None else None
        for tensor in state.values():
            if tensor.device.type != "cpu":
                if memory is None or not math.isfinite(memory.limit):
                    raise ValueError("device content identity requires a bounded process memory guard")
                if _canonical_device(tensor.device) != _canonical_device(memory.device):
                    raise ValueError("dictionary identity requires a guard for its device")
        if device_compare:
            memory = guard if isinstance(guard, _DeviceSnapshotMemory) else _DeviceSnapshotMemory(guard)
        return memory

    @staticmethod
    def _admit(memory, reserve, *, force=False, floor=0):
        from ._memory import CleanStop
        if memory is not None:
            if isinstance(memory, _DeviceSnapshotMemory):
                memory.reserve, memory.reserve_floor = reserve, floor
            memory.check("dictionary identity", force=force)
            if allocation_bytes(memory, floor=floor) + reserve > memory.limit:
                raise CleanStop("dictionary identity allocation exceeds the process memory limit")

    @classmethod
    def create(cls, dictionary, identifier, guard):
        from ._admission import GEMMA_ID
        from ._dictionary import _dense_state, _HASH_CHUNK_BYTES, _UNVERIFIED_CPU_BYTES
        from ._memory import autorelease_scope
        state = _dense_state(dictionary.state_dict())
        source_devices = {name: tensor.device for name, tensor in state.items()}
        device_compare = (dictionary.metadata.get("model_id") == GEMMA_ID
                          and any(device.type == "mps" for device in source_devices.values()))
        sizes = [tensor.numel() * tensor.element_size() for tensor in state.values()]
        total, largest = sum(sizes), max(sizes, default=0)
        memory = cls._memory(state, guard, device_compare=device_compare)
        # Account for the retained copy, hash chunks, and comparison workspace.
        scratch = max(4 * min(largest, _SNAPSHOT_COMPARE_BYTES), 4 * _HASH_CHUNK_BYTES)
        if device_compare:
            scratch = max(scratch, _snapshot_workspace(min(largest, _SNAPSHOT_COMPARE_BYTES),
                                                      _SNAPSHOT_BATCH_BLOCKS))
        if memory is None:
            if any(t.device.type != "cpu" for t in state.values()) or total + scratch > _UNVERIFIED_CPU_BYTES:
                return None
            initial = 0
        else:
            cls._admit(memory, 0, force=True)
            initial = allocation_bytes(memory)
            if not math.isfinite(memory.limit) or initial + total + scratch > memory.limit:
                return None
        snapshot, copied, result = {}, 0, None
        tensor = current = original = None
        try:
            for name, tensor in state.items():
                cls._admit(memory, total - copied + scratch, floor=initial + copied)
                if not device_compare or tensor.device.type != "mps":
                    with autorelease_scope(tensor.device):
                        snapshot[name] = tensor.detach().clone()
                    copied += tensor.numel() * tensor.element_size()
                    continue
                # Count the entire private destination before its first write.
                with autorelease_scope(tensor.device):
                    snapshot[name] = torch.empty_like(tensor)
                copied += tensor.numel() * tensor.element_size()
                current = tensor.detach().reshape(-1).view(torch.uint8)
                original = snapshot[name].reshape(-1).view(torch.uint8)
                stride = _SNAPSHOT_COMPARE_BYTES * _SNAPSHOT_BATCH_BLOCKS
                for batch in range(0, current.numel(), stride):
                    with _snapshot_device_batch(tensor.device):
                        for start in range(batch, min(batch + stride, current.numel()), _SNAPSHOT_COMPARE_BYTES):
                            stop = min(start + _SNAPSHOT_COMPARE_BYTES, current.numel())
                            cls._admit(memory, total - copied + scratch, floor=initial + copied)
                            original[start:stop].copy_(current[start:stop])
            if memory is not None:
                memory.floor = initial + copied
            cls._admit(memory, scratch, force=True, floor=initial + copied)
            content = _content_identity(snapshot, memory)
            result = cls(snapshot, content, source_devices, device_compare=device_compare)
            cls._admit(memory, scratch, force=True, floor=initial + copied)
            # The copied bytes must reproduce the configured full-content identity.
            if _dictionary_digest(dictionary, content) != identifier or not result.matches(dictionary, identifier, memory):
                raise InterventionError("dictionary content changed. Reconfigure before use",
                                        "intervene.unknown_dictionary")
            return result
        except BaseException as error:
            snapshot.clear()
            result = None
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            # A retained exception must not keep a partial destination alive.
            tensor = current = original = None

    def _device_equal(self, blocks, memory):
        """Compare one bounded byte batch with one flag readback."""
        largest = max(left.numel() for left, _ in blocks)
        reserve = _snapshot_workspace(largest, len(blocks))
        initial = allocation_bytes(memory)
        mask = flags = host = left = right = template = None
        try:
            with _snapshot_device_batch(blocks[0][0].device) as completed:
                self._admit(memory, reserve, floor=initial)
                template = max((left for left, _ in blocks), key=lambda value: value.numel())
                mask = torch.empty_like(template, dtype=torch.bool)
                self._admit(memory, reserve, floor=initial)
                flags = torch.empty_like(template[:1].expand(len(blocks)), dtype=torch.bool)
                for index, (left, right) in enumerate(blocks):
                    self._admit(memory, reserve, floor=initial)
                    torch.eq(left, right, out=mask[:left.numel()])
                    self._admit(memory, reserve, floor=initial)
                    torch.all(mask[:left.numel()], out=flags[index])
                self._admit(memory, reserve, floor=initial)
                host = flags.to(device="cpu")
                completed.append(True)
                equal = all(host.tolist())
                self._admit(memory, 0, force=True)
                return equal
        except BaseException as error:
            # Backend tracebacks can retain the output mask through out views.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            mask = flags = host = left = right = template = None

    def _device_matches(self, state, memory):
        def blocks():
            for name, tensor in state.items():
                if tensor.device.type != "mps":
                    continue
                current = tensor.detach().reshape(-1).view(torch.uint8)
                original = self.state[name].reshape(-1).view(torch.uint8)
                for start in range(0, current.numel(), _SNAPSHOT_COMPARE_BYTES):
                    yield current[start:start + _SNAPSHOT_COMPARE_BYTES], original[start:start + _SNAPSHOT_COMPARE_BYTES]

        pending = blocks()
        batch = []
        equal = True
        try:
            while batch := list(islice(pending, _SNAPSHOT_BATCH_BLOCKS)):
                # Do not short-circuit later batches after a content mismatch.
                equal = self._device_equal(batch, memory) and equal
                batch.clear()
            return equal
        finally:
            batch.clear()
            pending.close()

    def matches(self, dictionary, identifier, guard):
        from ._dictionary import _dense_state
        from ._memory import autorelease_scope
        state = _dense_state(dictionary.state_dict())
        if state.keys() != self.state.keys() or _dictionary_digest(dictionary, self.content) != identifier:
            return False
        memory = self._memory(state, guard, device_compare=self.device_compare)
        if self.device_compare:
            memory.floor = max(memory.floor, sum(t.numel() * t.element_size()
                                                for t in (*state.values(), *self.state.values())))
        self._admit(memory, 0, force=True)
        for name, tensor in state.items():
            saved = self.state[name]
            if (tensor.shape != saved.shape or tensor.dtype != saved.dtype
                    or tensor.device != self.source_devices[name]):
                return False
        equal = self._device_matches(state, memory) if self.device_compare else True
        for name, tensor in state.items():
            saved = self.state[name]
            if self.device_compare and tensor.device.type == "mps":
                continue
            # Byte views also distinguish signed zero and preserve NaN bits.
            current = tensor.detach().reshape(-1).view(torch.uint8)
            original = saved.reshape(-1).view(torch.uint8)
            for start in range(0, current.numel(), _SNAPSHOT_COMPARE_BYTES):
                stop = min(start + _SNAPSHOT_COMPARE_BYTES, current.numel())
                self._admit(memory, 4 * (stop - start))
                with autorelease_scope(tensor.device):
                    block = current[start:stop]
                    if tensor.device != saved.device:
                        block = block.to(device="cpu")
                        equal = _cpu_byte_equal(block, original[start:stop])
                    else:
                        equal = torch.equal(block, original[start:stop]) and equal
                    del block
                    if not equal:
                        return False
        return equal


def _indices(value, name, *, limit=None):
    if not isinstance(value, (list, tuple)) or any(
        type(i) is not int or i < 0 or (limit is not None and i >= limit) for i in value
    ) or len(set(value)) != len(value) or (limit is not None and len(value) > limit):
        raise InterventionError(f"{name} must contain unique nonnegative integer indices")
    return tuple(value)


def _text(value, name, maximum, *, code="intervene.bad_request"):
    if not metadata.text(value, maximum, False):
        raise InterventionError(f"{name} must contain 1 to {maximum} UTF-8 bytes", code)
    return value


def _finite(value, name):
    if not metadata.finite(value):
        raise InterventionError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise InterventionError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise InterventionError(f"{name} must be a finite number")
    return result


def selected_rows(model):
    """Return None for all rows, or a tuple for the current model context."""
    if id(model) in _SUSPENDED.get():
        return ()
    return _ROWS.get().get(id(model))


def _hook_muted(hook):
    return hook in _MUTED_HOOKS.get()


@contextmanager
def _mute_hooks(hooks):
    """Mute these hook objects in this context without a lifecycle change."""
    token = _MUTED_HOOKS.set(_MUTED_HOOKS.get().union(hooks))
    try:
        yield
    finally:
        _MUTED_HOOKS.reset(token)


@contextmanager
def suspend(model):
    """Suspend every intervention on this model within the current context."""
    token = _SUSPENDED.set(_SUSPENDED.get() | {id(model)})
    try:
        yield
    finally:
        _SUSPENDED.reset(token)


@contextmanager
def _preserve_captures(model):
    """Keep an internal source forward out of existing probe capture maps."""
    from ._points import _ACTIVE
    saved = [(values, dict(values), {key: dict(part) for key, part in values._pending.items()})
             for values in list(_ACTIVE.get(model, {}).values())]
    try:
        yield
    finally:
        for values, tensors, pending in saved:
            if values._closed:
                continue
            values.clear()
            values.update(tensors)
            for key, part in values._pending.items():
                part.clear()
                part.update(pending[key])


@contextmanager
def rows(model, indices):
    """Restrict interventions to batch rows, intersected with any outer scope."""
    chosen = _indices(indices, "rows")
    outer = _ROWS.get().get(id(model))
    if outer is not None:
        chosen = tuple(i for i in chosen if i in outer)
    token = _ROWS.set({**_ROWS.get(), id(model): chosen})
    try:
        yield
    finally:
        _ROWS.reset(token)


def row_mask(model, tensor):
    """Build the batch mask for legacy module and head ablation."""
    chosen = selected_rows(model)
    if chosen is None:
        return None
    if tensor.ndim == 0:
        raise InterventionError("batch row selection requires a batch dimension")
    if any(i >= tensor.shape[0] for i in chosen):
        raise InterventionError("a selected batch row is outside the tensor")
    mask = torch.zeros(tensor.shape[0], dtype=torch.bool, device=tensor.device)
    mask[list(chosen)] = True
    return mask.reshape((-1,) + (1,) * (tensor.ndim - 1))


class InterventionService:
    """Validate specifications and install removable hooks on one real model.

    Dictionary weights stay on their device. Specifications contain only JSON values.
    Clean patch sources use a separate forward pass under ``suspend()``.
    """

    def __init__(self, model, *, adapter=None):
        self.model = model
        self.adapter = adapter
        self._ablation_points = None
        self.dictionary = None
        self.dictionary_id = None
        self._default_dictionary_id = False
        self._dictionary_snapshot = None
        self._identity_lock = threading.Lock()
        self._identity_checks = 0
        self._identity_seconds = 0.
        self.tokenizer = None
        self.resource_guard = None
        self._cache = OrderedDict()
        self._cache_identity = None
        self.preparing_source = False
        self._hooks = weakref.WeakSet()
        self._retiring_sources = frozenset()

    def suspend(self):
        return suspend(self.model)

    def rows(self, indices):
        return rows(self.model, indices)

    @property
    def suspended(self):
        return id(self.model) in _SUSPENDED.get()

    def configure(self, *, dictionary=None, tokenizer=None, dictionary_id=None, resource_guard=None):
        """Bind a validated dictionary and tokenizer before requests arrive.

        Reconfiguration removes existing hooks. Controls must also end their records.
        A failed validation leaves the prior configuration and hooks intact.
        The default identifier covers dictionary content and metadata except quality.
        A content change requires reconfiguration before another feature request.
        The caller owns an explicit identifier and its restore compatibility.
        """
        if resource_guard is not None and not callable(getattr(resource_guard, "check", None)):
            raise InterventionError("resource_guard must provide check(stage, **kwargs)")
        default_dictionary_id = dictionary is not None and dictionary_id is None
        if dictionary is not None:
            from ._dictionary import Dictionary, model_identity
            if not isinstance(dictionary, Dictionary):
                raise InterventionError("dictionary must be a loaded Dictionary", "intervene.unknown_dictionary")
            meta = dictionary.metadata
            limit = None if resource_guard is None else family_memory_limit(self.model, memory=resource_guard)
            if meta.get("model_hash") != model_identity(self.model, memory=resource_guard, memory_limit_gb=limit):
                raise InterventionError("dictionary model identity mismatch", "intervene.unknown_dictionary")
            probes = _architecture(self.model)
            reads = [p.point for p in probes.values() if p.point.endswith(".mlp_in")]
            writes = [p.point for p in probes.values() if p.point.endswith(".mlp_out")]
            width = self.model.config.hidden_size
            if (meta.get("read_points") != reads or meta.get("write_points") != writes
                    or len(dictionary.layers) != len(writes) or meta.get("width") != width):
                raise InterventionError("dictionary probes or dimensions differ from the model")
            for layer in dictionary.layers:
                if (layer.W_enc.ndim != 2 or layer.W_dec.shape != layer.W_enc.shape
                        or layer.W_dec.shape[1] != width
                        or layer.b_enc.shape != (layer.W_enc.shape[0],)
                        or layer.b_dec.shape != (width,)):
                    raise InterventionError("dictionary weight dimensions differ from its probes")
            if default_dictionary_id:
                dictionary_id = _dictionary_identifier(dictionary, resource_guard)
            _text(dictionary_id, "dictionary identifier", 255)
        elif dictionary_id is not None:
            raise InterventionError("dictionary identifier requires a dictionary", "intervene.unknown_dictionary")
        if tokenizer is not None and not callable(tokenizer):
            raise InterventionError("tokenizer must be callable")
        self.close()
        self.dictionary, self.dictionary_id, self.tokenizer = dictionary, dictionary_id, tokenizer
        self._default_dictionary_id = default_dictionary_id
        self.resource_guard = resource_guard

    def invalidate_cache(self):
        """Release patch sources after a prompt or model change."""
        self._cache.clear()
        self._cache_identity = None

    def _prepare_dictionary_identity(self):
        """Admit an optional exact-byte snapshot for repeated hosted validation."""
        self._dictionary_snapshot = None
        if self._default_dictionary_id:
            self._dictionary_snapshot = _DictionarySnapshot.create(
                self.dictionary, self.dictionary_id, self.resource_guard)

    def _dictionary_matches(self):
        started = time.perf_counter()
        try:
            if self._dictionary_snapshot is not None:
                return self._dictionary_snapshot.matches(self.dictionary, self.dictionary_id, self.resource_guard)
            return _dictionary_identifier(self.dictionary, self.resource_guard) == self.dictionary_id
        finally:
            seconds = time.perf_counter() - started
            with self._identity_lock:
                self._identity_checks += 1
                self._identity_seconds += seconds

    def _dictionary_status(self):
        """Return cumulative validation costs, separate from request completion."""
        snapshot = self._dictionary_snapshot
        mode = "snapshot" if snapshot is not None else "hash" if self._default_dictionary_id else "caller"
        size = sum(t.numel() * t.element_size() for t in snapshot.state.values()) if snapshot is not None else 0
        with self._identity_lock:
            return {"mode": mode, "snapshot_bytes": size,
                    "checks": self._identity_checks, "seconds": self._identity_seconds}

    def close(self):
        for hook in list(self._hooks):
            hook()
        self.invalidate_cache()
        self._dictionary_snapshot = None

    def _target(self, point):
        from .._ablate import _find, _shapes
        try:
            probes = _architecture(self.model)
        except NotImplementedError:
            probes = {}
        probe = probes.get(point)
        if probe is not None:
            if probe.event == "pattern":
                raise InterventionError("attention patterns are read-only points")
            return (dict(self.model.named_modules())[probe.module], probe.event,
                    self.model.config.hidden_size, probe)
        try:
            module, head = _find(self.model, point)
        except KeyError:
            raise InterventionError("unknown intervention point", "intervene.unknown_point") from None
        if head is not None:
            raise InterventionError("head points support ablate only")
        shape = _shapes.get(self.model, {}).get(point)
        width = getattr(module, "out_features", shape[-1] if shape else None)
        if width is None:
            raise InterventionError("point width is unknown; capture its output first")
        return module, "output", width, None

    def validate(self, request, *, resources=True):
        """Return a detached, canonical specification without a hook or forward pass."""
        if not isinstance(request, dict):
            raise InterventionError("intervention set must be an object")
        op, kind = request.get("op"), request.get("kind")
        if op is not None and kind is not None and op != kind:
            raise InterventionError("op and kind conflict")
        kind = kind if kind is not None else op
        if not isinstance(kind, str) or kind not in KINDS:
            raise InterventionError("unsupported intervention kind", "intervene.op_reserved")
        fields = {"point", "op", "kind", "from_step", "until_step"}
        fields |= {"add": {"vector", "coefficient", "positions"},
                   "scale": {"feature", "multiplier", "positions", "dictionary"},
                   "clamp": {"feature", "value", "positions", "dictionary"},
                   "patch": {"source_prompt", "source_position", "positions"},
                   "ablate": set()}[kind]
        if set(request) - fields:
            raise InterventionError("unexpected intervention fields")
        point = request.get("point")
        _text(point, "point", 255, code="intervene.unknown_point")
        spec = deepcopy(request)
        spec.update(op=kind, kind=kind)
        for key in ("from_step", "until_step"):
            value = spec.get(key)
            if value is not None and (type(value) is not int or not 0 <= value < 2**53):
                raise InterventionError("schedule values must be nonnegative steps", "intervene.bad_range")
        if (spec.get("from_step") is not None and spec.get("until_step") is not None
                and spec["until_step"] <= spec["from_step"]):
            raise InterventionError("until_step must exceed from_step", "intervene.bad_range")
        # Check wire bounds before any integer-to-float conversion or resource access.
        if not metadata.specification(spec):
            raise InterventionError("intervention metadata exceeds the wire bounds")
        if not resources:
            return spec
        if kind == "ablate":
            from .._ablate import points
            names = self._ablation_points
            if names is None:
                try:
                    listed = points(self.model) if self.adapter is None else self.adapter.points(self.model)
                    names = {p["point"] for p in listed}
                except Exception as error:
                    raise InterventionError("cannot enumerate intervention points", "intervene.unknown_point") from error
            if point not in names:
                raise InterventionError("unknown intervention point", "intervene.unknown_point")
            if point.endswith(".attn_pattern"):
                raise InterventionError("attention patterns are read-only points")
            return spec
        module, _, width, probe = self._target(point)
        if "positions" in spec:
            spec["positions"] = list(_indices(spec["positions"], "positions", limit=MAX_POSITIONS))
        if kind == "add":
            vector = spec.get("vector")
            if not isinstance(vector, list) or len(vector) != width or not 0 < len(vector) <= MAX_VECTOR:
                raise InterventionError("vector must match the point width")
            spec["vector"] = [_finite(v, "vector element") for v in vector]
            spec["coefficient"] = _finite(spec.get("coefficient"), "coefficient")
            if not all(math.isfinite(v * spec["coefficient"]) for v in spec["vector"]):
                raise InterventionError("scaled vector must be finite")
        elif kind in ("scale", "clamp"):
            if self.dictionary is None or spec.get("dictionary", self.dictionary_id) != self.dictionary_id:
                raise InterventionError("unknown or absent dictionary", "intervene.unknown_dictionary")
            if self._default_dictionary_id and not self._dictionary_matches():
                raise InterventionError("dictionary content changed. Reconfigure before use",
                                        "intervene.unknown_dictionary")
            if probe is None or not point.endswith(".mlp_out"):
                raise InterventionError("features require an mlp_out point")
            from ._dictionary import fallback_metadata
            meta = self.dictionary.metadata
            fallback = fallback_metadata(meta) if "fallback" in meta.get("training", {}) else None
            if fallback is not None and probe.layer not in fallback["active_layers"]:
                raise InterventionError("dictionary excludes this feature layer", "intervene.unknown_feature")
            feature = spec.get("feature")
            if type(feature) is not int or not 0 <= feature < self.dictionary.layers[probe.layer].W_dec.shape[0]:
                raise InterventionError("unknown dictionary feature", "intervene.unknown_feature")
            key = "multiplier" if kind == "scale" else "value"
            spec[key] = _finite(spec.get(key), key)
            spec["dictionary"] = self.dictionary_id
        elif kind == "patch":
            prompt = spec.get("source_prompt")
            _text(prompt, "source_prompt", MAX_PROMPT_BYTES)
            if self.tokenizer is None:
                raise InterventionError("patch requires a configured tokenizer")
            if "source_position" in spec:
                _indices([spec["source_position"]], "source_position", limit=MAX_POSITIONS)
            self._source_inputs(spec)
        parameter = next(module.parameters(), next(self.model.parameters(), None))
        if parameter is not None and parameter.is_floating_point():
            maximum = torch.finfo(parameter.dtype).max
            values = ([*spec["vector"], spec["coefficient"],
                       *(v * spec["coefficient"] for v in spec["vector"])] if kind == "add" else
                      [spec["multiplier"]] if kind == "scale" else
                      [spec["value"]] if kind == "clamp" else [])
            if any(abs(v) > maximum for v in values):
                raise InterventionError("intervention value exceeds the model dtype range")
        if not metadata.specification(spec):
            raise InterventionError("canonical intervention metadata exceeds the wire bounds")
        return spec

    def _source_inputs(self, spec):
        from ._output import prompt_inputs
        try:
            inputs = prompt_inputs(self.model, self.tokenizer, spec["source_prompt"])
        except ValueError as error:
            raise InterventionError("invalid patch source: " + str(error)) from error
        except Exception as error:
            raise InterventionError("source prompt tokenization failed") from error
        ids = inputs["input_ids"]
        positions = [spec["source_position"]] if "source_position" in spec else spec.get("positions", [])
        if positions and max(positions) >= ids.shape[1]:
            raise InterventionError("source position is outside the clean prompt")
        return inputs

    def _identity(self):
        """Check exact state at each lookup, including writes between point hooks."""
        from ._forward_identity import source_settings
        # Include nonpersistent buffers and alias writes. Inference tensors have
        # no version counter, but their materialized bytes obey the same contract.
        state = {"parameter:" + name: value for name, value in self.model.named_parameters()}
        state.update({"buffer:" + name: value for name, value in self.model.named_buffers()})
        settings = source_settings(self.model, tensor_hash=lambda values: _content_identity(values, self.resource_guard))
        content = _content_identity(state, self.resource_guard)
        return (id(self.model), id(self.tokenizer),
                tuple((id(m), m.training) for m in self.model.modules()),
                tuple((name, id(value), value.device) for name, value in state.items()), content,
                tuple(sorted(settings.items())),
                tuple((name, id(module.act_fn)) for name, module in self.model.named_modules()
                      if hasattr(module, "act_fn")))

    def _source_keys(self):
        return {(hook.spec["source_prompt"], hook.spec["point"]) for hook in self._hooks
                if (isinstance(hook, Intervention) and not hook.removed and hook.spec["op"] == "patch"
                    and hook not in self._retiring_sources)}

    @contextmanager
    def _retire_sources(self, hooks):
        """Count final sources while the caller retains old hooks and cache for rollback."""
        previous = self._retiring_sources
        self._retiring_sources = previous.union(hook for hook in hooks if hook is not None)
        try:
            yield
        finally:
            self._retiring_sources = previous

    def _discard_unused_sources(self):
        required = self._source_keys()
        for key in list(self._cache):
            if key not in required:
                del self._cache[key]
        self._trim_source_inputs()

    def _trim_source_inputs(self):
        if self._cache_identity is not None:
            identity, inputs = self._cache_identity
            prompts = {key[0] for key in self._cache}
            self._cache_identity = identity, tuple((key, value) for key, value in inputs if key in prompts)

    def _source_space(self, sources):
        """Plan cache eviction after successful capture within fixed bounds."""
        sizes = {key: value.numel() * value.element_size() for key, value in self._cache.items()}
        extra = sum(value.numel() * value.element_size() for value in sources.values())
        total, count = sum(sizes.values()) + extra, len(sizes) + len(sources)
        pinned, evict = self._source_keys(), []
        for key, size in sizes.items():
            if total <= MAX_SOURCE_BYTES and count <= MAX_SOURCE_ENTRIES:
                break
            if key not in pinned:
                evict.append(key)
                total, count = total - size, count - 1
        if total > MAX_SOURCE_BYTES or count > MAX_SOURCE_ENTRIES:
            resource = "source_bytes" if total > MAX_SOURCE_BYTES else "source_entries"
            raise InterventionError("active patch sources exceed the bounded source cache", resource=resource)
        return evict

    @contextmanager
    def _capture_sources(self, points, *, prompt):
        """Capture generic tensor outputs without decoder or trace admission."""
        try:
            values = capture(self.model, points)
        except NotImplementedError:
            values, handles = {}, []
            memory = _IdentityMemory(self.resource_guard) if self.resource_guard is not None else None
            initial = None
            try:
                for point in points:
                    module, event, _, probe = self._target(point)
                    if event != "output" or probe is not None:
                        raise InterventionError("generic patch sources require module outputs")

                    def save(module, args, output, point=point):
                        nonlocal initial
                        if not isinstance(output, torch.Tensor):
                            raise InterventionError("patch source point must produce a tensor")
                        pending = {(prompt, name): value for name, value in values.items()}
                        pending[prompt, point] = output
                        try:
                            self._source_space(pending)
                        finally:
                            pending.clear()
                        if memory is not None:
                            memory.check("patch source capture")
                            if initial is None:
                                initial = allocation_bytes(memory)
                            retained = sum(value.numel() * value.element_size() for value in values.values())
                            reserve = output.numel() * output.element_size()
                            if allocation_bytes(memory, floor=initial + retained) + reserve > memory.limit:
                                raise CleanStop("patch source allocation exceeds the process memory limit")
                        # The next module can change this output in place. Preserve
                        # the point value now and retain prior copies in admission.
                        values[point] = output.detach().clone()

                    handles.append(module.register_forward_hook(save))
                yield values, True
            finally:
                for handle in reversed(handles):
                    handle.remove()
                values.clear()
        else:
            try:
                with values:
                    yield values, False
            finally:
                values.close(clear=True)

    def _source(self, spec):
        """Report source admission failures through the intervention protocol."""
        try:
            return self._prepare_source(spec)
        except InterventionError:
            raise
        except (TypeError, ValueError) as error:
            raise InterventionError("patch source refused: " + str(error)[:256],
                                    "intervene.not_supported") from error

    def _copy_sources(self, sources):
        """Admit all decoder copies while cache and capture tensors remain live."""
        memory = _IdentityMemory(self.resource_guard) if self.resource_guard is not None else None
        retained = {}
        if memory is not None:
            # Capture points can share storage. Old cache entries remain live
            # until publication, even when the cache plan selects their eviction.
            for value in (*self._cache.values(), *sources.values()):
                storage = value.untyped_storage()
                retained[value.device, storage.data_ptr()] = storage.nbytes()
        pending = sum(value.numel() * value.element_size() for value in sources.values())
        initial, copied, copies = None, 0, {}
        try:
            for key, value in sources.items():
                if memory is not None:
                    memory.check("patch source copy")
                    if initial is None:
                        initial = allocation_bytes(memory, floor=sum(retained.values()))
                    # Live readings already include prior copies. The floor also
                    # covers stale readings, and only pending copies need reserve.
                    if allocation_bytes(memory, floor=initial + copied) + pending > memory.limit:
                        raise CleanStop("patch source allocation exceeds the process memory limit")
                copies[key] = value.detach().clone()
                size = value.numel() * value.element_size()
                copied, pending = copied + size, pending - size
            return copies
        except BaseException:
            copies.clear()
            raise
        finally:
            # A retained exception keeps this frame after the caller clears its maps.
            value = storage = None

    def _prepare_source(self, spec):
        identity = self._identity()
        if self._cache_identity is None or identity != self._cache_identity[0]:
            self.invalidate_cache()
            self._cache_identity = identity, ()
        prompt, point = spec["source_prompt"], spec["point"]
        key = prompt, point
        inputs, input_identity = _source_input_identity(self._source_inputs(spec), self.resource_guard)
        input_identities = dict(self._cache_identity[1])
        if input_identities.get(prompt) != input_identity:
            for old_key in list(self._cache):
                if old_key[0] == prompt:
                    del self._cache[old_key]
            input_identities.pop(prompt, None)
        if key not in self._cache:
            from ._memory import autorelease_scope
            from ._output import internal_forward
            missing = {key} | {item for item in self._source_keys()
                               if item[0] == prompt and item not in self._cache}
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            config = getattr(self.model, "config", None)
            kwargs = {"use_cache": False} if config is not None else {}
            if getattr(config, "model_type", None) in ("qwen3_5_text", "gemma2"):
                kwargs["logits_to_keep"] = 1
            self.preparing_source = True
            sources = {}
            try:
                with (internal_forward(), self.suspend(), torch.no_grad(), autorelease_scope(device),
                      _preserve_captures(self.model)):
                    with self._capture_sources(sorted(item[1] for item in missing), prompt=prompt) as (values, copied):
                        if self.resource_guard is not None:
                            self.resource_guard.check("patch source forward start")
                        try:
                            self.model(**inputs, **kwargs)
                        finally:
                            if self.resource_guard is not None:
                                self.resource_guard.check("patch source forward complete")
                        if any(item[1] not in values for item in missing):
                            raise InterventionError("patch source point did not run")
                        sources = {item: values[item[1]] for item in sorted(missing)}
                        if any(value.ndim != 3 or not torch.isfinite(value).all().item()
                               for value in sources.values()):
                            raise InterventionError("patch source must be a finite batch-token-width tensor")
                        evict = self._source_space(sources)
                        # Admission precedes clones, and a failed capture retains no partial entries.
                        if not copied:
                            sources = self._copy_sources(sources)
            except BaseException:
                sources.clear()
                raise
            finally:
                self.preparing_source = False
            for old_key in evict:
                del self._cache[old_key]
            self._cache.update(sources)
            input_identities[prompt] = input_identity
            # Keep model and input evidence in one immutable transaction record.
            # Control rollback already restores this record with its cached tensors.
            self._cache_identity = identity, tuple(input_identities.items())
            self._trim_source_inputs()
        self._cache.move_to_end(key)
        return self._cache[key]

    def install(self, request, *, on_error=None):
        """Install a validated hook. The caller selects the step boundary."""
        spec = self.validate(request)
        if spec["op"] == "ablate":
            from .._ablate import ablate
            apply = ablate if self.adapter is None else self.adapter.ablate
            hook = apply(self.model, spec["point"])
            if hasattr(hook, "on_error"):
                hook.on_error = on_error
        else:
            module, event, width, probe = self._target(spec["point"])
            if spec["op"] == "patch":
                source = self._source(spec)
                positions = spec.get("positions")
                sources = [spec["source_position"]] if "source_position" in spec else positions
                if sources and max(sources) >= source.shape[1]:
                    raise InterventionError("source position is outside the clean prompt")
            hook = Intervention(self, spec, width, probe, on_error)
            try:
                if spec["op"] in ("scale", "clamp"):
                    read = self._target(self.dictionary.metadata["read_points"][probe.layer])
                    if read[0] is not module:
                        if read[1] != "input":
                            raise InterventionError("feature interventions require a semantic input probe")
                        hook.separate_read = True
                        block = self.model.get_submodule(
                            self.dictionary.metadata["read_points"][probe.layer].rsplit(".", 1)[0])
                        hook.handles.append(read[0].register_forward_hook(hook.read, with_kwargs=True))
                        hook.handles.append(block.register_forward_pre_hook(hook.enter_read))
                        hook.handles.append(block.register_forward_hook(hook.clear_read, always_call=True))
                if event == "input":
                    hook.handles.append(module.register_forward_pre_hook(hook.pre, with_kwargs=True))
                else:
                    hook.handles.append(module.register_forward_hook(hook.post, with_kwargs=True))
            except BaseException:
                hook()
                raise
        self._hooks.add(hook)
        return hook


class _InterventionReplay:
    """Private snapshot effects with borrowed resources and an independent source cache."""

    def __init__(self, executor):
        self.model = executor.model
        self.executor = InterventionService(self.model, adapter=executor.adapter)
        self.executor._ablation_points = executor._ablation_points
        self.executor.dictionary = executor.dictionary
        self.executor.dictionary_id = executor.dictionary_id
        self.executor._default_dictionary_id = executor._default_dictionary_id
        self.executor._dictionary_snapshot = executor._dictionary_snapshot
        self.executor.tokenizer = executor.tokenizer
        self.executor.resource_guard = executor.resource_guard
        self._effects = []

    @property
    def dictionary(self):
        return self.executor.dictionary

    @property
    def tokenizer(self):
        return self.executor.tokenizer

    def specifications(self):
        """Return detached specifications for effects that remain installed."""
        return [deepcopy(spec) for spec, hook in self._effects if not getattr(hook, "removed", False)]

    @contextmanager
    def _scope(self, live_hooks, specifications):
        outer = _ACTIVE_REPLAYS.get()
        hooks = [*live_hooks, *(hook for replay in outer if replay.model is self.model
                               for _, hook in replay._effects)]
        with _mute_hooks(hooks):
            token = _ACTIVE_REPLAYS.set((*outer, self))
            try:
                for spec in specifications:
                    self._effects.append((spec, self.executor.install(spec)))
                yield self
                for _, hook in self._effects:
                    if getattr(hook, "error", None) is not None:
                        raise InterventionError("snapshot hook failed: " + hook.error,
                                                "intervene.hook_failed")
            finally:
                try:
                    self.executor.close()
                finally:
                    _ACTIVE_REPLAYS.reset(token)


class Intervention:
    """One failure-contained hook with no retained live activation."""

    def __init__(self, service, spec, width, probe, on_error):
        self.service, self.spec = service, spec
        self.width, self.probe = width, probe
        self.on_error = on_error
        self.handles = []
        self.separate_read = False
        self.live_input = None
        self.read_stack = []
        self.calls, self.errors, self.error = 0, 0, None

    @property
    def removed(self):
        return not self.handles

    def __call__(self):
        self._remove()

    def _remove(self, *, retain_source=False):
        self.live_input = None
        self.read_stack.clear()
        while self.handles:
            self.handles.pop().remove()
        if self.spec["op"] == "patch" and not retain_source:
            key = self.spec["source_prompt"], self.spec["point"]
            if key not in self.service._source_keys():
                self.service._cache.pop(key, None)
                self.service._trim_source_inputs()
        self.on_error = None

    def _failed(self, error):
        self.errors += 1
        self.error = f"{type(error).__name__}: {error}"[:300]
        callback = self.on_error
        self()
        if callback is not None:
            try:
                callback(self.error)
            except CleanStop:
                raise
            except Exception:
                pass  # Error reporting must never escape into the model forward.

    def _apply(self, value, live_input):
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != self.width:
            raise InterventionError("point must produce a batch-token-width tensor of the expected width")
        if not value.is_floating_point():
            raise InterventionError("point tensor must be floating point")
        chosen = selected_rows(self.service.model)
        batch = list(range(value.shape[0])) if chosen is None else list(chosen)
        positions = self.spec.get("positions", list(range(value.shape[1])))
        if any(i >= value.shape[0] for i in batch) or any(i >= value.shape[1] for i in positions):
            raise InterventionError("selected row or position is outside the live tensor")
        if not batch or not positions:
            return value
        b = torch.tensor(batch, device=value.device)[:, None]
        p = torch.tensor(positions, device=value.device)[None, :]
        selected = value[b, p]
        kind = self.spec["op"]
        if kind == "add":
            vector = value.new_tensor(self.spec["vector"])
            changed = selected + self.spec["coefficient"] * vector
        elif kind in ("scale", "clamp"):
            if not isinstance(live_input, torch.Tensor) or live_input.shape != value.shape:
                raise InterventionError("live MLP input does not match the output shape")
            layer = self.probe.layer
            dictionary = self.service.dictionary
            weights = dictionary.layers[layer]
            x = live_input[b, p].to(device=weights.W_enc.device)
            activation = _trace_features(weights, x, self.service.resource_guard)[..., self.spec["feature"]]
            amount = ((self.spec["multiplier"] - 1) * activation if kind == "scale" else
                      self.spec["value"] - activation)
            decoder = weights.W_dec[self.spec["feature"]].to(x.dtype)
            delta = (amount.unsqueeze(-1) * decoder).to(device=value.device, dtype=value.dtype)
            changed = selected + delta
        else:
            source = self.service._source(self.spec)
            source_positions = [self.spec["source_position"]] * len(positions) if "source_position" in self.spec else positions
            if max(source_positions) >= source.shape[1] or source.shape[-1] != self.width:
                raise InterventionError("source positions or width do not match the patch")
            changed = source[:, source_positions].to(device=value.device, dtype=value.dtype).expand(len(batch), -1, -1)
        if not torch.isfinite(changed).all().item():
            raise InterventionError("intervention produced a nonfinite tensor")
        result = value.clone()
        result[b, p] = changed
        return result

    def pre(self, module, args, kwargs):
        if self.removed or _hook_muted(self) or selected_rows(self.service.model) == ():
            return None
        self.calls += 1
        try:
            value = self._apply(_input(args, kwargs), None)
            if args:
                return (value,) + args[1:], kwargs
            key = "hidden_states" if "hidden_states" in kwargs else "x"
            return args, dict(kwargs, **{key: value})
        except Exception as error:
            self._failed(error)
            if isinstance(error, CleanStop):
                raise
            return None

    def post(self, module, args, kwargs, output):
        live_input = self.live_input if self.separate_read else _input(args, kwargs)
        self.live_input = None
        if self.removed or _hook_muted(self) or selected_rows(self.service.model) == ():
            return None
        self.calls += 1
        try:
            return self._apply(output, live_input)
        except Exception as error:
            self._failed(error)
            if isinstance(error, CleanStop):
                raise
            return None

    def read(self, module, args, kwargs, output):
        self.live_input = None
        if not self.removed and not _hook_muted(self) and selected_rows(self.service.model) != ():
            self.live_input = _input(args, kwargs)

    def enter_read(self, module, args):
        self.read_stack.append(self.live_input)
        self.live_input = None

    def clear_read(self, module, args, output):
        self.live_input = self.read_stack.pop() if self.read_stack else None
