"""Exact model copies owned only by a prepared published transaction."""
from __future__ import annotations

import json
import math
import traceback
import weakref

import torch

from . import _dictionary, _intervene as identity
from ._dictionary import _dense_state
from ._forward_identity import source_settings
from ._memory import BudgetMemory, CleanStop, allocation_bytes, autorelease_scope


class _PublishedMemory(identity._DeviceSnapshotMemory):
    def check(self, stage, *, force=True):
        super().check(stage, force=force)
        self.headroom -= max(0, self.guard.host_reserve_bytes - 8_000_000_000)
        if self.headroom < self.reserve:
            raise CleanStop('published identity allocation exceeds the supplied host reserve')


def published_memory(model, executor, state, guard, *, floor=0):
    guards = [value for value in (guard, executor.resource_guard) if value is not None]
    tiny = model.config.model_type == 'caliper_tiny' and all(t.device.type == 'cpu' for t in state.values())
    if not guards:
        if not tiny:
            raise ValueError('published model snapshot requires a bounded memory guard')
        return None
    primary = guards[0]
    combined = BudgetMemory(primary, 40, extra=guards[-1] if guards[-1] is not primary else None)
    # Validate device ownership before any dispatch, even with the hard cap.
    if any(t.device.type != 'cpu' for t in state.values()) and not math.isfinite(
            identity._IdentityMemory(primary).limit):
        raise ValueError('published model snapshot requires a bounded device guard')
    identity._DictionarySnapshot._memory(state, combined)
    memory = identity._IdentityMemory(combined) if tiny else _PublishedMemory(combined)
    memory.floor = floor
    return memory


def _callable_identity(value):
    function = getattr(value, '__func__', value)
    return (id(function), id(getattr(value, '__self__', None)),
            id(getattr(function, '__code__', None)))


def _structure(model, state):
    tensors = tuple((name, id(value), type(value), value.dtype, tuple(value.shape), value.layout,
                     value.device, value.stride(), value.storage_offset(),
                     value.untyped_storage()._cdata, value.untyped_storage().nbytes(),
                     value.data_ptr(), value.requires_grad) for name, value in state.items())
    modules = tuple((name, id(module), type(module), module.training,
                     tuple((key, id(value)) for key, value in module._parameters.items()),
                     tuple((key, id(value)) for key, value in module._buffers.items()),
                     tuple(sorted(module._non_persistent_buffers_set)),
                     tuple((field, _callable_identity(getattr(module, field, None)))
                           for field in ('forward', '_call_impl', '_compiled_call_impl')))
                    for name, module in model.named_modules(remove_duplicate=False))
    return tensors, modules


def forward_hooks(model):
    """Bind hook order and dispatch flags across one fresh action forward."""
    from torch.nn.modules import module as global_hooks
    fields = ('_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_forward_hooks',
              '_forward_hooks_with_kwargs', '_forward_hooks_always_called')
    def entries(owner, names):
        return tuple((field, tuple((key, _callable_identity(value))
                                  for key, value in getattr(owner, field, {}).items())) for field in names)
    return (entries(global_hooks, tuple('_global' + field for field in fields)),
            tuple((name, id(module), entries(module, fields)) for name, module in model.named_modules()))


class _PublishedModelSnapshot(identity._DictionarySnapshot):
    """Reuse bounded byte comparisons without a general model identity cache."""

    def __init__(self, executor, original):
        super().__init__({}, original[4], {}, device_compare=True)
        self.executor, self.model, self.tokenizer = executor, executor.model, executor.tokenizer
        self.dictionary, self.dictionary_id = weakref.ref(executor.dictionary), executor.dictionary_id
        self.dictionary_snapshot = (None if executor._dictionary_snapshot is None else
                                    weakref.ref(executor._dictionary_snapshot))
        self.original = original
        self.structure = None
        self.settings_hashes = {}
        self.references = []
        self.floor = 0
        self.active = False
        self.checks = 0

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in self.state.values())

    def release(self):
        self.state.clear()
        self.active = False
        self.floor = 0

    def _state(self):
        state = identity._model_state(self.model)
        settings_state = {}
        expected = dict(self.original[5])
        def tensor_hash(values):
            # Source settings request one named tensor at a time.
            name, value = next(iter(values.items()))
            if len(values) != 1 or name not in expected:
                raise ValueError('published tensor settings changed')
            digest = json.loads(expected[name])
            if not isinstance(digest, dict) or set(digest) != {'tensor'}:
                raise ValueError('published tensor settings changed')
            settings_state['setting:' + name] = value
            self.settings_hashes[name] = digest['tensor']
            return digest['tensor']
        settings = source_settings(self.model, tensor_hash=tensor_hash)
        parts = identity._model_identity_parts(self.model, self.executor.tokenizer, state,
                                              self.content, settings)
        state.update(settings_state)
        return _dense_state(state), parts

    def _owners_match(self, executor, model, tokenizer):
        return (executor is self.executor and model is self.model and tokenizer is self.tokenizer
                and executor.model is self.model and executor.tokenizer is self.tokenizer
                and executor.dictionary is self.dictionary() and executor.dictionary_id == self.dictionary_id
                and executor._default_dictionary_id
                and (executor._dictionary_snapshot is None if self.dictionary_snapshot is None else
                     executor._dictionary_snapshot is not None
                     and executor._dictionary_snapshot is self.dictionary_snapshot()))

    def memory(self, state, guard):
        return published_memory(self.model, self.executor, state, guard, floor=self.floor)

    @classmethod
    def create(cls, executor, original, guard):
        if executor.model.config.model_type not in ('gemma2', 'caliper_tiny'):
            raise ValueError('model snapshots require the published Gemma profile or CPU fixture')
        result = cls(executor, original)
        state = tensor = current = saved = memory = None
        try:
            state, parts = result._state()
            if parts != original:
                raise ValueError('published source model changed before its snapshot')
            result.structure = _structure(result.model, state)
            # Retain owners to prevent address reuse after an object replacement.
            result.references = [*state.values(), *result.model.modules(),
                                 *(value.untyped_storage() for value in state.values())]
            for module in result.model.modules():
                for field in ('forward', '_call_impl', '_compiled_call_impl'):
                    value = getattr(module, field, None)
                    function = getattr(value, '__func__', value)
                    result.references.extend((function, getattr(function, '__code__', None)))
            memory = result.memory(state, guard)
            sizes = [t.numel() * t.element_size() for t in state.values()]
            total, largest = sum(sizes), max(sizes, default=0)
            scratch = max(4 * _dictionary._HASH_CHUNK_BYTES,
                          identity._snapshot_workspace(min(largest, identity._SNAPSHOT_COMPARE_BYTES), 1))
            if memory is None:
                if total + scratch > _dictionary._UNVERIFIED_CPU_BYTES:
                    raise CleanStop('published model snapshot exceeds the unguarded CPU limit')
                initial = 0
            else:
                cls._admit(memory, 0, force=True)
                dictionary_bytes = sum(t.numel() * t.element_size() for t in executor.dictionary.state_dict().values())
                if executor._dictionary_snapshot is not None:
                    dictionary_bytes += sum(t.numel() * t.element_size()
                                            for t in executor._dictionary_snapshot.state.values())
                initial = max(allocation_bytes(memory), total + dictionary_bytes)
            # Admit the complete retained set before the first destination exists.
            cls._admit(memory, total + scratch, force=True, floor=initial)
            copied = 0
            for name, tensor in state.items():
                cls._admit(memory, total - copied + scratch, floor=initial + copied)
                if tensor.device.type == 'cpu':
                    with autorelease_scope(tensor.device):
                        result.state[name] = tensor.detach().clone()
                    copied += tensor.numel() * tensor.element_size()
                    continue
                if tensor.device.type != 'mps':
                    raise ValueError('published model snapshots support CPU and MPS')
                current = tensor.detach().reshape(-1).view(torch.uint8)
                saved, start = None, 0
                while saved is None or start < current.numel():
                    blocks = [min(identity._SNAPSHOT_COMPARE_BYTES, current.numel() - offset)
                              for offset in range(start, min(current.numel(), start +
                                  identity._SNAPSHOT_COMPARE_BYTES * identity._SNAPSHOT_BATCH_BLOCKS),
                                  identity._SNAPSHOT_COMPARE_BYTES)]
                    count, workspace = cls._admit_batch(memory, blocks, retained=total - copied,
                                                       minimum=scratch, floor=initial + copied)
                    if saved is None:
                        with autorelease_scope(tensor.device):
                            result.state[name] = torch.empty_like(tensor)
                        copied += current.numel()
                        saved = result.state[name].reshape(-1).view(torch.uint8)
                    end = min(start + count * identity._SNAPSHOT_COMPARE_BYTES, current.numel())
                    if count:
                        with identity._snapshot_device_batch(tensor.device):
                            for offset in range(start, end, identity._SNAPSHOT_COMPARE_BYTES):
                                stop = min(offset + identity._SNAPSHOT_COMPARE_BYTES, end)
                                cls._admit(memory, total - copied + workspace, floor=initial + copied)
                                saved[offset:stop].copy_(current[offset:stop])
                    start = end
            result.floor = initial + copied
            if memory is not None:
                memory.floor = result.floor
            cls._admit(memory, 4 * _dictionary._HASH_CHUNK_BYTES, force=True, floor=result.floor)
            copied_model = {name: value for name, value in result.state.items() if not name.startswith('setting:')}
            if identity._content_identity(copied_model, memory) != original[4]:
                raise ValueError('published model copy differs from the preparation hash')
            for name, digest in result.settings_hashes.items():
                if identity._content_identity({name: result.state['setting:' + name]}, memory) != digest:
                    raise ValueError('published settings copy differs from the preparation hash')
            result.active = True
            if not result.matches(executor, executor.model, executor.tokenizer, guard):
                raise ValueError('published source model changed during its snapshot')
            return result
        except BaseException as error:
            result.release()
            result.references.clear()
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            # A retained exception must not retain private tensor views.
            state = tensor = current = saved = memory = None
            copied_model = None

    def matches(self, executor, model, tokenizer, guard, *, full_identity=None):
        if not self._owners_match(executor, model, tokenizer):
            return False
        state, parts = self._state()
        if parts != self.original or _structure(model, state) != self.structure:
            return False
        memory = self.memory(state, guard)
        if not self.active:
            self._admit(memory, 4 * _dictionary._HASH_CHUNK_BYTES, force=True)
            return full_identity is not None and full_identity() == self.original
        self._admit(memory, 0, force=True)
        # The device helper visits all batches even after a mismatch.
        equal = self._device_matches(state, memory)
        current = saved = None
        try:
            for name, tensor in state.items():
                if tensor.device.type != 'cpu':
                    continue
                current = tensor.detach().reshape(-1).view(torch.uint8)
                saved = self.state[name].reshape(-1).view(torch.uint8)
                for start in range(0, current.numel(), identity._SNAPSHOT_COMPARE_BYTES):
                    stop = min(start + identity._SNAPSHOT_COMPARE_BYTES, current.numel())
                    self._admit(memory, 4 * (stop - start))
                    with autorelease_scope(tensor.device):
                        equal = identity._cpu_byte_equal(current[start:stop], saved[start:stop]) and equal
            self._admit(memory, 0, force=True)
            self.checks += 1
            return equal
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            current = saved = None
