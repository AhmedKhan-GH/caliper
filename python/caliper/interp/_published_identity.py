"""Exact model copies owned only by a prepared published transaction."""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import math
import traceback
import weakref

import torch

from . import _dictionary, _intervene as identity
from ._dictionary import _dense_state
from ._forward_identity import source_settings
from ._memory import BudgetMemory, CleanStop, allocation_bytes, autorelease_scope


_PUBLIC_CHECK = ContextVar('caliper_published_identity_check', default=None)


class _ProofChanged(Exception):
    """Reject reuse without suppressing a caller or resource exception."""


class _CPUProofBudget:
    device = torch.device('cpu')
    limit = _dictionary._UNVERIFIED_CPU_BYTES
    current = peak = 0

    def check(self, stage):
        return 0


class _PublishedMemory(identity._DeviceSnapshotMemory):
    @property
    def host_reserve_bytes(self):
        return self.guard.host_reserve_bytes

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
        self._public = None
        self._public_attempted = False
        self._public_owner = self._public_service = None
        self._public_guard = self._public_resource_guard = None
        self._public_verification = self._public_config = None

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in self.state.values())

    def release(self):
        self.reject_public_identity()
        self._public_attempted = True
        reference = _dictionary._PUBLISHED_MODELS.get(self.model)
        if reference is not None and reference() is self:
            del _dictionary._PUBLISHED_MODELS[self.model]
        self._public_owner = self._public_service = None
        self._public_guard = self._public_resource_guard = None
        self._public_verification = self._public_config = None
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
        verified = self._public_verification
        if verified is not None and verified.memory is not None:
            guard = BudgetMemory(verified.memory, 40, extra=guard)
        return published_memory(self.model, self.executor, state, guard, floor=self.floor)

    @staticmethod
    def _admit(memory, reserve, *, force=False, floor=0):
        check = _PUBLIC_CHECK.get()
        if check is not None:
            check('model_identity')
        identity._DictionarySnapshot._admit(memory, reserve, force=force, floor=floor)
        if check is not None:
            check('model_identity')

    def reject_public_identity(self):
        self._public = None

    def _public_layout(self):
        # Public hashes include every persistent and transient registration.
        # Custom serializers and tensor subclasses retain the full hash path.
        names = {id(value): name for name, value in identity._model_state(self.model).items()}
        persistent, transient = [], []
        for prefix, module in self.model.named_modules(remove_duplicate=False):
            if (any(getattr(getattr(module, field), '__func__', None) is not getattr(torch.nn.Module, field)
                    for field in ('state_dict', '_save_to_state_dict', 'get_extra_state'))
                    or module._state_dict_hooks or module._state_dict_pre_hooks):
                return None
            for buffers, values in ((False, module._parameters), (True, module._buffers)):
                for key, value in values.items():
                    if value is None:
                        if buffers and key in module._non_persistent_buffers_set:
                            return None
                        continue
                    if type(value) not in (torch.Tensor, torch.nn.Parameter) or id(value) not in names:
                        return None
                    group = transient if buffers and key in module._non_persistent_buffers_set else persistent
                    group.append(((prefix + '.' if prefix else '') + key, names[id(value)]))
        return tuple(sorted(persistent)), tuple(sorted(transient))

    def _public_owned(self, *, preparing=False):
        owner = None if self._public_owner is None else self._public_owner()
        service = None if self._public_service is None else self._public_service()
        return (self.active and owner is not None and service is not None
                and (preparing or owner.preparation is not None)
                and owner._model_snapshot is self and owner.model is self.model
                and owner.tokenizer is self.tokenizer and owner.interventions is service
                and owner.guard is self._public_guard and service.executor is self.executor
                and self.executor.resource_guard is self._public_resource_guard
                and self.model.config is self._public_config
                and _dictionary._VERIFIED_MODELS.get(self.model) is self._public_verification
                and self._owners_match(self.executor, self.model, self.tokenizer))

    def public_memory(self, memory, limit_gb):
        # Retain preparation guards even after rejection, until explicit release.
        guards = [guard for guard in (self._public_guard, self._public_resource_guard, memory)
                  if guard is not None]
        combined = guards[0] if guards else _CPUProofBudget()
        for guard in guards[1:]:
            combined = BudgetMemory(combined, 40, extra=guard)
        if limit_gb is not None:
            combined = BudgetMemory(combined, limit_gb)
        memory = self.memory(self.state, combined)
        if isinstance(memory, _PublishedMemory):
            memory.reserve = 4 * _dictionary._HASH_CHUNK_BYTES
        return memory

    def bind_public_identity(self, owner):
        """Prove the public hash from existing snapshot storage exactly once."""
        if self._public_attempted or not self.active:
            raise ValueError('published identity proof cannot be rebound')
        self._public_attempted = True
        self._public_owner, self._public_service = weakref.ref(owner), weakref.ref(owner.interventions)
        self._public_guard, self._public_resource_guard = owner.guard, self.executor.resource_guard
        self._public_verification = verified = _dictionary._VERIFIED_MODELS.get(self.model)
        self._public_config = self.model.config
        tiny = self.model.config.model_type == 'caliper_tiny'
        layout = self._public_layout()
        if layout is None or (not tiny and verified is None):
            return

        def check(stage):
            if not self._public_owned(preparing=True):
                raise ValueError('published identity ownership changed during preparation')

        token = _PUBLIC_CHECK.set(check)
        try:
            check('model_identity')
            memory = self.public_memory(None, None)
            public = _dictionary._full_model_identity(self.model, memory=memory, check=check)
            check('model_identity')
            hashes = [_dictionary._state_hash({name: self.state[source] for name, source in group},
                                              memory=memory, check=check) for group in layout]
            if tiny:
                valid = hashes[0] == public
            else:
                settings = _dictionary.forward_settings(self.model, tensor_hash=lambda values: _dictionary._state_hash(
                    {name: self.state['setting:' + name] for name in values}, memory=memory, check=check))
                valid = (public == verified.identity == _dictionary._hf_identity(self.model.config)
                         and hashlib.sha256(''.join(hashes).encode('ascii')).hexdigest() == verified.digest
                         and settings == verified.settings)
            if not valid or not self.matches(self.executor, self.model, self.tokenizer, memory, _memory=memory):
                raise ValueError('published public identity differs from its snapshot')
            check('model_identity')
            state, parts = self._state()
            if (parts != self.original or _structure(self.model, state) != self.structure
                    or self._public_layout() != layout):
                raise ValueError('published public identity changed during preparation')
            check('model_identity')
            self._public = public, layout
        finally:
            _PUBLIC_CHECK.reset(token)

    def register_public_identity(self):
        if self._public is None:
            return
        if not self._public_owned():
            self.reject_public_identity()
            raise ValueError('published identity owner changed before registration')
        previous = _dictionary._PUBLISHED_MODELS.get(self.model)
        previous = None if previous is None else previous()
        if previous is not None and previous is not self:
            previous.reject_public_identity()
        _dictionary._PUBLISHED_MODELS[self.model] = weakref.ref(self)

    def public_identity(self, model, memory, check):
        """Compare every current byte before reuse of the preparation proof."""
        proof, accepted, caller_failed = self._public, False, False

        def current(stage):
            nonlocal caller_failed
            if model is not self.model or self._public is not proof or not self._public_owned():
                raise _ProofChanged()
            if check is not None:
                try:
                    check('model_identity')
                except BaseException:
                    caller_failed = True
                    raise
            if self._public is not proof or not self._public_owned():
                raise _ProofChanged()

        token = _PUBLIC_CHECK.set(current)
        try:
            if proof is None:
                return None
            current('model_identity')
            public, layout = proof
            if self._public_layout() != layout:
                return None
            if self._public_verification is not None and _dictionary._hf_identity(model.config) != public:
                return None
            # Settings and registrations surround all comparison and guard callbacks.
            if not self.matches(self.executor, model, self.tokenizer, memory, _memory=memory):
                return None
            current('model_identity')
            state, parts = self._state()
            if (parts != self.original or _structure(model, state) != self.structure
                    or self._public_layout() != layout
                    or (self._public_verification is not None and _dictionary._hf_identity(model.config) != public)
                    or not self._public_owned()):
                return None
            accepted = True
            return public
        except _ProofChanged:
            return None
        except (ValueError, KeyError, TypeError):
            if caller_failed:
                raise
            # Unsupported state must retain the ordinary serializer and hash.
            return None
        finally:
            _PUBLIC_CHECK.reset(token)
            if not accepted:
                self.reject_public_identity()
                if isinstance(memory, _PublishedMemory):
                    memory.reserve = max(memory.reserve, 4 * _dictionary._HASH_CHUNK_BYTES)

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

    def matches(self, executor, model, tokenizer, guard, *, full_identity=None, _memory=None):
        if not self._owners_match(executor, model, tokenizer):
            return False
        state, parts = self._state()
        if parts != self.original or _structure(model, state) != self.structure:
            return False
        memory = self.memory(state, guard) if _memory is None else _memory
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
