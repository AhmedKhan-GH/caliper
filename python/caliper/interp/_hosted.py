"""Offline hosted inference with deterministic fixture resources and safe forks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ._admission import (GEMMA_REVISION, _admit_model_load, _admit_model_transfer, _load_gemma_model)
from .._controls import intervention_key
from ._output import Output, bounded_text, internal_forward, prompt_inputs
from ._qwen_family import MODEL_ID as QWEN_ID, MODEL_REVISION as QWEN_REVISION
from ._service import InterpretationService, Refused, register_service, unregister_service

CHECKPOINT_FORMAT = "caliper.inference"

CHECKPOINT_BYTES = 4 * 1024 * 1024
CHECKPOINT_VALUES = 262144


def checkpoint_metadata(value):
    """Validate local JSON metadata, including full-width intervention vectors."""
    return _checkpoint_metadata_size(value) is not None


def _checkpoint_metadata_size(value):
    """Return value and byte counts for metadata within the checkpoint limits."""
    count = 0
    def valid(item, depth=0):
        nonlocal count
        count += 1
        if depth > 8 or count > CHECKPOINT_VALUES:
            return False
        if item is None or type(item) is bool:
            return True
        if type(item) is int:
            return abs(item) < 2**53
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is str:
            return len(item.encode("utf-8")) <= 4096
        if type(item) is list:
            return len(item) <= 16384 and all(valid(v, depth + 1) for v in item)
        if type(item) is dict:
            return len(item) <= 64 and all(type(k) is str and len(k.encode("utf-8")) <= 63
                                           and valid(v, depth + 1) for k, v in item.items())
        return False
    try:
        if type(value) is dict and valid(value):
            size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            if size <= CHECKPOINT_BYTES:
                return count, size
    except (ValueError, UnicodeError):
        pass
    return None


@lru_cache(maxsize=1)
def _checkpoint_reserve():
    """Reserve the largest envelope and lifecycle fields without model access."""
    from .._controls import MAX_PARAMS, NAME_MAX, PARAM_TEXT_MAX, POINT_MAX, TEXT_MAX
    # Escaped control characters cost six JSON bytes per UTF-8 byte. Reserve
    # every parameter slot because text and parameter registrations can change.
    text = "\x01" * PARAM_TEXT_MAX
    step = 2**53 - 1
    clamp = {"point": "\x01" * POINT_MAX, "kind": "clamp", "op": "clamp",
             "dictionary": "\x01" * POINT_MAX, "feature": 2**31 - 1,
             "value": -1.7976931348623157e308, "positions": [31] * 32,
             "from_step": step, "until_step": step}
    envelope = {"format": CHECKPOINT_FORMAT, "version": 1, "run": text, "parent": text,
                "step": step, "params": {str(i).zfill(NAME_MAX): text for i in range(MAX_PARAMS)},
                "bindings": {key: text for key in ("model", "model_hash", "dictionary",
                                                   "dictionary_hash", "tokenizer_hash",
                                                   "replication_profile", "dtype")},
                "interventions": {"version": 1, "interventions": []},
                "selection": {key: step for key in ("layer", "feature", "position")},
                "owned_clamp": clamp, "generation_id": step}
    record = {"specification": None, "applied_specification": None, "state": "pending",
              "since_step": step, "clear": False,
              "error": "\x01" * TEXT_MAX, "error_code": "\x01" * NAME_MAX}
    count, size = _checkpoint_metadata_size(envelope)
    # Parameter names can also contain characters with six-byte JSON escapes.
    size += 5 * MAX_PARAMS * NAME_MAX
    return (count, size), _checkpoint_metadata_size(record)


def _check_checkpoint_interventions(ivs):
    """Refuse records whose future saved state can exceed either metadata limit."""
    (count, size), (record_count, record_size) = _checkpoint_reserve()
    for index, iv in enumerate(ivs.values()):
        specifications = [iv.spec]
        if iv.applied_spec is not None:
            specifications.append(iv.applied_spec)
        sizes = [_checkpoint_metadata_size(spec) for spec in specifications]
        if any(item is None for item in sizes):
            raise ValueError("intervention exceeds the checkpoint metadata limits")
        # A pending request acquires an applied copy. A replacement can retain
        # its old effect, or revert to it. Reserve two copies of each maximum.
        # The largest byte count and value count can come from different specs.
        # Ended history cannot acquire an effect without another admission.
        copies = 1 if iv.state == "ended" and iv.applied_spec is None else 2
        count += record_count - copies + copies * max(item[0] for item in sizes)
        size += record_size - 4 * copies + copies * max(item[1] for item in sizes) + (2 if index else 0)
        if count > CHECKPOINT_VALUES or size > CHECKPOINT_BYTES:
            raise ValueError("interventions exceed the checkpoint metadata limits with lifecycle headroom")


class _RunGuard:
    """Retain a resource stop after an action or service catches its exception."""
    def __init__(self, guard, *, extra=None, on_stop=None):
        self.guard, self.extra, self.failure = guard, extra, None
        self.on_stop = on_stop
        self._readings = {}
        self._observed_peak = 0

    def _owners(self):
        return (owner for guard in (self.guard, self.extra) if guard is not None
                for owner in (guard, getattr(guard, "_guard", guard)))

    @property
    def limit(self):
        return min((getattr(owner, "limit", math.inf) for owner in self._owners()), default=math.inf)

    @property
    def peak(self):
        return max(self._observed_peak, *(getattr(owner, "peak", 0) for owner in self._owners()), 0)

    @property
    def current(self):
        values = []
        for index, guard in enumerate((self.guard, self.extra)):
            if guard is None:
                continue
            owners = (guard, getattr(guard, "_guard", guard))
            readings = [getattr(owner, "current", None) for owner in owners]
            readings.append(self._readings.get(index))
            readings = [value for value in readings if value is not None]
            values.append(max(readings) if readings else max(getattr(owner, "peak", 0) for owner in owners))
        return max(values, default=0)

    @property
    def device(self):
        guard = self.guard if self.guard is not None else self.extra
        return getattr(guard, "device", torch.device("cpu"))

    def check(self, stage, *, force=True):
        return self._check("check" if force else "_check_interval", stage)

    def _check_interval(self, stage):
        return self.check(stage, force=False)

    def _check(self, method, stage):
        import inspect
        from ._memory import CleanStop
        if self.failure is not None:
            raise self.failure
        try:
            for index, guard in enumerate((self.guard, self.extra)):
                if guard is not None:
                    interval = getattr(guard, "_check_interval", None)
                    if method == "_check_interval" and interval is not None:
                        reading = interval(stage)
                    else:
                        check = guard.check
                        try:
                            parameters = inspect.signature(check).parameters.values()
                        except (TypeError, ValueError):
                            parameters = ()
                        keywords = any(p.kind == p.VAR_KEYWORD or (
                            p.name == "force" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
                            for p in parameters)
                        reading = check(stage, force=method == "check") if keywords else check(stage)
                    if isinstance(reading, (int, float)):
                        self._readings[index] = reading
                        self._observed_peak = max(self._observed_peak, reading)
            self._observed_peak = max(self.peak, self.current)
            if self._observed_peak > self.limit:
                raise CleanStop("hosted allocation exceeds the process memory limit")
            return self.current
        except CleanStop as exc:
            self.failure = exc
            if self.on_stop is not None:
                self.on_stop(exc)
            raise


def _fixture_display_guard(model_name):
    """Enable fixture display copies only through the explicit demo environment."""
    enabled = os.environ.get("CALIPER_FIXTURE_DISPLAY_MPS", "0")
    if enabled == "0":
        return None
    if enabled != "1":
        raise ValueError("CALIPER_FIXTURE_DISPLAY_MPS must be 0 or 1")
    if model_name != "tiny-lm":
        raise ValueError("fixture display copies require tiny-lm")
    if not torch.backends.mps.is_available():
        raise ValueError("fixture display copies require MPS")
    from ._memory import MemoryGuard
    return MemoryGuard(12, "mps")


class _FixtureDisplay:
    """Copy only bounded display tensors while fixture math remains on the CPU."""
    MAX_BYTES = 256 * 1024
    TENSORS = {
        "trace.counts": ((4,), torch.int32),
        "trace.node_i": ((512, 4), torch.int32),
        "trace.node_f": ((512, 4), torch.float32),
        "trace.edge_i": ((8192, 2), torch.int32),
        "trace.edge_f": ((8192, 2), torch.float32),
        "trace.label_text": ((544, 16), torch.uint32),
        "trace.probs": ((2, 10), torch.float32),
        "trace.logit_tokens": ((10,), torch.int32),
        "trace.logit_label_text": ((10, 16), torch.uint32),
        "examples.tokens": ((16, 25), torch.int32),
        "examples.activations": ((16, 25), torch.float32),
        "examples.example_ids": ((16,), torch.int32),
        "examples.valid_tokens": ((16,), torch.int32),
        "examples.label_text": ((400, 16), torch.uint32),
        "output.probs": ((2, 10), torch.float32),
        "output.tokens": ((10,), torch.int32),
        "output.label_text": ((10, 16), torch.uint32),
        "patch.grid": ((4, None), torch.float32),
    }

    def __init__(self, guard, *, device="mps"):
        # The private device seam permits CPU tests without an MPS operation.
        self.guard, self.device = guard, torch.device(device)
        if self.device.type == "mps" and self.device.index is None:
            self.device = torch.device("mps:0")

    def _transfer(self, tensor):
        return tensor.detach().contiguous().to(device=self.device, copy=True)

    def prepare(self, values):
        """Validate and transfer private display tensors before publication."""
        from .. import _programs
        from . import _publish
        from ._memory import autorelease_scope
        with _programs._LOCK:
            retained = {name: _publish._TENSORS[name] for name in self.TENSORS
                        if name not in values and name in _publish._TENSORS}
        pending, total = [], 0
        for name, (shape, dtype) in self.TENSORS.items():
            tensor = values.get(name, retained.get(name))
            if tensor is None:
                continue
            if (tensor.dtype != dtype or tensor.ndim != len(shape)
                    or any(actual != limit if limit is not None else not 1 <= actual <= 32
                           for actual, limit in zip(tensor.shape, shape))):
                raise ValueError(f"fixture display tensor {name} has an invalid shape or dtype")
            total += tensor.numel() * tensor.element_size()
            if tensor.device.type == "cpu":
                if name in values:
                    pending.append((name, tensor))
            elif tensor.device != self.device:
                raise ValueError(f"fixture display tensor {name} has an invalid device")
        if total > self.MAX_BYTES:
            raise ValueError("fixture display tensors exceed the aggregate byte limit")
        if not pending:
            return dict(values), 0
        with autorelease_scope(self.device):
            self.guard.check("fixture display copy start")
            try:
                converted = [(name, self._transfer(tensor)) for name, tensor in pending]
            finally:
                self.guard.check("fixture display copy complete")
        return dict(values, **dict(converted)), sum(tensor.numel() * tensor.element_size() for _, tensor in pending)

    def publish(self):
        from .. import _programs
        from . import _publish
        with _publish._publication_scope():
            with _programs._LOCK:
                values = {name: _publish._TENSORS[name] for name in self.TENSORS if name in _publish._TENSORS}
            converted, size = self.prepare(values)
            if size:
                _publish._publish_many({name: tensor for name, tensor in converted.items() if tensor is not values[name]})
            return size


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        self.frozen_scale = None

    def forward(self, x):
        scale = self.frozen_scale
        if scale is None:
            scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        self.last_scale = scale.detach()
        return x * scale * self.weight


class Attention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.pattern = nn.Identity()
        self.frozen_pattern = None

    def forward(self, x):
        batch, length, width = x.shape
        def split(proj):
            return proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        pattern = self.frozen_pattern
        if pattern is None:
            scores = q @ k.transpose(-1, -2) / self.head_dim ** 0.5
            mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            pattern = scores.masked_fill(mask, -torch.inf).softmax(-1)
        pattern = self.pattern(pattern)
        self.last_pattern = pattern.detach()
        y = (pattern @ v).transpose(1, 2).reshape(batch, length, width)
        return self.o_proj(y)


class TranscoderMLP(nn.Module):
    def __init__(self, width=64, latents=256, k=8):
        super().__init__()
        self.k = k
        # Rows of W_enc read a feature. Rows of W_dec write a feature.
        self.W_enc = nn.Parameter(torch.randn(latents, width) / width ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(latents))
        self.W_dec = nn.Parameter(torch.randn(latents, width) / (latents * k) ** 0.5)
        self.b_dec = nn.Parameter(torch.zeros(width))
        self.features = nn.Identity()

    def encode(self, x):
        z = F.linear(x, self.W_enc, self.b_enc).relu()
        values, indices = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter(-1, indices, values)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.features(self.encode(x)))


class Block(nn.Module):
    def __init__(self, width=64, heads=4, latents=256, k=8):
        super().__init__()
        self.input_layernorm = RMSNorm(width)
        self.self_attn = Attention(width, heads)
        self.post_attention_layernorm = RMSNorm(width)
        self.mlp = TranscoderMLP(width, latents, k)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyLM(nn.Module):
    def __init__(self, *, seed=7, layers=4, width=64, heads=4, vocab=512,
                 latents=256, k=8, dtype=torch.float64):
        super().__init__()
        self.config = SimpleNamespace(model_type="caliper_tiny", _name_or_path="tiny-lm",
                                      num_hidden_layers=layers, hidden_size=width,
                                      num_attention_heads=heads, vocab_size=vocab,
                                      rms_norm_eps=1e-6, latents=latents, k=k)
        # Model construction must not change the caller's random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embed_tokens = nn.Embedding(vocab, width)
            self.layers = nn.ModuleList(Block(width, heads, latents, k) for _ in range(layers))
            self.norm = RMSNorm(width)
            self.lm_head = nn.Linear(width, vocab, bias=False)
        self.to(dtype=dtype)
        self.eval()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("supply input_ids or inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for block in self.layers:
            x = block(x)
        return SimpleNamespace(logits=self.lm_head(self.norm(x)))

    @contextmanager
    def frozen_attention(self, *, norms=False):
        """Freeze constants from the last forward pass, then restore prior values."""
        saved = []
        for module in self.modules():
            key = "frozen_pattern" if isinstance(module, Attention) else (
                "frozen_scale" if norms and isinstance(module, RMSNorm) else None)
            if key is not None:
                saved.append((module, key, getattr(module, key)))
                source = "last_pattern" if key == "frozen_pattern" else "last_scale"
                setattr(module, key, getattr(module, source).detach())
        try:
            yield self
        finally:
            for module, key, value in saved:
                setattr(module, key, value)


class TinyTokenizer:
    """A byte tokenizer with a fixed 512-token vocabulary and a BOS token."""
    vocab_size = 512
    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258

    def encode(self, text, add_special_tokens=True):
        ids = list(text.encode("utf-8"))
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return bytes(int(x) for x in ids if 0 <= int(x) < 256).decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.decode([ids]) if ids < 256 else f"<token{ids}>"
        return [self.convert_ids_to_tokens(int(x)) for x in ids]

    def __call__(self, text, return_tensors="pt", **kwargs):
        ids = self.encode(text, add_special_tokens=kwargs.get("add_special_tokens", True))
        return {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}



def fixture_resources(folder):
    """Create the seeded dictionary and a small real example index once."""
    from safetensors.torch import save_file
    from ._dictionary import load_dictionary, model_identity
    from ._index import build_index
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("fixture resource folder must be empty")
    model, tokenizer = TinyLM(), TinyTokenizer()
    count = len(model.layers)
    manifest = {
        "format_version": 1, "model_id": "tiny-lm", "model_hash": model_identity(model),
        "revision": "fixture-v1-seed-7", "layers": count, "width": model.config.hidden_size,
        "latents": model.config.latents, "activation": "topk", "k": model.config.k,
        "read_points": [f"layers.{i}.mlp_in" for i in range(count)],
        "write_points": [f"layers.{i}.mlp_out" for i in range(count)],
        "training": {"dataset_id": "fixture", "dataset_config": "seeded-text",
                     "dataset_revision": "fixture-v1", "split": "train", "partitions": {},
                     "tokens": 0, "elapsed_seconds": 0, "peak_memory_bytes": 0},
        "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None, "evaluated_tokens": 0},
    }
    for i, block in enumerate(model.layers):
        save_file({name: p.detach().clone() for name, p in block.mlp.named_parameters()}, folder / f"layer_{i:02d}.safetensors")
    (folder / "dictionary.json").write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    dictionary = load_dictionary(folder, model=model)
    texts = ("AB CD EF GH", "The red cube.", "A blue circle.", "AB CD AB CD", "A small test.",
             "The red cube is beside a tree.")
    documents = [(f"fixture-{i}-{copy}", tokenizer.encode(text))
                 for i, text in enumerate(texts) for copy in range(16)]
    build_index(folder, documents, model=model, tokenizer=tokenizer, dictionary=dictionary,
                provenance={"partition": "fixture-v1"}, context_length=32)
    return folder


def load_resources(model_name, dictionary_path, *, replication_profile=None):
    """Load local safe weights for a fixture or an exact pinned text model."""
    from ._dictionary import load_dictionary, _record_verified_hf_load
    from ._index import FeatureIndex
    from ._memory import MemoryGuard
    if replication_profile not in (None, "published_dallas_v1"):
        raise ValueError("unknown replication profile")
    reference = replication_profile == "published_dallas_v1"
    if reference:
        from ._reference_snapshots import model_snapshot, remember_model_snapshot, scope_snapshot
        snapshot = model_snapshot(model_name)
    folder = Path(dictionary_path).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("dictionary must name a local resource folder")
    guard = None
    if model_name == "tiny-lm":
        model, tokenizer = TinyLM(), TinyTokenizer()
    elif reference or Path(model_name).name == GEMMA_REVISION:
        from transformers import AutoTokenizer
        snapshot = Path(model_name).resolve(strict=True)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        guard = MemoryGuard(40, device)
        if reference:
            from ._dictionary import _read_metadata
            from ._gemma_import_identity import _REFERENCE_INPUT, _REFERENCE_PROFILE
            metadata, _ = _read_metadata(folder / "dictionary.json", guard)
            if metadata.get("reference_input") != _REFERENCE_INPUT:
                raise ValueError("published replication requires the BF16 original-coordinate artifact")
            scope_snapshot()
        options = {"dtype": torch.bfloat16, "reference_profile": _REFERENCE_PROFILE} if reference else {}
        model = _load_gemma_model(snapshot, device=device, memory=guard, **options)
        if reference:
            remember_model_snapshot(model, snapshot)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    else:
        from ._qwen_family import load_hosted_model
        model, tokenizer, guard = load_hosted_model(
            model_name, memory_type=MemoryGuard, admit_load=_admit_model_load,
            admit_transfer=_admit_model_transfer, record_load=_record_verified_hf_load)
    options = {"memory_limit_gb": 40} if getattr(model.config, "model_type", None) == "gemma2" else {}
    dictionary = load_dictionary(folder, model=model, device=next(model.parameters()).device,
                                 dtype=next(model.parameters()).dtype, memory=guard, **options)
    if guard is not None:
        guard.check("inference dictionary load")
    if reference:
        from ._qualification import qualify_import, require_reference_index
        qualify_import(folder, dictionary.metadata, model=model, memory=guard)
    index = FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary,
                         memory=guard, **options)
    if reference:
        try:
            require_reference_index(index.metadata)
        except BaseException:
            index.close()
            raise
    return model, tokenizer, dictionary, index, guard


class HostedService(InterpretationService):
    """Run steering updates and queued requests inside the same boundary."""
    def __init__(self, owner):
        self.owner = owner
        super().__init__(model=owner.model, tokenizer=owner.tokenizer, dictionary=owner.dictionary, index=owner.index,
                         handlers={"patch.sweep": owner.sweep, "trace.validate": owner.validate})
        if getattr(owner.model.config, "model_type", None) == "gemma2":
            self.factories["tracer"] = self._gemma_tracer
        self.boundary_running = False
        self.boundary_thread = None

    def _check_intervention_budget(self, ivs):
        _check_checkpoint_interventions(ivs)

    def _gemma_tracer(self):
        from . import Tracer
        return Tracer._for_service(
            self.model, self.tokenizer, self._resource("dictionary"),
            controls=getattr(self, "controls", None), memory_limit_gb=40, memory=self.owner.guard)

    def submit(self, metadata, reply, *, acknowledge=None):
        status, result = super().submit(metadata, reply, acknowledge=acknowledge)
        if metadata["method"] == "trace.status":
            result["inference"] = self.owner.inference_status()
        return status, result

    def stop(self, failure, step=None):
        error = super().stop(failure, step)
        self.owner._latch_failure(self.failure)
        self.owner.controls._terminate_actions(self.failure)
        return error

    def _execute(self, method, params, step):
        if method != "trace.run":
            return super()._execute(method, params, step)
        published = getattr(self.owner, "replication_profile", None) == "published_dallas_v1"
        if published:
            from ._replication_export import export_graph, trace_transaction
            from ._trace import _Deadline
            check = _Deadline()
        if method == "trace.run":
            prompt = self.owner.prompt
            with self.owner.controls.lock:
                prompt_version = prompt._version
            prompt_inputs(self.model, self.tokenizer, params["prompt"], profile=params.get("profile", "caliper"))
            # Check the control limits before trace admission without an edit.
            with self.owner.controls.lock:
                entry = prompt.entry()
                entry["value"] = params["prompt"]
                self.owner.controls._check_param_budget(prompt, entry)
            traced_specifications = self.owner.interventions.specifications()

        def complete(result, transaction=None):
            if transaction is not None:
                export_graph(self.graph, self.owner.replication.directory, memory=self.tracer.guard,
                             check=check, evidence=transaction.evidence)
            publish_prompt = False
            with self.owner.controls.lock:
                if transaction is not None:
                    transaction.controls()
                # Commit only after success. A script edit or pending viewer edit
                # keeps ownership even when its text equals the previous value.
                if (self.owner.controls.params.get(prompt.name) is prompt
                        and prompt._version == prompt_version and prompt.pending is None):
                    self.owner.controls._check_param_budget(prompt, entry)
                    prompt._value = params["prompt"]
                    prompt._version += 1
                    self.owner.checkpoint_prompt = params["prompt"]
                    self.owner.controls._update_busy()
                    publish_prompt = True
            self.owner.traced_prompt = params["prompt"]
            self.owner.traced_specifications = traced_specifications
            # Queued replacements may differ from the hooks that produced this graph.
            self.owner.traced_state = deepcopy(traced_specifications)
            self.owner._reconcile_selection(self.graph)
            if publish_prompt:
                self.owner.controls._publish()
        if published:
            with trace_transaction(self) as transaction:
                return super()._execute(method, params, step,
                                        complete=lambda result: complete(result, transaction))
        result = super()._execute(method, params, step)
        complete(result)
        return result

    def step(self, step=None):
        with self.lock:
            if self.running or self.boundary_running or self.closed:
                return 0
            self.boundary_running = True
            self.boundary_thread = threading.get_ident()
        try:
            self.owner._check_stop()
            with internal_forward():
                try:
                    self.owner.update_controls()
                except ValueError as exc:
                    self.owner.status.set(bounded_text(str(exc)))
                count = super().step(step)
                try:
                    self.owner.update_output()
                except ValueError as exc:
                    self.owner.status.set(bounded_text(str(exc)))
                return count
        except self.stop_exception as exc:
            self.stop(exc, step)
            raise self.failure
        finally:
            with self.idle:
                self.boundary_running = False
                self.boundary_thread = None
                self.idle.notify_all()

    def _check_close_thread(self):
        super()._check_close_thread()
        with self.lock:
            if self.boundary_thread == threading.get_ident():
                raise Refused("interpret.busy", "An active hosted boundary cannot close its own service.")

    def close(self):
        with self.idle:
            self._check_close_thread()
            self.closed = True
            while self.boundary_running:
                self.idle.wait()
        super().close()


class HostedInference:
    """Own one model, its controls, and safe checkpoint metadata."""
    def __init__(self, model_name, dictionary_path, prompt, *, resources=None,
                 replication_profile=None, private_output_directory=None):
        if replication_profile not in (None, "published_dallas_v1"):
            raise ValueError("unknown replication profile")
        if replication_profile is not None and private_output_directory is None:
            raise ValueError("published replication requires a private output directory")
        self.replication_profile = replication_profile
        self.private_output_directory = private_output_directory
        self.replication = None
        display_guard = _fixture_display_guard(model_name)
        self.model_name = model_name
        self.dictionary_path = str(Path(dictionary_path).resolve())
        self.model, self.tokenizer, self.dictionary, self.index, self.guard = (
            (load_resources(model_name, dictionary_path, replication_profile=replication_profile)
             if replication_profile is not None else load_resources(model_name, dictionary_path))
            if resources is None else resources)
        try:
            if display_guard is not None and self.guard is None:
                from ._memory import MemoryGuard
                self.guard = MemoryGuard(12, next(self.model.parameters()).device)
            self.guard = (_RunGuard(self.guard, extra=display_guard, on_stop=self._latch_failure)
                          if self.guard is not None or display_guard is not None else None)
            self.fixture_display = _FixtureDisplay(self.guard) if display_guard is not None else None
            prompt_inputs(self.model, self.tokenizer, prompt)
            self.initial_prompt = prompt
            # Rejected editor text stays visible, but checkpoints retain an accepted prompt.
            self.checkpoint_prompt = prompt
            self.restored = None
            self.handle = self.service = self.managed = None
            self._failure = None
            self._stop_checkpointed = False
            self._verified_bindings = None
            self._inference_lock = threading.Lock()
            self._inference_ready = False
            self._comparison = None
            self._restore_generation(0)
            self.selection = None
            self._owned_clamp = None
            self._owned_applied_spec = None
            self._clamp_selection = None
            self._selection_graph = None
            self.traced_prompt = None
            self.traced_specifications = None
            self.traced_state = None
            self._output_key = None
            self._output_action_id = None
            self._clamp_version = None
            self._selected_version = None
            self._retrace = False
            self.parameters, self.actions = [], []
        except BaseException as error:
            if resources is None:
                pending = [error]
                try:
                    self.index.close()
                except BaseException as cleanup_error:
                    # Cleanup must not replace the constructor refusal.
                    pending.append(cleanup_error)
                finally:
                    self.model = self.tokenizer = self.dictionary = self.index = self.guard = None
                    self.fixture_display = display_guard = None
                    # Retained helper frames must not keep owned resources alive.
                    seen = set()
                    while pending:
                        failure = pending.pop()
                        if id(failure) not in seen:
                            seen.add(id(failure))
                            traceback.clear_frames(failure.__traceback__)
                            pending.extend(link for link in (failure.__cause__, failure.__context__)
                                           if link is not None)
                    del failure
            raise

    def _latch_failure(self, failure):
        """Latch a stop without callbacks, including while the Controls lock is held."""
        if self._failure is None:
            self._failure = failure
        if self.guard is not None:
            self.guard.failure = self._failure
        if self.service is not None:
            with self.service.lock:
                if self.service.failure is None:
                    self.service.failure = self._failure

    def _bindings(self):
        from ._dictionary import model_identity
        from ._memory import CleanStop, family_memory_limit
        if self._failure is not None and self._verified_bindings is not None:
            return deepcopy(self._verified_bindings)
        try:
            limit = None if self.guard is None else family_memory_limit(self.model, memory=self.guard)
            identity = model_identity(self.model, memory=self.guard,
                                      memory_limit_gb=limit)
        except CleanStop as exc:
            self._latch_failure(exc)
            raise self._failure
        bindings = {"model": self.model_name, "model_hash": identity,
                    "dictionary": self.dictionary_path, "dictionary_hash": self.index.metadata["dictionary_hash"],
                    "tokenizer_hash": self.index.metadata["tokenizer_hash"]}
        if self.replication_profile is not None:
            bindings["replication_profile"] = self.replication_profile
            bindings["dtype"] = str(next(self.model.parameters()).dtype)
        self._verified_bindings = deepcopy(bindings)
        return bindings

    def load_checkpoint(self, record, *, params=None):
        if (not checkpoint_metadata(record) or record.get("format") != CHECKPOINT_FORMAT or type(record.get("version")) is not int
                or record["version"] != 1 or record.get("bindings") != self._bindings()):
            raise ValueError("inference checkpoint resource bindings do not match")
        if type(record.get("step")) is not int or not 0 <= record["step"] < 2**53:
            raise ValueError("inference checkpoint step is invalid")
        if type(record.get("params")) is not dict or type(record.get("interventions")) is not dict:
            raise ValueError("inference checkpoint state is invalid")
        # A fork can repair a rejected prompt from an older checkpoint. All other
        # saved state still passes its original validation before the run starts.
        prompt = (params or {}).get("prompt", record["params"].get("prompt"))
        prompt_inputs(self.model, self.tokenizer, prompt)
        selection = record.get("selection")
        if selection is not None:
            if type(selection) is not dict or set(selection) != {"layer", "feature", "position"}:
                raise ValueError("inference checkpoint selection is invalid")
            limits = {"layer": len(self.dictionary.layers), "feature": self.dictionary.metadata["latents"], "position": 32}
            if any(type(selection[k]) is not int or not 0 <= selection[k] < limit for k, limit in limits.items()):
                raise ValueError("inference checkpoint selection is out of range")
        generation_id = record.get("generation_id", 0)
        if type(generation_id) is not int or not 0 <= generation_id < 2**53:
            raise ValueError("inference checkpoint generation identity is invalid")
        owned = record.get("owned_clamp")
        if owned is not None:
            from .._intervention_protocol import specification
            if (not specification(owned) or owned.get("op", owned.get("kind")) != "clamp"
                    or not any(item.get("specification") == owned and not item.get("clear")
                               and item.get("state") in ("active", "pending")
                               for item in record["interventions"].get("interventions", [])
                               if type(item) is dict)):
                raise ValueError("inference checkpoint clamp ownership is invalid")
        self.restored = deepcopy(record)
        self.restored["params"]["prompt"] = prompt

    def state_dict(self):
        bindings = self._bindings()
        with self.interventions._ctl.lock, self._inference_lock:
            owned = self._owned_record()
            return {"bindings": bindings, "interventions": self.interventions._state_dict(),
                    "selection": deepcopy(self.selection),
                    "owned_clamp": deepcopy(owned.spec) if owned is not None else None,
                    "generation_id": self._generation["id"]}

    def _restore_generation(self, generation_id):
        with self._inference_lock:
            self._generation = {"id": generation_id, "prompt_hash": None, "interventions_hash": None,
                                "token_ids": [[], []], "counts": [0, 0], "completed_step": None,
                                "text_hashes": [None, None]}

    def _input_state(self):
        with self.controls.lock:
            specifications = [deepcopy(iv.applied_spec) for iv in self.controls._ivs.values()
                              if iv.state == "active" and iv.hook is not None and iv.applied_spec is not None]
            return self.prompt.value, specifications

    @staticmethod
    def _input_hashes(state):
        prompt, specifications = state
        encoded = json.dumps(specifications, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
        return {"prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "interventions_hash": hashlib.sha256(encoded).hexdigest()}

    def inference_status(self):
        with self._inference_lock:
            if not self._inference_ready:
                # Transport requests can precede prompt and executor setup.
                # Do not wait for dictionary copies or acquire the Controls lock.
                return {"state": "starting", "prompt_hash": None, "interventions_hash": None,
                        "comparison": None, "generation": deepcopy(self._generation),
                        "dictionary_validation": None}
        input_state = self._input_state()
        binding = self._input_hashes(input_state)
        if self.replication is not None:
            binding["replication"] = self.replication.status(*input_state)
        identity = self.interventions.executor._dictionary_status()
        with self._inference_lock:
            return dict(binding, state="ready", comparison=deepcopy(self._comparison), generation=deepcopy(self._generation),
                        dictionary_validation=identity)

    def _owned_record(self):
        """Resolve slider ownership by request or retained effect identity under the lock."""
        if self.interventions._ctl._iv_model is not self.interventions or self._owned_clamp is None:
            return None
        iv = self.interventions._ctl._ivs.get(intervention_key(self._owned_clamp))
        if iv is None or iv.clear or iv.state == "ended":
            self._owned_clamp = self._owned_applied_spec = None
            return None
        if iv.spec is not self._owned_clamp:
            # A refused external request can retain the exact slider effect.
            # A pending or accepted external request still owns its specification.
            if (not iv.replace and iv.hook is not None and iv.error_code == "intervene.hook_failed"
                    and iv.applied_spec is self._owned_applied_spec and iv.spec == iv.applied_spec):
                self._owned_clamp = iv.spec
            else:
                if not iv.replace:
                    self._owned_clamp = self._owned_applied_spec = None
                return None
        if iv.hook is not None:
            self._owned_applied_spec = iv.applied_spec
        return iv

    def _restore_owned_clamp(self, owned):
        with self.controls.lock:
            self._check_watch()
            self._owned_clamp = self._owned_applied_spec = None
            if owned is not None:
                iv = self.controls._ivs.get(intervention_key(owned))
                if iv is not None and iv.spec == owned and not iv.clear and iv.state != "ended":
                    self._owned_clamp = iv.spec

    def _discard_owned_clamp(self):
        """Mark only the owned record for removal with the Controls lock held."""
        iv = self._owned_record()
        if iv is None:
            return
        self._owned_clamp = self._owned_applied_spec = None
        iv.clear = True
        self.controls._iv_revision += 1
        self.controls._iv_due = True
        self.controls.counters["interventions_in"] += 1
        self.controls._update_busy()
        self.controls.cond.notify_all()

    def start(self):
        import caliper
        from .. import _managed
        self.controls = caliper._CTL
        self._bindings()
        self.managed = _managed.current()
        if self.managed is not None:
            self.managed.inference = self
            self.managed.kept.extension = ".json"
        try:
            # Managed watch starts the stream. Its first handshake must advertise
            # interpretation even when dictionary identity takes time to compute.
            self.service = register_service(HostedService(self), controls=self.controls)
            self.handle = caliper.watch(self.model, activations=False)
            self.interventions = self.controls.intervention_service
            self.interventions.configure(dictionary=self.dictionary, tokenizer=self.tokenizer,
                                         resource_guard=self.guard)
            self.interventions.executor._prepare_dictionary_identity()
            self.output = Output(self.model, self.tokenizer, self.interventions, guard=self.guard)
            if self.replication_profile is not None:
                from ._replication_actions import PublishedCases, PROMPTS
                if self.initial_prompt != PROMPTS["dallas"]:
                    raise ValueError("published replication requires the exact Dallas prompt")
                self.replication = PublishedCases(self.model, self.tokenizer, self.interventions, guard=self.guard,
                                                  output_directory=self.private_output_directory)
                self.replication.prepare()
                self.output.attention_scope = self.replication.output_scope
            def param(name, value, **kwargs):
                result = caliper.param(name, value, **kwargs)
                self.parameters.append(result)
                return result
            self.prompt = param("prompt", self.initial_prompt)
            prompt_inputs(self.model, self.tokenizer, self.prompt.value)
            self.checkpoint_prompt = self.prompt.value
            self.selected = param("selected_node", -1, min=-1,
                                  max=8191 if self.replication_profile == "published_dallas_v1" else 511)
            self.clamp = param("clamp", 0., min=-4., max=4.)
            self.clamp_status = param("clamp.status", "No feature selected. No slider clamp.")
            self.retrace = param("retrace", False)
            self.baseline_text = param("output.baseline", "")
            self.steered_text = param("output.steered", "")
            self.status = param("output.status", "ready")
            self.actions = [caliper.action("generate", self.generate), caliper.action("reset", self.reset),
                            caliper.action("apply_clamp", self.apply_clamp)]
            if self.replication is not None:
                from ._replication_actions import CASES, PROTOCOL
                for cell, case in CASES.items():
                    self.actions.append(caliper.action(f"{PROTOCOL}.cell{cell}",
                                        lambda cell=cell: self.apply_published_case(cell), help=case["label"]))
            if self.restored is not None:
                self.interventions.load_state_dict(self.restored["interventions"])
                self.selection = deepcopy(self.restored.get("selection"))
                self._restore_owned_clamp(self.restored.get("owned_clamp"))
                self._restore_generation(self.restored.get("generation_id", 0))
            # Fork overrides follow preparation and parent restore, before the first boundary.
            if self.managed is not None and self.managed.interventions:
                self.managed._intervene()
            self._clamp_version, self._selected_version = self.clamp._version, self.selected._version
            with self._inference_lock:
                self._inference_ready = True
            self.boundary()
            return self
        except BaseException:
            self.close()
            raise

    def apply_published_case(self, cell):
        """Apply one prepared case and publish its actual model output."""
        from ._replication_actions import PROMPTS
        self._check_stop()
        if self.replication is None:
            raise ValueError("published replication is not configured")
        with self._watch_resources():
            with self.controls.lock:
                if self.prompt.value != PROMPTS["dallas"] or self.prompt.pending is not None:
                    raise ValueError("published actions require the unchanged Dallas prompt")
                prompt_version = self.prompt._version
            def check():
                self._check_watch()
                if self.prompt._version != prompt_version or self.prompt.pending is not None:
                    raise ValueError("published prompt changed during the action")
            def prepare(logits):
                transform = (None if self.fixture_display is None else
                             lambda values: self.fixture_display.prepare(values)[0])
                publication, _ = self.output.prepare_logits(logits, transform=transform)
                if self.guard is not None:
                    self.guard.check("published case complete")
                return publication
            result = self.replication.run(cell, prepare=prepare, check=check)
            self._owned_clamp = self._owned_applied_spec = None
            key = self._input_state()
            with self._inference_lock:
                self._comparison = dict(self._input_hashes(key), argmax_token_ids=result["argmax_token_ids"],
                                        baseline_top_text=result["baseline_top_text"],
                                        baseline_top_probability=result["baseline_top_probability"],
                                        completed_step=self.controls.n, case_id=result["case_id"],
                                        compute_publish_seconds=result["action_seconds"])
            self._output_key = deepcopy(key)
        return f"{result['label']} completed. Evidence: {result['metadata_file']}"

    def _check_watch(self):
        """Require this watch and stable resources with the Controls lock held."""
        service = self.interventions
        if self.controls._iv_model is not service:
            raise ValueError("the intervention service is no longer attached")
        if (self.controls._iv_attaching or service._operation_owner is not None
                and service._operation_owner[0] in ("configure", "watch")):
            raise ValueError("an intervention lifecycle operation is in progress")
        if (service.executor.dictionary is not self.dictionary
                or service.executor.tokenizer is not self.tokenizer):
            raise ValueError("the intervention resources no longer match the hosted resources")

    @contextmanager
    def _watch_resources(self):
        """Pin this watch until hosted publication completes without the lock."""
        service = self.interventions
        with self.controls.lock:
            self._check_watch()
            service._readers += 1
        try:
            yield
            with self.controls.lock:
                self._check_watch()
        finally:
            with self.controls.lock:
                service._readers -= 1
                self.controls._finish_retired(service)

    def _apply(self):
        with self.controls.lock:
            self._check_watch()
            changed, ready = self.controls._intervention_boundary()
            # The boundary releases the lock. Detachment must precede any hosted
            # counter update or publication, including a deferred boundary.
            self._check_watch()
            if not ready:
                self.controls._iv_boundary_failures += 1
            self._owned_record()
            self.controls._update_busy()
        if changed:
            with self._watch_resources():
                self.controls._publish()

    def _reconcile_selection(self, graph):
        """Keep the semantic feature selected only while its displayed row exists."""
        row = -1
        if graph is not None and self.selection is not None:
            row = next((i for i, node in enumerate(graph.nodes) if node["kind"] == "feature"
                        and all(node[key] == value for key, value in self.selection.items())), -1)
        if row == -1:
            self.selection = None
        if self.selected.value != row:
            self.selected.set(row)
        self._selected_version = self.selected._version
        self._selection_graph = graph

    def _update_selection(self):
        graph = self.service.graph
        if graph is not self._selection_graph or (graph is None and self.selection is not None):
            self._reconcile_selection(graph)
        if self.selected._version != self._selected_version:
            node = self.selected.value
            self.selection = None
            self._selected_version = self.selected._version
            if node == -1:
                return
            if graph is None or node >= len(graph.nodes):
                raise ValueError("selected_node requires a completed trace")
            record = graph.nodes[node]
            if record["kind"] != "feature":
                raise ValueError("selected_node must identify a feature")
            self.selection = {k: record[k] for k in ("layer", "feature", "position")}

    def update_controls(self):
        with self.controls.lock:
            self._check_watch()
        clamp_changed = self.clamp._version != self._clamp_version
        # Consume the edit even when prompt, selection, or clamp validation fails.
        self._clamp_version = self.clamp._version
        try:
            prompt = self.prompt.value
            prompt_inputs(self.model, self.tokenizer, prompt)
            self.checkpoint_prompt = prompt
            self._update_selection()
            if clamp_changed:
                from .._controls import _cadence_excluded
                with _cadence_excluded():
                    if not self._set_clamp():
                        self.status.set("The selected feature clamp did not commit. Retry the edit.")
        finally:
            self._sync_clamp_control(preserve_edit=clamp_changed)

    def _cancel_clamp_request(self, record, request, previous):
        """Cancel only an uncommitted hosted request, with the Controls lock held."""
        key = intervention_key(record.spec)
        if (self.controls._iv_model is not self.interventions
                or self.controls._ivs.get(key) is not record or record.spec is not request
                or record.clear or record.error is not None):
            return
        hook = None if previous is None else previous.hook
        applied = None if previous is None else previous.applied_spec
        if record.hook is not hook or record.applied_spec is not applied:
            return
        if previous is None:
            del self.controls._ivs[key]
        else:
            for field in record.__slots__:
                setattr(record, field, getattr(previous, field))
        self.controls._iv_revision += 1
        self.controls._iv_due = True
        self.controls._update_busy()
        self.controls.cond.notify_all()

    def _set_clamp(self, *, require_commit=False):
        """Retain the old slider effect until its replacement commits.

        A cross-layer request cannot wait beyond this call because checkpoints
        identify one slider owner. Cancel its exact uncommitted request on deferral.
        Explicit actions also require a committed effect before output publication.
        """
        self._check_stop()
        if self.selection is None:
            raise ValueError("clamp requires a selected feature")
        layer, feature, position = (self.selection[k] for k in ("layer", "feature", "position"))
        maximum = float(self.index.show(layer, feature)["maximum"])
        if not torch.isfinite(torch.tensor(maximum)) or maximum <= 0:
            raise ValueError("selected feature has no positive indexed maximum")
        point = self.dictionary.metadata["write_points"][layer]
        with self.controls.lock:
            # All submissions, including changed clamps, belong to this watch.
            # Check before request construction can inspect detached resources.
            self._check_watch()
            spec = self.interventions.executor.validate({"kind": "clamp", "point": point, "feature": feature,
                "positions": [position], "value": self.clamp.value * maximum,
                "dictionary": self.interventions.executor.dictionary_id}, resources=False)
            owned = self._owned_record()
            if (owned is not None and owned.state == "active" and owned.hook is not None
                    and not owned.replace and owned.spec == owned.applied_spec == spec):
                # Pin resources while the unchanged clamp verifies dictionary content.
                service = self.interventions
                service._readers += 1
                try:
                    with self.controls._outside_lock():
                        service.executor.validate(spec)
                finally:
                    service._readers -= 1
                    self.controls._finish_retired(service)
                self._check_stop()
                if (self.controls._iv_model is not service or self._owned_record() is not owned
                        or owned.state != "active" or owned.hook is None or owned.replace
                        or owned.spec != spec or owned.applied_spec != spec):
                    raise ValueError("intervention state changed during clamp validation")
                return True
            # Admission and installation each validate resources. Do not hash the
            # same dictionary a third time merely to construct the request.
            previous = self.controls._ivs.get(intervention_key(spec))
            previous = copy(previous) if previous is not None else None
            owner, applied = self._owned_clamp, self._owned_applied_spec
            cross_layer = owner is not None and intervention_key(owner) != intervention_key(spec)
            refused = self.controls._intervene({"set": spec})
            if refused is not None:
                raise ValueError(refused[1])
            record = self.controls._ivs[intervention_key(spec)]
            request = record.spec
            failures = self.controls._iv_boundary_failures
            self.controls._update_busy()
            if not cross_layer:
                self._owned_clamp = request
        try:
            self._apply()
        except BaseException:
            if cross_layer:
                with self.controls.lock:
                    self._cancel_clamp_request(record, request, previous)
            raise
        with self.controls.lock:
            self._check_watch()
            committed = (self.controls._ivs.get(intervention_key(spec)) is record and record.spec is request
                         and record.state == "active" and record.hook is not None and not record.clear
                         and not getattr(record.hook, "removed", False)
                         and not record.replace and record.applied_spec == spec)
            if committed:
                if cross_layer:
                    self._discard_owned_clamp()
                self._owned_clamp, self._owned_applied_spec = request, record.applied_spec
            else:
                if self.controls._iv_boundary_failures == failures:
                    self.controls._iv_boundary_failures += 1
                if cross_layer or require_commit:
                    self._cancel_clamp_request(record, request, previous)
                if cross_layer or require_commit or record.error is not None:
                    self._owned_clamp, self._owned_applied_spec = owner, applied
                if require_commit:
                    raise ValueError(record.error or "The selected feature clamp did not commit. Retry the action.")
                return False
        if cross_layer:
            self._apply()
            with self.controls.lock:
                self._check_watch()
                if require_commit and (self.controls._ivs.get(intervention_key(spec)) is not record
                        or record.hook is None or getattr(record.hook, "removed", False)
                        or record.clear or record.applied_spec != spec):
                    raise ValueError("The selected feature clamp changed before completion. Retry the action.")
        return True

    def _sync_clamp_control(self, *, preserve_edit=False):
        with self.controls.lock:
            if self.controls._iv_model is not self.interventions:
                return
        with self._watch_resources():
            self._sync_clamp_status(preserve_edit=preserve_edit)

    def _sync_clamp_status(self, *, preserve_edit):
        reset_input = False
        if self.selection != self._clamp_selection:
            self._clamp_selection = deepcopy(self.selection)
            with self.controls.lock:
                if (not preserve_edit and self.clamp._version == self._clamp_version
                        and self.clamp.pending is None):
                    # Preserve a viewer edit that arrived after this boundary began.
                    self.clamp._value = 0.
                    self.clamp._version += 1
                    self._clamp_version = self.clamp._version
                    reset_input = True
            if reset_input:
                self.controls._publish()
        selected = "No feature selected."
        if self.selection is not None:
            selected = (f"Selected layer {self.selection['layer']}, feature {self.selection['feature']}, "
                        f"position {self.selection['position']}.")
        live = "No slider clamp."
        with self.controls.lock:
            owned = self._owned_record()
            if owned is not None and owned.hook is not None and owned.applied_spec is not None:
                spec = owned.applied_spec
                live = (f"Live slider clamp: {spec['point']}, feature {spec['feature']}, "
                        f"position {spec['positions'][0]}, value {spec['value']:g}.")
        text = bounded_text(f"{selected} Clamp input: {self.clamp.value:g} times the indexed maximum. {live}")
        if self.clamp_status.value != text:
            self.clamp_status.set(text)

    def apply_clamp(self):
        self._check_stop()
        with internal_forward():
            self.update_controls()
            try:
                self._set_clamp(require_commit=True)
            finally:
                self._sync_clamp_control()
            with self.controls.lock:
                action = self.controls.actions["apply_clamp"]
                refresh = not action.running or self._output_action_id != action.run_id
            self.update_output(force=refresh)
        return "The selected feature clamp is applied."

    def update_output(self, *, force=False):
        with self._watch_resources():
            self._publish_output(force=force)

    def _publish_output(self, *, force):
        try:
            self._update_output(force=force)
        except ValueError:
            # Completed requests can publish CPU tensors before comparison fails.
            # Actions and service boundaries must both synchronize those tensors.
            if self.fixture_display is not None:
                self.fixture_display.publish()
            raise
        if self.fixture_display is not None:
            self.fixture_display.publish()

    def _update_output(self, *, force=False):
        key = self._input_state()
        changed = self._output_key != key
        if changed or force:
            from .._controls import _cadence_excluded
            with _cadence_excluded():
                with self.controls.lock:
                    # A later request must not reuse a comparison that already began.
                    action_id = self.controls.actions["apply_clamp"].run_id
                started = time.perf_counter()
                comparison = self.output.compare(key[0])
                seconds = time.perf_counter() - started
                if self._input_state() != key:
                    raise ValueError("inference state changed during the output comparison")
                with self._inference_lock:
                    self._comparison = dict(self._input_hashes(key), argmax_token_ids=comparison["argmax_token_ids"],
                                            baseline_top_text=comparison["baseline_top_text"],
                                            baseline_top_probability=comparison["baseline_top_probability"],
                                            completed_step=self.controls.n, compute_publish_seconds=seconds)
                self._output_key = deepcopy(key)
                self._output_action_id = action_id
                self.status.set("ready")
        if self.retrace.value and (changed or not self._retrace):
            if (self.traced_prompt, self.traced_specifications) != key:
                from .._controls import _cadence_excluded
                with _cadence_excluded():
                    response = self.service._retrace({"prompt": key[0]}, self.controls.n)
                if response is not None and response[0] == "error":
                    self.status.set(response[2]["message"])
        self._retrace = self.retrace.value
        self._sync_clamp_control()
        if self.guard is not None:
            self.guard.check("inference boundary", force=False)

    def generate(self):
        self._check_stop()
        with self._watch_resources():
            return self._generate()

    def _generate(self):
        state = self._input_state()
        with self._inference_lock:
            generation_id = self._generation["id"] + 1
        if generation_id >= 2**53:
            raise ValueError("generation identity exceeds the metadata limit")
        with internal_forward():
            result = self.output.generate(state[0])
        ids = result.get("token_ids")
        if (type(ids) is not list or len(ids) != 2 or any(type(row) is not list or len(row) != 24
                or any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in row)
                for row in ids)):
            raise ValueError("generation requires exactly two rows of 24 token IDs")
        texts = [result.get("baseline"), result.get("steered")]
        expected = [bounded_text(self.tokenizer.decode(row)) for row in ids]
        if texts != expected:
            raise ValueError("generation text does not match its token IDs")
        if self._input_state() != state:
            raise ValueError("inference state changed during generation")
        self.baseline_text.set(texts[0])
        self.steered_text.set(texts[1])
        if self._input_state() != state:
            raise ValueError("inference state changed during generation publication")
        completion = dict(self._input_hashes(state), id=generation_id, token_ids=deepcopy(ids), counts=[24, 24],
                          completed_step=self.controls.n,
                          text_hashes=[hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts])
        with self._inference_lock:
            self._generation = completion
        return "Generated 24 tokens for each output."

    def reset(self):
        self._check_stop()
        with internal_forward():
            with self.controls.lock:
                self._discard_owned_clamp()
            self._apply()
            self.clamp.set(0.)
            self._clamp_version = self.clamp._version
            try:
                try:
                    prompt_inputs(self.model, self.tokenizer, self.prompt.value)
                except ValueError:
                    self.status.set("The clamp is reset. Correct the prompt to update the output.")
                else:
                    self.update_output()
            finally:
                self._sync_clamp_control()
        return "The feature clamp is reset."

    def sweep(self, *, clean_prompt, corrupted_prompt, metric):
        from ._sweep import patch_sweep
        with self._watch_resources():
            return patch_sweep(self.model, self.tokenizer, self.interventions, clean_prompt=clean_prompt,
                               corrupted_prompt=corrupted_prompt, metric=metric, guard=self.guard)

    def validate(self):
        from ._validate import validate_trace
        if self.service.graph is None:
            raise Refused("interpret.missing_trace", "No completed trace is available.")
        with internal_forward(), self.interventions.temporary_state(self.traced_state) as replay:
            return validate_trace(self.model, self.tokenizer, self.dictionary, replay,
                                  self.service.graph, prompt=self.traced_prompt, guard=self.guard)

    def _check_stop(self):
        failure = self._failure or getattr(self.guard, "failure", None)
        if failure is not None:
            raise failure

    @contextmanager
    def _stop_boundary(self):
        from ._memory import CleanStop
        try:
            self._check_stop()
            yield
            self._check_stop()
        except CleanStop as exc:
            if self.service is not None:
                self.service.stop(exc, self.controls.n)
            if self._failure is None:
                self._failure = exc
            if self.managed is not None and not self._stop_checkpointed:
                self._stop_checkpointed = True
                try:
                    self.managed.checkpoint(self.model, None)
                except Exception as checkpoint_error:
                    raise self._failure from checkpoint_error
            raise self._failure

    def boundary(self):
        import caliper
        from .. import _programs
        with self._stop_boundary(), _programs.inference_boundary():
            self.controls.busy = True
            caliper.step()
            if self.guard is not None:
                self.guard.check("inference actions", force=False)
            if self.managed is not None:
                if self.managed.stopping:
                    self.managed.exit()
                self.managed._keep(self.model, None)

    def run(self):
        from .. import _programs
        from .._controls import RUN_PAUSING, RUN_STEPPING
        from ._memory import CleanStop
        while True:
            def pending():
                # The condition already owns Controls.lock. Do not call locking service APIs here.
                due = any(iv.clear or iv.replace or (iv.state == "pending" and
                          (iv.from_step is None or iv.from_step <= self.controls.n + 1))
                          for iv in self.controls._ivs.values())
                return bool(self.controls._npending or self.controls.queue or self.service.pending or due
                            or self.controls.state in (RUN_PAUSING, RUN_STEPPING))
            with _programs.inference_boundary():
                with self.controls.cond:
                    publication_event = self.controls.publication_event
                # Inspect program state outside Controls.lock to preserve lock order.
                blocked = _programs.inference_blocked()
                publication = _programs.inference_pending() and not blocked
                with self.controls.cond:
                    if not publication:
                        with _programs.inference_excluded():
                            self.controls.cond.wait_for(
                                lambda: pending() or self.controls.publication_event != publication_event,
                                timeout=1.)
                    work = pending()
                # Events before or during the wait require a fresh capacity check.
                if not publication:
                    blocked = _programs.inference_blocked()
                    publication = _programs.inference_pending() and not blocked
                try:
                    with self._stop_boundary():
                        if work or publication:
                            self.boundary()
                        else:
                            # Reports and intervention expiry need no new prompt step or forward.
                            # No step measures this routine work. Snapshots and any reentrant
                            # publication checkpoint retain their own measured costs.
                            with _programs.inference_excluded():
                                self.controls.checkpoint()
                except CleanStop:
                    raise
                except Exception as exc:
                    self.status.set(bounded_text(f"{type(exc).__name__}: {exc}"))

    def close(self):
        if self.replication is not None:
            self.replication.close()
        if self.service is not None:
            unregister_service(self.service, controls=self.controls)
            self.service = None
        for item in self.actions + self.parameters:
            item.remove()
        self.actions.clear()
        self.parameters.clear()
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.index.close()
        if self.managed is not None and self.managed.inference is self:
            self.managed.inference = None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Inference Run")
    parser.add_argument("--model")
    parser.add_argument("--dictionary")
    parser.add_argument("--prompt")
    parser.add_argument("--fixture-dir")
    parser.add_argument("--replication-profile", choices=["published_dallas_v1"])
    parser.add_argument("--private-output-directory")
    args = parser.parse_args(argv)
    if args.fixture_dir:
        if any((args.model, args.dictionary, args.prompt)):
            parser.error("fixture creation takes no run arguments")
        fixture_resources(args.fixture_dir)
        return
    if not all((args.model, args.dictionary, args.prompt)):
        parser.error("model, dictionary, and prompt are required")
    run = HostedInference(args.model, args.dictionary, args.prompt,
                          replication_profile=args.replication_profile,
                          private_output_directory=args.private_output_directory)
    try:
        run.start()
        print(f"caliper: inference ready at step {run.controls.n}", flush=True)
        run.run()
    finally:
        run.close()


if __name__ == "__main__":
    # Keep durable fixture type names equal in imported and module entry paths.
    from caliper.interp._hosted import main as hosted_main
    hosted_main()
