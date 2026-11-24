"""Offline hosted inference with deterministic fixture resources and safe forks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ._admission import _admit_model_load, _admit_model_transfer
from ._output import Output, bounded_text, internal_forward, prompt_inputs
from ._service import InterpretationService, Refused, register_service, unregister_service

QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN_ID = "Qwen/Qwen3.5-0.8B"
CHECKPOINT_FORMAT = "caliper.inference"

CHECKPOINT_BYTES = 4 * 1024 * 1024


def checkpoint_metadata(value):
    """Validate local JSON metadata, including full-width intervention vectors."""
    count = 0
    def valid(item, depth=0):
        nonlocal count
        count += 1
        if depth > 8 or count > 262144:
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
        return (type(value) is dict and valid(value)
                and len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) <= CHECKPOINT_BYTES)
    except (ValueError, UnicodeError):
        return False


class _RunGuard:
    """Retain a resource stop after an action or service catches its exception."""
    def __init__(self, guard, *, extra=None, on_stop=None):
        self.guard, self.extra, self.failure = guard, extra, None
        self.on_stop = on_stop

    @property
    def limit(self):
        return min(getattr(guard, "limit", math.inf) for guard in (self.guard, self.extra))

    @property
    def peak(self):
        return max(getattr(guard, "peak", 0) for guard in (self.guard, self.extra))

    @property
    def device(self):
        guard = self.guard if self.guard is not None else self.extra
        return getattr(guard, "device", torch.device("cpu"))

    def check(self, *args, **kwargs):
        self._check("check", *args, **kwargs)

    def _check_interval(self, stage):
        self._check("_check_interval", stage)

    def _check(self, method, *args, **kwargs):
        from ._memory import CleanStop
        if self.failure is not None:
            raise self.failure
        try:
            for guard in (self.guard, self.extra):
                if guard is not None:
                    getattr(guard, method, guard.check)(*args, **kwargs)
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

    def publish(self):
        from . import publish
        from ._memory import autorelease_scope
        from ._publish import _resolve
        pending, total = [], 0
        for name, (shape, dtype) in self.TENSORS.items():
            tensor = _resolve(name)
            if tensor is None:
                continue
            if (tensor.dtype != dtype or tensor.ndim != len(shape)
                    or any(actual != limit if limit is not None else not 1 <= actual <= 32
                           for actual, limit in zip(tensor.shape, shape))):
                raise ValueError(f"fixture display tensor {name} has an invalid shape or dtype")
            total += tensor.numel() * tensor.element_size()
            if tensor.device.type == "cpu":
                pending.append((name, tensor))
            elif tensor.device != self.device:
                raise ValueError(f"fixture display tensor {name} has an invalid device")
        if total > self.MAX_BYTES:
            raise ValueError("fixture display tensors exceed the aggregate byte limit")
        if not pending:
            return 0
        with autorelease_scope(self.device):
            self.guard.check("fixture display copy start")
            try:
                converted = [(name, self._transfer(tensor)) for name, tensor in pending]
            finally:
                self.guard.check("fixture display copy complete")
            for name, tensor in converted:
                publish(name, tensor)
        return sum(tensor.numel() * tensor.element_size() for _, tensor in pending)


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


def load_resources(model_name, dictionary_path):
    """Load only local safe weights and the exact pinned Qwen text model."""
    from ._dictionary import load_dictionary, _record_verified_hf_load
    from ._index import FeatureIndex
    from ._memory import MemoryGuard
    folder = Path(dictionary_path).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("dictionary must name a local resource folder")
    guard = None
    if model_name == "tiny-lm":
        model, tokenizer = TinyLM(), TinyTokenizer()
    else:
        snapshot = Path(model_name).resolve(strict=True)
        if not snapshot.is_dir() or snapshot.name != QWEN_REVISION or snapshot.parent.name != "snapshots":
            raise ValueError("model must name the pinned local Qwen snapshot or tiny-lm")
        if snapshot.parent.parent.name != "models--Qwen--Qwen3.5-0.8B":
            raise ValueError("model snapshot does not identify the pinned Qwen repository")
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
        config = Qwen3_5TextConfig(**json.loads((snapshot / "config.json").read_text())["text_config"])
        if (config.num_hidden_layers, config.hidden_size) != (24, 1024):
            raise ValueError("the pinned text model has unexpected dimensions")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        guard = MemoryGuard(12, device)
        estimate = _admit_model_load(snapshot, guard, device)
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            snapshot, config=config, dtype=torch.float32, attn_implementation="eager",
            local_files_only=True, use_safetensors=True, output_loading_info=True,
            key_mapping={r"^model\.language_model\.": "model."})
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("local model weights do not match the pinned text architecture")
        model.config._name_or_path, model.config._commit_hash = QWEN_ID, QWEN_REVISION
        model.requires_grad_(False).eval()
        guard.check("inference model CPU load")
        _record_verified_hf_load(model, memory=guard)
        _admit_model_transfer(guard, estimate, device)
        model.to(device)
        guard.check("inference model load")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    dictionary = load_dictionary(folder, model=model, device=next(model.parameters()).device,
                                 dtype=next(model.parameters()).dtype)
    if guard is not None:
        guard.check("inference dictionary load")
    index = FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary)
    return model, tokenizer, dictionary, index, guard


class HostedService(InterpretationService):
    """Run steering updates and queued requests inside the same boundary."""
    def __init__(self, owner):
        self.owner = owner
        super().__init__(model=owner.model, tokenizer=owner.tokenizer, dictionary=owner.dictionary, index=owner.index,
                         handlers={"patch.sweep": owner.sweep, "trace.validate": owner.validate})
        self.boundary_running = False
        self.boundary_thread = None

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
        if method == "trace.run":
            prompt_inputs(self.model, self.tokenizer, params["prompt"])
            self.owner.prompt.set(params["prompt"])
            traced_specifications = self.owner.interventions.specifications()
        result = super()._execute(method, params, step)
        if method == "trace.run":
            self.owner.traced_prompt = params["prompt"]
            self.owner.traced_specifications = traced_specifications
            # Queued replacements may differ from the hooks that produced this graph.
            self.owner.traced_state = deepcopy(traced_specifications)
            self.owner._reconcile_selection(self.graph)
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
    def __init__(self, model_name, dictionary_path, prompt, *, resources=None):
        display_guard = _fixture_display_guard(model_name)
        self.model_name = model_name
        self.dictionary_path = str(Path(dictionary_path).resolve())
        self.model, self.tokenizer, self.dictionary, self.index, self.guard = (
            load_resources(model_name, dictionary_path) if resources is None else resources)
        self.guard = (_RunGuard(self.guard, extra=display_guard, on_stop=self._latch_failure)
                      if self.guard is not None or display_guard is not None else None)
        self.fixture_display = _FixtureDisplay(self.guard) if display_guard is not None else None
        prompt_inputs(self.model, self.tokenizer, prompt)
        self.initial_prompt = prompt
        self.restored = None
        self.handle = self.service = self.managed = None
        self._failure = None
        self._stop_checkpointed = False
        self._verified_bindings = None
        self._inference_lock = threading.Lock()
        self._comparison = None
        self._restore_generation(0)
        self.selection = None
        self._owned_clamp = None
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
        from ._memory import CleanStop
        if self._failure is not None and self._verified_bindings is not None:
            return deepcopy(self._verified_bindings)
        try:
            identity = model_identity(self.model)
        except CleanStop as exc:
            self._latch_failure(exc)
            raise self._failure
        bindings = {"model": self.model_name, "model_hash": identity,
                    "dictionary": self.dictionary_path, "dictionary_hash": self.index.metadata["dictionary_hash"],
                    "tokenizer_hash": self.index.metadata["tokenizer_hash"]}
        self._verified_bindings = deepcopy(bindings)
        return bindings

    def load_checkpoint(self, record):
        if (not checkpoint_metadata(record) or record.get("format") != CHECKPOINT_FORMAT or type(record.get("version")) is not int
                or record["version"] != 1 or record.get("bindings") != self._bindings()):
            raise ValueError("inference checkpoint resource bindings do not match")
        if type(record.get("step")) is not int or not 0 <= record["step"] < 2**53:
            raise ValueError("inference checkpoint step is invalid")
        if type(record.get("params")) is not dict or type(record.get("interventions")) is not dict:
            raise ValueError("inference checkpoint state is invalid")
        prompt_inputs(self.model, self.tokenizer, record["params"].get("prompt"))
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
        binding = self._input_hashes(self._input_state())
        identity = self.interventions.executor._dictionary_status()
        with self._inference_lock:
            return dict(binding, comparison=deepcopy(self._comparison), generation=deepcopy(self._generation),
                        dictionary_validation=identity)

    def _owned_record(self):
        """Resolve the exact slider request while the Controls lock is held."""
        if self._owned_clamp is None:
            return None
        iv = self.interventions._ctl._ivs.get((self._owned_clamp["point"], "clamp"))
        if iv is not None and iv.spec is self._owned_clamp and not iv.clear and iv.state != "ended":
            return iv
        return None

    def _restore_owned_clamp(self, owned):
        with self.controls.lock:
            self._owned_clamp = None
            if owned is not None:
                iv = self.controls._ivs.get((owned["point"], "clamp"))
                if iv is not None and iv.spec == owned and not iv.clear and iv.state != "ended":
                    self._owned_clamp = iv.spec

    def _discard_owned_clamp(self):
        """Mark only the owned record for removal with the Controls lock held."""
        iv = self._owned_record()
        self._owned_clamp = None
        if iv is None:
            return
        iv.clear = True
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
            def param(name, value, **kwargs):
                result = caliper.param(name, value, **kwargs)
                self.parameters.append(result)
                return result
            self.prompt = param("prompt", self.initial_prompt)
            self.selected = param("selected_node", -1, min=-1, max=511)
            self.clamp = param("clamp", 0., min=-4., max=4.)
            self.clamp_status = param("clamp.status", "No feature selected. No slider clamp.")
            self.retrace = param("retrace", False)
            self.baseline_text = param("output.baseline", "")
            self.steered_text = param("output.steered", "")
            self.status = param("output.status", "ready")
            self.actions = [caliper.action("generate", self.generate), caliper.action("reset", self.reset),
                            caliper.action("apply_clamp", self.apply_clamp)]
            if self.restored is not None:
                self.interventions.load_state_dict(self.restored["interventions"])
                self.selection = deepcopy(self.restored.get("selection"))
                self._restore_owned_clamp(self.restored.get("owned_clamp"))
                self._restore_generation(self.restored.get("generation_id", 0))
            # Fork overrides apply before any internal forward, after resource binding.
            if self.managed is not None and self.managed.interventions:
                self.managed._intervene()
            self._clamp_version, self._selected_version = self.clamp._version, self.selected._version
            self.boundary()
            return self
        except BaseException:
            self.close()
            raise

    def _apply(self):
        with self.controls.lock:
            changed = self.controls._apply_interventions()
            self.controls._update_busy()
        if changed:
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
        clamp_changed = self.clamp._version != self._clamp_version
        # Consume the edit even when prompt, selection, or clamp validation fails.
        self._clamp_version = self.clamp._version
        try:
            prompt_inputs(self.model, self.tokenizer, self.prompt.value)
            self._update_selection()
            if clamp_changed:
                self._set_clamp()
        finally:
            self._sync_clamp_control(preserve_edit=clamp_changed)

    def _set_clamp(self):
        self._check_stop()
        if self.selection is None:
            raise ValueError("clamp requires a selected feature")
        layer, feature, position = (self.selection[k] for k in ("layer", "feature", "position"))
        maximum = float(self.index.show(layer, feature)["maximum"])
        if not torch.isfinite(torch.tensor(maximum)) or maximum <= 0:
            raise ValueError("selected feature has no positive indexed maximum")
        point = self.dictionary.metadata["write_points"][layer]
        with self.controls.lock:
            spec = self.interventions.executor.validate({"kind": "clamp", "point": point, "feature": feature,
                "positions": [position], "value": self.clamp.value * maximum,
                "dictionary": self.interventions.executor.dictionary_id}, resources=False)
            owned = self._owned_record()
            if (owned is not None and owned.state == "active" and owned.hook is not None
                    and not owned.replace and owned.spec == owned.applied_spec == spec):
                # An unchanged clamp still verifies the current dictionary content.
                self.interventions.executor.validate(spec)
                return
            # Admission and installation each validate resources. Do not hash the
            # same dictionary a third time merely to construct the request.
            refused = self.controls._intervene({"set": spec})
            if refused is not None:
                raise ValueError(refused[1])
            self._discard_owned_clamp()
            self.controls._update_busy()
            self._owned_clamp = self.controls._ivs[(point, "clamp")].spec
        self._apply()

    def _sync_clamp_control(self, *, preserve_edit=False):
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
            if owned is not None:
                spec = owned.spec
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
                self._set_clamp()
            finally:
                self._sync_clamp_control()
            with self.controls.lock:
                action = self.controls.actions["apply_clamp"]
                refresh = not action.running or self._output_action_id != action.run_id
            self.update_output(force=refresh)
        return "The selected feature clamp is applied."

    def update_output(self, *, force=False):
        key = self._input_state()
        changed = self._output_key != key
        if changed or force:
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
                                        completed_step=self.controls.n, compute_publish_seconds=seconds)
            self._output_key = deepcopy(key)
            self._output_action_id = action_id
            self.status.set("ready")
        if self.retrace.value and (changed or not self._retrace):
            if (self.traced_prompt, self.traced_specifications) != key:
                self.service._execute("trace.run", {"prompt": self.prompt.value}, self.controls.n)
        self._retrace = self.retrace.value
        self._sync_clamp_control()
        if self.guard is not None:
            self.guard.check("inference boundary", force=False)
        if self.fixture_display is not None:
            self.fixture_display.publish()

    def generate(self):
        self._check_stop()
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
        with internal_forward():
            with self.controls.lock:
                self._discard_owned_clamp()
            self._apply()
            self.clamp.set(0.)
            self._clamp_version = self.clamp._version
            self.update_output()
        return "The feature clamp is reset."

    def sweep(self, *, clean_prompt, corrupted_prompt, metric):
        from ._sweep import patch_sweep
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
        with self._stop_boundary():
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
            # Inspect program state outside Controls.lock to preserve lock order.
            publication = _programs.inference_pending()
            with self.controls.cond:
                if not publication:
                    self.controls.cond.wait_for(pending, timeout=1.)
                work = pending()
            try:
                with self._stop_boundary():
                    if work or publication:
                        self.boundary()
                    else:
                        # Reports and intervention expiry need no new prompt step or forward.
                        self.controls.checkpoint()
            except CleanStop:
                raise
            except Exception as exc:
                self.status.set(bounded_text(f"{type(exc).__name__}: {exc}"))

    def close(self):
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
    args = parser.parse_args(argv)
    if args.fixture_dir:
        if any((args.model, args.dictionary, args.prompt)):
            parser.error("fixture creation takes no run arguments")
        fixture_resources(args.fixture_dir)
        return
    if not all((args.model, args.dictionary, args.prompt)):
        parser.error("model, dictionary, and prompt are required")
    run = HostedInference(args.model, args.dictionary, args.prompt)
    try:
        run.start()
        print(f"caliper: inference ready at step {run.controls.n}", flush=True)
        run.run()
    finally:
        run.close()


if __name__ == "__main__":
    main()
