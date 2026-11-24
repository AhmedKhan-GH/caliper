"""Safe sparse dictionaries with an independently supplied model identity."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from dataclasses import dataclass
import stat
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from ._memory import CleanStop, MemoryGuard, autorelease_scope
from ._forward_identity import checkpoint_settings, forward_settings, nonpersistent_buffers


_WEIGHTS = ("W_enc", "b_enc", "W_dec", "b_dec")
_GEMMA_IDENTITY = "hf:google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf"
_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_STORED_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16,
                  "F32": torch.float32, "F64": torch.float64}
_FINITE_CHUNK_ELEMENTS = 65_536
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_METADATA_DEPTH = 64


# One digest chunk retains at most one CPU copy and one byte view.
_HASH_CHUNK_BYTES = 1024 * 1024
_UNVERIFIED_CPU_BYTES = 64 * 1024 * 1024
_VERIFIED_MODELS = weakref.WeakKeyDictionary()


def weight_names(layer_or_metadata):
    """Return the tensor names for a layer or its dictionary metadata."""
    activation = (layer_or_metadata.get("activation") if isinstance(layer_or_metadata, dict)
                  else layer_or_metadata.activation)
    if activation == "topk":
        return _WEIGHTS
    if activation == "jumprelu":
        return (*_WEIGHTS, "threshold")
    raise ValueError("unsupported dictionary activation")


def _memory_limit(limit, identity, operation):
    if type(limit) not in (int, float) or not 0 < limit <= 28:
        raise ValueError(f"{operation} memory limit must be positive and at most 28 GB")
    if limit > 12 and identity != _GEMMA_IDENTITY:
        raise ValueError(f"{operation} memory limit above 12 GB requires the pinned Gemma model")


@dataclass(frozen=True)
class _VerifiedWeights:
    identity: str
    digest: str
    settings: dict
    memory: object


def _dense_state(state):
    for tensor in state.values():
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.device.type == "meta" or not tensor.is_contiguous()
                or tensor.is_conj() or tensor.is_neg()):
            raise ValueError("model identity requires materialized contiguous dense tensors")
    return state


def _hash_admission(memory, limit_gb=None, *, force=False):
    if memory is None:
        return
    memory.check("model_identity", force=force)
    limit = memory.limit if limit_gb is None else min(memory.limit, int(limit_gb * 1e9))
    if memory.peak + 4 * _HASH_CHUNK_BYTES > limit:
        raise CleanStop("model identity allocation exceeds the process memory limit")


def _canonical_device(device):
    """Resolve the two spellings of the sole MPS device without device access."""
    device = torch.device(device)
    if device.type == "mps" and device.index is None:
        return torch.device("mps", 0)
    return device


def _state_hash(state, *, memory=None, limit_gb=None):
    _dense_state(state)
    _hash_admission(memory, limit_gb, force=True)
    digest = hashlib.sha256(b"caliper.weights.v1\0")
    for name, tensor in sorted(state.items()):
        if tensor.device.type != "cpu" and memory is None:
            raise ValueError("device weight identity requires a process memory guard")
        if (tensor.device.type != "cpu"
                and _canonical_device(tensor.device) != _canonical_device(memory.device)):
            raise ValueError("verify HF weights again after a model device change")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        # Contiguity was checked before flattening. This view cannot copy weights.
        flat = tensor.detach().view(-1)
        count = max(1, _HASH_CHUNK_BYTES // tensor.element_size())
        for start in range(0, flat.numel(), count):
            _hash_admission(memory, limit_gb)
            with autorelease_scope(tensor.device):
                chunk = flat[start:start + count].to(device="cpu")
                raw = chunk.view(torch.uint8).numpy()
                digest.update(memoryview(raw))
                del raw, chunk
    return "sha256:" + digest.hexdigest()


def weights_hash(model, *, memory=None):
    """Hash every named parameter and persistent buffer, with bounded CPU copies.

    Tensor names sort lexically. The bytes match the original fixture digest.
    Noncontiguous tensors are refused before any flatten or device copy.
    """
    return _state_hash(model.state_dict(), memory=memory)


def dictionary_fingerprint(folder, metadata, *, memory=None):
    """Hash stable metadata and exact layer files, without the quality report."""
    _hash_admission(memory, force=True)
    digest = hashlib.sha256(json.dumps({k: v for k, v in metadata.items() if k != "quality"},
                                     sort_keys=True, allow_nan=False).encode())
    for i in range(metadata["layers"]):
        with (Path(folder) / f"layer_{i:02d}.safetensors").open("rb") as source:
            while True:
                _hash_admission(memory)
                chunk = source.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                del chunk
    return digest.hexdigest()


def _hf_identity(config):
    """Parse checkpoint metadata without a claim about the actual weights."""
    name = str(getattr(config, "_name_or_path", "")).replace("\\", "/")
    revision = getattr(config, "_commit_hash", None)
    snapshot = re.search(r"(?:^|/)models--([^/]+)/snapshots/([0-9a-f]{40})(?:/|$)", name)
    if snapshot:
        repository, pinned = snapshot.groups()
        name = repository.replace("--", "/")
        if revision is not None and revision != pinned:
            raise ValueError("model config revision conflicts with its snapshot")
        revision = pinned
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", name) or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model identity requires an exact HF repository and commit")
    return f"hf:{name}@{revision}"


def _hf_digest(model, *, memory=None, limit_gb=None):
    state = model.state_dict()
    # Nonpersistent buffers can affect the forward but do not occur in state_dict.
    buffers = nonpersistent_buffers(model)
    persistent = _state_hash(state, memory=memory, limit_gb=limit_gb)
    transient = _state_hash(buffers, memory=memory, limit_gb=limit_gb)
    return hashlib.sha256((persistent + transient).encode("ascii")).hexdigest()


def _record_verified_hf_load(model, *, memory):
    """Record a baseline only after the trusted loader verifies complete weights."""
    identity = _hf_identity(model.config)
    digest = _hf_digest(model, memory=memory)
    settings = forward_settings(model, tensor_hash=lambda state: _state_hash(state, memory=memory))
    _VERIFIED_MODELS[model] = _VerifiedWeights(identity, digest, settings, memory)


def model_identity(model, *, memory_limit_gb=None):
    """Return a verified checkpoint identity or a bounded CPU content identity.

    Configuration fields never prove weights. A trusted local load supplies the
    baseline for weights and forward settings. Every later check reads all weights
    in fixed-size chunks, including changes through ``.data`` aliases. A device or
    large external model requires ``verify_hf_model`` before it can use a pinned HF dictionary.
    """
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "caliper_tiny" or config is None:
        return weights_hash(model)
    identity = _hf_identity(config)
    if not isinstance(model, nn.Module):
        raise ValueError("HF configuration does not verify model weights")
    verified = _VERIFIED_MODELS.get(model)
    if verified is not None:
        settings = forward_settings(model, tensor_hash=lambda state: _state_hash(
            state, memory=verified.memory, limit_gb=memory_limit_gb))
        if settings != verified.settings:
            raise ValueError("dictionary model identity mismatch: verified checkpoint forward settings changed")
        if (identity != verified.identity
                or _hf_digest(model, memory=verified.memory, limit_gb=memory_limit_gb) != verified.digest):
            raise ValueError("dictionary model identity mismatch: verified checkpoint weights changed")
        return identity
    state = _dense_state(model.state_dict())
    buffers = _dense_state(dict(model.named_buffers()))
    tensors = [*state.values(), *buffers.values()]
    if (any(value.device.type != "cpu" for value in tensors)
            or sum(value.numel() * value.element_size() for value in tensors) > _UNVERIFIED_CPU_BYTES):
        raise ValueError("unverified HF weights: call verify_hf_model with the local checkpoint first")
    content = _hf_digest(model)
    settings = json.dumps(forward_settings(model, tensor_hash=_state_hash), sort_keys=True)
    return "sha256:" + hashlib.sha256((identity + content + settings).encode("utf-8")).hexdigest()


def _checkpoint_slices(shape, maximum, prefix=()):
    """Yield row-major slices whose materialized size stays within the cap."""
    if not shape:
        yield prefix
    elif math.prod(shape[1:]) <= maximum:
        step = max(1, maximum // max(1, math.prod(shape[1:])))
        for start in range(0, shape[0], step):
            yield prefix + (slice(start, start + step),) + (slice(None),) * (len(shape) - 1)
    else:
        for index in range(shape[0]):
            yield from _checkpoint_slices(shape[1:], maximum, prefix + (index,))


def _checkpoint_subset(model, identity, subset, key_mapping):
    """Declare exact non-text keys from the pinned Qwen3.5 checkpoint header."""
    if subset is None:
        return ({} if key_mapping is None else key_mapping), frozenset()
    pinned = "hf:Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17"
    if (subset != "qwen3.5-0.8b-text" or identity != pinned
            or getattr(model.config, "model_type", None) != "qwen3_5_text"):
        raise ValueError("unsupported checkpoint subset or revision")
    if key_mapping is not None:
        raise ValueError("the declared text subset supplies its own key mapping")
    visual = {f"model.visual.blocks.{i}.{module}.{field}"
              for i in range(12)
              for module in ("attn.proj", "attn.qkv", "mlp.linear_fc1", "mlp.linear_fc2", "norm1", "norm2")
              for field in ("weight", "bias")}
    visual.update(f"model.visual.{module}.{field}"
                  for module in ("merger.linear_fc1", "merger.linear_fc2", "merger.norm", "patch_embed.proj")
                  for field in ("weight", "bias"))
    visual.add("model.visual.pos_embed.weight")
    mtp = {f"mtp.{name}.weight" for name in (
        "fc", "norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden",
        "layers.0.input_layernorm", "layers.0.post_attention_layernorm",
        "layers.0.mlp.down_proj", "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj",
        "layers.0.self_attn.k_norm", "layers.0.self_attn.k_proj", "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_norm", "layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj",
    )}
    return {"model.language_model.": "model."}, frozenset(visual | mtp)


def verify_hf_model(model, snapshot, *, key_mapping=None, checkpoint_subset=None, memory_limit_gb=12):
    """Verify an external model against a local HF safetensors snapshot.

    The snapshot path must identify the same repository and exact commit as the
    model configuration. This function does not download or accept a claimed
    digest. It reads the local checkpoint as trusted source data. Each optional
    ``key_mapping`` entry maps a literal checkpoint prefix to a model prefix.
    Missing or extra tensors, ambiguous mappings, and changed weights are errors.
    ``checkpoint_subset="qwen3.5-0.8b-text"`` declares the text weights of the
    pinned Qwen3.5-0.8B revision. It supplies the prefix mapping and excludes
    exactly 153 vision and 15 MTP keys. All other unexpected keys remain errors.
    Tied parameters can share one checkpoint entry. Nonpersistent buffers must
    match a bounded trusted reconstruction. Qwen3.5 default text RoPE buffers
    are supported. Other nonpersistent buffers are refused.
    Forward settings must match an evaluation-mode reference from the local
    config. Eager and SDPA are supported.

    Converted slices and CPU transfers stay within fixed chunk limits. The
    function returns the verified HF identity and records private provenance for
    this model instance. Every later identity check reads all current weights.
    """
    from types import SimpleNamespace
    from safetensors import safe_open

    identity = _hf_identity(model.config)
    _memory_limit(memory_limit_gb, identity, "verification")
    snapshot = Path(snapshot)
    source_identity = _hf_identity(SimpleNamespace(_name_or_path=str(snapshot), _commit_hash=None))
    if identity != source_identity:
        raise ValueError("checkpoint snapshot identity differs from the model")
    state = _dense_state(model.state_dict(keep_vars=True))
    if not state:
        raise ValueError("checkpoint verification requires model weights")
    devices = {value.device for value in state.values()}
    if len(devices) != 1:
        raise ValueError("checkpoint verification requires one model device")
    memory = MemoryGuard(memory_limit_gb, next(iter(devices)))
    _hash_admission(memory)
    config_data, _ = _read_metadata(snapshot / "config.json", memory)

    def admit_buffers(size):
        memory.check("checkpoint_buffers", force=True)
        if memory.peak + size > memory.limit:
            raise CleanStop("checkpoint buffer reconstruction exceeds the process memory limit")

    settings, buffers = checkpoint_settings(
        model, config_data, tensor_hash=lambda state: _state_hash(state, memory=memory),
        admit_buffers=admit_buffers)

    def current_settings():
        return forward_settings(model, tensor_hash=lambda state: _state_hash(state, memory=memory))

    if current_settings() != settings:
        raise ValueError("model forward settings differ from the local checkpoint")
    if _state_hash(nonpersistent_buffers(model), memory=memory) != buffers:
        raise ValueError("model nonpersistent buffers differ from the local checkpoint")
    mapping, excluded = _checkpoint_subset(model, identity, checkpoint_subset, key_mapping)
    if (not isinstance(mapping, dict) or any(not isinstance(k, str) or not k
            or not isinstance(v, str) for k, v in mapping.items())):
        raise ValueError("checkpoint key mappings require nonempty literal prefixes")
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise ValueError("local checkpoint has no safetensors files")
    entries, source_keys = {}, set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key in source_keys:
                    raise ValueError("checkpoint has duplicate source weights")
                source_keys.add(key)
                if key in excluded:
                    continue
                prefixes = [prefix for prefix in mapping if key.startswith(prefix)]
                if len(prefixes) > 1:
                    raise ValueError("ambiguous checkpoint key mapping")
                name = mapping[prefixes[0]] + key[len(prefixes[0]):] if prefixes else key
                if name not in state or name in entries:
                    raise ValueError("checkpoint has unexpected or duplicate model weights")
                view = source.get_slice(key)
                stored = _STORED_DTYPES.get(view.get_dtype())
                if stored is None or state[name].dtype not in _DTYPES or tuple(view.get_shape()) != tuple(state[name].shape):
                    raise ValueError("checkpoint tensor shape or dtype differs from the model")
                entries[name] = (path, key, stored)
                del view
    if not excluded <= source_keys:
        raise ValueError("checkpoint is missing declared non-text weights")
    # Only an actual shared Parameter object can justify an omitted tied key.
    for name, value in state.items():
        if name not in entries:
            tied = next((other for other in entries if state[other] is value), None)
            if tied is None:
                raise ValueError("checkpoint is missing model weights")
            entries[name] = entries[tied]
    before = _hf_digest(model, memory=memory)
    for name, value in state.items():
        path, key, stored = entries[name]
        maximum = _HASH_CHUNK_BYTES // max(stored.itemsize, value.element_size())
        for selection in _checkpoint_slices(tuple(value.shape), maximum):
            _hash_admission(memory)
            # Close each mapping after its bounded slice to release source pages.
            with autorelease_scope(value.device), safe_open(path, framework="pt", device="cpu") as source:
                expected = source.get_slice(key)[selection].to(dtype=value.dtype).reshape(-1).view(torch.uint8)
                actual = value.detach()[selection].to(device="cpu").reshape(-1).view(torch.uint8)
                if not torch.equal(expected, actual):
                    raise ValueError("model weights differ from the local checkpoint")
                del expected, actual
    if _hf_identity(model.config) != identity or _hf_digest(model, memory=memory) != before:
        raise ValueError("model weights changed during checkpoint verification")
    if current_settings() != settings:
        raise ValueError("model forward settings changed during checkpoint verification")
    if _state_hash(nonpersistent_buffers(model), memory=memory) != buffers:
        raise ValueError("model nonpersistent buffers changed during checkpoint verification")
    _VERIFIED_MODELS[model] = _VerifiedWeights(identity, before, settings, memory)
    return identity


class TopKTranscoder(nn.Module):
    """One MLP replacement with feature rows in both weight matrices."""

    activation = "topk"

    def __init__(self, W_enc, b_enc, W_dec, b_dec, k, *, trainable=False):
        super().__init__()
        self.k = k
        for name, value in zip(_WEIGHTS, (W_enc, b_enc, W_dec, b_dec)):
            self.register_parameter(name, nn.Parameter(value, requires_grad=trainable))

    def encode(self, x):
        return self.activate(F.linear(x, self.W_enc, self.b_enc))

    def activate(self, preactivation):
        z = preactivation.relu()
        values, indices = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter(-1, indices, values)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


class JumpReLUTranscoder(TopKTranscoder):
    """One MLP replacement with a positive threshold for each feature."""

    activation = "jumprelu"

    def __init__(self, W_enc, b_enc, W_dec, b_dec, threshold, *, trainable=False):
        if (threshold.shape != b_enc.shape or not _finite(threshold)
                or not _positive(threshold)):
            raise ValueError("JumpReLU requires positive finite thresholds with one value per feature")
        super().__init__(W_enc, b_enc, W_dec, b_dec, None, trainable=trainable)
        self.register_parameter("threshold", nn.Parameter(threshold, requires_grad=trainable))

    def activate(self, preactivation):
        return preactivation * (preactivation > self.threshold)


class Dictionary(nn.Module):
    """A manifest and zero-based layer modules on the requested device.

    Direct access to ``layers[n].W_enc`` and ``layers[n].W_dec`` never copies data.
    The load freezes parameters but permits gradients with respect to inputs.
    """

    def __init__(self, metadata, layers):
        super().__init__()
        self.metadata = metadata
        self.layers = nn.ModuleList(layers)

    def encode(self, layer, x):
        return self.layers[layer].encode(x)

    def decode(self, layer, a):
        return self.layers[layer].decode(a)

    def error(self, layer, x, y):
        return y - self.decode(layer, self.encode(layer, x))


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"dictionary {name} must be an integer of at least {minimum}")
    return value


def fallback_metadata(data):
    """Validate the optional version-one upper-half contract without tensors."""
    training = data["training"]
    if "fallback" not in training:
        return None
    fallback = training["fallback"]
    fields = {"version", "mode", "active_layers", "zero_layers", "source_dictionary_hash",
              "source_checkpoint_generation", "source_training_tokens", "source_elapsed_seconds",
              "prior_wall_seconds", "aggregate_wall_limit_seconds", "token_budget", "warm_start",
              "source_id", "lower_behavior"}
    if not isinstance(fallback, dict) or set(fallback) != fields:
        raise ValueError("invalid fallback metadata fields")
    if (type(fallback["version"]) is not int or fallback["version"] != 1
            or fallback["mode"] != "upper_half"
            or fallback["warm_start"] != "upper_weights_fresh_adam"
            or fallback["lower_behavior"] != "zero_dictionary_original_mlp"):
        raise ValueError("unsupported fallback contract")
    layers = _integer(data["layers"], "layers", 2)
    for name, expected in (("active_layers", list(range(layers // 2, layers))),
                           ("zero_layers", list(range(layers // 2)))):
        indices = fallback[name]
        if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                or indices != expected):
            raise ValueError("fallback requires complete ordered upper and lower partitions")
    for name, pattern in (("source_dictionary_hash", r"[0-9a-f]{64}"),
                          ("source_checkpoint_generation", r"checkpoint-[0-9a-f]{32}")):
        if not isinstance(fallback[name], str) or not re.fullmatch(pattern, fallback[name]):
            raise ValueError(f"invalid fallback {name}")
    for name in ("source_training_tokens", "token_budget"):
        _integer(fallback[name], f"fallback.{name}")
    for name in ("source_elapsed_seconds", "prior_wall_seconds", "aggregate_wall_limit_seconds"):
        value = fallback[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid fallback {name}")
    if (not 0 < fallback["aggregate_wall_limit_seconds"] <= 36_000
            or fallback["prior_wall_seconds"] < fallback["source_elapsed_seconds"]):
        raise ValueError("fallback aggregate time conflicts with the source or ten-hour cap")
    if (not isinstance(fallback["source_id"], str) or not fallback["source_id"]
            or fallback["source_id"] != training.get("source_id")):
        raise ValueError("fallback text source identity differs")
    return fallback


def _manifest(data):
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("unsupported dictionary format_version")
    for field in ("model_id", "model_hash", "revision"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(f"dictionary requires {field}")
    for field in ("layers", "width", "latents"):
        _integer(data.get(field), field)
    if data.get("activation") == "topk":
        if _integer(data.get("k"), "k") > data["latents"]:
            raise ValueError("dictionary requires TopK activation and 1 <= k <= latents")
    elif data.get("activation") == "jumprelu":
        if "k" not in data or data["k"] is not None:
            raise ValueError("JumpReLU dictionary requires k=null")
    else:
        raise ValueError("dictionary requires TopK or JumpReLU activation")
    if data["model_hash"].startswith("hf:"):
        expected = f"hf:{data['model_id']}@{data['revision']}"
        if data["model_hash"] != expected or not re.fullmatch(r"[0-9a-f]{40}", data["revision"]):
            raise ValueError("dictionary HF identity conflicts with model ID or revision")
    elif not re.fullmatch(r"sha256:[0-9a-f]{64}", data["model_hash"]):
        raise ValueError("unsupported dictionary model hash")
    for field, suffix in (("read_points", "mlp_in"), ("write_points", "mlp_out")):
        rows = data.get(field)
        if not isinstance(rows, list) or len(rows) != data["layers"] or any(
            not isinstance(point, str) or not point.endswith(f".{i}.{suffix}")
            for i, point in enumerate(rows)
        ):
            raise ValueError(f"dictionary {field} must identify every layer in order")
    for read, write in zip(data["read_points"], data["write_points"]):
        if read.rsplit(".", 1)[0] != write.rsplit(".", 1)[0]:
            raise ValueError("dictionary input and output probes refer to different blocks")
    training = data.get("training")
    if not isinstance(training, dict):
        raise ValueError("dictionary requires training metadata")
    for field in ("dataset_id", "dataset_config", "dataset_revision", "split"):
        if not isinstance(training.get(field), str) or not training[field]:
            raise ValueError(f"dictionary training requires {field}")
    if not isinstance(training.get("partitions"), dict):
        raise ValueError("dictionary training requires document partitions")
    _integer(training.get("tokens"), "training.tokens", 0)
    for field in ("elapsed_seconds", "peak_memory_bytes"):
        if (type(training.get(field)) not in (int, float)
                or not math.isfinite(training[field]) or training[field] < 0):
            raise ValueError(f"dictionary training requires nonnegative {field}")
    fallback_metadata(data)
    quality = data.get("quality")
    if not isinstance(quality, dict) or any(field not in quality for field in (
        "fvu", "dead_fraction", "cross_entropy_delta", "evaluated_tokens",
    )):
        raise ValueError("dictionary requires quality metrics, or null metrics before evaluation")
    _integer(quality["evaluated_tokens"], "quality.evaluated_tokens", 0)
    for field in ("fvu", "dead_fraction", "cross_entropy_delta"):
        value = quality[field]
        if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
            raise ValueError(f"invalid quality metric {field}")
        if value is not None and ((field == "fvu" and value < 0)
                                  or (field == "dead_fraction" and not 0 <= value <= 1)):
            raise ValueError(f"invalid quality metric {field}")
    return data


def _reject_constant(value):
    raise ValueError(f"nonfinite dictionary metadata: {value}")


def _metadata_budget(size):
    # Reserve raw bytes, Unicode decode, parser containers, and retained objects.
    # Small JSON values have much more Python overhead than their source bytes.
    return 4096 + 64 * (size + 1)


def _metadata_depth(raw):
    """Bound parser recursion before JSON object allocation, with no token copies."""
    depth, quoted, escaped = 0, False, False
    for value in raw:
        if quoted:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == 34:
                quoted = False
        elif value == 34:
            quoted = True
        elif value in (91, 123):
            depth += 1
            if depth > _MAX_METADATA_DEPTH:
                raise CleanStop("dictionary metadata exceeds the nesting limit")
        elif value in (93, 125):
            depth -= 1


def _read_metadata(path, memory):
    """Admit bounded UTF-8 metadata before the read and JSON parser allocation."""
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("dictionary metadata requires a regular file")
    with path.open("rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("dictionary metadata requires a regular file")
        if info.st_size > _MAX_METADATA_BYTES:
            raise CleanStop("dictionary metadata exceeds the byte limit")
        reserve = _metadata_budget(info.st_size)
        _admit(memory, reserve, "dictionary_metadata_admission")
        # A concurrent append cannot turn this into an unbounded read.
        raw = source.read(info.st_size + 1)
        if len(raw) != info.st_size:
            raise ValueError("dictionary metadata size changed during the read")
    _metadata_depth(raw)
    data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError("dictionary metadata requires an object")
    memory.check("dictionary_metadata_parsed")
    return data, reserve


def _layer_header(source, index, shapes, dtype):
    if set(source.keys()) != set(shapes):
        raise ValueError(f"dictionary layer {index} has unexpected tensor names")
    stored = {}
    for name, shape in shapes.items():
        header = source.get_slice(name)
        if tuple(header.get_shape()) != shape:
            raise ValueError(f"dictionary layer {index} has incorrect {name} shape")
        stored[name] = _STORED_DTYPES.get(header.get_dtype())
        if stored[name] is None:
            raise ValueError(f"dictionary layer {index} has invalid {name} dtype")
    if dtype is None and len(set(stored.values())) != 1:
        raise ValueError("dictionary tensors must share one dtype per layer")
    return stored


def _layer_bytes(stored, shapes, device, dtype):
    counts = {name: math.prod(shape) for name, shape in shapes.items()}
    targets = {name: dtype or value for name, value in stored.items()}
    converted = device.type != "cpu" or targets != stored
    destination = sum(counts[name] * value.itemsize for name, value in targets.items())
    # A mapped source layer can retain all its pages until the handle closes.
    source = sum(counts[name] * value.itemsize for name, value in stored.items()) if converted else 0
    # Reserve one extra target tensor for device/dtype conversion staging.
    staging = max(counts[name] * value.itemsize for name, value in targets.items()) if converted else 0
    itemsize = max(value.itemsize for value in (*stored.values(), *targets.values()))
    finite = min(max(counts.values()), _FINITE_CHUNK_ELEMENTS) * (2 * itemsize + 4)
    return destination, source + staging + finite, converted


def _finite(value):
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), _FINITE_CHUNK_ELEMENTS):
        with autorelease_scope(value.device):
            if not torch.isfinite(flat[start:start + _FINITE_CHUNK_ELEMENTS]).all().item():
                return False
    return True


def _zero(value):
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), _FINITE_CHUNK_ELEMENTS):
        with autorelease_scope(value.device):
            if torch.count_nonzero(flat[start:start + _FINITE_CHUNK_ELEMENTS]).item():
                return False
    return True


def _positive(value):
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), _FINITE_CHUNK_ELEMENTS):
        with autorelease_scope(value.device):
            if not (flat[start:start + _FINITE_CHUNK_ELEMENTS] > 0).all().item():
                return False
    return True


def _admit(memory, required, stage):
    memory.check(stage)
    if memory.peak + required > memory.limit:
        raise CleanStop("dictionary allocation exceeds the process memory limit")


def load_dictionary(path, *, model=None, model_hash=None, device=None, dtype=None,
                    memory_limit_gb=12):
    """Load a matching dictionary within a default process limit of 12 GB.

    The pinned Gemma model permits an explicit limit of at most 28 GB.

    ``memory_limit_gb`` uses decimal GB and includes current process/model memory,
    destination weights, metadata, and temporary conversion storage. Metadata has
    a 1 MiB byte cap and 64 nesting levels. Admission precedes its read and parse.
    All headers pass byte admission before tensor materialization.
    Resource refusal raises ``CleanStop``.
    An external HF model requires ``verify_hf_model(model, snapshot)`` to match a
    pinned HF manifest. The snapshot is a trusted local safetensors checkpoint.
    """
    from safetensors import safe_open

    if type(memory_limit_gb) not in (int, float) or not 0 < memory_limit_gb <= 28:
        raise ValueError("dictionary memory limit must be positive and at most 28 GB")
    if model is None and model_hash is None:
        from .. import _programs
        model = _programs._STATE.get("model")
    if model is None and model_hash is None:
        raise ValueError("supply a model or expected model_hash, or watch a model first")
    actual = model_identity(model, memory_limit_gb=memory_limit_gb) if model is not None else model_hash
    _memory_limit(memory_limit_gb, actual, "dictionary")
    if model_hash is not None and actual != model_hash:
        raise ValueError("supplied model and expected model_hash differ")
    folder = Path(path)
    # Metadata admission needs no device initialization or tensor header access.
    metadata_memory = MemoryGuard(memory_limit_gb, "cpu")
    data, metadata_bytes = _read_metadata(folder / "dictionary.json", metadata_memory)
    data = _manifest(data)
    fallback = fallback_metadata(data)
    zero_layers = set(fallback["zero_layers"]) if fallback else set()
    if actual != data["model_hash"]:
        if model is not None and data["model_hash"].startswith("hf:") and model not in _VERIFIED_MODELS:
            raise ValueError("dictionary model identity mismatch: verify_hf_model(model, snapshot) "
                             "must verify external HF weights against the local checkpoint first")
        raise ValueError("dictionary model identity mismatch")
    if model is not None:
        config = getattr(model, "config", None)
        if config is None or (config.num_hidden_layers, config.hidden_size) != (data["layers"], data["width"]):
            raise ValueError("dictionary dimensions differ from the model")
        from ._points import semantic_points
        rows = semantic_points(model)
        for field, suffix in (("read_points", "mlp_in"), ("write_points", "mlp_out")):
            expected = [p["point"] for p in rows if p["probe"] == suffix]
            if data[field] != expected:
                raise ValueError(f"dictionary {field} differ from model probes")
        parameter = next(model.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
    if dtype is not None and dtype not in _DTYPES:
        raise ValueError("dictionary dtype must be floating point")
    device = torch.device(device or "cpu")
    shapes = {"W_enc": (data["latents"], data["width"]), "b_enc": (data["latents"],),
              "W_dec": (data["latents"], data["width"]), "b_dec": (data["width"],)}
    if data["activation"] == "jumprelu":
        shapes["threshold"] = (data["latents"],)
    headers, budgets = [], []
    for index in range(data["layers"]):
        # Header inspection uses CPU mappings and never requests tensor data.
        with safe_open(folder / f"layer_{index:02d}.safetensors", framework="pt", device="cpu") as source:
            stored = _layer_header(source, index, shapes, dtype)
        headers.append(stored)
        budgets.append(_layer_bytes(stored, shapes, device, dtype))
    remaining = sum(budget[0] for budget in budgets)
    temporary = max(budget[1] for budget in budgets)
    if metadata_memory.peak + metadata_bytes + remaining + temporary > int(memory_limit_gb * 1_000_000_000):
        raise CleanStop("dictionary allocation exceeds the process memory limit")
    memory = MemoryGuard(memory_limit_gb, device)
    _admit(memory, metadata_bytes + remaining + temporary, "dictionary_admission")
    layers = []
    for index in range(data["layers"]):
        _admit(memory, metadata_bytes + remaining + temporary, f"dictionary_layer_{index}")
        tensors = {}
        with autorelease_scope(device), safe_open(
            folder / f"layer_{index:02d}.safetensors", framework="pt", device="cpu",
        ) as source:
            if _layer_header(source, index, shapes, dtype) != headers[index]:
                raise ValueError(f"dictionary layer {index} header changed after memory admission")
            for name in weight_names(data):
                value = source.get_tensor(name)
                if not _finite(value):
                    raise ValueError(f"dictionary layer {index} has invalid or nonfinite {name}")
                if index in zero_layers and not _zero(value):
                    raise ValueError(f"fallback lower layer {index} has nonzero {name}")
                # Converted CPU layers must release the entire source mapping.
                value = value.to(device=device, dtype=dtype, copy=budgets[index][2])
                if not _finite(value):
                    raise ValueError(f"dictionary layer {index} overflows the requested dtype")
                if name == "threshold" and not _positive(value):
                    raise ValueError(f"dictionary layer {index} requires positive thresholds")
                tensors[name] = value
                del value
        layers.append(JumpReLUTranscoder(**tensors) if data["activation"] == "jumprelu"
                      else TopKTranscoder(**tensors, k=data["k"]))
        remaining -= budgets[index][0]
    memory.check("dictionary_loaded")
    return Dictionary(data, layers).eval()
