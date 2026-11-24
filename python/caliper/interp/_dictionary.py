"""Safe TopK dictionaries with an independently supplied model identity."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from ._memory import CleanStop, MemoryGuard, autorelease_scope


_WEIGHTS = ("W_enc", "b_enc", "W_dec", "b_dec")
_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_STORED_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16,
                  "F32": torch.float32, "F64": torch.float64}
_FINITE_CHUNK_ELEMENTS = 65_536


def weights_hash(model):
    """Hash every named parameter and persistent buffer, including shape and dtype.

    The byte order is little endian. Tensor names sort lexically. No forward sample
    or subset of weights establishes the fixture identity.
    """
    digest = hashlib.sha256(b"caliper.weights.v1\0")
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise ValueError("model identity requires materialized dense tensors")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return "sha256:" + digest.hexdigest()


def model_identity(model):
    """Return the fixture weights hash or an exact HF checkpoint identity.

    HF models must retain their resolved commit and repository name. A local HF
    snapshot path also supplies both fields. A branch name is not a revision.
    """
    config = getattr(model, "config", None)
    family = getattr(config, "model_type", None)
    if family == "caliper_tiny" or config is None:
        return weights_hash(model)
    name = str(getattr(config, "_name_or_path", ""))
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


class TopKTranscoder(nn.Module):
    """One MLP replacement with feature rows in both weight matrices."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec, k, *, trainable=False):
        super().__init__()
        self.k = k
        for name, value in zip(_WEIGHTS, (W_enc, b_enc, W_dec, b_dec)):
            self.register_parameter(name, nn.Parameter(value, requires_grad=trainable))

    def encode(self, x):
        z = F.linear(x, self.W_enc, self.b_enc).relu()
        values, indices = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter(-1, indices, values)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


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


def _manifest(data):
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("unsupported dictionary format_version")
    for field in ("model_id", "model_hash", "revision"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(f"dictionary requires {field}")
    for field in ("layers", "width", "latents", "k"):
        _integer(data.get(field), field)
    if data.get("activation") != "topk" or data["k"] > data["latents"]:
        raise ValueError("dictionary requires TopK activation and 1 <= k <= latents")
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


def _layer_header(source, index, shapes, dtype):
    if set(source.keys()) != set(_WEIGHTS):
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


def _admit(memory, required, stage):
    memory.check(stage)
    if memory.peak + required > memory.limit:
        raise CleanStop("dictionary allocation exceeds the process memory limit")


def load_dictionary(path, *, model=None, model_hash=None, device=None, dtype=None,
                    memory_limit_gb=12):
    """Load a matching dictionary within a total process limit of at most 12 GB.

    ``memory_limit_gb`` uses decimal GB and includes current process/model memory,
    destination weights, and temporary conversion storage. All headers pass byte
    admission before tensor materialization. Resource refusal raises ``CleanStop``.
    """
    from safetensors import safe_open

    if type(memory_limit_gb) not in (int, float) or not 0 < memory_limit_gb <= 12:
        raise ValueError("dictionary memory limit must be positive and at most 12 GB")
    if model is None and model_hash is None:
        from .. import _programs
        model = _programs._STATE.get("model")
    if model is None and model_hash is None:
        raise ValueError("supply a model or expected model_hash, or watch a model first")
    actual = model_identity(model) if model is not None else model_hash
    if model_hash is not None and actual != model_hash:
        raise ValueError("supplied model and expected model_hash differ")
    folder = Path(path)
    data = _manifest(json.loads((folder / "dictionary.json").read_text(encoding="utf-8"),
                                parse_constant=_reject_constant))
    if actual != data["model_hash"]:
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
    headers, budgets = [], []
    for index in range(data["layers"]):
        # Header inspection uses CPU mappings and never requests tensor data.
        with safe_open(folder / f"layer_{index:02d}.safetensors", framework="pt", device="cpu") as source:
            stored = _layer_header(source, index, shapes, dtype)
        headers.append(stored)
        budgets.append(_layer_bytes(stored, shapes, device, dtype))
    remaining = sum(budget[0] for budget in budgets)
    temporary = max(budget[1] for budget in budgets)
    if remaining + temporary > int(memory_limit_gb * 1_000_000_000):
        raise CleanStop("dictionary allocation exceeds the process memory limit")
    memory = MemoryGuard(memory_limit_gb, device)
    _admit(memory, remaining + temporary, "dictionary_admission")
    layers = []
    for index in range(data["layers"]):
        _admit(memory, remaining + temporary, f"dictionary_layer_{index}")
        tensors = {}
        with autorelease_scope(device), safe_open(
            folder / f"layer_{index:02d}.safetensors", framework="pt", device="cpu",
        ) as source:
            if _layer_header(source, index, shapes, dtype) != headers[index]:
                raise ValueError(f"dictionary layer {index} header changed after memory admission")
            for name in _WEIGHTS:
                value = source.get_tensor(name)
                if not _finite(value):
                    raise ValueError(f"dictionary layer {index} has invalid or nonfinite {name}")
                # Converted CPU layers must release the entire source mapping.
                value = value.to(device=device, dtype=dtype, copy=budgets[index][2])
                if not _finite(value):
                    raise ValueError(f"dictionary layer {index} overflows the requested dtype")
                tensors[name] = value
                del value
        layers.append(TopKTranscoder(**tensors, k=data["k"]))
        remaining -= budgets[index][0]
    memory.check("dictionary_loaded")
    return Dictionary(data, layers).eval()
