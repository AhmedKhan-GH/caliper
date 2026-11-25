"""Bounded local checkpoint admission without hosted runtime dependencies."""
from itertools import islice
import json
import math
from pathlib import Path

import torch

GEMMA_ID = "google/gemma-2-2b"
GEMMA_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"


def _admit_load(memory, required, stage, *, staged=False):
    from ._memory import CleanStop
    memory.check(stage)
    used = getattr(memory, "current", memory.peak) if staged else memory.peak
    if used + required > memory.limit:
        raise CleanStop("model allocation exceeds the process memory limit")
    _host_reserve(required, stage)


def _host_reserve(required, stage):
    """Leave eight decimal GB available after a prospective allocation."""
    import psutil
    from ._memory import CleanStop
    if psutil.virtual_memory().available - required < 8_000_000_000:
        raise CleanStop(f"{stage} allocation would consume the 8 GB host reserve")


def _admit_model_load(snapshot, memory, device, *, staged=False):
    """Reserve float32 weights, source pages, conversion work, and a device copy."""
    header_limit = 4 * 1024 * 1024
    _admit_load(memory, 128 * 1024 * 1024, "model header admission", staged=staged)
    files = list(islice(Path(snapshot).glob("*.safetensors"), 65))
    if not files or len(files) > 64:
        raise ValueError("model admission requires one to 64 local safetensors shards")
    sizes = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
             "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
    loaded = stored = largest = 0
    for path in files:
        size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            length = int.from_bytes(prefix, "little")
            if len(prefix) != 8 or not 2 <= length <= min(header_limit, size - 8):
                raise ValueError("model safetensors header exceeds the admission bounds")
            entries = json.loads(source.read(length))
        if type(entries) is not dict or len(entries) > 4096:
            raise ValueError("model safetensors header has too many entries")
        for name, entry in entries.items():
            if name == "__metadata__":
                continue
            if type(entry) is not dict:
                raise ValueError("model tensor header is invalid")
            shape, offsets = entry.get("shape"), entry.get("data_offsets")
            width = sizes.get(entry.get("dtype"))
            if (width is None or type(shape) is not list or len(shape) > 8
                    or any(type(dim) is not int or not 0 <= dim < 2**31 for dim in shape)
                    or type(offsets) is not list or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)):
                raise ValueError("model tensor header is invalid")
            count = math.prod(shape)
            if not 0 <= offsets[0] <= offsets[1] <= size - length - 8 or offsets[1] - offsets[0] != count * width:
                raise ValueError("model tensor storage differs from its header")
            allocation = count * max(4, width)
            loaded += allocation
            largest = max(largest, allocation)
        stored += size
        del entries
    if loaded == 0:
        raise ValueError("model admission requires nonempty weights")
    copies = 1 if torch.device(device).type == "cpu" else 2
    # The reserve covers loader metadata, tokenizer data, and allocator overhead.
    # Gemma releases checkpoint mappings before transfer. Admit both phases,
    # then measure current memory again before the separate device allocation.
    required = (max(loaded + stored, copies * loaded) if staged else copies * loaded + stored)
    _admit_load(memory, required + largest + 512_000_000, "model load admission", staged=staged)
    return loaded, largest


def _admit_model_transfer(memory, estimate, device, *, staged=False):
    loaded, largest = estimate
    required = 0 if torch.device(device).type == "cpu" else loaded + largest
    _admit_load(memory, required + 512_000_000, "model device admission", staged=staged)


def _load_gemma_model(snapshot, *, device, memory, dtype=torch.float32):
    """Load and verify the exact local Gemma snapshot before device transfer."""
    from transformers import Gemma2Config, Gemma2ForCausalLM
    from dataclasses import replace
    from ._dictionary import _VERIFIED_MODELS, _verification_memory, model_identity, verify_hf_model

    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Gemma load dtype must be float32 or bfloat16")
    snapshot = Path(snapshot).resolve(strict=True)
    if (snapshot.name != GEMMA_REVISION or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != "models--google--gemma-2-2b"):
        raise ValueError("model must name the pinned local Gemma snapshot")
    if not 0 < memory.limit <= 40_000_000_000:
        raise ValueError("Gemma memory limit is at most 40 GB")
    config = Gemma2Config(**json.loads((snapshot / "config.json").read_text()))
    if (config.model_type, config.num_hidden_layers, config.hidden_size, config.vocab_size) != (
            "gemma2", 26, 2304, 256000):
        raise ValueError("the pinned Gemma model has unexpected dimensions")
    estimate = _admit_model_load(snapshot, memory, device, staged=True)
    model, info = Gemma2ForCausalLM.from_pretrained(
        snapshot, config=config, dtype=dtype, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("local model weights do not match the pinned Gemma architecture")
    model.config._name_or_path, model.config._commit_hash = GEMMA_ID, GEMMA_REVISION
    model.requires_grad_(False).eval()
    memory.check("inference model CPU load")
    # Verify bytes, tied parameters, settings, and reconstructed buffers before
    # the trusted identity enters the cache. Loading information alone is insufficient.
    with _verification_memory(memory):
        verify_hf_model(model, snapshot, memory_limit_gb=memory.limit / 1e9)
    verified = _VERIFIED_MODELS[model]
    _admit_model_transfer(memory, estimate, device, staged=True)
    model.to(device=device, dtype=dtype)
    memory.check("inference model load")
    # Preserve the proven bytes and settings across transfer. Only the guard
    # changes, because future readbacks must use the destination device guard.
    _VERIFIED_MODELS[model] = replace(verified, memory=memory)
    model_identity(model, memory_limit_gb=memory.limit / 1e9)
    return model
