"""Bounded local checkpoint admission without hosted runtime dependencies."""
from itertools import islice
import json
import math
from pathlib import Path

import torch


def _admit_load(memory, required, stage):
    from ._memory import CleanStop
    memory.check(stage)
    if memory.peak + required > memory.limit:
        raise CleanStop("model allocation exceeds the process memory limit")


def _admit_model_load(snapshot, memory, device):
    """Reserve float32 weights, source pages, conversion work, and a device copy."""
    header_limit = 4 * 1024 * 1024
    _admit_load(memory, 128 * 1024 * 1024, "model header admission")
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
    _admit_load(memory, copies * loaded + stored + largest + 512_000_000, "model load admission")
    return loaded, largest


def _admit_model_transfer(memory, estimate, device):
    loaded, largest = estimate
    required = 0 if torch.device(device).type == "cpu" else loaded + largest
    _admit_load(memory, required + 512_000_000, "model device admission")
