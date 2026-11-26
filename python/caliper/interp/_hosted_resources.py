"""Resource loading, fixture setup, and checkpoint validation for hosted inference."""
from __future__ import annotations

from inspect import Parameter, signature
import json
import math
import os
from pathlib import Path

import torch

from ._backend import load_backend
from ._index import FeatureIndex, LiveFeatureIndex, build_index
from ._memory import MAX_PROCESS_GB, MemoryGuard, autorelease_scope

CHECKPOINT_FORMAT = "caliper.inference"
CHECKPOINT_BYTES = 4 * 1024 * 1024
CHECKPOINT_VALUES = 262144
DEFAULT_PROMPT = "He saw a carrot and had to grab it, his hunger was like a starving"
READ_ONLY_PARAMS = frozenset({
    "output.baseline",
    "output.steered",
    "output.status",
    "output.progress",
    "clamp.status",
})
ALL_PANELS = ("graph", "output", "examples")


def _accepts_kwarg(fn, name: str) -> bool:
    parameters = signature(fn).parameters
    return any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()) or (
        name in parameters
        and parameters[name].kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    )


def checkpoint_metadata(value) -> bool:
    """Validate bounded JSON checkpoint metadata."""
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
            return len(item) <= 64 and all(
                type(k) is str and len(k.encode("utf-8")) <= 63 and valid(v, depth + 1)
                for k, v in item.items()
            )
        return False

    try:
        if type(value) is dict and valid(value):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            return len(raw) <= CHECKPOINT_BYTES
    except (ValueError, UnicodeError):
        pass
    return False


class _FixtureDisplay:
    """Copy bounded CPU fixture display tensors onto MPS when ``CALIPER_FIXTURE_DISPLAY_MPS=1``."""

    TENSORS = (
        "trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f",
        "trace.label_text", "trace.probs", "trace.logit_tokens", "trace.logit_label_text",
        "examples.tokens", "examples.activations", "examples.example_ids",
        "examples.valid_tokens", "examples.label_text",
        "output.probs", "output.tokens", "output.label_text",
        "patch.grid",
    )

    def __init__(self, guard: MemoryGuard, *, device="mps"):
        self.guard = guard
        self.device = torch.device(device)
        if self.device.type == "mps" and self.device.index is None:
            self.device = torch.device("mps:0")

    def publish(self) -> int:
        from .. import _programs
        from . import _publish

        with _publish._publication_scope():
            with _programs._LOCK:
                values = {n: _publish._TENSORS[n] for n in self.TENSORS if n in _publish._TENSORS}
            pending = [(n, t) for n, t in values.items() if t.device.type == "cpu"]
            if not pending:
                return 0
            with autorelease_scope(self.device):
                self.guard.check("fixture display copy start")
                converted = {n: t.detach().contiguous().to(device=self.device, copy=True) for n, t in pending}
                self.guard.check("fixture display copy complete")
            _publish._publish_many(converted)
            return sum(t.numel() * t.element_size() for _, t in pending)


def fixture_resources(directory: str | Path) -> Path:
    """Materialize deterministic ``tiny-lm`` transcoders and an example index on disk."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    backend = load_backend("tiny-lm")
    backend.dictionary.raw.to_safetensors(str(root))
    if not (root / "index").exists():
        docs = [
            ("doc-0", backend.tokenize("ABCD")[0].tolist()),
            ("doc-1", backend.tokenize("EFGH")[0].tolist()),
            ("doc-2", backend.tokenize("ABEF")[0].tolist()),
        ]
        build_index(
            root,
            docs,
            model=backend.model,
            tokenizer=backend.tokenizer,
            dictionary=backend.dictionary,
            provenance={"partition": "tiny-lm-fixture"},
            context_length=16,
        )
    return root


def load_resources(model_name: str, dictionary_path: str | Path, *, device=None, memory_limit_gb: float = MAX_PROCESS_GB, progress=None):
    """Load ``(backend, index, guard)`` for hosted inference."""
    is_tiny = str(model_name) == "tiny-lm"
    if device is None:
        dev = torch.device("cpu") if is_tiny else None
    else:
        dev = torch.device(device)
    guard = None
    if not is_tiny:
        from ._backend import default_device
        actual_dev = dev if dev is not None else default_device()
        guard = MemoryGuard(memory_limit_gb, actual_dev)
        guard.check("hosted load start")

    backend = load_backend(model_name, dictionary_path, device=dev, guard=guard, progress=progress)
    if callable(progress):
        progress(stage="index", label="Loading example index (stage 3/3)", done=None, total=None)
    env_index = os.environ.get("CALIPER_FEATURE_INDEX", "").strip()
    resolved_index = Path(env_index).expanduser() if env_index else FeatureIndex.resolve(dictionary_path)
    if resolved_index is not None:
        index = FeatureIndex(
            resolved_index,
            model=backend.model,
            tokenizer=backend.tokenizer,
            dictionary=backend.dictionary,
            memory=guard,
        )
    else:
        index = LiveFeatureIndex(backend, memory=guard, allow_online=os.environ.get("CALIPER_AGENT_OFFLINE") != "1")
    if callable(progress):
        progress(stage="index", label="Model resources ready (3/3)", done=3, total=3)
    return backend, index, guard
