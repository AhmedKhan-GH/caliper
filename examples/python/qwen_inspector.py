#!/usr/bin/env python3
"""Run the local Qwen text model behind Caliper's optional request channel.

Start from the repository with PYTHONPATH=python .venv/bin/python
examples/python/qwen_inspector.py, then attach to this process in Caliper (F4).
The model folder is read-only. All edits and undo data live only in this process.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys

import numpy as np

# The J-space module sits beside this script (also when tests load it by path).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_jspace import DEFAULT_LENS, Intervention  # noqa: E402

DEFAULT_MODEL = Path.home() / "PycharmProjects/weight-space/models/Qwen3.5-0.8B"
CAPABILITIES = ["generate", "list_tensors", "show_tensor", "read_weight",
                "edit_weight", "revert_tensor", "list_edits", "tensor_regions", "tensor_values", "sample_data"]
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 1024
MAX_UNDO_BYTES = 64 * 1024 * 1024
MAX_UNDO_EDITS = 512
MAX_SAMPLE_SIDE = 2048
MAX_SAMPLES = 131072
# Cells read per step while averaging a region (64 MB as float32).
CHUNK_CELLS = 1 << 24
# Raw bytes per tensor_values reply; base64 keeps it under the 4 MB request frame.
MAX_VALUE_BYTES = 2 * 1024 * 1024
# Offered with a Jacobian lens: the lens grid while generating, with J-space interventions.
LENS_CAPABILITIES = ["trace", "tokenize"]
MAX_INTERVENTIONS = 16
INTERVENTION_KINDS = {"ablate": (0, 0), "remove": (1, 64), "swap": (2, 2), "inject": (1, 1)}


class Cancelled(Exception):
    pass


def integer(request, name, default=None, *, low=0, high=None):
    value = request.get(name, default)
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def number(request, name, default=None, *, low=None, high=None):
    value = request.get(name, default)
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (low is not None and value < low) or (high is not None and value > high)):
        raise ValueError(f"{name} must be a finite number between {low} and {high}")
    return float(value)


def matrix_shape(shape):
    """Vectors are N x 1; higher ranks flatten all but the last dimension."""
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return shape[0], 1
    return math.prod(shape[:-1]), shape[-1]


def area_sample(read, region, sample_rows, sample_cols):
    """The mean of the cells under each sample: what a pixel covering several cells should show.

    Taking one cell per pixel aliases a field of weights into moire stripes once a pixel covers many cells;
    a box filter does not. Bin i of n over a length L covers [floor(i L / n), floor((i + 1) L / n)), the
    same bins whose centres sample_plan returns. `read(r0, r1)` returns rows r0..r1 of the region's columns
    as a float32 array; whole bins are read about CHUNK_CELLS cells at a time, so a large tensor is never
    held at once.
    """
    height, width = region["rows"], region["cols"]
    row_starts = [(i * height) // sample_rows for i in range(sample_rows)] + [height]
    col_starts = np.array([(j * width) // sample_cols for j in range(sample_cols)])
    col_counts = np.diff(np.append(col_starts, width))
    out = np.empty((sample_rows, sample_cols), dtype=np.float64)
    per_chunk = max(1, CHUNK_CELLS // max(1, width))
    i = 0
    while i < sample_rows:
        j = i + 1
        while j < sample_rows and row_starts[j + 1] - row_starts[i] <= per_chunk:
            j += 1
        block = read(region["row"] + row_starts[i], region["row"] + row_starts[j])
        sums = np.add.reduceat(block, col_starts, axis=1, dtype=np.float64)
        sums = np.add.reduceat(sums, np.array(row_starts[i:j]) - row_starts[i], axis=0)
        counts = np.diff(np.array(row_starts[i:j + 1]))[:, None] * col_counts[None, :]
        out[i:j] = sums / counts
        i = j
    return out.astype(np.float32)


def sample_plan(rows, cols, request):
    """A bounded, evenly spaced grid in an exact tensor-coordinate region.

    A target measures display pixels in wire orientation: width is columns,
    height is rows. Indices are each bin's centre cell; show_tensor averages each bin (area_sample).
    The returned region remains unchanged when the sample budget is reduced.
    """
    region = request.get("region", dict(row=0, col=0, rows=rows, cols=cols))
    if not isinstance(region, dict):
        raise ValueError("region must contain row, col, rows and cols")
    row = integer(region, "row", high=rows - 1)
    col = integer(region, "col", high=cols - 1)
    height = integer(region, "rows", low=1, high=rows - row)
    width = integer(region, "cols", low=1, high=cols - col)
    target = request.get("target", dict(width=128, height=128))
    if not isinstance(target, dict):
        raise ValueError("target must contain width and height")
    target_width = integer(target, "width", low=1, high=MAX_SAMPLE_SIDE)
    target_height = integer(target, "height", low=1, high=MAX_SAMPLE_SIDE)
    sample_rows, sample_cols = min(height, target_height), min(width, target_width)
    if sample_rows * sample_cols > MAX_SAMPLES:
        scale = math.sqrt(MAX_SAMPLES / (sample_rows * sample_cols))
        sample_rows = max(1, math.floor(sample_rows * scale))
        sample_cols = max(1, math.floor(sample_cols * scale))
    # Integer bin centres avoid rounding drift at the final row/column and
    # return every cell, in order, when the visible region is fully resolved.
    row_indices = [row + ((2 * i + 1) * height) // (2 * sample_rows)
                   for i in range(sample_rows)]
    col_indices = [col + ((2 * i + 1) * width) // (2 * sample_cols)
                   for i in range(sample_cols)]
    return dict(row=row, col=col, rows=height, cols=width), row_indices, col_indices


@dataclass
class Undo:
    name: str
    index: tuple
    before: np.ndarray


class Inspector:
    def __init__(self, path: Path, lens: Path = DEFAULT_LENS):
        # Imports and all MLX evaluation stay on serve()'s thread. Socket I/O
        # never touches the model, including cancellation and disconnect.
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import load

        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"local model folder not found: {path}")
        config = json.loads((path / "config.json").read_text())
        self.layer_types = config.get("text_config", config).get("layer_types", [])
        mx.set_cache_limit(128 * 1024 * 1024)
        mx.set_memory_limit(3 * 1024 * 1024 * 1024)
        self.mx = mx
        self.model, self.tokenizer = load(str(path), tokenizer_config={"local_files_only": True})
        self.tokenizer.add_eos_token("<|im_end|>")
        self.weights = dict(tree_flatten(self.model.parameters()))
        self.revisions = dict.fromkeys(self.weights, 0)
        self.model_name = path.name
        self.undo: list[Undo] = []
        self.undo_bytes = 0
        self.history: list[dict] = []
        self.next_edit = 1
        self.tied = bool(self.model.language_model.args.tie_word_embeddings)
        self.lens_path = lens
        self._jspace = None

    @property
    def jspace(self):
        """The Jacobian lens over this model, loaded on first use (48 MB)."""
        if self._jspace is None:
            from qwen_jspace import JSpace
            if not self.lens_path.is_file():
                raise ValueError(f"no Jacobian lens at {self.lens_path}")
            self._jspace = JSpace(self.model, self.mx, self.lens_path)
        return self._jspace

    def _weight(self, request):
        name = request.get("name")
        if not isinstance(name, str) or name not in self.weights:
            raise ValueError("unknown tensor name")
        return name, self.weights[name]

    def _edits(self, name=None):
        return [edit for edit in self.history if name is None or edit["name"] == name]

    def _status(self, name):
        return dict(edited=any(patch.name == name for patch in self.undo),
                    revision=self.revisions[name])

    def _layer_type(self, index):
        if index < 0:
            return ""
        if index < len(self.layer_types):
            layer_type = self.layer_types[index]
            if layer_type in ("linear_attention", "full_attention"):
                return layer_type
        prefix = f".layers.{index}."
        names = [name for name in self.weights if prefix in name]
        if any(".linear_attn." in name for name in names):
            return "linear_attention"
        if any(".self_attn." in name for name in names):
            return "full_attention"
        return ""

    def _record(self, name, description):
        self.history.append(dict(id=self.next_edit, name=name, description=description))
        self.next_edit += 1
        # Undo patches have their own limits. History is a bounded recent list.
        if len(self.history) > 512:
            del self.history[:-512]

    def _replace(self, name, array):
        self.mx.eval(array)
        self.model.load_weights([(name, array)], strict=False)
        self.weights[name] = array

    def __call__(self, request, emit, cancelled):
        op = request["op"]
        if op == "generate":
            return self.generate(request, emit, cancelled)
        if op == "list_tensors":
            tensors = []
            for name, weight in self.weights.items():
                layer = re.search(r"\.layers\.(\d+)\.", name)
                layer_index = int(layer[1]) if layer else -1
                group = f"Layer {layer_index:02d}" if layer else "Model"
                role = "weight"
                if name.endswith("embed_tokens.weight"):
                    role = "embedding"
                elif name.endswith("lm_head.weight"):
                    role = "output"
                elif not layer and name.endswith("norm.weight"):
                    role = "final_norm"
                rows, cols = matrix_shape(weight.shape)
                tensors.append(dict(name=name, group=group, shape=list(weight.shape),
                                    dtype=str(weight.dtype).removeprefix("mlx.core."),
                                    size=weight.size, bytes=weight.nbytes, rows=rows, cols=cols,
                                    tied=self.tied and role == "embedding", role=role,
                                    layer_index=layer_index, layer_type=self._layer_type(layer_index),
                                    **self._status(name)))
            result = dict(model=self.model_name, tensors=tensors,
                          matrix_layout="Vectors: N x 1. Higher ranks: flatten all but last dimension.")
            embedding = next((tensor for tensor in tensors if tensor["tied"]), None)
            if embedding:
                result["output_alias"] = dict(name="Output (tied)", tensor=embedding["name"])
            return result
        if op == "list_edits":
            return dict(edits=self._edits(), undo_bytes=self.undo_bytes)
        if op == "show_tensor":
            name, weight = self._weight(request)
            rows, cols = matrix_shape(weight.shape)
            region, row_indices, col_indices = sample_plan(rows, cols, request)
            matrix = weight.reshape(rows, cols)
            col0, width = region["col"], region["cols"]

            def read(r0, r1):
                return np.array(matrix[r0:r1, col0:col0 + width].astype(self.mx.float32))

            if len(row_indices) == region["rows"] and len(col_indices) == width:
                sample = read(region["row"], region["row"] + region["rows"])     # every cell, exactly
            else:
                # Each sample is the mean of the cells it covers, read a chunk
                # at a time, so a downsampled view does not alias.
                sample = area_sample(read, region, len(row_indices), len(col_indices))
            if not np.isfinite(sample).all():
                raise ValueError("sample contains non-finite weights")
            # encoding float32 sends the values as little-endian float32 bytes in base64 (data): a JSON
            # list of 100,000 numbers costs most of a request's time to write and read.
            encoding = request.get("encoding", "list")
            if encoding not in ("list", "float32"):
                raise ValueError("encoding must be list or float32")
            values = (dict(data=base64.b64encode(sample.astype("<f4").tobytes()).decode("ascii"))
                      if encoding == "float32" else dict(values=sample.reshape(-1).tolist()))
            return dict(name=name, rows=rows, cols=cols,
                        sample_rows=len(row_indices), sample_cols=len(col_indices),
                        **values, row_indices=row_indices,
                        col_indices=col_indices, min=float(sample.min()), max=float(sample.max()),
                        region=region, exact=(len(row_indices) == region["rows"] and
                                              len(col_indices) == region["cols"]),
                        edits=self._edits(name), **self._status(name))
        if op == "tensor_values":
            return self.values(request)
        if op == "read_weight":
            name, weight = self._weight(request)
            rows, cols = matrix_shape(weight.shape)
            row = integer(request, "row", high=rows - 1)
            col = integer(request, "col", high=cols - 1)
            return dict(name=name, row=row, col=col,
                        value=float(weight.reshape(rows, cols)[row, col].item()),
                        **self._status(name))
        if op == "trace":
            return self.trace(request, emit, cancelled)
        if op == "tokenize":
            text = request.get("text")
            if not isinstance(text, str) or not 0 < len(text) <= 256:
                raise ValueError("text must contain 1 to 256 characters")
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return dict(ids=ids, words=[self.tokenizer.decode([i]) for i in ids])
        if op == "edit_weight":
            return self.edit(request)
        if op == "revert_tensor":
            return self.revert(request)
        raise ValueError(f"unknown operation: {op}")

    def values(self, request):
        """Every value of whole rows from `row` on, as many as fit in one reply.

        The app fetches a tensor this way once and then zooms and pans it on its own, with no request per
        view. bfloat16 goes as its raw bits, so every value arrives exactly; other types go as float32.
        """
        name, weight = self._weight(request)
        rows, cols = matrix_shape(weight.shape)
        row = integer(request, "row", 0, high=rows - 1)
        bfloat16 = weight.dtype == getattr(self.mx, "bfloat16", None)
        itemsize = 2 if bfloat16 else 4
        if cols * itemsize > MAX_VALUE_BYTES:
            raise ValueError("tensor rows are too wide to send whole")
        count = min(rows - row, MAX_VALUE_BYTES // (cols * itemsize))
        block = weight.reshape(rows, cols)[row:row + count]
        if bfloat16:
            data = np.array(block.view(self.mx.uint16)).astype("<u2")
            finite = ((data & 0x7F80) != 0x7F80).all()      # an all-ones exponent is inf or NaN
        else:
            data = np.array(block.astype(self.mx.float32)).astype("<f4")
            finite = np.isfinite(data).all()
        if not finite:
            raise ValueError("tensor contains non-finite weights")
        return dict(name=name, rows=rows, cols=cols, row=row, count=count,
                    encoding="bfloat16" if bfloat16 else "float32",
                    data=base64.b64encode(data.tobytes()).decode("ascii"), **self._status(name))

    def edit(self, request):
        name, weight = self._weight(request)
        rows, cols = matrix_shape(weight.shape)
        action = request.get("action")
        if action not in ("set", "zero_row", "scale_row", "zero_column", "scale_column"):
            raise ValueError("unknown edit action")
        row = integer(request, "row", 0, high=rows - 1)
        col = integer(request, "col", 0, high=cols - 1)
        value = number(request, "value", 0.0)
        matrix = weight.reshape(rows, cols)
        if action == "set":
            index = (row, col)
            replacement = self.mx.array(value, dtype=weight.dtype)
            description = f"Set ({row}, {col}) to {value:g}"
        elif action.endswith("row"):
            index = (row, slice(None))
            replacement = matrix[index] * value if action == "scale_row" else self.mx.zeros_like(matrix[index])
            description = f"Scale row {row} by {value:g}" if action == "scale_row" else f"Zero row {row}"
        else:
            index = (slice(None), col)
            replacement = matrix[index] * value if action == "scale_column" else self.mx.zeros_like(matrix[index])
            description = f"Scale column {col} by {value:g}" if action == "scale_column" else f"Zero column {col}"
        # Check the realized dtype, so a finite input cannot overflow bfloat16
        # and silently poison subsequent generation.
        replacement = replacement.astype(weight.dtype)
        if not bool(self.mx.all(self.mx.isfinite(replacement)).item()):
            raise ValueError("edit would produce non-finite weights")
        before = np.array(matrix[index].astype(self.mx.float32))
        if self.undo_bytes + before.nbytes > MAX_UNDO_BYTES or len(self.undo) >= MAX_UNDO_EDITS:
            raise ValueError("undo limit reached; revert an edited tensor before making more edits")
        # MLX assignment replaces this reshaped array handle; it does not
        # mutate the original weight handle retained by the model.
        matrix[index] = replacement
        updated = matrix.reshape(weight.shape)
        self._replace(name, updated)
        self.undo.append(Undo(name, index, before))
        self.undo_bytes += before.nbytes
        self.revisions[name] += 1
        actual = float(updated.reshape(rows, cols)[row, col].item())
        if action == "set":
            description = f"Set ({row}, {col}) to {actual:g}"
        self._record(name, description)
        return dict(name=name, value=actual, edits=self._edits(name), **self._status(name))

    def revert(self, request):
        name, weight = self._weight(request)
        patches = [patch for patch in self.undo if patch.name == name]
        rows, cols = matrix_shape(weight.shape)
        # Restore in reverse order: intersecting row/column/cell edits recover
        # the originally loaded tensor exactly, including loader sanitization.
        for patch in reversed(patches):
            matrix = self.weights[name].reshape(rows, cols)
            matrix[patch.index] = self.mx.array(patch.before, dtype=weight.dtype)
            self._replace(name, matrix.reshape(weight.shape))
        self.undo = [patch for patch in self.undo if patch.name != name]
        self.undo_bytes -= sum(patch.before.nbytes for patch in patches)
        if patches:
            self.revisions[name] += 1
        self._record(name, "Reverted to original")
        return dict(name=name, edits=self._edits(name), **self._status(name))

    def interventions(self, request):
        """Validated J-space interventions: bands within the lens's layers, token ids within the vocabulary."""
        items = request.get("interventions", [])
        if not isinstance(items, list) or len(items) > MAX_INTERVENTIONS:
            raise ValueError(f"interventions must be a list of at most {MAX_INTERVENTIONS}")
        layers = self.jspace.source_layers
        vocabulary = self.jspace.text.embed_tokens.weight.shape[0]
        result = []
        for item in items:
            if not isinstance(item, dict) or item.get("kind") not in INTERVENTION_KINDS:
                raise ValueError(f"intervention kind must be one of {sorted(INTERVENTION_KINDS)}")
            kind = item["kind"]
            first = integer(item, "first", layers[0], low=layers[0], high=layers[-1])
            last = integer(item, "last", layers[-1], low=first, high=layers[-1])
            tokens = item.get("tokens", [])
            least, most = INTERVENTION_KINDS[kind]
            if (not isinstance(tokens, list) or not least <= len(tokens) <= most
                    or any(type(t) is not int or not 0 <= t < vocabulary for t in tokens)):
                raise ValueError(f"{kind} takes {least} to {most} token ids from the vocabulary")
            if kind == "swap" and tokens[0] == tokens[1]:
                raise ValueError("swap needs two different tokens")
            positions = item.get("positions")
            if positions is not None and (not isinstance(positions, list) or len(positions) > 4096 or
                                          any(type(i) is not int or i < 0 for i in positions)):
                raise ValueError("positions must be a list of token positions")
            result.append(Intervention(kind, first, last, tokens=list(tokens),
                                       k=integer(item, "k", 10, low=1, high=50),
                                       strength=number(item, "strength", 1.0, low=-64, high=64),
                                       positions=positions))
        return result

    def chat_tokens(self, request):
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32768:
            raise ValueError("prompt must contain 1 to 32768 characters")
        thinking = request.get("thinking", False)
        if type(thinking) is not bool:
            raise ValueError("thinking must be true or false")
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking)
        if len(tokens) > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt exceeds {MAX_PROMPT_TOKENS} tokens including chat template")
        return tokens

    def trace(self, request, emit, cancelled):
        """Generate while streaming the lens grid (qwen_jspace.trace), with any interventions applied."""
        from mlx_lm.sample_utils import make_sampler
        from qwen_jspace import trace

        tokens = self.chat_tokens(request)
        max_tokens = integer(request, "max_new_tokens", 64, high=MAX_NEW_TOKENS)
        temperature = number(request, "temperature", 0.0, low=0, high=2)
        top_p = number(request, "top_p", 1.0, low=0, high=1)
        top_k = integer(request, "top_k", 0, high=1000)
        k = integer(request, "k", 5, low=1, high=10)
        seed = integer(request, "seed", 0, high=2**31 - 1)
        attention = request.get("attention", False)
        run_to_max = request.get("run_to_max", False)
        if type(attention) is not bool or type(run_to_max) is not bool:
            raise ValueError("attention and run_to_max must be true or false")
        changes = self.interventions(request)
        # One seed for every run, so an intervened run samples like the intact one until their
        # distributions differ.
        self.mx.random.seed(seed)
        try:
            return trace(self.jspace, self.tokenizer, tokens, max_new_tokens=max_tokens,
                         sampler=make_sampler(temp=temperature, top_p=top_p, top_k=top_k), k=k,
                         interventions=changes, emit=emit, cancelled=cancelled,
                         eos=set(self.tokenizer.eos_token_ids), attention=attention, run_to_max=run_to_max)
        finally:
            self.mx.clear_cache()

    def generate(self, request, emit, cancelled):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        max_tokens = integer(request, "max_new_tokens", 128, low=1, high=MAX_NEW_TOKENS)
        temperature = number(request, "temperature", 0.7, low=0, high=2)
        top_p = number(request, "top_p", 0.9, low=0, high=1)
        top_k = integer(request, "top_k", 20, high=1000)
        tokens = self.chat_tokens(request)
        run_to_max = request.get("run_to_max", False)
        if type(run_to_max) is not bool:
            raise ValueError("run_to_max must be true or false")
        processors = []
        if run_to_max:
            # The end tokens banned, the model writes until max_new_tokens.
            from qwen_jspace import end_ban
            banned = end_ban(self.mx, self.model.language_model.model.embed_tokens.weight.shape[0],
                             set(self.tokenizer.eos_token_ids))
            processors = [lambda _, logits: logits + banned]
        if cancelled.is_set():
            return dict(tokens=0, tokens_per_second=0.0, cancelled=True)

        def progress(done, total):
            if cancelled.is_set():
                raise Cancelled()

        count, tps, finish = 0, 0.0, "cancelled"
        # A new generator owns a fresh cache every request, so changed weights
        # take effect on both prompt processing and decoding immediately.
        generated = stream_generate(
            self.model, self.tokenizer, tokens, max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=top_p, top_k=top_k), logits_processors=processors,
            prefill_step_size=128, prompt_progress_callback=progress)
        try:
            for response in generated:
                if cancelled.is_set():
                    break
                count = response.generation_tokens
                tps = response.generation_tps
                finish = response.finish_reason or "cancelled"
                emit(dict(text=response.text, tokens=count, tokens_per_second=tps))
        except Cancelled:
            pass
        finally:
            # Closing stream_generate synchronizes its MLX stream and drops
            # its private cache before a weight edit can be admitted.
            generated.close()
            self.mx.clear_cache()
        return dict(tokens=count, tokens_per_second=tps, cancelled=cancelled.is_set(),
                    finish_reason=finish, peak_memory_bytes=self.mx.get_peak_memory())



@contextmanager
def model_lease(lock_path=None):
    """Only one inspector model per user. This does not inspect other ML jobs."""
    if lock_path is None:
        from caliper.requests import socket_path
        lock_path = Path(socket_path("qwen-inspector")).with_suffix(".lock")
    else:
        lock_path = Path(lock_path)
    directory = lock_path.parent
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError("inspector lock directory must be owned by this user with mode 0700")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("inspector lock must be a private file owned by this user")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another inspector holds the model; attach to that host or use --connect for checks") from None
        yield
    finally:
        os.close(fd)
        # Keep the empty file: unlinking a lock introduces an inode race.


def run(args):
    from caliper.live import SessionTransport

    print(f"Loading {args.model.expanduser()} with mlx-lm", flush=True)
    inspector = Inspector(args.model.expanduser(), args.lens.expanduser())
    with SessionTransport.open(script_name="qwen_inspector.py", session=args.session,
                               data_dir=args.data_dir) as session:
        lens = LENS_CAPABILITIES if inspector.lens_path.is_file() else []
        with session.request_server(CAPABILITIES + lens) as server:
            def shutdown(signum, frame):
                # Unwind the generator and both contexts on the model thread.
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, shutdown)
            print(f"Ready: pid={os.getpid()} session={session.session}", flush=True)
            print(f"Socket: {server.path}", flush=True)
            print("Caliper: F4, select qwen_inspector.py, Attach", flush=True)
            try:
                server.serve(inspector)
            except KeyboardInterrupt:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS,
                        help="Jacobian lens for the model as lens.npz (neuronpedia/jacobian-lens, converted)")
    parser.add_argument("--session", help="optional live session token (1 to 20 simple characters)")
    parser.add_argument("--data-dir", help="Caliper session discovery directory")
    args = parser.parse_args()

    try:
        with model_lease():
            run(args)
    except Exception as exc:
        print(f"Inspector failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
