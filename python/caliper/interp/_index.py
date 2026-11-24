"""Bounded feature examples with exact top-k order and document boundaries."""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ._dictionary import _canonical_device, _memory_limit, load_dictionary, model_identity, weights_hash
from ._memory import (BudgetMemory, CleanStop, MemoryGuard, allocation_bytes, family_memory_limit,
                      autorelease_scope as _device_scope)
from ._points import capture
from ._frozen import _trace_features

INDEX_VERSION = 2
ENCODER_POLICY = "live_input_v1"
TOP_EXAMPLES = 16
WINDOW_TOKENS = 33
CACHE_FEATURES = 256
LABEL_FEATURES = 512
LABEL_VOCAB_CHUNK = 4096
MAX_TOKENS = 2_000_000
_TABLE = np.dtype([("activation", "<f8"), ("offset", "<i8")])
_DOCUMENT = np.dtype([("id", "S128"), ("start", "<i8"), ("length", "<i8"),
                      ("truncated", "?")])
_FILES = ("table.npy", "counts.npy", "tokens.bin", "documents.bin")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _model_encoding(model, memory):
    from ._evaluation_identity import _evaluation_settings, _settings_digest
    device = next(model.parameters()).device.type
    autocast = torch.is_autocast_enabled(device)
    tensors = [("parameter:" + name, value) for name, value in model.named_parameters()]
    tensors.extend(("buffer:" + name, value) for name, value in model.named_buffers())
    return {"version": 1, "autocast_dtype": str(torch.get_autocast_dtype(device)) if autocast else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tensors": [[name, str(value.dtype), str(_canonical_device(value.device))] for name, value in tensors],
            "forward_settings_hash": _settings_digest(model, _evaluation_settings(model, memory)),
            "attention_backends": [[name, module.config._attn_implementation]
                                   for name, module in model.named_modules()
                                   if hasattr(getattr(module, "config", None), "_attn_implementation")]}


def _encoding_policy(manifest, model, dictionary, memory):
    version = manifest["format_version"]
    if type(version) is not int or version not in (1, INDEX_VERSION):
        raise ValueError("incomplete or unsupported index")
    if version == INDEX_VERSION:
        if manifest.get("encoder_policy") != ENCODER_POLICY:
            raise ValueError("unsupported index encoder policy; rebuild the index")
        if manifest.get("model_encoding") != _model_encoding(model, memory):
            raise ValueError("index model encoding differs; rebuild the index")
        return
    # Version 1 cast captured inputs to encoder storage precision. The two
    # policies agree only when the admitted model and encoder use one dtype.
    dtypes = {value.dtype for value in model.state_dict().values() if value.is_floating_point()}
    if ("encoder_policy" in manifest or "model_encoding" in manifest or len(dtypes) != 1 or
            any(value.dtype not in dtypes for layer in dictionary.layers
                for value in (layer.W_enc, layer.b_enc))):
        raise ValueError("legacy index encoder precision differs; rebuild the index")


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dictionary_hash(dictionary, *, memory=None, memory_limit_gb=None):
    device = next(dictionary.parameters()).device
    limit = (12 if memory is None else memory.limit / 1e9) if memory_limit_gb is None else memory_limit_gb
    if memory is None or memory_limit_gb is not None:
        _memory_limit(limit, dictionary.metadata.get("model_hash"), "index")
    memory = MemoryGuard(limit, device) if memory is None else BudgetMemory(memory, limit)
    digest = hashlib.sha256(_json(dictionary.metadata).encode())
    with _device_scope(device):
        digest.update(weights_hash(dictionary, memory=memory).encode())
    return digest.hexdigest()


def _source_hash(folder, layers):
    names = ["dictionary.json"] + [f"layer_{i:02d}.safetensors" for i in range(layers)]
    return hashlib.sha256(_json({name: _hash_file(folder / name) for name in names}).encode()).hexdigest()


def _tokenizer_hash(tokenizer, vocab):
    """Include token text, special IDs, and the tokenizer pipeline when available."""
    digest = hashlib.sha256()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        digest.update(backend.to_str().encode())
    digest.update(_json([getattr(tokenizer, name, None) for name in
                        ("bos_token_id", "eos_token_id", "pad_token_id")]).encode())
    for token_id in range(vocab):
        digest.update(_json([token_id, tokenizer.convert_ids_to_tokens(token_id)]).encode())
    return digest.hexdigest()


def _positive_integer(value, name, maximum=None, minimum=1):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def _compatible(model, dictionary, *, memory_limit_gb=None):
    data = dictionary.metadata
    if model_identity(model, memory_limit_gb=memory_limit_gb) != data["model_hash"]:
        raise ValueError("index model identity mismatch")
    if model.training:
        raise ValueError("index requires a model in eval mode")
    if len(dictionary.layers) != data["layers"]:
        raise ValueError("index dictionary layer count mismatch")
    from ._points import semantic_points
    expected = [p["point"] for p in semantic_points(model) if p["probe"] == "mlp_in"]
    if data["read_points"] != expected:
        raise ValueError("index dictionary probe mismatch")
    for layer in dictionary.layers:
        shape = (data["latents"], data["width"])
        if (tuple(layer.W_enc.shape) != shape or tuple(layer.W_dec.shape) != shape or
                tuple(layer.b_enc.shape) != (shape[0],) or tuple(layer.b_dec.shape) != (shape[1],)):
            raise ValueError("index dictionary dimensions mismatch")
        activation = getattr(layer, "activation", "topk")
        if activation != data["activation"]:
            raise ValueError("index dictionary activation mismatch")
        if activation == "jumprelu":
            if layer.k is not None or data["k"] is not None or tuple(layer.threshold.shape) != (shape[0],):
                raise ValueError("index dictionary JumpReLU mismatch")
        elif activation != "topk" or type(layer.k) is not int or layer.k != data["k"]:
            raise ValueError("index dictionary TopK mismatch")
    return data


def _stable_topk(values, offsets, k=TOP_EXAMPLES):
    """Use topk, repair boundary ties, then order only the bounded candidates.

    Rows represent features. Ties use the smallest global token offset.
    No epsilon changes an activation. Float64 fixture values stay exact.
    """
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("topk requires nonempty feature rows")
    if not torch.isfinite(values).all().item():
        raise ValueError("index activations must be finite")
    offsets = torch.broadcast_to(offsets, values.shape)
    take = min(k, values.shape[1])
    best, indices = values.topk(take, dim=1)
    threshold = best.min(dim=1, keepdim=True).values
    best_offsets = offsets.gather(1, indices)
    # topk may choose arbitrary boundary ties. Replace all of them explicitly.
    boundary = values == threshold
    sentinel = torch.iinfo(torch.int64).max
    tied_offsets = torch.where(boundary, offsets, sentinel)
    tied, tied_indices = tied_offsets.topk(take, dim=1, largest=False)
    tied_values = values.gather(1, tied_indices).masked_fill(tied == sentinel, -torch.inf)
    best = best.masked_fill(best == threshold, -torch.inf)
    candidates = torch.cat((best, tied_values), dim=1).cpu().double().numpy()
    locations = torch.cat((best_offsets, tied), dim=1).cpu().numpy()
    order = np.lexsort((locations, -candidates), axis=1)[:, :take]
    selected = np.take_along_axis(candidates, order, axis=1)
    selected_offsets = np.take_along_axis(locations, order, axis=1)
    return selected, selected_offsets


class _TopTable:
    def __init__(self, folder, layers, features):
        try:
            self.table = np.lib.format.open_memmap(folder / "table.npy", mode="w+", dtype=_TABLE,
                                                   shape=(layers, features, TOP_EXAMPLES))
            self.table["activation"] = 0
            self.table["offset"] = -1
            self.counts = np.lib.format.open_memmap(folder / "counts.npy", mode="w+", dtype="<u8",
                                                    shape=(layers, features))
            self.counts[:] = 0
        except BaseException:
            self.close()
            raise

    def update(self, layer, activations, start):
        values = activations.reshape(-1, activations.shape[-1]).T
        positions = torch.arange(start, start + values.shape[1], device=values.device)
        best, offsets = _stable_topk(values, positions)
        old = self.table[layer]
        candidates = torch.from_numpy(np.concatenate((old["activation"], best), axis=1))
        locations = torch.from_numpy(np.concatenate((old["offset"], offsets), axis=1))
        best, offsets = _stable_topk(candidates, locations)
        valid = best > 0
        old["activation"] = np.where(valid, best, 0)
        old["offset"] = np.where(valid, offsets, -1)
        self.counts[layer] += (values > 0).sum(dim=1).cpu().numpy().astype(np.uint64)

    def flush(self):
        self.table.flush()
        self.counts.flush()

    def close(self):
        for name in ("table", "counts"):
            value = getattr(self, name, None)
            setattr(self, name, None)
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


def _forward(model, input_ids, attention_mask):
    if model.config.model_type in ("qwen3_5_text", "gemma2"):
        # The decoder supplies MLP probes without a vocabulary-sized logit tensor.
        return model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)


def build_index(dictionary_path, documents, *, model, tokenizer, provenance,
                dictionary=None, max_tokens=MAX_TOKENS, context_length=256, memory_guard=None,
                memory=None, memory_limit_gb=None):
    """Index ``(document_id, token_ids)`` pairs in source order.

    IDs must be unique UTF-8 strings of at most 128 bytes. Each pair is one nonempty document.
    Context blocks reset within each document. The last document can end at the token cap.
    The caller supplies disjoint partition provenance. The tool validates T2 partitions.
    An existing index is never overwritten. A failed build has no completion manifest.
    """
    _positive_integer(max_tokens, "max_tokens", MAX_TOKENS)
    _positive_integer(context_length, "context_length", 256)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("index requires text partition provenance")
    _json(provenance)
    limit = family_memory_limit(model, memory_limit_gb, memory)
    device = next(model.parameters()).device
    memory = MemoryGuard(limit, device) if memory is None else BudgetMemory(memory, limit)
    if _canonical_device(memory.device) != _canonical_device(device):
        raise ValueError("index requires a memory guard for its model device")

    def check(stage):
        memory.check(stage)
        if memory_guard is not None:
            memory_guard(stage)

    check("index admission")
    root = Path(dictionary_path)
    dictionary = dictionary if dictionary is not None else load_dictionary(
        root, model=model, memory_limit_gb=limit, memory=memory)
    data = _compatible(model, dictionary, memory_limit_gb=limit)
    check("dictionary")
    identity = _dictionary_hash(dictionary, memory=memory)
    source_hash = _source_hash(root, data["layers"])
    vocab = model.get_output_embeddings().weight.shape[0]
    tokenizer_hash = _tokenizer_hash(tokenizer, vocab)
    model_encoding = _model_encoding(model, memory)
    folder = root / "index"
    folder.mkdir()  # An incomplete prior build also requires explicit removal by its owner.
    table = _TopTable(folder, data["layers"], data["latents"])
    device = next(model.parameters()).device
    total, document_count = 0, 0
    seen = set()
    with table, (folder / "tokens.bin").open("wb") as token_file, (folder / "documents.bin").open("wb") as doc_file:
        for document_id, token_ids in documents:
            check("document")
            if not isinstance(document_id, str) or not document_id or len(document_id.encode()) > 128 or "\0" in document_id:
                raise ValueError("invalid document ID")
            if document_id in seen:
                raise ValueError("duplicate document ID")
            tokens = np.asarray(token_ids)
            if tokens.ndim != 1 or (tokens.size and (tokens.dtype.kind not in "iu" or
                                                    tokens.min() < 0 or tokens.max() >= min(vocab, 2**32))):
                raise ValueError("invalid document token IDs")
            length = min(len(tokens), max_tokens - total)
            if not length:
                raise ValueError("document token IDs must not be empty")
            seen.add(document_id)
            truncated = length < len(tokens)
            tokens = tokens[:length].astype("<u4", copy=False)
            token_file.write(tokens.tobytes())
            doc_file.write(np.array([(document_id.encode(), total, length, truncated)], dtype=_DOCUMENT).tobytes())
            for begin in range(0, length, context_length):
                check("before batch")
                ids = torch.tensor(tokens[begin:begin + context_length].astype(np.int64), device=device).unsqueeze(0)
                with _device_scope(device), torch.inference_mode(), capture(model, data["read_points"]) as saved:
                    _forward(model, ids, torch.ones_like(ids))
                    check("model batch")
                    for layer, point in enumerate(data["read_points"]):
                        weights = dictionary.layers[layer].W_enc
                        x = saved.pop(point).to(device=weights.device)
                        activations = _trace_features(dictionary.layers[layer], x, check=check)
                        table.update(layer, activations, total + begin)
                        del x, activations
                        check("dictionary batch")
                del ids
            total += length
            document_count += 1
            if total == max_tokens:
                break
        table.flush()
    check("table")
    if _model_encoding(model, memory) != model_encoding:
        raise ValueError("model encoding changed during index construction")
    manifest = {
        "format_version": INDEX_VERSION, "encoder_policy": ENCODER_POLICY,
        "model_encoding": model_encoding,
        "complete": True, "top_examples": TOP_EXAMPLES,
        "window_tokens": WINDOW_TOKENS, "model_hash": data["model_hash"],
        "dictionary_hash": identity, "dictionary_source_hash": source_hash,
        "tokenizer_hash": tokenizer_hash, "layers": data["layers"], "features": data["latents"],
        "vocab_size": vocab, "tokens": total, "documents": document_count,
        "context_length": context_length, "token_budget": max_tokens,
        "provenance": provenance, "tie_order": "activation_desc,global_offset_asc",
        "files": {name: {"bytes": (folder / name).stat().st_size,
                         "sha256": _hash_file(folder / name)} for name in _FILES},
    }
    manifest["index_hash"] = hashlib.sha256(_json(manifest).encode()).hexdigest()
    check("before completion")
    # Write the completion record last. A partial or malformed record is refused.
    (folder / "index.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


class FeatureIndex:
    """Read an index with 256 cached windows and 512 cached feature summaries.

    ``show(layer, feature)`` returns JSON data with label, count, maximum, and token lists.
    ``examples(layer, feature, limit=16)`` returns rows sorted by indexed activation.
    Each row has 33 token slots. Missing document-edge slots contain null values.
    The first query recomputes all retained windows in admitted forward batches.
    A window has fresh context, thus its center value can differ from the indexed maximum.
    Model weights, dictionary weights, tokenizer, and forward settings must remain fixed.
    Admission verifies identity once. Queries do not hash these immutable resources again.
    Close and reopen the index after a resource change. Cache clearance does not admit changes.
    """

    def __init__(self, dictionary_path, *, model, tokenizer, dictionary=None, memory=None, memory_limit_gb=None):
        self._closed = False
        self._cache = OrderedDict()
        self._summaries = OrderedDict()
        self.root = Path(dictionary_path)
        self.model, self.tokenizer = model, tokenizer
        limit = family_memory_limit(model, memory_limit_gb, memory)
        device = next(model.parameters()).device
        self.memory = MemoryGuard(limit, device) if memory is None else BudgetMemory(memory, limit)
        if _canonical_device(self.memory.device) != _canonical_device(device):
            raise ValueError("index requires a memory guard for its model device")
        self.memory.check("index admission")
        self.dictionary = dictionary if dictionary is not None else load_dictionary(
            self.root, model=model, memory_limit_gb=limit, memory=self.memory)
        data = _compatible(model, self.dictionary, memory_limit_gb=limit)
        folder = self.root / "index"
        try:
            manifest = json.loads((folder / "index.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("malformed feature index manifest")
            claimed = manifest.pop("index_hash")
            if claimed != hashlib.sha256(_json(manifest).encode()).hexdigest():
                raise ValueError("index hash mismatch")
            manifest["index_hash"] = claimed
            _encoding_policy(manifest, model, self.dictionary, self.memory)
            if (manifest["complete"] is not True or
                    manifest["top_examples"] != TOP_EXAMPLES or manifest["window_tokens"] != WINDOW_TOKENS):
                raise ValueError("incomplete or unsupported index")
            if (manifest["dictionary_source_hash"] != _source_hash(self.root, data["layers"]) or
                    manifest["dictionary_hash"] != _dictionary_hash(self.dictionary, memory=self.memory) or
                    manifest["model_hash"] != data["model_hash"]):
                raise ValueError("index dictionary identity mismatch")
            vocab = model.get_output_embeddings().weight.shape[0]
            if manifest["vocab_size"] != vocab or manifest["tokenizer_hash"] != _tokenizer_hash(tokenizer, vocab):
                raise ValueError("index tokenizer identity mismatch")
            if manifest["layers"] != data["layers"] or manifest["features"] != data["latents"]:
                raise ValueError("index dimensions mismatch")
            for name in _FILES:
                record = manifest["files"][name]
                if (folder / name).stat().st_size != record["bytes"] or _hash_file(folder / name) != record["sha256"]:
                    raise ValueError(f"incomplete or corrupt index file {name}")
            self.table = np.load(folder / "table.npy", mmap_mode="r", allow_pickle=False)
            self.counts = np.load(folder / "counts.npy", mmap_mode="r", allow_pickle=False)
            shape = (data["layers"], data["latents"])
            if self.table.dtype != _TABLE or self.table.shape != (*shape, TOP_EXAMPLES):
                raise ValueError("index table layout mismatch")
            if self.counts.dtype != np.dtype("<u8") or self.counts.shape != shape:
                raise ValueError("index counts layout mismatch")
            for field in ("tokens", "documents"):
                _positive_integer(manifest[field], field, MAX_TOKENS, minimum=0)
            self.tokens = self._map(folder / "tokens.bin", "<u4", manifest["tokens"])
            self.documents = self._map(folder / "documents.bin", _DOCUMENT, manifest["documents"])
            if len(self.documents):
                ends = self.documents["start"] + self.documents["length"]
                if (self.documents[0]["start"] != 0 or np.any(self.documents["length"] <= 0) or
                        not np.array_equal(self.documents["start"][1:], ends[:-1]) or ends[-1] != len(self.tokens)):
                    raise ValueError("index document boundaries mismatch")
            elif len(self.tokens):
                raise ValueError("index lacks document boundaries")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.close()
            raise ValueError("incomplete or malformed feature index") from error
        except BaseException:
            self.close()
            raise
        self.metadata = manifest
        self._label_identity = ("caliper.feature-labels.v1", claimed)

    def close(self):
        """Release memory maps and cached results. Repeated calls are safe."""
        self._closed = True
        self._cache.clear()
        self._summaries.clear()
        for name in ("table", "counts", "tokens", "documents"):
            value = getattr(self, name, None)
            setattr(self, name, None)
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self):
        if self._closed:
            raise ValueError("feature index is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def _map(path, dtype, count):
        dtype = np.dtype(dtype)
        if path.stat().st_size != count * dtype.itemsize:
            raise ValueError("index token or document layout mismatch")
        return np.memmap(path, mode="r", dtype=dtype, shape=(count,)) if count else np.empty(0, dtype=dtype)

    def _check_open(self):
        if self._closed:
            raise ValueError("feature index is closed")
        if self.model.training:
            raise ValueError("index requires a model in eval mode")

    def _feature_key(self, layer, feature):
        self._check_open()
        _positive_integer(layer, "layer", self.metadata["layers"] - 1, minimum=0)
        _positive_integer(feature, "feature", self.metadata["features"] - 1, minimum=0)
        return layer, feature

    def _entry(self, layer, feature):
        key = self._feature_key(layer, feature)
        if key not in self._cache:
            self._cache[key] = {}
            if len(self._cache) > CACHE_FEATURES:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def clear_cache(self):
        """Discard feature summaries and window activations."""
        self._cache.clear()
        self._summaries.clear()

    @property
    def label_identity(self):
        """Return the admitted index identity and label version without a weight scan."""
        self._check_open()
        return self._label_identity

    def _token(self, token_id):
        from ._labels import token_label
        return token_label(self.tokenizer, token_id)

    def _window_bounds(self, offset):
        document_row = int(np.searchsorted(self.documents["start"], offset, side="right") - 1)
        if document_row < 0 or offset >= len(self.tokens):
            raise ValueError("index example offset outside token store")
        doc = self.documents[document_row]
        position = offset - int(doc["start"])
        begin, end = max(0, position - 16), min(int(doc["length"]), position + 17)
        return doc, position, begin, end

    def _windows(self, layer, feature):
        rows = []
        for rank, item in enumerate(self.table[layer, feature]):
            if item["activation"] <= 0:
                break
            offset = int(item["offset"])
            doc, position, begin, end = self._window_bounds(offset)
            left = 16 - (position - begin)
            ids = self.tokens[int(doc["start"]) + begin:int(doc["start"]) + end].astype(np.int64).tolist()
            slots = [None] * WINDOW_TOKENS
            slots[left:left + len(ids)] = ids
            positions = [None] * WINDOW_TOKENS
            positions[left:left + len(ids)] = list(range(begin, end))
            rows.append({"example_id": rank, "document_id": bytes(doc["id"]).decode("utf-8"),
                         "position": position, "activation": float(item["activation"]),
                         "window_start": begin, "center": 16, "token_ids": slots,
                         "tokens": [self._token(i) if i is not None else None for i in slots],
                         "positions": positions, "valid": [i is not None for i in slots],
                         "document_truncated": bool(doc["truncated"])})
        return rows

    def _promoted(self, layer, feature, chunk_size=LABEL_VOCAB_CHUNK):
        return FeatureIndex._promoted_many(self, [(layer, feature)], chunk_size=chunk_size)[0]

    def _promoted_many(self, features, *, chunk_size=LABEL_VOCAB_CHUNK, check=None):
        """Scan vocabulary chunks once for all requested decoder directions."""
        output = self.model.get_output_embeddings().weight
        best, ids = np.empty((len(features), 0)), np.empty((len(features), 0), dtype=np.int64)
        with _device_scope(output.device), torch.inference_mode():
            # Copy selected decoder rows only. The vocabulary stays on its original device.
            directions = torch.stack([self.dictionary.layers[layer].W_dec[feature].to(
                device=output.device, dtype=output.dtype) for layer, feature in features])
            for start in range(0, output.shape[0], chunk_size):
                if check is not None:
                    check("feature label chunk")
                with _device_scope(output.device):
                    chunk = output[start:start + chunk_size]
                    values = ((chunk @ directions[0]).unsqueeze(0) if len(features) == 1
                              else (chunk @ directions.T).T)
                    positions = torch.arange(start, start + values.shape[1], device=output.device)
                    values, positions = _stable_topk(values, positions, k=3)
                best, ids = _stable_topk(torch.from_numpy(np.concatenate((best, values), axis=1)),
                                        torch.from_numpy(np.concatenate((ids, positions), axis=1)), k=3)
        return [[{"token_id": int(i), "token": self._token(i), "score": float(v)}
                 for i, v in zip(row_ids, row_values)] for row_ids, row_values in zip(ids, best)]

    def _summary(self, layer, feature, promoted):
        frequency = Counter()
        for item in self.table[layer, feature]:
            if item["activation"] <= 0:
                break
            doc, _, begin, end = self._window_bounds(int(item["offset"]))
            start = int(doc["start"])
            frequency.update(self.tokens[start + begin:start + end].tolist())
        common = [{"token_id": i, "token": self._token(i), "count": count}
                  for i, count in sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:3]]
        label = "Output: " + " | ".join(t["token"] for t in promoted)
        label += " / Examples: " + (" | ".join(t["token"] for t in common) or "none")
        return {"layer": layer, "feature": feature, "label": label,
                "count": int(self.counts[layer, feature]),
                "maximum": float(self.table[layer, feature, 0]["activation"]),
                "promoted_tokens": promoted, "example_tokens": common}

    def _label_summaries(self, features, check=None):
        self._check_open()
        keys = {}
        for count, item in enumerate(features, 1):
            if count > LABEL_FEATURES:
                raise ValueError("feature labels accept at most 512 requests")
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("feature labels require layer and feature pairs")
            keys[self._feature_key(*item)] = None
        result = {key: self._summaries[key] for key in keys if key in self._summaries}
        missing = [key for key in keys if key not in result]
        if missing:
            output = self.model.get_output_embeddings().weight
            # Bound directions, score matrices, tie repair, host candidates, and allocator reserve.
            size = max(8, output.element_size())
            workspace = (len(missing) * (4 * output.shape[1] + 16 * min(
                LABEL_VOCAB_CHUNK, output.shape[0])) * size + 64_000_000)
            self.memory.check("feature labels admission")
            if allocation_bytes(self.memory) + workspace > self.memory.limit:
                raise CleanStop("feature label allocation exceeds the process memory limit")

            def progress(stage):
                self.memory.check(stage, force=False)
                if check is not None:
                    check(stage)

            promoted = self._promoted_many(missing, check=progress)
            for (layer, feature), tokens in zip(missing, promoted):
                progress("feature label examples")
                result[layer, feature] = self._summary(layer, feature, tokens)
            self.memory.check("feature labels complete")
        if check is not None:
            check("feature labels complete")
        # Commit summaries only after all work succeeds. A label batch never evicts windows.
        for key in keys:
            self._summaries[key] = result[key]
            self._summaries.move_to_end(key)
            if len(self._summaries) > LABEL_FEATURES:
                self._summaries.popitem(last=False)
        return result

    def labels(self, features, *, check=None):
        """Return labels by ``(layer, feature)`` for at most 512 displayed nodes.

        Repeated features share one projection. At most six candidates per feature
        cross to the CPU per vocabulary chunk. Example tokens come from mapped windows.
        This method needs no model forward pass and no full identity hash.
        ``check(stage)`` can enforce the caller's deadline or cancellation request.
        """
        return {key: value["label"] for key, value in self._label_summaries(features, check).items()}

    def show(self, layer, feature):
        return deepcopy(self._label_summaries([(layer, feature)])[layer, feature])

    def _example_bytes(self, rows):
        """Bound one no-gradient layer, retained probes, and dictionary encode work."""
        config = self.model.config
        width = self.dictionary.metadata["width"]
        latents = self.dictionary.metadata["latents"]
        heads = getattr(config, "num_attention_heads", 1)
        head_dim = getattr(config, "head_dim", None) or width // heads
        intermediate = getattr(config, "intermediate_size", latents)
        key_dim = getattr(config, "linear_key_head_dim", head_dim)
        value_dim = getattr(config, "linear_value_head_dim", head_dim)
        value_heads = getattr(config, "linear_num_value_heads", heads)
        key_heads = getattr(config, "linear_num_key_heads", heads)
        dimensions = (width, latents, heads, head_dim, intermediate, key_dim, value_dim, value_heads, key_heads)
        if any(type(value) is not int or not 1 <= value <= 2**20 for value in dimensions):
            raise ValueError("example admission requires bounded model dimensions")
        size = max(4, *(p.element_size() for p in self.model.parameters()),
                   *(layer.W_enc.element_size() for layer in self.dictionary.layers))
        tokens = rows * WINDOW_TOKENS
        mixer = max(width, heads * head_dim, key_heads * key_dim, value_heads * value_dim)
        workspace = tokens * (32 * width + 8 * intermediate + 8 * mixer + 4 * latents)
        # Attention patterns can remain on fixture modules after each layer.
        patterns = 4 * rows * heads * WINDOW_TOKENS**2 * len(self.dictionary.layers)
        recurrent = 4 * rows * value_heads * key_dim * value_dim
        logits = 0 if config.model_type in ("qwen3_5_text", "gemma2") else tokens * config.vocab_size
        # Include token tensors, output lists, temporary copies, and allocator reserve.
        return size * (workspace + patterns + recurrent + logits) + tokens * 32 + 512_000_000

    def _admit_examples(self, rows):
        self.memory.check("example windows admission")
        available = self.memory.limit - allocation_bytes(self.memory)
        for count in range(min(rows, TOP_EXAMPLES), 0, -1):
            if self._example_bytes(count) <= available:
                return count
        raise CleanStop("example allocation exceeds the process memory limit")

    def _example_values(self, rows, layer, feature, device):
        """Return CPU values and release batch tensors before the next admission."""
        pad = getattr(self.tokenizer, "pad_token_id", None)
        pad = 0 if pad is None else pad
        ids = torch.full((len(rows), WINDOW_TOKENS), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row_number, row in enumerate(rows):
            valid = [i for i in row["token_ids"] if i is not None]
            ids[row_number, :len(valid)] = torch.tensor(valid, device=device)
            mask[row_number, :len(valid)] = 1
        point = self.dictionary.metadata["read_points"][layer]
        with torch.inference_mode(), capture(self.model, [point]) as saved:
            _forward(self.model, ids, mask)
            weights = self.dictionary.layers[layer].W_enc
            values = _trace_features(self.dictionary.layers[layer], saved[point].to(device=weights.device),
                                     guard=self.memory)
            if not torch.isfinite(values).all().item():
                raise ValueError("index window activations must be finite")
            result = values[..., feature].cpu().tolist()
            # Measure while batch tensors and captured probes still exist.
            self.memory.check("example windows complete")
        return result

    def examples(self, layer, feature, limit=TOP_EXAMPLES):
        entry = self._entry(layer, feature)
        _positive_integer(limit, "limit", TOP_EXAMPLES, minimum=0)
        if limit == 0:
            return []
        if "examples" not in entry:
            rows = self._windows(layer, feature)
            if rows:
                device = next(self.model.parameters()).device
                start = 0
                while start < len(rows):
                    count = self._admit_examples(len(rows) - start)
                    batch = rows[start:start + count]
                    with _device_scope(device):
                        values = self._example_values(batch, layer, feature, device)
                    self.memory.check("example windows released")
                    for row, values_row in zip(batch, values):
                        iterator = iter(values_row)
                        row["activations"] = [next(iterator) if valid else None for valid in row["valid"]]
                    start += count
            entry["examples"] = rows
        return deepcopy(entry["examples"][:limit])
