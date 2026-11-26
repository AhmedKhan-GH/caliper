"""Bounded feature examples and promoted-token labels over Backend and disk indexes."""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from ._labels import token_label
from ._memory import CleanStop, MemoryGuard, autorelease_scope as _device_scope

INDEX_VERSION = 1
TOP_EXAMPLES = 16
WINDOW_TOKENS = 33
CACHE_FEATURES = 256
LABEL_FEATURES = 512
LABEL_VOCAB_CHUNK = 4096
MAX_TOKENS = 2_000_000
_TABLE = np.dtype([("activation", "<f8"), ("offset", "<i8")])
_DOCUMENT = np.dtype([("id", "S128"), ("start", "<i8"), ("length", "<i8"), ("truncated", "?")])
_FILES = ("table.npy", "counts.npy", "tokens.bin", "documents.bin")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_hash(tokenizer, vocab: int) -> str:
    """Include token text, special IDs, and the tokenizer pipeline when available."""
    digest = hashlib.sha256()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        digest.update(backend.to_str().encode())
    digest.update(_json([getattr(tokenizer, name, None) for name in ("bos_token_id", "eos_token_id", "pad_token_id")]).encode())
    for token_id in range(vocab):
        digest.update(_json([token_id, tokenizer.convert_ids_to_tokens(token_id)]).encode())
    return digest.hexdigest()


def _positive_integer(value, name: str, maximum: int | None = None, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def _vocab_size(model) -> int:
    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    vocab = getattr(cfg, "d_vocab", None) or getattr(cfg, "vocab_size", None)
    if isinstance(vocab, int) and vocab > 0:
        return vocab
    return int(_output_embedding_weight(model).shape[0])


def _output_embedding_weight(model) -> torch.Tensor:
    if hasattr(model, "W_U"):
        # TransformerLens W_U has shape (d_model, d_vocab); transpose to (d_vocab, d_model).
        return model.W_U.T
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        return get_out().weight
    return model.lm_head.weight


def _stable_topk(values: torch.Tensor, offsets: torch.Tensor, k: int = TOP_EXAMPLES):
    """Select top-k rows with deterministic smallest-offset tie breaking."""
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("topk requires nonempty feature rows")
    if not torch.isfinite(values).all().item():
        raise ValueError("index activations must be finite")
    offsets = torch.broadcast_to(offsets, values.shape)
    take = min(k, values.shape[1])
    best, indices = values.topk(take, dim=1)
    threshold = best.min(dim=1, keepdim=True).values
    best_offsets = offsets.gather(1, indices)
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
    def __init__(self, folder: Path, layers: int, features: int):
        try:
            self.table = np.lib.format.open_memmap(
                folder / "table.npy", mode="w+", dtype=_TABLE, shape=(layers, features, TOP_EXAMPLES)
            )
            self.table["activation"] = 0
            self.table["offset"] = -1
            self.counts = np.lib.format.open_memmap(
                folder / "counts.npy", mode="w+", dtype="<u8", shape=(layers, features)
            )
            self.counts[:] = 0
        except BaseException:
            self.close()
            raise

    def update(self, layer: int, activations: torch.Tensor, start: int) -> None:
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

    def flush(self) -> None:
        self.table.flush()
        self.counts.flush()

    def close(self) -> None:
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


def _encode_batch_layers(model, dictionary, ids: torch.Tensor, layers=None) -> dict[int, torch.Tensor]:
    """Run ``model`` on ``ids`` ``(B, T)`` and return ``{layer: encoded_acts}``."""
    covered = getattr(dictionary, "covered_layers", tuple(range(dictionary.metadata["layers"])))
    target_layers = covered if layers is None else tuple(layers)
    if dictionary.kind == "transcoder":
        in_hook = dictionary.raw.feature_input_hook
        hook_map = {l: f"blocks.{l}.{in_hook}" for l in target_layers}
    else:
        hook_map = {l: str(dictionary.raw[l].cfg.metadata.hook_name) for l in target_layers}
    _, cache = model.run_with_cache(ids, prepend_bos=False, names_filter=list(hook_map.values()))
    return {l: dictionary.encode(l, cache[h]) for l, h in hook_map.items()}


def build_index(
    dictionary_path,
    documents,
    *,
    model,
    tokenizer,
    dictionary,
    provenance: dict,
    max_tokens: int = MAX_TOKENS,
    context_length: int = 256,
    memory_guard=None,
    memory=None,
) -> dict:
    """Index ``(document_id, token_ids)`` pairs into an on-disk ``FeatureIndex`` folder."""
    _positive_integer(max_tokens, "max_tokens", MAX_TOKENS)
    _positive_integer(context_length, "context_length", 256)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("index requires text partition provenance")
    _json(provenance)
    if model.training:
        raise ValueError("index requires a model in eval mode")
    check = memory_guard if memory_guard is not None else lambda stage: None
    root = Path(dictionary_path)
    out_dir = root / "index"
    if out_dir.exists():
        raise ValueError("the dictionary already has an index folder")
    staging = root / "index.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    vocab = _vocab_size(model)
    n_layers = int(dictionary.metadata["layers"])
    n_latents = int(dictionary.metadata["latents"])
    device = next(model.parameters()).device
    seen_ids: set[str] = set()
    doc_records = []
    total_tokens = 0

    try:
        with _TopTable(staging, n_layers, n_latents) as top, (staging / "tokens.bin").open("wb") as tok_out:
            for item in documents:
                check("document")
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError("each document must be a (document_id, token_ids) pair")
                doc_id, raw_tokens = item
                if not isinstance(doc_id, str) or not doc_id or len(doc_id.encode("utf-8")) > 128 or "\x00" in doc_id:
                    raise ValueError("document_id must be 1 to 128 UTF-8 bytes without NUL")
                if doc_id in seen_ids:
                    raise ValueError(f"duplicate document_id {doc_id!r}")
                seen_ids.add(doc_id)
                tok_list = list(raw_tokens)
                if not tok_list:
                    raise ValueError("indexed documents must not be empty")
                if any(type(t) is not int or not 0 <= t < vocab for t in tok_list):
                    raise ValueError("document contains an out-of-vocabulary token")
                remaining = max_tokens - total_tokens
                if remaining <= 0:
                    break
                truncated = len(tok_list) > remaining
                if truncated:
                    tok_list = tok_list[:remaining]
                doc_start = total_tokens
                tok_out.write(np.asarray(tok_list, dtype="<u4").tobytes())
                for seg_start in range(0, len(tok_list), context_length):
                    check("before batch")
                    chunk = tok_list[seg_start:seg_start + context_length]
                    ids = torch.tensor([chunk], dtype=torch.long, device=device)
                    with _device_scope(device), torch.inference_mode():
                        acts_by_layer = _encode_batch_layers(model, dictionary, ids)
                        for layer, acts in acts_by_layer.items():
                            top.update(layer, acts[0], doc_start + seg_start)
                    check("model batch")
                total_tokens += len(tok_list)
                doc_records.append((doc_id.encode("utf-8"), doc_start, len(tok_list), truncated))
                if total_tokens >= max_tokens:
                    break

            check("table")
            top.flush()

        doc_arr = np.array(doc_records, dtype=_DOCUMENT) if doc_records else np.empty((0,), dtype=_DOCUMENT)
        with (staging / "documents.bin").open("wb") as doc_out:
            doc_out.write(doc_arr.tobytes())

        check("before completion")
        tok_hash = _tokenizer_hash(tokenizer, vocab)
        file_hashes = {name: _hash_file(staging / name) for name in _FILES}
        index_hash = hashlib.sha256(_json({
            "dictionary_id": dictionary.metadata["dictionary_id"],
            "tokenizer_hash": tok_hash,
            "files": file_hashes,
        }).encode()).hexdigest()
        manifest = {
            "version": INDEX_VERSION,
            "dictionary_id": dictionary.metadata["dictionary_id"],
            "layers": n_layers,
            "latents": n_latents,
            "tokens": total_tokens,
            "documents": len(doc_records),
            "context_length": context_length,
            "tokenizer_hash": tok_hash,
            "provenance": provenance,
            "files": file_hashes,
            "index_hash": index_hash,
        }
        (staging / "index.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        os.replace(staging, out_dir)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


class _PromotedLabelsMixin:
    """Shared chunked ``W_dec @ W_U`` promoted-token scoring and bounded LRU label cache."""

    def _token(self, token_id: int) -> str:
        return token_label(self.tokenizer, token_id)

    def _feature_key(self, layer, feature) -> tuple[int, int]:
        n_layers = int(self.dictionary.metadata["layers"])
        n_latents = int(self.dictionary.metadata["latents"])
        if type(layer) is not int or not 0 <= layer < n_layers:
            raise ValueError("feature layer outside dictionary")
        if type(feature) is not int or not 0 <= feature < n_latents:
            raise ValueError("feature index outside dictionary")
        if hasattr(self.dictionary, "covered_layers") and layer not in self.dictionary.covered_layers:
            raise ValueError(f"dictionary has no layer {layer}")
        return layer, feature

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("feature index is closed")
        if getattr(self.model, "training", False):
            raise ValueError("feature index requires a model in eval mode")

    def _promoted_many(self, features: list[tuple[int, int]], *, chunk_size: int = LABEL_VOCAB_CHUNK, check=None):
        output = _output_embedding_weight(self.model)
        best = np.empty((len(features), 0))
        ids = np.empty((len(features), 0), dtype=np.int64)
        with _device_scope(output.device), torch.inference_mode():
            directions = torch.stack([
                self.dictionary.layers[layer].W_dec[feature].to(device=output.device, dtype=output.dtype)
                for layer, feature in features
            ])
            for start in range(0, output.shape[0], chunk_size):
                if check is not None:
                    check("feature label chunk")
                with _device_scope(output.device):
                    chunk = output[start:start + chunk_size]
                    values = ((chunk @ directions[0]).unsqueeze(0) if len(features) == 1
                              else (chunk @ directions.T).T)
                    positions = torch.arange(start, start + values.shape[1], device=output.device)
                    values, positions = _stable_topk(values, positions, k=3)
                best, ids = _stable_topk(
                    torch.from_numpy(np.concatenate((best, values), axis=1)),
                    torch.from_numpy(np.concatenate((ids, positions), axis=1)),
                    k=3,
                )
        return [
            [{"token_id": int(i), "token": self._token(int(i)), "score": float(v)}
             for i, v in zip(row_ids, row_values)]
            for row_ids, row_values in zip(ids, best)
        ]


def _index_latents(metadata: dict):
    """Return the latent count from ``index.json`` using ``latents`` or ``features``."""
    latents = metadata.get("latents")
    features = metadata.get("features")
    if latents is not None and features is not None and latents != features:
        raise ValueError("index metadata has conflicting latents and features counts")
    return latents if latents is not None else features


def resolve_index_dir(path: str | os.PathLike | None) -> Path | None:
    """Resolve an on-disk ``FeatureIndex`` folder from an index or dictionary path.

    Checks ``path`` itself, ``path / "index"``, and (when ``path`` exists on disk)
    the sibling ``path.parent / "index"``. Returns the first directory containing
    ``index.json``, or ``None`` if no index is found.
    """
    if path is None or not isinstance(path, (str, os.PathLike)):
        return None
    raw = os.fspath(path)
    if not raw:
        return None
    root = Path(raw).expanduser()
    if (root / "index.json").is_file():
        return root
    nested = root / "index"
    if (nested / "index.json").is_file():
        return nested
    if root.exists():
        abs_root = Path(os.path.abspath(root))
        parent = abs_root.parent
        if parent != abs_root:
            sibling = parent / "index"
            if (sibling / "index.json").is_file():
                return sibling
    return None


class FeatureIndex(_PromotedLabelsMixin):
    """Memory-mapped on-disk feature example table and promoted-token summaries."""

    @staticmethod
    def resolve(path: str | os.PathLike | None) -> Path | None:
        """Resolve an index folder from an index path or dictionary folder."""
        return resolve_index_dir(path)

    def __init__(self, dictionary_path, *, model, tokenizer, dictionary, memory=None):
        self.root = Path(dictionary_path)
        self.model = model
        self.tokenizer = tokenizer
        self.dictionary = dictionary
        dev = next(model.parameters()).device
        self.memory = memory if memory is not None else MemoryGuard(12, dev)
        self.closed = False
        self._cache: OrderedDict = OrderedDict()
        self._summaries: OrderedDict = OrderedDict()

        idx_dir = resolve_index_dir(self.root)
        if idx_dir is None:
            raise FileNotFoundError(f"feature index not found for {self.root}")
        self.index_dir = idx_dir
        self.metadata = json.loads((idx_dir / "index.json").read_text(encoding="utf-8"))
        index_layers = self.metadata.get("layers")
        index_latents = _index_latents(self.metadata)
        if index_layers != dictionary.metadata["layers"] or index_latents != dictionary.metadata["latents"]:
            raise ValueError("index dimensions differ from dictionary")
        try:
            self.table = np.lib.format.open_memmap(idx_dir / "table.npy", mode="r")
            self.counts = np.lib.format.open_memmap(idx_dir / "counts.npy", mode="r")
            if self.table.shape[:2] != (index_layers, index_latents) or self.counts.shape != (index_layers, index_latents):
                raise ValueError("index table dimensions differ from dictionary")
            tok_bytes = (idx_dir / "tokens.bin").stat().st_size
            self.tokens = (np.memmap(idx_dir / "tokens.bin", mode="r", dtype="<u4", shape=(tok_bytes // 4,))
                           if tok_bytes else np.empty((0,), dtype="<u4"))
            doc_bytes = (idx_dir / "documents.bin").stat().st_size
            self.documents = (np.memmap(idx_dir / "documents.bin", mode="r", dtype=_DOCUMENT,
                                        shape=(doc_bytes // _DOCUMENT.itemsize,))
                              if doc_bytes else np.empty((0,), dtype=_DOCUMENT))
            self._label_identity = ("caliper.feature-labels.v1", self.metadata["index_hash"])
        except BaseException:
            self.close()
            raise

    @property
    def label_identity(self):
        self._check_open()
        return self._label_identity

    def clear_cache(self) -> None:
        self._cache.clear()
        self._summaries.clear()

    def close(self) -> None:
        self.closed = True
        self.clear_cache()
        for name in ("table", "counts", "tokens", "documents"):
            val = getattr(self, name, None)
            setattr(self, name, None)
            mmap_obj = getattr(val, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _window_bounds(self, offset: int):
        doc_row = int(np.searchsorted(self.documents["start"], offset, side="right") - 1)
        if doc_row < 0 or offset >= len(self.tokens):
            raise ValueError("index example offset outside token store")
        doc = self.documents[doc_row]
        position = offset - int(doc["start"])
        begin, end = max(0, position - 16), min(int(doc["length"]), position + 17)
        return doc, position, begin, end

    def _windows(self, layer: int, feature: int) -> list[dict]:
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
            rows.append({
                "example_id": rank,
                "document_id": bytes(doc["id"]).decode("utf-8"),
                "position": position,
                "activation": float(item["activation"]),
                "window_start": begin,
                "center": 16,
                "token_ids": slots,
                "tokens": [self._token(i) if i is not None else None for i in slots],
                "positions": positions,
                "valid": [i is not None for i in slots],
                "document_truncated": bool(doc["truncated"]),
            })
        return rows

    def _summary(self, layer: int, feature: int, promoted: list[dict]) -> dict:
        frequency: Counter = Counter()
        for item in self.table[layer, feature]:
            if item["activation"] <= 0:
                break
            doc, _, begin, end = self._window_bounds(int(item["offset"]))
            start = int(doc["start"])
            frequency.update(self.tokens[start + begin:start + end].tolist())
        common = [
            {"token_id": int(i), "token": self._token(int(i)), "count": count}
            for i, count in sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:3]
        ]
        label = "Output: " + " | ".join(t["token"] for t in promoted)
        label += " / Examples: " + (" | ".join(t["token"] for t in common) or "none")
        return {
            "layer": layer,
            "feature": feature,
            "label": label,
            "count": int(self.counts[layer, feature]),
            "maximum": float(self.table[layer, feature, 0]["activation"]),
            "promoted_tokens": promoted,
            "example_tokens": common,
        }

    def _label_summaries(self, features, check=None) -> dict:
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
            output = _output_embedding_weight(self.model)
            size = max(8, output.element_size())
            workspace = len(missing) * (4 * output.shape[1] + 16 * min(LABEL_VOCAB_CHUNK, output.shape[0])) * size + 64_000_000
            self.memory.check("feature labels admission")
            if self.memory.peak + workspace > self.memory.limit:
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
        for key in keys:
            self._summaries[key] = result[key]
            self._summaries.move_to_end(key)
            if len(self._summaries) > LABEL_FEATURES:
                self._summaries.popitem(last=False)
        return result

    def labels(self, features, *, check=None) -> dict:
        return {key: value["label"] for key, value in self._label_summaries(features, check).items()}

    def show(self, layer: int, feature: int) -> dict:
        return deepcopy(self._label_summaries([(layer, feature)])[layer, feature])

    def examples(self, layer: int, feature: int, limit: int = TOP_EXAMPLES) -> list[dict]:
        self._check_open()
        key = self._feature_key(layer, feature)
        _positive_integer(limit, "limit", TOP_EXAMPLES, minimum=0)
        if limit == 0:
            return []
        entry = self._cache.get(key)
        if entry is None:
            entry = {}
            self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_FEATURES:
            self._cache.popitem(last=False)
        if "examples" not in entry:
            rows = self._windows(layer, feature)
            if rows:
                self.memory.check("example windows admission")
                device = next(self.model.parameters()).device
                pad = getattr(self.tokenizer, "pad_token_id", None)
                pad = 0 if pad is None else int(pad)
                ids = torch.full((len(rows), WINDOW_TOKENS), pad, dtype=torch.long, device=device)
                for r_idx, row in enumerate(rows):
                    valid = [i for i in row["token_ids"] if i is not None]
                    ids[r_idx, :len(valid)] = torch.tensor(valid, dtype=torch.long, device=device)
                with _device_scope(device), torch.inference_mode():
                    acts_map = _encode_batch_layers(self.model, self.dictionary, ids, layers=(layer,))
                    vals = acts_map[layer][..., feature].float().cpu()
                    if not torch.isfinite(vals).all().item():
                        raise ValueError("index window activations must be finite")
                    vals_list = vals.tolist()
                self.memory.check("example windows complete")
                for row, values_row in zip(rows, vals_list):
                    it = iter(values_row)
                    row["activations"] = [next(it) if v else None for v in row["valid"]]
            entry["examples"] = rows
        return deepcopy(entry["examples"][:limit])


class LiveFeatureIndex(_PromotedLabelsMixin):
    """Zero-disk-setup index backed by ``Backend.top_examples`` and ``W_dec @ W_U``."""

    def __init__(self, backend, *, memory=None, allow_online: bool = False):
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.dictionary = backend.dictionary
        self.memory = memory
        self.allow_online = allow_online
        self.closed = False
        self._cache: OrderedDict = OrderedDict()
        self._summaries: OrderedDict = OrderedDict()
        dict_id = self.dictionary.metadata["dictionary_id"]
        self.metadata = {
            "version": INDEX_VERSION,
            "dictionary_id": dict_id,
            "layers": self.dictionary.metadata["layers"],
            "latents": self.dictionary.metadata["latents"],
            "index_hash": hashlib.sha256(f"live:{dict_id}".encode()).hexdigest(),
        }
        self._label_identity = ("caliper.live-feature-labels.v1", self.metadata["index_hash"])

    @property
    def label_identity(self):
        self._check_open()
        return self._label_identity

    def clear_cache(self) -> None:
        self._cache.clear()
        self._summaries.clear()

    def close(self) -> None:
        self.closed = True
        self.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def labels(self, features, *, check=None) -> dict:
        self._check_open()
        keys = {}
        for count, item in enumerate(features, 1):
            if count > LABEL_FEATURES:
                raise ValueError("feature labels accept at most 512 requests")
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("feature labels require layer and feature pairs")
            keys[self._feature_key(*item)] = None
        result = {k: self._summaries[k] for k in keys if k in self._summaries}
        missing = [k for k in keys if k not in result]
        if missing:
            promoted = self._promoted_many(missing, check=check)
            for key, toks in zip(missing, promoted):
                result[key] = "Output: " + " | ".join(t["token"] for t in toks)
        if check is not None:
            check("feature labels complete")
        for key in keys:
            self._summaries[key] = result[key]
            self._summaries.move_to_end(key)
            if len(self._summaries) > LABEL_FEATURES:
                self._summaries.popitem(last=False)
        return result

    def examples(self, layer: int, feature: int, limit: int = TOP_EXAMPLES) -> list[dict]:
        self._check_open()
        key = self._feature_key(layer, feature)
        _positive_integer(limit, "limit", TOP_EXAMPLES, minimum=0)
        if limit == 0:
            return []
        prompts_sig = tuple(self.backend.recent_prompts)
        cached = self._cache.get(key)
        if cached is None or cached[0] != prompts_sig:
            rows = self.backend.top_examples(layer, feature, limit=TOP_EXAMPLES, allow_online=self.allow_online)
            self._cache[key] = (prompts_sig, rows)
            self._cache.move_to_end(key)
            while len(self._cache) > CACHE_FEATURES:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
            rows = cached[1]
        return deepcopy(rows[:limit])

    def show(self, layer: int, feature: int) -> dict:
        self._check_open()
        key = self._feature_key(layer, feature)
        promoted = self._promoted_many([key])[0]
        rows = self.examples(layer, feature, limit=TOP_EXAMPLES)
        freq: Counter = Counter(
            tok_id for r in rows for tok_id in r["token_ids"] if tok_id is not None
        )
        common = [
            {"token_id": int(i), "token": self._token(int(i)), "count": c}
            for i, c in sorted(freq.items(), key=lambda it: (-it[1], it[0]))[:3]
        ]
        label = "Output: " + " | ".join(t["token"] for t in promoted)
        label += " / Examples: " + (" | ".join(t["token"] for t in common) or "none")
        maximum = max((float(r["activation"]) for r in rows), default=0.0)
        return {
            "layer": layer,
            "feature": feature,
            "label": label,
            "count": len(rows),
            "maximum": maximum,
            "promoted_tokens": promoted,
            "example_tokens": common,
        }
