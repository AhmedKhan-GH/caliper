"""Reproducible document partitions and resumable token streams.

The source is FineWeb-Edu sample-10BT at one fixed revision. Equal text always
belongs to the same partition, even if the source contains duplicate documents.
Only token IDs and cursor metadata enter a checkpoint.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

from .memory_limits import CleanStop


DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
_CHECKSUM_CHUNK_BYTES = 8 * 1024 * 1024
_PARQUET_RESERVE_BYTES = 32 * 1024 * 1024
PARTITIONS = {
    "algorithm": "sha256-utf8-text-mod10000-v1",
    "train": [0, 9400], "quality": [9400, 9500], "index": [9500, 10000],
    "quality_tokens": 200_000, "index_tokens": 2_000_000,
    "max_document_characters": 1_000_000,
    "tokenization": "no-special-tokens-then-eos",
}


def partition_metadata():
    """Return a fresh manifest entry. Interval upper bounds are exclusive."""
    return copy.deepcopy(PARTITIONS)


def document_identity(text):
    """Return the partition and a stable signed-int64 document ID."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest, "big") % 10000
    partition = next(name for name in ("train", "quality", "index")
                     if PARTITIONS[name][0] <= bucket < PARTITIONS[name][1])
    return partition, int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _parquet_units(source, file_bytes):
    """Bound one-row decode units from trusted, well-formed Parquet metadata.

    Only flat UTF-8 BYTE_ARRAY columns with PLAIN/dictionary data and bounded
    codecs are supported. A dictionary value can occupy almost all U bytes,
    where U is the column chunk's uncompressed size. Row count cannot bound it.
    One-row batches keep both Arrow output and scalar conversion proportional
    to U, even when every dictionary index refers to that same value.

    The allowance is 2C + 20U + 64(R + 1) + 512 + 32 MiB. Two C cover compressed
    input and its copy. Eight U cover pages, dictionaries, and value descriptors.
    Four U cover Arrow output and buffer growth. Eight U cover Python Unicode,
    UTF-8 copies, and conversion overlap. R covers row offsets and level buffers.
    The fixed reserve covers decoder workspace, object headers, and alignment.
    This is an admission bound, not a sandbox for corrupt files or native codecs.
    """
    import pyarrow as arrow

    schema = source.schema
    if (source.schema_arrow.names.count("text") != 1
            or not arrow.types.is_string(source.schema_arrow.field("text").type)
            or schema.names.count("text") != 1):
        raise ValueError("local Parquet source requires one flat string text column")
    column = schema.names.index("text")
    field = schema.column(column)
    if (field.path != "text" or field.physical_type != "BYTE_ARRAY"
            or field.max_repetition_level != 0 or field.max_definition_level > 1):
        raise ValueError("unsupported Parquet text column layout")
    units = []
    for index in range(source.num_row_groups):
        group = source.metadata.row_group(index)
        chunk = group.column(column)
        rows, compressed, uncompressed = (group.num_rows, chunk.total_compressed_size,
                                           chunk.total_uncompressed_size)
        if (type(rows) is not int or rows < 0 or chunk.num_values != rows
                or any(type(size) is not int or size <= 0 for size in (compressed, uncompressed))
                or compressed > file_bytes or uncompressed >= 2**31):
            raise ValueError("invalid or impossible Parquet decode size bound")
        encodings = set(chunk.encodings)
        if (chunk.path_in_schema != "text" or chunk.physical_type != "BYTE_ARRAY"
                or chunk.file_path not in (None, "")
                or chunk.compression not in {"UNCOMPRESSED", "SNAPPY"}
                or "PLAIN" not in encodings
                or not encodings <= {"PLAIN", "RLE", "PLAIN_DICTIONARY", "RLE_DICTIONARY"}):
            raise ValueError("unsupported Parquet text encoding or codec")
        required = 2 * compressed + 20 * uncompressed + 64 * (rows + 1) + 512 + _PARQUET_RESERVE_BYTES
        units.append((rows, required))
    if sum(rows for rows, _ in units) != source.metadata.num_rows:
        raise ValueError("inconsistent Parquet row counts")
    return units


def _admit_parquet_decode(memory, required):
    """Require fresh process headroom and host memory above a 20 percent reserve."""
    if memory is None:
        raise CleanStop("Parquet decode requires a process memory guard")
    import psutil

    memory.check("parquet_decode", force=False)
    driver = (memory.driver_reader() if memory.driver_reader else
              torch.mps.driver_allocated_memory() if memory.device.type == "mps" else 0)
    rss, footprint = memory.resident(), memory.footprint()
    process = max(rss, driver, footprint if footprint is not None else rss + driver)
    memory.peak = max(memory.peak, process, memory.peak_reader())
    if memory.peak > memory.limit:
        raise CleanStop("process or MPS allocation exceeds the memory limit")
    host = psutil.virtual_memory()
    headroom = min(memory.limit - process, host.available - host.total // 5)
    if required > headroom:
        raise CleanStop("Parquet decode exceeds available memory headroom")


class LocalParquetDocuments:
    """Pinned local shards with no Arrow worker pool and an exact row cursor."""

    def __init__(self, manifest, *, health=None, memory=None):
        self.health, self.memory = health, memory
        if health is not None:
            health()
        import pyarrow.parquet as parquet
        manifest = Path(manifest)
        data = json.loads(manifest.read_text())
        required = {"dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
                    "dataset_revision": DATASET_REVISION, "split": "train"}
        if any(data.get(key) != value for key, value in required.items()):
            raise ValueError("local dataset manifest does not identify the pinned FineWeb source")
        if not isinstance(data.get("files"), list) or not data["files"]:
            raise ValueError("local dataset manifest requires ordered Parquet files")
        self.files = []
        self.units = []
        identities = []
        root = manifest.parent.resolve()
        for entry in data["files"]:
            if health is not None:
                health()
            relative = Path(entry["path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root):
                raise ValueError("local Parquet files must stay below the manifest folder")
            expected = entry.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("local Parquet files require exact SHA256 checksums")
            if type(entry.get("size_bytes")) is not int or path.stat().st_size != entry["size_bytes"]:
                raise ValueError("local Parquet size mismatch")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while True:
                    if health is not None:
                        health()
                    chunk = source.read(_CHECKSUM_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("local Parquet checksum mismatch")
            source = parquet.ParquetFile(path, pre_buffer=False)
            try:
                units = _parquet_units(source, entry["size_bytes"])
                if memory is not None and any(required > memory.limit for _, required in units):
                    raise CleanStop("Parquet decode bound exceeds the process memory limit")
                rows = source.metadata.num_rows
            finally:
                source.close()
            if health is not None:
                health()
            self.files.append((path, rows))
            self.units.append(units)
            identities.append(expected)
        fingerprint = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
        self.source_id = f"local-parquet:{DATASET_ID}/{DATASET_CONFIG}@{DATASET_REVISION}:train:{fingerprint}"
        self.shard, self.row = 0, 0

    def state_dict(self):
        return {"version": 1, "source_id": self.source_id, "shard": self.shard, "row": self.row}

    def load_state_dict(self, state):
        shard, row = state["shard"], state["row"]
        if (state.get("version") != 1 or state.get("source_id") != self.source_id
                or type(shard) is not int or type(row) is not int or not 0 <= shard <= len(self.files)
                or row < 0 or (row != 0 if shard == len(self.files) else row > self.files[shard][1])):
            raise ValueError("invalid local Parquet cursor or source identity")
        self.shard, self.row = shard, row

    def __iter__(self):
        import pyarrow.parquet as parquet
        for shard in range(self.shard, len(self.files)):
            path, _ = self.files[shard]
            skip = self.row if shard == self.shard else 0
            source = parquet.ParquetFile(path, pre_buffer=False)
            try:
                units = _parquet_units(source, path.stat().st_size)
                if units != self.units[shard]:
                    raise ValueError("Parquet decode metadata changed after admission")
                # Skip whole row groups without decoding their text on resume.
                for group, (rows, required) in enumerate(units):
                    if skip >= rows:
                        skip -= rows
                        continue
                    batches = None
                    try:
                        for _ in range(rows):
                            if self.health is not None:
                                self.health()
                            _admit_parquet_decode(self.memory, required)
                            if batches is None:
                                batches = source.iter_batches(batch_size=1, row_groups=[group],
                                                              columns=["text"], use_threads=False)
                            batch = next(batches)
                            if skip:
                                skip -= 1
                            else:
                                # Convert one scalar, never a full batch. The admission
                                # includes both the live Arrow buffers and Python copies.
                                text = batch.column(0)[0].as_py()
                                self.shard = shard
                                self.row += 1
                                yield {"text": text}
                                del text
                            del batch
                    finally:
                        if batches is not None:
                            batches.close()
            finally:
                source.close()
            self.shard, self.row = shard + 1, 0


def fineweb_documents(*, cache_dir=None, local_manifest=None, health=None, memory=None):
    """Open the pinned stream. Local decode requires memory and calls health()."""
    if local_manifest is not None:
        return LocalParquetDocuments(local_manifest, health=health, memory=memory)
    from datasets import load_dataset
    return load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION,
                        split="train", streaming=True, cache_dir=cache_dir)


@dataclass
class TokenBatch:
    """Flat tokens with original document IDs and document token positions."""

    input_ids: torch.Tensor
    document_ids: torch.Tensor
    positions: torch.Tensor

    def __len__(self):
        return self.input_ids.numel()


class TextStream:
    """A cursor over one partition with exact partial-document resume.

    ``documents`` is a re-iterable sequence or a HF IterableDataset. HF state
    preserves its shard and row position. A plain sequence resumes by row count.
    The caller must provide the same source and tokenizer after a restart.
    """

    def __init__(self, documents, tokenizer, partition, *, state=None, source_id=None):
        if partition not in ("train", "quality", "index"):
            raise ValueError("unknown text partition")
        self.documents = documents
        self.tokenizer = tokenizer
        self.partition = partition
        self.source_id = source_id or getattr(documents, "source_id", None) or f"{DATASET_ID}/{DATASET_CONFIG}@{DATASET_REVISION}:train"
        self.rows = 0
        self.emitted = 0
        self.skipped_long = 0
        self.pending = []
        self.offset = 0
        self.document_id = 0
        self.iterator = None
        self.buffer = [[], [], []]
        self.closed = False
        self._closed_state = None
        if state is not None:
            self.load_state_dict(state)

    def _iterator(self, source_state=None):
        if source_state is not None:
            if not hasattr(self.documents, "load_state_dict"):
                raise ValueError("the source cannot restore its saved cursor")
            self.documents.load_state_dict(source_state)
        self.iterator = iter(self.documents)
        if source_state is None:
            for _ in range(self.rows):
                next(self.iterator)

    def state_dict(self):
        if self.closed:
            return copy.deepcopy(self._closed_state)
        source_state = self.documents.state_dict() if self.iterator is not None and hasattr(self.documents, "state_dict") else None
        return {"version": 1, "source_id": self.source_id, "partition": self.partition,
                "partitions": partition_metadata(), "rows": self.rows, "emitted": self.emitted,
                "skipped_long": self.skipped_long, "pending": list(self.pending),
                "offset": self.offset, "document_id": self.document_id,
                "source_state": copy.deepcopy(source_state), "buffer": copy.deepcopy(self.buffer)}

    def load_state_dict(self, state):
        if self.closed:
            raise RuntimeError("cannot restore a closed text stream")
        if (state.get("version"), state.get("source_id"), state.get("partition"), state.get("partitions")) != (
            1, self.source_id, self.partition, partition_metadata(),
        ):
            raise ValueError("stream source or partition changed across resume")
        for key in ("rows", "emitted", "skipped_long", "offset", "document_id"):
            value = state[key]
            if type(value) is not int or value < 0:
                raise ValueError("invalid text cursor")
            setattr(self, key, value)
        self.pending = list(state["pending"])
        self.buffer = copy.deepcopy(state.get("buffer", [[], [], []]))
        if not 0 <= self.offset <= len(self.pending) or any(type(t) is not int or t < 0 for t in self.pending):
            raise ValueError("invalid token cursor")
        self._iterator(state.get("source_state"))

    def next_tokens(self, count, *, health=None):
        """Return at most count tokens. Exhaustion returns a shorter batch."""
        if self.closed:
            raise RuntimeError("cannot read a closed text stream")
        if type(count) is not int or count < 1:
            raise ValueError("token count must be positive")
        if self.iterator is None:
            self._iterator()
        tokens, documents, positions = self.buffer
        while len(tokens) < count:
            if self.offset >= len(self.pending):
                if health:
                    health()
                self.pending, self.offset = [], 0
                try:
                    record = next(self.iterator)
                except StopIteration:
                    break
                self.rows += 1
                text = record["text"]
                if not isinstance(text, str):
                    raise ValueError("the text source contains a non-text document")
                if len(text) > PARTITIONS["max_document_characters"]:
                    self.skipped_long += 1
                    continue
                partition, self.document_id = document_identity(text)
                if partition != self.partition or not text:
                    continue
                self.pending = list(self.tokenizer.encode(text, add_special_tokens=False))
                eos = self.tokenizer.eos_token_id
                if eos is not None:
                    self.pending.append(int(eos))
                if not self.pending:
                    continue
            end = min(len(self.pending), self.offset + count - len(tokens))
            tokens.extend(self.pending[self.offset:end])
            documents.extend([self.document_id] * (end - self.offset))
            positions.extend(range(self.offset, end))
            self.offset = end
        values = [part[:count] for part in self.buffer]
        self.buffer = [part[count:] for part in self.buffer]
        self.emitted += len(values[0])
        return TokenBatch(*(torch.tensor(part, dtype=torch.int64) for part in values))

    def close(self):
        """Release source generators after the final checkpoint preserves their cursor."""
        if self.closed:
            return
        self._closed_state = self.state_dict()
        iterator, self.iterator = self.iterator, None
        self.documents = None
        self.closed = True
        if iterator is not None and hasattr(iterator, "close"):
            iterator.close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("cannot enter a closed text stream")
        return self

    def __exit__(self, *_):
        self.close()
