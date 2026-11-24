"""A bounded held-out example index and local feature queries.

Run with ``python -m tools.interp.build_index --help``. Real jobs require the
shared heavy-job wrapper. Model and tokenizer loads use the local cache only.

Indexes require the exact recorded source and the disjoint index partition.
Imported dictionaries use an independent evaluation source. Their publisher
training overlap remains unknown.
The JSONL option serves only tiny-lm fixture dictionaries in CPU test harnesses,
which supply a fixture model loader. Their manifests must record the JSONL
SHA256 source identity and the same document partition rules.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

import torch

from caliper.interp import load_dictionary
from caliper.interp._index import FeatureIndex, MAX_TOKENS, _hash_file, build_index
from .memory_limits import CleanStop, MemoryGuard, StopControl
from .model_load import GEMMA_ID, MODEL_ID, load_text_model
from .stop_records import clean_stop_record
from .text_stream import (DATASET_CONFIG, DATASET_ID, DATASET_REVISION, TextStream,
                          evaluation_source_metadata, fineweb_documents, partition_metadata)

MEMORY_LIMIT_GB = 12


def validate_partitions(metadata):
    """Require the pinned evaluation source and unchanged document partitions."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("training"), dict):
        raise ValueError("dictionary requires text source metadata")
    training = evaluation_source_metadata(metadata)
    expected = {"dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
                "dataset_revision": DATASET_REVISION, "split": "train",
                "partitions": partition_metadata()}
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("dictionary text source or partitions differ from the training dataset")
    if not isinstance(training.get("source_id"), str) or not training["source_id"]:
        raise ValueError("dictionary requires training source identity")
    if "import_source" in metadata:
        expected["publisher_training_overlap"] = "unknown"
    return expected


def stream_documents(stream, *, batch_tokens=4096, check=None):
    """Restore complete documents across T2 token batches and skip duplicate text.

    The numeric T2 document ID becomes a decimal string for lossless JSON transport.
    Positions must start at zero and increase by one. This adapter cannot resume
    from the middle of a document. At most one document and one batch stay in memory.
    """
    if stream.partition != "index":
        raise ValueError("the example index requires the index partition")
    if type(batch_tokens) is not int or not 1 <= batch_tokens <= 4096:
        raise ValueError("stream batch size must be between one and 4096")
    current, tokens, seen = None, [], set()
    for_batch = check if check is not None else lambda stage: None
    while True:
        for_batch("text batch")
        batch = stream.next_tokens(batch_tokens, health=lambda: for_batch("text document"))
        if not (batch.input_ids.ndim == batch.document_ids.ndim == batch.positions.ndim == 1 and
                len(batch.input_ids) == len(batch.document_ids) == len(batch.positions)):
            raise ValueError("invalid text batch shape")
        if any(t.dtype != torch.int64 or t.device.type != "cpu" for t in
               (batch.input_ids, batch.document_ids, batch.positions)):
            raise ValueError("text batches must contain CPU int64 tensors")
        for token, document_id, position in zip(batch.input_ids.tolist(), batch.document_ids.tolist(), batch.positions.tolist()):
            if document_id < 0 or token < 0 or position < 0:
                raise ValueError("invalid text batch value")
            if current is None or document_id != current or position == 0:
                if current is not None and current not in seen:
                    seen.add(current)
                    yield str(current), tokens
                current, tokens = document_id, []
            if position != len(tokens):
                raise ValueError("text stream document positions are not contiguous")
            tokens.append(token)
        if len(batch) < batch_tokens:
            if current is not None and current not in seen:
                yield str(current), tokens
            return


def jsonl_documents(path):
    """Read a fixture source without retention of raw text between rows."""
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("each source row requires a text field")
            yield {"text": row["text"]}


def _index_stream(args, metadata, provenance, resources, *, health=None, memory=None):
    """Admit the exact source before model load, without token consumption."""
    provenance["partition"] = args.partition
    provenance["duplicate_documents"] = "first occurrence only"
    if args.text_jsonl is not None:
        if metadata.get("model_id") != "tiny-lm":
            raise ValueError("--text-jsonl is only supported for tiny-lm fixture dictionaries. Use the exact training source.")
        source_hash = _hash_file(args.text_jsonl)
        source = jsonl_documents(args.text_jsonl)
        source_id = "jsonl:sha256:" + source_hash
        provenance.update(source_kind="staged_jsonl", source_hash=source_hash)
    elif args.local_dataset_manifest is not None:
        source = fineweb_documents(local_manifest=args.local_dataset_manifest, health=health, memory=memory)
        source_id = None
        provenance["source_kind"] = "pinned_local_parquet"
    else:
        source = fineweb_documents(cache_dir=args.dataset_cache, health=health, memory=memory)
        source_id = None
        provenance["source_kind"] = "pinned_dataset"
    if hasattr(source, "close"):
        resources.callback(source.close)
    if args.local_dataset_manifest is not None:
        source_id = source.source_id
    stream = resources.enter_context(TextStream(source, None, args.partition, source_id=source_id))
    if stream.source_id != evaluation_source_metadata(metadata)["source_id"]:
        label = "evaluation" if "import_source" in metadata else "training"
        raise ValueError(f"index source differs from the {label} source")
    provenance["source_id"] = stream.source_id
    return stream


def _bounded_int(maximum):
    def parse(value):
        number = int(value)
        if not 1 <= number <= maximum:
            raise argparse.ArgumentTypeError(f"value must be between one and {maximum}")
        return number
    return parse


def _memory_gb(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 28:
        raise argparse.ArgumentTypeError("memory limit must be positive and at most 28 GB")
    return number


class _IndexParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        ceiling = 28 if result.model == GEMMA_ID else MEMORY_LIMIT_GB
        result.memory_gb = ceiling if result.memory_gb is None else result.memory_gb
        if result.memory_gb > ceiling:
            self.error(f"memory limit must be at most {ceiling} GB for this model")
        return result


def parser():
    result = _IndexParser(description=__doc__)
    result.add_argument("--dictionary", type=Path, required=True)
    result.add_argument("--model", choices=(MODEL_ID, GEMMA_ID), default=MODEL_ID)
    result.add_argument("--device", choices=("cpu", "mps"), default="mps")
    result.add_argument("--tokens", type=_bounded_int(MAX_TOKENS), default=MAX_TOKENS)
    result.add_argument("--context", type=_bounded_int(256), default=256)
    result.add_argument("--memory-gb", type=_memory_gb)
    result.add_argument("--partition", choices=("index",), default="index")
    result.add_argument("--dataset-cache", type=Path)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--text-jsonl", type=Path,
                        help="Fixture harness only: tiny-lm source with one text field per row and the exact training SHA256 identity")
    source.add_argument("--local-dataset-manifest", type=Path, help="Pinned local Parquet source manifest")
    result.add_argument("--query", type=int, nargs=2, metavar=("LAYER", "FEATURE"))
    result.add_argument("--limit", type=int, choices=range(17), default=16)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    # Refuse provenance and output errors before a real model enters memory.
    metadata = json.loads((args.dictionary / "dictionary.json").read_text(encoding="utf-8"))
    provenance = validate_partitions(metadata)
    if args.query is None and (args.dictionary / "index").exists():
        raise ValueError("the dictionary already has an index folder")
    if args.query is not None and not (0 <= args.query[0] < metadata["layers"] and
                                      0 <= args.query[1] < metadata["latents"]):
        raise ValueError("query layer or feature is outside the dictionary")
    try:
        with StopControl() as stop, ExitStack() as resources:
            stop.check()
            memory = MemoryGuard(args.memory_gb, args.device)
            memory.check("start")
            stop.check()
            def check(stage):
                stop.check()
                # Resource checks occur before and after each bounded model batch.
                memory.check(stage, force=stage in ("dictionary", "before batch", "model batch", "table", "before completion"))

            stream = None
            if args.query is None:
                stream = _index_stream(args, metadata, provenance, resources,
                                       health=lambda: check("source admission"), memory=memory)
                check("source")
                stop.check()
            selection = {"model_id": args.model} if args.model != MODEL_ID else {}
            model, tokenizer = load_text_model(device=args.device, memory=memory, **selection)
            stop.check()
            dictionary = load_dictionary(args.dictionary, model=model, memory_limit_gb=args.memory_gb, memory=memory)
            check("dictionary")
            if args.query is not None:
                with FeatureIndex(args.dictionary, model=model, tokenizer=tokenizer,
                                  dictionary=dictionary, memory=memory) as index:
                    memory.check("index")
                    layer, feature = args.query
                    response = {"show": index.show(layer, feature), "examples": index.examples(layer, feature, args.limit)}
                memory.check("query")
                print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
                return 0
            stream.tokenizer = tokenizer
            documents = stream_documents(stream, check=check)
            resources.callback(documents.close)
            manifest = build_index(
                args.dictionary, documents, model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance=provenance, max_tokens=args.tokens,
                context_length=args.context, memory_guard=check, memory=memory,
            )
            memory.check("complete")
            print(json.dumps({"stage": "index_ready", "tokens": manifest["tokens"],
                              "documents": manifest["documents"], "index_hash": manifest["index_hash"],
                              "peak_memory_bytes": memory.peak}), flush=True)
            return 0
    except CleanStop as error:
        print(json.dumps(clean_stop_record(error)), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
