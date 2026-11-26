"""Build or query an offline feature example index over a TransformerLens backend."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

import torch

from caliper.interp._backend import load_backend
from caliper.interp._index import FeatureIndex, MAX_TOKENS, build_index
from caliper.interp._memory import CleanStop, MAX_PROCESS_GB, MemoryGuard, StopControl
from .stop_records import clean_stop_record
from .text_stream import TextStream, fineweb_documents

DEFAULT_MEMORY_GB = 24


def stream_documents(stream, *, batch_tokens: int = 4096, check=None):
    """Restore complete documents across token batches and skip duplicates."""
    if stream.partition != "index":
        raise ValueError("the example index requires the index partition")
    if type(batch_tokens) is not int or not 1 <= batch_tokens <= 4096:
        raise ValueError("stream batch size must be between one and 4096")
    current, tokens, seen = None, [], set()
    for_batch = check if check is not None else lambda stage: None
    while True:
        for_batch("text batch")
        batch = stream.next_tokens(batch_tokens, health=lambda: for_batch("text document"))
        if not (batch.input_ids.ndim == batch.document_ids.ndim == batch.positions.ndim == 1
                and len(batch.input_ids) == len(batch.document_ids) == len(batch.positions)):
            raise ValueError("invalid text batch shape")
        if any(t.dtype != torch.int64 or t.device.type != "cpu"
               for t in (batch.input_ids, batch.document_ids, batch.positions)):
            raise ValueError("text batches must contain CPU int64 tensors")
        for token, document_id, position in zip(
            batch.input_ids.tolist(), batch.document_ids.tolist(), batch.positions.tolist()
        ):
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


def jsonl_documents(path: Path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("each source row requires a text field")
            yield {"text": row["text"]}


def _bounded_int(maximum: int):
    def parse(value):
        number = int(value)
        if not 1 <= number <= maximum:
            raise argparse.ArgumentTypeError(f"value must be between one and {maximum}")
        return number
    return parse


def _memory_gb(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= MAX_PROCESS_GB:
        raise argparse.ArgumentTypeError(f"memory limit must be positive and at most {MAX_PROCESS_GB} GB")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--dictionary", type=Path, required=True)
    result.add_argument("--sae-id")
    result.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    result.add_argument("--tokens", type=_bounded_int(MAX_TOKENS), default=MAX_TOKENS)
    result.add_argument("--context", type=_bounded_int(256), default=256)
    result.add_argument("--memory-gb", type=_memory_gb, default=DEFAULT_MEMORY_GB)
    result.add_argument("--partition", choices=("index",), default="index")
    result.add_argument("--dataset-cache", type=Path)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--text-jsonl", type=Path)
    source.add_argument("--local-dataset-manifest", type=Path)
    result.add_argument("--query", type=int, nargs=2, metavar=("LAYER", "FEATURE"))
    result.add_argument("--limit", type=int, choices=range(17), default=16)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.query is None and (args.dictionary / "index").exists():
        raise ValueError("the dictionary already has an index folder")
    try:
        with StopControl() as stop, ExitStack() as resources:
            memory = MemoryGuard(args.memory_gb, args.device)
            memory.check("start")

            def check(stage):
                stop.check()
                memory.check(stage, force=stage in ("before batch", "model batch", "table", "before completion"))

            backend = load_backend(args.model, args.dictionary, sae_id=args.sae_id, device=args.device, guard=memory)
            if args.query is not None:
                with FeatureIndex(
                    args.dictionary,
                    model=backend.model,
                    tokenizer=backend.tokenizer,
                    dictionary=backend.dictionary,
                    memory=memory,
                ) as index:
                    layer, feature = args.query
                    response = {
                        "show": index.show(layer, feature),
                        "examples": index.examples(layer, feature, args.limit),
                    }
                print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
                return 0

            provenance = {"partition": args.partition}
            if args.text_jsonl is not None:
                raw_source = jsonl_documents(args.text_jsonl)
                provenance["source_kind"] = "jsonl"
                source_id = "jsonl"
            elif args.local_dataset_manifest is not None:
                raw_source = fineweb_documents(local_manifest=args.local_dataset_manifest, health=lambda: check("source"), memory=memory)
                provenance["source_kind"] = "local_parquet"
                source_id = raw_source.source_id
            else:
                raw_source = fineweb_documents(cache_dir=args.dataset_cache, health=lambda: check("source"), memory=memory)
                provenance["source_kind"] = "fineweb"
                source_id = None
            stream = resources.enter_context(TextStream(raw_source, backend.tokenizer, args.partition, source_id=source_id))
            provenance["source_id"] = stream.source_id
            documents = stream_documents(stream, check=check)
            resources.callback(documents.close)
            manifest = build_index(
                args.dictionary,
                documents,
                model=backend.model,
                tokenizer=backend.tokenizer,
                dictionary=backend.dictionary,
                provenance=provenance,
                max_tokens=args.tokens,
                context_length=args.context,
                memory_guard=check,
                memory=memory,
            )
            memory.check("complete")
            print(json.dumps({
                "stage": "index_ready",
                "tokens": manifest["tokens"],
                "documents": manifest["documents"],
                "index_hash": manifest["index_hash"],
                "peak_memory_bytes": memory.peak,
            }), flush=True)
            return 0
    except CleanStop as error:
        print(json.dumps(clean_stop_record(error)), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
