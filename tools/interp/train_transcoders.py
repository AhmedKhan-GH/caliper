"""Online TopK transcoders with serial layer updates and bounded checkpoints.

Run with ``python -m tools.interp.train_transcoders --help``. Real model jobs
require the shared heavy-job wrapper. The model and tokenizer use local files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from caliper.interp import capture, points
from caliper.interp._dictionary import TopKTranscoder, model_identity
from .checkpoint_state import export_dictionary, restore_checkpoint, save_checkpoint, write_json
from .memory_limits import CleanStop, MemoryGuard, StopControl, autorelease_scope
from .model_load import MODEL_ID, MODEL_REVISION, hidden_forward, load_text_model
from .text_stream import DATASET_CONFIG, DATASET_ID, DATASET_REVISION, TextStream, fineweb_documents, partition_metadata


def make_manifest(model, latents, k):
    rows = points(model)
    return {"format_version": 1, "model_id": MODEL_ID, "model_hash": model_identity(model),
            "revision": MODEL_REVISION, "layers": model.config.num_hidden_layers,
            "width": model.config.hidden_size, "latents": latents, "activation": "topk", "k": k,
            "read_points": [p["point"] for p in rows if p.get("probe") == "mlp_in"],
            "write_points": [p["point"] for p in rows if p.get("probe") == "mlp_out"],
            "training": {"dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
                         "dataset_revision": DATASET_REVISION, "split": "train",
                         "partitions": partition_metadata(), "tokens": 0, "elapsed_seconds": 0,
                         "peak_memory_bytes": 0},
            "quality": {"fvu": None, "dead_fraction": None, "cross_entropy_delta": None,
                        "evaluated_tokens": 0, "status": "not_evaluated"}}


class Trainer:
    """Fresh random transcoders and one Adam optimizer per layer."""

    def __init__(self, metadata, *, device="cpu", learning_rate=3e-4, microbatch_tokens=256,
                 auxiliary_weight=1 / 32, dead_tokens=100_000, context=256, batch_tokens=4096):
        self.metadata = metadata
        self.device = torch.device(device)
        self.settings = {"learning_rate": learning_rate, "microbatch_tokens": microbatch_tokens,
                         "auxiliary_weight": auxiliary_weight, "dead_tokens": dead_tokens,
                         "context": context, "batch_tokens": batch_tokens,
                         "decoder_normalization": "initial_directions_only",
                         "layers": metadata["layers"], "width": metadata["width"],
                         "latents": metadata["latents"], "k": metadata["k"], "dtype": "float32"}
        if (not 0 < learning_rate < 1 or microbatch_tokens < 1 or dead_tokens < 1
                or not 0 <= auxiliary_weight <= 1 or context < 2 or batch_tokens < context
                or not 1 <= metadata["k"] <= metadata["latents"]):
            raise ValueError("invalid trainer settings")
        self.layers, self.optimizers, self.last_active = [], [], []
        self.initialized = [False] * metadata["layers"]
        self.metadata["training"]["optimizer"] = {"name": "Adam", **self.settings,
                                                   "decay_start": 0.8, "final_lr_ratio": 0.1}
        width, latents = metadata["width"], metadata["latents"]
        for _ in range(metadata["layers"]):
            # Separate random encoder and decoder directions. No teacher weights enter here.
            with autorelease_scope(self.device):
                encoder = torch.randn(latents, width, device=self.device) / math.sqrt(width)
                decoder = 0.1 * F.normalize(torch.randn(latents, width, device=self.device), dim=-1)
                layer = TopKTranscoder(encoder, torch.zeros(latents, device=self.device), decoder,
                                      torch.zeros(width, device=self.device), metadata["k"], trainable=True)
                self.layers.append(layer)
                self.optimizers.append(torch.optim.Adam(layer.parameters(), lr=learning_rate, foreach=False))
                self.last_active.append(torch.zeros(latents, dtype=torch.int64, device=self.device))

    def step_layer(self, index, x, y, tokens_before, *, health=None):
        """Accumulate microbatch gradients, then update one layer once."""
        layer, optimizer = self.layers[index], self.optimizers[index]
        x, y = x.detach().reshape(-1, x.shape[-1]), y.detach().reshape(-1, y.shape[-1])
        if x.shape != y.shape or not x.numel():
            raise ValueError("MLP input and target shapes differ")
        if not self.initialized[index]:
            with torch.no_grad():
                layer.b_dec.copy_(y.mean(0))
            self.initialized[index] = True
        optimizer.zero_grad(set_to_none=True)
        variance = y.var(dim=0, unbiased=False).mean().clamp_min(1e-8)
        dead = tokens_before - self.last_active[index] >= self.settings["dead_tokens"]
        use_aux = self.settings["auxiliary_weight"] > 0 and bool(dead.any())
        active = torch.zeros_like(dead)
        mse = torch.zeros((), device=self.device)
        microbatch = self.settings["microbatch_tokens"]
        for start in range(0, len(x), microbatch):
            if health:
                health()
            with autorelease_scope(self.device):
                inputs, targets = x[start:start + microbatch], y[start:start + microbatch]
                pre = F.linear(inputs, layer.W_enc, layer.b_enc)
                values, indices = pre.relu().topk(layer.k, dim=-1)
                features = torch.zeros_like(pre).scatter(-1, indices, values)
                prediction = layer.decode(features)
                residual = targets - prediction
                loss = residual.square().mean() / variance
                mse += residual.detach().square().sum() / y.numel()
                active |= (features.detach() > 0).any(0)
                if use_aux:
                    aux_values, aux_indices = pre.masked_fill(~dead, -torch.inf).topk(min(2 * layer.k, pre.shape[-1]), dim=-1)
                    aux = torch.zeros_like(pre).scatter(-1, aux_indices, aux_values.relu())
                    aux_prediction = aux @ layer.W_dec
                    loss = loss + self.settings["auxiliary_weight"] * (residual.detach() - aux_prediction).square().mean() / variance
                (loss * (len(inputs) / len(x))).backward()
        # Decoder directions start normalized. Their magnitudes remain trainable:
        # TopK selection uses encoder scores, so rescaling encoders changes gates.
        with autorelease_scope(self.device):
            norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            with torch.no_grad():
                self.last_active[index].masked_fill_(active, tokens_before + len(x))
            optimizer.zero_grad(set_to_none=True)
            return {"fvu": float((mse / variance).item()), "gradient_norm": float(norm.item())}

    def set_learning_rate(self, fraction, *, decay_start=0.8, final_ratio=0.1):
        """Linearly reduce Adam's rate during the last part of the token budget."""
        decay = min(1.0, max(0.0, (fraction - decay_start) / (1 - decay_start)))
        rate = self.settings["learning_rate"] * (1 - decay * (1 - final_ratio))
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = rate


def initial_progress():
    return {"tokens": 0, "steps": 0, "next_layer": 0, "pending_ids": None, "elapsed_seconds": 0.0}


def train_online(model, trainer, stream, output, *, token_budget, progress=None, memory=None,
                 stop=None, save_seconds=240, max_wall_seconds=36_000, run_seconds=None,
                 started=None, clock=time.monotonic):
    """Train one model batch at a time and save partial layer progress on stop."""
    if not 0 < save_seconds <= 240 or not 0 < max_wall_seconds <= 36_000:
        raise ValueError("save interval is at most 240 seconds and total wall time is at most ten hours")
    if token_budget < 1 or (run_seconds is not None and run_seconds <= 0):
        raise ValueError("token and runtime budgets must be positive")
    output = Path(output)
    progress = initial_progress() if progress is None else progress
    start = clock() if started is None else started
    elapsed_before = progress["elapsed_seconds"]
    elapsed = lambda: elapsed_before + clock() - start
    invocation_limit = min(max_wall_seconds - elapsed_before, run_seconds or max_wall_seconds)
    # A bounded reserve covers a final checkpoint and dictionary export.
    reserve = min(60.0, invocation_limit / 4)
    last_save = clock()
    checkpoint_seconds = 0.0
    last_metrics = {}
    model.requires_grad_(False).eval()

    def health():
        if stop:
            stop.check()
        if clock() - start >= invocation_limit - max(reserve, 2 * checkpoint_seconds):
            raise CleanStop("wall time reserve")
        if clock() - last_save >= 270:
            raise CleanStop("checkpoint deadline")
        if memory:
            memory.check("train", force=False)

    def save():
        nonlocal last_save, checkpoint_seconds
        for optimizer in trainer.optimizers:
            optimizer.zero_grad(set_to_none=True)
        if memory:
            trainer.metadata["training"]["peak_memory_bytes"] = max(
                trainer.metadata["training"]["peak_memory_bytes"], memory.peak)
        before = clock()
        save_checkpoint(output / "checkpoints", trainer, stream, progress, elapsed=elapsed)
        checkpoint_seconds = clock() - before
        last_save = before
        print(json.dumps({"stage": "checkpoint", "tokens": progress["tokens"],
                          "next_layer": progress["next_layer"], "save_seconds": checkpoint_seconds,
                          "elapsed_seconds": elapsed()}), flush=True)

    reason = "token_budget"
    trainer.metadata["training"]["token_budget"] = token_budget
    captured = None
    # An initial complete checkpoint also covers an unsupported model operator.
    if not (output / "checkpoints" / "latest.json").exists():
        save()
    try:
        while progress["tokens"] < token_budget:
            health()
            if progress["pending_ids"] is None:
                requested = min(trainer.settings["batch_tokens"], token_budget - progress["tokens"])
                batch = stream.next_tokens(requested, health=health)
                if not len(batch):
                    raise CleanStop("text source exhausted")
                progress["pending_ids"] = batch.input_ids.tolist()
            trainer.set_learning_rate(progress["tokens"] / token_budget)
            count = len(progress["pending_ids"])
            context = trainer.settings["context"]
            ids = torch.tensor(progress["pending_ids"], dtype=torch.int64, device=trainer.device)
            padded = F.pad(ids, (0, (-len(ids)) % context)).reshape(-1, context)
            requested = trainer.metadata["read_points"] + trainer.metadata["write_points"]
            with autorelease_scope(trainer.device), capture(model, requested) as captured, torch.no_grad():
                hidden_forward(model, padded)
            if memory:
                memory.check("after_forward", force=False)
            for index in range(progress["next_layer"], len(trainer.layers)):
                health()
                read, write = trainer.metadata["read_points"][index], trainer.metadata["write_points"][index]
                x = captured[read].reshape(-1, trainer.metadata["width"])[:count]
                y = captured[write].reshape(-1, trainer.metadata["width"])[:count]
                metrics = trainer.step_layer(index, x, y, progress["tokens"], health=health)
                progress["next_layer"] = index + 1
                last_metrics[str(index)] = metrics
                del x, y
                captured.pop(read)
                captured.pop(write)
                if clock() - last_save >= save_seconds:
                    save()
            captured.close(clear=True)
            progress["tokens"] += count
            progress["steps"] += 1
            progress["next_layer"] = 0
            progress["pending_ids"] = None
            rate = progress["tokens"] / max(elapsed(), 1e-9)
            print(json.dumps({"stage": "train_step", "tokens": progress["tokens"], "step": progress["steps"],
                              "elapsed_seconds": elapsed(), "tokens_per_second": rate,
                              "projected_seconds": token_budget / max(rate, 1e-9), "layers": last_metrics}), flush=True)
    except CleanStop as error:
        reason = str(error)
    finally:
        if captured is not None:
            captured.close(clear=True)
    # Other exceptions retain the prior complete checkpoint and propagate.
    trainer.metadata["training"]["stop_reason"] = reason
    trainer.metadata["training"]["last_batch_metrics"] = last_metrics
    trainer.metadata["training"]["partial_batch_tokens"] = len(progress["pending_ids"] or [])
    trainer.metadata["training"]["next_layer"] = progress["next_layer"]
    save()
    progress["elapsed_seconds"] = elapsed()
    export_dictionary(output, trainer, progress)
    progress["elapsed_seconds"] = elapsed()
    trainer.metadata["training"]["elapsed_seconds"] = elapsed()
    write_json(output / "dictionary.json", trainer.metadata)
    generation = json.loads((output / "checkpoints" / "latest.json").read_text())["generation"]
    write_json(output / "runtime.json", {"generation": generation, "elapsed_seconds": elapsed()})
    print(json.dumps({"stage": "train_end", "reason": reason, "tokens": progress["tokens"],
                      "elapsed_seconds": elapsed()}), flush=True)
    return progress


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("mps", "cpu"), default="mps")
    result.add_argument("--latents", type=int, default=16_384)
    result.add_argument("--k", type=int, default=32)
    result.add_argument("--tokens", type=int, default=40_000_000)
    result.add_argument("--context", type=int, default=256)
    result.add_argument("--batch-tokens", type=int, default=4096)
    result.add_argument("--microbatch-tokens", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--auxiliary-weight", type=float, default=1 / 32)
    result.add_argument("--dead-tokens", type=int, default=100_000)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--save-seconds", type=float, default=240)
    result.add_argument("--max-wall-seconds", type=float, default=36_000)
    result.add_argument("--run-seconds", type=float)
    result.add_argument("--memory-gb", type=float, default=24)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dataset-cache", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.monotonic()
    if (not 1 <= args.latents <= 16_384 or not 1 <= args.k <= args.latents
            or not 2 <= args.context <= 256 or not args.context <= args.batch_tokens <= 4096
            or not 1 <= args.microbatch_tokens <= 256):
        raise ValueError("requested dimensions exceed the bounded trainer allocations")
    if (not 0 < args.save_seconds <= 240 or not 0 < args.max_wall_seconds <= 36_000
            or args.tokens < 1 or (args.run_seconds is not None and not 0 < args.run_seconds <= 36_000)):
        raise ValueError("invalid token, checkpoint, or wall-time budget")
    if args.output.exists() and (args.output / "checkpoints" / "latest.json").exists() and not args.resume:
        raise ValueError("output has a checkpoint; use --resume or a new output folder")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    memory = MemoryGuard(args.memory_gb, args.device)
    memory.check("start")
    with StopControl() as stop:
        model, tokenizer = load_text_model(device=args.device, memory=memory)
        metadata = make_manifest(model, args.latents, args.k)
        metadata["training"]["seed"] = args.seed
        trainer = Trainer(metadata, device=args.device, learning_rate=args.learning_rate,
                          microbatch_tokens=args.microbatch_tokens, auxiliary_weight=args.auxiliary_weight,
                          dead_tokens=args.dead_tokens, context=args.context, batch_tokens=args.batch_tokens)
        memory.check("after_dictionary_init")
        with TextStream(fineweb_documents(cache_dir=args.dataset_cache), tokenizer, "train") as stream:
            progress = restore_checkpoint(args.output / "checkpoints", trainer, stream) if args.resume else None
            memory.check("after_resume")
            train_online(model, trainer, stream, args.output, token_budget=args.tokens, progress=progress,
                         memory=memory, stop=stop, save_seconds=args.save_seconds,
                         max_wall_seconds=args.max_wall_seconds, run_seconds=args.run_seconds, started=started)


if __name__ == "__main__":
    main()
