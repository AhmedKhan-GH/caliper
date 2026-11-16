"""Train a small transformer and stream its attention to the Caliper app.

    python examples/build_a_dashboard/train.py [--seconds 300] [--cpu]

This is step 1 of the walk-through in README.md beside this file. The script
trains for real, on the GPU through MPS (or on the CPU with --cpu), on
synthetic data, so nothing is downloaded.

The task: each sequence is 24 random tokens followed by the same 24 tokens
again, and the model predicts every next token. The first half cannot be
predicted, but the second half can: the next token is the one that stands 23
positions back. A small transformer learns to look there, and while it does,
the attention of some heads sharpens onto one key, so their entropy falls. That
is what the app shows.

The script streams to the app with caliper.serve(): a stream server on
127.0.0.1 that prints its link. In Caliper, press F4 on the landing screen,
paste the link into the field under the session list and press Enter. Its
views, by name:

    loss           series: the training loss of every step
    attn           heatmap: the attention of the batch's first sequence, with
                   the dims (layer, head, query, key); by default the stream
                   takes the mean over the heads and tiles the layers, so the
                   app receives one query-by-key map per layer
    attn.entropy   heatmap: the entropy of each head's attention, in nats,
                   averaged over the batch and the queries, (layer, head)
    entropy.l<k>   series: layer k's mean entropy over its heads, every step

A heatmap on the stream carries at most three dimensions, and a dashboard's
spec cannot reshape, so the per-head entropy is computed here, in the script,
rather than from attn in the app.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper

VOCAB, HALF = 64, 24          # 64 token ids; 24 random tokens, then the same 24 again
LAYERS, HEADS, WIDTH = 4, 4, 128
BATCH, LR = 64, 3e-5


def make_batch(device: torch.device) -> torch.Tensor:
    """BATCH sequences of 2 * HALF tokens whose second half repeats the first."""
    first = torch.randint(0, VOCAB, (BATCH, HALF), device=device)
    return torch.cat([first, first], dim=1)


class Block(nn.Module):
    """One pre-norm transformer block. The attention is written out by hand,
    rather than with F.scaled_dot_product_attention, so that its weights can be
    returned and shown."""

    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, x: torch.Tensor, future: torch.Tensor):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, HEADS, WIDTH // HEADS).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(WIDTH // HEADS)
        attn = scores.masked_fill(future, float("-inf")).softmax(-1)      # (batch, head, query, key)
        x = x + self.out((attn @ v).transpose(1, 2).reshape(b, t, WIDTH))
        x = x + self.mlp(self.norm2(x))
        return x, attn


class Transformer(nn.Module):
    def __init__(self, length: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.position = nn.Parameter(torch.randn(length, WIDTH) * 0.02)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB)
        # A query may look at its own position and the ones before it.
        self.register_buffer("future", torch.ones(length, length, dtype=torch.bool).triu(1))

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens) + self.position[: tokens.shape[1]]
        maps = []
        for block in self.blocks:
            x, attn = block(x, self.future)
            maps.append(attn)
        # The logits, and every layer's attention: (batch, layer, head, query, key).
        return self.head(self.norm(x)), torch.stack(maps, dim=1)


def entropy(attn: torch.Tensor) -> torch.Tensor:
    """The entropy of each attention row, in nats, averaged over the batch and the
    queries: (layer, head). A query sees itself and the keys before it, so a head
    that spreads evenly over them averages about 2.9 over the 47 queries; one that
    looks at a single key has 0."""
    rows = -(attn * attn.clamp_min(1e-12).log()).sum(-1)               # (batch, layer, head, query)
    return rows.mean(dim=(0, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=300.0, help="how long to train (default 300)")
    ap.add_argument("--cpu", action="store_true", help="train on the CPU instead of MPS")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    torch.manual_seed(0)
    model = Transformer(2 * HALF - 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # The stream server. It prints its link; the app attaches to it at any time,
    # and a viewer that attaches late still receives every series value so far.
    server = caliper.serve()
    print(f"training on {device} for {args.seconds:.0f} s; attach Caliper with the link above", flush=True)

    step, start, last_print = 0, time.monotonic(), 0.0
    try:
        while time.monotonic() - start < args.seconds:
            tokens = make_batch(device)
            logits, attn = model(tokens[:, :-1])
            # Only the second half can be predicted, so only it counts in the loss.
            loss = F.cross_entropy(logits[:, HALF - 1:].reshape(-1, VOCAB), tokens[:, HALF:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            # What the app shows. Each call returns at once; a view takes a snapshot
            # only while a viewer watches it, at most 60 times a second.
            with torch.no_grad():
                ent = entropy(attn)                                      # (layer, head)
                caliper.series(loss.detach(), "loss")
                caliper.show(attn[0], "attn", cmap="magma", dims=("layer", "head", "query", "key"),
                             data_view={"reduce": {"head": "mean"}, "tile": "layer"})
                caliper.show(ent, "attn.entropy", cmap="magma", dims=("layer", "head"))
                for layer in range(LAYERS):
                    caliper.series(ent[layer].mean(), f"entropy.l{layer}")
            # The run's checkpoint: the app's run bar can pause and step it here.
            caliper.step(step)

            if time.monotonic() - last_print >= 10.0:
                last_print = time.monotonic()
                per_layer = " ".join(f"{e:.2f}" for e in ent.mean(dim=1).tolist())
                print(f"step {step:6d}  loss {loss.item():.3f}  entropy by layer {per_layer}", flush=True)
    except KeyboardInterrupt:
        print("stopped", flush=True)

    print(f"done after {step} steps; viewers keep the last frames", flush=True)
    server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
