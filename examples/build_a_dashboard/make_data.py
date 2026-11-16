"""Writes the two data files of the build-a-dashboard walk-through.

    python examples/build_a_dashboard/make_data.py [--out DIR]

data/run_metrics.parquet
    The logged metrics of a small character-level language model's training
    run, one row every 10 steps for 3,000 steps (300 rows): step (int64),
    loss, accuracy and lr (float64). The learning rate warms up linearly for
    200 steps and then follows a cosine decay; the loss falls from ln(65), the
    loss of a uniform guess over a 65-character vocabulary, as the summed
    learning rate grows, and the accuracy rises as the loss falls.

data/attention.safetensors
    One tensor, attn, float32 [4, 4, 32, 32]: the attention weights of four
    layers of four heads over a sequence of 32 tokens, as layers x heads x
    query x key. Every row is causal (a query attends only to itself and
    earlier keys) and sums to 1. Each head has one of the patterns that small
    trained transformers show, named in the file's __metadata__: previous
    token, current token, local window, first token, induction (the token
    after the previous occurrence of the current one, in a sequence that
    repeats every 12 tokens) and broad. Layer 1, head 2 is an induction head.

The values are synthetic and made from a fixed seed, so the files are the same
on every run (with the same NumPy and PyArrow). No downloads; NumPy and
PyArrow only. The safetensors file is written directly: an 8-byte
little-endian header length, the JSON header, then the raw tensor bytes.
"""

import argparse
import json
import math
import os
import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VOCAB = 65
STEPS = 3000
EVERY = 10
WARMUP = 200
LR_MAX = 3e-4
LR_MIN = 3e-5

LAYERS, HEADS, T = 4, 4, 32
PERIOD = 12   # the sequence repeats every PERIOD tokens, which an induction head follows

# The pattern of each head, layer by layer.
KINDS = [
    ["previous", "current", "local", "broad"],
    ["first", "previous", "induction", "local"],
    ["induction", "broad", "previous", "first"],
    ["local", "induction", "first", "broad"],
]


def metrics(rng):
    step = np.arange(0, STEPS, EVERY, dtype=np.int64)
    s = step.astype(np.float64)
    warm = LR_MAX * (s + 1.0) / WARMUP
    cos = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + np.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP)))
    lr = np.where(s < WARMUP, warm, cos)
    # The loss falls with the learning rate summed over the steps so far.
    progress = np.cumsum(lr * EVERY) / np.sum(lr * EVERY)
    start, floor = math.log(VOCAB), 1.36
    clean = floor + (start - floor) * (0.85 * np.exp(-7.0 * progress) + 0.15 * np.exp(-1.5 * progress))
    noise = 0.035 * (0.4 + 0.6 * (clean - floor) / (start - floor))
    loss = clean * (1.0 + noise * rng.standard_normal(step.size))
    loss[0] = start
    # The accuracy rises as the loss falls, from a uniform guess.
    learned = 1.0 - (clean - floor) / (start - floor)
    accuracy = 1.0 / VOCAB + (0.585 - 1.0 / VOCAB) * learned ** 1.4
    accuracy = np.clip(accuracy + 0.008 * rng.standard_normal(step.size), 0.0, 1.0)
    accuracy[0] = 1.0 / VOCAB
    return pa.table({
        "step": pa.array(step, pa.int64()),
        "loss": pa.array(np.round(loss, 6), pa.float64()),
        "accuracy": pa.array(np.round(accuracy, 6), pa.float64()),
        "lr": pa.array(np.round(lr, 9), pa.float64()),
    })


def head_scores(kind, rng):
    q = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    if kind == "previous":
        s = np.where(k == q - 1, 7.0, 0.0)
    elif kind == "current":
        s = np.where(k == q, 6.0, 0.0)
    elif kind == "local":
        s = -0.7 * (q - k)
    elif kind == "first":
        s = np.where(k == 0, 5.0, 0.0)
    elif kind == "induction":
        s = np.where(k == q - PERIOD + 1, 7.0, 0.0) + np.where(k == q - 2 * PERIOD + 1, 4.0, 0.0)
    else:   # broad
        s = 0.6 * np.sin(k / 4.0)
    return s + 0.35 * rng.standard_normal((T, T))


def attention(rng):
    q = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    causal = k <= q
    attn = np.zeros((LAYERS, HEADS, T, T), np.float32)
    for layer in range(LAYERS):
        for head in range(HEADS):
            s = np.where(causal, head_scores(KINDS[layer][head], rng), -np.inf)
            e = np.exp(s - s.max(axis=1, keepdims=True))
            attn[layer, head] = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    return attn


def write_safetensors(path, name, array, metadata):
    data = np.ascontiguousarray(array, dtype="<f4").tobytes()
    header = {
        "__metadata__": metadata,
        name: {"dtype": "F32", "shape": list(array.shape), "data_offsets": [0, len(data)]},
    }
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(data)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(here, "data"), help="the folder to write into (default: data/ beside this script)")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261116)

    table = metrics(rng)
    parquet = os.path.join(out, "run_metrics.parquet")
    pq.write_table(table, parquet, compression="zstd")

    attn = attention(rng)
    meta = {
        "layout": "layer, head, query, key",
        "heads": json.dumps({f"{l}.{h}": KINDS[l][h] for l in range(LAYERS) for h in range(HEADS)}, sort_keys=True),
    }
    tensors = os.path.join(out, "attention.safetensors")
    write_safetensors(tensors, "attn", attn, meta)

    for path in (parquet, tensors):
        print(f"wrote {os.path.relpath(path)} ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
