"""Train a two-layer transformer until induction heads form, for the Attention Arc Tower.

    python tests/acceptance/arc_tower_train.py [--steps N] [--pace S] [--device mps|cpu]

The model is an attention-only transformer with two layers of four heads, a width of 64 and a
vocabulary of 64 words, and with no layer norm after the last layer, as in the first studies of
induction heads. Thus each head adds its own share to the logits, and the ablation of one head
shows in the loss. Each training sequence holds 32 random tokens, and a segment of 12 of them
repeats after a random gap of 0 to 8 tokens. The loss stays near ln 64 while the model can only
guess, and it drops at a bump when heads of layer 1 learn to copy: from a token of the second copy
they attend to the token after the same token's first occurrence, and predict it. The seed is
fixed, and on an Apple GPU the bump comes between steps 600 and 1,000, which the default pace puts
about a minute and a half after the start.

Row 0 of every batch is the fixed sentence ``SENTENCE``, whose tokens 4 to 15 repeat at 16 to 27,
and the training loss leaves it out. Thus the attention that a viewer shows is always that of the
same sentence, and the model never trains on it.

The script calls ``caliper.watch(model, optimizer)``, which publishes these sources for a managed
run:

- ``live:blocks.<i>.attn.weights.act``: the attention weights of layer ``i`` after the softmax, of
  shape (64 * 4 * 32, 32): the (64, 4, 32, 32) weights of batch, head, query token and key token,
  with the first three axes as rows. A (64, 4, 32, 32) tensor would show as RGBA images of its
  four heads, which a viewer cannot bind as weights; the rows keep the same memory, thus batch
  row 0 is still the first 4 * 32 * 32 values;
- ``live:token_loss.act``: the loss of each token, of shape (64, 31), where column ``j`` is the
  loss of the prediction of token ``j + 1``. Row 0, columns 16 to 26, is the in-context loss: the
  tokens 17 to 27 of the sentence, which only a copy from the first occurrence predicts;
- the intervention points ``blocks.<i>.attn#<h>`` of each head, through the attention module's
  integer ``n_head`` and its output projection ``proj``.

After each optimizer step, ``caliper.series`` publishes the induction score of each head of each
layer as the series ``induction_l<i>_h<h>``: the head's mean weight of the sentence from a token of
the second copy to the token after the same token's first occurrence. It is the score that the
``scores`` pass of the piece ``attention_arcs`` computes. The calls pass 0-d tensors on the device,
thus they add no wait for the GPU to a step.

The script prints ``step <n> loss <value> in-context <value>`` every 50 steps.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper

SEED = 0
LAYERS = 2
HEADS = 4
WIDTH = 64
TOKENS = 32
SEGMENT = 12        # the length of the repeated segment
BATCH = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
STEPS = 20_000
PACE_S = 0.1        # the pause after each step, so that a person and an agent can follow the bump

#: The fixed sentence of row 0. Its tokens 4 to 15 repeat at 16 to 27, thus an induction head
#: attends from token t of the second copy to token t - 11.
SENTENCE = ("the old map shows red fox jumps over lazy dogs near green hills under blue skies "
            "red fox jumps over lazy dogs near green hills under blue skies and then it ends").split()

#: The 64 words of the vocabulary: the 20 words of the sentence, then 44 others.
WORDS = list(dict.fromkeys(SENTENCE)) + (
    "a an big small cold warm dark light fast slow tall short new long wide deep soft hard "
    "loud calm wet dry bird fish tree rock road door wall roof river lake hill sand snow rain "
    "wind star moon sun cat cow owl bee").split()
VOCAB = len(WORDS)
assert VOCAB == 64 and len(SENTENCE) == TOKENS


class Attention(nn.Module):
    """Causal self-attention whose weights pass through the leaf ``weights``."""

    def __init__(self):
        super().__init__()
        self.n_head = HEADS
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.weights = nn.Identity()
        self.register_buffer("mask", torch.tril(torch.ones(TOKENS, TOKENS, dtype=torch.bool)))

    def forward(self, x):
        b, t, c = x.shape
        d = c // self.n_head
        q, k, v = (z.view(b, t, self.n_head, d).transpose(1, 2) for z in self.qkv(x).split(c, dim=2))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(~self.mask[:t, :t], float("-inf"))
        w = self.weights(F.softmax(scores, dim=-1).reshape(b * self.n_head * t, t)).view(b, self.n_head, t, t)
        self.last = w[0].detach()                          # the sentence's weights, for the head scores
        y = (w @ v).transpose(1, 2).reshape(b, t, c)     # the heads side by side, head 0 first
        return self.proj(y)


class Block(nn.Module):
    """One layer: a layer norm and the attention, added to the residual stream."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(WIDTH)
        self.attn = Attention()

    def forward(self, x):
        return x + self.attn(self.ln1(x))


class Model(nn.Module):
    """The transformer. Its forward pass returns the loss of each token through ``token_loss``."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.position = nn.Embedding(TOKENS, WIDTH)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.head = nn.Linear(WIDTH, VOCAB)
        self.token_loss = nn.Identity()

    def forward(self, tokens):
        b, t = tokens.shape
        x = self.embed(tokens) + self.position(torch.arange(t, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1), reduction="none")
        return self.token_loss(loss.view(b, t - 1))


def batch(gen: torch.Generator) -> torch.Tensor:
    """One batch of random sequences with a repeated segment, and the sentence in row 0.

    The gap between the two copies is random too, from 0 to 8 tokens, so that no head can predict
    a token from a fixed distance: only a copy from the token after the previous occurrence of the
    same token predicts the second copy.
    """
    x = torch.randint(0, VOCAB, (BATCH, TOKENS), generator=gen)
    free = TOKENS - 2 * SEGMENT
    gap = torch.randint(0, free + 1, (BATCH, 1), generator=gen)
    start = (torch.rand(BATCH, 1, generator=gen) * (free - gap + 1)).long()
    first = start + torch.arange(SEGMENT)
    rows = torch.arange(BATCH)[:, None]
    x[rows, first + SEGMENT + gap] = x[rows, first]
    x[0] = torch.tensor([WORDS.index(w) for w in SENTENCE])
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=STEPS, help="the steps to train")
    ap.add_argument("--pace", type=float, default=PACE_S, help="the pause after each step in seconds")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                    help="the device; the derive halves of pieces read tensors on mps")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED + 1)
    model = Model().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    caliper.watch(model, optimizer)
    second = torch.arange(4 + SEGMENT, 4 + 2 * SEGMENT, device=args.device)   # the tokens of the second copy

    for step in range(1, args.steps + 1):
        per_token = model(batch(gen).to(args.device))
        loss = per_token[1:].mean()                        # row 0, the sentence, never trains
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # The induction score of each head: its mean weight from a token of the second copy to the
        # token after the same token's first occurrence, published as one series for each head.
        for layer, block in enumerate(model.blocks):
            scores = block.attn.last[:, second, second - SEGMENT + 1].mean(dim=1)
            for head in range(HEADS):
                caliper.series(scores[head], name=f"induction_l{layer}_h{head}")
        if step % 50 == 0:
            in_context = per_token[0, 16:27].mean()
            print(f"step {step} loss {loss.item():.4f} in-context {in_context.item():.4f}", flush=True)
        if args.pace > 0:
            time.sleep(args.pace)


if __name__ == "__main__":
    main()
