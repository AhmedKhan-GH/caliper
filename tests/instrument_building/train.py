"""A small transformer on counting sequences, for the acceptance test of instrument building: an instrument is built on its live run.

Seed 0, vocabulary 128, context 64, width 64, 4 blocks of 4 heads, batch 8, Adam at 3e-4. The attention of each
block ends in a Linear named proj that takes the tokens flattened, so its output is a live view of shape
(512, 64). The output layer head takes the hidden state flattened to (512, 64). The script prints `step <n> loss <value>` every 10 steps and runs until
something stops it.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper

VOCAB, CONTEXT, WIDTH, BLOCKS, HEADS, BATCH = 128, 64, 64, 4, 4, 8


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, HEADS, w // HEADS).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # The projection takes the tokens flattened to (b * t, w), so its live view is the whole (512, 64)
        # tensor; a rank-3 output would show as a channel mosaic.
        return self.proj(y.transpose(1, 2).reshape(b * t, w)).view(b, t, w)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(WIDTH)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, WIDTH)
        self.pos = nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.ModuleList([Block() for _ in range(BLOCKS)])
        self.ln = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln(x).reshape(-1, WIDTH)   # (512, 64): the input of head
        return self.head(x)


torch.manual_seed(0)
dev = "mps"
model = Model().to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
caliper.watch(model, optimizer)

for step in range(1, 10_000_001):
    # Each sequence counts from a random start with a random stride of 1 to 3, so the loss falls as the
    # model learns to read the stride from the context.
    start = torch.randint(0, VOCAB, (BATCH, 1), device=dev)
    stride = torch.randint(1, 4, (BATCH, 1), device=dev)
    data = (start + stride * torch.arange(CONTEXT + 1, device=dev)) % VOCAB
    logits = model(data[:, :-1])
    loss = F.cross_entropy(logits, data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step} loss {loss.item():.6f}", flush=True)
    time.sleep(0.01)   # the pace of a slow loop, so that a viewer keeps up
