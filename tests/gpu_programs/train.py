"""The training script of the GPU programs acceptance test: a small model on the GPU with a fixed seed.

Its only Caliper line is `caliper.watch(model, optimizer)`. The model runner of the app starts it as a managed
run. It prints `step <n> loss <value>` every 10 steps, so that the log of the run shows the loss before, during
and after an ablation. The weight `0.weight` (64 x 8) feeds a geometry program and `2.weight` (64 x 64) feeds
a pixels program.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper

torch.manual_seed(0)
dev = "mps"
model = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)).to(dev)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
caliper.watch(model, optimizer)

x = torch.randn(256, 8, device=dev)
y = x.sum(dim=1, keepdim=True)
for step in range(1, 1000001):
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step} loss {loss.item():.6f}", flush=True)
    time.sleep(0.005)   # the pace of a slow training loop, so that a viewer keeps up
