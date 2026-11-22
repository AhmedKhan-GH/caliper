"""The training script of the hosted runs acceptance test: a small model and optimizer with a fixed seed.

Its only Caliper line is `caliper.watch(model, optimizer)`. The model runner starts it as a managed run,
so the script reads no argument and starts no server of its own. It prints a line each 50 steps.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
caliper.watch(model, optimizer)

x = torch.randn(64, 8)
y = x.sum(dim=1, keepdim=True)
for step in range(1, 60001):
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step} loss {loss.item():.6f}", flush=True)
    time.sleep(0.01)   # the pace of a slow training loop, so that a viewer keeps up
