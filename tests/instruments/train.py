"""The training script of the instrument acceptance test: a small model and optimizer with a fixed seed.

Its only Caliper line is `caliper.watch(model, optimizer)`. After each optimizer step it reads the
learning rate that the step used. It prints `lr <value> step <n>` when that rate changes, so that
a test reads the new rate at the script's next step. The first output line is the stream link
that `caliper.serve()` prints.
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
last_lr = None
for step in range(1, 60001):
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr = optimizer.param_groups[0]["lr"]
    if lr != last_lr:
        print(f"lr {lr:.6g} step {step}", flush=True)
        last_lr = lr
    time.sleep(0.01)   # the pace of a slow training loop, so that a viewer keeps up
