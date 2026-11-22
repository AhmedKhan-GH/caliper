"""The inference script of the hosted runs acceptance test: a small model that runs forward passes.

Its only Caliper line is `caliper.watch(model)`. Each forward pass of the model is one step of the run.
"""
import time

import torch
import torch.nn as nn

import caliper

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
caliper.watch(model)

x = torch.randn(64, 8)
with torch.no_grad():
    for step in range(1, 60001):
        out = model(x)
        if step % 50 == 0:
            print(f"pass {step} mean {out.mean().item():.6f}", flush=True)
        time.sleep(0.01)
