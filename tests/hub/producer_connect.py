"""A training script for `tests/hub/e2e.py --connect`: the model and views of
tests/hub/producer.py, streamed to the hub by the package itself, with
caliper.connect(), caliper.show() and caliper.series().

It connects with no argument, so it finds the hub through the hub file
(CALIPER_HUB_FILE, which e2e.py sets). Its session id is the one the package
makes, clp-<pid>-<6 hex>; e2e.py finds it in the session list by its pid.

    CALIPER_HUB_FILE=<hub.json> python tests/hub/producer_connect.py --seconds 30

When it exits it prints one JSON line: {"id", "pid", "steps", "loss_values",
"state"}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import caliper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    import torch
    hub = caliper.connect()                      # the hub file, as a script would
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(os.getpid() % 1000)
    x = torch.randn(4096, 32, device=device)
    y = (x[:, :4].sum(dim=1) > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    end = time.monotonic() + a.seconds
    steps = 0
    while time.monotonic() < end:
        idx = torch.randint(0, x.shape[0], (256,), device=device)
        loss = torch.nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        caliper.series(loss.detach(), name="loss")
        caliper.show(model[0].weight, name="fc1.weight")
        caliper.show(model[2].weight, name="fc2.weight")
        time.sleep(0.004)
    session, state = hub.session, hub.state
    caliper.close()                              # delivers what is pending, then ends the connection
    print(json.dumps({"id": session, "pid": os.getpid(), "steps": steps, "loss_values": steps,
                      "state": state}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
