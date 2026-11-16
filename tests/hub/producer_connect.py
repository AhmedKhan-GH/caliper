"""A training script for `tests/hub/e2e.py --connect`: the model and views of
tests/hub/producer.py, streamed to the hub by the package itself, with
caliper.connect(), caliper.show() and caliper.series().

It connects with no argument, so it finds the hub through the hub file
(CALIPER_HUB_FILE, which e2e.py sets). Its session id is the one the package
makes, clp-<pid>-<6 hex>; e2e.py finds it in the session list by its pid.

    CALIPER_HUB_FILE=<hub.json> python tests/hub/producer_connect.py --seconds 30

With --controls it takes the controls of protocol 3.1 through the package, as
tests/hub/producer.py --controls does by hand: caliper.param("lr") is the
optimiser's learning rate, caliper.action("save") counts a save and answers
"saved at step N", and caliper.step() is the checkpoint of each step.

When it exits it prints one JSON line: {"id", "pid", "steps", "loss_values",
"state", "lr", "saves"}.
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
    ap.add_argument("--controls", action="store_true", help="take the controls of protocol 3.1")
    a = ap.parse_args()
    import torch
    steps = 0
    saves = 0
    lr = None
    if a.controls:                               # registered before connecting: the first
        lr = caliper.param("lr", 1e-3, min=1e-5, max=0.1, log=True,   # CONTROLS has them
                           help="the learning rate")

        @caliper.action("save", help="save a checkpoint")
        def save():
            nonlocal saves
            saves += 1
            return f"saved at step {steps}"
    hub = caliper.connect()                      # the hub file, as a script would
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(os.getpid() % 1000)
    x = torch.randn(4096, 32, device=device)
    y = (x[:, :4].sum(dim=1) > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    end = time.monotonic() + a.seconds
    while time.monotonic() < end:
        if lr is not None:
            caliper.step(steps)
            for group in opt.param_groups:
                group["lr"] = lr.value
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
                      "state": state, "lr": lr.value if lr is not None else None,
                      "saves": saves}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
