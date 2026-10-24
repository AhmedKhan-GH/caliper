"""Train a tiny CNN and watch it live in Caliper.

    .venv/bin/python examples/python/train_show.py [--mode window|attach|none]
                                                   [--seconds 20] [--cpu]

Synthetic data (no downloads): 28x28 images with one Gaussian blob whose
quadrant is the label. Every training step makes four caliper calls:

    caliper.show(model.fc1.weight, "fc1.weight")        heatmap (784 x 64)
    caliper.show(conv1_out, "conv1.act")                channel mosaic (forward hook)
    caliper.points(hidden @ P, labels=y, classes=4)     3-D embedding of the 64-d hidden
                                                        layer (fixed random projection P,
                                                        on the device, no sklearn)
    caliper.series(loss, "loss")                        loss curve (every step)

Each call returns at once: the snapshot is a few queued torch ops into a slot on
the tensor's device (at most 60 per second per view), nothing reads back.

Modes:
  window  caliper.window() + caliper.run(train): the in-process window, training on a
          worker thread so the main thread keeps the window interactive.
  attach  caliper.session(auto_launch=True): the live session and a caliper-viewer
          process attached to it (the "attach to app" route).
  none    no explicit transport: the first call opens the session and nothing attaches,
          so every call is the near-free unattached path.

Prints steps/s without caliper, with caliper and (MPS) with only a 60 Hz
torch.mps.Event.record (the commit effect: recording an event commits torch's
command buffer, which by itself changes this small model's throughput), and the
per-call cost of every view, split into calls that took a snapshot and calls
that were skipped (rate limit / nothing attached).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import caliper


class TinyCNN(nn.Module):
    def __init__(self, classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def make_batch(n: int, device: torch.device):
    labels = torch.randint(0, 4, (n,), device=device)
    cy = 7.0 + 14.0 * (labels // 2).float() + torch.randn(n, device=device) * 2
    cx = 7.0 + 14.0 * (labels % 2).float() + torch.randn(n, device=device) * 2
    g = torch.arange(28, device=device, dtype=torch.float32)
    yy = (g[None, :, None] - cy[:, None, None]) ** 2
    xx = (g[None, None, :] - cx[:, None, None]) ** 2
    img = torch.exp(-(yy + xx) / 18.0) + 0.15 * torch.randn(n, 28, 28, device=device)
    return img[:, None], labels


class Trainer:
    def __init__(self, device: torch.device):
        torch.manual_seed(0)
        self.device = device
        self.model = TinyCNN().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.acts: dict[str, torch.Tensor] = {}
        self.model.conv1.register_forward_hook(lambda m, i, o: self.acts.__setitem__("conv1", o))
        self.model.fc1.register_forward_hook(lambda m, i, o: self.acts.__setitem__("hidden", o))
        g = torch.Generator().manual_seed(1)
        self.proj = (torch.randn(64, 3, generator=g) / 8.0).to(device)   # fixed (64, 3)
        self.costs: dict[str, dict[bool, list]] = {}

    def _call(self, view: str, fn, *args, **kw):
        a = time.perf_counter_ns()
        took = fn(*args, **kw)
        self.costs.setdefault(view, {True: [], False: []})[bool(took)].append(
            time.perf_counter_ns() - a)

    def phase(self, seconds: float, mode: str) -> tuple[int, float, float]:
        """Train for `seconds`. mode: "off" | "caliper" | "commit" (60 Hz Event.record
        only). Returns (steps, elapsed s incl. the final device sync, last loss)."""
        steps, loss = 0, None
        ev = torch.mps.Event() if self.device.type == "mps" else None
        last_rec = 0.0
        t0 = time.perf_counter()
        t_end = t0 + seconds
        while time.perf_counter() < t_end:
            x, y = make_batch(64, self.device)
            loss = F.cross_entropy(self.model(x), y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            steps += 1
            if mode == "caliper":
                self._call("fc1.weight", caliper.show, self.model.fc1.weight, "fc1.weight",
                           cmap="rdbu")
                self._call("conv1.act", caliper.show, self.acts["conv1"], "conv1.act")
                self._call("embedding", caliper.points, self.acts["hidden"] @ self.proj, y,
                           "embedding", classes=4)
                self._call("loss", caliper.series, loss.detach(), "loss")
            elif mode == "commit" and ev is not None:
                now = time.perf_counter()
                if now - last_rec >= 1.0 / 60.0:
                    last_rec = now
                    ev.record()
        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        return steps, elapsed, loss.item() if loss is not None else float("nan")


def run_phases(tr: Trainer, args) -> list:
    rows = []
    tr.phase(0.5, "off")                                   # warm-up: kernels, allocator
    tr.phase(0.5, "caliper")                               # warm-up: views, shaders
    tr.costs.clear()
    print(f"baseline, no caliper ({args.baseline:.0f} s)")
    rows.append(("no caliper", *tr.phase(args.baseline, "off")))
    if tr.device.type == "mps":
        print(f"control, 60 Hz Event.record only ({args.baseline:.0f} s)")
        rows.append(("60 Hz Event.record", *tr.phase(args.baseline, "commit")))
    print(f"with caliper, 4 calls per step ({args.seconds:.0f} s)")
    rows.append(("with caliper", *tr.phase(args.seconds, "caliper")))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("window", "attach", "none"), default="window")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--baseline", type=float, default=4.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--linger", type=float, default=3.0,
                    help="window mode: keep the window open this long after training")
    args = ap.parse_args()

    use_mps = not args.cpu and torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    print(f"device {device}, torch {torch.__version__}, mode {args.mode}")
    tr = Trainer(device)

    if args.mode == "attach":
        try:
            from caliper.live import SessionTransport  # noqa: F401
        except ImportError as e:
            print(f"attach mode needs the live session transport (caliper.live.SessionTransport): "
                  f"{e}", file=sys.stderr)
            return 2
        sess = caliper.session(auto_launch=True)
        print(f"session {sess.session}: waiting for the viewer to attach")
        if not sess.wait_attached(15.0):
            print("the viewer did not attach within 15 s", file=sys.stderr)
            return 3
        rows = run_phases(tr, args)
    elif args.mode == "window":
        caliper.window(1280, 800, title="caliper: train_show")
        rows = caliper.run(run_phases, tr, args)       # training on a worker thread
    else:
        rows = run_phases(tr, args)

    base = rows[0][1] / rows[0][2]
    print()
    for label, steps, el, loss in rows:
        rate = steps / el
        print(f"  {label:20s} {rate:8.1f} steps/s  ({100 * rate / base:5.1f} % of no caliper, "
              f"loss {loss:.4f})")
    print("\nper-call cost, median us (p95) [calls]:")
    for view, d in tr.costs.items():
        def fmt(xs):
            if not xs:
                return "       -        "
            xs = sorted(xs)
            return (f"{statistics.median(xs) / 1e3:7.1f} ({xs[int(0.95 * (len(xs) - 1))] / 1e3:6.1f}) "
                    f"[{len(xs)}]")
        print(f"  {view:11s} snapshot {fmt(d[True])}   skipped {fmt(d[False])}")
    st = caliper.stats()
    for route, d in st.items():
        ring = d.get("ring", {})
        line = (f"\n{route}: snapshots {ring.get('taken', 0)}, published {ring.get('published', 0)}, "
                f"ring full {ring.get('ring_full', 0)}, series values {ring.get('appended', 0)}")
        if "frames" in d:
            lat = sorted(d.get("latency_ms") or [0.0])
            line += (f"; render thread {d['frames']} frames, drawn {d['drawn']}, errors "
                     f"{d['errors']}, snapshot-to-frame latency median {lat[len(lat) // 2]:.1f} ms, "
                     f"max frame gap {d['max_frame_gap_ms']:.1f} ms")
        print(line)
        if d.get("errors"):
            print(f"  last error: {d['last_error']}")
    if args.mode == "window" and caliper._native_core() is not None:
        for v in ("fc1.weight", "conv1.act"):
            i = caliper.view_info(v)
            print(f"  {v:11s} {i['width']}x{i['height']} {i['path']} imported={i['imported']} "
                  f"updates={i['updates']}")
        for v in ("embedding", "loss"):
            i = caliper.live_view_info(v)
            print(f"  {v:11s} {i['path']} count={i['count']} updates={i['updates']} "
                  f"latency {i['latency_ms']:.1f} ms")
        if args.linger > 0:
            print(f"window open for {args.linger:.0f} s (close it to exit early)")
            caliper.spin(args.linger)
    elif args.mode == "none":
        print(f"\nsession opened implicitly: {caliper._session_route is not None}, "
              f"attached: {bool(caliper._session_route and caliper._session_route.transport.attached())}")
    caliper.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
