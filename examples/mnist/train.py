"""Train the standard PyTorch MNIST CNN and watch it live in Caliper.

    python examples/mnist/train.py [--mode window|app|attach|none]
                                   [--steps N] [--device auto|mps|cpu]

Model, optimizer, learning-rate schedule, and normalization follow the official
PyTorch MNIST example:
https://github.com/pytorch/examples/blob/main/mnist/main.py

Caliper views published during training:
  loss              training NLL loss curve (from caliper.watch)
  lr                learning rate curve and live slider (from caliper.watch)
  accuracy          training batch accuracy curve (caliper.series)
  conv1.filters     32 first-layer 3x3 filters as a mosaic (caliper.show)
  conv1.act         32 conv1 feature maps on one fixed test digit (caliper.show)
  embedding         128-d hidden layer projected to 3-D, colored by digit (caliper.points)
  fc1.weight.hist   distribution of fc1 weights (caliper.histogram)

Sections marked with `# --8<-- [start:<name>]` are included verbatim in the
documentation through pymdownx.snippets.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# --8<-- [start:imports]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import caliper
# --8<-- [end:imports]

# Standard PyTorch MNIST normalization constants (pytorch/examples/mnist/main.py).
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEFAULT_DATA_DIR = Path(
    os.environ.get("CALIPER_MNIST_DATA_DIR", str(Path.home() / ".cache" / "caliper" / "mnist"))
).expanduser()


# --8<-- [start:model]
# Standard PyTorch MNIST CNN from pytorch/examples/mnist/main.py:
# https://github.com/pytorch/examples/blob/main/mnist/main.py
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# --8<-- [end:model]


# --8<-- [start:data]
def load_mnist(
    data_dir: Path,
    batch_size: int,
    test_batch_size: int,
) -> tuple[DataLoader, DataLoader, datasets.MNIST]:
    """Download MNIST into ``data_dir`` if absent and return train/test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    data_dir = data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader, test_set
# --8<-- [end:data]


def make_probe(
    test_set: datasets.MNIST,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a fixed test batch (item 0 is the fixed digit) and an orthonormal (128, 3) projection."""
    n = min(count, len(test_set))
    images, labels = zip(*(test_set[i] for i in range(n)))
    probe_x = torch.stack(images).to(device)
    probe_y = torch.tensor(labels, dtype=torch.long, device=device)
    gen = torch.Generator().manual_seed(0)
    proj, _ = torch.linalg.qr(torch.randn(128, 3, generator=gen))
    return probe_x, probe_y, proj.to(device)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Compute average NLL loss and accuracy on the full MNIST test split."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    if was_training:
        model.train()
    return total_loss / max(total, 1), correct / max(total, 1)


def wait_for_app(timeout: float) -> bool:
    """Wait until a viewer attaches to the active Caliper stream server."""
    server = caliper._server
    if server is None:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(server.transport, "n_attached", 0) > 0:
            return True
        time.sleep(0.05)
    return False


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    train_loader: DataLoader,
    test_loader: DataLoader,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
    proj: torch.Tensor,
    device: torch.device,
    *,
    epochs: int,
    max_steps: int | None,
    probe_every: int,
    pace: float,
) -> dict[str, float | int]:
    """Run the MNIST training loop and publish live Caliper views."""
    probe_acts: dict[str, torch.Tensor] = {}
    capturing = False

    def save_conv1(_mod, _inp, out):
        if capturing:
            probe_acts["conv1"] = F.relu(out[0].detach())

    def save_hidden(_mod, _inp, out):
        if capturing:
            probe_acts["hidden"] = F.relu(out.detach())

    h_conv1 = model.conv1.register_forward_hook(save_conv1)
    h_fc1 = model.fc1.register_forward_hook(save_hidden)

    def publish_probe() -> None:
        nonlocal capturing
        was_training = model.training
        model.eval()
        capturing = True
        try:
            with torch.no_grad():
                model(probe_x)
        finally:
            capturing = False
            if was_training:
                model.train()
        caliper.show(probe_acts["conv1"], "conv1.act")
        caliper.points(probe_acts["hidden"] @ proj, probe_y, "embedding", classes=10)

    step = 0
    last_loss = float("nan")
    t0 = time.perf_counter()
    model.train()
    try:
        for epoch in range(1, epochs + 1):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                # --8<-- [start:train-step]
                output = model(data)
                loss = F.nll_loss(output, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                step += 1
                # --8<-- [start:views]
                accuracy = (output.argmax(dim=1) == target).float().mean()
                caliper.series(accuracy, "accuracy")
                caliper.show(
                    model.conv1.weight.squeeze(1),
                    "conv1.filters",
                    cmap="rdbu",
                    dims=("filter", "height", "width"),
                    data_view={"tile": "filter"},
                )
                caliper.histogram(model.fc1.weight, "fc1.weight.hist")
                if step == 1 or step % probe_every == 0 or (max_steps is not None and step == max_steps):
                    publish_probe()
                # --8<-- [end:views]
                # --8<-- [end:train-step]

                if pace > 0:
                    time.sleep(pace)
                if max_steps is not None and step >= max_steps:
                    break

            last_loss = loss.item()
            if max_steps is not None and step >= max_steps:
                break
            scheduler.step()
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(
                f"epoch {epoch}/{epochs}  step {step}  train_loss {last_loss:.4f}  "
                f"test_loss {test_loss:.4f}  test_acc {100.0 * test_acc:.2f}%",
                flush=True,
            )
    finally:
        h_conv1.remove()
        h_fc1.remove()

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    test_loss, test_acc = evaluate(model, test_loader, device)
    return {
        "steps": step,
        "elapsed_s": elapsed,
        "train_loss": last_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def build_parser() -> argparse.ArgumentParser:
    default_mode = "none" if os.environ.get("CALIPER_RUN_DIR") else "window"
    ap = argparse.ArgumentParser(description="Train the PyTorch MNIST CNN with live Caliper views.")
    ap.add_argument("--mode", choices=("none", "app", "attach", "window"), default=default_mode)
    ap.add_argument("--steps", type=int, default=None,
                    help="stop after this many optimizer steps (default: full epochs)")
    ap.add_argument("--device", default="auto", help="compute device: auto, mps, cuda, or cpu")
    ap.add_argument("--epochs", type=int, default=14,
                    help="number of epochs when --steps is not set (PyTorch default: 14)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--test-batch-size", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1.0, help="Adadelta learning rate (PyTorch default: 1.0)")
    ap.add_argument("--gamma", type=float, default=0.7, help="StepLR decay per epoch (PyTorch default: 0.7)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="cache folder for the MNIST dataset")
    ap.add_argument("--probe-every", type=int, default=10,
                    help="steps between fixed-digit activation and embedding updates")
    ap.add_argument("--probe-size", type=int, default=512,
                    help="number of fixed test digits in the hidden-layer point cloud")
    ap.add_argument("--linger", type=float, default=3.0,
                    help="window mode: seconds to keep the window open after training")
    ap.add_argument("--wait", type=float, default=120.0,
                    help="app mode: seconds to wait for the Caliper app to attach")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="optional pause in seconds after each step")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps is not None and args.steps < 1:
        print("--steps must be >= 1", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device {device}, torch {torch.__version__}, mode {args.mode}", flush=True)

    train_loader, test_loader, test_set = load_mnist(
        args.data_dir, args.batch_size, args.test_batch_size
    )
    probe_x, probe_y, proj = make_probe(test_set, args.probe_size, device)

    # --8<-- [start:setup]
    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # --8<-- [start:watch]
    watch = caliper.watch(model, optimizer, activations=False, groups=False)
    # --8<-- [end:watch]
    # --8<-- [end:setup]

    run_kwargs = dict(
        epochs=args.epochs,
        max_steps=args.steps,
        probe_every=max(1, args.probe_every),
        pace=args.pace,
    )

    try:
        # --8<-- [start:modes]
        if args.mode == "attach":
            try:
                from caliper.live import SessionTransport  # noqa: F401
            except ImportError as exc:
                print(f"attach mode needs caliper.live.SessionTransport: {exc}", file=sys.stderr)
                return 2
            sess = caliper.session(auto_launch=True)
            print(f"session {sess.session}: waiting for caliper-viewer to attach", flush=True)
            if not sess.wait_attached(15.0):
                print("caliper-viewer did not attach within 15 s", file=sys.stderr)
                return 3
            result = train(model, optimizer, scheduler, train_loader, test_loader,
                           probe_x, probe_y, proj, device, **run_kwargs)
        elif args.mode == "app":
            server = caliper._server
            print(
                f"stream {server.session} (pid {os.getpid()}): in Caliper, press F4 "
                f"(Attach to process) and open this script's row, or open {server.url}",
                flush=True,
            )
            if not wait_for_app(args.wait):
                print(f"Caliper did not attach within {args.wait:.0f} s", file=sys.stderr)
                return 3
            result = train(model, optimizer, scheduler, train_loader, test_loader,
                           probe_x, probe_y, proj, device, **run_kwargs)
        elif args.mode == "window":
            caliper.window(1280, 800, title="caliper: mnist")
            result = caliper.run(
                train, model, optimizer, scheduler, train_loader, test_loader,
                probe_x, probe_y, proj, device, **run_kwargs,
            )
            if args.linger > 0:
                print(f"window open for {args.linger:.0f} s (close it to exit early)", flush=True)
                caliper.spin(args.linger)
        else:
            result = train(model, optimizer, scheduler, train_loader, test_loader,
                           probe_x, probe_y, proj, device, **run_kwargs)
        # --8<-- [end:modes]
    finally:
        watch.remove()
        caliper.close()

    rate = result["steps"] / max(result["elapsed_s"], 1e-9)
    print(
        f"done  steps={result['steps']}  elapsed={result['elapsed_s']:.2f}s  "
        f"rate={rate:.1f} steps/s  train_loss={result['train_loss']:.4f}  "
        f"test_loss={result['test_loss']:.4f}  test_acc={100.0 * result['test_acc']:.2f}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
