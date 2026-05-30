#!/usr/bin/env python3
"""Export PerLeadCNN state_dict to TorchScript for C++ inference.

Infers architecture parameters (filters, kernels, n_leads, n_classes) directly
from the state_dict tensor shapes so no manual configuration is needed.

Usage:
    python scripts/export_torchscript.py
    python scripts/export_torchscript.py --input multisplit_dbb6f49/best_model.pt --output multisplit_dbb6f49/best_model_scripted.pt
"""

import argparse
import torch
import torch.nn as nn


class PerLeadCNN(nn.Module):
    def __init__(self, n_leads=12, filters=(16, 32, 48), kernels=(31, 21, 11),
                 dropout=0.15, n_classes=2):
        super().__init__()
        layers = []
        in_ch = 1
        for f, k in zip(filters, kernels):
            layers.extend([
                nn.Conv1d(in_ch, f, k, stride=2, padding=k // 2, bias=False),
                nn.BatchNorm1d(f), nn.Mish(),
            ])
            in_ch = f
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.n_leads = n_leads
        self.head_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters[-1] * n_leads, n_classes)

    def forward(self, x):
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        x = x.reshape(B, L * x.shape[-1])
        return self.fc(self.head_drop(x))


def detect_architecture(state_dict):
    """Auto-detect PerLeadCNN architecture from state_dict keys.

    Conv weights live at backbone.{0,3,6,...}.weight with shape (out_ch, in_ch, kernel).
    BatchNorm params at backbone.{1,4,7,...}.  fc.weight shape gives n_classes and
    n_leads (via filters[-1] * n_leads == fc.weight.shape[1]).
    """
    filters = []
    kernels = []
    # Conv1d layers are at indices 0, 3, 6, ... (each stage is conv+bn+mish = 3 items)
    stage = 0
    while True:
        key = f"backbone.{stage * 3}.weight"
        if key not in state_dict:
            break
        w = state_dict[key]
        filters.append(w.shape[0])
        kernels.append(w.shape[2])
        stage += 1

    if not filters:
        raise RuntimeError("Could not detect any backbone conv stages in state_dict")

    fc_w = state_dict["fc.weight"]
    n_classes = fc_w.shape[0]
    fc_in = fc_w.shape[1]
    n_leads = fc_in // filters[-1]

    return {
        "filters": tuple(filters),
        "kernels": tuple(kernels),
        "n_leads": n_leads,
        "n_classes": n_classes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export PerLeadCNN to TorchScript")
    parser.add_argument("--input", default="multisplit_dbb6f49/best_model.pt",
                        help="Path to state_dict checkpoint")
    parser.add_argument("--output", default="multisplit_dbb6f49/best_model_scripted.pt",
                        help="Output TorchScript path")
    parser.add_argument("--seq-len", type=int, default=2500,
                        help="Trace sequence length (default: 2500)")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    arch = detect_architecture(checkpoint)

    print(f"Detected architecture:")
    print(f"  Filters:  {arch['filters']}")
    print(f"  Kernels:  {arch['kernels']}")
    print(f"  Leads:    {arch['n_leads']}")
    print(f"  Classes:  {arch['n_classes']}")

    model = PerLeadCNN(
        n_leads=arch["n_leads"],
        filters=arch["filters"],
        kernels=arch["kernels"],
        n_classes=arch["n_classes"],
    )
    model.load_state_dict(checkpoint)
    model.eval()

    dummy = torch.randn(1, arch["n_leads"], args.seq_len)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    traced.save(args.output)
    print(f"\nSaved TorchScript model to: {args.output}")
    print(f"  Input shape:  (1, {arch['n_leads']}, {args.seq_len})")

    with torch.no_grad():
        diff = (model(dummy) - traced(dummy)).abs().max().item()
        out = traced(dummy)
        print(f"  Output shape: {tuple(out.shape)}")
        print(f"  Max diff:     {diff:.2e}")


if __name__ == "__main__":
    main()
