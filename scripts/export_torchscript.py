#!/usr/bin/env python3
"""Export RepNetCrossLeadDeeper state_dict to TorchScript for C++ inference.

Infers architecture parameters (filters, kernels, attention stages) directly
from the state_dict tensor shapes so no manual configuration is needed.

Usage:
    python scripts/export_torchscript.py
    python scripts/export_torchscript.py --input data/best_model.pt --output data/best_model_scripted.pt
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Export RepNetCrossLeadDeeper to TorchScript")
    parser.add_argument("--input", default="data/best_model.pt",
                        help="Path to state_dict checkpoint")
    parser.add_argument("--output", default="data/best_model_scripted.pt",
                        help="Output TorchScript path")
    parser.add_argument("--seq-len", type=int, default=2500,
                        help="Trace sequence length (default: 2500)")
    args = parser.parse_args()

    import torch

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")

    init_file = os.path.join(data_dir, "__init__.py")
    created_init = not os.path.exists(init_file)
    if created_init:
        open(init_file, "w").close()

    sys.path.insert(0, project_root)

    try:
        from data.repnet_crosslead_deeper import RepNetCrossLeadDeeper

        checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)

        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

        stage_indices = sorted({
            int(k.split(".")[1])
            for k in checkpoint
            if k.startswith("stages.")
        })
        n_stages = len(stage_indices)

        filters, kernels, attn_stages = [], [], []
        for i in stage_indices:
            w = checkpoint[f"stages.{i}.conv.conv1.weight"]
            filters.append(w.shape[0])
            kernels.append(w.shape[2])
            attn_stages.append(
                any(k.startswith(f"stages.{i}.attn.") for k in checkpoint))

        n_classes = checkpoint["fc.weight"].shape[0]

        print(f"Detected architecture:")
        print(f"  Stages:    {n_stages}")
        print(f"  Filters:   {tuple(filters)}")
        print(f"  Kernels:   {tuple(kernels)}")
        print(f"  Attention: {tuple(attn_stages)}")
        print(f"  Classes:   {n_classes}")

        model = RepNetCrossLeadDeeper(
            stage_filters=tuple(filters),
            kernels=tuple(kernels),
            attn_stages=tuple(attn_stages),
            n_classes=n_classes,
        )
        model.load_state_dict(checkpoint)
        model.eval()

        dummy = torch.randn(1, 12, args.seq_len)
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)

        traced.save(args.output)
        print(f"\nSaved TorchScript model to: {args.output}")
        print(f"  Input shape:  (1, 12, {args.seq_len})")

        with torch.no_grad():
            diff = (model(dummy) - traced(dummy)).abs().max().item()
            out = traced(dummy)
            print(f"  Output shape: {tuple(out.shape)}")
            print(f"  Max diff:     {diff:.2e}")

    finally:
        if created_init and os.path.exists(init_file):
            os.remove(init_file)


if __name__ == "__main__":
    main()
