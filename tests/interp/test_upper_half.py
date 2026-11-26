"""Synthetic metadata for legacy upper-half consumer checks."""
import torch

from .fixture import TinyLM
from .test_dictionary import fixture_manifest


def tiny_model(layers=4):
    return TinyLM(layers=layers, width=4, heads=2, latents=8, k=2, dtype=torch.float32)


def fallback_manifest(model):
    metadata = fixture_manifest(model)
    count = metadata["layers"]
    metadata["training"]["source_id"] = "synthetic-text-v1"
    metadata["training"]["fallback"] = {
        "version": 1, "mode": "upper_half",
        "active_layers": list(range(count // 2, count)), "zero_layers": list(range(count // 2)),
        "source_dictionary_hash": "a" * 64, "source_checkpoint_generation": "checkpoint-" + "b" * 32,
        "source_training_tokens": 16, "source_elapsed_seconds": 100.0, "prior_wall_seconds": 120.0,
        "aggregate_wall_limit_seconds": 36000.0, "token_budget": 16,
        "warm_start": "upper_weights_fresh_adam", "source_id": "synthetic-text-v1",
        "lower_behavior": "zero_dictionary_original_mlp",
    }
    return metadata
