"""Activation patch recovery over Backend batched ``hook_resid_pre`` sweeps."""
from __future__ import annotations

from ._publish import _publication_scope


def patch_sweep(backend, interventions, *, clean_prompt: str, corrupted_prompt: str, metric: dict, guard=None) -> dict:
    """Patch each ``hook_resid_pre`` cell (up to 32 cells per batch) and publish ``patch.grid``."""
    if type(metric) is not dict or set(metric) != {"positive_token", "negative_token"}:
        raise ValueError("metric requires two distinct vocabulary token IDs")
    with _publication_scope(backend.model):
        res = backend.patch_sweep(
            clean_prompt,
            corrupted_prompt,
            positive_token=metric["positive_token"],
            negative_token=metric["negative_token"],
            batch_size=32,
            guard=guard,
        )
        from . import publish
        publish("patch.grid", res["grid"].to(backend.device))
        return {
            "clean_metric": res["clean_metric"],
            "corrupted_metric": res["corrupted_metric"],
            "normalizer": res["normalizer"],
            "recovery": res["recovery"],
            "layers": res["layers"],
            "positions": res["positions"],
            "binding": "live:patch.grid",
        }
