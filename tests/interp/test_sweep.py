"""Tests for patch_sweep recovery grid publication and validation."""
import pytest
import torch

from caliper import _controls
from caliper.interp import _publish
from caliper.interp._backend import make_tiny_backend
from caliper.interp._sweep import patch_sweep


def test_patch_sweep_publishes_grid_and_validates_inputs(monkeypatch):
    published = {}
    monkeypatch.setattr(_publish, "_TENSORS", published)
    backend = make_tiny_backend()
    ctl = _controls.Controls()
    ivs = ctl.interventions_on(backend.model)
    try:
        res = patch_sweep(
            backend,
            ivs,
            clean_prompt="ABCD",
            corrupted_prompt="EFGH",
            metric={"positive_token": 40, "negative_token": 41},
        )
        assert res["layers"] == 2 and res["positions"] == 5
        assert res["binding"] == "live:patch.grid"
        assert "patch.grid" in published
        assert published["patch.grid"].shape == (2, 5)
        assert torch.isfinite(published["patch.grid"]).all()

        with pytest.raises(ValueError, match="equal token counts"):
            patch_sweep(
                backend, ivs, clean_prompt="AB", corrupted_prompt="EFGH",
                metric={"positive_token": 40, "negative_token": 41},
            )
        with pytest.raises(ValueError, match="distinct vocabulary"):
            patch_sweep(
                backend, ivs, clean_prompt="ABCD", corrupted_prompt="EFGH",
                metric={"positive_token": 40, "negative_token": 40},
            )
    finally:
        ivs.remove()
