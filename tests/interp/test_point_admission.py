"""Semantic probe priority at the bounded controls advertisement."""
from types import SimpleNamespace

import pytest
import torch

from caliper import _wire
from caliper._adapter import TORCH
from caliper._controls import Controls, MAX_POINTS_LISTED
from caliper.interp._points import semantic_points

from .fixture import TinyLM
from .test_points import hooks


@pytest.mark.parametrize("excess", [-1, 0, 1, 16])
def test_semantic_points_survive_controls_admission_and_refresh(monkeypatch, excess):
    model = TinyLM(layers=1, width=8, heads=2, vocab=16, latents=16, k=2)
    original = TORCH.points(model)
    probes = semantic_points(model)
    ordinary = [row for row in original if "probe" not in row]
    modules = list(model.named_modules())
    # Metadata aliases reach the wire limit without a large module tree or tensors.
    placeholder = torch.nn.Identity()
    extra = [(f"synthetic.{i}", placeholder)
             for i in range(MAX_POINTS_LISTED + excess - len(original))]

    def point_rows(model):
        with monkeypatch.context() as patch:
            patch.setattr(model, "named_modules", lambda: iter(modules + extra))
            return TORCH.points(model)

    rows = point_rows(model)
    assert len(rows) == MAX_POINTS_LISTED + excess
    assert rows == point_rows(model)
    # Existing module and head rows retain their fields and relative order.
    assert [row for row in rows if "probe" not in row][:len(ordinary)] == ordinary
    assert {row["point"] for row in ordinary if row["kind"] == "head"} == {
        "layers.0.self_attn#0", "layers.0.self_attn#1"}
    controls = Controls()
    adapter = SimpleNamespace(points=point_rows, ablate=TORCH.ablate)
    handle = controls.interventions_on(model, adapter=adapter)

    def advertised():
        message = _wire.Controls.from_frame(
            _wire.Frame(_wire.MsgType.CONTROLS, controls.controls_meta()))
        decoded = _wire.decode(_wire.encode(message))
        listed = decoded.to_meta()["interventions"]["points"]
        assert len(listed) == min(MAX_POINTS_LISTED, len(rows))
        return listed

    try:
        listed = advertised()
        names = {row["point"] for row in listed}
        assert {row["point"] for row in probes} <= names
        assert listed == rows[:MAX_POINTS_LISTED]
        assert listed[:len(probes)] == probes
        assert handle.names == frozenset(row["point"] for row in rows)
        # Native validation reads these advertised names before it sends a command.
        assert {"layers.0.self_attn", "layers.0.self_attn#1", "layers.0.mlp_out"} <= names
        model(torch.tensor([[1, 2]]))
        controls.step()
        refreshed = advertised()
        assert [row["point"] for row in refreshed] == [row["point"] for row in listed]
        assert next(row for row in refreshed if row["point"] == "layers.0.mlp")["shape"] == [1, 2, 8]
        assert refreshed[:len(probes)] == probes
        with controls.lock:
            assert controls._intervene({"set": {"point": "layers.0.mlp_out", "op": "ablate"}}) is None
            controls._update_busy()
        controls.step()
        assert model.layers[0].mlp(torch.ones(1, 2, 8, dtype=torch.float64)).count_nonzero() == 0
    finally:
        handle.remove()
    assert hooks(model) == 0
