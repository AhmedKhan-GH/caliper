"""The ablation of intervention points through module hooks (``design/gpu-programs.md``, Interventions).

``TorchAdapter.points`` and ``TorchAdapter.ablate`` in ``caliper._adapter`` call these two
functions, which are the only torch code of the interventions. ``caliper._controls`` applies a
CONTROL of op ``intervene`` at the next step boundary through them, and sends the points and the
interventions in the key ``interventions`` of CONTROLS. Both raise ``NotImplementedError`` until
the interventions are built.
"""
from __future__ import annotations

__all__ = ["points", "ablate", "HEAD_SEPARATOR"]

#: The separator of a head's point: ``<module>#<h>``.
HEAD_SEPARATOR = "#"


def points(model) -> list:
    """List the intervention points of ``model``; see ``TorchAdapter.points``."""
    raise NotImplementedError("the intervention points are not built yet (design/gpu-programs.md)")


def ablate(model, point: str):
    """Ablate one point of ``model`` and return the function that removes it; see
    ``TorchAdapter.ablate``."""
    raise NotImplementedError("the ablation is not built yet (design/gpu-programs.md)")
