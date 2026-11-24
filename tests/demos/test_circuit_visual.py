"""Independent failures for the circuit visual acceptance gates."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[2] / "demos/circuit_trace"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
from _visual import visual_checks


def scene(name):
    panels = {key: {"composition": {"rect": rect}} for key, rect in
              (("graph", [0, 0, 200, 180]), ("examples", [200, 0, 300, 90]), ("output", [200, 90, 300, 180]))}
    metadata = {"origin": [0, 0], "pixel_scale": [1, 1], "width": 300, "height": 200, "panels": panels}
    panel = panels[name]
    panel.update(programs=[{"pick_id": 1}], clip_from_world=np.eye(4).reshape(-1).tolist(),
                 labels=[{"text": "France", "box": [1, 1, 10, 5]}, {"text": "Belgium", "box": [12, 1, 22, 5]}])
    ids = np.zeros((100, 100, 2), dtype=np.uint32)
    if name == "graph":
        for index, (x, y) in enumerate(((10, 10), (85, 85), (40, 40))):
            ids[y:y + 5, x:x + 5] = [1, index]
    elif name == "examples":
        for row in range(8):
            ids[10 + row * 10:18 + row * 10, 10:90] = [1, row]
    else:
        for bar in range(10):
            ids[20:90, 5 + bar * 9:11 + bar * 9] = [1, bar]
    return panel, ids, metadata


@pytest.mark.parametrize("name", ["graph", "examples", "output"])
def test_readable_visual_layout_passes(name):
    panel, ids, metadata = scene(name)
    gates = visual_checks(name, panel, ids, metadata, node_count=3)
    assert all(gate["passed"] for gate in gates), gates


@pytest.mark.parametrize("name", ["graph", "examples", "output"])
@pytest.mark.parametrize("defect", ["perspective", "tilt", "overlap", "raw_token"])
def test_visual_gates_reject_projection_and_text_defects(name, defect):
    panel, ids, metadata = scene(name)
    if defect == "perspective":
        panel["clip_from_world"][15] = 5
    elif defect == "tilt":
        panel["clip_from_world"][1] = .5
    elif defect == "overlap":
        panel["labels"][1]["box"] = deepcopy(panel["labels"][0]["box"])
    else:
        panel["labels"][0]["text"] = "ĠFrance"
    gates = visual_checks(name, panel, ids, metadata, node_count=3)
    assert any(not gate["passed"] for gate in gates)


@pytest.mark.parametrize("name", ["graph", "examples", "output"])
def test_visual_gates_reject_small_geometry_cluster(name):
    panel, ids, metadata = scene(name)
    small = np.zeros_like(ids)
    small[40:60, 40:60] = ids[::5, ::5]
    gates = visual_checks(name, panel, small, metadata, node_count=3)
    expected = {"graph": "graph_framing", "examples": "example_row_layout", "output": "output_bar_layout"}[name]
    assert not next(gate for gate in gates if gate["name"] == expected)["passed"]


def test_visual_gates_reject_large_disclosure_footer():
    panel, ids, metadata = scene("graph")
    metadata["height"] = 250
    gates = visual_checks("graph", panel, ids, metadata, node_count=3)
    assert not next(gate for gate in gates if gate["name"] == "compact_disclosures")["passed"]
