"""Circuit Trace spec builder with published Dallas one-click controls."""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent / "published_dallas_v1.json"


def load_dallas_manifest() -> dict:
    """Load the static published Dallas prompt, feature group, and intervention manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def group_text(manifest: dict, index: int) -> str:
    """Format one published feature group's `(layer, feature, position)` triples."""
    name, nodes = list(manifest["groups"].items())[index]
    return name + ": " + "; ".join(f"L{n['layer']} F{n['feature']} P{n['position']}" for n in nodes)


def build_instrument(session, *, dallas: bool = False) -> dict | None:
    """Create the three-panel Circuit Trace spec and optional Dallas one-click buttons."""
    m = session.mcp
    manifest = load_dallas_manifest() if dallas else None
    title = "Circuit Trace — Published Dallas" if dallas else "Circuit Trace"
    desc = (
        "Live attribution graph, feature examples, and paired next-token probabilities "
        "powered by TransformerLens, circuit-tracer, and SAELens."
    )
    m.body("spec.new", id="circuit-trace", title=title, description=desc)
    m.body("state.create", id="selected_node", type="integer", default=-1, min=-1, max=8191)
    m.body("state.create", id="example_page", type="integer", default=0, min=0, max=1)
    m.body("state.create", id="activation_max", type="number", default=1.0, min=1e-6, max=1e20)

    for panel_id, panel_title in (
        ("graph", "Attribution Graph"),
        ("examples", "Feature Examples"),
        ("output", "Output Probabilities"),
    ):
        empty = "No feature is selected." if panel_id == "examples" else "The trace is not available."
        m.body("panel.create", id=panel_id, primitive="text", title=panel_title, options={"text": empty})

    controls = [
        ("prompt", "text", "param:prompt", "Prompt", {}),
        ("clamp", "slider", "param:clamp", "Feature Clamp", {"min": -4, "max": 4}),
        ("apply_clamp", "button", "action:apply_clamp", "Apply Clamp", {}),
        ("reset", "button", "action:reset", "Feature Reset", {}),
        ("generate", "button", "action:generate", "Token Generation", {}),
        ("page", "segmented", "state:example_page", "Example Page",
         {"choices": [{"value": 0, "label": "1-8"}, {"value": 1, "label": "9-16"}]}),
        ("baseline_text", "text", "param:output.baseline", "Baseline Tokens", {}),
        ("steered_text", "text", "param:output.steered", "Steered Tokens", {}),
    ]
    for cid, kind, bind, ctitle, opts in controls:
        m.body("control.create", id=cid, kind=kind, bind=bind, title=ctitle, options=opts)

    m.body("control.create", id="clamp_status_value", kind="text", bind="param:clamp.status", title="Clamp Status")
    m.body(
        "control.create",
        id="clamp_status",
        kind="readout",
        title="Clamp Status",
        options={"format": "{status:s}", "fields": {"status": {"control": "clamp_status_value"}}},
    )

    control_ids = ["clamp", "apply_clamp", "reset", "generate", "page", "baseline_text", "steered_text"]
    footer_ids = []
    if dallas and manifest is not None:
        m.body("state.create", id="published_group", type="integer", default=0, min=0, max=len(manifest["groups"]) - 1)
        m.body("state.create", id="published_members", type="text", default=group_text(manifest, 0))
        m.body("control.create", id="published_members_val", kind="text", bind="state:published_members", title="Group Coordinates")
        m.body(
            "control.create",
            id="published_members",
            kind="readout",
            title="Group Coordinates",
            options={"format": "{text:s}", "fields": {"text": {"control": "published_members_val"}}},
        )
        m.body(
            "control.create",
            id="published_group",
            kind="picker",
            bind="state:published_group",
            title="Published Feature Groups",
            options={"choices": [{"value": i, "label": name} for i, name in enumerate(manifest["groups"])]},
        )
        cell_buttons = []
        for case in manifest["interventions"]:
            cell = int(case["cell"])
            cid = f"cell{cell}"
            m.body("control.create", id=cid, kind="button", bind=f"action:dallas.cell{cell}", title=case["label"])
            cell_buttons.append(cid)
        footer_ids = ["published_group", "published_members"]
    else:
        cell_buttons = []

    nodes = [
        {"id": "main", "split": "vertical", "ratio": 0.11, "children": ["header", "body"]},
        {"id": "header", "split": "vertical", "children": ["prompt", "clamp_status"]},
        {"id": "body", "split": "vertical", "ratio": 0.68, "children": ["plots", "examples"]},
        {"id": "plots", "split": "horizontal", "ratio": 0.66, "children": ["graph", "right"]},
        {"id": "right", "split": "vertical", "ratio": 0.46, "children": ["output", "controls"]},
    ]
    if cell_buttons:
        nodes.extend([
            {"id": "controls", "split": "horizontal", "ratio": 0.52, "children": ["cells_col", "live_col"]},
            {"id": "cells_col", "split": "vertical", "children": cell_buttons},
            {"id": "live_col", "split": "vertical", "children": control_ids},
            {"id": "root_split", "split": "vertical", "ratio": 0.91, "children": ["main", "footer"]},
            {"id": "footer", "split": "horizontal", "children": footer_ids},
        ])
        root_id = "root_split"
    else:
        nodes.append({"id": "controls", "split": "vertical", "children": control_ids})
        root_id = "main"

    m.body("layout.set", layout={"root": root_id, "nodes": nodes})
    return manifest
