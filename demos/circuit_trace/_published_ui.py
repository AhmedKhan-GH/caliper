"""Circuit Trace spec builder with real-time controls, read-only outputs, and Dallas preset."""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent / "published_dallas_v1.json"


def create_readout(
    mcp,
    *,
    id: str,
    bind: str,
    title: str,
    field: str = "text",
    value_id: str | None = None,
) -> str:
    """Create a read-only readout control backed by an unmounted value control."""
    backing_id = value_id or f"{id}_value"
    mcp.body("control.create", id=backing_id, kind="text", bind=bind, title=title)
    mcp.body(
        "control.create",
        id=id,
        kind="readout",
        title=title,
        options={"format": f"{{{field}:s}}", "fields": {field: {"control": backing_id}}},
    )
    return backing_id

DEFAULT_PROMPT = "He saw a carrot and had to grab it, his hunger was like a starving"
DALLAS_PROMPT = "Fact: the capital of the state containing Dallas is"

PROMPT_PRESETS: tuple[tuple[str, str, str], ...] = (
    ("default", "Default", DEFAULT_PROMPT),
    ("dallas", "Published Dallas", DALLAS_PROMPT),
)
CUSTOM_PRESET_INDEX = len(PROMPT_PRESETS)

SHOW_MORE_NODE_THRESHOLD = 0.78
SHOW_MORE_EDGE_THRESHOLD = 0.93
DEFAULT_MAX_NODES = 8192
DEFAULT_MAX_EDGES = 262144
DEFAULT_UPDATE_INTERVAL = 8


def load_dallas_manifest() -> dict:
    """Load the static published Dallas prompt, feature group, and intervention manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def published_trace_options(*, show_more: bool = False) -> dict:
    """Return trace.run pruning and generation thresholds from the published Dallas profile."""
    graph = load_dallas_manifest()["graph"]
    generation = graph["generation"]
    pruning = graph["pruning"]
    return {
        "max_targets": int(generation["max_feature_nodes"]),
        "max_nodes": DEFAULT_MAX_NODES,
        "max_edges": DEFAULT_MAX_EDGES,
        "max_n_logits": int(generation["max_n_logits"]),
        "desired_logit_prob": float(generation["desired_logit_prob"]),
        "node_threshold": SHOW_MORE_NODE_THRESHOLD if show_more else float(pruning["node_threshold"]),
        "edge_threshold": SHOW_MORE_EDGE_THRESHOLD if show_more else float(pruning["edge_threshold"]),
        "update_interval": DEFAULT_UPDATE_INTERVAL,
    }


def preset_index_for_prompt(prompt: str | None) -> int:
    """Map a prompt string to its preset picker index (or Custom)."""
    for idx, (_key, _label, text) in enumerate(PROMPT_PRESETS):
        if prompt == text:
            return idx
    return CUSTOM_PRESET_INDEX


def preset_prompt(index: int) -> str | None:
    """Return the prompt text for a preset picker index, or None for Custom."""
    if 0 <= index < len(PROMPT_PRESETS):
        return PROMPT_PRESETS[index][2]
    return None


def preset_prompt_by_name(name: str) -> str:
    """Return the prompt text for a named CLI preset."""
    for key, _label, text in PROMPT_PRESETS:
        if key == name:
            return text
    valid = ", ".join(key for key, _label, _text in PROMPT_PRESETS)
    raise ValueError(f"Unknown prompt preset {name!r}; expected one of: {valid}.")


def group_text(manifest: dict, index: int) -> str:
    """Format one published feature group's `(layer, feature, position)` triples."""
    name, nodes = list(manifest["groups"].items())[index]
    return name + ": " + "; ".join(f"L{n['layer']} F{n['feature']} P{n['position']}" for n in nodes)


def build_instrument(session, *, dallas: bool = False, prompt: str | None = None) -> dict | None:
    """Create the three-panel Circuit Trace spec with live controls and read-only outputs."""
    m = session.mcp
    manifest = load_dallas_manifest() if dallas else None
    title = "Circuit Trace — Published Dallas" if dallas else "Circuit Trace"
    desc = (
        "Live attribution graph, feature examples, and paired next-token probabilities "
        "powered by TransformerLens, circuit-tracer, and SAELens."
    )
    initial_preset = preset_index_for_prompt(DALLAS_PROMPT if dallas and prompt is None else (prompt if prompt is not None else DEFAULT_PROMPT))

    m.body("spec.new", id="circuit-trace", title=title, description=desc)
    m.body("state.create", id="selected_node", type="integer", default=-1, min=-1, max=8191)
    m.body("state.create", id="example_page", type="integer", default=0, min=0, max=1)
    m.body("state.create", id="activation_max", type="number", default=1.0, min=1e-6, max=1e20)
    m.body("state.create", id="prompt_preset", type="integer", default=initial_preset, min=0, max=CUSTOM_PRESET_INDEX)

    for panel_id, panel_title in (
        ("graph", "Attribution Graph"),
        ("examples", "Feature Examples"),
        ("output", "Output Probabilities"),
    ):
        empty = "No feature is selected." if panel_id == "examples" else "The trace is not available."
        m.body("panel.create", id=panel_id, primitive="text", title=panel_title, options={"text": empty})

    preset_choices = [
        {"value": idx, "label": label}
        for idx, (_key, label, _text) in enumerate(PROMPT_PRESETS)
    ]
    preset_choices.append({"value": CUSTOM_PRESET_INDEX, "label": "Custom"})

    controls = [
        ("prompt", "text", "param:prompt", "Prompt", {}),
        ("run", "button", "action:run", "Run", {}),
        ("preset", "picker", "state:prompt_preset", "Prompt Preset", {"choices": preset_choices}),
        ("show_more", "toggle", "param:show_more", "Show More Nodes", {}),
        ("clamp", "slider", "param:clamp", "Feature Clamp", {"min": -4, "max": 4}),
        ("apply_clamp", "button", "action:apply_clamp", "Apply Clamp", {}),
        ("reset", "button", "action:reset", "Feature Reset", {}),
        ("generate", "button", "action:generate", "Token Generation", {}),
        (
            "page",
            "segmented",
            "state:example_page",
            "Example Page",
            {"choices": [{"value": 0, "label": "1-8"}, {"value": 1, "label": "9-16"}]},
        ),
    ]
    for cid, kind, bind, ctitle, opts in controls:
        m.body("control.create", id=cid, kind=kind, bind=bind, title=ctitle, options=opts)

    create_readout(m, id="trace_status", bind="param:output.status", title="Status", field="status")
    create_readout(m, id="clamp_status", bind="param:clamp.status", title="Clamp Status", field="status")
    create_readout(m, id="baseline_text", bind="param:output.baseline", title="Baseline Output")
    create_readout(m, id="steered_text", bind="param:output.steered", title="Steered Output")

    control_rows = ["clamp", "clamp_actions", "page", "baseline_text", "steered_text"]
    footer_ids = []
    if dallas and manifest is not None:
        m.body("state.create", id="published_group", type="integer", default=0, min=0, max=len(manifest["groups"]) - 1)
        m.body("state.create", id="published_members", type="text", default=group_text(manifest, 0))
        create_readout(
            m,
            id="published_members",
            bind="state:published_members",
            title="Group Coordinates",
            value_id="published_members_val",
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
        {"id": "main", "split": "vertical", "ratio": 0.14, "children": ["header", "body"]},
        {"id": "header", "split": "vertical", "children": ["prompt_row", "status_row"]},
        {"id": "prompt_row", "split": "horizontal", "ratio": 0.60, "children": ["prompt", "prompt_tools"]},
        {"id": "prompt_tools", "split": "horizontal", "children": ["run", "preset", "show_more"]},
        {"id": "status_row", "split": "horizontal", "ratio": 0.25, "children": ["trace_status", "clamp_status"]},
        {"id": "body", "split": "vertical", "ratio": 0.68, "children": ["plots", "examples"]},
        {"id": "plots", "split": "horizontal", "ratio": 0.66, "children": ["graph", "right"]},
        {"id": "right", "split": "vertical", "ratio": 0.44, "children": ["output", "controls"]},
        {"id": "clamp_actions", "split": "horizontal", "children": ["apply_clamp", "reset", "generate"]},
    ]
    if cell_buttons:
        nodes.extend([
            {"id": "controls", "split": "horizontal", "ratio": 0.52, "children": ["cells_col", "live_col"]},
            {"id": "cells_col", "split": "vertical", "children": cell_buttons},
            {"id": "live_col", "split": "vertical", "children": control_rows},
            {"id": "root_split", "split": "vertical", "ratio": 0.91, "children": ["main", "footer"]},
            {"id": "footer", "split": "horizontal", "children": footer_ids},
        ])
        root_id = "root_split"
    else:
        nodes.append({"id": "controls", "split": "vertical", "children": control_rows})
        root_id = "main"

    m.body("layout.set", layout={"root": root_id, "nodes": nodes})
    return manifest
