"""Published group controls through the registered instrument commands."""
from _published import CASES, PROFILE, PROMPT, SOURCE, action_name


def group_text(index):
    name, nodes = list(SOURCE["groups"].items())[index]
    return name + ": " + "; ".join(f"L{n['layer']} F{n['feature']} P{n['position']}" for n in nodes)


def instrument(session):
    m = session.mcp
    m.body("spec.new", id="circuit-trace", title="Published Dallas", description=(
        "The instrument uses the lowercase published prompt and exact feature groups. "
        "Each case uses independent Caliper source activations. British Columbia has a published ineffective outcome. "
        "Replication requires independent reference comparison and all Caliper acceptance checks."))
    for name, default, maximum in (("selected_node", -1, 8191), ("example_page", 0, 1), ("published_group", 0, len(SOURCE["groups"]) - 1)):
        m.body("state.create", id=name, type="integer", default=default, min=default, max=maximum)
    m.body("state.create", id="activation_max", type="number", default=1., min=1e-6, max=1e20)
    for name, value, title in (("published_status", "No published case has completed.", "Case Status"),
                               ("published_members", group_text(0), "Layer, Feature, And Position")):
        m.body("state.create", id=name, type="text", default=value)
        m.body("control.create", id=name + "_value", kind="text", bind="state:" + name, title=title)
        m.body("control.create", id="clamp_status" if name == "published_status" else name, kind="readout", title=title,
               options={"format": "{text:s}", "fields": {"text": {"control": name + "_value"}}})
    m.body("control.create", id="prompt", kind="readout", title="Published Prompt", options={"format": PROMPT})
    for identifier, title in (("graph", "Attribution Graph"), ("examples", "Feature Examples"), ("output", "Actual Output Probabilities")):
        m.body("panel.create", id=identifier, primitive="text", title=title, options={"text": "The published trace is not available."})
    for cell, case in CASES.items():
        m.body("control.create", id=f"cell{cell}", kind="button", bind="action:" + action_name(cell), title=case["label"])
    m.body("control.create", id="published_group", kind="picker", bind="state:published_group", title="Published Feature Groups",
           options={"choices": [{"value": i, "label": name} for i, name in enumerate(SOURCE["groups"])]})
    m.body("control.create", id="page", kind="segmented", bind="state:example_page", title="Example Page",
           options={"choices": [{"value": 0, "label": "1–8"}, {"value": 1, "label": "9–16"}]})
    m.body("layout.set", layout={"root": "instrument", "nodes": [
        {"id": "instrument", "split": "vertical", "ratio": .90, "children": ["main", "groups"]},
        {"id": "main", "split": "vertical", "ratio": .10225, "children": ["header", "body"]},
        {"id": "header", "split": "vertical", "children": ["prompt", "clamp_status"]},
        {"id": "body", "split": "vertical", "ratio": .70, "children": ["plots", "examples"]},
        {"id": "plots", "split": "horizontal", "ratio": .67, "children": ["graph", "right"]},
        {"id": "right", "split": "vertical", "ratio": .45, "children": ["output", "controls"]},
        {"id": "controls", "split": "vertical", "children": [f"cell{cell}" for cell in CASES]},
        {"id": "groups", "split": "vertical", "children": ["published_group", "published_members", "page"]},
    ]})


def display_result(session, evidence):
    record = evidence["result"]
    before, after = [row[0] for row in record["top_five"]]
    text = (f"{record['case_id']} | {record['label']} | "
            f"Before: {before['text']!r} ({before['probability']:.6f}) | "
            f"After: {after['text']!r} ({after['probability']:.6f})")
    if record["cell"] == 29:
        text += " | Published outcome: ineffective."
    session.mcp.body("state.set", id="published_status", value=text)
