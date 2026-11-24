"""Measured circuit gates, with separate fixture and learned-model results."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from _mcp import Failure

FIXTURE_HASH = "2464295694bd3690f156dcc7248882f0ad289f55972caafbbd7bd23f4e3d109e"
FIXTURE_PROMPT = "The red cube is beside a tree."
QWEN_PROMPT = "Paris is the capital of"


def selected_feature(graph):
    candidates = [n for n in graph["nodes"] if n["kind"] == "feature"
                  and n["layer"] < graph["counts"]["layers"] - 1]
    if not candidates:
        raise Failure("The actual graph has no selectable feature before the last layer.")
    return max(candidates, key=lambda n: (n["influence"], -n["index"]))


def token_path(graph):
    logits = [n for n in graph["nodes"] if n["kind"] == "logit"]
    if not logits:
        return False
    top = max(logits, key=lambda n: n["probability"])["index"]
    reached = {n["index"] for n in graph["nodes"] if n["kind"] == "token"}
    edges = graph["edges"]
    for _ in range(len(graph["nodes"])):
        extended = reached | {e["target"] for e in edges if e["source"] in reached and e["weight"] != 0}
        if extended == reached:
            break
        reached = extended
    return top in reached


def check(name, passed, measured, threshold, *, required=True):
    return {"name": name, "passed": bool(passed), "measured": measured,
            "threshold": threshold, "required": required}


def program_budget_checks(snapshot):
    """Keep absent GPU measurements unverified and enforce the declared budgets."""
    by_id = {program["id"]: program for program in snapshot["status"]["programs"]}
    checks = []
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    def byte_limit(value):
        return value * (1 << 20) if number(value) else None
    for identifier in snapshot["expected_steps"]:
        program = by_id.get(identifier, {})
        draw = program.get("draw") or {}
        manifest = snapshot["pieces"].get(identifier, {})
        derive_limits, draw_limits = manifest.get("derive", {}), manifest.get("draw", {})
        metrics = [
            ("derive_time_ms", program.get("last_pass_ms"), derive_limits.get("time_ms"), None, 50),
            ("derive_memory_bytes", program.get("memory_bytes"), byte_limit(derive_limits.get("memory_mb")),
             program.get("memory_limit_bytes"), 64 << 20),
            ("draw_time_ms", draw.get("frame_ms"), draw_limits.get("frame_ms"), draw.get("frame_limit_ms"), 4),
            ("draw_memory_bytes", draw.get("memory_bytes"), byte_limit(draw_limits.get("memory_mb")),
             draw.get("memory_limit_bytes"), 64 << 20),
        ]
        for name, measured, declared, reported, planned in metrics:
            verified = number(measured) and number(declared)
            if name != "derive_time_ms":
                verified = verified and number(reported)
            passed = verified and 0 < declared <= planned and measured <= declared
            if reported is not None:
                passed = passed and reported == declared
            gate = check(f"{identifier}.{name}", passed, measured,
                         {"planned": planned, "declared": declared, "reported": reported})
            gate["evidence"] = "measured" if verified else "unverified"
            if not verified:
                gate["reason"] = "A GPU measurement or declared limit is unavailable or invalid."
            checks.append(gate)
    return checks


def math_checks(graph, examples, before, after, validation, trace_seconds, steering_seconds, *, fixture):
    rows = examples["examples"]
    strengths = [r["activation"] for r in rows]
    ids = [r["example_id"] for r in rows]
    top = max(before, key=lambda r: r["baseline"])
    changed = next((r for r in after if r["token"] == top["token"]), None)
    if changed is None:
        raise Failure("The steered bars omit the baseline top token.")
    fall = top["baseline"] - changed["probability"]
    new_top = max(after, key=lambda r: r["probability"])["token"]
    semantic = not fixture
    residual = graph["completeness"]["max_absolute_residual"]
    tolerance = 1e-9 if fixture else 0.02
    return [
        check("completeness", math.isfinite(residual) and residual <= tolerance, residual, tolerance),
        check("feature_share", graph["feature_share"] >= 0.30, graph["feature_share"], 0.30),
        check("graph_size", 20 <= len(graph["nodes"]) <= 512, len(graph["nodes"]), [20, 512]),
        check("token_path", token_path(graph), token_path(graph), True),
        check("examples", len(rows) == 16 and ids == list(range(16)) and all(a > 0 for a in strengths)
              and strengths == sorted(strengths, reverse=True),
              {"ids": ids, "activations": strengths}, "16 positive examples in indexed order"),
        check("fixture_output_effect", fall > 0, fall, ">0", required=fixture),
        check("steering", fall >= 0.05 or new_top != top["token"],
              {"probability_fall": fall, "baseline_top": top["token"], "steered_top": new_top},
              "fall >=0.05 or top-token change", required=semantic),
        check("validation", validation.get("count") == 10 and validation.get("correct_signs", 0) >= 8
              and validation.get("spearman") is not None and validation["spearman"] >= 0.6,
              validation, "10 features, >=8 correct signs, Spearman >=0.6"),
        check("trace_time", trace_seconds <= 20, trace_seconds, 20),
        check("steering_time", steering_seconds <= 1, steering_seconds, 1),
    ]


def bar_rows(reply):
    """Decode the real logit_bars resource, in the declared field order."""
    rows = []
    for element in reply["elements"]:
        # program.read flattens vector fields into each element's scalar list.
        if len(element) != 8:
            raise Failure("The published bar layout differs from logit_bars@1.")
        rows.append({"rect": element[:4], "baseline": element[4], "probability": element[5],
                     "token": int(element[6]), "index": int(element[7])})
    if len(rows) != 10:
        raise Failure("The output resource does not contain ten bars.")
    return rows


def example_page_check(examples, rows, cells, page):
    """Compare all 200 slots with the real API windows and valid prefixes."""
    records = examples["examples"][page * 8:page * 8 + 8]
    valid = len(rows["elements"]) == 8 and len(cells["elements"]) == 200
    for i, (record, row) in enumerate(zip(records, rows["elements"])):
        source = [j for j in range(4, 29) if record["valid"][j]]
        valid &= row == [record["example_id"], page * 8 + i, i, len(source)]
        for col, cell in enumerate(cells["elements"][i * 25:(i + 1) * 25]):
            valid &= int(cell[6]) == i and bool(cell[7]) == (col < len(source))
            if col < len(source):
                j = source[col]
                valid &= int(cell[5]) == record["token_ids"][j]
                valid &= math.isclose(cell[4], record["activations"][j], rel_tol=1e-6, abs_tol=1e-7)
    return check(f"example_page_{page + 1}_records", valid,
                 {"rows": rows["elements"], "slots": len(cells["elements"])}, "8 rows of 25 API token slots")


def raw_panel(folder, metadata, name):
    panel = metadata["panels"][name]
    width, height = panel["width"], panel["height"]
    if not 0 < width * height <= 8_388_608:
        raise Failure("The diagnostic exceeds the pixel limit.")
    root = Path(folder)
    ids = np.fromfile(root / panel["ids"], dtype="<u4").reshape(height, width, 2)
    color = np.fromfile(root / panel["color"], dtype=np.uint8).reshape(height, width, 4)
    return panel, ids, color


def capture_checks(folder, metadata, png, graph, bars, *, selected):
    """Read renderer targets and the exact PNG pixels from one capture."""
    root = Path(folder)
    if metadata.get("error"):
        raise Failure(metadata["error"])
    window = np.asarray(Image.open(png).convert("RGBA"))
    raw = np.fromfile(root / metadata["window"], dtype=np.uint8).reshape(window.shape)
    same = np.array_equal(window, raw)
    panel, ids, _ = raw_panel(root, metadata, "graph")
    pick = panel["programs"][0]["pick_id"]
    node_ids = sorted(int(i) for i in np.unique(ids[:, :, 1][ids[:, :, 0] == pick]))
    label_text = [label["text"] for label in panel["labels"]]
    required_labels = [t["text"] for t in graph["tokens"] if t["text"].strip()]
    required_labels.append(selected["label"])
    # The diagnostic records placed host labels. Nonuniform pixels in each box
    # establish that host text reached the full PNG, beside the geometry target.
    scale = metadata["pixel_scale"]
    origin = metadata["origin"]
    drawn = []
    for label in panel["labels"]:
        x0, y0, x1, y1 = label["box"]
        x0, x1 = [max(0, min(window.shape[1], round((x-origin[0])*scale[0]))) for x in (x0, x1)]
        y0, y1 = [max(0, min(window.shape[0], round((y-origin[1])*scale[1]))) for y in (y0, y1)]
        region = window[y0:y1, x0:x1, :3]
        if region.size and np.any(region.max(axis=(0, 1)) != region.min(axis=(0, 1))):
            drawn.append(label["text"])
    ep, ei, _ = raw_panel(root, metadata, "examples")
    example_pick = ep["programs"][0]["pick_id"]
    row_ids = sorted(int(i) for i in np.unique(ei[:, :, 1][ei[:, :, 0] == example_pick]))
    bp, bi, bc = raw_panel(root, metadata, "output")
    bar_pick = bp["programs"][0]["pick_id"]
    top = max(bars, key=lambda row: row["baseline"])
    # Teal comes from the actual current bar, while ghost-only sides remain distinct.
    mask = ((bi[:, :, 0] == bar_pick) & (bi[:, :, 1] == top["index"])
            & (bc[:, :, 1] > bc[:, :, 0] * 1.5) & (bc[:, :, 2] > bc[:, :, 0] * 1.5)
            & (bc[:, :, 1] > 100))
    ys = np.flatnonzero(mask.any(axis=1))
    measured = int(ys[-1] - ys[0] + 1) if len(ys) else 0
    # The last frame's clip matrix includes the real pan/zoom camera.
    bar_scale = abs(bp["clip_from_world"][5]) * bp["height"] * 0.72
    expected = top["probability"] * bar_scale
    gates = [
        check("capture_generation", same, metadata["capture"], "PNG equals diagnostic window pixels"),
        check("rendered_nodes", node_ids == list(range(len(graph["nodes"]))), node_ids, len(graph["nodes"])),
        check("host_labels", all(t in label_text and t in drawn for t in required_labels),
              {"required": required_labels, "placed": label_text, "pixel_boxes": drawn}, "prompt and selected labels"),
        check("example_page", len(row_ids) == 8, row_ids, "8 actual row IDs"),
        check("bar_height", abs(measured - expected) <= 2,
              {"pixels": measured, "expected": expected, "scale": bar_scale}, "within 2 pixels"),
    ]
    for name, target in metadata["panels"].items():
        for program in target["programs"]:
            minimum = metadata["expected_steps"][program["id"]]
            gates.append(check(name + "_resource_generation", target["generation"] > 0
                               and program["step"] is not None and program["step"] >= minimum,
                               {"generation": target["generation"], "step": program["step"]}, minimum))
    return gates, {"row_ids": row_ids, "height": measured, "expected_height": expected}


def finish(result):
    failed = [g["name"] for g in result.get("checks", []) if g["required"] and not g["passed"]]
    result["failed_checks"] = failed
    result["passed"] = not failed and not result.get("error") and bool(result.get("checks"))
    result["qwen_acceptance"] = result["passed"] and result["mode"] == "qwen" and result.get("pixel_evidence_collected", False)
    return 0 if result["passed"] else 1
