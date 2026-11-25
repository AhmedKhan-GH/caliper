"""Measured circuit gates, with separate fixture and learned-model results."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from _mcp import Failure
from _composition import panel_composition, read_rgba, required_text, MAX_PIXELS, MAX_ATLAS_PIXELS
from _qualification import dictionary_qualification

FIXTURE_HASH = "2464295694bd3690f156dcc7248882f0ad289f55972caafbbd7bd23f4e3d109e"
FIXTURE_PROMPT = "The red cube is beside a tree."
MODEL_PROMPT = "Fact: the capital of the state containing Dallas is"


def selected_feature(graph, *, rule="effect"):
    """Select by absolute top-logit effect, or retain the influence comparison."""
    candidates = [n for n in graph["nodes"] if n["kind"] == "feature"
                  and n["layer"] < graph["counts"]["layers"] - 1]
    if not candidates:
        raise Failure("The actual graph has no selectable feature before the last layer.")
    if rule == "effect":
        return max(candidates, key=lambda n: (abs(n["effect"]), -n["index"]))
    if rule == "influence":
        return max(candidates, key=lambda n: (n["influence"], -n["index"]))
    raise ValueError("Unknown feature selection rule.")


def secondary_selection(graph, validation=None):
    """Keep the original selection and its independently measured ablation."""
    node = selected_feature(graph, rule="influence")
    validation = validation or {}
    identity = ("layer", "position", "feature")
    if "secondary_measurement" in validation:
        row = validation["secondary_measurement"]
        if not isinstance(row, dict) or any(row.get(key) != node[key] for key in identity):
            raise Failure("The secondary measurement does not match the influence-selected feature.")
    else:
        row = next((row for row in validation.get("measurements", [])
                    if all(row[key] == node[key] for key in identity)), None)
    return {"rule": "largest influence before the last layer", "feature": node,
            "predicted_ablation_effect": -node["effect"],
            "measured_ablation_effect": None if row is None else row["measured"],
            "measurement_source": "trace.validate" if row is not None else None,
            "units": "centered top-token logits"}


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


def effect_direction(value):
    """Return a direction only for a finite, nonzero effect."""
    if not math.isfinite(value) or value == 0:
        return None
    return "increase" if value > 0 else "decrease"


def direction_evidence(predicted_ablation_effect, measured_change):
    """Compare an intervention prediction with an observed change."""
    predicted = effect_direction(predicted_ablation_effect)
    measured = effect_direction(measured_change)
    return {"predicted_ablation_effect": predicted_ablation_effect if math.isfinite(predicted_ablation_effect) else None,
            "prediction_finite": math.isfinite(predicted_ablation_effect),
            "prediction_units": "centered original-top-token logits",
            "predicted_direction": predicted, "measured_direction": measured,
            "direction_matches": predicted is not None and predicted == measured}


def bar_direction_check(before_height, after_height, *, predicted_ablation_effect, fixture):
    change = after_height - before_height
    evidence = direction_evidence(predicted_ablation_effect, change)
    return check("bar_change_matches_prediction", evidence["direction_matches"],
                 {**evidence, "before_pixels": before_height, "after_pixels": after_height,
                  "pixel_change": change}, "nonzero pixel change in the predicted direction", required=not fixture)


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


def dallas_preflight_check(status, prompt):
    """Require the exact Dallas answer without an additional confidence threshold."""
    from _inference_evidence import baseline_evidence
    if prompt != MODEL_PROMPT:
        raise Failure("Dallas preflight requires the exact default prompt.")
    comparison = baseline_evidence(status, prompt)
    mismatch = comparison["baseline_top_text"] != " Austin"
    return check("dallas_preflight", not mismatch, {**comparison, "mismatch": mismatch},
                 "exact decoded baseline top token ' Austin'")


def prompt_confidence_check(prompt, before, baseline_top_token):
    """Apply the default-prompt exception to the actual baseline argmax."""
    rows = [row for row in before if row["token"] == baseline_top_token]
    if len(rows) != 1:
        raise Failure("Prompt confidence requires one actual baseline top-token row.")
    probability = rows[0]["baseline"]
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise Failure("Prompt confidence requires a finite baseline probability.")
    default = prompt == MODEL_PROMPT
    return check("prompt_confidence", default or probability > 0.5,
                 {"default_prompt": default, "baseline_top": baseline_top_token,
                  "baseline_probability": probability},
                 "default Dallas prompt or actual baseline top-token probability >0.5")


def math_checks(graph, examples, before, after, validation, trace_seconds, steering_seconds, *, fixture,
                argmax_token_ids, predicted_ablation_effect):
    rows = examples["examples"]
    strengths = [r["activation"] for r in rows]
    ids = [r["example_id"] for r in rows]
    baseline_top, new_top = argmax_token_ids
    top = next((r for r in before if r["token"] == baseline_top), None)
    if top is None:
        raise Failure("The baseline bars omit the actual baseline top token.")
    changed = next((r for r in after if r["token"] == top["token"]), None)
    if changed is None:
        raise Failure("The steered bars omit the baseline top token.")
    change = changed["probability"] - top["baseline"]
    direction = direction_evidence(predicted_ablation_effect, change)
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
        check("fixture_output_effect", -change > 0, -change, ">0", required=fixture),
        check("steering_change_matches_prediction",
              direction["direction_matches"] and (abs(change) >= 0.05 or new_top != top["token"]),
              {**direction, "probability_change": change, "absolute_probability_change": abs(change),
               "baseline_probability": top["baseline"], "steered_probability": changed["probability"],
               "baseline_top": top["token"], "steered_top": new_top},
              "(absolute probability change >=0.05 or top-token change) and direction matches prediction",
              required=semantic),
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
    if type(width) is not int or type(height) is not int or min(width, height) <= 0 or width * height > MAX_PIXELS:
        raise Failure("The diagnostic exceeds the pixel limit.")
    root = Path(folder)
    if (root / panel["ids"]).stat().st_size != width * height * 8:
        raise Failure("The capture ID target has incomplete bytes.")
    ids = np.fromfile(root / panel["ids"], dtype="<u4").reshape(height, width, 2)
    color = read_rgba(root, panel, "color")
    return panel, ids, color


def capture_checks(folder, metadata, png, graph, bars, *, selected, baseline_top_token, expanded=False):
    """Read renderer targets and the exact PNG pixels from one capture."""
    if type(baseline_top_token) is not int or baseline_top_token < 0:
        raise Failure("The capture lacks the original baseline top token ID.")
    top = next((row for row in bars if row["token"] == baseline_top_token), None)
    if top is None:
        raise Failure("The captured bars omit the original baseline top token.")
    root = Path(folder)
    if metadata.get("error"):
        raise Failure(metadata["error"])
    with Image.open(png) as image:
        if image.size != (metadata["width"], metadata["height"]) or image.width * image.height > MAX_PIXELS:
            raise Failure("The captured window size is invalid.")
        window = np.asarray(image.convert("RGBA"))
    raw = read_rgba(root, metadata, "window")
    same = np.array_equal(window, raw)
    atlas = read_rgba(root, metadata["font_atlas"], "file", limit=MAX_ATLAS_PIXELS)
    compositions = {}
    for name in ("graph", "examples", "output"):
        target, target_ids, target_color = raw_panel(root, metadata, name)
        compositions[name] = panel_composition(window, target, target_color, target_ids, atlas, metadata)
    panel, ids, _ = raw_panel(root, metadata, "graph")
    pick = panel["programs"][0]["pick_id"]
    node_ids = sorted(int(i) for i in np.unique(ids[:, :, 1][ids[:, :, 0] == pick]))
    required_labels = [t["text"] for t in graph["tokens"] if t["text"].strip()]
    required_labels.append(selected["label"])
    ep, ei, _ = raw_panel(root, metadata, "examples")
    example_pick = ep["programs"][0]["pick_id"]
    row_ids = sorted(int(i) for i in np.unique(ei[:, :, 1][ei[:, :, 0] == example_pick]))
    bp = metadata["panels"]["output"]
    output = compositions["output"]
    bi, bc = output["ids"], output["actual"].astype(np.float32)
    bar_pick = bp["programs"][0]["pick_id"]
    # Teal comes from the actual current bar, while ghost-only sides remain distinct.
    mask = (output["geometry"] & (bi[:, :, 0] == bar_pick) & (bi[:, :, 1] == top["index"])
            & (bc[:, :, 1] > bc[:, :, 0] * 1.5) & (bc[:, :, 2] > bc[:, :, 0] * 1.5)
            & (bc[:, :, 1] > 100))
    ys = np.flatnonzero(mask.any(axis=1))
    measured = int(ys[-1] - ys[0] + 1) if len(ys) else 0
    # The last frame's clip matrix includes the real pan/zoom camera.
    bar_scale = abs(bp["clip_from_world"][5]) * output["height_scale"] * 0.72
    expected = top["probability"] * bar_scale
    gates = [
        check("capture_generation", same, metadata["capture"], "PNG equals diagnostic window pixels"),
        check("rendered_nodes", node_ids == list(range(len(graph["nodes"]))), node_ids, len(graph["nodes"])),
        check("host_labels", required_text(compositions["graph"]["labels"], required_labels),
              {"required": required_labels, "glyphs": compositions["graph"]["labels"]},
              "expected prompt and selected glyph pixels within two channel levels"),
        check("example_page", len(row_ids) == 8, row_ids, "8 actual row IDs"),
        check("bar_height", abs(measured - expected) <= 2,
              {"token": baseline_top_token, "pixels": measured, "expected": expected, "scale": bar_scale}, "within 2 pixels"),
    ]
    for name, composition in compositions.items():
        gates.append(check(name + "_composition", composition["passed"], composition["summary"],
                           "two channel levels in interiors, at most one percent differing edges, every item visible"))
    for name, target in metadata["panels"].items():
        for program in target["programs"]:
            minimum = metadata["expected_steps"][program["id"]]
            gates.append(check(name + "_resource_generation", target["generation"] > 0
                               and program["step"] is not None and program["step"] >= minimum,
                               {"generation": target["generation"], "step": program["step"]}, minimum))
    from _visual import visual_checks
    for name in ("graph", "examples", "output"):
        target, target_ids, _ = raw_panel(root, metadata, name)
        gates.extend(visual_checks(name, target, target_ids, metadata, node_count=len(graph["nodes"]), expanded=expanded))
    return gates, {"row_ids": row_ids, "token": baseline_top_token, "height": measured, "expected_height": expected}


def finish(result):
    has_evidence = any(g["name"] != "dictionary_qualification" for g in result.get("checks", []))
    qualification = dictionary_qualification(result.get("dictionary_evidence"),
                                              quality_override=result.get("dictionary_quality_override"),
                                              family="gemma" if result["mode"] == "gemma" else "qwen")
    result["dictionary_qualification"] = qualification
    if result["mode"] in ("qwen", "gemma"):
        # Recompute this gate from evidence. A cached status cannot qualify a run.
        result["checks"] = [g for g in result.get("checks", []) if g["name"] != "dictionary_qualification"]
        sample_requirement = ("at least 50000 research source positions" if result["mode"] == "gemma"
                              else "200000 quality tokens")
        result["checks"].append(check("dictionary_qualification", qualification["qualified"], qualification,
                                      f"exact current dictionary, {sample_requirement}, 2000000 index tokens",
                                      required=result["mode"] == "gemma" or result.get("pixel_evidence_collected") is not False))
    result["accepted_replication"] = False
    if result["mode"] == "gemma":
        from _published import PROFILE
        result["checks"] = [gate for gate in result["checks"] if gate["name"] not in
                            ("published_profile", "external_replication_review")]
        explicit = result.get("replication_profile") == PROFILE
        result["checks"].append(check("published_profile", explicit, result.get("replication_profile"), PROFILE))
        evidence = {"verified": False, "reason": "Independent reference, outcomes, browser, parity, and memory evidence are required."}
        if explicit and all(result.get(key) for key in ("replication_evidence_root", "replication_collection", "replication_review")):
            from _replication_review import verify_review
            try:
                evidence = verify_review(result["replication_evidence_root"], result["replication_collection"], result["replication_review"])
            except Failure as error:
                evidence["reason"] = str(error)
        result["checks"].append(check("external_replication_review", evidence["verified"], evidence,
                                      "verified independent agreement and every required review artifact"))
    failed = [g["name"] for g in result.get("checks", []) if g["required"] and not g["passed"]]
    result["failed_checks"] = failed
    result["passed"] = (result.get("completed") is True and not failed
                        and not result.get("error") and has_evidence)
    result["qwen_acceptance"] = (result["passed"] and result["mode"] == "qwen"
                                 and qualification["qualified"] and result.get("pixel_evidence_collected", False))
    result["qwen_acceptance_scope"] = qualification["scope"] if result["qwen_acceptance"] else None
    result["qwen_full_mlp_acceptance"] = result["qwen_acceptance"] and qualification["scope"] == "all_layers"
    result["gemma_acceptance"] = (result["passed"] and result["mode"] == "gemma"
                                 and qualification["qualified"] and result.get("pixel_evidence_collected", False))
    result["accepted_replication"] = result["gemma_acceptance"]
    return 0 if result["passed"] else 1
