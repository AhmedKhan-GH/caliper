"""Visual acceptance from the renderer's actual IDs, camera, and label boxes."""
import math

import numpy as np

from _checks import check
from _composition import physical


def extent(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def visual_checks(name, panel, ids, metadata, *, node_count):
    height, width = ids.shape[:2]
    pick = panel["programs"][0]["pick_id"]
    mask = ids[:, :, 0] == pick
    bounds = extent(mask)
    coverage = [0., 0.] if bounds is None else [(bounds[2] - bounds[0]) / width, (bounds[3] - bounds[1]) / height]
    matrix = panel["clip_from_world"]
    flat = (len(matrix) == 16 and all(math.isfinite(value) for value in matrix)
            and abs(matrix[15] - 1) < 1e-6
            and all(abs(matrix[index]) < 1e-6 for index in (1, 2, 3, 4, 6, 7, 8, 9, 11)))
    gates = [check(name + "_orthographic", flat, matrix, "axis-aligned orthographic projection")]
    labels = [label for label in panel["labels"] if label["text"].strip()]
    boxes = [label.get("box") for label in labels]
    valid_boxes = all(isinstance(box, list) and len(box) == 4 and all(math.isfinite(v) for v in box)
                      and box[0] < box[2] and box[1] < box[3] for box in boxes)
    overlaps = []
    if valid_boxes:
        for i, left in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                right = boxes[j]
                if min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1]):
                    overlaps.append([i, j])
    gates.append(check(name + "_label_spacing", bool(labels) and valid_boxes and not overlaps,
                       {"labels": len(labels), "overlap_count": len(overlaps), "first_overlaps": overlaps[:16]},
                       "nonempty label boxes with no overlap"))
    raw = [label["text"] for label in labels if any(marker in label["text"] for marker in ("Ġ", "Ċ", "▁"))]
    gates.append(check(name + "_decoded_text", not raw, raw[:16], "no raw tokenizer markers"))
    if name == "graph":
        gates.append(check("graph_framing", min(coverage) >= .70, coverage, ">=70 percent of each panel dimension"))
        rect = physical(panel["composition"]["rect"], metadata)
        rectangles = [physical(item["composition"]["rect"], metadata) for item in metadata["panels"].values()]
        total = sum((item[2] - item[0]) * (item[3] - item[1]) for item in rectangles)
        area = float((rect[2] - rect[0]) * (rect[3] - rect[1]) / total)
        footer = float((metadata["height"] - max(item[3] for item in rectangles)) / metadata["height"])
        gates.append(check("graph_panel_share", area >= .50, area, ">=50 percent of the three panel areas"))
        gates.append(check("compact_disclosures", footer <= .15, footer, "footer and status bar <=15 percent of window height"))
        visible = set(int(value) for value in np.unique(ids[:, :, 1][mask]))
        gates.append(check("graph_visible_nodes", visible == set(range(node_count)), len(visible), node_count))
    elif name == "examples":
        rows = [extent(mask & (ids[:, :, 1] == index)) for index in range(8)]
        ordered = all(row is not None for row in rows)
        if ordered:
            ordered = all(rows[i][3] <= rows[i + 1][1] for i in range(7))
        gates.append(check("example_row_layout", ordered and coverage[0] >= .70 and coverage[1] >= .60,
                           {"bounds": rows, "coverage": coverage}, "8 separated rows, >=70 percent width and >=60 percent height"))
    else:
        bars = [extent(mask & (ids[:, :, 1] == index)) for index in range(10)]
        ordered = all(bar is not None for bar in bars)
        if ordered:
            ordered = all(bars[i][2] <= bars[i + 1][0] for i in range(9))
        gates.append(check("output_bar_layout", ordered and coverage[0] >= .70,
                           {"bounds": bars, "coverage": coverage}, "10 separated bars across >=70 percent of panel width"))
    return gates
