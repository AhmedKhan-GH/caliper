"""Bounded CPU reconstruction of panel placement and expected font glyphs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from _mcp import Failure

MAX_PIXELS = 8_388_608
MAX_ATLAS_PIXELS = 16_777_216


def read_rgba(root, record, key, *, limit=MAX_PIXELS):
    width, height = record["width"], record["height"]
    if type(width) is not int or type(height) is not int or min(width, height) <= 0 or width * height > limit:
        raise Failure("The capture texture exceeds its pixel limit.")
    path = Path(root) / record[key]
    if path.stat().st_size != width * height * 4:
        raise Failure("The capture texture has incomplete bytes.")
    return np.fromfile(path, dtype=np.uint8).reshape(height, width, 4)


def physical(rect, metadata):
    values = np.asarray(rect, dtype=np.float32)
    origin = np.asarray(metadata["origin"] * 2, dtype=np.float32)
    scale = np.asarray(metadata["pixel_scale"] * 2, dtype=np.float32)
    if values.shape != (4,) or origin.shape != (4,) or scale.shape != (4,) or not np.isfinite(values).all():
        raise Failure("The capture transform is invalid.")
    if not np.isfinite(origin).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise Failure("The capture DPI or origin is invalid.")
    return (values - origin) * scale


def scissor(rect, metadata, width, height):
    # Match the Metal ImGui backend: truncate the origin and the clipped extent.
    x0, y0, x1, y1 = physical(rect, metadata)
    x0, y0, x1, y1 = max(0., x0), max(0., y0), min(width, x1), min(height, y1)
    return int(x0), int(y0), int(x0) + max(0, int(x1 - x0)), int(y0) + max(0, int(y1 - y0))


def sample(texture, u, v):
    """Linear sampling at normalized UV coordinates, with edge clamping."""
    height, width = texture.shape[:2]
    x = np.clip(u * width - .5, 0, width - 1)
    y = np.clip(v * height - .5, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return ((texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx) * (1 - fy)
            + (texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx) * fy).astype(np.float32)


def fragments(quad, metadata, window_size, bounds):
    """Return covered pixel centers and UV coordinates in one bounded region."""
    width, height = window_size
    x0, y0, x1, y1 = physical(quad["rect"], metadata)
    if x1 <= x0 or y1 <= y0:
        raise Failure("The capture quad has no positive area.")
    clip = scissor(quad["clip"], metadata, width, height)
    left = max(bounds[0], clip[0], int(np.ceil(x0 - .5)))
    top = max(bounds[1], clip[1], int(np.ceil(y0 - .5)))
    right = min(bounds[2], clip[2], int(np.ceil(x1 - .5)))
    bottom = min(bounds[3], clip[3], int(np.ceil(y1 - .5)))
    if right <= left or bottom <= top:
        return None
    u0, v0, u1, v1 = quad["uv"]
    if not np.isfinite([u0, v0, u1, v1]).all() or not all(0 <= p <= 1 for p in (u0, v0, u1, v1)):
        raise Failure("The capture UV coordinates are invalid.")
    u = u0 + (np.arange(left, right, dtype=np.float32)[None, :] + .5 - x0) / (x1 - x0) * (u1 - u0)
    v = v0 + (np.arange(top, bottom, dtype=np.float32)[:, None] + .5 - y0) / (y1 - y0) * (v1 - v0)
    return (slice(top - bounds[1], bottom - bounds[1]), slice(left - bounds[0], right - bounds[0])), u, v


def edge_mask(values):
    """Mark only a one-pixel neighborhood of reference color or ID transitions."""
    edge = np.zeros(values.shape[:2], dtype=bool)
    horizontal = np.any(values[:, 1:] != values[:, :-1], axis=2)
    vertical = np.any(values[1:] != values[:-1], axis=2)
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    return edge


def panel_composition(window, panel, color, ids, atlas, metadata):
    placement = panel.get("composition")
    if not isinstance(placement, dict) or placement.get("sampler") != "linear" or placement.get("blend") != "source_alpha":
        raise Failure("The capture lacks supported panel composition evidence.")
    height, width = window.shape[:2]
    rect = physical(placement["rect"], metadata)
    clip = scissor(placement["clip"], metadata, width, height)
    bounds = (max(0, clip[0], int(np.floor(rect[0]))), max(0, clip[1], int(np.floor(rect[1]))),
              min(width, clip[2], int(np.ceil(rect[2]))), min(height, clip[3], int(np.ceil(rect[3]))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise Failure("The captured panel is fully clipped.")
    shape = (bounds[3] - bounds[1], bounds[2] - bounds[0])
    if shape[0] * shape[1] > MAX_PIXELS:
        raise Failure("The composed panel exceeds its pixel limit.")
    expected = np.zeros((*shape, 4), dtype=np.float32)
    mapped = np.zeros((*shape, 2), dtype=np.uint32)
    visible = np.zeros(shape, dtype=bool)
    fragment = fragments(placement, metadata, (width, height), bounds)
    if fragment is None:
        raise Failure("The captured panel has no visible samples.")
    region, u, v = fragment
    tint = np.asarray(placement["tint"], dtype=np.float32) / 255
    source = sample(color, u, v) * tint
    # Scene targets clear to opaque WindowBg. A changed alpha contract needs
    # an explicit destination reference instead of assumed window colors.
    if not np.all(source[:, :, 3] == 255):
        raise Failure("The panel requires an opaque composition reference.")
    expected[region] = source
    mapped[region] = ids[np.clip((v * ids.shape[0]).astype(int), 0, ids.shape[0] - 1),
                         np.clip((u * ids.shape[1]).astype(int), 0, ids.shape[1] - 1)]
    visible[region] = True
    base = expected.copy()
    text_mask = np.zeros(shape, dtype=bool)
    label_regions = []
    complete_labels = []
    labels = panel["labels"]
    if len(labels) > 512:
        raise Failure("The panel exceeds 512 reference labels.")
    for label in labels:
        quads = label.get("quads")
        if not isinstance(quads, list) or len(quads) > 512:
            raise Failure("The label lacks bounded reference glyphs.")
        label_box = None
        complete = True
        for quad in quads:
            full = np.ceil(physical(quad["rect"], metadata) - .5).astype(int)
            fragment = fragments(quad, metadata, (width, height), bounds)
            if fragment is None:
                complete = False
                continue
            region, u, v = fragment
            ys, xs = region
            complete &= (xs.start + bounds[0] == full[0] and ys.start + bounds[1] == full[1]
                         and xs.stop + bounds[0] == full[2] and ys.stop + bounds[1] == full[3])
            rgba = sample(atlas, u, v) * (np.asarray(quad["tint"], dtype=np.float32) / 255)
            alpha = rgba[:, :, 3:4] / 255
            destination = expected[region]
            destination[:, :, :3] = rgba[:, :, :3] * alpha + destination[:, :, :3] * (1 - alpha)
            destination[:, :, 3:4] = rgba[:, :, 3:4] + destination[:, :, 3:4] * (1 - alpha)
            # The render target quantizes after each glyph pass, including its shadow.
            expected[region] = np.rint(destination)
            text_mask[region] = True
            y, x = region
            box = (x.start, y.start, x.stop, y.stop)
            label_box = box if label_box is None else (min(label_box[0], box[0]), min(label_box[1], box[1]),
                                                       max(label_box[2], box[2]), max(label_box[3], box[3]))
        label_regions.append(label_box)
        complete_labels.append(bool(complete))
    expected = np.rint(expected).astype(np.uint8)
    actual = window[bounds[1]:bounds[3], bounds[0]:bounds[2]]
    difference = np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max(axis=2)
    edge = edge_mask(expected) | edge_mask(mapped)
    over = visible & (difference > 2)
    interior_over = int(np.count_nonzero(over & ~edge))
    edge_over = int(np.count_nonzero(over & edge))
    compared = int(visible.sum())
    # Keep the existing two-channel-level / one-percent edge bounds. No area is
    # excluded because the actual screenshot differs from the reference.
    matched = visible & ~text_mask & (difference <= 2)
    source_ids = {tuple(row) for row in np.unique(ids.reshape(-1, 2), axis=0) if row[0]}
    window_ids = {tuple(row) for row in np.unique(mapped[matched], axis=0) if row[0]}
    text_reports = []
    for label, box, complete in zip(labels, label_regions, complete_labels):
        passed, maximum, ink_pixels = False, None, 0
        if box is not None:
            x0, y0, x1, y1 = box
            local = np.s_[y0:y1, x0:x1]
            valid = visible[local]
            ink = np.abs(expected[local].astype(np.float32) - base[local]).max(axis=2) > 2
            ink_pixels = int(np.count_nonzero(ink & valid))
            maximum = int(difference[local][valid].max()) if valid.any() else None
            passed = complete and ink_pixels > 0 and maximum is not None and maximum <= 2
        text_reports.append({"text": label["text"], "passed": passed, "maximum_difference": maximum,
                             "ink_pixels": ink_pixels, "complete": complete})
    summary = {"pixels": compared, "interior_over_two": interior_over,
               "edge_over_two": edge_over, "edge_difference_share": edge_over / compared,
               "source_items": [[int(value) for value in row] for row in sorted(source_ids)],
               "window_items": [[int(value) for value in row] for row in sorted(window_ids)],
               "rect_pixels": rect.tolist(), "clip_pixels": clip}
    return {"passed": interior_over == 0 and edge_over / compared <= .01 and source_ids <= window_ids,
            "summary": summary, "labels": text_reports, "ids": mapped,
            "actual": actual, "geometry": visible & ~text_mask, "height_scale": float(rect[3] - rect[1])}


def required_text(reports, expected):
    required = Counter(expected)
    proved = Counter(row["text"] for row in reports if row["passed"])
    return bool(required) and all(proved[text] >= count for text, count in required.items())
