#!/usr/bin/env python3
"""The Python streaming figure on the Caliper landing page (site/index.html).

    python3 scripts/site_streaming_figure.py [--write site/index.html]

What happens between a training script and the viewer when the script calls caliper.show(): the snapshot is a
GPU copy on torch's own stream into POSIX shared memory, and the viewer draws the newest completed slot from the
same pages. Facts from docs/wiki/explanation/live-streaming.md (sessions, snapshots, sampling, sync).

Every node, arrow, slot and heatmap carries an id the page's script animates: illustrative weights are computed,
copied into a slot, published, claimed and drawn, one training step per cycle, with the matching lines lit in the
code above. Without the script the figure shows one moment of that cycle.
"""
import math
import random
import re
import sys
from pathlib import Path

W, H = 1100, 560
ROWS, COLS = 8, 12
SLOT_Y = [210 + 64 * k for k in range(3)]          # top of each slot; the arrows meet a slot at its middle
out = []


def rect(x, y, w, h, cls, r=4, ident=""):
    i = f' id="{ident}"' if ident else ""
    out.append(f'<rect{i} x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>')


def text(x, y, s, cls, anchor="start", ident=""):
    i = f' id="{ident}"' if ident else ""
    out.append(f'<text{i} x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>')


def node(x, y, w, h, title, lines, cls="blk", ident=""):
    rect(x, y, w, h, cls, ident=ident)
    text(x + 16, y + 24, title, "mono")
    for i, l in enumerate(lines):
        text(x + 16, y + 44 + i * 18, l, "n")


def arrow(d, cls="arrow", ident=""):
    i = f' id="{ident}"' if ident else ""
    out.append(f'<path{i} d="{d}" class="{cls}" marker-end="url(#{"tip-data" if "data" in cls else "tip"})"/>')


# The illustrative weights: a learned pattern that noise resolves into as training runs. The page's script uses
# the same formula; this static copy is the state late in the run.
def target(r, c):
    return 0.95 * math.sin(0.95 * c - 0.7 * r) * math.exp(-((r - 3.5) ** 2) / 14)


def colour(v):
    v = max(-1.0, min(1.0, v))
    base, pos, neg = (42, 41, 37), (179, 139, 89), (70, 119, 173)
    end = pos if v >= 0 else neg
    t = abs(v)
    return "#%02x%02x%02x" % tuple(round(b + (e - b) * t) for b, e in zip(base, end))


def heatmap(ident, x, y, w, h, noise):
    rng = random.Random(ident)
    cw, ch = w / COLS, h / ROWS
    cells, values = [], []
    for r in range(ROWS):
        for c in range(COLS):
            v = target(r, c) * (1 - noise) + rng.uniform(-1, 1) * noise
            values.append(v)
            cells.append(f'<rect x="{x + c * cw:.2f}" y="{y + r * ch:.2f}" width="{cw - 0.8:.2f}" height="{ch - 0.8:.2f}" '
                         f'fill="{colour(v)}"/>')
    out.append(f'<g id="{ident}" class="hm">' + "".join(cells) + "</g>")
    return values


out.append('<defs>'
           '<marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,1 L10,5 L0,9 Z" class="tip"/></marker>'
           '<marker id="tip-data" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,1 L10,5 L0,9 Z" class="tip-data"/></marker>'
           '</defs>')

# The three columns: the training process, the shared pages, the viewer process.
for x, w, title, sub in ((16, 340, "TRAINING PROCESS", "Python, torch on MPS"),
                         (386, 328, "SHARED MEMORY", "POSIX segments, mode 0600"),
                         (744, 340, "VIEWER PROCESS", "Caliper app or caliper-viewer")):
    rect(x, 44, w, 492, "pkg", 10)
    text(x + 18, 28, title, "k")
    text(x + w - 18, 28, sub, "n dim", "end")

# Training process, top to bottom.
node(36, 64, 300, 62, "caliper.show(W)", ["60 Hz rate limit, then the watcher check"], ident="n-show")
node(36, 150, 300, 62, "SnapshotRing", ["a free slot of three, or the snapshot drops"], ident="n-ring")
node(36, 236, 300, 82, "torch MPS stream", ["slot.copy_(W)", "torch.aminmax(W, out=stats)"], "blk hot", ident="n-torch")
heatmap("hm-train", 244, 247, 80, 60, 0.12)
node(36, 342, 300, 70, "torch.mps.Event", ["record(), then a publisher thread", "polls query() every 2 ms"], ident="n-event")
for y0, y1 in ((126, 150), (212, 236), (318, 342)):
    arrow(f"M 186 {y0} L 186 {y1 - 2}")
text(52, 524, "step 1,216 · loss 0.074", "mono small dim", ident="t-step")

# Shared memory: the control table and one view's segment with its three slots.
node(406, 64, 288, 62, "/session-ctl", ["view rows, the viewer's heartbeat"], ident="n-ctl")
rect(406, 150, 288, 262, "blk")
text(422, 174, "/session-v1", "mono")
text(422, 194, "header: published, claimed", "n")
for k, (state, label) in enumerate((("writing", "being written"), ("published", "published"), ("claimed", "claimed"))):
    y = SLOT_Y[k]
    rect(422, y, 196, 50, "slot " + state, 3, ident=f"s{k}")
    rect(624, y, 54, 50, "slot " + state, 3, ident=f"s{k}st")
    text(434, y + 22, f"slot {k}", "mono small")
    text(434, y + 40, label, "n", ident=f"s{k}lbl")
    heatmap(f"hm-s{k}", 546, y + 7, 64, 36, (0.15, 0.25, 0.35)[k])
    text(651, y + 30, "stats", "n dim", "middle")
# The ring's key: what each slot state means, in the slots' own colours.
for i, (state, line) in enumerate((("writing", "being written: the GPU copy is filling it"),
                                   ("published", "published: the newest snapshot, drawn next"),
                                   ("claimed", "claimed: the viewer is reading it for a frame"),
                                   ("", "free: the next copy can land here"))):
    y = 430 + i * 18
    rect(422, y - 9, 14, 10, "slot " + state if state else "slot", 2)
    text(444, y, line, "n")
text(422, 504, "none free: the new snapshot is dropped, never waited for", "n dim")
text(550, 526, "one MTLBuffer per process over the same pages", "n dim", "middle")

# Viewer process, top to bottom; the frame shows what the viewer draws.
node(764, 64, 300, 62, "ShmSource", ["heartbeat, newest completed slot"], ident="n-shm")
node(764, 168, 300, 62, "caliper_core_live_apply", ["colormap pass reads the slot in place"], "blk hot", ident="n-apply")
# The frame draws what the script's calls make: the weights as a heatmap, and graphics derived from tensors on the
# GPU: the weights' histogram, the hidden layer projected to 3-D points, the loss series.
rect(764, 256, 300, 270, "blk frame", ident="n-frame")
text(780, 280, "ImGui frame", "mono")
text(1048, 280, "seq 152", "n dim", "end", ident="t-seq")
text(780, 302, "fc1.weight", "n")
view = heatmap("hm-view", 780, 310, 132, 92, 0.35)
text(930, 302, "fc1.weight.hist", "n")
rect(930, 310, 118, 92, "spark", 2)
counts = [0] * 20
for v in view:
    counts[min(19, int((max(-1, min(1, v)) + 1) / 2 * 20))] += 1
out.append('<g id="hist">' + "".join(
    f'<rect x="{934 + i * 5.5:.1f}" y="{398 - 80 * n / max(counts):.1f}" width="4.6" height="{80 * n / max(counts):.1f}" class="bar"/>'
    for i, n in enumerate(counts)) + "</g>")
text(780, 420, "embedding", "n")
rect(780, 428, 132, 86, "spark", 2)
rng = random.Random("emb")
points = []
for i in range(40):
    k = i % 4
    cx, cy = (-1, 1, -1, 1)[k] * 0.55, (-1, -1, 1, 1)[k] * 0.5
    px, py = 846 + (cx + rng.uniform(-0.25, 0.25)) * 54, 471 + (cy + rng.uniform(-0.25, 0.25)) * 36
    points.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.4" class="pt c{k}"/>')
out.append('<g id="emb">' + "".join(points) + "</g>")
text(930, 420, "loss", "n")
rect(930, 428, 118, 86, "spark", 2)
out.append('<polyline id="loss-line" class="loss" points="'
           + " ".join(f"{934 + i * 4.5:.1f},{510 - 78 * (0.04 + 0.96 * math.exp(-i / 6)):.1f}" for i in range(25)) + '"/>')
for y0, y1 in ((126, 168), (230, 256)):
    arrow(f"M 914 {y0} L 914 {y1 - 2}")

# Across the processes: data in bronze, control faint. The script re-aims the slot arrows at the slot in play.
arrow("M 336 277 L 420 235", "arrow data", ident="a-copy")
text(368, 246, "GPU copy", "lbl", "middle")
arrow("M 336 377 L 420 299", ident="a-publish")
text(366, 372, "publish", "lbl", "middle")
arrow("M 764 95 L 696 95", ident="a-heartbeat")
text(730, 87, "heartbeat", "lbl", "middle")
arrow("M 406 95 L 338 95", ident="a-watched")
text(372, 87, "watched?", "lbl", "middle")
arrow("M 678 363 C 722 363, 722 199, 762 199", "arrow data", ident="a-read")
text(733, 290, "read", "lbl", "middle")
out.append('<g id="packet" class="packet"><rect x="-9" y="-6" width="18" height="12" rx="2"/></g>')

SVG = (f'<svg class="hw flow" viewBox="0 0 {W} {H}" data-slot-y="{",".join(str(y + 25) for y in SLOT_Y)}" '
       f'data-rows="{ROWS}" data-cols="{COLS}" role="img" aria-label="Python streaming, animated with illustrative '
       f'data: each training step computes the weights; caliper.show queues a GPU copy on torch\'s MPS stream into a '
       f'slot of a shared-memory ring and returns; a publisher marks the slot published when its event completes; the '
       f'viewer process, which writes a heartbeat the script checks, claims the newest published slot, reads it in '
       f'place with a colormap pass and draws the heatmap and the loss curve in the ImGui frame">\n      '
       + "\n      ".join(out) + "\n    </svg>")

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--write":
        page = Path(sys.argv[2])
        s = page.read_text()
        s, n = re.subn(r'<svg class="hw flow".*?</svg>', lambda m: SVG, s, flags=re.S)
        if n != 1:
            sys.exit(f"expected one streaming figure in {page}, found {n}")
        page.write_text(s)
    else:
        print(SVG)
