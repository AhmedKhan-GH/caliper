#!/usr/bin/env python3
"""The Python streaming figure on the Caliper landing page (site/index.html).

    python3 scripts/site_streaming_figure.py [--write site/index.html]

What happens between a training script and the viewer when the script calls caliper.show(): the snapshot is a
GPU copy on torch's own stream into POSIX shared memory, and the viewer draws the newest completed slot from the
same pages. Facts from docs/wiki/explanation/live-streaming.md (sessions, snapshots, sampling, sync).
"""
import re
import sys
from pathlib import Path

W, H = 1100, 452
out = []


def rect(x, y, w, h, cls, r=4):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>')


def text(x, y, s, cls, anchor="start"):
    out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>')


def node(x, y, w, h, title, lines, cls="blk"):
    rect(x, y, w, h, cls)
    text(x + 16, y + 24, title, "mono")
    for i, l in enumerate(lines):
        text(x + 16, y + 44 + i * 18, l, "n")


def arrow(d, cls="arrow"):
    out.append(f'<path d="{d}" class="{cls}" marker-end="url(#{"tip-data" if "data" in cls else "tip"})"/>')


out.append('<defs>'
           '<marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,1 L10,5 L0,9 Z" class="tip"/></marker>'
           '<marker id="tip-data" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,1 L10,5 L0,9 Z" class="tip-data"/></marker>'
           '</defs>')

# The three columns: the training process, the shared pages, the viewer process.
for x, w, title, sub in ((16, 340, "TRAINING PROCESS", "Python, torch on MPS"),
                         (386, 328, "SHARED MEMORY", "POSIX segments, mode 0600"),
                         (744, 340, "VIEWER PROCESS", "Caliper app or caliper-viewer")):
    rect(x, 44, w, 392, "pkg", 10)
    text(x + 18, 28, title, "k")
    text(x + w - 18, 28, sub, "n dim", "end")

# Training process, top to bottom.
node(36, 64, 300, 62, "caliper.show(W)", ["60 Hz rate limit, then the watcher check"])
node(36, 150, 300, 62, "SnapshotRing", ["a free slot of three, or the snapshot drops"])
node(36, 236, 300, 82, "torch MPS stream", ["slot.copy_(W)", "torch.aminmax(W, out=stats)"], "blk hot")
node(36, 342, 300, 70, "torch.mps.Event", ["record(), then a publisher thread", "polls query() every 2 ms"])
for y0, y1 in ((126, 150), (212, 236), (318, 342)):
    arrow(f"M 186 {y0} L 186 {y1 - 2}")

# Shared memory: the control table and one view's segment with its three slots.
node(406, 64, 288, 62, "/session-ctl", ["view rows, the viewer's heartbeat"])
rect(406, 150, 288, 262, "blk")
text(422, 174, "/session-v1", "mono")
text(422, 194, "header: published, claimed", "n")
for i, (state, label) in enumerate((("writing", "being written"), ("published", "published"), ("claimed", "claimed by the viewer"))):
    y = 210 + i * 64
    rect(422, y, 196, 50, "slot " + state, 3)
    rect(624, y, 54, 50, "slot " + state, 3)
    text(434, y + 22, f"slot {i}", "mono small")
    text(434, y + 40, label, "n")
    text(651, y + 30, "stats", "n dim", "middle")

# Viewer process, top to bottom.
node(764, 64, 300, 62, "ShmSource", ["heartbeat, newest completed slot"])
node(764, 168, 300, 62, "caliper_core_live_apply", ["colormap pass reads the slot in place"], "blk hot")
node(764, 272, 300, 62, "ImGui frame", ["up to 60 fps"], "blk frame")
for y0, y1 in ((126, 168), (230, 272)):
    arrow(f"M 914 {y0} L 914 {y1 - 2}")

# Across the processes: data in bronze, control faint.
arrow("M 336 270 L 420 235", "arrow data")                 # the GPU copy lands in the slot being written
text(372, 240, "GPU copy", "lbl", "middle")
arrow("M 336 377 L 420 300")                               # the publisher marks a completed slot published
text(366, 372, "publish", "lbl", "middle")
arrow("M 764 95 L 696 95")                                 # the viewer writes its heartbeat
text(730, 87, "heartbeat", "lbl", "middle")
arrow("M 406 95 L 338 95")                                 # show() reads it: is anyone watching
text(372, 87, "watched?", "lbl", "middle")
arrow("M 678 299 C 722 299, 722 199, 762 199", "arrow data")  # the viewer reads the published slot in place
text(731, 262, "read", "lbl", "middle")
text(550, 432, "one MTLBuffer per process over the same pages", "n dim", "middle")

SVG = (f'<svg class="hw flow" viewBox="0 0 {W} {H}" role="img" aria-label="Python streaming: caliper.show queues a GPU copy '
       f'on torch\'s MPS stream into a slot of a shared-memory ring and returns; a publisher marks the slot published when '
       f'its event completes; the viewer process, which writes a heartbeat the script checks, reads the newest published '
       f'slot in place with a colormap pass and draws it in the ImGui frame">\n      ' + "\n      ".join(out) + "\n    </svg>")

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
