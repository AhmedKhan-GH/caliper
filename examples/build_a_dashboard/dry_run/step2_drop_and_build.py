"""Step 2 of the walk-through in the hidden app: drop the files and build the
dashboard by hand.

    python examples/build_a_dashboard/dry_run/step2_drop_and_build.py [--out DIR] [--sizes 1440x900,2560x1440]

From the landing screen, ⌘K and the title in the command line, then F3 New
dashboard, which opens an empty spec in build mode. The two data files are
dropped on the window and listed in Sources. ⌘T opens the spec text under the
canvas. A series of loss and accuracy over step, a heatmap of one attention
head (two slices of the tensor, set in Properties) and a table of the metrics
are built in the builder pane, and the spec text follows each one.

Each moment is captured as step2-<n>-<what>-<W>x<H>.png in DIR/shots. The
script checks that the input script ran to its end and that the journal holds
the import from the person's origin and the build from the builder's.

What a person does differently: a hidden window receives no real drop, so the
`drop` step calls the function that the window's drop callback calls, with the
same paths a drop from Finder would give. Every other step is a click, a drag,
a key or typed text, as a person makes them.
"""

import argparse
import sys
from pathlib import Path

from hidden_app import REPO, SIZES, HiddenApp, build_steps, check_data, journal_summary, png_size, settle_and_shot

NAMES = ["landing", "new-dashboard", "sources", "spec-text-empty", "series", "heatmap-refused", "slices", "heatmap", "table"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / "build" / "build_a_dashboard" / "step2"))
    ap.add_argument("--sizes", default=",".join(SIZES), help="window sizes in points, comma separated")
    args = ap.parse_args()
    check_data()
    out = Path(args.out)
    failed = False
    for size in args.sizes.split(","):
        steps = build_steps(settle_and_shot) + ["quit"]
        app = HiddenApp(out / size, size, steps, out / "shots")
        rc = app.wait()
        print(f"{size}: exit {rc}, {'done' if app.ui_done() else 'stopped'}")
        for line in app.ui_lines():
            if "stopped" in line or "shot " in line:
                print(" ", line)
        shots = app.collect(2, NAMES)
        for s in shots:
            w, h = png_size(s)
            print(f"  {s.name} {w}x{h}")
        counts = journal_summary(out / size / "workspace")
        imported = counts.get(("ui", "data.import"), 0)
        built = sum(n for (origin, _), n in counts.items() if origin == "builder")
        print(f"  journal: data.import from ui {imported}, commands from builder {built}")
        ok = rc == 0 and app.ui_done() and len(shots) == len(NAMES) and imported == 1 and built > 10
        failed = failed or not ok
    print("step 2: " + ("FAILED" if failed else "ok"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
