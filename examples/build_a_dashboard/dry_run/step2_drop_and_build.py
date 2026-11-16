"""Step 2 of the walk-through in the hidden app: drop the files and build the
dashboard by hand.

    python examples/build_a_dashboard/dry_run/step2_drop_and_build.py [--out DIR] [--work DIR]
        [--size 1440x900[,2560x1440]] [--caliper APP] [--workspace DIR --app-data DIR]

From the landing screen, ⌘K and the title in the command line, then F3 New
dashboard, which opens an empty spec in build mode. The two data files are
dropped on the window and listed in Sources. ⌘T opens the spec text under the
canvas. A series of loss and accuracy over step, a heatmap of one attention
head (two slices of the tensor, set in Properties) and a table of the metrics
are built in the builder pane, and the spec text follows each one.

Each moment is captured as step2-<n>-<what>-<W>x<H>.png in --out. The script
checks that the input script ran to its end and that the journal holds the
import from the person's origin and the build from the builder's.

With --workspace and --app-data, as in run_all.py --story, the app works in
that workspace, and the dashboard is saved there with ⌘S at the end, as the
spec applet training-run that steps 3, 5 and 6 continue with; the saved spec
must equal expected.spec.toml beside this script. Otherwise the workspace is a
new one in the step's own folder. The options are those of hidden_app.py.

What a person does differently: a hidden window receives no real drop, so the
`drop` step calls the function that the window's drop callback calls, with the
same paths a drop from Finder would give. Every other step is a click, a drag,
a key or typed text, as a person makes them.
"""

import argparse
import sys
from pathlib import Path

from hidden_app import (EXPECTED_SPEC, FAILED, INTERRUPTED, MISSING, OK, Children, HiddenApp, Stop, add_options,
                        build_steps, caliper_problem, data_problem, fresh_folder, journal_summary, png_size,
                        saved_spec, settle_and_shot, work_folder)

NAMES = ["landing", "new-dashboard", "sources", "spec-text-empty", "series", "heatmap-refused", "slices", "heatmap", "table"]


def run(children, args, size, work, out):
    story = bool(args.workspace)
    root = fresh_folder(work, size)
    workspace = Path(args.workspace).resolve() if story else root / "workspace"
    steps = build_steps(settle_and_shot)
    if story:
        steps += ["key cmd+s", "wait 20"]          # saved as the applet training-run
    app = HiddenApp(children, args.caliper, root, size, steps + ["quit"], workspace=workspace,
                    app_data=Path(args.app_data).resolve() if args.app_data else None)
    rc = app.wait()
    print(f"{size}: exit {rc}, {'done' if app.ui_done() else 'stopped'}")
    for line in app.ui_lines():
        if "stopped" in line or "shot " in line:
            print(" ", line)
    shots = app.collect(2, NAMES, out)
    for s in shots:
        w, h = png_size(s)
        print(f"  {s.name} {w}x{h}")
    counts = journal_summary(workspace)
    imported = counts.get(("ui", "data.import"), 0)
    built = sum(n for (origin, _), n in counts.items() if origin == "builder")
    print(f"  journal: data.import from ui {imported}, commands from builder {built}")
    ok = rc == 0 and app.ui_done() and len(shots) == len(NAMES) and imported == 1 and built > 10
    if story:
        saved = saved_spec(workspace)
        same = saved.is_file() and saved.read_text() == EXPECTED_SPEC.read_text()
        print(f"  saved spec: {'equals expected.spec.toml' if same else 'missing or differs from expected.spec.toml'}")
        ok = ok and same
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_options(ap, sizes=True)
    args = ap.parse_args()
    sizes = args.size.split(",")
    for problem in (data_problem(), caliper_problem(args.caliper),
                    "--workspace and --app-data go together" if bool(args.workspace) != bool(args.app_data) else "",
                    "one --size with --workspace" if args.workspace and len(sizes) > 1 else ""):
        if problem:
            print(f"step 2: {problem}", file=sys.stderr)
            return MISSING
    out = Path(args.out).resolve()
    failed = False
    with Children() as children:
        try:
            work = work_folder(args, 2)
            print(f"step 2: files in {work}")
            for size in sizes:
                failed = not run(children, args, size, work, out) or failed
        except Stop as why:
            print(f"step 2: interrupted by {why}")
            return INTERRUPTED
    print("step 2: " + ("FAILED" if failed else "ok"))
    return FAILED if failed else OK


if __name__ == "__main__":
    sys.exit(main())
