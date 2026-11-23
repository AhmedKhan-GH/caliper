#!/usr/bin/env python3
"""Tests of the shader generator, on a copy of the shader sources in a temporary folder.

    python3 tools/shaders/test_generate.py --driver PATH

It checks that the committed outputs pass --check, that a hand-written output is left alone
and included by its backend's table, that a WGSL source that does not compile fails the
check with Naga's message, and that a changed, missing or extra generated file fails it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GENERATE = REPO / "tools" / "shaders" / "generate.py"
REL = Path("src") / "host" / "gfx" / "shaders"


def run(driver: str, root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GENERATE), "--driver", driver, "--root", str(root), *extra],
                          capture_output=True, text=True)


def fresh(tmp: Path, name: str) -> Path:
    root = tmp / name
    shutil.copytree(REPO / REL, root / REL)
    return root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    driver = ap.parse_args().driver
    failures = []

    def expect(cond: bool, what: str) -> None:
        print(("ok    " if cond else "FAIL  ") + what)
        if not cond:
            failures.append(what)

    with tempfile.TemporaryDirectory(prefix="caliper-shaders-test-") as t:
        tmp = Path(t)

        root = fresh(tmp, "clean")
        r = run(driver, root, "--check")
        expect(r.returncode == 0, "the committed outputs match their sources")

        # A hand-written MSL for fill_texture: not generated, included by the Metal table.
        root = fresh(tmp, "hand")
        manifest = root / REL / "manifest.json"
        m = json.loads(manifest.read_text())
        prog = next(p for p in m["programs"] if p["name"] == "fill_texture")
        prog["outputs"] = ["wgsl", "spirv"]
        prog["hand_written"] = {"msl": "gfx/shaders/hand/fill_texture_msl.h"}
        manifest.write_text(json.dumps(m, indent=2))
        hand = root / "src" / "host" / "gfx" / "shaders" / "hand" / "fill_texture_msl.h"
        hand.parent.mkdir(parents=True)
        hand.write_text("// hand-written\n")
        r = run(driver, root)
        gen = root / REL / "generated"
        expect(r.returncode == 0, "a program with a hand-written MSL generates")
        expect(not (gen / "fill_texture_msl.h").exists(), "the hand-written output is not generated")
        expect('#include "gfx/shaders/hand/fill_texture_msl.h"' in (gen / "programs_metal.h").read_text(),
               "the Metal table includes the hand-written header")
        expect(hand.read_text() == "// hand-written\n", "the hand-written file is left alone")
        expect(run(driver, root, "--check").returncode == 0, "the check passes with the hand-written output")
        hand.unlink()
        r = run(driver, root, "--check")
        expect(r.returncode != 0 and "does not exist" in r.stderr, "a missing hand-written file fails")

        # A WGSL source that does not compile.
        root = fresh(tmp, "broken")
        src = root / REL / "compute" / "fill_texture.wgsl"
        src.write_text(src.read_text() + "\nfn broken() -> u32 { return 1.5; }\n")
        r = run(driver, root, "--check")
        expect(r.returncode != 0 and "fill_texture does not compile" in r.stderr,
               "a WGSL source that does not compile fails the check")
        print("      " + "\n      ".join(r.stderr.strip().splitlines()[:4]))

        # A changed, a missing and an extra generated file.
        root = fresh(tmp, "stale")
        gen = root / REL / "generated"
        (gen / "uniforms.h").write_text((gen / "uniforms.h").read_text() + "// edited\n")
        (gen / "fill_texture_spv.h").unlink()
        (gen / "stray.h").write_text("\n")
        r = run(driver, root, "--check")
        expect(r.returncode != 0 and "differs: uniforms.h" in r.stderr and "missing: fill_texture_spv.h" in r.stderr
               and "not generated: stray.h" in r.stderr, "changed, missing and extra files fail the check")

    print(f"{len(failures)} failed" if failures else "all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
