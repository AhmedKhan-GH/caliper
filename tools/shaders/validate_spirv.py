#!/usr/bin/env python3
"""Validates every generated SPIR-V program with spirv-val, for Vulkan 1.1.

    python3 tools/shaders/validate_spirv.py [--spirv-val PATH]

It reads the words of each src/host/gfx/shaders/generated/*_spv.h, writes them to a
temporary .spv file and runs spirv-val --target-env vulkan1.1 on it, as the Vulkan backend
will load them. spirv-val comes from SPIRV-Tools and is needed only here.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GENERATED = Path(__file__).resolve().parents[2] / "src" / "host" / "gfx" / "shaders" / "generated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirv-val", default=shutil.which("spirv-val") or "spirv-val")
    args = ap.parse_args()
    headers = sorted(GENERATED.glob("*_spv.h"))
    if not headers:
        print("no SPIR-V headers", file=sys.stderr)
        return 1
    bad = 0
    with tempfile.TemporaryDirectory(prefix="caliper-spirv-") as tmp:
        for h in headers:
            words = [int(w, 16) for w in re.findall(r"0x([0-9a-f]{8})", h.read_text())]
            if not words or words[0] != 0x07230203:
                print(f"{h.name}: not SPIR-V words", file=sys.stderr)
                bad += 1
                continue
            spv = Path(tmp) / (h.stem + ".spv")
            spv.write_bytes(b"".join(w.to_bytes(4, "little") for w in words))
            r = subprocess.run([args.spirv_val, "--target-env", "vulkan1.1", str(spv)], capture_output=True, text=True)
            print(f"{h.name}: {'valid' if r.returncode == 0 else 'INVALID'} ({len(words)} words)")
            if r.returncode != 0:
                print(r.stdout + r.stderr, file=sys.stderr)
                bad += 1
    print(f"{len(headers) - bad} of {len(headers)} SPIR-V programs pass spirv-val for Vulkan 1.1")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
