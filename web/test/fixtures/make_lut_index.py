#!/usr/bin/env python3
"""Write web/test/fixtures/lut_index.json: colormap indices of float32 values,
as the native reference map_f32_to_rgba8 gives them, for the browser views'
lutIndex test (web/test/colormap.spec.ts).

    python web/test/fixtures/make_lut_index.py --build cmake-build-release

It compiles lut_index_ref.cpp against the native build's libcaliper.a with the
system's clang++ and runs it on every case: ranges that increase, decrease,
have zero, infinite or NaN width or NaN ends, and values on each side of
index boundaries, NaN, the infinities, zeros, subnormals and the extremes.
Each case is [v, vmin, vmax, index], the floats as float32 bit patterns in hex.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def flt(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def f32(x: float) -> float:
    """x rounded to float32 (inf beyond its range)."""
    try:
        return flt(bits(x))
    except OverflowError:
        return float("inf") if x > 0 else float("-inf")


def up(b: int) -> int:
    """The next float32 towards +inf, by bit pattern (finite inputs)."""
    x = flt(b)
    if x != x or x == float("inf"):
        return b
    if b == 0x80000000:
        return 1
    return b + 1 if b < 0x80000000 else b - 1


def down(b: int) -> int:
    x = flt(b)
    if x != x or x == float("-inf"):
        return b
    if b == 0:
        return 0x80000001
    return b - 1 if b < 0x80000000 and b != 0 else b + 1


INF = float("inf")
NAN_BITS = [0x7FC00000, 0xFFC00000, 0x7F800001]           # quiet, negative, signalling
RANGES = [
    (0.0, 1.0), (-3.0, 5.0), (1.0, -1.0), (2.0, 2.0), (0.0, 255.0), (-1e-30, 1e-30),
    (0.0, INF), (-INF, 0.0), (-INF, INF), (INF, INF), (-INF, -INF), (INF, 0.0), (0.0, -INF),
    (0.0, "nan"), ("nan", 1.0), ("nan", "nan"),
    (-3e38, 3e38), (0.0, 3e38), (1e-40, 2e-40), (1.0, "next1"),
]
SPECIAL = [INF, -INF, 0.0, -0.0, 1e30, -1e30, 3.4e38, -3.4e38, 1e-45, -1e-45, 0.5, -0.5, 1.0, -1.0]


def range_bits(r) -> list[int]:
    out = []
    for e in r:
        if e == "nan":
            out.append(0x7FC00000)
        elif e == "next1":
            out.append(up(bits(1.0)))
        else:
            out.append(bits(f32(e)))
    return out


def cases() -> list[tuple[int, int, int]]:
    out = []
    for r in RANGES:
        lo_b, hi_b = range_bits(r)
        lo, hi = flt(lo_b), flt(hi_b)
        vals = set(NAN_BITS) | {bits(f32(x)) for x in SPECIAL} | {lo_b, hi_b}
        for k in (0, 1, 2, 63, 127, 128, 129, 254, 255, 256):
            b = f32(f32(lo) + f32(f32(f32(k - 0.5) / 255.0) * f32(hi - lo))) if lo == lo and hi == hi else float("nan")
            if b == b:
                vb = bits(b)
                vals |= {vb, up(vb), down(vb), up(up(vb)), down(down(vb))}
        out += [(v, lo_b, hi_b) for v in sorted(vals)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, default=ROOT / "cmake-build-release",
                    help="the native build folder holding libcaliper.a")
    args = ap.parse_args()
    lib = args.build.resolve() / "libcaliper.a"
    if not lib.is_file():
        print(f"make_lut_index: no {lib}; build the native target libcaliper first", file=sys.stderr)
        return 2
    cs = cases()
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "lut_index_ref"
        subprocess.run(["clang++", "-std=c++17", "-O2", str(HERE / "lut_index_ref.cpp"), str(lib), "-o", str(exe)],
                       check=True)
        text = "".join(f"{v:08x} {lo:08x} {hi:08x}\n" for v, lo, hi in cs)
        r = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True)
    idx = [int(x) for x in r.stdout.split()]
    assert len(idx) == len(cs), (len(idx), len(cs))
    table = [[f"{v:08x}", f"{lo:08x}", f"{hi:08x}", i] for (v, lo, hi), i in zip(cs, idx)]
    path = HERE / "lut_index.json"
    path.write_text(json.dumps({"source": "map_f32_to_rgba8 (src/host/tensor_bridge.cpp)", "cases": table},
                               separators=(",", ":")) + "\n")
    print(f"make_lut_index: {len(table)} cases into {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
