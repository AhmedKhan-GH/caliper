#!/usr/bin/env python3
"""Fail when the app theme holds a raw colour literal.

The theme reads every colour from the generated header of the shared design
tokens (sternbild_tokens.h, fetched by CMake). This check scans the theme
sources for colour literals written by hand: IM_COL32(...) and ImColor(...)
calls, ImVec4 and ImColor constructions with parentheses or braces, declarations
such as `ImVec4 c(1, 0, 0, 1)`, three or four numbers in braces of which one is
a float literal (`{0.9f, 0.8f, 0.7f, 1}`), and 0xRRGGBB or 0xAARRGGBB constants.
Comments and the contents of string literals are skipped. The macros
IM_COL32_BLACK_TRANS and IM_COL32_A_MASK name no colour and pass.

Exactly one region of the theme may hold literals: the lines between

    // BEGIN colours without a token
    // END colours without a token

Each marker must stand on a line of its own. A second region, a marker after
code on its line, an END without its BEGIN and a BEGIN without its END are
findings. The region lists the colours that the tokens do not cover, apart from
everything generated.

Usage: check_theme_colors.py [--root <repository root>] [files...]
Without files the check reads the theme sources (THEME_FILES below).
Exit status: 0 when the theme holds no raw colour, 1 when it does, 2 on bad input.
"""

import argparse
import pathlib
import re
import sys

THEME_FILES = (
    "src/host/ui/sternbild_theme.h",
    "src/host/ui/sternbild_theme_colors.inc",
    "src/host/ui/sternbild_theme.cpp",
    "src/host/ui/sternbild_fonts.h",
)

BEGIN = "colours without a token"
REGION_BEGIN = re.compile(r"^\s*//\s*BEGIN\s+" + BEGIN + r"\s*$")
REGION_END = re.compile(r"^\s*//\s*END\s+" + BEGIN + r"\s*$")
MARKER = re.compile(r"\b(?:BEGIN|END)\s+" + BEGIN)

# A colour written by hand. IM_COL32_BLACK_TRANS and IM_COL32_A_MASK do not match
# because the call needs its opening parenthesis right after the macro name.
LITERAL = re.compile(
    r"\bIM_COL32\s*\("
    r"|\bImColor\s*\("
    r"|\b(?:ImVec4|ImColor)\s*[({]"
    r"|\b(?:ImVec4|ImColor)\s+\w+\s*[({]\s*[-+]?\.?\d"
    r"|\bIM_COL32_(?:WHITE|BLACK)\b"
    r"|\b0[xX][0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?[uUlL]*\b"
)


def code_only(line, in_block):
    """Returns the line with comments and string contents blanked, and the block-comment state."""
    out = []
    i = 0
    n = len(line)
    while i < n:
        two = line[i:i + 2]
        if in_block:
            if two == "*/":
                in_block = False
                i += 2
            else:
                i += 1
            continue
        if two == "/*":
            in_block = True
            i += 2
            continue
        if two == "//":
            break
        ch = line[i]
        if ch in "\"'":
            quote = ch
            out.append(quote)
            i += 1
            while i < n and line[i] != quote:
                i += 2 if line[i] == "\\" else 1
            out.append(quote)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), in_block


# Three or four numbers in braces, an initializer of a colour when one is a float literal.
NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?"
NUMBER_GROUP = re.compile(r"\{\s*(" + NUMBER + r"(?:\s*,\s*" + NUMBER + r"){2,3})\s*\}")


def float_group(code):
    """Whether the code holds three or four numbers in braces, one of them a float literal."""
    for match in NUMBER_GROUP.finditer(code):
        if any(re.search(r"[.eEfF]", n) for n in match.group(1).split(",")):
            return True
    return False


def scan(text, regions_before=0):
    """Returns (line number, text) of each raw colour literal outside the allowed region, and of
    each misplaced marker. `regions_before` counts the regions of the files read before this one."""
    hits = []
    in_block = False
    in_region = False
    regions = regions_before
    for number, raw in enumerate(text.splitlines(), 1):
        if REGION_BEGIN.search(raw):
            regions += 1
            if in_region or regions > 1:
                hits.append((number, "only one region 'BEGIN " + BEGIN + "' is allowed: " + raw.strip()))
            in_region = True
            continue
        if REGION_END.search(raw):
            if not in_region:
                hits.append((number, "an END without its BEGIN: " + raw.strip()))
            in_region = False
            continue
        if MARKER.search(raw):
            hits.append((number, "a marker must stand on a line of its own: " + raw.strip()))
        code, in_block = code_only(raw, in_block)
        if not in_region and (LITERAL.search(code) or float_group(code)):
            hits.append((number, "raw colour: " + raw.strip()))
    if in_region:
        hits.append((0, "a region 'BEGIN " + BEGIN + "' has no END"))
    return hits


def count_regions(text):
    """The number of regions that a file opens."""
    return sum(1 for raw in text.splitlines() if REGION_BEGIN.search(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent,
                        help="repository root (default: the parent of scripts/)")
    parser.add_argument("files", nargs="*", help="files to read, relative to the root")
    args = parser.parse_args()
    root = args.root.resolve()
    names = args.files or THEME_FILES
    found = 0
    checked = 0
    regions = 0
    for name in names:
        path = root / name
        if not path.is_file():
            print(f"error: {name} does not exist", file=sys.stderr)
            return 2
        checked += 1
        text = path.read_text(encoding="utf-8")
        hits = scan(text, regions)
        regions += count_regions(text)
        for number, text in hits:
            found += 1
            print(f"{name}:{number}: {text}")
    if found:
        print(f"The theme check has {found} finding(s). Read a colour from "
              "sternbild_tokens.h, or list a colour without a token in the region "
              f"'{BEGIN}'.")
        return 1
    if not args.files:
        scripts_dir = str(pathlib.Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import shared_colors  # noqa: E402
        try:
            drift = shared_colors.check_files(root)
        except Exception as exc:  # noqa: BLE001
            print(f"error: shared colour check failed: {exc}", file=sys.stderr)
            return 2
        if drift:
            for item in drift:
                print(f"shared_colors: {item}", file=sys.stderr)
            print("Run `python3 scripts/shared_colors.py --write` to regenerate the copies.",
                  file=sys.stderr)
            return 1
    print(f"The theme holds no raw colour literal outside the generated header ({checked} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
