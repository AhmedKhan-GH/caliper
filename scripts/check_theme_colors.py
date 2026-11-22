#!/usr/bin/env python3
"""Fail when the app theme holds a raw colour literal.

The theme reads every colour from the generated header of the shared design
tokens (sternbild_tokens.h, fetched by CMake). This check scans the theme
sources for colour literals written by hand: IM_COL32(...) and ImColor(...)
calls, ImVec4(...) constructions, and 0xRRGGBB or 0xAARRGGBB constants. Comments
and the contents of string literals are skipped. The macros IM_COL32_BLACK_TRANS
and IM_COL32_A_MASK name no colour and pass.

One region may hold literals: the lines between

    // BEGIN colours without a token
    // END colours without a token

in a theme file. It lists the colours that the tokens do not cover, apart from
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
    "src/host/ui/sternbild_theme.cpp",
    "src/host/ui/sternbild_fonts.h",
)

BEGIN = "colours without a token"
REGION_BEGIN = re.compile(r"//\s*BEGIN\s+" + BEGIN)
REGION_END = re.compile(r"//\s*END\s+" + BEGIN)

# A colour written by hand. IM_COL32_BLACK_TRANS and IM_COL32_A_MASK do not match
# because the call needs its opening parenthesis right after the macro name.
LITERAL = re.compile(
    r"\bIM_COL32\s*\("
    r"|\bImColor\s*\("
    r"|\bImVec4\s*\("
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


def scan(text):
    """Returns (line number, text) of each raw colour literal outside the allowed region."""
    hits = []
    in_block = False
    in_region = False
    for number, raw in enumerate(text.splitlines(), 1):
        if REGION_BEGIN.search(raw):
            in_region = True
            continue
        if REGION_END.search(raw):
            in_region = False
            continue
        code, in_block = code_only(raw, in_block)
        if not in_region and LITERAL.search(code):
            hits.append((number, raw.strip()))
    if in_region:
        hits.append((0, "a region 'BEGIN " + BEGIN + "' has no END"))
    return hits


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
    for name in names:
        path = root / name
        if not path.is_file():
            print(f"error: {name} does not exist", file=sys.stderr)
            return 2
        checked += 1
        for number, text in scan(path.read_text(encoding="utf-8")):
            found += 1
            print(f"{name}:{number}: raw colour: {text}")
    if found:
        print(f"The theme holds {found} raw colour literal(s). Read the value from "
              "sternbild_tokens.h, or list a colour without a token in the region "
              f"'{BEGIN}'.")
        return 1
    print(f"The theme holds no raw colour literal outside the generated header ({checked} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
