#!/usr/bin/env python3
"""Fail when the host names an instrument.

The host must work with any instrument and name none. This check collects, from
the example instruments in this repository:

  * the [instrument] id and name of every *.caliper.toml under examples/;
  * every window title passed as a string literal to ImGui::Begin(...) in the
    sources under examples/;

and fails if any of them appears as a whole C or C++ string literal in the
code of a source file under src/ or include/. Matching whole literals, not
substrings, keeps ordinary words in host strings from matching, and comments
are skipped, so a header may still cite an instrument id as an example.

Usage: check_host_names_no_instrument.py [--root <repository root>]
Exit status: 0 when the host names no instrument, 1 when it does, 2 on bad input.
"""

import argparse
import pathlib
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python older than 3.11
    tomllib = None

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".m", ".mm"}

# A C/C++ string literal on one line, with escapes. Raw strings are not used
# for window titles or ids, so they are not handled.
STRING_LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
BEGIN_TITLE = re.compile(r'\bImGui::Begin\s*\(\s*"((?:[^"\\\n]|\\.)*)"')


def manifest_names(path):
    """Returns the [instrument] id and name of one manifest."""
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        try:
            instrument = tomllib.loads(text).get("instrument", {})
        except tomllib.TOMLDecodeError as error:
            raise SystemExit(f"error: {path}: {error}") from None
        values = [instrument.get("id"), instrument.get("name")]
    else:
        values = []
        table = None
        for line in text.splitlines():
            line = line.strip()
            header = re.match(r"\[\s*([^\]]+?)\s*\]", line)
            if header:
                table = header.group(1)
                continue
            pair = re.match(r'(id|name)\s*=\s*"([^"]*)"', line)
            if table == "instrument" and pair:
                values.append(pair.group(2))
    return [v for v in values if isinstance(v, str) and v.strip()]


def strip_comments(text):
    """Replaces // and /* */ comments with spaces, keeping line breaks and
    leaving string and character literals untouched."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def source_files(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES)


def collect_needles(root):
    """Maps each instrument id, name and window title to where it came from."""
    needles = {}
    for top in ("examples",):
        folder = root / top
        if not folder.is_dir():
            continue
        for manifest in sorted(folder.rglob("*.caliper.toml")):
            for value in manifest_names(manifest):
                needles.setdefault(value, f"{manifest.relative_to(root)}")
        for source in source_files(folder):
            text = source.read_text(encoding="utf-8", errors="replace")
            for match in BEGIN_TITLE.finditer(text):
                title = match.group(1)
                if title.strip():
                    line = text.count("\n", 0, match.start()) + 1
                    needles.setdefault(title, f"{source.relative_to(root)}:{line}")
    return needles


def find_hits(root, needles):
    hits = []
    for top in ("src", "include"):
        for source in source_files(root / top):
            text = strip_comments(source.read_text(encoding="utf-8", errors="replace"))
            for number, line in enumerate(text.splitlines(), start=1):
                for match in STRING_LITERAL.finditer(line):
                    literal = match.group(1)
                    if literal in needles:
                        hits.append((source.relative_to(root), number, literal, needles[literal]))
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent,
                        help="repository root (default: the parent of scripts/)")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "src").is_dir():
        print(f"error: {root} has no src/ folder", file=sys.stderr)
        return 2

    needles = collect_needles(root)
    hits = find_hits(root, needles)
    if hits:
        print(f"The host names {len(hits)} instrument string(s); it must name none:")
        for path, number, literal, origin in hits:
            print(f'  {path}:{number}: "{literal}" (from {origin})')
        return 1
    print(f"The host names no instrument ({len(needles)} instrument ids, names and window titles checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
