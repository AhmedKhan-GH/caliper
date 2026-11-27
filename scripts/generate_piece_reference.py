#!/usr/bin/env python3
"""Generate or verify docs/reference/piece-format.md from piece headers, checks, and schema.

Reads:
  - src/builder/include/caliper_builder/spec/piece.h
  - src/builder/include/caliper_builder/programs/pieces.h
  - src/builder/programs/piece_checks.cpp
  - src/builder/spec/schema/caliper-spec-v1.schema.json

In --check mode (the default), fails when any source cannot be parsed, when
piece_checks.cpp emits an undocumented check code, or when
docs/reference/piece-format.md is out of date.

Usage:
  generate_piece_reference.py [--root <repo_root>] [--check | --write]
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import sys

PIECE_H_REL = "src/builder/include/caliper_builder/spec/piece.h"
PIECES_H_REL = "src/builder/include/caliper_builder/programs/pieces.h"
CHECKS_CPP_REL = "src/builder/programs/piece_checks.cpp"
SCHEMA_REL = "src/builder/spec/schema/caliper-spec-v1.schema.json"
OUTPUT_REL = "docs/reference/piece-format.md"

LIMIT_RE = re.compile(
    r"^\s*inline constexpr\s+[A-Za-z0-9_:]+\s+(k[A-Za-z0-9_]+)\s*=\s*([^;]+);(?:\s*//\s*(.*))?$"
)
CODE_RE = re.compile(
    r'^\s*inline constexpr const char\*\s+(k[A-Za-z0-9_]+)\s*=\s*"([^"]+)";(?:\s*//\s*(.*))?$'
)
ENUM_RE = re.compile(r"^\s*enum class\s+([A-Za-z0-9_]+)\s*\{\s*([^}]+)\};")
ARRAY_STR_RE = re.compile(
    r'inline constexpr std::array<const char\*,\s*\d+>\s+(k[A-Za-z0-9_]+)\{([^}]+)\};'
)
FAIL_CALL_RE = re.compile(r"\bfail\(\s*(k[A-Za-z0-9_]+)\b")


def md_cell(text: str) -> str:
    escaped = html.escape(" ".join(str(text).split()), quote=False)
    return escaped.replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")


def eval_cpp_expr(expr: str) -> str:
    cleaned = expr.strip().replace("'", "")
    m_shift = re.fullmatch(r"(\d+)(?:u|ull|ul)?\s*<<\s*(\d+)", cleaned)
    if m_shift:
        val = int(m_shift.group(1)) << int(m_shift.group(2))
        if val % (1 << 20) == 0:
            return "%s (%d MiB)" % ("{:,}".format(val), val >> 20)
        return "{:,}".format(val)
    m_int = re.fullmatch(r"(\d+)(?:u|ull|ul)?", cleaned)
    if m_int:
        return "{:,}".format(int(m_int.group(1)))
    m_flt = re.fullmatch(r"([0-9]+\.[0-9]+)", cleaned)
    if m_flt:
        return m_flt.group(1)
    return cleaned


def parse_piece_h(text: str) -> tuple[list[tuple[str, str, str]], dict[str, list[str]], list[tuple[str, list[str]]]]:
    limits: list[tuple[str, str, str]] = []
    in_limits = False
    pending_comment: list[str] = []
    for line in text.splitlines():
        if "namespace piece_limits {" in line:
            in_limits = True
            pending_comment.clear()
            continue
        if in_limits and line.strip().startswith("}"):
            in_limits = False
            continue
        if in_limits:
            stripped = line.strip()
            if stripped.startswith("//"):
                pending_comment.append(stripped[2:].strip())
                continue
            m = LIMIT_RE.match(line)
            if m:
                note = (m.group(3) or "").strip()
                if not note and pending_comment:
                    note = " ".join(pending_comment)
                limits.append((m.group(1), eval_cpp_expr(m.group(2)), note))
                pending_comment.clear()
            elif not stripped:
                pending_comment.clear()

    arrays: dict[str, list[str]] = {}
    for m in ARRAY_STR_RE.finditer(text):
        vals = re.findall(r'"([^"]+)"', m.group(2))
        arrays[m.group(1)] = vals

    enums: list[tuple[str, list[str]]] = []
    for m in ENUM_RE.finditer(text):
        ename = m.group(1)
        members = [x.strip() for x in m.group(2).split(",") if x.strip()]
        members = ["inline" if x == "inline_piece" else x for x in members]
        enums.append((ename, members))

    return limits, arrays, enums


def parse_pieces_h(text: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    codes: list[tuple[str, str, str]] = []
    bindings: list[tuple[str, str, str]] = []
    pending_comment: list[str] = []
    in_bindings = False
    for line in text.splitlines():
        stripped = line.strip()
        if "bind groups of a composed module" in stripped:
            in_bindings = True
            pending_comment.clear()
            continue
        if in_bindings and stripped.startswith("// ----"):
            in_bindings = False
        if stripped.startswith("//") and not stripped.startswith("// ----"):
            pending_comment.append(stripped[2:].strip())
            continue
        cm = CODE_RE.match(line)
        if cm:
            note = (cm.group(3) or "").strip()
            if not note and pending_comment:
                note = " ".join(pending_comment)
            codes.append((cm.group(1), cm.group(2), note))
            pending_comment.clear()
            continue
        if in_bindings:
            lm = LIMIT_RE.match(line)
            if lm:
                note = (lm.group(3) or "").strip()
                bindings.append((lm.group(1), eval_cpp_expr(lm.group(2)), note))
                pending_comment.clear()
                continue
        if not stripped:
            pending_comment.clear()
    return codes, bindings


def format_schema_type(prop: dict) -> str:
    if "$ref" in prop:
        return "`%s`" % prop["$ref"].split("/")[-1]
    t = prop.get("type")
    if isinstance(t, list):
        t_str = " | ".join("`%s`" % x for x in t)
    elif isinstance(t, str):
        if t == "array":
            items = prop.get("items", {})
            if "$ref" in items:
                t_str = "array of `%s`" % items["$ref"].split("/")[-1]
            elif "type" in items:
                t_str = "array of `%s`" % items["type"]
            else:
                t_str = "array"
        else:
            t_str = "`%s`" % t
    else:
        t_str = "object"
    return t_str


def format_schema_constraints(prop: dict) -> str:
    parts: list[str] = []
    if "enum" in prop:
        parts.append("enum: " + ", ".join("`%s`" % v for v in prop["enum"]))
    if "pattern" in prop:
        parts.append("pattern `%s`" % prop["pattern"])
    if "minimum" in prop and "maximum" in prop:
        parts.append("range `%s` to `%s`" % (prop["minimum"], prop["maximum"]))
    elif "minimum" in prop:
        parts.append("min `%s`" % prop["minimum"])
    elif "maximum" in prop:
        parts.append("max `%s`" % prop["maximum"])
    if "minLength" in prop and "maxLength" in prop:
        parts.append("length `%s` to `%s`" % (prop["minLength"], prop["maxLength"]))
    elif "maxLength" in prop:
        parts.append("max length `%s`" % prop["maxLength"])
    if "minItems" in prop and "maxItems" in prop:
        parts.append("items `%s` to `%s`" % (prop["minItems"], prop["maxItems"]))
    elif "maxItems" in prop:
        parts.append("max items `%s`" % prop["maxItems"])
    items = prop.get("items")
    if isinstance(items, dict):
        if "pattern" in items:
            parts.append("item pattern `%s`" % items["pattern"])
        elif "maxLength" in items:
            parts.append("item max length `%s`" % items["maxLength"])
    if "description" in prop:
        parts.append(str(prop["description"]).rstrip("."))
    return md_cell("; ".join(parts)) if parts else "None"


SCHEMA_SECTIONS = [
    ("piece", "Manifest Root Keys", "Top-level keys in `piece.toml` or an inline `[[piece]]` table."),
    ("piece_input", "Input Table Keys", "Keys of each `[[inputs]]` entry for derive tensors and draw buffer inputs."),
    ("piece_param", "Parameter Table Keys", "Keys of each `[[params]]` entry in the generated `Params` uniform struct."),
    ("piece_resource", "Resource Table Keys", "Keys of each `[[resources]]` entry for published or private buffers and textures."),
    ("piece_derive", "Derive Table Keys", "Keys of the `[derive]` compute table that runs in the producer on `caliper.step()`."),
    ("piece_draw", "Draw Table Keys", "Keys of the `[draw]` table that runs each viewer frame on Metal, Vulkan, or WebGPU."),
    ("piece_draw_pass", "Draw Pass Keys", "Keys of each `[[draw.passes]]` compute or render pass entry."),
]


def load_and_check(root: pathlib.Path) -> tuple[dict, list[str]]:
    problems: list[str] = []
    for rel in (PIECE_H_REL, PIECES_H_REL, CHECKS_CPP_REL, SCHEMA_REL):
        if not (root / rel).is_file():
            problems.append("%s: file does not exist" % rel)
    if problems:
        return {}, problems

    piece_h = (root / PIECE_H_REL).read_text(encoding="utf-8")
    pieces_h = (root / PIECES_H_REL).read_text(encoding="utf-8")
    checks_cpp = (root / CHECKS_CPP_REL).read_text(encoding="utf-8")
    schema = json.loads((root / SCHEMA_REL).read_text(encoding="utf-8"))

    limits, arrays, enums = parse_piece_h(piece_h)
    codes, bindings = parse_pieces_h(pieces_h)
    if not limits:
        problems.append("%s: no piece_limits constants parsed" % PIECE_H_REL)
    if not codes:
        problems.append("%s: no check/error codes parsed" % PIECES_H_REL)

    known_code_consts = {cname for cname, _, _ in codes}
    used_in_checks = set(FAIL_CALL_RE.findall(checks_cpp))
    unknown_consts = sorted(used_in_checks - known_code_consts)
    if unknown_consts:
        problems.append(
            "%s: fail() uses code constants missing from %s: %s"
            % (CHECKS_CPP_REL, PIECES_H_REL, ", ".join(unknown_consts))
        )

    defs = schema.get("$defs") or {}
    for def_key, _, _ in SCHEMA_SECTIONS:
        if def_key not in defs:
            problems.append("%s: missing $defs.%s" % (SCHEMA_REL, def_key))

    parsed = {
        "limits": limits,
        "arrays": arrays,
        "enums": enums,
        "codes": codes,
        "bindings": bindings,
        "used_in_checks": used_in_checks,
        "defs": defs,
    }
    return parsed, problems


def render_markdown(parsed: dict) -> str:
    limits = parsed["limits"]
    arrays = parsed["arrays"]
    enums = parsed["enums"]
    codes = parsed["codes"]
    bindings = parsed["bindings"]
    used_in_checks = parsed["used_in_checks"]
    defs = parsed["defs"]

    piece_files = ", ".join("`%s`" % f for f in arrays.get("kPieceFiles", []))
    tensor_dtypes = ", ".join("`%s`" % d for d in arrays.get("kTensorDtypes", []))
    tex_formats = ", ".join("`%s`" % f for f in arrays.get("kTextureFormats", []))

    lines = [
        "<!-- Generated by scripts/generate_piece_reference.py. Do not edit by hand. -->",
        "",
        "# Piece Format Reference",
        "",
        "A piece defines a portable unit of programmable graphics with `piece.toml` and up to three WGSL files (%s)." % piece_files,
        "Library pieces, workspace pieces, and inline `[[piece]]` entries in a spec share one schema, one manifest validator, and one SHA-256 content hash.",
        "The script `scripts/generate_piece_reference.py` generates this page from `piece.h`, `pieces.h`, `piece_checks.cpp`, and `caliper-spec-v1.schema.json`.",
        "Continuous integration runs `python scripts/generate_piece_reference.py --check` and fails when this page drifts from those sources.",
        "Run `python3 scripts/generate_piece_reference.py --write` after any change to piece limits, manifest schema keys, or check codes.",
        "",
        "## Schema Definitions",
        "",
        "The canonical JSON Schema in `src/builder/spec/schema/caliper-spec-v1.schema.json` validates piece keys, types, patterns, and array bounds.",
        "Tensor inputs accept element types %s, or distinct combinations joined by `|`." % tensor_dtypes,
        "Texture resources accept formats %s." % tex_formats,
        "See [Pieces](../agents/pieces.md) for authoring workflows and [MCP Command Reference](mcp-commands.md#piece-library-commands) for `piece.*` tools.",
        "",
    ]

    for def_key, heading, intro in SCHEMA_SECTIONS:
        dobj = defs[def_key]
        req = set(dobj.get("required") or [])
        props = dobj.get("properties") or {}
        lines.extend(
            [
                "### %s" % heading,
                "",
                intro,
                "",
                "| Key | Type | Required | Constraints |",
                "|---|---|---|---|",
            ]
        )
        for pname, pspec in props.items():
            req_str = "Yes" if pname in req else "No"
            lines.append(
                "| `%s` | %s | %s | %s |"
                % (pname, format_schema_type(pspec), req_str, format_schema_constraints(pspec))
            )
        lines.append("")

    lines.extend(
        [
            "## Manifest Enumerations",
            "",
            "The value types in `src/builder/include/caliper_builder/spec/piece.h` define fixed string enumerations for piece manifests and program pins.",
            "",
            "| Enumeration | Allowed Values |",
            "|---|---|",
        ]
    )
    for ename, vals in enums:
        lines.append("| `%s` | %s |" % (ename, ", ".join("`%s`" % v for v in vals)))
    lines.append("")

    lines.extend(
        [
            "## Quantitative Limits",
            "",
            "Every limit in `caliper_builder::spec::piece_limits` (`src/builder/include/caliper_builder/spec/piece.h`) bounds memory, pass counts, and WebGPU bindings.",
            "The manifest validator in `src/builder/programs/piece_checks.cpp` checks every product in 64-bit arithmetic before buffer allocation.",
            "",
            "| Constant | Limit | Purpose |",
            "|---|---:|---|",
        ]
    )
    for cname, val_str, note in limits:
        lines.append("| `%s` | `%s` | %s |" % (cname, val_str, md_cell(note or "Piece limit")))
    lines.append("")

    lines.extend(
        [
            "## Composed Module Bindings",
            "",
            "Piece WGSL files never declare `@group` or `@binding` attributes because the composer generates all bindings from `piece.toml`.",
            "The constants in `src/builder/include/caliper_builder/programs/pieces.h` fix the bind group and prelude layout across Metal, Vulkan, and WebGPU.",
            "",
            "| Constant | Value | Binding Role |",
            "|---|---:|---|",
        ]
    )
    for cname, val_str, note in bindings:
        lines.append("| `%s` | `%s` | %s |" % (cname, val_str, md_cell(note or "Composer constant")))
    lines.append("")

    lines.extend(
        [
            "## Validation and Host Refusal Codes",
            "",
            "The composer, manifest validator (`piece_checks.cpp`), WGSL reflection checker, and host runtimes return structured `piece.*` and `program.*` codes.",
            "",
            "| Code | Constant | Stage | Condition |",
            "|---|---|---|---|",
        ]
    )
    for cname, code_str, note in codes:
        stage = "Manifest check" if cname in used_in_checks else (
            "Host runtime" if code_str.startswith("program.resources_") or code_str.startswith("program.resource_") or code_str.startswith("program.draw_") or code_str == "program.browser_refused"
            else "Composer and reflection"
        )
        lines.append("| `%s` | `%s` | %s | %s |" % (code_str, cname, stage, md_cell(note)))
    lines.append("")

    return "\n".join(lines)


def run(root: pathlib.Path, write: bool) -> tuple[list[str], str]:
    parsed, problems = load_and_check(root)
    if problems:
        return problems, ""
    rendered = render_markdown(parsed)
    out_path = root / OUTPUT_REL
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        return [], rendered
    if not out_path.is_file():
        return ["%s: generated file does not exist (run scripts/generate_piece_reference.py --write)" % OUTPUT_REL], rendered
    if out_path.read_text(encoding="utf-8") != rendered:
        return ["%s: out of date (run scripts/generate_piece_reference.py --write)" % OUTPUT_REL], rendered
    return [], rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify docs/reference/piece-format.md is up to date")
    mode.add_argument("--write", action="store_true", help="write docs/reference/piece-format.md")
    args = parser.parse_args(argv)

    problems, _ = run(args.root.resolve(), write=args.write)
    if problems:
        for p in problems:
            print(p)
        print("%d problem(s) in piece format reference" % len(problems))
        return 1
    if args.write:
        print("Wrote %s" % OUTPUT_REL)
    else:
        print("Piece format reference complete and up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
