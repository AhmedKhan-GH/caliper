#!/usr/bin/env python3
"""Generate or verify docs/development/feature-index.md from roadmap/*.yml.

Reads roadmap/*.yml and roadmap/index/feature_code_tests.yml so every feature
in the roadmap registry gets one lookup row with its documentation page, code
locations, and test suites. In --check mode (the default), exits with status 1
when:

  1. any completed (status: shipped) feature lacks a docs link, code location,
     or test location;
  2. any referenced docs page, heading anchor, code path, or test path does not
     exist in the repository;
  3. roadmap/index/feature_code_tests.yml names an unknown feature ID or trips
     a privacy pattern;
  4. docs/development/feature-index.md is out of date relative to the sources.

Usage:
  generate_feature_index.py [--root <repo_root>] [--check | --write]
"""

from __future__ import annotations

import argparse
import html
import os
import pathlib
import re
import sys

import yaml
from markdown.extensions.toc import slugify

OUTPUT_REL = "docs/development/feature-index.md"
INDEX_MAP_REL = "roadmap/index/feature_code_tests.yml"

STATUS_LABELS = {
    "shipped": "Shipped",
    "built": "Built",
    "progress": "In Progress",
    "planned": "Planned",
    "idea": "Idea",
    "decision": "Decision",
}

FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
ATX_RE = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
EXPLICIT_ANCHOR_RE = re.compile(r"\{#([A-Za-z0-9_.:-]+)\}")

_MAC_HOME = "/".join(["", "Users", ""])
_LINUX_HOME = "/".join(["", "home", ""])
_WIN_HOME = "\\".join(["C:", "Users", ""])
PRIVACY_RE = re.compile(
    r"(?:%s|%s|%s)[A-Za-z0-9._-]+|[A-Za-z0-9._%%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|https?://github\.com/[A-Za-z0-9-]+/"
    % (re.escape(_MAC_HOME), re.escape(_LINUX_HOME), re.escape(_WIN_HOME))
)


def id_key(fid: str) -> tuple[str, int, str]:
    prefix, _, num = fid.partition("-")
    return (prefix, int(num) if num.isdigit() else 0, fid)


def md_cell(text: str) -> str:
    escaped = html.escape(str(text).strip(), quote=False)
    return escaped.replace("|", "\\|")


def heading_text(raw: str) -> str:
    text = re.sub(r"(?<!`)<[A-Za-z/][^<>`]*>(?!`)", "", raw)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.strip()


def parse_page_anchors(path: pathlib.Path) -> tuple[str | None, dict[str, str]] | None:
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() in ("---", "..."):
                start = idx + 1
                break
    fence = None
    title = None
    anchors: dict[str, str] = {}
    seen: set[str] = set()
    for line in lines[start:]:
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        for exp in EXPLICIT_ANCHOR_RE.finditer(line):
            anchors.setdefault(exp.group(1), exp.group(1))
        hm = ATX_RE.match(line)
        if not hm:
            continue
        htext = heading_text(hm.group(2))
        if len(hm.group(1)) == 1 and title is None:
            title = htext
        slug = base = slugify(htext, "-")
        suffix = 0
        while slug in seen:
            suffix += 1
            slug = "%s_%d" % (base, suffix)
        seen.add(slug)
        anchors[slug] = htext
    return title, anchors


def load_sources(root: pathlib.Path) -> tuple[list[dict], dict[str, dict], list[str]]:
    problems: list[str] = []
    rm_dir = root / "roadmap"
    if not rm_dir.is_dir():
        return [], {}, ["roadmap: directory does not exist"]

    pillars: list[dict] = []
    all_features: dict[str, tuple[str, dict]] = {}
    for yml_path in sorted(rm_dir.glob("*.yml")):
        rel = yml_path.relative_to(root).as_posix()
        data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "features" not in data:
            problems.append("%s: invalid pillar YAML" % rel)
            continue
        feats = sorted(data.get("features") or [], key=lambda f: id_key(f["id"]))
        pillar = {
            "file": rel,
            "slug": yml_path.stem,
            "prefix": data["prefix"],
            "title": data["title"],
            "order": data.get("order", 999),
            "summary": str(data.get("summary", "")).strip(),
            "features": feats,
        }
        pillars.append(pillar)
        for feat in feats:
            all_features[feat["id"]] = (rel, feat)

    pillars.sort(key=lambda p: (p["order"], p["prefix"]))

    map_path = root / INDEX_MAP_REL
    if not map_path.is_file():
        problems.append("%s: mapping file does not exist" % INDEX_MAP_REL)
        return pillars, {}, problems

    raw_map_text = map_path.read_text(encoding="utf-8")
    if PRIVACY_RE.search(raw_map_text):
        problems.append("%s: privacy check failed" % INDEX_MAP_REL)

    code_tests = yaml.safe_load(raw_map_text) or {}
    if not isinstance(code_tests, dict):
        problems.append("%s: root value must be a mapping" % INDEX_MAP_REL)
        return pillars, {}, problems

    for fid, entry in sorted(code_tests.items(), key=lambda kv: id_key(kv[0])):
        if fid not in all_features:
            problems.append("%s: %s: unknown feature ID" % (INDEX_MAP_REL, fid))
            continue
        if not isinstance(entry, dict):
            problems.append("%s: %s: entry must be a mapping" % (INDEX_MAP_REL, fid))
            continue
        extra_keys = set(entry) - {"code", "tests"}
        if extra_keys:
            problems.append(
                "%s: %s: unexpected keys %s" % (INDEX_MAP_REL, fid, ", ".join(sorted(extra_keys)))
            )
        for kind in ("code", "tests"):
            vals = entry.get(kind)
            if vals is None:
                continue
            if not isinstance(vals, list) or not all(isinstance(x, str) and x.strip() for x in vals):
                problems.append("%s: %s: %s must be a list of non-empty paths" % (INDEX_MAP_REL, fid, kind))
                continue
            for rel_item in vals:
                if rel_item.startswith("/") or ".." in rel_item.split("/"):
                    problems.append("%s: %s: %s path %r must be repository-relative" % (INDEX_MAP_REL, fid, kind, rel_item))
                elif not (root / rel_item).exists():
                    problems.append("%s: %s: %s path %s does not exist" % (INDEX_MAP_REL, fid, kind, rel_item))

    anchor_cache: dict[str, tuple[str | None, dict[str, str]] | None] = {}

    def check_docs_ref(fid: str, file_label: str, ref: str) -> None:
        doc_path, _, anchor = ref.partition("#")
        if doc_path not in anchor_cache:
            anchor_cache[doc_path] = parse_page_anchors(root / doc_path)
        parsed = anchor_cache[doc_path]
        if parsed is None:
            problems.append("%s: %s: docs page %s does not exist" % (file_label, fid, doc_path))
            return
        if anchor and anchor not in parsed[1]:
            problems.append("%s: %s: docs anchor #%s does not exist on %s" % (file_label, fid, anchor, doc_path))

    for fid, (rel, feat) in sorted(all_features.items(), key=lambda kv: id_key(kv[0])):
        docs_ref = feat.get("docs")
        entry = code_tests.get(fid) if isinstance(code_tests.get(fid), dict) else {}
        code_list = entry.get("code") or []
        tests_list = entry.get("tests") or []
        if docs_ref:
            check_docs_ref(fid, rel, docs_ref)
        if feat.get("status") == "shipped":
            if not docs_ref:
                problems.append("%s: %s: shipped feature lacks docs link" % (rel, fid))
            if not code_list:
                problems.append("%s: %s: shipped feature lacks code locations in %s" % (rel, fid, INDEX_MAP_REL))
            if not tests_list:
                problems.append("%s: %s: shipped feature lacks test locations in %s" % (rel, fid, INDEX_MAP_REL))

    return pillars, code_tests, problems


def format_docs_cell(root: pathlib.Path, docs_ref: str | None, cache: dict[str, tuple[str | None, dict[str, str]] | None]) -> str:
    if not docs_ref:
        return "None"
    doc_path, _, anchor = docs_ref.partition("#")
    if doc_path not in cache:
        cache[doc_path] = parse_page_anchors(root / doc_path)
    parsed = cache[doc_path]
    page_title = (parsed[0] if parsed and parsed[0] else doc_path)
    label = page_title
    if anchor and parsed and anchor in parsed[1]:
        sec = parsed[1][anchor]
        if sec != page_title:
            label = "%s (%s)" % (page_title, sec)
    if doc_path.startswith("docs/"):
        rel_target = os.path.relpath(doc_path[len("docs/"):], "development").replace(os.sep, "/")
        if anchor:
            rel_target += "#" + anchor
        return "[%s](%s)" % (md_cell(label), rel_target)
    return "`%s`" % md_cell(docs_ref)


def format_paths_cell(paths: list[str] | None) -> str:
    if not paths:
        return "None"
    return ", ".join("`%s`" % p for p in paths)


def shown_status(feat: dict) -> str:
    if feat.get("status") == "built" and feat.get("in_progress"):
        return "progress"
    return feat.get("status", "planned")


def render_markdown(root: pathlib.Path, pillars: list[dict], code_tests: dict[str, dict]) -> str:
    cache: dict[str, tuple[str | None, dict[str, str]] | None] = {}
    lines = [
        "<!-- Generated by scripts/generate_feature_index.py. Do not edit by hand. -->",
        "",
        "# Feature Index",
        "",
        "This page maps every feature in `roadmap/*.yml` to its documentation page, source files, and test suites.",
        "The script `scripts/generate_feature_index.py` builds this index from `roadmap/*.yml` and `roadmap/index/feature_code_tests.yml`.",
        "Continuous integration runs `python scripts/generate_feature_index.py --check` on every documentation pull request.",
        "The check fails when any shipped feature lacks documentation, code locations, or tests, or when any linked path disappears.",
        "Run `python3 scripts/generate_feature_index.py --write` after roadmap or file layout changes to update this page.",
        "",
        "## Pillar Summary",
        "",
        "Ten roadmap pillars group Caliper features by subsystem.",
        "",
        "| Pillar | Prefix | Total Features | Shipped Features |",
        "|---|---|---:|---:|",
    ]
    for pillar in pillars:
        total = len(pillar["features"])
        shipped_count = sum(1 for f in pillar["features"] if f.get("status") == "shipped")
        lines.append(
            "| [%s](../roadmap/%s.md) | `%s` | %d | %d |"
            % (md_cell(pillar["title"]), pillar["slug"], pillar["prefix"], total, shipped_count)
        )
    lines.append("")

    for pillar in pillars:
        lines.append("## %s" % pillar["title"])
        lines.append("")
        if pillar["summary"]:
            lines.append(pillar["summary"])
            lines.append("")
        lines.extend(
            [
                "| Feature | Status | Documentation | Code Locations | Test Suites |",
                "|---|---|---|---|---|",
            ]
        )
        for feat in pillar["features"]:
            fid = feat["id"]
            st = STATUS_LABELS.get(shown_status(feat), feat.get("status", ""))
            entry = code_tests.get(fid) or {}
            doc_cell = format_docs_cell(root, feat.get("docs"), cache)
            code_cell = format_paths_cell(entry.get("code"))
            test_cell = format_paths_cell(entry.get("tests"))
            feat_cell = "[`%s`](../roadmap/%s.md) %s" % (fid, fid.lower(), md_cell(feat["title"]))
            lines.append("| %s | %s | %s | %s | %s |" % (feat_cell, st, doc_cell, code_cell, test_cell))
        lines.append("")

    return "\n".join(lines)


def run(root: pathlib.Path, write: bool) -> tuple[list[str], str]:
    pillars, code_tests, problems = load_sources(root)
    if problems:
        return problems, ""
    rendered = render_markdown(root, pillars, code_tests)
    out_path = root / OUTPUT_REL
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        return [], rendered
    if not out_path.is_file():
        return ["%s: generated file does not exist (run scripts/generate_feature_index.py --write)" % OUTPUT_REL], rendered
    current = out_path.read_text(encoding="utf-8")
    if current != rendered:
        return ["%s: out of date (run scripts/generate_feature_index.py --write)" % OUTPUT_REL], rendered
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
    mode.add_argument("--check", action="store_true", help="verify completeness and check for drift (default)")
    mode.add_argument("--write", action="store_true", help="write docs/development/feature-index.md")
    args = parser.parse_args(argv)

    problems, _ = run(args.root.resolve(), write=args.write)
    if problems:
        for p in problems:
            print(p)
        print("%d problem(s) in feature index" % len(problems))
        return 1
    if args.write:
        print("Wrote %s" % OUTPUT_REL)
    else:
        print("Feature index complete and up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
