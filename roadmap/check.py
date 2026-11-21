"""Checks the roadmap registry: `python roadmap/check.py [--root REPO_ROOT]`.

Prints one line per problem as `<file>: <id>: <message>`. Exits 1 when there is an error and 0
otherwise. Warnings are printed with the word warning and do not change the exit code.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import jsonschema

import build

STATUSES = build.STATUS_ORDER


def _has_todo(value):
    if isinstance(value, str):
        return "TODO" in value
    if isinstance(value, dict):
        return any(_has_todo(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_todo(v) for v in value)
    return False


def _todo_fields(feature):
    return [k for k, v in feature.items() if _has_todo(v)]


def _find_cycle(graph):
    """Returns one cycle as a list of ids, first id repeated at the end, or None."""
    state = {}
    stack = []

    def visit(node):
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, [])):
            if nxt not in graph:
                continue
            if state.get(nxt) == 1:
                return stack[stack.index(nxt):] + [nxt]
            if nxt not in state:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if node not in state:
            found = visit(node)
            if found:
                return found
    return None


def _check_ref(root, name, fid, field, ref, errors):
    path, anchor = build.split_ref(ref)
    headings = build.read_headings(root / path)
    if headings is None:
        errors.append("%s: %s: %s page %s does not exist" % (name, fid, field, path))
        return
    if anchor and anchor not in [slug for _, _, slug in headings]:
        errors.append("%s: %s: %s anchor #%s is not a heading on %s"
                      % (name, fid, field, anchor, path))


def run_checks(root):
    """Returns (errors, warnings, summary) for the registry under root."""
    root = Path(root)
    errors, warnings = [], []
    try:
        files = build.read_files(root)
    except build.RegistryError as err:
        return [str(err)], [], ""
    for extra in sorted((root / "roadmap").glob("*.yaml")):
        errors.append("roadmap/%s: -: pillar files end in .yml, so this file is not read" % extra.name)
    schema = json.loads((root / "roadmap" / "schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    valid = []
    for name, data in files:
        problems = sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.path)))
        for err in problems:
            where = "/".join(str(p) for p in err.absolute_path) or "-"
            fid = "-"
            parts = list(err.absolute_path)
            if len(parts) >= 2 and parts[0] == "features" and isinstance(data, dict):
                try:
                    fid = data["features"][parts[1]].get("id", "-")
                except (KeyError, IndexError, AttributeError, TypeError):
                    pass
            errors.append("%s: %s: schema: %s: %s" % (name, fid, where, err.message))
        if not problems:
            valid.append((name, data))
    if errors:
        return errors, [], ""

    prefixes, orders = {}, {}
    for name, data in valid:
        if data["prefix"] in prefixes:
            errors.append("%s: -: duplicate prefix %s, also used by %s"
                          % (name, data["prefix"], prefixes[data["prefix"]]))
        prefixes.setdefault(data["prefix"], name)
        if data["order"] in orders:
            errors.append("%s: -: duplicate order %d, also used by %s"
                          % (name, data["order"], orders[data["order"]]))
        orders.setdefault(data["order"], name)
        if _has_todo({k: v for k, v in data.items() if k != "features"}):
            errors.append("%s: -: contains TODO" % name)

    # The graph page's pillar list, which narrow screens show instead of the graph, and the
    # section's navigation are written by hand; each must name every pillar page.
    for rel in ("docs/roadmap/index.md", "docs/roadmap/.nav.yml"):
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for name, _ in valid:
            page = Path(name).stem + ".md"
            if rel.endswith(".md"):
                found = "](%s)" % page in text
            else:
                found = re.search(r"^\s*-\s*%s\s*$" % re.escape(page), text, re.M)
            if not found:
                errors.append("%s: -: %s does not list the pillar page %s" % (name, rel, page))

    features = {}
    for name, data in valid:
        for f in data["features"]:
            fid = f["id"]
            if fid in features:
                errors.append("%s: %s: duplicate id, also in %s" % (name, fid, features[fid][0]))
                continue
            features[fid] = (name, f)
            if not fid.startswith(data["prefix"] + "-"):
                errors.append("%s: %s: id does not use the prefix %s" % (name, fid, data["prefix"]))
            for field in _todo_fields(f):
                errors.append("%s: %s: %s contains TODO" % (name, fid, field))

    graph = {}
    for fid, (name, f) in sorted(features.items()):
        deps = f.get("depends_on") or []
        graph[fid] = deps
        for dep in deps:
            if dep == fid:
                errors.append("%s: %s: depends on itself" % (name, fid))
            elif dep not in features:
                errors.append("%s: %s: depends_on %s, which does not exist" % (name, fid, dep))
    cycle = _find_cycle({k: [d for d in v if d != k] for k, v in graph.items()})
    if cycle:
        errors.append("%s: %s: dependency cycle %s"
                      % (features[cycle[0]][0], cycle[0], " -> ".join(cycle)))

    rank = build.STATUS_RANK
    for fid, (name, f) in sorted(features.items()):
        if f["status"] == "shipped" and not f.get("docs"):
            errors.append("%s: %s: shipped feature has no docs" % (name, fid))
        for field in ("docs", "design"):
            if f.get(field):
                _check_ref(root, name, fid, field, f[field], errors)
        for dep in f.get("depends_on") or []:
            if dep in features and dep != fid:
                dstatus = features[dep][1]["status"]
                if rank[dstatus] > rank[f["status"]]:
                    errors.append("%s: %s: %s depends on %s, which is %s and has a higher "
                                  "status rank" % (name, fid, f["status"], dep, dstatus))

    try:
        _, loaded = build.load_registry(root)
    except build.RegistryError:
        loaded = None  # already reported above
    if loaded is not None:
        for fid, f in sorted(loaded.items()):
            if f["status"] == "decision" and not f["unblocks"]:
                errors.append("%s: %s: decision that no feature depends on" % (f["file"], fid))

    # Redundant edges: a dependency already reachable through another dependency. An edge from a
    # decision is never redundant: it names a feature that the decision blocks directly.
    def reach(start, seen):
        for nxt in graph.get(start, []):
            if nxt in graph and nxt not in seen:
                seen.add(nxt)
                reach(nxt, seen)
        return seen

    if not cycle:
        for fid, (name, f) in sorted(features.items()):
            deps = [d for d in graph[fid] if d in features and d != fid]
            for dep in sorted(deps):
                if features[dep][1]["status"] == "decision":
                    continue
                for other in sorted(deps):
                    if other != dep and dep in reach(other, set()):
                        warnings.append("%s: %s: warning: depends_on %s is already implied "
                                        "through %s" % (name, fid, dep, other))
                        break

    counts = Counter(f["status"] for _, f in features.values())
    edges = sum(len(v) for v in graph.values())
    summary = "%d features, %d edges (%s)" % (
        len(features), edges, ", ".join("%d %s" % (counts[s], s) for s in STATUSES))
    return errors, warnings, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the roadmap registry.")
    parser.add_argument("--root", default=str(build.ROOT), help="repository root")
    args = parser.parse_args(argv)
    errors, warnings, summary = run_checks(args.root)
    for line in errors + warnings:
        print(line)
    if errors:
        return 1
    print("ok: " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
