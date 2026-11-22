"""Checks the roadmap registry and its coverage ledger: `python roadmap/check.py [--root REPO_ROOT]`.

Prints one line per problem as `<file>: <id>: <message>`. Exits 1 when there is an error and 0
otherwise. Warnings are printed with the word warning and do not change the exit code.

Candidate files under roadmap/candidates/ are checked on their own and stay out of the registry,
the graph, the coverage check and the counts.

The ledger under roadmap/coverage/ names where each feature comes from. Every entry maps to ids
that exist or gives a reason, and every `##` and `###` heading of each plan page under design/
appears in that page's ledger file, so a new plan heading without an entry fails the check.
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import jsonschema
import yaml

import build

STATUSES = build.STATUS_ORDER
# Pages under design/ that index the folder rather than plan work.
DESIGN_INDEXES = ("README.md", "index.md")
CANDIDATES = "candidates"
# A title is a short name, so it never starts with an article.
ARTICLE = re.compile(r"^(the|a|an)\s", re.I)


def _article(title):
    return isinstance(title, str) and ARTICLE.match(title) is not None


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


_JUDGE = []


def _judge(title):
    """The docs title rule of scripts/check_doc_headings.py: None when the title passes."""
    if not _JUDGE:
        path = build.ROOT / "scripts" / "check_doc_headings.py"
        spec = importlib.util.spec_from_file_location("check_doc_headings", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("check_doc_headings", module)
        spec.loader.exec_module(module)
        _JUDGE.append(module.judge)
    return _JUDGE[0](title)


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


def _heading_key(text):
    return build.heading_text(str(text))


def check_coverage(root, features, errors, warnings):
    """Checks the ledger under root/roadmap/coverage against the registry and design/.

    Returns (groups, entries): the number of ledger files read and of their entries.
    """
    folder = root / "roadmap" / build.COVERAGE
    if not folder.is_dir():
        errors.append("roadmap/%s: -: the coverage ledger is missing" % build.COVERAGE)
        return 0, 0
    for extra in sorted(folder.glob("*.yaml")):
        errors.append("roadmap/%s/%s: -: ledger files end in .yml, so this file is not read"
                      % (build.COVERAGE, extra.name))
    schema_path = folder / "schema.json"
    if not schema_path.is_file():
        errors.append("roadmap/%s/schema.json: -: the ledger's schema is missing" % build.COVERAGE)
        return 0, 0
    validator = jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8")))

    groups = []
    for path in sorted(folder.glob("*.yml")):
        name = "roadmap/%s/%s" % (build.COVERAGE, path.name)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as err:
            errors.append("%s: -: not valid YAML: %s" % (name, err))
            continue
        problems = sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.path)))
        for err in problems:
            where = "/".join(str(p) for p in err.absolute_path) or "-"
            errors.append("%s: -: schema: %s: %s" % (name, where, err.message))
        if not problems:
            groups.append((name, data))
            if _article(data["title"]):
                errors.append("%s: -: the title starts with an article" % name)
            if _has_todo(data):
                errors.append("%s: -: contains TODO" % name)

    sourced = set()
    for name, group in groups:
        for n, entry in enumerate(group["entries"], 1):
            for fid in entry.get("ids") or []:
                if fid in features:
                    sourced.add(fid)
                else:
                    errors.append("%s: %s: entry %d maps to %s, which does not exist"
                                  % (name, fid, n, fid))
            if entry.get("page"):
                _check_ref(root, name, "entry %d" % n, "docs", entry["page"], errors)

    # Plan pages: each heading appears in the page's ledger file.
    pages = {}
    for name, group in groups:
        if group["kind"] != "page":
            continue
        if group["source"] in pages:
            errors.append("%s: -: %s already has the ledger file %s"
                          % (name, group["source"], pages[group["source"]][0]))
            continue
        pages[group["source"]] = (name, group)
    design = root / "design"
    plan_pages = sorted(p for p in design.glob("*.md") if p.name not in DESIGN_INDEXES) \
        if design.is_dir() else []
    for page in plan_pages:
        rel = "design/" + page.name
        if rel not in pages:
            errors.append("%s: -: the plan page has no ledger file under roadmap/%s"
                          % (rel, build.COVERAGE))
            continue
        name, group = pages[rel]
        group_ref = group.get("ref", "main")
        found = build.read_headings(page) or []
        headings = [text for level, text, _ in found if level in (2, 3)]
        on_page = {text for _, text, _ in found}  # an entry may sit under the page's title
        skipped = [_heading_key(h) for h in group.get("not_features") or []]
        named = [(_heading_key(e["heading"]), e.get("ref") or group_ref)
                 for e in group["entries"]]
        covered = set(skipped) | {h for h, _ in named}
        for heading in headings:
            if heading not in covered:
                errors.append("%s: -: the heading \"%s\" of %s has no ledger entry and is not "
                              "listed under not_features" % (name, heading, rel))
        for heading in skipped:
            if heading not in on_page:
                errors.append("%s: -: not_features names \"%s\", which is not a heading of %s"
                              % (name, heading, rel))
        for heading, ref in sorted(set(named)):
            if heading not in on_page and ref == "main":
                errors.append("%s: -: an entry names the heading \"%s\", which is not on %s; "
                              "give the entry the ref of the branch it was read from"
                              % (name, heading, rel))
    for rel, (name, group) in sorted(pages.items()):
        if not (root / rel).is_file() and group.get("ref", "main") == "main":
            errors.append("%s: -: %s is not on this branch, so the ledger must give the ref of "
                          "the branch it was read from" % (name, rel))

    for fid in sorted(features, key=build.id_key):
        if fid not in sourced:
            warnings.append("%s: %s: warning: no ledger entry names this feature"
                            % (features[fid][0], fid))
    return len(groups), sum(len(g["entries"]) for _, g in groups)


def check_candidates(root, features, pillars, errors):
    """Checks roadmap/candidates/*.yml against the registry. Candidates are not features.

    pillars maps each registry prefix to its file stem. Returns (candidates, files) read.
    """
    folder = root / "roadmap" / CANDIDATES
    if not folder.is_dir():
        return 0, 0
    for extra in sorted(folder.glob("*.yaml")):
        errors.append("roadmap/%s/%s: -: candidate files end in .yml, so this file is not read"
                      % (CANDIDATES, extra.name))
    schema_path = folder / "schema.json"
    if not schema_path.is_file():
        errors.append("roadmap/%s/schema.json: -: the candidates' schema is missing" % CANDIDATES)
        return 0, 0
    validator = jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8")))

    groups = []
    for path in sorted(folder.glob("*.yml")):
        name = "roadmap/%s/%s" % (CANDIDATES, path.name)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as err:
            errors.append("%s: -: not valid YAML: %s" % (name, err))
            continue
        problems = sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.path)))
        for err in problems:
            where = "/".join(str(p) for p in err.absolute_path) or "-"
            errors.append("%s: -: schema: %s: %s" % (name, where, err.message))
        if not problems:
            groups.append((name, path.stem, data))

    titles = {f["title"].casefold(): fid for fid, (_, f) in features.items()}
    prefixes, keys = {}, {}
    for name, stem, data in groups:
        pillar = data["pillar"]
        if pillar in prefixes:
            errors.append("%s: -: duplicate pillar %s, also used by %s"
                          % (name, pillar, prefixes[pillar]))
        prefixes.setdefault(pillar, name)
        if data.get("proposed"):
            if pillar in pillars:
                errors.append("%s: -: proposed pillar %s is already a registry prefix"
                              % (name, pillar))
        elif pillar not in pillars:
            errors.append("%s: -: pillar %s is not a registry prefix" % (name, pillar))
        elif pillars[pillar] != stem:
            errors.append("%s: -: the file for pillar %s must be named %s.yml"
                          % (name, pillar, pillars[pillar]))
        reason = _judge(data["title"])
        if reason:
            errors.append("%s: -: the title \"%s\" %s" % (name, data["title"], reason))
        if _has_todo(data):
            errors.append("%s: -: contains TODO" % name)
        for c in data["candidates"]:
            key = c["key"]
            if key in keys:
                errors.append("%s: %s: duplicate key, also in %s" % (name, key, keys[key]))
            keys.setdefault(key, name)
            if key.casefold() in {fid.casefold() for fid in features}:
                errors.append("%s: %s: the key is a registry feature id" % (name, key))
            reason = _judge(c["title"])
            if reason:
                errors.append("%s: %s: the title \"%s\" %s" % (name, key, c["title"], reason))
            if c["title"].casefold() in titles:
                errors.append("%s: %s: the title is the title of the feature %s"
                              % (name, key, titles[c["title"].casefold()]))
            for dep in c.get("depends_on") or []:
                if dep not in features:
                    errors.append("%s: %s: depends_on %s, which does not exist" % (name, key, dep))
    return sum(len(d["candidates"]) for _, _, d in groups), len(groups)


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
        if _article(data["title"]):
            errors.append("%s: -: the pillar title starts with an article" % name)

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
            if _article(f["title"]):
                errors.append("%s: %s: the title starts with an article" % (name, fid))

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

    ledger_files, ledger_entries = check_coverage(root, features, errors, warnings)

    candidates, candidate_files = check_candidates(
        root, features, {d["prefix"]: Path(n).stem for n, d in valid}, errors)

    counts = Counter(f["status"] for _, f in features.values())
    in_progress = sum(1 for _, f in features.values() if f.get("in_progress"))
    edges = sum(len(v) for v in graph.values())
    summary = "%d features, %d edges (%s; %d in progress)" % (
        len(features), edges, ", ".join("%d %s" % (counts[s], s) for s in STATUSES), in_progress)
    if ledger_files:
        summary += ", %d ledger entries in %d files" % (ledger_entries, ledger_files)
    if candidate_files:
        summary += ", %d candidates in %d files" % (candidates, candidate_files)
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
