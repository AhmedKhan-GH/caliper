"""Generates the roadmap's pages and data files from the YAML registry.

Used two ways. As an MkDocs hook (`hooks:` in mkdocs.yml) it adds the generated files to the
site in memory. As a script, `python roadmap/build.py --out DIR`, it writes the same files under
DIR at their source paths, so the output can be read without MkDocs.

Standard library plus PyYAML and Markdown only, because it runs inside `mkdocs build`.
"""

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import yaml
from markdown.extensions.toc import slugify

REGISTRY = Path(__file__).resolve().parent
ROOT = REGISTRY.parent


def _site_url():
    """The docs site's address from mkdocs.yml, for the script; inside MkDocs the config gives it."""
    try:
        with open(ROOT / "mkdocs.yml", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("site_url") or "/"
    except (OSError, yaml.YAMLError):
        return "/"


DEFAULT_SITE = _site_url()
GITHUB = "https://github.com/AhmedKhan-GH/caliper"

STATUS_ORDER = ["shipped", "built", "planned", "idea", "decision"]
STATUS_RANK = {"shipped": 0, "built": 1, "planned": 2, "idea": 2, "decision": 2}
# What the pages show: built work whose branches are unfinished shows as in progress.
SHOWN_ORDER = ["shipped", "built", "progress", "planned", "idea", "decision"]
SECTION_TITLES = {"shipped": "Shipped", "built": "Built", "progress": "In progress",
                  "planned": "Planned", "idea": "Ideas", "decision": "Decisions"}
BADGE_LABELS = {"shipped": "Shipped", "built": "Built", "progress": "In progress",
                "planned": "Planned", "idea": "Idea", "decision": "Decision"}
FEATURE_KEYS = ["id", "title", "status", "summary", "depends_on", "sprint", "docs",
                "branches", "in_progress", "design", "issue", "draft"]
COVERAGE = "coverage"


class RegistryError(Exception):
    """The registry cannot be loaded; the message names the file and the id."""


def id_key(fid):
    """Sorts ids by prefix and then by number, so that COR-99 comes before COR-100."""
    prefix, _, number = fid.partition("-")
    return (prefix, int(number) if number.isdigit() else 0, fid)


_MD_SPECIAL = re.compile(r"([\\`*_\[\]|#])")


def md(text):
    """Registry text for a Markdown page: HTML-like text such as <folder> shows as typed, and
    Markdown characters, the table's pipe included, are escaped."""
    return _MD_SPECIAL.sub(r"\\\1", html.escape(str(text).strip(), quote=False))


# Headings

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_ATX = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")


def heading_text(raw):
    """The text of a heading as MkDocs slugs it: inline code, link syntax and HTML tags removed.

    A placeholder such as `<folder>` outside backticks is an HTML tag to Markdown, so MkDocs
    leaves it out of the heading's text and its slug.
    """
    text = re.sub(r"(?<!`)<[A-Za-z/][^<>`]*>(?!`)", "", raw)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.strip()


def parse_headings(markdown):
    """Returns [(level, text, slug)] for the ATX headings of a Markdown source.

    Fenced code blocks are skipped. A repeated slug gets _1, _2 and so on, as MkDocs does.
    """
    found = []
    seen = set()
    fence = None
    for line in markdown.splitlines():
        m = _FENCE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        m = _ATX.match(line)
        if not m:
            continue
        text = heading_text(m.group(2))
        slug = base = slugify(text, "-")
        n = 0
        while slug in seen:
            n += 1
            slug = "%s_%d" % (base, n)
        seen.add(slug)
        found.append((len(m.group(1)), text, slug))
    return found


def read_headings(path):
    """Headings of the file at path, or None when it cannot be read."""
    try:
        return parse_headings(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None


def first_heading(headings):
    for level, text, _ in headings:
        if level == 1:
            return text
    return None


def shown_status(feature):
    """The status a page shows: 'progress' for built work that is unfinished."""
    if feature["status"] == "built" and feature.get("in_progress"):
        return "progress"
    return feature["status"]


def split_ref(ref):
    """'docs/a.md#x' -> ('docs/a.md', 'x'); no anchor gives None."""
    path, _, anchor = ref.partition("#")
    return path, (anchor or None)


# Registry

def read_files(root):
    """Returns [(file name, data)] for every pillar file, sorted by name."""
    folder = Path(root) / "roadmap"
    result = []
    for path in sorted(folder.glob("*.yml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as err:
            raise RegistryError("roadmap/%s: not valid YAML: %s" % (path.name, err))
        result.append(("roadmap/" + path.name, data))
    return result


def load_registry(root):
    """Returns (pillars, features) from the YAML files under root/roadmap.

    Pillars are sorted by order and each holds its features, sorted by id. Every feature has
    its pillar prefix and a computed `unblocks` list. An unknown dependency raises RegistryError.
    """
    pillars = []
    features = {}
    for name, data in read_files(root):
        if not isinstance(data, dict):
            raise RegistryError("%s: not a pillar mapping" % name)
        pillar = {"prefix": data["prefix"], "title": data["title"], "slug": Path(name).stem,
                  "order": data["order"], "summary": data["summary"], "file": name,
                  "features": []}
        for raw in data.get("features") or []:
            fid = raw["id"]
            if fid in features:
                raise RegistryError("%s: %s: duplicate id" % (name, fid))
            feature = {key: raw.get(key) for key in FEATURE_KEYS}
            feature["depends_on"] = sorted(raw.get("depends_on") or [], key=id_key)
            feature["branches"] = list(raw.get("branches") or [])
            feature["in_progress"] = bool(raw.get("in_progress"))
            feature["draft"] = bool(raw.get("draft"))
            feature["sources"] = []
            feature["pillar"] = pillar["prefix"]
            feature["file"] = name
            feature["unblocks"] = []
            features[fid] = feature
            pillar["features"].append(feature)
        pillar["features"].sort(key=lambda f: id_key(f["id"]))
        pillars.append(pillar)
    pillars.sort(key=lambda p: (p["order"], p["prefix"]))
    for fid in sorted(features, key=id_key):
        for dep in features[fid]["depends_on"]:
            if dep not in features:
                raise RegistryError("%s: %s: depends_on %s, which does not exist"
                                    % (features[fid]["file"], fid, dep))
            features[dep]["unblocks"].append(fid)
    for feature in features.values():
        feature["unblocks"].sort(key=id_key)
    for group in read_coverage(root):
        for entry in group["entries"]:
            for fid in entry.get("ids") or []:
                if fid in features:
                    features[fid]["sources"].append((group, entry))
    return pillars, features


# Coverage ledger

def read_coverage(root):
    """Returns the ledger groups under root/roadmap/coverage, sorted by file name.

    Each group is its YAML mapping with its file name under the key `file`. A file that is not
    a mapping is skipped here; the checker reports it.
    """
    folder = Path(root) / "roadmap" / COVERAGE
    groups = []
    for path in sorted(folder.glob("*.yml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as err:
            raise RegistryError("roadmap/%s/%s: not valid YAML: %s" % (COVERAGE, path.name, err))
        if isinstance(data, dict):
            data = dict(data)
            data["file"] = "roadmap/%s/%s" % (COVERAGE, path.name)
            data["entries"] = [e for e in data.get("entries") or [] if isinstance(e, dict)]
            groups.append(data)
    return groups


# Pages

def _heading_link(root, base_dir, ref, kind, base):
    """Markdown link for a docs or design reference, texted by its page heading."""
    path, anchor = split_ref(ref)
    headings = read_headings(Path(root) / path) or []
    title = first_heading(headings) or path
    text = title
    if anchor:
        for _, htext, slug in headings:
            if slug == anchor:
                text = "%s, %s" % (title, htext)
                break
    if kind == "docs" and path.startswith("docs/"):
        target = os.path.relpath(path[len("docs/"):], base_dir).replace(os.sep, "/")
        if anchor:
            target += "#" + anchor
    elif kind == "design":
        target = "%s/blob/main/%s" % (base, ref)
    else:
        return md(ref)  # not a page path (a placeholder); no link
    return "[%s](%s)" % (md(text), target)


def _feature_links(ids, features):
    if not ids:
        return "None"
    return "<br>".join("[%s](%s.md) (`%s`)" % (md(features[i]["title"]), i.lower(), i) for i in ids)


def pillar_page(pillar):
    lines = ["---", "template: roadmap.html", "---", "", "# " + md(pillar["title"]), "",
             md(pillar["summary"]), "",
             "The [roadmap graph](index.md) shows how these features depend on features in "
             "other pillars.", ""]
    for status in SHOWN_ORDER:
        items = [f for f in pillar["features"] if shown_status(f) == status]
        if not items:
            continue
        lines += ["## " + SECTION_TITLES[status], ""]
        for f in items:
            line = '- <span class="roadmap-id">%s</span> [%s](%s.md). %s' % (
                f["id"], md(f["title"]), f["id"].lower(), md(f["summary"]))
            if f["sprint"]:
                line += " Sprint %s." % f["sprint"]
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def feature_page(feature, pillars_by_prefix, features, root, base):
    pillar = pillars_by_prefix[feature["pillar"]]
    lines = ["---", "template: roadmap.html", "---", "", "# " + md(feature["title"]), ""]
    if feature["draft"]:
        lines += ['!!! note "Draft"',
                  "    The dependencies of this entry are a first draft and still need "
                  "checking.", ""]
    lines += [md(feature["summary"]), ""]
    docs = (_heading_link(root, "roadmap", feature["docs"], "docs", base)
            if feature["docs"] else "None")
    design = (_heading_link(root, "roadmap", feature["design"], "design", base)
              if feature["design"] else "None")
    # Most work branches are not published, so a branch is named rather than linked.
    branches = ("<br>".join("`%s`" % b for b in feature["branches"])
                if feature["branches"] else "None")
    shown = shown_status(feature)
    issue = ("[#%d](%s/issues/%d)" % (feature["issue"], base, feature["issue"])
             if feature["issue"] else "None")
    rows = [
        ("Id", "`%s`" % feature["id"]),
        ("Status", '<span class="roadmap-badge roadmap-badge--%s">%s</span>'
         % (shown, BADGE_LABELS[shown])),
        ("Pillar", "[%s](%s.md)" % (md(pillar["title"]), pillar["slug"])),
        ("Sprint", str(feature["sprint"]) if feature["sprint"] else "None"),
        ("Depends on", _feature_links(feature["depends_on"], features)),
        ("Unblocks", _feature_links(feature["unblocks"], features)),
        ("Docs", docs), ("Branches", branches), ("Design", design), ("Issue", issue),
    ]
    lines += ["| Field | Value |", "| --- | --- |"]
    lines += ["| %s | %s |" % row for row in rows]
    if feature["in_progress"]:
        lines += ["", "The work on these branches is unfinished, so the feature shows as in "
                  "progress."]
    lines += [""]
    if feature["sources"]:
        lines += ["## Sources", "",
                  "The [coverage ledger](%s/tree/main/roadmap/coverage) records where this "
                  "feature comes from:" % base, ""]
        lines += ["- " + source_line(root, group, entry, base)
                  for group, entry in feature["sources"]]
        lines += [""]
    lines += ["Back to the [roadmap graph](index.md).", ""]
    return "\n".join(lines)


def _sentence(text):
    text = md(text)
    return text if text.endswith((".", "?", "!")) else text + "."


def source_line(root, group, entry, base):
    """One line of a feature page's Sources list: where the item is, then the item."""
    kind = group.get("kind")
    ref = entry.get("ref") or group.get("ref") or "main"
    if kind == "page":
        path = group["source"]
        heading = entry.get("heading")
        on_main = ref == "main" and (Path(root) / path).is_file()
        if on_main:
            anchor = ""
            if heading:
                for _, text, slug in read_headings(Path(root) / path) or []:
                    if text == heading_text(heading):
                        anchor = "#" + slug
                        break
            where = "[%s](%s/blob/main/%s%s)" % (md(path), base, path, anchor)
        else:
            where = "`%s` on the branch `%s`" % (path, ref)
        if heading:
            where += ", under *%s*" % md(heading_text(heading))
    elif kind == "docs":
        where = _heading_link(root, "roadmap", entry["page"], "docs", base)
    elif kind == "github":
        number = entry.get("pr")
        where = "[Pull request #%d](%s/pull/%d)" % (number, base, number)
    elif kind in ("branches", "worktrees"):
        label = "Branch" if kind == "branches" else "Uncommitted work on the branch"
        where = "%s `%s`" % (label, entry["branch"])
        if entry.get("commits"):
            where += ", %s" % md(entry["commits"])
    elif kind == "notes":
        where = "%s, `%s`" % (md(group.get("title", "Notes")), entry["file"])
    else:
        where = md(group.get("title", group["file"]))
    return "%s. %s" % (where, _sentence(entry["item"]))


def _dump(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def graph_json(pillars, features):
    order = {p["prefix"]: p["order"] for p in pillars}
    nodes = []
    for f in sorted(features.values(), key=lambda f: (order[f["pillar"]], id_key(f["id"]))):
        nodes.append({"id": f["id"], "title": f["title"], "status": f["status"],
                      "in_progress": f["in_progress"], "pillar": f["pillar"],
                      "summary": f["summary"], "sprint": f["sprint"],
                      "branches": f["branches"], "draft": f["draft"],
                      "url": f["id"].lower() + "/"})
    edges = []
    for f in sorted(features.values(), key=lambda f: id_key(f["id"])):
        for dep in f["depends_on"]:
            edges.append({"id": "%s>%s" % (dep, f["id"]), "source": dep, "target": f["id"],
                          "draft": f["draft"]})
    edges.sort(key=lambda e: (id_key(e["target"]), id_key(e["source"])))
    return _dump({
        "pillars": [{"prefix": p["prefix"], "title": p["title"], "slug": p["slug"],
                     "order": p["order"], "url": p["slug"] + "/"} for p in pillars],
        "nodes": nodes, "edges": edges})


def _source_ref(group, entry):
    """A ledger entry for roadmap.json: its ledger file and the fields that place the item."""
    ref = {"ledger": group["file"], "item": entry["item"]}
    for key in ("heading", "ref", "page", "pr", "branch", "state", "file"):
        if entry.get(key) is not None:
            ref[key] = entry[key]
    if group.get("kind") == "page":
        ref["source"] = group["source"]
        ref.setdefault("ref", group.get("ref") or "main")
    return ref


def registry_json(pillars, features, site):
    return _dump({
        "site": site,
        "pillars": [{"prefix": p["prefix"], "title": p["title"], "slug": p["slug"],
                     "order": p["order"], "summary": p["summary"],
                     "url": "%sroadmap/%s/" % (site, p["slug"]),
                     "features": [f["id"] for f in p["features"]]} for p in pillars],
        "features": [
            {"id": f["id"], "title": f["title"], "status": f["status"], "pillar": f["pillar"],
             "summary": f["summary"], "depends_on": f["depends_on"], "unblocks": f["unblocks"],
             "sprint": f["sprint"], "docs": f["docs"], "branches": f["branches"],
             "in_progress": f["in_progress"], "design": f["design"], "issue": f["issue"],
             "draft": f["draft"], "sources": [_source_ref(g, e) for g, e in f["sources"]],
             "url": "%sroadmap/%s/" % (site, f["id"].lower())}
            for f in sorted(features.values(), key=lambda f: id_key(f["id"]))]})


def generate(root, site=DEFAULT_SITE):
    """Returns [(src_uri, content, in_nav)] for every generated file, in a stable order."""
    if not site.endswith("/"):
        site += "/"
    pillars, features = load_registry(root)
    by_prefix = {p["prefix"]: p for p in pillars}
    out = []
    for p in pillars:
        out.append(("roadmap/%s.md" % p["slug"], pillar_page(p), True))
    for fid in sorted(features, key=id_key):
        out.append(("roadmap/%s.md" % fid.lower(),
                    feature_page(features[fid], by_prefix, features, root, GITHUB), False))
    out.append(("roadmap/graph.json", graph_json(pillars, features), True))
    out.append(("roadmap.json", registry_json(pillars, features, site), True))
    return out


# MkDocs hook

def on_files(files, config):
    from mkdocs.structure.files import File, InclusionLevel

    site = config.get("site_url") or DEFAULT_SITE
    for src_uri, content, in_nav in generate(ROOT, site):
        kwargs = {} if in_nav else {"inclusion": InclusionLevel.NOT_IN_NAV}
        files.append(File.generated(config, src_uri, content=content, **kwargs))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the generated roadmap files to a folder.")
    parser.add_argument("--out", required=True, help="folder that receives the files")
    parser.add_argument("--site-url", default=DEFAULT_SITE)
    parser.add_argument("--root", default=str(ROOT), help="repository root")
    args = parser.parse_args(argv)
    try:
        generated = generate(Path(args.root), args.site_url)
    except RegistryError as err:
        print(err, file=sys.stderr)
        return 1
    for src_uri, content, _ in generated:
        target = Path(args.out) / src_uri
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    print("wrote %d files under %s" % (len(generated), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
