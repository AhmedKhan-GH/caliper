"""Generates the roadmap's pages and data files from the YAML registry.

Used two ways. As an MkDocs hook (`hooks:` in mkdocs.yml) it adds the generated files to the
site in memory. As a script, `python roadmap/build.py --out DIR`, it writes the same files under
DIR at their source paths, so the output can be read without MkDocs.

Standard library plus PyYAML and Markdown only, because it runs inside `mkdocs build`.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml
from markdown.extensions.toc import slugify

REGISTRY = Path(__file__).resolve().parent
ROOT = REGISTRY.parent
DEFAULT_SITE = "https://ahmedkhan-gh.github.io/caliper/docs/"
GITHUB = "https://github.com/AhmedKhan-GH/caliper"

STATUS_ORDER = ["shipped", "built", "planned", "idea", "decision"]
STATUS_RANK = {"shipped": 0, "built": 1, "planned": 2, "idea": 2, "decision": 2}
SECTION_TITLES = {"shipped": "Shipped", "built": "Built", "planned": "Planned",
                  "idea": "Ideas", "decision": "Decisions"}
BADGE_LABELS = {"shipped": "Shipped", "built": "Built", "planned": "Planned",
                "idea": "Idea", "decision": "Decision"}
FEATURE_KEYS = ["id", "title", "status", "summary", "depends_on", "sprint", "docs",
                "branch", "design", "issue", "draft"]


class RegistryError(Exception):
    """The registry cannot be loaded; the message names the file and the id."""


# Headings

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_ATX = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")


def heading_text(raw):
    """The text of a heading as MkDocs slugs it: inline code and link syntax removed."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", raw)
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
            feature["depends_on"] = sorted(raw.get("depends_on") or [])
            feature["draft"] = bool(raw.get("draft"))
            feature["pillar"] = pillar["prefix"]
            feature["file"] = name
            feature["unblocks"] = []
            features[fid] = feature
            pillar["features"].append(feature)
        pillar["features"].sort(key=lambda f: f["id"])
        pillars.append(pillar)
    pillars.sort(key=lambda p: (p["order"], p["prefix"]))
    for fid in sorted(features):
        for dep in features[fid]["depends_on"]:
            if dep not in features:
                raise RegistryError("%s: %s: depends_on %s, which does not exist"
                                    % (features[fid]["file"], fid, dep))
            features[dep]["unblocks"].append(fid)
    for feature in features.values():
        feature["unblocks"].sort()
    return pillars, features


# Pages

def _cell(text):
    return text.replace("|", "\\|")


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
        return _cell(ref)  # not a page path (a placeholder); no link
    return "[%s](%s)" % (_cell(text), target)


def _feature_links(ids, features):
    if not ids:
        return "None"
    return "<br>".join("[%s](%s.md) (`%s`)" % (features[i]["title"], i.lower(), i) for i in ids)


def pillar_page(pillar):
    lines = ["---", "template: roadmap.html", "---", "", "# " + pillar["title"], "",
             pillar["summary"].strip(), "",
             "The [roadmap graph](index.md) shows how these features depend on features in "
             "other pillars.", ""]
    for status in STATUS_ORDER:
        items = [f for f in pillar["features"] if f["status"] == status]
        if not items:
            continue
        lines += ["## " + SECTION_TITLES[status], ""]
        for f in items:
            line = '- <span class="roadmap-id">%s</span> [%s](%s.md). %s' % (
                f["id"], f["title"], f["id"].lower(), f["summary"].strip())
            if f["sprint"]:
                line += " Sprint %d." % f["sprint"]
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def feature_page(feature, pillars_by_prefix, features, root, base):
    pillar = pillars_by_prefix[feature["pillar"]]
    lines = ["---", "template: roadmap.html", "---", "", "# " + feature["title"], ""]
    if feature["draft"]:
        lines += ['!!! note "Draft"',
                  "    The dependencies of this entry are a first draft and still need "
                  "checking.", ""]
    lines += [feature["summary"].strip(), ""]
    docs = (_heading_link(root, "roadmap", feature["docs"], "docs", base)
            if feature["docs"] else "None")
    design = (_heading_link(root, "roadmap", feature["design"], "design", base)
              if feature["design"] else "None")
    branch = ("[`%s`](%s/tree/%s)" % (feature["branch"], base, feature["branch"])
              if feature["branch"] else "None")
    issue = ("[#%d](%s/issues/%d)" % (feature["issue"], base, feature["issue"])
             if feature["issue"] else "None")
    rows = [
        ("Id", "`%s`" % feature["id"]),
        ("Status", '<span class="roadmap-badge roadmap-badge--%s">%s</span>'
         % (feature["status"], BADGE_LABELS[feature["status"]])),
        ("Pillar", "[%s](%s.md)" % (pillar["title"], pillar["slug"])),
        ("Sprint", str(feature["sprint"]) if feature["sprint"] else "None"),
        ("Depends on", _feature_links(feature["depends_on"], features)),
        ("Unblocks", _feature_links(feature["unblocks"], features)),
        ("Docs", docs), ("Branch", branch), ("Design", design), ("Issue", issue),
    ]
    lines += ["| Field | Value |", "| --- | --- |"]
    lines += ["| %s | %s |" % row for row in rows]
    lines += ["", "Back to the [roadmap graph](index.md).", ""]
    return "\n".join(lines)


def _dump(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def graph_json(pillars, features):
    order = {p["prefix"]: p["order"] for p in pillars}
    nodes = []
    for f in sorted(features.values(), key=lambda f: (order[f["pillar"]], f["id"])):
        nodes.append({"id": f["id"], "title": f["title"], "status": f["status"],
                      "pillar": f["pillar"], "summary": f["summary"], "sprint": f["sprint"],
                      "draft": f["draft"], "url": f["id"].lower() + "/"})
    edges = []
    for f in sorted(features.values(), key=lambda f: f["id"]):
        for dep in f["depends_on"]:
            edges.append({"id": "%s>%s" % (dep, f["id"]), "source": dep, "target": f["id"],
                          "draft": f["draft"]})
    edges.sort(key=lambda e: (e["target"], e["source"]))
    return _dump({
        "pillars": [{"prefix": p["prefix"], "title": p["title"], "slug": p["slug"],
                     "order": p["order"], "url": p["slug"] + "/"} for p in pillars],
        "nodes": nodes, "edges": edges})


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
             "sprint": f["sprint"], "docs": f["docs"], "branch": f["branch"],
             "design": f["design"], "issue": f["issue"], "draft": f["draft"],
             "url": "%sroadmap/%s/" % (site, f["id"].lower())}
            for f in sorted(features.values(), key=lambda f: f["id"])]})


def generate(root, site=DEFAULT_SITE):
    """Returns [(src_uri, content, in_nav)] for every generated file, in a stable order."""
    if not site.endswith("/"):
        site += "/"
    pillars, features = load_registry(root)
    by_prefix = {p["prefix"]: p for p in pillars}
    out = []
    for p in pillars:
        out.append(("roadmap/%s.md" % p["slug"], pillar_page(p), True))
    for fid in sorted(features):
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
