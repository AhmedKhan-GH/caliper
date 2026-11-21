"""Tests for roadmap/check.py and roadmap/build.py, on small fixture repositories."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build  # noqa: E402
import check  # noqa: E402

CORE = """\
prefix: COR
title: Core runtime
order: 1
summary: The host and its ABI.
features:
  - id: COR-01
    title: The host
    status: shipped
    summary: The native application.
    depends_on: []
    docs: docs/guide.md#first-part
    issue: null
  - id: COR-02
    title: The ABI
    status: shipped
    summary: The frozen header.
    depends_on:
      - COR-01
    docs: docs/guide.md
    issue: 7
  - id: COR-03
    title: The packaging
    status: built
    summary: Packages applets.
    depends_on:
      - COR-02
    sprint: 4
    branch: feature/packaging
    design: design/plan.md#the-plan
    issue: null
    draft: true
"""

RENDERING = """\
prefix: REN
title: Rendering
order: 2
summary: Drawing scenes.
features:
  - id: REN-01
    title: The scene core
    status: planned
    summary: Holds the scene.
    depends_on:
      - COR-03
    issue: null
"""

GUIDE = """\
# The guide

## First part

Text.

```
# not a heading
```

## `Code` and [link](x.md)

## First part
"""

PLAN = "# The plan page\n\n## The plan\n"


def make_repo(tmp, pillars=None):
    root = Path(tmp)
    (root / "roadmap").mkdir()
    shutil.copy(HERE / "schema.json", root / "roadmap" / "schema.json")
    for name, text in (pillars or {"core": CORE, "rendering": RENDERING}).items():
        (root / "roadmap" / (name + ".yml")).write_text(text, encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text(GUIDE, encoding="utf-8")
    (root / "design").mkdir()
    (root / "design" / "plan.md").write_text(PLAN, encoding="utf-8")
    return root


class CheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def run_check(self, pillars=None, edit=None):
        root = make_repo(self.tmp, pillars)
        if edit:
            for name, (old, new) in edit.items():
                path = root / "roadmap" / (name + ".yml")
                text = path.read_text(encoding="utf-8")
                self.assertIn(old, text)
                path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return check.run_checks(root)

    def assertError(self, errors, *fragments):
        self.assertTrue(any(all(f in e for f in fragments) for e in errors),
                        "no error with %r in %r" % (fragments, errors))

    def test_valid_registry(self):
        errors, warnings, summary = self.run_check()
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertIn("4 features, 3 edges", summary)
        self.assertIn("2 shipped, 1 built, 1 planned", summary)

    def test_schema_failure(self):
        errors, _, _ = self.run_check(edit={"core": ("status: built", "status: finished")})
        self.assertError(errors, "roadmap/core.yml", "COR-03", "schema")

    def test_schema_requires_docs_when_shipped(self):
        errors, _, _ = self.run_check(edit={"core": ("    docs: docs/guide.md\n", "")})
        self.assertError(errors, "COR-02", "schema")

    def test_duplicate_prefix(self):
        errors, _, _ = self.run_check(edit={"rendering": ("prefix: REN", "prefix: COR")})
        self.assertError(errors, "duplicate prefix COR")

    def test_duplicate_order(self):
        errors, _, _ = self.run_check(edit={"rendering": ("order: 2", "order: 1")})
        self.assertError(errors, "duplicate order 1")

    def test_duplicate_id(self):
        errors, _, _ = self.run_check(edit={"core": ("id: COR-02", "id: COR-01")})
        self.assertError(errors, "COR-01", "duplicate id")

    def test_wrong_prefix(self):
        errors, _, _ = self.run_check(edit={"rendering": ("id: REN-01", "id: COR-09")})
        self.assertError(errors, "COR-09", "does not use the prefix REN")

    def test_unknown_dependency(self):
        errors, _, _ = self.run_check(edit={"rendering": ("- COR-03", "- COR-77")})
        self.assertError(errors, "REN-01", "COR-77", "does not exist")

    def test_self_dependency(self):
        errors, _, _ = self.run_check(edit={"rendering": ("- COR-03", "- REN-01")})
        self.assertError(errors, "REN-01", "depends on itself")

    def test_cycle(self):
        errors, _, _ = self.run_check(edit={
            "core": ("    depends_on: []\n    docs: docs/guide.md#first-part",
                     "    depends_on:\n      - COR-02\n    docs: docs/guide.md#first-part")})
        self.assertError(errors, "dependency cycle", "COR-01 -> COR-02 -> COR-01")

    def test_shipped_without_docs(self):
        errors, _, _ = self.run_check(edit={"core": ("docs: docs/guide.md\n", "docs: null\n")})
        self.assertError(errors, "COR-02")

    def test_missing_docs_page(self):
        errors, _, _ = self.run_check(edit={"core": ("docs: docs/guide.md\n",
                                                     "docs: docs/missing.md\n")})
        self.assertError(errors, "COR-02", "docs page docs/missing.md does not exist")

    def test_missing_docs_anchor(self):
        errors, _, _ = self.run_check(edit={"core": ("#first-part", "#nope")})
        self.assertError(errors, "COR-01", "anchor #nope")

    def test_repeated_heading_anchor_is_valid(self):
        errors, _, _ = self.run_check(edit={"core": ("#first-part", "#first-part_1")})
        self.assertEqual(errors, [])

    def test_code_and_link_heading_anchor(self):
        errors, _, _ = self.run_check(edit={"core": ("#first-part", "#code-and-link")})
        self.assertEqual(errors, [])

    def test_fenced_comment_is_not_a_heading(self):
        errors, _, _ = self.run_check(edit={"core": ("#first-part", "#not-a-heading")})
        self.assertError(errors, "anchor #not-a-heading")

    def test_missing_design_page(self):
        errors, _, _ = self.run_check(edit={"core": ("design/plan.md#the-plan",
                                                     "design/absent.md")})
        self.assertError(errors, "COR-03", "design page design/absent.md does not exist")

    def test_missing_design_anchor(self):
        errors, _, _ = self.run_check(edit={"core": ("#the-plan", "#other")})
        self.assertError(errors, "COR-03", "design anchor #other")

    def test_higher_rank_dependency(self):
        errors, _, _ = self.run_check(edit={"core": (
            "      - COR-01\n    docs: docs/guide.md\n", "      - COR-03\n    docs: docs/guide.md\n")})
        self.assertError(errors, "COR-02", "COR-03", "higher status rank")

    def test_built_may_depend_on_built(self):
        errors, _, _ = self.run_check(edit={"rendering": ("status: planned", "status: built\n    branch: x")})
        self.assertEqual(errors, [])

    def test_unused_decision(self):
        extra = RENDERING + """\
  - id: REN-02
    title: An open question
    status: decision
    summary: Which way to go.
    depends_on: []
    issue: null
"""
        errors, _, _ = self.run_check(pillars={"core": CORE, "rendering": extra})
        self.assertError(errors, "REN-02", "decision that no feature depends on")

    def test_used_decision(self):
        extra = RENDERING.replace("      - COR-03\n", "      - COR-03\n      - REN-02\n") + """\
  - id: REN-02
    title: An open question
    status: decision
    summary: Which way to go.
    depends_on: []
    issue: null
"""
        errors, _, _ = self.run_check(pillars={"core": CORE, "rendering": extra})
        self.assertEqual(errors, [])

    def test_todo(self):
        errors, _, _ = self.run_check(edit={"rendering": ("Holds the scene.", "TODO")})
        self.assertError(errors, "REN-01", "summary contains TODO")

    def test_todo_in_pillar_summary(self):
        errors, _, _ = self.run_check(edit={"rendering": ("Drawing scenes.", "TODO")})
        self.assertError(errors, "roadmap/rendering.yml", "contains TODO")

    def test_redundant_edge_warning(self):
        _, warnings, _ = self.run_check(edit={"rendering": ("      - COR-03\n",
                                                            "      - COR-03\n      - COR-01\n")})
        self.assertEqual(len(warnings), 1)
        self.assertIn("REN-01", warnings[0])
        self.assertIn("warning", warnings[0])
        self.assertIn("COR-01", warnings[0])

    def test_point_sprint(self):
        errors, _, _ = self.run_check(edit={"core": ("    sprint: 4\n", '    sprint: "3.2"\n')})
        self.assertEqual(errors, [])

    def test_unquoted_point_sprint(self):
        # Unquoted, 3.2 is a YAML float, which the schema refuses.
        errors, _, _ = self.run_check(edit={"core": ("    sprint: 4\n", "    sprint: 3.2\n")})
        self.assertError(errors, "COR-03", "schema")

    def test_decision_edge_is_never_redundant(self):
        extra = RENDERING.replace("      - COR-03\n",
                                  "      - COR-03\n      - REN-02\n      - REN-03\n") + """\
  - id: REN-02
    title: An open question
    status: decision
    summary: Which way to go.
    depends_on: []
    issue: null
  - id: REN-03
    title: A step after the question
    status: planned
    summary: Waits for the question.
    depends_on:
      - REN-02
    issue: null
"""
        errors, warnings, _ = self.run_check(pillars={"core": CORE, "rendering": extra})
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_command_line_exit_codes(self):
        root = make_repo(self.tmp)
        script = str(HERE / "check.py")
        ok = subprocess.run([sys.executable, script, "--root", str(root)],
                            capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertTrue(ok.stdout.startswith("ok: 4 features"))
        (root / "docs" / "guide.md").unlink()
        bad = subprocess.run([sys.executable, script, "--root", str(root)],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("roadmap/core.yml: COR-01: docs page docs/guide.md does not exist",
                      bad.stdout)


class BuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = make_repo(self._tmp.name)
        self.out = Path(self._tmp.name) / "out"

    def run_build(self):
        result = subprocess.run([sys.executable, str(HERE / "build.py"), "--root",
                                 str(self.root), "--out", str(self.out)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_dependency_names_file_and_id(self):
        path = self.root / "roadmap" / "rendering.yml"
        path.write_text(path.read_text().replace("COR-03", "COR-77"), encoding="utf-8")
        with self.assertRaises(build.RegistryError) as ctx:
            build.load_registry(self.root)
        self.assertIn("roadmap/rendering.yml", str(ctx.exception))
        self.assertIn("REN-01", str(ctx.exception))

    def test_unblocks(self):
        _, features = build.load_registry(self.root)
        self.assertEqual(features["COR-02"]["unblocks"], ["COR-03"])
        self.assertEqual(features["COR-03"]["unblocks"], ["REN-01"])

    def test_parse_headings(self):
        slugs = [s for _, _, s in build.parse_headings(GUIDE)]
        self.assertEqual(slugs, ["the-guide", "first-part", "code-and-link", "first-part_1"])

    def test_html_placeholder_is_left_out_of_the_slug(self):
        # MkDocs reads <folder> outside backticks as an HTML tag and drops it, as on cli.md.
        headings = build.parse_headings("### caliper check <folder>\n\n## The `<folder>` form\n")
        self.assertEqual([s for _, _, s in headings], ["caliper-check", "the-folder-form"])

    def test_output_files(self):
        self.run_build()
        for name in ("core.md", "rendering.md", "cor-01.md", "cor-03.md", "ren-01.md",
                     "graph.json"):
            self.assertTrue((self.out / "roadmap" / name).is_file(), name)
        self.assertTrue((self.out / "roadmap.json").is_file())

        pillar = (self.out / "roadmap" / "core.md").read_text()
        self.assertTrue(pillar.startswith("---\ntemplate: roadmap.html\n---\n\n# Core runtime\n"))
        self.assertIn("The [roadmap graph](index.md) shows how these features depend on "
                      "features in other pillars.", pillar)
        self.assertLess(pillar.index("## Shipped"), pillar.index("## Built"))
        self.assertNotIn("## Planned", pillar)
        self.assertIn('- <span class="roadmap-id">COR-03</span> [The packaging](cor-03.md). '
                      "Packages applets. Sprint 4.", pillar)

        page = (self.out / "roadmap" / "cor-03.md").read_text()
        self.assertIn('!!! note "Draft"', page)
        self.assertIn("| Field | Value |", page)
        self.assertIn('| Status | <span class="roadmap-badge roadmap-badge--built">Built</span> |',
                      page)
        self.assertIn("| Pillar | [Core runtime](core.md) |", page)
        self.assertIn("| Sprint | 4 |", page)
        self.assertIn("| Depends on | [The ABI](cor-02.md) (`COR-02`) |", page)
        self.assertIn("| Unblocks | [The scene core](ren-01.md) (`REN-01`) |", page)
        self.assertIn("| Docs | None |", page)
        self.assertIn("| Branch | [`feature/packaging`]"
                      "(https://github.com/AhmedKhan-GH/caliper/tree/feature/packaging) |", page)
        self.assertIn("| Design | [The plan page, The plan]"
                      "(https://github.com/AhmedKhan-GH/caliper/blob/main/design/plan.md"
                      "#the-plan) |", page)
        self.assertTrue(page.rstrip().endswith("Back to the [roadmap graph](index.md)."))

        shipped = (self.out / "roadmap" / "cor-01.md").read_text()
        self.assertNotIn("Draft", shipped)
        self.assertIn("| Docs | [The guide, First part](../guide.md#first-part) |", shipped)
        second = (self.out / "roadmap" / "cor-02.md").read_text()
        self.assertIn("| Issue | [#7](https://github.com/AhmedKhan-GH/caliper/issues/7) |",
                      second)

    def test_graph_json(self):
        self.run_build()
        graph = json.loads((self.out / "roadmap" / "graph.json").read_text())
        self.assertEqual([n["id"] for n in graph["nodes"]],
                         ["COR-01", "COR-02", "COR-03", "REN-01"])
        self.assertEqual([e["id"] for e in graph["edges"]],
                         ["COR-01>COR-02", "COR-02>COR-03", "COR-03>REN-01"])
        node = graph["nodes"][2]
        self.assertEqual((node["sprint"], node["draft"], node["url"], node["pillar"]),
                         (4, True, "cor-03/", "COR"))
        self.assertEqual(graph["nodes"][0]["sprint"], None)
        self.assertEqual([e["draft"] for e in graph["edges"]], [False, True, False])
        self.assertEqual(graph["pillars"][1], {"prefix": "REN", "title": "Rendering",
                                               "slug": "rendering", "order": 2,
                                               "url": "rendering/"})

    def test_registry_json(self):
        self.run_build()
        data = json.loads((self.out / "roadmap.json").read_text())
        self.assertEqual(data["site"], build.DEFAULT_SITE)
        self.assertEqual(data["pillars"][0]["features"], ["COR-01", "COR-02", "COR-03"])
        self.assertEqual(data["pillars"][0]["url"], build.DEFAULT_SITE + "roadmap/core/")
        feature = data["features"][2]
        self.assertEqual(feature["unblocks"], ["REN-01"])
        self.assertEqual(feature["url"], build.DEFAULT_SITE + "roadmap/cor-03/")

    def test_output_is_deterministic(self):
        first = build.generate(self.root)
        second = build.generate(self.root)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
