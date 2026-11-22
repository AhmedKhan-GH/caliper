"""Tests for roadmap/check.py and roadmap/build.py, on small fixture repositories."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

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
    title: Host
    status: shipped
    summary: The native application.
    depends_on: []
    docs: docs/guide.md#first-part
    issue: null
  - id: COR-02
    title: ABI
    status: shipped
    summary: The frozen header.
    depends_on:
      - COR-01
    docs: docs/guide.md
    issue: 7
  - id: COR-03
    title: Packaging
    status: built
    summary: Packages applets.
    depends_on:
      - COR-02
    sprint: 4
    branches: [feature/packaging]
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
    title: Scene core
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

PLAN_LEDGER = """\
title: Plan page
kind: page
source: design/plan.md
summary: The one plan page.
entries:
  - heading: The plan
    item: The packaging step and the scene core after it.
    ids: [COR-03, REN-01]
"""

DOCS_LEDGER = """\
title: Docs
kind: docs
summary: The pages of shipped features.
entries:
  - page: docs/guide.md#first-part
    item: The host and its ABI.
    ids: [COR-01, COR-02]
"""


def make_repo(tmp, pillars=None, ledger=None):
    root = Path(tmp)
    (root / "roadmap").mkdir()
    shutil.copy(HERE / "schema.json", root / "roadmap" / "schema.json")
    for name, text in (pillars or {"core": CORE, "rendering": RENDERING}).items():
        (root / "roadmap" / (name + ".yml")).write_text(text, encoding="utf-8")
    (root / "roadmap" / "coverage").mkdir()
    shutil.copy(HERE / "coverage" / "schema.json", root / "roadmap" / "coverage" / "schema.json")
    for name, text in (ledger or {"design-plan": PLAN_LEDGER, "docs": DOCS_LEDGER}).items():
        (root / "roadmap" / "coverage" / (name + ".yml")).write_text(text, encoding="utf-8")
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
        self.assertIn("0 in progress", summary)
        self.assertIn("2 ledger entries in 2 files", summary)

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
        self.assertError(errors, "COR-02", "docs")

    def test_yaml_extension_is_reported(self):
        root = make_repo(self.tmp)
        (root / "roadmap" / "extra.yaml").write_text(RENDERING.replace("REN", "EXT"))
        errors, _, _ = check.run_checks(root)
        self.assertError(errors, "extra.yaml", ".yml")

    def test_pillar_missing_from_the_section_navigation(self):
        root = make_repo(self.tmp)
        (root / "docs" / "roadmap").mkdir()
        (root / "docs" / "roadmap" / "index.md").write_text("# Roadmap\n")
        (root / "docs" / "roadmap" / ".nav.yml").write_text("nav:\n  - index.md\n  - core.md\n")
        errors, _, _ = check.run_checks(root)
        self.assertEqual(len(errors), 1, errors)
        self.assertError(errors, "rendering.yml", ".nav.yml", "rendering.md")

    def test_graph_page_needs_no_pillar_list(self):
        # The graph's rail links every pillar, so the page itself lists none.
        root = make_repo(self.tmp)
        (root / "docs" / "roadmap").mkdir()
        (root / "docs" / "roadmap" / "index.md").write_text("# Roadmap\n")
        (root / "docs" / "roadmap" / ".nav.yml").write_text(
            "nav:\n  - index.md\n  - core.md\n  - rendering.md\n")
        errors, _, _ = check.run_checks(root)
        self.assertEqual(errors, [])

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
        errors, _, _ = self.run_check(edit={"rendering": ("status: planned",
                                                          "status: built\n    branches: [x]")})
        self.assertEqual(errors, [])

    def test_unused_decision(self):
        extra = RENDERING + """\
  - id: REN-02
    title: Open question
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
    title: Open question
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
    title: Open question
    status: decision
    summary: Which way to go.
    depends_on: []
    issue: null
  - id: REN-03
    title: Step after the question
    status: planned
    summary: Waits for the question.
    depends_on:
      - REN-02
    issue: null
"""
        errors, warnings, _ = self.run_check(pillars={"core": CORE, "rendering": extra})
        self.assertEqual(errors, [])
        self.assertEqual([w for w in warnings if "implied" in w], [])

    def test_branches_must_be_a_list(self):
        errors, _, _ = self.run_check(edit={"core": ("branches: [feature/packaging]",
                                                     "branches: feature/packaging")})
        self.assertError(errors, "COR-03", "schema")

    def test_several_branches_and_in_progress(self):
        errors, _, summary = self.run_check(edit={"core": (
            "branches: [feature/packaging]", "branches: [feature/packaging, feature/more]\n"
            "    in_progress: true")})
        self.assertEqual(errors, [])
        self.assertIn("1 in progress", summary)

    def test_in_progress_only_on_built_work(self):
        errors, _, _ = self.run_check(edit={"rendering": ("status: planned",
                                                          "status: planned\n    in_progress: true")})
        self.assertError(errors, "REN-01", "schema")

    def test_title_with_an_article(self):
        for title in ("The ABI", "A packaging", "An ABI", "the ABI"):
            with self.subTest(title=title), tempfile.TemporaryDirectory() as self.tmp:
                errors, _, _ = self.run_check(edit={"core": ("title: Packaging", "title: " + title)})
                self.assertError(errors, "COR-03", "starts with an article")

    def test_title_starting_with_a_word_like_an_article(self):
        for title in ("Another packaging", "API packaging", "Theme packaging"):
            with self.subTest(title=title), tempfile.TemporaryDirectory() as self.tmp:
                errors, _, _ = self.run_check(edit={"core": ("title: Packaging", "title: " + title)})
                self.assertEqual(errors, [])

    def test_pillar_title_with_an_article(self):
        errors, _, _ = self.run_check(edit={"rendering": ("title: Rendering", "title: The rendering")})
        self.assertError(errors, "roadmap/rendering.yml", "pillar title starts with an article")

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


class LedgerTest(unittest.TestCase):
    """The coverage ledger: every entry maps to ids or gives a reason, and every plan heading
    under design/ has an entry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = make_repo(self._tmp.name)

    def ledger(self, name, text):
        (self.root / "roadmap" / "coverage" / (name + ".yml")).write_text(text, encoding="utf-8")

    def edit(self, rel, old, new):
        path = self.root / rel
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def check(self):
        return check.run_checks(self.root)

    def assertError(self, errors, *fragments):
        self.assertTrue(any(all(f in e for f in fragments) for e in errors),
                        "no error with %r in %r" % (fragments, errors))

    def test_valid(self):
        errors, warnings, _ = self.check()
        self.assertEqual((errors, warnings), ([], []))

    def test_new_plan_heading_fails(self):
        self.edit("design/plan.md", "## The plan\n", "## The plan\n\n### A new step\n")
        errors, _, _ = self.check()
        self.assertEqual(len(errors), 1, errors)
        self.assertError(errors, "roadmap/coverage/design-plan.yml", '"A new step"',
                         "design/plan.md", "no ledger entry")

    def test_level_four_headings_are_not_plan_headings(self):
        self.edit("design/plan.md", "## The plan\n", "## The plan\n\n#### A detail\n")
        self.assertEqual(self.check()[0], [])

    def test_heading_entry_covers_it(self):
        self.edit("design/plan.md", "## The plan\n", "## The plan\n\n### `A` new step\n")
        self.ledger("design-plan", PLAN_LEDGER + """\
  - heading: "`A` new step"
    item: A step that only tidies the code.
    reason: maintenance
""")
        self.assertEqual(self.check()[0], [])

    def test_not_features_covers_a_heading(self):
        self.edit("design/plan.md", "## The plan\n", "## The plan\n\n## Risks\n")
        self.ledger("design-plan", PLAN_LEDGER.replace("entries:", "not_features: [Risks]\nentries:"))
        self.assertEqual(self.check()[0], [])

    def test_stale_not_features(self):
        self.ledger("design-plan", PLAN_LEDGER.replace("entries:", "not_features: [Risks]\nentries:"))
        self.assertError(self.check()[0], "not_features names \"Risks\"")

    def test_stale_heading(self):
        self.ledger("design-plan", PLAN_LEDGER.replace("heading: The plan", "heading: Gone"))
        errors, _, _ = self.check()
        self.assertError(errors, '"The plan"', "no ledger entry")
        self.assertError(errors, '"Gone"', "not on design/plan.md")

    def test_entry_under_the_page_title(self):
        self.ledger("design-plan", PLAN_LEDGER + """\
  - heading: The plan page
    item: The page's opening paragraph.
    reason: context
""")
        self.assertEqual(self.check()[0], [])

    def test_heading_from_a_newer_version_on_a_branch(self):
        self.ledger("design-plan", PLAN_LEDGER + """\
  - heading: A step only the branch has
    ref: feature/newer
    item: The newer version's step.
    ids: [REN-01]
""")
        self.assertEqual(self.check()[0], [])

    def test_plan_page_without_ledger(self):
        (self.root / "design" / "other.md").write_text("# Other\n\n## Step\n")
        self.assertError(self.check()[0], "design/other.md", "no ledger file")

    def test_index_pages_are_not_plans(self):
        (self.root / "design" / "README.md").write_text("# Design\n\n## Pages\n")
        self.assertEqual(self.check()[0], [])

    def test_page_on_another_branch_needs_its_ref(self):
        text = PLAN_LEDGER.replace("design/plan.md", "design/later.md")
        self.ledger("design-later", text)
        self.assertError(self.check()[0], "design-later.yml", "design/later.md is not on this branch")
        self.ledger("design-later", text.replace("kind: page", "kind: page\nref: feature/later"))
        self.assertEqual(self.check()[0], [])

    def test_two_ledgers_for_one_page(self):
        self.ledger("design-plan-2", PLAN_LEDGER)
        self.assertError(self.check()[0], "design-plan-2.yml", "already has the ledger file")

    def test_unknown_id(self):
        self.ledger("design-plan", PLAN_LEDGER.replace("[COR-03, REN-01]", "[COR-03, REN-09]"))
        self.assertError(self.check()[0], "design-plan.yml", "REN-09", "does not exist")

    def test_entry_needs_ids_or_a_reason(self):
        self.ledger("design-plan", PLAN_LEDGER.replace("    ids: [COR-03, REN-01]\n", ""))
        self.assertError(self.check()[0], "design-plan.yml", "schema")

    def test_entry_has_not_both(self):
        self.ledger("design-plan", PLAN_LEDGER + "    reason: maintenance\n")
        self.assertError(self.check()[0], "design-plan.yml", "schema")

    def test_unknown_reason(self):
        self.ledger("design-plan", PLAN_LEDGER.replace("    ids: [COR-03, REN-01]\n",
                                                       "    reason: later\n"))
        self.assertError(self.check()[0], "design-plan.yml", "schema")

    def test_branch_entries_need_branch_and_state(self):
        self.ledger("branches", """\
title: Branch map
kind: branches
summary: Every branch with commits that main lacks.
entries:
  - branch: feature/packaging
    item: The packaging.
    ids: [COR-03]
""")
        self.assertError(self.check()[0], "branches.yml", "schema", "state")

    def test_docs_entry_page_must_exist(self):
        self.ledger("docs", DOCS_LEDGER.replace("#first-part", "#nope"))
        self.assertError(self.check()[0], "docs.yml", "anchor #nope")

    def test_ledger_title_with_an_article(self):
        self.ledger("docs", DOCS_LEDGER.replace("title: Docs", "title: A docs group"))
        self.assertError(self.check()[0], "docs.yml", "title starts with an article")

    def test_feature_without_source_warns(self):
        self.ledger("docs", DOCS_LEDGER.replace("[COR-01, COR-02]", "[COR-01]"))
        errors, warnings, _ = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("COR-02: warning: no ledger entry", warnings[0])

    def test_missing_ledger(self):
        shutil.rmtree(self.root / "roadmap" / "coverage")
        self.assertError(self.check()[0], "roadmap/coverage", "missing")


CANDIDATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["pillar", "title", "proposed", "candidates"],
    "properties": {
        "pillar": {"type": "string"},
        "title": {"type": "string"},
        "proposed": {"type": "boolean"},
        "description": {"type": "string"},
        "candidates": {"type": "array", "items": {
            "type": "object",
            "required": ["key", "title", "summary", "origin"],
            "properties": {
                "key": {"type": "string"},
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "origin": {"enum": ["cut", "plan-idea", "adjacent-work", "field"]},
                "depends_on": {"type": "array", "items": {"type": "string"}},
            },
        }},
    },
}

CAND_CORE = """\
pillar: COR
title: Core Runtime
proposed: false
candidates:
  - key: host-tracing
    title: Host Tracing
    summary: Traces the host.
    origin: cut
    depends_on: [COR-01]
  - key: abi-diff
    title: ABI Diff
    summary: Diffs two headers.
    origin: field
"""

CAND_DATA = """\
pillar: DAT
title: Datasets
proposed: true
description: Dataset tools.
candidates:
  - key: dataset-health
    title: Dataset Health Report
    summary: Reports on a dataset.
    origin: field
    depends_on: [REN-01]
"""


class CandidateTest(unittest.TestCase):
    """Candidate files under candidates/ are checked and stay out of the registry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = make_repo(self._tmp.name)
        self.folder = self.root / "candidates"
        self.folder.mkdir()
        (self.folder / "schema.json").write_text(json.dumps(CANDIDATE_SCHEMA), encoding="utf-8")
        self.write("core", CAND_CORE)
        self.write("datasets", CAND_DATA)

    def write(self, stem, text, ext=".yml"):
        (self.folder / (stem + ext)).write_text(text, encoding="utf-8")

    def edit(self, stem, old, new):
        path = self.folder / (stem + ".yml")
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def check(self):
        return check.run_checks(self.root)

    def assertError(self, errors, *fragments):
        self.assertTrue(any(all(f in e for f in fragments) for e in errors),
                        "no error with %r in %r" % (fragments, errors))

    def test_valid(self):
        errors, warnings, summary = self.check()
        self.assertEqual((errors, warnings), ([], []))
        self.assertIn("4 features, 3 edges", summary)
        self.assertIn("2 ledger entries in 2 files", summary)
        self.assertTrue(summary.endswith(", 3 candidates in 2 files"), summary)

    def test_no_folder_changes_nothing(self):
        shutil.rmtree(self.folder)
        errors, _, summary = self.check()
        self.assertEqual(errors, [])
        self.assertNotIn("candidates", summary)

    def test_candidates_stay_out_of_the_counts(self):
        _, _, with_candidates = self.check()
        shutil.rmtree(self.folder)
        _, _, without = self.check()
        self.assertTrue(with_candidates.startswith(without), (with_candidates, without))

    def test_schema_failure(self):
        self.edit("core", "origin: cut", "origin: other")
        self.assertError(self.check()[0], "candidates/core.yml", "schema", "origin")

    def test_invalid_yaml(self):
        self.write("core", "pillar: [COR\n")
        self.assertError(self.check()[0], "candidates/core.yml", "not valid YAML")

    def test_missing_schema(self):
        (self.folder / "schema.json").unlink()
        self.assertError(self.check()[0], "candidates/schema.json", "schema is missing")

    def test_yaml_extension_is_not_read(self):
        self.write("rendering", CAND_CORE, ext=".yaml")
        self.assertError(self.check()[0], "candidates/rendering.yaml", "not read")

    def test_existing_pillar_must_be_a_registry_prefix(self):
        self.edit("core", "pillar: COR", "pillar: XYZ")
        self.assertError(self.check()[0], "candidates/core.yml", "XYZ is not a registry prefix")

    def test_existing_pillar_file_stem(self):
        self.write("rendering", CAND_CORE.replace("key: host", "key: rhost").replace(
            "key: abi", "key: rabi"))
        self.assertError(self.check()[0], "candidates/rendering.yml", "COR", "core.yml")

    def test_proposed_prefix_is_new(self):
        self.edit("datasets", "pillar: DAT", "pillar: REN")
        self.assertError(self.check()[0], "candidates/datasets.yml",
                         "proposed pillar REN is already a registry prefix")

    def test_prefixes_are_unique(self):
        self.write("more-data", CAND_DATA.replace("dataset-health", "other-key").replace(
            "Dataset Health Report", "Other Report"))
        self.assertError(self.check()[0], "more-data.yml", "duplicate pillar DAT", "datasets.yml")

    def test_keys_are_unique_across_files(self):
        self.write("more-data", CAND_DATA.replace("pillar: DAT", "pillar: SCI").replace(
            "Dataset Health Report", "Other Report"))
        self.assertError(self.check()[0], "more-data.yml", "dataset-health", "duplicate key",
                         "datasets.yml")

    def test_depends_on_must_exist(self):
        self.edit("core", "[COR-01]", "[COR-99]")
        self.assertError(self.check()[0], "candidates/core.yml", "host-tracing",
                         "depends_on COR-99, which does not exist")

    def test_candidate_title_rule(self):
        for title, reason in (("The Host Tracing", "article"), ("Host tracing", "Title Case"),
                              ("Host Tracing?", "question"), ("Host: Tracing", "colon")):
            with self.subTest(title=title):
                self.write("core", CAND_CORE)
                self.edit("core", "title: Host Tracing", "title: '%s'" % title)
                errors, _, _ = self.check()
                self.assertError(errors, "host-tracing", title, reason)

    def test_file_title_rule(self):
        self.edit("core", "title: Core Runtime", "title: Core runtime")
        self.assertError(self.check()[0], "candidates/core.yml", "-:", "Core runtime")

    def test_key_is_not_a_feature_id(self):
        self.edit("core", "key: abi-diff", "key: cor-03")
        self.assertError(self.check()[0], "cor-03", "registry feature id")

    def test_title_is_not_a_feature_title(self):
        self.edit("core", "title: ABI Diff", "title: packaging")
        self.assertError(self.check()[0], "abi-diff", "title of the feature COR-03")

    def test_no_todo(self):
        self.edit("core", "Traces the host.", "TODO write this.")
        self.assertError(self.check()[0], "candidates/core.yml", "contains TODO")

    def test_candidates_do_not_satisfy_the_registry(self):
        self.edit("core", "[COR-01]", "[REN-01]")
        errors, _, _ = self.check()
        self.assertEqual(errors, [])
        self.edit("datasets", "[REN-01]", "[DAT-01]")
        self.assertError(self.check()[0], "dataset-health", "DAT-01, which does not exist")


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
        self.assertIn('- <span class="roadmap-id">COR-03</span> [Packaging](cor-03.md). '
                      "Packages applets. Sprint 4.", pillar)

        page = (self.out / "roadmap" / "cor-03.md").read_text()
        self.assertIn('!!! note "Draft"', page)
        self.assertIn("| Field | Value |", page)
        self.assertIn('| Status | <span class="roadmap-badge roadmap-badge--built">Built</span> |',
                      page)
        self.assertIn("| Pillar | [Core runtime](core.md) |", page)
        self.assertIn("| Sprint | 4 |", page)
        self.assertIn("| Depends on | [ABI](cor-02.md) (`COR-02`) |", page)
        self.assertIn("| Unblocks | [Scene core](ren-01.md) (`REN-01`) |", page)
        self.assertIn("| Docs | None |", page)
        self.assertIn("| Branches | `feature/packaging` |", page)
        self.assertIn("| Design | [The plan page, The plan]"
                      "(https://github.com/AhmedKhan-GH/caliper/blob/main/design/plan.md"
                      "#the-plan) |", page)
        self.assertTrue(page.rstrip().endswith(
            'Back to the [roadmap graph](index.md), or to <a href="../#COR-03">this feature in it</a>.'))
        self.assertIn("## Sources", page)
        self.assertIn("- [design/plan.md](https://github.com/AhmedKhan-GH/caliper/blob/main/"
                      "design/plan.md#the-plan), under *The plan*. The packaging step and the "
                      "scene core after it.", page)
        self.assertNotIn("in progress", page)

        shipped = (self.out / "roadmap" / "cor-01.md").read_text()
        self.assertNotIn("Draft", shipped)
        self.assertIn("| Docs | [The guide, First part](../guide.md#first-part) |", shipped)
        self.assertIn("- [The guide, First part](../guide.md#first-part). The host and its ABI.",
                      shipped)
        second = (self.out / "roadmap" / "cor-02.md").read_text()
        self.assertIn("| Issue | [#7](https://github.com/AhmedKhan-GH/caliper/issues/7) |",
                      second)

    def test_in_progress_and_branch_sources(self):
        path = self.root / "roadmap" / "core.yml"
        path.write_text(path.read_text().replace(
            "branches: [feature/packaging]",
            "branches: [feature/packaging, feature/more]\n    in_progress: true"), encoding="utf-8")
        (self.root / "roadmap" / "coverage" / "branches.yml").write_text("""\
title: Branch map
kind: branches
summary: Every branch with commits that main lacks.
entries:
  - branch: feature/more
    state: unique
    wip: true
    commits: 2 commits, abc1234 and def5678
    item: "WIP: the packaging's second half."
    ids: [COR-03]
""", encoding="utf-8")
        self.run_build()
        page = (self.out / "roadmap" / "cor-03.md").read_text()
        self.assertIn('| Status | <span class="roadmap-badge roadmap-badge--progress">In progress'
                      '</span> |', page)
        self.assertIn("| Branches | `feature/packaging`<br>`feature/more` |", page)
        self.assertIn("unfinished, so the feature shows as in progress", page)
        self.assertIn("- Branch `feature/more`, 2 commits, abc1234 and def5678. WIP: the "
                      "packaging's second half.", page)
        pillar = (self.out / "roadmap" / "core.md").read_text()
        self.assertIn("## In progress", pillar)
        self.assertNotIn("## Built", pillar)
        graph = json.loads((self.out / "roadmap" / "graph.json").read_text())
        node = graph["nodes"][2]
        self.assertEqual((node["status"], node["in_progress"], node["branches"]),
                         ("built", True, ["feature/packaging", "feature/more"]))
        data = json.loads((self.out / "roadmap.json").read_text())
        sources = data["features"][2]["sources"]
        self.assertEqual([s["ledger"] for s in sources],
                         ["roadmap/coverage/branches.yml", "roadmap/coverage/design-plan.yml"])
        self.assertEqual(sources[1]["source"], "design/plan.md")

    def test_docs_source_left_out_of_the_site(self):
        (self.root / "docs" / "draft.md").write_text("# Draft\n")
        (self.root / "mkdocs.yml").write_text("site_name: Test\nexclude_docs: |\n  draft.md\n")
        (self.root / "roadmap" / "coverage" / "docs.yml").write_text(
            DOCS_LEDGER + "  - page: docs/draft.md\n    item: A draft.\n    ids: [COR-01]\n")
        self.run_build()
        page = (self.out / "roadmap" / "cor-01.md").read_text()
        self.assertIn("- `docs/draft.md`, which the docs site leaves out. A draft.", page)

    def test_page_read_from_another_branch(self):
        (self.root / "roadmap" / "coverage" / "design-later.yml").write_text(
            PLAN_LEDGER.replace("design/plan.md", "design/later.md")
            .replace("kind: page", "kind: page\nref: feature/later")
            .replace("[COR-03, REN-01]", "[REN-01]"), encoding="utf-8")
        self.run_build()
        page = (self.out / "roadmap" / "ren-01.md").read_text()
        self.assertIn("- `design/later.md` on the branch `feature/later`, under *The plan*. "
                      "The packaging step and the scene core after it.", page)

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

    def write_candidates(self, **texts):
        folder = self.root / "candidates"
        folder.mkdir()
        (folder / "schema.json").write_text(json.dumps(CANDIDATE_SCHEMA), encoding="utf-8")
        for stem, text in texts.items():
            (folder / (stem + ".yml")).write_text(text, encoding="utf-8")

    def test_candidates_json(self):
        in_progress = self.root / "roadmap" / "core.yml"
        in_progress.write_text(in_progress.read_text().replace(
            "branches: [feature/packaging]",
            "branches: [feature/packaging]\n    in_progress: true"), encoding="utf-8")
        core = CAND_CORE.replace("depends_on: [COR-01]", "serves: Debugging.\n"
                                 "    depends_on: [COR-03, COR-01]")
        data = CAND_DATA.replace("candidates:", "alternative: Fold into the core.\ncandidates:")
        self.write_candidates(datasets=data, core=core)
        self.run_build()
        out = json.loads((self.out / "roadmap" / "candidates.json").read_text())
        self.assertEqual([o["origin"] for o in out["origins"]],
                         [origin for origin, _ in build.ORIGINS])
        self.assertEqual(out["origins"][0], {"origin": "cut", "title": "Scope Cuts"})
        # Existing pillars come first, then the proposed ones.
        self.assertEqual([p["prefix"] for p in out["pillars"]], ["COR", "DAT"])
        core, data = out["pillars"]
        self.assertEqual((core["slug"], core["proposed"], core["url"]),
                         ("core", False, "candidates/core/"))
        self.assertEqual((core["description"], core["alternative"]), (None, None))
        self.assertEqual((data["description"], data["alternative"]),
                         ("Dataset tools.", "Fold into the core."))
        # Candidates keep their file order, and dependencies sort by number.
        self.assertEqual([c["key"] for c in core["candidates"]], ["host-tracing", "abi-diff"])
        self.assertEqual(core["candidates"][0]["depends_on"], ["COR-01", "COR-03"])
        self.assertEqual(core["candidates"][0]["serves"], "Debugging.")
        self.assertEqual(core["candidates"][1]["depends_on"], [])
        # Only the features that a candidate depends on, in id order.
        self.assertEqual(list(out["features"]), ["COR-01", "COR-03", "REN-01"])
        self.assertEqual(out["features"]["COR-03"],
                         {"title": "Packaging", "status": "progress", "url": "cor-03/",
                          "graph": "./#COR-03"})
        self.assertEqual(out["features"]["COR-01"]["status"], "shipped")
        self.assertEqual(out["features"]["REN-01"]["status"], "planned")

    def test_no_candidates_json_without_the_folder(self):
        self.run_build()
        self.assertFalse((self.out / "roadmap" / "candidates.json").exists())

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

    def test_ids_sort_by_number(self):
        path = self.root / "roadmap" / "rendering.yml"
        path.write_text(path.read_text() + """\
  - id: REN-100
    title: Hundredth
    status: planned
    summary: Comes last.
    depends_on: []
    issue: null
  - id: REN-99
    title: Ninety-ninth
    status: planned
    summary: Comes before the hundredth.
    depends_on: []
    issue: null
""", encoding="utf-8")
        pillars, _ = build.load_registry(self.root)
        self.assertEqual([f["id"] for f in pillars[1]["features"]], ["REN-01", "REN-99", "REN-100"])
        graph = json.loads(dict((u, c) for u, c, _ in build.generate(self.root))["roadmap/graph.json"])
        self.assertEqual([n["id"] for n in graph["nodes"]][-3:], ["REN-01", "REN-99", "REN-100"])


class EscapingTest(unittest.TestCase):
    """Registry text reaches the pages as typed, whatever Markdown or HTML characters it holds."""

    SHOWN = "Pipes | &lt;folder&gt; *stars* [brackets] snake_case"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        core = CORE.replace("title: ABI",
                            'title: "Pipes | <folder> *stars* [brackets] snake_case"')
        core = core.replace("summary: The frozen header.",
                            'summary: "Reads a | b and <folder> as text, with *x* and [y]."')
        self.root = make_repo(self._tmp.name, {"core": core, "rendering": RENDERING})

    def render(self, src_uri):
        import markdown
        pages = {uri: text for uri, text, _ in build.generate(self.root)}
        body = pages[src_uri].split("---\n", 2)[2]
        return markdown.markdown(body, extensions=["tables", "admonition"])

    def test_feature_page(self):
        page = self.render("roadmap/cor-02.md")
        self.assertIn("<h1>%s</h1>" % self.SHOWN, page)
        self.assertIn("Reads a | b and &lt;folder&gt; as text, with *x* and [y].", page)
        self.assertEqual(page.count("<tr>"), 11)  # the header and ten rows

    def test_a_dependent_page_keeps_its_table_rows(self):
        page = self.render("roadmap/cor-03.md")  # COR-03 depends on COR-02
        self.assertEqual(page.count("<tr>"), 11)
        self.assertIn(self.SHOWN + "</a>", page)

    def test_pillar_page(self):
        page = self.render("roadmap/core.md")
        self.assertIn(self.SHOWN + "</a>", page)
        self.assertNotIn("<em>", page)


class HookTest(unittest.TestCase):
    """The MkDocs hook adds every generated file of the real registry to the site."""

    def test_on_files(self):
        try:
            from mkdocs.config.defaults import MkDocsConfig
            from mkdocs.structure.files import Files, InclusionLevel
        except ImportError:
            self.skipTest("MkDocs is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "docs").mkdir()
            config = MkDocsConfig()
            # Named under hooks, as in mkdocs.yml, and run through MkDocs's own event.
            config.load_dict({"site_name": "Test", "docs_dir": tmp + "/docs",
                              "site_dir": tmp + "/site", "site_url": "https://example.org/docs/",
                              "hooks": [str(HERE / "build.py")]})
            errors, _ = config.validate()
            self.assertEqual(errors, [])
            files = config.plugins.on_files(Files([]), config=config)
            uris = {f.src_uri: f for f in files}
            pillars, features = build.load_registry(build.ROOT)
            # One page per candidate file, the candidates index and its data file, when the
            # folder exists.
            folder = build.ROOT / "candidates"
            extra = len(list(folder.glob("*.yml"))) + 2 if folder.is_dir() else 0
            self.assertEqual(len(uris), len(pillars) + len(features) + 2 + extra)
            self.assertEqual(uris["roadmap/cor-01.md"].inclusion, InclusionLevel.NOT_IN_NAV)
            self.assertNotEqual(uris["roadmap/core.md"].inclusion, InclusionLevel.NOT_IN_NAV)
            if folder.is_dir():
                # The navigator is its own header tab; its pillar pages stay out of the sidebar.
                self.assertNotEqual(uris["roadmap/candidates/index.md"].inclusion, InclusionLevel.NOT_IN_NAV)
                for uri in ["roadmap/candidates/%s.md" % f.stem for f in folder.glob("*.yml")]:
                    self.assertEqual(uris[uri].inclusion, InclusionLevel.NOT_IN_NAV, uri)
            site = json.loads(uris["roadmap.json"].content_string)["site"]
            self.assertEqual(site, "https://example.org/docs/")
            self.assertIn("nodes", json.loads(uris["roadmap/graph.json"].content_string))
            if folder.is_dir():
                data = json.loads(uris["roadmap/candidates.json"].content_string)
                total = sum(len(yaml.safe_load(f.read_text(encoding="utf-8"))["candidates"] or [])
                            for f in folder.glob("*.yml"))
                self.assertEqual(sum(len(p["candidates"]) for p in data["pillars"]), total)


if __name__ == "__main__":
    unittest.main()
