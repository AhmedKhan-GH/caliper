"""Unit tests for scripts/check_doc_references.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_doc_references as chk  # noqa: E402


class CheckDocReferencesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._write(
            "mkdocs.yml",
            "site_name: Test\nexclude_docs: |\n  ignored.md\nmarkdown_extensions:\n  - pymdownx.superfences:\n      custom_fences:\n        - format: !!python/name:pymdownx.superfences.fence_code_format\n",
        )
        self._write("src/host/host_cli.h", 'if (arg == "dev") {} else if (arg == "list") {}\n')
        self._write("src/host/real.cpp", "int x = 0;\n")
        self._write(
            "src/builder/commands/command_table.json",
            json.dumps({"version": 0, "commands": [{"name": "panel.create"}]}),
        )
        self._write(
            "python/caliper/__init__.py",
            '__all__ = ["show"]\ndef show(t):\n    """Show."""\n',
        )
        self._write(
            "python/caliper/interp/__init__.py",
            '__all__ = ["publish"]\ndef publish(n, t):\n    """Publish."""\n',
        )
        self._write("docs/target.md", "# Target Page\n\n## Details\n\nText.\n")
        self._write(
            "docs/guide.md",
            "# Guide\n\nSee [Target](target.md#details), `src/host/real.cpp`, "
            "`panel.create`, `caliper list`, `caliper.show`, and `caliper.interp.publish`.\n",
        )
        self._write("docs/ignored.md", "# Ignored\n\n`src/host/does_not_exist.cpp`\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def test_valid_references_pass(self):
        self.assertEqual(chk.check(self.root), [])

    def test_broken_relative_link_fails(self):
        self._write("docs/guide.md", "# Guide\n\nSee [Missing](missing.md).\n")
        problems = chk.check(self.root)
        self.assertTrue(any("broken link `missing.md`" in p for p in problems), problems)

    def test_broken_anchor_fails(self):
        self._write("docs/guide.md", "# Guide\n\nSee [Target](target.md#wrong-anchor).\n")
        problems = chk.check(self.root)
        self.assertTrue(any("broken anchor `#wrong-anchor`" in p for p in problems), problems)

    def test_stale_repo_path_fails(self):
        self._write("docs/guide.md", "# Guide\n\nChanged `src/host/deleted.cpp`.\n")
        problems = chk.check(self.root)
        self.assertTrue(any("stale path `src/host/deleted.cpp`" in p for p in problems), problems)

    def test_unknown_command_fails(self):
        self._write("docs/guide.md", "# Guide\n\nRun `panel.nonexistent`.\n")
        problems = chk.check(self.root)
        self.assertTrue(any("unknown command or code `panel.nonexistent`" in p for p in problems), problems)

    def test_unknown_cli_subcommand_fails(self):
        self._write("docs/guide.md", "# Guide\n\nRun `caliper vanish`.\n")
        problems = chk.check(self.root)
        self.assertTrue(any("unknown CLI subcommand `caliper vanish`" in p for p in problems), problems)

    def test_unknown_python_symbol_fails(self):
        self._write("docs/guide.md", "# Guide\n\nCall `caliper.nonexistent`.\n")
        problems = chk.check(self.root)
        self.assertTrue(any("unknown Python symbol `caliper.nonexistent`" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
