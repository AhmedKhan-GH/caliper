"""Unit tests for generate_feature_index.py, generate_mcp_reference.py, and generate_piece_reference.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_feature_index as gen_fi  # noqa: E402
import generate_mcp_reference as gen_mcp  # noqa: E402
import generate_piece_reference as gen_piece  # noqa: E402


class FeatureIndexGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._write(
            "roadmap/core.yml",
            """\
prefix: COR
title: Core
order: 1
summary: Core runtime features.
features:
  - id: COR-01
    title: Host
    status: shipped
    scope: core
    summary: Host binary.
    docs: docs/intro/arch.md#components
  - id: COR-02
    title: Future Idea
    status: planned
    scope: supporting
    summary: Planned feature.
""",
        )
        self._write(
            "docs/intro/arch.md",
            "# Architecture\n\n## Components\n\nHost details.\n",
        )
        self._write("src/main.cpp", "int main() { return 0; }\n")
        self._write("tests/test_main.cpp", "int main() { return 0; }\n")
        self._write(
            "roadmap/index/feature_code_tests.yml",
            """\
COR-01:
  code:
    - src/main.cpp
  tests:
    - tests/test_main.cpp
""",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_write_then_check_passes(self):
        problems, _ = gen_fi.run(self.root, write=True)
        self.assertEqual(problems, [])
        problems, _ = gen_fi.run(self.root, write=False)
        self.assertEqual(problems, [])

    def test_drift_fails_check(self):
        gen_fi.run(self.root, write=True)
        out = self.root / gen_fi.OUTPUT_REL
        out.write_text(out.read_text(encoding="utf-8") + "\nextra\n", encoding="utf-8")
        problems, _ = gen_fi.run(self.root, write=False)
        self.assertTrue(any("out of date" in p for p in problems), problems)

    def test_shipped_missing_tests_fails(self):
        self._write(
            "roadmap/index/feature_code_tests.yml",
            "COR-01:\n  code:\n    - src/main.cpp\n  tests: []\n",
        )
        problems, _ = gen_fi.run(self.root, write=True)
        self.assertTrue(any("lacks test locations" in p for p in problems), problems)

    def test_shipped_missing_code_path_on_disk_fails(self):
        self._write(
            "roadmap/index/feature_code_tests.yml",
            "COR-01:\n  code:\n    - src/missing.cpp\n  tests:\n    - tests/test_main.cpp\n",
        )
        problems, _ = gen_fi.run(self.root, write=True)
        self.assertTrue(any("src/missing.cpp does not exist" in p for p in problems), problems)

    def test_shipped_missing_docs_anchor_fails(self):
        self._write("docs/intro/arch.md", "# Architecture\n\n## Other\n")
        problems, _ = gen_fi.run(self.root, write=True)
        self.assertTrue(any("#components does not exist" in p for p in problems), problems)


class McpGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        table = {
            "version": 0,
            "commands": [
                {
                    "name": "app.info",
                    "agents": True,
                    "effect": "read",
                    "mutates": False,
                    "destructive": False,
                    "frame_thread": False,
                    "args": {"type": "object", "properties": {}, "required": []},
                    "result": {"type": "object", "properties": {"host": {"type": "string"}}},
                    "summary": "Return host info.",
                },
                {
                    "name": "history.undo",
                    "agents": False,
                    "effect": "edit",
                    "mutates": True,
                    "destructive": False,
                    "frame_thread": False,
                    "args": {"type": "object", "properties": {}, "required": []},
                    "result": {"type": "object", "properties": {}},
                    "summary": "Undo last step.",
                },
            ],
        }
        p = self.root / gen_mcp.TABLE_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(table), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_check_passes(self):
        problems, rendered = gen_mcp.run(self.root, write=True)
        self.assertEqual(problems, [])
        self.assertIn("`app_info`", rendered)
        self.assertIn("`history.undo`", rendered)
        problems, _ = gen_mcp.run(self.root, write=False)
        self.assertEqual(problems, [])

    def test_drift_fails(self):
        gen_mcp.run(self.root, write=True)
        out = self.root / gen_mcp.OUTPUT_REL
        out.write_text("changed\n", encoding="utf-8")
        problems, _ = gen_mcp.run(self.root, write=False)
        self.assertTrue(any("out of date" in p for p in problems), problems)


class PieceGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._write(
            gen_piece.PIECE_H_REL,
            """\
namespace piece_limits {
inline constexpr std::size_t kInputs = 8; // max inputs
}
inline constexpr std::array<const char*, 3> kPieceFiles{"common.wgsl", "derive.wgsl", "draw.wgsl"};
inline constexpr std::array<const char*, 2> kTensorDtypes{"f32", "i32"};
inline constexpr std::array<const char*, 1> kTextureFormats{"rgba8unorm"};
enum class InputKind { tensor, buffer };
""",
        )
        self._write(
            gen_piece.PIECES_H_REL,
            """\
inline constexpr const char* kPieceBadManifest = "piece.bad_manifest"; // invalid manifest
// ---- bind groups of a composed module
inline constexpr std::uint32_t kPreludeVersion = 1; // prelude version
""",
        )
        self._write(
            gen_piece.CHECKS_CPP_REL,
            'Verdict f() { return fail(kPieceBadManifest, "bad"); }\n',
        )
        defs = {k: {"type": "object", "required": [], "properties": {}} for k, _, _ in gen_piece.SCHEMA_SECTIONS}
        self._write(gen_piece.SCHEMA_REL, json.dumps({"$defs": defs}))

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def test_write_and_check_passes(self):
        problems, rendered = gen_piece.run(self.root, write=True)
        self.assertEqual(problems, [])
        self.assertIn("`piece.bad_manifest`", rendered)
        problems, _ = gen_piece.run(self.root, write=False)
        self.assertEqual(problems, [])

    def test_undocumented_check_code_fails(self):
        self._write(gen_piece.CHECKS_CPP_REL, 'Verdict f() { return fail(kPieceMystery, "bad"); }\n')
        problems, _ = gen_piece.run(self.root, write=True)
        self.assertTrue(any("kPieceMystery" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
