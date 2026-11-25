"""Tests for scripts/shared_colors.py."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import shared_colors  # noqa: E402


def stage_managed_tree(dst_root: Path) -> None:
    rels = [
        shared_colors.SOURCE_REL,
        shared_colors.STERNBILD_CMAKE_REL,
        *(rel for rel, _ in shared_colors.FULL_TARGETS),
        *(rel for rel, _, _, _ in shared_colors.BLOCK_TARGETS),
    ]
    for rel in rels:
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)


class SharedColorsTest(unittest.TestCase):
    def test_repository_matches_source(self):
        self.assertEqual(shared_colors.check_files(REPO), [])
        self.assertEqual(shared_colors.main(["--root", str(REPO), "--check"]), 0)

    def test_write_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_managed_tree(root)
            self.assertEqual(shared_colors.write_files(root), [])
            self.assertEqual(shared_colors.check_files(root), [])

    def test_cpp_drift_fails_and_write_repairs(self):
        for rel in (
            "src/builder/spec/colormaps.cpp",
            "src/host/gfx/colormap_coeffs.h",
            "src/host/gfx/colormap_tables.h",
            "src/host/ui/sternbild_theme_colors.inc",
            "src/host/scene/scene_palette.h",
        ):
            with self.subTest(target=rel):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    stage_managed_tree(root)
                    target = root / rel
                    target.write_text(target.read_text(encoding="utf-8") + "// drift\n", encoding="utf-8")
                    issues = shared_colors.check_files(root)
                    self.assertEqual(len(issues), 1)
                    self.assertIn(rel, issues[0])
                    self.assertEqual(shared_colors.main(["--root", str(root), "--check"]), 1)
                    self.assertEqual(shared_colors.main(["--root", str(root), "--write"]), 0)
                    self.assertEqual(shared_colors.check_files(root), [])

    def test_typescript_and_html_drift_fails_and_write_repairs(self):
        mutations = (
            ("web/src/render/palette.gen.ts", "[18, 17, 15]", "[19, 17, 15]"),
            ("web/src/render/colormaps.gen.ts", "'berlin'", "'berlyn'"),
            ("web/src/render/colormap.ts", "103, 0, 31", "104, 0, 31"),
            ("web/src/views/theme.ts", "'#272622'", "'#272623'"),
            ("web/src/spec/index.html", "#201f1c", "#201f1d"),
        )
        for rel, old, new in mutations:
            with self.subTest(target=rel):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    stage_managed_tree(root)
                    target = root / rel
                    text = target.read_text(encoding="utf-8")
                    self.assertIn(old, text)
                    target.write_text(text.replace(old, new, 1), encoding="utf-8")
                    issues = shared_colors.check_files(root)
                    self.assertEqual(len(issues), 1)
                    self.assertIn(rel, issues[0])
                    self.assertEqual(shared_colors.main(["--root", str(root), "--check"]), 1)
                    self.assertEqual(shared_colors.main(["--root", str(root), "--write"]), 0)
                    self.assertEqual(shared_colors.check_files(root), [])

    def test_python_drift_fails_and_write_repairs(self):
        rel = "python/caliper/_colors_gen.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_managed_tree(root)
            target = root / rel
            text = target.read_text(encoding="utf-8")
            target.write_text(text.replace("'viridis': 0", "'viridis': 99", 1), encoding="utf-8")
            issues = shared_colors.check_files(root)
            self.assertEqual(len(issues), 1)
            self.assertIn(rel, issues[0])
            self.assertEqual(shared_colors.main(["--root", str(root), "--check"]), 1)
            self.assertEqual(shared_colors.main(["--root", str(root), "--write"]), 0)
            self.assertEqual(shared_colors.check_files(root), [])

    def test_sternbild_pin_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_managed_tree(root)
            cmake_file = root / shared_colors.STERNBILD_CMAKE_REL
            text = cmake_file.read_text(encoding="utf-8")
            cmake_file.write_text(text.replace("design-v2.0.0", "design-v9.9.9", 1), encoding="utf-8")
            issues = shared_colors.check_files(root)
            self.assertEqual(len(issues), 1)
            self.assertIn(shared_colors.STERNBILD_CMAKE_REL, issues[0])
            self.assertEqual(shared_colors.main(["--root", str(root), "--check"]), 1)


if __name__ == "__main__":
    unittest.main()
