"""Tests for scripts/check_doc_headings.py."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_doc_headings as chk  # noqa: E402

PASSING = [
    "Applet Contract", "Tensor and Arrow Headers", "Torch per Platform",
    "Caliper LIB API", "C++ Sugar", "Zero-Copy Tensors", "In-Tree Applets",
    "`caliper.ui.v1`", "caliper.metrics.v1 and v1_1",
    "caliper.tensor_bridge.v1, v1_1 and v1_2", "caliper.geometry.v1 to v1_3",
    "HTTP API", "Vulkan and CUDA", "Web and Embedding", "Wire Protocol v3",
    "1. Threading Spine", "10. Threading Spine", "`caliper list`",
    "`key: value` Format", "Build on macOS and iOS", "Up-to-Date Notes",
    "[Applet Contract](applet.md)", "**Bold** Heading {#bold-heading}",
    "Install with pip", "Using `--flag` Options", "Tools & Plugins",
]

FAILING = [
    ("The applet contract", "starts with an article"),
    ("An Applet", "starts with an article"),
    ("a Note", "starts with an article"),
    ("1. The Spine", "starts with an article"),
    ("Your first applet", "starts with Your"),
    ("Input: the play/pause button", "contains a colon"),
    ("See it in the app", "not Title Case (it)"),
    ("Attach to a running script", "not Title Case (running)"),
    ("Zero-copy tensors", "not Title Case (Zero-copy)"),
    ("What is a hub?", "question"),
    ("A note", "starts with an article"),
    ("of Mice and Men", "not Title Case (of)"),
]


class JudgeTest(unittest.TestCase):
    def test_passing(self):
        for heading in PASSING:
            with self.subTest(heading=heading):
                self.assertIsNone(chk.judge(heading))

    def test_failing(self):
        for heading, reason in FAILING:
            with self.subTest(heading=heading):
                self.assertEqual(chk.judge(heading), reason)

    def test_colon_in_code_span_passes(self):
        self.assertIsNone(chk.judge("`key: value` Format"))

    def test_colon_outside_code_span_fails(self):
        self.assertEqual(chk.judge("`x` Format: Details"), "contains a colon")


class HeadingsTest(unittest.TestCase):
    def collect(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.md"
            path.write_text(text, encoding="utf-8")
            return list(chk.headings(path))

    def test_front_matter_and_fences_ignored(self):
        text = ("---\ntitle: the bad: one\n# not a heading\n---\n"
                "# Real Heading\n"
                "```md\n# the bad one\n```\n"
                "~~~~\n## the other bad one\n~~~\n### still inside\n~~~~\n"
                "````\n```\n# nested bad\n```\n--8<-- \"x\"\n````\n"
                "## After Fences\n")
        found = [(n, lvl, t) for n, lvl, t in self.collect(text)]
        self.assertEqual(found, [(5, 1, "Real Heading"), (20, 2, "After Fences")])

    def test_deep_headings_ignored(self):
        text = "#### the bad one\n##### also bad\n### Fine Heading\n"
        self.assertEqual(self.collect(text), [(3, 3, "Fine Heading")])

    def test_closing_hashes_removed(self):
        self.assertEqual(self.collect("## Title ##\n"), [(1, 2, "Title")])


class MainTest(unittest.TestCase):
    def run_main(self, files):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = chk.main(["--root", str(root)])
            return code, out.getvalue()

    def test_failure_returns_one(self):
        code, out = self.run_main({
            "docs/a.md": "# Good Page\n\n## The bad one\n",
            "docs/sub/b.md": "# Fine\n\n### Input: here\n",
        })
        self.assertEqual(code, 1)
        self.assertIn("docs/a.md:3: The bad one: starts with an article", out)
        self.assertIn("docs/sub/b.md:3: Input: here: contains a colon", out)
        self.assertIn("2 of 4 docs headings", out)

    def test_clean_returns_zero(self):
        code, out = self.run_main({"docs/a.md": "# Good Page\n\n## Tensor and Arrow Headers\n"})
        self.assertEqual(code, 0)
        self.assertIn("0 of 2", out)

    def test_missing_docs_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(chk.main(["--root", tmp]), 2)


if __name__ == "__main__":
    unittest.main()
