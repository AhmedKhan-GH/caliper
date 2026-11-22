"""Tests for scripts/check_theme_colors.py."""

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_theme_colors as chk  # noqa: E402


def lines(text):
    return [n for n, _ in chk.scan(textwrap.dedent(text))]


class ScanTest(unittest.TestCase):
    def test_tokens_pass(self):
        self.assertEqual(lines("""\
            constexpr ImU32 ground = tok::role::background_app.imu32;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
            c = (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
            """), [])

    def test_raw_literals_fail(self):
        self.assertEqual(lines("""\
            ImU32 a = IM_COL32(1, 2, 3, 255);
            ImU32 b = tok::role::text.imu32;
            ImVec4 c = ImVec4(1, 0, 0, 1);
            ImU32 d = 0xFF8800;
            ImU32 e = 0x80112233u;
            ImU32 f = IM_COL32_WHITE;
            ImU32 g = ImColor(1, 2, 3);
            """), [1, 3, 4, 5, 6, 7])

    def test_comments_and_strings_pass(self):
        self.assertEqual(lines("""\
            // bronze is IM_COL32(0xb3, 0x8b, 0x59, 255)
            /* 0x123456
               ImVec4(1, 2, 3, 4) */
            const char* s = "0xb38b59 IM_COL32(";
            int x = 3;  // 0xb38b59
            """), [])

    def test_region_allows_literals(self):
        self.assertEqual(lines("""\
            // BEGIN colours without a token
            constexpr ImU32 warn = IM_COL32(255, 196, 120, 255);
            // END colours without a token
            ImU32 b = IM_COL32(1, 2, 3, 255);
            """), [4])

    def test_open_region_fails(self):
        self.assertEqual(lines("""\
            // BEGIN colours without a token
            int x;
            """), [0])

    def test_short_hex_numbers_pass(self):
        self.assertEqual(lines("int mask = 0xFF; int big = 0x1000;"), [])


class MainTest(unittest.TestCase):
    def run_main(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "theme.h").write_text(text, encoding="utf-8")
            argv = sys.argv
            sys.argv = ["check_theme_colors.py", "--root", tmp, "theme.h"]
            try:
                return chk.main()
            finally:
                sys.argv = argv

    def test_clean_file_exits_zero(self):
        self.assertEqual(self.run_main("constexpr ImU32 a = tok::role::text.imu32;\n"), 0)

    def test_raw_colour_exits_one(self):
        self.assertEqual(self.run_main("constexpr ImU32 a = IM_COL32(1, 2, 3, 255);\n"), 1)

    def test_missing_file_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = sys.argv
            sys.argv = ["check_theme_colors.py", "--root", tmp, "absent.h"]
            try:
                self.assertEqual(chk.main(), 2)
            finally:
                sys.argv = argv


if __name__ == "__main__":
    unittest.main()
