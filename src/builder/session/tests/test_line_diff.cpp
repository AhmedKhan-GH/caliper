// Tests of unified_line_diff(): the format of diff -u, context and merged
// hunks, texts without a final newline, empty texts, and the bound that keeps a
// large diff cheap.
#include <caliper_builder/session/line_diff.h>

#include <doctest/doctest.h>

#include <string>

using caliper_builder::unified_line_diff;

namespace {

std::string lines(int from, int to, const std::string& prefix = "line ") {
    std::string out;
    for (int i = from; i <= to; ++i) out += prefix + std::to_string(i) + "\n";
    return out;
}

}  // namespace

TEST_CASE("identical texts give no diff") {
    CHECK(unified_line_diff("", "").empty());
    CHECK(unified_line_diff("a\nb\n", "a\nb\n").empty());
}

TEST_CASE("one changed line has three lines of context on each side") {
    const std::string a = lines(1, 10);
    std::string b = a;
    b.replace(b.find("line 5\n"), 7, "line five\n");
    const std::string d = unified_line_diff(a, b, "a/x.spec.toml", "b/x.spec.toml");
    CHECK(d ==
          "--- a/x.spec.toml\n+++ b/x.spec.toml\n"
          "@@ -2,7 +2,7 @@\n line 2\n line 3\n line 4\n-line 5\n+line five\n line 6\n line 7\n line 8\n");
}

TEST_CASE("additions at the start and the end, and to an empty text") {
    CHECK(unified_line_diff("b\n", "a\nb\n") == "--- before\n+++ after\n@@ -1 +1,2 @@\n+a\n b\n");
    CHECK(unified_line_diff("a\n", "a\nb\n") == "--- before\n+++ after\n@@ -1 +1,2 @@\n a\n+b\n");
    CHECK(unified_line_diff("", "x\ny\n") == "--- before\n+++ after\n@@ -0,0 +1,2 @@\n+x\n+y\n");
    CHECK(unified_line_diff("x\ny\n", "") == "--- before\n+++ after\n@@ -1,2 +0,0 @@\n-x\n-y\n");
}

TEST_CASE("changes close together share a hunk and changes far apart do not") {
    const std::string a = lines(1, 30);
    std::string near = a;
    near.replace(near.find("line 10\n"), 8, "ten\n");
    near.replace(near.find("line 15\n"), 8, "fifteen\n");
    const std::string d1 = unified_line_diff(a, near);
    CHECK(d1.find("@@ -7,12 +7,12 @@") != std::string::npos);
    CHECK(d1.find("\n@@ ", d1.find("@@ -7") + 1) == std::string::npos);  // one hunk

    std::string far = a;
    far.replace(far.find("line 3\n"), 7, "three\n");
    far.replace(far.find("line 25\n"), 8, "twenty-five\n");
    const std::string d2 = unified_line_diff(a, far);
    CHECK(d2.find("@@ -1,6 +1,6 @@") != std::string::npos);
    CHECK(d2.find("@@ -22,7 +22,7 @@") != std::string::npos);
}

TEST_CASE("a text without a final newline is marked") {
    const std::string d = unified_line_diff("a\nb", "a\nb\n");
    CHECK(d == "--- before\n+++ after\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+b\n");
}

TEST_CASE("a large rewrite is bounded and still correct") {
    const std::string a = lines(1, 5000, "a");
    const std::string b = lines(1, 5000, "b");
    const std::string d = unified_line_diff(a, b);
    CHECK(d.find("@@ -1,5000 +1,5000 @@") != std::string::npos);
    std::size_t minus = 0;
    std::size_t plus = 0;
    for (std::size_t p = 0; (p = d.find('\n', p)) != std::string::npos; ++p) {
        if (p + 1 < d.size() && d[p + 1] == '-') ++minus;
        if (p + 1 < d.size() && d[p + 1] == '+') ++plus;
    }
    CHECK(minus == 5000);     // every removed line (the --- line starts the text)
    CHECK(plus == 5000 + 1);  // the +++ line and every added line
}

TEST_CASE("a reordering is shown with the longest common part kept") {
    const std::string d = unified_line_diff("a\nb\nc\nd\n", "b\nc\nd\na\n", "x", "y");
    CHECK(d == "--- x\n+++ y\n@@ -1,4 +1,4 @@\n-a\n b\n c\n d\n+a\n");
}
