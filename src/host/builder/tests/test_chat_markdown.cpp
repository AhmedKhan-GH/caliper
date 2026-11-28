// The side chat's Markdown subset (chat_markdown.h), headless: the summary a
// local model wrote at the end of the side chat's turn in the walk-through
// (tests/fixtures/agent_summary.md, transcribed from the capture of that run),
// every prefix of it as it streams in, the inline marks, the blocks, what is
// shown as plain text, hostile input, and the layout with a fake measure.
#include "builder/chat_markdown.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

using namespace caliper_host;

namespace {

std::string fixture() {
    std::ifstream in(std::string(CALIPER_CHAT_FIXTURES) + "/agent_summary.md", std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string spans_text(const MdBlock& b) {
    std::string out;
    for (const MdSpan& s : b.spans) out += s.text;
    return out;
}

// "plain:Done|bold:Spec|code:x" for a quick look at a block's spans.
std::string shape(const MdBlock& b) {
    std::string out;
    for (const MdSpan& s : b.spans) {
        if (!out.empty()) out += "|";
        out += s.style == MdStyle::bold ? "bold:" : s.style == MdStyle::code ? "code:" : "plain:";
        out += s.text;
    }
    return out;
}

bool raw_marks(const std::string& plain) {
    return plain.find("**") != std::string::npos || plain.find('`') != std::string::npos ||
           plain.find("__") != std::string::npos;
}

// A fake measure: 7 points a character of text, 8 of code.
float fake(const std::string& t, MdFont f) {
    size_t chars = 0;
    for (unsigned char c : t) chars += (c & 0xC0) != 0x80;
    return (float)chars * (f == MdFont::mono ? 8.0f : 7.0f);
}

}  // namespace

TEST_CASE("chat markdown: the agent's summary renders as bold, code, a list and paragraphs") {
    const std::string text = fixture();
    REQUIRE(text.find("**Spec**") != std::string::npos);
    const std::vector<MdBlock> blocks = chat_markdown_blocks(text);
    REQUIRE(blocks.size() == 5);
    CHECK(blocks[0].kind == MdBlockKind::paragraph);
    CHECK(shape(blocks[0]) == "plain:Done \xE2\x80\x94 the plot is live and updating. What I built:");
    CHECK(blocks[1].kind == MdBlockKind::bullet);
    CHECK(blocks[1].indent == 0);
    REQUIRE(blocks[1].spans.size() >= 4);
    CHECK(shape(blocks[1]).rfind("bold:Spec|plain: |code:attention-entropy-per-layer|plain: with six sources", 0) == 0);
    CHECK(shape(blocks[1]).find("code:live:entropy.l0|plain: through |code:live:entropy.l5") != std::string::npos);
    CHECK(blocks[2].kind == MdBlockKind::bullet);
    CHECK(shape(blocks[2]) == "bold:One series panel|plain: with six lines, one per layer, over training steps.");
    CHECK(blocks[3].kind == MdBlockKind::paragraph);
    CHECK(blocks[4].kind == MdBlockKind::paragraph);
    // "[6, 8]" is not a link: it stays as text.
    CHECK(spans_text(blocks[4]).find("attn.entropy [6, 8] heatmap") != std::string::npos);

    const std::string plain = chat_markdown_plain(text);
    CHECK_FALSE(raw_marks(plain));
    CHECK(plain.find("\xE2\x80\xA2 Spec attention-entropy-per-layer with six sources") != std::string::npos);
}

TEST_CASE("chat markdown: every prefix of the summary, as it streams in, shows no raw marks") {
    const std::string text = fixture();
    size_t checked = 0;
    for (size_t n = 0; n <= text.size(); ++n) {
        // Prefixes that end inside a character are not sent by the chat; skip them.
        if (n < text.size() && ((unsigned char)text[n] & 0xC0) == 0x80) continue;
        const std::string plain = chat_markdown_plain(text.substr(0, n));
        INFO("prefix of " << n << " bytes: " << plain);
        CHECK_FALSE(raw_marks(plain));
        ++checked;
    }
    // One prefix ends before each character, and one is the whole text.
    size_t characters = 0;
    for (unsigned char c : text) characters += (c & 0xC0) != 0x80;
    CHECK(checked == characters + 1);
    // Mid-stream: an opening mark with no closing one yet, and a closing mark half there.
    CHECK(chat_markdown_plain("- **Spe") == "\xE2\x80\xA2 Spe");
    CHECK(chat_markdown_plain("- **Spec*") == "\xE2\x80\xA2 Spec");
    CHECK(chat_markdown_plain("the live `attn.ent") == "the live attn.ent");
    CHECK(chat_markdown_plain("- **") == "\xE2\x80\xA2 ");
    CHECK(chat_markdown_plain("a **") == "a ");
    CHECK(chat_markdown_plain("a **\n") == "a **");   // a finished line keeps a lone pair of marks
    CHECK(shape(chat_markdown_blocks("**Spec**")[0]) == "bold:Spec");
}

TEST_CASE("chat markdown: inline marks, and what shows as plain text") {
    auto one = [](const std::string& t) { return shape(chat_markdown_blocks(t).at(0)); };
    CHECK(one("**bold** and __bold__") == "bold:bold|plain: and |bold:bold");
    CHECK(one("*em* and _em_ are plain") == "plain:em and em are plain");
    CHECK(one("***both***") == "bold:both");
    CHECK(one("2 * 3 = 6, snake_case_names and a*b") == "plain:2 * 3 = 6, snake_case_names and a*b");
    CHECK(one("\\*not em\\* and \\`not code\\`") == "plain:*not em* and `not code`");
    CHECK(one("`a **b** c` and ``x ` y``") == "code:a **b** c|plain: and |code:x ` y");
    // Links and images keep their text; the target is dropped and nothing is clickable.
    CHECK(one("see [the docs](https://example.org/x) and ![a chart](file:///etc/passwd)") ==
          "plain:see the docs and a chart");
    CHECK(one("an autolink <https://example.org> stays as text") == "plain:an autolink https://example.org stays as text");
    CHECK(one("<b>tags</b> go, <script>x</script> too, <id> and a < b stay") ==
          "plain:tags go, x too, <id> and a < b stay");
    CHECK(one("**[bold link](http://x)**") == "bold:bold link");
}

TEST_CASE("chat markdown: blocks, and what shows as plain blocks") {
    const auto b = chat_markdown_blocks(
        "# A heading #\n"
        "\n"
        "1. first\n"
        "2) second, on\n"
        "   two lines\n"
        "  - nested\n"
        "* star\n"
        "\n"
        "> quoted **text**\n"
        "\n"
        "---\n"
        "| a | b |\n"
        "|---|:-:|\n"
        "| 1 | 2 |\n"
        "\n"
        "```python\n"
        "x = 1  # keep  spaces\n"
        "\n"
        "print(x)\n"
        "```\n"
        "after");
    REQUIRE(b.size() == 10);
    CHECK(shape(b[0]) == "plain:A heading");
    CHECK(b[1].kind == MdBlockKind::number);
    CHECK(b[1].label == "1.");
    CHECK(b[2].label == "2.");
    CHECK(shape(b[2]) == "plain:second, on two lines");
    CHECK(b[3].kind == MdBlockKind::bullet);
    CHECK(b[3].indent == 1);
    CHECK(b[4].kind == MdBlockKind::bullet);
    CHECK(b[4].indent == 0);
    CHECK(shape(b[5]) == "plain:quoted |bold:text");
    CHECK(shape(b[6]) == "plain:a   b");
    CHECK(shape(b[7]) == "plain:1   2");
    REQUIRE(b[8].kind == MdBlockKind::code);
    CHECK(b[8].lines == std::vector<std::string>{"x = 1  # keep  spaces", "", "print(x)"});
    CHECK(shape(b[9]) == "plain:after");

    // An unclosed fence: its line goes, and what follows is plain text.
    const auto open = chat_markdown_blocks("Here:\n```\nx = `1`\n**y**");
    REQUIRE(open.size() == 2);
    CHECK(shape(open[0]) == "plain:Here:");
    CHECK(shape(open[1]) == "plain:x = |code:1|plain: |bold:y");
    CHECK_FALSE(raw_marks(chat_markdown_plain("```js\nalert(1)")));
}

TEST_CASE("chat markdown: hostile text is bounded and falls back to plain text") {
    // Thousands of nested and unbalanced marks.
    // (in lines of 1,500 bytes, under the line bound, in one block under the inline bound)
    std::string nested;
    for (int line = 0; line < 5; ++line) {
        for (int i = 0; i < 300; ++i) nested += "**a *";
        nested += "\n";
    }
    const auto n = chat_markdown_blocks(nested);
    REQUIRE(n.size() == 1);
    CHECK(spans_text(n[0]).size() < nested.size() / 2);
    // Only the lone marks where the lines join, "a * **a" read as "2 * 3", stay:
    // the last mark of each of the five lines.
    const std::string kept = spans_text(n[0]);
    CHECK(std::count(kept.begin(), kept.end(), '*') == 5);
    std::string stars(7000, '*');
    CHECK(chat_markdown_blocks("x " + stars + " y").size() == 1);
    std::string ticks;
    for (int i = 0; i < 2000; ++i) ticks += std::string((size_t)(i % 7 + 1), '`') + "a";
    CHECK(chat_markdown_blocks(ticks).size() == 1);
    // A very long line shows as it is, in one block.
    const std::string line(200000 / 4, 'x');
    const auto l = chat_markdown_blocks("**" + line + "**");
    REQUIRE(l.size() == 1);
    CHECK(spans_text(l[0]).size() == line.size() + 4);
    // Many lines: past the block bound the text shows as it is, in plain blocks
    // of a bounded size, none of it lost.
    std::string many;
    for (int i = 0; i < 5000; ++i) many += "- item " + std::to_string(i) + "\n\n";
    const auto m = chat_markdown_blocks(many);
    REQUIRE(m.size() > MdLimits::max_blocks);
    for (size_t i = MdLimits::max_blocks; i < m.size(); ++i) {
        CHECK(m[i].kind == MdBlockKind::paragraph);
        CHECK(spans_text(m[i]).size() <= MdLimits::max_plain_block);
    }
    CHECK(spans_text(m.back()).find("- item 4999") != std::string::npos);
    // Text over the Markdown bound: the rest shows as it is, none of it lost.
    const std::string big((size_t)MdLimits::max_text + 1, '*');
    size_t bytes = 0;
    for (const MdBlock& b : chat_markdown_blocks(big)) bytes += spans_text(b).size();
    CHECK(bytes == big.size());
    // Fences that never close, a bracket that never closes, a tag that never closes.
    // (300 of them, under the block bound, past which the text shows as it is)
    std::string fences;
    for (int i = 0; i < 300; ++i) fences += "```a\n[x](\n<b\n";
    CHECK(chat_markdown_plain(fences).find("```") == std::string::npos);
}

namespace {

// The steps the parser and the layout take for `text`, laid out at 300 points.
size_t steps_of(const std::string& text) {
    chat_markdown_reset_steps();
    MdDocument doc;
    doc.set_text(text);
    doc.layout(300.0f, MdMetrics{}, fake);
    return chat_markdown_steps();
}

// `unit` repeated up to `bytes` bytes.
std::string repeat(const std::string& unit, size_t bytes) {
    std::string out;
    while (out.size() + unit.size() <= bytes) out += unit;
    return out;
}

// Lines of `line` bytes of `unit`, in paragraphs of `per` lines, up to `bytes`.
std::string paragraphs(const std::string& unit, size_t bytes, size_t line = 1900, size_t per = 4) {
    const std::string one = repeat(unit, line) + "\n";
    std::string out;
    size_t n = 0;
    while (out.size() + one.size() + 1 <= bytes) {
        out += one;
        if (++n % per == 0) out += "\n";
    }
    return out;
}

// Linear: twice the text takes at most a little more than twice the steps,
// and at most `per_byte` steps a byte.
void check_linear(const std::string& name, const std::function<std::string(size_t)>& make, size_t n,
                  double per_byte = 48.0) {
    const std::string a = make(n), b = make(2 * n);
    const size_t sa = steps_of(a), sb = steps_of(b);
    INFO(name << ": " << a.size() << " bytes take " << sa << " steps, " << b.size() << " bytes " << sb);
    CHECK((double)sb <= 2.3 * (double)sa + 4096.0);
    CHECK((double)sb <= per_byte * (double)b.size());
}

}  // namespace

// The parser and the layout count their steps (chat_markdown_steps()); a
// quadratic path shows as a count that grows faster than the text, on any
// machine, without timing anything.
TEST_CASE("chat markdown: hostile text takes steps linear in its size") {
    const size_t n = 24 * 1024;   // and 48 KiB, both under the Markdown bound
    check_linear("unclosed fences", [](size_t b) { return repeat("```a\n", b); }, n);
    check_linear("unclosed tilde and backtick fences", [](size_t b) { return repeat("~~~x\n```y\nz\n", b); }, n);
    check_linear("opening brackets", [](size_t b) { return paragraphs("[", b); }, n);
    check_linear("links without a target's end", [](size_t b) { return paragraphs("[x](", b); }, n);
    check_linear("angle brackets", [](size_t b) { return paragraphs("<a ", b); }, n);
    check_linear("tags without an end", [](size_t b) { return paragraphs("<b", b); }, n);
    check_linear("backtick runs of every length", [](size_t b) {
        std::string unit;
        for (int k = 1; k <= 9; ++k) unit += std::string((size_t)k, '`') + "a";
        return paragraphs(unit, b);
    }, n);
    check_linear("nested emphasis", [](size_t b) { return paragraphs("**a *", b); }, n);
    check_linear("one long word", [](size_t b) { return std::string(b, 'x'); }, n);
    check_linear("long lines", [](size_t b) { return repeat(std::string(5000, 'y') + "\n", b); }, n);
    // An item of the chat's largest size, 256 KiB: Markdown up to the bound,
    // plain text after it.
    check_linear("a 256 KiB item", [](size_t b) { return repeat("word **bold** `code` [l](t)\n", b); }, 128 * 1024);
    check_linear("a 256 KiB word", [](size_t b) { return std::string(b, 'z'); }, 128 * 1024);
}

TEST_CASE("chat markdown: a streamed text is parsed again only from its last finished block") {
    // The summary, many times, in paragraphs, as it streams in pieces of 37 bytes.
    const std::string summary = fixture();
    auto stream = [&](size_t bytes, size_t* last_append) {
        const std::string text = repeat(summary + "\n", bytes);
        chat_markdown_reset_steps();
        MdDocument doc;
        size_t before = 0;
        for (size_t at = 0; at < text.size();) {
            at = std::min(text.size(), at + 37);
            while (at < text.size() && ((unsigned char)text[at] & 0xC0) == 0x80) ++at;
            before = chat_markdown_steps();
            doc.set_text(text.substr(0, at));
            doc.layout(300.0f, MdMetrics{}, fake);
        }
        *last_append = chat_markdown_steps() - before;
        // The end result is that of the whole text read at once.
        CHECK(chat_markdown_plain(text) == [&] {
            std::string out;
            for (const MdBlock& b : doc.blocks()) {
                if (!out.empty()) out += '\n';
                if (b.kind == MdBlockKind::bullet) out += "\xE2\x80\xA2 ";
                for (const MdSpan& s : b.spans) out += s.text;
            }
            return out;
        }());
        return chat_markdown_steps();
    };
    size_t last_small = 0, last_big = 0;
    const size_t small = stream(16 * 1024, &last_small), big = stream(32 * 1024, &last_big);
    INFO("16 KiB streamed: " << small << " steps, 32 KiB: " << big);
    CHECK((double)big <= 2.3 * (double)small);
    CHECK((double)big <= 64.0 * 32 * 1024);
    // The last piece costs about the same whatever came before it.
    CHECK(last_big <= 2 * last_small + 2048);
    CHECK(last_big < 8000);
}

TEST_CASE("chat markdown: a fence that arrives later closes an earlier one, as if read at once") {
    MdDocument doc;
    const std::string text = "Before\n\n```py\nx = 1\n\ny = **2**\n";
    doc.set_text(text);
    // Unclosed so far: the fence's line goes, the lines after it are plain.
    REQUIRE(doc.blocks().size() == 3);
    CHECK(doc.blocks()[1].kind == MdBlockKind::paragraph);
    doc.set_text(text + "```\nAfter");
    REQUIRE(doc.blocks().size() == 3);
    CHECK(doc.blocks()[1].kind == MdBlockKind::code);
    CHECK(doc.blocks()[1].lines == std::vector<std::string>{"x = 1", "", "y = **2**"});
    CHECK(shape(doc.blocks()[2]) == "plain:After");
    // Any other change than an append reads the whole text again.
    doc.set_text("Other");
    REQUIRE(doc.blocks().size() == 1);
    CHECK(shape(doc.blocks()[0]) == "plain:Other");
}

TEST_CASE("chat markdown: a long word is broken from the width of each character, measured once") {
    // The bytes the layout asks the measure for: a word of n characters wider
    // than the line costs about 2n, not one measure of every prefix.
    size_t measured = 0;
    const MdMeasure counting = [&](const std::string& t, MdFont f) {
        measured += t.size();
        return fake(t, f);
    };
    for (const size_t n : {2000u, 8000u}) {
        measured = 0;
        layout_chat_markdown(chat_markdown_blocks(std::string(n, 'w')), 120.0f, MdMetrics{}, counting);
        INFO(n << " characters measured " << measured << " bytes");
        CHECK(measured <= 3 * n);
    }
}

TEST_CASE("chat markdown: a document lays out again only the blocks that changed") {
    size_t measured = 0;
    const MdMeasure counting = [&](const std::string& t, MdFont f) {
        measured += t.size();
        return fake(t, f);
    };
    const std::string summary = fixture();
    MdDocument doc;
    doc.set_text(repeat(summary + "\n", 16 * 1024));
    const MdLayout full = doc.layout(300.0f, MdMetrics{}, counting);
    const size_t first = measured;
    measured = 0;
    doc.set_text(doc.text() + "One more line.");
    const MdLayout& more = doc.layout(300.0f, MdMetrics{}, counting);
    CHECK(measured < first / 20);
    CHECK(more.height > full.height);
    // The same as a layout of the whole text at once.
    const MdLayout whole = layout_chat_markdown(chat_markdown_blocks(doc.text()), 300.0f, MdMetrics{}, fake);
    REQUIRE(whole.runs.size() == more.runs.size());
    CHECK(whole.height == more.height);
    for (size_t i = 0; i < whole.runs.size(); ++i) {
        CHECK(whole.runs[i].text == more.runs[i].text);
        CHECK(whole.runs[i].y == more.runs[i].y);
    }
    // Another width lays out every block again.
    measured = 0;
    doc.layout(200.0f, MdMetrics{}, counting);
    CHECK(measured >= first / 2);
}

TEST_CASE("chat markdown: the layout wraps words in the width, indents lists and plates code") {
    MdMetrics m;   // line 18, code line 17, gap 6, indent 14, marker 14, pad 6
    const auto blocks = chat_markdown_blocks(
        "one two three four five six seven\n\n- **bold** and `code`\n  - deeper\n\n```\nabcdefghij\n```");
    const MdLayout lay = layout_chat_markdown(blocks, 100.0f, m, fake);
    // No run passes the width, and none starts left of its block.
    for (const MdRun& r : lay.runs) {
        CHECK(r.x >= 0.0f);
        CHECK(r.x + r.w <= 100.0f + 0.01f);
    }
    // The paragraph: 33 bytes at 7 points do not fit 100 points; three lines.
    std::vector<float> ys;
    for (const MdRun& r : lay.runs)
        if (r.y < 54.0f && r.style == MdStyle::plain && !r.marker) ys.push_back(r.y);
    REQUIRE(!ys.empty());
    CHECK(ys.front() == 0.0f);
    CHECK(ys.back() == 36.0f);
    // The bullet: its marker at the left, its text after the marker's room; bold
    // a little wider for its second stroke; code in the mono measure.
    const MdRun* marker = nullptr;
    const MdRun* bold = nullptr;
    const MdRun* code = nullptr;
    const MdRun* deeper = nullptr;
    for (const MdRun& r : lay.runs) {
        if (r.marker && !marker) marker = &r;
        if (r.style == MdStyle::bold) bold = &r;
        if (r.style == MdStyle::code && r.text == "code") code = &r;
        if (r.text == "deeper") deeper = &r;
    }
    REQUIRE(marker);
    REQUIRE(bold);
    REQUIRE(code);
    REQUIRE(deeper);
    CHECK(marker->x == 0.0f);
    CHECK(bold->x == 14.0f);
    CHECK(bold->w == doctest::Approx(4 * 7.0f + 0.6f));
    CHECK(code->w == doctest::Approx(4 * 8.0f));
    CHECK(deeper->x == 28.0f);
    // The code block: one plate the width of the column, its line inside it.
    REQUIRE(lay.plates.size() == 1);
    CHECK(lay.plates[0].w == 100.0f);
    CHECK(lay.plates[0].h == doctest::Approx(6 + 17 + 6));
    CHECK(lay.height == doctest::Approx(lay.plates[0].y + lay.plates[0].h));
    // A word wider than the line is broken between characters.
    const MdLayout narrow = layout_chat_markdown(chat_markdown_blocks("abcdefghijklmnopqrstuvwxyz"), 50.0f, m, fake);
    CHECK(narrow.runs.size() == 4);   // 7 characters of 7 points fit 50
    for (const MdRun& r : narrow.runs) CHECK(r.w <= 50.0f);
    // A width too small for one character still places each, one a line.
    const MdLayout tiny = layout_chat_markdown(chat_markdown_blocks("abc"), 1.0f, m, fake);
    CHECK(tiny.runs.size() == 3);
}
