#pragma once
// ============================================================================
// chat_markdown.h: the small subset of Markdown that the side chat renders in
// an agent's text (side_chat.h), as display only: the conversation keeps the
// text as the agent sent it.
//
// Rendered: paragraphs, **bold** (and __bold__), `inline code`, fenced code
// blocks, and bulleted and numbered lists, nested up to three levels. Shown
// as plain text with the markup taken out: headings, emphasis (*em*, _em_),
// links and images (their text; a link's target is dropped and nothing is
// made clickable), autolinks, the common HTML tags (so "<id>" stays text),
// block quotes, table rows (the cells side by side) and rules (left out).
// Nothing here opens a link, loads a file or makes a request; the output is
// text and styles.
//
// Unfinished or hostile text falls back to plain text: an unclosed fence
// leaves its lines as paragraphs, an opening mark or backtick with no closing
// one is taken out (so text still streaming in, "**Spe", shows "Spe"), and
// every mark of a run that formed a pair goes ("**Spec*" shows "Spec").
//
// What it costs. Parsing and layout are linear in the text: one pass over
// the lines, with whether a closing fence exists further on read from a table
// built backwards in the same pass, never by searching again; the closing
// bracket, parenthesis, angle bracket and backtick run of each position read
// from tables built in one pass over a block; the delimiter stack's search
// bounded by MdLimits::max_search; and a long word broken from the widths of
// its characters, each measured once. Only the first MdLimits::max_text
// bytes of a text are read as Markdown; the rest, and a text's blocks past
// MdLimits::max_blocks, are shown as plain text in blocks of at most
// MdLimits::max_plain_block bytes. A line longer than max_line, and a
// paragraph that grows past max_inline, show as plain text too. The tests
// count the steps (chat_markdown_steps()) rather than time them.
//
// MdDocument keeps a text's blocks and layout between frames: when the text
// grows by an append, as a streamed reply does, it parses again only from its
// last finished block (a blank line, a closed code block or a heading ends
// one), and lays out again only the blocks that changed; it parses again from
// an unclosed fence once a fence of its kind arrives that may close it.
//
// The parts are without ImGui, so that they are tested headless: the blocks,
// the plain text and a layout in a width from text widths the caller measures.
// ============================================================================
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace caliper_host {

enum class MdStyle { plain, bold, code };

struct MdSpan {
    MdStyle style = MdStyle::plain;
    std::string text;
};

enum class MdBlockKind { paragraph, bullet, number, code };

struct MdBlock {
    MdBlockKind kind = MdBlockKind::paragraph;
    int indent = 0;                  // bullet, number: the list's nesting, 0 to 3
    std::string label;               // number: "1." as the text gave it
    std::vector<MdSpan> spans;       // paragraph, bullet, number
    std::vector<std::string> lines;  // code: its lines
};

struct MdLimits {
    static constexpr std::size_t max_text = 64 * 1024;      // read as Markdown; the rest is plain text
    static constexpr std::size_t max_line = 2000;           // a longer line is a plain block of its own
    static constexpr std::size_t max_inline = 8000;         // a paragraph that grows past it shows as plain text
    static constexpr std::size_t max_blocks = 400;          // the text after them is plain text
    static constexpr std::size_t max_plain_block = 4096;    // a plain block past the limits, at most
    static constexpr std::size_t max_search = 64;           // openers looked at for one closing mark
    static constexpr std::size_t max_link_text = 512;       // a link's text, at most
    static constexpr std::size_t max_link_target = 2048;    // a link's target, at most
    static constexpr std::size_t max_tag = 256;             // an HTML tag or autolink, at most
};

std::vector<MdBlock> chat_markdown_blocks(const std::string& text);

// The blocks as plain text: one line per block ("• " before a bullet, the
// label before a number, a code block's lines as they are).
std::string chat_markdown_plain(const std::string& text);

// The work done by the parser and the layout in this thread since the last
// reset, in steps of a few operations: the tests check that it grows
// linearly with the text.
std::size_t chat_markdown_steps();
void chat_markdown_reset_steps();

// ---- the layout ---------------------------------------------------------------------------

enum class MdFont { text, mono };
using MdMeasure = std::function<float(const std::string& text, MdFont font)>;

struct MdMetrics {
    float line = 18.0f;       // a line of text (also of a line that holds code)
    float code_line = 17.0f;  // a line of a code block
    float gap = 6.0f;         // between two blocks
    float indent = 14.0f;     // one level of list nesting
    float marker = 14.0f;     // the room of a bullet before its text
    float code_pad = 6.0f;    // inside a code block's plate
    float bold_extra = 0.6f;  // what bold adds to a run's width (the app draws it twice, offset)
    bool operator==(const MdMetrics& o) const {
        return line == o.line && code_line == o.code_line && gap == o.gap && indent == o.indent &&
               marker == o.marker && code_pad == o.code_pad && bold_extra == o.bold_extra;
    }
};

struct MdRun {
    float x = 0, y = 0, w = 0;   // the top-left of its line and its width
    MdStyle style = MdStyle::plain;
    std::string text;
    bool marker = false;         // a list's bullet or number
    bool block = false;          // a line of a code block, on its plate
};

struct MdPlate {
    float x = 0, y = 0, w = 0, h = 0;   // a code block's background
};

struct MdLayout {
    std::vector<MdRun> runs;
    std::vector<MdPlate> plates;
    float height = 0.0f;
};

MdLayout layout_chat_markdown(const std::vector<MdBlock>& blocks, float width, const MdMetrics& m,
                              const MdMeasure& measure);

// ---- a text kept between frames -----------------------------------------------------------

class MdDocument {
public:
    // The text as it is now. An append is parsed from the last finished
    // block; any other change parses the whole text.
    void set_text(const std::string& text);
    const std::string& text() const { return text_; }
    const std::vector<MdBlock>& blocks() const { return blocks_; }
    // The layout in a width; the blocks laid out before at the same width and
    // metrics keep their runs.
    const MdLayout& layout(float width, const MdMetrics& m, const MdMeasure& measure);

    struct Pending {          // a fence with no closing one yet
        std::size_t offset;   // the start of its line
        std::size_t blocks;   // the blocks before it
        char fence;           // ` or ~
    };

private:
    void parse_from(std::size_t offset, std::size_t blocks);

    std::string text_;
    std::vector<MdBlock> blocks_;
    // The last point an append cannot change: a line start, with the blocks before it.
    std::size_t stable_offset_ = 0, stable_blocks_ = 0;
    std::vector<Pending> pending_;   // the earliest unclosed fence of each kind before the stable point
    // The layout: valid for the first laid_blocks_ blocks, at width_ and metrics_.
    MdLayout layout_;
    float width_ = -1.0f;
    MdMetrics metrics_;
    std::size_t laid_blocks_ = 0;
    std::vector<std::size_t> block_runs_, block_plates_;   // where each laid-out block's runs and plates start
    std::vector<float> block_y_;                           // the bottom of each laid-out block
};

}  // namespace caliper_host
