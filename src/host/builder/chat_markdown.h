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
// made clickable), autolinks, HTML tags, block quotes, table rows (the cells
// side by side) and rules (left out). Nothing here opens a link, loads a file
// or makes a request; the output is text and styles.
//
// HTML tags are only the common ones, so "<id>" stays text. Hostile or
// unfinished text stays bounded and falls back to plain text: an unclosed
// fence leaves its lines as paragraphs, an opening mark or backtick with no
// closing one is taken out (so text still streaming in, "**Spe", shows
// "Spe"), every mark of a run that formed a pair goes ("**Spec*" shows
// "Spec"), and the inline pass skips a block longer than
// MdLimits::max_inline or a line longer than max_line, which then shows as it
// is. The block count, the delimiter stack and the
// search for a closing mark are bounded, so the work is linear in the text.
//
// Two parts, both without ImGui so that they are tested headless:
//   - chat_markdown_blocks() parses a text into blocks of styled spans, and
//     chat_markdown_plain() gives the same as plain text.
//   - layout_chat_markdown() places the blocks in a width as runs of one
//     style each, from text widths the caller measures, word-wrapped, a word
//     wider than the line broken between characters.
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
    static constexpr std::size_t max_text = 64 * 1024;   // beyond it, the text shows as it is
    static constexpr std::size_t max_line = 2000;        // a longer line is a block of its own, as it is
    static constexpr std::size_t max_inline = 8000;      // a longer block shows its text as it is
    static constexpr std::size_t max_blocks = 400;       // the rest of the text is one block as it is
    static constexpr std::size_t max_search = 64;        // openers looked at for one closing mark
    static constexpr std::size_t max_code_lines = 1000;  // lines looked at for a fence's closing one
};

std::vector<MdBlock> chat_markdown_blocks(const std::string& text);

// The blocks as plain text: one line per block ("• " before a bullet, the
// label before a number, a code block's lines as they are).
std::string chat_markdown_plain(const std::string& text);

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

}  // namespace caliper_host
