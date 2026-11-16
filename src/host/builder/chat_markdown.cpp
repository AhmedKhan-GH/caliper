// chat_markdown.cpp: the side chat's small Markdown subset (chat_markdown.h).
#include "builder/chat_markdown.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace caliper_host {

namespace {

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool is_alnum(char c) { return std::isalnum((unsigned char)c) != 0; }
bool is_ascii_punct(char c) { return (unsigned char)c < 0x80 && std::ispunct((unsigned char)c) != 0; }

// The tags of HTML a model writes in prose; anything else in angle brackets,
// such as "<id>", stays as text.
bool is_html_tag(const std::string& inner) {
    static const char* const kTags[] = {"a",  "b",  "i",  "u",   "s",      "em",   "strong", "code", "pre",
                                        "br", "p",  "div", "span", "img",  "ul",   "ol",     "li",   "hr",
                                        "h1", "h2", "h3", "h4", "h5",     "h6",   "table",  "tr",   "td",
                                        "th", "sup", "sub", "del", "kbd", "details", "summary", "blockquote",
                                        "script", "style", "iframe", "font", "center", "small", "big", "mark"};
    size_t i = inner.size() && (inner[0] == '/' || inner[0] == '!') ? 1 : 0;
    size_t j = i;
    while (j < inner.size() && std::isalnum((unsigned char)inner[j])) ++j;
    if (j == i) return false;
    if (j < inner.size() && inner[j] != ' ' && inner[j] != '/') return false;
    std::string name = inner.substr(i, j - i);
    for (char& c : name) c = (char)std::tolower((unsigned char)c);
    for (const char* t : kTags)
        if (name == t) return true;
    return false;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) ++a;
    while (b > a && is_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

size_t leading_spaces(const std::string& s) {
    size_t n = 0;
    for (char c : s) {
        if (c == ' ') ++n;
        else if (c == '\t') n += 4;
        else break;
    }
    return n;
}

std::string strip_left(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

// The length of the UTF-8 sequence that starts with byte `c` (1 for a stray byte).
size_t utf8_len(unsigned char c) {
    if (c >= 0xF0 && c < 0xF8) return 4;
    if (c >= 0xE0) return c < 0xF0 ? 3 : 1;
    if (c >= 0xC0) return 2;
    return 1;
}

void push_span(std::vector<MdSpan>& spans, MdStyle style, const std::string& text) {
    if (text.empty()) return;
    if (!spans.empty() && spans.back().style == style) spans.back().text += text;
    else spans.push_back({style, text});
}

// ---- inline ---------------------------------------------------------------------------------

// A character of a text segment, after escapes, links, tags and code spans: a
// literal one is text, never a mark.
struct Ch {
    char c;
    bool literal;
};

struct Segment {
    bool code = false;
    std::vector<Ch> chars;   // a text segment
    std::string text;        // a code span
};

// Splits a block's text into code spans and text, taking out links, images,
// HTML tags and escapes on the way.
std::vector<Segment> segments(const std::string& s) {
    std::vector<Segment> out(1);
    auto text = [&]() -> std::vector<Ch>& {
        if (out.back().code) out.emplace_back();
        return out.back().chars;
    };
    auto emit = [&](const std::string& t, bool literal) {
        for (char c : t) text().push_back({c, literal});
    };
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const char c = s[i];
        if (c == '\\' && i + 1 < n && is_ascii_punct(s[i + 1])) {
            text().push_back({s[i + 1], true});
            i += 2;
            continue;
        }
        if (c == '`') {
            size_t k = 0;
            while (i + k < n && s[i + k] == '`') ++k;
            // The closing run of exactly k backticks.
            size_t j = i + k, close = std::string::npos;
            while (j < n) {
                if (s[j] != '`') {
                    ++j;
                    continue;
                }
                size_t m = 0;
                while (j + m < n && s[j + m] == '`') ++m;
                if (m == k) {
                    close = j;
                    break;
                }
                j += m;
            }
            if (close == std::string::npos) {
                i += k;   // an opening run with no closing one is taken out
                continue;
            }
            std::string inner = s.substr(i + k, close - i - k);
            for (char& x : inner)
                if (x == '\n') x = ' ';
            if (inner.size() >= 2 && inner.front() == ' ' && inner.back() == ' ' && trim(inner).size())
                inner = inner.substr(1, inner.size() - 2);
            Segment code;
            code.code = true;
            code.text = inner;
            out.push_back(std::move(code));
            i = close + k;
            continue;
        }
        if (c == '[' || (c == '!' && i + 1 < n && s[i + 1] == '[')) {
            // [text](target) and ![alt](target): the text only, the target dropped.
            const size_t open = c == '!' ? i + 1 : i;
            const size_t end_text = s.find(']', open + 1);
            if (end_text != std::string::npos && end_text - open <= 512 && end_text + 1 < n && s[end_text + 1] == '(') {
                const size_t end_target = s.find(')', end_text + 2);
                if (end_target != std::string::npos && end_target - end_text <= 2048 &&
                    s.substr(end_text + 2, end_target - end_text - 2).find('\n') == std::string::npos) {
                    emit(s.substr(open + 1, end_text - open - 1), c == '!');
                    i = end_target + 1;
                    continue;
                }
            }
            text().push_back({c, true});
            ++i;
            continue;
        }
        if (c == '<') {
            const size_t close = s.find('>', i + 1);
            if (close != std::string::npos && close - i <= 256) {
                const std::string inner = s.substr(i + 1, close - i - 1);
                const bool spaced = inner.find_first_of(" \t\n<") != std::string::npos;
                const size_t colon = inner.find(':');
                if (!spaced && colon != std::string::npos && colon > 0 && std::isalpha((unsigned char)inner[0])) {
                    emit(inner, true);   // an autolink: its address as text, never a link
                    i = close + 1;
                    continue;
                }
                if (is_html_tag(inner) && inner.find('<') == std::string::npos) {
                    i = close + 1;   // an HTML tag is taken out
                    continue;
                }
            }
            text().push_back({c, true});
            ++i;
            continue;
        }
        text().push_back({c, false});
        ++i;
    }
    return out;
}

// Resolves * and _ in a text segment: a pair of runs of two or more is bold,
// a pair of single marks is emphasis, shown plain; the marks of each pair and
// every opening run left without a closing one are taken out.
void emphasis(const std::vector<Ch>& chars, std::vector<MdSpan>& spans, bool at_end) {
    const size_t n = chars.size();
    struct Run {
        size_t pos, len;
        char c;
        bool open, close;
    };
    std::vector<Run> runs;
    for (size_t i = 0; i < n;) {
        const char c = chars[i].c;
        if ((c != '*' && c != '_') || chars[i].literal) {
            ++i;
            continue;
        }
        size_t k = 0;
        while (i + k < n && chars[i + k].c == c && !chars[i + k].literal) ++k;
        const char prev = i > 0 ? chars[i - 1].c : ' ';
        const char next = i + k < n ? chars[i + k].c : ' ';
        Run r{i, k, c, !is_space(next), !is_space(prev)};
        if (c == '_') {
            r.open = r.open && !is_alnum(prev);
            r.close = r.close && !is_alnum(next);
        }
        runs.push_back(r);
        i += k;
    }
    std::vector<char> removed(n, 0);
    std::vector<int> bold(n + 1, 0);   // +1 where a bold range starts, -1 where it ends
    std::vector<size_t> stack;         // the indexes of runs that may open
    for (size_t r = 0; r < runs.size(); ++r) {
        Run& closer = runs[r];
        bool matched = false;
        if (closer.close) {
            const size_t lowest = stack.size() > MdLimits::max_search ? stack.size() - MdLimits::max_search : 0;
            for (size_t k = stack.size(); k-- > lowest;) {
                Run& opener = runs[stack[k]];
                if (opener.c != closer.c) continue;
                const size_t used = std::min(opener.len, closer.len);
                // Every mark of both runs goes, also those the pair did not
                // use, such as the first * of "**Spec*" while it streams in.
                for (size_t q = 0; q < opener.len; ++q) removed[opener.pos + q] = 1;
                for (size_t q = 0; q < closer.len; ++q) removed[closer.pos + q] = 1;
                if (used >= 2) {
                    ++bold[opener.pos + opener.len];
                    --bold[closer.pos];
                }
                // The openers in between are left without a pair, and go too,
                // unless they stand inside a word ("a*b").
                for (size_t j = k + 1; j < stack.size(); ++j)
                    if (!runs[stack[j]].close)
                        for (size_t q = 0; q < runs[stack[j]].len; ++q) removed[runs[stack[j]].pos + q] = 1;
                stack.resize(k);
                matched = true;
                break;
            }
        }
        if (!matched && closer.open) stack.push_back(r);
    }
    // An opening run that found no closing one is taken out, so that no raw
    // mark shows, also while the text still streams in ("**Spe"), unless it
    // stands inside a word ("a*b"); so is a run that ends the whole text,
    // which may be the first half of a pair still coming.
    for (size_t k : stack)
        if (!runs[k].close)
            for (size_t q = 0; q < runs[k].len; ++q) removed[runs[k].pos + q] = 1;
    if (at_end && !runs.empty() && runs.back().pos + runs.back().len == n)
        for (size_t q = 0; q < runs.back().len; ++q) removed[runs.back().pos + q] = 1;
    int depth = 0;
    std::string run_text;
    bool run_bold = false;
    for (size_t i = 0; i < n; ++i) {
        depth += bold[i];
        if (removed[i]) continue;
        const bool b = depth > 0;
        if (b != run_bold && !run_text.empty()) {
            push_span(spans, run_bold ? MdStyle::bold : MdStyle::plain, run_text);
            run_text.clear();
        }
        run_bold = b;
        run_text += chars[i].c;
    }
    push_span(spans, run_bold ? MdStyle::bold : MdStyle::plain, run_text);
}

std::vector<MdSpan> inline_spans(const std::string& text, bool at_end) {
    std::vector<MdSpan> spans;
    if (text.size() > MdLimits::max_inline) {
        push_span(spans, MdStyle::plain, text);
        return spans;
    }
    for (const Segment& seg : segments(text)) {
        if (seg.code) push_span(spans, MdStyle::code, seg.text);
        else emphasis(seg.chars, spans, at_end);
    }
    return spans;
}

// ---- blocks ---------------------------------------------------------------------------------

bool is_fence(const std::string& line, char* fence_char, size_t* fence_len) {
    if (leading_spaces(line) > 3) return false;
    const std::string t = strip_left(line);
    if (t.size() < 3 || (t[0] != '`' && t[0] != '~')) return false;
    size_t k = 0;
    while (k < t.size() && t[k] == t[0]) ++k;
    if (k < 3) return false;
    if (t[0] == '`' && t.find('`', k) != std::string::npos) return false;
    *fence_char = t[0];
    *fence_len = k;
    return true;
}

// Without allocating: the search runs over many lines.
bool is_closing_fence(const std::string& line, char fence_char, size_t fence_len) {
    size_t i = 0;
    while (i < line.size() && i < 4 && line[i] == ' ') ++i;
    if (i > 3) return false;
    size_t k = 0;
    while (i + k < line.size() && line[i + k] == fence_char) ++k;
    if (k < fence_len) return false;
    for (size_t j = i + k; j < line.size(); ++j)
        if (!is_space(line[j])) return false;
    return true;
}

bool is_rule(const std::string& line) {
    const std::string t = trim(line);
    if (t.size() < 3) return false;
    const char c = t[0];
    if (c != '-' && c != '*' && c != '_') return false;
    size_t marks = 0;
    for (char x : t) {
        if (x == c) ++marks;
        else if (x != ' ') return false;
    }
    return marks >= 3;
}

// A heading's text, or false.
bool heading(const std::string& line, std::string* text) {
    if (leading_spaces(line) > 3) return false;
    const std::string t = strip_left(line);
    size_t k = 0;
    while (k < t.size() && t[k] == '#') ++k;
    if (k == 0 || k > 6 || (k < t.size() && t[k] != ' ' && t[k] != '\t')) return false;
    std::string rest = trim(t.substr(k));
    while (!rest.empty() && rest.back() == '#') rest.pop_back();
    *text = trim(rest);
    return true;
}

// A list item: its kind, nesting, label and text.
bool list_item(const std::string& line, MdBlockKind* kind, int* indent, std::string* label, std::string* text) {
    const size_t sp = leading_spaces(line);
    const std::string t = strip_left(line);
    if (t.size() >= 2 && (t[0] == '-' || t[0] == '*' || t[0] == '+') && (t[1] == ' ' || t[1] == '\t')) {
        *kind = MdBlockKind::bullet;
        *text = t.substr(2);
    } else if (t.size() == 1 && (t[0] == '-' || t[0] == '*' || t[0] == '+')) {
        *kind = MdBlockKind::bullet;
        text->clear();
    } else {
        size_t d = 0;
        while (d < t.size() && d < 9 && std::isdigit((unsigned char)t[d])) ++d;
        if (d == 0 || d >= t.size() || (t[d] != '.' && t[d] != ')') ||
            (d + 1 < t.size() && t[d + 1] != ' ' && t[d + 1] != '\t'))
            return false;
        *kind = MdBlockKind::number;
        *label = t.substr(0, d) + ".";
        *text = d + 2 <= t.size() ? t.substr(d + 2) : std::string();
    }
    *indent = (int)std::min<size_t>(3, sp / 2);
    return true;
}

// A table row's cells side by side, or false; a separator row gives "".
bool table_row(const std::string& line, std::string* text) {
    const std::string t = trim(line);
    if (t.size() < 2 || t[0] != '|') return false;
    std::vector<std::string> cells;
    std::string cell;
    for (size_t i = 1; i < t.size(); ++i) {
        if (t[i] == '\\' && i + 1 < t.size() && t[i + 1] == '|') {
            cell += '|';
            ++i;
        } else if (t[i] == '|') {
            cells.push_back(trim(cell));
            cell.clear();
        } else {
            cell += t[i];
        }
    }
    if (!trim(cell).empty()) cells.push_back(trim(cell));
    bool separator = !cells.empty();
    for (const std::string& c : cells)
        for (char x : c)
            if (x != '-' && x != ':' && x != ' ') separator = false;
    text->clear();
    if (separator) return true;
    for (const std::string& c : cells) {
        if (c.empty()) continue;
        if (!text->empty()) *text += "   ";
        *text += c;
    }
    return true;
}

MdBlock as_it_is(const std::string& text) {
    MdBlock b;
    push_span(b.spans, MdStyle::plain, text);
    return b;
}

}  // namespace

std::vector<MdBlock> chat_markdown_blocks(const std::string& source) {
    std::vector<MdBlock> blocks;
    if (source.size() > MdLimits::max_text) {
        blocks.push_back(as_it_is(source));
        return blocks;
    }
    const std::string& text = source;
    std::vector<std::string> lines;
    for (size_t at = 0;;) {
        const size_t nl = text.find('\n', at);
        std::string l = text.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        lines.push_back(std::move(l));
        if (nl == std::string::npos) break;
        at = nl + 1;
    }

    // The text of the block being gathered (a paragraph or a list item).
    std::string open_text;
    bool open = false;
    auto flush = [&](bool at_end = false) {
        if (!open) return;
        blocks.back().spans = inline_spans(trim(open_text), at_end);
        open_text.clear();
        open = false;
    };
    auto start = [&](MdBlock b, const std::string& t) {
        flush();
        blocks.push_back(std::move(b));
        open_text = t;
        open = true;
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string& line = lines[i];
        if (blocks.size() >= MdLimits::max_blocks) {
            flush();
            std::string rest;
            for (size_t j = i; j < lines.size(); ++j) rest += (j > i ? "\n" : "") + lines[j];
            blocks.push_back(as_it_is(rest));
            break;
        }
        if (line.size() > MdLimits::max_line) {
            flush();
            blocks.push_back(as_it_is(line));
            continue;
        }
        if (trim(line).empty()) {
            flush();
            continue;
        }
        char fence_char = 0;
        size_t fence_len = 0;
        if (is_fence(line, &fence_char, &fence_len)) {
            // Its closing fence, looked for over a bounded number of lines.
            const size_t last = std::min(lines.size(), i + 1 + MdLimits::max_code_lines);
            size_t close = i + 1;
            while (close < last && !is_closing_fence(lines[close], fence_char, fence_len)) ++close;
            if (close == last) close = lines.size();
            flush();
            if (close < lines.size()) {
                MdBlock code;
                code.kind = MdBlockKind::code;
                for (size_t j = i + 1; j < close; ++j) code.lines.push_back(lines[j].substr(0, MdLimits::max_line));
                blocks.push_back(std::move(code));
                i = close;
            }
            // An unclosed fence: its line goes, and the lines after it are plain.
            continue;
        }
        std::string body = line;
        // A block quote's marks go; what follows is read as any line.
        while (true) {
            const std::string t = strip_left(body);
            if (leading_spaces(body) > 3 || t.empty() || t[0] != '>') break;
            body = t.substr(1);
            if (!body.empty() && body[0] == ' ') body.erase(0, 1);
        }
        if (trim(body).empty()) {
            flush();
            continue;
        }
        std::string t;
        if (is_rule(body)) {
            flush();
            continue;
        }
        if (heading(body, &t)) {
            start(MdBlock{}, t);
            flush();
            continue;
        }
        if (table_row(body, &t)) {
            if (!t.empty()) {
                start(MdBlock{}, t);
                flush();
            }
            continue;
        }
        MdBlockKind kind;
        int indent = 0;
        std::string label;
        if (list_item(body, &kind, &indent, &label, &t)) {
            MdBlock b;
            b.kind = kind;
            b.indent = indent;
            b.label = label;
            start(std::move(b), t);
            continue;
        }
        // A line of the open paragraph or list item, or a new paragraph.
        if (open) open_text += " " + trim(body);
        else start(MdBlock{}, trim(body));
    }
    // The text's last block, unless the text ends a line: its last marks may
    // be the first half of a pair still streaming in.
    flush(!text.empty() && text.back() != '\n');
    return blocks;
}

std::string chat_markdown_plain(const std::string& text) {
    std::string out;
    for (const MdBlock& b : chat_markdown_blocks(text)) {
        if (!out.empty()) out += '\n';
        if (b.kind == MdBlockKind::code) {
            for (size_t i = 0; i < b.lines.size(); ++i) out += (i ? "\n" : "") + b.lines[i];
            continue;
        }
        out += std::string(2 * (size_t)b.indent, ' ');
        if (b.kind == MdBlockKind::bullet) out += "\xE2\x80\xA2 ";
        if (b.kind == MdBlockKind::number) out += b.label + " ";
        for (const MdSpan& s : b.spans) out += s.text;
    }
    return out;
}

// ---- the layout -----------------------------------------------------------------------------

namespace {

struct Flow {
    MdLayout& out;
    const MdMetrics& m;
    const MdMeasure& measure;
    float left, right, x, y;
    size_t line_start;   // the first run of the current line
    bool block = false;  // the lines of a code block

    float width_of(const std::string& t, MdStyle style) const {
        return measure(t, style == MdStyle::code ? MdFont::mono : MdFont::text) +
               (style == MdStyle::bold && !t.empty() ? m.bold_extra : 0.0f);
    }
    void newline() {
        y += m.line;
        x = left;
        line_start = out.runs.size();
    }
    void add(const std::string& t, MdStyle style, float w) {
        if (out.runs.size() > line_start) {
            MdRun& last = out.runs.back();
            if (!last.marker && last.style == style && std::abs(last.x + last.w - x) < 0.01f) {
                last.text += t;
                last.w += w;
                x += w;
                return;
            }
        }
        out.runs.push_back({x, y, w, style, t, false, block});
        x += w;
    }
    void word(const std::string& t, MdStyle style) {
        const float w = width_of(t, style);
        if (x + w > right && x > left) newline();
        if (w <= right - x) {
            add(t, style, w);
            return;
        }
        // Wider than the line: broken between characters, at least one a line.
        size_t at = 0;
        while (at < t.size()) {
            size_t end = at, fit = at;
            float fw = 0.0f;
            while (end < t.size()) {
                const size_t next = std::min(t.size(), end + utf8_len((unsigned char)t[end]));
                const float nw = width_of(t.substr(at, next - at), style);
                if (nw > right - x && end > at) break;
                end = next;
                fit = end;
                fw = nw;
            }
            add(t.substr(at, fit - at), style, fw);
            at = fit;
            if (at < t.size()) newline();
        }
    }
    void spans(const std::vector<MdSpan>& list) {
        for (const MdSpan& s : list) {
            size_t i = 0;
            const std::string& t = s.text;
            while (i < t.size()) {
                if (t[i] == '\n') {
                    newline();
                    ++i;
                    continue;
                }
                if (t[i] == ' ' || t[i] == '\t') {
                    size_t j = i;
                    while (j < t.size() && (t[j] == ' ' || t[j] == '\t')) ++j;
                    if (x > left) {
                        const float w = width_of(" ", s.style);
                        if (x + w <= right) add(" ", s.style, w);
                    }
                    i = j;
                    continue;
                }
                size_t j = i;
                while (j < t.size() && t[j] != ' ' && t[j] != '\t' && t[j] != '\n') ++j;
                word(t.substr(i, j - i), s.style);
                i = j;
            }
        }
    }
};

}  // namespace

MdLayout layout_chat_markdown(const std::vector<MdBlock>& blocks, float width, const MdMetrics& m,
                              const MdMeasure& measure) {
    MdLayout out;
    width = std::max(width, 1.0f);
    float y = 0.0f;
    bool first = true;
    for (const MdBlock& b : blocks) {
        if (!first) y += m.gap;
        first = false;
        if (b.kind == MdBlockKind::code) {
            const float pad = m.code_pad;
            MdMetrics cm = m;
            cm.line = m.code_line;
            const float top = y + pad;
            Flow c{out, cm, measure, pad, std::max(pad + 1.0f, width - pad), pad, top, out.runs.size(), true};
            for (size_t i = 0; i < b.lines.size(); ++i) {
                if (i) c.newline();
                if (!b.lines[i].empty()) c.word(b.lines[i], MdStyle::code);   // keeps its spaces, broken where it must be
            }
            const float h = 2.0f * pad + (c.y - top) + m.code_line;
            out.plates.push_back({0.0f, y, width, h});
            y += h;
            continue;
        }
        float left = 0.0f;
        if (b.kind == MdBlockKind::bullet || b.kind == MdBlockKind::number) {
            const float xi = (float)b.indent * m.indent;
            const std::string mark = b.kind == MdBlockKind::bullet ? std::string("\xE2\x80\xA2") : b.label;
            const float mw = measure(mark, MdFont::text);
            out.runs.push_back({xi, y, mw, MdStyle::plain, mark, true, false});
            left = std::min(width - 1.0f, xi + std::max(m.marker, mw + measure(" ", MdFont::text)));
        }
        Flow f{out, m, measure, left, width, left, y, out.runs.size()};
        f.spans(b.spans);
        y = f.y + m.line;
    }
    out.height = y;
    return out;
}

}  // namespace caliper_host
