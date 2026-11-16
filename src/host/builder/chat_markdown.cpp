// chat_markdown.cpp: the side chat's small Markdown subset (chat_markdown.h).
#include "builder/chat_markdown.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace caliper_host {

namespace {

thread_local std::size_t g_steps = 0;
inline void step(std::size_t n = 1) { g_steps += n; }

constexpr std::size_t kNone = ~std::size_t(0);

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool is_alnum(char c) { return std::isalnum((unsigned char)c) != 0; }
bool is_ascii_punct(char c) { return (unsigned char)c < 0x80 && std::ispunct((unsigned char)c) != 0; }

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

// The tags of HTML a model writes in prose; anything else in angle brackets,
// such as "<id>", stays as text. Only the name is read, at most 16 bytes.
bool is_html_tag(const std::string& s, size_t from, size_t to) {
    static const char* const kTags[] = {"a",  "b",  "i",  "u",   "s",      "em",   "strong", "code", "pre",
                                        "br", "p",  "div", "span", "img",  "ul",   "ol",     "li",   "hr",
                                        "h1", "h2", "h3", "h4", "h5",     "h6",   "table",  "tr",   "td",
                                        "th", "sup", "sub", "del", "kbd", "details", "summary", "blockquote",
                                        "script", "style", "iframe", "font", "center", "small", "big", "mark"};
    size_t i = from < to && (s[from] == '/' || s[from] == '!') ? from + 1 : from;
    size_t j = i;
    while (j < to && j - i < 16 && std::isalnum((unsigned char)s[j])) ++j;
    if (j == i || (j < to && s[j] != ' ' && s[j] != '/')) return false;
    std::string name = s.substr(i, j - i);
    for (char& c : name) c = (char)std::tolower((unsigned char)c);
    for (const char* t : kTags)
        if (name == t) return true;
    return false;
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

// For each position, the next position at or after it that holds one of
// `chars`, or n: one backward pass.
std::vector<uint32_t> next_of(const std::string& s, const char* chars) {
    const size_t n = s.size();
    std::vector<uint32_t> out(n + 1, (uint32_t)n);
    for (size_t i = n; i-- > 0;) {
        out[i] = std::strchr(chars, s[i]) && s[i] ? (uint32_t)i : out[i + 1];
    }
    step(n);
    return out;
}

// Splits a block's text into code spans and text, taking out links, images,
// HTML tags and escapes on the way. Every search is read from a table built
// in one pass, so the work is linear in the text.
std::vector<Segment> segments(const std::string& s) {
    const size_t n = s.size();
    const std::vector<uint32_t> bracket = next_of(s, "]"), paren = next_of(s, ")"), gt = next_of(s, ">"),
                                space = next_of(s, " \t\n"), colon = next_of(s, ":"), lt = next_of(s, "<");
    // The backtick runs, and for each the next run of the same length.
    struct Tick {
        size_t pos, len, next_same;
    };
    std::vector<Tick> ticks;
    for (size_t i = 0; i < n;) {
        if (s[i] != '`') {
            ++i;
            continue;
        }
        size_t k = 0;
        while (i + k < n && s[i + k] == '`') ++k;
        ticks.push_back({i, k, kNone});
        i += k;
    }
    {
        std::unordered_map<size_t, size_t> last;
        for (size_t r = ticks.size(); r-- > 0;) {
            const auto it = last.find(ticks[r].len);
            if (it != last.end()) ticks[r].next_same = it->second;
            last[ticks[r].len] = r;
        }
        step(ticks.size());
    }
    std::vector<Segment> out(1);
    auto text = [&]() -> std::vector<Ch>& {
        if (out.back().code) out.emplace_back();
        return out.back().chars;
    };
    size_t ti = 0;   // the first backtick run at or after i
    auto skip_ticks_to = [&](size_t pos) {
        while (ti < ticks.size() && ticks[ti].pos < pos) ++ti;
    };
    size_t i = 0;
    while (i < n) {
        step();
        const char c = s[i];
        if (c == '\\' && i + 1 < n && is_ascii_punct(s[i + 1])) {
            text().push_back({s[i + 1], true});
            i += 2;
            skip_ticks_to(i);
            continue;
        }
        if (c == '`') {
            skip_ticks_to(i);
            if (ti >= ticks.size() || ticks[ti].pos != i) {
                ++i;   // a backtick of a run that an escape broke: taken out
                continue;
            }
            const Tick t = ticks[ti];
            if (t.next_same == kNone) {
                i += t.len;   // an opening run with no closing one is taken out
                ++ti;
                continue;
            }
            const Tick& close = ticks[t.next_same];
            std::string inner = s.substr(i + t.len, close.pos - i - t.len);
            step(inner.size());
            for (char& x : inner)
                if (x == '\n') x = ' ';
            if (inner.size() >= 2 && inner.front() == ' ' && inner.back() == ' ' && !trim(inner).empty())
                inner = inner.substr(1, inner.size() - 2);
            Segment code;
            code.code = true;
            code.text = std::move(inner);
            out.push_back(std::move(code));
            i = close.pos + close.len;
            ti = t.next_same + 1;
            continue;
        }
        if (c == '[' || (c == '!' && i + 1 < n && s[i + 1] == '[')) {
            // [text](target) and ![alt](target): the text only, the target dropped.
            const size_t open = c == '!' ? i + 1 : i;
            const size_t end_text = bracket[open + 1];
            if (end_text < n && end_text - open <= MdLimits::max_link_text && end_text + 1 < n &&
                s[end_text + 1] == '(') {
                const size_t end_target = paren[end_text + 2];
                if (end_target < n && end_target - end_text <= MdLimits::max_link_target &&
                    space[end_text + 2] >= end_target) {
                    const bool literal = c == '!';
                    for (size_t q = open + 1; q < end_text; ++q) text().push_back({s[q], literal});
                    step(end_target - i);
                    i = end_target + 1;
                    skip_ticks_to(i);
                    continue;
                }
            }
            text().push_back({c, true});
            ++i;
            continue;
        }
        if (c == '<') {
            const size_t close = gt[i + 1];
            if (close < n && close - i <= MdLimits::max_tag && lt[i + 1] > close) {
                const bool spaced = space[i + 1] < close;
                if (!spaced && colon[i + 1] < close && colon[i + 1] > i + 1 && std::isalpha((unsigned char)s[i + 1])) {
                    for (size_t q = i + 1; q < close; ++q) text().push_back({s[q], true});   // an autolink: its address as text
                    step(close - i);
                    i = close + 1;
                    skip_ticks_to(i);
                    continue;
                }
                if (is_html_tag(s, i + 1, close)) {
                    i = close + 1;   // an HTML tag is taken out
                    skip_ticks_to(i);
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
    step(n);
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
    auto remove_run = [&](const Run& r) {
        for (size_t q = 0; q < r.len; ++q) removed[r.pos + q] = 1;
    };
    for (size_t r = 0; r < runs.size(); ++r) {
        Run& closer = runs[r];
        bool matched = false;
        if (closer.close) {
            const size_t lowest = stack.size() > MdLimits::max_search ? stack.size() - MdLimits::max_search : 0;
            for (size_t k = stack.size(); k-- > lowest;) {
                step();
                Run& opener = runs[stack[k]];
                if (opener.c != closer.c) continue;
                // Every mark of both runs goes, also those the pair did not
                // use, such as the first * of "**Spec*" while it streams in.
                remove_run(opener);
                remove_run(closer);
                if (std::min(opener.len, closer.len) >= 2) {
                    ++bold[opener.pos + opener.len];
                    --bold[closer.pos];
                }
                // The openers in between are left without a pair, and go too,
                // unless they stand inside a word ("a*b").
                for (size_t j = k + 1; j < stack.size(); ++j)
                    if (!runs[stack[j]].close) remove_run(runs[stack[j]]);
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
        if (!runs[k].close) remove_run(runs[k]);
    if (at_end && !runs.empty() && runs.back().pos + runs.back().len == n) remove_run(runs.back());
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
        step(text.size());
        return spans;
    }
    for (const Segment& seg : segments(text)) {
        if (seg.code) push_span(spans, MdStyle::code, seg.text);
        else emphasis(seg.chars, spans, at_end);
    }
    return spans;
}

// ---- lines ----------------------------------------------------------------------------------

// A fence line: its mark and its length, the mark run alone or with an info
// string (an opening fence), or the run alone (`bare`, which may close one).
bool fence(const std::string& line, char* mark, size_t* len, bool* bare) {
    size_t i = 0;
    while (i < line.size() && i < 4 && line[i] == ' ') ++i;
    if (i > 3 || i >= line.size() || (line[i] != '`' && line[i] != '~')) return false;
    const char c = line[i];
    size_t k = 0;
    while (i + k < line.size() && line[i + k] == c) ++k;
    if (k < 3) return false;
    bool only_space = true;
    for (size_t j = i + k; j < line.size(); ++j) {
        if (c == '`' && line[j] == '`') return false;   // a backtick fence's info has no backtick
        if (!is_space(line[j])) only_space = false;
    }
    *mark = c;
    *len = k;
    *bare = only_space;
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

// The last newline in [from, to), looked for backwards over at most `span` bytes.
size_t newline_before(const std::string& text, size_t from, size_t to, size_t span) {
    const size_t low = to > from + span ? to - span : from;
    for (size_t i = to; i-- > low;) {
        step();
        if (text[i] == '\n') return i;
    }
    return std::string::npos;
}

// Where Markdown ends in a text: past max_text, at the line start before it
// when one is near.
size_t markdown_end(const std::string& text) {
    if (text.size() <= MdLimits::max_text) return text.size();
    const size_t nl = newline_before(text, 0, MdLimits::max_text, MdLimits::max_line);
    return nl == std::string::npos ? MdLimits::max_text : nl + 1;
}

}  // namespace

std::size_t chat_markdown_steps() { return g_steps; }
void chat_markdown_reset_steps() { g_steps = 0; }

// ---- the parse ------------------------------------------------------------------------------

void MdDocument::parse_from(std::size_t offset, std::size_t base) {
    const std::string& text = text_;
    blocks_.resize(base);
    stable_offset_ = offset;
    stable_blocks_ = base;
    const size_t md_end = markdown_end(text);

    // The lines from `offset`, as [start, end) with whether a newline ends them.
    struct Line {
        size_t start, end;
        bool nl;
    };
    std::vector<Line> lines;
    for (size_t at = offset; at < text.size() || (at == offset && at == text.size());) {
        const size_t nl = text.find('\n', at);
        const size_t end = nl == std::string::npos ? text.size() : nl;
        lines.push_back({at, end, nl != std::string::npos});
        step(end - at + 1);
        if (nl == std::string::npos) break;
        at = nl + 1;
        if (at == text.size()) break;
    }
    // For each line, the longest bare fence of each mark after it, built
    // backwards once: whether a closing fence exists is read, never searched.
    std::vector<uint32_t> after_tick(lines.size() + 1, 0), after_tilde(lines.size() + 1, 0);
    for (size_t k = lines.size(); k-- > 0;) {
        after_tick[k] = after_tick[k + 1];
        after_tilde[k] = after_tilde[k + 1];
        if (lines[k].start >= md_end || lines[k].end - lines[k].start > MdLimits::max_line) continue;
        char mark = 0;
        size_t len = 0;
        bool bare = false;
        const std::string l = text.substr(lines[k].start, lines[k].end - lines[k].start);
        if (fence(l, &mark, &len, &bare) && bare) {
            uint32_t& m = mark == '`' ? after_tick[k] : after_tilde[k];
            m = std::max<uint32_t>(m, (uint32_t)len);
        }
        step();
    }

    std::string open_text;
    bool open = false;
    auto flush = [&](bool at_end = false) {
        if (!open) return;
        blocks_.back().spans = inline_spans(trim(open_text), at_end);
        open_text.clear();
        open = false;
    };
    auto start = [&](MdBlock b, const std::string& t) {
        flush();
        blocks_.push_back(std::move(b));
        open_text = t;
        open = true;
    };
    // A line start that an append cannot change: nothing open before it.
    auto safe = [&](size_t k) {
        if (open || k >= lines.size() || !lines[k].nl) return;
        stable_offset_ = lines[k].end + 1;
        stable_blocks_ = blocks_.size();
    };
    auto pending = [&](size_t at, char mark) {
        for (const Pending& p : pending_)
            if (p.fence == mark) return;   // the earliest of each kind is enough
        pending_.push_back({at, blocks_.size(), mark});
    };

    for (size_t k = 0; k < lines.size(); ++k) {
        const Line& ln = lines[k];
        if (ln.start >= md_end || blocks_.size() >= MdLimits::max_blocks) {
            // Plain text from here, in blocks of at most max_plain_block bytes,
            // cut at a line's end where one is near.
            flush();
            size_t at = ln.start;
            while (at < text.size()) {
                size_t end = std::min(text.size(), at + MdLimits::max_plain_block);
                if (end < text.size()) {
                    const size_t nl = newline_before(text, at, end, MdLimits::max_plain_block / 2);
                    if (nl != std::string::npos) end = nl + 1;
                }
                std::string chunk = text.substr(at, end - at);
                while (!chunk.empty() && (chunk.back() == '\n' || chunk.back() == '\r')) chunk.pop_back();
                step(end - at);
                if (end < text.size()) {
                    blocks_.push_back(as_it_is(chunk));
                    stable_offset_ = end;
                    stable_blocks_ = blocks_.size();
                } else {
                    blocks_.push_back(as_it_is(chunk));
                }
                at = end;
            }
            return;
        }
        std::string line = text.substr(ln.start, ln.end - ln.start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        step();
        if (line.size() > MdLimits::max_line) {
            flush();
            blocks_.push_back(as_it_is(line));
            safe(k);
            continue;
        }
        if (trim(line).empty()) {
            flush();
            safe(k);
            continue;
        }
        char mark = 0;
        size_t len = 0;
        bool bare = false;
        if (fence(line, &mark, &len, &bare)) {
            flush();
            const uint32_t further = (mark == '`' ? after_tick : after_tilde)[k + 1];
            if (further >= len) {
                // A closing fence exists further on: the lines up to it are code.
                MdBlock code;
                code.kind = MdBlockKind::code;
                size_t j = k + 1;
                for (; j < lines.size(); ++j) {
                    const std::string l = text.substr(lines[j].start, lines[j].end - lines[j].start);
                    char m2 = 0;
                    size_t len2 = 0;
                    bool bare2 = false;
                    if (fence(l, &m2, &len2, &bare2) && bare2 && m2 == mark && len2 >= len) break;
                    std::string cl = l.substr(0, MdLimits::max_line);
                    if (!cl.empty() && cl.back() == '\r') cl.pop_back();
                    code.lines.push_back(std::move(cl));
                    step();
                }
                blocks_.push_back(std::move(code));
                k = j;
                safe(k);
                continue;
            }
            // An unclosed fence: its line goes, and the lines after it are
            // plain. A fence of its kind arriving later may close it.
            pending(ln.start, mark);
            safe(k);
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
        if (trim(body).empty() || is_rule(body)) {
            flush();
            safe(k);
            continue;
        }
        std::string t;
        if (heading(body, &t) || (table_row(body, &t) && !t.empty())) {
            start(MdBlock{}, t);
            flush();
            safe(k);
            continue;
        }
        if (table_row(body, &t)) {   // a separator row
            safe(k);
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
        // A line of the open paragraph or list item, or a new paragraph; one
        // that has grown past max_inline is shown as it is and ends there.
        if (open) open_text += " " + trim(body);
        else start(MdBlock{}, trim(body));
        if (open_text.size() > MdLimits::max_inline) {
            flush();
            safe(k);
        }
    }
    // The text's last block, unless the text ends a line: its last marks may
    // be the first half of a pair still streaming in.
    flush(!text.empty() && text.back() != '\n');
}

void MdDocument::set_text(const std::string& text) {
    if (text.size() == text_.size() && text == text_) return;
    const bool append = text.size() > text_.size() && text.compare(0, text_.size(), text_) == 0;
    if (!append) {
        text_ = text;
        pending_.clear();
        parse_from(0, 0);
        laid_blocks_ = 0;
        return;
    }
    size_t offset = stable_offset_, base = stable_blocks_;
    text_ = text;
    // A fence of a pending fence's kind in what is read again may close it:
    // read again from that fence.
    for (const Pending& p : pending_) {
        if (p.offset >= offset) continue;
        bool closes = false;
        for (size_t at = offset; at < text_.size() && !closes;) {
            const size_t nl = text_.find('\n', at);
            const size_t end = nl == std::string::npos ? text_.size() : nl;
            char mark = 0;
            size_t len = 0;
            bool bare = false;
            step(end - at + 1);
            if (end - at <= MdLimits::max_line && fence(text_.substr(at, end - at), &mark, &len, &bare) && bare &&
                mark == p.fence)
                closes = true;
            step();
            if (nl == std::string::npos) break;
            at = nl + 1;
        }
        if (closes && p.offset < offset) {
            offset = p.offset;
            base = p.blocks;
        }
    }
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [&](const Pending& p) { return p.offset >= offset; }),
                   pending_.end());
    parse_from(offset, base);
    laid_blocks_ = std::min(laid_blocks_, base);
}

std::vector<MdBlock> chat_markdown_blocks(const std::string& text) {
    MdDocument doc;
    doc.set_text(text);
    return doc.blocks();
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
    float space[2] = {-1.0f, -1.0f};

    float width_of(const std::string& t, MdStyle style) const {
        step(t.size() + 1);
        return measure(t, style == MdStyle::code ? MdFont::mono : MdFont::text) +
               (style == MdStyle::bold && !t.empty() ? m.bold_extra : 0.0f);
    }
    float space_of(MdStyle style) {
        float& s = space[style == MdStyle::code ? 1 : 0];
        if (s < 0.0f) s = measure(" ", style == MdStyle::code ? MdFont::mono : MdFont::text);
        return s;
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
        // Wider than the line: broken between characters, from the width of
        // each character measured once, at least one a line.
        const float extra = style == MdStyle::bold ? m.bold_extra : 0.0f;
        const MdFont font = style == MdStyle::code ? MdFont::mono : MdFont::text;
        size_t from = 0, at = 0;
        float acc = 0.0f;
        while (at < t.size()) {
            const size_t len = std::min(t.size() - at, utf8_len((unsigned char)t[at]));
            const float cw = measure(t.substr(at, len), font);
            step();
            if (acc + cw + extra > right - x && at > from) {
                add(t.substr(from, at - from), style, acc + extra);
                newline();
                from = at;
                acc = 0.0f;
            }
            acc += cw;
            at += len;
        }
        if (at > from) add(t.substr(from, at - from), style, acc + extra);
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
                    step(j - i);
                    if (x > left) {
                        const float w = space_of(s.style);
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

// One block laid out with its top at y; returns its bottom.
float layout_block(const MdBlock& b, float width, const MdMetrics& m, const MdMeasure& measure, float y,
                   MdLayout& out) {
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
        return y + h;
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
    return f.y + m.line;
}

}  // namespace

MdLayout layout_chat_markdown(const std::vector<MdBlock>& blocks, float width, const MdMetrics& m,
                              const MdMeasure& measure) {
    MdLayout out;
    width = std::max(width, 1.0f);
    float y = 0.0f;
    for (size_t i = 0; i < blocks.size(); ++i) y = layout_block(blocks[i], width, m, measure, i ? y + m.gap : 0.0f, out);
    out.height = y;
    return out;
}

const MdLayout& MdDocument::layout(float width, const MdMetrics& m, const MdMeasure& measure) {
    width = std::max(width, 1.0f);
    if (width != width_ || !(m == metrics_)) {
        laid_blocks_ = 0;
        width_ = width;
        metrics_ = m;
    }
    laid_blocks_ = std::min({laid_blocks_, blocks_.size(), block_y_.size()});
    if (laid_blocks_ == blocks_.size() && block_y_.size() == blocks_.size() && !blocks_.empty()) return layout_;
    // Keep the runs and plates of the blocks laid out before; lay out the rest.
    const size_t keep = laid_blocks_;
    layout_.runs.resize(keep > 0 && keep < block_runs_.size() ? block_runs_[keep] : keep > 0 ? layout_.runs.size() : 0);
    layout_.plates.resize(keep > 0 && keep < block_plates_.size() ? block_plates_[keep]
                                                                  : keep > 0 ? layout_.plates.size() : 0);
    block_runs_.resize(keep);
    block_plates_.resize(keep);
    block_y_.resize(keep);
    float y = keep > 0 ? block_y_[keep - 1] : 0.0f;
    for (size_t i = keep; i < blocks_.size(); ++i) {
        const float top = i ? y + m.gap : 0.0f;
        block_runs_.push_back(layout_.runs.size());
        block_plates_.push_back(layout_.plates.size());
        y = layout_block(blocks_[i], width, m, measure, top, layout_);
        block_y_.push_back(y);
    }
    layout_.height = blocks_.empty() ? 0.0f : y;
    laid_blocks_ = blocks_.size();
    return layout_;
}

}  // namespace caliper_host
