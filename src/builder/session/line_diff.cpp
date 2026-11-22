#include <caliper_builder/session/line_diff.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <vector>

namespace caliper_builder {

namespace {

constexpr std::size_t kMaxCells = std::size_t(1) << 22;

struct Lines {
    std::vector<std::string_view> lines;
    bool final_newline = true;  // the last line ends with a line feed (or the text is empty)
};

Lines split_lines(std::string_view text) {
    Lines out;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos) {
            out.lines.push_back(text.substr(start));
            out.final_newline = false;
            break;
        }
        out.lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

enum class Op : std::uint8_t { equal, remove, add };

struct Step {
    Op op;
    std::size_t a;  // the line of `before` (equal, remove)
    std::size_t b;  // the line of `after` (equal, add)
};

// The edit script of the middle part, a[a0, a1) against b[b0, b1); `same`
// compares line i of `before` with line j of `after`.
template <class Same>
void middle_script(std::size_t a0, std::size_t a1, std::size_t b0, std::size_t b1, const Same& same,
                   std::vector<Step>& out) {
    const std::size_t n = a1 - a0;
    const std::size_t m = b1 - b0;
    if (n == 0 || m == 0 || (n + 1) > kMaxCells / (m + 1)) {
        for (std::size_t i = a0; i < a1; ++i) out.push_back({Op::remove, i, b0});
        for (std::size_t j = b0; j < b1; ++j) out.push_back({Op::add, a1, j});
        return;
    }
    // lcs[i][j]: the length of the longest common subsequence of a[a0+i..a1)
    // and b[b0+j..b1). The bound keeps min(n, m) below 2^11, so 16 bits hold it.
    const std::size_t width = m + 1;
    std::vector<std::uint16_t> lcs((n + 1) * width, 0);
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = m; j-- > 0;) {
            std::uint16_t& cell = lcs[i * width + j];
            if (same(a0 + i, b0 + j)) cell = static_cast<std::uint16_t>(lcs[(i + 1) * width + j + 1] + 1);
            else cell = std::max(lcs[(i + 1) * width + j], lcs[i * width + j + 1]);
        }
    }
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < n && j < m) {
        if (same(a0 + i, b0 + j)) {
            out.push_back({Op::equal, a0 + i, b0 + j});
            ++i;
            ++j;
        } else if (lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) {
            out.push_back({Op::remove, a0 + i, b0 + j});
            ++i;
        } else {
            out.push_back({Op::add, a0 + i, b0 + j});
            ++j;
        }
    }
    for (; i < n; ++i) out.push_back({Op::remove, a0 + i, b0 + m});
    for (; j < m; ++j) out.push_back({Op::add, a0 + n, b0 + j});
}

std::string range_text(std::size_t start, std::size_t count) {
    // diff -u numbers lines from 1; an empty range names the line before it.
    const std::size_t first = count == 0 ? start : start + 1;
    if (count == 1) return std::to_string(first);
    return std::to_string(first) + "," + std::to_string(count);
}

std::string diff_impl(std::string_view before, std::string_view after, std::string_view before_label,
                      std::string_view after_label, std::size_t context) {
    if (before == after) return {};
    const Lines la = split_lines(before);
    const Lines lb = split_lines(after);
    const auto& a = la.lines;
    const auto& b = lb.lines;

    // The last lines differ when only one text ends with a line feed, even if
    // their characters match.
    auto same = [&](std::size_t i, std::size_t j) {
        if (a[i] != b[j]) return false;
        const bool a_last = i + 1 == a.size();
        const bool b_last = j + 1 == b.size();
        const bool a_nl = !a_last || la.final_newline;
        const bool b_nl = !b_last || lb.final_newline;
        return a_nl == b_nl;
    };
    std::size_t prefix = 0;
    while (prefix < a.size() && prefix < b.size() && same(prefix, prefix)) ++prefix;
    std::size_t suffix = 0;
    while (suffix < a.size() - prefix && suffix < b.size() - prefix &&
           same(a.size() - 1 - suffix, b.size() - 1 - suffix))
        ++suffix;

    std::vector<Step> steps;
    steps.reserve(a.size() + b.size());
    for (std::size_t i = 0; i < prefix; ++i) steps.push_back({Op::equal, i, i});
    middle_script(prefix, a.size() - suffix, prefix, b.size() - suffix, same, steps);
    for (std::size_t k = suffix; k > 0; --k) steps.push_back({Op::equal, a.size() - k, b.size() - k});

    std::string out;
    out += "--- ";
    out += before_label;
    out += "\n+++ ";
    out += after_label;
    out += "\n";
    auto line = [&](char mark, std::string_view text, bool last_without_newline) {
        out += mark;
        out += text;
        out += '\n';
        if (last_without_newline) out += "\\ No newline at end of file\n";
    };

    std::size_t k = 0;
    while (k < steps.size()) {
        if (steps[k].op == Op::equal) {
            ++k;
            continue;
        }
        // A hunk: from `context` lines before this change to `context` lines
        // after the last change that follows within twice the context.
        const std::size_t start = k > context ? k - context : 0;
        std::size_t last = k;
        std::size_t j = k;
        while (j < steps.size()) {
            if (steps[j].op != Op::equal) {
                last = j++;
                continue;
            }
            std::size_t run_end = j;
            while (run_end < steps.size() && steps[run_end].op == Op::equal) ++run_end;
            if (run_end == steps.size() || run_end - j > 2 * context) break;
            j = run_end;
        }
        const std::size_t end = std::min(steps.size(), last + 1 + context);
        std::size_t a_count = 0;
        std::size_t b_count = 0;
        for (std::size_t s = start; s < end; ++s) {
            if (steps[s].op != Op::add) ++a_count;
            if (steps[s].op != Op::remove) ++b_count;
        }
        out += "@@ -" + range_text(steps[start].a, a_count) + " +" + range_text(steps[start].b, b_count) + " @@\n";
        for (std::size_t s = start; s < end; ++s) {
            const Step& st = steps[s];
            const bool a_last = st.a + 1 == a.size() && !la.final_newline;
            const bool b_last = st.b + 1 == b.size() && !lb.final_newline;
            if (st.op == Op::equal) line(' ', a[st.a], a_last);
            else if (st.op == Op::remove) line('-', a[st.a], a_last);
            else line('+', b[st.b], b_last);
        }
        k = end;
    }
    return out;
}

}  // namespace

std::string unified_line_diff(std::string_view before, std::string_view after, std::string_view before_label,
                              std::string_view after_label, std::size_t context) {
    try {
        return diff_impl(before, after, before_label, after_label, std::min<std::size_t>(context, 1000));
    } catch (const std::exception&) {
        return "--- " + std::string(before_label) + "\n+++ " + std::string(after_label) +
               "\n@@ the diff could not be computed @@\n";
    }
}

}  // namespace caliper_builder
