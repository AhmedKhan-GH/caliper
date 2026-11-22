// The runs panel of Hosted Runs (runs_panel.h, design/hosted-runs.md).
#include "builder/runs_panel.h"

#include "spec_runtime/spec_runtime.h"

#include <caliper_builder/commands/command.h>
#include <caliper_builder/session/session.h>

#include <imgui.h>

#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <set>

namespace caliper_host {

namespace cb = caliper_builder;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// One word of a field: its text with the quotes taken away and their escapes
// read, the place in it of its first '=' outside quotes, whether quotes
// follow that '=', and the word as the person typed it.
struct Word {
    std::string text;
    std::size_t eq = std::string::npos;
    bool quoted = false;
    std::string typed;
};

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// The `n` hex digits at text[i], or -1.
long hex_at(const std::string& text, std::size_t i, std::size_t n) {
    if (i + n > text.size()) return -1;
    long v = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const int d = hex_digit(text[i + k]);
        if (d < 0) return -1;
        v = v * 16 + d;
    }
    return v;
}

void put_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Reads the escape at text[i], a backslash in quotes, as the card writes it
// (cb::run_card_text()): the escapes of JSON, with a surrogate pair as one
// character. Appends the character to *out and moves i past the escape; false
// with *problem for any other escape, also for the card's \xHH of a byte that
// is not UTF-8, which the session refuses before any card shows.
bool read_escape(const std::string& text, std::size_t& i, const std::string& field, std::string* out,
                 std::string* problem) {
    if (i + 1 >= text.size()) {
        *problem = "A quote in " + field + " is not closed.";
        return false;
    }
    const char e = text[i + 1];
    const char* simple = nullptr;
    switch (e) {
        case '"': simple = "\""; break;
        case '\\': simple = "\\"; break;
        case '/': simple = "/"; break;
        case 'b': simple = "\b"; break;
        case 'f': simple = "\f"; break;
        case 'n': simple = "\n"; break;
        case 'r': simple = "\r"; break;
        case 't': simple = "\t"; break;
        default: break;
    }
    if (simple) {
        *out += simple;
        i += 2;
        return true;
    }
    if (e == 'u') {
        const long first = hex_at(text, i + 2, 4);
        if (first < 0) {
            *problem = field + " has a \\u in quotes without four hex digits.";
            return false;
        }
        std::uint32_t cp = static_cast<std::uint32_t>(first);
        std::size_t len = 6;
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            const long second = text.compare(i + 6, 2, "\\u") == 0 ? hex_at(text, i + 8, 4) : -1;
            if (second >= 0xDC00 && second <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (static_cast<std::uint32_t>(second) - 0xDC00);
                len = 12;
            }
        }
        if (cp >= 0xD800 && cp <= 0xDFFF) {
            *problem = field + " has the escape " + text.substr(i, 6) + " in quotes without its other half.";
            return false;
        }
        put_utf8(*out, cp);
        i += len;
        return true;
    }
    const unsigned char lead = static_cast<unsigned char>(e);
    const std::size_t n = (lead & 0xE0) == 0xC0 ? 2 : (lead & 0xF0) == 0xE0 ? 3 : (lead & 0xF8) == 0xF0 ? 4 : 1;
    *problem = field + " has the escape \\" + text.substr(i + 1, n) +
               " in quotes, which is not known; write \\\\ for a backslash.";
    return false;
}

// Splits a field into words as the card writes its texts: at spaces (and at
// commas when `commas`), where double quotes keep spaces, commas and '='
// inside one word and read the card's escapes (read_escape()). Outside quotes
// every character is itself, a backslash too. False with *problem for an
// open quote or an escape that the card does not write.
bool words_of(const std::string& text, bool commas, const std::string& field, std::vector<Word>* out,
              std::string* problem) {
    out->clear();
    Word w;
    bool in_word = false, in_quote = false;
    std::size_t start = 0;
    for (std::size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (in_quote) {
            if (c == '"') {
                in_quote = false;
                ++i;
            } else if (c == '\\') {
                if (!read_escape(text, i, field, &w.text, problem)) return false;
            } else {
                w.text += c;
                ++i;
            }
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || (commas && c == ',')) {
            if (in_word) {
                w.typed = text.substr(start, i - start);
                out->push_back(std::move(w));
            }
            w = Word();
            in_word = false;
            ++i;
            continue;
        }
        if (!in_word) start = i;
        in_word = true;
        if (c == '"') {
            in_quote = true;
            if (w.eq != std::string::npos) w.quoted = true;
        } else {
            if (c == '=' && w.eq == std::string::npos) w.eq = w.text.size();
            w.text += c;
        }
        ++i;
    }
    if (in_quote) {
        *problem = "A quote in " + field + " is not closed.";
        return false;
    }
    if (in_word) {
        w.typed = text.substr(start);
        out->push_back(std::move(w));
    }
    return true;
}

json value_of(const std::string& v, bool quoted) {
    if (quoted) return v;
    if (v == "true") return true;
    if (v == "false") return false;
    const char* b = v.c_str();
    char* end = nullptr;
    const bool integral = !v.empty() && v.find_first_not_of("+-0123456789") == std::string::npos;
    if (integral) {
        errno = 0;
        const long long n = std::strtoll(b, &end, 10);
        if (end && *end == '\0' && errno == 0) return n;
    }
    end = nullptr;
    const double d = std::strtod(b, &end);
    if (!v.empty() && end && *end == '\0' && std::isfinite(d)) return d;
    return v;
}

// The base of a run id, as the model runner makes it from a script's name.
std::string id_base(const std::string& script) {
    std::string stem = fs::u8path(script).stem().u8string();
    std::string out;
    for (char c : stem) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
            out += c;
        else if (!out.empty() && out.back() != '-')
            out += '-';
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    if (out.size() > 32) out.resize(32);
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "run" : out;
}

bool finished(const std::string& state) { return state == "exited" || state == "failed"; }

const cb::RunEntry* find_run(const std::vector<cb::RunEntry>& runs, const std::string& id) {
    for (const auto& e : runs)
        if (e.id == id) return &e;
    return nullptr;
}

std::string upper(std::string s) {
    for (char& c : s)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    return s;
}

// The cells of a row: Id, State, Script, Parent, Step, Detail.
std::vector<std::string> cells_of(const cb::RunEntry& e) {
    std::string detail;
    if (e.error && !e.error->empty())
        detail = *e.error;
    else if (e.stopping && !finished(e.state))
        detail = "stopping";
    else if (e.exit_code)
        detail = "exit " + std::to_string(*e.exit_code);
    else if (e.attached)
        detail = "attached";
    return {e.id, upper(e.state), e.script, e.parent ? *e.parent : std::string(),
            e.step ? std::to_string(*e.step) : std::string(), detail};
}

const char* kColumnTitles[RunsPanelLayout::kColumns] = {"Id", "State", "Script", "Parent", "Step", "Detail"};
const float kColumnShares[RunsPanelLayout::kColumns] = {0.16f, 0.12f, 0.26f, 0.14f, 0.10f, 0.22f};

// The card's text in lines no wider than `width`, broken at spaces, and
// inside a word wider than a line after its last character that fits. The
// text is not read: every character shows, in order, quotes and escapes too,
// and the lines joined at their breaks give the text back.
std::vector<std::string> wrap(const std::string& text, float width, const RunsMeasure& measure) {
    std::vector<std::string> lines;
    std::string line;
    for (std::size_t i = 0; i <= text.size();) {
        std::size_t j = text.find(' ', i);
        if (j == std::string::npos) j = text.size();
        std::string word = text.substr(i, j - i);
        i = j + 1;
        while (measure(word) > width) {
            if (!line.empty()) lines.push_back(std::move(line));
            line.clear();
            std::size_t cut = 0;
            while (cut < word.size()) {
                std::size_t next = cut + 1;
                while (next < word.size() && (static_cast<unsigned char>(word[next]) & 0xC0) == 0x80) ++next;
                if (cut > 0 && measure(word.substr(0, next)) > width) break;
                cut = next;
            }
            lines.push_back(word.substr(0, cut));
            word.erase(0, cut);
        }
        const std::string next = line.empty() ? word : line + " " + word;
        if (!line.empty() && measure(next) > width) {
            lines.push_back(std::move(line));
            line = word;
        } else {
            line = next;
        }
    }
    if (!line.empty()) lines.push_back(line);
    return lines;
}

}  // namespace

// ---- the fields ---------------------------------------------------------------------------

bool parse_run_params(const std::string& text, json* pairs, std::string* problem) {
    *pairs = json::array();
    std::vector<Word> words;
    if (!words_of(text, true, "Parameters", &words, problem)) return false;
    std::set<std::string> seen;
    for (const auto& w : words) {
        if (w.eq == std::string::npos) {
            *problem = "Parameter " + w.typed + " has no value; write name=value.";
            return false;
        }
        const std::string name = w.text.substr(0, w.eq), value = w.text.substr(w.eq + 1);
        if (name.empty()) {
            *problem = "A parameter has no name.";
            return false;
        }
        if (name.size() > 63) {
            *problem = "A parameter name has at most 63 characters.";
            return false;
        }
        if (value.empty() && !w.quoted) {
            *problem = "Parameter " + name + " has no value.";
            return false;
        }
        if (!seen.insert(name).second) {
            *problem = "Parameter " + name + " is given twice.";
            return false;
        }
        if (pairs->size() >= cb::kRunMaxParams) {
            *problem = "A run takes at most " + std::to_string(cb::kRunMaxParams) + " parameters.";
            return false;
        }
        pairs->push_back({{"name", name}, {"value", value_of(value, w.quoted)}});
    }
    return true;
}

bool split_run_args(const std::string& text, std::vector<std::string>* args, std::string* problem) {
    args->clear();
    std::vector<Word> words;
    if (!words_of(text, false, "Arguments", &words, problem)) return false;
    if (words.size() > cb::kRunMaxArgs) {
        *problem = "A run takes at most " + std::to_string(cb::kRunMaxArgs) + " arguments.";
        return false;
    }
    for (auto& w : words) args->push_back(std::move(w.text));
    return true;
}

std::string suggest_run_id(const std::string& script, const std::vector<cb::RunEntry>& runs,
                           const std::string& workspace) {
    const std::string base = id_base(script);
    std::error_code ec;
    for (int n = 1; n < 100000; ++n) {
        const std::string c = base + "-" + std::to_string(n);
        if (find_run(runs, c)) continue;
        if (!workspace.empty() && fs::exists(fs::u8path(workspace) / "runs" / c, ec)) continue;
        return c;
    }
    return {};
}

std::vector<std::string> workspace_scripts(const std::string& workspace) {
    std::vector<std::string> out;
    if (workspace.empty()) return out;
    std::error_code ec;
    const fs::path root = fs::u8path(workspace);
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
    // A workspace may hold a large dataset: the walk looks at 20,000 entries at most.
    for (int seen = 0; !ec && it != end && seen < 20000 && out.size() < 200; it.increment(ec), ++seen) {
        const fs::path& p = it->path();
        const std::string name = p.filename().u8string();
        if (it->is_directory(ec)) {
            const bool skip = name.empty() || name[0] == '.' || name == "__pycache__" || name == "node_modules" ||
                              (it.depth() == 0 && name == "runs");
            if (skip || it.depth() >= 3) it.disable_recursion_pending();
            continue;
        }
        if (p.extension() != ".py" || !it->is_regular_file(ec)) continue;
        out.push_back(p.lexically_relative(root).generic_u8string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---- the layout ---------------------------------------------------------------------------

RunsPanelMetrics RunsPanelMetrics::for_text_scale(float s) {
    if (!(s > 0.0f)) s = 1.0f;
    RunsPanelMetrics m;
    auto r = [s](float v) { return std::round(v * s); };
    m.row = r(m.row);
    m.line = r(m.line);
    m.pad = r(m.pad);
    m.gap = r(m.gap);
    m.button_pad = r(m.button_pad);
    m.label = r(m.label);
    m.min_w = r(m.min_w);
    m.margin = r(m.margin);
    return m;
}

RunsPanelLayout layout_runs_panel(const RunsPanelLayoutInput& in, const RunsPanelMetrics& m,
                                  const RunsMeasure& measure) {
    RunsPanelLayout L;
    const RunsRect& A = in.area;
    auto button_w = [&](const std::string& t) { return std::round(measure(t) + 2.0f * m.button_pad); };

    // The width first: a share of the work area, at least the minimum, inside the margins.
    float w = std::max(m.min_w, std::round(A.w * m.share_w));
    w = std::max(0.0f, std::min(w, A.w - 2.0f * m.margin));
    const float iw = std::max(0.0f, w - 2.0f * m.pad);

    // The card's lines and the height of everything but the rows.
    if (!in.card.empty()) L.card_lines = wrap(in.card, std::max(1.0f, iw - 2.0f * m.gap), measure);
    const float top = m.pad + m.row + 2.0f * m.gap + m.row;   // title, rule, header
    float form = 2.0f * m.gap + m.row + m.gap;                // rule, form title
    if (!in.fork) form += (m.row + m.gap) * (in.scripts.empty() ? 2.0f : 3.0f);   // script, choices, arguments
    form += (m.row + m.gap) + m.row;                          // parameters, id
    if (!in.card.empty()) form += m.gap + m.gap + L.card_lines.size() * m.line + m.gap + m.row + m.gap;
    if (in.problem) form += m.gap + m.line;
    form += m.pad;
    const float max_h = std::max(0.0f, A.h - 2.0f * m.margin);
    int fit = in.runs;
    float rows_h = in.runs == 0 ? m.row : in.runs * m.row;
    if (top + rows_h + form > max_h && in.runs > 0) {
        fit = std::max(0, (int)std::floor((max_h - top - form - m.row) / m.row));
        fit = std::min(fit, in.runs);
        rows_h = fit * m.row + m.row;   // and the "+N" line
    }
    const float h = top + rows_h + form;
    float y0 = std::round(A.y + A.h * m.top_share);
    if (y0 + h > A.bottom() - m.margin) y0 = std::max(A.y + m.margin, A.bottom() - m.margin - h);
    L.window = {std::round(A.x + (A.w - w) * 0.5f), y0, w, h};

    const float x0 = L.window.x + m.pad, x1 = L.window.right() - m.pad;
    float y = L.window.y + m.pad;
    const float close_w = button_w("esc Close");
    L.close = {x1 - close_w, y, close_w, m.row};
    L.title = {x0, y, std::max(0.0f, iw - close_w - m.gap), m.row};
    y += m.row + 2.0f * m.gap;

    // The table: the text columns as shares, then Fork and Stop.
    const float fork_w = button_w("Fork"), stop_w = button_w("Stop");
    const float tw = std::max(0.0f, iw - fork_w - stop_w - 2.0f * m.gap);
    L.header = {x0, y, iw, m.row};
    float cx = x0;
    for (int i = 0; i < RunsPanelLayout::kColumns; ++i) {
        L.column_x[i] = std::round(cx);
        L.column_w[i] = std::round(tw * kColumnShares[i]);
        cx += tw * kColumnShares[i];
    }
    y += m.row;
    for (int i = 0; i < fit; ++i) {
        RunsPanelLayout::Row r;
        r.row = {x0, y, iw, m.row};
        r.stop = {x1 - stop_w, y + 2.0f, stop_w, m.row - 4.0f};
        r.fork = {r.stop.x - m.gap - fork_w, y + 2.0f, fork_w, m.row - 4.0f};
        L.rows.push_back(r);
        y += m.row;
    }
    L.rows_folded = in.runs - fit;
    if (in.runs == 0) {
        L.empty = {x0, y, iw, m.row};
        y += m.row;
    } else if (L.rows_folded > 0) {
        L.folded = {x0, y, iw, m.row};
        y += m.row;
    }

    // The form.
    y += 2.0f * m.gap;
    const float cancel_w = button_w("Cancel");
    L.form_title = {x0, y, iw, m.row};
    if (in.fork) L.cancel_fork = {x1 - cancel_w, y, cancel_w, m.row};
    y += m.row + m.gap;
    const float fx = x0 + m.label, fw = std::max(0.0f, iw - m.label);
    if (!in.fork) {
        L.script_label = {x0, y, m.label, m.row};
        L.script = {fx, y, fw, m.row};
        y += m.row + m.gap;
        if (!in.scripts.empty()) {
            const float more_w = measure("+999") + m.gap;
            float bx = fx;
            for (std::size_t i = 0; i < in.scripts.size(); ++i) {
                const float bw = button_w(in.scripts[i]);
                const bool last = i + 1 == in.scripts.size();
                if (bx + bw > x1 - (last ? 0.0f : more_w)) break;
                L.scripts.push_back({bx, y, bw, m.row});
                bx += bw + m.gap;
            }
            L.scripts_folded = (int)(in.scripts.size() - L.scripts.size());
            y += m.row + m.gap;
        }
        L.args_label = {x0, y, m.label, m.row};
        L.args = {fx, y, fw, m.row};
        y += m.row + m.gap;
    }
    L.params_label = {x0, y, m.label, m.row};
    L.params = {fx, y, fw, m.row};
    y += m.row + m.gap;
    const float go_w = std::max(button_w("Start"), button_w("Fork"));
    L.id_label = {x0, y, m.label, m.row};
    L.id = {fx, y, std::max(0.0f, fw - go_w - m.gap), m.row};
    L.go = {x1 - go_w, y, go_w, m.row};
    y += m.row;
    if (!in.card.empty()) {
        y += m.gap;
        const float ch = m.gap + L.card_lines.size() * m.line + m.gap + m.row + m.gap;
        L.card = {x0, y, iw, ch};
        const float by = y + m.gap + L.card_lines.size() * m.line + m.gap;
        const float approve_w = button_w("Approve");
        L.approve = {x0 + m.gap, by, approve_w, m.row};
        L.cancel = {L.approve.right() + m.gap, by, cancel_w, m.row};
        y += ch;
    }
    if (in.problem) {
        y += m.gap;
        L.problem = {x0, y, iw, m.line};
    }
    return L;
}

// ---- the state ----------------------------------------------------------------------------

void RunsPanel::set_open(bool open) {
    if (open && !open_) {
        focus_ = true;
        workspace_.clear();   // read the scripts again
        refresh();
    }
    open_ = open;
}

void RunsPanel::refresh() {
    runs_ = runner_ ? runner_->list() : std::vector<cb::RunEntry>();
    const std::string ws = session_ ? session_->workspace() : std::string();
    if (ws != workspace_) {
        workspace_ = ws;
        scripts_ = workspace_scripts(ws);
        if (script_.empty() && !scripts_.empty()) script_ = scripts_.front();
        if (!id_edited_) id_.clear();
    }
    // A suggested id that a run took since gives way to the next free one.
    const std::string base_script = fork_of_.empty() ? script_ : [&] {
        const cb::RunEntry* p = find_run(runs_, fork_of_);
        return p ? p->script : script_;
    }();
    if (!id_edited_ && !base_script.empty() && (id_.empty() || find_run(runs_, id_)))
        id_ = suggest_run_id(base_script, runs_, workspace_);
}

void RunsPanel::frame(cb::Session* session, cb::ModelRunner* runner, bool busy) {
    if (session != session_) forget_card();
    session_ = session;
    runner_ = runner;
    bool answered = false;
    if (pending_.valid() && pending_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const cb::CommandResult r = pending_.get();
        answered = true;
        forget_card();
        if (!r.ok) {
            problem_ = r.error.message.empty() ? r.error.code : r.error.message;
        } else {
            problem_.clear();
            if (pending_name_ == "run.start" || pending_name_ == "run.fork") {
                if (r.result.contains("run") && r.result["run"].contains("id") && r.result["run"]["id"].is_string())
                    last_run_ = r.result["run"]["id"].get<std::string>();
                fork_of_.clear();
                id_edited_ = false;
                id_.clear();
            }
        }
        pending_name_.clear();
    } else if (pending_.valid() && session_ && !busy) {
        read_card();
    }
    if (open_ || answered) refresh();
}

void RunsPanel::read_card() {
    // The session parks a start or a fork on its card for every origin
    // (design/hosted-runs.md, Review Card); the panel shows that card.
    card_.clear();
    card_approval_.clear();
    for (const cb::PendingApproval& p : session_->registry().pending_approvals()) {
        if (p.invocation.id != sent_id_ || p.approval == decided_approval_) continue;
        card_approval_ = p.approval;
        card_ = cb::run_card_text(p.invocation, card_script_);
        return;
    }
}

void RunsPanel::forget_card() {
    card_.clear();
    card_approval_.clear();
    decided_approval_.clear();
}

void RunsPanel::set_script(const std::string& script) {
    script_ = script;
    if (!id_edited_) id_ = script.empty() ? std::string() : suggest_run_id(script, runs_, workspace_);
}

void RunsPanel::set_id(const std::string& id) {
    id_ = id;
    id_edited_ = !id.empty();
}

void RunsPanel::begin_fork(const std::string& run) {
    const cb::RunEntry* e = find_run(runs_, run);
    if (!e || !can_fork(*e)) return;
    fork_of_ = run;
    params_ = run_params_text(e->params);
    id_edited_ = false;
    id_ = suggest_run_id(e->script, runs_, workspace_);
    problem_.clear();
}

void RunsPanel::cancel_fork() {
    fork_of_.clear();
    id_edited_ = false;
    id_ = script_.empty() ? std::string() : suggest_run_id(script_, runs_, workspace_);
}

bool RunsPanel::review() {
    problem_.clear();
    if (busy()) return false;
    if (!session_) {
        problem_ = "No workspace is open.";
        return false;
    }
    if (!runner_) {
        problem_ = "This app has no model runner.";
        return false;
    }
    json pairs;
    if (!parse_run_params(params_, &pairs, &problem_)) return false;
    if (!id_.empty() && !cb::valid_run_id(id_)) {
        problem_ = "A run id has 1 to 40 characters, each a-z, 0-9 or -.";
        return false;
    }
    if (fork_of_.empty()) {
        if (script_.empty()) {
            problem_ = "No script.";
            return false;
        }
        std::vector<std::string> args;
        if (!split_run_args(args_, &args, &problem_)) return false;
        json a{{"script", script_}};
        if (!args.empty()) a["args"] = args;
        if (!pairs.empty()) a["params"] = pairs;
        if (!id_.empty()) a["id"] = id_;
        card_script_.clear();
        return send("run.start", std::move(a));
    }
    const cb::RunEntry* parent = find_run(runs_, fork_of_);
    if (!parent || !can_fork(*parent)) {
        problem_ = "The run " + fork_of_ + " does not run.";
        return false;
    }
    json a{{"run", fork_of_}};
    if (!pairs.empty()) a["params"] = pairs;
    if (!id_.empty()) a["id"] = id_;
    card_script_ = parent->script;
    return send("run.fork", std::move(a));
}

bool RunsPanel::decide(bool approve) {
    if (!session_ || card_approval_.empty()) return false;
    // The decision runs at the queue's next drain; the card does not come
    // back while the approval waits for it.
    if (approve)
        session_->queue().approve(card_approval_);
    else
        session_->queue().deny(card_approval_);
    decided_approval_ = card_approval_;
    card_.clear();
    card_approval_.clear();
    return true;
}

bool RunsPanel::approve() { return decide(true); }

bool RunsPanel::cancel_review() { return decide(false); }

bool RunsPanel::stop(const std::string& run) {
    const cb::RunEntry* e = find_run(runs_, run);
    if (!e || !can_stop(*e) || busy()) return false;
    return send("run.stop", {{"run", run}});
}

bool RunsPanel::send(const char* name, json args) {
    if (!session_) {
        problem_ = "No workspace is open.";
        return false;
    }
    cb::CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = cb::CommandOrigin::ui;
    inv.id = "runs-panel-" + std::to_string(++sent_count_);   // finds the session's card of this call
    sent_id_ = inv.id;
    forget_card();
    pending_ = session_->queue().submit(std::move(inv));
    pending_name_ = name;
    problem_.clear();
    return true;
}

bool RunsPanel::can_fork(const cb::RunEntry& e) {
    return e.managed && !e.stopping && (e.state == "running" || e.state == "paused");
}

bool RunsPanel::can_stop(const cb::RunEntry& e) {
    return e.managed && !e.stopping && (e.state == "starting" || e.state == "running" || e.state == "paused");
}

std::vector<std::string> RunsPanel::visible_text() const {
    std::vector<std::string> out = {"Runs", "esc Close"};
    for (const char* t : kColumnTitles) out.push_back(t);
    if (runs_.empty()) out.push_back("No runs");
    for (const auto& e : runs_) {
        for (const auto& c : cells_of(e))
            if (!c.empty()) out.push_back(c);
        if (can_fork(e)) out.push_back("Fork");
        if (can_stop(e)) out.push_back("Stop");
    }
    if (fork_of_.empty()) {
        out.push_back("Start");
        for (const char* t : {"Script", "Arguments"}) out.push_back(t);
        out.push_back(script_);
        out.push_back(args_);
        for (const auto& s : scripts_) out.push_back(s);
    } else {
        out.push_back("Fork " + fork_of_);
        out.push_back("Cancel");
    }
    for (const char* t : {"Parameters", "Id"}) out.push_back(t);
    out.push_back(params_);
    out.push_back(id_);
    out.push_back(fork_of_.empty() ? "Start" : "Fork");
    if (!card_.empty()) {
        out.push_back(card_);
        out.push_back("Approve");
        out.push_back("Cancel");
    }
    if (!problem_.empty()) out.push_back(problem_);
    out.erase(std::remove(out.begin(), out.end(), std::string()), out.end());
    return out;
}

RunsPanelLayoutInput RunsPanel::layout_input(const RunsRect& area) const {
    RunsPanelLayoutInput in;
    in.area = area;
    in.runs = (int)runs_.size();
    in.fork = !fork_of_.empty();
    in.scripts = scripts_;
    in.card = card_;
    in.problem = !problem_.empty();
    return in;
}

// ---- the ImGui side -----------------------------------------------------------------------

namespace {

int resize_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* s = static_cast<std::string*>(data->UserData);
        s->resize((size_t)data->BufTextLen);
        data->Buf = &(*s)[0];
    }
    return 0;
}

ImVec2 extent(ImFont* font, float size, const std::string& t) {
    if (!font) font = ImGui::GetFont();
    return font->CalcTextSizeA(size, FLT_MAX, 0.0f, t.c_str());
}

void text_at(ImDrawList* dl, ImFont* font, float size, ImVec2 p, ImU32 col, const std::string& t) {
    if (!font) font = ImGui::GetFont();
    dl->AddText(font, size, ImVec2(std::round(p.x), std::round(p.y)), col, t.c_str());
}

// `t` cut to `w` with an ellipsis at its end.
std::string fit(const SpecLook& L, const std::string& t, float w) {
    if (extent(L.ui, L.ui_size, t).x <= w) return t;
    std::string s = t;
    while (!s.empty() && extent(L.ui, L.ui_size, s + "\xe2\x80\xa6").x > w) s.pop_back();
    while (!s.empty() && (static_cast<unsigned char>(s.back()) & 0xC0) == 0x80) s.pop_back();   // a whole character
    if (!s.empty() && static_cast<unsigned char>(s.back()) >= 0xC0) s.pop_back();
    return s + "\xe2\x80\xa6";
}

// A button in the house's look: a 1 px outline, bronze when lit.
bool house_button(const SpecLook& L, ImDrawList* dl, const RunsRect& r, const char* id, const std::string& label,
                  bool enabled = true, bool lit = false) {
    if (r.w < 1.0f) return false;
    ImGui::SetCursorScreenPos(ImVec2(r.x, r.y));
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(r.w, r.h)) && enabled;
    const bool hovered = enabled && ImGui::IsItemHovered();
    if (hovered) dl->AddRectFilled(ImVec2(r.x, r.y), ImVec2(r.right(), r.bottom()), L.plate_hover);
    dl->AddRect(ImVec2(r.x, r.y), ImVec2(r.right(), r.bottom()), lit ? L.accent : enabled ? L.rule : L.rule_faint, 0.0f,
                0, 1.0f);
    const ImVec2 e = extent(L.ui, L.ui_size, label);
    text_at(dl, L.ui, L.ui_size, ImVec2(r.x + (r.w - e.x) * 0.5f, r.y + (r.h - e.y) * 0.5f),
            !enabled ? L.faint : lit ? L.text : L.secondary, label);
    return clicked;
}

// A text field over `value`; true when it changed.
bool field(const SpecLook& L, const RunsRect& r, const char* id, std::string* value) {
    if (r.w < 1.0f) return false;
    ImGui::SetCursorScreenPos(ImVec2(r.x, r.y));
    ImGui::PushItemWidth(r.w);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::ColorConvertU32ToFloat4(L.deep));
    ImGui::PushStyleColor(ImGuiCol_Border, ImGui::ColorConvertU32ToFloat4(L.rule));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(std::round(8.0f * L.text_scale), std::max(0.0f, (r.h - L.ui_size) * 0.5f)));
    if (L.ui) ImGui::PushFont(L.ui, L.ui_size);
    if (value->capacity() < 64) value->reserve(64);
    const bool changed = ImGui::InputText(id, &(*value)[0], value->capacity() + 1, ImGuiInputTextFlags_CallbackResize,
                                          resize_callback, value);
    if (L.ui) ImGui::PopFont();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    ImGui::PopItemWidth();
    return changed;
}

}  // namespace

void RunsPanel::read_keys() {
    // ImGuiMod_Ctrl is the Command key on macOS, as the page's ⌘K reads it.
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_R)) set_open(!open_);
}

void RunsPanel::draw(float x, float y, float w, float h, const SpecLook& L) {
    if (!open_) return;
    const RunsPanelMetrics m = RunsPanelMetrics::for_text_scale(L.text_scale);
    const RunsMeasure measure = [&L](const std::string& t) { return extent(L.ui, L.ui_size, t).x; };
    const RunsPanelLayout lay = layout_runs_panel(layout_input({x, y, w, h}), m, measure);
    const RunsRect& W = lay.window;
    ImGui::SetNextWindowPos(ImVec2(W.x, W.y));
    ImGui::SetNextWindowSize(ImVec2(W.w, W.h));
    if (focus_) ImGui::SetNextWindowFocus();
    focus_ = false;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImGui::ColorConvertU32ToFloat4(L.ground));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoScrollWithMouse;
    const bool shown = ImGui::Begin("##runs.panel", nullptr, flags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    if (!shown) {
        ImGui::End();
        return;
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRect(ImVec2(W.x, W.y), ImVec2(W.right(), W.bottom()), L.rule, 0.0f, 0, 1.0f);
    const float ty = (m.row - extent(L.ui, L.ui_size, "Ag").y) * 0.5f;
    auto label = [&](const RunsRect& r, const std::string& t, ImU32 col) {
        if (r.w > 0.0f) text_at(dl, L.ui, L.ui_size, ImVec2(r.x, r.y + ty), col, t);
    };
    auto rule = [&](float ry) { dl->AddLine(ImVec2(W.x + m.pad, ry), ImVec2(W.right() - m.pad, ry), L.rule_faint); };
    const bool idle = !busy();

    // The title strip.
    label(lay.title, "Runs", L.text);
    if (house_button(L, dl, lay.close, "##runs.close", "esc Close")) set_open(false);
    rule(lay.title.bottom() + m.gap);

    // The table.
    for (int i = 0; i < RunsPanelLayout::kColumns; ++i)
        label({lay.column_x[i], lay.header.y, lay.column_w[i], m.row}, kColumnTitles[i], L.faint);
    std::string fork_click, stop_click;
    for (std::size_t r = 0; r < lay.rows.size() && r < runs_.size(); ++r) {
        const cb::RunEntry& e = runs_[r];
        const RunsPanelLayout::Row& row = lay.rows[r];
        dl->AddLine(ImVec2(row.row.x, row.row.y), ImVec2(row.row.right(), row.row.y), L.rule_faint);
        const std::vector<std::string> cells = cells_of(e);
        for (int i = 0; i < RunsPanelLayout::kColumns; ++i) {
            const ImU32 col = i == 0 || (i == 1 && (e.state == "running" || e.state == "failed")) ? L.text : L.secondary;
            label({lay.column_x[i], row.row.y, lay.column_w[i], m.row},
                  fit(L, cells[i], lay.column_w[i] - m.gap), i == 5 ? L.faint : col);
        }
        if (can_fork(e) && house_button(L, dl, row.fork, ("##runs.fork." + e.id).c_str(), "Fork", idle,
                                        fork_of_ == e.id))
            fork_click = e.id;
        if (can_stop(e) && house_button(L, dl, row.stop, ("##runs.stop." + e.id).c_str(), "Stop", idle))
            stop_click = e.id;
    }
    if (lay.empty.w > 0.0f) label(lay.empty, "No runs", L.faint);
    if (lay.folded.w > 0.0f) label(lay.folded, "+" + std::to_string(lay.rows_folded), L.faint);

    // The form.
    rule(lay.form_title.y - m.gap);
    label(lay.form_title, fork_of_.empty() ? std::string("Start") : "Fork " + fork_of_, L.text);
    if (!fork_of_.empty() && house_button(L, dl, lay.cancel_fork, "##runs.fork.cancel", "Cancel")) cancel_fork();
    if (fork_of_.empty()) {
        label(lay.script_label, "Script", L.faint);
        script_edit_ = script_;
        if (field(L, lay.script, "##runs.script", &script_edit_)) set_script(script_edit_);
        for (std::size_t i = 0; i < lay.scripts.size() && i < scripts_.size(); ++i)
            if (house_button(L, dl, lay.scripts[i], ("##runs.script." + std::to_string(i)).c_str(), scripts_[i], true,
                             scripts_[i] == script_))
                set_script(scripts_[i]);
        if (lay.scripts_folded > 0 && !lay.scripts.empty())
            text_at(dl, L.ui, L.ui_size, ImVec2(lay.scripts.back().right() + m.gap, lay.scripts.back().y + ty), L.faint,
                    "+" + std::to_string(lay.scripts_folded));
        label(lay.args_label, "Arguments", L.faint);
        args_edit_ = args_;
        if (field(L, lay.args, "##runs.args", &args_edit_)) set_args(args_edit_);
    }
    label(lay.params_label, "Parameters", L.faint);
    params_edit_ = params_;
    if (field(L, lay.params, "##runs.params", &params_edit_)) set_params(params_edit_);
    label(lay.id_label, "Id", L.faint);
    id_edit_ = id_;
    if (field(L, lay.id, "##runs.id", &id_edit_)) set_id(id_edit_);
    if (house_button(L, dl, lay.go, "##runs.go", fork_of_.empty() ? "Start" : "Fork", idle && session_ && runner_,
                     !card_.empty()))
        review();

    // The card.
    if (lay.card.w > 0.0f) {
        dl->AddRect(ImVec2(lay.card.x, lay.card.y), ImVec2(lay.card.right(), lay.card.bottom()), L.accent, 0.0f, 0,
                    1.0f);
        for (std::size_t i = 0; i < lay.card_lines.size(); ++i)
            text_at(dl, L.ui, L.ui_size, ImVec2(lay.card.x + m.gap, lay.card.y + m.gap + i * m.line), L.text,
                    lay.card_lines[i]);
        if (house_button(L, dl, lay.approve, "##runs.approve", "Approve", !card_approval_.empty(), true)) approve();
        if (house_button(L, dl, lay.cancel, "##runs.cancel", "Cancel")) cancel_review();
    }
    if (lay.problem.w > 0.0f)
        text_at(dl, L.ui, L.ui_size, ImVec2(lay.problem.x, lay.problem.y), L.text, fit(L, problem_, lay.problem.w));

    // The clicks of the rows act after the drawing, which reads runs_.
    if (!fork_click.empty()) begin_fork(fork_click);
    if (!stop_click.empty()) stop(stop_click);

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::IsKeyPressed(ImGuiKey_Escape, false) &&
        !ImGui::GetIO().WantTextInput)
        set_open(false);
    ImGui::End();
}

}  // namespace caliper_host
