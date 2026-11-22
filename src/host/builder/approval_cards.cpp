// approval_cards.cpp: the approval cards (approval_cards.h): the cards'
// logic over the session's pending approvals, their layout as plain
// rectangles, and the strip and the panel mark drawn with ImGui.
#include "builder/approval_cards.h"

#include <iostream>

#if CALIPER_WITH_BUILDER
#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"
#include "ui/work_area.h"   // JobsTrayPresence

#include <caliper_builder/commands/permissions.h>
#include <caliper_builder/commands/queue.h>
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <nlohmann/json.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#endif

namespace caliper_host {

#if CALIPER_WITH_BUILDER

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using nlohmann::json;

namespace {

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

bool bool_arg(const json& args, const char* key, bool fallback) {
    const auto it = args.find(key);
    return it != args.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

int steps_arg(const json& args) {
    const auto it = args.find("steps");
    return it != args.end() && it->is_number_integer() ? std::max(1, it->get<int>()) : 1;
}

std::string steps_text(int n) { return std::to_string(n) + (n == 1 ? " step" : " steps"); }

// The last part of a path, as a file's name.
std::string file_name(const std::string& path) {
    const size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? path : path.substr(cut + 1);
}

std::string panel_title(const spec::Spec* open, const std::string& id) {
    if (open)
        if (const spec::Panel* p = open->find_panel(id); p && p->title && !p->title->empty()) return *p->title;
    return id;
}

std::string upper(std::string s) {
    for (char& c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

}  // namespace

// ---- one card -----------------------------------------------------------------------------

std::string approval_who(const cb::CommandInvocation& inv) {
    switch (inv.origin) {
        case cb::CommandOrigin::mcp: return "MCP agent";
        case cb::CommandOrigin::agent: return inv.preview.empty() ? "Side chat" : "Widget agent";
        case cb::CommandOrigin::api: return "API";
        case cb::CommandOrigin::builder: return "Builder";
        case cb::CommandOrigin::ui: return "App";
        case cb::CommandOrigin::viewer: return "Viewer";
    }
    return "Agent";
}

std::string approval_what(const cb::CommandInvocation& inv, const spec::Spec* open) {
    const std::string& n = inv.name;
    const json& a = inv.args;
    std::string target;
    if (n == "panel.patch" || n == "panel.remove") {
        target = panel_title(open, str_arg(a, "id"));
    } else if (n == "panel.create") {
        target = str_arg(a, "title");
        if (target.empty()) target = str_arg(a, "id");
    } else if (n.rfind("source.", 0) == 0 || n.rfind("loader.", 0) == 0) {
        target = str_arg(a, "id");
    } else if (n == "layout.place") {
        target = panel_title(open, str_arg(a, "panel"));
    } else if (n == "layout.set") {
        target = "layout";
    } else if (n == "spec.set_text") {
        if (open) target = open->applet.title && !open->applet.title->empty() ? *open->applet.title : open->applet.id;
    } else if (n == "applet.save") {
        target = str_arg(a, "id");
        if (target.empty() && open) target = open->applet.id;
    } else if (n == "data.import") {
        const auto it = a.find("paths");
        if (it != a.end() && it->is_array() && !it->empty() && (*it)[0].is_string()) {
            target = file_name((*it)[0].get<std::string>());
            if (it->size() > 1) target += " +" + std::to_string(it->size() - 1);
        }
    } else if (n == "history.undo" || n == "history.redo") {
        target = steps_text(steps_arg(a));
    } else if (n == "run.set_param" || n == "run.action") {
        target = str_arg(a, "name");
    } else if (n == "run.pause") {
        target = str_arg(a, "op");
    } else if (n == "run.start") {
        target = str_arg(a, "script");
    } else if (n == "run.fork" || n == "run.stop") {
        target = str_arg(a, "run");
    }
    return target.empty() ? n : n + " \xC2\xB7 " + target;   // ·
}

std::string approval_run_text(const cb::CommandInvocation& inv, const std::string& script) {
    const std::string& n = inv.name;
    const json& a = inv.args;
    // The model runner's commands (design/hosted-runs.md, Review Card).
    if (n == "run.start" || n == "run.fork" || n == "run.stop") return cb::run_card_text(inv, script);
    const std::string in = script.empty() ? std::string("the attached script") : script;
    if (n == "run.set_param") {
        const auto v = a.find("value");
        const std::string value = v == a.end() ? std::string("?") : v->is_string() ? v->get<std::string>() : v->dump();
        return "Set parameter " + str_arg(a, "name") + " to " + value + " in " + in +
               ". The change applies at the next step.";
    }
    if (n == "run.action") return "Run action " + str_arg(a, "name") + " in " + in + " at the next step.";
    if (n == "run.pause") {
        const std::string op = str_arg(a, "op");
        if (op == "resume") return "Resume " + in + ".";
        if (op == "step") {
            const auto c = a.find("count");
            const long long count = c != a.end() && c->is_number_integer() ? c->get<long long>() : 1;
            return "Run " + std::to_string(count) + (count == 1 ? " step" : " steps") + " of " + in + ", then pause.";
        }
        return "Pause " + in + " at the next step.";
    }
    return "";
}

std::string approval_loss(const cb::CommandInvocation& inv, const spec::Spec* open) {
    const std::string& n = inv.name;
    const json& a = inv.args;
    std::string id = str_arg(a, "id");
    if (n == "applet.save" && id.empty() && open) id = open->applet.id;
    if (n == "loader.update")
        return "Loader " + id + (bool_arg(a, "run", false) ? " overwritten and run" : " overwritten");
    if (n == "loader.run") return "Tables of loader " + id + " replaced";
    if (n == "loader.delete")
        return "Loader " + id + (bool_arg(a, "keep_outputs", false) ? " deleted" : " and its tables deleted");
    if (n == "loader.create") return "Python script run";   // it asks only for a Python loader with run set
    if (n == "applet.save") return "Applet " + (id.empty() ? std::string("of the open spec") : id) + " overwritten";
    if (n == "history.undo") return steps_text(steps_arg(a)) + " undone";
    if (n == "history.redo") return steps_text(steps_arg(a)) + " redone";
    return {};
}

const char* approval_state_tag(ApprovalState s) {
    switch (s) {
        case ApprovalState::waiting: return "";
        case ApprovalState::deciding: return "DECIDING";
        case ApprovalState::approved: return "APPROVED";
        case ApprovalState::denied: return "DENIED";
        case ApprovalState::timed_out: return "TIMED OUT";
        case ApprovalState::conflict: return "CONFLICT";
        case ApprovalState::failed: return "FAILED";
        case ApprovalState::withdrawn: return "WITHDRAWN";
    }
    return "";
}

std::vector<WidgetDiffLine> approval_diff_from_unified(const std::string& diff, size_t max_value) {
    // "key = value" lines of TOML; a header or a line of an array is kept whole.
    auto split = [&](const std::string& body, std::string* key, std::string* value) {
        const size_t eq = body.find(" = ");
        if (eq == std::string::npos || body.empty() || body[0] == '[') {
            *key = body;
            value->clear();
            return;
        }
        *key = body.substr(0, eq);
        *value = body.substr(eq + 3);
        if (value->size() > max_value && max_value > 3) *value = value->substr(0, max_value - 1) + "\xE2\x80\xA6";
    };
    std::vector<WidgetDiffLine> out;
    std::map<std::string, size_t> removed;   // key to its line in `out`, waiting for its added line
    size_t start = 0;
    int line_no = 0;
    while (start < diff.size()) {
        size_t end = diff.find('\n', start);
        if (end == std::string::npos) end = diff.size();
        const std::string l = diff.substr(start, end - start);
        start = end + 1;
        if (++line_no <= 2 || l.empty() || l.rfind("@@", 0) == 0) continue;
        if (l[0] != '-' && l[0] != '+') continue;
        std::string key, value;
        split(l.substr(1), &key, &value);
        if (key.empty()) continue;
        if (l[0] == '-') {
            removed[key] = out.size();
            out.push_back({WidgetDiffLine::Kind::removed, key, value.empty() ? key : value, ""});
            continue;
        }
        const auto it = removed.find(key);
        if (it != removed.end()) {
            WidgetDiffLine& r = out[it->second];
            r.kind = WidgetDiffLine::Kind::changed;
            r.after = value;
            removed.erase(it);
        } else {
            out.push_back({WidgetDiffLine::Kind::added, key, "", value.empty() ? key : value});
        }
    }
    return out;
}

// ---- the logic ----------------------------------------------------------------------------

ApprovalCards::ApprovalCards() = default;

ApprovalCards::ApprovalCards(Hooks hooks) : hooks_(std::move(hooks)) {}

void ApprovalCards::set_decisions(const std::string& path) {
    decisions_path_ = path;
    decisions_read_ = 0;
    decisions_waiting_.clear();
}

const ApprovalCard* ApprovalCards::find(const std::string& id) const {
    for (const ApprovalCard& c : cards_)
        if (c.id == id) return &c;
    return nullptr;
}

size_t ApprovalCards::waiting() const {
    size_t n = 0;
    for (const ApprovalCard& c : cards_) n += c.state == ApprovalState::waiting || c.state == ApprovalState::deciding;
    return n;
}

const ApprovalCard* ApprovalCards::card_for_panel(const std::string& panel_id) const {
    for (const ApprovalCard& c : cards_) {
        if (c.state != ApprovalState::waiting && c.state != ApprovalState::deciding) continue;
        if (std::find(c.panels.begin(), c.panels.end(), panel_id) != c.panels.end()) return &c;
    }
    return nullptr;
}

void ApprovalCards::clear() {
    cards_.clear();
    decisions_made_.clear();
    session_ = nullptr;
    read_now_ = true;
}

void ApprovalCards::line(const ApprovalCard& c, const std::string& detail) {
    if (hooks_.log) hooks_.log(c.origin + " approval", c.what, detail);
    std::ostream& o = hooks_.err ? *hooks_.err : std::cerr;
    o << "[approval] " << c.id << " " << c.name << " (" << c.origin << ") " << detail << std::endl;
}

void ApprovalCards::sync(cb::Session* s, bool busy, double now) {
    now_ = now;
    if (s != session_) {
        cards_.clear();
        decisions_made_.clear();
        session_ = s;
        read_now_ = true;
    }
    if (!s) return;
    if (!busy && (read_now_ || last_read_ < 0.0 || now - last_read_ >= kReadInterval)) read_pending(*s, now);
    apply_decisions();
    for (ApprovalCard& c : cards_)
        if (c.state == ApprovalState::waiting || c.state == ApprovalState::deciding)
            c.time_left = s->queue().approval_time_left(c.id);
    cards_.erase(std::remove_if(cards_.begin(), cards_.end(),
                                [&](const ApprovalCard& c) {
                                    return c.state != ApprovalState::waiting && c.state != ApprovalState::deciding &&
                                           now - c.decided_at >= kKeepDecided;
                                }),
                 cards_.end());
}

void ApprovalCards::read_pending(cb::Session& s, double now) {
    const std::vector<cb::PendingApproval> pending = s.registry().pending_approvals();
    last_read_ = now;
    read_now_ = false;
    std::set<std::string> ids;
    bool changed = false;
    for (const cb::PendingApproval& p : pending) {
        ids.insert(p.approval);
        if (find(p.approval)) continue;
        cards_.push_back(make_card(s, p));
        line(cards_.back(), "waiting");
        changed = true;
    }
    for (ApprovalCard& c : cards_) {
        if ((c.state == ApprovalState::waiting || c.state == ApprovalState::deciding) && !ids.count(c.id)) {
            settle(s, c, now);
            changed = true;
        }
    }
    // The waiting cards first, oldest first, then the decided ones.
    if (changed)
        std::stable_partition(cards_.begin(), cards_.end(), [](const ApprovalCard& c) {
            return c.state == ApprovalState::waiting || c.state == ApprovalState::deciding;
        });
}

ApprovalCard ApprovalCards::make_card(cb::Session& s, const cb::PendingApproval& p) const {
    const cb::CommandInvocation& inv = p.invocation;
    ApprovalCard c;
    c.id = p.approval;
    c.name = inv.name;
    c.origin = cb::command_origin_name(inv.origin);
    c.who = approval_who(inv);
    const std::optional<cb::OpenSpec> open = s.open_spec();
    const spec::Spec* before = open ? &open->spec : nullptr;
    c.what = approval_what(inv, before);
    c.loss = approval_loss(inv, before);
    if (inv.name.rfind("run.", 0) == 0) {
        // The script of the run that the command names (a managed run), else
        // of the attached script.
        std::string script;
        const std::string target = str_arg(inv.args, "run");
        cb::ModelRunner* runner = s.model_runner();
        if (!target.empty() && target != cb::kUserRunId && runner) {
            for (const cb::RunEntry& e : runner->list())
                if (e.id == target) script = e.script;
        } else if (cb::RunControl* run = s.run_control()) {
            script = run->state().script;
        }
        c.loss = approval_run_text(inv, script);
    }
    c.invocation = inv;
    c.rule = cb::permission_rule_key(inv);
    c.rules = s.approval_offers_rules(inv);
    if (!p.review.is_object()) return c;
    c.review = true;
    c.loss.clear();
    if (const auto d = p.review.find("diff"); d != p.review.end() && d->is_string()) c.full_diff = d->get<std::string>();
    if (const auto w = p.review.find("warnings"); w != p.review.end() && w->is_array())
        for (const json& x : *w) {
            if (!x.is_object()) continue;
            const std::string code = str_arg(x, "code"), message = str_arg(x, "message");
            c.warnings.push_back(message.empty() ? code : code + ": " + message);
        }
    // The compact diff of the edit made on the open spec as it is now; the
    // review's unified diff when it no longer applies.
    if (before && cb::is_spec_edit_command(inv.name)) {
        const cb::SpecEdit e = cb::apply_spec_edit(*before, inv.name, inv.args, s.source_resolver());
        if (!e.error) {
            std::string target;
            if (inv.name == "panel.patch" || inv.name == "panel.remove") target = str_arg(inv.args, "id");
            if (inv.name == "panel.create") {
                // The new panel's keys, then whatever else changed.
                for (const spec::Panel& q : e.spec.panels)
                    if (!before->find_panel(q.id)) target = q.id;
                if (const spec::Panel* q = e.spec.find_panel(target)) c.diff = widget_item_lines(spec::to_json(*q));
                for (WidgetDiffLine& l : widget_compact_diff(*before, e.spec, std::string()))
                    if (l.key != "panel " + target) c.diff.push_back(std::move(l));
            } else {
                c.diff = widget_compact_diff(*before, e.spec, target);
            }
            for (const spec::Panel& q : before->panels) {
                const spec::Panel* after = e.spec.find_panel(q.id);
                if (!after || spec::to_json(*after) != spec::to_json(q)) c.panels.push_back(q.id);
            }
        }
    }
    if (c.diff.empty()) c.diff = approval_diff_from_unified(c.full_diff);
    return c;
}

void ApprovalCards::settle(cb::Session& s, ApprovalCard& c, double now) {
    std::optional<cb::CommandResult> r;
    if (const auto it = decisions_made_.find(c.id); it != decisions_made_.end()) {
        if (it->second.valid() && it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            r = it->second.get();
            if (r->error.code == "command.unknown_approval") r.reset();
        }
        decisions_made_.erase(it);
    }
    if (!r) r = s.queue().approval_outcome(c.id);
    std::string detail;
    if (r && r->error.code == "command.needs_approval" && !r->approval.empty() && r->approval != c.id) {
        // What the card showed changed while it waited, so the approval is
        // void and the command waits again on a new card, for the new
        // content, which the pending approvals already list: this card goes
        // at once, as no outcome.
        c.state = ApprovalState::withdrawn;
        c.time_left.reset();
        c.decided_at = now - kKeepDecided;
        line(c, "reissued as " + r->approval);
        return;
    }
    if (!r) {
        c.state = ApprovalState::withdrawn;
        detail = "withdrawn";
    } else if (r->ok) {
        c.state = ApprovalState::approved;
        detail = "approved";
    } else if (r->error.code == "command.denied") {
        c.state = ApprovalState::denied;
        detail = "denied";
    } else if (r->error.code == "command.approval_timeout") {
        c.state = ApprovalState::timed_out;
        detail = "timed out";
    } else if (r->error.code == "command.shutdown") {
        c.state = ApprovalState::withdrawn;
        detail = "withdrawn";
    } else {
        // Approved, and then the command failed, such as an edit whose change
        // is no longer the one shown.
        c.state = r->error.code == "history.conflict" ? ApprovalState::conflict : ApprovalState::failed;
        c.error = r->error.code;
        detail = r->error.code;
    }
    c.time_left.reset();
    c.decided_at = now;
    line(c, detail);
}

bool ApprovalCards::decide(const std::string& id, bool approve) {
    if (!session_) return false;
    for (ApprovalCard& c : cards_) {
        if (c.id != id || c.state != ApprovalState::waiting) continue;
        cb::CommandQueue& q = session_->queue();
        decisions_made_[id] = approve ? q.approve(id) : q.deny(id);
        c.state = ApprovalState::deciding;
        read_now_ = true;
        return true;
    }
    return false;
}

bool ApprovalCards::decide_rule(const std::string& id, bool saved) {
    if (!session_) return false;
    for (ApprovalCard& c : cards_) {
        if (c.id != id || c.state != ApprovalState::waiting || !c.rules) continue;
        const cb::Error e = session_->allow_always(c.invocation, saved ? cb::RuleScope::saved : cb::RuleScope::session);
        if (e) {
            std::ostream& o = hooks_.err ? *hooks_.err : std::cerr;
            o << "[approval] " << c.name << ": no rule added: " << e.code << std::endl;
            return false;
        }
        if (hooks_.log)
            hooks_.log(saved ? "rule saved" : "rule for this session", c.name, c.origin == "mcp" ? "mcp" : "side chat");
        std::ostream& o = hooks_.err ? *hooks_.err : std::cerr;
        o << "[approval] " << c.id << " " << c.name << " (" << c.origin << ") "
          << (saved ? "rule saved" : "rule for this session") << std::endl;
        return decide(id, true);
    }
    return false;
}

bool ApprovalCards::allow_always(const std::string& id, bool saved) { return decide_rule(id, saved); }

bool ApprovalCards::approve(const std::string& id) { return decide(id, true); }

bool ApprovalCards::deny(const std::string& id) { return decide(id, false); }

void ApprovalCards::apply_decisions() {
    if (decisions_path_.empty()) return;
    if (decisions_checked_ < 0.0 || now_ - decisions_checked_ >= kReadInterval) {
        decisions_checked_ = now_;
        std::error_code ec;
        const std::uintmax_t size = std::filesystem::file_size(decisions_path_, ec);
        if (!ec && size > decisions_read_) {
            std::ifstream in(decisions_path_, std::ios::binary);
            in.seekg((std::streamoff)decisions_read_);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            // Whole lines only; a line still being written waits.
            const size_t last = text.rfind('\n');
            if (last != std::string::npos) {
                decisions_read_ += last + 1;
                size_t start = 0;
                while (start <= last) {
                    const size_t end = text.find('\n', start);
                    std::string l = text.substr(start, end - start);
                    start = end + 1;
                    while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
                    if (!l.empty() && l[0] != '#') decisions_waiting_.push_back(l);
                }
            }
        }
    }
    // Each decision goes to the oldest waiting card that matches; one that
    // finds none waits for a card.
    for (auto it = decisions_waiting_.begin(); it != decisions_waiting_.end();) {
        char verb[16] = "", name[72] = "", origin[16] = "";
        const int got = std::sscanf(it->c_str(), "%15s %71s %15s", verb, name, origin);
        const std::string v = verb;
        if (got < 2 || (v != "approve" && v != "deny" && v != "session" && v != "always")) {
            std::ostream& o = hooks_.err ? *hooks_.err : std::cerr;
            o << "[approval] decision line ignored: " << *it << std::endl;
            it = decisions_waiting_.erase(it);
            continue;
        }
        std::string id;
        bool rules = false;
        for (const ApprovalCard& c : cards_)
            if (c.state == ApprovalState::waiting && c.name == name && (got < 3 || c.origin == origin)) {
                id = c.id;
                rules = c.rules;
                break;
            }
        if (id.empty()) {
            ++it;
            continue;
        }
        it = decisions_waiting_.erase(it);
        if (v == "session" || v == "always") {
            if (!rules) {
                std::ostream& o = hooks_.err ? *hooks_.err : std::cerr;
                o << "[approval] decision line ignored, the card offers no rule: " << v << " " << name << std::endl;
                continue;
            }
            decide_rule(id, v == "always");
        } else {
            decide(id, v == "approve");
        }
    }
}

// ---- the layout ---------------------------------------------------------------------------

ApprovalLayoutMetrics ApprovalLayoutMetrics::for_text_scale(float s) {
    if (!(s > 0.0f)) s = 1.0f;
    ApprovalLayoutMetrics m;
    auto r = [s](float v) { return std::round(v * s); };
    m.line = r(20);
    m.pad = r(8);
    m.gap = r(6);
    m.button = r(22);
    m.button_pad = r(10);
    m.min_card = r(340);
    m.max_card = r(620);
    return m;
}

namespace {

constexpr const char* kApprove = "Approve";
constexpr const char* kAllowOnce = "Allow once";
constexpr const char* kAllowSession = "Allow always for this session";
constexpr const char* kDeny = "Deny";
constexpr const char* kDiff = "Diff";
constexpr const char* kLabel = "APPROVALS";
constexpr const char* kTimeSample = "0:00";

}  // namespace

ApprovalStripLayout layout_approval_strip(const ApprovalLayoutInput& in, const ApprovalLayoutMetrics& m,
                                          const WidgetMeasure& measure) {
    ApprovalStripLayout L;
    const int n = (int)in.cards.size();
    if (n == 0 || in.width <= 0.0f) return L;

    // Across: the label, the cards at their minimum width, "+N".
    const float label_w = std::ceil(std::max(measure(kLabel, WidgetFont::tag), measure("99", WidgetFont::mono)));
    const float more_w = std::ceil(measure("+" + std::to_string(n), WidgetFont::mono) + 2.0f * m.button_pad);
    const float right = in.width - m.pad;
    auto fit_count = [&](float area) {
        return area < m.min_card ? 0 : (int)std::floor((area + m.gap) / (m.min_card + m.gap));
    };
    bool label = true;
    float x0 = m.pad + label_w + 2.0f * m.pad;
    float area = right - x0;
    if (fit_count(area) < 1) {
        // The label goes before the card does.
        label = false;
        x0 = m.pad;
        area = right - x0;
    }
    int shown = std::min(n, std::max(1, fit_count(area)));
    if (shown < n) shown = std::min(n, std::max(1, fit_count(area - more_w - m.gap)));
    L.hidden = n - shown;
    const float cards_area = std::max(0.0f, area - (L.hidden > 0 ? more_w + m.gap : 0.0f));
    const float card_w =
        std::floor(std::max(0.0f, std::min(m.max_card, (cards_area - m.gap * (float)(shown - 1)) / (float)shown)));

    // Down: the body is as tall as the tallest shown card needs.
    bool any_multi = false;
    int lines = 1;
    bool warn = false;
    bool rules = false;   // a shown card has the row of rule answers
    for (int i = 0; i < shown; ++i) rules = rules || in.cards[(size_t)i].rules;
    for (int i = 0; i < shown; ++i) {
        const ApprovalCardShape& c = in.cards[(size_t)i];
        if (c.decided || !c.review) continue;
        lines = std::max(lines, std::min(c.diff_lines, m.max_diff_lines));
        any_multi = any_multi || c.diff_lines > 1;
        warn = warn || c.warnings;
    }
    auto card_h = [&](int body, bool w) {
        return m.gap + m.line + m.line * (float)body + (w ? m.line : 0.0f) + m.gap + m.button + m.gap +
               (rules ? m.button + m.gap : 0.0f);
    };
    float cap = in.window_h > 0.0f ? std::floor(in.window_h * m.max_share) : 1e9f;
    if (in.max_height > 0.0f) cap = std::min(cap, in.max_height);
    const int min_lines = any_multi ? 2 : 1;   // a first line and "+N"
    while (lines > min_lines && 2.0f * m.pad + card_h(lines, warn) > cap) --lines;
    if (warn && 2.0f * m.pad + card_h(lines, warn) > cap) warn = false;
    float ch = card_h(lines, warn);
    L.height = 2.0f * m.pad + ch;
    if (in.fill_height > L.height) {
        // A strip kept taller for a moment: the cards fill it, their buttons
        // at the bottom.
        L.height = in.fill_height;
        ch = L.height - 2.0f * m.pad;
    }

    if (label) {
        L.label = {m.pad, m.pad + m.gap, label_w, m.line};
        if (in.waiting > 1) L.count = {m.pad, L.label.bottom(), label_w, m.line};
    }
    if (L.hidden > 0) L.more = {right - more_w, m.pad + m.gap, more_w, m.button};

    const float time_w = std::ceil(measure(kTimeSample, WidgetFont::mono));
    auto button_w = [&](const char* label) { return std::ceil(measure(label, WidgetFont::label) + 2.0f * m.button_pad); };
    float x = x0;
    for (int i = 0; i < shown; ++i) {
        const ApprovalCardShape& c = in.cards[(size_t)i];
        ApprovalCardLayout cl;
        cl.card = {x, m.pad, card_w, ch};
        const float ix = x + m.pad, iw = std::max(0.0f, card_w - 2.0f * m.pad);
        float y = cl.card.y + m.gap;
        // The header: who, what, the time left at the right end.
        const float who_w = std::min(iw, std::ceil(measure(upper(c.who), WidgetFont::tag)));
        cl.who = {ix, y, who_w, m.line};
        const float what_x = cl.who.right() + m.pad * 1.5f;
        float what_r = ix + iw;
        if (what_r - time_w - m.pad - what_x >= 4.0f * time_w) {
            cl.time = {what_r - time_w, y, time_w, m.line};
            what_r = cl.time.x - m.pad;
        }
        cl.what = {what_x, y, std::max(0.0f, what_r - what_x), m.line};
        y += m.line;
        // The body: the compact diff, folding past `lines`, or one line.
        if (c.review && !c.decided) {
            if (c.diff_lines <= lines) {
                cl.diff_shown = c.diff_lines;
            } else {
                cl.diff_shown = std::max(1, lines - 1);
                cl.diff_folded = true;
            }
            const int rows = std::max(1, cl.diff_shown + (cl.diff_folded ? 1 : 0));
            cl.body = {ix, y, iw, m.line * (float)rows};
        } else {
            cl.body = {ix, y, iw, m.line};
        }
        y += m.line * (float)lines;
        if (warn && c.warnings && !c.decided) cl.warnings = {ix, y, iw, m.line};
        // The buttons on the card's last row: Approve, Deny, and Diff at the
        // right end; when they do not fit, they share the row.
        if (!c.decided) {
            const float by = cl.card.bottom() - m.gap - m.button;
            float aw = button_w(c.rules ? kAllowOnce : kApprove), dw = button_w(kDeny),
                  fw = c.review ? button_w(kDiff) : 0.0f;
            const float need = aw + m.gap + dw + (c.review ? m.gap + fw : 0.0f);
            if (need > iw) {
                const int k = c.review ? 3 : 2;
                const float each = std::floor((iw - m.gap * (float)(k - 1)) / (float)k);
                aw = dw = each;
                fw = c.review ? each : 0.0f;
            }
            cl.approve = {ix, by, aw, m.button};
            cl.deny = {cl.approve.right() + m.gap, by, dw, m.button};
            if (c.review) cl.diff = {std::max(cl.deny.right() + m.gap, ix + iw - fw), by, fw, m.button};
            // The rule answers in a row of their own above: for this session,
            // then always; they share the row when both do not fit.
            if (c.rules) {
                const float ry = by - m.gap - m.button;
                float sw = button_w(approval_session_label(c.what_command).c_str()),
                      lw = button_w(approval_always_label(c.what_command).c_str());
                if (sw + m.gap + lw > iw) sw = lw = std::floor((iw - m.gap) * 0.5f);
                cl.session = {ix, ry, sw, m.button};
                cl.always = {cl.session.right() + m.gap, ry, lw, m.button};
            }
        }
        L.cards.push_back(cl);
        x += card_w + m.gap;
    }
    return L;
}

const char* approval_approve_label(bool rules) { return rules ? kAllowOnce : kApprove; }
const char* approval_session_label() { return kAllowSession; }

namespace {

// The target of a rule key, as a noun phrase: "changes of lr", "action save";
// "" for a key without one.
std::string rule_target_words(const std::string& rule) {
    const std::size_t colon = rule.find(':');
    if (colon == std::string::npos) return "";
    const std::string command = rule.substr(0, colon), target = rule.substr(colon + 1);
    if (command == "run.set_param") return "changes of " + target;
    if (command == "run.action") return "action " + target;
    return "";
}

}  // namespace

std::string approval_session_label(const std::string& rule) {
    const std::string words = rule_target_words(rule);
    return words.empty() ? std::string(kAllowSession) : "Allow " + words + " for this session";
}

std::string approval_always_label(const std::string& rule) {
    const std::string words = rule_target_words(rule);
    return "Always allow " + (words.empty() ? rule : words);
}

ApprovalLayoutInput approval_layout_input(const ApprovalCards& cards, float width, float window_h) {
    ApprovalLayoutInput in;
    in.width = width;
    in.window_h = window_h;
    in.waiting = (int)cards.waiting();
    for (const ApprovalCard& c : cards.cards()) {
        ApprovalCardShape s;
        s.who = c.who;
        s.review = c.review;
        s.diff_lines = (int)c.diff.size();
        s.warnings = !c.warnings.empty();
        s.decided = c.state != ApprovalState::waiting && c.state != ApprovalState::deciding;
        s.rules = c.rules && !s.decided;
        s.what_command = c.rule.empty() ? c.name : c.rule;
        in.cards.push_back(std::move(s));
    }
    return in;
}

float ApprovalStripHeight::update(double now, bool open, float need) {
    if (!open) {
        height_ = 0.0f;
        asked_since_ = -1.0;
        return height_;
    }
    if (height_ <= 0.0f || need == height_) {
        // On opening, the strip takes what it needs at once.
        if (height_ <= 0.0f) height_ = need;
        asked_since_ = -1.0;
        return height_;
    }
    if (asked_since_ < 0.0 || need != asked_) {
        asked_ = need;
        asked_since_ = now;
    }
    const double delay = need > height_ ? JobsTrayPresence::kOpenDelay : JobsTrayPresence::kCloseDelay;
    if (now - asked_since_ >= delay) {
        height_ = need;
        asked_since_ = -1.0;
    }
    return height_;
}

// ---- the ImGui side -----------------------------------------------------------------------

namespace {

ImRect rr(const WidgetRect& r, ImVec2 o) { return ImRect(ImVec2(o.x + r.x, o.y + r.y), ImVec2(o.x + r.right(), o.y + r.bottom())); }

float tracking(const SpecLook& L) { return std::round(L.small_size * 0.08f); }

WidgetMeasure measure_for(const SpecLook& L) {
    return [&L](const std::string& t, WidgetFont font) {
        switch (font) {
            case WidgetFont::label: return card_text_extent(L.ui, L.ui_size, t).x;
            case WidgetFont::key:
            case WidgetFont::mono: return card_text_extent(L.mono, L.mono_size, t).x;
            case WidgetFont::tag: return sternbild::text_tracked_width(L.ui, L.small_size, t.c_str(), tracking(L));
        }
        return 0.0f;
    };
}

std::string time_text(const std::optional<std::chrono::milliseconds>& left) {
    if (!left) return {};
    const long long s = (left->count() + 999) / 1000;
    char buf[16];
    std::snprintf(buf, sizeof buf, "%lld:%02lld", s / 60, s % 60);
    return buf;
}

// A button in the house's look: the primary on a cut plate, the others in a
// 1 px outline; faint while `enabled` is false.
bool card_button(const SpecLook& L, ImDrawList* dl, const ImRect& b, const char* label, bool primary, bool enabled,
                 bool lit = false) {
    if (b.GetWidth() < 1.0f) return false;
    ImGui::SetCursorScreenPos(b.Min);
    const bool clicked = ImGui::InvisibleButton(label, b.GetSize()) && enabled;
    const bool hovered = enabled && ImGui::IsItemHovered(), held = enabled && ImGui::IsItemActive();
    ImU32 outline = !enabled ? L.rule_faint : primary ? L.text : L.rule, fill = 0, ink = enabled ? L.text : L.faint;
    if (held && hovered) {
        fill = L.text;
        outline = fill;
        ink = L.deep;
    } else if (hovered) {
        fill = L.plate_hover;
    }
    if (lit) outline = L.accent;
    const float s = L.text_scale;
    if (primary) {
        sternbild::cut_plate(dl, b.Min, b.Max, outline, fill, std::round(4.0f * s));
    } else {
        if (fill) dl->AddRectFilled(b.Min, b.Max, fill);
        dl->AddRect(b.Min, b.Max, outline, 0.0f, 0, 1.0f);
    }
    const float ui_h = card_text_extent(L.ui, L.ui_size, "Ag").y;
    const std::string t = card_fit_text(L.ui, L.ui_size, label, b.GetWidth() - std::round(8.0f * s));
    const float tw = card_text_extent(L.ui, L.ui_size, t).x;
    card_draw_text(dl, L.ui, L.ui_size, ImVec2(b.GetCenter().x - tw * 0.5f, b.Min.y + (b.GetHeight() - ui_h) * 0.5f), ink,
                   t);
    return clicked;
}

}  // namespace

#endif  // CALIPER_WITH_BUILDER

// ---- the app's part -----------------------------------------------------------------------

#if CALIPER_WITH_BUILDER

struct ApprovalTray::Impl {
    Hooks hooks;
    ApprovalCards cards;
    JobsTrayPresence presence;
    ApprovalStripHeight height;
    double now = 0.0;
    bool started = false;
    bool closed = false;
    std::string diff_open;   // the card whose full diff is open
    std::string pointed;     // the card the side chat points at, until pointed_until
    double pointed_until = 0.0;
    int drawn_waiting = 0;   // the waiting cards of the last frame's strip at its full height (shown_waiting())

    void draw_card(const SpecLook& L, ImDrawList* dl, ImVec2 o, const ApprovalCard& c, const ApprovalCardLayout& cl);
    void draw_diff_popup(const SpecLook& L, const ImRect& card, float strip_top);
    void draw_more_popup(const SpecLook& L);
};

ApprovalTray::ApprovalTray() : impl_(std::make_unique<Impl>()) {}

ApprovalTray::~ApprovalTray() = default;

void ApprovalTray::start(Hooks hooks) {
    Impl& im = *impl_;
    im.hooks = std::move(hooks);
    ApprovalCards::Hooks ch;
    ch.log = im.hooks.log;
    ch.err = im.hooks.err;
    im.cards = ApprovalCards(ch);
    im.cards.set_decisions(im.hooks.decisions);
    im.started = true;
}

void ApprovalTray::frame(caliper_builder::Session* session, bool busy, double now) {
    Impl& im = *impl_;
    if (!im.started || im.closed) return;
    im.now = now;
    im.cards.sync(session, busy, now);
    im.presence.update(now, im.cards.waiting() > 0);
}

void ApprovalTray::close() {
    impl_->closed = true;
    impl_->cards.clear();
}

void ApprovalTray::Impl::draw_card(const SpecLook& L, ImDrawList* dl, ImVec2 o, const ApprovalCard& c,
                                   const ApprovalCardLayout& cl) {
    const float s = L.text_scale;
    const ImRect card = rr(cl.card, o);
    dl->AddRectFilled(card.Min, card.Max, L.deep);
    if (c.id == pointed && now < pointed_until) dl->AddRect(card.Min, card.Max, L.accent, 0.0f, 0, 1.0f);
    const float ui_h = card_text_extent(L.ui, L.ui_size, "Ag").y;
    const float small_h = card_text_extent(L.ui, L.small_size, "Ag").y;
    const float mono_h = card_text_extent(L.mono, L.mono_size, "Ag").y;
    const bool waiting = c.state == ApprovalState::waiting || c.state == ApprovalState::deciding;
    auto mid = [](const ImRect& r, float h) { return std::round(r.Min.y + (r.GetHeight() - h) * 0.5f); };

    // The header: who asks, what, and the time left.
    const ImRect who = rr(cl.who, o);
    const std::string who_tag = upper(c.who);
    sternbild::text_tracked(dl, L.ui, L.small_size, ImVec2(who.Min.x, mid(who, small_h)), L.secondary,
                            who_tag.c_str(), tracking(L));
    const ImRect what = rr(cl.what, o);
    card_draw_text(dl, L.ui, L.ui_size, ImVec2(what.Min.x, mid(what, ui_h)), L.text,
                   card_fit_text(L.ui, L.ui_size, c.what, what.GetWidth()));
    if (!cl.time.empty() && waiting && c.time_left) {
        const ImRect t = rr(cl.time, o);
        const std::string tt = time_text(c.time_left);
        const float tw = card_text_extent(L.mono, L.mono_size, tt).x;
        const bool soon = c.time_left->count() <= 30000;
        card_draw_text(dl, L.mono, L.mono_size, ImVec2(t.Max.x - tw, mid(t, mono_h)), soon ? L.text : L.faint, tt);
    }

    // The body: the compact diff, what is lost, or the outcome.
    const ImRect body = rr(cl.body, o);
    if (!waiting) {
        const std::string tag = approval_state_tag(c.state);
        sternbild::text_tracked(dl, L.ui, L.small_size, ImVec2(body.Min.x, mid(body, small_h)), L.secondary,
                                tag.c_str(), tracking(L));
        if (!c.error.empty()) {
            const float dx =
                body.Min.x + sternbild::text_tracked_width(L.ui, L.small_size, tag.c_str(), tracking(L)) + std::round(12.0f * s);
            card_draw_text(dl, L.mono, L.mono_size, ImVec2(dx, mid(body, mono_h)), L.faint,
                           card_fit_text(L.mono, L.mono_size, c.error, body.Max.x - dx));
        }
    } else if (c.review) {
        draw_compact_diff(dl, L, body, c.diff, cl.diff_shown, cl.diff_folded);
    } else if (!c.loss.empty()) {
        card_draw_text(dl, L.ui, L.ui_size, ImVec2(body.Min.x, mid(body, ui_h)), L.text,
                       card_fit_text(L.ui, L.ui_size, c.loss, body.GetWidth()));
    }
    if (!cl.warnings.empty() && waiting && !c.warnings.empty()) {
        const ImRect w = rr(cl.warnings, o);
        const std::string tag = c.warnings.size() == 1 ? "WARNING" : std::to_string(c.warnings.size()) + " WARNINGS";
        sternbild::text_tracked(dl, L.ui, L.small_size, ImVec2(w.Min.x, mid(w, small_h)), L.faint, tag.c_str(),
                                tracking(L));
        const float dx = w.Min.x + sternbild::text_tracked_width(L.ui, L.small_size, tag.c_str(), tracking(L)) +
                         std::round(12.0f * s);
        card_draw_text(dl, L.ui, L.small_size, ImVec2(dx, mid(w, small_h)), L.secondary,
                       card_fit_text(L.ui, L.small_size, c.warnings.front(), w.Max.x - dx));
    }

    // The buttons.
    if (waiting) {
        const bool enabled = c.state == ApprovalState::waiting;
        ImGui::PushID(c.id.c_str());
        if (card_button(L, dl, rr(cl.approve, o), c.rules ? kAllowOnce : kApprove, true, enabled)) cards.approve(c.id);
        if (card_button(L, dl, rr(cl.deny, o), kDeny, false, enabled)) cards.deny(c.id);
        const std::string rule = c.rule.empty() ? c.name : c.rule;
        if (!cl.session.empty() &&
            card_button(L, dl, rr(cl.session, o), approval_session_label(rule).c_str(), false, enabled))
            cards.allow_always(c.id, false);
        if (!cl.always.empty() &&
            card_button(L, dl, rr(cl.always, o), approval_always_label(rule).c_str(), false, enabled))
            cards.allow_always(c.id, true);
        if (!cl.diff.empty() && card_button(L, dl, rr(cl.diff, o), kDiff, false, true, diff_open == c.id)) {
            diff_open = diff_open == c.id ? std::string() : c.id;
            if (!diff_open.empty()) ImGui::OpenPopup("##approval.diff");
        }
        ImGui::PopID();
    }
}

void ApprovalTray::Impl::draw_diff_popup(const SpecLook& L, const ImRect& card, float strip_top) {
    const ApprovalCard* c = cards.find(diff_open);
    ImGui::PushID(diff_open.c_str());
    if (!c || !(c->state == ApprovalState::waiting || c->state == ApprovalState::deciding)) {
        diff_open.clear();
        ImGui::PopID();
        return;
    }
    const float s = L.text_scale;
    const float lh = std::round(20.0f * s);
    int n = 1;
    for (char ch : c->full_diff) n += ch == '\n';
    const ImVec2 display = ImGui::GetMainViewport()->Size;
    const float w = std::min(display.x - 2.0f * std::round(8.0f * s), std::max(card.GetWidth(), std::round(560.0f * s)));
    const float h = std::min(std::floor((float)n * lh + 3.0f * std::round(20.0f * s)), std::floor(display.y * 0.6f));
    ImGui::SetNextWindowPos(ImVec2(std::min(card.Min.x, display.x - w - std::round(8.0f * s)), strip_top - 4.0f),
                            ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, L.deep);
    ImGui::PushStyleColor(ImGuiCol_Border, L.rule);
    if (ImGui::BeginPopup("##approval.diff", ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const std::string who = upper(c->who);
        sternbild::text_tracked(dl, L.ui, L.small_size, p, L.secondary, who.c_str(), tracking(L));
        const float wx = p.x + sternbild::text_tracked_width(L.ui, L.small_size, who.c_str(), tracking(L)) + std::round(12.0f * s);
        card_draw_text(dl, L.ui, L.ui_size, ImVec2(wx, p.y - 1.0f), L.text,
                       card_fit_text(L.ui, L.ui_size, c->what, p.x + ImGui::GetContentRegionAvail().x - wx));
        ImGui::Dummy(ImVec2(1.0f, lh));
        if (ImGui::BeginChild("##approval.diff.text", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_HorizontalScrollbar)) {
            draw_unified_diff(L, c->full_diff, lh);
            for (const std::string& warning : c->warnings) {
                const ImVec2 q = ImGui::GetCursorScreenPos();
                card_draw_text(ImGui::GetWindowDrawList(), L.ui, L.small_size, q, L.secondary, warning);
                ImGui::Dummy(ImVec2(1.0f, lh));
            }
        }
        ImGui::EndChild();
        ImGui::EndPopup();
    } else {
        diff_open.clear();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopID();
}

void ApprovalTray::Impl::draw_more_popup(const SpecLook& L) {
    ImGui::PushStyleColor(ImGuiCol_PopupBg, L.deep);
    ImGui::PushStyleColor(ImGuiCol_Border, L.rule);
    if (ImGui::BeginPopup("##approval.more")) {
        const float s = L.text_scale;
        const float lh = std::round(24.0f * s);
        const SpecLook& look = L;
        for (const ApprovalCard& c : cards.cards()) {
            if (c.state != ApprovalState::waiting && c.state != ApprovalState::deciding) continue;
            ImGui::PushID(c.id.c_str());
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float small_h = card_text_extent(look.ui, look.small_size, "Ag").y;
            const float ui_h = card_text_extent(look.ui, look.ui_size, "Ag").y;
            const std::string who = upper(c.who);
            sternbild::text_tracked(dl, look.ui, look.small_size, ImVec2(p.x, std::round(p.y + (lh - small_h) * 0.5f)),
                                    look.secondary, who.c_str(), tracking(look));
            const float wx = p.x + std::round(110.0f * s);
            card_draw_text(dl, look.ui, look.ui_size, ImVec2(wx, p.y + (lh - ui_h) * 0.5f), look.text,
                           card_fit_text(look.ui, look.ui_size, c.what, std::round(300.0f * s)));
            const float tx = wx + std::round(310.0f * s);
            card_draw_text(dl, look.mono, look.mono_size, ImVec2(tx, p.y + (lh - ui_h) * 0.5f), look.faint,
                           time_text(c.time_left));
            const float bw = std::round(84.0f * s), bh = std::round(22.0f * s), by = p.y + (lh - bh) * 0.5f;
            const float bx = tx + std::round(50.0f * s);
            const bool enabled = c.state == ApprovalState::waiting;
            if (card_button(look, dl, ImRect(ImVec2(bx, by), ImVec2(bx + bw, by + bh)), kApprove, true, enabled))
                cards.approve(c.id);
            if (card_button(look, dl, ImRect(ImVec2(bx + bw + std::round(6.0f * s), by),
                                             ImVec2(bx + 2.0f * bw + std::round(6.0f * s), by + bh)),
                            kDeny, false, enabled))
                cards.deny(c.id);
            ImGui::SetCursorScreenPos(p);
            ImGui::Dummy(ImVec2(bx + 2.0f * bw + std::round(6.0f * s) - p.x, lh));
            ImGui::PopID();
        }
        if (cards.waiting() == 0) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor(2);
}

void ApprovalTray::draw_strip() {
    Impl& im = *impl_;
    im.drawn_waiting = 0;
    if (!im.started || im.closed || !im.presence.open() || im.cards.cards().empty()) {
        im.height.update(im.now, false, 0.0f);
        return;
    }
    const SpecLook L = house_spec_look();
    const ApprovalLayoutMetrics m = ApprovalLayoutMetrics::for_text_scale(L.text_scale);
    const WidgetMeasure measure = measure_for(L);
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ApprovalLayoutInput in = approval_layout_input(im.cards, vp->Size.x, vp->Size.y);
    const float need = layout_approval_strip(in, m, measure).height;
    const float h = im.height.update(im.now, true, need);
    in.max_height = h;
    in.fill_height = h;
    const ApprovalStripLayout lay = layout_approval_strip(in, m, measure);
    if (h < 1.0f) return;
    if (h >= need) im.drawn_waiting = (int)im.cards.waiting();

    // A viewport side bar at the bottom, after the status bar and the jobs
    // tray, so it lands above them and the work area ends above it.
    ImGui::PushStyleColor(ImGuiCol_WindowBg, L.ground);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0.0f, 0.0f));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoScrollWithMouse;
    const bool open = ImGui::BeginViewportSideBar("##caliper_approvals", ImGui::GetMainViewport(), ImGuiDir_Down, h, flags);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    if (open) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 o = ImGui::GetWindowPos();
        const ImVec2 size = ImGui::GetWindowSize();
        dl->PushClipRect(o, ImVec2(o.x + size.x, o.y + size.y), false);
        sternbild::rule_h(dl, o.x, o.x + size.x, o.y);   // the rule above, as the tray's

        // The label, and the count when several wait.
        if (!lay.label.empty()) {
            const ImRect l = rr(lay.label, o);
            const float small_h = card_text_extent(L.ui, L.small_size, "Ag").y;
            sternbild::text_tracked(dl, L.ui, L.small_size, ImVec2(l.Min.x, std::round(l.Min.y + (l.GetHeight() - small_h) * 0.5f)),
                                    L.secondary, kLabel, tracking(L));
            if (!lay.count.empty()) {
                const ImRect c = rr(lay.count, o);
                const float mono_h = card_text_extent(L.mono, L.mono_size, "Ag").y;
                card_draw_text(dl, L.mono, L.mono_size, ImVec2(c.Min.x, c.Min.y + (c.GetHeight() - mono_h) * 0.5f), L.text,
                               std::to_string(im.cards.waiting()));
            }
        }
        const std::vector<ApprovalCard>& cards = im.cards.cards();
        ImRect diff_card;
        for (size_t i = 0; i < lay.cards.size() && i < cards.size(); ++i) {
            im.draw_card(L, dl, o, cards[i], lay.cards[i]);
            if (cards[i].id == im.diff_open) diff_card = rr(lay.cards[i].card, o);
        }
        if (!lay.more.empty()) {
            const std::string more = "+" + std::to_string(lay.hidden);
            if (card_button(L, dl, rr(lay.more, o), more.c_str(), false, true)) ImGui::OpenPopup("##approval.more");
            const ImRect b = rr(lay.more, o);
            ImGui::SetNextWindowPos(ImVec2(b.Max.x, o.y - 4.0f), ImGuiCond_Always, ImVec2(1.0f, 1.0f));
            im.draw_more_popup(L);
        }
        if (!im.diff_open.empty()) {
            if (diff_card.GetWidth() <= 0.0f) diff_card = ImRect(o, ImVec2(o.x + m.max_card, o.y));
            im.draw_diff_popup(L, diff_card, o.y);
        }
        dl->PopClipRect();
    }
    ImGui::End();
}

int ApprovalTray::shown_waiting() const { return impl_->drawn_waiting; }

void ApprovalTray::point_to(const std::string& approval_id) {
    impl_->pointed = approval_id;
    impl_->pointed_until = impl_->now + 2.0;
}

void ApprovalTray::mark_panel(const std::string& panel_id, const std::string& title, float x, float y, float w,
                              float h, float strip) {
    Impl& im = *impl_;
    if (!im.started || im.closed) return;
    const ApprovalCard* c = im.cards.card_for_panel(panel_id);
    if (!c || w < 1.0f || h < 1.0f) return;
    const SpecLook L = house_spec_look();
    const float s = L.text_scale;
    // A REVIEW tag in bronze after the panel's title, where a preview's tag
    // goes, with the time left, on the strip's plate over the info string,
    // and a bronze rule under the strip, as a preview's; a tab, whose tab
    // carries its title, has the tag framed in its top right corner. Drawn
    // in the panel's own window, which the page's overlay is called in.
    const bool tab = strip < 1.0f;
    const float band = tab ? std::min(h, std::round(L.ui_size + 9.0f * s)) : strip;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const std::string tag = "REVIEW";
    const std::string left = time_text(c->time_left);
    const float tw = sternbild::text_tracked_width(L.ui, L.small_size, tag.c_str(), tracking(L));
    const float lw = left.empty() ? 0.0f : card_text_extent(L.mono, L.mono_size, left).x + std::round(8.0f * s);
    const float pad = std::round(8.0f * s);
    const float mark_w = tw + lw + 2.0f * pad;
    float x0 = x + w - std::min(w, mark_w);
    if (!tab) {
        // Right after the title as the runtime fits it, unless the title
        // leaves no room.
        const float title_w = std::min(card_text_extent(L.ui, L.ui_size, title).x, std::max(0.0f, w - 2.0f * pad));
        x0 = std::min(x0, std::max(x, x + pad + title_w + std::round(4.0f * s)));
    }
    const ImVec2 b0(std::round(x0), y), b1(x + w, y + band - (tab ? 0.0f : 1.0f));
    dl->AddRectFilled(b0, b1, L.plate);
    const float small_h = card_text_extent(L.ui, L.small_size, "Ag").y;
    const float mono_h = card_text_extent(L.mono, L.mono_size, "Ag").y;
    sternbild::text_tracked(dl, L.ui, L.small_size, ImVec2(b0.x + pad, std::round(y + (band - small_h) * 0.5f)), L.accent,
                            tag.c_str(), tracking(L));
    if (!left.empty())
        card_draw_text(dl, L.mono, L.mono_size, ImVec2(b0.x + pad + tw + std::round(8.0f * s), y + (band - mono_h) * 0.5f),
                       L.faint, left);
    if (!tab) dl->AddRectFilled(ImVec2(x, y + band - 1.0f), ImVec2(x + w, y + band), L.accent);
    else dl->AddRect(b0, b1, L.accent, 0.0f, 0, 1.0f);
}

#else  // !CALIPER_WITH_BUILDER

// Without the builder there are no approvals to show.
struct ApprovalTray::Impl {};

ApprovalTray::ApprovalTray() : impl_(std::make_unique<Impl>()) {}
ApprovalTray::~ApprovalTray() = default;
void ApprovalTray::start(Hooks) {}
void ApprovalTray::frame(caliper_builder::Session*, bool, double) {}
void ApprovalTray::draw_strip() {}
void ApprovalTray::mark_panel(const std::string&, const std::string&, float, float, float, float, float) {}
void ApprovalTray::point_to(const std::string&) {}
void ApprovalTray::close() {}
int ApprovalTray::shown_waiting() const { return 0; }

#endif

}  // namespace caliper_host
