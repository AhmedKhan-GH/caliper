// widget_surface.cpp: the widget agent's inline surface (widget_surface.h):
// the compact diff, the state machine, the layout as plain rectangles, and
// the ImGui side in the house style.
#include "builder/widget_surface.h"

#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"

#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <set>

namespace caliper_host {

using nlohmann::json;
namespace spec = caliper_builder::spec;
using caliper_builder::CommandResult;
using caliper_builder::Error;

// ---- the session ------------------------------------------------------------------------

SessionWidgetSession::SessionWidgetSession(caliper_builder::Session& session, std::function<bool()> busy)
    : session_(session), busy_(std::move(busy)) {}

bool SessionWidgetSession::ready() const { return !busy_ || !busy_(); }

CommandResult SessionWidgetSession::run(const std::string& name, json args) {
    caliper_builder::CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = caliper_builder::CommandOrigin::ui;
    return session_.invoke(std::move(inv));
}

std::optional<caliper_builder::OpenSpec> SessionWidgetSession::open_spec() const { return session_.open_spec(); }

caliper_builder::PreviewState SessionWidgetSession::preview(const std::string& id) const { return session_.preview(id); }

caliper_builder::PanelData SessionWidgetSession::resolve(const std::string& panel, const std::string& preview) const {
    return session_.resolve(panel, preview);
}

caliper_builder::ContextBundle SessionWidgetSession::bundle(const WidgetTarget& target, const std::string& preview) {
    caliper_builder::ContextBundleOptions o = bundle_options;
    o.preview = preview;
    o.origin = caliper_builder::CommandOrigin::ui;
    o.selection = target.kind == WidgetTarget::Kind::selection ? target.selection : std::nullopt;
    return caliper_builder::build_context_bundle(session_, target.to_builder(), o);
}

// ---- names ------------------------------------------------------------------------------

const char* widget_state_name(WidgetState state) {
    switch (state) {
        case WidgetState::idle: return "idle";
        case WidgetState::prompt: return "prompt";
        case WidgetState::running: return "running";
        case WidgetState::preview: return "preview";
        case WidgetState::applied: return "applied";
        case WidgetState::declined: return "declined";
        case WidgetState::handoff: return "handoff";
        case WidgetState::unavailable: return "unavailable";
        case WidgetState::conflict: return "conflict";
        case WidgetState::failed: return "failed";
    }
    return "idle";
}

const char* widget_action_label(WidgetAction action) {
    switch (action) {
        case WidgetAction::accept: return "Accept";
        case WidgetAction::reject: return "Reject";
        case WidgetAction::refine: return "Refine";
        case WidgetAction::diff: return "Diff";
        case WidgetAction::ask_in_chat: return "Ask in chat";
        case WidgetAction::cancel: return "Cancel";
        case WidgetAction::retry: return "Retry";
        case WidgetAction::rephrase: return "Rephrase";
        case WidgetAction::model: return "Model";
        case WidgetAction::setup: return "Setup";
        case WidgetAction::close: return "Close";
    }
    return "";
}

const char* widget_action_key(WidgetAction action) {
    switch (action) {
        case WidgetAction::accept: return "\xE2\x8F\x8E";   // ⏎
        case WidgetAction::reject:
        case WidgetAction::cancel:
        case WidgetAction::close: return "esc";
        default: return "";
    }
}

namespace {

const char* handoff_tag(const std::string& kind) {
    if (kind == "build_from_scratch") return "NEW DASHBOARD";
    if (kind == "loader") return "LOADER";
    if (kind == "project_code") return "PROJECT CODE";
    if (kind == "explanation") return "EXPLANATION";
    if (kind == "another_target") return "ANOTHER TARGET";
    return "OUTSIDE SCOPE";
}

constexpr size_t kMaxReply = 4000;
constexpr size_t kMaxThought = 4000;

void append_bounded(std::string* s, const std::string& more, size_t limit) {
    if (s->size() >= limit) return;
    s->append(more, 0, limit - s->size());
}

std::string trimmed(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

}  // namespace

// ---- the state machine ------------------------------------------------------------------

struct WidgetSurface::Sink : WidgetPreviewSink {
    explicit Sink(SpecRuntime& r) : runtime(r) {}
    void show(const spec::Panel& panel, const caliper_builder::PanelData& data, const spec::Spec& s) override {
        runtime.set_preview(panel, data, &s);
    }
    void clear(const std::string& panel) override { runtime.clear_preview(panel); }
    void show_spec(const spec::Spec& s, const std::set<std::string>& added) override {
        runtime.set_spec_preview(s, added);
    }
    void clear_spec() override { runtime.clear_spec_preview(); }
    SpecRuntime& runtime;
};

WidgetSurface::WidgetSurface(WidgetSession& session, WidgetAgent& agent, Hooks hooks)
    : session_(session), agent_(agent), hooks_(std::move(hooks)) {
    std::random_device rd;
    char hex[8];
    std::snprintf(hex, sizeof hex, "%06x", (unsigned)(rd() & 0xFFFFFF));
    preview_prefix_ = std::string("widget-") + hex + "-";
}

WidgetSurface::~WidgetSurface() {
    // A preview left open would be dropped by the session's idle time; drop
    // it now, and end the agent's session.
    if (state_ == WidgetState::running) agent_.cancel();
    if (!preview_.empty()) session_.run("preview.discard", {{"id", preview_}});
    if (session_open_) agent_.close();
    if (runtime_) detach();
}

std::chrono::steady_clock::time_point WidgetSurface::now() const {
    return hooks_.now ? hooks_.now() : std::chrono::steady_clock::now();
}

std::string WidgetSurface::next_preview_id() {
    if (hooks_.make_preview_id) return hooks_.make_preview_id();
    return preview_prefix_ + std::to_string(++preview_count_);
}

void WidgetSurface::enter(WidgetState s) {
    state_ = s;
    if (s == WidgetState::prompt || s == WidgetState::unavailable) focus_input_ = true;
    if (s == WidgetState::applied) applied_at_ = now();
}

void WidgetSurface::reset_turn() {
    request_.clear();
    progress_.clear();
    reply_.clear();
    thought_.clear();
    handoff_.reset();
    error_ = Error();
    limit_ = false;
    show_full_diff_ = false;
}

std::string WidgetSurface::model_label() const {
    if (!model_.empty()) return model_;
    return agent_.model_label();
}

bool WidgetSurface::open(const WidgetTarget& target) {
    if (state_ != WidgetState::idle && state_ != WidgetState::applied) close();
    if (!target.panel.empty()) {
        const auto open = session_.open_spec();
        if (!open || !open->spec.find_panel(target.panel)) return false;
    }
    target_ = target;
    text_.clear();
    reset_turn();
    conversation_.clear();
    fresh_session_ = true;
    enter(agent_.available() ? WidgetState::prompt : WidgetState::unavailable);
    return true;
}

void WidgetSurface::set_text(std::string text) {
    if (state_ == WidgetState::prompt || state_ == WidgetState::unavailable) text_ = std::move(text);
}

void WidgetSurface::submit() {
    if (state_ == WidgetState::unavailable) {
        ask_in_chat();
        return;
    }
    if (state_ != WidgetState::prompt) return;
    const std::string t = trimmed(text_);
    if (t.empty()) return;
    begin_turn(t);
}

bool WidgetSurface::ensure_preview() {
    if (!preview_.empty()) return true;
    const std::string id = next_preview_id();
    json args = target_.preview_scope();
    args["id"] = id;
    CommandResult r = session_.run("preview.begin", args);
    if (!r.ok && r.error.code == "spec.no_open_spec" && target_.kind == WidgetTarget::Kind::dropped_file) {
        // A preview needs an open spec: a new one named after the loader,
        // from the ui origin, which Reject undoes again.
        std::string sid;
        for (char c : target_.loader) {
            const char l = (char)std::tolower((unsigned char)c);
            sid += (std::isalnum((unsigned char)l) || l == '-' || l == '_') ? l : '-';
        }
        if (sid.empty() || !std::isalpha((unsigned char)sid[0])) sid = "spec-" + sid;
        if (sid.size() > 64) sid.resize(64);
        const CommandResult n = session_.run("spec.new", {{"id", sid}, {"title", target_.loader}});
        if (n.ok) {
            created_spec_ = true;
            r = session_.run("preview.begin", args);
        }
    }
    if (!r.ok) {
        error_ = r.error;
        return false;
    }
    preview_ = r.result.value("id", id);
    edits_ = 0;
    diff_.clear();
    full_diff_.clear();
    const auto open = session_.open_spec();
    seen_open_text_ = open ? open->text : std::string();
    return true;
}

void WidgetSurface::begin_turn(const std::string& text) {
    const std::string request = text;
    progress_.clear();
    reply_.clear();
    thought_.clear();
    handoff_.reset();
    error_ = Error();
    limit_ = false;
    request_ = request;
    if (!session_.ready()) {
        // A long command holds the session: the turn starts at the next
        // update() that finds it free.
        pending_turn_ = request;
        enter(WidgetState::running);
        return;
    }
    if (!ensure_preview()) {
        enter(WidgetState::failed);
        return;
    }
    json context;
    if (!session_open_ || fresh_session_) {
        // A new session: the context bundle goes before its first request,
        // and its scope with session/new. Without one, the sidecar derives
        // the scope itself and the model reads what it needs.
        caliper_builder::ContextBundle b = session_.bundle(target_, preview_);
        bundle_error_ = b.error;
        bundle_cuts_ = b.cuts;
        std::vector<std::string> scope;
        if (!b.error) {
            context = std::move(b.blocks);
            scope = std::move(b.scope);
        }
        agent_.start(target_, preview_, scope);
        session_open_ = true;
        fresh_session_ = false;
    }
    conversation_.push_back({request, ""});
    agent_.prompt(request, context);
    text_.clear();
    enter(WidgetState::running);
}

void WidgetSurface::on_agent(const WidgetAgentEvent& e) {
    using K = WidgetAgentEvent::Kind;
    if (e.kind == K::exited) session_open_ = false;
    if (e.kind == K::started) {
        model_ = e.effort.empty() ? e.model : e.model + " \xC2\xB7 " + e.effort;   // ·
        return;
    }
    if (state_ != WidgetState::running) return;   // a turn the surface has left
    switch (e.kind) {
        case K::progress: progress_ = e.text; break;
        case K::thought: {
            // The agent's summarized thinking shows as the progress line: its
            // latest line, until the next tool call names a command.
            append_bounded(&thought_, e.text, kMaxThought);
            const std::string t = trimmed(thought_);
            const size_t nl = t.find_last_of('\n');
            const std::string line = trimmed(nl == std::string::npos ? t : t.substr(nl + 1));
            if (!line.empty()) progress_ = line.size() > 160 ? line.substr(0, 160) : line;
            break;
        }
        case K::message: append_bounded(&reply_, e.text, kMaxReply); break;
        case K::finished: finish_turn(e); break;
        case K::failed:
        case K::exited:
            error_ = e.error;
            if (e.kind == K::failed) end_session();
            drop_preview();
            enter(WidgetState::failed);
            break;
        case K::started: break;
    }
}

void WidgetSurface::finish_turn(const WidgetAgentEvent& e) {
    if (e.stop_reason == "cancelled") return;   // the surface left the turn already
    reply_ = trimmed(reply_);
    if (!conversation_.empty()) conversation_.back().reply = reply_;
    progress_.clear();
    if (e.handoff) {
        // A preview ends with a handoff; the conversation goes with Ask in chat.
        handoff_ = e.handoff;
        drop_preview();
        end_session();
        enter(WidgetState::handoff);
        return;
    }
    if (!read_preview()) {
        enter(error_.code == "history.conflict" ? WidgetState::conflict : WidgetState::failed);
        return;
    }
    show_preview();
    if (e.stop_reason == "refusal") {
        // The declined prompt leaves the conversation, as ACP's refusal means.
        if (!conversation_.empty()) conversation_.pop_back();
        enter(WidgetState::declined);
        return;
    }
    limit_ = e.limit || e.stop_reason == "max_turn_requests" || e.stop_reason == "max_tokens";
    enter(WidgetState::preview);
}

bool WidgetSurface::read_preview() {
    if (preview_.empty()) return false;
    const CommandResult r = session_.run("preview.diff", {{"id", preview_}});
    if (!r.ok) {
        error_ = r.error;
        return false;
    }
    edits_ = r.result.value("edits", 0);
    full_diff_ = r.result.value("diff", std::string());
    const caliper_builder::PreviewState ps = session_.preview(preview_);
    const auto open = session_.open_spec();
    if (ps.error || !open) {
        error_ = ps.error ? ps.error : Error("spec.no_open_spec", "No spec is open.", "");
        return false;
    }
    diff_ = widget_compact_diff(open->spec, ps.spec.spec, target_.panel);
    seen_open_text_ = open->text;
    return true;
}

void WidgetSurface::clear_shown() {
    if (sink_ && !shown_panel_.empty()) sink_->clear(shown_panel_);
    if (sink_ && shown_spec_) sink_->clear_spec();
    shown_panel_.clear();
    shown_spec_ = false;
}

void WidgetSurface::show_preview() {
    if (!sink_) return;
    if (edits_ <= 0) {
        clear_shown();
        return;
    }
    const caliper_builder::PreviewState ps = session_.preview(preview_);
    const auto open = session_.open_spec();
    if (ps.error || !open) return;
    // A change that adds, removes or moves panels is drawn as the whole
    // spec's preview, the added panels in a dashed frame; a change inside
    // the target panel is drawn in that panel.
    std::set<std::string> added;
    bool whole = false;
    for (const auto& p : ps.spec.spec.panels)
        if (!open->spec.find_panel(p.id)) added.insert(p.id);
    for (const auto& p : open->spec.panels)
        if (!ps.spec.spec.find_panel(p.id)) whole = true;
    if (!added.empty()) whole = true;
    if (caliper_builder::effective_layout(ps.spec.spec) != caliper_builder::effective_layout(open->spec)) whole = true;
    if (whole) {
        if (!shown_panel_.empty()) sink_->clear(shown_panel_);
        shown_panel_.clear();
        sink_->show_spec(ps.spec.spec, added);
        shown_spec_ = true;
        return;
    }
    if (shown_spec_) {
        sink_->clear_spec();
        shown_spec_ = false;
    }
    if (target_.panel.empty()) return;
    const spec::Panel* p = ps.spec.spec.find_panel(target_.panel);
    if (!p) {
        if (!shown_panel_.empty()) sink_->clear(shown_panel_);
        shown_panel_.clear();
        return;
    }
    sink_->show(*p, session_.resolve(target_.panel, preview_), ps.spec.spec);
    shown_panel_ = target_.panel;
}

void WidgetSurface::drop_preview() {
    if (!preview_.empty()) {
        if (session_.ready()) session_.run("preview.discard", {{"id", preview_}});
        else pending_discards_.push_back(preview_);
    }
    preview_.clear();
    edits_ = 0;
    diff_.clear();
    full_diff_.clear();
    show_full_diff_ = false;
    clear_shown();
    if (created_spec_ && session_.ready()) session_.run("history.undo", json::object());   // the spec.new it made
    created_spec_ = false;
}

void WidgetSurface::end_session() {
    if (!session_open_) return;
    agent_.close();
    session_open_ = false;
}

void WidgetSurface::accept() {
    if (state_ != WidgetState::preview || edits_ <= 0) return;
    if (!session_.ready()) {
        pending_commit_ = true;
        return;
    }
    pending_commit_ = false;
    const CommandResult r = session_.run("preview.commit", {{"id", preview_}});
    if (!r.ok) {
        error_ = r.error;
        if (r.error.code == "history.conflict") {
            enter(WidgetState::conflict);
        } else {
            drop_preview();
            end_session();
            enter(WidgetState::failed);
        }
        return;
    }
    // The preview ended with the commit; the panel draws the open spec again.
    preview_.clear();
    edits_ = 0;
    created_spec_ = false;
    clear_shown();
    end_session();
    enter(WidgetState::applied);
}

void WidgetSurface::reject() {
    if (state_ == WidgetState::idle || state_ == WidgetState::applied) return;
    close();
}

void WidgetSurface::refine() {
    if (state_ == WidgetState::preview) {
        text_.clear();
        enter(WidgetState::prompt);
    } else if (state_ == WidgetState::conflict) {
        // A new preview from the spec as it is now, in a new agent session.
        drop_preview();
        end_session();
        conversation_.clear();
        fresh_session_ = true;
        text_ = request_;
        enter(WidgetState::prompt);
    }
}

void WidgetSurface::escape() {
    switch (state_) {
        case WidgetState::idle: break;
        case WidgetState::applied: enter(WidgetState::idle); break;
        case WidgetState::prompt:
            if (refining()) enter(WidgetState::preview);   // back to the card
            else close();
            break;
        case WidgetState::running: cancel(); break;
        default: close(); break;
    }
}

void WidgetSurface::cancel() {
    if (state_ != WidgetState::running) return;
    if (!pending_turn_.empty()) pending_turn_.clear();
    else agent_.cancel();
    drop_preview();
    end_session();
    conversation_.clear();
    fresh_session_ = true;
    text_ = request_;
    progress_.clear();
    reply_.clear();
    enter(WidgetState::prompt);
}

void WidgetSurface::ask_in_chat() {
    if (state_ == WidgetState::idle || state_ == WidgetState::applied) return;
    WidgetChatRequest req;
    const bool typing = state_ == WidgetState::prompt || state_ == WidgetState::unavailable;
    req.request = typing && !trimmed(text_).empty() ? trimmed(text_) : request_;
    req.target = target_;
    req.conversation = conversation_;
    req.diff = full_diff_;
    req.handoff = handoff_;
    close();
    if (hooks_.ask_in_chat) hooks_.ask_in_chat(req);
}

void WidgetSurface::retry() {
    if (state_ != WidgetState::declined && state_ != WidgetState::failed) return;
    if (state_ == WidgetState::failed) fresh_session_ = true;
    if (request_.empty()) {
        close();
        return;
    }
    begin_turn(request_);
}

void WidgetSurface::rephrase() {
    if (state_ != WidgetState::declined && state_ != WidgetState::handoff) return;
    if (state_ == WidgetState::handoff) fresh_session_ = true;
    text_ = request_;
    enter(WidgetState::prompt);
}

void WidgetSurface::open_model() {
    if (hooks_.open_settings) hooks_.open_settings();
}

void WidgetSurface::toggle_diff() {
    if (state_ == WidgetState::preview || state_ == WidgetState::conflict) show_full_diff_ = !show_full_diff_;
}

void WidgetSurface::close() {
    if (state_ == WidgetState::running && pending_turn_.empty()) agent_.cancel();
    pending_turn_.clear();
    pending_commit_ = false;
    drop_preview();
    end_session();
    text_.clear();
    conversation_.clear();
    reset_turn();
    fresh_session_ = true;
    enter(WidgetState::idle);
}

void WidgetSurface::abandon() {
    if (state_ == WidgetState::running && pending_turn_.empty()) agent_.cancel();
    pending_turn_.clear();
    pending_discards_.clear();
    pending_commit_ = false;
    preview_.clear();
    edits_ = 0;
    diff_.clear();
    full_diff_.clear();
    show_full_diff_ = false;
    if (sink_ && !shown_panel_.empty()) sink_->clear(shown_panel_);
    shown_panel_.clear();
    end_session();
    text_.clear();
    conversation_.clear();
    reset_turn();
    fresh_session_ = true;
    enter(WidgetState::idle);
}

void WidgetSurface::act(WidgetAction action) {
    switch (action) {
        case WidgetAction::accept: accept(); break;
        case WidgetAction::reject: reject(); break;
        case WidgetAction::refine: refine(); break;
        case WidgetAction::diff: toggle_diff(); break;
        case WidgetAction::ask_in_chat: ask_in_chat(); break;
        case WidgetAction::cancel: cancel(); break;
        case WidgetAction::retry: retry(); break;
        case WidgetAction::rephrase: rephrase(); break;
        case WidgetAction::model:
        case WidgetAction::setup: open_model(); break;
        case WidgetAction::close: close(); break;
    }
}

void WidgetSurface::update() {
    if (session_.ready()) {
        for (const std::string& id : pending_discards_) session_.run("preview.discard", {{"id", id}});
        pending_discards_.clear();
        if (!pending_turn_.empty() && state_ == WidgetState::running) {
            const std::string t = pending_turn_;
            pending_turn_.clear();
            begin_turn(t);
        }
        if (pending_commit_) accept();
    }
    WidgetAgentEvent e;
    while (agent_.poll(&e)) on_agent(e);
    // The open spec changed under the preview (the side chat, the person):
    // the preview is the open spec with its edits applied again, or a
    // conflict when one no longer applies.
    if ((state_ == WidgetState::preview || refining()) && !preview_.empty() && session_.ready()) {
        const auto open = session_.open_spec();
        if (open && open->text != seen_open_text_) {
            if (read_preview()) show_preview();
            else enter(error_.code == "history.conflict" ? WidgetState::conflict : WidgetState::failed);
        }
    }
}

std::string WidgetSurface::tag() const {
    switch (state_) {
        case WidgetState::declined: return "DECLINED";
        case WidgetState::handoff: return handoff_ ? handoff_tag(handoff_->kind) : "OUTSIDE SCOPE";
        case WidgetState::conflict: return "CONFLICT";
        case WidgetState::failed: return error_.code == "acp.agent_exited" ? "STOPPED" : "FAILED";
        case WidgetState::preview: return limit_ ? "LIMIT" : (edits_ <= 0 && reply_.empty() ? "NO CHANGE" : "");
        default: return "";
    }
}

std::vector<WidgetAction> WidgetSurface::card_actions() const {
    using A = WidgetAction;
    switch (state_) {
        case WidgetState::preview:
            if (edits_ > 0) return {A::accept, A::reject, A::refine, A::diff};
            return {A::refine, A::close};
        case WidgetState::declined: return {A::rephrase, A::retry, A::model};
        case WidgetState::handoff: return {A::ask_in_chat, A::rephrase, A::close};
        case WidgetState::conflict: return {A::refine, A::reject, A::diff};
        case WidgetState::failed: return {A::retry, A::close};
        default: return {};
    }
}

std::vector<WidgetAction> WidgetSurface::row_actions() const {
    using A = WidgetAction;
    switch (state_) {
        case WidgetState::prompt: return {A::ask_in_chat};
        case WidgetState::unavailable: return {A::setup, A::ask_in_chat};
        case WidgetState::running: return {A::cancel};
        case WidgetState::preview:
        case WidgetState::declined:
        case WidgetState::conflict:
        case WidgetState::failed: return {A::ask_in_chat};
        default: return {};
    }
}

// ---- what the page owns -----------------------------------------------------------------

InlineWidget::InlineWidget(Options o)
    : session_(std::make_unique<SessionWidgetSession>(*o.session, std::move(o.busy))),
      agent_(std::move(o.agent)),
      surface_(std::make_unique<WidgetSurface>(*session_, *agent_, std::move(o.hooks))) {}

InlineWidget::~InlineWidget() {
    surface_.reset();   // discards an open preview and closes the session first
    agent_.reset();
}

// ---- the layout -------------------------------------------------------------------------

WidgetLayoutMetrics WidgetLayoutMetrics::for_text_scale(float s) {
    if (!(s > 0.0f)) s = 1.0f;
    WidgetLayoutMetrics m;
    auto r = [s](float v) { return std::round(v * s); };
    m.row = r(30);
    m.line = r(20);
    m.pad = r(8);
    m.gap = r(6);
    m.button = r(22);
    m.button_pad = r(8);
    m.key_gap = r(6);
    m.mark = r(28);
    m.live = r(16);
    m.min_input = r(96);
    m.min_content = r(48);
    m.min_width = r(240);
    return m;
}

namespace {

float button_width(WidgetAction a, bool keys, const WidgetLayoutMetrics& m, const WidgetMeasure& measure) {
    float w = 2.0f * m.button_pad + measure(widget_action_label(a), WidgetFont::label);
    const char* key = widget_action_key(a);
    if (keys && key[0]) w += measure(key, WidgetFont::key) + m.key_gap;
    return std::ceil(w);
}

// Lays buttons out from x to the right; returns their total width.
float row_of_buttons(const std::vector<WidgetAction>& actions, bool keys, const WidgetLayoutMetrics& m,
                     const WidgetMeasure& measure) {
    float w = 0;
    for (size_t i = 0; i < actions.size(); ++i) w += button_width(actions[i], keys, m, measure) + (i ? m.gap : 0.0f);
    return w;
}

}  // namespace

WidgetLayout layout_widget_surface(const WidgetLayoutInput& in, const WidgetLayoutMetrics& m,
                                   const WidgetMeasure& measure) {
    WidgetLayout L;
    const float x0 = in.panel.x + m.pad;
    const float w = std::max(0.0f, in.panel.w - 2.0f * m.pad);
    const float y0 = in.panel.y + in.strip;
    const float avail = std::max(0.0f, in.panel.h - in.strip);

    // The buttons' keys go when either row's buttons do not fit.
    const float row_room = w - m.mark - m.min_input - m.gap;
    L.keys = row_of_buttons(in.card_actions, true, m, measure) <= w &&
             row_of_buttons(in.row_actions, true, m, measure) <= row_room;

    // The prompt row: the mark, the input, the status, the row's buttons.
    L.prompt = {x0, y0 + m.gap, w, m.row};
    L.mark = {x0, L.prompt.y, std::min(m.mark, w), m.row};
    float right = L.prompt.right() - m.gap;
    const float by = L.prompt.y + std::round((m.row - m.button) * 0.5f);
    std::vector<WidgetRect> rb(in.row_actions.size());
    for (size_t i = in.row_actions.size(); i-- > 0;) {
        const float bw = button_width(in.row_actions[i], L.keys, m, measure);
        rb[i] = {right - bw, by, bw, m.button};
        right -= bw + m.gap;
    }
    const float input_x = L.mark.right();
    float input_w = right - input_x;
    if (!in.status.empty()) {
        // While the agent works the progress has the live diamond before it.
        const float mark = in.state == WidgetState::running ? m.live : 0.0f;
        const float sw = std::ceil(measure(in.status, WidgetFont::mono) + mark) + m.gap;
        if (input_w - sw >= m.min_input) {
            L.status = {right - sw + m.gap, L.prompt.y, sw - m.gap, m.row};
            input_w -= sw;
        }
    }
    if (input_w < m.min_input) {
        // Too narrow for the row's buttons too: they go before the input does.
        input_w = std::max(0.0f, L.prompt.right() - m.gap - input_x);
        rb.clear();
    }
    L.row_buttons = rb;
    L.input = {input_x, L.prompt.y, std::max(0.0f, input_w), m.row};

    // The card under the row: tag, reply, diff, actions.
    const bool any_card = !in.tag.empty() || in.reply || in.diff_lines > 0 || (in.full_diff && in.full_diff_lines > 0) ||
                          !in.card_actions.empty();
    bool reply = in.reply;
    int diff_lines = in.full_diff ? std::min(in.full_diff_lines, m.max_full_lines)
                                  : std::min(in.diff_lines, m.max_diff_lines);
    auto card_height = [&](bool with_reply, int lines, bool folded) {
        if (!any_card) return 0.0f;
        float h = m.gap;
        if (!in.tag.empty()) h += m.line;
        if (with_reply) h += m.line * (float)std::max(1, in.reply_lines);
        h += m.line * (float)(lines + (folded ? 1 : 0));
        if (!in.card_actions.empty()) h += m.gap + m.button;
        return h + m.gap;
    };
    auto folded_for = [&](int lines) { return !in.full_diff && in.diff_lines > lines; };
    auto band_for = [&](bool with_reply, int lines) {
        const float card = card_height(with_reply, lines, folded_for(lines));
        return m.gap + m.row + (card > 0 ? m.gap + card : 0.0f) + m.gap;
    };
    // What collapses first as the panel shrinks: the reply, then the diff
    // lines past the first.
    if (band_for(reply, diff_lines) + m.min_content > avail) reply = false;
    while (diff_lines > 1 && band_for(reply, diff_lines) + m.min_content > avail) --diff_lines;
    float band = band_for(reply, diff_lines);
    if (band + m.min_content > avail) {
        // Below the minimum: the band takes the panel, the content is hidden.
        L.content_hidden = true;
        band = avail;
    }
    L.band = std::min(band, avail);

    if (any_card) {
        float y = L.prompt.bottom() + m.gap;
        L.card = {x0, y, w, card_height(reply, diff_lines, folded_for(diff_lines))};
        y += m.gap;
        if (!in.tag.empty()) {
            L.tag = {x0 + m.pad, y, std::max(0.0f, w - 2.0f * m.pad), m.line};
            y += m.line;
        }
        if (reply) {
            const float rh = m.line * (float)std::max(1, in.reply_lines);
            L.reply = {x0 + m.pad, y, std::max(0.0f, w - 2.0f * m.pad), rh};
            y += rh;
        }
        L.diff_shown = in.full_diff ? 0 : diff_lines;
        L.diff_folded = folded_for(diff_lines);
        const int lines = diff_lines + (L.diff_folded ? 1 : 0);
        L.diff = {x0 + m.pad, y, std::max(0.0f, w - 2.0f * m.pad), m.line * (float)lines};
        y += L.diff.h;
        if (!in.card_actions.empty()) {
            y += m.gap;
            L.actions = {x0 + m.pad, y, std::max(0.0f, w - 2.0f * m.pad), m.button};
            float x = L.actions.x;
            for (WidgetAction a : in.card_actions) {
                float bw = button_width(a, L.keys, m, measure);
                // Diff sits at the row's right end.
                if (a == WidgetAction::diff) {
                    const float rx = L.actions.right() - bw;
                    L.buttons.push_back({std::max(x, rx), y, bw, m.button});
                } else {
                    L.buttons.push_back({x, y, bw, m.button});
                }
                x = L.buttons.back().right() + m.gap;
            }
            // Past the right edge the buttons share what is there.
            if (!L.buttons.empty() && L.buttons.back().right() > L.actions.right()) {
                const float each = (L.actions.w - m.gap * (float)(L.buttons.size() - 1)) / (float)L.buttons.size();
                float bx = L.actions.x;
                for (WidgetRect& b : L.buttons) {
                    b.x = std::round(bx);
                    b.w = std::max(0.0f, std::floor(each));
                    bx += each + m.gap;
                }
            }
        }
    }
    // Nothing leaves the band.
    const WidgetRect band_rect{in.panel.x, y0, in.panel.w, L.band};
    auto clip = [&](WidgetRect& r) {
        if (r.empty()) return;
        if (!band_rect.contains(r)) {
            const float b = std::min(r.bottom(), band_rect.bottom());
            if (r.y >= b) r = {};
            else r.h = b - r.y;
        }
    };
    clip(L.prompt);
    clip(L.mark);
    clip(L.input);
    clip(L.status);
    for (auto& r : L.row_buttons) clip(r);
    clip(L.card);
    clip(L.tag);
    clip(L.reply);
    clip(L.diff);
    clip(L.actions);
    for (auto& r : L.buttons) clip(r);
    return L;
}

WidgetLayoutInput widget_layout_input(const WidgetSurface& s, const WidgetRect& panel, float strip, int reply_lines) {
    WidgetLayoutInput in;
    in.panel = panel;
    in.strip = strip;
    in.reply_lines = reply_lines;
    in.state = s.state();
    in.refining = s.refining();
    in.status = s.state() == WidgetState::running && !s.progress().empty() ? s.progress() : s.model_label();
    if (s.state() == WidgetState::unavailable) in.status = "NO PROVIDER";
    in.tag = s.tag();
    const bool cardful = s.state() == WidgetState::preview || s.refining() || s.state() == WidgetState::conflict ||
                         s.state() == WidgetState::declined;
    in.reply = (s.state() == WidgetState::preview || s.refining()) ? !s.reply().empty()
               : s.state() == WidgetState::handoff                ? s.handoff() && !s.handoff()->reason.empty()
               : s.state() == WidgetState::failed                 ? true
                                                                  : false;
    in.diff_lines = cardful ? (int)s.diff().size() : 0;
    in.full_diff = cardful && s.showing_full_diff();
    if (in.full_diff) {
        // The full diff holds the whole reply too, wrapped, so the reply's
        // one elided line goes.
        int n = 0;
        for (char c : s.full_diff()) n += c == '\n';
        if (!s.reply().empty()) n += 1 + (int)(s.reply().size() / 90);
        in.full_diff_lines = std::max(1, n);
        in.reply = false;
    }
    in.card_actions = s.card_actions();
    in.row_actions = s.row_actions();
    return in;
}

// ---- the ImGui side ---------------------------------------------------------------------

void WidgetSurface::attach(SpecRuntime& runtime) {
    if (runtime_) detach();
    runtime_ = &runtime;
    runtime_sink_ = std::make_unique<Sink>(runtime);
    sink_ = runtime_sink_.get();
    runtime.set_reserve([this](const SpecPanelFrame& f) { return reserve(f); });
    runtime.set_overlay([this](const SpecPanelFrame& f) { overlay(f); });
}

void WidgetSurface::detach() {
    if (!runtime_) return;
    if (!shown_panel_.empty()) runtime_->clear_preview(shown_panel_);
    shown_panel_.clear();
    runtime_->set_reserve({});
    runtime_->set_overlay({});
    runtime_ = nullptr;
    sink_ = nullptr;
    runtime_sink_.reset();
}

namespace {

// The card's text helpers (diff_card.h).
ImVec2 text_extent(ImFont* f, float size, const std::string& s) { return card_text_extent(f, size, s); }

std::string fit(ImFont* f, float size, const std::string& s, float w) { return card_fit_text(f, size, s, w); }

void draw_text(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 c, const std::string& s) {
    card_draw_text(dl, f, size, p, c, s);
}

ImVec2 v2(float x, float y) { return ImVec2(x, y); }

int resize_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* s = static_cast<std::string*>(data->UserData);
        s->resize((size_t)data->BufTextLen);
        data->Buf = const_cast<char*>(s->c_str());
    }
    return 0;
}

}  // namespace

namespace {

WidgetMeasure look_measure(const SpecLook& L) {
    return [&L](const std::string& s, WidgetFont font) {
        switch (font) {
            case WidgetFont::label: return text_extent(L.ui, L.ui_size, s).x;
            case WidgetFont::key:
            case WidgetFont::mono: return text_extent(L.mono, L.mono_size, s).x;
            case WidgetFont::tag:
                return sternbild::text_tracked_width(L.ui, L.small_size, s.c_str(), std::round(L.small_size * 0.08f));
        }
        return 0.0f;
    };
}

}  // namespace

bool WidgetSurface::open_at(const std::string& anchor) const {
    if (!look_ || state_ == WidgetState::idle || state_ == WidgetState::applied) return false;
    return target_.anchor == anchor;
}

std::string WidgetSurface::reply_body() const {
    if (state_ == WidgetState::handoff && handoff_) return handoff_->reason;
    if (state_ == WidgetState::failed) return error_.message;
    return reply_;
}

int WidgetSurface::reply_lines(float card_width) const {
    // An answer without a change is the card's content: its reply wraps over
    // up to six lines. Otherwise the reply is one elided line.
    if (!look_ || state_ != WidgetState::preview || edits_ > 0 || reply_.empty()) return 1;
    const SpecLook& L = *look_;
    const float w = std::max(40.0f, card_width);
    ImFont* f = L.ui ? L.ui : ImGui::GetFont();
    const float h = f->CalcTextSizeA(L.small_size, FLT_MAX, w, reply_.c_str(), reply_.c_str() + reply_.size()).y;
    const float lh = f->CalcTextSizeA(L.small_size, FLT_MAX, 0.0f, "Ag").y;
    return std::max(1, std::min(6, (int)std::ceil(h / std::max(1.0f, lh) - 0.01f)));
}

WidgetLayout WidgetSurface::layout_in(const WidgetRect& panel, float strip, bool row) const {
    const SpecLook& L = *look_;
    WidgetLayoutMetrics m = WidgetLayoutMetrics::for_text_scale(L.text_scale);
    if (row) m.min_content = 0.0f;   // under a row nothing else shares the space
    const float card_w = panel.w - 4.0f * m.pad;
    return layout_widget_surface(widget_layout_input(*this, panel, strip, reply_lines(card_w)), m, look_measure(L));
}

float WidgetSurface::reserve(const SpecPanelFrame& f) const {
    if (!look_ || state_ == WidgetState::idle || state_ == WidgetState::applied) return 0.0f;
    const std::string fp = target_.frame_panel();
    if (fp.empty() || f.id != fp) return 0.0f;
    const WidgetRect panel{f.rect.Min.x, f.rect.Min.y, f.rect.GetWidth(), f.rect.GetHeight()};
    return layout_in(panel, f.header.GetHeight(), false).band;
}

float WidgetSurface::row_height(const std::string& anchor, float width) const {
    if (!open_at(anchor) || anchor.empty()) return 0.0f;
    return layout_in(WidgetRect{0.0f, 0.0f, width, 100000.0f}, 0.0f, true).band;
}

void WidgetSurface::draw_row(const std::string& anchor, const ImRect& rect) {
    if (!open_at(anchor) || anchor.empty() || rect.GetHeight() < 1.0f) return;
    draw_block(WidgetRect{rect.Min.x, rect.Min.y, rect.GetWidth(), rect.GetHeight()}, 0.0f, rect, true);
}

void WidgetSurface::frame(const SpecLook& look, const std::string& focused, bool keys) {
    if (!look_) look_ = std::make_unique<SpecLook>();
    *look_ = look;
    if (!focused.empty()) focused_ = focused;
    if (ImGui::GetCurrentContext()) {
        // ⌘K on macOS (ImGui maps Cmd to its Ctrl there), Ctrl+K elsewhere,
        // and Escape, unless the page routes them itself (keys false).
        if (keys && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_K) && !focused_.empty())
            open(WidgetTarget::for_panel(focused_));
        else if (keys && state_ != WidgetState::idle && ImGui::IsKeyPressed(ImGuiKey_Escape, false))
            escape();
        else if ((state_ == WidgetState::preview || state_ == WidgetState::handoff) && !ImGui::GetIO().WantTextInput &&
                 (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)))
            act(card_actions().front());
    }
    update();
}

void WidgetSurface::overlay(const SpecPanelFrame& f) {
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImGui::IsMouseHoveringRect(f.rect.Min, f.rect.Max, false))
        focused_ = f.id;
    if (!look_ || state_ == WidgetState::idle || state_ == WidgetState::applied) return;
    const std::string fp = target_.frame_panel();
    if (fp.empty() || f.id != fp || f.band.GetHeight() < 1.0f) return;
    draw_block(WidgetRect{f.rect.Min.x, f.rect.Min.y, f.rect.GetWidth(), f.rect.GetHeight()}, f.header.GetHeight(),
               f.band, false);
}

void WidgetSurface::draw_block(const WidgetRect& panel, float strip, const ImRect& band, bool at_row) {
    const SpecLook& L = *look_;
    const float s = L.text_scale;
    const WidgetLayoutMetrics metrics = WidgetLayoutMetrics::for_text_scale(s);
    WidgetLayoutInput in = widget_layout_input(*this, panel, strip, reply_lines(panel.w - 4.0f * metrics.pad));
    const WidgetLayout lay = layout_in(panel, strip, at_row);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::PushClipRect(band.Min, band.Max, true);
    ImGui::PushID("##widget.surface");

    // The band: the panel's ground, closed by a rule over the content.
    dl->AddRectFilled(band.Min, band.Max, L.ground);
    if (!lay.content_hidden || at_row) dl->AddRectFilled(v2(band.Min.x, band.Max.y - 1.0f), band.Max, L.rule_faint);

    auto rr = [](const WidgetRect& r) { return ImRect(v2(r.x, r.y), v2(r.right(), r.bottom())); };
    const float ui_h = text_extent(L.ui, L.ui_size, "Ag").y;
    const float mono_h = text_extent(L.mono, L.mono_size, "Ag").y;
    const float small_h = text_extent(L.ui, L.small_size, "Ag").y;

    // A button in the house's key-button look; Accept is the primary plate.
    auto button = [&](const WidgetRect& r, WidgetAction a) {
        if (r.empty()) return;
        const ImRect b = rr(r);
        ImGui::SetCursorScreenPos(b.Min);
        const bool clicked = ImGui::InvisibleButton(widget_action_label(a), b.GetSize());
        const bool hovered = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
        const bool primary = a == WidgetAction::accept || (a == WidgetAction::ask_in_chat && state_ == WidgetState::handoff);
        ImU32 outline = primary ? L.text : L.rule, fill = 0, key_ink = primary ? L.accent : L.secondary, ink = L.text;
        if (held && hovered) {
            fill = L.text;
            outline = fill;
            key_ink = ink = L.deep;
        } else if (hovered) {
            fill = L.plate_hover;
            key_ink = L.accent;
        }
        if (a == WidgetAction::diff && show_full_diff_) outline = L.accent;
        if (primary) sternbild::cut_plate(dl, b.Min, b.Max, outline, fill, std::round(4.0f * s));
        else {
            if (fill) dl->AddRectFilled(b.Min, b.Max, fill);
            dl->AddRect(b.Min, b.Max, outline, 0.0f, 0, 1.0f);
        }
        float x = b.Min.x + std::round(8.0f * s);
        const char* key = widget_action_key(a);
        if (lay.keys && key[0]) {
            draw_text(dl, L.mono, L.mono_size, v2(x, b.Min.y + (b.GetHeight() - mono_h) * 0.5f), key_ink, key);
            x += text_extent(L.mono, L.mono_size, key).x + std::round(6.0f * s);
        }
        const std::string label = fit(L.ui, L.ui_size, widget_action_label(a), b.Max.x - x - std::round(4.0f * s));
        draw_text(dl, L.ui, L.ui_size, v2(x, b.Min.y + (b.GetHeight() - ui_h) * 0.5f), ink, label);
        if (clicked) act(a);
    };

    // The prompt row: a deep well with a 1 px outline, bronze while typing.
    const bool typing = state_ == WidgetState::prompt || state_ == WidgetState::unavailable;
    const ImRect row = rr(lay.prompt);
    dl->AddRectFilled(row.Min, row.Max, L.deep);
    dl->AddRect(row.Min, row.Max, typing ? L.accent : L.rule, 0.0f, 0, 1.0f);
    if (!lay.mark.empty()) {
        const std::string mark = "\xE2\x8C\x98K";   // ⌘K
        const ImVec2 ms = text_extent(L.mono, L.mono_size, mark);
        draw_text(dl, L.mono, L.mono_size, v2(lay.mark.x + (lay.mark.w - ms.x) * 0.5f, row.Min.y + (row.GetHeight() - mono_h) * 0.5f),
             typing ? L.accent : L.faint, mark);
    }
    const ImRect input = rr(lay.input);
    if (typing && !lay.input.empty()) {
        ImGui::SetCursorScreenPos(v2(input.Min.x, std::round(input.Min.y + (input.GetHeight() - ui_h) * 0.5f) - 1.0f));
        ImGui::PushItemWidth(input.GetWidth());
        ImGui::PushStyleColor(ImGuiCol_FrameBg, 0);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, 0);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, 0);
        ImGui::PushStyleColor(ImGuiCol_Text, L.text);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, sternbild::with_alpha(L.accent, 90));
        ImGui::PushStyleColor(ImGuiCol_NavCursor, 0);   // the row's own outline marks the focus
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, v2(0, 1));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
        ImGui::PushFont(L.ui, L.ui_size);
        if (focus_input_) {
            ImGui::SetKeyboardFocusHere();
            focus_input_ = false;
        }
        if (text_.capacity() < 64) text_.reserve(64);
        const bool entered = ImGui::InputText("##widget.prompt", const_cast<char*>(text_.c_str()), text_.capacity() + 1,
                                              ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackResize,
                                              resize_callback, &text_);
        ImGui::PopFont();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(6);
        ImGui::PopItemWidth();
        if (entered) {
            submit();
            focus_input_ = state_ == WidgetState::prompt;
        }
    } else if (!lay.input.empty()) {
        // The request as it was sent.
        const std::string shown = fit(L.ui, L.ui_size, request_, input.GetWidth());
        draw_text(dl, L.ui, L.ui_size, v2(input.Min.x, input.Min.y + (input.GetHeight() - ui_h) * 0.5f), L.secondary, shown);
    }
    if (!lay.status.empty()) {
        const ImRect st = rr(lay.status);
        const bool running = state_ == WidgetState::running;
        float x = st.Min.x;
        if (running) {
            // The live diamond while the agent works.
            const float half = std::round(3.5f * s);
            sternbild::diamond(dl, v2(x + half, st.GetCenter().y), half, L.accent, true);
            x += 2.0f * half + std::round(6.0f * s);
        }
        const std::string label = fit(L.mono, L.mono_size, in.status, st.Max.x - x);
        // The model is the one place to choose it: a click lists the models
        // the session offers, when it offers a choice.
        const std::vector<WidgetModelChoice> choices = running ? std::vector<WidgetModelChoice>() : agent_.models();
        bool hovered = false;
        if (choices.size() > 1) {
            ImGui::SetCursorScreenPos(st.Min);
            if (ImGui::InvisibleButton("##widget.model", st.GetSize())) ImGui::OpenPopup("##widget.models");
            hovered = ImGui::IsItemHovered();
        }
        draw_text(dl, L.mono, L.mono_size, v2(x, st.Min.y + (st.GetHeight() - mono_h) * 0.5f),
                  state_ == WidgetState::unavailable ? L.secondary : hovered ? L.text : L.faint, label);
        if (hovered) {
            const float tw = text_extent(L.mono, L.mono_size, label).x;
            const float uy = std::round(st.Min.y + (st.GetHeight() + mono_h) * 0.5f);
            dl->AddRectFilled(v2(x, uy), v2(x + tw, uy + 1.0f), L.text);
        }
        if (ImGui::BeginPopup("##widget.models")) {
            for (const WidgetModelChoice& c : choices)
                if (ImGui::Selectable(c.name.c_str(), c.current) && !c.current) agent_.set_model(c.value);
            ImGui::EndPopup();
        }
    }
    for (size_t i = 0; i < lay.row_buttons.size() && i < in.row_actions.size(); ++i) button(lay.row_buttons[i], in.row_actions[i]);

    // The card: a deep well under the row.
    if (!lay.card.empty()) {
        const ImRect card = rr(lay.card);
        dl->AddRectFilled(card.Min, card.Max, L.deep);
        if (!lay.tag.empty()) {
            const ImRect t = rr(lay.tag);
            const float tracking = std::round(L.small_size * 0.08f);
            const std::string tg = tag();
            const float ty = t.Min.y + (t.GetHeight() - small_h) * 0.5f;
            sternbild::text_tracked(dl, L.ui, L.small_size, v2(t.Min.x, std::round(ty)), L.secondary, tg.c_str(), tracking);
            const float tw = sternbild::text_tracked_width(L.ui, L.small_size, tg.c_str(), tracking);
            std::string detail;
            if (state_ == WidgetState::declined) detail = model_label();
            else if (state_ == WidgetState::failed || state_ == WidgetState::conflict) detail = error_.code;
            else if (state_ == WidgetState::preview && limit_) detail = std::to_string(edits_) + (edits_ == 1 ? " edit" : " edits");
            if (!detail.empty()) {
                const float dx = t.Min.x + tw + std::round(12.0f * s);
                const std::string d = fit(L.mono, L.mono_size, detail, t.Max.x - dx);
                draw_text(dl, L.mono, L.mono_size, v2(dx, t.Min.y + (t.GetHeight() - mono_h) * 0.5f), L.faint, d);
            }
        }
        if (!lay.reply.empty()) {
            const ImRect r = rr(lay.reply);
            const std::string body = reply_body();
            if (in.reply_lines > 1) {
                // An answer: wrapped over its lines, clipped to them.
                const float lh = r.GetHeight() / (float)in.reply_lines;
                ImGui::PushClipRect(r.Min, r.Max, true);
                ImFont* f = L.ui ? L.ui : ImGui::GetFont();
                dl->AddText(f, L.small_size, v2(r.Min.x, std::round(r.Min.y + (lh - small_h) * 0.5f)), L.secondary,
                            body.c_str(), body.c_str() + body.size(), r.GetWidth());
                ImGui::PopClipRect();
            } else {
                draw_text(dl, L.ui, L.small_size, v2(r.Min.x, r.Min.y + (r.GetHeight() - small_h) * 0.5f), L.secondary,
                          fit(L.ui, L.small_size, body, r.GetWidth()));
            }
        }
        if (!lay.diff.empty()) {
            const ImRect d = rr(lay.diff);
            const float lh = lay.diff_shown > 0 || lay.diff_folded ? d.GetHeight() / (float)(lay.diff_shown + (lay.diff_folded ? 1 : 0))
                                                                   : std::round(20.0f * s);
            if (show_full_diff_) {
                // The full diff of the canonical TOML, scrolling past what fits.
                ImGui::SetCursorScreenPos(d.Min);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, 0);
                if (ImGui::BeginChild("##widget.diff", d.GetSize(), ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                    if (!reply_.empty()) {
                        // The agent's reply in full, which the card's one line elides.
                        ImGui::PushFont(L.ui, L.small_size);
                        ImGui::PushStyleColor(ImGuiCol_Text, L.secondary);
                        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + d.GetWidth() - std::round(16.0f * s));
                        ImGui::TextUnformatted(reply_.c_str(), reply_.c_str() + reply_.size());
                        ImGui::PopTextWrapPos();
                        ImGui::PopStyleColor();
                        ImGui::PopFont();
                        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
                    }
                    draw_unified_diff(L, full_diff_, lh);
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();
            } else {
                // The compact diff (diff_card.h).
                draw_compact_diff(dl, L, d, diff_, lay.diff_shown, lay.diff_folded);
            }
        }
        for (size_t i = 0; i < lay.buttons.size() && i < in.card_actions.size(); ++i) button(lay.buttons[i], in.card_actions[i]);
    }

    ImGui::PopID();
    ImGui::PopClipRect();
}

}  // namespace caliper_host
