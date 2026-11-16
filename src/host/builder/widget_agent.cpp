// widget_agent.cpp: the widget target and the scripted widget agent
// (widget_agent.h).
#include "builder/widget_agent.h"

#include <utility>

namespace caliper_host {

using nlohmann::json;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandResult;

// ---- the target ------------------------------------------------------------------------

WidgetTarget WidgetTarget::for_panel(std::string id) {
    WidgetTarget t;
    t.kind = Kind::panel;
    t.panel = std::move(id);
    return t;
}

const char* widget_target_kind_name(WidgetTarget::Kind kind) {
    switch (kind) {
        case WidgetTarget::Kind::panel: return "panel";
        case WidgetTarget::Kind::selection: return "selection";
        case WidgetTarget::Kind::dropped_file: return "dropped_file";
        case WidgetTarget::Kind::property: return "property";
    }
    return "panel";
}

caliper_builder::WidgetTarget WidgetTarget::to_builder() const {
    caliper_builder::WidgetTarget t;
    switch (kind) {
        case Kind::panel: t.kind = caliper_builder::WidgetTargetKind::panel; break;
        case Kind::selection: t.kind = caliper_builder::WidgetTargetKind::selection; break;
        case Kind::dropped_file: t.kind = caliper_builder::WidgetTargetKind::dropped_file; break;
        case Kind::property: t.kind = caliper_builder::WidgetTargetKind::property; break;
    }
    t.panel = panel;
    t.source = source;
    t.loader = loader;
    t.catalog = catalog;
    t.path = path;
    return t;
}

json WidgetTarget::to_json() const { return caliper_builder::widget_target_to_json(to_builder()); }

json WidgetTarget::preview_scope() const {
    switch (kind) {
        case Kind::panel:
        case Kind::selection: return {{"panel", panel}};
        case Kind::property:
            if (!panel.empty()) return {{"panel", panel}};
            return {{"source", source}};
        case Kind::dropped_file: return json::object();
    }
    return json::object();
}

const char* widget_agent_event_name(WidgetAgentEvent::Kind kind) {
    switch (kind) {
        case WidgetAgentEvent::Kind::started: return "started";
        case WidgetAgentEvent::Kind::progress: return "progress";
        case WidgetAgentEvent::Kind::thought: return "thought";
        case WidgetAgentEvent::Kind::message: return "message";
        case WidgetAgentEvent::Kind::finished: return "finished";
        case WidgetAgentEvent::Kind::failed: return "failed";
        case WidgetAgentEvent::Kind::exited: return "exited";
    }
    return "progress";
}

// ---- the scripted agent ----------------------------------------------------------------

ScriptedWidgetAgent::ScriptedWidgetAgent(WidgetScript script, Send send, Clock now, CountEdits count_edits)
    : script_(std::move(script)), send_(std::move(send)), now_(std::move(now)), count_edits_(std::move(count_edits)) {}

std::chrono::steady_clock::time_point ScriptedWidgetAgent::now() const {
    return now_ ? now_() : std::chrono::steady_clock::now();
}

bool ScriptedWidgetAgent::available(std::string* why) const {
    if (!script_.available && why) *why = "no provider";
    return script_.available;
}

std::string ScriptedWidgetAgent::model_label() const {
    if (script_.model.empty()) return {};
    return script_.effort.empty() ? script_.model : script_.model + " · " + script_.effort;
}

void ScriptedWidgetAgent::start(const WidgetTarget&, const std::string& preview, const std::vector<std::string>& scope) {
    scopes_.push_back(scope);
    if (open_) close();
    open_ = true;
    started_ = false;
    gone_ = false;
    preview_ = preview;
    started_at_ = now();
    turn_ = -1;
    sent_ok_ = 0;
    ++sessions_;
    previews_.push_back(preview);
}

void ScriptedWidgetAgent::prompt(const std::string& text, const json& context) {
    prompts_.push_back(text);
    contexts_.push_back(context);
    if (!open_) {
        WidgetAgentEvent e;
        e.kind = WidgetAgentEvent::Kind::failed;
        e.error = {"acp.no_session", "No widget session is open.", ""};
        events_.push_back(std::move(e));
        return;
    }
    queued_.push_back(text);
    advance();
}

void ScriptedWidgetAgent::cancel() {
    queued_.clear();
    if (turn_ < 0) return;
    ++cancels_;
    turn_ = -1;
    WidgetAgentEvent e;
    e.kind = WidgetAgentEvent::Kind::finished;
    e.stop_reason = "cancelled";
    e.edits = count_edits_ ? count_edits_(preview_) : sent_ok_;
    events_.push_back(std::move(e));
}

void ScriptedWidgetAgent::close() {
    if (!open_) return;
    ++closed_;
    open_ = false;
    started_ = false;
    turn_ = -1;
    queued_.clear();
    events_.clear();
}

void ScriptedWidgetAgent::advance() {
    if (!open_) return;
    const auto t = now();
    if (!started_) {
        if (t < started_at_ + script_.start_after) return;
        started_ = true;
        WidgetAgentEvent e;
        e.kind = WidgetAgentEvent::Kind::started;
        e.model = script_.model;
        e.effort = script_.effort;
        events_.push_back(std::move(e));
    }
    for (;;) {
        if (turn_ < 0) {
            if (queued_.empty()) return;
            queued_.pop_front();
            turn_ = next_turn_++;
            turn_at_ = t;
            next_step_ = 0;
        }
        static const WidgetScriptTurn kDefault = [] {
            WidgetScriptTurn d;
            WidgetScriptStep m;
            m.kind = WidgetScriptStep::Kind::message;
            m.text = "No change.";
            d.steps.push_back(m);
            return d;
        }();
        const WidgetScriptTurn& turn = (size_t)turn_ < script_.turns.size() ? script_.turns[(size_t)turn_] : kDefault;
        while (next_step_ < turn.steps.size() && t >= turn_at_ + turn.steps[next_step_].at) {
            const WidgetScriptStep& s = turn.steps[next_step_++];
            WidgetAgentEvent e;
            switch (s.kind) {
                case WidgetScriptStep::Kind::progress: e.kind = WidgetAgentEvent::Kind::progress; e.text = s.text; break;
                case WidgetScriptStep::Kind::thought: e.kind = WidgetAgentEvent::Kind::thought; e.text = s.text; break;
                case WidgetScriptStep::Kind::message: e.kind = WidgetAgentEvent::Kind::message; e.text = s.text; break;
                case WidgetScriptStep::Kind::edit: {
                    // As the sidecar sends an edit: the agent origin and the
                    // preview's id (X-Caliper-Preview), no dry run.
                    CommandInvocation inv = s.edit;
                    inv.origin = CommandOrigin::agent;
                    inv.preview = preview_;
                    e.kind = WidgetAgentEvent::Kind::progress;
                    e.text = inv.name;
                    if (send_) {
                        const CommandResult r = send_(std::move(inv));
                        if (r.ok) ++sent_ok_;
                        else e.text = r.error.code;
                    }
                    break;
                }
            }
            events_.push_back(std::move(e));
        }
        if (next_step_ < turn.steps.size() || t < turn_at_ + turn.end_at) return;
        WidgetAgentEvent e;
        if (turn.exit) {
            e.kind = WidgetAgentEvent::Kind::exited;
            e.error = {"acp.agent_exited", "The agent stopped during the turn.", "session/prompt"};
            events_.push_back(std::move(e));
            open_ = false;
            gone_ = true;
            turn_ = -1;
            queued_.clear();
            return;
        }
        e.kind = WidgetAgentEvent::Kind::finished;
        e.stop_reason = turn.stop_reason;
        e.handoff = turn.handoff;
        e.limit = turn.limit;
        e.edits = count_edits_ ? count_edits_(preview_) : sent_ok_;
        events_.push_back(std::move(e));
        turn_ = -1;
    }
}

bool ScriptedWidgetAgent::poll(WidgetAgentEvent* out) {
    advance();
    if (events_.empty()) return false;
    *out = std::move(events_.front());
    events_.pop_front();
    return true;
}

}  // namespace caliper_host
