#pragma once
// ============================================================================
// widget_agent.h: the agent side of the widget agent's inline surface
// (design/widget-agent.md, widget_surface.h). The surface talks to its agent
// through WidgetAgent: start a session for one target and one stacked
// preview, prompt it, cancel a turn, close the session, and poll the events
// that come back. Two implementations:
//   - ScriptedWidgetAgent (here), for tests and screenshots: it plays a
//     script of turns, each a few timed progress steps, the edits it sends to
//     the session with the preview's id as the sidecar does, and the turn's
//     answer (a reply, a refusal, a handoff or a limit). It needs no model,
//     no key and no process.
//   - AcpWidgetAgent (widget_agent_acp.h), over AcpClient and the
//     caliper-agent sidecar.
//
// The contract with the sidecar (design/widget-agent.md, How it stays an ACP
// agent): session/new carries _meta.caliper.widget = {target, preview,
// scope?}; each prompt's answer carries _meta.caliper.widget = {preview,
// edits, handoff: null | {kind, reason}, limit}; a refusal ends the turn with
// the stop reason `refusal`. The sidecar never commits: the surface does.
//
// Threads: every call and poll() run on the frame thread. An implementation
// that blocks (a process start) does it on a thread of its own and reports
// through poll().
//
// Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>
#include <caliper_builder/session/context_bundle.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace caliper_host {

// What a widget prompt acts on: a panel, a selection in a panel, a dropped
// file's loader and its catalog ids, or one property of a panel or a source.
struct WidgetTarget {
    enum class Kind { panel, selection, dropped_file, property };
    Kind kind = Kind::panel;
    std::string panel;                  // panel, selection, property of a panel
    std::string source;                 // property of a source
    std::string path;                   // property: such as options.colormap
    std::string loader;                 // dropped_file
    std::vector<std::string> catalog;   // dropped_file: the loader's outputs

    static WidgetTarget for_panel(std::string id);
    // The panel whose frame the surface draws in, or "" (a dropped file, a
    // property of a source).
    const std::string& frame_panel() const { return panel; }
    // The same target as the session library's context bundle takes it.
    caliper_builder::WidgetTarget to_builder() const;
    // {"kind", ...} as session/new's _meta.caliper.widget.target takes it
    // (caliper_builder::widget_target_to_json()).
    nlohmann::json to_json() const;
    // The scope of preview.begin: {"panel": id}, {"source": id} or {}.
    nlohmann::json preview_scope() const;
};

const char* widget_target_kind_name(WidgetTarget::Kind kind);

// Why the agent sent a request to the side chat (the sidecar's handoff tool).
struct WidgetHandoff {
    std::string kind;     // build_from_scratch, loader, project_code, explanation, another_target, outside_scope
    std::string reason;   // the model's short reason
};

struct WidgetAgentEvent {
    enum class Kind {
        started,    // the session is open: model and effort set
        progress,   // a step of the turn: text is a short noun phrase or a command name
        thought,    // a chunk of the agent's summarized thinking
        message,    // a chunk of the agent's reply
        finished,   // the turn ended: stop_reason, edits, handoff, limit
        failed,     // the session could not start or the turn failed: error
        exited,     // the agent's process ended: error
    };
    Kind kind = Kind::progress;
    std::string text;
    std::string model, effort;            // started
    std::string stop_reason;              // finished: end_turn, refusal, max_turn_requests, cancelled
    int edits = -1;                       // finished: the preview's edits after the turn; -1 unknown
    std::optional<WidgetHandoff> handoff; // finished
    bool limit = false;                   // finished: a budget limit ended the turn
    caliper_builder::Error error;         // failed, exited
};

const char* widget_agent_event_name(WidgetAgentEvent::Kind kind);

class WidgetAgent {
public:
    virtual ~WidgetAgent() = default;
    // Whether a provider is set up; when not, `why` names what is missing and
    // the surface offers only Ask in chat.
    virtual bool available(std::string* why = nullptr) const = 0;
    // Opens a session for the target on the preview the surface began, with
    // the tool scope of the context bundle (empty: the sidecar derives it). A
    // session already open is closed first. The answer arrives as started or
    // failed.
    virtual void start(const WidgetTarget& target, const std::string& preview,
                       const std::vector<std::string>& scope = {}) = 0;
    // One prompt in the open session (queued until it has started), with the
    // context bundle's ACP content blocks before the request on a session's
    // first prompt (null or empty: none). The turn reports progress, thoughts
    // and messages, then finished.
    virtual void prompt(const std::string& text, const nlohmann::json& context = nlohmann::json()) = 0;
    // Cancels the running turn, which then finishes with `cancelled`.
    virtual void cancel() = 0;
    // Ends the session (session/close); no event follows.
    virtual void close() = 0;
    // Takes the next event, on the frame thread.
    virtual bool poll(WidgetAgentEvent* out) = 0;
    // The model and effort shown in the prompt row, once known.
    virtual std::string model_label() const { return {}; }
};

// ---- the scripted agent --------------------------------------------------------------

// One step of a scripted turn, at `at` after the prompt.
struct WidgetScriptStep {
    std::chrono::milliseconds at{0};
    enum class Kind { progress, thought, message, edit } kind = Kind::progress;
    std::string text;                 // progress, thought, message
    caliper_builder::CommandInvocation edit;   // edit: name and args; origin and preview are set when sent
};

// One scripted turn: its steps in time order, then its end at `end_at`.
struct WidgetScriptTurn {
    std::vector<WidgetScriptStep> steps;
    std::chrono::milliseconds end_at{0};
    std::string stop_reason = "end_turn";
    std::optional<WidgetHandoff> handoff;
    bool limit = false;
    bool exit = false;                 // the agent's process ends instead of the turn
};

struct WidgetScript {
    // The name the prompt row shows. A scripted agent runs no model, so it
    // never borrows a real model's name.
    std::string model = "Scripted";
    std::string effort;
    std::chrono::milliseconds start_after{0};   // from start() to started
    std::vector<WidgetScriptTurn> turns;        // one per prompt, in order; past the last, a turn that only replies
    bool available = true;                      // a provider is set up
};

// Plays a WidgetScript. Edits go to `send` with the origin `agent` and the
// preview's id, as the sidecar sends them over Caliper API with
// X-Caliper-Preview; the result's failure becomes a progress line of its
// code, and the turn's edits count is what `count_edits` says after it (the
// session's preview), or the number of edits sent that succeeded. Time comes
// from `now`, which tests and the screenshot tool advance by hand.
class ScriptedWidgetAgent : public WidgetAgent {
public:
    using Clock = std::function<std::chrono::steady_clock::time_point()>;
    using Send = std::function<caliper_builder::CommandResult(caliper_builder::CommandInvocation)>;
    using CountEdits = std::function<int(const std::string& preview)>;

    ScriptedWidgetAgent(WidgetScript script, Send send, Clock now = {}, CountEdits count_edits = {});

    bool available(std::string* why) const override;
    void start(const WidgetTarget& target, const std::string& preview, const std::vector<std::string>& scope) override;
    void prompt(const std::string& text, const nlohmann::json& context) override;
    void cancel() override;
    void close() override;
    bool poll(WidgetAgentEvent* out) override;
    std::string model_label() const override;

    // What the surface asked, for the tests.
    const std::vector<std::string>& prompts() const { return prompts_; }
    const std::vector<nlohmann::json>& contexts() const { return contexts_; }   // one per prompt
    const std::vector<std::vector<std::string>>& scopes() const { return scopes_; }   // one per session
    const std::vector<std::string>& previews() const { return previews_; }
    int sessions_started() const { return sessions_; }
    int sessions_closed() const { return closed_; }
    int cancels() const { return cancels_; }
    bool turn_active() const { return turn_ >= 0; }

private:
    void advance();
    std::chrono::steady_clock::time_point now() const;

    WidgetScript script_;
    Send send_;
    Clock now_;
    CountEdits count_edits_;
    std::deque<WidgetAgentEvent> events_;
    std::string preview_;
    bool open_ = false, started_ = false, gone_ = false;
    std::chrono::steady_clock::time_point started_at_{}, turn_at_{};
    int turn_ = -1;              // the playing turn's index, -1 when none
    size_t next_step_ = 0;
    int next_turn_ = 0;
    int sent_ok_ = 0;
    std::deque<std::string> queued_;
    std::vector<std::string> prompts_, previews_;
    std::vector<nlohmann::json> contexts_;
    std::vector<std::vector<std::string>> scopes_;
    int sessions_ = 0, closed_ = 0, cancels_ = 0;
};

}  // namespace caliper_host
