#pragma once
// ============================================================================
// widget_surface.h: the widget agent's inline surface (design/widget-agent.md,
// The inline prompt and the preview; design/builder-overview.md, the
// placement sketch). ⌘K on a panel opens a one-line prompt inside that panel,
// at its top under the title strip. While the agent works, the row shows its
// progress and Escape cancels. When it is done, the panel draws the
// preview's version of itself in place, its title strip marked PREVIEW, and
// a compact diff under the prompt row lists the changed keys with their old
// and new values, with Accept, Reject and Refine; the full diff of the
// spec's canonical TOML is behind Diff. Nothing covers another pane: the
// prompt row and the diff take a band of the panel itself, and the panel's
// content is drawn below it (SpecRuntime::set_reserve()).
//
// Three parts, so that each is tested without a window:
//   - WidgetSurface, the state machine: idle, prompt, running, preview,
//     applied, declined, handoff, unavailable, conflict and failed. Input
//     events (open, the text, submit, accept, reject, refine, escape, cancel,
//     ask in chat, retry, rephrase) and the agent's events drive it. It owns
//     the preview's life on the command layer: preview.begin from the ui
//     origin scoped to the target when the first prompt is submitted,
//     preview.discard on Reject, Escape, cancel, a handoff or a failure, and
//     preview.commit from the ui origin on Accept, which the session files as
//     one undo step. Refine sends the follow-up in the same agent session on
//     the same preview. It reaches the session through WidgetSession and the
//     drawing through WidgetPreviewSink, so tests give it fakes or a real
//     Session.
//   - layout_widget_surface(): the prompt row, the diff and the action row as
//     plain rectangles inside the panel's rectangle, computed without ImGui
//     from text widths the caller measures.
//   - attach(), frame() and overlay(): the ImGui side on the frame thread, in
//     the house style.
//
// The page's part: own one WidgetSurface, call attach(runtime) once, then
// each frame call frame() before the runtime draws (it reads ⌘K, Ctrl+K on
// Windows, on the focused panel and Escape, and polls the agent) and
// overlay() from the runtime's overlay callback for every panel (attach()
// installs that callback itself; a page with an overlay of its own calls
// overlay() from it instead).
//
// Labels are nouns, as the UI principles ask: Accept, Reject, Refine, Diff,
// Ask in chat, Retry, Rephrase, Model, Setup, Cancel, Close; states are tags:
// PREVIEW, DECLINED, CONFLICT, LIMIT, STOPPED, NO PROVIDER. The surface writes
// no sentences of its own; the agent's one-sentence reply and its handoff
// reason are the agent's content.
//
// Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include "builder/widget_agent.h"

#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace caliper_host {

class SpecRuntime;
struct SpecPanelFrame;
struct SpecLook;

// ---- what the surface needs of the app ---------------------------------------------------

// The session, as the surface uses it on the frame thread: commands from the
// ui origin, the open spec, a preview's spec and a preview's panel data.
class WidgetSession {
public:
    virtual ~WidgetSession() = default;
    // False while the session must be left alone this frame (a long command
    // runs on the app's worker); what the surface would run waits.
    virtual bool ready() const { return true; }
    virtual caliper_builder::CommandResult run(const std::string& name, nlohmann::json args) = 0;
    virtual std::optional<caliper_builder::OpenSpec> open_spec() const = 0;
    virtual caliper_builder::PreviewState preview(const std::string& id) const = 0;
    virtual caliper_builder::PanelData resolve(const std::string& panel, const std::string& preview) const = 0;
    // The context bundle for a session's first prompt; an empty one by default.
    virtual caliper_builder::ContextBundle bundle(const WidgetTarget& target, const std::string& preview) {
        (void)target;
        (void)preview;
        return {};
    }
};

// A WidgetSession over a builder Session: run() is Session::invoke() from the
// ui origin; `busy`, when given, says when ready() is false
// (AppSession::worker_busy()).
class SessionWidgetSession : public WidgetSession {
public:
    explicit SessionWidgetSession(caliper_builder::Session& session, std::function<bool()> busy = {});
    bool ready() const override;
    caliper_builder::CommandResult run(const std::string& name, nlohmann::json args) override;
    std::optional<caliper_builder::OpenSpec> open_spec() const override;
    caliper_builder::PreviewState preview(const std::string& id) const override;
    caliper_builder::PanelData resolve(const std::string& panel, const std::string& preview) const override;
    // build_context_bundle() from the ui origin, read on the preview.
    caliper_builder::ContextBundle bundle(const WidgetTarget& target, const std::string& preview) override;
    caliper_builder::ContextBundleOptions bundle_options;   // the model's context length and vision

private:
    caliper_builder::Session& session_;
    std::function<bool()> busy_;
};

// Where the surface draws a preview in place: the spec runtime
// (SpecRuntime::set_preview()), or a fake in tests.
class WidgetPreviewSink {
public:
    virtual ~WidgetPreviewSink() = default;
    virtual void show(const caliper_builder::spec::Panel& panel, const caliper_builder::PanelData& data,
                      const caliper_builder::spec::Spec& spec) = 0;
    virtual void clear(const std::string& panel) = 0;
};

// ---- the compact diff -------------------------------------------------------------------

// One changed key between the open spec and the preview: a key of the target
// panel relative to it (options without their "options." prefix: y_log,
// title, bind.y), or of another item ("panel rows", "source metrics.catalog",
// "layout").
struct WidgetDiffLine {
    enum class Kind { changed, added, removed };
    Kind kind = Kind::changed;
    std::string key;
    std::string before;   // the old value, compact; empty when added
    std::string after;    // the new value, compact; empty when removed
};

// The changed keys of `panel` and of the rest of the spec, the panel's first.
// Values are compact JSON, cut at `max_value` characters.
std::vector<WidgetDiffLine> widget_compact_diff(const caliper_builder::spec::Spec& before,
                                                const caliper_builder::spec::Spec& after, const std::string& panel,
                                                size_t max_value = 40);

// ---- the state machine ------------------------------------------------------------------

enum class WidgetState { idle, prompt, running, preview, applied, declined, handoff, unavailable, conflict, failed };
const char* widget_state_name(WidgetState state);

// The actions a state offers, in their order on the surface.
enum class WidgetAction { accept, reject, refine, diff, ask_in_chat, cancel, retry, rephrase, model, setup, close };
const char* widget_action_label(WidgetAction action);   // the noun on the surface
const char* widget_action_key(WidgetAction action);     // its key ("⏎", "esc") or ""

// One request and its reply in a widget conversation.
struct WidgetExchange {
    std::string request;
    std::string reply;
};

// What "Ask in chat" hands to the side chat (design/widget-agent.md, Handoff
// to the side chat): the request, the target, the conversation so far and
// the preview's diff, marked as not applied.
struct WidgetChatRequest {
    std::string request;
    WidgetTarget target;
    std::vector<WidgetExchange> conversation;
    std::string diff;                        // the preview's unified diff, not applied; empty when none
    std::optional<WidgetHandoff> handoff;    // the agent's reason, when it asked
};

class WidgetSurface {
public:
    struct Hooks {
        // The handoff to the side chat; the surface then closes.
        std::function<void(const WidgetChatRequest&)> ask_in_chat;
        // The widget agent's settings: the model (Model), or the provider's
        // setup when none is set up (Setup).
        std::function<void()> open_settings;
        std::function<std::chrono::steady_clock::time_point()> now;
        // The id of the next preview; "widget-<hex>-<n>" by default. The
        // session's preview ids are [a-z][a-z0-9_-]{0,63}.
        std::function<std::string()> make_preview_id;
    };

    WidgetSurface(WidgetSession& session, WidgetAgent& agent, Hooks hooks = {});
    ~WidgetSurface();
    WidgetSurface(const WidgetSurface&) = delete;
    WidgetSurface& operator=(const WidgetSurface&) = delete;

    void set_preview_sink(WidgetPreviewSink* sink) { sink_ = sink; }

    // ---- input events
    // ⌘K on a target: the prompt, or with no provider the prompt that offers
    // only Ask in chat. Whatever was open on another target is closed first.
    // False when the target is not in the open spec.
    bool open(const WidgetTarget& target);
    void set_text(std::string text);
    // Enter in the prompt: begins the preview on the first prompt, then
    // prompts the agent. With no provider it is Ask in chat.
    void submit();
    void accept();           // preview: preview.commit from the ui origin
    void reject();           // preview, conflict: preview.discard
    void refine();           // preview: the prompt for a follow-up on the same preview;
                             // conflict: a new preview from the current spec
    void escape();           // leaves the innermost thing open (see the .cpp)
    void cancel();           // running: cancels the turn and discards the preview
    void ask_in_chat();      // hands the request to the side chat
    void retry();            // declined, failed: the same request again
    void rephrase();         // declined, handoff: the prompt with the request to edit
    void open_model();       // declined: the widget agent's settings
    void toggle_diff();      // preview: the full diff or the compact one
    void close();            // back to idle, discarding the preview and closing the session
    void act(WidgetAction action);

    // ---- per frame: the agent's events, the open spec changing under the
    // preview, and what waited for the session.
    void update();

    // ---- what is shown
    WidgetState state() const { return state_; }
    const WidgetTarget& target() const { return target_; }
    const std::string& text() const { return text_; }
    const std::string& request() const { return request_; }   // the prompt being answered
    const std::string& preview_id() const { return preview_; }
    bool refining() const { return state_ == WidgetState::prompt && !preview_.empty(); }
    const std::string& progress() const { return progress_; }
    const std::string& reply() const { return reply_; }
    const std::string& thought() const { return thought_; }
    const std::vector<WidgetDiffLine>& diff() const { return diff_; }
    const std::string& full_diff() const { return full_diff_; }
    bool showing_full_diff() const { return show_full_diff_; }
    int edits() const { return edits_; }
    bool limit() const { return limit_; }
    const std::optional<WidgetHandoff>& handoff() const { return handoff_; }
    const caliper_builder::Error& error() const { return error_; }
    const std::vector<WidgetExchange>& conversation() const { return conversation_; }
    // What the last session's bundle cut, or why it could not be built.
    const std::vector<std::string>& bundle_cuts() const { return bundle_cuts_; }
    const caliper_builder::Error& bundle_error() const { return bundle_error_; }
    std::string model_label() const;
    // The tag of the state's card ("DECLINED", ...) or "".
    std::string tag() const;
    // The actions of the action row, and of the prompt row's right end.
    std::vector<WidgetAction> card_actions() const;
    std::vector<WidgetAction> row_actions() const;

    // ---- the ImGui side (frame thread)
    // Installs the band and the overlay on the runtime and draws previews
    // there; detach() takes them off.
    void attach(SpecRuntime& runtime);
    void detach();
    // Reads ⌘K (Ctrl+K) on `focused` (or the panel last clicked) and Escape
    // unless `keys` is false (a page that routes them itself), Enter on the
    // card, then update(). `look` sizes the band.
    void frame(const SpecLook& look, const std::string& focused = {}, bool keys = true);
    // The band the surface keeps under a panel's strip.
    float reserve(const SpecPanelFrame& frame) const;
    // Draws the surface inside the target panel's band; for any panel,
    // notes a click as its focus.
    void overlay(const SpecPanelFrame& frame);
    // Puts the keyboard into the prompt's input on its next draw.
    void focus_input() { focus_input_ = true; }

private:
    bool ensure_preview();
    void begin_turn(const std::string& text);
    void on_agent(const WidgetAgentEvent& e);
    void finish_turn(const WidgetAgentEvent& e);
    bool read_preview();
    void show_preview();
    void drop_preview();
    void end_session();
    void reset_turn();
    void enter(WidgetState s);
    std::chrono::steady_clock::time_point now() const;
    std::string next_preview_id();

    WidgetSession& session_;
    WidgetAgent& agent_;
    Hooks hooks_;
    WidgetPreviewSink* sink_ = nullptr;

    WidgetState state_ = WidgetState::idle;
    WidgetTarget target_;
    std::string text_, request_, preview_, progress_, reply_, thought_, full_diff_;
    std::string seen_open_text_;   // the open spec the preview was last read against
    std::vector<WidgetDiffLine> diff_;
    std::vector<WidgetExchange> conversation_;
    std::optional<WidgetHandoff> handoff_;
    caliper_builder::Error error_, bundle_error_;
    std::vector<std::string> bundle_cuts_;
    int edits_ = 0;
    bool limit_ = false;
    bool show_full_diff_ = false;
    bool session_open_ = false;     // the agent has a session on preview_
    bool fresh_session_ = false;    // the next submit starts a new agent session
    bool pending_commit_ = false;
    std::string pending_turn_;      // a prompt that waits for the session
    std::vector<std::string> pending_discards_;   // previews to discard once the session is free
    std::chrono::steady_clock::time_point applied_at_{};
    int preview_count_ = 0;
    std::string preview_prefix_;
    std::string model_;

    // ImGui side
    SpecRuntime* runtime_ = nullptr;
    std::unique_ptr<SpecLook> look_;   // the look of the last frame()
    std::string focused_;
    bool focus_input_ = false;
    std::string shown_panel_;       // the panel the sink shows a preview in
    struct Sink;
    std::unique_ptr<WidgetPreviewSink> runtime_sink_;
};

// ---- what the page owns -----------------------------------------------------------------

// The inline widget as one object for a page: the session adapter, the agent
// and the surface. The page makes one when the session is open, calls
// attach() with its spec runtime once, frame() each frame before the runtime
// draws, and overlay() from the runtime's overlay when it has one of its own
// (attach() installs the surface's otherwise).
//
// The agent is the page's choice: an AcpWidgetAgent (widget_agent_acp.h) over
// the packaged caliper-agent, or a ScriptedWidgetAgent for tests and
// screenshots. A hidden app under test must never be given an agent that
// reads the system key store.
class InlineWidget {
public:
    struct Options {
        caliper_builder::Session* session = nullptr;
        std::function<bool()> busy;              // AppSession::worker_busy()
        std::unique_ptr<WidgetAgent> agent;
        WidgetSurface::Hooks hooks;
    };
    explicit InlineWidget(Options options);
    ~InlineWidget();

    WidgetSurface& surface() { return *surface_; }
    void attach(SpecRuntime& runtime) { surface_->attach(runtime); }
    void frame(const SpecLook& look, const std::string& focused = {}, bool keys = true) {
        surface_->frame(look, focused, keys);
    }
    void overlay(const SpecPanelFrame& frame) { surface_->overlay(frame); }

private:
    std::unique_ptr<SessionWidgetSession> session_;
    std::unique_ptr<WidgetAgent> agent_;
    std::unique_ptr<WidgetSurface> surface_;
};

// ---- the layout -------------------------------------------------------------------------

struct WidgetRect {
    float x = 0, y = 0, w = 0, h = 0;
    float right() const { return x + w; }
    float bottom() const { return y + h; }
    bool empty() const { return w <= 0.0f || h <= 0.0f; }
    bool contains(const WidgetRect& r) const {
        return r.x >= x && r.y >= y && r.right() <= right() + 0.01f && r.bottom() <= bottom() + 0.01f;
    }
    bool overlaps(const WidgetRect& r) const {
        return !empty() && !r.empty() && r.x < right() && x < r.right() && r.y < bottom() && y < r.bottom();
    }
};

// The sizes of the surface's rows, which grow with the text scale.
struct WidgetLayoutMetrics {
    float row = 30;          // the prompt row
    float line = 20;         // a reply, tag or diff line
    float pad = 8;           // inside the band, left and right
    float gap = 6;           // between rows and between buttons
    float button = 22;       // a button's height
    float button_pad = 8;    // inside a button, left and right
    float key_gap = 6;       // between a button's key and its label
    float mark = 26;         // the ⌘K mark at the row's left
    float live = 16;         // the live diamond before the progress, and its gap
    float min_input = 96;    // the narrowest input before the status collapses
    float min_content = 48;  // the content kept under the band before the band gives way
    float min_width = 240;   // the narrowest panel the surface lays out in full
    int max_diff_lines = 6;  // compact lines before they fold into "+N"
    int max_full_lines = 12; // full-diff lines shown before it scrolls
    static WidgetLayoutMetrics for_text_scale(float s);
};

enum class WidgetFont { label, key, mono, tag };
using WidgetMeasure = std::function<float(const std::string& text, WidgetFont font)>;

struct WidgetLayoutInput {
    WidgetRect panel;       // the whole panel
    float strip = 0;        // its title strip's height
    WidgetState state = WidgetState::prompt;
    bool refining = false;  // the prompt over a preview
    std::string status;     // the prompt row's right label: the model or the progress
    std::string tag;        // the card's tag, or ""
    bool reply = false;     // a reply line to show
    int diff_lines = 0;     // compact diff lines
    bool full_diff = false;
    int full_diff_lines = 0;
    std::vector<WidgetAction> card_actions;
    std::vector<WidgetAction> row_actions;
};

struct WidgetLayout {
    float band = 0;               // the height kept under the strip
    WidgetRect prompt;            // the prompt row
    WidgetRect mark;              // the ⌘K mark
    WidgetRect input;             // the text field, or the request as it was sent
    WidgetRect status;            // the model or the progress; empty when collapsed
    std::vector<WidgetRect> row_buttons;   // at the row's right, in row_actions order
    WidgetRect card;              // under the row: tag, reply, diff, actions; empty in prompt and running
    WidgetRect tag;               // empty when there is none
    WidgetRect reply;             // empty when collapsed or none
    WidgetRect diff;              // the compact lines, or the full diff
    int diff_shown = 0;           // compact lines shown; the rest fold into "+N"
    bool diff_folded = false;
    WidgetRect actions;           // the action row
    std::vector<WidgetRect> buttons;   // in card_actions order
    bool keys = true;             // the buttons show their keys
    bool content_hidden = false;  // the panel is below the minimum: the band takes it all
};

// Lays the surface out in the panel. What collapses first, as the panel
// shrinks: the reply line, then compact diff lines past the first (folded
// into "+N"), then the status label of the prompt row, then the buttons'
// keys; last, below the minimum height, the band takes the whole panel and
// the content is hidden.
WidgetLayout layout_widget_surface(const WidgetLayoutInput& in, const WidgetLayoutMetrics& m,
                                   const WidgetMeasure& measure);

// The layout input of a surface's state for a panel.
WidgetLayoutInput widget_layout_input(const WidgetSurface& surface, const WidgetRect& panel, float strip);

}  // namespace caliper_host
