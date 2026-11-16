#pragma once
// ============================================================================
// approval_cards.h: the approval cards (design/acp-integration.md, Approvals
// and diffs for every agent; design/command-layer.md, Approval). An agent's
// command that needs the person's approval is parked by the command layer:
// a destructive command from agent, mcp or api, and, with review on (the
// default for agent and mcp), an edit of the open spec, which carries its
// diff. Caliper API holds a call from agent or mcp for up to 120 seconds
// while it waits. The cards are where the person sees and decides it.
//
// One card per pending approval says who asks (the origin as a noun: MCP
// agent, Side chat, Widget agent, API), what (the command and its target,
// "panel.patch · Loss and accuracy"), the review when there is one (the
// compact diff of diff_card.h, as the inline surface draws it: the changed
// keys with their old and new values, the validator's warnings, and the full
// unified diff on demand), or what a destructive command loses as a noun
// phrase ("Applet run42-dashboard overwritten"), the time left before the
// call times out, and two actions, Approve and Deny. Both go through the
// session's queue (CommandQueue::approve() and deny()), whose decision runs
// at the next drain on the frame thread. There is no Approve all for one
// agent's turn: the command layer has no safe notion of one (a group is the
// agent's own label, and a held call answers before the agent sends its next
// edit, so one turn never has two cards at once).
//
// Where they sit. A strip at the window's bottom edge, above the jobs tray,
// on every page: it takes its own space as the tray does (a viewport side
// bar, work_area.h), so it covers no panel, and it comes and goes with the
// tray's delays (JobsTrayPresence), so the panes do not jump; its height
// changes while it is open only after the same delays. The cards stand side
// by side, oldest first, since the oldest times out first; those that do not
// fit fold into "+N" at the right end, which lists them all. On the spec
// page, a panel that a waiting edit changes carries a REVIEW mark in its
// title strip. A decided card stays in the strip for a moment with its
// outcome (APPROVED, DENIED, TIMED OUT, CONFLICT, FAILED). Approve and Deny
// have no keys, so that no keystroke meant for another pane decides.
//
// Three parts, so that each is tested without a window:
//   - ApprovalCards, the logic: reads the session's pending approvals on the
//     frame thread (at most every 100 ms, never while a long command holds
//     the registry), builds a card once for each new one, reads the time left
//     from the queue each frame, and writes one Activity line when a request
//     starts waiting and one when it is decided, times out or is withdrawn.
//   - layout_approval_strip(): the strip and its cards as plain rectangles,
//     computed without ImGui from text widths the caller measures, with
//     minimums that grow with the text scale.
//   - ApprovalTray, the app's part: the strip drawn with ImGui in the house
//     style, and the panel mark. The side chat will draw the same cards in
//     its pane, so nothing here depends on a page.
//
// A test-only way to decide: a hidden app (--hidden) whose environment names
// a file in CALIPER_TEST_APPROVALS reads decisions from it, one line each,
// "approve <command> [<origin>]" or "deny <command> [<origin>]", or, for a
// card that offers rules, "session <command> [<origin>]" (Allow always for
// this session) or "always <command> [<origin>]" (Always allow), applied to
// the oldest waiting card that matches through the same path as the buttons.
// The app reads the variable only when hidden; nothing over Caliper API or
// MCP can decide, so an agent cannot approve its own request.
//
// The app's part is declared in every build; without CALIPER_WITH_BUILDER it
// does nothing. The rest is built only with the builder. Frame thread only.
// ============================================================================
#include <functional>
#include <memory>
#include <ostream>
#include <string>

namespace caliper_builder {
class Session;
}

namespace caliper_host {

// ---- the app's part -----------------------------------------------------------------------

class ApprovalTray {
public:
    struct Hooks {
        // One Activity line (WHEN / EVENT / OBJECT / DETAIL).
        std::function<void(const std::string& event, const std::string& object, const std::string& detail)> log;
        std::ostream* err = nullptr;   // the [approval] lines; standard error when null
        // CALIPER_TEST_APPROVALS, which the app passes only when hidden.
        std::string decisions;
    };

    ApprovalTray();
    ~ApprovalTray();
    ApprovalTray(const ApprovalTray&) = delete;
    ApprovalTray& operator=(const ApprovalTray&) = delete;

    void start(Hooks hooks);
    // Once per frame, after the session's queue has drained and before the
    // pages draw: the session (null when none is open), whether a long
    // command holds its registry (AppSession::worker_busy()), and a monotonic
    // time in seconds.
    void frame(caliper_builder::Session* session, bool busy, double now);
    // The strip, from the jobs tray's draw (JobsTray::set_next_strip()), so
    // that it lands above the tray on every page.
    void draw_strip();
    // The spec page's overlay of each shown panel, in the panel's own window
    // after its content (SpecPage::set_panel_overlay()): the title its strip
    // shows, its rectangle and its title strip's height (0 for a tab).
    // Marks the panel a waiting edit changes.
    void mark_panel(const std::string& panel_id, const std::string& title, float x, float y, float w, float h,
                    float strip);
    // Points at an approval's card for a moment with a bronze outline: the
    // side chat's row of a waiting command (side_chat.h) calls it.
    void point_to(const std::string& approval_id);
    // Before the session closes: the cards go, and nothing is read again.
    void close();
    // How many waiting cards the strip drew at its full height in its last
    // frame: 0 while it is closed, still opening or growing. The hidden app's
    // input script waits on it (spec_ui_driver.h, `cards`), so that a capture
    // of a card does not depend on the strip's delays.
    int shown_waiting() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host

#if CALIPER_WITH_BUILDER

#include "builder/diff_card.h"
#include "builder/widget_surface.h"   // WidgetRect, WidgetMeasure, WidgetFont

#include <caliper_builder/commands/approval.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/spec/spec.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <optional>
#include <vector>

namespace caliper_host {

// ---- one card -----------------------------------------------------------------------------

// The origin as a noun: "MCP agent", "Side chat", "Widget agent" (the agent
// origin with a stacked preview, which only the widget agent sends), "API",
// "Builder" or "App".
std::string approval_who(const caliper_builder::CommandInvocation& invocation);
// The command and its target: "panel.patch · Loss and accuracy", with a
// panel's title from `open` when it has one, a source's, loader's or
// applet's id, "layout", a file's name, or the number of steps.
std::string approval_what(const caliper_builder::CommandInvocation& invocation, const caliper_builder::spec::Spec* open);
// What a destructive command loses, as a noun phrase: "Applet
// run42-dashboard overwritten" (the open spec's applet when the call names
// none), "Loader rows deleted", "Tables of loader rows replaced", "2 steps
// undone"; "" for a command that loses nothing.
std::string approval_loss(const caliper_builder::CommandInvocation& invocation,
                          const caliper_builder::spec::Spec* open = nullptr);

enum class ApprovalState { waiting, deciding, approved, denied, timed_out, conflict, failed, withdrawn };
const char* approval_state_tag(ApprovalState state);   // "APPROVED", ...; "" while waiting

struct ApprovalCard {
    std::string id;       // the approval id
    std::string name;     // the command
    std::string origin;   // the origin's name: agent, mcp, api
    std::string who, what, loss;
    bool review = false;                  // the pending approval carries a review
    std::vector<WidgetDiffLine> diff;     // the compact diff of the review
    std::string full_diff;                // its unified diff of the canonical TOML
    std::vector<std::string> warnings;    // the validator's warnings, "code: message"
    std::vector<std::string> panels;      // the open spec's panels the edit changes or removes
    std::optional<std::chrono::milliseconds> time_left;   // none: the call waits without a limit
    ApprovalState state = ApprovalState::waiting;
    // The card offers Allow once, Allow always for this session and Always
    // allow <command> (Session::approval_offers_rules()).
    bool rules = false;
    caliper_builder::CommandInvocation invocation;   // the parked call, whose command a rule names
    std::string error;                    // conflict, failed: the command's error code
    double decided_at = 0.0;              // when the card learnt its outcome
};

// The compact lines of a unified diff of canonical TOML, for a review whose
// edit no longer applies to the open spec: a removed and an added line of the
// same key pair up as a change.
std::vector<WidgetDiffLine> approval_diff_from_unified(const std::string& diff, size_t max_value = 40);

// ---- the logic ----------------------------------------------------------------------------

class ApprovalCards {
public:
    struct Hooks {
        std::function<void(const std::string& event, const std::string& object, const std::string& detail)> log;
        std::ostream* err = nullptr;
    };
    static constexpr double kReadInterval = 0.1;   // seconds between two reads of the pending list
    static constexpr double kKeepDecided = 2.0;    // seconds a decided card stays, with its outcome

    ApprovalCards();
    explicit ApprovalCards(Hooks hooks);

    // The test-only decisions file (CALIPER_TEST_APPROVALS); "" reads none.
    void set_decisions(const std::string& path);

    // Once per frame on the frame thread: the session (null: none open),
    // whether a long command holds its registry, and a monotonic time in
    // seconds. Reads the pending list when it is due and the registry is
    // free, and the time left of each waiting card always.
    void sync(caliper_builder::Session* session, bool busy, double now);
    // Drops every card, without Activity lines (the session closed).
    void clear();

    // The person's decisions, through the session's queue; false when the
    // card is not waiting.
    bool approve(const std::string& id);
    bool deny(const std::string& id);
    // Allow always for this session (`saved` false) or Always allow (true):
    // the rule is added for the card's own command through the session, then
    // the card is approved. False when the card is not waiting or offers no rule.
    bool allow_always(const std::string& id, bool saved);

    // The waiting cards, oldest first, then the decided ones still kept.
    const std::vector<ApprovalCard>& cards() const { return cards_; }
    const ApprovalCard* find(const std::string& id) const;
    size_t waiting() const;   // waiting or deciding
    // The oldest waiting card whose edit changes the panel, for its mark.
    const ApprovalCard* card_for_panel(const std::string& panel_id) const;

private:
    void read_pending(caliper_builder::Session& s, double now);
    ApprovalCard make_card(caliper_builder::Session& s, const caliper_builder::PendingApproval& p) const;
    void settle(caliper_builder::Session& s, ApprovalCard& card, double now);
    void apply_decisions();
    bool decide(const std::string& id, bool approve);
    bool decide_rule(const std::string& id, bool saved);
    void line(const ApprovalCard& card, const std::string& detail);

    Hooks hooks_;
    caliper_builder::Session* session_ = nullptr;
    std::vector<ApprovalCard> cards_;
    std::map<std::string, std::future<caliper_builder::CommandResult>> decisions_made_;
    double last_read_ = -1.0;
    bool read_now_ = true;
    // The decisions file: its path, how much was read, and the lines not
    // applied yet because no card matched.
    std::string decisions_path_;
    std::uintmax_t decisions_read_ = 0;
    double decisions_checked_ = -1.0;
    std::deque<std::string> decisions_waiting_;
    double now_ = 0.0;
};

// ---- the layout ---------------------------------------------------------------------------

// The sizes of the strip, which grow with the text scale.
struct ApprovalLayoutMetrics {
    float line = 20;          // a header, diff, warning or outcome line
    float pad = 8;            // around the strip's content and inside a card
    float gap = 6;            // between cards, between rows and between buttons
    float button = 22;        // a button's height
    float button_pad = 10;    // inside a button, left and right
    float min_card = 340;     // the narrowest card before the next folds into "+N"
    float max_card = 620;     // the widest card
    int max_diff_lines = 4;   // compact lines before the rest fold into "+N"
    float max_share = 0.4f;   // of the window's height, the most the strip takes
    static ApprovalLayoutMetrics for_text_scale(float s);
};

// What one card needs, for the layout.
struct ApprovalCardShape {
    std::string who;          // the origin as a noun, drawn as a tag
    bool review = false;      // diff lines and the Diff button
    int diff_lines = 0;
    bool warnings = false;
    bool decided = false;     // an outcome line and no buttons
    bool rules = false;       // a row of the two rule answers above the buttons
    std::string what_command; // the command a rule names, for the Always allow label
};

struct ApprovalLayoutInput {
    float width = 0;          // the strip's width: the window's
    float window_h = 0;       // the window's height
    float max_height = 0;     // a lower bound than max_share, or 0 (ApprovalStripHeight)
    float fill_height = 0;    // a strip kept taller than it needs: the cards fill it, or 0
    int waiting = 0;          // the count beside the label when more than one waits
    std::vector<ApprovalCardShape> cards;   // in order, oldest first
};

struct ApprovalCardLayout {
    WidgetRect card;          // the card's well
    WidgetRect who;           // the origin's tag
    WidgetRect what;          // the command and its target
    WidgetRect time;          // the time left; empty when there is no room
    WidgetRect body;          // the compact diff, the loss or the outcome
    int diff_shown = 0;       // compact lines shown; the rest fold into "+N"
    bool diff_folded = false;
    WidgetRect warnings;      // empty when none or no room
    WidgetRect approve, deny, diff;   // the buttons; empty when decided (diff: without a review)
    WidgetRect session, always;       // the rule answers, a row above; empty without rules
};

struct ApprovalStripLayout {
    float height = 0;          // the strip's height; 0 with no card
    WidgetRect label;          // "Approvals", with the count under it; empty when there is no room
    WidgetRect count;          // the count; empty with one card waiting
    WidgetRect more;           // "+N", the cards that do not fit; empty when all fit
    int hidden = 0;
    std::vector<ApprovalCardLayout> cards;   // the shown cards, oldest first
};

// Lays the strip out in (0, 0, width, height). Across: the label, then as
// many cards as fit at their minimum width side by side, the rest folded
// into "+N" at the right end; as the window narrows the label goes first,
// and one card always shows. Down: one row for the header, the body, a row
// for the warnings when a shown card has any, and the buttons; the strip
// takes at most max_share of the window's height, and as it shrinks the
// compact lines past the first fold, then the warnings row goes.
ApprovalStripLayout layout_approval_strip(const ApprovalLayoutInput& in, const ApprovalLayoutMetrics& m,
                                          const WidgetMeasure& measure);

// The labels of a card's answers: "Approve" or, with rules, "Allow once";
// "Allow always for this session"; "Always allow <command>".
const char* approval_approve_label(bool rules);
const char* approval_session_label();
std::string approval_always_label(const std::string& command);

// The layout input of the cards for a window.
ApprovalLayoutInput approval_layout_input(const ApprovalCards& cards, float width, float window_h);

// The strip's height while it is open: a taller one is taken after it has
// been asked for during JobsTrayPresence::kOpenDelay, a lower one after
// kCloseDelay, so the panes move only after the strips' usual delays.
class ApprovalStripHeight {
public:
    float update(double now, bool open, float need);
    float height() const { return height_; }

private:
    float height_ = 0;
    float asked_ = 0;
    double asked_since_ = -1.0;
};

}  // namespace caliper_host

#endif  // CALIPER_WITH_BUILDER
