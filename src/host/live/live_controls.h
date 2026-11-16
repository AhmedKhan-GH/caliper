// live_controls.h: the viewer side of wire protocol 3.1 on a live page. A
// script that calls a checkpoint can be paused, resumed and stepped, the
// parameters it registered can be set and its actions run, and a selection
// made in one view is shared with every view of the same item space, with
// the script's details of the selected items.
//
// RunControls is the state a page keeps between frames: the latest CONTROLS
// of the attached session, the values a person has set that the script has
// not confirmed yet, the requests waiting for their CONTROL_REPLY, the
// results of actions, and the current selections. It reads and writes the
// stream only through the Source's 3.1 seam (live_source.h), so it never
// touches a socket and works on whichever thread draws the page (the frame
// thread). The stream carries only these requests: never a host command.
//
// The rules it keeps (wire protocol 3.1):
//   - A control is sent only when Source::control_access() allows it (the
//     server announced `controls` and set WELCOME's `control`, the session
//     lists `controls` and is live). Otherwise every control is disabled and
//     why_not() says why. Pause, step and actions also need a checkpoint:
//     with the run state `none` they are disabled with that reason. A
//     parameter may be set before the first checkpoint, as the protocol
//     allows (the script reads it at once).
//   - `set` goes out at most CALIPER_NET_CONTROL_RATE_HZ times a second per
//     parameter. While a slider is dragged the newest value waits for its
//     turn, so the value where the drag stops is always sent last.
//   - A value that was set shows as pending until a CONTROLS that arrived
//     after its `ok` reply confirms it (as the value, or as the script's own
//     `pending`). A refused value is dropped and the reason shown.
//
// The drawing functions use plain ImGui widgets with a ControlsLook, so the
// app draws them in its house style and caliper-viewer in ImGui's own.
#pragma once

#include "live/live_source.h"

#include <imgui.h>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

struct CaliperCore;

namespace caliper_live {

class RunControls {
public:
    // Once per frame, before drawing: reads the source's CONTROLS, replies and
    // selections and sends the `set` values whose turn has come. `src` null:
    // nothing is attached and every control is disabled. `now_ns` is a
    // monotonic clock (caliper_live::now_ns()).
    void update(Source* src, uint64_t now_ns);
    // Forgets everything (a detach).
    void reset();

    // A CONTROLS has arrived for the attached session.
    bool have() const { return have_; }
    const Controls& state() const { return controls_; }
    const ControlAccess& access() const { return access_; }
    // Why `op` (pause, resume, step, set, run) cannot be sent now; "" when it can.
    std::string why_not(const char* op) const;

    // The requests. False (and the notice says why) when it was not sent.
    bool pause();
    bool resume();
    bool step(uint32_t count = 1);
    bool set(const std::string& name, const caliper_net::Json& value);
    bool run(const std::string& name);

    // What a parameter's control shows: the value set here until the script
    // confirms it, else the script's pending value, else its value.
    caliper_net::Json shown_value(const ControlParam& p) const;
    // A value set here waits for its reply or for CONTROLS to confirm it.
    bool param_waiting(const ControlParam& p) const;
    // The script holds a pending value it reads at its next checkpoint.
    bool param_pending(const ControlParam& p) const { return !p.pending.is_null(); }
    // An action waits for its reply, is queued or runs.
    bool action_busy(const ControlAction& a) const;

    struct Result {
        std::string text;
        bool error = false;
        uint64_t at_ns = 0;
    };
    // The last result of action `name` (its message, or why it failed); null none.
    const Result* action_result(const std::string& name) const;
    // The last refused request (pause, resume, step, set) and why.
    const Result& notice() const { return notice_; }
    // The clock of the last update.
    uint64_t now_ns() const { return now_ns_; }

    // ---- linked selection -----------------------------------------------------------
    uint64_t selection_version() const { return sel_version_; }
    const std::vector<LinkedSelection>& selections() const { return selections_; }
    // The selection whose details show: the last one to change that has
    // details or a details error and selects something. Null: none.
    const LinkedSelection* details() const;
    const std::vector<ItemAxis>& item_axes() const { return item_axes_; }

    // What changed, as plain lines ("run paused at step 12", "set lr = 0.02",
    // "save: saved at step 14", "selection units: 4 items"), oldest first; a
    // host may log them. Cleared by the call.
    std::vector<std::string> take_log();

    struct Counts {
        uint64_t sent = 0;        // CONTROLs handed to the source
        uint64_t sets = 0;        // of them `set`
        uint64_t replies = 0;     // CONTROL_REPLYs read
        uint64_t refused = 0;     // replies with the status error
    };
    const Counts& counts() const { return counts_; }

private:
    struct Local {                     // a parameter value set here
        caliper_net::Json value;
        bool unsent = false;           // waits for its turn (the rate limit)
        uint64_t sent_ns = 0;          // when the last `set` of it went out
        uint32_t request = 0;          // that `set`'s request id
        bool replied = false;          // its `ok` arrived
        uint64_t reply_controls = 0;   // Controls::version when it did
    };
    void send_set(const std::string& name, Local& l);
    void on_reply(const ControlReply& r);
    void on_controls(const Controls& before);
    uint32_t send(const char* op, const std::string& name, const caliper_net::Json& value, uint32_t count);
    void refuse(const std::string& what, const std::string& why);

    Source* src_ = nullptr;
    uint64_t now_ns_ = 0;
    bool have_ = false;
    Controls controls_;
    ControlAccess access_{false, "not attached to a session"};
    std::map<std::string, Local> locals_;
    std::map<uint32_t, std::string> runs_;     // run request -> action
    std::map<uint32_t, std::string> ops_;      // pause/resume/step request -> op
    std::map<std::string, Result> results_;
    Result notice_;
    uint64_t sel_version_ = 0;
    std::vector<LinkedSelection> selections_;
    std::string details_space_;
    std::vector<ItemAxis> item_axes_;
    std::vector<std::string> log_;
    Counts counts_;
};

// ---- drawing ----------------------------------------------------------------------------
// How the bar and the details look in a host. Empty functions fall back to
// ImGui's own widgets.
struct ControlsLook {
    // A button holding `label`; disabled, it shows `reason` on hover. True when clicked.
    std::function<bool(const char* label, bool enabled, const char* reason)> button;
    // A small field label before a control ("lr").
    std::function<void(const char* text)> label;
    // Opens the details table (columns, then TableSetupColumn and the header
    // row are the caller's); false: not drawn. Paired with end_table.
    std::function<bool(const char* id, int columns)> begin_table;
    std::function<void()> end_table;
    std::function<void()> header_row;
    ImU32 text = IM_COL32(255, 255, 255, 255);
    ImU32 secondary = IM_COL32(190, 190, 190, 255);
    ImU32 faint = IM_COL32(128, 128, 128, 255);
    ImU32 accent = IM_COL32(110, 170, 255, 255);   // the selection highlight
};

// The run bar on the current line of a window the host opened, at most
// `width` wide: the run state, Pause or Resume and Step, every parameter as
// a slider, field, toggle or menu by its type, every action as a button with
// its state and result, then the last refusal or why the controls are off.
// What does not fit goes into a "+N more" popup.
void draw_run_bar(RunControls& rc, const ControlsLook& look, float width);

// The selection's details as a table, one row per item with the script's
// fields, or why there are none.
void draw_selection_details(const RunControls& rc, const ControlsLook& look);

// ==== THE SELECTION HOOK =================================================================
// Hands the linked selections to the view layer: every view whose axis indexes
// an item space shows that space's selection. This is the one place where
// received selections reach the views. Today it draws the highlight itself, on
// top of the views, from what the view layer publishes (live_views_state.h:
// the payload map and geo_area / geo_window of the last frame): bands over the
// selected rows or columns of a heatmap or an image and over the selected steps
// of a series. Points and mosaics are not highlighted here: that needs the view
// layer's own drawing, which replaces this function's body when it lands.
void draw_linked_selection(CaliperCore* core, const RunControls& rc, const ControlsLook& look);

// ---- testing aid -------------------------------------------------------------------------
// Where each widget of the bar was drawn in the last frame, in screen units,
// by name: "pause", "resume", "step", "param:<name>", "action:<name>",
// "more". A UI test aims its injected mouse events at these.
struct WidgetRect {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};
const std::map<std::string, WidgetRect>& bar_widgets();

}  // namespace caliper_live
