#pragma once
// ============================================================================
// The applet page of the caliper executable, shared by the app and by
// `caliper shot`. One frame of the page is, in this order: the page bar and
// the status bar (unless the page is chromeless), the dock space over the
// work area between them, docking by role (dock_layout.h), and the applet's
// own frame between UI checkpoints (ui_recovery.h). The jobs tray, drawn on
// every page of the app while jobs exist, is here too.
//
// The app draws these into its window; `caliper shot` draws the same frames
// into an offscreen canvas and writes the last one as a PNG, so a shot shows
// what the app shows. Navigation (Esc, Home, quarantine back to the landing) stays with the
// caller: draw() only reports what happened.
// ============================================================================
#include "applet_loader.h"
#include "frame_watchdog.h"
#include "job_system.h"
#include "ui/dock_layout.h"

#include <caliper/abi.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace caliper_host {

// The CaliperHost block every launch hands the loader; the loader fills in
// the applet's data folder.
CaliperHost applet_host_proto();

// The page bar of the applet and live pages: a main menu bar one frame tall,
// with the landing command bar's height. Call ImGui::EndMainMenuBar() when it
// returns true.
bool begin_page_bar();

class AppletPage {
public:
    struct Result {
        bool home = false;    // the page bar's Home was clicked
        bool alive = true;    // false: the applet faulted this frame and is quarantined
    };

    // A freshly launched applet: the watchdog starts again.
    void enter() { watchdog_.reset(); }

    // Draws one frame of the page for loader entry `applet`, inside an ImGui
    // frame the caller opened. `fi` is the frame passed to the applet. When
    // the applet faults, the UI state it left open is unwound before this
    // returns, and the caller tears the applet down after rendering the frame.
    Result draw(AppletLoader& loader, int applet, const CaliperFrameInfo& fi,
                const char* renderer_name, bool chromeless);

    // The layout that the applet's dock hints go to (set_dock_hint_sink).
    DockLayout& dock_layout() { return dock_layout_; }

private:
    bool draw_bars(const AppletLoader& loader, int applet, const char* renderer_name);

    DockLayout dock_layout_;
    FrameWatchdog watchdog_;
};

// The jobs tray: a plate pinned to the bottom right of the work area while
// jobs exist, with running jobs first, then failures (kept until cleared),
// then the jobs that finished.
class JobsTray {
public:
    struct Failure {
        uint64_t id = 0;
        std::string label;
        std::string reason;   // the job's message without its "failed: " prefix
    };

    // Keeps every failed job in `views` that it has not seen before, and calls
    // `on_new` once for each. The job system's own entry does not last long
    // enough: going Home joins the applet's jobs and drops every entry.
    void note_failures(const std::vector<JobSystem::JobView>& views,
                       const std::function<void(const Failure&)>& on_new = {});

    // Draws the tray when `views` or the kept failures are not empty.
    void draw(const std::vector<JobSystem::JobView>& views);

private:
    std::vector<Failure> failed_;   // oldest first
};

} // namespace caliper_host
