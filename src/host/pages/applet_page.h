#pragma once
// ============================================================================
// The applet page of the caliper executable, shared by the app and by
// `caliper shot`. One frame of the page is, in this order: the page bar and
// the status bar, then the jobs tray's strip above the status bar while it is
// open (none of the three when the page is chromeless), the dock space over
// the work area they leave (work_area.h), docking by role (dock_layout.h),
// and the applet's own frame between UI checkpoints (ui_recovery.h). The jobs
// tray, which the app draws on every page, is here too.
//
// The app draws these into its window; `caliper shot` draws the same frames
// into an offscreen canvas and writes the last one as a PNG, so a shot shows
// what the app shows. Navigation (Esc, Home, quarantine back to the landing)
// stays with the caller: draw() only reports what happened.
// ============================================================================
#include "applet_loader.h"
#include "frame_watchdog.h"
#include "job_system.h"
#include "ui/dock_layout.h"
#include "ui/work_area.h"

#include <caliper/abi.h>

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace caliper_host {

// The CaliperHost block every launch hands the loader; the loader fills in
// the applet's data folder.
CaliperHost applet_host_proto();

// The page bar of the applet and live pages: a main menu bar one frame tall,
// with the landing command bar's height. Call ImGui::EndMainMenuBar() when it
// returns true.
bool begin_page_bar();

class JobsTray;

class AppletPage {
public:
    struct Result {
        bool home = false;    // the page bar's Home was clicked
        bool alive = true;    // false: the applet faulted this frame and is quarantined
    };

    // A freshly launched applet: the watchdog starts again.
    void enter() { watchdog_.reset(); }

    // Draws one frame of the page for loader entry `applet`, inside an ImGui
    // frame the caller opened. `fi` is the frame passed to the applet, and
    // `tray` (may be null) the jobs tray, whose strip is drawn between the
    // status bar and the dock space unless the page is chromeless. When the
    // applet faults, the UI state it left open is unwound before this
    // returns, and the caller tears the applet down after rendering the frame.
    Result draw(AppletLoader& loader, int applet, const CaliperFrameInfo& fi,
                const char* renderer_name, bool chromeless, JobsTray* tray);

    // The layout that the applet's dock hints go to (set_dock_hint_sink).
    DockLayout& dock_layout() { return dock_layout_; }

private:
    bool draw_bars(const AppletLoader& loader, int applet, const char* renderer_name);

    DockLayout dock_layout_;
    FrameWatchdog watchdog_;
};

// The jobs tray: a strip above the status bar, kJobsTrayHeight tall, while a
// job runs or a failure is kept (JobsTrayPresence in work_area.h decides when,
// so the work area does not jump). It is one line: the running jobs, each
// with its progress and a cancel button, then the kept failures, each marked
// FAILED with its reason, then how many jobs finished and "clear finished" at
// the right end. Entries that do not fit are behind "+N more", which opens the
// whole list in a popup. The strip reserves its height from the work area,
// so the dock space and every applet window end above it, and nothing of the
// applet is covered unless the user opens the popup.
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

    // Once per frame, before any page is drawn: the frame's jobs, after
    // note_failures(), and a monotonic time in seconds, which decide whether
    // the strip is open in this frame.
    void begin_frame(std::vector<JobSystem::JobView> views, double now);
    bool open() const { return presence_.open(); }

    // Draws the strip when it is open. Call it after the status bar and
    // before the dock space (dock_space_over_work_area), so the strip lands
    // above the status bar and the dock space ends above the strip.
    void draw();

    // A strip that draw() draws right after the tray's own, whether the tray
    // is open or not, so that it lands above the tray on every page that
    // draws the tray: the approval cards' strip (builder/approval_cards.h).
    void set_next_strip(std::function<void()> draw) { next_strip_ = std::move(draw); }

private:
    void draw_tray();
    void draw_list();   // every entry, one per row, in the "+N more" popup
    void clear_finished();

    std::vector<Failure> failed_;   // oldest first
    std::vector<JobSystem::JobView> views_;
    JobsTrayPresence presence_;
    std::function<void()> next_strip_;
};

} // namespace caliper_host
