#pragma once
// ============================================================================
// The landing screen: a workstation home in the Sternbild house style, drawn
// with Dear ImGui for every renderer backend (Metal, Vulkan, GL). It replaced
// the GL-only IntroScreen and the plain fallback launcher list.
//
// Layout (top to bottom):
//   command bar   masthead (mark + CALIPER), the command line (filters the
//                 instruments, Enter launches), the live-run area (jobs), and
//                 the line's actions as key buttons: ⌘K Command, ↑ ↓, esc Clear
//   dock space    four panes in ImGui docking, first-run layout built with
//                 DockBuilder and persisted in the app-data imgui.ini:
//                   Projects     (empty state; F2 Open folder disabled with
//                                 the reason "not yet available"; F4 Attach
//                                 to process opens the live sessions of
//                                 training scripts here: script, pid, start
//                                 time, state, views; ↑↓ select, Enter /
//                                 double-click attach, esc back)
//                   Instruments  (installed applets: name, id, version, tag,
//                                 status; ↑↓ select, Enter / double-click
//                                 launch)
//                   Inspector    (the selected applet's manifest: id, version,
//                                 tag, summary, required / optional services,
//                                 status text, binary path; ⏎ Launch plate,
//                                 and ⌘R Restart beside it for an applet
//                                 quarantined by a C++ exception)
//                   Activity     (host log: renderer, scans, launches, jobs,
//                                 live sessions attached, ended, detached)
//   status bar    backend · GPU · memory · applet count (facts, no actions)
//
// Patterns named in the implementation: status bar (VS Code / JetBrains),
// command bar of key buttons (Bloomberg command line and function keys),
// dense tables with keyboard navigation and one accent for selection
// (Bloomberg, W&B run tables), docked panes (Launchpad, Blender areas),
// overview first / details on demand (Shneiderman), honest disabled actions.
// ============================================================================

#include <functional>
#include <string>
#include <vector>

namespace caliper_host { class AppletLoader; struct AppletEntry; }
namespace caliper_live { class SessionList; }

class LandingScreen {
public:
    // Everything the screen reads each frame. The loader is read live so
    // statuses are always fresh; `launch` reproduces a row activation.
    struct Context {
        caliper_host::AppletLoader* loader = nullptr;
        const char* renderer_name = "";          // "metal" / "gl" / "vulkan"
        std::function<void(int)> launch;         // launch applet by loader index
        // Unload and relaunch a quarantined applet (loader index); offered
        // only when AppletLoader::restartable says so.
        std::function<void(int)> restart;
        // F4 Attach to process: the live sessions (relisted once a second
        // while the landing is drawn) and attach-by-token. Null when the build
        // has no live sessions; the F4 button then says why.
        caliper_live::SessionList* sessions = nullptr;
        std::function<void(const std::string&)> attach;
    };

    // Draws one frame of the landing page. Owns its windows and dock space;
    // the caller has already called renderer->new_frame().
    void draw(const Context& ctx);

    // Appends an Activity row (landing/final: WHEN / EVENT / OBJECT / DETAIL,
    // newest first, relative time). `event` is a short verb ("launch",
    // "scanned", "home"); `object` the thing it acted on; `detail` optional.
    void log(std::string event, std::string object = {},
             std::string detail = {});

    // Index into the loader of the selected row, or -1. Survives launches so
    // Home returns to the same row.
    int selected_index() const { return selected_; }
    // Selects a row by loader index (a launch that did not come from a row,
    // so Home still returns to it); kept in view on the next draw.
    void select(int index) {
        selected_ = index;
        scroll_to_selected_ = true;
    }

private:
    struct LogLine {
        double time_sec = 0.0;   // wall clock at log(), seconds since the epoch
        std::string event;
        std::string object;
        std::string detail;
    };

    // One frame, top to bottom (landing_screen.cpp).
    void draw_command_bar();
    void draw_dock_space();
    void draw_projects(const Context& ctx);
    void draw_sessions(const Context& ctx);   // Projects in attach mode (F4)
    void draw_instruments(const Context& ctx);
    void draw_inspector(const Context& ctx);
    void draw_inspector_rows(const caliper_host::AppletEntry& e);
    void draw_activity();
    void draw_status_bar(const Context& ctx);
    // Filters the loader by command_ and sorts by the table's sort spec.
    void rebuild_rows(const caliper_host::AppletLoader& loader);

    int selected_ = -1;
    std::string command_;            // command-line text (filters instruments)
    bool focus_command_ = false;     // ⌘K / Ctrl+K: focus the command line
    bool layout_built_ = false;      // first-run DockBuilder layout attempted
    std::vector<LogLine> log_;

    std::vector<int> rows_;          // visible loader indices, filtered + sorted
    int sort_column_ = 3;            // Instruments sort column (TAG) ...
    bool sort_descending_ = false;   // ... and direction, from the table spec
    bool command_active_ = false;    // the command line is being edited
    bool enter_requested_ = false;   // Enter on the command line or the table
    bool enter_from_command_ = false;
    bool scroll_to_selected_ = false;  // keyboard moved the selection
    int focus_instruments_ = 0;      // frames left to hand focus to the table
    const char* focus_pane_ = nullptr;  // ⌘1–⌘4: pane to focus after the frame
    int pending_launch_ = -1;        // activation, run after the frame is drawn
    int pending_restart_ = -1;       // Restart, run after the frame like a launch
    // F4 Attach to process: Projects lists the live sessions instead of the
    // ways in. The selection is the session token, so a relist keeps it.
    bool attach_mode_ = false;
    bool open_sessions_ = false;     // F4 / the button: enter attach mode after the frame
    std::string session_selected_;
    std::string pending_attach_;     // Enter / double-click: attach after the frame
    bool scroll_to_session_ = false;
    bool projects_focused_ = false;  // Projects had focus last frame (owns Esc then)
    int focus_projects_ = 0;         // frames left to hand focus to Projects
    int last_frame_ = -1;            // ImGui frame of the previous draw()
    int select_step_ = 0;            // ↑ ↓ buttons: rows to move this frame
    float command_right_w_ = 0.0f;   // right-aligned block, measured last frame
    float updown_w_ = 0.0f;          // command-bar ↑ ↓ buttons + gap, when drawn
};

// Status-bar items every page shares (VS Code status bar): the backend,
// uppercased, then the device name. Call between sternbild::status_bar_begin()
// and status_bar_end(); page-specific items follow.
void draw_host_status_items(const char* renderer_name);
