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
//                                 the reason "not yet available"; F3 New
//                                 dashboard, a new spec in build mode, titled
//                                 by "new <title>" on the command line; F4 Attach
//                                 to process opens the live sessions of
//                                 training scripts here: script, pid, start
//                                 time, state, views; ↑↓ select, Enter /
//                                 double-click attach, esc back)
//                   Instruments  (installed applets: name, id, version, tag,
//                                 status; ↑↓ select, Enter / double-click
//                                 launch)
//                   Inspector    (the selected applet's manifest: id, version,
//                                 tag, summary, libtorch, required / optional
//                                 services, data schema, SDK, status text,
//                                 binary path; ⏎ Launch plate,
//                                 and ⌘R Restart beside it for an applet
//                                 quarantined by a C++ exception)
//                   Activity     (host log: renderer, scans, launches, jobs,
//                                 live sessions attached, ended, detached)
//   jobs tray     above the status bar while a job runs or a failure is kept
//                 (drawn by the host through Context::after_bars)
//   status bar    backend · GPU · memory · applet count (facts, no actions)
//
// Patterns named in the implementation: status bar (VS Code / JetBrains),
// command bar of key buttons (Bloomberg command line and function keys),
// dense tables with keyboard navigation and one accent for selection
// (Bloomberg, W&B run tables), docked panes (Launchpad, Blender areas),
// overview first / details on demand (Shneiderman), honest disabled actions.
// ============================================================================

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace caliper_host { class AppletLoader; struct AppletEntry; }
namespace caliper_live { class SessionList; }

// What the Inspector shows of a spec applet, read from its spec by the app
// (src/host/builder/spec_page.h): the spec's format version (caliper_spec)
// and its numbers of panels and sources, or why the spec does not read.
struct SpecAppletFacts {
    long long format = -1;
    size_t panels = 0;
    size_t sources = 0;
    std::string error;
};

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
        // Called after the command bar and the status bar, before the dock
        // space: the host's jobs tray takes its strip above the status bar
        // here, and the panes fill what is left.
        std::function<void()> after_bars;
        // Attach to a network stream by its link (wire protocol v3): the ws://
        // link, or the http:// page link that caliper.serve() prints. Null
        // hides the link field under the session list.
        std::function<void(const std::string&)> attach_url;
        // A spec applet's facts for the Inspector, from its spec file. Null,
        // or false, leaves the rows that need them out.
        std::function<bool(const caliper_host::AppletEntry&, SpecAppletFacts*)> spec_facts;
        // F3 New dashboard, the Projects way in: a new spec titled `title`
        // ("" for one without a title), opened in build mode. The command
        // line's "new <title>" and Enter does the same. Null disables it.
        std::function<void(const std::string& title)> new_spec;
        // A file's path as the Inspector shows it: relative to the workspace
        // for a file in it, else with the home folder written "~", as the
        // Activity pane writes paths. Null shows the path as it is.
        std::function<std::string(const std::string&)> shown_path;
        // ⌘, Agent (Ctrl+, elsewhere): the widget agent's settings
        // (builder/agent_settings.h), at the command bar's right end. Null
        // leaves the button out.
        std::function<void()> agent_settings;
    };

    // Draws one frame of the landing page. Owns its windows and dock space;
    // the caller has already called renderer->new_frame().
    void draw(const Context& ctx);

    // Appends an Activity row (landing/final: WHEN / EVENT / OBJECT / DETAIL,
    // newest first, relative time). `event` is a short verb ("launch",
    // "scanned", "home"); `object` the thing it acted on; `detail` optional.
    void log(std::string event, std::string object = {},
             std::string detail = {});

    // The command line's text, as the line holds it (the hidden input
    // driver's `command-line` step reads it).
    const std::string& command_text() const { return command_; }

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
    void draw_inspector_rows(const caliper_host::AppletEntry& e, const Context& ctx);
    void draw_activity();
    void draw_status_bar(const Context& ctx);
    // Filters the loader by command_ and sorts by the table's sort spec.
    void rebuild_rows(const caliper_host::AppletLoader& loader);

    int selected_ = -1;
    std::function<void()> agent_settings_;   // Context::agent_settings, for the command bar
    std::string command_;            // command-line text (filters instruments)
    unsigned int command_id_ = 0;    // the command line's ImGuiID, for its edit state
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
    bool pending_new_ = false;       // F3 / "new <title>": a new spec after the frame
    std::string pending_new_title_;
    // F4 Attach to process: Projects lists the live sessions instead of the
    // ways in. The selection is the session token, so a relist keeps it.
    bool attach_mode_ = false;
    bool open_sessions_ = false;     // F4 / the button: enter attach mode after the frame
    std::string session_selected_;
    std::string pending_attach_;     // Enter / double-click: attach after the frame
    std::string stream_link_;        // the stream link typed or pasted under the list
    bool stream_link_active_ = false;   // the link field had the keys last frame
    std::string pending_attach_url_; // Attach link: attach after the frame
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
// A fact of the app's that every page's status bar shows after the machine,
// such as the address of Caliper API while it runs; "" shows nothing.
void set_host_status_note(std::function<std::string()> note);
