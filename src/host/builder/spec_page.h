#pragma once
// ============================================================================
// spec_page.h: the spec page of the app and of `caliper shot`
// (design/builder-session.md, The spec page). It draws the builder session's
// open spec with the spec runtime (spec_runtime/spec_runtime.h) in the work
// area that the host's strips leave: the page bar with Home and the spec's
// title where an applet's name goes, the status bar with the machine and the
// page's facts (the workspace, the number of panels and of sources), and the
// jobs tray's strip above the status bar while it is open.
//
// When it opens. Every applet.open that succeeds, from any origin (the
// person's UI, a command script, Caliper API, an agent over MCP), asks the
// app to show the page: take_open() is true once for each, read from a
// counter of the session's journal (AppSession::applets_opened()), and the
// app leaves the page it shows and enters this one. request_open() is the
// seam for a list of spec applets, such as the landing screen's later: it
// sends applet.open from the ui origin through the session's queue, so the
// page opens by the same path.
//
// Its data. The page reads the open spec again only when the session's
// journal has recorded a command since it last looked
// (AppSession::journaled()), and hands it to the runtime, which resolves only
// the panels whose fragment changed, and every panel again when the
// catalog's revision grows (a loader ran). Nothing is resolved per frame.
// While a long command runs on the session's worker, the page reads neither
// the session nor its catalog or store: it draws what it has, a panel that
// must be resolved meanwhile waits in a "Data busy" state, and every panel is
// resolved again once the worker is done.
//
// panel.render. make_panel_renderer() gives the session its PanelRenderSeam
// (panel_renderer.h): the runtime's OffscreenPanelRenderer, whose canvas is
// made on the first call, on the frame thread between two frames while the
// caller's ImGui context is current, so it shares that context's font atlas
// and draws in the house faces the app loaded.
//
// Frame thread only. Built with CALIPER_WITH_BUILDER; without it the page is
// a stub that never opens, so the app builds and behaves as before.
// ============================================================================
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

struct SpecAppletFacts;   // ui/landing_screen.h

namespace caliper_host {

class AppSession;
class JobsTray;
class PanelRenderSeam;

// How the spec page routes ⌘K (Ctrl K elsewhere) and Escape, one meaning
// each. ⌘K opens the command input of the page: with a panel focused, the
// inline prompt inside that panel, which is the widget agent's surface;
// with none, the page's own command input, as ⌘K opens the landing screen's.
// Escape first gives the panel's focus up, and goes Home only when no panel
// has it. A text field that has the keyboard, such as an open prompt, takes
// Escape before the page does (the app's Escape test leaves it alone then).
// A click inside a panel focuses it, and the page draws its frame in bronze.
enum class SpecKeyAction { none, panel_prompt, page_command, unfocus, home };

inline SpecKeyAction spec_key_action(bool command_key, bool escape, bool panel_focused) {
    if (command_key) return panel_focused ? SpecKeyAction::panel_prompt : SpecKeyAction::page_command;
    if (escape) return panel_focused ? SpecKeyAction::unfocus : SpecKeyAction::home;
    return SpecKeyAction::none;
}

class SpecPage {
public:
    struct Result {
        bool home = false;   // the page bar's Home was clicked
    };

    SpecPage();
    ~SpecPage();
    SpecPage(const SpecPage&) = delete;
    SpecPage& operator=(const SpecPage&) = delete;

    // False when the builder is not in this build.
    static bool built();

    // True once for each applet.open that succeeded since the last call.
    bool take_open(const AppSession& session);

    // Sends applet.open for the workspace's spec applet `id` from the ui
    // origin through the session's queue, without waiting; the page opens
    // when it succeeds. False when the session is not open.
    bool request_open(AppSession& session, const std::string& id);
    // The answer of the last request_open(), once it has come: true, with
    // `error` empty when it succeeded and holding the reason otherwise.
    bool request_answered(std::string* error);

    // Shows the page from the next draw() on, with the session's open spec.
    void enter(AppSession& session);
    // Shows the page with the spec of a spec applet found outside the
    // workspace, read from `spec_path` and drawn as it is, over the
    // session's data, without becoming the open spec; an applet.open later
    // shows the open spec again. False, with the reason, when the file does
    // not read or validate.
    bool enter_file(AppSession& session, const std::string& spec_path, std::string* error);
    // One frame of the page, inside an ImGui frame the caller opened.
    // `escape`: Escape was pressed and nothing else of the app takes it.
    Result draw(AppSession& session, const char* renderer_name, JobsTray* tray, bool escape = false);

    // What ⌘K opens (spec_key_action): the inline prompt inside the focused
    // panel, which the widget agent's surface sets, and the page's command
    // input. Unset, ⌘K does nothing there yet.
    struct Seams {
        std::function<void(const std::string& panel_id)> panel_prompt;
        std::function<void()> page_command;
    };
    void set_seams(Seams seams);
    // Whether the app runs hidden (--hidden): the widget agent then never
    // touches the key store, and runs the sidecar only on a test's script
    // (builder/widget_app.h).
    void set_hidden(bool hidden);
    // For the hidden app's input driver (builder/spec_ui_driver.h): a shown
    // panel's rectangle in window points, and the inline surface's state
    // ("idle" before any ⌘K).
    bool panel_rect(const std::string& panel_id, float rect[4]) const;
    std::string widget_state() const;
    // The focused panel's id, "" for none; focus_panel("") gives it up.
    const std::string& focused_panel() const;
    void focus_panel(const std::string& panel_id);
    // After the last frame that drew the page has been rendered: releases
    // the page's textures and forgets the spec.
    void leave();
    // The open spec's title, or its applet id when it has none.
    const std::string& title() const;
    // The shown spec's applet id ([applet].id), "" when none is shown.
    const std::string& applet_id() const;

    // What the landing screen's Inspector shows of a spec applet: its format
    // version and its numbers of panels and sources, read from its spec file
    // and kept until the file changes. False without the builder.
    static bool facts(const std::string& spec_path, SpecAppletFacts* out);

    // After the renderer has shut down and before the UI context is
    // destroyed: frees what the page made in that context.
    void shutdown();

    // panel.render's renderer for AppSession::set_panel_renderer(); null
    // without the builder.
    static std::shared_ptr<PanelRenderSeam> make_panel_renderer();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
