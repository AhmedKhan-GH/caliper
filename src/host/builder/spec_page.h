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
// person's UI, a command script, Caliper API, an agent over MCP), and every
// spec.new or spec.open of an agent (the side chat's or one over MCP), asks
// the app to show the page: take_open() is true once for each, read from a
// counter of the session's journal (AppSession::specs_opened()), and the app
// leaves the page it shows and enters this one, so that the person sees the
// spec an agent builds while they review its edits. request_open() is the
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
// Build mode (builder_pane.h, design/builder-pane.md). The page owns one
// BuilderPane over the session. ⌘B Build in the page bar, after the title,
// turns it on for the open spec: the pane lays the page out with
// layout_spec_page() (spec_page_layout.h), draws the runtime on its canvas,
// Sources and Primitives at the left, Properties at the right with the side
// chat under it while the chat is open, and the spec text under the canvas,
// and its other items (⌘T Text, ⌘Z Undo, ⇧⌘Z Redo, ⌘S Save) follow Build in
// the page bar, before ⌘L Chat and ⌘K Command at the right end. The focused
// panel is the page's in both modes, so ⌘K opens the inline prompt on the
// panel focused in build mode too. Files dropped on the window reach
// drop_files(), which turns build mode on and imports them from the ui
// origin.
//
// Frame thread only. Built with CALIPER_WITH_BUILDER; without it the page is
// a stub that never opens, so the app builds and behaves as before.
// ============================================================================
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct SpecAppletFacts;   // ui/landing_screen.h
namespace caliper_builder {
class Session;
}
struct CaliperCore;       // embed/embed_core.h: the core whose live views hold an attached script's scenes

namespace caliper_host {

class AgentSettings;
class AppSession;
class BuilderPane;
class ChatPane;
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

    // True once for each applet.open, or agent's spec.new or spec.open, that
    // succeeded since the last call.
    bool take_open(const AppSession& session);
    // Whether every open that the last true take_open() took was an agent's
    // (AppSession::agent_specs_opened()), not the person's own.
    bool opened_by_agent() const;

    // Sends applet.open for the workspace's spec applet `id` through the
    // session's queue, without waiting, from `origin`: "ui" for the person's
    // own launch, or the origin of the agent or program that asked ("api",
    // "mcp", "agent"), whose replacement of unsaved changes then waits on an
    // approval card as its own command would. The page opens when it
    // succeeds. False when the session is not open.
    bool request_open(AppSession& session, const std::string& id, const std::string& origin = "ui");
    // What an applet.open from `origin` would meet now, by the command
    // layer's own condition for replacing an open spec (review on for that
    // origin, and an open spec with unsaved changes): `opens` at once,
    // `waits` for the person's approval, or `unknown` while the session's
    // worker runs a command and the open spec cannot be read. It is the state
    // at the time of the call; the open runs a frame or two later.
    enum class OpenReview { opens, waits, unknown };
    static OpenReview open_review(AppSession& session, const std::string& origin);
    // An answer of the last request_open() or the last request_new(), once
    // it has come, one per call (spec.new's first): true, with `error` empty
    // when it succeeded and holding the reason otherwise, and `was_new` true
    // for spec.new's. The two are kept apart, so an open sent while a new
    // spec is on its way never drops the new spec's answer.
    bool request_answered(std::string* error, bool* was_new = nullptr);

    // A new spec from the app (the landing's F3 New dashboard, or a drop with
    // no open spec): spec.new from the ui origin through the session's queue,
    // with the applet id pane_new_spec_id() derives from `title` among the
    // workspace's saved applets. request_answered() reports it as it does an
    // open, and take_new() is true once after it succeeded; the app then
    // shows the page in build mode. False when the session is not open.
    bool request_new(AppSession& session, const std::string& title);
    bool take_new();
    // The session holds an open spec (spec.new or applet.open ran).
    static bool has_open_spec(AppSession& session);
    // ... and it has changes that are not saved (a new spec that was never
    // saved has).
    static bool has_unsaved_spec(AppSession& session);
    // The open spec's title, or its applet id when it has none; "" when no
    // spec is open or a long command holds the session.
    static std::string open_title(AppSession& session);
    // An import from drop_files() still runs.
    bool importing() const;

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
    // The side chat (chat_pane.h): its toggle in the page bar beside ⌘K
    // Command, its column after the page's strips, its key, and the inline
    // surface's "Ask in chat", which opens it with the handoff draft. Null
    // leaves the page without it.
    void set_chat(ChatPane* chat);
    // The widget agent's sidecar reaches the session's commands through an API
    // file of its own: issue() gives one with a token of the origin agent
    // (AppApi::issue_agent_file(); "" when the API is not running), and
    // revoke() ends it when the inline surface goes.
    struct ApiFiles {
        std::function<std::string()> issue;
        std::function<void(const std::string& file)> revoke;
    };
    void set_api_files(ApiFiles files);
    // Whether the app runs hidden (--hidden): the widget agent then never
    // touches the key store, and runs the sidecar only on a test's script
    // (builder/widget_app.h).
    void set_hidden(bool hidden);
    // The widget agent's settings (agent_settings.h): the inline surface's
    // Setup and Model open them, and its agent is made from them. A change
    // applies at once, on any page: the open prompt closes (a running turn is
    // cancelled, a preview discarded) and the agent goes with its sidecar and
    // its API token; the next ⌘K makes it again from the settings. The page
    // listens until shutdown(). Null: no settings.
    void set_agent_settings(AgentSettings* settings);
    // ⌘O Open in browser in the page bar: the saved dashboard the page shows,
    // in the person's default browser, at the spec page of Caliper API with
    // a viewer token (api_server.h, THE SPEC PAGE). The app does the opening
    // (src/main.cpp): `open` gets the dashboard's workspace id and returns ""
    // once the link has gone to the browser, or why not, which the page keeps
    // for the button's tooltip. Null leaves the button out. The button is
    // disabled, with the reason, for a dashboard that was never saved and for
    // a spec applet's file from outside the workspace.
    void set_open_in_browser(std::function<std::string(const std::string& applet_id)> open);
    // Where Properties reads the names of a catalog source's dims, such as a
    // live view's (AppProviders::dim_names()); none without it.
    void set_dim_names(std::function<std::vector<std::string>(const std::string& catalog_id)> names);
    // Where the scene panels of spec version 1 find their live scenes: the
    // core of the attached script (src/main.cpp gives its LiveAttachment's
    // core), asked at each frame. Without it every scene panel shows that
    // its source is offline.
    void set_live_core(std::function<CaliperCore*()> core);
    // The app's test seam for a pick in a scene panel: a click at (u, v),
    // each from 0 to 1 across the panel's content, at the next frame. Its
    // result is a pick event with the panel's id (design/instruments.md).
    // False when the open spec has no scene panel `panel_id`.
    bool pick_scene(const std::string& panel_id, float u, float v);
    // For the same seam: whether the scene panel `panel_id` of the open spec has drawn its scene,
    // its entities' own scene or the live scene that it binds.
    bool scene_ready(const std::string& panel_id);
    // For the hidden app's input driver: a named rectangle of the page's own
    // ("page:browser"), in window points, as the last frame drew it.
    bool item_rect(const std::string& name, float rect[4]) const;
    // What the page says about ⌘O Open in browser in its status bar: that
    // the browser shows the saved dashboard, once a link was given out while
    // the open spec has unsaved changes; "" otherwise. For the input driver.
    std::string browser_note() const;
    // For the hidden app's input driver (builder/spec_ui_driver.h): a shown
    // panel's rectangle in window points, and the inline surface's state
    // ("idle" before any ⌘K).
    bool panel_rect(const std::string& panel_id, float rect[4]) const;
    std::string widget_state() const;
    // What the inline surface shows besides its state, for the driver's log:
    // its tag, its edits, the model, an error's code and message, the input
    // tokens of each request of the last turn and its output tokens, and the
    // start of the reply, such as "tag LIMIT, 1 edit, model qwen, reply ...";
    // "" before any ⌘K.
    std::string widget_facts() const;
    // Presses the inline surface's action of that label ("Setup", "Accept",
    // ...) when the surface offers it now, as a click on its button does;
    // false otherwise. For the hidden app's input driver.
    bool widget_action(const std::string& label);
    // The spec runtime's selection as {"panel", "kind", "x0", "x1", "y0",
    // "y1"}, or "" when there is none.
    std::string selection_json() const;
    // Draws over each shown panel in the panel's own window after its
    // content, as the page's overlay does the focus frame, with the panel's
    // id, the title its strip shows, its rectangle and its title strip's
    // height (0 for a tab): the approval cards mark the panel a waiting edit
    // changes (approval_cards.h).
    using PanelOverlay = std::function<void(const std::string& panel_id, const std::string& title, float x, float y,
                                            float w, float h, float strip)>;
    void set_panel_overlay(PanelOverlay overlay);
    // Build mode, for the open spec the page shows (not for a spec applet's
    // file drawn from outside the workspace).
    bool build_mode() const;
    void set_build_mode(bool on);
    // Files dropped on the window: build mode on, then data.import of the
    // paths from the ui origin through the builder pane, which selects the
    // sources they make. False, with the reason, when the page shows no open
    // spec or another import still runs.
    bool drop_files(AppSession& session, const std::vector<std::string>& paths, std::string* why);
    // The builder pane, once the page has a session; null before and
    // without the builder. For the hidden app's input driver and tests.
    BuilderPane* builder();
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
    // without the builder. With the app's builder session, a panel's
    // annotations, cursor line and bound texts read its values.
    // `session` gives the adopted session on the frame thread, or null.
    static std::shared_ptr<PanelRenderSeam> make_panel_renderer(
        std::function<caliper_builder::Session*()> session = nullptr);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
