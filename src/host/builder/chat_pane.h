#pragma once
// ============================================================================
// chat_pane.h: the side chat in the app (design/acp-integration.md, The side
// chat as built; design/builder-overview.md, the placement sketch). The
// component is side_chat.h; this is the app's part of it, owned once by the
// app and shown on the spec page and the live page:
//
//   - The toggle. "Chat" with its key, ⌘L (Ctrl+L elsewhere), in the page
//     bar beside ⌘K Command, outlined in bronze while the pane is open. The
//     key works on the pages that have the pane.
//   - The pane. The right column of the window, claimed as a viewport side
//     bar after the host's strips (the page bar, the status bar, the jobs tray
//     and the approval cards), so the strips keep their places at the window's
//     edges and every page fills the work area that is left (work_area.h).
//     Its width is layout_chat_column()'s: a share of the window with a
//     minimum that grows with the text scale; the column's upper part
//     (Selection and Properties) and the builder's left column are not open
//     yet, so the chat takes the whole column.
//   - The agent. The chat is made when the pane first opens, with the
//     workspace of the app's session (opened then if it is not), this
//     build's caliper-mcp and the settings below. Its agent starts with the
//     first message; Caliper API must run for caliper-mcp, so the pane asks
//     the app to start it then (Hooks::start_api) when the app was started
//     without --api: on a free port of 127.0.0.1 only, with a new session
//     token and the API file, as --api does; no setting chooses another
//     address. The chat then says so in a line of its own ("Caliper API
//     started on 127.0.0.1:<port> for this chat"), the app writes an
//     Activity line and shows the API in the status bar, and when the chat's
//     agent ends the app stops the API it started for it
//     (Hooks::agent_ended). An API started with --api is left alone.
//   - The settings. side-chat.json in the app-data folder: the chosen agent's
//     id, the custom agent's name and command line, whether an agent's
//     allow_always options are offered, the agents marked as running on a
//     local model (the picker's Local model row), and whether the pane is
//     open. Never a key: the agent logs in on its own.
//   - The GPU note. "Shared GPU" in the chat's header while a live run is
//     attached (Hooks::live_attached) and the chosen agent is marked as
//     running on a local model (design/widget-agent.md, GPU contention).
//   - The meeting points. A command approval of the agent's turn is a row in
//     the conversation that points to its card on the approval cards' strip
//     (Hooks::show_approval); "Ask in chat" from the ⌘K surface opens the pane
//     with the handoff draft (handoff()).
//
// Frame thread only. Without CALIPER_WITH_BUILDER the pane is a stub that
// never opens, so the app builds and behaves as before.
// ============================================================================
#include <functional>
#include <memory>
#include <string>

namespace caliper_host {

class AppSession;

class ChatPane {
public:
    struct Hooks {
        // Starts Caliper API when it is not running, and returns an API file of
        // the agent's own, with a token of the origin agent issued for this
        // session (AppApi::issue_agent_file()), or "" with the reason;
        // `started` gets its address (127.0.0.1:<port>) when this call
        // started the API.
        std::function<std::string(std::string* why, std::string* started)> start_api;
        // Ends a file that start_api() gave: its token is revoked and the
        // file removed (AppApi::revoke_agent_file()). The pane calls it when
        // the agent ends, before a restart's new file, and at shutdown.
        std::function<void(const std::string& file)> end_api_file;
        // The chat's agent has ended (side_chat.h, Hooks::agent_ended): the app
        // stops the API it started for the chat.
        std::function<void()> agent_ended;
        // A row of the conversation was clicked: point at that approval's card.
        std::function<void(const std::string& approval)> show_approval;
        // One Activity line.
        std::function<void(const std::string& event, const std::string& object, const std::string& detail)> log;
        // A live run is attached (the app's live session): with the agent
        // marked as running on a local model, the header's GPU note.
        std::function<bool()> live_attached;
    };
    struct Options {
        std::string settings_path;   // side-chat.json in the app-data folder
        std::string exe_dir;         // the executable's folder, where caliper-mcp ships
        std::string path;            // PATH for finding the agents; this process's when empty
    };

    ChatPane();
    ~ChatPane();
    ChatPane(const ChatPane&) = delete;
    ChatPane& operator=(const ChatPane&) = delete;

    // False when the builder is not in this build.
    static bool built();

    void start(Options options, Hooks hooks);
    // Once per frame after AppSession::frame(): makes the chat when the pane
    // first opens, polls its agent, and reads the pending approvals of the
    // session while no long command holds its registry.
    void frame(AppSession& app);

    bool open() const;
    void set_open(bool open);
    void toggle() { set_open(!open()); }

    // In a page bar (a menu bar): the toggle, at the cursor.
    void draw_toggle();
    // The toggle's width in the page bar, to align it at the right.
    static float toggle_width();
    // ⌘L (Ctrl+L elsewhere) toggles the pane; inside a frame.
    void read_keys();
    // After the page's bars and strips, before its content: the right
    // column as a viewport side bar, with the chat in it, while open.
    void draw_column();
    // The chat in a rectangle the page gives, in its current window, while
    // open: in build mode the builder pane's right column holds Properties
    // above and the chat below (BuilderPaneLayout::chat_rect), and the page
    // calls this in place of draw_column().
    void draw_in(float x, float y, float w, float h);

    // For the hidden app's input driver (spec_ui_driver.h): the chat's state
    // ("none" before it is made), whether it has an item of a kind (with a
    // command, and a tool call's status, when given), and focus on its input.
    std::string state() const;
    bool has_item(const std::string& kind, const std::string& command = {}, const std::string& status = {}) const;
    void focus_input();
    // The chips of the draft in the input (a handoff's context), 0 for none.
    int draft_chips() const;
    // The header's GPU note ("Shared GPU"), "" for none or before the chat is made.
    std::string note() const;
    // The picker's Local model row for the chosen agent, as a click on it
    // does (SideChat::set_local_model()), and the picker opened on the next
    // draw; false before the chat is made or with no agent.
    bool set_local_model(bool on);
    void open_picker();

    // Before the session and the UI context go: stops the agent.
    void shutdown();

    struct Impl;
    Impl& impl() { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host

#if CALIPER_WITH_BUILDER

#include "builder/side_chat.h"

namespace caliper_host {

// The pane's settings file (side-chat.json): the chat's settings and whether
// the pane is open. read_chat_settings() leaves the defaults for a file that
// is missing or does not parse (false, with the reason, for the second).
struct ChatPaneSettings {
    ChatSettings chat;
    bool open = false;
};
bool read_chat_settings(const std::string& path, ChatPaneSettings* out, std::string* why = nullptr);
bool write_chat_settings(const std::string& path, const ChatPaneSettings& settings, std::string* why = nullptr);

// "Ask in chat": opens the pane with the draft of the widget request and the
// context bundle the inline surface had (chat_handoff_draft()).
void chat_pane_handoff(ChatPane& pane, const WidgetChatRequest& request, const caliper_builder::ContextBundle& bundle,
                       const caliper_builder::spec::Spec* open_spec);
// The chat, once it is made; null before.
SideChat* chat_pane_chat(ChatPane& pane);

}  // namespace caliper_host

#endif
