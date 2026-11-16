#pragma once
// ============================================================================
// side_chat.h: the side chat (design/acp-integration.md, The design of the
// side chat; design/builder-overview.md, item 8 and the placement sketch).
// The person's own agent (Claude Code through its adapter, Codex, Gemini
// CLI, opencode, or any ACP agent given by its command) runs as a child
// process over AcpClient, with this build's caliper-mcp handed to it in
// session/new, and the pane shows the conversation: the person's messages,
// the agent's streamed text, its thoughts collapsed to one line, its tool
// calls as compact rows (a Caliper tool's call names its command), its plan,
// its own permission requests as cards with their diff and options, and the
// stop reason of each turn in a word. A stalled agent, an agent that exited,
// one that needs a login and a protocol fault each have a clear state, with
// Cancel while a turn runs and Restart where the agent must start again.
//
// Four parts, so that each is tested without a window:
//   - the agents: detect_chat_agents() finds the four presets of
//     caliper_builder::acp::known_agents() by looking for their commands on
//     PATH, without running anything; custom_chat_agent() splits a custom
//     command line as a shell would split words, and runs it without one;
//     build_chat_launch() makes the client's options and session/new's
//     arguments: the workspace as the working folder and the session's cwd,
//     and one MCP server, caliper-mcp --origin agent --api-file <the app's
//     API file>, by absolute path. AcpClient declines the file-system and
//     terminal capabilities itself.
//   - SideChat, the state machine: no agent, idle, starting, ready, running,
//     cancelling, needs login, exited and failed, with a stalled flag while a
//     turn runs. Input events (choose an agent, the text, send, cancel,
//     restart, answer a permission, a handoff draft) and the client's events,
//     polled on the frame thread in update(), drive it. start() and
//     session/new block, so they run on a worker thread; shutting a client
//     down also runs on one, so the frame thread never waits for an agent.
//   - layout_chat_column() and layout_chat_pane(): the right column beside
//     the canvas and the pane's header, conversation, draft and input row as
//     plain rectangles, computed without ImGui from text widths the caller
//     measures.
//   - draw(): the ImGui side on the frame thread, in the house style.
//
// What the page owns: one SideChat, made with the workspace, this build's
// caliper-mcp and the app's API file. Each frame it calls update() (polls
// the agent; cheap when nothing came), claims the right column's lower part
// with layout_chat_column(), and calls draw() with that rectangle. The
// handoff from ⌘K is open_handoff() with chat_handoff_draft(), which puts a
// draft in the input for the person to send; nothing is sent by itself.
//
// Command approvals. A command the agent sends through caliper-mcp that
// needs the person's approval is held by the command layer, and the app's
// approval cards (approval_cards.h, another component) show it in their
// strip; the side chat draws no second card. While a turn runs it mirrors
// each pending approval of the origin `agent` as one row in the conversation,
// at the point where it appeared, with the command and "waiting", then
// "decided": Hooks::pending_approvals lists them (the registry's pending
// approvals), and a click on the row calls Hooks::show_approval, which points
// at the card.
//
// Labels are nouns, as the UI principles ask: Send, Cancel, Restart, Custom
// command, Clear; states are tags: RUNNING, STALLED, EXITED, LOGIN, FAILED,
// NO AGENT, PERMISSION, APPROVAL, PLAN, THOUGHT, and the stop words DONE,
// STOPPED, DECLINED, LIMIT. The pane writes no sentences of its own; the
// agent's text, its option names and its login methods are the agent's
// content.
//
// Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include "builder/widget_surface.h"

#include <caliper_builder/acp/acp_client.h>
#include <caliper_builder/acp/agents.h>
#include <caliper_builder/commands/approval.h>
#include <caliper_builder/error.h>
#include <caliper_builder/session/context_bundle.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct ImRect;

namespace caliper_host {

struct SpecLook;

// ---- the agents ---------------------------------------------------------------------------

// One agent the person can pick: a detected preset or their custom command.
struct ChatAgent {
    std::string id;                    // claude, codex, gemini, opencode; "custom" for the person's command
    std::string title;                 // "Claude Code", ...; the custom agent's name
    std::vector<std::string> argv;     // the launch command, argv[0] absolute once found on PATH
    std::vector<std::string> needs;    // the commands looked for on PATH
    std::string login_hint;            // the preset's, empty for a custom agent
    bool custom = false;
    bool found = false;                // every command of `needs` is on PATH
};

// The commands a preset needs on PATH: its launch command, and for an agent
// started through an adapter the agent's own command as well, since the
// adapter's launcher (npx) alone says nothing about the agent being there:
// claude needs npx and claude, codex npx and codex, gemini gemini, opencode
// opencode.
std::vector<std::string> chat_agent_needs(const caliper_builder::acp::AgentPreset& preset);

// The four presets, each marked found when every command it needs is on
// `path_list` (a PATH value). Nothing is run.
std::vector<ChatAgent> detect_chat_agents(const std::string& path_list);

// A custom agent from one command line (split_command_line(): words as a
// shell splits them, no expansion), run without a shell; its program is
// looked for on `path_list` when it has no slash. acp.bad_command when the
// line does not split.
caliper_builder::Error custom_chat_agent(const std::string& name, const std::string& line, const std::string& path_list,
                                         ChatAgent* out);

// What the side chat needs of the app to start an agent.
struct ChatEnvironment {
    std::string workspace;     // absolute: the agent's working folder and session/new's cwd
    std::string mcp_command;   // this build's caliper-mcp, absolute (find_caliper_mcp())
    std::string api_file;      // the app's API file (AppApi::file()), which caliper-mcp reads
    std::string path;          // PATH for detection and for the agent; this process's when empty
};

// The side chat's settings. They are the app's own, never a workspace's,
// because a workspace from elsewhere must not choose a program to run.
struct ChatSettings {
    std::string agent;                         // the chosen agent's id
    std::string custom_name = "Custom agent";  // the custom agent's name in the picker
    std::string custom_command;                // one line; empty: no custom agent
    bool allow_always = false;                 // offer an agent's allow_always options
    bool local_model = false;                  // the agent runs on a local model
};

// The MCP server's name in session/new, and so the prefix agents give its tools.
inline constexpr const char* kChatMcpServerName = "caliper";

// The client's options and session/new's arguments for one agent.
struct ChatLaunch {
    caliper_builder::acp::AcpClientOptions client;   // agent_argv, agent_cwd, agent_env
    std::string cwd;                                  // session/new's cwd
    std::vector<caliper_builder::acp::McpServer> mcp_servers;
};

// Builds the launch: the agent's argument vector as it is (no shell), the
// workspace as the agent's working folder and the session's cwd, PATH in
// the agent's environment when `env.path` is given, and one MCP server,
// kChatMcpServerName = mcp_command --origin agent --api-file <api_file>.
// `base` gives the client's timeouts and limits. Errors: chat.no_agent (an
// agent with no command or not found), chat.no_workspace (not an absolute
// folder), chat.no_mcp (caliper-mcp not given or not there), chat.no_api (no
// API file: the app's API is off).
caliper_builder::Error build_chat_launch(const ChatAgent& agent, const ChatEnvironment& env, ChatLaunch* out,
                                         const caliper_builder::acp::AcpClientOptions& base = {});

// This build's caliper-mcp: beside the executable in `exe_dir`, else the
// build tree's (a development build), else "".
std::string find_caliper_mcp(const std::string& exe_dir);

// The command a tool call runs when it is a call of the Caliper MCP server's
// tool, as the agents name such calls in a title or in rawInput:
// "mcp__caliper__panel_create" (Claude Code), "caliper__panel_create",
// "caliper.panel_create", "caliper/panel_create", "caliper_panel_create",
// "panel_create (caliper MCP Server)", or rawInput {"server": "caliper",
// "tool": ...}; the tool name maps back to a command of the table
// ("panel.create"). "" for any other tool.
std::string caliper_command_of(const caliper_builder::acp::ToolCallInfo& call,
                               const std::string& server = kChatMcpServerName);

// ---- the handoff from ⌘K ------------------------------------------------------------------

// A draft in the input: the request, which the person can edit, and the
// context blocks that go with it, each shown as a chip.
struct ChatDraft {
    std::string text;
    nlohmann::json blocks = nlohmann::json::array();   // ACP content blocks, each for the assistant
    std::vector<std::string> chips;                    // one noun per block
    bool empty() const { return text.empty() && blocks.empty(); }
};

// The draft of a widget request handed to the chat (design/widget-agent.md,
// Handoff to the side chat): the context bundle's blocks as the inline
// surface had them, the widget conversation so far, the open preview's diff
// marked as not applied, and the target's ids (the panel, its sources and
// their catalog ids, read from `open_spec` when given), each block with
// annotations.audience ["assistant"]; the request is the draft's text.
ChatDraft chat_handoff_draft(const WidgetChatRequest& request, const caliper_builder::ContextBundle& bundle,
                             const caliper_builder::spec::Spec* open_spec = nullptr);

// The blocks of one prompt: the draft's blocks shaped for the agent (a
// resource block becomes a text block unless the agent declared
// promptCapabilities.embeddedContext; an image is left out unless it declared
// promptCapabilities.image, and named in `dropped`), then the person's text
// last, since the context says that only the request says what to do.
nlohmann::json chat_prompt_blocks(const std::string& text, const ChatDraft* draft,
                                  const caliper_builder::acp::AgentInfo& agent,
                                  std::vector<std::string>* dropped = nullptr);

// The opening context of a session, sent once before its first message as a
// block for the assistant: change specs through the Caliper tools, and do not
// ask before a Caliper command, since Caliper reviews them itself.
nlohmann::json chat_opening_block();

// ---- the conversation ---------------------------------------------------------------------

enum class ChatState { no_agent, idle, starting, ready, running, cancelling, needs_login, exited, failed };
const char* chat_state_name(ChatState state);

enum class ChatAction { send, cancel, restart, custom, clear };
const char* chat_action_label(ChatAction action);   // the noun on the pane
const char* chat_action_key(ChatAction action);     // its key ("⏎", "esc") or ""

// The stop reason of a turn in a word: DONE, STOPPED, DECLINED, LIMIT, or
// FAILED for a turn that ended in an error.
std::string chat_stop_word(const std::string& stop_reason, const caliper_builder::Error& error);

// The status of a tool call in a word: pending, running, done, failed, stopped.
std::string chat_tool_status_word(const std::string& status);

struct ChatItem {
    enum class Kind { person, agent, thought, tool, plan, permission, approval, stop, notice };
    Kind kind = Kind::agent;
    int turn = 0;
    std::string text;      // person, agent, thought: the text; stop, notice: the tag
    std::string detail;    // stop: the error code; notice: the error code or the exit status
    std::vector<std::string> chips;                 // person: the context sent with it
    std::vector<std::string> lines;                 // notice: the agent's stderr tail, its login methods
    caliper_builder::acp::ToolCallInfo tool;        // tool, permission: the call, updates merged
    std::string command;                            // tool, permission: the Caliper command, or ""
    std::vector<caliper_builder::acp::PlanEntry> plan;
    std::uint64_t permission = 0;
    std::vector<caliper_builder::acp::PermissionOption> options;   // the options offered
    std::string answer;    // permission: the option's name, or "withdrawn"; approval: "decided"; "" while open
    std::string diff;      // permission, tool: the unified diff of its diff content
    std::string diff_path; // the diff's file
    std::string approval;  // approval: its id
    bool expanded = false; // thought: shown in full
};

class SideChat {
public:
    struct Options {
        ChatEnvironment env;
        ChatSettings settings;
        // The agents to offer: detect_chat_agents() and, when the settings
        // hold one, the custom agent. Empty: made from env.path and settings.
        std::vector<ChatAgent> agents;
        caliper_builder::acp::AcpClientOptions client;   // timeouts and limits; the command comes from the agent
        bool opening_prompt = true;                       // chat_opening_block() before a session's first message
    };
    struct Hooks {
        // The command layer's pending approvals (the registry's); the chat
        // mirrors those of the origin agent while a turn runs.
        std::function<std::vector<caliper_builder::PendingApproval>()> pending_approvals;
        // A click on an approval's row: the app points at its card in the
        // approval cards' strip.
        std::function<void(const std::string& approval)> show_approval;
        // The settings changed (an agent chosen, a custom command): the app saves them.
        std::function<void(const ChatSettings&)> settings_changed;
    };

    SideChat(Options options, Hooks hooks = {});
    ~SideChat();
    SideChat(const SideChat&) = delete;
    SideChat& operator=(const SideChat&) = delete;

    // ---- input events
    // Picks an agent by id; a running agent is stopped first. False for an
    // unknown or missing one.
    bool choose_agent(const std::string& id);
    // Sets the custom agent's name and command line and picks it.
    caliper_builder::Error set_custom_agent(const std::string& name, const std::string& line);
    void set_text(std::string text);
    // Enter: sends the text and the draft's blocks. The agent starts first
    // when it is not running, and the message waits for its session.
    void send();
    void cancel();           // running: session/cancel
    void restart();          // a new process and a new session; the conversation stays
    void answer_permission(std::uint64_t permission, const std::string& option_id);
    // The handoff: the draft goes into the input, to be sent by the person.
    void open_handoff(ChatDraft draft);
    void clear_draft();
    void act(ChatAction action);

    // ---- per frame: the start worker, the agent's events and the mirrored approvals
    void update();

    // ---- what is shown
    ChatState state() const { return state_; }
    bool stalled() const { return stalled_; }
    const std::vector<ChatItem>& items() const { return items_; }
    const std::vector<ChatAgent>& agents() const { return agents_; }
    const ChatAgent* agent() const;          // the chosen agent, or null
    const ChatSettings& settings() const { return options_.settings; }
    const std::string& text() const { return text_; }
    const ChatDraft& draft() const { return draft_; }
    const caliper_builder::acp::AgentInfo& agent_info() const { return info_; }
    const caliper_builder::Error& error() const { return error_; }
    const std::string& mode() const { return mode_; }   // the agent's current mode id, "" when unknown
    // True for a mode that skips the agent's permission prompts.
    bool mode_skips_permission() const;
    const std::vector<caliper_builder::Error>& faults() const { return faults_; }
    int turns() const { return turn_; }
    // The last turn's prompt as sent, for tests.
    const nlohmann::json& last_prompt() const { return last_prompt_; }
    // The agent's name for the picker and the header: agentInfo's title, else the agent's.
    std::string agent_title() const;
    // The tag of the state ("RUNNING", ...) or "".
    std::string tag() const;
    // The header's actions, and the input row's (Send, or Cancel while a turn runs).
    std::vector<ChatAction> header_actions() const;
    ChatAction input_action() const;

    // ---- the ImGui side (frame thread)
    // Draws the pane into `pane` of the current window.
    void draw(const ImRect& pane, const SpecLook& look);
    // Opens the agent picker on the next draw (the captures show it).
    void open_picker() { picker_open_ = true; }
    void focus_input() { focus_input_ = true; }

    struct StartJob;

private:
    void start_agent();
    void retire_client();
    void send_now();
    void on_event(const caliper_builder::acp::AcpEvent& e);
    void end_turn(const caliper_builder::acp::AcpEvent& e);
    void finish_start();
    void mirror_approvals();
    ChatItem& add(ChatItem::Kind kind);
    ChatItem* find_tool(const std::string& id);
    ChatItem* last_of(ChatItem::Kind kind);
    void notice(const std::string& tag, const caliper_builder::Error& error, std::vector<std::string> lines = {});
    void refresh_agents();

    Options options_;
    Hooks hooks_;
    std::vector<ChatAgent> agents_;
    ChatState state_ = ChatState::no_agent;
    bool stalled_ = false;
    bool offer_restart_ = false;   // a cancel the agent never answered
    std::vector<ChatItem> items_;
    int turn_ = 0;
    std::string text_;
    ChatDraft draft_;
    // The message that waits for the agent's session.
    bool pending_send_ = false;
    std::string pending_text_;
    ChatDraft pending_draft_;
    bool opened_ = false;          // the session's first message has gone
    nlohmann::json last_prompt_;
    caliper_builder::acp::AgentInfo info_;
    caliper_builder::Error error_;
    std::string mode_;
    std::vector<caliper_builder::Error> faults_;
    std::shared_ptr<caliper_builder::acp::AcpClient> client_;
    std::shared_ptr<StartJob> job_;
    std::vector<std::thread> workers_;   // start and shutdown threads, joined at the end

    // ImGui side
    bool picker_open_ = false;
    bool focus_input_ = false;
    bool custom_open_ = false;
    std::string custom_line_, custom_name_;
    caliper_builder::Error custom_error_;
    bool diagnostics_open_ = false;
    size_t drawn_items_ = 0;
    bool follow_ = true;
};

// ---- the layout ---------------------------------------------------------------------------

// The sizes of the pane's rows, which grow with the text scale.
struct ChatLayoutMetrics {
    float header = 26;         // the header strip
    float row = 30;            // the input row at one line
    float line = 18;           // one more line of the input
    float chip = 22;           // the draft's chips
    float pad = 8;             // inside the pane, left and right
    float gap = 6;             // between buttons and chips
    float button = 22;         // a button's height
    float button_pad = 8;      // inside a button, left and right
    float key_gap = 6;         // between a button's key and its label
    float picker_mark = 14;    // the picker's ▾ mark and its gap
    float min_input = 120;     // the narrowest input before the picker's name goes
    float min_width = 300;     // the column's minimum width
    float min_height = 220;    // the pane's minimum height
    float min_canvas = 480;    // the canvas's minimum width beside the column
    float min_upper = 160;     // the upper part's minimum height (Selection, Properties)
    int max_input_lines = 6;
    static ChatLayoutMetrics for_text_scale(float s);
};

// The right column beside the canvas: the canvas keeps the rest of the
// window. With the upper part (Selection and Properties) open it takes
// `upper_share` of the column's height and the chat the rest; closed, the
// chat takes the whole column. What gives way as the window shrinks: the
// column's share down to its minimum, then the upper part down to its
// minimum and then away, so the chat keeps its minimum; the canvas gives
// way last, below the two minimums.
struct ChatColumnInput {
    WidgetRect window;          // the work area the host's strips leave
    float share = 0.28f;        // the column's share of the window's width
    bool upper_open = false;    // Selection or Properties has something to show
    float upper_share = 0.45f;
};
struct ChatColumnLayout {
    WidgetRect canvas;
    WidgetRect column;
    WidgetRect upper;           // empty when closed or given way
    WidgetRect chat;
};
ChatColumnLayout layout_chat_column(const ChatColumnInput& in, const ChatLayoutMetrics& m);

enum class ChatFont { label, key, mono, tag };
using ChatMeasure = std::function<float(const std::string& text, ChatFont font)>;

struct ChatPaneInput {
    WidgetRect pane;
    std::string title = "Chat";       // the pane's title, drawn as a tag
    std::string agent;                // the agent's name
    std::string version;              // its version, or ""
    std::string tag;                  // the state's tag, or ""
    std::vector<ChatAction> header_actions;
    int faults = 0;                   // the diagnostics' count; 0 shows none
    std::string picker;               // the picker's label: the agent's name
    int input_lines = 1;              // the lines of the input's text
    std::vector<std::string> chips;   // the draft's chips
    ChatAction input_action = ChatAction::send;
};

struct ChatPaneLayout {
    WidgetRect header;
    WidgetRect title;
    WidgetRect agent;                 // empty when collapsed
    WidgetRect version;               // empty when collapsed or none
    WidgetRect tag;                   // empty when none
    WidgetRect faults;                // empty when none
    std::vector<WidgetRect> header_buttons;   // in header_actions order
    WidgetRect conversation;
    WidgetRect chips;                 // the draft's row, empty when none
    std::vector<WidgetRect> chip_rects;
    int chips_shown = 0;              // the rest fold into "+N"
    WidgetRect clear;                 // the draft's Clear
    WidgetRect input_row;
    WidgetRect picker;
    bool picker_label = true;         // false: the ▾ mark only
    WidgetRect input;
    WidgetRect send;                  // Send or Cancel
    bool keys = true;                 // the buttons show their keys
};

// Lays the pane out. What collapses first as it narrows: the agent's version
// in the header, then the agent's name in the header, then the picker's name
// (the ▾ mark stays), then the buttons' keys; in the header the faults chip,
// then the state's tag, give way to the buttons when they need the room; the
// draft's chips past what fits fold into "+N". The conversation takes every
// row the others leave.
ChatPaneLayout layout_chat_pane(const ChatPaneInput& in, const ChatLayoutMetrics& m, const ChatMeasure& measure);

// The layout input of a chat's state for a pane.
ChatPaneInput chat_pane_input(const SideChat& chat, const WidgetRect& pane);

}  // namespace caliper_host
