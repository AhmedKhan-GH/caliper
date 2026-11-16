// The side chat's state machine against the stub agents, headless: a plain
// turn, a thought, a plan and streamed text, tool calls, a permission request
// allowed and denied, cancel, a stalled agent, an agent that exits in the
// middle of a turn, a wrong protocol version, an agent that needs a login,
// restart, protocol faults, the opening prompt, the mirrored command
// approvals, the handoff draft, and no agent found.
#include "builder/side_chat.h"

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <thread>

using namespace caliper_host;
namespace acp = caliper_builder::acp;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct Fixture {
    fs::path root, ws, api;
    explicit Fixture(const std::string& name) {
        root = fs::path(CALIPER_SPEC_TEST_TMP) / name;
        std::error_code ec;
        fs::remove_all(root, ec);
        ws = root / "workspace";
        fs::create_directories(ws);
        root = fs::canonical(root);
        ws = fs::canonical(ws);
        api = root / "api.json";
        std::ofstream(api) << "{}\n";
    }
    // A chat with a stub as the person's custom command.
    SideChat::Options options(const std::string& agent_args = "", const char* exe = CALIPER_ACP_STUB_AGENT) const {
        SideChat::Options o;
        o.env.workspace = ws.string();
        o.env.mcp_command = CALIPER_MCP_EXE;
        o.env.api_file = api.string();
        o.env.path = (root / "empty").string();
        o.settings.custom_name = "Stub agent";
        o.settings.custom_command = std::string("'") + exe + "'" + (agent_args.empty() ? "" : " " + agent_args);
        o.settings.agent = "custom";
        o.opening_prompt = false;
        o.client.request_timeout_ms = 20000;
        o.client.cancel_timeout_ms = 2000;
        return o;
    }
};

bool pump_until(SideChat& chat, const std::function<bool()>& cond, int ms = 20000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        chat.update();
        if (cond()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    chat.update();
    return cond();
}

bool turn_over(SideChat& chat) {
    return chat.state() == ChatState::ready && !chat.items().empty() && chat.items().back().kind == ChatItem::Kind::stop;
}

const ChatItem* find(const SideChat& chat, ChatItem::Kind kind, int turn = -1) {
    for (const ChatItem& it : chat.items())
        if (it.kind == kind && (turn < 0 || it.turn == turn)) return &it;
    return nullptr;
}

std::vector<ChatItem::Kind> kinds(const SideChat& chat, int turn) {
    std::vector<ChatItem::Kind> out;
    for (const ChatItem& it : chat.items())
        if (it.turn == turn) out.push_back(it.kind);
    return out;
}

void say(SideChat& chat, const std::string& text) {
    chat.set_text(text);
    chat.send();
}

using K = ChatItem::Kind;

}  // namespace

TEST_CASE("side chat: a plain turn starts the agent, sends the message and ends with its stop word") {
    Fixture f("chat-plain");
    SideChat chat(f.options());
    CHECK(chat.state() == ChatState::idle);
    CHECK(chat.agent_title() == "Stub agent");
    CHECK(chat.tag().empty());
    CHECK(chat.input_action() == ChatAction::send);
    say(chat, "  hi there  ");
    CHECK(chat.text().empty());
    CHECK(chat.state() == ChatState::starting);
    CHECK(chat.tag() == "STARTING");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(chat.turns() == 1);
    CHECK(kinds(chat, 1) == std::vector<K>{K::person, K::agent, K::stop});
    CHECK(chat.items()[0].text == "hi there");
    CHECK(chat.items()[1].text == "You said: hi there");
    CHECK(chat.items()[2].text == "DONE");
    CHECK(chat.agent_info().version == "1");
    CHECK(chat.mode() == "default");
    CHECK(chat.last_prompt() == json::array({{{"type", "text"}, {"text", "hi there"}}}));

    // The next message goes to the same session.
    say(chat, "again");
    CHECK(chat.state() == ChatState::running);
    CHECK(chat.tag() == "RUNNING");
    CHECK(chat.input_action() == ChatAction::cancel);
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(kinds(chat, 2) == std::vector<K>{K::person, K::agent, K::stop});

    // An empty message sends nothing.
    say(chat, "   ");
    CHECK(chat.turns() == 2);
}

TEST_CASE("side chat: a thought, a plan, streamed text and a mode change") {
    Fixture f("chat-hello");
    SideChat chat(f.options());
    say(chat, "hello");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(kinds(chat, 1) == std::vector<K>{K::person, K::thought, K::plan, K::agent, K::stop});
    CHECK(find(chat, K::thought)->text == "Thinking about hello.");
    REQUIRE(find(chat, K::plan)->plan.size() == 2);
    CHECK(find(chat, K::plan)->plan[0].content == "Say hello");
    CHECK(find(chat, K::plan)->plan[0].status == "in_progress");
    CHECK(find(chat, K::agent)->text == "Hi, world.");   // three chunks, one message
    CHECK(chat.mode() == "plan");
    CHECK_FALSE(chat.mode_skips_permission());
}

TEST_CASE("side chat: a tool call and its permission request, allowed") {
    Fixture f("chat-allow");
    SideChat chat(f.options());
    say(chat, "edit");
    REQUIRE(pump_until(chat, [&] { return find(chat, K::permission) != nullptr; }));
    const ChatItem* tool = find(chat, K::tool);
    REQUIRE(tool);
    CHECK(tool->tool.id == "call_1");
    CHECK(tool->tool.title == "Edit demo.spec.toml");
    CHECK(tool->tool.status == "pending");
    CHECK(tool->command.empty());   // the agent's own tool, not a Caliper command
    const ChatItem* p = find(chat, K::permission);
    CHECK(p->answer.empty());
    CHECK(p->diff_path == "demo.spec.toml");
    CHECK(p->diff.find("-title = \"Old\"") != std::string::npos);
    CHECK(p->diff.find("+title = \"New\"") != std::string::npos);
    // allow_always only when the settings allow it.
    REQUIRE(p->options.size() == 2);
    CHECK(p->options[0].kind == "allow_once");
    CHECK(p->options[1].kind == "reject_once");
    CHECK(chat.state() == ChatState::running);

    chat.answer_permission(p->permission, "allow");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(find(chat, K::permission)->answer == "Allow");
    CHECK(find(chat, K::tool)->tool.status == "completed");
    CHECK(find(chat, K::agent)->text == "Applied the change.");
    CHECK(chat.items().back().text == "DONE");
}

TEST_CASE("side chat: a permission request denied, and allow_always offered by the setting") {
    Fixture f("chat-deny");
    SideChat::Options o = f.options();
    o.settings.allow_always = true;
    SideChat chat(o);
    say(chat, "edit");
    REQUIRE(pump_until(chat, [&] { return find(chat, K::permission) != nullptr; }));
    const ChatItem* p = find(chat, K::permission);
    REQUIRE(p->options.size() == 3);
    CHECK(p->options[1].kind == "allow_always");
    chat.answer_permission(p->permission, "no-such-option");   // ignored
    CHECK(find(chat, K::permission)->answer.empty());
    chat.answer_permission(p->permission, "reject");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(find(chat, K::permission)->answer == "Reject");
    CHECK(find(chat, K::tool)->tool.status == "failed");
    CHECK(find(chat, K::agent)->text == "Left the spec unchanged.");
    CHECK(chat.items().back().text == "DONE");
}

TEST_CASE("side chat: cancel ends the turn as stopped, and withdraws an open permission card") {
    Fixture f("chat-cancel");
    SideChat chat(f.options());
    say(chat, "slow");
    REQUIRE(pump_until(chat, [&] { return find(chat, K::agent) != nullptr; }));
    chat.cancel();
    CHECK(chat.state() == ChatState::cancelling);
    CHECK(chat.tag() == "STOPPING");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(chat.items().back().text == "STOPPED");

    say(chat, "edit");
    REQUIRE(pump_until(chat, [&] { return find(chat, K::permission, 2) != nullptr; }));
    chat.cancel();
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(find(chat, K::permission, 2)->answer == "withdrawn");
    CHECK(chat.items().back().text == "STOPPED");
    CHECK(chat.header_actions().empty());
}

TEST_CASE("side chat: a stalled agent shows STALLED with Restart until it speaks again") {
    Fixture f("chat-stall");
    SideChat::Options o = f.options();
    o.client.stall_timeout_ms = 300;
    SideChat chat(o);
    say(chat, "stall");
    REQUIRE(pump_until(chat, [&] { return chat.stalled(); }));
    CHECK(chat.tag() == "STALLED");
    CHECK(chat.header_actions() == std::vector<ChatAction>{ChatAction::restart});
    CHECK(chat.input_action() == ChatAction::cancel);
    const ChatItem* n = find(chat, K::notice);
    REQUIRE(n);
    CHECK(n->text == "STALLED");
    CHECK(n->detail == "acp.stalled");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK_FALSE(chat.stalled());
    CHECK(chat.items().back().text == "DONE");
    CHECK(chat.header_actions().empty());
}

TEST_CASE("side chat: an agent that exits in the middle of a turn, then Restart") {
    Fixture f("chat-crash");
    SideChat chat(f.options());
    say(chat, "crash");
    REQUIRE(pump_until(chat, [&] { return chat.state() == ChatState::exited; }));
    CHECK(chat.tag() == "EXITED");
    CHECK(chat.header_actions() == std::vector<ChatAction>{ChatAction::restart});
    CHECK(chat.input_action() == ChatAction::send);
    CHECK(find(chat, K::agent)->text == "About to exit.");
    CHECK(find(chat, K::stop) == nullptr);   // the exit says it, not a stop word
    const ChatItem* n = find(chat, K::notice);
    REQUIRE(n);
    CHECK(n->text == "EXITED");
    CHECK(n->detail == "exit code 3");
    CHECK(chat.error().code == "acp.agent_exited");

    // Restart: a new process and a new session; the conversation stays.
    const size_t before = chat.items().size();
    chat.restart();
    CHECK(chat.state() == ChatState::starting);
    REQUIRE(pump_until(chat, [&] { return chat.state() == ChatState::ready; }));
    CHECK(chat.items().size() == before);
    say(chat, "back");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(chat.items()[chat.items().size() - 2].text == "You said: back");

    // A message sent to an exited agent starts it again.
    say(chat, "crash");
    REQUIRE(pump_until(chat, [&] { return chat.state() == ChatState::exited; }));
    say(chat, "once more");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(chat.items()[chat.items().size() - 2].text == "You said: once more");
}

TEST_CASE("side chat: an agent of another protocol version fails with Restart") {
    Fixture f("chat-version");
    SideChat chat(f.options("--version-reply 2"));
    say(chat, "hi");
    REQUIRE(pump_until(chat, [&] { return chat.state() == ChatState::failed; }));
    CHECK(chat.error().code == "acp.version_unsupported");
    CHECK(chat.tag() == "FAILED");
    CHECK(chat.header_actions() == std::vector<ChatAction>{ChatAction::restart});
    const ChatItem* n = find(chat, K::notice);
    REQUIRE(n);
    CHECK(n->text == "FAILED");
    CHECK(n->detail == "acp.version_unsupported");
    REQUIRE_FALSE(n->lines.empty());
    CHECK(n->lines[0].find("version 2") != std::string::npos);
}

TEST_CASE("side chat: an agent that needs a login shows its own methods and never asks for a key") {
    Fixture f("chat-login");
    SideChat chat(f.options("--auth-required"));
    say(chat, "hi");
    REQUIRE(pump_until(chat, [&] { return chat.state() == ChatState::needs_login; }));
    CHECK(chat.tag() == "LOGIN");
    CHECK(chat.error().code == "acp.auth_required");
    CHECK(chat.header_actions() == std::vector<ChatAction>{ChatAction::restart});
    const ChatItem* n = find(chat, K::notice);
    REQUIRE(n);
    CHECK(n->text == "LOGIN");
    REQUIRE_FALSE(n->lines.empty());
    CHECK(n->lines[0] == "Stub login");   // the agent's own method, by its name
    // The message waits: after the person logs in, Restart sends it.
    CHECK(chat.turns() == 1);
}

TEST_CASE("side chat: protocol faults go to the diagnostics, not the conversation") {
    Fixture f("chat-faults");
    SideChat chat(f.options());
    say(chat, "malformed");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(chat.faults().size() >= 5);
    CHECK(kinds(chat, 1) == std::vector<K>{K::person, K::agent, K::stop});
    CHECK(find(chat, K::agent)->text == "Still here.");
    CHECK(chat.items().back().text == "DONE");
}

TEST_CASE("side chat: the opening context goes once, before a session's first message, for the assistant") {
    Fixture f("chat-opening");
    SideChat::Options o = f.options();
    o.opening_prompt = true;
    SideChat chat(o);
    say(chat, "first");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    REQUIRE(chat.last_prompt().size() == 2);
    CHECK(chat.last_prompt()[0] == chat_opening_block());
    CHECK(chat.last_prompt()[0]["annotations"]["audience"] == json::array({"assistant"}));
    CHECK(chat.last_prompt()[0]["text"].get<std::string>().find("do not ask for permission") != std::string::npos);
    CHECK(chat.last_prompt()[1]["text"] == "first");
    say(chat, "second");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    CHECK(chat.last_prompt().size() == 1);
}

TEST_CASE("side chat: a pending command approval of the agent's turn is mirrored in its slot") {
    Fixture f("chat-approvals");
    std::vector<caliper_builder::PendingApproval> pending;
    SideChat::Hooks hooks;
    hooks.pending_approvals = [&pending] { return pending; };
    SideChat chat(f.options(), hooks);
    say(chat, "slow");
    REQUIRE(pump_until(chat, [&] { return chat.state() == ChatState::running; }));
    caliper_builder::PendingApproval a;
    a.approval = "ap-7";
    a.invocation.name = "panel.create";
    a.invocation.origin = caliper_builder::CommandOrigin::agent;
    caliper_builder::PendingApproval other = a;
    other.approval = "ap-8";
    other.invocation.origin = caliper_builder::CommandOrigin::mcp;   // an outside agent's: not this chat's
    pending = {a, other};
    chat.update();
    const ChatItem* slot = find(chat, K::approval);
    REQUIRE(slot);
    CHECK(slot->approval == "ap-7");
    CHECK(slot->command == "panel.create");
    CHECK(slot->answer.empty());
    int slots = 0;
    for (const ChatItem& it : chat.items()) slots += it.kind == K::approval;
    CHECK(slots == 1);
    pending.clear();   // the person decided on the approval card
    chat.update();
    CHECK(find(chat, K::approval)->answer == "decided");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
}

TEST_CASE("side chat: an approval reissued for changed content keeps its row waiting on the new approval") {
    Fixture f("chat-reissued");
    std::vector<caliper_builder::PendingApproval> pending;
    std::map<std::string, std::string> reissued;   // old approval -> new, as the queue records it
    SideChat::Hooks hooks;
    hooks.pending_approvals = [&pending] { return pending; };
    hooks.reissued_as = [&reissued](const std::string& id) {
        const auto it = reissued.find(id);
        return it == reissued.end() ? std::string() : it->second;
    };
    hooks.approval_outcome = [](const std::string& id) { return id == "ap-3" ? std::string("approved") : std::string(); };
    SideChat chat(f.options(), hooks);
    say(chat, "slow");
    REQUIRE(pump_until(chat, [&] { return chat.state() == ChatState::running; }));
    caliper_builder::PendingApproval a;
    a.approval = "ap-2";
    a.invocation.name = "applet.save";
    a.invocation.origin = caliper_builder::CommandOrigin::agent;
    pending = {a};
    chat.update();
    REQUIRE(find(chat, K::approval));
    CHECK(find(chat, K::approval)->approval == "ap-2");
    // The content changed while the card waited, and the person approved:
    // the approval is void and reissued as ap-3, which waits.
    a.approval = "ap-3";
    pending = {a};
    reissued["ap-2"] = "ap-3";
    chat.update();
    int rows = 0;
    for (const ChatItem& it : chat.items()) rows += it.kind == K::approval;
    CHECK(rows == 1);
    CHECK(find(chat, K::approval)->approval == "ap-3");
    CHECK(find(chat, K::approval)->answer.empty());
    // The person approves the new card.
    pending.clear();
    chat.update();
    CHECK(find(chat, K::approval)->answer == "approved");
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
}

TEST_CASE("side chat: the handoff from ⌘K is a draft the person sends, shaped for the agent") {
    WidgetChatRequest r;
    r.request = "Build a loader for the validation logs";
    r.target = WidgetTarget::for_panel("curves");
    r.conversation = {{"Use a linear y axis", "Turns off the log scale."}};
    r.diff = "--- before\n+++ after\n@@ -1 +1 @@\n-y_log = true\n+y_log = false\n";
    r.handoff = WidgetHandoff{"loader", "A loader is work for the side chat."};
    caliper_builder::ContextBundle bundle;
    bundle.blocks = json::array(
        {{{"type", "text"}, {"text", "Context for a widget request about the panel curves."}},
         {{"type", "resource"},
          {"resource", {{"uri", "caliper://context/spec/curves"}, {"mimeType", "text/plain"}, {"text", "<caliper-data part=\"spec\">x</caliper-data>"}}}},
         {{"type", "text"}, {"text", "<caliper-data part=\"source\" about=\"1\">\nrows\n</caliper-data>"}},
         {{"type", "image"}, {"mimeType", "image/png"}, {"data", "iVBORw0KGgo="}}});
    caliper_builder::spec::Spec spec;
    spec.applet.id = "run42-dashboard";
    caliper_builder::spec::Source metrics;
    metrics.id = "metrics";
    metrics.catalog = "table:run42_metrics";
    spec.sources.push_back(metrics);
    caliper_builder::spec::Panel curves;
    curves.id = "curves";
    curves.bind["x"].bindings = {"metrics.step"};
    curves.bind["y"].bindings = {"metrics.loss", "metrics.accuracy"};
    spec.panels.push_back(curves);

    const ChatDraft d = chat_handoff_draft(r, bundle, &spec);
    CHECK(d.text == r.request);
    CHECK(d.chips == std::vector<std::string>{"context", "spec curves", "source 1", "render", "conversation", "preview diff", "ids"});
    REQUIRE(d.blocks.size() == 7);
    for (const json& b : d.blocks) CHECK(b["annotations"]["audience"] == json::array({"assistant"}));
    CHECK(d.blocks[4]["text"].get<std::string>().find("Turns off the log scale.") != std::string::npos);
    CHECK(d.blocks[4]["text"].get<std::string>().find("(loader): A loader is work for the side chat.") != std::string::npos);
    CHECK(d.blocks[5]["text"].get<std::string>().find("NOT APPLIED") != std::string::npos);
    CHECK(d.blocks[5]["text"].get<std::string>().find("+y_log = false") != std::string::npos);
    const std::string ids = d.blocks[6]["text"];
    CHECK(ids.find("\"panel\":\"curves\"") != std::string::npos);
    CHECK(ids.find("\"catalog\":\"table:run42_metrics\"") != std::string::npos);

    // Shaped for an agent without embedded context or images: the resource as
    // text, the render left out, the request last.
    acp::AgentInfo plain;
    std::vector<std::string> dropped;
    json blocks = chat_prompt_blocks(d.text, &d, plain, &dropped);
    REQUIRE(blocks.size() == 7);
    CHECK(blocks[1]["type"] == "text");
    CHECK(blocks[1]["text"] == "<caliper-data part=\"spec\">x</caliper-data>");
    CHECK(blocks[1]["annotations"]["audience"] == json::array({"assistant"}));
    for (const json& b : blocks) CHECK(b["type"] != "image");
    CHECK(dropped == std::vector<std::string>{"render"});
    CHECK(blocks.back() == json{{"type", "text"}, {"text", r.request}});
    acp::AgentInfo rich;
    rich.prompt_embedded_context = true;
    rich.prompt_image = true;
    blocks = chat_prompt_blocks(d.text, &d, rich);
    REQUIRE(blocks.size() == 8);
    CHECK(blocks[1]["type"] == "resource");
    CHECK(blocks[3]["type"] == "image");

    // In the chat: a draft in the input, sent only by the person.
    Fixture f("chat-handoff");
    SideChat chat(f.options("--embedded-context", CALIPER_CHAT_STUB_AGENT));
    chat.open_handoff(d);
    CHECK(chat.text() == r.request);
    CHECK(chat.draft().chips.size() == 7);
    for (int i = 0; i < 20; ++i) chat.update();
    CHECK(chat.turns() == 0);
    CHECK(chat.items().empty());
    CHECK(chat.state() == ChatState::idle);
    chat.send();
    CHECK(chat.draft().empty());
    REQUIRE(pump_until(chat, [&] { return turn_over(chat); }));
    const ChatItem* p = find(chat, K::person);
    REQUIRE(p);
    CHECK(p->text == r.request);
    CHECK(p->chips.size() == 7);
    // The stub declared embedded context and no images.
    const json& sent = chat.last_prompt();
    REQUIRE(sent.size() == 7);
    CHECK(sent[1]["type"] == "resource");
    CHECK(sent.back()["text"] == r.request);
    CHECK(find(chat, K::agent)->text == "You said: " + r.request);

    // Clear drops the draft's blocks.
    chat.open_handoff(d);
    chat.clear_draft();
    CHECK(chat.draft().blocks.empty());
}

TEST_CASE("side chat: no agent found, then a custom command") {
    Fixture f("chat-none");
    SideChat::Options o = f.options();
    o.settings.custom_command.clear();
    o.settings.agent.clear();
    SideChat chat(o);
    CHECK(chat.state() == ChatState::no_agent);
    CHECK(chat.tag() == "NO AGENT");
    CHECK(chat.agent() == nullptr);
    REQUIRE(chat.agents().size() == 4);
    for (const ChatAgent& a : chat.agents()) CHECK_FALSE(a.found);
    CHECK_FALSE(chat.choose_agent("claude"));
    say(chat, "hi");
    CHECK(chat.items().empty());

    ChatSettings saved;
    SideChat::Hooks hooks;
    hooks.settings_changed = [&saved](const ChatSettings& s) { saved = s; };
    SideChat chat2(o, hooks);
    CHECK(chat2.set_custom_agent("Stub agent", "say 'unclosed").code == "acp.bad_command");
    CHECK(chat2.set_custom_agent("Stub agent", "/no/such/agent").code == "chat.no_agent");
    CHECK(chat2.state() == ChatState::no_agent);
    REQUIRE(chat2.set_custom_agent("Stub agent", std::string("'") + CALIPER_ACP_STUB_AGENT + "'").ok());
    CHECK(chat2.state() == ChatState::idle);
    CHECK(saved.agent == "custom");
    CHECK(saved.custom_name == "Stub agent");
    say(chat2, "hi");
    REQUIRE(pump_until(chat2, [&] { return turn_over(chat2); }));
    CHECK(chat2.items()[1].text == "You said: hi");
}

TEST_CASE("side chat: the stop words and the tool status words") {
    CHECK(chat_stop_word("end_turn", {}) == "DONE");
    CHECK(chat_stop_word("cancelled", {}) == "STOPPED");
    CHECK(chat_stop_word("cancelled", {"acp.cancel_unanswered", "x"}) == "STOPPED");
    CHECK(chat_stop_word("refusal", {}) == "DECLINED");
    CHECK(chat_stop_word("max_tokens", {}) == "LIMIT");
    CHECK(chat_stop_word("max_turn_requests", {}) == "LIMIT");
    CHECK(chat_stop_word("", {"acp.prompt_failed", "x"}) == "FAILED");
    CHECK(chat_tool_status_word("pending") == "pending");
    CHECK(chat_tool_status_word("in_progress") == "running");
    CHECK(chat_tool_status_word("completed") == "done");
    CHECK(chat_tool_status_word("failed") == "failed");
    CHECK(chat_tool_status_word("cancelled") == "stopped");
    CHECK(std::string(chat_action_label(ChatAction::custom)) == "Custom command");
    CHECK(std::string(chat_action_key(ChatAction::send)) == "\xE2\x8F\x8E");
}

TEST_CASE("side chat: the compact diff of a spec file's edit, as the cards show it") {
    const std::string before = R"(caliper_spec = 0
[applet]
id = "d"
[[source]]
id = "metrics"
catalog = "table:run42_metrics"
[[panel]]
id = "curves"
primitive = "series"
title = "Loss"
bind = { x = "metrics.step", y = ["metrics.loss"] }
options = { y_log = true }
)";
    std::string after = before;
    after.replace(after.find("y_log = true"), 12, "y_log = false");
    after += R"([[panel]]
id = "lr"
primitive = "series"
title = "Learning rate"
bind = { x = "metrics.step", y = ["metrics.lr"] }
)";
    const std::vector<WidgetDiffLine> d = chat_compact_diff("/w/applets/d/d.spec.toml", before, after);
    REQUIRE(d.size() >= 4);
    CHECK(d[0].key == "primitive");
    CHECK(d[0].after == "\"series\"");
    CHECK(d[1].key == "title");
    CHECK(d[1].after == "\"Learning rate\"");
    bool y_log = false;
    for (const WidgetDiffLine& l : d)
        if (l.key.find("y_log") != std::string::npos) {
            y_log = true;
            CHECK(l.before == "true");
            CHECK(l.after == "false");
        }
    CHECK(y_log);
    for (const WidgetDiffLine& l : d) CHECK(l.key != "panel lr");   // the new panel's keys, not one line
    // Not a spec file, or not a spec: no compact diff, the card shows the unified diff.
    CHECK(chat_compact_diff("/w/loader.py", before, after).empty());
    CHECK(chat_compact_diff("/w/x.spec.toml", "title = \"Old\"\n", "title = \"New\"\n").empty());
}

namespace {

ChatItem call(const std::string& id, const std::string& command, const std::string& status) {
    ChatItem it;
    it.kind = ChatItem::Kind::tool;
    it.tool.id = id;
    it.tool.title = "mcp__caliper__" + caliper_builder::command_tool_name(command);
    it.tool.status = status;
    it.command = command;
    return it;
}

ChatItem approval(const std::string& id, const std::string& command, const std::string& answer) {
    ChatItem it;
    it.kind = ChatItem::Kind::approval;
    it.approval = id;
    it.command = command;
    it.answer = answer;
    return it;
}

ChatItem said(const std::string& text) {
    ChatItem it;
    it.kind = ChatItem::Kind::agent;
    it.text = text;
    return it;
}

}  // namespace

TEST_CASE("side chat rows: a run of one command folds into one group with its approvals") {
    std::vector<ChatItem> items;
    for (int i = 0; i < 7; ++i) {
        items.push_back(call("s" + std::to_string(i), "source.add", "completed"));
        items.push_back(approval("ap-" + std::to_string(i), "source.add", "approved"));
    }
    items.push_back(said("Opened it."));
    const std::vector<ChatRow> rows = chat_rows(items);
    REQUIRE(rows.size() == 2);
    const ChatRow& g = rows[0];
    CHECK(g.kind == ChatRow::Kind::group);
    CHECK(g.command == "source.add");
    CHECK(g.caliper);
    CHECK(g.count == 7);
    CHECK(g.members.size() == 14);   // the calls and their decided approvals
    CHECK(g.approvals == "approved");
    CHECK(g.status == "done");
    CHECK(chat_group_summary(g) == "approved \xC2\xB7 done");
    CHECK_FALSE(g.alert);
    CHECK_FALSE(g.expanded);   // closed until the person opens it
    CHECK(rows[1].kind == ChatRow::Kind::item);
    CHECK(rows[1].item == 14);
    // The person opens it.
    CHECK(chat_rows(items, {{g.key, true}})[0].expanded);
    // A lone call without approvals stays a plain row.
    const std::vector<ChatItem> one = {call("x", "spec.get", "completed")};
    REQUIRE(chat_rows(one).size() == 1);
    CHECK(chat_rows(one)[0].kind == ChatRow::Kind::item);
    // A lone call with its approval is a group of one.
    const std::vector<ChatItem> approved = {call("p", "panel.create", "completed"), approval("ap-9", "panel.create", "approved")};
    REQUIRE(chat_rows(approved).size() == 1);
    CHECK(chat_rows(approved)[0].count == 1);
    CHECK(chat_group_summary(chat_rows(approved)[0]) == "approved \xC2\xB7 done");
}

TEST_CASE("side chat rows: a run is broken by another command, and by the agent's text") {
    const std::vector<ChatItem> items = {call("a", "source.add", "completed"), call("b", "source.add", "completed"),
                                         call("c", "panel.create", "completed"), call("d", "source.add", "completed"),
                                         said("And one more."), call("e", "source.add", "completed"),
                                         call("f", "source.add", "completed")};
    const std::vector<ChatRow> rows = chat_rows(items);
    REQUIRE(rows.size() == 5);
    CHECK(rows[0].kind == ChatRow::Kind::group);
    CHECK(rows[0].count == 2);
    CHECK(rows[0].approvals.empty());
    CHECK(chat_group_summary(rows[0]) == "done");
    CHECK(rows[1].kind == ChatRow::Kind::item);   // panel.create alone
    CHECK(rows[1].item == 2);
    CHECK(rows[2].kind == ChatRow::Kind::item);   // source.add alone again
    CHECK(rows[2].item == 3);
    CHECK(rows[3].kind == ChatRow::Kind::item);   // the text
    CHECK(rows[4].kind == ChatRow::Kind::group);
    CHECK(rows[4].count == 2);
    CHECK(rows[4].members == std::vector<size_t>{5, 6});
}

TEST_CASE("side chat rows: a waiting approval shows on its own after its group until it is decided") {
    std::vector<ChatItem> items = {call("a", "source.add", "completed"), approval("ap-1", "source.add", "approved"),
                                   call("b", "source.add", "in_progress"), approval("ap-2", "source.add", "")};
    std::vector<ChatRow> rows = chat_rows(items);
    REQUIRE(rows.size() == 2);
    CHECK(rows[0].kind == ChatRow::Kind::group);
    CHECK(rows[0].count == 2);
    CHECK(rows[0].status == "running");
    CHECK(rows[0].mark == "in_progress");
    CHECK(rows[0].members == std::vector<size_t>{0, 1, 2});
    CHECK(rows[1].kind == ChatRow::Kind::waiting_approval);
    CHECK(rows[1].item == 3);
    // Decided: it folds into the group.
    items[3].answer = "approved";
    items[2].tool.status = "completed";
    rows = chat_rows(items);
    REQUIRE(rows.size() == 1);
    CHECK(rows[0].members.size() == 4);
    CHECK(chat_group_summary(rows[0]) == "approved \xC2\xB7 done");
}

TEST_CASE("side chat rows: a denied approval or a failed call makes the group an alert, open by default") {
    const std::vector<ChatItem> items = {call("a", "source.add", "completed"), approval("ap-1", "source.add", "approved"),
                                         call("b", "source.add", "failed"), approval("ap-2", "source.add", "denied")};
    const std::vector<ChatRow> rows = chat_rows(items);
    REQUIRE(rows.size() == 1);
    CHECK(rows[0].alert);
    CHECK(rows[0].expanded);
    CHECK(rows[0].status == "failed");
    CHECK(rows[0].approvals == "1 denied \xC2\xB7 1 approved");
    CHECK(chat_group_summary(rows[0]) == "1 denied \xC2\xB7 1 approved \xC2\xB7 failed");
    // The person may close it.
    CHECK_FALSE(chat_rows(items, {{rows[0].key, false}})[0].expanded);
    // A call that failed without an approval is an alert too.
    const std::vector<ChatItem> failed = {call("a", "layout.set", "completed"), call("b", "layout.set", "failed")};
    CHECK(chat_rows(failed)[0].alert);
    CHECK(chat_rows(failed)[0].expanded);
}
