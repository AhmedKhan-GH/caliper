// The side chat's agents and their launch: detection on a fake PATH without
// running anything, the custom command, the launch built for each detected
// agent and for a custom command (no shell, the MCP server with --origin
// agent, the workspace as the working folder), the declined capabilities as
// the agent sees them, and the names of Caliper's tool calls.
#include "builder/side_chat.h"

#include <caliper_builder/acp/acp_client.h>

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

using namespace caliper_host;
namespace acp = caliper_builder::acp;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

fs::path fresh(const std::string& name) {
    const fs::path dir = fs::path(CALIPER_SPEC_TEST_TMP) / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return fs::canonical(dir);
}

// A file that looks like a program on PATH. Detection must never run it: it
// writes a file when it runs, and the tests check that the file is absent.
void fake_program(const fs::path& dir, const std::string& name) {
    const fs::path p = dir / name;
    std::ofstream(p) << "#!/bin/sh\ntouch \"" << (dir / ("ran-" + name)).string() << "\"\n";
    fs::permissions(p, fs::perms::owner_all, fs::perm_options::replace);
}

const ChatAgent* by_id(const std::vector<ChatAgent>& agents, const std::string& id) {
    for (const ChatAgent& a : agents)
        if (a.id == id) return &a;
    return nullptr;
}

}  // namespace

TEST_CASE("side chat: the four agents are detected on PATH without running them") {
    const fs::path bin = fresh("chat-detect/bin");
    const fs::path other = fresh("chat-detect/other");
    fake_program(bin, "npx");
    fake_program(bin, "claude");
    fake_program(other, "gemini");
    fake_program(other, "codex");   // without npx on PATH, Codex's adapter has no launcher
    const std::string path = bin.string() + ":" + other.string();

    std::vector<ChatAgent> agents = detect_chat_agents(path);
    REQUIRE(agents.size() == 4);
    CHECK(agents[0].id == "claude");
    CHECK(agents[1].id == "codex");
    CHECK(agents[2].id == "gemini");
    CHECK(agents[3].id == "opencode");
    CHECK(by_id(agents, "claude")->found);
    CHECK(by_id(agents, "claude")->needs == std::vector<std::string>{"npx", "claude"});
    CHECK(by_id(agents, "claude")->argv == std::vector<std::string>{(bin / "npx").string(), "-y",
                                                                     "@agentclientprotocol/claude-agent-acp"});
    CHECK(by_id(agents, "gemini")->found);
    CHECK(by_id(agents, "gemini")->argv == std::vector<std::string>{(other / "gemini").string(), "--acp"});
    CHECK(by_id(agents, "codex")->found);   // npx and codex are both on PATH
    CHECK_FALSE(by_id(agents, "opencode")->found);
    CHECK(by_id(agents, "opencode")->argv == std::vector<std::string>{"opencode", "acp"});

    // Without npx, neither adapter is found, though claude and codex are there.
    agents = detect_chat_agents(other.string());
    CHECK_FALSE(by_id(agents, "claude")->found);
    CHECK_FALSE(by_id(agents, "codex")->found);
    CHECK(by_id(agents, "gemini")->found);

    // An empty PATH finds nothing.
    agents = detect_chat_agents(fresh("chat-detect/empty").string());
    for (const ChatAgent& a : agents) CHECK_FALSE(a.found);

    // Nothing ran.
    for (const char* n : {"npx", "claude", "gemini", "codex"}) {
        CHECK_FALSE(fs::exists(bin / (std::string("ran-") + n)));
        CHECK_FALSE(fs::exists(other / (std::string("ran-") + n)));
    }
}

TEST_CASE("side chat: a custom command is split like shell words and run without a shell") {
    const fs::path bin = fresh("chat-custom/bin");
    fake_program(bin, "my-agent");
    ChatAgent a;
    REQUIRE(custom_chat_agent("Local agent", "my-agent --model 'example model' $HOME", bin.string(), &a).ok());
    CHECK(a.custom);
    CHECK(a.id == "custom");
    CHECK(a.title == "Local agent");
    CHECK(a.found);
    CHECK(a.argv == std::vector<std::string>{(bin / "my-agent").string(), "--model", "example model", "$HOME"});

    REQUIRE(custom_chat_agent("", "nowhere-agent acp", bin.string(), &a).ok());
    CHECK(a.title == "Custom agent");
    CHECK_FALSE(a.found);
    CHECK(custom_chat_agent("x", "say 'unclosed", bin.string(), &a).code == "acp.bad_command");
    CHECK(custom_chat_agent("x", "   ", bin.string(), &a).code == "acp.bad_command");
}

TEST_CASE("side chat: the launch of each detected agent and of a custom command") {
    const fs::path root = fresh("chat-launch");
    const fs::path bin = root / "bin";
    fs::create_directories(bin);
    for (const char* n : {"npx", "claude", "codex", "gemini", "opencode"}) fake_program(bin, n);
    const fs::path ws = root / "workspace";
    fs::create_directories(ws);
    const fs::path api = root / "api.json";
    std::ofstream(api) << "{}\n";
    ChatEnvironment env;
    env.workspace = ws.string();
    env.mcp_command = CALIPER_MCP_EXE;
    env.api_file = api.string();
    env.path = bin.string();
    acp::AcpClientOptions base;
    base.stall_timeout_ms = 1234;

    for (const ChatAgent& a : detect_chat_agents(bin.string())) {
        INFO(a.id);
        REQUIRE(a.found);
        ChatLaunch l;
        REQUIRE(build_chat_launch(a, env, &l, base).ok());
        // The preset's argument vector as it is, its program by absolute path: no shell.
        REQUIRE(l.client.agent_argv.size() == a.argv.size());
        CHECK(fs::path(l.client.agent_argv[0]).is_absolute());
        CHECK(l.client.agent_argv == a.argv);
        CHECK(l.client.agent_argv[0].find("sh -c") == std::string::npos);
        CHECK(l.client.agent_cwd == ws.string());
        CHECK(l.cwd == ws.string());
        CHECK(l.client.stall_timeout_ms == 1234);
        REQUIRE(l.client.agent_env.size() == 1);
        CHECK(l.client.agent_env[0].name == "PATH");
        CHECK(l.client.agent_env[0].value == bin.string());
        // One MCP server: this build's caliper-mcp for the origin agent, with the app's API file.
        REQUIRE(l.mcp_servers.size() == 1);
        CHECK(l.mcp_servers[0].name == "caliper");
        CHECK(l.mcp_servers[0].command == CALIPER_MCP_EXE);
        CHECK(l.mcp_servers[0].args == std::vector<std::string>{"--origin", "agent", "--api-file", api.string()});
        CHECK(l.mcp_servers[0].env.empty());
    }
    CHECK(detect_chat_agents(bin.string())[0].argv ==
          std::vector<std::string>{(bin / "npx").string(), "-y", "@agentclientprotocol/claude-agent-acp"});
    CHECK(detect_chat_agents(bin.string())[3].argv == std::vector<std::string>{(bin / "opencode").string(), "acp"});

    ChatAgent custom;
    REQUIRE(custom_chat_agent("Mine", "'" + (bin / "opencode").string() + "' acp --print-logs", "", &custom).ok());
    ChatLaunch l;
    REQUIRE(build_chat_launch(custom, env, &l).ok());
    CHECK(l.client.agent_argv == std::vector<std::string>{(bin / "opencode").string(), "acp", "--print-logs"});
    CHECK(l.mcp_servers[0].args[1] == "agent");

    // What it refuses.
    ChatAgent missing = custom;
    missing.found = false;
    CHECK(build_chat_launch(missing, env, &l).code == "chat.no_agent");
    ChatEnvironment bad = env;
    bad.workspace = "relative/folder";
    CHECK(build_chat_launch(custom, bad, &l).code == "chat.no_workspace");
    bad = env;
    bad.mcp_command = (root / "no-such-mcp").string();
    CHECK(build_chat_launch(custom, bad, &l).code == "chat.no_mcp");
    bad.mcp_command.clear();
    CHECK(build_chat_launch(custom, bad, &l).code == "chat.no_mcp");
    bad = env;
    bad.api_file = (root / "no-api.json").string();
    CHECK(build_chat_launch(custom, bad, &l).code == "chat.no_api");
    // No PATH given: the agent inherits the app's.
    bad = env;
    bad.path.clear();
    REQUIRE(build_chat_launch(custom, bad, &l).ok());
    CHECK(l.client.agent_env.empty());
}

TEST_CASE("side chat: the agent sees the declined capabilities and the MCP server of the launch") {
    const fs::path root = fresh("chat-caps");
    const fs::path ws = root / "workspace";
    fs::create_directories(ws);
    const fs::path api = root / "api.json";
    std::ofstream(api) << "{}\n";
    SideChat::Options o;
    o.env.workspace = ws.string();
    o.env.mcp_command = CALIPER_MCP_EXE;
    o.env.api_file = api.string();
    o.settings.custom_name = "Stub agent";
    o.settings.custom_command = std::string("'") + CALIPER_ACP_STUB_AGENT + "'";
    o.settings.agent = "custom";
    o.opening_prompt = false;
    SideChat chat(o);
    auto turn = [&](const std::string& text) -> json {
        chat.set_text(text);
        chat.send();
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (std::chrono::steady_clock::now() < end) {
            chat.update();
            if (!chat.items().empty() && chat.items().back().kind == ChatItem::Kind::stop) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        for (const ChatItem& it : chat.items())
            if (it.kind == ChatItem::Kind::agent && it.turn == chat.turns()) return json::parse(it.text, nullptr, false);
        return nullptr;
    };
    const json caps = turn("caps");
    INFO(caps.dump());
    REQUIRE(caps.is_object());
    CHECK(caps["protocolVersion"] == 1);
    CHECK(caps["clientCapabilities"]["fs"]["readTextFile"] == false);
    CHECK(caps["clientCapabilities"]["fs"]["writeTextFile"] == false);
    CHECK(caps["clientCapabilities"]["terminal"] == false);
    CHECK_FALSE(caps["clientCapabilities"].contains("auth"));
    CHECK_FALSE(caps["clientCapabilities"].contains("elicitation"));

    const json mcp = turn("mcp");
    INFO(mcp.dump());
    REQUIRE(mcp.is_object());
    CHECK(mcp["cwd"] == ws.string());
    REQUIRE(mcp["mcpServers"].size() == 1);
    CHECK(mcp["mcpServers"][0]["name"] == "caliper");
    CHECK(mcp["mcpServers"][0]["command"] == CALIPER_MCP_EXE);
    CHECK(mcp["mcpServers"][0]["args"] == json::array({"--origin", "agent", "--api-file", api.string()}));
    CHECK(mcp["mcpServers"][0]["env"] == json::array());
    CHECK(mcp["_meta"].is_null());
}

TEST_CASE("side chat: a Caliper tool's call names its command, as each agent names it") {
    auto call = [](const std::string& title, json raw = nullptr) {
        acp::ToolCallInfo t;
        t.title = title;
        t.raw_input = std::move(raw);
        return t;
    };
    CHECK(caliper_command_of(call("mcp__caliper__panel_create")) == "panel.create");
    CHECK(caliper_command_of(call("caliper__spec_get")) == "spec.get");
    CHECK(caliper_command_of(call("caliper.catalog_describe")) == "catalog.describe");
    CHECK(caliper_command_of(call("caliper/layout_set")) == "layout.set");
    CHECK(caliper_command_of(call("caliper_review_status")) == "review.status");
    CHECK(caliper_command_of(call("panel_patch (caliper MCP Server)")) == "panel.patch");
    CHECK(caliper_command_of(call("Tool: `mcp__caliper__applet_save`")) == "applet.save");
    CHECK(caliper_command_of(call("MCP", {{"server", "caliper"}, {"tool", "spec_validate"}})) == "spec.validate");
    // Another server's tool, a tool the table does not have, or no prefix.
    CHECK(caliper_command_of(call("mcp__github__create_issue")).empty());
    CHECK(caliper_command_of(call("mcp__caliper__no_such")).empty());
    CHECK(caliper_command_of(call("panel_create")).empty());
    CHECK(caliper_command_of(call("Edit demo.spec.toml")).empty());
    CHECK(caliper_command_of(call("MCP", {{"server", "other"}, {"tool", "spec_get"}})).empty());
    // caliper-agent names the command in the call's _meta, whatever the title.
    acp::ToolCallInfo own = call("panel_create");
    own.caliper_command = "panel.create";
    CHECK(caliper_command_of(own) == "panel.create");
    own.title = "panel.create heads";
    CHECK(caliper_command_of(own) == "panel.create");
    own.caliper_command = "panel.no_such";
    CHECK(caliper_command_of(own).empty());
}

TEST_CASE("side chat: this build's caliper-mcp is found beside the executable, else in the build tree") {
    const fs::path dir = fresh("chat-find-mcp");
    CHECK(find_caliper_mcp(dir.string()) == CALIPER_MCP_EXE);   // the build tree's
    fake_program(dir, "caliper-mcp");
    CHECK(find_caliper_mcp(dir.string()) == (dir / "caliper-mcp").string());
}

// What the person allows on a permission card is the agent's own call, which
// its file and diff show; a Caliper command named in the call's _meta or its
// title must not stand in for it, or an agent could make a file write read as
// a harmless command.
TEST_CASE("side chat: a permission card shows the agent's own title, never a command the call names") {
    ChatItem p;
    p.kind = ChatItem::Kind::permission;
    p.tool.title = "Edit notes.txt";
    p.tool.caliper_command = "spec.get";
    p.command = caliper_command_of(p.tool);
    REQUIRE(p.command == "spec.get");   // the tool row may show it; the card does not
    CHECK(chat_permission_label(p) == "Edit notes.txt");
    p.tool.title = "mcp__caliper__spec_get";
    CHECK(chat_permission_label(p) == "mcp__caliper__spec_get");
    p.tool.title.clear();
    p.diff_path = "notes.txt";
    CHECK(chat_permission_label(p) == "notes.txt");
    p.diff_path.clear();
    CHECK(chat_permission_label(p) == "Tool call");
}
