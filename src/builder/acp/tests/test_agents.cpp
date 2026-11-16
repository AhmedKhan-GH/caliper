// The agent presets, the command-line splitter, describe_spawn and, on
// Windows, the stubs that refuse to spawn.
#include <caliper_builder/acp/acp_client.h>
#include <caliper_builder/acp/agents.h>
#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>

using namespace caliper_builder::acp;

TEST_CASE("split_command_line splits words like a shell, without running one") {
    std::vector<std::string> argv;
    CHECK(split_command_line("npx -y @agentclientprotocol/claude-agent-acp", &argv).ok());
    CHECK(argv == std::vector<std::string>{"npx", "-y", "@agentclientprotocol/claude-agent-acp"});

    CHECK(split_command_line("  'a b'  \"c \\\"d\\\"\" e\\ f ''", &argv).ok());
    CHECK(argv == std::vector<std::string>{"a b", "c \"d\"", "e f", ""});

    // Nothing is expanded.
    CHECK(split_command_line("echo $HOME `id` *.txt > out", &argv).ok());
    CHECK(argv == std::vector<std::string>{"echo", "$HOME", "`id`", "*.txt", ">", "out"});

    CHECK(split_command_line("a'b'\"c\"", &argv).ok());
    CHECK(argv == std::vector<std::string>{"abc"});
}

TEST_CASE("split_command_line refuses broken input") {
    std::vector<std::string> argv;
    CHECK(split_command_line("", &argv).code == "acp.bad_command");
    CHECK(split_command_line("   ", &argv).code == "acp.bad_command");
    CHECK(split_command_line("say 'unclosed", &argv).code == "acp.bad_command");
    CHECK(split_command_line("say \"unclosed", &argv).code == "acp.bad_command");
    CHECK(split_command_line("trailing\\", &argv).code == "acp.bad_command");
    CHECK(split_command_line(std::string("nul\0byte", 8), &argv).code == "acp.bad_command");
    CHECK(split_command_line(std::string(70 * 1024, 'a'), &argv).code == "acp.bad_command");
}

TEST_CASE("the known agents have launch commands") {
    const auto& agents = known_agents();
    REQUIRE(agents.size() == 4);
    for (const auto& a : agents) {
        CHECK_FALSE(a.id.empty());
        CHECK_FALSE(a.argv.empty());
        CHECK_FALSE(a.login_hint.empty());
    }
    REQUIRE(find_agent_preset("gemini") != nullptr);
    CHECK(find_agent_preset("gemini")->argv == std::vector<std::string>{"gemini", "--acp"});
    CHECK(find_agent_preset("claude")->argv.back() == "@agentclientprotocol/claude-agent-acp");
    REQUIRE(find_agent_preset("opencode") != nullptr);
    CHECK(find_agent_preset("opencode")->argv == std::vector<std::string>{"opencode", "acp"});
    CHECK(find_agent_preset("nobody") == nullptr);
}

TEST_CASE("describe_spawn names variables but never shows their values") {
    SpawnOptions so;
    so.argv = {"agent", "--flag", "two words"};
    so.env = {{"API_KEY", "sk-very-secret"}, {"MODE", "quiet"}};
    so.env_remove = {"HOME"};
    so.cwd = "/work space";
    const std::string d = describe_spawn(so);
    CHECK(d.find("API_KEY") != std::string::npos);
    CHECK(d.find("MODE") != std::string::npos);
    CHECK(d.find("HOME") != std::string::npos);
    CHECK(d.find("'two words'") != std::string::npos);
    CHECK(d.find("sk-very-secret") == std::string::npos);
    CHECK(d.find("quiet") == std::string::npos);

    AcpClientOptions o;
    o.agent_argv = {"agent"};
    o.agent_env = {{"ANTHROPIC_API_KEY", "sk-also-secret"}};
    AcpClient client(o);
    CHECK(client.describe_command().find("sk-also-secret") == std::string::npos);
    CHECK(client.describe_command().find("ANTHROPIC_API_KEY") != std::string::npos);
}

TEST_CASE("event_to_json gives each kind its type") {
    AcpEvent ev;
    ev.kind = AcpEventKind::turn_ended;
    ev.stop_reason = "end_turn";
    auto j = event_to_json(ev);
    CHECK(j["type"] == "turn_ended");
    CHECK(j["stop_reason"] == "end_turn");
    CHECK_FALSE(j.contains("error"));

    ev = AcpEvent{};
    ev.kind = AcpEventKind::tool_call;
    ev.tool_call.id = "t1";
    ev.tool_call.has_content = true;
    ToolCallContent c;
    c.type = "diff";
    c.path = "/w/a.toml";
    c.new_text = "b";
    ev.tool_call.content.push_back(c);
    j = event_to_json(ev);
    CHECK(j["tool_call"]["content"][0]["old_text"].is_null());
    CHECK(j["tool_call"]["content"][0]["new_text"] == "b");
}

#ifdef _WIN32
TEST_CASE("spawning is refused on Windows with a clear error") {
    caliper_builder::Error e;
    SpawnOptions so;
    so.argv = {"cmd.exe"};
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK(e.code == "process.unsupported");
    CHECK(make_pipe_channel_pair().error.code == "process.unsupported");
}
#endif
