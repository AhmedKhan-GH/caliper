// caliper-acp-harness end to end against the stub agent: the transcript it
// prints and its exit codes. POSIX only.
#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

using namespace caliper_builder::acp;
using nlohmann::json;

namespace {

const std::string kHarness = CALIPER_ACP_HARNESS;
const std::string kStub = CALIPER_ACP_STUB_AGENT;

std::string workspace() {
    std::filesystem::path p = std::filesystem::path(CALIPER_BUILDER_TEST_TMP) / "harness-workspace";
    std::filesystem::create_directories(p);
    return std::filesystem::canonical(p).string();
}

struct Transcript {
    RunResult run;
    std::vector<json> lines;

    std::vector<std::string> types() const {
        std::vector<std::string> t;
        for (const auto& l : lines) t.push_back(l.value("type", ""));
        return t;
    }
    const json* first(const std::string& type) const {
        for (const auto& l : lines)
            if (l.value("type", "") == type) return &l;
        return nullptr;
    }
    std::string message() const {
        std::string s;
        for (const auto& l : lines)
            if (l.value("type", "") == "agent_message_chunk") s += l.value("text", "");
        return s;
    }
};

Transcript run_harness(std::vector<std::string> args, bool parse = true) {
    SpawnOptions so;
    so.argv = {kHarness};
    so.argv.insert(so.argv.end(), args.begin(), args.end());
    RunLimits limits;
    limits.timeout_ms = 60000;
    Transcript t;
    t.run = run_process(so, limits);
    if (!parse) return t;
    std::istringstream in(t.run.out);
    std::string line;
    while (std::getline(in, line)) {
        json j = json::parse(line, nullptr, false);
        REQUIRE_MESSAGE(!j.is_discarded(), "not JSON: " << line);
        t.lines.push_back(std::move(j));
    }
    return t;
}

}  // namespace

TEST_CASE("the harness runs an edit end to end and allows it") {
    Transcript t = run_harness({"--cwd", workspace(), "--mcp", "caliper=" + kStub + " --unused", "--prompt", "edit",
                                "--permission", "allow", "--env", "CALIPER_TEST_SECRET=hunter2-value", "--timeout",
                                "30", "--", kStub});
    REQUIRE(t.run.error.ok());
    CHECK(t.run.status.code == 0);
    const auto types = t.types();
    REQUIRE(types.size() >= 8);
    CHECK(types[0] == "harness");
    CHECK(types[1] == "initialized");
    CHECK(types[2] == "session");
    CHECK(types[3] == "prompt");
    CHECK(types.back() == "result");

    const json* h = t.first("harness");
    CHECK((*h)["env_names"] == json::array({"CALIPER_TEST_SECRET"}));
    CHECK((*h)["cwd"] == workspace());
    CHECK(t.run.out.find("hunter2-value") == std::string::npos);
    CHECK(t.run.err.find("hunter2-value") == std::string::npos);

    CHECK((*t.first("initialized"))["protocol_version"] == 1);
    CHECK((*t.first("initialized"))["agent"]["name"] == "caliper-acp-stub");
    CHECK((*t.first("session"))["session_id"] == "stub-session-1");

    const json* call = t.first("tool_call");
    REQUIRE(call);
    CHECK((*call)["tool_call"]["content"][0]["type"] == "diff");
    CHECK((*call)["tool_call"]["content"][0]["path"] == workspace() + "/demo.spec.toml");
    const json* req = t.first("permission_request");
    REQUIRE(req);
    CHECK((*req)["options"].size() == 3);
    const json* answer = t.first("permission_answer");
    REQUIRE(answer);
    CHECK((*answer)["option_id"] == "allow");
    CHECK((*answer)["permission_id"] == (*req)["permission_id"]);
    CHECK(t.message() == "Applied the change.");
    CHECK((*t.first("turn_ended"))["stop_reason"] == "end_turn");
    const json* result = t.first("result");
    CHECK((*result)["ok"] == true);
    CHECK((*result)["stop_reason"] == "end_turn");
    CHECK((*result)["exit_code"] == 0);
}

TEST_CASE("the harness passes MCP servers and denies by default") {
    Transcript t = run_harness({"--cwd", workspace(), "--mcp", "caliper=" + kStub + " --flag 'two words'", "--prompt",
                                "mcp", "--agent", "'" + kStub + "'"});
    CHECK(t.run.status.code == 0);
    json report = json::parse(t.message());
    REQUIRE(report["mcpServers"].size() == 1);
    CHECK(report["mcpServers"][0]["name"] == "caliper");
    CHECK(report["mcpServers"][0]["args"] == json::array({"--flag", "two words"}));

    t = run_harness({"--cwd", workspace(), "--prompt", "edit", "--", kStub});
    CHECK(t.run.status.code == 0);
    CHECK((*t.first("permission_answer"))["option_id"] == "reject");
    CHECK(t.message() == "Left the spec unchanged.");
}

TEST_CASE("the harness reports failures with their exit codes") {
    Transcript t = run_harness({"--cwd", workspace(), "--prompt", "hello", "--", kStub, "--version-reply", "3"});
    CHECK(t.run.status.code == 3);
    const json* err = t.first("error");
    REQUIRE(err);
    CHECK((*err)["stage"] == "initialize");
    CHECK((*err)["error"]["code"] == "acp.version_unsupported");

    t = run_harness({"--cwd", workspace(), "--prompt", "hello", "--", kStub, "--auth-required"});
    CHECK(t.run.status.code == 3);
    CHECK((*t.first("error"))["error"]["code"] == "acp.auth_required");

    t = run_harness({"--cwd", workspace(), "--prompt", "crash", "--", kStub});
    CHECK(t.run.status.code == 4);
    CHECK((*t.first("turn_ended"))["error"]["code"] == "acp.agent_exited");
    const json* exited = t.first("agent_exited");
    REQUIRE(exited);
    CHECK((*exited)["exit_code"] == 3);
    CHECK((*t.first("result"))["ok"] == false);

    t = run_harness({"--cwd", workspace(), "--prompt", "slow", "--timeout", "0.5", "--", kStub});
    CHECK(t.run.status.code == 5);
    CHECK(t.first("timeout"));
    CHECK((*t.first("turn_ended"))["stop_reason"] == "cancelled");

    t = run_harness({"--prompt", "hello", "--", "caliper-no-such-agent"});
    CHECK(t.run.status.code == 3);
    CHECK((*t.first("error"))["error"]["code"] == "process.not_found");
}

TEST_CASE("the harness refuses bad usage") {
    auto usage = [](std::vector<std::string> args) { return run_harness(std::move(args), false).run.status.code; };
    CHECK(usage({"--prompt", "x"}) == 2);
    CHECK(usage({"--", kStub}) == 2);
    CHECK(usage({"--prompt", "x", "--permission", "maybe", "--", kStub}) == 2);
    CHECK(usage({"--prompt", "x", "--preset", "nobody"}) == 2);
    CHECK(usage({"--prompt", "x", "--agent", "'unclosed"}) == 2);
    CHECK(usage({"--prompt", "x", "--mcp", "noequals", "--", kStub}) == 2);
    CHECK(usage({"--prompt", "x", "--env", "CALIPER_SURELY_UNSET_VARIABLE", "--", kStub}) == 2);
    CHECK(usage({"--prompt", "x", "--timeout", "-1", "--", kStub}) == 2);
    CHECK(usage({"--prompt", "x", "--cwd", "/nonexistent/folder", "--", kStub}) == 2);
    CHECK(usage({"--help"}) == 0);
}

TEST_CASE("the harness passes --meta through as the _meta of session/new") {
    Transcript t = run_harness({"--cwd", workspace(), "--meta", R"({"caliper": {"widget": {"preview": "widget-1"}}})",
                                "--prompt", "mcp", "--timeout", "30", "--", kStub});
    REQUIRE(t.run.error.ok());
    CHECK(t.run.status.code == 0);
    const json* session = t.first("session");
    REQUIRE(session != nullptr);
    CHECK((*session)["meta"]["echo"]["caliper"]["widget"]["preview"] == "widget-1");
    CHECK(json::parse(t.message())["_meta"]["caliper"]["widget"]["preview"] == "widget-1");
    CHECK(run_harness({"--meta", "[1]", "--prompt", "x", "--", kStub}, false).run.status.code == 2);
}
