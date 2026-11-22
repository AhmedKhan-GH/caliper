// caliper-mcp as a child process, driven over its standard input and output
// against an in-process fake of Caliper API that implements API v0 and the
// two command endpoints. The commands are picked from the table generically,
// and the fake answers each with the smallest result its schema accepts.
// POSIX only.
#include "../codec.h"
#include "fake_http.h"
#include "mcp_child.h"
#include "schema_sample.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>
#include <caliper_builder/mcp/tools.h>

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <unistd.h>

using namespace caliper_builder;
using namespace caliper_builder::mcp;
using caliper_builder::acp::EnvVar;
using caliper_builder::acp::ExitStatus;
using caliper_builder::acp::IoStatus;
using caliper_builder::acp::Process;
using caliper_builder::acp::SpawnOptions;
using caliper_builder::mcp::test::FakeConnection;
using caliper_builder::mcp::test::FakeHttpServer;
using caliper_builder::mcp::test::FakeRequest;
using caliper_builder::mcp::test::http_response;
using caliper_builder::mcp::test::McpChild;
using caliper_builder::mcp::test::modern_meta;
using nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

const char* const kV0Commands[] = {"app.info", "app.applets", "app.launch", "capture.frame"};

std::string tiny_png() {
    return std::string("\x89PNG\r\n\x1a\n", 8) + std::string("\0\0\0\x0d", 4) + "IHDR" + std::string("\0\0\0\x02", 4) +
           std::string("\0\0\0\x03", 4) + std::string(17, '\0');
}

std::string new_token() {
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";
    std::random_device rd;
    std::string t;
    for (int i = 0; i < 32; ++i) t += alphabet[rd() % 64];
    return t;
}

fs::path test_dir(const std::string& name) {
    const fs::path p = fs::path(CALIPER_BUILDER_TEST_TMP) / "process" / name;
    fs::remove_all(p);
    fs::create_directories(p);
    return fs::canonical(p);
}

bool is_v0(const std::string& name) {
    for (const char* n : kV0Commands)
        if (name == n) return true;
    return false;
}

// The first command of the table, other than the four of API v0, that has an
// example and whose result schema the sample satisfies.
const CommandDescriptor* pick(const std::function<bool(const CommandDescriptor&)>& wanted) {
    for (const auto& d : command_table()) {
        if (is_v0(d.name) || d.examples.empty() || command_returns_png(d) || !wanted(d)) continue;
        if (validate_command_value(d.result, test::sample_for_schema(d.result), "result").ok()) return &d;
    }
    return nullptr;
}

json find_descriptor(const json& table, const std::string& name) {
    for (const auto& c : table["commands"])
        if (c["name"] == name) return c;
    return nullptr;
}

// A fake of Caliper API: API v0 and, when command_layer is set, the command
// endpoints. Every token it ever issued is kept, to check none leaks.
class FakeApi {
public:
    explicit FakeApi(const std::string& name) : dir(test_dir(name)) {}
    ~FakeApi() {
        if (server) server->stop();
    }

    fs::path dir;
    std::atomic<bool> command_layer{true};
    std::atomic<bool> capture_501{false};
    std::atomic<bool> deny_held{false};
    std::atomic<int> hold_ms{1500};
    std::atomic<bool> held_peer_closed{false};
    std::string hold_command;
    std::string expected_origin = "mcp";

    std::string api_file() const { return (dir / "api.json").string(); }

    // Starts a server on a new port with a new token and writes the API file.
    void start() {
        auto s = std::make_unique<FakeHttpServer>();
        FakeHttpServer* raw = s.get();
        {
            std::lock_guard<std::mutex> g(mu_);
            token_ = new_token();
            tokens_.push_back(token_);
        }
        REQUIRE(s->start([this, raw](const FakeRequest& r, FakeConnection& c) { handle(raw, r, c); }));
        std::unique_ptr<FakeHttpServer> old = std::move(server);
        server = std::move(s);
        if (old) old->stop();
        write_file();
    }

    void write_file() {
        const int port = server->port();
        const json j = {{"version", 1},
                        {"pid", int(getpid())},
                        {"port", port},
                        {"token", token()},
                        {"url", "http://127.0.0.1:" + std::to_string(port) + "/v0"},
                        {"stream_url", "ws://127.0.0.1:" + std::to_string(port) + "/v3/stream"}};
        const fs::path tmp = dir / "api.json.tmp";
        {
            std::ofstream out(tmp);
            out << j.dump();
        }
        fs::rename(tmp, dir / "api.json");
    }

    // The server takes a new token, but the API file keeps the old one.
    void rotate_token_only() {
        std::lock_guard<std::mutex> g(mu_);
        token_ = new_token();
        tokens_.push_back(token_);
    }

    void set_table(json t) {
        std::lock_guard<std::mutex> g(mu_);
        table_ = std::move(t);
    }
    json table() const {
        std::lock_guard<std::mutex> g(mu_);
        return table_;
    }
    std::string token() const {
        std::lock_guard<std::mutex> g(mu_);
        return token_;
    }
    std::vector<std::string> tokens() const {
        std::lock_guard<std::mutex> g(mu_);
        return tokens_;
    }
    std::vector<FakeRequest> log() const {
        std::lock_guard<std::mutex> g(mu_);
        return log_;
    }
    std::size_t log_size() const {
        std::lock_guard<std::mutex> g(mu_);
        return log_.size();
    }

    std::unique_ptr<FakeHttpServer> server;

private:
    void handle(FakeHttpServer* srv, const FakeRequest& r, FakeConnection& c) {
        {
            std::lock_guard<std::mutex> g(mu_);
            log_.push_back(r);
        }
        auto answer = [&](int status, const json& body) { c.write(http_response(status, "application/json", body.dump())); };
        if (r.header("host") != "127.0.0.1:" + std::to_string(srv->port()))
            return answer(403, {{"error", "this Host is not allowed"}});
        if (r.header("authorization") != "Bearer " + token()) {
            c.write(http_response(401, "application/json", "{\"error\":\"the session token is missing or wrong\"}",
                                  {{"WWW-Authenticate", "Bearer realm=\"caliper\""}}));
            return;
        }
        const json t = table();
        auto sample_of = [&](const std::string& name) {
            const json d = find_descriptor(t, name);
            return d.is_null() ? json::object() : test::sample_for_schema(d["result"]);
        };
        const std::string& p = r.target;
        if (r.method == "GET" && p == "/v0/info") return answer(200, sample_of("app.info"));
        if (r.method == "GET" && p == "/v0/applets") return answer(200, sample_of("app.applets"));
        if (r.method == "POST" && p.rfind("/v0/applets/", 0) == 0) {
            if (!r.body.empty()) return answer(413, {{"error", "Caliper API v0 takes no request body"}});
            if (p == "/v0/applets/com.example.missing/launch") return answer(404, {{"error", "no applet with that id"}});
            return answer(200, sample_of("app.launch"));
        }
        if (r.method == "GET" && p == "/v0/capture.png") {
            if (capture_501) return answer(501, {{"error", "the GL renderer cannot read the window back"}});
            c.write(http_response(200, "image/png", tiny_png()));
            return;
        }
        if (r.method == "GET" && p == "/v0/commands") {
            if (!command_layer) return answer(404, {{"error", "no such endpoint in Caliper API v0"}});
            return answer(200, t);
        }
        if (r.method == "POST" && p.rfind("/v0/commands/", 0) == 0) {
            if (!command_layer) return answer(404, {{"error", "no such endpoint in Caliper API v0"}});
            if (r.header("content-type") != "application/json") return answer(415, {{"error", "JSON only"}});
            if (r.header("x-caliper-origin") != expected_origin)
                return answer(400, {{"error", "the test expects another origin"}});
            const std::string name = p.substr(13);
            const json args = json::parse(r.body, nullptr, false);
            if (!args.is_object())
                return answer(200, {{"id", "c-1"}, {"ok", false},
                                    {"error", {{"code", "command.invalid_args"}, {"message", "not an object"}, {"where", "args"}}}});
            if (find_descriptor(t, name).is_null())
                return answer(200, {{"id", "c-1"}, {"ok", false},
                                    {"error", {{"code", "command.unknown"}, {"message", "no such command"}, {"where", ""}}}});
            if (name == hold_command) {
                // Held while the person decides, as the app holds an agent's call.
                const auto end = Clock::now() + std::chrono::milliseconds(hold_ms.load());
                while (Clock::now() < end) {
                    if (c.peer_closed()) {
                        held_peer_closed = true;
                        return;
                    }
                    c.sleep_ms(20);
                }
                if (deny_held)
                    return answer(200, {{"id", "c-1"}, {"ok", false},
                                        {"error", {{"code", "command.denied"}, {"message", "The person denied it."}, {"where", ""}}}});
            }
            return answer(200, {{"id", "c-1"}, {"ok", true}, {"result", sample_of(name)}});
        }
        answer(404, {{"error", "no such endpoint in Caliper API v0"}});
    }

    mutable std::mutex mu_;
    std::string token_;
    std::vector<std::string> tokens_;
    json table_ = command_table_to_json();
    std::vector<FakeRequest> log_;
};

json error_of(const json& tool_result) {
    REQUIRE_MESSAGE(tool_result.value("isError", false), tool_result.dump());
    return json::parse(tool_result["content"][0]["text"].get<std::string>())["error"];
}

void check_no_token(const McpChild& child, const FakeApi& api) {
    const std::string out = child.all_out(), err = child.all_err();
    for (const auto& t : api.tokens()) {
        CHECK_MESSAGE(out.find(t) == std::string::npos, "a token appeared on stdout");
        CHECK_MESSAGE(err.find(t) == std::string::npos, "a token appeared on stderr");
    }
}

fs::path docs_dir(const fs::path& under) {
    const fs::path d = under / "docs";
    fs::create_directories(d / "agents");
    std::ofstream(d / "index.md") << "# Caliper\n";
    std::ofstream(d / "agents" / "http-api.md") << "# Caliper API\n";
    return d;
}

}  // namespace

TEST_CASE("a modern session drives every route of the fake API") {
    FakeApi api("modern");
    api.start();
    const fs::path docs = docs_dir(api.dir);
    McpChild child({"--api-file", api.api_file(), "--docs", docs.string(), "--command-timeout", "20"});

    const json d = child.request("d", "server/discover", json::object());
    CHECK(d["result"]["supportedVersions"][0] == "2026-07-28");

    const json list = child.request("l", "tools/list", json::object());
    json app_table = api.table();
    json agents_may = json::array();  // the person's undo and redo are no tools
    for (const auto& c : app_table["commands"])
        if (c.value("agents", true)) agents_may.push_back(c);
    app_table["commands"] = agents_may;
    REQUIRE(list["result"]["tools"].size() == app_table["commands"].size());
    for (std::size_t i = 0; i < app_table["commands"].size(); ++i)
        CHECK(list["result"]["tools"][i]["name"] == command_tool_name(app_table["commands"][i]["name"].get<std::string>()));

    // The four tools of API v0.
    const json info = child.call(1, "app_info", json::object());
    CHECK(info["isError"] == false);
    CHECK(info["structuredContent"] == test::sample_for_schema(find_descriptor(app_table, "app.info")["result"]));
    CHECK(json::parse(info["content"][0]["text"].get<std::string>()) == info["structuredContent"]);
    CHECK(child.call(2, "app_applets", json::object())["isError"] == false);
    CHECK(child.call(3, "app_launch", {{"id", "com.example.viewer"}})["isError"] == false);
    CHECK(error_of(child.call(4, "app_launch", {{"id", "com.example.missing"}}))["code"] == "app.applet_not_found");
    std::size_t before = api.log_size();
    CHECK(error_of(child.call(5, "app_launch", {{"id", "../../etc"}}))["code"] == "command.invalid_args");
    CHECK(api.log_size() == before);

    const json frame = child.call(6, "capture_frame", json::object());
    REQUIRE(frame["isError"] == false);
    REQUIRE(frame["content"][0]["type"] == "image");
    CHECK(frame["content"][0]["mimeType"] == "image/png");
    std::string png;
    REQUIRE(detail::base64_decode(frame["content"][0]["data"].get<std::string>(), &png));
    CHECK(png == tiny_png());
    CHECK(frame["structuredContent"] == json{{"width", 2}, {"height", 3}});
    api.capture_501 = true;
    const json no_frame = error_of(child.call(7, "capture_frame", json::object()));
    CHECK(no_frame["code"] == "capture.unavailable");
    CHECK(no_frame["message"].get<std::string>().find("GL") != std::string::npos);
    api.capture_501 = false;

    // A command of the command layer, picked from the table.
    const CommandDescriptor* plain = pick([](const CommandDescriptor& c) { return !c.mutates && !c.destructive; });
    REQUIRE(plain);
    CAPTURE(plain->name);
    const json r = child.call(8, command_tool_name(plain->name), plain->examples[0]);
    CHECK(r["isError"] == false);
    CHECK(r["structuredContent"] == test::sample_for_schema(plain->result));
    const FakeRequest sent = api.log().back();
    CHECK(sent.method == "POST");
    CHECK(sent.target == "/v0/commands/" + plain->name);
    CHECK(sent.header("content-type") == "application/json");
    CHECK(sent.header("x-caliper-origin") == "mcp");
    CHECK(sent.header("authorization") == "Bearer " + api.token());
    CHECK(sent.header("host") == "127.0.0.1:" + std::to_string(api.server->port()));
    CHECK(json::parse(sent.body) == plain->examples[0]);

    // Arguments that fail the schema never leave caliper-mcp.
    before = api.log_size();
    CHECK(error_of(child.call(9, command_tool_name(plain->name), {{"no_such_argument", true}}))["code"] ==
          "command.invalid_args");
    CHECK(api.log_size() == before);

    // Resources.
    const json page = child.request("r1", "resources/read", {{"uri", "caliper://docs/agents/http-api.md"}});
    CHECK(page["result"]["contents"][0]["text"] == "# Caliper API\n");
    const json table = child.request("r2", "resources/read", {{"uri", "caliper://commands/command_table.json"}});
    CHECK(json::parse(table["result"]["contents"][0]["text"].get<std::string>()) == app_table);

    const ExitStatus st = child.finish();
    CHECK(st.success());
    check_no_token(child, api);
}

TEST_CASE("a call held for approval waits while other requests go on, and can be denied, cancelled or timed out") {
    const CommandDescriptor* held = pick([](const CommandDescriptor& c) { return c.destructive; });
    REQUIRE(held);
    CAPTURE(held->name);
    const std::string tool = command_tool_name(held->name);
    FakeApi api("held");
    api.hold_command = held->name;
    api.hold_ms = 1500;
    api.start();
    McpChild child({"--api-file", api.api_file(), "--command-timeout", "20"});

    // Approved after 1.5 s; a ping and a duplicate id are answered meanwhile.
    json meta = modern_meta();
    const auto t0 = Clock::now();
    child.send({{"jsonrpc", "2.0"}, {"id", "held"}, {"method", "tools/call"},
                {"params", {{"name", tool}, {"arguments", held->examples[0]}, {"_meta", meta}}}});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const json pong = child.request("ping", "ping", json::object());
    CHECK(pong["result"]["resultType"] == "complete");
    CHECK(Clock::now() - t0 < std::chrono::milliseconds(1400));
    const json dup = child.request("held", "ping", json::object());
    CHECK(dup["error"]["code"] == -32600);
    const json approved = child.await("held");
    CHECK(Clock::now() - t0 >= std::chrono::milliseconds(1400));
    CHECK(approved["result"]["isError"] == false);
    CHECK(approved["result"]["structuredContent"] == test::sample_for_schema(held->result));

    // Denied.
    api.deny_held = true;
    CHECK(error_of(child.call("denied", tool, held->examples[0]))["code"] == "command.denied");
    api.deny_held = false;

    // Cancelled: no answer, and the held connection is closed.
    api.hold_ms = 8000;
    child.send({{"jsonrpc", "2.0"}, {"id", "gone"}, {"method", "tools/call"},
                {"params", {{"name", tool}, {"arguments", held->examples[0]}, {"_meta", meta}}}});
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    child.send({{"jsonrpc", "2.0"}, {"method", "notifications/cancelled"},
                {"params", {{"requestId", "gone"}, {"reason", "the user stopped it"}}}});
    const auto t1 = Clock::now();
    while (!api.held_peer_closed && Clock::now() - t1 < std::chrono::seconds(3))
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(api.held_peer_closed);
    CHECK(child.request("after-cancel", "ping", json::object()).contains("result"));
    for (json m = child.next(500); !m.is_null(); m = child.next(500)) CHECK(m.value("id", json()) != "gone");
    for (const auto& m : child.notes) CHECK(m.value("id", json()) != "gone");

    const ExitStatus st = child.finish();
    CHECK(st.success());
    check_no_token(child, api);

    // A call that outlasts --command-timeout.
    api.hold_ms = 4000;
    McpChild quick({"--api-file", api.api_file(), "--command-timeout", "1"});
    const auto t2 = Clock::now();
    const json late = error_of(quick.call("late", tool, held->examples[0]));
    CHECK(late["code"] == "mcp.timeout");
    CHECK(Clock::now() - t2 < std::chrono::milliseconds(3000));
    CHECK(quick.finish().success());
    check_no_token(quick, api);
}

TEST_CASE("the app restarting, a stale token, a changed table and an app without a command layer") {
    FakeApi api("restart");
    api.start();
    McpChild child({"--api-file", api.api_file()});

    // A legacy client and a modern subscription on the same connection.
    const json init = child.request(1, "initialize",
                                    {{"protocolVersion", "2025-11-25"},
                                     {"capabilities", json::object()},
                                     {"clientInfo", {{"name", "legacy-test"}, {"version", "1"}}}},
                                    false);
    CHECK(init["result"]["protocolVersion"] == "2025-11-25");
    child.send({{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}});
    child.send({{"jsonrpc", "2.0"}, {"id", "sub"}, {"method", "subscriptions/listen"},
                {"params", {{"notifications", {{"toolsListChanged", true}}}, {"_meta", modern_meta()}}}});
    const json ack = child.next(5000);
    CHECK(ack["method"] == "notifications/subscriptions/acknowledged");
    CHECK(ack["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == "sub");

    CHECK(child.call(2, "app_info", json::object(), false)["isError"] == false);
    const std::size_t full = api.table()["commands"].size() - 2;  // not history.undo, history.redo
    CHECK(child.request(3, "tools/list", json::object(), false)["result"]["tools"].size() == full);

    // The app's token changes but the file still names the old one.
    api.rotate_token_only();
    CHECK(error_of(child.call(4, "app_info", json::object(), false))["code"] == "mcp.unauthorized");
    api.write_file();
    CHECK(child.call(5, "app_info", json::object(), false)["isError"] == false);

    // The app restarts on another port with another token.
    const int old_port = api.server->port();
    api.start();
    REQUIRE(api.server->port() != old_port);
    CHECK(child.call(6, "app_info", json::object(), false)["isError"] == false);
    CHECK(api.log().back().header("host") == "127.0.0.1:" + std::to_string(api.server->port()));

    // A newer app with a smaller table: the tool list changes, and both eras hear of it.
    json smaller = command_table_to_json();
    smaller["commands"].erase(smaller["commands"].size() - 1);
    api.set_table(smaller);
    api.start();
    child.notes.clear();
    CHECK(child.call(7, "app_info", json::object(), false)["isError"] == false);
    int legacy_notes = 0, modern_notes = 0;
    for (const auto& m : child.notes) {
        if (m.value("method", "") != "notifications/tools/list_changed") continue;
        if (m.contains("params") && m["params"].contains("_meta")) {
            CHECK(m["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == "sub");
            ++modern_notes;
        } else {
            ++legacy_notes;
        }
    }
    CHECK(legacy_notes == 1);
    CHECK(modern_notes == 1);
    CHECK(child.request(8, "tools/list", json::object(), false)["result"]["tools"].size() == full - 1);

    // An app without a command layer: the built-in table, and a clear error.
    api.command_layer = false;
    api.start();
    const CommandDescriptor* plain = pick([](const CommandDescriptor& c) { return !c.mutates; });
    REQUIRE(plain);
    const json none = error_of(child.call(9, command_tool_name(plain->name), plain->examples[0], false));
    CHECK(none["code"] == "mcp.no_command_layer");
    CHECK(none["message"].get<std::string>().find("no command layer yet") != std::string::npos);
    CHECK(child.request(10, "tools/list", json::object(), false)["result"]["tools"].size() ==
          command_table().size() - 2);  // not history.undo, history.redo

    // The app exits and leaves its file, then the file goes.
    api.server->stop();
    CHECK(error_of(child.call(11, "app_info", json::object(), false))["code"] == "mcp.app_unreachable");
    fs::remove(api.api_file());
    CHECK(error_of(child.call(12, "app_info", json::object(), false))["code"] == "mcp.no_api_file");

    child.notes.clear();
    const ExitStatus st = child.finish();
    CHECK(st.success());
    bool closed = false;
    for (const auto& m : child.notes)
        if (m.value("id", json()) == "sub") {
            closed = true;
            CHECK(m["result"]["resultType"] == "complete");
        }
    CHECK(closed);
    check_no_token(child, api);
}

TEST_CASE("--origin is mcp or agent, and agent is sent as X-Caliper-Origin") {
    SpawnOptions so;
    so.argv = {CALIPER_MCP_EXE, "--origin", "ui"};
    acp::RunLimits limits;
    limits.timeout_ms = 10000;
    const acp::RunResult bad = acp::run_process(so, limits);
    REQUIRE(bad.error.ok());
    CHECK(bad.status.exited);
    CHECK(bad.status.code == 2);
    CHECK(bad.err.find("--origin is mcp or agent") != std::string::npos);
    CHECK(bad.out.empty());

    FakeApi api("origin-agent");
    api.expected_origin = "agent";
    api.start();
    const CommandDescriptor* plain = pick([](const CommandDescriptor& c) { return !c.mutates; });
    REQUIRE(plain);
    McpChild child({"--api-file", api.api_file(), "--origin", "agent"});
    CHECK(child.call(1, command_tool_name(plain->name), plain->examples[0])["isError"] == false);
    CHECK(api.log().back().header("x-caliper-origin") == "agent");
    CHECK(child.finish().success());
    check_no_token(child, api);
}

TEST_CASE("the API file is found through CALIPER_DATA_DIR") {
    FakeApi api("data-dir");
    api.start();
    McpChild child({}, {{"CALIPER_DATA_DIR", api.dir.string()}});
    CHECK(child.call(1, "app_info", json::object())["isError"] == false);
    CHECK(child.finish().success());
    check_no_token(child, api);
}

TEST_CASE("hostile input over the pipe and a closed stdin") {
    FakeApi api("hostile");
    const CommandDescriptor* held = pick([](const CommandDescriptor& c) { return c.destructive; });
    REQUIRE(held);
    api.hold_command = held->name;
    api.hold_ms = 20000;
    api.start();
    McpChild child({"--api-file", api.api_file()});

    child.send_raw("this is not json\n");
    json m = child.next(5000);
    CHECK(m["error"]["code"] == -32700);
    CHECK_FALSE(m.contains("id"));
    child.send_raw(std::string("{\"a\":\0}\n", 8));
    CHECK(child.next(5000)["error"]["code"] == -32700);
    child.send_raw(std::string(9u << 20, ' ') + "{}\n");
    m = child.next(10000);
    CHECK(m["error"]["code"] == -32600);
    CHECK(m["error"]["message"].get<std::string>().find("longer than") != std::string::npos);
    child.send_raw("{\"jsonrpc\":\"2.0\",\"id\":\"crlf\",\"method\":\"ping\"}\r\n");
    CHECK(child.await("crlf").contains("result"));

    // stdin closes while a call is held: the server stops within its grace
    // period and answers the call with mcp.cancelled.
    child.send({{"jsonrpc", "2.0"}, {"id", "held"}, {"method", "tools/call"},
                {"params", {{"name", command_tool_name(held->name)}, {"arguments", held->examples[0]}, {"_meta", modern_meta()}}}});
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const auto t0 = Clock::now();
    const ExitStatus st = child.finish(10000);
    CHECK(st.success());
    CHECK(Clock::now() - t0 < std::chrono::milliseconds(5000));
    bool answered = false;
    for (const auto& n : child.notes)
        if (n.value("id", json()) == "held") {
            answered = true;
            CHECK(error_of(n["result"])["code"] == "mcp.cancelled");
        }
    CHECK(answered);
    check_no_token(child, api);
}
