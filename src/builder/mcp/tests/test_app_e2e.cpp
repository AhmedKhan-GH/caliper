// caliper-mcp end to end against the real app, after contracts sections 14
// and 17. When the CMake variable CALIPER_BUILDER_TEST_APP names a caliper
// executable, the test starts it with --hidden --api --api-port 0 and a
// CALIPER_DATA_DIR of its own, runs caliper-mcp against the API file the app
// writes, calls app_info, app_applets, capture_frame and one command tool,
// which the app's builder session answers, and stops the app with SIGTERM. The app is never started without --hidden.
// Without the variable the test reports itself skipped. POSIX only.
#include "../codec.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/mcp/api_file.h>

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>

using namespace caliper_builder;
using caliper_builder::acp::ExitStatus;
using caliper_builder::acp::IoStatus;
using caliper_builder::acp::Process;
using caliper_builder::acp::SpawnOptions;
using nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

const std::string kApp = CALIPER_BUILDER_TEST_APP;
const std::string kExe = CALIPER_MCP_EXE;

// Collects a child's two output streams on threads, so it never blocks on a
// full pipe. `stdout_lines` gives complete lines of standard output.
class Drain {
public:
    Drain(Process& p, bool keep_stdout_for_reading) : p_(p), read_stdout_(keep_stdout_for_reading) {
        err_ = std::thread([this] { pump(false); });
        if (!read_stdout_) out_ = std::thread([this] { pump(true); });
    }
    ~Drain() { join(); }
    void join() {
        stop_ = true;
        if (err_.joinable()) err_.join();
        if (out_.joinable()) out_.join();
    }
    std::string text() {
        std::lock_guard<std::mutex> g(mu_);
        return text_;
    }

private:
    void pump(bool is_out) {
        char b[8192];
        for (;;) {
            std::size_t got = 0;
            const IoStatus st = is_out ? p_.read_stdout(b, sizeof b, &got, 100) : p_.read_stderr(b, sizeof b, &got, 100);
            if (st == IoStatus::ok) {
                std::lock_guard<std::mutex> g(mu_);
                if (text_.size() < (4u << 20)) text_.append(b, got);
            } else if (st != IoStatus::timeout || stop_) {
                return;
            }
        }
    }
    Process& p_;
    bool read_stdout_;
    std::atomic<bool> stop_{false};
    std::thread err_, out_;
    std::mutex mu_;
    std::string text_;
};

json modern_meta() {
    return {{"io.modelcontextprotocol/protocolVersion", "2026-07-28"},
            {"io.modelcontextprotocol/clientCapabilities", json::object()},
            {"io.modelcontextprotocol/clientInfo", {{"name", "caliper-e2e"}, {"version", "1"}}}};
}

// One MCP request over the child's pipes, waiting for its response.
json request(Process& p, std::string* out_log, std::string* buf, const json& id, const std::string& method,
             json params, int timeout_ms = 60000) {
    params["_meta"] = modern_meta();
    const std::string line = json{{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}}.dump() + "\n";
    REQUIRE(p.write_stdin_all(line.data(), line.size(), 10000) == IoStatus::ok);
    const auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const auto nl = buf->find('\n');
        if (nl != std::string::npos) {
            const std::string got = buf->substr(0, nl);
            buf->erase(0, nl + 1);
            *out_log += got + "\n";
            const json m = json::parse(got, nullptr, false);
            REQUIRE_MESSAGE(m.is_object(), "stdout carried a line that is not a JSON object");
            if (m.contains("id") && m["id"] == id) return m;
            continue;
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
        REQUIRE_MESSAGE(left > 0, "no answer to " << method);
        char b[65536];
        std::size_t n = 0;
        const IoStatus st = p.read_stdout(b, sizeof b, &n, int(std::min<long long>(left, 100)));
        if (st == IoStatus::ok) buf->append(b, n);
        REQUIRE_MESSAGE(st != IoStatus::eof, "caliper-mcp closed its output");
    }
}

json tool_error(const json& result) {
    return json::parse(result["content"][0]["text"].get<std::string>())["error"];
}

}  // namespace

TEST_CASE("caliper-mcp drives the real app end to end") {
    if (kApp.empty()) {
        std::printf("caliper-mcp end-to-end test skipped: set the CMake variable CALIPER_BUILDER_TEST_APP to a "
                    "caliper executable to run it\n");
        MESSAGE("caliper-mcp end-to-end test skipped: CALIPER_BUILDER_TEST_APP is not set");
        return;
    }
    REQUIRE_MESSAGE(fs::exists(kApp), "CALIPER_BUILDER_TEST_APP does not name a file: " << kApp);

    const fs::path work = fs::path(CALIPER_BUILDER_TEST_TMP) / "app-e2e";
    fs::remove_all(work);
    const fs::path data = work / "data";
    fs::create_directories(data);
    const std::string api_file = (data / "api.json").string();

    // The app, hidden, with the API on a free port and its own app-data folder.
    SpawnOptions app_opts;
    app_opts.argv = {kApp, "--hidden", "--api", "--api-port", "0"};
    app_opts.env_remove = {"CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_USER_APPLETS_DIR",
                           "CALIPER_APPLET_PATH"};
    // The app passes its environment on to any caliper-agent it starts: the
    // switch keeps that agent off every model server off loopback.
    app_opts.env = {{"CALIPER_DATA_DIR", data.string()}, {"CALIPER_HIDDEN", "1"}, {"CALIPER_AGENT_OFFLINE", "1"}};
    Error err;
    std::unique_ptr<Process> app = Process::spawn(app_opts, &err);
    REQUIRE_MESSAGE(app, "the app did not start: " << err.message);
    Drain app_out(*app, false);

    // The API file appears once the API listens.
    mcp::ApiFileRead endpoint;
    const auto wait_end = Clock::now() + std::chrono::seconds(60);
    for (;;) {
        endpoint = mcp::read_api_file(api_file);
        if (endpoint.error.ok()) break;
        ExitStatus early;
        if (app->try_wait(&early)) FAIL("the app exited before it wrote its API file: " << early.describe());
        if (Clock::now() > wait_end) FAIL("the app wrote no API file within 60 seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const std::string token = endpoint.endpoint.token;
    REQUIRE(token.size() >= 16);

    // caliper-mcp against that file.
    SpawnOptions mcp_opts;
    mcp_opts.argv = {kExe, "--api-file", api_file, "--docs",
                     (fs::path(CALIPER_BUILDER_SOURCE_DIR) / ".." / ".." / "docs").string()};
    mcp_opts.env_remove = {"CALIPER_DATA_DIR"};
    std::unique_ptr<Process> server = Process::spawn(mcp_opts, &err);
    REQUIRE_MESSAGE(server, "caliper-mcp did not start: " << err.message);
    Drain server_err(*server, true);
    std::string out_log, buf;

    const json discover = request(*server, &out_log, &buf, "d", "server/discover", json::object());
    CHECK(discover["result"]["supportedVersions"][0] == "2026-07-28");
    const json list = request(*server, &out_log, &buf, "l", "tools/list", json::object());
    CHECK(list["result"]["tools"].size() == command_table().size() - 2);  // not history.undo, history.redo

    const json info = request(*server, &out_log, &buf, 1, "tools/call", {{"name", "app_info"}, {"arguments", json::object()}});
    REQUIRE_MESSAGE(info["result"]["isError"] == false, info.dump());
    CHECK(info["result"]["structuredContent"]["host"].is_string());
    CHECK(info["result"]["structuredContent"]["services"].is_array());
    MESSAGE("app_info: " << info["result"]["structuredContent"].dump());

    const json applets = request(*server, &out_log, &buf, 2, "tools/call", {{"name", "app_applets"}, {"arguments", json::object()}});
    REQUIRE_MESSAGE(applets["result"]["isError"] == false, applets.dump());
    CHECK(applets["result"]["structuredContent"]["applets"].is_array());
    MESSAGE("app_applets lists " << applets["result"]["structuredContent"]["applets"].size() << " applets");

    const json frame = request(*server, &out_log, &buf, 3, "tools/call", {{"name", "capture_frame"}, {"arguments", json::object()}});
    if (frame["result"]["isError"] == true) {
        // A renderer that cannot read its window back (GL, Vulkan) answers 501.
        CHECK(tool_error(frame["result"])["code"] == "capture.unavailable");
        MESSAGE("capture_frame: " << tool_error(frame["result"]).dump());
    } else {
        const json& content = frame["result"]["content"];
        REQUIRE(content[0]["type"] == "image");
        std::string png;
        REQUIRE(mcp::detail::base64_decode(content[0]["data"].get<std::string>(), &png));
        int w = 0, h = 0, n = 0;
        unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(png.data()), int(png.size()),
                                                  &w, &h, &n, 4);
        REQUIRE_MESSAGE(px, "the capture does not decode as a PNG");
        CHECK(w == frame["result"]["structuredContent"]["width"].get<int>());
        CHECK(h == frame["result"]["structuredContent"]["height"].get<int>());
        std::size_t differing = 0;
        for (std::size_t i = 4; i < std::size_t(w) * std::size_t(h) * 4; i += 4)
            if (px[i] != px[0] || px[i + 1] != px[1] || px[i + 2] != px[2]) ++differing;
        stbi_image_free(px);
        CHECK_MESSAGE(differing > 0, "the capture is blank: every pixel has one colour");
        MESSAGE("capture_frame: a " << w << " by " << h << " PNG, " << differing << " pixels differ from the first");
    }

    // A tool of the command layer, which the app serves over its builder
    // session; the default workspace under the app-data folder opens for it.
    const CommandDescriptor* command = nullptr;
    for (const auto& d : command_table())
        if (d.name.rfind("app.", 0) != 0 && d.name != "capture.frame" && !d.examples.empty() && !d.mutates) {
            command = &d;
            break;
        }
    REQUIRE(command);
    const json cmd = request(*server, &out_log, &buf, 4, "tools/call",
                             {{"name", command_tool_name(command->name)}, {"arguments", command->examples[0]}});
    INFO(cmd.dump());
    CHECK(cmd["result"]["isError"] == false);
    CHECK(cmd["result"]["structuredContent"].is_object());
    CHECK(fs::exists(data / "workspace" / "journal"));
    MESSAGE(command->name << ": " << cmd["result"]["structuredContent"].dump().substr(0, 200));

    const json page = request(*server, &out_log, &buf, 5, "resources/read", {{"uri", "caliper://docs/references/manifest.md"}});
    CHECK(page["result"]["contents"][0]["text"].get<std::string>().rfind("# Manifest", 0) == 0);

    // caliper-mcp stops when its input ends, and never showed the token.
    server->close_stdin();
    ExitStatus mcp_exit;
    REQUIRE(server->wait_for(10000, &mcp_exit));
    CHECK(mcp_exit.success());
    server_err.join();
    CHECK(out_log.find(token) == std::string::npos);
    CHECK(server_err.text().find(token) == std::string::npos);

    // SIGTERM closes the hidden window as its close button would, and the app
    // removes its API file.
    const ExitStatus app_exit = app->terminate(20000);
    app_out.join();
    MESSAGE("the app ended with " << app_exit.describe());
    CHECK(app_exit.exited);
    CHECK_FALSE(fs::exists(api_file));
}

TEST_CASE("caliper-mcp's app_launch over unsaved work says that the open waits for the person's approval") {
    if (kApp.empty()) {
        std::printf("caliper-mcp end-to-end test skipped: set the CMake variable CALIPER_BUILDER_TEST_APP to a "
                    "caliper executable to run it\n");
        MESSAGE("caliper-mcp end-to-end test skipped: CALIPER_BUILDER_TEST_APP is not set");
        return;
    }
    const fs::path work = fs::path(CALIPER_BUILDER_TEST_TMP) / "app-e2e-launch";
    fs::remove_all(work);
    const fs::path data = work / "data";
    fs::create_directories(data);
    const std::string api_file = (data / "api.json").string();
    SpawnOptions app_opts;
    app_opts.argv = {kApp, "--hidden", "--api", "--api-port", "0"};
    app_opts.env_remove = {"CALIPER_API", "CALIPER_AUTOLAUNCH", "CALIPER_EXIT_AFTER", "CALIPER_USER_APPLETS_DIR",
                           "CALIPER_APPLET_PATH"};
    app_opts.env = {{"CALIPER_DATA_DIR", data.string()}, {"CALIPER_HIDDEN", "1"}, {"CALIPER_AGENT_OFFLINE", "1"}};
    Error err;
    std::unique_ptr<Process> app = Process::spawn(app_opts, &err);
    REQUIRE_MESSAGE(app, "the app did not start: " << err.message);
    Drain app_out(*app, false);
    mcp::ApiFileRead endpoint;
    const auto wait_end = Clock::now() + std::chrono::seconds(60);
    for (;;) {
        endpoint = mcp::read_api_file(api_file);
        if (endpoint.error.ok()) break;
        ExitStatus early;
        if (app->try_wait(&early)) FAIL("the app exited before it wrote its API file: " << early.describe());
        if (Clock::now() > wait_end) FAIL("the app wrote no API file within 60 seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    SpawnOptions mcp_opts;
    mcp_opts.argv = {kExe, "--api-file", api_file};
    mcp_opts.env_remove = {"CALIPER_DATA_DIR"};
    std::unique_ptr<Process> server = Process::spawn(mcp_opts, &err);
    REQUIRE_MESSAGE(server, "caliper-mcp did not start: " << err.message);
    Drain server_err(*server, true);
    std::string out_log, buf;
    request(*server, &out_log, &buf, "d", "server/discover", json::object());
    int n = 0;
    auto call = [&](const std::string& tool, const json& args) {
        const json r = request(*server, &out_log, &buf, ++n, "tools/call", {{"name", tool}, {"arguments", args}});
        INFO(tool << ": " << r.dump());
        REQUIRE(r.contains("result"));
        return r["result"];
    };
    // The agent saves One, then starts Draft, which it never saves: the open
    // spec has unsaved changes. Over a saved spec nothing of this asks.
    // No spec open yet: spec.get answers so, as a result that its output schema allows.
    const json none = call("spec_get", json::object());
    CHECK(none["isError"] == false);
    CHECK(none["structuredContent"] == json({{"open", false}}));
    CHECK(call("spec_new", {{"id", "one"}, {"title", "One"}})["isError"] == false);
    CHECK(call("applet_save", json::object())["isError"] == false);
    CHECK(call("spec_new", {{"id", "draft"}, {"title", "Draft"}})["isError"] == false);
    // The app lists One once it has read its workspace's applets again.
    bool listed = false;
    for (const auto end = Clock::now() + std::chrono::seconds(30); !listed && Clock::now() < end;) {
        const json applets = call("app_applets", json::object());
        listed = applets["structuredContent"]["applets"].dump().find("\"local.one\"") != std::string::npos;
        if (!listed) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    REQUIRE(listed);
    // The launch is accepted, and says that it is not open yet and why, in the
    // structured result and in the text an agent reads.
    const json launch = call("app_launch", {{"id", "local.one"}});
    CHECK(launch["isError"] == false);
    CHECK(launch["structuredContent"]["status"] == "PENDING_APPROVAL");
    const std::string text = launch["content"][0]["text"].get<std::string>();
    CHECK(text.find("PENDING_APPROVAL") != std::string::npos);
    CHECK(text.find("waits for the person's approval") != std::string::npos);
    MESSAGE("app_launch: " << text);
    // Nothing has changed: Draft is still the open spec.
    CHECK(call("spec_get", json::object())["structuredContent"]["id"] == "draft");
    CHECK(call("spec_get", json::object())["structuredContent"]["open"] == true);

    server->close_stdin();
    ExitStatus mcp_exit;
    REQUIRE(server->wait_for(10000, &mcp_exit));
    server_err.join();
    app->terminate(20000);
    app_out.join();
}
