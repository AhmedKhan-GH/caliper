// The widget agent's settings in the hidden app (suite "app"), with the
// person's input played by CALIPER_TEST_SPEC_UI:
//   - ⌘K on a panel with no provider set up offers Setup, and Setup opens the
//     settings, which then show "No provider"; nothing starts an agent and no
//     key store but the hidden app's in-memory one is used;
//   - with the Ollama preset on loopback chosen in widget-agent.json, the
//     settings show the Shared GPU note while the app is attached to the
//     stream of caliper_attention_producer, and not after the stream is
//     detached (esc Home on the live page).
// It needs the caliper executable (CALIPER_TEST_CALIPER_EXE), and the second
// case the producer (CALIPER_TEST_PRODUCER_EXE) and the built applets
// (CALIPER_TEST_APPLETS), as the side chat's live case; without Metal both
// say they are skipped. No case reaches a model or uses a key.
#include "builder/agent_settings.h"

#include "spec_test_workspace.h"

#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace spec_test;
using nlohmann::json;
namespace acp = caliper_builder::acp;

#ifndef _WIN32

namespace {

bool metal(const std::string& err) { return err.find("[renderer] metal") != std::string::npos; }

}  // namespace

TEST_CASE("agent settings app: Setup in ⌘K's no-provider state opens the settings" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_agent_settings_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("agent-settings-app");
    const std::string ws = root + "/ws";
    {
        auto s = dashboard_workspace(ws, root + "/data-files");
        ok(*s, "applet.save", {{"overwrite", true}});
        s->close();
    }
    std::filesystem::create_directories(root + "/data");
    std::filesystem::create_directories(root + "/apps");
    std::filesystem::create_directories(root + "/shots");
    acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--api", "--api-port", "0", "--workspace", ws};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_APPLETS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_AUTOLAUNCH", "local.run42-dashboard"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_AGENT_PYTHON", CALIPER_BUILDER_TEST_PYTHON},
             {"CALIPER_TEST_SPEC_UI",
              "wait 60; focus curves; wait 3; key cmd+k; until unavailable; wait 5; widget Setup; settings until open; "
              "settings until state No provider; wait 5; shot setup; key escape; wait 5; settings close; quit"},
             {"CALIPER_TEST_SPEC_UI_SHOTS", root + "/shots"},
             {"CALIPER_EXIT_AFTER", "100"}};
    o.env_remove = {"CALIPER_AGENT_TEST_SCRIPT", "ANTHROPIC_API_KEY", "CALIPER_MODEL_API_KEY", "OPENAI_API_KEY"};
    acp::RunLimits lim;
    lim.timeout_ms = 150000;
    const auto run = acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 6000 ? run.err.size() - 6000 : 0));
    REQUIRE(run.error.ok());
    if (!metal(run.err)) {
        std::printf("caliper_agent_settings_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] the surface is unavailable") != std::string::npos);
    CHECK(run.err.find("[ui] pressed Setup") != std::string::npos);
    CHECK(run.err.find("[ui] the settings are open") != std::string::npos);
    CHECK(run.err.find("[ui] the settings' state is No provider") != std::string::npos);
    CHECK(run.err.find("settings open") != std::string::npos);   // the shot's facts
    CHECK(run.err.find("[ui] done") != std::string::npos);
    // Nothing was stored: no preset was chosen, so the settings file is absent
    // or holds no provider, and never a key.
    const std::string file = root + "/data/widget-agent.json";
    if (std::filesystem::exists(file)) CHECK(read_text(file).find("sk-") == std::string::npos);
}

TEST_CASE("agent settings app: a changed setting ends the widget agent at once, and the next ⌘K makes it anew" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_agent_settings_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("agent-settings-change");
    const std::string ws = root + "/ws";
    {
        auto s = dashboard_workspace(ws, root + "/data-files");
        ok(*s, "applet.save", {{"overwrite", true}});
        s->close();
    }
    const std::string data = root + "/data";
    std::filesystem::create_directories(data);
    std::filesystem::create_directories(root + "/apps");
    // ⌘K with no provider, then Setup; choosing a preset with the prompt still
    // open closes it at once. ⌘K after the settings close makes the agent from
    // them (a server on loopback, which is never asked anything here), and
    // holds an API file of the sidecar's own; changing the model again drops
    // that agent at once, and its file and token go with it.
    const std::string files = "files " + data + " api-agent- ";
    acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--api", "--api-port", "0", "--workspace", ws};
    o.env = {{"CALIPER_DATA_DIR", data},
             {"CALIPER_USER_APPLETS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_AUTOLAUNCH", "local.run42-dashboard"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_AGENT_PYTHON", CALIPER_BUILDER_TEST_PYTHON},
             {"CALIPER_TEST_SPEC_UI",
              "wait 60; focus curves; wait 3; key cmd+k; until unavailable; " + files + "0; widget Setup; "
              "settings until open; settings preset ollama; until idle; settings model local-model; settings close; "
              "wait 3; key cmd+k; until prompt; " + files + "1; settings model other-model; until idle; " + files +
              "0; key cmd+k; until prompt; " + files + "1; quit"},
             {"CALIPER_EXIT_AFTER", "100"}};
    o.env_remove = {"CALIPER_AGENT_TEST_SCRIPT", "ANTHROPIC_API_KEY", "CALIPER_MODEL_API_KEY", "OPENAI_API_KEY"};
    acp::RunLimits lim;
    lim.timeout_ms = 150000;
    const auto run = acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 6000 ? run.err.size() - 6000 : 0));
    REQUIRE(run.error.ok());
    if (!metal(run.err)) {
        std::printf("caliper_agent_settings_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    const size_t unavailable = run.err.find("[ui] the surface is unavailable");
    const size_t chosen = run.err.find("[ui] settings preset ollama");
    const size_t closed = run.err.find("[ui] the surface is idle", chosen == std::string::npos ? 0 : chosen);
    const size_t prompt = run.err.find("[ui] the surface is prompt");
    CHECK(unavailable != std::string::npos);
    CHECK(chosen != std::string::npos);
    CHECK(closed != std::string::npos);
    CHECK(prompt != std::string::npos);
    CHECK(unavailable < chosen);
    CHECK(chosen < closed);
    CHECK(closed < prompt);
    CHECK(run.err.find("[ui] 1 file api-agent-*") != std::string::npos);
    CHECK(run.err.find("[ui] done") != std::string::npos);
    // The settings hold the last choice, and never a key.
    const std::string saved = read_text(data + "/widget-agent.json");
    CHECK(saved.find("other-model") != std::string::npos);
    CHECK(saved.find("sk-") == std::string::npos);
}

TEST_CASE("agent settings app: the Shared GPU note while a live run is attached" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* producer_exe = std::getenv("CALIPER_TEST_PRODUCER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !producer_exe || !*producer_exe || !applets || !*applets) {
        // Not a skip of the whole entry: the first case still counts.
        std::printf("caliper_agent_settings_app: the live case is not run: it needs CALIPER_TEST_PRODUCER_EXE and "
                    "CALIPER_TEST_APPLETS (macOS)\n");
        return;
    }
    const std::string root = fresh_dir("agent-settings-live");
    std::filesystem::create_directories(root + "/data");
    std::filesystem::create_directories(root + "/apps");
    std::filesystem::create_directories(root + "/ws");
    // The Ollama preset on loopback; no request is made to it.
    caliper_host::AgentSettingsData d;
    d.preset = "ollama";
    d.providers["ollama"] = {"http://127.0.0.1:9/v1", "local-model", ""};
    REQUIRE(caliper_host::write_agent_settings(root + "/data/widget-agent.json", d));

    // The training script's stand-in.
    acp::SpawnOptions po;
    po.argv = {producer_exe, "--layers", "2", "--heads", "4", "--tokens", "16", "--hz", "8"};
    caliper_builder::Error e;
    std::unique_ptr<acp::Process> producer = acp::Process::spawn(po, &e);
    REQUIRE_MESSAGE(producer, e.message);
    std::string out, link;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (link.empty() && std::chrono::steady_clock::now() < until) {
        char b[4096];
        std::size_t got = 0;
        if (producer->read_stdout(b, sizeof b, &got, 100) == acp::IoStatus::ok) out.append(b, got);
        if (producer->read_stderr(b, sizeof b, &got, 10) == acp::IoStatus::ok) out.append(b, got);
        const size_t at = out.find("[producer] hub ");
        if (at != std::string::npos && out.find('\n', at) != std::string::npos) {
            const size_t s = at + std::string("[producer] hub ").size();
            link = out.substr(s, out.find('\n', at) - s);
        }
    }
    REQUIRE_MESSAGE(!link.empty(), out);

    acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--applets", applets, "--url", link, "--workspace", root + "/ws"};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_APPLETS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI",
              "page live; settings until note Shared GPU; settings open; wait 5; shot live; settings close; "
              "key escape; page landing; settings until note none; quit"},
             {"CALIPER_TEST_SPEC_UI_SHOTS", root},
             {"CALIPER_EXIT_AFTER", "100"}};
    o.env_remove = {"CALIPER_AGENT_TEST_SCRIPT", "ANTHROPIC_API_KEY", "CALIPER_MODEL_API_KEY", "OPENAI_API_KEY"};
    acp::RunLimits lim;
    lim.timeout_ms = 150000;
    const auto run = acp::run_process(o, lim);
    producer->terminate(2000);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 6000 ? run.err.size() - 6000 : 0));
    REQUIRE(run.error.ok());
    if (!metal(run.err)) {
        std::printf("caliper_agent_settings_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    const size_t attached = run.err.find("[ui] the settings' note is Shared GPU");
    const size_t detached = run.err.find("[ui] the settings' note is none");
    CHECK(attached != std::string::npos);
    CHECK(detached != std::string::npos);
    CHECK(attached < detached);
    CHECK(run.err.find("[ui] done") != std::string::npos);
}

#endif
