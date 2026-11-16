// ⌘K in the hidden app: `caliper --hidden --api` on a workspace that holds the
// dashboard of dashboard.spec.toml saved as a spec applet, its spec page opened
// by CALIPER_AUTOLAUNCH, the real caliper-agent sidecar playing a test script
// (CALIPER_AGENT_TEST_SCRIPT, a fake key, CALIPER_AGENT_OFFLINE=1, no key
// store), and the person's input played by CALIPER_TEST_SPEC_UI: a click on
// the curves panel, ⌘K, the request typed, Enter, Enter on the preview. The
// app reads four frames back, and the workspace's journal holds the agent's
// edit committed by the person as one step.
//
// It needs the caliper executable (CALIPER_TEST_CALIPER_EXE, set by its ctest
// entry), a Metal device and CALIPER_BUILDER_TEST_PYTHON; without them it says
// it is skipped.
#include <stb_image.h>

#include "spec_test_workspace.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/journal.h>

#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

using namespace spec_test;
using nlohmann::json;

#ifndef _WIN32

TEST_CASE("widget app: ⌘K on the hidden app's spec page, through the sidecar, to one undo step" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe || std::string(CALIPER_BUILDER_TEST_PYTHON).empty()) {
        std::printf("caliper_widget_app skipped: needs CALIPER_TEST_CALIPER_EXE and CALIPER_BUILDER_TEST_PYTHON\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("widget-app");
    const std::string ws = root + "/ws";
    {
        auto s = dashboard_workspace(ws, root + "/data-files");
        ok(*s, "applet.save", {{"overwrite", true}});
        s->close();
    }
    const std::string shots = root + "/shots", script = root + "/script.json", record = root + "/requests.jsonl";
    std::filesystem::create_directories(shots);
    std::filesystem::create_directories(root + "/data");
    std::filesystem::create_directories(root + "/apps");
    {
        const json steps = json::array(
            {{{"message",
               {{"content", json::array({{{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "panel_patch"},
                                          {"input", {{"id", "curves"}, {"patch", {{"options", {{"y_log", false}}}}}}}}})},
                {"stop_reason", "tool_use"},
                {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}},
             {{"message",
               {{"content", json::array({{{"type", "text"}, {"text", "Turns off the log scale of the y axis."}}})},
                {"stop_reason", "end_turn"},
                {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}}});
        std::ofstream(script) << json{{"steps", steps}, {"record", record}, {"backoff_scale", 0}}.dump();
    }
    const char* const fake_key = "sk-ant-api03-TESTONLY-inline-app-0123456789abcdef-fakeAA";
    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--api", "--api-port", "0", "--workspace", ws};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_APPLETS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_AUTOLAUNCH", "local.run42-dashboard"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_AGENT_PYTHON", CALIPER_BUILDER_TEST_PYTHON},
             {"CALIPER_AGENT_TEST_SCRIPT", script},
             {"ANTHROPIC_API_KEY", fake_key},
             {"CALIPER_TEST_SPEC_UI",
              "wait 60; focus curves; wait 3; leave; wait 10; shot focus; key cmd+k; until prompt; wait 3; "
              "type Use a linear y axis; wait 10; shot prompt; key enter; until preview; wait 10; shot preview; "
              "key enter; until applied; wait 10; shot accepted; quit"},
             {"CALIPER_TEST_SPEC_UI_SHOTS", shots},
             {"CALIPER_EXIT_AFTER", "100"}};
    o.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 150000;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 6000 ? run.err.size() - 6000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_widget_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
    CHECK(run.err.find("shot accepted") != std::string::npos);
    CHECK(run.err.find("surface preview, focus curves, undo 1") != std::string::npos);
    CHECK(run.err.find("surface applied, focus curves, undo 2") != std::string::npos);
    for (const char* name : {"focus", "prompt", "preview", "accepted"}) {
        int w = 0, h = 0, n = 0;
        const std::string file = shots + "/" + name + ".png";
        unsigned char* px = stbi_load(file.c_str(), &w, &h, &n, 4);
        CHECK_MESSAGE(px != nullptr, file);
        if (px) stbi_image_free(px);
        CHECK(((w == 960 && h == 600) || (w == 1920 && h == 1200)));
    }
    // The scripted model answered, and the key never reached a request.
    REQUIRE(std::filesystem::is_regular_file(record));
    CHECK(read_text(record).find(fake_key) == std::string::npos);
    // One step: the person's commit, then the agent's edit in the preview's group.
    const auto journal = caliper_builder::read_journal(ws + "/journal/commands.jsonl");
    bool commit = false, edit = false;
    for (const auto& e : journal.events) {
        if (e.name == "preview.commit") commit = commit || e.origin == caliper_builder::CommandOrigin::ui;
        if (e.name == "panel.patch" && e.origin == caliper_builder::CommandOrigin::agent)
            edit = edit || e.group.rfind("widget-", 0) == 0;
    }
    CHECK(commit);
    CHECK(edit);
}

#endif
