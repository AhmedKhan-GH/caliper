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
#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

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

namespace {

// One hidden app run of a ⌘K target: the dashboard saved as a spec applet,
// the sidecar on `steps`, the input `ui`. Returns false when it cannot run
// here (no caliper executable, no Python, no Metal), after saying so.
struct AppRun {
    std::string root, ws, shots, record, err;
    bool ran = false;
    int code = -1;
};

json tool_step(json tools) {
    return {{"message", {{"content", std::move(tools)}, {"stop_reason", "tool_use"},
                         {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}};
}

json text_step(const std::string& text) {
    return {{"message", {{"content", json::array({{{"type", "text"}, {"text", text}}})}, {"stop_reason", "end_turn"},
                         {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}};
}

json tool_use(const std::string& id, const std::string& name, json input) {
    return {{"type", "tool_use"}, {"id", id}, {"name", name}, {"input", std::move(input)}};
}

AppRun run_target(const std::string& name, const json& steps, const std::string& ui) {
    AppRun out;
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe || std::string(CALIPER_BUILDER_TEST_PYTHON).empty()) {
        std::printf("caliper_widget_app skipped: needs CALIPER_TEST_CALIPER_EXE and CALIPER_BUILDER_TEST_PYTHON\n");
        return out;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    out.root = fresh_dir("widget-app-" + name);
    out.ws = out.root + "/ws";
    {
        auto s = dashboard_workspace(out.ws, out.root + "/data-files");
        ok(*s, "applet.save", {{"overwrite", true}});
        s->close();
    }
    out.shots = out.root + "/shots";
    out.record = out.root + "/requests.jsonl";
    const std::string script = out.root + "/script.json";
    std::filesystem::create_directories(out.shots);
    std::filesystem::create_directories(out.root + "/data");
    std::filesystem::create_directories(out.root + "/apps");
    std::ofstream(script) << json{{"steps", steps}, {"record", out.record}, {"backoff_scale", 0}}.dump();
    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--api", "--api-port", "0", "--workspace", out.ws};
    o.env = {{"CALIPER_DATA_DIR", out.root + "/data"},
             {"CALIPER_USER_APPLETS_DIR", out.root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "1280x800"},
             {"CALIPER_AUTOLAUNCH", "local.run42-dashboard"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_AGENT_PYTHON", CALIPER_BUILDER_TEST_PYTHON},
             {"CALIPER_AGENT_TEST_SCRIPT", script},
             {"ANTHROPIC_API_KEY", "sk-ant-api03-TESTONLY-inline-app-0123456789abcdef-fakeAA"},
             {"CALIPER_TEST_SPEC_UI", ui},
             {"CALIPER_TEST_SPEC_UI_SHOTS", out.shots},
             {"CALIPER_EXIT_AFTER", "100"}};
    o.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 150000;
    const auto run = caliper_builder::acp::run_process(o, lim);
    REQUIRE(run.error.ok());
    out.err = run.err;
    out.code = run.status.exited ? run.status.code : -1;
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_widget_app skipped: no Metal device\n");
        return out;
    }
    out.ran = true;
    return out;
}

// The journal's agent edits in a widget preview's group, and the person's commit.
struct Committed {
    std::vector<std::string> edits;
    bool commit = false;
};

Committed committed(const std::string& ws) {
    Committed c;
    for (const auto& e : caliper_builder::read_journal(ws + "/journal/commands.jsonl").events) {
        if (e.name == "preview.commit") c.commit = c.commit || e.origin == caliper_builder::CommandOrigin::ui;
        if (e.origin == caliper_builder::CommandOrigin::agent && e.group.rfind("widget-", 0) == 0) c.edits.push_back(e.name);
    }
    return c;
}

}  // namespace

TEST_CASE("widget app: a selection's table, through the sidecar, previewed beside the panel and committed" *
          doctest::test_suite("app")) {
    const json steps = json::array(
        {tool_step(json::array(
             {tool_use("toolu_1", "source_add",
                       {{"id", "selected_steps"},
                        {"sql", "SELECT step, loss, accuracy FROM run42_metrics WHERE step BETWEEN 60 AND 120"}}),
              tool_use("toolu_2", "panel_create",
                       {{"id", "selected_rows"}, {"primitive", "table"}, {"bind", {{"rows", "selected_steps"}}}}),
              tool_use("toolu_3", "layout_place", {{"panel", "selected_rows"}, {"target", "curves"}, {"where", "beside"}})})),
         text_step("A table of the selected steps beside the curves.")});
    const AppRun r = run_target("selection", steps,
                                "wait 60; select curves 0.35 0.6; wait 10; leave; key cmd+k; until prompt; wait 3; "
                                "type Make a table of these points; wait 5; key enter; until preview; wait 10; "
                                "shot preview; key enter; until applied; wait 5; quit");
    if (!r.ran) return;
    INFO("stderr tail: " << r.err.substr(r.err.size() > 6000 ? r.err.size() - 6000 : 0));
    CHECK(r.code == 0);
    CHECK(r.err.find("[ui] done") != std::string::npos);
    CHECK(r.err.find("selected on curves") != std::string::npos);
    CHECK(r.err.find("surface preview, focus curves, undo 1") != std::string::npos);
    CHECK(std::filesystem::is_regular_file(r.shots + "/preview.png"));
    // The selection's rows reached the model, with the selection's tools.
    REQUIRE(std::filesystem::is_regular_file(r.record));
    const std::string requests = read_text(r.record);
    CHECK(requests.find("selection") != std::string::npos);
    CHECK(requests.find("\"panel_create\"") != std::string::npos);
    CHECK(requests.find("\"panel_patch\"") == std::string::npos);
    const Committed c = committed(r.ws);
    CHECK(c.commit);
    CHECK(c.edits == std::vector<std::string>{"source.add", "panel.create", "layout.place"});
}

TEST_CASE("widget app: a dropped file's question, through the sidecar, answered in its row's card" *
          doctest::test_suite("app")) {
    const json steps = json::array({text_step("A table of 50 rows with two columns, step and val_loss.")});
    // The file to drop, written before the app starts.
    const std::string files = fresh_dir("widget-app-file-input");
    write_parquet(files + "/validation.parquet",
                  "SELECT range AS step, 0.5 + 0.1 * cos(range / 7.0) AS val_loss FROM range(50)");
    const AppRun r = run_target("file", steps,
                                "wait 60; drop " + files + "/validation.parquet; pane source table:validation; wait 10; "
                                "click cmdk:source:table:validation; until prompt; wait 3; type What's in this?; wait 5; "
                                "key enter; until preview; wait 10; shot answer; quit");
    if (!r.ran) return;
    INFO("stderr tail: " << r.err.substr(r.err.size() > 6000 ? r.err.size() - 6000 : 0));
    CHECK(r.code == 0);
    CHECK(r.err.find("[ui] done") != std::string::npos);
    CHECK(r.err.find("shot answer") != std::string::npos);
    CHECK(r.err.find("surface preview") != std::string::npos);
    REQUIRE(std::filesystem::is_regular_file(r.record));
    const std::string requests = read_text(r.record);
    CHECK(requests.find("validation") != std::string::npos);
    CHECK(requests.find("\"loader_describe\"") != std::string::npos);   // the dropped file's scope
    CHECK(committed(r.ws).edits.empty());   // an answer, no change
}

TEST_CASE("widget app: a property's edit, through the sidecar, previewed in its panel from Properties" *
          doctest::test_suite("app")) {
    const json steps = json::array(
        {tool_step(json::array({tool_use("toolu_1", "panel_patch",
                                         {{"id", "attention"}, {"patch", {{"options", {{"colormap", "rdbu"}}}}}})})),
         text_step("The attention uses the diverging colormap.")});
    const AppRun r = run_target("property", steps,
                                "wait 60; key cmd+b; wait 20; click panel:attention 0.5 0.7; pane focus attention; "
                                "wait 10; click cmdk:prop:options.colormap; until prompt; wait 3; "
                                "type Use the diverging colormap; wait 5; key enter; until preview; wait 10; "
                                "shot preview; key enter; until applied; wait 5; quit");
    if (!r.ran) return;
    INFO("stderr tail: " << r.err.substr(r.err.size() > 6000 ? r.err.size() - 6000 : 0));
    CHECK(r.code == 0);
    CHECK(r.err.find("[ui] done") != std::string::npos);
    CHECK(r.err.find("surface preview, focus attention") != std::string::npos);
    REQUIRE(std::filesystem::is_regular_file(r.record));
    const std::string requests = read_text(r.record);
    CHECK(requests.find("options.colormap") != std::string::npos);
    CHECK(requests.find("\"panel_patch\"") != std::string::npos);
    CHECK(requests.find("\"panel_create\"") == std::string::npos);   // one property: panel.patch only
    const Committed c = committed(r.ws);
    CHECK(c.commit);
    CHECK(c.edits == std::vector<std::string>{"panel.patch"});
}


namespace {

// The crash reports of the caliper executable in the system's folder for
// them, on macOS; empty elsewhere.
std::set<std::string> crash_reports() {
    std::set<std::string> out;
#ifdef __APPLE__
    const char* home = std::getenv("HOME");
    if (!home) return out;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(std::string(home) + "/Library/Logs/DiagnosticReports", ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("caliper-", 0) == 0) out.insert(name);
    }
#endif
    return out;
}

// Starts the hidden app on the dashboard with `ui`, waits for the input
// driver's "[ui] done", sends SIGTERM, and returns how it ended with what it
// wrote on standard error. `ran` is false when it cannot run here.
struct QuitRun {
    bool ran = false;
    caliper_builder::acp::ExitStatus status;
    std::string err;
};

QuitRun quit_while_open(const std::string& name, const std::string& ui, const json& steps) {
    QuitRun out;
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe || std::string(CALIPER_BUILDER_TEST_PYTHON).empty()) {
        std::printf("caliper_widget_app skipped: needs CALIPER_TEST_CALIPER_EXE and CALIPER_BUILDER_TEST_PYTHON\n");
        return out;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("widget-quit-" + name);
    const std::string ws = root + "/ws";
    {
        auto s = dashboard_workspace(ws, root + "/data-files");
        ok(*s, "applet.save", {{"overwrite", true}});
        s->close();
    }
    std::filesystem::create_directories(root + "/data");
    std::filesystem::create_directories(root + "/apps");
    const std::string script = root + "/script.json";
    std::ofstream(script) << json{{"steps", steps}, {"record", root + "/requests.jsonl"}, {"backoff_scale", 0}}.dump();
    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--api", "--api-port", "0", "--workspace", ws};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_APPLETS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_AUTOLAUNCH", "local.run42-dashboard"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_AGENT_PYTHON", CALIPER_BUILDER_TEST_PYTHON},
             {"CALIPER_AGENT_TEST_SCRIPT", script},
             {"ANTHROPIC_API_KEY", "sk-ant-api03-TESTONLY-inline-quit-0123456789abcdef-fakeAA"},
             {"CALIPER_TEST_SPEC_UI", ui},
             {"CALIPER_EXIT_AFTER", "100"}};
    o.env_remove = {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"};
    caliper_builder::Error error;
    std::unique_ptr<caliper_builder::acp::Process> p = caliper_builder::acp::Process::spawn(o, &error);
    REQUIRE_MESSAGE(p, error.message);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
    char buf[4096];
    while (std::chrono::steady_clock::now() < deadline && out.err.find("[ui] done") == std::string::npos &&
           out.err.find("[ui] stopped") == std::string::npos) {
        std::size_t got = 0;
        if (p->read_stderr(buf, sizeof buf, &got, 50) == caliper_builder::acp::IoStatus::ok) out.err.append(buf, got);
        if (p->read_stdout(buf, sizeof buf, &got, 0) != caliper_builder::acp::IoStatus::ok) got = 0;
        caliper_builder::acp::ExitStatus early;
        if (p->try_wait(&early)) break;
    }
    // SIGTERM, as the system sends it on logout: the app closes its window
    // and runs its teardown; more than 30 s and it is killed.
    out.status = p->terminate(30000);
    for (;;) {
        std::size_t got = 0;
        if (p->read_stderr(buf, sizeof buf, &got, 0) != caliper_builder::acp::IoStatus::ok || got == 0) break;
        out.err.append(buf, got);
    }
    if (out.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_widget_app skipped: no Metal device\n");
        return out;
    }
    out.ran = true;
    return out;
}

json linear_steps() {
    return json::array(
        {{{"message",
           {{"content", json::array({{{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "panel_patch"},
                                      {"input", {{"id", "curves"}, {"patch", {{"options", {{"y_log", false}}}}}}}}})},
            {"stop_reason", "tool_use"},
            {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}},
         {{"message",
           {{"content", json::array({{{"type", "text"}, {"text", "Turns off the log scale of the y axis."}}})},
            {"stop_reason", "end_turn"},
            {"usage", {{"input_tokens", 30}, {"output_tokens", 12}}}}}}});
}

void check_clean_quit(const QuitRun& r, const std::set<std::string>& reports_before) {
    INFO("stderr tail: " << r.err.substr(r.err.size() > 4000 ? r.err.size() - 4000 : 0));
    CHECK(r.err.find("[ui] done") != std::string::npos);
    CHECK(r.status.exited);
    CHECK(r.status.code == 0);
    CHECK(r.status.signal == 0);
    // No crash report: the system writes one within moments of a crash.
    std::set<std::string> after;
    for (int i = 0; i < 20; ++i) {
        after = crash_reports();
        if (after != reports_before) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::vector<std::string> fresh;
    for (const auto& n : after)
        if (!reports_before.count(n)) fresh.push_back(n);
    CHECK_MESSAGE(fresh.empty(), (fresh.empty() ? std::string() : fresh.front()));
}

}  // namespace

TEST_CASE("widget app: quitting with a ⌘K prompt open on a panel exits 0" * doctest::test_suite("app")) {
    const std::set<std::string> before = crash_reports();
    const QuitRun r = quit_while_open("prompt",
                                      "wait 60; focus curves; wait 3; leave; key cmd+k; until prompt; wait 3; "
                                      "type Use a linear y axis; wait 10",
                                      linear_steps());
    if (!r.ran) return;
    CHECK(r.err.find("the surface is prompt") != std::string::npos);
    check_clean_quit(r, before);
}

TEST_CASE("widget app: quitting with a ⌘K preview showing exits 0" * doctest::test_suite("app")) {
    const std::set<std::string> before = crash_reports();
    const QuitRun r = quit_while_open("preview",
                                      "wait 60; focus curves; wait 3; leave; key cmd+k; until prompt; wait 3; "
                                      "type Use a linear y axis; wait 5; key enter; until preview; wait 10",
                                      linear_steps());
    if (!r.ran) return;
    CHECK(r.err.find("the surface is preview") != std::string::npos);
    check_clean_quit(r, before);
}

#endif
