// Building by hand in the hidden app: `caliper --hidden` on an empty
// workspace, with the person's input played by CALIPER_TEST_SPEC_UI. A
// Parquet file and a safetensors file are dropped on the landing (the drop
// step calls the function GLFW's drop callback calls, since a hidden window
// receives no real drop), which makes a new spec and opens it in build mode
// with the two sources imported; the spec is named, the three-panel
// dashboard is built through the builder pane, the text strip and the side
// chat are opened in build mode, the applet is saved with ⌘S, the page goes
// Home, and the dashboard is opened again from the landing's command line.
// The saved spec equals dashboard.spec.toml.
//
// It needs the caliper executable (CALIPER_TEST_CALIPER_EXE, set by its ctest
// entry) and a Metal device; without them it says it is skipped.
#include <stb_image.h>

#include "spec_test_workspace.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace spec_test;
using nlohmann::json;

#ifndef _WIN32

namespace {

// The acceptance by hand, as one input script.
std::string acceptance_script(const std::string& data) {
    const std::string steps[] = {
        "wait 60",
        "drop " + data + "/run42_metrics.parquet " + data + "/attn.safetensors",
        "page spec",
        "pane source table:run42_metrics",
        "pane source tensor:attn_weights",
        "wait 20", "leave", "wait 5", "shot app-builder-1-dropped",
        // The spec's id, title and description, with nothing focused.
        "key escape",
        "click prop:spec.id", "type run42-dashboard", "key enter",
        "click prop:spec.title", "key cmd+a", "type Run 42", "key enter",
        "click prop:spec.description",
        "type Loss and accuracy of run 42, the attention of its first layer,\\nand the metrics as a table.",
        "click canvas 0.5 0.5",
        // The curves: the metrics table, named, as a series.
        "click source:table:run42_metrics 0.6 0.5",
        "click prop:source.id", "type metrics", "key enter",
        "click primitive:series", "pane focus series",
        "click prop:id", "type curves", "key enter", "pane focus curves",
        "click prop:title", "type Loss and accuracy", "key enter",
        "click option:x_label", "type Step", "key enter",
        "click option:y_log",
        "wait 10", "leave", "wait 10", "shot app-builder-2-series",
        // The attention: its source and three transforms, the heatmap beside the curves.
        "click source:tensor:attn_weights 0.6 0.5",
        "click prop:source.id", "type attn", "key enter",
        "click transform:add:slice", "click transform:add:reduce", "click transform:add:normalise",
        "drag primitive:heatmap to panel:curves -40 0.55", "pane focus heatmap",
        "click prop:id", "type attention", "key enter", "pane focus attention",
        "click prop:title", "type Mean attention of layer 0", "key enter",
        "click colormap:magma",
        "click option:range:min", "type 0", "key enter",
        "click option:x_label", "type Key", "key enter",
        "click option:y_label", "type Query", "key enter",
        // The table under everything, its columns, the root divider.
        "click source:spec:metrics 0.6 0.5",
        "drag primitive:table to canvas 0.5 -4", "pane focus table",
        "click prop:id", "type rows", "key enter", "pane focus rows",
        "click option:columns:accuracy",
        "drag divider: 0.5 0.0 to canvas 0.5 0.65",
        "wait 10", "leave", "wait 10", "shot app-builder-3-three-panels",
        // The spec text, then the side chat under Properties.
        "click bar:text", "click panel:curves 0.5 0.6", "wait 10", "leave", "wait 10", "shot app-builder-4-text",
        "chat open", "wait 40", "leave", "wait 10", "shot app-builder-5-with-chat", "chat close",
        // Saved, Home, and opened again from the landing's command line.
        "key cmd+s", "wait 20",
        "key escape", "wait 5", "key escape",
        "page landing", "wait 30",
        "key cmd+k", "wait 5", "type Run 42", "wait 5", "key enter",
        "page spec", "wait 40", "leave", "wait 10", "shot app-builder-6-reopened",
        "quit"};
    std::string out;
    for (const auto& s : steps) out += (out.empty() ? "" : "; ") + s;
    return out;
}

}  // namespace

TEST_CASE("builder app: the dashboard built by hand in the hidden app, saved and opened again" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_builder_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("builder-app");
    const std::string ws = root + "/ws", data = root + "/files", shots = root + "/shots";
    for (const auto& d : {shots, root + "/data", root + "/apps"}) std::filesystem::create_directories(d);
    write_parquet(data + "/run42_metrics.parquet",
                  "SELECT range AS step, 0.1 + 2.0 * exp(-range / 50.0) AS loss, "
                  "0.1 + 0.8 * (1 - exp(-range / 40.0)) AS accuracy FROM range(200)");
    std::vector<float> w(2 * 3 * 8 * 8, 0.0f);
    for (int l = 0; l < 2; ++l)
        for (int h = 0; h < 3; ++h)
            for (int q = 0; q < 8; ++q) w[((l * 3 + h) * 8 + q) * 8 + q] = 1.0f + (float)h;
    write_safetensors_f32(data + "/attn.safetensors", "weights", {2, 3, 8, 8}, w);

    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--workspace", ws};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_APPLETS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "1440x900"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", acceptance_script(data)},
             {"CALIPER_TEST_SPEC_UI_SHOTS", shots},
             {"CALIPER_EXIT_AFTER", "200"}};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 300000;
    lim.max_output_bytes = 64u << 20;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 8000 ? run.err.size() - 8000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);

    // The saved applet is the fixture.
    const std::string saved = ws + "/applets/run42-dashboard/run42-dashboard.spec.toml";
    REQUIRE(std::filesystem::is_regular_file(saved));
    auto parsed = caliper_builder::spec::parse_toml(read_text(saved));
    REQUIRE(parsed.ok());
    CHECK(parsed.spec == dashboard_fixture());

    // The journal: the new spec and the import from the person's ui, the
    // build from the builder, the save, and the applet opened again after it.
    const auto journal = caliper_builder::read_journal(ws + "/journal/commands.jsonl");
    bool made = false, imported = false, saved_line = false, reopened = false;
    size_t built = 0;
    for (const auto& e : journal.events) {
        if (e.name == "spec.new") made = made || e.origin == caliper_builder::CommandOrigin::ui;
        if (e.name == "data.import") imported = imported || (e.ok && e.origin == caliper_builder::CommandOrigin::ui);
        if (e.origin == caliper_builder::CommandOrigin::builder && e.ok) ++built;
        if (e.name == "applet.save" && e.ok) saved_line = true;
        if (e.name == "applet.open" && e.ok && saved_line) reopened = true;
    }
    CHECK(made);
    CHECK(imported);
    CHECK(built > 20);
    CHECK(saved_line);
    CHECK(reopened);

    // The six captures, at the window's size in points or pixels.
    for (const char* name : {"app-builder-1-dropped", "app-builder-2-series", "app-builder-3-three-panels",
                             "app-builder-4-text", "app-builder-5-with-chat", "app-builder-6-reopened"}) {
        int iw = 0, ih = 0, n = 0;
        const std::string file = shots + "/" + name + ".png";
        unsigned char* px = stbi_load(file.c_str(), &iw, &ih, &n, 4);
        CHECK_MESSAGE(px != nullptr, file);
        if (px) stbi_image_free(px);
        CHECK(((iw == 1440 && ih == 900) || (iw == 2880 && ih == 1800)));
    }
}

// New dashboard takes the landing's command line as its title, so the line is
// empty when the person comes back Home: with F3, and with "new <title>" and
// Enter. The line used to keep the title while it had the keyboard, since the
// line's own text won over the cleared buffer, and the instruments came back
// filtered by it.
TEST_CASE("builder app: New dashboard leaves the landing's command line empty" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_builder_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("builder-app-new");
    for (const auto& d : {root + "/data", root + "/apps"}) std::filesystem::create_directories(d);
    const std::string script =
        "wait 60; page landing; "
        "key cmd+k; type Run 42; key f3; page spec; wait 10; key escape; page landing; wait 10; command-line none; "
        "key cmd+k; type new Run 43; key enter; page spec; wait 10; key escape; page landing; wait 10; command-line none; "
        "quit";
    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--workspace", root + "/ws"};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_APPLETS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "1440x900"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", script},
             {"CALIPER_EXIT_AFTER", "120"}};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 180000;
    lim.max_output_bytes = 16u << 20;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
    // Both dashboards were made from the person's origin, with their titles.
    std::vector<std::string> titles;
    for (const auto& e : caliper_builder::read_journal(root + "/ws/journal/commands.jsonl").events)
        if (e.name == "spec.new" && e.ok && e.origin == caliper_builder::CommandOrigin::ui)
            titles.push_back(e.args.value("title", std::string()));
    CHECK(titles == std::vector<std::string>{"Run 42", "Run 43"});
}

#endif
