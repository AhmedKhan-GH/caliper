// caliper-runs-panel-shot: a development tool that draws the runs panel
// (runs_panel.h) offscreen over a session and a fake model runner, and writes
// a PNG of the canvas, as caliper-agent-settings-shot does for the agent's
// settings. It opens no window, is not installed and starts no process.
//
//   caliper-runs-panel-shot --scenario NAME --out FILE --workspace DIR [options]
//
// Scenarios:
//   table      four runs (a run, its fork, an evaluation that ended and a
//              start that failed) and the form of a start
//   fork-card  the same runs, with Fork on train-1, lr changed to 0.02 and
//              the card open
// Options:
//   --size WxH   the canvas (1440x900)
//   --frames N   frames drawn (4)
//
// The workspace folder is made new, with scripts/train.py and
// scripts/eval.py. It prints one JSON object: the file, the scenario and
// what the panel shows. Without a canvas (no GPU) it prints
// "caliper_runs_panel_shot skipped: <why>" and exits with 0.
#include "builder/runs_panel.h"
#include "builder/tests/runs_panel_fake_runner.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"

#include "export_service.h"

#include <caliper_builder/session/session.h>

#include <imgui.h>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace caliper_host;
using nlohmann::json;

namespace {

int usage(const char* why) {
    std::fprintf(stderr,
                 "caliper-runs-panel-shot: %s\n"
                 "usage: caliper-runs-panel-shot --scenario table|fork-card --out FILE --workspace DIR\n"
                 "                               [--size WxH] [--frames N]\n",
                 why);
    return 2;
}

int fail(const std::string& message) {
    std::fprintf(stderr, "caliper-runs-panel-shot: %s\n", message.c_str());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string scenario, out, workspace;
    int width = 1440, height = 900, frames = 4;
    for (int i = 1; i < argc; i += 2) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!v) return usage(("missing a value after " + a).c_str());
        if (a == "--scenario") scenario = v;
        else if (a == "--out") out = v;
        else if (a == "--workspace") workspace = v;
        else if (a == "--frames") frames = std::atoi(v);
        else if (a == "--size") {
            if (std::sscanf(v, "%dx%d", &width, &height) != 2) return usage("--size takes WxH, such as 1440x900");
        } else {
            return usage(("unknown option " + a).c_str());
        }
    }
    if (scenario != "table" && scenario != "fork-card") return usage("--scenario must be table or fork-card");
    if (out.empty() || workspace.empty()) return usage("--out and --workspace are required");
    if (width < 64 || height < 64 || width > 8192 || height > 8192) return usage("--size must be from 64x64 to 8192x8192");
    if (frames < 1 || frames > 120) return usage("--frames must be from 1 to 120");

    std::error_code ec;
    fs::remove_all(fs::u8path(workspace), ec);
    fs::create_directories(fs::u8path(workspace) / "scripts", ec);
    for (const char* name : {"train.py", "eval.py"})
        std::ofstream(fs::u8path(workspace) / "scripts" / name) << "print('" << name << "')\n";
    caliper_builder::Session::Open opened = caliper_builder::Session::open(workspace);
    if (opened.error) return fail("cannot open the workspace: " + opened.error.message);
    caliper_builder::Session& session = *opened.session;
    runs_panel_test::FakeRunner runner;
    runner.runs = runs_panel_test::sample_runs();
    session.set_model_runner(&runner);

    RunsPanel panel;
    panel.frame(&session, &runner);
    panel.set_open(true);
    panel.set_script("scripts/train.py");
    panel.set_args("2000");
    panel.set_params("lr=0.05");
    if (scenario == "fork-card") {
        panel.begin_fork("train-1");
        panel.set_params("lr=0.02");
        if (!panel.review()) return fail("the card did not open: " + panel.problem());
    }

    std::string why;
    std::unique_ptr<SpecOffscreen> canvas = SpecOffscreen::create(width, height, 1.0f, &why);
    if (!canvas) {
        session.set_model_runner(nullptr);
        std::printf("caliper_runs_panel_shot skipped: %s\n", why.c_str());
        return 0;
    }
    std::vector<uint8_t> rgba;
    const bool ok = canvas->render(frames, [&](const ImRect& area) {
        panel.frame(&session, &runner);
        panel.draw(area.Min.x, area.Min.y, area.GetWidth(), area.GetHeight(), canvas->look());
    }, &rgba, &why);
    if (!ok) return fail("the canvas: " + why);
    if (!export_write_png_atomic(out, rgba.data(), (uint32_t)width, (uint32_t)height, &why))
        return fail("cannot write " + out + ": " + why);

    json shown = json::array();
    for (const auto& t : panel.visible_text()) shown.push_back(t);
    const json result{{"file", out},       {"scenario", scenario},  {"size", {width, height}},
                      {"card", panel.card()}, {"runs", panel.runs().size()}, {"shown", shown}};
    canvas.reset();
    session.set_model_runner(nullptr);
    std::printf("%s\n", result.dump().c_str());
    return 0;
}
