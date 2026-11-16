// caliper-widget-shot as a process, on an offscreen canvas: a workspace with
// the dashboard of dashboard.spec.toml built through commands and saved as a
// spec applet, then the ⌘K scenario on its curves panel with the scripted
// agent. The images say what the surface did: the preview differs from the
// idle state inside the panel's rectangle and nowhere else, and the state
// after Accept equals a fresh render of the changed spec.
#include <stb_image.h>

#include "spec_test_workspace.h"

#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace spec_test;
using nlohmann::json;

namespace {

struct Png {
    int w = 0, h = 0;
    std::vector<unsigned char> px;
};

Png load(const std::string& path) {
    Png img;
    int ch = 0;
    unsigned char* px = stbi_load(path.c_str(), &img.w, &img.h, &ch, 4);
    REQUIRE_MESSAGE(px, path);
    img.px.assign(px, px + (size_t)img.w * img.h * 4);
    stbi_image_free(px);
    return img;
}

// Pixels that differ, inside and outside a rectangle.
struct Diff {
    size_t inside = 0, outside = 0;
};

Diff diff(const Png& a, const Png& b, int x0, int y0, int x1, int y1) {
    REQUIRE(a.w == b.w);
    REQUIRE(a.h == b.h);
    Diff d;
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < a.w; ++x) {
            const size_t i = ((size_t)y * a.w + x) * 4;
            const bool same = a.px[i] == b.px[i] && a.px[i + 1] == b.px[i + 1] && a.px[i + 2] == b.px[i + 2];
            if (same) continue;
            if (x >= x0 && x < x1 && y >= y0 && y < y1) ++d.inside;
            else ++d.outside;
        }
    return d;
}

}  // namespace

TEST_CASE("caliper-widget-shot: the ⌘K scenario on a panel of the dashboard" * doctest::test_suite("gpu")) {
    const std::string root = fresh_dir("widget-shot");
    const std::string ws = root + "/ws";
    {
        auto s = dashboard_workspace(ws, root + "/data");
        ok(*s, "applet.save", {{"overwrite", true}});
        s->close();
    }
    const std::string out = root + "/shots";
    std::filesystem::create_directories(out);
    caliper_builder::acp::SpawnOptions o;
    o.argv = {CALIPER_WIDGET_SHOT_EXE, "--workspace", ws,        "--applet", "run42-dashboard", "--panel", "curves",
              "--prompt",              "Use a linear y axis",     "--out-dir", out,  "--size", "960x600", "--prefix", "t"};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 120000;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr: " << run.err);
    REQUIRE(run.error.ok());
    if (!run.status.success() && run.err.find("cannot draw offscreen") != std::string::npos) {
        MESSAGE("skipped: no offscreen canvas");
        return;
    }
    REQUIRE(run.status.success());
    const size_t nl = run.out.rfind('\n', run.out.size() - 2);
    const json summary = json::parse(nl == std::string::npos ? run.out : run.out.substr(nl + 1));
    CHECK(summary["final_state"] == "applied");
    CHECK(summary["open_spec_unchanged_before_accept"] == true);
    CHECK(summary["undo_steps_after_accept"].get<int>() == summary["undo_steps_before_accept"].get<int>() + 1);
    REQUIRE(summary["diff"].size() == 1);
    CHECK(summary["diff"][0] == json::array({"y_log", "true", "false"}));
    std::vector<std::string> names;
    for (const auto& f : summary["files"]) names.push_back(f["name"]);
    CHECK(names == std::vector<std::string>{"idle", "prompt", "running", "preview", "accepted", "fresh"});

    const json r = summary["panel_rect"];
    const int x0 = (int)r[0].get<double>(), y0 = (int)r[1].get<double>();
    const int x1 = x0 + (int)r[2].get<double>(), y1 = y0 + (int)r[3].get<double>();
    const Png idle = load(out + "/t-idle.png"), prompt = load(out + "/t-prompt.png"), preview = load(out + "/t-preview.png");
    const Png accepted = load(out + "/t-accepted.png"), fresh = load(out + "/t-fresh.png");
    CHECK(idle.w == 960);
    CHECK(idle.h == 600);
    // The prompt and the preview change the curves panel and nothing else.
    const Diff dp = diff(idle, prompt, x0, y0, x1, y1);
    CHECK(dp.inside > 500);
    CHECK(dp.outside == 0);
    const Diff dv = diff(idle, preview, x0, y0, x1, y1);
    CHECK(dv.inside > 5000);
    CHECK(dv.outside == 0);
    // After Accept the panel is the changed spec drawn afresh, and it is not
    // the idle panel.
    const Diff da = diff(accepted, fresh, 0, 0, 0, 0);
    CHECK(da.inside + da.outside == 0);
    const Diff di = diff(idle, accepted, x0, y0, x1, y1);
    CHECK(di.inside > 1000);
    CHECK(di.outside == 0);
}

TEST_CASE("caliper-widget-shot: the command line's refusals" * doctest::test_suite("gpu")) {
    caliper_builder::acp::SpawnOptions o;
    o.argv = {CALIPER_WIDGET_SHOT_EXE, "--workspace", "x"};
    const auto run = caliper_builder::acp::run_process(o, {});
    REQUIRE(run.error.ok());
    CHECK(run.status.code == 2);
    CHECK(run.err.find("usage") != std::string::npos);
}
