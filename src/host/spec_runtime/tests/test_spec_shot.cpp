// caliper-spec-shot as a process: a workspace with a Parquet file and a
// safetensors file, the dashboard of dashboard.spec.toml built through
// commands and saved as an instrument, then drawn by the harness into a PNG.
#include <stb_image.h>

#include "spec_runtime/spec_prepare.h"

#include "spec_test_canvas.h"
#include "spec_test_workspace.h"

#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>

using namespace caliper_host;
using namespace spec_test;
using nlohmann::json;

namespace {

caliper_builder::acp::RunResult run_shot(const std::vector<std::string>& args) {
    caliper_builder::acp::SpawnOptions o;
    o.argv = {CALIPER_SPEC_SHOT_EXE};
    o.argv.insert(o.argv.end(), args.begin(), args.end());
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 120000;
    return caliper_builder::acp::run_process(o, lim);
}

Image load(const std::string& path) {
    Image img;
    int ch = 0;
    unsigned char* px = stbi_load(path.c_str(), &img.w, &img.h, &ch, 4);
    REQUIRE_MESSAGE(px, path);
    img.px.assign(px, px + (size_t)img.w * img.h * 4);
    stbi_image_free(px);
    return img;
}

}  // namespace

TEST_CASE("caliper-spec-shot: the command line's refusals") {
    auto r = run_shot({});
    CHECK(r.status.exited);
    CHECK(r.status.code == 2);
    CHECK(r.err.find("give either --instrument") != std::string::npos);
    r = run_shot({"--instrument", "x"});
    CHECK(r.status.code == 2);
    r = run_shot({"--spec", "x.toml", "--size", "10x10"});
    CHECK(r.status.code == 2);
    const std::string root = fresh_dir("shot-refusals");
    r = run_shot({"--workspace", root, "--instrument", "missing", "--out", root + "/x.png"});
    CHECK(r.status.code == 1);
    CHECK(r.err.find("no instrument missing") != std::string::npos);
    r = run_shot({"--spec", root + "/none.spec.toml", "--out", root + "/nowhere/x.png"});
    CHECK(r.status.code == 1);
}

TEST_SUITE("gpu") {

TEST_CASE("caliper-spec-shot: the dashboard over Parquet and safetensors, built through commands, saved and drawn") {
    if (!canvas()) return;
    const std::string root = fresh_dir("shot-dashboard");
    const std::string ws = root + "/workspace";
    {
        auto s = dashboard_workspace(ws, root);
        CHECK(s->open_spec()->spec == dashboard_fixture());
        ok(*s, "instrument.save");
        s->close();
    }
    const std::string out = root + "/dashboard.png";
    const auto r = run_shot({"--workspace", ws, "--instrument", "run42-dashboard", "--size", "960x600", "--out", out});
    INFO(r.err);
    REQUIRE(r.status.success());
    CHECK(r.out.find("run42-dashboard at 960x600") != std::string::npos);
    const Image img = load(out);
    CHECK(img.w == 960);
    CHECK(img.h == 600);
    // The heatmap in magma, both lines in their key colours, and the table's
    // text.
    const uint32_t* lut = spec_colormap_lut(kCmapMagma);
    CHECK(img.count(near(lut[255], 1)).n > 50);
    CHECK(img.count(near(spec_palette_color(0), 30)).n > 100);
    CHECK(img.count(near(spec_palette_color(1), 30)).n > 100);
    CHECK(img.count(bright, 0, 400, 960, 600).n > 200);
    // The workspace is read, not changed: no command ran.
    CHECK(read_text(ws + "/journal/commands.jsonl").find("panel.render") == std::string::npos);

    // A spec file with no workspace draws each panel's state.
    const std::string bare = root + "/bare.png";
    const auto b = run_shot({"--spec", std::string(CALIPER_SPEC_FIXTURES) + "/dashboard.spec.toml", "--size", "640x400",
                             "--out", bare});
    INFO(b.err);
    REQUIRE(b.status.success());
    CHECK(b.out.find("No data source") != std::string::npos);
    const Image empty = load(bare);
    CHECK(empty.count(near(lut[255], 1)).n == 0);
}

}  // TEST_SUITE("gpu")
