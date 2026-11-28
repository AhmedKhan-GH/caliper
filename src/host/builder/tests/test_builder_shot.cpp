// caliper-builder-shot as a process, on an offscreen canvas: the three-panel
// dashboard built by hand through the pane's own input over a workspace that
// imports a Parquet table and a safetensors tensor by the pane's file drop.
// The spec it ends with, and the instrument it saves, equal dashboard.spec.toml,
// as for the other three ways of building it; every command came from the
// builder origin, the dropped files' import from ui; two undos took two steps; and the captures have the size
// asked for.
#include <stb_image.h>

#include "spec_test_workspace.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace spec_test;
using nlohmann::json;
namespace spec = caliper_builder::spec;

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

size_t differing(const Png& a, const Png& b) {
    REQUIRE(a.px.size() == b.px.size());
    size_t n = 0;
    for (size_t i = 0; i < a.px.size(); i += 4)
        if (a.px[i] != b.px[i] || a.px[i + 1] != b.px[i + 1] || a.px[i + 2] != b.px[i + 2]) ++n;
    return n;
}

}  // namespace

TEST_CASE("caliper-builder-shot: the dashboard built by hand equals the fixture" * doctest::test_suite("gpu")) {
    const std::string root = fresh_dir("builder-shot");
    const std::string data = root + "/data", ws = root + "/ws", out = root + "/shots";
    std::filesystem::create_directories(out);
    write_parquet(data + "/run42_metrics.parquet",
                  "SELECT range AS step, 0.1 + 2.0 * exp(-range / 50.0) AS loss, "
                  "0.1 + 0.8 * (1 - exp(-range / 40.0)) AS accuracy FROM range(200)");
    std::vector<float> w(2 * 3 * 8 * 8, 0.0f);
    for (int l = 0; l < 2; ++l)
        for (int h = 0; h < 3; ++h)
            for (int q = 0; q < 8; ++q) w[((l * 3 + h) * 8 + q) * 8 + q] = 1.0f + (float)h;
    write_safetensors_f32(data + "/attn.safetensors", "weights", {2, 3, 8, 8}, w);

    caliper_builder::acp::SpawnOptions o;
    o.argv = {CALIPER_BUILDER_SHOT_EXE,
              "--workspace", ws,
              "--instrument-from", std::string(CALIPER_SPEC_FIXTURES) + "/dashboard.spec.toml",
              "--files", data + "/run42_metrics.parquet", data + "/attn.safetensors",
              "--out-dir", out, "--size", "1440x900", "--prefix", "t", "--suffix", "-small"};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 240000;
    lim.max_output_bytes = 64u << 20;
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

    // The spec the pane ends with, and the instrument it saved, are the fixture.
    const spec::Spec d = dashboard_fixture();
    auto finished = spec::parse_toml(summary["finished_text"].get<std::string>());
    REQUIRE(finished.ok());
    CHECK(finished.spec == d);
    auto last = spec::parse_toml(summary["final_text"].get<std::string>());
    REQUIRE(last.ok());
    CHECK(last.spec == d);
    REQUIRE(summary["final_path"] == "instruments/run42-dashboard/run42-dashboard.spec.toml");
    auto saved = spec::parse_toml(read_text(ws + "/" + summary["final_path"].get<std::string>()));
    REQUIRE(saved.ok());
    CHECK(saved.spec == d);
    // The two undos took two steps and changed the spec.
    CHECK(summary["undo_steps_after_undo"].get<int>() == summary["undo_steps_before_undo"].get<int>() - 2);
    CHECK(summary["undone_text"] != summary["finished_text"]);
    // Every command from the builder origin, and so every line of the journal,
    // but the import of the dropped files, which is the person's ui.
    REQUIRE(summary["commands"].size() > 20);
    for (const auto& c : summary["commands"]) CHECK(c["origin"] == (c["name"] == "data.import" ? "ui" : "builder"));
    std::ifstream journal(ws + "/journal/commands.jsonl");
    std::string line;
    size_t lines = 0;
    while (std::getline(journal, line)) {
        if (line.empty()) continue;
        ++lines;
        const json e = json::parse(line);
        CHECK(e["origin"] == (e["name"] == "data.import" ? "ui" : "builder"));
    }
    CHECK(lines > 20);

    // The six captures, at the size asked for; the undone dashboard differs
    // from the finished one.
    std::vector<std::string> names;
    for (const auto& f : summary["files"]) names.push_back(f["name"]);
    CHECK(names == std::vector<std::string>{"1-empty", "2-series", "3-heatmap", "4-divider", "5-text", "6-undo"});
    for (const auto& n : names) {
        const Png img = load(out + "/t-" + n + "-small.png");
        CHECK(img.w == 1440);
        CHECK(img.h == 900);
    }
    CHECK(differing(load(out + "/t-5-text-small.png"), load(out + "/t-6-undo-small.png")) > 1000);
    CHECK(differing(load(out + "/t-1-empty-small.png"), load(out + "/t-2-series-small.png")) > 10000);
}

TEST_CASE("caliper-builder-shot: the command line's refusals" * doctest::test_suite("gpu")) {
    caliper_builder::acp::SpawnOptions o;
    o.argv = {CALIPER_BUILDER_SHOT_EXE, "--workspace", "x"};
    const auto run = caliper_builder::acp::run_process(o, {});
    REQUIRE(run.error.ok());
    CHECK(run.status.code == 2);
    CHECK(run.err.find("usage") != std::string::npos);
}
