// scene_parity_app.mm: the app's side of the scene comparison between the app and the browser
// module (scene_parity.h).
//
//   caliper_scene_parity_app <out>
//
// draws each fixture of scene_parity::fixtures() through the app's scene view, whose shapes the
// library pieces draw, on its Metal device in the offscreen mode of the graphics layer, no window,
// at 256 by 256, exactly as the graphics suite's rows apply and draw it (tests/scene/primitive_rows.h).
// For each it writes into <out>/scene/:
//
//   <fixture>.color   the view's color target after its first frame, RGBA8, top row first
//   <fixture>.id      the pieces' ID target of that frame, RG32Uint (pick id, item), top row first
//   <fixture>.json    the facts of that frame (scene_parity::frame_json), the pick points and the hit
//                     of a pick at each
//   <fixture>.png     the color target, for the captures
//
// and <out>/scene/report.txt with one line for each fixture, or <out>/scene/skipped.txt with the
// reason when it drew nothing. web/test/draw_parity.spec.ts draws the same fixtures with the browser
// module in headless Chromium and compares. Without a Metal device it exits 77, which ctest reports
// as skipped.
#import <Metal/Metal.h>

#include "scene_parity.h"

#include "../scene/primitive_rows.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;
namespace gfx = caliper_gfx;

bool write_file(const fs::path& p, const void* data, std::size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(n));
    return bool(f);
}

std::vector<std::uint8_t> read(gfx::Backend& g, gfx::Texture t, std::uint32_t size, std::string* why) {
    std::vector<std::uint8_t> out;
    if (!t) return out;
    const gfx::Readback rb = g.read_texture(t, 0, 0, size, size);
    if (!rb || !g.wait(g.last_use(t)) || !g.take_readback(rb, &out)) *why = "a readback failed: " + g.last_error();
    return out;
}

// The answer to a pick that the view asked, once its readbacks are in: the pick pass's and the
// pieces' window of the ID and depth targets.
caliper_scene::PickPass::Result take_pick(gfx::Backend& g, caliper_live_views::SceneView& v, caliper_scene::Hit* hit) {
    caliper_scene::PickPass::Result r = caliper_scene::PickPass::Result::Pending;
    for (int tries = 0; tries < 10000 && r == caliper_scene::PickPass::Result::Pending; ++tries) {
        @autoreleasepool {
            g.wait(v.pick_pass()->last_use());
            if (v.id_target()) g.wait(g.last_use(v.id_target()));
            if (v.depth_target()) g.wait(g.last_use(v.depth_target()));
            g.poll();
            r = v.poll_pick(hit);
        }
    }
    return r;
}

// One fixture: draw, read the targets of the first frame, pick, write.
bool one(gfx::Backend& g, const primitive_fixtures::Fixture& f, const fs::path& dir, std::string* line) {
    const std::uint32_t S = primitive_fixtures::kSize;
    caliper_live_views::SceneView view(g, caliper_gfx::shaders::kMetalPrograms);
    view.set_shape_draw(caliper_live_views::ShapeDraw::pieces);
    std::string why;
    if (!view.init(&why)) return (*line = f.name + ": init: " + why, false);
    primitive_rows::Row row;
    row.fixture = f;
    if (!row.draw(view)) return (*line = row.error, false);
    nlohmann::json facts = scene_parity::frame_json(view);
    std::vector<std::uint8_t> color, ids;
    @autoreleasepool {
        color = read(g, view.color(), S, &why);
        ids = read(g, view.id_target(), S, &why);
    }
    if (color.size() != std::size_t(S) * S * 4) return (*line = f.name + ": the color readback: " + why, false);
    if (ids.size() != std::size_t(S) * S * 8) ids.assign(std::size_t(S) * S * 8, 0);   // no piece drew
    const std::vector<std::array<int, 2>> points = scene_parity::pick_points(scene_parity::pair_ids(ids, std::size_t(S) * S), int(S));
    nlohmann::json picks = nlohmann::json::array();
    for (const auto& p : points) {
        if (!view.request_pick(float(p[0]) + 0.5f, float(p[1]) + 0.5f, &why))
            return (*line = f.name + ": pick: " + why, false);
        caliper_scene::Hit hit;
        const caliper_scene::PickPass::Result r = take_pick(g, view, &hit);
        nlohmann::json j = scene_parity::hit_json(view, r, hit);
        j["at"] = {p[0], p[1]};
        picks.push_back(j);
    }
    facts["picks"] = picks;
    const fs::path base = dir / f.name;
    const std::string text = facts.dump(1);
    if (!write_file(base.string() + ".color", color.data(), color.size()) ||
        !write_file(base.string() + ".id", ids.data(), ids.size()) ||
        !write_file(base.string() + ".json", text.data(), text.size()) ||
        !stbi_write_png((base.string() + ".png").c_str(), int(S), int(S), 4, color.data(), int(S) * 4))
        return (*line = f.name + ": cannot write into " + dir.string(), false);
    *line = f.name + ": frame " + std::to_string(facts["frame"].get<std::uint64_t>()) + ", " +
            std::to_string(facts["pieces"].get<std::uint32_t>()) + " shapes by pieces, " +
            std::to_string(facts["builtin"].get<std::uint32_t>()) + " by the built-in " + facts["fallbacks"].dump() + ", " +
            std::to_string(points.size()) + " picks";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: caliper_scene_parity_app <out>\n");
        return 2;
    }
    const fs::path dir = fs::path(argv[1]) / "scene";
    std::error_code ec;
    fs::create_directories(dir, ec);
    for (const auto& e : fs::directory_iterator(dir, ec)) fs::remove(e.path(), ec);   // no stale picture
    auto skip = [&](const std::string& why) {
        write_file(dir / "skipped.txt", why.data(), why.size());
        std::printf("skipped: %s\n", why.c_str());
        return 77;
    };
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return skip("no Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::unique_ptr<gfx::Backend> g = gfx::make_metal_backend((__bridge void*)device, (__bridge void*)queue);
    if (!g) return skip("no Metal device (the graphics layer made no Metal backend)");
    std::string report;
    int failed = 0;
    for (const primitive_fixtures::Fixture& f : scene_parity::fixtures()) {
        std::string line;
        const bool ok = one(*g, f, dir, &line);
        failed += ok ? 0 : 1;
        report += (ok ? "drawn " : "failed ") + line + "\n";
        std::printf("%s %s\n", ok ? "drawn" : "failed", line.c_str());
    }
    write_file(dir / "report.txt", report.data(), report.size());
    return failed == 0 ? 0 : 1;
}
