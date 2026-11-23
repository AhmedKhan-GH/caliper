// scene_selftest.cpp: the scene rows in the page (scene_selftest.h).
//
// The report holds the fixture name, piece count, nonempty pixels, notes, and device level.
#include "scene_selftest.h"

#include "../../../tests/scene/primitive_rows.h"
#include "renderer/webgpu_backend.h"

#include <memory>
#include <cstring>

namespace caliper_web {

namespace {

struct Run {
    std::unique_ptr<caliper_live_views::SceneView> view;
    primitive_rows::Row row;
    caliper_gfx::Readback readback;
    std::vector<uint8_t> pixels;
    bool running = false;
};
Run& run() {
    static Run r;
    return r;
}

}  // namespace

std::string scene_selftest_names() {
    std::string out;
    for (const primitive_fixtures::Fixture& f : primitive_fixtures::all()) out += (out.empty() ? "" : ",") + f.name;
    return out;
}

bool scene_selftest_start(caliper_gfx::Backend& b, const caliper_gfx::ProgramDesc* programs, const std::string& name,
                          std::string* why) {
    Run& r = run();
    if (r.running) {
        *why = "a row is running";
        return false;
    }
    scene_selftest_stop();
    bool found = false;
    for (primitive_fixtures::Fixture& f : primitive_fixtures::all())
        if (f.name == name) {
            r.row = primitive_rows::Row{};
            r.row.fixture = std::move(f);
            found = true;
        }
    if (!found) {
        *why = "no fixture '" + name + "' (" + scene_selftest_names() + ")";
        return false;
    }
    r.view = std::make_unique<caliper_live_views::SceneView>(b, programs);
    if (!r.view->init(why)) return false;
    if (!r.row.draw(*r.view)) {
        *why = r.row.error;
        return false;
    }
    r.readback = b.read_texture(r.view->color(), 0, 0, r.view->width(), r.view->height());
    if (!r.readback) {
        *why = "no readback: " + b.last_error();
        return false;
    }
    r.running = true;
    return true;
}

bool scene_selftest_poll(caliper_gfx::Backend& b, std::string* report) {
    Run& r = run();
    if (!r.running) return false;
    if (!b.take_readback(r.readback, &r.pixels)) return false;   // not yet
    r.running = false;
    const std::string level = static_cast<caliper_gfx::WebGpuBackend&>(b).compatibility() ? "compatibility" : "core";
    if (r.pixels.size() != size_t(r.view->width()) * r.view->height() * 4) {
        *report = "fail " + r.row.fixture.name + ": the readback has " + std::to_string(r.pixels.size()) + " bytes";
        return true;
    }
    const auto count = r.view->shape_counts();
    size_t colored = 0;
    for (size_t at = 0; at + 4 <= r.pixels.size(); at += 4) {
        uint32_t pixel = 0;
        std::memcpy(&pixel, r.pixels.data() + at, 4);
        colored += pixel != primitive_fixtures::kClear;
    }
    const bool ok = count.pieces > 0 && colored > 0 && r.view->notes().empty();
    *report = std::string(ok ? "ok " : "fail ") + r.row.fixture.name + ": pieces=" + std::to_string(count.pieces) +
              " pixels=" + std::to_string(colored) + " notes=" + std::to_string(r.view->notes().size()) + " level=" + level;
    if (!r.view->notes().empty()) *report += " note=" + r.view->notes().back();
    return true;
}

bool scene_selftest_running() { return run().running; }
const std::vector<uint8_t>& scene_selftest_pixels() { return run().pixels; }

void scene_selftest_stop() {
    Run& r = run();
    r.running = false;
    r.view.reset();
}

}  // namespace caliper_web
