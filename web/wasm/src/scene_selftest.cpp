// scene_selftest.cpp: the scene rows in the page (scene_selftest.h).
//
// The report is "ok" or "fail", the fixture's name and a colon, then the comparison's sentence (how
// many pixels differ from the CPU reference and why), then: draws, the draws the view compiled;
// clipped, triangles the reference would have had to clip (none in a fixture); notes, what the view
// noted; problems, what the reference could not rasterise; and level, the device's feature level.
#include "scene_selftest.h"

#include "../../../tests/scene/primitive_rows.h"
#include "renderer/webgpu_backend.h"

#include <memory>

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
    const primitive_rows::Row::Result res = r.row.compare(*r.view, r.pixels);
    size_t uncovered = 0;
    for (const size_t c : res.covered) uncovered += c == 0 ? 1 : 0;
    const bool ok = res.c.failed == 0 && res.clipped == 0 && res.problems.empty() && r.view->notes().empty() &&
                    uncovered == 0;
    *report = std::string(ok ? "ok " : "fail ") + res.sentence + "; draws=" + std::to_string(res.draws) +
              " uncovered=" + std::to_string(uncovered) + " clipped=" + std::to_string(res.clipped) +
              " notes=" + std::to_string(r.view->notes().size()) + " problems=" + std::to_string(res.problems.size()) +
              " level=" + level;
    if (!res.c.first.empty()) *report += " first=" + res.c.first;
    if (!r.view->notes().empty()) *report += " note=" + r.view->notes().back();
    if (!res.problems.empty()) *report += " problem=" + res.problems.front();
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
