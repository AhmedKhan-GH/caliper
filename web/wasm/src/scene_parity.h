// scene_parity.h: the browser's side of the scene comparison between the app and the browser module
// (tests/draw_parity/scene_parity.h), for web/test/draw_parity.spec.ts.
//
// The page names a fixture of tests/scene/primitive_fixtures.h, optionally the switch of the shapes
// ("pieces" or "builtin"; without it the view keeps the module's switch, caliperConfig.shapes) and
// the points to pick at, which the app's side chose. The module applies the fixture to a scene view
// on its WebGPU backend exactly as the scene rows apply it (tests/scene/primitive_rows.h), draws it
// once at 256 x 256 and keeps the facts of that frame: what
// drew each shape, with the reason of each fallback, and the program of each pick id. It then reads
// the color target and the pieces' ID target back with the layer's asynchronous readback, picks at
// each point, one pick at a time over the next frames, and reports.
#pragma once

#include "gfx/gfx.h"

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_web {

struct SceneParityResult {
    std::string facts;                  // JSON: scene_parity::frame_json and "picks", each with "at"
    std::vector<std::uint8_t> color;    // RGBA8, top row first
    std::vector<std::uint8_t> ids;      // RG32Uint (pick id, item), top row first; zeros without a piece
};

// Starts the comparison's draw of one fixture from the config JSON {"fixture": name, "shapes":
// "pieces" or "builtin" (optional), "points": [[x, y], ...] (optional)}. False, with why, when it
// could not start.
bool scene_parity_start(caliper_gfx::Backend& b, const caliper_gfx::ProgramDesc* programs, const std::string& config,
                        std::string* why);
// Called every frame while it runs. True once the readbacks and the picks are in, with the report:
// "ok" or "fail", the fixture's name, then the frame's counts.
bool scene_parity_poll(caliper_gfx::Backend& b, std::string* report);
bool scene_parity_running();
const SceneParityResult& scene_parity_result();
// Releases the scene view; before the backend goes.
void scene_parity_stop();

}  // namespace caliper_web
