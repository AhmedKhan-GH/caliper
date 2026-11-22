// scene_selftest.h: the scene rows in the page, for the browser tests (web/test/scenes.spec.ts).
//
// A fixture of tests/scene/primitive_fixtures.h (one scene per primitive) goes through a scene view on
// the module's WebGPU backend exactly as the graphics suite's Metal rows send it through one: each
// message encoded with the C++ codec, decoded again and applied, the view drawn at 256 x 256. The
// target comes back through the layer's asynchronous readback a frame or two later, and the module
// compares it with the CPU reference of tests/scene/draw_reference.h, rasterised from the parameter
// blocks the module itself compiled, under web geometry's parity rules.
#pragma once

#include "gfx/gfx.h"

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_web {

// Starts the row of fixture `name` on the backend. False, with why, when it could not start.
bool scene_selftest_start(caliper_gfx::Backend& b, const caliper_gfx::ProgramDesc* programs, const std::string& name,
                          std::string* why);

// Called every frame while a row runs. True once its readback is in, with the report: "ok" or
// "fail", the fixture's name, then the sentence of the comparison and key=value facts.
bool scene_selftest_poll(caliper_gfx::Backend& b, std::string* report);

bool scene_selftest_running();
// The row's picture, tightly packed RGBA8, top row first.
const std::vector<uint8_t>& scene_selftest_pixels();
// Releases the row's scene view; before the backend goes.
void scene_selftest_stop();
// The fixtures' names, comma-separated.
std::string scene_selftest_names();

}  // namespace caliper_web
