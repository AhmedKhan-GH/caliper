// scene_selftest.h: the scene rows in the page, for the browser tests (web/test/scenes.spec.ts).
//
// Each fixture draws through library pieces. The asynchronous readback checks that the
// scene has pixels, pieces, and no error notes, or an explicit refusal for each unsupported piece.
// App/browser parity tests compare the targets.
#pragma once

#include "gfx/gfx.h"

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_web {

// Starts the row of fixture `name` on the backend. False, with why, when it could not start.
bool scene_selftest_start(caliper_gfx::Backend& b, const caliper_gfx::ProgramDesc* programs, const std::string& name,
                          std::string* why);

// Called every frame while a row runs. True once its readback is in, with the report: "ok",
// "refused", or "fail", the fixture name, key=value facts, and JSON error details.
bool scene_selftest_poll(caliper_gfx::Backend& b, std::string* report);

bool scene_selftest_running();
// The row's picture, tightly packed RGBA8, top row first.
const std::vector<uint8_t>& scene_selftest_pixels();
// Releases the row's scene view; before the backend goes.
void scene_selftest_stop();
// The fixtures' names, comma-separated.
std::string scene_selftest_names();

}  // namespace caliper_web
