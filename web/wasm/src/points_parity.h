// points_parity.h: the points view drawn by points@1 against its initial payload draw in the page, for the
// browser test (web/test/points3d.spec.ts).
//
// Each case is one set of points through the points view's scene (src/host/live/live_points.h) on
// the module's WebGPU backend, drawn twice at 320 x 240: once by the library piece points@1, the
// view's default, and once by the initial payload points program. Both targets come back through the
// layer's asynchronous readback a frame or two later, and the module holds the piece's picture to
// the tolerances of the native match tests: a pixel whose 3 x 3 neighbourhood in the initial payload
// picture holds one colour has that colour, and any other pixel lies within 1 level of a initial payload
// colour of its neighbourhood. The cases: a 2-D lattice of class labels with labels outside 0 to
// 255 and NaN, 3-D values through viridis with NaN values and the depth cues, a 3-D selection over
// the dimmed rest, and 20,000 flat 2-D points.
#pragma once

#include "gfx/gfx.h"

#include <string>

namespace caliper_web {

// Starts the comparison on the backend. False, with why, when it could not start.
bool points_parity_start(caliper_gfx::Backend& b, const caliper_gfx::ProgramDesc* programs, std::string* why);

// Called every frame while it runs. True once every readback is in, with the report: "ok" or "fail",
// then one clause for each case with its counts.
bool points_parity_poll(caliper_gfx::Backend& b, std::string* report);

bool points_parity_running();
// Releases the scenes; before the backend goes.
void points_parity_stop();

}  // namespace caliper_web
