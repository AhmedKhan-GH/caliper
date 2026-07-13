// Pure logic behind the export.v1 E2 affordances (frame budget, wall-clock
// pacing, on-disk path assembly). Torch-free / UI-free, so it rides the fast
// caliper_tests suite (mirrors pulse_ring.h / instance_field.h).
#include <doctest/doctest.h>

#include "export_affordance.h"

using namespace caliper::exportui;

TEST_CASE("export frame budget: 10 s @ 30 fps is 300 frames") {
    CHECK(frame_budget(10.0, 30.0) == 300u);
    CHECK(frame_budget(0.5, 30.0) == 15u);
    CHECK(frame_budget(1.0, 60.0) == 60u);
}

TEST_CASE("export frame budget: rounds to nearest, clamps to >=1 for any positive duration") {
    CHECK(frame_budget(0.99, 30.0) == 30u);   // 29.7 -> 30
    CHECK(frame_budget(0.001, 30.0) == 1u);   // 0.03 rounds to 0 -> clamp to 1
}

TEST_CASE("export frame budget: non-positive duration or fps yields zero") {
    CHECK(frame_budget(0.0, 30.0) == 0u);
    CHECK(frame_budget(10.0, 0.0) == 0u);
    CHECK(frame_budget(-1.0, 30.0) == 0u);
}

TEST_CASE("export pacing: the first capture is always due") {
    CHECK(capture_due(0, 0, 30.0, /*first=*/true) == true);
    // Even with no elapsed time, first forces a start.
    CHECK(capture_due(1000, 1000, 30.0, true) == true);
}

TEST_CASE("export pacing: a capture is due only after one 1000/fps ms interval") {
    // 30 fps -> 33.33 ms interval.
    CHECK(capture_due(1000, 970, 30.0, false) == false);   // 30 ms < 33.33
    CHECK(capture_due(1000, 966, 30.0, false) == true);    // 34 ms >= 33.33
    CHECK(capture_due(1000, 1000, 30.0, false) == false);  // 0 ms
    CHECK(capture_due(1033, 1000, 30.0, false) == false);  // 33 ms < 33.33
    CHECK(capture_due(1034, 1000, 30.0, false) == true);   // 34 ms
}

TEST_CASE("export pacing: fps<=0 is never due (except the forced first)") {
    CHECK(capture_due(1000, 0, 0.0, false) == false);
}

TEST_CASE("export paths: figure PNG lands under <root>/exports/") {
    CHECK(figure_png_path("/data", "twin_scope", "20260712T010203") ==
          "/data/exports/twin_scope_20260712T010203.png");
    CHECK(record_dir_path("/data", "twin_scope", "20260712T010203") ==
          "/data/exports/twin_scope_20260712T010203");
}

TEST_CASE("export paths: an empty root degrades to a writable relative path") {
    CHECK(exports_dir("") == "./exports");
    CHECK(figure_png_path("", "mesh", "T") == "./exports/mesh_T.png");
}
