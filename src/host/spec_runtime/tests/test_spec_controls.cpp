// Tests of the control widgets of the spec runtime (spec_instruments.h,
// design/instruments.md, Control Layout). The M1 track "controls" owns this
// file; until it fills spec_controls.cpp, a control takes one row and draws
// nothing.
#include "spec_runtime/spec_instruments.h"

#include <doctest/doctest.h>

TEST_CASE("spec controls: a control leaf takes a row of the look's height") {
    caliper_builder::spec::Control slider;
    slider.id = "lr";
    slider.bind = "param:lr";
    const caliper_host::SpecLook look;
    CHECK(caliper_host::spec_control_height(slider, look) > 0.0f);
}
