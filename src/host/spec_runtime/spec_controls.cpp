// The controls of a spec (spec_instruments.h, design/instruments.md, Control
// Table).
//
// Stubs of milestone M1's interface commit: a control takes one row and draws
// nothing until the track "controls" fills this file.
#include "spec_runtime/spec_instruments.h"

namespace caliper_host {

float spec_control_height(const caliper_builder::spec::Control& control, const SpecLook& look) {
    (void)control;
    return look.ui_size * look.text_scale * 2.0f;   // stub: track controls
}

bool draw_spec_control(const caliper_builder::spec::Control& control, const ImRect& rect, const SpecLook& look,
                       SpecInstrumentHost& host) {
    (void)control;
    (void)rect;
    (void)look;
    (void)host;
    return false;   // stub: track controls
}

}  // namespace caliper_host
