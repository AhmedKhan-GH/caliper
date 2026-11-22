// The time transport in the spec runtime (spec_transport.h). Not built yet:
// the transport kind stays refused by the validator
// (spec.unsupported_control), and the cursor is always live.
#include "spec_runtime/spec_transport.h"

namespace caliper_host {

bool draw_transport_control(const caliper_builder::spec::Control& control, const ImRect& rect, const SpecLook& look,
                            SpecInstrumentHost& host) {
    (void)control;
    (void)host;
    ImGui::GetWindowDrawList()->AddText(rect.Min, look.faint, "transport is not built yet");
    return false;
}

bool handle_transport_keys(const caliper_builder::spec::Spec& spec, SpecInstrumentHost& host) {
    (void)spec;
    (void)host;
    return false;
}

void draw_cursor_line(const caliper_builder::spec::Panel& panel, const SpecInstrumentHost& host) {
    (void)panel;
    (void)host;
}

void tick_transport(SpecInstrumentHost& host, double seconds) {
    (void)host;
    (void)seconds;
}

}  // namespace caliper_host
