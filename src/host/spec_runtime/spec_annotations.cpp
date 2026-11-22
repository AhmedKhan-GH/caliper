// Annotations, legends and bound texts in the spec runtime
// (spec_annotations.h). Not built yet: the validator refuses them
// (spec.not_built), so nothing is drawn.
#include "spec_runtime/spec_annotations.h"

namespace caliper_host {

void draw_plot_annotations(const caliper_builder::spec::Panel& panel, const SpecInstrumentHost& host) {
    (void)panel;
    (void)host;
}

void draw_panel_overlays(const caliper_builder::spec::Panel& panel, const ImRect& content, const SpecLook& look,
                         const SpecInstrumentHost& host) {
    (void)panel;
    (void)content;
    (void)look;
    (void)host;
}

}  // namespace caliper_host
