#pragma once
// Annotations, legends and bound texts in the spec runtime
// (design/instrument-interface.md, Annotations and Markers, Legends and
// Scale Bars, Bound Text).
//
// draw_plot_annotations() draws a 2-D panel's thresholds, bands, markers and
// labels in data units inside its ImPlot plot; draw_panel_overlays() draws
// what lies over a whole panel after it: its legend and the placed labels
// (label_layout.h). A scene panel's thresholds and bands go to its
// SpecScene as overlay batches (spec_entities.h). Defined in
// spec_annotations.cpp.
#include "spec_runtime/spec_instruments.h"

namespace caliper_host {

// Inside the current ImPlot plot of a 2-D panel, between BeginPlot and
// EndPlot.
void draw_plot_annotations(const caliper_builder::spec::Panel& panel, const SpecInstrumentHost& host);

// Over the panel's content rectangle, after the panel draws.
void draw_panel_overlays(const caliper_builder::spec::Panel& panel, const ImRect& content, const SpecLook& look,
                         const SpecInstrumentHost& host);

}  // namespace caliper_host
