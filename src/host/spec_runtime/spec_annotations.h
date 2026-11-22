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

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <vector>

namespace caliper_host {

// Inside the current ImPlot plot of a 2-D panel, between BeginPlot and
// EndPlot.
void draw_plot_annotations(const caliper_builder::spec::Panel& panel, const SpecInstrumentHost& host);

// Over the panel's content rectangle, after the panel draws.
void draw_panel_overlays(const caliper_builder::spec::Panel& panel, const ImRect& content, const SpecLook& look,
                         const SpecInstrumentHost& host);

// The annotations of a scene panel that can draw now, in spec order: a mark
// whose point, item or value has no value yet (a selection before the first
// pick) is left out. The runtime maps them through the scene's data box
// (apply_annotation_marks() in spec_prepare.h).
std::vector<SceneAnnotationMark> scene_annotation_marks(const caliper_builder::spec::Panel& panel,
                                                        const SpecInstrumentHost& host);

// The value that a reference reads now: the session's value, else its
// default, else null.
nlohmann::json spec_reference_value(const caliper_builder::spec::Reference& ref, const SpecInstrumentHost& host);
// The value of each field of a bound text, by name.
std::map<std::string, nlohmann::json> spec_bound_text_values(const caliper_builder::spec::BoundText& text,
                                                             const SpecInstrumentHost& host);
// A bound text formatted with the values of its fields and the `extra`
// values (such as a text panel's `value`); empty when the format does not
// parse.
std::string spec_bound_text_string(const caliper_builder::spec::BoundText& text, const SpecInstrumentHost& host,
                                   const std::map<std::string, nlohmann::json>& extra = {});

}  // namespace caliper_host
