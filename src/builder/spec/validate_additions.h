#pragma once
// Internal to caliper_builder_spec: the validator's entry points for the keys
// of design/instrument-interface.md.
//
// validate.cpp calls each check once for a spec of version 1, after its own
// checks of the existing keys and before the extension tables, and calls the
// two shape functions from its shape walk for the new transform ops. Each
// function is defined in the file named beside it, which owns the codes of
// its feature. Until a feature is built, its check reports each use of its
// keys with not_built(), so that a spec that uses an unbuilt key never
// reaches code that ignores it, and every spec without the new keys stays
// valid.
#include <string>

#include <caliper_builder/spec/validate.h>

#include "internal.h"

namespace caliper_builder::spec::detail {

// What a check reads: the spec, the resolver of catalog sources, what the
// validator learned of each source so far (after its transforms), and the
// sink of the report.
struct AdditionsCheck {
    const Spec& spec;
    const SourceResolver& resolver;
    const ValidationReport& report;
    Sink& sink;
};

// Reports spec.not_built at `path`: "Caliper reads <what> but does not
// build it yet."
void not_built(Sink& sink, const std::string& what, const std::string& path);

// validate_states.cpp: the `state` array, the option `select`, and the
// controls bound to "state:<id>".
void validate_states(const AdditionsCheck& c);
// validate_references.cpp: every reference at a value place, the places
// that take only literals, and the cycle graph (spec.reference_cycle).
void validate_references(const AdditionsCheck& c);
// validate_controls.cpp: the options of the control kinds range, picker,
// text and segmented, their choices and shortcuts.
void validate_controls(const AdditionsCheck& c);
// validate_history.cpp: a source's `history`, the `window` step and the
// transport control.
void validate_history(const AdditionsCheck& c);
// validate_operators.cpp: the `surface` and `path` steps, `params` of
// programs and stages, and the `tensor` output.
void validate_operators(const AdditionsCheck& c);
// validate_entities.cpp: the `entity` array of a scene panel, and the scene
// options `box` and `views`.
void validate_entities(const AdditionsCheck& c);
// validate_annotations.cpp: the `annotation` array, the option `legend`, and
// `scale_bar` on image and points panels.
void validate_annotations(const AdditionsCheck& c);
// validate_bound_text.cpp: the options `format` and `fields` of a text panel
// and of a readout.
void validate_bound_text(const AdditionsCheck& c);
// validate_layout.cpp: `shares` of a split and `min` of a leaf.
void validate_layout(const AdditionsCheck& c);

// The shape walk of the new ops: each applies one step to `info` (dtype and
// shape, as validate.cpp's apply() does for the shipped steps) and returns
// false when the step does not apply to the shape. A step whose result is
// not known resets info.shape.
bool apply_window_shape(const Transform& t, SourceInfo& info, const std::string& path, Sink& sink);    // validate_history.cpp
bool apply_operator_shape(const Transform& t, SourceInfo& info, const std::string& path, Sink& sink);  // validate_operators.cpp

}  // namespace caliper_builder::spec::detail
