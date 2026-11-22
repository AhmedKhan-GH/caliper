// The validator's checks of the options of the control kinds range, picker,
// text and segmented, their choices and their shortcuts
// (design/instrument-interface.md). Until the feature is built, the check
// reports each use of its keys with spec.not_built (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void validate_controls(const AdditionsCheck& c) {
    // The kinds themselves stay spec.unsupported_control until
    // control_kind_built() names them (instrument.cpp); this covers the new
    // options on the kinds that are built.
    for (std::size_t i = 0; i < c.spec.controls.size(); ++i) {
        const ControlOptions& o = c.spec.controls[i].options;
        const std::string path = index_path("control", i) + ".options";
        if (!o.value_other.is_null()) not_built(c.sink, "a value that is not a number", path + ".value");
        if (!o.choices.empty()) not_built(c.sink, "the option choices", path + ".choices");
        if (o.choices_from) not_built(c.sink, "the option choices_from", path + ".choices_from");
        if (o.multiple) not_built(c.sink, "the option multiple", path + ".multiple");
        if (o.placeholder) not_built(c.sink, "the option placeholder", path + ".placeholder");
        if (o.max_length) not_built(c.sink, "the option max_length", path + ".max_length");
        if (o.min_span) not_built(c.sink, "the option min_span", path + ".min_span");
        if (o.follow) not_built(c.sink, "the option follow", path + ".follow");
    }
}

}  // namespace caliper_builder::spec::detail
