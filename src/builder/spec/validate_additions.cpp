// The shared helper of the validator's entry points for the keys of
// design/instrument-interface.md (validate_additions.h).
#include "validate_additions.h"

namespace caliper_builder::spec::detail {

void not_built(Sink& sink, const std::string& what, const std::string& path) {
    sink.error("spec.not_built", "Caliper reads " + what + " but does not build it yet.", path);
}

}  // namespace caliper_builder::spec::detail
