// The keyboard shortcuts of an instrument (spec_shortcuts.h). Not built
// yet: no key acts.
#include "spec_runtime/spec_shortcuts.h"

namespace caliper_host {

bool handle_instrument_keys(const caliper_builder::spec::Spec& spec, SpecInstrumentHost& host) {
    (void)spec;
    (void)host;
    return false;
}

}  // namespace caliper_host
