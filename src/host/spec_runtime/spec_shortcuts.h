#pragma once
// The keyboard shortcuts of an instrument (design/instrument-interface.md,
// Keyboard Shortcuts): the keys of the choices of segmented groups.
//
// The runtime calls handle_instrument_keys() once a frame, before it draws
// the panels. The shortcuts act only while the window has focus and no text
// field holds the keyboard, and the app's own shortcuts always win. The
// transport's keys are in spec_transport.h. Defined in spec_shortcuts.cpp.
#include "spec_runtime/spec_instruments.h"

namespace caliper_host {

// Applies the shortcut keys pressed in this frame. Returns true when one
// changed a control; the change has then been sent.
bool handle_instrument_keys(const caliper_builder::spec::Spec& spec, SpecInstrumentHost& host);

}  // namespace caliper_host
