#pragma once
// The instruments of the spec runtime (design/instruments.md): what the
// runtime needs from the app to draw spec version 1, and the seam where the
// runtime's layout hands a control leaf to the code that draws it.
//
// The runtime lays a control out as a leaf of the layout, like a panel, and
// calls draw_spec_control() for it inside the leaf's rectangle. A control
// bound to param:, action: or run:pause acts through the app's RunControl.
// It pushes one `control` event when the person finishes an edit: at the end
// of a slider's drag or typed entry, and at a button's click. A scene panel
// pushes its picks and selections as `pick` and `select` events with its
// panel id. Both reach the session's EventRing through SpecInstrumentHost.
//
// Threads: everything here runs on the frame thread, as the runtime does.
//
// The spec runtime and its layout call these functions, and spec_controls.cpp
// defines them.
#include "spec_runtime/spec_runtime.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/spec/instrument.h>

#include <imgui.h>
#include <imgui_internal.h>   // ImRect

#include <string>

namespace caliper_host {

// What the app gives the runtime for instruments. Either pointer may be null:
// without events nothing is pushed, and without a run control a control
// bound to the script shows that no script is attached.
struct SpecInstrumentHost {
    caliper_builder::EventSink* events = nullptr;      // the session's EventRing
    caliper_builder::RunControl* run = nullptr;        // the app's run control
};

// The height that a control of this kind takes in a vertical split: one row
// of the look's type with its padding. The layout gives a control leaf this
// height and the panels the rest.
float spec_control_height(const caliper_builder::spec::Control& control, const SpecLook& look);

// The label of a button. A button bound to an action shows its title and its
// bind string, "Save (action:save)", so that a person sees what a click runs,
// then "· queued" or "· running" while the action waits or runs. A button
// bound to run:pause reads "Pause", or "Resume" while the run is paused.
std::string spec_button_label(const caliper_builder::spec::Control& control, const caliper_builder::RunState& state);

// Draws one control into `rect` of the current window: its title and its
// widget, with the value it reads from the run control (param:), the state of
// the run (run:pause) or the value it holds (value). Returns true when the
// person changed it in this frame; the change has then been sent. The control
// event goes to the ring once, in the frame where the edit ends.
bool draw_spec_control(const caliper_builder::spec::Control& control, const ImRect& rect, const SpecLook& look,
                       SpecInstrumentHost& host);

}  // namespace caliper_host
