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
// panel id. Both reach the session's EventRing through SpecLiveHost.
//
// Threads: everything here runs on the frame thread, as the runtime does.
//
// The spec runtime and its layout call these functions, and spec_controls.cpp
// defines them.
#include "spec_runtime/spec_runtime.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/control.h>

#include <imgui.h>
#include <imgui_internal.h>   // ImRect

#include <string>

namespace caliper_host {

// What the app gives the runtime for instruments. Either pointer may be null:
// without events nothing is pushed, and without a run control a control
// bound to the script shows that no script is attached.
struct SpecLiveHost {
    caliper_builder::EventSink* events = nullptr;      // the session's EventRing
    caliper_builder::RunControl* run = nullptr;        // the app's run control
    // design/instrument-interface.md: the session's values of controls,
    // state, picks and the cursor, its kept steps, and the spec shown. Null
    // when the app has no session; references then read their defaults.
    caliper_builder::ValueStore* values = nullptr;
    caliper_builder::StepHistory* history = nullptr;
    const caliper_builder::spec::Spec* spec = nullptr;
};

// Records a pick of a scene panel in the session's values (the panel's last
// pick and its `select` write), after the runtime pushed its `pick` event.
// Defined in spec_controls.cpp.
void record_spec_pick(const std::string& panel, const caliper_builder::Event& pick, SpecLiveHost& host);

// The height that a control of this kind takes in a vertical split: one row
// of the look's type with its padding. The layout gives a control leaf this
// height and the panels the rest.
float spec_control_height(const caliper_builder::spec::Control& control, const SpecLook& look);

// An action button shows its title, or its id if no title exists, then
// "· queued" or "· running" while the action waits or runs.
// The action target appears only in the tooltip.
// A button bound to run:pause reads "Pause", or "Resume" when the run is paused.
std::string spec_button_label(const caliper_builder::spec::Control& control, const caliper_builder::RunState& state);

// Draws one control into `rect` of the current window: its title and its
// widget, with the value it reads from the run control (param:), the state of
// the run (run:pause) or the value it holds (value). Returns true when the
// person changed it in this frame; the change has then been sent. The control
// event goes to the ring once, in the frame where the edit ends.
bool draw_spec_control(const caliper_builder::spec::Control& control, const ImRect& rect, const SpecLook& look,
                       SpecLiveHost& host);

}  // namespace caliper_host
