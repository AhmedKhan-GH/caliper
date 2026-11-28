#pragma once
// The time transport in the spec runtime (design/instrument-interface.md,
// Transport Control and Latest at the Cursor).
//
// The transport control draws a step back, play or hold, a step forward, a
// live button, a speed menu and the step as text, and moves the cursor of
// the session's StepHistory. A series panel draws a vertical line at the
// cursor when it is not live. Defined in spec_transport.cpp.
#include "spec_runtime/spec_live_host.h"

namespace caliper_host {

// Draws a control of the kind `transport` into `rect`. Returns true when the
// person moved the cursor in this frame.
bool draw_transport_control(const caliper_builder::spec::Control& control, const ImRect& rect, const SpecLook& look,
                            SpecLiveHost& host);

// Applies the transport's keys pressed in this frame (Space, the left and
// right arrows, Home and End) when the spec has a transport with `keys`.
bool handle_transport_keys(const caliper_builder::spec::Spec& spec, SpecLiveHost& host);

// Draws the cursor's line inside the current ImPlot plot of a series panel,
// between BeginPlot and EndPlot, when the cursor is not live.
void draw_cursor_line(const caliper_builder::spec::Panel& panel, const SpecLiveHost& host);

// Advances a playing cursor by the frame's time; called once a frame.
void tick_transport(SpecLiveHost& host, double seconds);

}  // namespace caliper_host
