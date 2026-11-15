// web_input.h: the ImGui platform side of the browser's views: the canvas
// size and the device pixel ratio, pointer, wheel, keyboard and focus input,
// and the mouse cursor. Every listener sits on this instance's own canvas
// (Module.canvas), with pointer capture for drags, so several instances on
// one page never see each other's input. Keys reach an instance only while
// its canvas has focus; a click gives it focus.
#pragma once

namespace caliper_web {

// Adds the listeners. Call once after the ImGui context exists.
void input_install();
// Removes them (the handle's destroy()).
void input_uninstall();

struct FrameSize {
    float css_w = 0.0f, css_h = 0.0f;   // the canvas's CSS box (ImGui's display size)
    int fb_w = 0, fb_h = 0;             // its backing store in device pixels
};

// Before ImGui::NewFrame: sizes the canvas's backing store to its CSS box
// times the device pixel ratio (so nothing is scaled by the browser and text
// is sharp), and sets io.DisplaySize, io.DisplayFramebufferScale and
// io.DeltaTime.
FrameSize input_new_frame();

// After ImGui::Render: shows ImGui's mouse cursor on the canvas.
void input_end_frame();

// Counters for the tests: input events handed to ImGui, and the ImGui name
// of the last key pressed ("" when none).
struct InputCounts {
    unsigned pointer = 0, buttons = 0, wheel = 0, keys = 0, chars = 0;
    const char* last_key = "";
};
InputCounts input_counts();

}  // namespace caliper_web
