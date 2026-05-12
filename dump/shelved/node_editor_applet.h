#pragma once

// ============================================================================
// Caliper node-editor sandbox applet.
//
// Wraps thedmd/imgui-node-editor with a small graph of signal-processing
// flavored example nodes — a place to experiment with the API before wiring
// it into a real DSP pipeline.
//
// Frame lifecycle (driven by main.cpp):
//   initialize()            one-time editor-context creation
//   draw_ui(win_w, win_h)   full-window canvas + side panel
//   cleanup()               destroys the editor context
//
//   should_exit() / reset_exit_flag() — set when the "Back to Menu" button
//   is clicked, drives the page transition in CaliperApp.
// ============================================================================

class NodeEditorApplet {
public:
    bool initialize();
    void draw_ui(int win_w, int win_h);
    void cleanup();

    bool should_exit()      const { return exit_requested_; }
    void reset_exit_flag()        { exit_requested_ = false; }

private:
    struct State;
    State* s_ = nullptr;
    bool   exit_requested_ = false;
};
