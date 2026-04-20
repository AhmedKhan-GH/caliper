#pragma once

struct GLFWwindow;

// Registered applets that the landing screen can launch.
enum class AppletKind {
    None,
    ECGExplorer,
};

// ============================================================================
// Caliper intro / landing screen.
//
//   OpenGL visualizer (radial ECG-ribbon field, procedural grid, FBO bloom
//   post-processing) rendered behind an ImGui applet selector overlay.
//
// Frame lifecycle driven by main.cpp:
//   initialize()      one-time GL setup (shaders, FBOs, geometry)
//   update(win)       per-frame simulation, input
//   render_3d(w,h)    3D scene + post-processing (renders into default FB)
//   draw_ui(w,h)      ImGui overlay; sets launch_requested_ on confirm
//   cleanup()         releases all owned GL resources
// ============================================================================

class IntroScreen {
public:
    bool initialize();
    void update(GLFWwindow* window);
    void render_3d(int fb_w, int fb_h);
    void draw_ui(int win_w, int win_h);
    void cleanup();

    bool       should_launch()     const { return launch_requested_; }
    void       reset_launch_flag()       { launch_requested_ = false; }
    AppletKind selected_applet()   const;

private:
    struct State;
    State* s_ = nullptr;

    bool launch_requested_ = false;
};
