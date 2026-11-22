#pragma once
#include <memory>
#include <string>

#include "renderer/host_renderer.h"

// Core lifecycle (libcaliper / Compass R4, L1). The ordered init/teardown that
// used to sit inline in CaliperApp::initialize()/cleanup() (src/main.cpp) —
// renderer backend selection and the applet-canvas ImGui/ImPlot/ImPlot3D
// context lifecycle — lives here so every embedder of libcaliper gets the same
// policy in the same order. The host still owns the GLFWwindow, the event loop,
// and its chrome (intro screen, docking shell, jobs tray, styling); it calls
// these at the exact points it used to run the inline code.
//
// ORDERING IS LOAD-BEARING (docs/applets/rendering.md): the ImGui contexts
// must exist before HostRenderer::init() wires the ImGui backend; the contexts
// are destroyed AFTER HostRenderer::shutdown(). Do not reorder — every
// documented reordering crashes (MTLTexture UAF / ImGui-backend teardown).
namespace caliper_host {

// The platform's HostRenderer, not yet started: Metal on Apple, Vulkan on
// Win32. There is no fallback renderer until the WebGPU fallback through Dawn
// lands, so this returns nullptr with *why set to the refusal where there is
// none: CALIPER_RENDERER=gl, which asks for the removed OpenGL renderer, or a
// platform with neither Metal nor Vulkan. Any other value of CALIPER_RENDERER
// leaves the platform's renderer. The caller starts it (init() after creating
// the window, or canvas_init()) and refuses with renderer_start_refusal() if
// that fails.
std::unique_ptr<HostRenderer> core_select_renderer(std::string* why);

// The refusal when the renderer `name` could not start for the reason `why`
// (HostRenderer::start_error()): it names the renderer and the reason, and
// says that there is no fallback yet. The app prints it after "caliper: ",
// caliper shot after "caliper: shot cannot draw offscreen: ", and the
// embedding core sets it as the last error after "attach_canvas: ".
std::string renderer_start_refusal(const char* name, const std::string& why);

// Creates the applet-canvas ImGui/ImPlot/ImPlot3D contexts and sets the base
// config flags (keyboard nav + docking). MUST run before renderer->init().
// Host chrome styling (StyleColorsDark/style_ui) is applied by the host AFTER
// this returns, so the ordering relative to the renderer is preserved.
void core_create_ui_context();

// Destroys the ImPlot3D/ImPlot/ImGui contexts in reverse creation order. Runs
// during host cleanup AFTER renderer->shutdown().
void core_destroy_ui_context();

}  // namespace caliper_host
