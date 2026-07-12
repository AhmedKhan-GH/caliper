#pragma once
#include <memory>

#include "renderer/host_renderer.h"

// Core lifecycle (libcaliper / Compass R4, L1). The ordered init/teardown that
// used to sit inline in CaliperApp::initialize()/cleanup() (src/main.cpp) —
// renderer backend selection and the applet-canvas ImGui/ImPlot/ImPlot3D
// context lifecycle — lives here so every embedder of libcaliper gets the same
// policy in the same order. The host still owns the GLFWwindow, the event loop,
// and its chrome (intro screen, docking shell, jobs tray, styling); it calls
// these at the exact points it used to run the inline code.
//
// ORDERING IS LOAD-BEARING (PLATFORM.md §5.4, l1-survey §6): the ImGui contexts
// must exist before HostRenderer::init() wires the ImGui backend; the contexts
// are destroyed AFTER HostRenderer::shutdown(). Do not reorder — every
// documented reordering crashes (MTLTexture UAF / ImGui-backend teardown).
namespace caliper_host {

// Selects the HostRenderer backend from CALIPER_RENDERER + the platform default
// (Metal on Apple, Vulkan on Win32, GL elsewhere); GL is the guaranteed
// fallback. The host still creates the window and calls renderer->init()
// itself, and still recreates with make_renderer("gl") if that init fails.
std::unique_ptr<HostRenderer> core_select_renderer();

// Creates the applet-canvas ImGui/ImPlot/ImPlot3D contexts and sets the base
// config flags (keyboard nav + docking). MUST run before renderer->init().
// Host chrome styling (StyleColorsDark/style_ui) is applied by the host AFTER
// this returns, so the ordering relative to the renderer is preserved.
void core_create_ui_context();

// Destroys the ImPlot3D/ImPlot/ImGui contexts in reverse creation order. Runs
// during host cleanup AFTER renderer->shutdown().
void core_destroy_ui_context();

}  // namespace caliper_host
