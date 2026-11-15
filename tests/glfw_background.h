#pragma once
// glfwInit() for the test binaries that open hidden windows (caliper_gfx_tests,
// caliper_export_tests), without a Dock icon or a menu bar on macOS.
//
// GLFW's Cocoa init (third_party/glfw/src/cocoa_init.m) creates the shared
// NSApplication, runs it until it has finished launching, and then, unless the
// init hint GLFW_COCOA_MENUBAR is false, builds a menu bar and sets the
// activation policy to NSApplicationActivationPolicyRegular, which gives the
// test process a Dock icon while ctest runs it. With the hint false GLFW does
// neither, and the policy is set to Prohibited before glfwInit() and again
// after it (src/host/background_process.h), so it holds from before the first
// window to the end of the process. The windows are created with GLFW_VISIBLE
// false and are never shown or focused, so GLFW never orders them front or
// activates the process (cocoa_window.m does both only for a visible window).
#include <GLFW/glfw3.h>

#include "background_process.h"

namespace caliper_test {

inline bool glfw_init_in_background() {
    caliper_host::use_background_activation_policy();
    glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
    const bool ok = glfwInit() == GLFW_TRUE;
    caliper_host::use_background_activation_policy();
    return ok;
}

} // namespace caliper_test
