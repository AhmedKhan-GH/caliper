#pragma once
// ============================================================================
// Background processes on macOS. A process that draws without a visible
// window, such as `caliper shot` or a test binary that opens hidden GLFW
// windows, must not show a Dock icon or a menu bar, and must never take the
// focus from the user's own work.
//
// use_background_activation_policy() creates the shared NSApplication if it
// does not exist and sets its activation policy to
// NSApplicationActivationPolicyProhibited: the process has no Dock icon and no
// menu bar, and AppKit never activates it. Call it before the first window
// exists, and again after any library call that may change the policy:
// glfwInit() sets NSApplicationActivationPolicyRegular unless the init hint
// GLFW_COCOA_MENUBAR is false (third_party/glfw/src/cocoa_init.m), so a
// program that uses GLFW sets that hint as well and calls this function both
// before and after glfwInit().
//
// background_activation_policy_active() reports whether the policy holds, for
// the tests that check it.
//
// Off macOS a process has no Dock icon or menu bar to suppress: a hidden
// window has no taskbar button on Windows. Both functions do nothing there,
// and the second returns true.
// ============================================================================

namespace caliper_host {

#ifdef __APPLE__
void use_background_activation_policy();
bool background_activation_policy_active();
#else
inline void use_background_activation_policy() {}
inline bool background_activation_policy_active() { return true; }
#endif

} // namespace caliper_host
