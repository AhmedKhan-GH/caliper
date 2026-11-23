#pragma once

struct GLFWwindow;

namespace caliper_host {

// CALIPER_MOUSE_LOG records native mouse timestamps, GLFW callbacks, ImGui consumption,
// and the first camera change of each press. Start after the backend installs its callbacks.
// Stop before backend shutdown. All calls use the main thread and the desktop ImGui context.
#ifdef __APPLE__
void mouse_latency_start(GLFWwindow* window);
void mouse_latency_stop();
bool mouse_latency_enabled();
void mouse_latency_camera(int button);
#else
inline void mouse_latency_start(GLFWwindow*) {}
inline void mouse_latency_stop() {}
inline bool mouse_latency_enabled() { return false; }
inline void mouse_latency_camera(int) {}
#endif

} // namespace caliper_host
