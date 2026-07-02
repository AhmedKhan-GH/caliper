#pragma once
#include <caliper/tensor.h>
#include <cstdint>
#include <memory>
struct GLFWwindow;

namespace caliper_host {

// Host-internal renderer seam (PLATFORM.md §5.4). The ABI never sees this;
// backends are swappable forever because applets only see ImGui + bridge ids.
class HostRenderer {
public:
    virtual ~HostRenderer() = default;
    virtual bool init(GLFWwindow* window) = 0;      // after glfwCreateWindow
    virtual void new_frame() = 0;                   // backend NewFrame calls
    virtual void render(int fb_w, int fb_h) = 0;    // clear + RenderDrawData + present
    virtual void shutdown() = 0;
    virtual const char* name() const = 0;           // "gl" / "metal"

    // Texture ops the bridge builds on. data is tightly-packed RGBA8.
    virtual uint64_t tex_create_rgba8(int w, int h) = 0;            // 0 on fail
    virtual bool tex_upload_rgba8(uint64_t tex, const void* data,
                                  int w, int h) = 0;                // full update
    virtual void tex_release(uint64_t tex) = 0;
    virtual uint64_t tex_imtexture_id(uint64_t tex) = 0;            // for ImGui::Image

    // Device-resident update: src is a CaliperTensor whose data lives on this
    // backend's device (METAL buffer for metal). Returns false -> caller
    // falls back to CPU staging. GL always returns false (frozen fallback).
    virtual bool tex_update_from_device(uint64_t tex, const CaliperTensor& t,
                                        const uint32_t* lut256 /*nullable*/,
                                        float vmin, float vmax) = 0;

    // GLFW pre-window hint setup for this backend (GL profile vs NO_API).
    virtual void window_hints() = 0;
};

std::unique_ptr<HostRenderer> make_renderer(const char* name); // "gl"|"metal"|nullptr->default
}
