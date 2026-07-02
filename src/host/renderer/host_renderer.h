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
    virtual void new_frame() = 0;                   // frame CLEAR + backend NewFrame
    virtual void render(int fb_w, int fb_h) = 0;    // RenderDrawData + present
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

    // Diagnostic: which path the last device update took ("compute"/"blit" on
    // Metal). Lifted onto the interface (C2 review) so the bridge and gfx tests
    // read it without dynamic_cast; GL and other CPU-staged backends default.
    virtual const char* last_device_path() const { return "cpu-staged"; }

    // GLFW pre-window hint setup for this backend (GL profile vs NO_API).
    virtual void window_hints() = 0;
};

// GL factory (gl_renderer.cpp, C1). Any name -> GL; GL is the default backend
// until the 2D migration. The Metal backend is a SEPARATE factory below rather
// than a branch inside make_renderer(), because make_renderer() lives in the
// frozen (do-not-touch) gl_renderer.cpp and GLRenderer is file-local there;
// main.cpp does the env-driven selection between the two factories (C2).
std::unique_ptr<HostRenderer> make_renderer(const char* name); // "gl"|"metal"|nullptr->default

// Metal factory (metal_renderer.mm, C2; Apple-only translation unit). Returns
// the Metal backend, still un-init'd — caller runs window_hints()/init() and
// falls back to make_renderer("gl") if init() fails. Not defined on non-Apple.
std::unique_ptr<HostRenderer> make_metal_renderer();
}
