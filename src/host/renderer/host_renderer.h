#pragma once
#include <caliper/tensor.h>
#include <cstdint>
#include <memory>
#include <vector>
struct GLFWwindow;

namespace caliper_host {

// Host-side resolved draw for caliper.geometry.v1_1. Applet-facing allocation
// ids and colormap ids have already been resolved by TensorBridge; backends
// still re-check liveness/bounds against their own tables before encoding.
struct HostGeomDraw {
    uint64_t pos_alloc = 0;
    uint64_t pos_offset = 0;
    uint64_t vertex_count = 0;
    uint64_t index_alloc = 0;
    uint64_t index_offset = 0;
    uint64_t index_count = 0;
    uint64_t normal_alloc = 0;
    uint64_t normal_offset = 0;
    uint64_t attr_alloc = 0;
    uint64_t attr_offset = 0;
    uint32_t topology = 0;
    uint32_t color_mode = 0;
    uint32_t shade_mode = 0;
    uint32_t blend_mode = 0;
    uint32_t depth_flags = 0;
    uint32_t flat_rgba = 0xffffffffu;
    const uint32_t* lut256 = nullptr;
    float vmin = 0.0f;
    float vmax = 1.0f;
    float size_px = 1.0f;
    float model[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
};

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

    // Which device kind this backend imports on its zero-copy path (spec §3.4).
    // The bridge maps this to active_device_ instead of matching on name(),
    // so a new backend never grows a strcmp arm. Metal -> METAL; Vulkan -> CUDA
    // when a UUID-matched CUDA device is paired, else CPU; everyone else CPU.
    virtual CaliperDeviceKind interop_device() const { return CALIPER_DEV_CPU; }

    // Literal zero-copy alloc_shared (spec §3.5): make texture `tex`'s interop
    // buffer be the tensor's backing store and return a device pointer to it
    // (a CUDA device ptr on Vulkan). The applet's kernels write there directly
    // and the update pass reads it in place — zero data copies. Default false:
    // backends without device interop keep the CPU-vector alloc_shared.
    virtual bool alloc_device_shared(uint64_t /*tex*/, uint64_t /*bytes*/,
                                     void** /*out_device_ptr*/) { return false; }

    // External-allocation import (bridge v1.2). Default: unsupported — the
    // bridge then never grants CALIPER_BRIDGE_CAP_IMPORT_ALLOC. import_external_
    // allocation dups the OS shareable handle and returns a renderer-internal
    // id (0 on failure); the update pass runs a device texture update FROM the
    // imported allocation at offset_bytes, with the imported bytes as the
    // address (desc->data ignored). Colormap/vmin/vmax are the texture's stored
    // (pinned-at-create) mapping values, same as tex_update_from_device.
    virtual bool supports_external_import() const { return false; }
    virtual uint64_t import_external_allocation(void* /*os_handle*/,
                                                uint64_t /*size_bytes*/,
                                                uint32_t /*handle_type*/) { return 0; }
    virtual void release_external_allocation(uint64_t /*id*/) {}
    virtual bool tex_update_from_imported(uint64_t /*tex*/, uint64_t /*alloc*/,
                                          uint64_t /*offset_bytes*/,
                                          const CaliperTensor& /*desc*/,
                                          int32_t /*colormap*/,
                                          float /*vmin*/, float /*vmax*/) { return false; }

    // Imported 3-D geometry (caliper.geometry.v1). Default: unsupported — the
    // service then never grants CALIPER_GEOM_CAP_IMPORTED_POINTS. A view is an
    // offscreen render target registered in the SAME texture table as bridge
    // textures (ImGui-drawable, debug-readable); it is released via
    // tex_release like any texture. geom_draw_points renders ONE view frame
    // atomically: clear to clear_rgba, then `count` instanced points pulled
    // from the imported allocation (positions (count,3) f32 at pos_offset;
    // optional (count,) f32 attr colormapped via lut256, lut null = flat
    // white). Additive blend, no depth. count 0 = pure clear (alloc ids may
    // be 0 then). false = view pixels untouched.
    virtual bool supports_geometry() const { return false; }
    virtual uint64_t geom_create_view(int /*w*/, int /*h*/) { return 0; }
    virtual bool geom_draw_points(uint64_t /*view_tex*/,
                                  const float* /*view16*/, const float* /*proj16*/,
                                  uint64_t /*pos_alloc*/, uint64_t /*pos_offset*/,
                                  uint64_t /*count*/,
                                  uint64_t /*attr_alloc*/, uint64_t /*attr_offset*/,
                                  const uint32_t* /*lut256*/,
                                  float /*vmin*/, float /*vmax*/,
                                  float /*size_px*/, uint32_t /*clear_rgba*/) {
        return false;
    }
    virtual bool supports_geometry_primitives() const { return false; }
    virtual uint64_t geom_create_view_ex(int /*w*/, int /*h*/,
                                         uint32_t /*flags*/) { return 0; }
    virtual bool geom_draw_primitives(uint64_t /*view_tex*/,
                                      const float* /*view16*/, const float* /*proj16*/,
                                      const HostGeomDraw* /*draws*/,
                                      uint32_t /*count*/,
                                      uint32_t /*clear_rgba*/) {
        return false;
    }

    // GLFW pre-window hint setup for this backend (GL profile vs NO_API).
    virtual void window_hints() = 0;

    // Test-only: copy a texture's pixels back to host RGBA8 (w*h*4 bytes), or
    // empty on failure. GL/Metal read back their handle directly in the gfx
    // harness; Vulkan's CaliperTextureId is an opaque descriptor set, so the
    // backend must do the copy. Not used on any hot path.
    virtual std::vector<uint8_t> debug_readback_rgba8(uint64_t /*id*/,
                                                      int /*w*/, int /*h*/) {
        return {};
    }

    // D24 (docs/metal-pipelining.md §4): true when this backend honors a
    // non-NULL CaliperTensor.stream by GPU-ordering the device update after
    // the producer's stream/queue. Surfaced to applets as bridge-v1.1 caps()
    // bit 0. Default false: a backend that ignores stream must never let an
    // adapter skip its drain.
    virtual bool honors_stream_ordered_handoff() const { return false; }
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

// Vulkan factory (vulkan_renderer.cpp, Phase 4; Windows-only translation unit
// today). Same contract as the Metal factory: un-init'd, caller falls back to
// GL if init() fails. Device-resident CUDA tensors via VK_KHR_external_memory
// + the runtime-loaded CUDA driver API (ZEROCOPY.md). Not defined elsewhere.
std::unique_ptr<HostRenderer> make_vulkan_renderer();
}
