// webgpu_renderer.h: the web viewer spike's WebGPU backend behind
// HostRenderer (docs/plans/web-viewer-spike.md). It runs on the browser's own
// WebGPU through Emscripten's emdawnwebgpu port (webgpu.h).
//
// Only what the heatmap and series live views call is implemented: the
// swapchain path (init/new_frame/render/shutdown) drawing ImGui through
// imgui_impl_wgpu, RGBA8 textures, and the WGSL compute colormap behind
// tex_update_mapped_from_host. Everything else keeps the interface's default,
// which is an honest refusal: no device tensors, no imported allocations, no
// geometry (so the live views refuse POINTS with their own reason), no embed
// canvas. Each such path is marked "not implemented in the spike" below.
#pragma once

#include "renderer/host_renderer.h"

#include <webgpu/webgpu.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace caliper_host {

class WebGpuRenderer final : public HostRenderer {
public:
    // The device is acquired asynchronously by the page host before the
    // renderer exists (a browser cannot block for it without Asyncify), so the
    // renderer is built around it. `canvas_selector` names the canvas element.
    WebGpuRenderer(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device,
                   std::string canvas_selector);
    ~WebGpuRenderer() override;

    bool init(GLFWwindow* window) override;
    void new_frame() override;
    void render(int fb_w, int fb_h) override;
    void shutdown() override;
    const char* name() const override { return "webgpu"; }

    uint64_t tex_create_rgba8(int w, int h) override;
    bool tex_upload_rgba8(uint64_t tex, const void* data, int w, int h) override;
    void tex_release(uint64_t tex) override;
    uint64_t tex_imtexture_id(uint64_t tex) override;

    // Not implemented in the spike: a browser has no device tensors (every
    // snapshot arrives as bytes in WebAssembly memory), so the bridge never
    // reaches this with a payload the renderer could read in place.
    bool tex_update_from_device(uint64_t, const CaliperTensor&, const uint32_t*, float,
                                float) override {
        return false;
    }
    bool tex_update_mapped_from_host(uint64_t tex, const float* src, int w, int h,
                                     const uint32_t* lut256, float vmin,
                                     float vmax) override;
    const char* last_device_path() const override { return last_path_; }

    // GLFW_NO_API is set by the page host before glfwCreateWindow.
    void window_hints() override {}

    // ---- spike instrumentation (not part of HostRenderer) -----------------
    // Copy texture `tex` (renderer id or ImTextureID) back to tightly packed
    // RGBA8 and hand it to `done` once the GPU finished (asynchronous: the
    // browser cannot block on a buffer map). `done` gets an empty vector on
    // failure.
    void readback_async(uint64_t tex, std::function<void(std::vector<uint8_t>)> done);
    // Mean milliseconds from queue submit to "work done" over the frames
    // rendered so far, and how many frames that covers.
    double gpu_ms_mean() const { return gpu_n_ ? gpu_ms_sum_ / (double)gpu_n_ : 0.0; }
    uint64_t gpu_frames() const { return gpu_n_; }
    uint64_t colormap_dispatches() const { return dispatches_; }
    WGPUTextureFormat surface_format() const { return surface_format_; }

private:
    struct Tex {
        WGPUTexture     texture = nullptr;
        WGPUTextureView view = nullptr;
        int             w = 0, h = 0;
        // WGSL colormap inputs, created on the first mapped update.
        WGPUBuffer      src = nullptr;      // (h,w) f32
        WGPUBuffer      lut = nullptr;      // 256 x u32
        WGPUBuffer      params = nullptr;   // CmapParams uniform
        WGPUBindGroup   bind = nullptr;
        uint32_t        lut_cache[256] = {};
        bool            lut_valid = false;
    };

    Tex* find(uint64_t id);
    bool ensure_colormap_pipeline();
    void configure_surface(int w, int h);
    void on_work_done(double submitted_ms);

    WGPUInstance instance_ = nullptr;
    WGPUAdapter adapter_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    std::string selector_;
    WGPUSurface surface_ = nullptr;
    WGPUTextureFormat surface_format_ = WGPUTextureFormat_Undefined;
    int surf_w_ = 0, surf_h_ = 0;
    bool imgui_ready_ = false;

    WGPUComputePipeline cmap_pipeline_ = nullptr;
    WGPUBindGroupLayout cmap_layout_ = nullptr;

    std::unordered_map<uint64_t, Tex> tex_;
    uint64_t next_id_ = 1;
    const char* last_path_ = "cpu-staged";
    uint64_t dispatches_ = 0;

    double gpu_ms_sum_ = 0.0;
    uint64_t gpu_n_ = 0;
};

}  // namespace caliper_host
