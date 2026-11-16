// webgpu_renderer.h: the WebGPU backend behind HostRenderer for Caliper's
// views in the browser (web/wasm). It runs on the browser's own WebGPU through
// Emscripten's emdawnwebgpu port (webgpu.h).
//
// The renderer itself keeps the surface, the frame loop and ImGui, drawn
// through imgui_impl_wgpu. Every resource and draw of the live views goes
// through the graphics layer's WebGPU backend (webgpu_backend.h, design/
// renderer-core.md, Migration, step 4), made on the renderer's device and
// queue: RGBA8 textures, the compute colormap behind
// tex_update_mapped_from_host, and the points of the geometry path: host
// allocations (import_host_allocation, a copy the layer reports), render
// targets with depth (geom_create_view_ex) and point draws
// (geom_draw_primitives with the points topology), each point an instanced
// quad because WebGPU points are one pixel. The programs are those of
// webgpu_programs.h. Everything else keeps the interface's default, which is
// an honest refusal: no device tensors, no imported OS allocations, no lines,
// triangles, textured or instanced geometry, no embed canvas. Each such path
// is marked "not in the browser renderer" below.
#pragma once

#include "renderer/host_renderer.h"
#include "renderer/webgpu_backend.h"

#include <webgpu/webgpu.h>

#include <cstdint>
#include <functional>
#include <memory>
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

    // Not in the browser renderer: a browser has no device tensors (every
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

    // ---- the geometry path, points only ---------------------------------------
    uint64_t import_host_allocation(const void* data, uint64_t bytes) override;
    void release_external_allocation(uint64_t id) override;
    bool supports_geometry() const override { return true; }
    bool supports_geometry_primitives() const override { return true; }
    uint64_t geom_create_view(int w, int h) override { return geom_create_view_ex(w, h, 0); }
    uint64_t geom_create_view_ex(int w, int h, uint32_t flags) override;
    // Not in the browser renderer: lines, triangles, indices, normals, UVs,
    // textures and instance matrices (the live views draw points only). A
    // draw of another topology is refused and the target left untouched.
    bool geom_draw_primitives(uint64_t view_tex, const float* view16, const float* proj16,
                              const HostGeomDraw* draws, uint32_t count,
                              uint32_t clear_rgba) override;

    // No GLFW in the browser: init() takes no window (the canvas is named by
    // the selector), and there are no window hints.
    void window_hints() override {}

    // The graphics layer's backend, from init() until shutdown().
    caliper_gfx::WebGpuBackend* gfx() { return gfx_.get(); }

    // ---- instrumentation (not part of HostRenderer) ------------------------
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
    // What the renderer holds now (a long session must not grow these): live
    // textures, host allocations and the layer's spare buffers, and released
    // texture views waiting for the end of the frame.
    size_t texture_count() const { return tex_.size(); }
    size_t allocation_count() const { return allocs_.size() + (gfx_ ? gfx_->spare_buffer_count() : 0); }
    size_t retired_view_count() const { return gfx_ ? gfx_->retired_view_count() : 0; }
    // Hands finished readbacks to their callbacks and releases the texture
    // views released since the last frame was recorded. render() calls it; a
    // host that skips a frame (a hidden canvas) calls it instead, outside any
    // ImGui frame.
    void release_retired();
    WGPUTextureFormat surface_format() const { return surface_format_; }

private:
    struct Tex {
        caliper_gfx::Texture color, depth;      // depth: a geometry target's
        int w = 0, h = 0;
        // The colormap's inputs, made on the first mapped update.
        caliper_gfx::Buffer src, lut, th;
        caliper_gfx::BindGroup cmap;
        uint32_t lut_cache[256] = {};
        bool lut_valid = false;
        // The colormap's index thresholds for the range they were made for.
        float th_vmin = 0.0f, th_vmax = 0.0f;
        uint32_t th_mode = 0;
        bool th_valid = false;
    };
    struct Alloc {
        caliper_gfx::Buffer buffer;
        uint64_t bytes = 0;
    };
    struct Pending {
        caliper_gfx::Readback rb;
        std::function<void(std::vector<uint8_t>)> done;
    };

    Tex* find(uint64_t id);
    void free_tex(Tex& t);
    bool init_layer();
    caliper_gfx::Pipeline points_pipeline(bool depth_target, uint32_t depth_flags, bool cmap);
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

    std::unique_ptr<caliper_gfx::WebGpuBackend> gfx_;
    caliper_gfx::Program cmap_prog_, points_prog_;
    caliper_gfx::Pipeline cmap_pipe_;
    caliper_gfx::Pipeline points_pipes_[10] = {};   // [(depth target ? 1 + flags : 0) * 2 + colormap]
    caliper_gfx::BindGroup cmap_frame_group_, points_group_;
    caliper_gfx::Buffer zero16_;                    // the stride-0 value stream of a flat draw

    std::unordered_map<uint64_t, Tex> tex_;
    std::unordered_map<uint64_t, Alloc> allocs_;
    std::vector<Pending> readbacks_;
    uint64_t next_id_ = 1;
    const char* last_path_ = "cpu-staged";
    uint64_t dispatches_ = 0;

    double gpu_ms_sum_ = 0.0;
    uint64_t gpu_n_ = 0;
    // False once this renderer is gone: a work-done callback that arrives
    // after the page destroyed it must not touch it.
    std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};

}  // namespace caliper_host
