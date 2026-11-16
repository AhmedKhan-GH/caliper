// webgpu_renderer.cpp: the browser's WebGPU backend (see
// webgpu_renderer.h). Browser-only: it targets Emscripten's emdawnwebgpu port,
// so the surface is a canvas, presentation happens when the frame callback
// returns to the browser, and every completion is a callback, never a wait.
//
// The surface, the frame and ImGui are drawn here directly; every resource and
// draw of the live views goes through the graphics layer (webgpu_backend.h):
//   RGBA8 textures     create_texture (sampled, storage, copy source and
//                      destination); an upload is write_texture; the
//                      ImTextureID is imgui_texture_id.
//   The colormap       per texture a source buffer, the colour table and the
//                      thresholds as layer buffers with one bind group; each
//                      update writes the bytes that changed and submits a Frame
//                      of one dispatch whose parameters are a uniform in the
//                      frame's arena.
//   Readback           read_texture, handed over once the layer has it
//                      (release_retired).
// The points and scene views draw through the scene core on the same backend
// (src/host/live/live_points.h, live_scene.h).
#include "webgpu_renderer.h"
#include "webgpu_programs.h"

#include <imgui.h>
#include <imgui_impl_wgpu.h>

#include <emscripten/emscripten.h>

#include "tensor_bridge.h"   // colormap_thresholds, from the colormap's reference

#include <cmath>
#include <cstdio>
#include <cstring>

namespace caliper_host {

namespace gfx = caliper_gfx;
namespace prog = webgpu_programs;

namespace {

// The colormap's thresholds (webgpu_programs.h, kColormap). The
// shaders give exactly the bytes of the CPU reference map_f32_to_rgba8
// (tensor_bridge.cpp) on every adapter: WGSL allows a division 2.5 ULP of
// error and lets an implementation assume no NaN or infinity, so recomputing
// idx = clamp((v - vmin)/(vmax - vmin), 0, 1) * 255 + 0.5 on the GPU differed
// from the reference by one level for a few texels on Metal. Instead the CPU
// finds, for each index k in 1..255, the threshold T[k]: the first float at
// which the reference gives index k or more (the last one, for a decreasing
// range), with colormap_thresholds (tensor_bridge.h), which the Metal renderer
// shares. The reference is monotone in v, because every IEEE operation in it
// is, so the index of v is the number of thresholds it has passed, which the
// shader finds with a binary search of exact float comparisons. mode 0 is a
// range whose width vmax - vmin is zero, infinite or NaN, where the reference
// maps every value to index 0 (t is 0 or NaN), 1 an increasing and 2 a
// decreasing one; NaN takes index 0, tested on the bits.

WGPUStringView sv(const char* s) { return WGPUStringView{s, WGPU_STRLEN}; }

void report(gfx::Backend& b, const char* what) {
    std::fprintf(stderr, "[webgpu] %s: %s\n", what, b.last_error().c_str());
}

// Releases a handle that is set; a null one would only leave an error behind.
template <class H> void drop(gfx::Backend& b, H& h) {
    if (h) b.release(h);
    h = H{};
}

}  // namespace

WebGpuRenderer::WebGpuRenderer(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device,
                               std::string canvas_selector)
    : instance_(instance), adapter_(adapter), device_(device),
      selector_(std::move(canvas_selector)) {
    queue_ = wgpuDeviceGetQueue(device_);
}

WebGpuRenderer::~WebGpuRenderer() {
    *alive_ = false;
    shutdown();
}

bool WebGpuRenderer::init(GLFWwindow*) {
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas =
        WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    canvas.selector = sv(selector_.c_str());
    WGPUSurfaceDescriptor sd = WGPU_SURFACE_DESCRIPTOR_INIT;
    sd.nextInChain = &canvas.chain;
    surface_ = wgpuInstanceCreateSurface(instance_, &sd);
    if (!surface_) {
        std::fprintf(stderr, "[webgpu] no surface for canvas %s\n", selector_.c_str());
        return false;
    }
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    wgpuSurfaceGetCapabilities(surface_, adapter_, &caps);
    if (caps.formatCount == 0) {
        std::fprintf(stderr, "[webgpu] the canvas reports no surface format\n");
        wgpuSurfaceCapabilitiesFreeMembers(caps);
        return false;
    }
    surface_format_ = caps.formats[0];   // the browser's preferred canvas format
    wgpuSurfaceCapabilitiesFreeMembers(caps);

    ImGui_ImplWGPU_InitInfo ii;
    ii.Device = device_;
    ii.NumFramesInFlight = 3;
    ii.RenderTargetFormat = surface_format_;
    ii.DepthStencilFormat = WGPUTextureFormat_Undefined;
    if (!ImGui_ImplWGPU_Init(&ii)) return false;
    imgui_ready_ = true;
    return init_layer();
}

void WebGpuRenderer::configure_surface(int w, int h) {
    WGPUSurfaceConfiguration c = WGPU_SURFACE_CONFIGURATION_INIT;
    c.device = device_;
    c.format = surface_format_;
    c.usage = WGPUTextureUsage_RenderAttachment;
    c.width = (uint32_t)w;
    c.height = (uint32_t)h;
    c.alphaMode = WGPUCompositeAlphaMode_Opaque;
    c.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(surface_, &c);
    surf_w_ = w;
    surf_h_ = h;
}

void WebGpuRenderer::new_frame() {
    if (imgui_ready_) ImGui_ImplWGPU_NewFrame();
}

void WebGpuRenderer::release_retired() {
    if (!gfx_) return;
    // The readbacks the layer has finished.
    for (size_t i = 0; i < readbacks_.size();) {
        std::vector<uint8_t> out;
        if (!gfx_->alive(readbacks_[i].rb)) {   // refused or cancelled
            auto done = std::move(readbacks_[i].done);
            readbacks_.erase(readbacks_.begin() + (std::ptrdiff_t)i);
            done({});
        } else if (gfx_->take_readback(readbacks_[i].rb, &out)) {
            auto done = std::move(readbacks_[i].done);
            readbacks_.erase(readbacks_.begin() + (std::ptrdiff_t)i);
            done(std::move(out));
        } else {
            ++i;
        }
    }
    gfx_->release_retired();
}

void WebGpuRenderer::render(int fb_w, int fb_h) {
    if (!surface_ || fb_w <= 0 || fb_h <= 0) return release_retired();
    if (fb_w != surf_w_ || fb_h != surf_h_) configure_surface(fb_w, fb_h);
    WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface_, &st);
    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        if (st.texture) wgpuTextureRelease(st.texture);
        return release_retired();
    }
    WGPUTextureView view = wgpuTextureCreateView(st.texture, nullptr);

    WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    ca.view = view;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = WGPUColor{0x18 / 255.0, 0x17 / 255.0, 0x15 / 255.0, 1.0};   // sternbild::col::ground
    WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
    wgpuRenderPassEncoderEnd(pass);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);

    // Submit -> done latency of this frame's work (instrumentation).
    // No present call: the browser presents the canvas when this callback
    // returns to its event loop.
    struct Submitted {
        std::shared_ptr<bool> alive;
        WebGpuRenderer* renderer;
        double t0;
    };
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView, void* u1, void*) {
        Submitted* s = static_cast<Submitted*>(u1);
        if (status == WGPUQueueWorkDoneStatus_Success && *s->alive) s->renderer->on_work_done(s->t0);
        delete s;
    };
    cb.userdata1 = new Submitted{alive_, this, emscripten_get_now()};
    wgpuQueueOnSubmittedWorkDone(queue_, cb);

    wgpuCommandBufferRelease(cmd);
    wgpuRenderPassEncoderRelease(pass);
    wgpuCommandEncoderRelease(enc);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(st.texture);
    // The views released during this frame: its bind groups are gone now.
    release_retired();
}

void WebGpuRenderer::on_work_done(double submitted_ms) {
    gpu_ms_sum_ += emscripten_get_now() - submitted_ms;
    ++gpu_n_;
}

void WebGpuRenderer::shutdown() {
    if (gfx_) {
        for (auto& kv : tex_) free_tex(kv.second);
        for (auto& r : readbacks_) {
            drop(*gfx_, r.rb);
            r.done({});
        }
        drop(*gfx_, cmap_frame_group_);
        drop(*gfx_, cmap_pipe_);
        drop(*gfx_, cmap_prog_);
        gfx_->release_retired();
        gfx_.reset();
    }
    tex_.clear();
    readbacks_.clear();
    if (imgui_ready_) ImGui_ImplWGPU_Shutdown();
    imgui_ready_ = false;
    if (surface_) {
        wgpuSurfaceUnconfigure(surface_);
        wgpuSurfaceRelease(surface_);
        surface_ = nullptr;
    }
}

// The graphics layer's backend on this renderer's device and queue, and what the textures' colormap
// needs of it: its program, its pipeline and the bind group of its uniform arena.
bool WebGpuRenderer::init_layer() {
    std::string adapter;
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (adapter_ && wgpuAdapterGetInfo(adapter_, &info) == WGPUStatus_Success) {
        if (info.description.data)
            adapter.assign(info.description.data, info.description.length == WGPU_STRLEN
                                                      ? std::strlen(info.description.data)
                                                      : info.description.length);
        wgpuAdapterInfoFreeMembers(info);
    }
    gfx_ = gfx::WebGpuBackend::create(device_, queue_, adapter.c_str());
    if (!gfx_) {
        std::fprintf(stderr, "[webgpu] the graphics layer's backend could not be made\n");
        return false;
    }
    cmap_prog_ = gfx_->create_program(prog::kColormap);
    gfx::ComputePipelineDesc cd;
    cd.program = cmap_prog_;
    cd.entry = "cmap_f32";
    cd.label = "caliper.cmap_f32";
    cmap_pipe_ = gfx_->create_compute_pipeline(cd);
    // The dynamic uniform is the arena of the pass it is used in: one bind group serves every
    // dispatch.
    gfx::BindGroupDesc g;
    g.group = 0;
    g.entry_count = 1;
    g.entries[0].binding = 0;
    g.entries[0].buffer = gfx::kArena;
    g.program = cmap_prog_;
    cmap_frame_group_ = gfx_->create_bind_group(g);
    if (!cmap_prog_ || !cmap_pipe_ || !cmap_frame_group_) {
        report(*gfx_, "graphics layer set-up");
        return false;
    }
    return true;
}

WebGpuRenderer::Tex* WebGpuRenderer::find(uint64_t id) {
    auto it = tex_.find(id);
    if (it != tex_.end()) return &it->second;
    for (auto& kv : tex_)   // an ImTextureID (the view pointer) works too
        if (gfx_->imgui_texture_id(kv.second.color) == id) return &kv.second;
    return nullptr;
}

// ---- textures ------------------------------------------------------------------------------------

uint64_t WebGpuRenderer::tex_create_rgba8(int w, int h) {
    if (!gfx_ || w <= 0 || h <= 0) return 0;
    gfx::TextureDesc d;
    d.width = (uint32_t)w;
    d.height = (uint32_t)h;
    d.format = gfx::Format::RGBA8Unorm;
    d.usage = gfx::kSampled | gfx::kStorageTex | gfx::kCopySrcTex | gfx::kCopyDstTex;
    d.label = "caliper.texture";
    Tex t;
    t.color = gfx_->create_texture(d);
    if (!t.color) {
        report(*gfx_, "tex_create_rgba8");
        return 0;
    }
    t.w = w;
    t.h = h;
    const uint64_t id = next_id_++;
    tex_.emplace(id, t);
    return id;
}

bool WebGpuRenderer::tex_upload_rgba8(uint64_t id, const void* data, int w, int h) {
    if (!gfx_) return false;
    Tex* t = find(id);
    if (!t || !data || w != t->w || h != t->h) return false;
    if (!gfx_->write_texture(t->color, data, (uint32_t)w * 4)) {
        report(*gfx_, "tex_upload_rgba8");
        return false;
    }
    last_path_ = "cpu-staged";
    return true;
}

// The layer destroys a released texture once the last submit that used it completed, at once
// when none is pending, and keeps its view to the end of the frame (WebGpuBackend::
// release_retired). The ImGui backend makes one bind group per texture view while it records a
// frame, keyed by the view's address, and releases all of them at the end of that frame, so no
// bind group outlives the frame and a later view at the same address gets its own.
void WebGpuRenderer::tex_release(uint64_t id) {
    if (!gfx_) return;
    auto it = tex_.find(id);
    if (it == tex_.end()) return;
    free_tex(it->second);
    tex_.erase(it);
}

uint64_t WebGpuRenderer::tex_imtexture_id(uint64_t id) {
    Tex* t = gfx_ ? find(id) : nullptr;
    return t ? gfx_->imgui_texture_id(t->color) : 0;
}

void WebGpuRenderer::free_tex(Tex& t) {
    drop(*gfx_, t.cmap);
    drop(*gfx_, t.src);
    drop(*gfx_, t.lut);
    drop(*gfx_, t.th);
    drop(*gfx_, t.color);
}

bool WebGpuRenderer::tex_update_mapped_from_host(uint64_t id, const float* data, int w, int h,
                                                 const uint32_t* lut256, float vmin, float vmax) {
    if (!gfx_) return false;
    Tex* t = find(id);
    if (!t || !data || !lut256 || w != t->w || h != t->h) return false;
    const uint64_t bytes = (uint64_t)w * (uint64_t)h * 4;
    if (!t->src) {
        gfx::BufferDesc bd;
        bd.usage = gfx::kStorage | gfx::kCopyDst;
        bd.size = bytes;
        bd.label = "caliper.cmap.src";
        t->src = gfx_->create_buffer(bd);
        bd.size = 1024;
        bd.label = "caliper.cmap.lut";
        t->lut = gfx_->create_buffer(bd);
        bd.label = "caliper.cmap.thresholds";
        t->th = gfx_->create_buffer(bd);
        gfx::BindGroupDesc g;
        g.program = cmap_prog_;
        g.group = 1;
        g.entry_count = 4;
        g.entries[0].binding = 0;
        g.entries[0].buffer = t->src;
        g.entries[0].size = bytes;
        g.entries[1].binding = 1;
        g.entries[1].buffer = t->lut;
        g.entries[1].size = 1024;
        g.entries[2].binding = 2;
        g.entries[2].buffer = t->th;
        g.entries[2].size = 1024;
        g.entries[3].binding = 3;
        g.entries[3].texture = t->color;
        g.label = "caliper.cmap";
        if (t->src && t->lut && t->th) t->cmap = gfx_->create_bind_group(g);
        if (!t->cmap) {
            report(*gfx_, "colormap set-up");
            drop(*gfx_, t->src);
            drop(*gfx_, t->lut);
            drop(*gfx_, t->th);
            t->src = t->lut = t->th = {};
            return false;
        }
        t->lut_valid = false;
        t->th_valid = false;
    }
    gfx_->write_buffer(t->src, 0, data, bytes);
    if (!t->lut_valid || std::memcmp(t->lut_cache, lut256, 1024) != 0) {
        std::memcpy(t->lut_cache, lut256, 1024);
        t->lut_valid = true;
        gfx_->write_buffer(t->lut, 0, lut256, 1024);
    }
    // The thresholds change with the range only (an AUTO range changes with most snapshots;
    // computing them takes some 8,000 reference evaluations).
    if (!t->th_valid || t->th_vmin != vmin || t->th_vmax != vmax) {
        float T[256];
        t->th_mode = colormap_thresholds(vmin, vmax, T);
        if (t->th_mode) gfx_->write_buffer(t->th, 0, T, sizeof T);
        t->th_vmin = vmin;
        t->th_vmax = vmax;
        t->th_valid = true;
    }
    const prog::CmapParams p{(uint32_t)w, (uint32_t)h, 1u, (uint32_t)w, vmin, vmax, t->th_mode, 0u};
    gfx::Frame f;
    const uint32_t off = f.uniform(&p, sizeof p);
    const gfx::BindGroup groups[2] = {cmap_frame_group_, t->cmap};
    const uint32_t offsets[1] = {off};
    const gfx::Texture writes[1] = {t->color};
    const uint32_t n = prog::kColormapGroup;
    f.dispatch(cmap_pipe_, groups, offsets, ((uint32_t)w + n - 1) / n, ((uint32_t)h + n - 1) / n, 1, writes);
    if (!gfx_->submit(std::move(f))) {
        report(*gfx_, "colormap submit");
        return false;
    }
    ++dispatches_;
    last_path_ = "webgpu-compute";
    return true;
}

// ---- readback ------------------------------------------------------------------------------------

void WebGpuRenderer::readback_async(gfx::Texture tex, uint32_t w, uint32_t h,
                                    std::function<void(std::vector<uint8_t>)> done) {
    if (!gfx_ || !gfx_->alive(tex)) {
        done({});
        return;
    }
    const gfx::Readback rb = gfx_->read_texture(tex, 0, 0, w, h);
    if (!rb) {
        report(*gfx_, "readback");
        done({});
        return;
    }
    readbacks_.push_back({rb, std::move(done)});
}

void WebGpuRenderer::readback_async(uint64_t id, std::function<void(std::vector<uint8_t>)> done) {
    Tex* t = gfx_ ? find(id) : nullptr;
    if (!t) {
        done({});
        return;
    }
    const gfx::Readback rb = gfx_->read_texture(t->color, 0, 0, (uint32_t)t->w, (uint32_t)t->h);
    if (!rb) {
        report(*gfx_, "readback");
        done({});
        return;
    }
    readbacks_.push_back({rb, std::move(done)});
}

}  // namespace caliper_host
