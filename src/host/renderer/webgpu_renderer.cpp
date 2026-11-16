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
//   Host allocations   import_buffer of host memory: a copy, which the layer
//                      reports as the route copied-from-host.
//   Render targets     an RGBA8 texture and a Depth32Float texture, cleared by
//                      an empty draw list.
//   Points             one DrawList per call: per draw a 2,160-byte uniform
//                      block (matrix, colour rule, table and thresholds, depth
//                      cues) in the list's arena, the positions and values
//                      bound as instance-step vertex buffers at the draw's
//                      byte offsets.
//   Readback           read_texture, handed over once the layer has it
//                      (release_retired).
#include "webgpu_renderer.h"
#include "webgpu_programs.h"

#include <imgui.h>
#include <imgui_impl_wgpu.h>

#include <emscripten/emscripten.h>

#include "tensor_bridge.h"   // colormap_thresholds, from the colormap's reference

#include <caliper/services/geometry_v1_1.h>   // CALIPER_GEOM_*

#include <cmath>
#include <cstdio>
#include <cstring>

namespace caliper_host {

namespace gfx = caliper_gfx;
namespace prog = webgpu_programs;

namespace {

// The colormap's thresholds (webgpu_programs.h, kColormap and kPoints). The
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
// decreasing one; NaN takes index 0, tested on the bits. The points shader
// colours by value the same way.

WGPUStringView sv(const char* s) { return WGPUStringView{s, WGPU_STRLEN}; }

void mul4(const float* a, const float* b, float* out) {   // out = a * b, column-major
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float acc = 0.0f;
            for (int k = 0; k < 4; ++k) acc += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = acc;
        }
}

bool geom_fail(const char* why) {
    std::fprintf(stderr, "[webgpu] geometry refused: %s\n", why);
    return false;
}

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
        for (auto& kv : allocs_) drop(*gfx_, kv.second.buffer);
        for (auto& r : readbacks_) {
            drop(*gfx_, r.rb);
            r.done({});
        }
        for (auto& p : points_pipes_) drop(*gfx_, p);
        drop(*gfx_, points_group_);
        drop(*gfx_, cmap_frame_group_);
        drop(*gfx_, cmap_pipe_);
        drop(*gfx_, zero16_);
        drop(*gfx_, points_prog_);
        drop(*gfx_, cmap_prog_);
        gfx_->release_retired();
        gfx_.reset();
    }
    tex_.clear();
    allocs_.clear();
    readbacks_.clear();
    if (imgui_ready_) ImGui_ImplWGPU_Shutdown();
    imgui_ready_ = false;
    if (surface_) {
        wgpuSurfaceUnconfigure(surface_);
        wgpuSurfaceRelease(surface_);
        surface_ = nullptr;
    }
}

// The graphics layer's backend on this renderer's device and queue, and what the live views'
// draws need of it: the two programs, the colormap's pipeline, the bind groups of their uniform
// arena and the 16-byte constant that a flat-coloured point draw binds as its value stream.
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
    points_prog_ = gfx_->create_program(prog::kPoints);
    gfx::ComputePipelineDesc cd;
    cd.program = cmap_prog_;
    cd.entry = "cmap_f32";
    cd.label = "caliper.cmap_f32";
    cmap_pipe_ = gfx_->create_compute_pipeline(cd);
    // The dynamic uniform of each program is the arena of the pass it is used in: one bind group
    // serves every dispatch and every draw.
    gfx::BindGroupDesc g;
    g.group = 0;
    g.entry_count = 1;
    g.entries[0].binding = 0;
    g.entries[0].buffer = gfx::kArena;
    g.program = cmap_prog_;
    cmap_frame_group_ = gfx_->create_bind_group(g);
    g.program = points_prog_;
    points_group_ = gfx_->create_bind_group(g);
    gfx::BufferDesc zd;
    zd.size = 16;
    zd.usage = gfx::kVertex | gfx::kCopyDst;
    zd.label = "caliper.points.zero";
    zero16_ = gfx_->create_buffer(zd);
    if (!cmap_prog_ || !points_prog_ || !cmap_pipe_ || !cmap_frame_group_ || !points_group_ || !zero16_) {
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
    drop(*gfx_, t.depth);
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

// ---- the geometry path: points as instanced quads ------------------------------------------------

uint64_t WebGpuRenderer::import_host_allocation(const void* data, uint64_t bytes) {
    if (!gfx_ || !data || bytes == 0 || bytes > (1ull << 31)) return 0;
    gfx::ExternalBuffer ext;
    ext.kind = gfx::ExternalKind::HostMemory;
    ext.usage = gfx::kVertex;
    ext.label = "caliper.points.alloc";
    std::vector<uint8_t> padded;
    if (bytes % 4 == 0) {
        ext.handle = const_cast<void*>(data);
        ext.size = bytes;
    } else {   // the layer takes multiples of 4
        padded.assign((bytes + 3) & ~uint64_t(3), 0);
        std::memcpy(padded.data(), data, (size_t)bytes);
        ext.handle = padded.data();
        ext.size = padded.size();
    }
    const gfx::ImportResult r = gfx_->import_buffer(ext);
    if (!r.buffer) {
        std::fprintf(stderr, "[webgpu] import refused: %s\n", r.reason.c_str());
        return 0;
    }
    const uint64_t id = next_id_++;
    allocs_.emplace(id, Alloc{r.buffer, bytes});
    return id;
}

void WebGpuRenderer::release_external_allocation(uint64_t id) {
    if (!gfx_) return;
    auto it = allocs_.find(id);
    if (it == allocs_.end()) return;
    drop(*gfx_, it->second.buffer);   // the layer keeps it for reuse once its draws completed
    allocs_.erase(it);
}

uint64_t WebGpuRenderer::geom_create_view_ex(int w, int h, uint32_t flags) {
    if (!gfx_ || (flags & ~CALIPER_GEOM_VIEW_DEPTH) != 0u || w <= 0 || h <= 0) return 0;
    gfx::TextureDesc d;
    d.width = (uint32_t)w;
    d.height = (uint32_t)h;
    d.format = gfx::Format::RGBA8Unorm;
    d.usage = gfx::kSampled | gfx::kRenderTarget | gfx::kCopySrcTex;
    d.label = "caliper.geom.view";
    Tex t;
    t.color = gfx_->create_texture(d);
    if (!t.color) return 0;
    if (flags & CALIPER_GEOM_VIEW_DEPTH) {
        d.format = gfx::Format::Depth32Float;
        d.usage = gfx::kRenderTarget;
        d.label = "caliper.geom.depth";
        t.depth = gfx_->create_texture(d);
        if (!t.depth) {
            drop(*gfx_, t.color);
            return 0;
        }
    }
    t.w = w;
    t.h = h;
    const uint64_t id = next_id_++;
    tex_.emplace(id, t);
    // Black until the first draw, as the native targets start.
    const float I[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    geom_draw_primitives(id, I, I, nullptr, 0, 0xff000000u);
    return id;
}

// [(depth target ? 1 + flags : 0) * 2 + colormap]: a target without depth, or with a depth buffer
// tested and written as the draw's depth flags say (less-equal when testing, as Metal); the value
// stream at stride 4 for a colormapped draw and at stride 0 for a flat one.
gfx::Pipeline WebGpuRenderer::points_pipeline(bool depth_target, uint32_t depth_flags, bool cmap) {
    const uint32_t f = depth_flags & (CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE);
    const int slot = (depth_target ? 1 + (int)f : 0) * 2 + (cmap ? 1 : 0);
    if (points_pipes_[slot]) return points_pipes_[slot];
    gfx::RenderPipelineDesc d;
    d.program = points_prog_;
    d.vertex_entry = "vs";
    d.fragment_entry = "fs";
    d.vertex_layout_count = 2;
    d.vertex_layouts[0].stride = 12;
    d.vertex_layouts[0].step = gfx::Step::Instance;
    d.vertex_layouts[0].attribute_count = 1;
    d.vertex_layouts[0].attributes[0] = {0, gfx::VertexFormat::F32x3, 0};
    d.vertex_layouts[1].stride = cmap ? 4 : 0;
    d.vertex_layouts[1].step = gfx::Step::Instance;
    d.vertex_layouts[1].attribute_count = 1;
    d.vertex_layouts[1].attributes[0] = {1, gfx::VertexFormat::U32, 0};
    d.topology = gfx::Topology::TriangleList;
    d.color_format = gfx::Format::RGBA8Unorm;
    d.blend = gfx::Blend::Opaque;
    d.depth = depth_target;
    d.depth_test = depth_target && (f & CALIPER_GEOM_DEPTH_TEST);
    d.depth_write = depth_target && (f & CALIPER_GEOM_DEPTH_WRITE);
    d.label = "caliper.points";
    points_pipes_[slot] = gfx_->create_render_pipeline(d);
    if (!points_pipes_[slot]) report(*gfx_, "points pipeline");
    return points_pipes_[slot];
}

const WebGpuRenderer::PointThresholds& WebGpuRenderer::point_thresholds(float vmin, float vmax) {
    // The thresholds change with the range only, and computing them takes some 8,000 reference
    // evaluations, so the last few ranges keep theirs; the range is compared by its bits.
    uint32_t key[2];
    std::memcpy(&key[0], &vmin, 4);
    std::memcpy(&key[1], &vmax, 4);
    for (auto& e : point_th_)
        if (e.valid && e.key[0] == key[0] && e.key[1] == key[1]) return e;
    PointThresholds& e = point_th_[point_th_next_];
    point_th_next_ = (point_th_next_ + 1) % (uint32_t)(sizeof point_th_ / sizeof point_th_[0]);
    e.mode = colormap_thresholds(vmin, vmax, e.T);
    e.key[0] = key[0];
    e.key[1] = key[1];
    e.valid = true;
    return e;
}

bool WebGpuRenderer::geom_draw_primitives(uint64_t view_tex, const float* view16, const float* proj16,
                                          const HostGeomDraw* draws, uint32_t count,
                                          uint32_t clear_rgba) {
    if (!gfx_) return geom_fail("the renderer is not initialised");
    auto it = tex_.find(view_tex);
    if (it == tex_.end()) return geom_fail("unknown render target");
    Tex& t = it->second;
    if (!view16 || !proj16) return geom_fail("null camera matrices");
    if (count > 0 && !draws) return geom_fail("null draw array");

    gfx::RenderTarget target;
    target.color = t.color;
    target.depth = t.depth;
    target.clear_rgba = clear_rgba;
    target.clear = true;
    gfx::DrawList list(target);
    list.reserve(count, (size_t)count * 2304);   // each 2,160-byte block padded to 256
    float mvp[16];
    mul4(proj16, view16, mvp);
    // Every draw is checked and recorded before anything is submitted, so a refusal leaves the
    // target's pixels as they were; the layer refuses a bad list as a whole too.
    for (uint32_t i = 0; i < count; ++i) {
        const HostGeomDraw& d = draws[i];
        if (d.topology != CALIPER_GEOM_TOPO_POINTS) return geom_fail("only points are drawn in the browser");
        if (d.index_alloc || d.normal_alloc || d.uv_alloc || d.texture || d.instance_alloc)
            return geom_fail("indices, normals, UVs, textures and instances are not drawn in the browser");
        auto pa = allocs_.find(d.pos_alloc);
        if (pa == allocs_.end()) return geom_fail("unknown position allocation");
        if (d.vertex_count == 0 || d.vertex_count > UINT32_MAX || d.pos_offset % 4 ||
            d.pos_offset + d.vertex_count * 12 > pa->second.bytes)
            return geom_fail("positions out of the allocation's bounds");
        const bool cmap = d.color_mode == CALIPER_GEOM_COLOR_COLORMAP;
        gfx::VertexBinding attr{zero16_, 0};
        if (cmap) {
            auto aa = allocs_.find(d.attr_alloc);
            if (aa == allocs_.end() || !d.lut256) return geom_fail("colormap draw without attr or LUT");
            if (d.attr_offset % 4 || d.attr_offset + d.vertex_count * 4 > aa->second.bytes)
                return geom_fail("attr out of the allocation's bounds");
            attr = {aa->second.buffer, d.attr_offset};
        } else if (d.color_mode != CALIPER_GEOM_COLOR_FLAT) {
            return geom_fail("color mode other than flat or colormap");
        }
        prog::PointParams pp{};
        std::memcpy(pp.mvp, mvp, sizeof mvp);
        pp.color_mode = cmap ? 1u : 0u;
        pp.flat_rgba = d.flat_rgba;
        pp.vmin = d.vmin;
        pp.vmax = d.vmax;
        pp.size_px = d.size_px > 0.0f ? d.size_px : 1.0f;
        pp.viewport[0] = (float)t.w;
        pp.viewport[1] = (float)t.h;
        pp.att_ref = d.att_ref;
        pp.fade_w[0] = d.fade_w0;
        pp.fade_w[1] = d.fade_w1;
        if (cmap) {
            std::memcpy(pp.lut, d.lut256, sizeof pp.lut);
            // Every value-coloured draw takes map_f32_to_rgba8's exact index (d.exact_colormap
            // is the Metal renderer's switch; this one always does).
            const PointThresholds& th = point_thresholds(d.vmin, d.vmax);
            pp.cmap_mode = th.mode;
            if (th.mode) std::memcpy(pp.th, th.T, sizeof pp.th);
        }
        gfx::DrawRecord r;
        r.pipeline = points_pipeline(t.depth.id != 0, d.depth_flags, cmap);
        if (!r.pipeline) return geom_fail("pipeline creation failed");
        r.groups[0] = points_group_;
        r.dynamic_offsets[0] = list.uniform(&pp, sizeof pp);
        r.vertex[0] = {pa->second.buffer, d.pos_offset};
        r.vertex[1] = attr;
        r.count = 6;
        r.instance_count = (uint32_t)d.vertex_count;
        list.draw(r);
    }
    gfx::Frame f;
    f.render(std::move(list));
    if (!gfx_->submit(std::move(f))) {
        std::fprintf(stderr, "[webgpu] geometry refused: %s\n", gfx_->last_error().c_str());
        return false;
    }
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
