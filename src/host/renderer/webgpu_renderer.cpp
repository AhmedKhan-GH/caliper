// webgpu_renderer.cpp: the browser's WebGPU backend (see
// webgpu_renderer.h). Browser-only: it targets Emscripten's emdawnwebgpu port,
// so the surface is a canvas, presentation happens when the frame callback
// returns to the browser, and every completion is a callback, never a wait.
#include "webgpu_renderer.h"

#include <imgui.h>
#include <imgui_impl_wgpu.h>

#include <emscripten/emscripten.h>

#include <cstdio>
#include <cstring>

namespace caliper_host {

namespace {

// WGSL port of the Metal cmap_f32 kernel (metal_renderer.mm) and of the Vulkan
// shaders/colormap.comp, written by hand. Same index rule as the CPU reference
// map_f32_to_rgba8 (idx = clamp((v - vmin)/(vmax - vmin), 0, 1) * 255 + 0.5,
// truncated; vmax == vmin -> 0; NaN -> 0) and the same packed LUT. WGSL lets
// an implementation assume no NaN reaches float arithmetic, so the NaN test is
// done on the bits rather than with v != v. Push constants do not exist in
// WebGPU, so the parameters are a uniform buffer.
const char* kColormapWgsl = R"wgsl(
struct CmapParams {
    w: u32, h: u32, sx: u32, sy: u32,
    vmin: f32, vmax: f32, pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> lut: array<u32>;
@group(0) @binding(2) var<uniform> p: CmapParams;
@group(0) @binding(3) var dst: texture_storage_2d<rgba8unorm, write>;

fn is_nan_bits(v: f32) -> bool {
    let b = bitcast<u32>(v);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

@compute @workgroup_size(16, 16)
fn cmap_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let v = src[gid.y * p.sy + gid.x * p.sx];
    var idx = 0u;
    if (!is_nan_bits(v)) {
        let denom = p.vmax - p.vmin;
        var t = 0.0;
        if (denom != 0.0) { t = (v - p.vmin) / denom; }
        t = clamp(t, 0.0, 1.0);
        idx = u32(t * 255.0 + 0.5);
    }
    let c = lut[idx];
    let rgba = vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu),
                         f32((c >> 16u) & 0xffu), f32((c >> 24u) & 0xffu)) / 255.0;
    textureStore(dst, vec2<i32>(gid.xy), rgba);
}
)wgsl";

struct CmapParams {   // matches the WGSL struct (32 bytes)
    uint32_t w, h, sx, sy;
    float vmin, vmax;
    uint32_t pad0, pad1;
};
static_assert(sizeof(CmapParams) == 32, "WGSL uniform layout");

WGPUStringView sv(const char* s) { return WGPUStringView{s, WGPU_STRLEN}; }

template <class T>
void release(T*& h, void (*fn)(T*)) {
    if (h) fn(h);
    h = nullptr;
}

void release_tex(WGPUTexture& t) { if (t) { wgpuTextureDestroy(t); wgpuTextureRelease(t); } t = nullptr; }
void release_view(WGPUTextureView& v) { if (v) wgpuTextureViewRelease(v); v = nullptr; }
void release_buf(WGPUBuffer& b) { if (b) { wgpuBufferDestroy(b); wgpuBufferRelease(b); } b = nullptr; }
void release_bg(WGPUBindGroup& g) { if (g) wgpuBindGroupRelease(g); g = nullptr; }

WGPUBuffer make_buffer(WGPUDevice d, uint64_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = (size + 3) & ~uint64_t(3);
    bd.usage = usage;
    return wgpuDeviceCreateBuffer(d, &bd);
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
    return ensure_colormap_pipeline();
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

void WebGpuRenderer::render(int fb_w, int fb_h) {
    if (!surface_ || fb_w <= 0 || fb_h <= 0) return;
    if (fb_w != surf_w_ || fb_h != surf_h_) configure_surface(fb_w, fb_h);
    WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface_, &st);
    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        if (st.texture) wgpuTextureRelease(st.texture);
        return;
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
}

void WebGpuRenderer::on_work_done(double submitted_ms) {
    gpu_ms_sum_ += emscripten_get_now() - submitted_ms;
    ++gpu_n_;
}

void WebGpuRenderer::shutdown() {
    for (auto& kv : tex_) {
        Tex& t = kv.second;
        release_bg(t.bind);
        release_buf(t.src);
        release_buf(t.lut);
        release_buf(t.params);
        release_view(t.view);
        release_tex(t.texture);
    }
    tex_.clear();
    if (imgui_ready_) ImGui_ImplWGPU_Shutdown();
    imgui_ready_ = false;
    release(cmap_layout_, wgpuBindGroupLayoutRelease);
    release(cmap_pipeline_, wgpuComputePipelineRelease);
    if (surface_) {
        wgpuSurfaceUnconfigure(surface_);
        wgpuSurfaceRelease(surface_);
        surface_ = nullptr;
    }
}

// ---- textures ----------------------------------------------------------------
WebGpuRenderer::Tex* WebGpuRenderer::find(uint64_t id) {
    auto it = tex_.find(id);
    if (it != tex_.end()) return &it->second;
    for (auto& kv : tex_)   // an ImTextureID (the view pointer) works too
        if ((uint64_t)(uintptr_t)kv.second.view == id) return &kv.second;
    return nullptr;
}

uint64_t WebGpuRenderer::tex_create_rgba8(int w, int h) {
    if (w <= 0 || h <= 0) return 0;
    WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
    d.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst |
              WGPUTextureUsage_CopySrc | WGPUTextureUsage_StorageBinding;
    d.dimension = WGPUTextureDimension_2D;
    d.size = WGPUExtent3D{(uint32_t)w, (uint32_t)h, 1};
    d.format = WGPUTextureFormat_RGBA8Unorm;
    Tex t;
    t.texture = wgpuDeviceCreateTexture(device_, &d);
    if (!t.texture) return 0;
    t.view = wgpuTextureCreateView(t.texture, nullptr);
    t.w = w;
    t.h = h;
    const uint64_t id = next_id_++;
    tex_.emplace(id, t);
    return id;
}

bool WebGpuRenderer::tex_upload_rgba8(uint64_t id, const void* data, int w, int h) {
    Tex* t = find(id);
    if (!t || !data || w != t->w || h != t->h) return false;
    WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst.texture = t->texture;
    WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    layout.bytesPerRow = (uint32_t)w * 4;
    layout.rowsPerImage = (uint32_t)h;
    const WGPUExtent3D size{(uint32_t)w, (uint32_t)h, 1};
    wgpuQueueWriteTexture(queue_, &dst, data, (size_t)w * (size_t)h * 4, &layout, &size);
    last_path_ = "cpu-staged";
    return true;
}

// The ImGui backend caches one bind group per texture view and never learns
// that a view was released; a later view at the same address would reuse a
// stale bind group. The views never release a texture while drawing it; a
// release that could race a draw must invalidate that cache here.
void WebGpuRenderer::tex_release(uint64_t id) {
    auto it = tex_.find(id);
    if (it == tex_.end()) return;
    Tex& t = it->second;
    release_bg(t.bind);
    release_buf(t.src);
    release_buf(t.lut);
    release_buf(t.params);
    release_view(t.view);
    release_tex(t.texture);
    tex_.erase(it);
}

uint64_t WebGpuRenderer::tex_imtexture_id(uint64_t id) {
    Tex* t = find(id);
    return t ? (uint64_t)(uintptr_t)t->view : 0;
}

// ---- the WGSL colormap -------------------------------------------------------
bool WebGpuRenderer::ensure_colormap_pipeline() {
    if (cmap_pipeline_) return true;
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = sv(kColormapWgsl);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &src.chain;
    md.label = sv("caliper.cmap_f32");
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device_, &md);
    if (!mod) return false;
    WGPUComputePipelineDescriptor pd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pd.label = sv("caliper.cmap_f32");
    pd.compute.module = mod;
    pd.compute.entryPoint = sv("cmap_f32");
    cmap_pipeline_ = wgpuDeviceCreateComputePipeline(device_, &pd);
    wgpuShaderModuleRelease(mod);
    if (!cmap_pipeline_) return false;
    cmap_layout_ = wgpuComputePipelineGetBindGroupLayout(cmap_pipeline_, 0);
    return cmap_layout_ != nullptr;
}

bool WebGpuRenderer::tex_update_mapped_from_host(uint64_t id, const float* data, int w, int h,
                                                 const uint32_t* lut256, float vmin,
                                                 float vmax) {
    Tex* t = find(id);
    if (!t || !data || !lut256 || w != t->w || h != t->h) return false;
    if (!ensure_colormap_pipeline()) return false;
    const uint64_t bytes = (uint64_t)w * (uint64_t)h * 4;
    if (!t->src) {
        t->src = make_buffer(device_, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        t->lut = make_buffer(device_, 1024, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        t->params = make_buffer(device_, sizeof(CmapParams),
                                WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        WGPUBindGroupEntry e[4];
        for (auto& x : e) x = WGPU_BIND_GROUP_ENTRY_INIT;
        e[0].binding = 0; e[0].buffer = t->src;    e[0].size = bytes;
        e[1].binding = 1; e[1].buffer = t->lut;    e[1].size = 1024;
        e[2].binding = 2; e[2].buffer = t->params; e[2].size = sizeof(CmapParams);
        e[3].binding = 3; e[3].textureView = t->view;
        WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bd.layout = cmap_layout_;
        bd.entryCount = 4;
        bd.entries = e;
        t->bind = wgpuDeviceCreateBindGroup(device_, &bd);
        t->lut_valid = false;
    }
    wgpuQueueWriteBuffer(queue_, t->src, 0, data, (size_t)bytes);
    if (!t->lut_valid || std::memcmp(t->lut_cache, lut256, 1024) != 0) {
        std::memcpy(t->lut_cache, lut256, 1024);
        t->lut_valid = true;
        wgpuQueueWriteBuffer(queue_, t->lut, 0, lut256, 1024);
    }
    const CmapParams p{(uint32_t)w, (uint32_t)h, 1u, (uint32_t)w, vmin, vmax, 0u, 0u};
    wgpuQueueWriteBuffer(queue_, t->params, 0, &p, sizeof p);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(cp, cmap_pipeline_);
    wgpuComputePassEncoderSetBindGroup(cp, 0, t->bind, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(cp, ((uint32_t)w + 15) / 16, ((uint32_t)h + 15) / 16, 1);
    wgpuComputePassEncoderEnd(cp);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuComputePassEncoderRelease(cp);
    wgpuCommandEncoderRelease(enc);
    ++dispatches_;
    last_path_ = "webgpu-compute";
    return true;
}

// ---- instrumentation -----------------------------------------------------------
void WebGpuRenderer::readback_async(uint64_t id, std::function<void(std::vector<uint8_t>)> done) {
    Tex* t = find(id);
    if (!t) { done({}); return; }
    const uint32_t w = (uint32_t)t->w, h = (uint32_t)t->h;
    const uint32_t bpr = (w * 4 + 255) & ~255u;   // copies need 256-byte rows
    WGPUBuffer buf = make_buffer(device_, (uint64_t)bpr * h,
                                 WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = t->texture;
    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = buf;
    dst.layout.bytesPerRow = bpr;
    dst.layout.rowsPerImage = h;
    const WGPUExtent3D size{w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    struct Pending {
        WGPUBuffer buf;
        uint32_t w, h, bpr;
        std::function<void(std::vector<uint8_t>)> done;
    };
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        Pending* p = static_cast<Pending*>(u1);
        std::vector<uint8_t> out;
        if (status == WGPUMapAsyncStatus_Success) {
            const uint8_t* m = static_cast<const uint8_t*>(
                wgpuBufferGetConstMappedRange(p->buf, 0, (size_t)p->bpr * p->h));
            if (m) {
                out.resize((size_t)p->w * p->h * 4);
                for (uint32_t y = 0; y < p->h; ++y)
                    std::memcpy(out.data() + (size_t)y * p->w * 4, m + (size_t)y * p->bpr,
                                (size_t)p->w * 4);
            }
            wgpuBufferUnmap(p->buf);
        }
        wgpuBufferDestroy(p->buf);
        wgpuBufferRelease(p->buf);
        p->done(std::move(out));
        delete p;
    };
    cb.userdata1 = new Pending{buf, w, h, bpr, std::move(done)};
    wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, (size_t)bpr * h, cb);
}

}  // namespace caliper_host
