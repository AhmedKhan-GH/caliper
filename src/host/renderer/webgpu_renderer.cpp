// webgpu_renderer.cpp: the browser's WebGPU backend (see
// webgpu_renderer.h). Browser-only: it targets Emscripten's emdawnwebgpu port,
// so the surface is a canvas, presentation happens when the frame callback
// returns to the browser, and every completion is a callback, never a wait.
#include "webgpu_renderer.h"

#include <imgui.h>
#include <imgui_impl_wgpu.h>

#include <emscripten/emscripten.h>

#include <caliper/services/geometry_v1_1.h>   // CALIPER_GEOM_*

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

void WebGpuRenderer::free_tex(Tex& t) {
    release_bg(t.bind);
    release_buf(t.src);
    release_buf(t.lut);
    release_buf(t.params);
    release_view(t.view);
    release_tex(t.texture);
    release_view(t.depth_view);
    release_tex(t.depth);
}

void WebGpuRenderer::shutdown() {
    for (auto& kv : tex_) free_tex(kv.second);
    tex_.clear();
    for (auto& v : retired_views_) release_view(v);
    retired_views_.clear();
    for (auto& kv : allocs_) release_buf(kv.second.buffer);
    allocs_.clear();
    for (auto& a : spare_allocs_) release_buf(a.buffer);
    spare_allocs_.clear();
    for (auto& s : slots_) {
        release_buf(s.params);
        release_buf(s.lut);
    }
    slots_.clear();
    for (auto& p : points_pipelines_) release(p, wgpuRenderPipelineRelease);
    release(points_pipeline_layout_, wgpuPipelineLayoutRelease);
    release(points_layout_, wgpuBindGroupLayoutRelease);
    release(points_module_, wgpuShaderModuleRelease);
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

// The ImGui backend caches one bind group per texture view (keyed by the
// view's address) and is never told that a view went away, so a new view at a
// released one's address would be drawn with the old bind group. A released
// texture is therefore destroyed (its memory goes back at once), but its view
// object is kept until shutdown, so no address is ever reused. A texture is
// released only on a shape change, a data view change or a resize, so the
// kept objects stay few and small.
void WebGpuRenderer::tex_release(uint64_t id) {
    auto it = tex_.find(id);
    if (it == tex_.end()) return;
    Tex& t = it->second;
    retired_views_.push_back(t.view);
    t.view = nullptr;
    free_tex(t);
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

// ---- the geometry path: points as instanced quads ------------------------------
namespace {

// WebGPU has only one-pixel points, so each point is two triangles of
// size_px x size_px pixels around its projected centre (the vertex index picks
// the corner, the instance index the point), the same squares Metal and
// Vulkan draw with point_size. The colour rule is the native one: an attr
// value maps through the LUT with idx = clamp((v - vmin)/(vmax - vmin), 0, 1)
// * 255 + 0.5, NaN and a degenerate range taking index 0 (categorical labels
// arrive as values 0..255 with vmin 0 and vmax 255, so idx is the label).
const char* kPointsWgsl = R"wgsl(
struct Params {
    mvp: mat4x4<f32>,
    pos_base: u32, attr_base: u32, color_mode: u32, flat_rgba: u32,
    vmin: f32, vmax: f32, size_px: f32, pad0: u32,
    viewport: vec2<f32>, pad1: vec2<f32>,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> attr: array<f32>;
@group(0) @binding(3) var<storage, read> lut: array<u32>;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu),
                     f32((c >> 24u) & 0xffu)) / 255.0;
}

fn is_nan_bits(v: f32) -> bool {
    let b = bitcast<u32>(v);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let b = p.pos_base + 3u * iid;
    var c = p.mvp * vec4<f32>(pos[b], pos[b + 1u], pos[b + 2u], 1.0);
    // Half the size in NDC is (size_px / 2) / (viewport / 2); times w in clip space.
    let k = corners[vid] * (p.size_px / p.viewport) * c.w;
    c = vec4<f32>(c.x + k.x, c.y + k.y, c.z, c.w);
    var col = unpack(p.flat_rgba);
    if (p.color_mode == 1u) {
        let v = attr[p.attr_base + iid];
        var t = 0.0;
        if (!is_nan_bits(v) && p.vmax > p.vmin) { t = clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0, 1.0); }
        col = unpack(lut[u32(t * 255.0 + 0.5)]);
    }
    return VOut(c, col);
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> { return i.color; }
)wgsl";

struct PointParams {   // matches the WGSL struct (112 bytes)
    float mvp[16];
    uint32_t pos_base, attr_base, color_mode, flat_rgba;
    float vmin, vmax, size_px;
    uint32_t pad0;
    float viewport[2];
    float pad1[2];
};
static_assert(sizeof(PointParams) == 112, "WGSL uniform layout");

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

WGPUColor unpack_clear(uint32_t c) {
    return WGPUColor{(double)(c & 0xffu) / 255.0, (double)((c >> 8) & 0xffu) / 255.0,
                     (double)((c >> 16) & 0xffu) / 255.0, (double)((c >> 24) & 0xffu) / 255.0};
}

}  // namespace

uint64_t WebGpuRenderer::import_host_allocation(const void* data, uint64_t bytes) {
    if (!data || bytes == 0 || bytes > (1ull << 31)) return 0;
    Alloc a;
    a.bytes = bytes;
    const uint64_t cap = (bytes + 3) & ~uint64_t(3);
    for (size_t i = 0; i < spare_allocs_.size(); ++i) {
        if (((spare_allocs_[i].bytes + 3) & ~uint64_t(3)) == cap) {
            a.buffer = spare_allocs_[i].buffer;
            spare_allocs_.erase(spare_allocs_.begin() + (std::ptrdiff_t)i);
            break;
        }
    }
    if (!a.buffer) a.buffer = make_buffer(device_, bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    if (!a.buffer) return 0;
    if (bytes % 4 == 0) {
        wgpuQueueWriteBuffer(queue_, a.buffer, 0, data, (size_t)bytes);
    } else {   // writeBuffer takes multiples of 4
        std::vector<uint8_t> padded((bytes + 3) & ~uint64_t(3), 0);
        std::memcpy(padded.data(), data, (size_t)bytes);
        wgpuQueueWriteBuffer(queue_, a.buffer, 0, padded.data(), padded.size());
    }
    const uint64_t id = next_id_++;
    allocs_.emplace(id, a);
    return id;
}

void WebGpuRenderer::release_external_allocation(uint64_t id) {
    auto it = allocs_.find(id);
    if (it == allocs_.end()) return;
    // Up to four spare buffers are kept; queue writes into a reused one are
    // ordered after the draws already submitted that read it.
    spare_allocs_.push_back(it->second);
    if (spare_allocs_.size() > 4) {
        release_buf(spare_allocs_.front().buffer);
        spare_allocs_.erase(spare_allocs_.begin());
    }
    allocs_.erase(it);
}

uint64_t WebGpuRenderer::geom_create_view_ex(int w, int h, uint32_t flags) {
    if ((flags & ~CALIPER_GEOM_VIEW_DEPTH) != 0u || w <= 0 || h <= 0) return 0;
    WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
    d.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    d.dimension = WGPUTextureDimension_2D;
    d.size = WGPUExtent3D{(uint32_t)w, (uint32_t)h, 1};
    d.format = WGPUTextureFormat_RGBA8Unorm;
    Tex t;
    t.texture = wgpuDeviceCreateTexture(device_, &d);
    if (!t.texture) return 0;
    t.view = wgpuTextureCreateView(t.texture, nullptr);
    t.w = w;
    t.h = h;
    if (flags & CALIPER_GEOM_VIEW_DEPTH) {
        WGPUTextureDescriptor dd = d;
        dd.usage = WGPUTextureUsage_RenderAttachment;
        dd.format = WGPUTextureFormat_Depth32Float;
        t.depth = wgpuDeviceCreateTexture(device_, &dd);
        if (!t.depth) {
            free_tex(t);
            return 0;
        }
        t.depth_view = wgpuTextureCreateView(t.depth, nullptr);
    }
    const uint64_t id = next_id_++;
    tex_.emplace(id, t);
    // Black until the first draw, as the native targets start.
    const float I[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    geom_draw_primitives(id, I, I, nullptr, 0, 0xff000000u);
    return id;
}

bool WebGpuRenderer::ensure_points_pipelines() {
    if (points_layout_) return true;
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = sv(kPointsWgsl);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &src.chain;
    md.label = sv("caliper.points");
    points_module_ = wgpuDeviceCreateShaderModule(device_, &md);
    if (!points_module_) return false;
    WGPUBindGroupLayoutEntry e[4];
    for (auto& x : e) x = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    for (int i = 0; i < 4; ++i) {
        e[i].binding = (uint32_t)i;
        e[i].visibility = WGPUShaderStage_Vertex;
    }
    e[0].buffer.type = WGPUBufferBindingType_Uniform;
    e[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    e[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    e[3].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    WGPUBindGroupLayoutDescriptor bl = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bl.entryCount = 4;
    bl.entries = e;
    points_layout_ = wgpuDeviceCreateBindGroupLayout(device_, &bl);
    WGPUPipelineLayoutDescriptor pl = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pl.bindGroupLayoutCount = 1;
    pl.bindGroupLayouts = &points_layout_;
    points_pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device_, &pl);
    return points_layout_ && points_pipeline_layout_;
}

// [0]: a target without depth; [1 + flags]: with a depth buffer, test and
// write as the draw's depth flags say (less-equal when testing, as Metal).
WGPURenderPipeline WebGpuRenderer::points_pipeline(bool depth_target, uint32_t depth_flags) {
    const uint32_t f = depth_flags & (CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE);
    const int slot = depth_target ? 1 + (int)f : 0;
    if (points_pipelines_[slot]) return points_pipelines_[slot];
    if (!ensure_points_pipelines()) return nullptr;
    WGPUColorTargetState ct = WGPU_COLOR_TARGET_STATE_INIT;
    ct.format = WGPUTextureFormat_RGBA8Unorm;
    ct.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
    fs.module = points_module_;
    fs.entryPoint = sv("fs");
    fs.targetCount = 1;
    fs.targets = &ct;
    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    ds.format = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = (f & CALIPER_GEOM_DEPTH_WRITE) ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    ds.depthCompare = (f & CALIPER_GEOM_DEPTH_TEST) ? WGPUCompareFunction_LessEqual : WGPUCompareFunction_Always;
    WGPURenderPipelineDescriptor rp = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    rp.label = sv("caliper.points");
    rp.layout = points_pipeline_layout_;
    rp.vertex.module = points_module_;
    rp.vertex.entryPoint = sv("vs");
    rp.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    rp.primitive.cullMode = WGPUCullMode_None;
    rp.depthStencil = depth_target ? &ds : nullptr;
    rp.fragment = &fs;
    points_pipelines_[slot] = wgpuDeviceCreateRenderPipeline(device_, &rp);
    return points_pipelines_[slot];
}

bool WebGpuRenderer::geom_draw_primitives(uint64_t view_tex, const float* view16, const float* proj16,
                                          const HostGeomDraw* draws, uint32_t count,
                                          uint32_t clear_rgba) {
    auto it = tex_.find(view_tex);
    if (it == tex_.end()) return geom_fail("unknown render target");
    Tex& t = it->second;
    if (!view16 || !proj16) return geom_fail("null camera matrices");
    if (count > 0 && !draws) return geom_fail("null draw array");
    if (count > 0 && !ensure_points_pipelines()) return geom_fail("pipeline creation failed");

    // Validate and bind every draw before encoding anything, so a refusal
    // leaves the target's pixels as they were.
    struct Enc {
        WGPURenderPipeline pipeline = nullptr;
        WGPUBindGroup bind = nullptr;
        uint32_t n = 0;
    };
    std::vector<Enc> encs;
    encs.reserve(count);
    if (slots_.size() < count) slots_.resize(count);
    float mvp[16];
    mul4(proj16, view16, mvp);
    auto drop = [&]() {
        for (auto& e : encs) release_bg(e.bind);
    };
    for (uint32_t i = 0; i < count; ++i) {
        const HostGeomDraw& d = draws[i];
        if (d.topology != CALIPER_GEOM_TOPO_POINTS) { drop(); return geom_fail("only points are drawn in the browser"); }
        if (d.index_alloc || d.normal_alloc || d.uv_alloc || d.texture || d.instance_alloc) {
            drop();
            return geom_fail("indices, normals, UVs, textures and instances are not drawn in the browser");
        }
        auto pa = allocs_.find(d.pos_alloc);
        if (pa == allocs_.end()) { drop(); return geom_fail("unknown position allocation"); }
        if (d.vertex_count == 0 || d.vertex_count > UINT32_MAX || d.pos_offset % 4 ||
            d.pos_offset + d.vertex_count * 12 > pa->second.bytes) {
            drop();
            return geom_fail("positions out of the allocation's bounds");
        }
        WGPUBuffer attr = pa->second.buffer;
        uint64_t attr_bytes = pa->second.bytes;
        const bool cmap = d.color_mode == CALIPER_GEOM_COLOR_COLORMAP;
        if (cmap) {
            auto aa = allocs_.find(d.attr_alloc);
            if (aa == allocs_.end() || !d.lut256) { drop(); return geom_fail("colormap draw without attr or LUT"); }
            if (d.attr_offset % 4 || d.attr_offset + d.vertex_count * 4 > aa->second.bytes) {
                drop();
                return geom_fail("attr out of the allocation's bounds");
            }
            attr = aa->second.buffer;
            attr_bytes = aa->second.bytes;
        } else if (d.color_mode != CALIPER_GEOM_COLOR_FLAT) {
            drop();
            return geom_fail("color mode other than flat or colormap");
        }
        DrawSlot& sl = slots_[i];
        if (!sl.params) sl.params = make_buffer(device_, sizeof(PointParams), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        if (!sl.lut) sl.lut = make_buffer(device_, 1024, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        PointParams pp{};
        std::memcpy(pp.mvp, mvp, sizeof mvp);
        pp.pos_base = (uint32_t)(d.pos_offset / 4);
        pp.attr_base = cmap ? (uint32_t)(d.attr_offset / 4) : 0u;
        pp.color_mode = cmap ? 1u : 0u;
        pp.flat_rgba = d.flat_rgba;
        pp.vmin = d.vmin;
        pp.vmax = d.vmax;
        pp.size_px = d.size_px > 0.0f ? d.size_px : 1.0f;
        pp.viewport[0] = (float)t.w;
        pp.viewport[1] = (float)t.h;
        wgpuQueueWriteBuffer(queue_, sl.params, 0, &pp, sizeof pp);
        static const uint32_t kZeroLut[256] = {};
        wgpuQueueWriteBuffer(queue_, sl.lut, 0, cmap ? d.lut256 : kZeroLut, 1024);
        WGPUBindGroupEntry be[4];
        for (auto& x : be) x = WGPU_BIND_GROUP_ENTRY_INIT;
        be[0].binding = 0; be[0].buffer = sl.params; be[0].size = sizeof(PointParams);
        be[1].binding = 1; be[1].buffer = pa->second.buffer; be[1].size = (pa->second.bytes + 3) & ~uint64_t(3);
        be[2].binding = 2; be[2].buffer = attr; be[2].size = (attr_bytes + 3) & ~uint64_t(3);
        be[3].binding = 3; be[3].buffer = sl.lut; be[3].size = 1024;
        WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bd.layout = points_layout_;
        bd.entryCount = 4;
        bd.entries = be;
        Enc e;
        e.pipeline = points_pipeline(t.depth != nullptr, d.depth_flags);
        e.bind = wgpuDeviceCreateBindGroup(device_, &bd);
        e.n = (uint32_t)d.vertex_count;
        if (!e.pipeline || !e.bind) {
            release_bg(e.bind);
            drop();
            return geom_fail("pipeline or bind group creation failed");
        }
        encs.push_back(e);
    }

    WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    ca.view = t.view;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = unpack_clear(clear_rgba);
    WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    da.view = t.depth_view;
    da.depthLoadOp = WGPULoadOp_Clear;
    da.depthStoreOp = WGPUStoreOp_Discard;
    da.depthClearValue = 1.0f;
    WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    rp.depthStencilAttachment = t.depth_view ? &da : nullptr;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
    for (const Enc& e : encs) {
        wgpuRenderPassEncoderSetPipeline(pass, e.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, e.bind, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 6, e.n, 0, 0);
    }
    wgpuRenderPassEncoderEnd(pass);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuRenderPassEncoderRelease(pass);
    wgpuCommandEncoderRelease(enc);
    for (auto& e : encs) release_bg(e.bind);
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
