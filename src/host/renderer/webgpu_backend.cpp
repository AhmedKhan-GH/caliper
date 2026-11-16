// webgpu_backend.cpp: the WebGPU backend of the graphics layer (webgpu_backend.h).
//
// How each hook maps onto webgpu.h. The layer's core (src/host/gfx/core.cpp) has validated every
// call against Caps::limits and its handle tables before a hook runs, so the hooks only translate.
// Every table here is indexed by the handle's slot.
//
//   on_create_buffer          wgpuDeviceCreateBuffer with the usage bits mapped one to one, plus
//                             CopyDst always (writeBuffer needs it). The GPU buffer's capacity is the
//                             size rounded up to a power of two, at least 4 KiB, and a released buffer
//                             of the same capacity and usage is reused from a spare pool of at most
//                             64 MiB and 16 buffers (web-geometry.md, "Uploads and reuse"); a reused
//                             buffer is zeroed over its size by a queue write, as a new one is.
//   on_create_texture         wgpuDeviceCreateTexture (2-D, one mip, one sample) and one default view.
//   on_create_sampler         wgpuDeviceCreateSampler: clamp to edge, linear or nearest, no mipmaps.
//   on_create_program         wgpuDeviceCreateShaderModule from ProgramDesc::wgsl, an explicit
//                             WGPUBindGroupLayout per group from the program's bindings (uniform with
//                             hasDynamicOffset and minBindingSize, read-only and read-write storage, float or uint
//                             textures, filtering samplers, write-only storage textures), empty layouts
//                             for unused groups below the highest, and one WGPUPipelineLayout.
//   on_create_render_pipeline wgpuDeviceCreateRenderPipeline on the program's layout: vertex layouts
//                             one to one (stride 0 is a constant stream, as WebGPU allows), strip
//                             topologies with a u32 strip index format, cull none, the colour target
//                             and, with id_target, a second RG32Uint target; Opaque, Alpha
//                             (src-alpha, one-minus-src-alpha; alpha one, one-minus-src-alpha) and
//                             Additive (one, one) as the native geometry pipelines; Depth32Float with
//                             less-equal when testing and always when only writing; override
//                             constants keyed by their numeric id. Created synchronously: the core
//                             caches pipelines by descriptor, so each is made once. Refused on a
//                             compatibility-mode device when it blends and writes the ID target,
//                             which that level cannot do (one blend state for all targets).
//   on_create_compute_pipeline wgpuDeviceCreateComputePipeline on the program's layout; refused when
//                             the entry's workgroup is above the device's compute limits (128
//                             invocations on a compatibility-mode device), which Limits lacks.
//   on_create_bind_group      wgpuDeviceCreateBindGroup with the program's layout of that group. A
//                             kArena entry binds the arena buffer at offset 0 for the binding's size;
//                             the group is rebuilt (same handle) if the arena buffer has to grow.
//   on_destroy                buffers go to the spare pool or are destroyed; a texture is destroyed
//                             at once and its view kept to the end of the frame (release_retired),
//                             as the sprint 2 renderer does for imgui_impl_wgpu; the rest is released.
//   on_write_buffer           wgpuQueueWriteBuffer. The queue orders it after earlier submits, so
//                             in_use changes nothing here.
//   on_write_texture          wgpuQueueWriteTexture of the whole texture with the caller's row pitch.
//   on_mapped                 nullptr: a browser has no persistent mapping (kCapPersistentMap unset).
//   on_import_buffer          host memory only: a new or spare buffer and one wgpuQueueWriteBuffer,
//                             route copied-from-host with the bytes copied and the reason. A browser
//                             has no MTLBuffer or Win32 import, and the core refuses those by the caps.
//   on_import_texture         refused: no external textures in a browser.
//   on_submit                 one command encoder per submit. The arenas of the frame and of each draw
//                             list are written into the arena buffer with one wgpuQueueWriteBuffer,
//                             each at a 256-byte-aligned base, and the dynamic offsets of kArena
//                             bindings are rebased onto it. Consecutive dispatches share one compute
//                             pass; a copy is wgpuCommandEncoderCopyBufferToTexture; a draw list is one
//                             render pass (clear or load, the ID target cleared to 0, depth to 1.0)
//                             recorded record by record with redundant state skipped. Completion is
//                             wgpuQueueOnSubmittedWorkDone, which calls complete(serial). The upload,
//                             the encoding and the submit run in a validation error scope, so that a
//                             frame WebGPU refuses lands in device_errors() as "submit <serial>: ...".
//   on_wait                   false: a browser cannot block (kCapBlockingWait unset).
//   on_read_texture           wgpuCommandEncoderCopyTextureToBuffer into a MapRead staging buffer
//                             with 256-byte rows, submitted on the queue, then wgpuBufferMapAsync; the
//                             map callback packs the rows tightly and calls complete(serial).
//   on_take_readback          hands over the packed bytes.
//   on_imgui_texture_id       the WGPUTextureView, which imgui_impl_wgpu takes as an ImTextureID.
//
// What WebGPU changes against the header: creation errors of shader modules and pipelines, and the
// refusal of a frame that the core passed, come back asynchronously, through an error scope, so
// those hooks cannot refuse at once and the error is kept in device_errors(); gpu_ms is not reported, because timestamp queries are a feature a
// default device does not have (kCapTimestamps unset).
#include "webgpu_backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace caliper_gfx {

namespace {

WGPUStringView sv(const char* s) { return WGPUStringView{s, WGPU_STRLEN}; }

std::string text(WGPUStringView v) {
    if (!v.data) return {};
    return std::string(v.data, v.length == WGPU_STRLEN ? std::strlen(v.data) : v.length);
}

uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

uint64_t pow2_at_least(uint64_t v) {
    uint64_t c = 1;
    while (c < v) c <<= 1;
    return c;
}

WGPUBufferUsage buffer_usage(uint32_t u) {
    WGPUBufferUsage r = WGPUBufferUsage_CopyDst;
    if (u & kVertex) r |= WGPUBufferUsage_Vertex;
    if (u & kIndex) r |= WGPUBufferUsage_Index;
    if (u & kUniform) r |= WGPUBufferUsage_Uniform;
    if (u & kStorage) r |= WGPUBufferUsage_Storage;
    if (u & kCopySrc) r |= WGPUBufferUsage_CopySrc;
    return r;
}

WGPUTextureUsage texture_usage(uint32_t u) {
    WGPUTextureUsage r = WGPUTextureUsage_None;
    if (u & kSampled) r |= WGPUTextureUsage_TextureBinding;
    if (u & kStorageTex) r |= WGPUTextureUsage_StorageBinding;
    if (u & kRenderTarget) r |= WGPUTextureUsage_RenderAttachment;
    if (u & kCopySrcTex) r |= WGPUTextureUsage_CopySrc;
    if (u & kCopyDstTex) r |= WGPUTextureUsage_CopyDst;
    return r;
}

WGPUTextureFormat texture_format(Format f) {
    switch (f) {
        case Format::RGBA8Unorm: return WGPUTextureFormat_RGBA8Unorm;
        case Format::BGRA8Unorm: return WGPUTextureFormat_BGRA8Unorm;
        case Format::R8Unorm: return WGPUTextureFormat_R8Unorm;
        case Format::R16Uint: return WGPUTextureFormat_R16Uint;
        case Format::R32Uint: return WGPUTextureFormat_R32Uint;
        case Format::RG32Uint: return WGPUTextureFormat_RG32Uint;
        case Format::R32Float: return WGPUTextureFormat_R32Float;
        case Format::Depth32Float: return WGPUTextureFormat_Depth32Float;
    }
    return WGPUTextureFormat_Undefined;
}

uint32_t texel_bytes(Format f) {
    switch (f) {
        case Format::R8Unorm: return 1;
        case Format::R16Uint: return 2;
        case Format::RG32Uint: return 8;
        default: return 4;
    }
}

WGPUVertexFormat vertex_format(VertexFormat f) {
    switch (f) {
        case VertexFormat::F32: return WGPUVertexFormat_Float32;
        case VertexFormat::F32x2: return WGPUVertexFormat_Float32x2;
        case VertexFormat::F32x3: return WGPUVertexFormat_Float32x3;
        case VertexFormat::F32x4: return WGPUVertexFormat_Float32x4;
        case VertexFormat::Unorm8x4: return WGPUVertexFormat_Unorm8x4;
        case VertexFormat::U8: return WGPUVertexFormat_Uint8;
        case VertexFormat::U32: return WGPUVertexFormat_Uint32;
        case VertexFormat::S32: return WGPUVertexFormat_Sint32;
        case VertexFormat::U32x2: return WGPUVertexFormat_Uint32x2;
    }
    return WGPUVertexFormat_Float32;
}

WGPUPrimitiveTopology topology(Topology t) {
    switch (t) {
        case Topology::TriangleList: return WGPUPrimitiveTopology_TriangleList;
        case Topology::TriangleStrip: return WGPUPrimitiveTopology_TriangleStrip;
        case Topology::LineList: return WGPUPrimitiveTopology_LineList;
        case Topology::LineStrip: return WGPUPrimitiveTopology_LineStrip;
        case Topology::PointList: return WGPUPrimitiveTopology_PointList;
    }
    return WGPUPrimitiveTopology_TriangleList;
}

WGPUShaderStage stages(uint8_t s) {
    WGPUShaderStage r = WGPUShaderStage_None;
    if (s & kStageVertex) r |= WGPUShaderStage_Vertex;
    if (s & kStageFragment) r |= WGPUShaderStage_Fragment;
    if (s & kStageCompute) r |= WGPUShaderStage_Compute;
    return r;
}

WGPUColor unpack_clear(uint32_t c) {
    return WGPUColor{double(c & 0xffu) / 255.0, double((c >> 8) & 0xffu) / 255.0,
                     double((c >> 16) & 0xffu) / 255.0, double((c >> 24) & 0xffu) / 255.0};
}

void destroy_buffer(WGPUBuffer& b) {
    if (b) {
        wgpuBufferDestroy(b);
        wgpuBufferRelease(b);
    }
    b = nullptr;
}

// The text Caps::adapter points at (Caps keeps a pointer; one backend per module instance).
std::string& adapter_text() {
    static std::string s;
    return s;
}

}  // namespace

template <class T> T* WebGpuBackend::at(std::vector<T>& v, uint32_t slot) {
    return slot < v.size() ? &v[slot] : nullptr;
}

template <class T> T& WebGpuBackend::grow(std::vector<T>& v, uint32_t slot) {
    if (slot >= v.size()) v.resize(size_t(slot) + 1);
    v[slot] = T{};
    return v[slot];
}

// ---- construction --------------------------------------------------------------------------------

std::unique_ptr<WebGpuBackend> WebGpuBackend::create(WGPUDevice device, WGPUQueue queue, const char* adapter) {
    if (!device || !queue) return nullptr;
    WGPULimits L = WGPU_LIMITS_INIT;
    WGPUCompatibilityModeLimits cl = WGPU_COMPATIBILITY_MODE_LIMITS_INIT;
    L.nextInChain = &cl.chain;
    if (wgpuDeviceGetLimits(device, &L) != WGPUStatus_Success) return nullptr;

    // WebGPU's defaults are the ceiling: the backend asks for nothing more (web-geometry.md,
    // "Default WebGPU limits only"); a compatibility-mode device reports lower values.
    Caps caps;
    Limits& o = caps.limits;
    const Limits d;
    auto lo32 = [](uint32_t a, uint32_t b) { return std::min(a, b); };
    o.max_texture_dimension_2d = lo32(d.max_texture_dimension_2d, L.maxTextureDimension2D);
    o.max_bind_groups = lo32(d.max_bind_groups, L.maxBindGroups);
    o.max_dynamic_uniform_buffers = lo32(d.max_dynamic_uniform_buffers, L.maxDynamicUniformBuffersPerPipelineLayout);
    o.max_storage_buffers_per_stage = lo32(d.max_storage_buffers_per_stage, L.maxStorageBuffersPerShaderStage);
    const uint32_t vs = cl.maxStorageBuffersInVertexStage == WGPU_LIMIT_U32_UNDEFINED
                            ? L.maxStorageBuffersPerShaderStage
                            : cl.maxStorageBuffersInVertexStage;
    o.max_storage_buffers_in_vertex_stage = lo32(d.max_storage_buffers_in_vertex_stage, vs);
    o.max_uniform_buffer_binding_size =
        uint32_t(std::min<uint64_t>(d.max_uniform_buffer_binding_size, L.maxUniformBufferBindingSize));
    o.max_storage_buffer_binding_size = std::min(d.max_storage_buffer_binding_size, L.maxStorageBufferBindingSize);
    o.max_buffer_size = std::min(d.max_buffer_size, L.maxBufferSize);
    o.min_uniform_buffer_offset_alignment = std::max(d.min_uniform_buffer_offset_alignment, L.minUniformBufferOffsetAlignment);
    o.min_storage_buffer_offset_alignment = std::max(d.min_storage_buffer_offset_alignment, L.minStorageBufferOffsetAlignment);
    o.max_vertex_buffers = lo32(d.max_vertex_buffers, L.maxVertexBuffers);
    o.max_vertex_attributes = lo32(d.max_vertex_attributes, L.maxVertexAttributes);
    o.max_vertex_stride = lo32(d.max_vertex_stride, L.maxVertexBufferArrayStride);
    o.max_color_attachments = lo32(d.max_color_attachments, L.maxColorAttachments);
    const bool compat = o.max_storage_buffers_in_vertex_stage == 0;
    caps.bits = o.max_storage_buffers_in_vertex_stage > 0 ? uint64_t(kCapVertexStorage) : 0;
    caps.backend = "webgpu";
    adapter_text() = adapter ? adapter : "";
    caps.adapter = adapter_text().c_str();
    auto b = std::unique_ptr<WebGpuBackend>(new WebGpuBackend(device, queue, caps, compat));
    b->max_invocations_ = L.maxComputeInvocationsPerWorkgroup;
    b->max_group_[0] = L.maxComputeWorkgroupSizeX;
    b->max_group_[1] = L.maxComputeWorkgroupSizeY;
    b->max_group_[2] = L.maxComputeWorkgroupSizeZ;
    return b;
}

WebGpuBackend::WebGpuBackend(WGPUDevice device, WGPUQueue queue, const Caps& caps, bool compat)
    : Backend(caps), device_(device), queue_(queue), compat_(compat) {
    wgpuDeviceAddRef(device_);
    wgpuQueueAddRef(queue_);
}

WebGpuBackend::~WebGpuBackend() {
    *alive_ = false;
    for (auto& g : groups_) {
        if (g.group) wgpuBindGroupRelease(g.group);
        if (g.layout) wgpuBindGroupLayoutRelease(g.layout);
    }
    for (auto& p : pipelines_) {
        if (p.render) wgpuRenderPipelineRelease(p.render);
        if (p.compute) wgpuComputePipelineRelease(p.compute);
    }
    for (auto& p : programs_) {
        for (auto& l : p.groups)
            if (l) wgpuBindGroupLayoutRelease(l);
        if (p.layout) wgpuPipelineLayoutRelease(p.layout);
        if (p.module) wgpuShaderModuleRelease(p.module);
    }
    for (auto& s : samplers_)
        if (s) wgpuSamplerRelease(s);
    for (auto& t : textures_) {
        if (t.view) wgpuTextureViewRelease(t.view);
        if (t.texture) {
            wgpuTextureDestroy(t.texture);
            wgpuTextureRelease(t.texture);
        }
    }
    for (auto& b : buffers_) destroy_buffer(b.buffer);
    for (auto& s : spare_) destroy_buffer(s.buffer);
    for (auto& v : retired_views_) wgpuTextureViewRelease(v);
    destroy_buffer(arena_);
    // A readback in flight keeps its staging buffer until its map callback runs.
    wgpuQueueRelease(queue_);
    wgpuDeviceRelease(device_);
}

// ---- errors that WebGPU reports later ------------------------------------------------------------

void WebGpuBackend::push_scope() { wgpuDevicePushErrorScope(device_, WGPUErrorFilter_Validation); }

void WebGpuBackend::pop_scope(const char* what) {
    struct Ctx {
        std::shared_ptr<std::vector<std::string>> errors;
        std::string what;
    };
    WGPUPopErrorScopeCallbackInfo cb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* u1, void*) {
        Ctx* c = static_cast<Ctx*>(u1);
        if (status == WGPUPopErrorScopeStatus_Success && type != WGPUErrorType_NoError) {
            const std::string m = c->what + ": " + text(message);
            std::fprintf(stderr, "[webgpu gfx] %s\n", m.c_str());
            if (c->errors->size() < 64) c->errors->push_back(m);
        }
        delete c;
    };
    cb.userdata1 = new Ctx{errors_, what ? what : ""};
    wgpuDevicePopErrorScope(device_, cb);
}

// ---- buffers ---------------------------------------------------------------------------------------

WGPUBuffer WebGpuBackend::acquire_buffer(uint64_t size, WGPUBufferUsage usage, uint64_t* capacity,
                                         const char* label) {
    uint64_t cap = pow2_at_least(std::max<uint64_t>(size, kMinCapacity));
    cap = std::min(cap, std::max(caps().limits.max_buffer_size, (size + 3) & ~uint64_t(3)));
    for (size_t i = spare_.size(); i-- > 0;) {
        if (spare_[i].capacity == cap && spare_[i].usage == usage) {
            WGPUBuffer b = spare_[i].buffer;
            spare_bytes_ -= cap;
            spare_.erase(spare_.begin() + std::ptrdiff_t(i));
            *capacity = cap;
            return b;
        }
    }
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.label = sv(label);
    bd.size = cap;
    bd.usage = usage;
    WGPUBuffer b = wgpuDeviceCreateBuffer(device_, &bd);
    *capacity = b ? cap : 0;
    return b;
}

// Called once the buffer's last submit completed: its memory may be written again at once.
void WebGpuBackend::retire_buffer(Buf& b) {
    if (!b.buffer) return;
    if (b.capacity <= kSparePoolBytes) {
        spare_.push_back({b.buffer, b.capacity, b.usage});
        spare_bytes_ += b.capacity;
        while (!spare_.empty() && (spare_bytes_ > kSparePoolBytes || spare_.size() > kSparePoolCount)) {
            spare_bytes_ -= spare_.front().capacity;
            destroy_buffer(spare_.front().buffer);
            spare_.erase(spare_.begin());
        }
    } else {
        destroy_buffer(b.buffer);
    }
    b = Buf{};
}

size_t WebGpuBackend::live_buffer_count() const {
    size_t n = 0;
    for (const auto& b : buffers_) n += b.buffer != nullptr;
    return n;
}

bool WebGpuBackend::on_create_buffer(uint32_t slot, const BufferDesc& d, std::string* error) {
    Buf& b = grow(buffers_, slot);
    b.usage = buffer_usage(d.usage);
    const size_t spares = spare_.size();
    b.buffer = acquire_buffer(d.size, b.usage, &b.capacity, d.label);
    if (!b.buffer) {
        *error = "wgpuDeviceCreateBuffer failed";
        return false;
    }
    b.size = d.size;
    if (spare_.size() < spares && d.size > 0) {   // reused: zero it, as a new buffer is
        static const std::vector<uint8_t> zeros(1 << 16, 0);
        for (uint64_t off = 0; off < d.size; off += zeros.size())
            wgpuQueueWriteBuffer(queue_, b.buffer, off, zeros.data(),
                                 size_t(std::min<uint64_t>(zeros.size(), d.size - off)));
    }
    return true;
}

void WebGpuBackend::on_write_buffer(uint32_t slot, uint64_t offset, const void* data, uint64_t size, bool) {
    Buf* b = at(buffers_, slot);
    if (b && b->buffer) wgpuQueueWriteBuffer(queue_, b->buffer, offset, data, size_t(size));
}

void* WebGpuBackend::on_mapped(uint32_t) { return nullptr; }

bool WebGpuBackend::on_import_buffer(uint32_t slot, const ExternalBuffer& ext, ImportResult* r) {
    if (ext.kind != ExternalKind::HostMemory) {
        r->route = Route::Refused;
        r->reason = "a browser can import only host memory, by a copy";
        return false;
    }
    Buf& b = grow(buffers_, slot);
    b.usage = buffer_usage(ext.usage);
    b.buffer = acquire_buffer(ext.size, b.usage, &b.capacity, ext.label);
    if (!b.buffer) {
        r->route = Route::Refused;
        r->reason = "wgpuDeviceCreateBuffer failed";
        return false;
    }
    b.size = ext.size;
    wgpuQueueWriteBuffer(queue_, b.buffer, 0, ext.handle, size_t(ext.size));
    r->route = Route::CopiedFromHost;
    r->bytes_copied = ext.size;
    r->reason = "WebGPU in a browser cannot wrap host memory, so the bytes were copied into the GPU process";
    return true;
}

bool WebGpuBackend::on_import_texture(uint32_t, const ExternalTexture&, ImportResult* r) {
    r->route = Route::Refused;
    r->reason = "a browser has no external textures";
    return false;
}

// ---- textures and samplers -------------------------------------------------------------------------

bool WebGpuBackend::on_create_texture(uint32_t slot, const TextureDesc& d, std::string* error) {
    Tex& t = grow(textures_, slot);
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.label = sv(d.label);
    td.usage = texture_usage(d.usage);
    td.dimension = WGPUTextureDimension_2D;
    td.size = WGPUExtent3D{d.width, d.height, 1};
    td.format = texture_format(d.format);
    t.texture = wgpuDeviceCreateTexture(device_, &td);
    if (!t.texture) {
        *error = "wgpuDeviceCreateTexture failed";
        return false;
    }
    t.view = wgpuTextureCreateView(t.texture, nullptr);
    t.w = d.width;
    t.h = d.height;
    t.format = d.format;
    return true;
}

void WebGpuBackend::on_write_texture(uint32_t slot, const void* data, uint32_t row_pitch, bool) {
    Tex* t = at(textures_, slot);
    if (!t || !t->texture) return;
    WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst.texture = t->texture;
    WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    layout.bytesPerRow = row_pitch;
    layout.rowsPerImage = t->h;
    const WGPUExtent3D size{t->w, t->h, 1};
    const size_t bytes = size_t(row_pitch) * (t->h - 1) + size_t(t->w) * texel_bytes(t->format);
    wgpuQueueWriteTexture(queue_, &dst, data, bytes, &layout, &size);
}

bool WebGpuBackend::on_create_sampler(uint32_t slot, const SamplerDesc& d, std::string* error) {
    WGPUSampler& s = grow(samplers_, slot);
    WGPUSamplerDescriptor sd = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sd.addressModeU = sd.addressModeV = sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = sd.minFilter = d.linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    s = wgpuDeviceCreateSampler(device_, &sd);
    if (!s) *error = "wgpuDeviceCreateSampler failed";
    return s != nullptr;
}

uint64_t WebGpuBackend::on_imgui_texture_id(uint32_t slot) {
    Tex* t = at(textures_, slot);
    return t ? uint64_t(uintptr_t(t->view)) : 0;
}

void WebGpuBackend::release_retired() {
    for (auto& v : retired_views_) wgpuTextureViewRelease(v);
    retired_views_.clear();
}

// ---- programs and pipelines ------------------------------------------------------------------------

bool WebGpuBackend::on_create_program(uint32_t slot, const ProgramDesc& d, std::string* error) {
    Prog& p = grow(programs_, slot);
    if (!d.wgsl) {
        *error = "the program has no WGSL source, which the WebGPU backend needs";
        return false;
    }
    p.bindings.assign(d.bindings, d.bindings + d.binding_count);
    p.entries.assign(d.entries, d.entries + d.entry_count);
    std::sort(p.bindings.begin(), p.bindings.end(), [](const BindingDesc& a, const BindingDesc& b) {
        return a.group != b.group ? a.group < b.group : a.binding < b.binding;
    });
    push_scope();
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = sv(d.wgsl);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &src.chain;
    md.label = sv(d.label);
    p.module = wgpuDeviceCreateShaderModule(device_, &md);

    uint32_t top = 0;   // groups 0 .. top - 1 make up the pipeline layout
    for (const BindingDesc& b : p.bindings) {
        top = std::max<uint32_t>(top, uint32_t(b.group) + 1);
        p.group_mask |= uint8_t(1u << b.group);
    }
    bool ok = p.module != nullptr;
    for (uint32_t g = 0; g < top && ok; ++g) {
        WGPUBindGroupLayoutEntry e[kMaxBindingsPerGroup];
        uint32_t n = 0;
        for (const BindingDesc& b : p.bindings) {
            if (b.group != g || n >= kMaxBindingsPerGroup) continue;
            WGPUBindGroupLayoutEntry& x = e[n++];
            x = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            x.binding = b.binding;
            x.visibility = stages(b.stages);
            switch (b.type) {
                case BindingType::Uniform:
                    x.buffer.type = WGPUBufferBindingType_Uniform;
                    x.buffer.hasDynamicOffset = b.dynamic;
                    x.buffer.minBindingSize = b.size;
                    break;
                case BindingType::Storage:
                    x.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                    x.buffer.minBindingSize = b.size;
                    break;
                case BindingType::StorageReadWrite:
                    x.buffer.type = WGPUBufferBindingType_Storage;
                    x.buffer.minBindingSize = b.size;
                    break;
                case BindingType::Texture:
                    x.texture.sampleType = WGPUTextureSampleType_Float;
                    x.texture.viewDimension = WGPUTextureViewDimension_2D;
                    break;
                case BindingType::UintTexture:
                    x.texture.sampleType = WGPUTextureSampleType_Uint;
                    x.texture.viewDimension = WGPUTextureViewDimension_2D;
                    break;
                case BindingType::Sampler:
                    x.sampler.type = WGPUSamplerBindingType_Filtering;
                    break;
                case BindingType::StorageTexture:
                    x.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
                    x.storageTexture.format = texture_format(b.storage_format);
                    x.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
                    break;
                default:
                    *error = "a binding of a type the WebGPU backend does not know";
                    ok = false;
                    break;
            }
        }
        if (!ok) break;
        WGPUBindGroupLayoutDescriptor bl = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bl.label = sv(d.label);
        bl.entryCount = n;
        bl.entries = e;
        p.groups[g] = wgpuDeviceCreateBindGroupLayout(device_, &bl);
        ok = p.groups[g] != nullptr;
    }
    if (ok) {
        WGPUPipelineLayoutDescriptor pl = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pl.label = sv(d.label);
        pl.bindGroupLayoutCount = top;
        pl.bindGroupLayouts = p.groups;
        p.layout = wgpuDeviceCreatePipelineLayout(device_, &pl);
        ok = p.layout != nullptr;
    }
    pop_scope((std::string("program ") + (d.label ? d.label : "")).c_str());
    if (!ok && error->empty()) *error = "the shader module or its layouts could not be created";
    return ok;
}

bool WebGpuBackend::on_create_render_pipeline(uint32_t slot, const RenderPipelineDesc& d, std::string* error) {
    Pipe& pp = grow(pipelines_, slot);
    Prog* pr = at(programs_, handle_slot(d.program.id));
    if (!pr || !pr->layout) {
        *error = "the program is gone";
        return false;
    }
    if (!d.vertex_entry || !d.fragment_entry) {
        *error = "a render pipeline needs a vertex and a fragment entry";
        return false;
    }
    // A compatibility-mode device has one blend state for all colour targets, and an RG32Uint
    // target cannot blend, so a pipeline that writes the ID target cannot blend there.
    if (compat_ && d.id_target && d.blend != Blend::Opaque) {
        *error = "a compatibility-mode device cannot blend a pipeline that writes the ID target";
        return false;
    }
    WGPUVertexAttribute attrs[kMaxVertexBuffers][kMaxAttributesPerBuffer];
    WGPUVertexBufferLayout vbl[kMaxVertexBuffers];
    for (uint32_t i = 0; i < d.vertex_layout_count; ++i) {
        const VertexLayout& l = d.vertex_layouts[i];
        vbl[i] = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
        vbl[i].stepMode = l.step == Step::Instance ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
        vbl[i].arrayStride = l.stride;
        vbl[i].attributeCount = l.attribute_count;
        vbl[i].attributes = attrs[i];
        for (uint32_t a = 0; a < l.attribute_count; ++a) {
            attrs[i][a] = WGPU_VERTEX_ATTRIBUTE_INIT;
            attrs[i][a].format = vertex_format(l.attributes[a].format);
            attrs[i][a].offset = l.attributes[a].offset;
            attrs[i][a].shaderLocation = l.attributes[a].location;
        }
    }
    char keys[kMaxConstants][8];
    WGPUConstantEntry consts[kMaxConstants];
    for (uint32_t i = 0; i < d.constant_count; ++i) {
        std::snprintf(keys[i], sizeof keys[i], "%u", unsigned(d.constants[i].id));
        consts[i] = WGPU_CONSTANT_ENTRY_INIT;
        consts[i].key = sv(keys[i]);
        consts[i].value = d.constants[i].value;
    }
    WGPUBlendState blend;
    blend.color.operation = blend.alpha.operation = WGPUBlendOperation_Add;
    if (d.blend == Blend::Alpha) {
        blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
        blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend.alpha.srcFactor = WGPUBlendFactor_One;
        blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    } else {
        blend.color.srcFactor = blend.alpha.srcFactor = WGPUBlendFactor_One;
        blend.color.dstFactor = blend.alpha.dstFactor = WGPUBlendFactor_One;
    }
    WGPUColorTargetState ct[2] = {WGPU_COLOR_TARGET_STATE_INIT, WGPU_COLOR_TARGET_STATE_INIT};
    ct[0].format = texture_format(d.color_format);
    ct[0].blend = d.blend == Blend::Opaque ? nullptr : &blend;
    ct[0].writeMask = WGPUColorWriteMask_All;
    ct[1].format = WGPUTextureFormat_RG32Uint;
    ct[1].writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
    fs.module = pr->module;
    fs.entryPoint = sv(d.fragment_entry);
    fs.constantCount = d.constant_count;
    fs.constants = consts;
    fs.targetCount = d.id_target ? 2 : 1;
    fs.targets = ct;
    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    ds.format = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = d.depth_write ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    ds.depthCompare = d.depth_test ? WGPUCompareFunction_LessEqual : WGPUCompareFunction_Always;
    WGPURenderPipelineDescriptor rp = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    rp.label = sv(d.label);
    rp.layout = pr->layout;
    rp.vertex.module = pr->module;
    rp.vertex.entryPoint = sv(d.vertex_entry);
    rp.vertex.constantCount = d.constant_count;
    rp.vertex.constants = consts;
    rp.vertex.bufferCount = d.vertex_layout_count;
    rp.vertex.buffers = vbl;
    rp.primitive.topology = topology(d.topology);
    if (d.topology == Topology::TriangleStrip || d.topology == Topology::LineStrip)
        rp.primitive.stripIndexFormat = WGPUIndexFormat_Uint32;
    rp.primitive.cullMode = WGPUCullMode_None;
    rp.depthStencil = d.depth ? &ds : nullptr;
    rp.fragment = &fs;
    push_scope();
    pp.render = wgpuDeviceCreateRenderPipeline(device_, &rp);
    pop_scope((std::string("render pipeline ") + (d.label ? d.label : "")).c_str());
    pp.group_mask = pr->group_mask;
    pp.vertex_layout_count = d.vertex_layout_count;
    if (!pp.render) *error = "wgpuDeviceCreateRenderPipeline failed";
    return pp.render != nullptr;
}

bool WebGpuBackend::on_create_compute_pipeline(uint32_t slot, const ComputePipelineDesc& d, std::string* error) {
    Pipe& pp = grow(pipelines_, slot);
    Prog* pr = at(programs_, handle_slot(d.program.id));
    if (!pr || !pr->layout || !d.entry) {
        *error = "the program is gone or the entry is missing";
        return false;
    }
    for (const EntryDesc& e : pr->entries)
        if (e.name && std::strcmp(e.name, d.entry) == 0) {
            const uint32_t x = e.workgroup_size[0], y = e.workgroup_size[1], z = e.workgroup_size[2];
            if (x > max_group_[0] || y > max_group_[1] || z > max_group_[2] || x * y * z > max_invocations_) {
                char b[160];
                std::snprintf(b, sizeof b, "the workgroup %u x %u x %u is above the device's limits (%u invocations)",
                              x, y, z, max_invocations_);
                *error = b;
                return false;
            }
        }
    WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    cd.label = sv(d.label);
    cd.layout = pr->layout;
    cd.compute.module = pr->module;
    cd.compute.entryPoint = sv(d.entry);
    push_scope();
    pp.compute = wgpuDeviceCreateComputePipeline(device_, &cd);
    pop_scope((std::string("compute pipeline ") + (d.label ? d.label : "")).c_str());
    pp.group_mask = pr->group_mask;
    if (!pp.compute) *error = "wgpuDeviceCreateComputePipeline failed";
    return pp.compute != nullptr;
}

// ---- bind groups and the arena ---------------------------------------------------------------------

bool WebGpuBackend::build_group(Group& g, std::string* error) {
    Prog* pr = at(programs_, g.program_slot);
    WGPUBindGroupEntry e[kMaxBindingsPerGroup];
    for (uint32_t i = 0; i < g.desc.entry_count; ++i) {
        const BindEntry& be = g.desc.entries[i];
        e[i] = WGPU_BIND_GROUP_ENTRY_INIT;
        e[i].binding = be.binding;
        uint64_t declared = 0;   // a uniform's block; a storage binding of size 0 takes the rest
        if (pr)
            for (const BindingDesc& b : pr->bindings)
                if (b.group == g.index && b.binding == be.binding && b.type == BindingType::Uniform) declared = b.size;
        if (be.buffer == kArena) {
            if (!ensure_arena(kUniformAlign)) return false;
            e[i].buffer = arena_;
            e[i].offset = 0;
            e[i].size = be.size ? be.size : declared;
        } else if (be.buffer) {
            Buf* b = alive(be.buffer.id) ? at(buffers_, handle_slot(be.buffer.id)) : nullptr;
            if (!b || !b->buffer) {
                if (error) *error = "a bind group entry's buffer is gone";
                return false;
            }
            e[i].buffer = b->buffer;
            e[i].offset = be.offset;
            e[i].size = be.size ? be.size : declared ? declared : b->size - be.offset;
        } else if (be.texture) {
            Tex* t = alive(be.texture.id) ? at(textures_, handle_slot(be.texture.id)) : nullptr;
            if (!t || !t->view) {
                if (error) *error = "a bind group entry's texture is gone";
                return false;
            }
            e[i].textureView = t->view;
        } else if (be.sampler) {
            WGPUSampler* s = alive(be.sampler.id) ? at(samplers_, handle_slot(be.sampler.id)) : nullptr;
            if (!s || !*s) {
                if (error) *error = "a bind group entry's sampler is gone";
                return false;
            }
            e[i].sampler = *s;
        }
    }
    WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bd.layout = g.layout;
    bd.entryCount = g.desc.entry_count;
    bd.entries = e;
    WGPUBindGroup made = wgpuDeviceCreateBindGroup(device_, &bd);
    if (!made) {
        if (error) *error = "wgpuDeviceCreateBindGroup failed";
        return false;
    }
    if (g.group) wgpuBindGroupRelease(g.group);
    g.group = made;
    return true;
}

bool WebGpuBackend::on_create_bind_group(uint32_t slot, const BindGroupDesc& d, std::string* error) {
    Group& g = grow(groups_, slot);
    Prog* pr = at(programs_, handle_slot(d.program.id));
    if (!pr || d.group >= kMaxBindGroups || !pr->groups[d.group]) {
        *error = "the program or its group layout is gone";
        return false;
    }
    g.program_slot = handle_slot(d.program.id);
    g.index = d.group;
    g.layout = pr->groups[d.group];
    wgpuBindGroupLayoutAddRef(g.layout);   // a rebuild may come after the program's release
    g.desc = d;
    g.desc.label = nullptr;
    for (uint32_t i = 0; i < d.entry_count; ++i) g.uses_arena |= d.entries[i].buffer == kArena;
    for (const BindingDesc& b : pr->bindings) {   // sorted by binding within the group
        if (b.group != d.group || !b.dynamic) continue;
        bool arena = false;
        for (uint32_t i = 0; i < d.entry_count; ++i)
            if (d.entries[i].binding == b.binding) arena = d.entries[i].buffer == kArena;
        if (g.dynamic_count < kMaxBindingsPerGroup) g.dynamic_arena[g.dynamic_count++] = arena;
    }
    return build_group(g, error);
}

bool WebGpuBackend::ensure_arena(uint64_t bytes) {
    if (arena_ && arena_cap_ >= bytes) return true;
    const uint64_t cap = std::min(pow2_at_least(std::max<uint64_t>(bytes, 64 * 1024)), caps().limits.max_buffer_size);
    if (cap < bytes) return false;
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.label = sv("caliper.gfx.arena");
    bd.size = cap;
    bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    WGPUBuffer b = wgpuDeviceCreateBuffer(device_, &bd);
    if (!b) return false;
    // Submits in flight still read the old buffer; releasing it only drops this reference.
    if (arena_) wgpuBufferRelease(arena_);
    const bool rebuild = arena_ != nullptr;
    arena_ = b;
    arena_cap_ = cap;
    if (rebuild)
        for (Group& g : groups_)
            if (g.group && g.uses_arena) build_group(g, nullptr);
    return true;
}

// ---- destruction -------------------------------------------------------------------------------------

void WebGpuBackend::on_destroy(uint64_t handle) {
    const uint32_t slot = handle_slot(handle);
    switch (handle_kind(handle)) {
        case HandleKind::Buffer:
            if (Buf* b = at(buffers_, slot)) retire_buffer(*b);
            break;
        case HandleKind::Texture:
            if (Tex* t = at(textures_, slot)) {
                if (t->view) retired_views_.push_back(t->view);
                if (t->texture) {
                    wgpuTextureDestroy(t->texture);
                    wgpuTextureRelease(t->texture);
                }
                *t = Tex{};
            }
            break;
        case HandleKind::Sampler:
            if (WGPUSampler* s = at(samplers_, slot)) {
                if (*s) wgpuSamplerRelease(*s);
                *s = nullptr;
            }
            break;
        case HandleKind::Program:
            if (Prog* p = at(programs_, slot)) {
                for (auto& l : p->groups)
                    if (l) wgpuBindGroupLayoutRelease(l);
                if (p->layout) wgpuPipelineLayoutRelease(p->layout);
                if (p->module) wgpuShaderModuleRelease(p->module);
                *p = Prog{};
            }
            break;
        case HandleKind::Pipeline:
            if (Pipe* p = at(pipelines_, slot)) {
                if (p->render) wgpuRenderPipelineRelease(p->render);
                if (p->compute) wgpuComputePipelineRelease(p->compute);
                *p = Pipe{};
            }
            break;
        case HandleKind::BindGroup:
            if (Group* g = at(groups_, slot)) {
                if (g->group) wgpuBindGroupRelease(g->group);
                if (g->layout) wgpuBindGroupLayoutRelease(g->layout);
                *g = Group{};
            }
            break;
        case HandleKind::Readback:
            // A map in flight holds its own reference and finishes on its own.
            if (slot < readbacks_.size()) readbacks_[slot].reset();
            break;
        case HandleKind::None:
            break;
    }
}

// ---- submit ------------------------------------------------------------------------------------------

bool WebGpuBackend::on_submit(Serial s, const Frame& frame, Stats*, std::string* error) {
    // Where each pass's arena goes in the arena buffer: the frame's first, then each draw list's.
    const View<Pass> passes = frame.passes();
    std::vector<uint64_t> base(passes.size, 0);
    uint64_t total = round_up(frame.arena().size, kUniformAlign);
    for (size_t i = 0; i < passes.size; ++i)
        if (const auto* r = std::get_if<RenderPass>(&passes[i])) {
            base[i] = total;
            total += round_up(r->list.arena().size, kUniformAlign);
        }

    // Everything the frame names must exist here before anything is encoded (decision 13).
    auto pipe_ok = [&](Pipeline p, bool compute) {
        Pipe* pp = at(pipelines_, handle_slot(p.id));
        return pp && (compute ? pp->compute != nullptr : pp->render != nullptr);
    };
    auto groups_ok = [&](Pipeline p, const BindGroup* gs) {
        const Pipe* pp = at(pipelines_, handle_slot(p.id));
        for (uint32_t g = 0; g < kMaxBindGroups; ++g) {
            if (!(pp->group_mask & (1u << g))) continue;
            const Group* gr = at(groups_, handle_slot(gs[g].id));
            if (!gr || !gr->group) return false;
        }
        return true;
    };
    auto buf_ok = [&](Buffer b) {
        const Buf* x = at(buffers_, handle_slot(b.id));
        return x && x->buffer;
    };
    auto tex_ok = [&](Texture t) {
        const Tex* x = at(textures_, handle_slot(t.id));
        return x && x->view;
    };
    for (const Pass& pass : passes) {
        if (const auto* w = std::get_if<WaitPass>(&pass)) {
            if (w->sync.kind != ExternalSync::Drained) {
                *error = "a browser cannot wait on a producer's GPU event";
                return false;
            }
        } else if (const auto* d = std::get_if<DispatchPass>(&pass)) {
            if (!pipe_ok(d->pipeline, true) || !groups_ok(d->pipeline, d->groups)) {
                *error = "a dispatch names a pipeline or bind group the backend does not hold";
                return false;
            }
        } else if (const auto* c = std::get_if<CopyPass>(&pass)) {
            if (!buf_ok(c->src) || !tex_ok(c->dst)) {
                *error = "a copy names a buffer or texture the backend does not hold";
                return false;
            }
        } else if (const auto* r = std::get_if<RenderPass>(&pass)) {
            const RenderTarget& tg = r->list.target();
            if (!tex_ok(tg.color) || (tg.id && !tex_ok(tg.id)) || (tg.depth && !tex_ok(tg.depth))) {
                *error = "a draw list's target is a texture the backend does not hold";
                return false;
            }
            for (const DrawRecord& rec : r->list.records()) {
                if (!pipe_ok(rec.pipeline, false) || !groups_ok(rec.pipeline, rec.groups)) {
                    *error = "a draw names a pipeline or bind group the backend does not hold";
                    return false;
                }
                const Pipe* pp = at(pipelines_, handle_slot(rec.pipeline.id));
                for (uint32_t i = 0; i < pp->vertex_layout_count; ++i)
                    if (!buf_ok(rec.vertex[i].buffer)) {
                        *error = "a draw names a vertex buffer the backend does not hold";
                        return false;
                    }
                if (rec.index_buffer && !buf_ok(rec.index_buffer)) {
                    *error = "a draw names an index buffer the backend does not hold";
                    return false;
                }
            }
        }
    }
    if (total > 0 && !ensure_arena(total)) {
        *error = "the frame's uniform blocks do not fit one buffer";
        return false;
    }

    // Everything from here to the submit is in one error scope: a frame WebGPU refuses is reported.
    push_scope();

    // The arenas, uploaded once.
    if (total > 0) {
        arena_stage_.assign(size_t(total), 0);
        if (frame.arena().size) std::memcpy(arena_stage_.data(), frame.arena().data, frame.arena().size);
        for (size_t i = 0; i < passes.size; ++i)
            if (const auto* r = std::get_if<RenderPass>(&passes[i]))
                if (r->list.arena().size)
                    std::memcpy(arena_stage_.data() + base[i], r->list.arena().data, r->list.arena().size);
        wgpuQueueWriteBuffer(queue_, arena_, 0, arena_stage_.data(), arena_stage_.size());
    }

    // The dynamic offsets of the groups a pipeline declares, in (group, binding) order (decision 7),
    // with those of kArena bindings rebased by the pass's base.
    struct Bound {
        WGPUBindGroup group = nullptr;
        uint32_t count = 0;
        uint32_t offsets[kMaxDynamicOffsets] = {};
    };
    auto resolve = [&](Pipeline p, const BindGroup* gs, const uint32_t* dyn, uint64_t arena_base, Bound out[kMaxBindGroups]) {
        const Pipe* pp = at(pipelines_, handle_slot(p.id));
        uint32_t k = 0;
        for (uint32_t g = 0; g < kMaxBindGroups; ++g) {
            out[g] = Bound{};
            if (!(pp->group_mask & (1u << g))) continue;
            const Group* gr = at(groups_, handle_slot(gs[g].id));
            out[g].group = gr->group;
            for (uint32_t j = 0; j < gr->dynamic_count && k < kMaxDynamicOffsets; ++j, ++k)
                out[g].offsets[out[g].count++] = uint32_t(dyn[k] + (gr->dynamic_arena[j] ? arena_base : 0));
        }
    };

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPUComputePassEncoder cp = nullptr;
    auto end_compute = [&]() {
        if (!cp) return;
        wgpuComputePassEncoderEnd(cp);
        wgpuComputePassEncoderRelease(cp);
        cp = nullptr;
    };
    for (size_t i = 0; i < passes.size; ++i) {
        const Pass& pass = passes[i];
        if (const auto* d = std::get_if<DispatchPass>(&pass)) {
            if (!cp) cp = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            wgpuComputePassEncoderSetPipeline(cp, at(pipelines_, handle_slot(d->pipeline.id))->compute);
            Bound b[kMaxBindGroups];
            resolve(d->pipeline, d->groups, d->dynamic_offsets, 0, b);
            for (uint32_t g = 0; g < kMaxBindGroups; ++g)
                if (b[g].group) wgpuComputePassEncoderSetBindGroup(cp, g, b[g].group, b[g].count, b[g].offsets);
            wgpuComputePassEncoderDispatchWorkgroups(cp, d->x, d->y, d->z);
            continue;
        }
        end_compute();
        if (const auto* c = std::get_if<CopyPass>(&pass)) {
            const Tex* t = at(textures_, handle_slot(c->dst.id));
            WGPUTexelCopyBufferInfo src = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
            src.buffer = at(buffers_, handle_slot(c->src.id))->buffer;
            src.layout.offset = c->offset;
            src.layout.bytesPerRow = c->row_pitch;
            src.layout.rowsPerImage = t->h;
            WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            dst.texture = t->texture;
            const WGPUExtent3D size{t->w, t->h, 1};
            wgpuCommandEncoderCopyBufferToTexture(enc, &src, &dst, &size);
        } else if (const auto* r = std::get_if<RenderPass>(&pass)) {
            const DrawList& list = r->list;
            const RenderTarget& tg = list.target();
            const Tex* color = at(textures_, handle_slot(tg.color.id));
            WGPURenderPassColorAttachment ca[2] = {WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT,
                                                   WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT};
            const WGPULoadOp load = tg.clear ? WGPULoadOp_Clear : WGPULoadOp_Load;
            ca[0].view = color->view;
            ca[0].loadOp = load;
            ca[0].storeOp = WGPUStoreOp_Store;
            ca[0].clearValue = unpack_clear(tg.clear_rgba);
            if (tg.id) {
                ca[1].view = at(textures_, handle_slot(tg.id.id))->view;
                ca[1].loadOp = load;
                ca[1].storeOp = WGPUStoreOp_Store;
                ca[1].clearValue = WGPUColor{0, 0, 0, 0};
            }
            WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
            if (tg.depth) {
                da.view = at(textures_, handle_slot(tg.depth.id))->view;
                da.depthLoadOp = load;
                da.depthStoreOp = WGPUStoreOp_Store;
                da.depthClearValue = 1.0f;
            }
            WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
            rp.colorAttachmentCount = tg.id ? 2 : 1;
            rp.colorAttachments = ca;
            rp.depthStencilAttachment = tg.depth ? &da : nullptr;
            WGPURenderPassEncoder pe = wgpuCommandEncoderBeginRenderPass(enc, &rp);

            // State already set in this pass, so that an unchanged one is not set again.
            uint64_t cur_pipe = 0;
            Bound cur[kMaxBindGroups];
            WGPUBuffer cur_vb[kMaxVertexBuffers] = {};
            uint64_t cur_vo[kMaxVertexBuffers] = {};
            WGPUBuffer cur_ib = nullptr;
            uint32_t cur_sc[4] = {0, 0, color->w, color->h};
            for (const DrawRecord& rec : list.records()) {
                const Pipe* pp = at(pipelines_, handle_slot(rec.pipeline.id));
                if (rec.pipeline.id != cur_pipe) {
                    wgpuRenderPassEncoderSetPipeline(pe, pp->render);
                    cur_pipe = rec.pipeline.id;
                }
                Bound b[kMaxBindGroups];
                resolve(rec.pipeline, rec.groups, rec.dynamic_offsets, base[i], b);
                for (uint32_t g = 0; g < kMaxBindGroups; ++g) {
                    if (!b[g].group) continue;
                    if (b[g].group == cur[g].group && b[g].count == cur[g].count &&
                        std::equal(b[g].offsets, b[g].offsets + b[g].count, cur[g].offsets))
                        continue;
                    wgpuRenderPassEncoderSetBindGroup(pe, g, b[g].group, b[g].count, b[g].offsets);
                    cur[g] = b[g];
                }
                for (uint32_t v = 0; v < pp->vertex_layout_count; ++v) {
                    const Buf* vb = at(buffers_, handle_slot(rec.vertex[v].buffer.id));
                    if (vb->buffer == cur_vb[v] && rec.vertex[v].offset == cur_vo[v]) continue;
                    wgpuRenderPassEncoderSetVertexBuffer(pe, v, vb->buffer, rec.vertex[v].offset,
                                                         vb->size - rec.vertex[v].offset);
                    cur_vb[v] = vb->buffer;
                    cur_vo[v] = rec.vertex[v].offset;
                }
                // The scissor, clamped to the target.
                uint32_t sc[4];
                sc[0] = std::min<uint32_t>(rec.scissor[0], color->w);
                sc[1] = std::min<uint32_t>(rec.scissor[1], color->h);
                sc[2] = std::min<uint32_t>(rec.scissor[2], color->w - sc[0]);
                sc[3] = std::min<uint32_t>(rec.scissor[3], color->h - sc[1]);
                if (!std::equal(sc, sc + 4, cur_sc)) {
                    wgpuRenderPassEncoderSetScissorRect(pe, sc[0], sc[1], sc[2], sc[3]);
                    std::copy(sc, sc + 4, cur_sc);
                }
                if (rec.index_buffer) {
                    const Buf* ib = at(buffers_, handle_slot(rec.index_buffer.id));
                    if (ib->buffer != cur_ib) {
                        wgpuRenderPassEncoderSetIndexBuffer(pe, ib->buffer, WGPUIndexFormat_Uint32, 0, ib->size);
                        cur_ib = ib->buffer;
                    }
                    wgpuRenderPassEncoderDrawIndexed(pe, rec.count, rec.instance_count, uint32_t(rec.index_offset / 4),
                                                     int32_t(rec.first_vertex), rec.first_instance);
                } else {
                    wgpuRenderPassEncoderDraw(pe, rec.count, rec.instance_count, rec.first_vertex, rec.first_instance);
                }
            }
            wgpuRenderPassEncoderEnd(pe);
            wgpuRenderPassEncoderRelease(pe);
        }
    }
    end_compute();
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    pop_scope(("submit " + std::to_string(s)).c_str());

    struct Done {
        std::shared_ptr<bool> alive;
        WebGpuBackend* self;
        Serial s;
    };
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* u1, void*) {
        Done* d = static_cast<Done*>(u1);
        // Completed or not (a lost device), the serial is over: nothing will run it later.
        if (*d->alive) d->self->complete(d->s);
        delete d;
    };
    cb.userdata1 = new Done{alive_, this, s};
    wgpuQueueOnSubmittedWorkDone(queue_, cb);
    return true;
}

bool WebGpuBackend::on_wait(Serial) { return false; }

// ---- readback ----------------------------------------------------------------------------------------

bool WebGpuBackend::on_read_texture(Serial s, uint32_t readback_slot, uint32_t texture_slot, uint32_t x,
                                    uint32_t y, uint32_t w, uint32_t h, std::string* error) {
    Tex* t = at(textures_, texture_slot);
    if (!t || !t->texture) {
        *error = "the texture is gone";
        return false;
    }
    auto p = std::make_shared<Pending>();
    p->w = w;
    p->h = h;
    p->bpp = texel_bytes(t->format);
    p->row_pitch = uint32_t(round_up(uint64_t(w) * p->bpp, kCopyRowPitchAlign));
    const uint64_t bytes = uint64_t(p->row_pitch) * h;
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.label = sv("caliper.gfx.readback");
    bd.size = bytes;
    bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    p->staging = wgpuDeviceCreateBuffer(device_, &bd);
    if (!p->staging) {
        *error = "wgpuDeviceCreateBuffer failed for the staging buffer";
        return false;
    }
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = t->texture;
    src.origin = WGPUOrigin3D{x, y, 0};
    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = p->staging;
    dst.layout.bytesPerRow = p->row_pitch;
    dst.layout.rowsPerImage = h;
    const WGPUExtent3D size{w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    if (readback_slot >= readbacks_.size()) readbacks_.resize(size_t(readback_slot) + 1);
    readbacks_[readback_slot] = p;

    struct Map {
        std::shared_ptr<Pending> p;
        std::shared_ptr<bool> alive;
        WebGpuBackend* self;
        Serial s;
        uint64_t bytes;
    };
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        Map* m = static_cast<Map*>(u1);
        Pending& p = *m->p;
        if (status == WGPUMapAsyncStatus_Success) {
            const uint8_t* src = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(p.staging, 0, size_t(m->bytes)));
            if (src) {
                const size_t row = size_t(p.w) * p.bpp;
                p.bytes.resize(row * p.h);
                for (uint32_t r = 0; r < p.h; ++r)
                    std::memcpy(p.bytes.data() + r * row, src + size_t(r) * p.row_pitch, row);
            }
            wgpuBufferUnmap(p.staging);
        }
        destroy_buffer(p.staging);
        p.ready = true;
        // Read or failed, the serial is over; a failed readback hands over no bytes.
        if (*m->alive) m->self->complete(m->s);
        delete m;
    };
    cb.userdata1 = new Map{p, alive_, this, s, bytes};
    wgpuBufferMapAsync(p->staging, WGPUMapMode_Read, 0, size_t(bytes), cb);
    return true;
}

void WebGpuBackend::on_take_readback(uint32_t readback_slot, std::vector<uint8_t>* out) {
    if (readback_slot < readbacks_.size() && readbacks_[readback_slot]) *out = std::move(readbacks_[readback_slot]->bytes);
    else out->clear();
}

}  // namespace caliper_gfx
