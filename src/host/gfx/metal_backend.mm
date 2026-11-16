// metal_backend.mm: the Metal backend of the graphics layer (metal_backend.h), compiled with
// ARC. It runs on the renderer's device and queue and keeps Metal's fast paths behind the
// layer's interface:
//
//  - A bind group is no Metal object: its entries are resolved once, at creation, into the
//    MTLBuffer, MTLTexture and MTLSamplerState they name and their MSL indices, and bound with
//    setVertexBuffer and its relatives at draw time.
//  - Every arena of a submit is copied once into a shared ring buffer; a draw whose bind
//    groups did not change moves only its dynamic offsets, with setVertexBufferOffset and
//    setFragmentBufferOffset. Pipelines, bind groups, vertex buffers and scissors that repeat
//    from one record to the next are not set again.
//  - Command buffers are made with unretained references: the core destroys nothing before
//    the last submit that used it completed, so Metal need not retain each bound object.
//  - Writes to memory that no unfinished submit uses go straight to it (memcpy into a shared
//    buffer, replaceRegion on a shared texture); otherwise a blit orders them on the queue. While
//    such a blit has not completed, later writes to the same resource are queued too: written
//    directly, they could land before it, and the older bytes would win.
//  - Programs compile with fast math off, so float results follow IEEE rules as the CPU
//    references do, unless the program's description sets fast_math (the colormap's compute
//    form, whose output no float rounding can change).
//  - An MTLBuffer of this device is imported by retaining it, and page-aligned host memory by
//    newBufferWithBytesNoCopy: no copy on either route.
#include "gfx/metal_backend.h"
#include "gfx/backend_impl.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace caliper_gfx {
namespace {

MTLPixelFormat pixel_format(Format f) {
    switch (f) {
        case Format::RGBA8Unorm: return MTLPixelFormatRGBA8Unorm;
        case Format::BGRA8Unorm: return MTLPixelFormatBGRA8Unorm;
        case Format::R8Unorm: return MTLPixelFormatR8Unorm;
        case Format::R16Uint: return MTLPixelFormatR16Uint;
        case Format::R32Uint: return MTLPixelFormatR32Uint;
        case Format::RG32Uint: return MTLPixelFormatRG32Uint;
        case Format::R32Float: return MTLPixelFormatR32Float;
        case Format::Depth32Float: return MTLPixelFormatDepth32Float;
    }
    return MTLPixelFormatInvalid;
}

MTLVertexFormat vertex_format(VertexFormat f) {
    switch (f) {
        case VertexFormat::F32: return MTLVertexFormatFloat;
        case VertexFormat::F32x2: return MTLVertexFormatFloat2;
        case VertexFormat::F32x3: return MTLVertexFormatFloat3;
        case VertexFormat::F32x4: return MTLVertexFormatFloat4;
        case VertexFormat::Unorm8x4: return MTLVertexFormatUChar4Normalized;
        case VertexFormat::U8: return MTLVertexFormatUChar;
        case VertexFormat::U32: return MTLVertexFormatUInt;
        case VertexFormat::S32: return MTLVertexFormatInt;
        case VertexFormat::U32x2: return MTLVertexFormatUInt2;
    }
    return MTLVertexFormatInvalid;
}

MTLPrimitiveType primitive(Topology t) {
    switch (t) {
        case Topology::TriangleList: return MTLPrimitiveTypeTriangle;
        case Topology::TriangleStrip: return MTLPrimitiveTypeTriangleStrip;
        case Topology::LineList: return MTLPrimitiveTypeLine;
        case Topology::LineStrip: return MTLPrimitiveTypeLineStrip;
        case Topology::PointList: return MTLPrimitiveTypePoint;
    }
    return MTLPrimitiveTypeTriangle;
}

MTLPrimitiveTopologyClass topology_class(Topology t) {
    switch (t) {
        case Topology::PointList: return MTLPrimitiveTopologyClassPoint;
        case Topology::LineList: case Topology::LineStrip: return MTLPrimitiveTopologyClassLine;
        default: return MTLPrimitiveTopologyClassTriangle;
    }
}

std::string ns_error(NSError* e) { return e ? std::string([[e localizedDescription] UTF8String]) : "unknown error"; }

struct MEntry {
    BindingType type = BindingType::Uniform;
    uint8_t stages = 0, msl = 0, rank = 0;   // rank: index among the group's dynamic bindings
    bool dynamic = false, arena = false;
    id<MTLBuffer> buffer = nil;
    uint64_t offset = 0, size = 0;
    id<MTLTexture> texture = nil;
    id<MTLSamplerState> sampler = nil;
};
struct MGroup { std::vector<MEntry> entries; };

struct MPipe {
    id<MTLRenderPipelineState> render = nil;
    id<MTLComputePipelineState> compute = nil;
    id<MTLDepthStencilState> depth = nil;
    MTLPrimitiveType prim = MTLPrimitiveTypeTriangle;
    uint64_t program = 0;
    uint8_t group_mask = 0;
    uint8_t dynamic_base[kMaxBindGroups] = {};
    uint32_t vertex_layouts = 0;
    std::vector<std::pair<uint8_t, uint8_t>> storage;   // (group, entry) of each storage binding
    uint8_t storage_stages = 0;
    MTLSize workgroup = MTLSizeMake(1, 1, 1);
};

struct MTex {
    id<MTLTexture> texture = nil;
    Format format = Format::RGBA8Unorm;
    bool shared = false;
};

// A ring of shared memory for the uniform arenas. A region is reused once the submit that
// read it completed; when no region is free the ring grows into a new buffer and the old one
// is dropped once its last submit completed. Nothing ever waits.
class UniformRing {
public:
    void init(id<MTLDevice> d) { device_ = d; }
    bool alloc(uint64_t n, Serial s, Serial through, id<MTLBuffer> __strong* out, uint64_t* off) {
        while (!inflight_.empty() && inflight_.front().serial <= through) inflight_.pop_front();
        retired_.erase(std::remove_if(retired_.begin(), retired_.end(),
                                      [&](const Retired& r) { return r.serial <= through; }),
                       retired_.end());
        if (inflight_.empty()) head_ = tail_ = 0;
        else tail_ = inflight_.front().start;
        uint64_t at = UINT64_MAX;
        if (buffer_ != nil) {
            const bool full = !inflight_.empty() && head_ == tail_;
            if (!full && head_ >= tail_) {
                if (capacity_ - head_ >= n) at = head_;
                else if (tail_ >= n) at = 0;
            } else if (!full && tail_ - head_ >= n) {
                at = head_;
            }
        }
        if (at == UINT64_MAX) {   // grow
            if (buffer_ != nil && !inflight_.empty()) retired_.push_back({buffer_, inflight_.back().serial});
            capacity_ = std::max<uint64_t>(std::max<uint64_t>(capacity_ * 2, 1 << 20), align_up(n, 1 << 16) * 2);
            buffer_ = [device_ newBufferWithLength:(NSUInteger)capacity_ options:MTLResourceStorageModeShared];
            if (buffer_ == nil) return false;
            inflight_.clear();
            head_ = tail_ = at = 0;
        }
        head_ = at + n;
        inflight_.push_back({at, s});
        *out = buffer_;
        *off = at;
        return true;
    }
    uint64_t capacity() const { return capacity_; }

private:
    struct Region { uint64_t start; Serial serial; };
    struct Retired { id<MTLBuffer> buffer; Serial serial; };
    id<MTLDevice> device_ = nil;
    id<MTLBuffer> buffer_ = nil;
    uint64_t capacity_ = 0, head_ = 0, tail_ = 0;
    std::deque<Region> inflight_;
    std::vector<Retired> retired_;
};

template <class V> void grow(V& v, uint32_t slot) {
    if (v.size() <= slot) v.resize(size_t(slot) + 1);
}

Caps metal_caps(id<MTLDevice> device) {
    static std::string adapter;
    adapter = device ? [[device name] UTF8String] : "";
    Caps c;
    c.bits = kCapVertexStorage | kCapPersistentMap | kCapBlockingWait | kCapTimestamps | kCapImportMetalBuffer |
             kCapImportHostMemory | kCapImportMetalTexture | kCapWaitMetalEvent;
    c.backend = "metal";
    c.adapter = adapter.c_str();
    if (device) {
        c.limits.max_texture_dimension_2d = 16384;
        c.limits.max_buffer_size = (uint64_t)device.maxBufferLength;
        c.limits.max_storage_buffer_binding_size = (uint64_t)device.maxBufferLength;
    }
    return c;
}

class MetalBackend final : public Backend {
public:
    MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue)
        : Backend(metal_caps(device)), device_(device), queue_(queue) {
        ring_.init(device);
        unified_ = device.hasUnifiedMemory;
        page_ = (uint64_t)getpagesize();
    }

    ~MetalBackend() override {
        if (!inflight_.empty()) [inflight_.back().cb waitUntilCompleted];
    }

protected:
    bool on_create_buffer(uint32_t slot, const BufferDesc& d, std::string* error) override {
        grow(buffers_, slot);
        id<MTLBuffer> b = [device_ newBufferWithLength:(NSUInteger)d.size options:MTLResourceStorageModeShared];
        if (b == nil) { *error = "newBufferWithLength failed"; return false; }
        if (d.label) b.label = [NSString stringWithUTF8String:d.label];
        buffers_[slot] = b;
        return true;
    }

    bool on_create_texture(uint32_t slot, const TextureDesc& d, std::string* error) override {
        grow(textures_, slot);
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format(d.format)
                                                                                      width:d.width
                                                                                     height:d.height
                                                                                  mipmapped:NO];
        MTLTextureUsage usage = 0;
        if (d.usage & kSampled) usage |= MTLTextureUsageShaderRead;
        if (d.usage & kStorageTex) usage |= MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
        if (d.usage & kRenderTarget) usage |= MTLTextureUsageRenderTarget;
        td.usage = usage;
        // Render targets and depth stay private, where the GPU keeps them compressed; the
        // others are shared on unified memory, so an idle upload is a plain replaceRegion.
        const bool shared = unified_ && !(d.usage & kRenderTarget) && !is_depth_format(d.format);
        td.storageMode = shared ? MTLStorageModeShared : MTLStorageModePrivate;
        id<MTLTexture> t = [device_ newTextureWithDescriptor:td];
        if (t == nil) { *error = "newTextureWithDescriptor failed"; return false; }
        if (d.label) t.label = [NSString stringWithUTF8String:d.label];
        textures_[slot] = MTex{t, d.format, shared};
        return true;
    }

    bool on_create_sampler(uint32_t slot, const SamplerDesc& d, std::string* error) override {
        grow(samplers_, slot);
        MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = d.linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
        sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
        sd.normalizedCoordinates = YES;
        id<MTLSamplerState> s = [device_ newSamplerStateWithDescriptor:sd];
        if (s == nil) { *error = "newSamplerStateWithDescriptor failed"; return false; }
        samplers_[slot] = s;
        return true;
    }

    bool on_create_program(uint32_t slot, const ProgramDesc& d, std::string* error) override {
        if (d.msl == nullptr) { *error = "the program has no MSL source"; return false; }
        grow(libraries_, slot);
        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = d.fast_math ? MTLMathModeFast : MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = d.fast_math ? YES : NO;
#pragma clang diagnostic pop
        }
        NSError* err = nil;
        id<MTLLibrary> lib = [device_ newLibraryWithSource:[NSString stringWithUTF8String:d.msl] options:opts error:&err];
        if (lib == nil) { *error = "MSL compile failed: " + ns_error(err); return false; }
        for (uint32_t i = 0; i < d.entry_count; ++i)
            if ([lib newFunctionWithName:[NSString stringWithUTF8String:d.entries[i].name]] == nil) {
                *error = std::string("the MSL has no function ") + d.entries[i].name;
                return false;
            }
        libraries_[slot] = lib;
        return true;
    }

    // Override constants never reach a backend (decision 16 of gfx.h).
    static id<MTLFunction> function(id<MTLLibrary> lib, const char* name) {
        return [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    }

    bool on_create_render_pipeline(uint32_t slot, const RenderPipelineDesc& d, std::string* error) override {
        grow(pipes_, slot);
        id<MTLLibrary> lib = libraries_[handle_slot(d.program.id)];
        id<MTLFunction> vs = function(lib, d.vertex_entry);
        id<MTLFunction> fs = function(lib, d.fragment_entry);
        if (vs == nil || fs == nil) {
            *error = "a function is missing from the program's MSL";
            return false;
        }
        MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = vs;
        pd.fragmentFunction = fs;
        if (d.label) pd.label = [NSString stringWithUTF8String:d.label];
        if (d.vertex_layout_count > 0) {
            MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
            for (uint32_t i = 0; i < d.vertex_layout_count; ++i) {
                const VertexLayout& l = d.vertex_layouts[i];
                MTLVertexBufferLayoutDescriptor* bl = vd.layouts[kMetalVertexBufferBase + i];
                uint32_t end = 4;
                for (uint32_t a = 0; a < l.attribute_count; ++a)
                    end = std::max(end, l.attributes[a].offset + vertex_format_bytes(l.attributes[a].format));
                if (l.stride == 0) {   // a constant stream: one element for every vertex
                    bl.stepFunction = MTLVertexStepFunctionConstant;
                    bl.stepRate = 0;
                    bl.stride = (NSUInteger)align_up(end, 4);
                } else {
                    bl.stepFunction = l.step == Step::Instance ? MTLVertexStepFunctionPerInstance
                                                               : MTLVertexStepFunctionPerVertex;
                    bl.stepRate = 1;
                    bl.stride = l.stride;
                }
                for (uint32_t a = 0; a < l.attribute_count; ++a) {
                    MTLVertexAttributeDescriptor* ad = vd.attributes[l.attributes[a].location];
                    ad.format = vertex_format(l.attributes[a].format);
                    ad.offset = l.attributes[a].offset;
                    ad.bufferIndex = kMetalVertexBufferBase + i;
                }
            }
            pd.vertexDescriptor = vd;
        }
        pd.inputPrimitiveTopology = topology_class(d.topology);
        MTLRenderPipelineColorAttachmentDescriptor* ca = pd.colorAttachments[0];
        ca.pixelFormat = pixel_format(d.color_format);
        if (d.blend != Blend::Opaque) {
            ca.blendingEnabled = YES;
            ca.rgbBlendOperation = ca.alphaBlendOperation = MTLBlendOperationAdd;
            if (d.blend == Blend::Alpha) {   // the geometry service's equations
                ca.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
                ca.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                ca.sourceAlphaBlendFactor = MTLBlendFactorOne;
                ca.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            } else {
                ca.sourceRGBBlendFactor = ca.destinationRGBBlendFactor = MTLBlendFactorOne;
                ca.sourceAlphaBlendFactor = ca.destinationAlphaBlendFactor = MTLBlendFactorOne;
            }
        }
        if (d.id_target) pd.colorAttachments[1].pixelFormat = MTLPixelFormatRG32Uint;
        if (d.depth) pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        NSError* err = nil;
        MPipe p;
        p.render = [device_ newRenderPipelineStateWithDescriptor:pd error:&err];
        if (p.render == nil) { *error = "render pipeline: " + ns_error(err); return false; }
        if (d.depth) {
            MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
            dd.depthCompareFunction = d.depth_test ? MTLCompareFunctionLessEqual : MTLCompareFunctionAlways;
            dd.depthWriteEnabled = d.depth_write;
            p.depth = [device_ newDepthStencilStateWithDescriptor:dd];
        }
        p.prim = primitive(d.topology);
        fill_layout(p, d.program);
        p.vertex_layouts = d.vertex_layout_count;
        pipes_[slot] = p;
        return true;
    }

    void fill_layout(MPipe& p, Program program) {
        const ProgramInfo& info = program_info(program);
        p.program = program.id;
        p.group_mask = info.group_mask;
        std::copy(std::begin(info.dynamic_base), std::end(info.dynamic_base), p.dynamic_base);
        // Naga's sizes buffer: one uint per storage binding, in (group, binding) order.
        uint8_t index_in_group[kMaxBindGroups] = {};
        for (const BindingDesc& b : info.bindings) {
            const uint8_t j = index_in_group[b.group]++;
            if (b.type == BindingType::Storage || b.type == BindingType::StorageReadWrite) {
                p.storage.push_back({b.group, j});
                p.storage_stages |= b.stages;
            }
        }
    }

    // Binds the sizes buffer Naga's MSL reads for runtime-sized storage arrays.
    template <class Enc, class GroupOf>
    void bind_sizes(Enc enc, const MPipe& p, GroupOf group_of, bool compute) {
        if (p.storage.empty()) return;
        uint32_t sizes[kMaxBindGroups * kMaxBindingsPerGroup];
        size_t n = 0;
        for (const auto& s : p.storage)
            sizes[n++] = uint32_t(std::min<uint64_t>(group_of(s.first).entries[s.second].size, 0xffffffffu));
        const NSUInteger len = NSUInteger(n * sizeof(uint32_t));
        if constexpr (std::is_same_v<Enc, id<MTLComputeCommandEncoder>>) {
            (void)compute;
            [enc setBytes:sizes length:len atIndex:kMetalSizesIndex];
        } else {
            if (p.storage_stages & kStageVertex) [enc setVertexBytes:sizes length:len atIndex:kMetalSizesIndex];
            if (p.storage_stages & kStageFragment) [enc setFragmentBytes:sizes length:len atIndex:kMetalSizesIndex];
        }
    }

    bool on_create_compute_pipeline(uint32_t slot, const ComputePipelineDesc& d, std::string* error) override {
        grow(pipes_, slot);
        id<MTLLibrary> lib = libraries_[handle_slot(d.program.id)];
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:d.entry]];
        NSError* err = nil;
        MPipe p;
        p.compute = f ? [device_ newComputePipelineStateWithFunction:f error:&err] : nil;
        if (p.compute == nil) { *error = "compute pipeline: " + ns_error(err); return false; }
        fill_layout(p, d.program);
        const EntryInfo* e = program_info(d.program).entry(d.entry, kStageCompute);
        p.workgroup = MTLSizeMake(e->workgroup_size[0], e->workgroup_size[1], e->workgroup_size[2]);
        pipes_[slot] = p;
        return true;
    }

    bool on_create_bind_group(uint32_t slot, const BindGroupDesc& d, std::string*) override {
        grow(groups_, slot);
        // The core validated the group and keeps its entries sorted beside the program's
        // bindings; this resolves them once into Metal objects and indices.
        MGroup g;
        const ProgramInfo& info = program_info(d.program);
        uint8_t rank = 0;
        for (const BindingDesc& b : info.bindings) {
            if (b.group != d.group) continue;
            const BindEntry* e = nullptr;
            for (uint32_t i = 0; i < d.entry_count; ++i)
                if (d.entries[i].binding == b.binding) e = &d.entries[i];
            MEntry m;
            m.type = b.type;
            m.stages = b.stages;
            m.msl = b.msl_index;
            m.dynamic = b.dynamic;
            if (b.dynamic) m.rank = rank++;
            if (is_buffer_binding(b.type)) {
                m.arena = e->buffer == kArena;
                m.buffer = m.arena ? nil : buffers_[handle_slot(e->buffer.id)];
                m.offset = e->offset;
                m.size = e->size;   // resolved by the core
            } else if (b.type == BindingType::Sampler) {
                m.sampler = samplers_[handle_slot(e->sampler.id)];
            } else {
                m.texture = textures_[handle_slot(e->texture.id)].texture;
            }
            g.entries.push_back(m);
        }
        groups_[slot] = std::move(g);
        return true;
    }

    void on_destroy(uint64_t handle) override {
        const uint32_t slot = handle_slot(handle);
        switch (handle_kind(handle)) {
            case HandleKind::Buffer:
                if (slot < buffers_.size()) buffers_[slot] = nil;
                if (slot < buffer_writes_.size()) buffer_writes_[slot] = nil;   // the blit holds its own reference
                break;
            case HandleKind::Texture:
                if (slot < textures_.size()) textures_[slot] = MTex{};
                if (slot < texture_writes_.size()) texture_writes_[slot] = nil;
                break;
            case HandleKind::Sampler: if (slot < samplers_.size()) samplers_[slot] = nil; break;
            case HandleKind::Program: if (slot < libraries_.size()) libraries_[slot] = nil; break;
            case HandleKind::Pipeline: if (slot < pipes_.size()) pipes_[slot] = MPipe{}; break;
            case HandleKind::BindGroup: if (slot < groups_.size()) groups_[slot] = MGroup{}; break;
            case HandleKind::Readback: if (slot < readbacks_.size()) readbacks_[slot] = nil; break;
            case HandleKind::None: break;
        }
    }

    // Whether the last queued write of a resource may still be ahead on the queue; forgets it once
    // it completed.
    static bool queued_write_pending(std::vector<id<MTLCommandBuffer>>& writes, uint32_t slot) {
        if (slot >= writes.size() || writes[slot] == nil) return false;
        if (writes[slot].status >= MTLCommandBufferStatusCompleted) {   // completed, or failed
            writes[slot] = nil;
            return false;
        }
        return true;
    }

    void on_write_buffer(uint32_t slot, uint64_t offset, const void* data, uint64_t size, bool in_use) override {
        id<MTLBuffer> dst = buffers_[slot];
        if (!in_use && !queued_write_pending(buffer_writes_, slot)) {
            std::memcpy(static_cast<uint8_t*>(dst.contents) + offset, data, (size_t)size);
            return;
        }
        id<MTLBuffer> staging = [device_ newBufferWithBytes:data length:(NSUInteger)size
                                                    options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:(NSUInteger)offset
                        size:(NSUInteger)size];
        [blit endEncoding];
        [cb commit];
        grow(buffer_writes_, slot);
        buffer_writes_[slot] = cb;
    }

    void on_write_texture(uint32_t slot, const void* data, uint32_t row_pitch, bool in_use) override {
        const MTex& t = textures_[slot];
        const NSUInteger w = t.texture.width, h = t.texture.height;
        if (t.shared && !in_use && !queued_write_pending(texture_writes_, slot)) {
            [t.texture replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:data bytesPerRow:row_pitch];
            return;
        }
        // The caller gives (h - 1) row pitches and one row: no more is read from its memory.
        const size_t tight = size_t(row_pitch) * (h - 1) + size_t(w) * format_bytes(t.format);
        id<MTLBuffer> staging = [device_ newBufferWithLength:(NSUInteger)row_pitch * h
                                                     options:MTLResourceStorageModeShared];
        std::memcpy(staging.contents, data, tight);
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:row_pitch sourceBytesPerImage:row_pitch * h
                  sourceSize:MTLSizeMake(w, h, 1) toTexture:t.texture destinationSlice:0 destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];
        grow(texture_writes_, slot);
        texture_writes_[slot] = cb;
    }

    void* on_mapped(uint32_t slot) override { return buffers_[slot].contents; }

    bool on_import_buffer(uint32_t slot, const ExternalBuffer& ext, ImportResult* r) override {
        grow(buffers_, slot);
        if (ext.kind == ExternalKind::MetalBuffer) {
            id<MTLBuffer> b = (__bridge id<MTLBuffer>)ext.handle;
            if (b.device.registryID != device_.registryID) {
                r->reason = "the MTLBuffer belongs to another device";
                return false;
            }
            if (b.length < ext.size) {
                r->reason = "the MTLBuffer is shorter than the size claimed";
                return false;
            }
            buffers_[slot] = b;   // the retain is the import
            r->route = Route::InPlace;
            r->bytes_copied = 0;
            return true;
        }
        if (ext.kind != ExternalKind::HostMemory) {
            r->reason = "the Metal backend imports MTLBuffers and host memory";
            return false;
        }
        const bool aligned = reinterpret_cast<uintptr_t>(ext.handle) % page_ == 0 && ext.size % page_ == 0;
        if (aligned) {
            id<MTLBuffer> b = [device_ newBufferWithBytesNoCopy:ext.handle length:(NSUInteger)ext.size
                                                        options:MTLResourceStorageModeShared deallocator:nil];
            if (b != nil) {
                buffers_[slot] = b;
                r->route = Route::InPlace;
                r->bytes_copied = 0;
                return true;
            }
        }
        if (!ext.allow_copy) {
            r->reason = "host memory that is not page-aligned can only be copied";
            return false;
        }
        id<MTLBuffer> b = [device_ newBufferWithBytes:ext.handle length:(NSUInteger)ext.size
                                              options:MTLResourceStorageModeShared];
        if (b == nil) { r->reason = "newBufferWithBytes failed"; return false; }
        buffers_[slot] = b;
        r->route = Route::CopiedFromHost;
        r->bytes_copied = ext.size;
        r->reason = aligned ? "the wrap failed" : "the memory is not page-aligned";
        return true;
    }

    bool on_import_texture(uint32_t slot, const ExternalTexture& ext, ImportResult* r) override {
        grow(textures_, slot);
        id<MTLTexture> t = (__bridge id<MTLTexture>)ext.handle;
        if (t.device.registryID != device_.registryID || t.width != ext.width || t.height != ext.height ||
            t.pixelFormat != pixel_format(ext.format) || t.textureType != MTLTextureType2D) {
            r->reason = "the MTLTexture's device, size, format or type differs from the description";
            return false;
        }
        textures_[slot] = MTex{t, ext.format, t.storageMode == MTLStorageModeShared};
        r->route = Route::InPlace;
        return true;
    }

    void bind_entries(id<MTLRenderCommandEncoder> enc, const MGroup& g, const MPipe& p, uint32_t group,
                      const uint32_t* dyn, id<MTLBuffer> arena, uint64_t arena_base, bool rebind,
                      uint64_t* applied) {
        for (size_t j = 0; j < g.entries.size(); ++j) {
            const MEntry& e = g.entries[j];
            if (is_buffer_binding(e.type)) {
                id<MTLBuffer> buf = e.arena ? arena : e.buffer;
                uint64_t off = e.arena ? arena_base : e.offset;
                if (e.dynamic) off += dyn[p.dynamic_base[group] + e.rank];
                if (rebind) {
                    if (e.stages & kStageVertex) [enc setVertexBuffer:buf offset:(NSUInteger)off atIndex:e.msl];
                    if (e.stages & kStageFragment) [enc setFragmentBuffer:buf offset:(NSUInteger)off atIndex:e.msl];
                } else if (e.dynamic && applied[j] != off) {
                    if (e.stages & kStageVertex) [enc setVertexBufferOffset:(NSUInteger)off atIndex:e.msl];
                    if (e.stages & kStageFragment) [enc setFragmentBufferOffset:(NSUInteger)off atIndex:e.msl];
                }
                if (j < kMaxBindingsPerGroup) applied[j] = off;
            } else if (rebind && e.type == BindingType::Sampler) {
                if (e.stages & kStageVertex) [enc setVertexSamplerState:e.sampler atIndex:e.msl];
                if (e.stages & kStageFragment) [enc setFragmentSamplerState:e.sampler atIndex:e.msl];
            } else if (rebind) {
                if (e.stages & kStageVertex) [enc setVertexTexture:e.texture atIndex:e.msl];
                if (e.stages & kStageFragment) [enc setFragmentTexture:e.texture atIndex:e.msl];
            }
        }
    }

    void encode_render(id<MTLCommandBuffer> cb, const DrawList& list, id<MTLBuffer> arena, uint64_t base) {
        const RenderTarget& tg = list.target();
        const MTex& color = textures_[handle_slot(tg.color.id)];
        const NSUInteger W = color.texture.width, H = color.texture.height;
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        const MTLLoadAction load = tg.clear ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.colorAttachments[0].texture = color.texture;
        rp.colorAttachments[0].loadAction = load;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        const bool unorm = color.format == Format::RGBA8Unorm || color.format == Format::BGRA8Unorm ||
                           color.format == Format::R8Unorm;
        const uint32_t c = tg.clear_rgba;
        rp.colorAttachments[0].clearColor =
            unorm ? MTLClearColorMake((c & 0xff) / 255.0, ((c >> 8) & 0xff) / 255.0, ((c >> 16) & 0xff) / 255.0,
                                      ((c >> 24) & 0xff) / 255.0)
                  : MTLClearColorMake(0, 0, 0, 0);
        if (tg.id) {
            rp.colorAttachments[1].texture = textures_[handle_slot(tg.id.id)].texture;
            rp.colorAttachments[1].loadAction = load;
            rp.colorAttachments[1].storeAction = MTLStoreActionStore;
            rp.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
        }
        if (tg.depth) {
            rp.depthAttachment.texture = textures_[handle_slot(tg.depth.id)].texture;
            rp.depthAttachment.loadAction = load;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
            rp.depthAttachment.clearDepth = 1.0;
        }
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        [enc setViewport:(MTLViewport){0.0, 0.0, (double)W, (double)H, 0.0, 1.0}];

        uint64_t cur_pipe = 0, cur_program = 0;
        uint64_t cur_group[kMaxBindGroups] = {};
        uint64_t applied[kMaxBindGroups][kMaxBindingsPerGroup] = {};
        __unsafe_unretained id<MTLBuffer> cur_vb[kMaxVertexBuffers] = {};
        uint64_t cur_vb_off[kMaxVertexBuffers] = {};
        MTLScissorRect cur_sc = {0, 0, W, H};
        const MPipe* p = nullptr;
        for (const DrawRecord& r : list.records()) {
            const bool pipeline_changed = r.pipeline.id != cur_pipe;
            if (pipeline_changed) {
                p = &pipes_[handle_slot(r.pipeline.id)];
                [enc setRenderPipelineState:p->render];
                if (p->depth) [enc setDepthStencilState:p->depth];
                if (p->program != cur_program) {   // another program may use the indices differently
                    std::fill(std::begin(cur_group), std::end(cur_group), 0);
                    cur_program = p->program;
                }
                cur_pipe = r.pipeline.id;
            }
            bool rebound = pipeline_changed;
            for (uint32_t g = 0; g < kMaxBindGroups; ++g) {
                if (!(p->group_mask & (1u << g))) continue;
                const bool rebind = r.groups[g].id != cur_group[g];
                bind_entries(enc, groups_[handle_slot(r.groups[g].id)], *p, g, r.dynamic_offsets, arena, base,
                             rebind, applied[g]);
                cur_group[g] = r.groups[g].id;
                rebound = rebound || rebind;
            }
            if (rebound)
                bind_sizes(enc, *p, [&](uint8_t g) -> const MGroup& { return groups_[handle_slot(r.groups[g].id)]; },
                           false);
            for (uint32_t i = 0; i < p->vertex_layouts; ++i) {
                id<MTLBuffer> b = buffers_[handle_slot(r.vertex[i].buffer.id)];
                const uint64_t off = r.vertex[i].offset;
                if (b != cur_vb[i]) {
                    [enc setVertexBuffer:b offset:(NSUInteger)off atIndex:kMetalVertexBufferBase + i];
                    cur_vb[i] = b;
                    cur_vb_off[i] = off;
                } else if (off != cur_vb_off[i]) {
                    [enc setVertexBufferOffset:(NSUInteger)off atIndex:kMetalVertexBufferBase + i];
                    cur_vb_off[i] = off;
                }
            }
            const NSUInteger sx = std::min<NSUInteger>(r.scissor[0], W), sy = std::min<NSUInteger>(r.scissor[1], H);
            const MTLScissorRect sc = {sx, sy, std::min<NSUInteger>(r.scissor[2], W - sx),
                                       std::min<NSUInteger>(r.scissor[3], H - sy)};
            if (sc.width == 0 || sc.height == 0 || r.count == 0 || r.instance_count == 0) continue;
            if (sc.x != cur_sc.x || sc.y != cur_sc.y || sc.width != cur_sc.width || sc.height != cur_sc.height) {
                [enc setScissorRect:sc];
                cur_sc = sc;
            }
            if (r.index_buffer) {
                [enc drawIndexedPrimitives:p->prim
                                indexCount:r.count
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:buffers_[handle_slot(r.index_buffer.id)]
                         indexBufferOffset:(NSUInteger)r.index_offset
                             instanceCount:r.instance_count
                                baseVertex:(NSInteger)r.first_vertex
                              baseInstance:r.first_instance];
            } else {
                [enc drawPrimitives:p->prim
                        vertexStart:r.first_vertex
                        vertexCount:r.count
                      instanceCount:r.instance_count
                       baseInstance:r.first_instance];
            }
        }
        [enc endEncoding];
    }

    void encode_dispatch(id<MTLCommandBuffer> cb, const DispatchPass& d, id<MTLBuffer> arena, uint64_t base) {
        const MPipe& p = pipes_[handle_slot(d.pipeline.id)];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p.compute];
        for (uint32_t g = 0; g < kMaxBindGroups; ++g) {
            if (!(p.group_mask & (1u << g))) continue;
            for (const MEntry& e : groups_[handle_slot(d.groups[g].id)].entries) {
                if (is_buffer_binding(e.type)) {
                    uint64_t off = e.arena ? base : e.offset;
                    if (e.dynamic) off += d.dynamic_offsets[p.dynamic_base[g] + e.rank];
                    [enc setBuffer:(e.arena ? arena : e.buffer) offset:(NSUInteger)off atIndex:e.msl];
                } else if (e.type == BindingType::Sampler) {
                    [enc setSamplerState:e.sampler atIndex:e.msl];
                } else {
                    [enc setTexture:e.texture atIndex:e.msl];
                }
            }
        }
        bind_sizes(enc, p, [&](uint8_t g) -> const MGroup& { return groups_[handle_slot(d.groups[g].id)]; }, true);
        [enc dispatchThreadgroups:MTLSizeMake(d.x, d.y, d.z) threadsPerThreadgroup:p.workgroup];
        [enc endEncoding];
    }

    void prune_inflight() {
        const Serial through = completion_sink()->completed_through();
        while (!inflight_.empty() && inflight_.front().serial <= through) inflight_.pop_front();
    }

    void track(Serial s, id<MTLCommandBuffer> cb) {
        std::shared_ptr<CompletionSink> sink = completion_sink();
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            const double ms = done.status == MTLCommandBufferStatusCompleted
                                  ? (done.GPUEndTime - done.GPUStartTime) * 1000.0 : -1.0;
            sink->complete(s, ms);
        }];
        prune_inflight();
        inflight_.push_back({s, cb});
    }

    bool on_submit(Serial s, const Frame& frame, Stats*, std::string* error) override {
        @autoreleasepool {
            // One upload for every arena of the frame.
            std::vector<uint64_t> bases(frame.passes().size, 0);
            uint64_t total = align_up(frame.arena().size, kUniformAlign);
            for (size_t i = 0; i < frame.passes().size; ++i)
                if (const auto* r = std::get_if<RenderPass>(&frame.passes()[i])) {
                    bases[i] = total;
                    total += align_up(r->list.arena().size, kUniformAlign);
                }
            id<MTLBuffer> arena = nil;
            uint64_t base = 0;
            if (total > 0) {
                if (!ring_.alloc(total, s, completion_sink()->completed_through(), &arena, &base)) {
                    *error = "the uniform ring could not grow";
                    return false;
                }
                uint8_t* dst = static_cast<uint8_t*>(arena.contents) + base;
                if (frame.arena().size) std::memcpy(dst, frame.arena().data, frame.arena().size);
                for (size_t i = 0; i < frame.passes().size; ++i)
                    if (const auto* r = std::get_if<RenderPass>(&frame.passes()[i]))
                        if (r->list.arena().size)
                            std::memcpy(dst + bases[i], r->list.arena().data, r->list.arena().size);
            }
            id<MTLCommandBuffer> cb = [queue_ commandBufferWithUnretainedReferences];
            if (cb == nil) { *error = "no command buffer"; return false; }
            for (size_t i = 0; i < frame.passes().size; ++i) {
                const Pass& pass = frame.passes()[i];
                if (const auto* w = std::get_if<WaitPass>(&pass)) {
                    if (w->sync.kind == ExternalSync::MetalEvent)
                        [cb encodeWaitForEvent:(__bridge id<MTLEvent>)w->sync.object value:w->sync.value];
                } else if (const auto* c = std::get_if<CopyPass>(&pass)) {
                    const MTex& t = textures_[handle_slot(c->dst.id)];
                    const NSUInteger w = t.texture.width, h = t.texture.height;
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit copyFromBuffer:buffers_[handle_slot(c->src.id)] sourceOffset:(NSUInteger)c->offset
                         sourceBytesPerRow:c->row_pitch sourceBytesPerImage:(NSUInteger)c->row_pitch * h
                                sourceSize:MTLSizeMake(w, h, 1) toTexture:t.texture destinationSlice:0
                          destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                    [blit endEncoding];
                } else if (const auto* d = std::get_if<DispatchPass>(&pass)) {
                    encode_dispatch(cb, *d, arena, base);
                } else if (const auto* r = std::get_if<RenderPass>(&pass)) {
                    encode_render(cb, r->list, arena, base + bases[i]);
                }
            }
            track(s, cb);
            [cb commit];
            return true;
        }
    }

    bool on_wait(Serial s) override {
        id<MTLCommandBuffer> cb = nil;
        for (const Inflight& f : inflight_)
            if (f.serial >= s) { cb = f.cb; break; }
        if (cb == nil) return true;   // already completed and pruned
        [cb waitUntilCompleted];
        // Queue order: everything committed before it has completed too, even where a
        // completion handler has not run yet.
        std::shared_ptr<CompletionSink> sink = completion_sink();
        for (const Inflight& f : inflight_)
            if (f.serial <= s) sink->complete(f.serial, -1);
        return true;
    }

    bool on_read_texture(Serial s, uint32_t readback_slot, uint32_t texture_slot, uint32_t x, uint32_t y,
                         uint32_t w, uint32_t h, std::string* error) override {
        @autoreleasepool {
            grow(readbacks_, readback_slot);
            const MTex& t = textures_[texture_slot];
            const NSUInteger bpp = format_bytes(t.format), row = w * bpp;
            id<MTLBuffer> staging = [device_ newBufferWithLength:row * h options:MTLResourceStorageModeShared];
            if (staging == nil) { *error = "no staging buffer"; return false; }
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:t.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x, y, 0)
                       sourceSize:MTLSizeMake(w, h, 1) toBuffer:staging destinationOffset:0
           destinationBytesPerRow:row destinationBytesPerImage:row * h
                          options:is_depth_format(t.format) ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
            [blit endEncoding];
            readbacks_[readback_slot] = staging;
            track(s, cb);
            [cb commit];
            return true;
        }
    }

    void on_take_readback(uint32_t readback_slot, std::vector<uint8_t>* out) override {
        id<MTLBuffer> b = readbacks_[readback_slot];
        const uint8_t* p = static_cast<const uint8_t*>(b.contents);
        out->assign(p, p + b.length);
    }

    uint64_t on_imgui_texture_id(uint32_t slot) override {
        return (uint64_t)(__bridge void*)textures_[slot].texture;
    }

private:
    struct Inflight { Serial serial; id<MTLCommandBuffer> cb; };
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    bool unified_ = true;
    uint64_t page_ = 16384;
    UniformRing ring_;
    std::deque<Inflight> inflight_;
    std::vector<id<MTLBuffer>> buffers_;
    std::vector<MTex> textures_;
    std::vector<id<MTLSamplerState>> samplers_;
    std::vector<id<MTLLibrary>> libraries_;
    std::vector<MPipe> pipes_;
    std::vector<MGroup> groups_;
    std::vector<id<MTLBuffer>> readbacks_;
    // The command buffer of each resource's last queued write, until it completed.
    std::vector<id<MTLCommandBuffer>> buffer_writes_, texture_writes_;
};

}  // namespace

std::unique_ptr<Backend> make_metal_backend(void* mtl_device, void* mtl_command_queue) {
    if (mtl_device == nullptr || mtl_command_queue == nullptr) return nullptr;
    return std::make_unique<MetalBackend>((__bridge id<MTLDevice>)mtl_device,
                                          (__bridge id<MTLCommandQueue>)mtl_command_queue);
}

}  // namespace caliper_gfx
