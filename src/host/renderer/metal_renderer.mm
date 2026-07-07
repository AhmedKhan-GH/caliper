// Metal backend for the HostRenderer seam (PLATFORM.md §5.4). Sibling to
// gl_renderer.cpp; make_metal_renderer() (declared in host_renderer.h)
// constructs it and main.cpp selects via CALIPER_RENDERER=metal, falling back
// to the GL backend if init() fails. Apple-only translation unit.
//
// Frame ordering matches the C1 contract: the CLEAR happens in new_frame()
// (the Metal render pass is opened there with loadAction=Clear), the applet
// draws its ImGui frame, then render() flushes RenderDrawData and presents.
// This mirrors the GL backend's top-of-frame clear exactly.
//
// Compiled with ARC (-fobjc-arc, set in CMakeLists): ObjC strong references
// live as members / in an internal id table; the uint64 texture ids handed to
// callers are sequential handles, never raw retained pointers (§5.4).
#include "host_renderer.h"
#include <caliper/services/tensor_bridge_v1_2.h>   // CALIPER_ALLOC_HANDLE_MTLBUFFER

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_metal.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace caliper_host {

// Resolve a colormap id to its 256-entry RGBA8 LUT (tensor_bridge.cpp, present
// in every exe/test link scope that pulls this backend). Unlike
// tex_update_from_device — where the bridge hands the renderer a resolved
// lut256 — the v1.2 imported path receives the raw colormap id (the texture's
// pinned mapping), so the renderer resolves it here.
const uint32_t* colormap_lut(int32_t colormap);

namespace {

// Cross-backend determinism contract (§16): the compute colormap must produce
// the SAME bytes as the CPU reference. The index math below is byte-identical
//   idx = clamp((v - vmin)/(vmax - vmin), 0, 1) * 255 + 0.5  (floor)
// to the staged path. LUT is 256 RGBA8 entries packed little-endian as
// r | g<<8 | b<<16 | a<<24; RGBA8Unorm store rounds byte/255 back to byte.
static const char* kColormapShaderSrc = R"metal(
#include <metal_stdlib>
using namespace metal;

struct CmapParams {
    uint  w;
    uint  h;
    uint  sx;    // element stride along x
    uint  sy;    // element stride along y
    float vmin;
    float vmax;
};

kernel void cmap_f32(device const float*  src  [[buffer(0)]],
                     device const uint*   lut  [[buffer(1)]],
                     constant CmapParams& p    [[buffer(2)]],
                     texture2d<float, access::write> dst [[texture(0)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.w || gid.y >= p.h) return;
    float v = src[gid.y * p.sy + gid.x * p.sx];
    float denom = p.vmax - p.vmin;
    float t = denom != 0.0f ? (v - p.vmin) / denom : 0.0f;
    t = clamp(t, 0.0f, 1.0f);
    uint idx = (uint)(t * 255.0f + 0.5f);
    uint packed = lut[idx];
    float4 c = float4(float(packed        & 0xffu),
                      float((packed >> 8)  & 0xffu),
                      float((packed >> 16) & 0xffu),
                      float((packed >> 24) & 0xffu)) / 255.0f;
    dst.write(c, gid);
}
)metal";

struct CmapParams {
    uint32_t w, h, sx, sy;
    float    vmin, vmax;
};

// Vertex-pulled point pipeline (caliper.geometry.v1) — MSL port of
// shaders/points.vert + points.frag. Element-base addressing (byte offset / 4)
// into whole-bound buffers, same 4-byte alignment gate as Vulkan. Color index
// math byte-identical to points.vert / map_f32_to_rgba8 (NaN->0, degenerate
// range->0). Square points, no discard — deterministic rasterization.
static const char* kPointsShaderSrc = R"metal(
#include <metal_stdlib>
using namespace metal;

struct GeomParams {
    float4x4 mvp;        // proj*view, premultiplied host-side (column-major)
    uint  pos_base;      // element base = byte offset / 4
    uint  attr_base;
    uint  use_attr;
    float vmin;
    float vmax;
    float size_px;
};

struct VOut {
    float4 pos   [[position]];
    float  size  [[point_size]];
    float4 color;
};

vertex VOut points_vs(uint vid [[vertex_id]],
                      device const float* pos  [[buffer(0)]],
                      device const float* attr [[buffer(1)]],
                      device const uint*  lut  [[buffer(2)]],
                      constant GeomParams& p   [[buffer(3)]])
{
    VOut o;
    float3 wp = float3(pos[p.pos_base + 3u * vid + 0u],
                       pos[p.pos_base + 3u * vid + 1u],
                       pos[p.pos_base + 3u * vid + 2u]);
    o.pos  = p.mvp * float4(wp, 1.0f);
    o.size = p.size_px;
    if (p.use_attr != 0u) {
        float v = attr[p.attr_base + vid];
        float t = (v == v && p.vmax > p.vmin)
                ? clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0f, 1.0f) : 0.0f;
        uint packed = lut[(uint)(t * 255.0f + 0.5f)];
        o.color = float4(float(packed         & 0xffu),
                         float((packed >> 8)  & 0xffu),
                         float((packed >> 16) & 0xffu),
                         float((packed >> 24) & 0xffu)) / 255.0f;
    } else {
        o.color = float4(1.0f);
    }
    return o;
}

fragment float4 points_fs(VOut in [[stage_in]]) { return in.color; }
)metal";

struct GeomParams {          // must match the MSL struct byte-for-byte (88 B)
    float    mvp[16];
    uint32_t pos_base, attr_base, use_attr;
    float    vmin, vmax, size_px;
};
static_assert(sizeof(GeomParams) == 88, "MSL constant-buffer layout");

// Byte extent a tensor addresses: (max linear element index + 1) * elem_size,
// from shape×strides. The bridge already bounds this in *elements* against a
// sane cap, but only the backend sees the actual id<MTLBuffer> — so the device
// paths bound the buffer's real byte length here before dispatch. A short
// buffer (e.g. a caller passing a half-sized MTLBuffer) is rejected rather than
// letting the shader/blit read past the allocation and fault the GPU.
static uint64_t tensor_extent_bytes(const CaliperTensor& t, uint64_t elem_size) {
    uint64_t maxidx = 0;
    for (int i = 0; i < t.ndim; ++i)
        maxidx += (uint64_t)(t.shape[i] - 1) * (uint64_t)t.strides[i];
    return (maxidx + 1) * elem_size;
}

class MetalRenderer final : public HostRenderer {
public:
    const char* name() const override { return "metal"; }
    const char* last_device_path() const override { return last_device_path_; }
    CaliperDeviceKind interop_device() const override { return CALIPER_DEV_METAL; }

    // M2b shipped: a non-NULL t.stream is GPU-ordered after the producer
    // queue in both device paths above (D24).
    bool honors_stream_ordered_handoff() const override { return true; }

    // Metal owns no GL context: the window must be created with NO_API so GLFW
    // does not attach an OpenGL context we would fight over. Runs before
    // glfwCreateWindow (sibling to the GL backend's profile hints).
    void window_hints() override {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }

    bool init(GLFWwindow* window) override {
        window_ = window;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) return false;
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) return false;

        NSWindow* nswin = glfwGetCocoaWindow(window);
        if (nswin == nil) return false;
        layer_ = [CAMetalLayer layer];
        layer_.device = device_;
        layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer_.framebufferOnly = YES;
        nswin.contentView.layer = layer_;
        nswin.contentView.wantsLayer = YES;

        // ImGui core/ImPlot contexts are host-owned and already created; only
        // the GLFW + Metal *backends* belong to this renderer.
        if (!ImGui_ImplGlfw_InitForOther(window, true)) return false;
        if (!ImGui_ImplMetal_Init(device_)) {
            ImGui_ImplGlfw_Shutdown();   // undo the GLFW backend for a clean fallback
            return false;
        }

        textures_ = [NSMutableDictionary dictionary];
        events_ = [NSMutableDictionary dictionary];
        pass_desc_ = [MTLRenderPassDescriptor new];
        return true;
    }

    // Start-of-frame: size the layer, acquire a drawable, open the render pass
    // with a CLEAR matching the GL backend's background, then ImGui NewFrame.
    // The clear lives here (not render()) per the C1 contract — anything the
    // app draws this frame lands on top of it.
    void new_frame() override {
        @autoreleasepool {
            int w = 0, h = 0;
            glfwGetFramebufferSize(window_, &w, &h);
            if (w > 0 && h > 0)
                layer_.drawableSize = CGSizeMake((CGFloat)w, (CGFloat)h);

            drawable_ = [layer_ nextDrawable];   // may be nil under load

            pass_desc_.colorAttachments[0].texture = drawable_ ? drawable_.texture : nil;
            pass_desc_.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass_desc_.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass_desc_.colorAttachments[0].clearColor =
                MTLClearColorMake(0.05, 0.05, 0.08, 1.0);   // == GL glClearColor

            frame_cmd_ = [queue_ commandBuffer];
            if (drawable_) {
                frame_enc_ = [frame_cmd_ renderCommandEncoderWithDescriptor:pass_desc_];
            } else {
                frame_enc_ = nil;   // still keep ImGui frame balanced below
            }

            ImGui_ImplMetal_NewFrame(pass_desc_);
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
        }
    }

    void render(int /*fb_w*/, int /*fb_h*/) override {
        @autoreleasepool {
            ImGui::Render();                       // consume the frame regardless
            if (frame_enc_ && drawable_) {
                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
                                               frame_cmd_, frame_enc_);
                [frame_enc_ endEncoding];
                [frame_cmd_ presentDrawable:drawable_];
                [frame_cmd_ commit];
            } else if (frame_cmd_) {
                [frame_cmd_ commit];               // nothing to draw; flush
            }
            frame_enc_ = nil;
            frame_cmd_ = nil;
            drawable_ = nil;
        }
    }

    void shutdown() override {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        [events_ removeAllObjects]; events_ = nil; event_values_.clear();
        [textures_ removeAllObjects];
        textures_ = nil;
        imported_.clear();
        cmap_pipeline_ = nil;
        points_pipeline_ = nil;
        pass_desc_ = nil;
        queue_ = nil;
        layer_ = nil;
        device_ = nil;
    }

    // ---- Texture ops. id table maps an opaque uint64 -> strong MTLTexture;
    // the raw Metal handle never leaves this file as an id. ----
    uint64_t tex_create_rgba8(int w, int h) override {
        if (w <= 0 || h <= 0 || device_ == nil) return 0;
        MTLTextureDescriptor* d =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:(NSUInteger)w
                                                              height:(NSUInteger)h
                                                           mipmapped:NO];
        d.storageMode = MTLStorageModeShared;                  // CPU upload + readback
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;  // compute write
        id<MTLTexture> tex = [device_ newTextureWithDescriptor:d];
        if (tex == nil) return 0;

        uint64_t id = next_id_++;
        textures_[@(id)] = tex;
        return id;
    }

    bool tex_upload_rgba8(uint64_t tex, const void* data, int w, int h) override {
        id<MTLTexture> t = lookup(tex);
        if (t == nil || data == nullptr || w <= 0 || h <= 0) return false;
        [t replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)w, (NSUInteger)h)
             mipmapLevel:0
               withBytes:data
             bytesPerRow:(NSUInteger)w * 4];
        return true;
    }

    void tex_release(uint64_t tex) override {
        [textures_ removeObjectForKey:@(tex)];
        [events_ removeObjectForKey:@(tex)]; event_values_.erase(tex);
    }

    uint64_t tex_imtexture_id(uint64_t tex) override {
        id<MTLTexture> t = lookup(tex);
        return t == nil ? 0 : (uint64_t)(__bridge void*)t;
    }

    // Device-resident update: t.data is an id<MTLBuffer> living on this device.
    // No CPU roundtrip on either path. Returns false -> caller CPU-stages.
    bool tex_update_from_device(uint64_t tex, const CaliperTensor& t,
                                const uint32_t* lut256,
                                float vmin, float vmax) override {
        id<MTLTexture> dst = lookup(tex);
        if (dst == nil) return false;
        if (t.device != CALIPER_DEV_METAL || t.data == nullptr) return false;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)t.data;
        if (src == nil) return false;

        if (t.dtype == CALIPER_DT_F32 && lut256 != nullptr)
            return colormap_compute_from(tex, dst, src, 0, t, lut256, vmin, vmax,
                                         /*imported=*/false);
        if (t.dtype == CALIPER_DT_U8)
            return blit_u8_from(tex, dst, src, 0, t, /*imported=*/false);
        return false;
    }

    // ---- v1.2 imported allocations (in-process MTLBuffer import) -----------
    // The Metal analog of Vulkan's DuplicateHandle + VkImportMemory: there is
    // no OS handle transfer on Apple unified memory, so the "dup" is an ObjC
    // strong retain. Lights up CALIPER_BRIDGE_CAP_IMPORT_ALLOC via the bridge.
    bool supports_external_import() const override { return device_ != nil; }

    uint64_t import_external_allocation(void* os_handle, uint64_t size_bytes,
                                        uint32_t handle_type) override {
        if (handle_type != CALIPER_ALLOC_HANDLE_MTLBUFFER) return 0;
        if (os_handle == nullptr || size_bytes == 0 || device_ == nil) return 0;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)os_handle;
        if (buf == nil) return 0;
        if (buf.device.registryID != device_.registryID) return 0;  // wrong GPU
        if (buf.length < size_bytes) return 0;   // caller overclaims — refuse
        const uint64_t iid = next_import_id_++;
        imported_[iid] = buf;                    // ARC strong ref IS the dup
        return iid;
    }

    void release_external_allocation(uint64_t iid) override { imported_.erase(iid); }

    // Colormap/blit a texture FROM an imported buffer at a byte offset, with NO
    // data copy (the applet's kernels already wrote the bytes). Guards mirror
    // tex_update_from_device; the byte offset rides setBuffer:offset: (f32) or
    // sourceOffset: (u8). Colormap/vmin/vmax are the texture's pinned mapping.
    bool tex_update_from_imported(uint64_t tex, uint64_t alloc, uint64_t offset_bytes,
                                  const CaliperTensor& desc, int32_t colormap,
                                  float vmin, float vmax) override {
        id<MTLTexture> dst = lookup(tex);
        id<MTLBuffer>  src = lookup_import(alloc);
        if (dst == nil || src == nil) return false;
        if (offset_bytes % 4 != 0 || offset_bytes > src.length) return false;
        if (desc.dtype == CALIPER_DT_F32) {
            const uint32_t* lut = colormap_lut(colormap);
            if (lut == nullptr) return false;
            return colormap_compute_from(tex, dst, src, offset_bytes, desc,
                                         lut, vmin, vmax, /*imported=*/true);
        }
        if (desc.dtype == CALIPER_DT_U8)
            return blit_u8_from(tex, dst, src, offset_bytes, desc, /*imported=*/true);
        return false;
    }

    // ---- caliper.geometry.v1: instanced points from imported allocations ----
    // Same gate as the imported-texture path: point data lives in v1.2 imported
    // MTLBuffers, which only exist when external import is up.
    bool supports_geometry() const override { return supports_external_import(); }

    // An offscreen render target that is ALSO an ordinary sampled texture: it
    // lives in textures_, so tex_imtexture_id / debug_readback / tex_release
    // work unchanged; the RenderTarget usage makes it drawable by the point
    // pass. Cleared to opaque black at create so ImGui sampling before the
    // first draw is defined.
    uint64_t geom_create_view(int w, int h) override {
        if (w <= 0 || h <= 0 || device_ == nil) return 0;
        MTLTextureDescriptor* d =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:(NSUInteger)w
                                                              height:(NSUInteger)h
                                                           mipmapped:NO];
        d.storageMode = MTLStorageModeShared;                // unified memory: renderable + readback
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        id<MTLTexture> tex = [device_ newTextureWithDescriptor:d];
        if (tex == nil) return 0;
        @autoreleasepool {                                    // defined pre-first-draw: opaque black
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture     = tex;
            rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0, 0, 1);
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
            [enc endEncoding];
            [cb commit];
        }
        uint64_t tid = next_id_++;
        textures_[@(tid)] = tex;
        return tid;
    }

    // One view frame, atomically: clear + draw `count` vertex-pulled points.
    // Positions/attr are element bases into whole-bound imported buffers, so the
    // offset gate is 4-byte alignment (same as Vulkan). Additive blend, no
    // depth. Metal NDC is +y up with a positive-height viewport, which lands the
    // GL-style ndc_for_pixel mapping directly (NO y-flip, unlike Vulkan). Every
    // gate runs BEFORE the encoder exists — creating the encoder performs the
    // clear, so any late refusal would violate pixels-untouched.
    bool geom_draw_points(uint64_t view_tex, const float* view16, const float* proj16,
                          uint64_t pos_alloc, uint64_t pos_offset, uint64_t count,
                          uint64_t attr_alloc, uint64_t attr_offset,
                          const uint32_t* lut256, float vmin, float vmax,
                          float size_px, uint32_t clear_rgba) override {
        @autoreleasepool {
            // ---- every gate BEFORE the encoder exists: false = pixels untouched ----
            id<MTLTexture> t = lookup(view_tex);
            if (t == nil || view16 == nullptr || proj16 == nullptr) return false;
            id<MTLBuffer> pos = nil, attr = nil;
            if (count > 0) {
                pos = lookup_import(pos_alloc);
                if (pos == nil || pos_offset % 4 != 0) return false;
                if (count > UINT64_MAX / 12) return false;
                if (pos_offset > pos.length || count * 12 > pos.length - pos_offset)
                    return false;
                if (attr_alloc != 0) {
                    attr = lookup_import(attr_alloc);
                    if (attr == nil || attr_offset % 4 != 0 || lut256 == nullptr)
                        return false;
                    if (attr_offset > attr.length || count * 4 > attr.length - attr_offset)
                        return false;
                }
                if (!ensure_points_pipeline()) return false;
            }

            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture     = t;
            rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor  = MTLClearColorMake(
                (double)( clear_rgba        & 0xFFu) / 255.0,
                (double)((clear_rgba >> 8)  & 0xFFu) / 255.0,
                (double)((clear_rgba >> 16) & 0xFFu) / 255.0,
                (double)((clear_rgba >> 24) & 0xFFu) / 255.0);
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
            if (enc == nil) return false;   // nothing encoded, nothing cleared

            if (count > 0) {
                GeomParams p{};
                // mvp = proj * view, column-major — same loop as vulkan_renderer.cpp.
                for (int c = 0; c < 4; ++c)
                    for (int r = 0; r < 4; ++r) {
                        float acc = 0.f;
                        for (int k = 0; k < 4; ++k)
                            acc += proj16[k * 4 + r] * view16[c * 4 + k];
                        p.mvp[c * 4 + r] = acc;
                    }
                p.pos_base  = (uint32_t)(pos_offset / 4);
                p.attr_base = (uint32_t)(attr_offset / 4);
                p.use_attr  = attr != nil ? 1u : 0u;
                p.vmin = vmin; p.vmax = vmax;
                p.size_px = std::min(std::max(size_px, 1.0f), 511.0f);  // Metal point-size cap

                static const uint32_t kZeroLut[256] = {};   // valid-but-unread when flat
                [enc setRenderPipelineState:points_pipeline_];
                MTLViewport vp = {0.0, 0.0, (double)t.width, (double)t.height, 0.0, 1.0};
                [enc setViewport:vp];                        // positive height: no Y flip on Metal
                [enc setVertexBuffer:pos offset:0 atIndex:0];
                [enc setVertexBuffer:(attr != nil ? attr : pos) offset:0 atIndex:1];
                [enc setVertexBytes:(lut256 ? lut256 : kZeroLut)
                             length:256 * sizeof(uint32_t) atIndex:2];  // 1 KB < 4 KB setBytes cap
                [enc setVertexBytes:&p length:sizeof(p) atIndex:3];
                [enc drawPrimitives:MTLPrimitiveTypePoint
                        vertexStart:0 vertexCount:(NSUInteger)count];
            }
            [enc endEncoding];
            [cb commit];   // no CPU wait: same-queue_ commit order covers the frame's
                           // sampling; producer (MPS) writes are already CPU-drained
                           // before publish (flow_scope sync contract).
            last_device_path_ = "points-imported";
            return true;
        }
    }

    // Test-only (spec §3.4 / M1): copy a texture back on the RENDERER's own
    // queue — commit order retires every previously committed tensor op, so
    // this reads fully-updated texels without the hot path ever waiting. The
    // gfx harness passes the PUBLIC bridge id (the bridged texture pointer),
    // so resolve by pointer value against the id table; internal renderer ids
    // resolve via lookup(). NB: parameter is tex_id, never `id` (ObjC keyword).
    std::vector<uint8_t> debug_readback_rgba8(uint64_t tex_id, int w, int h) override {
        @autoreleasepool {
            id<MTLTexture> t = lookup(tex_id);
            if (t == nil) {
                for (NSNumber* key in textures_) {
                    id<MTLTexture> cand = textures_[key];
                    if ((uint64_t)(__bridge void*)cand == tex_id) { t = cand; break; }
                }
            }
            if (t == nil || w <= 0 || h <= 0) return {};
            const NSUInteger bpr = (NSUInteger)w * 4;
            id<MTLBuffer> out = [device_ newBufferWithLength:bpr * (NSUInteger)h
                                                     options:MTLResourceStorageModeShared];
            if (out == nil) return {};
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:t
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
                         toBuffer:out
                destinationOffset:0
           destinationBytesPerRow:bpr
         destinationBytesPerImage:bpr * (NSUInteger)h];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];   // waits live in test readbacks, not the hot path
            std::vector<uint8_t> px((size_t)w * h * 4);
            std::memcpy(px.data(), out.contents, px.size());
            return px;
        }
    }

private:
    id<MTLTexture> lookup(uint64_t id) {
        if (id == 0) return nil;
        return textures_[@(id)];
    }

    id<MTLBuffer> lookup_import(uint64_t iid) {
        auto it = imported_.find(iid);
        return it == imported_.end() ? nil : it->second;
    }

    bool ensure_pipeline() {
        if (cmap_pipeline_ != nil) return true;
        NSError* err = nil;
        id<MTLLibrary> lib =
            [device_ newLibraryWithSource:[NSString stringWithUTF8String:kColormapShaderSrc]
                                  options:nil
                                    error:&err];
        if (lib == nil) return false;
        id<MTLFunction> fn = [lib newFunctionWithName:@"cmap_f32"];
        if (fn == nil) return false;
        cmap_pipeline_ = [device_ newComputePipelineStateWithFunction:fn error:&err];
        return cmap_pipeline_ != nil;
    }

    // Lazy point pipeline (once; released at shutdown). Additive ONE/ONE blend
    // on both channels, matching the Vulkan geom pipeline: a 1-px point at a
    // pixel center lands exactly the LUT color on top of the cleared background.
    bool ensure_points_pipeline() {
        if (points_pipeline_ != nil) return true;
        NSError* err = nil;
        id<MTLLibrary> lib =
            [device_ newLibraryWithSource:[NSString stringWithUTF8String:kPointsShaderSrc]
                                  options:nil error:&err];
        if (lib == nil) return false;
        id<MTLFunction> vs = [lib newFunctionWithName:@"points_vs"];
        id<MTLFunction> fs = [lib newFunctionWithName:@"points_fs"];
        if (vs == nil || fs == nil) return false;
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction   = vs;
        d.fragmentFunction = fs;
        d.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
        d.colorAttachments[0].pixelFormat         = MTLPixelFormatRGBA8Unorm;
        d.colorAttachments[0].blendingEnabled     = YES;   // additive ONE/ONE, both channels
        d.colorAttachments[0].rgbBlendOperation   = MTLBlendOperationAdd;
        d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        d.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorOne;
        d.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOne;
        d.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorOne;
        d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
        points_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        return points_pipeline_ != nil;
    }

    // M2b (spec §4): GPU-order this texture's update after the producer
    // queue's already-committed work. A tiny command buffer on the PRODUCER's
    // queue signals value v (queue order puts it after the producer's
    // committed kernels); the tensor-op command buffer waits v before any
    // encoder runs. No CPU block. If the event can't be created, fall back to
    // a CPU wait on the producer queue — slower, never silently unordered.
    void order_after_producer(uint64_t tex, id<MTLCommandBuffer> cb, void* stream) {
        id<MTLCommandQueue> producer = (__bridge id<MTLCommandQueue>)stream;
        if (producer == nil) return;
        id<MTLSharedEvent> ev = events_[@(tex)];
        if (ev == nil) {
            ev = [device_ newSharedEvent];
            if (ev != nil) events_[@(tex)] = ev;
        }
        id<MTLCommandBuffer> sig = [producer commandBuffer];
        if (ev != nil && sig != nil) {
            const uint64_t v = ++event_values_[tex];
            [sig encodeSignalEvent:ev value:v];
            [sig commit];
            [cb encodeWaitForEvent:ev value:v];
        } else if (sig != nil) {
            [sig commit];
            [sig waitUntilCompleted];   // rare fallback: CPU-ordered, still correct
        }
    }

    // f32 + LUT -> runtime-compiled compute shader. Sources from src at
    // src_offset bytes (0 on the direct path). Records "compute" (direct) or
    // "compute-imported" (v1.2 imported path).
    bool colormap_compute_from(uint64_t tex, id<MTLTexture> dst, id<MTLBuffer> src,
                               uint64_t src_offset, const CaliperTensor& t,
                               const uint32_t* lut256, float vmin, float vmax,
                               bool imported) {
        if (!ensure_pipeline()) return false;
        if (tensor_extent_bytes(t, sizeof(float)) > src.length - src_offset)
            return false;   // buffer too short for the declared extent at offset

        CmapParams p{};
        p.w = (uint32_t)dst.width;
        p.h = (uint32_t)dst.height;
        // Strides are in elements (§7.2). Fall back to row-major contiguous.
        p.sx = (t.ndim >= 1) ? (uint32_t)t.strides[t.ndim - 1] : 1u;
        p.sy = (t.ndim >= 2) ? (uint32_t)t.strides[t.ndim - 2] : p.w;
        p.vmin = vmin;
        p.vmax = vmax;

        id<MTLBuffer> lutbuf = [device_ newBufferWithBytes:lut256
                                                    length:256 * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        if (lutbuf == nil) return false;

        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (t.stream != nullptr) order_after_producer(tex, cb, t.stream);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cmap_pipeline_];
        [enc setBuffer:src offset:(NSUInteger)src_offset atIndex:0];
        [enc setBuffer:lutbuf offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc setTexture:dst atIndex:0];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize groups = MTLSizeMake((p.w + 15) / 16, (p.h + 15) / 16, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
        [enc endEncoding];
        // No CPU wait (M1/D23): the frame's command buffer commits AFTER this
        // one on the same queue_, so draw ordering is free by commit order; cb
        // retains src, lutbuf, and dst until the GPU retires it (default,
        // non-`unretained` encoding), so lifetime is free too. Test readbacks
        // retire the queue themselves (debug_readback_rgba8).
        [cb commit];

        last_device_path_ = imported ? "compute-imported" : "compute";
        return true;
    }

    // u8 HWC (RGBA8) -> blit straight into the texture. Sources from src at
    // src_offset bytes (0 on the direct path). Records "blit" (direct) or
    // "blit-imported" (v1.2 imported path).
    bool blit_u8_from(uint64_t tex, id<MTLTexture> dst, id<MTLBuffer> src,
                      uint64_t src_offset, const CaliperTensor& t, bool imported) {
        if (t.ndim < 2) return false;
        NSUInteger h = (NSUInteger)t.shape[0];
        NSUInteger w = (NSUInteger)t.shape[1];
        NSUInteger c = (t.ndim >= 3) ? (NSUInteger)t.shape[2] : 1;
        if (c != 4 || w != dst.width || h != dst.height) return false;  // RGBA8 only
        if (tensor_extent_bytes(t, 1) > src.length - src_offset)
            return false;   // buffer too short for the declared extent at offset

        NSUInteger bytesPerRow = w * 4;
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (t.stream != nullptr) order_after_producer(tex, cb, t.stream);
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src
                sourceOffset:(NSUInteger)src_offset
           sourceBytesPerRow:bytesPerRow
         sourceBytesPerImage:bytesPerRow * h
                  sourceSize:MTLSizeMake(w, h, 1)
                   toTexture:dst
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];   // no CPU wait (M1/D23): same-queue commit order + retention

        last_device_path_ = imported ? "blit-imported" : "blit";
        return true;
    }

    GLFWwindow*                 window_ = nullptr;
    id<MTLDevice>               device_ = nil;
    id<MTLCommandQueue>         queue_  = nil;
    CAMetalLayer*               layer_  = nil;
    MTLRenderPassDescriptor*    pass_desc_ = nil;
    id<MTLComputePipelineState> cmap_pipeline_ = nil;
    id<MTLRenderPipelineState>  points_pipeline_ = nil;  // caliper.geometry.v1 point pass

    // Per-frame transients created in new_frame(), consumed in render().
    id<CAMetalDrawable>          drawable_  = nil;
    id<MTLCommandBuffer>         frame_cmd_ = nil;
    id<MTLRenderCommandEncoder>  frame_enc_ = nil;

    NSMutableDictionary<NSNumber*, id<MTLTexture>>* textures_ = nil;

    // M2b: per-texture producer-ordering events (D23 — MTLSharedEvent appears
    // ONLY where cross-queue ordering genuinely exists). Values are a per-
    // texture monotonic timeline, the Metal analog of Vulkan's semaphores.
    NSMutableDictionary<NSNumber*, id<MTLSharedEvent>>* events_ = nil;
    std::unordered_map<uint64_t, uint64_t> event_values_;

    // v1.2 imported allocations: in-process MTLBuffers, strong-retained (ARC) —
    // the Metal analog of Vulkan's DuplicateHandle+VkImportMemory. 0 invalid.
    std::unordered_map<uint64_t, id<MTLBuffer>> imported_;
    uint64_t next_import_id_ = 1;

    uint64_t next_id_ = 1;          // 0 is the invalid id
    const char* last_device_path_ = "";
};

}  // namespace

std::unique_ptr<HostRenderer> make_metal_renderer() {
    return std::make_unique<MetalRenderer>();
}

}  // namespace caliper_host
