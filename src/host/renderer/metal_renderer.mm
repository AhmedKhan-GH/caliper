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

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_metal.h>

#include <cstdio>

namespace caliper_host {
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
        [textures_ removeAllObjects];
        textures_ = nil;
        cmap_pipeline_ = nil;
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
            return colormap_compute(dst, src, t, lut256, vmin, vmax);
        if (t.dtype == CALIPER_DT_U8)
            return blit_u8(dst, src, t);
        return false;
    }

private:
    id<MTLTexture> lookup(uint64_t id) {
        if (id == 0) return nil;
        return textures_[@(id)];
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

    // f32 + LUT -> runtime-compiled compute shader. Records "compute".
    bool colormap_compute(id<MTLTexture> dst, id<MTLBuffer> src,
                          const CaliperTensor& t, const uint32_t* lut256,
                          float vmin, float vmax) {
        if (!ensure_pipeline()) return false;
        if (src.length < tensor_extent_bytes(t, sizeof(float)))
            return false;   // buffer too short for the declared extent

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
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cmap_pipeline_];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:lutbuf offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc setTexture:dst atIndex:0];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize groups = MTLSizeMake((p.w + 15) / 16, (p.h + 15) / 16, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];   // texture must be ready for the frame/readback

        last_device_path_ = "compute";
        return true;
    }

    // u8 HWC (RGBA8) -> blit straight into the texture. Records "blit".
    bool blit_u8(id<MTLTexture> dst, id<MTLBuffer> src, const CaliperTensor& t) {
        if (t.ndim < 2) return false;
        NSUInteger h = (NSUInteger)t.shape[0];
        NSUInteger w = (NSUInteger)t.shape[1];
        NSUInteger c = (t.ndim >= 3) ? (NSUInteger)t.shape[2] : 1;
        if (c != 4 || w != dst.width || h != dst.height) return false;  // RGBA8 only
        if (src.length < tensor_extent_bytes(t, 1))
            return false;   // buffer too short for the declared extent

        NSUInteger bytesPerRow = w * 4;
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src
                sourceOffset:0
           sourceBytesPerRow:bytesPerRow
         sourceBytesPerImage:bytesPerRow * h
                  sourceSize:MTLSizeMake(w, h, 1)
                   toTexture:dst
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        last_device_path_ = "blit";
        return true;
    }

    GLFWwindow*                 window_ = nullptr;
    id<MTLDevice>               device_ = nil;
    id<MTLCommandQueue>         queue_  = nil;
    CAMetalLayer*               layer_  = nil;
    MTLRenderPassDescriptor*    pass_desc_ = nil;
    id<MTLComputePipelineState> cmap_pipeline_ = nil;

    // Per-frame transients created in new_frame(), consumed in render().
    id<CAMetalDrawable>          drawable_  = nil;
    id<MTLCommandBuffer>         frame_cmd_ = nil;
    id<MTLRenderCommandEncoder>  frame_enc_ = nil;

    NSMutableDictionary<NSNumber*, id<MTLTexture>>* textures_ = nil;
    uint64_t next_id_ = 1;          // 0 is the invalid id
    const char* last_device_path_ = "";
};

}  // namespace

std::unique_ptr<HostRenderer> make_metal_renderer() {
    return std::make_unique<MetalRenderer>();
}

}  // namespace caliper_host
