// Metal backend for the HostRenderer seam (docs/applets/rendering.md).
// make_metal_renderer() (declared in host_renderer.h) constructs it and
// core_select_renderer() chooses it on macOS. There is no fallback: if its
// start fails, the app, caliper shot and the embedding core refuse with
// start_error(). Apple-only translation unit.
//
// Frame ordering matches the C1 contract: the CLEAR happens in new_frame()
// (the Metal render pass is opened there with loadAction=Clear), the applet
// draws its ImGui frame, then render() flushes RenderDrawData and presents.
//
// Compiled with ARC (-fobjc-arc, set in CMakeLists): ObjC strong references
// live as members / in an internal id table; the uint64 texture ids handed to
// callers are sequential handles, never raw retained pointers (§5.4).
#include "host_renderer.h"
#include "gfx/gfx.h"
#include "gfx/metal_backend.h"
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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
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

    FrameWaits frame_waits() const override {
        FrameWaits w;
        w.drawable_ms = drawable_wait_ms_;
        w.drawable = had_drawable_;
        @autoreleasepool {
            NSWindow* win = window_ ? glfwGetCocoaWindow(window_) : nil;
            if (win != nil) w.occluded = ([win occlusionState] & NSWindowOcclusionStateVisible) ? 0 : 1;
        }
        return w;
    }
    const char* last_device_path() const override { return last_device_path_; }
    CaliperDeviceKind interop_device() const override { return CALIPER_DEV_METAL; }

    // M2b shipped: a non-NULL t.stream is GPU-ordered after the producer
    // queue in both device paths above.
    bool honors_stream_ordered_handoff() const override { return true; }

    // Metal owns no GL context: the window must be created with NO_API so GLFW
    // does not attach an OpenGL context we would fight over. Runs before
    // glfwCreateWindow.
    void window_hints() override {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }

    // ---- Embed canvas seam (libcaliper R4 L2a) ---------------------------
    // Parallel to init()/new_frame()/render(): a GLFW-free path that composites
    // the applet's ImGui frame onto an offscreen RGBA8 texture (readable) or a
    // CAMetalLayer attached to the host's native NSView. Shares device_/queue_/
    // textures_ with the texture ops, so the bridge works unchanged.
    // The ImGui *render* backend (imgui_impl_metal) is wired here; the *platform*
    // half is NOT (the embedder feeds io via caliper_core_event). Only ONE of
    // init() / canvas_init() is ever called on a given instance.
    bool canvas_init(void* native_view, CanvasMode mode, int w, int h) override {
        if (forced_start_failure()) return false;
        if (w <= 0 || h <= 0) return start_failed("the canvas size is not positive");
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) return start_failed("no Metal device");
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) { device_ = nil; return start_failed("no Metal command queue"); }

        canvas_mode_ = mode;
        canvas_w_ = w;
        canvas_h_ = h;

        if (mode == CANVAS_OFFSCREEN) {
            if (!make_canvas_target(w, h)) {
                queue_ = nil; device_ = nil;
                return start_failed("no " + std::to_string(w) + "x" + std::to_string(h) +
                                    " offscreen target");
            }
        } else {
            // Window mode: attach a CAMetalLayer to the host's native NSView —
            // the embed analog of init()'s glfwGetCocoaWindow path, but from a
            // view the EMBEDDER owns (no GLFW). BGRA8 to match a display layer.
            NSView* view = (__bridge NSView*)native_view;
            if (view == nil) { queue_ = nil; device_ = nil; return start_failed("no view to draw into"); }
            canvas_layer_ = [CAMetalLayer layer];
            canvas_layer_.device = device_;
            canvas_layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
            canvas_layer_.framebufferOnly = YES;
            canvas_layer_.drawableSize = CGSizeMake((CGFloat)w, (CGFloat)h);
            // HiDPI (L2b W2): the host passes w/h in PHYSICAL pixels, so the
            // layer's contentsScale must be the backing scale for the drawable
            // to map 1:1 — leaving the CALayer default (1.0) makes a Retina
            // window composite the physical-size drawable into a point-size
            // layer and present it quarter-scale / blurry. AppKit does NOT set
            // contentsScale on a layer-hosting view (unlike a layer-BACKED one),
            // so we set it from the window the host already parented the view in.
            CGFloat backing = view.window ? view.window.backingScaleFactor
                                          : (w > 0 && view.bounds.size.width > 0
                                                 ? (CGFloat)w / view.bounds.size.width
                                                 : 1.0);
            if (backing <= 0.0) backing = 1.0;
            canvas_layer_.contentsScale = backing;
            canvas_layer_.frame = view.bounds;
            view.layer = canvas_layer_;
            view.wantsLayer = YES;
        }

        // Host-owned ImGui/ImPlot contexts already exist (core_create_ui_context);
        // wire ONLY the Metal render backend. No ImGui_ImplGlfw here.
        if (!ImGui_ImplMetal_Init(device_)) {
            canvas_layer_ = nil; canvas_tex_ = nil; queue_ = nil; device_ = nil;
            return start_failed("ImGui's Metal backend did not start");
        }

        textures_ = [NSMutableDictionary dictionary];
        events_ = [NSMutableDictionary dictionary];
        pass_desc_ = [MTLRenderPassDescriptor new];
        gfx_ = caliper_gfx::make_metal_backend((__bridge void*)device_, (__bridge void*)queue_);
        canvas_ready_ = true;
        return true;
    }

    void canvas_resize(int w, int h) override {
        if (!canvas_ready_ || w <= 0 || h <= 0) return;
        canvas_w_ = w;
        canvas_h_ = h;
        if (canvas_mode_ == CANVAS_OFFSCREEN)
            make_canvas_target(w, h);
        else if (canvas_layer_ != nil)
            canvas_layer_.drawableSize = CGSizeMake((CGFloat)w, (CGFloat)h);
    }

    // Start-of-frame: acquire the target, open the render pass with a CLEAR
    // (== the exe backend's background), then ImGui NewFrame. Mirrors new_frame()
    // exactly, minus the GLFW platform NewFrame. Anything drawn this frame
    // lands on top.
    void canvas_new_frame() override {
        if (!canvas_ready_) return;
        @autoreleasepool {
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            if (canvas_mode_ == CANVAS_OFFSCREEN) {
                rp.colorAttachments[0].texture = canvas_tex_;
            } else {
                canvas_drawable_ = [canvas_layer_ nextDrawable];   // may be nil under load
                rp.colorAttachments[0].texture =
                    canvas_drawable_ ? canvas_drawable_.texture : nil;
            }
            rp.colorAttachments[0].loadAction = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0);

            canvas_cmd_ = [queue_ commandBuffer];
            if (canvas_mode_ == CANVAS_OFFSCREEN || canvas_drawable_ != nil)
                canvas_enc_ = [canvas_cmd_ renderCommandEncoderWithDescriptor:rp];
            else
                canvas_enc_ = nil;   // window with no drawable; keep frame balanced

            ImGui_ImplMetal_NewFrame(rp);
            ImGui::NewFrame();
        }
    }

    void canvas_render() override {
        @autoreleasepool {
            ImGui::Render();   // consume the frame regardless (matches render())
            if (!canvas_ready_) return;
            if (canvas_enc_ != nil) {
                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
                                               canvas_cmd_, canvas_enc_);
                [canvas_enc_ endEncoding];
                if (canvas_mode_ == CANVAS_WINDOW && canvas_drawable_ != nil)
                    [canvas_cmd_ presentDrawable:canvas_drawable_];
                stats_gpu_time(canvas_cmd_, kGpuFrame);
                [canvas_cmd_ commit];
                // Offscreen readback must observe THIS frame, so drain it (no
                // swapchain to pace us). Window mode presents and rides on.
                if (canvas_mode_ == CANVAS_OFFSCREEN)
                    [canvas_cmd_ waitUntilCompleted];
            } else if (canvas_cmd_ != nil) {
                [canvas_cmd_ commit];   // nothing to draw; flush
            }
            canvas_enc_ = nil;
            canvas_cmd_ = nil;
            canvas_drawable_ = nil;
            stats_frame();
        }
    }

    bool canvas_read_pixels(uint8_t* dst, int stride) override {
        if (!canvas_ready_ || canvas_mode_ != CANVAS_OFFSCREEN) return false;
        if (canvas_tex_ == nil || dst == nullptr) return false;
        const int w = canvas_w_, h = canvas_h_;
        const int tight = w * 4;
        if (stride < tight) return false;
        @autoreleasepool {
            const NSUInteger bpr = (NSUInteger)w * 4;
            id<MTLBuffer> out = [device_ newBufferWithLength:bpr * (NSUInteger)h
                                                     options:MTLResourceStorageModeShared];
            if (out == nil) return false;
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:canvas_tex_
                      sourceSlice:0 sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
                         toBuffer:out destinationOffset:0
           destinationBytesPerRow:bpr
         destinationBytesPerImage:bpr * (NSUInteger)h];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const uint8_t* src = (const uint8_t*)out.contents;
            for (int y = 0; y < h; ++y)
                std::memcpy(dst + (size_t)y * (size_t)stride,
                            src + (size_t)y * (size_t)tight, (size_t)tight);
            return true;
        }
    }

    void canvas_shutdown() override {
        if (!canvas_ready_) return;
        stats_print();
        gfx_.reset();   // waits for its last submit; nothing of it may outlive the queue
        ImGui_ImplMetal_Shutdown();
        [events_ removeAllObjects]; events_ = nil; event_values_.clear();
        [textures_ removeAllObjects]; textures_ = nil;
        imported_.clear();
        last_update_cb_ = nil;
        cmap_pipeline_ = nil;
        pass_desc_ = nil;
        canvas_tex_ = nil;
        canvas_layer_ = nil;
        canvas_cmd_ = nil;
        canvas_enc_ = nil;
        canvas_drawable_ = nil;
        queue_ = nil;
        device_ = nil;
        canvas_ready_ = false;
    }

    bool init(GLFWwindow* window) override {
        if (forced_start_failure()) return false;
        window_ = window;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) return start_failed("no Metal device");
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) return start_failed("no Metal command queue");

        NSWindow* nswin = glfwGetCocoaWindow(window);
        if (nswin == nil) return start_failed("the window has no Cocoa window");
        layer_ = [CAMetalLayer layer];
        layer_.device = device_;
        layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer_.framebufferOnly = YES;
        nswin.contentView.layer = layer_;
        nswin.contentView.wantsLayer = YES;

        // ImGui core/ImPlot contexts are host-owned and already created; only
        // the GLFW + Metal *backends* belong to this renderer.
        if (!ImGui_ImplGlfw_InitForOther(window, true))
            return start_failed("ImGui's GLFW backend did not start");
        if (!ImGui_ImplMetal_Init(device_)) {
            ImGui_ImplGlfw_Shutdown();   // undo the GLFW backend before the refusal
            return start_failed("ImGui's Metal backend did not start");
        }

        textures_ = [NSMutableDictionary dictionary];
        events_ = [NSMutableDictionary dictionary];
        pass_desc_ = [MTLRenderPassDescriptor new];
        gfx_ = caliper_gfx::make_metal_backend((__bridge void*)device_, (__bridge void*)queue_);
        return true;
    }

    // Start-of-frame: size the layer, acquire a drawable, open the render pass
    // with a CLEAR matching the GL backend's background, then ImGui NewFrame.
    // The clear lives here (not render()) per the C1 contract — anything the
    // app draws this frame lands on top of it.
    void new_frame() override { new_frame_with_input({}); }

    void new_frame_with_input(const std::function<void()>& input) override {
        @autoreleasepool {
            int w = 0, h = 0;
            glfwGetFramebufferSize(window_, &w, &h);
            if (w > 0 && h > 0)
                layer_.drawableSize = CGSizeMake((CGFloat)w, (CGFloat)h);

            const auto wait0 = std::chrono::steady_clock::now();
            drawable_ = [layer_ nextDrawable];   // may be nil under load
            drawable_wait_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait0).count();
            had_drawable_ = drawable_ != nil;

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
            if (input) input();
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
                copy_for_capture();                // before present, in the same command buffer
                [frame_cmd_ presentDrawable:drawable_];
                stats_gpu_time(frame_cmd_, kGpuFrame);
                [frame_cmd_ commit];
            } else if (frame_cmd_) {
                [frame_cmd_ commit];               // nothing to draw; flush
            }
            frame_enc_ = nil;
            frame_cmd_ = nil;
            drawable_ = nil;
            stats_frame();
        }
    }

    // ---- Window capture: the presented frame, copied in the frame's own
    // command buffer after the ImGui pass and before present, never waited
    // for on the frame thread. ----
    bool window_capture_supported() const override { return layer_ != nil; }

    bool window_capture_request() override {
        if (layer_ == nil) return false;
        const int s = cap_state_->load();
        if (s == kCapArmed || s == kCapCopying) return true;   // the next result serves it
        cap_buf_ = nil;
        if (!cap_readable_) {
            layer_.framebufferOnly = NO;   // drawables from now on can be a blit source
            cap_readable_ = true;
        }
        cap_state_->store(kCapArmed);
        return true;
    }

    WindowCapture window_capture_poll(std::vector<uint8_t>* rgba, int* w, int* h) override {
        const int s = cap_state_->load();
        if (s == kCapNone) return WindowCapture::None;
        if (s == kCapArmed || s == kCapCopying) return WindowCapture::Pending;
        if (s == kCapFailed || cap_buf_ == nil) {
            cap_buf_ = nil;
            cap_state_->store(kCapNone);
            return WindowCapture::Failed;
        }
        // BGRA8Unorm (the layer's format) to opaque RGBA8.
        const size_t n = (size_t)cap_w_ * (size_t)cap_h_;
        const uint8_t* src = (const uint8_t*)cap_buf_.contents;
        rgba->resize(n * 4);
        uint8_t* dst = rgba->data();
        for (size_t i = 0; i < n; ++i) {
            dst[4 * i + 0] = src[4 * i + 2];
            dst[4 * i + 1] = src[4 * i + 1];
            dst[4 * i + 2] = src[4 * i + 0];
            dst[4 * i + 3] = 255;
        }
        *w = cap_w_;
        *h = cap_h_;
        cap_buf_ = nil;
        cap_state_->store(kCapNone);
        return WindowCapture::Ready;
    }

    // In render(), for an armed capture: blit this frame's drawable into a
    // shared buffer before it is presented. A drawable still framebuffer-only
    // (made before the request) leaves the capture armed for the next frame.
    void copy_for_capture() {
        if (cap_state_->load() != kCapArmed || drawable_ == nil) return;
        id<MTLTexture> t = drawable_.texture;
        if (t == nil || t.framebufferOnly || t.pixelFormat != MTLPixelFormatBGRA8Unorm) return;
        const NSUInteger w = t.width, h = t.height, bpr = w * 4;
        id<MTLBuffer> buf = [device_ newBufferWithLength:bpr * h options:MTLResourceStorageModeShared];
        if (buf == nil) {
            cap_state_->store(kCapFailed);
            return;
        }
        id<MTLBlitCommandEncoder> blit = [frame_cmd_ blitCommandEncoder];
        [blit copyFromTexture:t
                  sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(w, h, 1)
                     toBuffer:buf destinationOffset:0
       destinationBytesPerRow:bpr
     destinationBytesPerImage:bpr * h];
        [blit endEncoding];
        cap_buf_ = buf;
        cap_w_ = (int)w;
        cap_h_ = (int)h;
        cap_state_->store(kCapCopying);
        std::shared_ptr<std::atomic<int>> state = cap_state_;
        [frame_cmd_ addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            state->store(cb.status == MTLCommandBufferStatusCompleted ? kCapReady : kCapFailed);
        }];
    }

    void shutdown() override {
        stats_print();
        gfx_.reset();   // waits for its last submit; nothing of it may outlive the queue
        cap_buf_ = nil;
        cap_state_->store(kCapNone);
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        [events_ removeAllObjects]; events_ = nil; event_values_.clear();
        [textures_ removeAllObjects];
        textures_ = nil;
        imported_.clear();
        last_update_cb_ = nil;
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
        stats_update("cpu-staged");   // the copy is counted by the caller that staged it
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

    // Embed v1.2 views (CALIPER_VIEW_FLAG_WAIT_GPU): CPU-wait the last device
    // update's command buffer, then drop the reference (it was retaining the
    // source buffer + LUT until retirement anyway). nil-safe: no pending update
    // -> immediate return. Only callers that ASK pay the wait.
    void device_update_wait() override {
        if (last_update_cb_ != nil) {
            [last_update_cb_ waitUntilCompleted];
            last_update_cb_ = nil;
        }
    }

    // Live views: a +1 handle on the most recent update's command buffer, polled
    // with status (never waited on) and released by the caller.
    void* device_update_retain_last() override {
        return last_update_cb_ != nil ? (__bridge_retained void*)last_update_cb_ : nullptr;
    }
    bool device_update_is_done(void* h) const override {
        if (h == nullptr) return true;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)h;
        return cb.status >= MTLCommandBufferStatusCompleted;   // Completed or Error
    }
    void device_update_release(void* h) override {
        if (h != nullptr) { id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)h; (void)cb; }
    }
    void* native_device() const override { return (__bridge void*)device_; }
    void* native_queue() const override { return (__bridge void*)queue_; }
    caliper_gfx::Backend* gfx() override { return gfx_.get(); }
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

    // Test-only (spec §3.4): copy a texture back on the RENDERER's own
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

    // CALIPER_METAL_STATS: the window's frame times, GPU times and updates by
    // path (HostRenderer::FrameStats). false while the variable is unset.
    bool frame_stats(FrameStats* out, bool reset_window) override {
        if (!stats_on_ || out == nullptr) return false;
        StatSamples gpu;
        {
            std::lock_guard<std::mutex> lk(gpu_times_->mu);
            gpu = gpu_times_->window;
            if (reset_window) gpu_times_->window = StatSamples{};
        }
        *out = FrameStats{};
        out->frames = stats_window_.frames;
        stats_fill(stats_window_.frame_ms, &out->frame_ms_p50, &out->frame_ms_p99, &out->frame_ms_max);
        out->update_gpu_n = stats_fill(gpu.update_gpu_ms, &out->update_gpu_ms_p50, &out->update_gpu_ms_p99,
                                       &out->update_gpu_ms_max);
        out->draw_gpu_n = stats_fill(gpu.draw_gpu_ms, &out->draw_gpu_ms_p50, &out->draw_gpu_ms_p99,
                                     &out->draw_gpu_ms_max);
        out->frame_gpu_n = stats_fill(gpu.frame_gpu_ms, &out->frame_gpu_ms_p50, &out->frame_gpu_ms_p99,
                                      &out->frame_gpu_ms_max);
        for (const auto& kv : stats_window_.updates) out->updates.emplace_back(kv.first, kv.second);
        if (reset_window) stats_window_ = StatSamples{};
        return true;
    }

    void note_draw_gpu_ms(double ms, uint32_t submits) override {
        if (!stats_on_ || !(ms > 0.0) || submits == 0) return;
        const float each = (float)(ms / (double)submits);
        std::lock_guard<std::mutex> lk(gpu_times_->mu);
        for (uint32_t k = 0; k < submits; ++k) {
            gpu_times_->window.draw_gpu_ms.push_back(each);
            if (gpu_times_->run.draw_gpu_ms.size() < kStatsRunCap) gpu_times_->run.draw_gpu_ms.push_back(each);
        }
    }

private:
    // ---- CALIPER_METAL_STATS (docs/applets/rendering.md, "Copy Counters
    // and Frame Statistics"): frame times, GPU times from the command buffers'
    // GPUStartTime and GPUEndTime, and updates by path, over a window that
    // frame_stats() resets and over the whole run, which shutdown prints in one
    // line beside the renderer's copy count. Read once at construction; while
    // unset nothing is recorded and no completion handler is added. ----
    struct StatSamples {
        std::vector<float> frame_ms, update_gpu_ms, draw_gpu_ms, frame_gpu_ms;
        std::map<std::string, uint64_t> updates;
        uint64_t frames = 0;
    };
    // GPU times arrive on Metal's completion threads: guarded, and shared with
    // the handlers so that a late one never touches a destroyed renderer.
    struct GpuTimes {
        std::mutex mu;
        StatSamples window, run;   // only the *_gpu_ms vectors are used
    };
    enum GpuKind { kGpuUpdate, kGpuDraw, kGpuFrame };
    static constexpr size_t kStatsRunCap = size_t(1) << 20;   // samples kept per series for the run

    static std::vector<float>& gpu_series(StatSamples& s, GpuKind kind) {
        return kind == kGpuUpdate ? s.update_gpu_ms : kind == kGpuDraw ? s.draw_gpu_ms : s.frame_gpu_ms;
    }
    // Before the command buffer is committed.
    void stats_gpu_time(id<MTLCommandBuffer> cb, GpuKind kind) {
        if (!stats_on_ || cb == nil) return;
        std::shared_ptr<GpuTimes> g = gpu_times_;
        [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
            const double ms = (done.GPUEndTime - done.GPUStartTime) * 1000.0;
            if (!(ms > 0.0)) return;   // unknown (0) or failed
            std::lock_guard<std::mutex> lk(g->mu);
            gpu_series(g->window, kind).push_back((float)ms);
            std::vector<float>& run = gpu_series(g->run, kind);
            if (run.size() < kStatsRunCap) run.push_back((float)ms);
        }];
    }
    void stats_update(const char* path) {
        if (!stats_on_ || path == nullptr) return;
        ++stats_window_.updates[path];
        ++stats_run_.updates[path];
    }
    // At the end of every frame (render, canvas_render).
    void stats_frame() {
        if (!stats_on_) return;
        const auto now = std::chrono::steady_clock::now();
        if (stats_last_frame_ != std::chrono::steady_clock::time_point{}) {
            const float ms = std::chrono::duration<float, std::milli>(now - stats_last_frame_).count();
            stats_window_.frame_ms.push_back(ms);
            if (stats_run_.frame_ms.size() < kStatsRunCap) stats_run_.frame_ms.push_back(ms);
        }
        stats_last_frame_ = now;
        ++stats_window_.frames;
        ++stats_run_.frames;
    }
    // Nearest-rank p50, p99 and maximum of v (as CALIPER_VULKAN_STATS); the count.
    static uint64_t stats_fill(std::vector<float> v, double* p50, double* p99, double* mx) {
        if (v.empty()) { *p50 = *p99 = *mx = 0.0; return 0; }
        std::sort(v.begin(), v.end());
        const auto at = [&](double q) { return (double)v[(size_t)(q * (double)(v.size() - 1))]; };
        *p50 = at(0.5);
        *p99 = at(0.99);
        *mx = v.back();
        return v.size();
    }
    // The run's summary, once, at shutdown.
    void stats_print() {
        if (!stats_on_ || stats_printed_) return;
        stats_printed_ = true;
        StatSamples gpu;
        {
            std::lock_guard<std::mutex> lk(gpu_times_->mu);
            gpu = gpu_times_->run;
        }
        double f50, f99, fmx, u50, u99, umx, d50, d99, dmx, g50, g99, gmx;
        stats_fill(stats_run_.frame_ms, &f50, &f99, &fmx);
        const uint64_t un = stats_fill(gpu.update_gpu_ms, &u50, &u99, &umx);
        const uint64_t dn = stats_fill(gpu.draw_gpu_ms, &d50, &d99, &dmx);
        const uint64_t gn = stats_fill(gpu.frame_gpu_ms, &g50, &g99, &gmx);
        std::string upd;
        for (const auto& kv : stats_run_.updates)
            upd += " " + kv.first + " " + std::to_string(kv.second);
        const CopyCount c = copy_count();
        std::fprintf(stderr,
                     "[metal] stats: %llu frames, ms p50 %.2f p99 %.2f max %.2f; updates%s; "
                     "gpu ms update p50 %.3f p99 %.3f max %.3f (%llu), view draw p50 %.3f p99 %.3f "
                     "max %.3f (%llu), frame p50 %.3f p99 %.3f max %.3f (%llu); copies %llu (%llu bytes)\n",
                     (unsigned long long)stats_run_.frames, f50, f99, fmx, upd.empty() ? " none" : upd.c_str(),
                     u50, u99, umx, (unsigned long long)un, d50, d99, dmx, (unsigned long long)dn, g50, g99,
                     gmx, (unsigned long long)gn, (unsigned long long)c.copies, (unsigned long long)c.bytes);
    }

    const bool stats_on_ = std::getenv("CALIPER_METAL_STATS") != nullptr;
    bool stats_printed_ = false;
    StatSamples stats_window_, stats_run_;   // frame thread: frame times, updates, frames
    std::shared_ptr<GpuTimes> gpu_times_ = std::make_shared<GpuTimes>();
    std::chrono::steady_clock::time_point stats_last_frame_{};

    // Offscreen canvas target: renderable AND host-readable (unified memory).
    // RGBA8 so canvas_read_pixels hands back RGBA with no swizzle. Rebuilt on
    // resize; the old texture is ARC-released.
    bool make_canvas_target(int w, int h) {
        MTLTextureDescriptor* d =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:(NSUInteger)w
                                                              height:(NSUInteger)h
                                                           mipmapped:NO];
        d.storageMode = MTLStorageModeShared;
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        canvas_tex_ = [device_ newTextureWithDescriptor:d];
        return canvas_tex_ != nil;
    }

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
        // No CPU wait: the frame's command buffer commits AFTER this
        // one on the same queue_, so draw ordering is free by commit order; cb
        // retains src, lutbuf, and dst until the GPU retires it (default,
        // non-`unretained` encoding), so lifetime is free too. Test readbacks
        // retire the queue themselves (debug_readback_rgba8).
        stats_gpu_time(cb, kGpuUpdate);
        [cb commit];
        last_update_cb_ = cb;   // device_update_wait() target (never waited here)

        last_device_path_ = imported ? "compute-imported" : "compute";
        stats_update(last_device_path_);
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
        stats_gpu_time(cb, kGpuUpdate);
        [cb commit];   // no CPU wait: same-queue commit order + retention
        last_update_cb_ = cb;   // device_update_wait() target (never waited here)

        last_device_path_ = imported ? "blit-imported" : "blit";
        stats_update(last_device_path_);
        return true;
    }

    GLFWwindow*                 window_ = nullptr;
    double                      drawable_wait_ms_ = -1.0;   // the last new_frame()'s wait for its drawable
    bool                        had_drawable_ = true;
    std::unique_ptr<caliper_gfx::Backend> gfx_;   // the graphics layer on device_ and queue_
    id<MTLDevice>               device_ = nil;
    id<MTLCommandQueue>         queue_  = nil;
    CAMetalLayer*               layer_  = nil;
    MTLRenderPassDescriptor*    pass_desc_ = nil;
    id<MTLComputePipelineState> cmap_pipeline_ = nil;

    // Window capture (window_capture_*). The state is shared with the copy's
    // completion handler, which may run after a frame and on another thread:
    // 0 none, 1 armed (the next presented frame is copied), 2 copying,
    // 3 ready, 4 failed. framebufferOnly is turned off for good on the first
    // request, since a drawable that is framebuffer-only cannot be copied.
    enum : int { kCapNone = 0, kCapArmed = 1, kCapCopying = 2, kCapReady = 3, kCapFailed = 4 };
    std::shared_ptr<std::atomic<int>> cap_state_ = std::make_shared<std::atomic<int>>(kCapNone);
    id<MTLBuffer>                cap_buf_ = nil;
    int                          cap_w_ = 0, cap_h_ = 0;
    bool                         cap_readable_ = false;

    // Per-frame transients created in new_frame(), consumed in render().
    id<CAMetalDrawable>          drawable_  = nil;
    id<MTLCommandBuffer>         frame_cmd_ = nil;
    id<MTLRenderCommandEncoder>  frame_enc_ = nil;

    // Embed canvas state (L2a). Populated only on the canvas_init() path; the
    // swapchain members above stay nil there and vice-versa.
    bool                         canvas_ready_ = false;
    CanvasMode                   canvas_mode_  = CANVAS_OFFSCREEN;
    int                          canvas_w_ = 0, canvas_h_ = 0;
    id<MTLTexture>               canvas_tex_ = nil;    // offscreen target (readable)
    CAMetalLayer*                canvas_layer_ = nil;  // window mode
    id<MTLCommandBuffer>         canvas_cmd_ = nil;
    id<MTLRenderCommandEncoder>  canvas_enc_ = nil;
    id<CAMetalDrawable>          canvas_drawable_ = nil;

    NSMutableDictionary<NSNumber*, id<MTLTexture>>* textures_ = nil;

    // M2b: per-texture producer-ordering events (MTLSharedEvent appears
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
    // Most recent device texture-update command buffer (compute/blit, direct or
    // imported) — what device_update_wait() blocks on. Strong (ARC); cleared by
    // the wait and on shutdown.
    id<MTLCommandBuffer> last_update_cb_ = nil;
};

}  // namespace

std::unique_ptr<HostRenderer> make_metal_renderer() {
    return std::make_unique<MetalRenderer>();
}

}  // namespace caliper_host
