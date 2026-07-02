// caliper_gfx_tests — windowed, pixel-exact proof of the tensor->texture path
// (PLATFORM.md §16). NOT part of the headless ctest default; registered under
// the "gfx" label and requires a GUI session. If glfwInit() fails (headless
// CI), every case is reported skipped so the label stays green rather than red.
//
// This runs the SAME §16 matrix on BOTH backends: the GL run and (on Apple, when
// a Metal device is available) the Metal run. Both compare the GPU readback
// byte-for-byte against the shared CPU reference (map_f32_to_rgba8 /
// expand_u8_to_rgba8) — the single source of truth the staging path also uses.
//
// Metal additionally exercises the DEVICE paths that nothing had ever run: a
// METAL-device CaliperTensor is built from a raw MTLBuffer allocated *in the
// test* (the C-ABI contract is data = bridge-cast id<MTLBuffer>; no torch here),
// then f32+LUT must take the "compute" path and u8 HWC the "blit" path, each
// pixel-exact vs the identical CPU reference. Readback is test-only: a Metal
// blit texture->shared-buffer + waitUntilCompleted (never the render path).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include "tensor_bridge.h"
#include "renderer/host_renderer.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>
#endif

using namespace caliper_host;

namespace {

// One GLFW init/terminate for the whole binary regardless of how many backend
// envs come up — each env owns only its window + ImGui context. A function-local
// guard is initialized by the first env that constructs, so it tears down after
// every env (reverse static-destruction order), avoiding a double-terminate.
struct GlfwGuard {
    bool ok;
    GlfwGuard() : ok(glfwInit() == GLFW_TRUE) {}
    ~GlfwGuard() { if (ok) glfwTerminate(); }
};
GlfwGuard& glfw_guard() { static GlfwGuard g; return g; }

// ---- GL backend env: hidden GL window + GLRenderer + TensorBridge ----------
struct GlEnv {
    bool ok = false;
    GLFWwindow* window = nullptr;
    ImGuiContext* imgui_ctx = nullptr;
    std::unique_ptr<HostRenderer> renderer;
    std::unique_ptr<TensorBridge> bridge;

    GlEnv() {
        if (!glfw_guard().ok) return;             // headless -> ok stays false
        renderer = make_renderer("gl");
        glfwDefaultWindowHints();                 // clear any prior backend hints
        renderer->window_hints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(64, 64, "caliper_gfx_tests(gl)", nullptr, nullptr);
        if (!window) return;
        imgui_ctx = ImGui::CreateContext();
        ImGui::SetCurrentContext(imgui_ctx);
        if (!renderer->init(window)) {
            ImGui::DestroyContext(imgui_ctx);
            glfwDestroyWindow(window);
            return;
        }
        bridge = std::make_unique<TensorBridge>(*renderer);
        ok = true;
    }
    ~GlEnv() {
        if (!ok) return;
        ImGui::SetCurrentContext(imgui_ctx);
        bridge.reset();
        renderer->shutdown();
        ImGui::DestroyContext(imgui_ctx);
        glfwDestroyWindow(window);
    }
};

GlEnv& gl_env() { static GlEnv e; return e; }

// Read an RGBA8 texture back off the GL GPU. The bridge id maps (via the
// renderer) to the GL name; the raw handle never escaped the renderer (§5.4).
std::vector<uint8_t> gl_readback(HostRenderer& r, CaliperTextureId id, int w, int h) {
    GLuint name = (GLuint)r.tex_imtexture_id(id);
    std::vector<uint8_t> px((size_t)w * h * 4, 0xAB);
    glBindTexture(GL_TEXTURE_2D, name);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return px;
}

// ---- CPU-tensor builders (shared by both backends) -------------------------
CaliperTensor f32_2d(const float* d, int64_t h, int64_t w) {
    CaliperTensor t{};
    t.struct_size = sizeof(t); t.data = (void*)d; t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_CPU; return t;
}
CaliperTensor u8_3d(const uint8_t* d, int64_t h, int64_t w, int64_t c) {
    CaliperTensor t{};
    t.struct_size = sizeof(t); t.data = (void*)d; t.dtype = CALIPER_DT_U8;
    t.ndim = 3; t.shape[0] = h; t.shape[1] = w; t.shape[2] = c;
    t.strides[0] = w * c; t.strides[1] = c; t.strides[2] = 1;
    t.device = CALIPER_DEV_CPU; return t;
}

// A backend under test: its bridge + renderer + a readback for verification.
struct Backend {
    TensorBridge* bridge = nullptr;
    HostRenderer* renderer = nullptr;
    std::function<std::vector<uint8_t>(CaliperTextureId, int, int)> readback;
};

Backend gl_backend() {
    Backend b;
    b.bridge = gl_env().bridge.get();
    b.renderer = gl_env().renderer.get();
    HostRenderer* r = b.renderer;
    b.readback = [r](CaliperTextureId id, int w, int h) {
        return gl_readback(*r, id, w, h);
    };
    return b;
}

// ===========================================================================
// Shared §16 CPU-tensor matrix — identical assertions on every backend.
// ===========================================================================
void mat_f32_viridis(Backend& bk) {
    float ramp[16];
    for (int i = 0; i < 16; ++i) ramp[i] = (float)i;   // 0..15
    CaliperTensor t = f32_2d(ramp, 4, 4);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, 15.0f, 0);
    REQUIRE(id != 0);
    std::vector<uint8_t> ref(16 * 4);
    map_f32_to_rgba8(ramp, 4, 4, colormap_lut(CALIPER_CMAP_VIRIDIS), 0.0f, 15.0f, ref.data());
    CHECK(bk.readback(id, 4, 4) == ref);
    bk.bridge->release_texture(id);
}

void mat_f32_magma_rdbu(Backend& bk) {
    float v[16];
    for (int i = 0; i < 16; ++i) v[i] = -1.0f + 2.0f * (i / 15.0f);  // -1..1
    CaliperTensor t = f32_2d(v, 4, 4);
    for (int cm : {CALIPER_CMAP_MAGMA, CALIPER_CMAP_RDBU}) {
        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(&t, cm, -1.0f, 1.0f, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(16 * 4);
        map_f32_to_rgba8(v, 4, 4, colormap_lut(cm), -1.0f, 1.0f, ref.data());
        CHECK(bk.readback(id, 4, 4) == ref);
        bk.bridge->release_texture(id);
    }
}

void mat_u8_direct(Backend& bk) {
    const int h = 2, w = 3;
    {   // C=1 gray replicate
        uint8_t g[6]; for (int i = 0; i < 6; ++i) g[i] = (uint8_t)(i * 40);
        CaliperTensor t = u8_3d(g, h, w, 1);
        CaliperTextureId id = bk.bridge->texture_from_tensor(&t, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(w * h * 4);
        expand_u8_to_rgba8(g, w, h, 1, ref.data());
        CHECK(bk.readback(id, w, h) == ref);
        bk.bridge->release_texture(id);
    }
    {   // C=3 rgb, alpha forced 255
        uint8_t rgb[18]; for (int i = 0; i < 18; ++i) rgb[i] = (uint8_t)(i * 11);
        CaliperTensor t = u8_3d(rgb, h, w, 3);
        CaliperTextureId id = bk.bridge->texture_from_tensor(&t, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(w * h * 4);
        expand_u8_to_rgba8(rgb, w, h, 3, ref.data());
        CHECK(bk.readback(id, w, h) == ref);
        bk.bridge->release_texture(id);
    }
    {   // C=4 passthrough
        uint8_t rgba[24]; for (int i = 0; i < 24; ++i) rgba[i] = (uint8_t)(255 - i * 7);
        CaliperTensor t = u8_3d(rgba, h, w, 4);
        CaliperTextureId id = bk.bridge->texture_from_tensor(&t, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(w * h * 4);
        expand_u8_to_rgba8(rgba, w, h, 4, ref.data());
        CHECK(bk.readback(id, w, h) == ref);
        bk.bridge->release_texture(id);
    }
}

void mat_update(Backend& bk) {
    float a[16]; for (int i = 0; i < 16; ++i) a[i] = (float)i;
    CaliperTensor ta = f32_2d(a, 4, 4);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &ta, CALIPER_CMAP_VIRIDIS, 0.0f, 15.0f, 0);
    REQUIRE(id != 0);

    float b[16]; for (int i = 0; i < 16; ++i) b[i] = (float)(15 - i);   // reversed
    CaliperTensor tb = f32_2d(b, 4, 4);
    REQUIRE(bk.bridge->update_texture(id, &tb));

    std::vector<uint8_t> ref(16 * 4);
    map_f32_to_rgba8(b, 4, 4, colormap_lut(CALIPER_CMAP_VIRIDIS), 0.0f, 15.0f, ref.data());
    CHECK(bk.readback(id, 4, 4) == ref);
    bk.bridge->release_texture(id);
}

void mat_invalid(Backend& bk) {
    float f[16] = {0};
    CaliperTensor wrong_ndim = f32_2d(f, 4, 4); wrong_ndim.ndim = 3;
    CHECK(bk.bridge->texture_from_tensor_mapped(&wrong_ndim, 0, 0, 1, 0) == 0);
    CaliperTensor noncontig = f32_2d(f, 4, 4); noncontig.strides[0] = 5;   // gap
    CHECK(bk.bridge->texture_from_tensor_mapped(&noncontig, 0, 0, 1, 0) == 0);
    CaliperTensor f16 = f32_2d(f, 4, 4); f16.dtype = CALIPER_DT_F16;
    CHECK(bk.bridge->texture_from_tensor_mapped(&f16, 0, 0, 1, 0) == 0);
    CHECK(bk.bridge->texture_from_tensor(nullptr, 0) == 0);
}

void mat_alloc_shared(Backend& bk) {
    int64_t shape[3] = {2, 3, 4};   // (H,W,4) u8 unified buffer
    CaliperTensor out{};
    CaliperTextureId tex = 0;
    REQUIRE(bk.bridge->alloc_shared(CALIPER_DT_U8, 3, shape, &out, &tex));
    REQUIRE(tex != 0);
    REQUIRE(out.data != nullptr);
    uint8_t* buf = (uint8_t*)out.data;
    for (int i = 0; i < 2 * 3 * 4; ++i) buf[i] = (uint8_t)(i * 5 + 1);
    REQUIRE(bk.bridge->update_texture(tex, &out));
    std::vector<uint8_t> ref(2 * 3 * 4);
    expand_u8_to_rgba8(buf, 3, 2, 4, ref.data());
    CHECK(bk.readback(tex, 3, 2) == ref);
    bk.bridge->free_shared(tex);
}

#define REQUIRE_GL() do { \
    if (!gl_env().ok) { MESSAGE("no GUI/GL context — skipping gfx case"); return; } \
} while (0)

}  // namespace

// ---------------------------------------------------------------------------
// GL run of the §16 matrix.
// ---------------------------------------------------------------------------
TEST_CASE("gfx/GL: 4x4 f32 ramp mapped through viridis is pixel-exact") {
    REQUIRE_GL(); Backend bk = gl_backend(); mat_f32_viridis(bk);
}
TEST_CASE("gfx/GL: f32 mapped through magma and RdBu is pixel-exact") {
    REQUIRE_GL(); Backend bk = gl_backend(); mat_f32_magma_rdbu(bk);
}
TEST_CASE("gfx/GL: 2x3 u8 direct (C=1,3,4) expands pixel-exact") {
    REQUIRE_GL(); Backend bk = gl_backend(); mat_u8_direct(bk);
}
TEST_CASE("gfx/GL: update_texture changes the pixels on the GPU") {
    REQUIRE_GL(); Backend bk = gl_backend(); mat_update(bk);
}
TEST_CASE("gfx/GL: invalid tensors return id 0") {
    REQUIRE_GL(); Backend bk = gl_backend(); mat_invalid(bk);
}
TEST_CASE("gfx/GL: alloc_shared roundtrip — write buffer, update, readback") {
    REQUIRE_GL(); Backend bk = gl_backend(); mat_alloc_shared(bk);
}
TEST_CASE("gfx/GL: last_device_path is the frozen CPU-staged fallback") {
    REQUIRE_GL();
    // GL never reads device memory; the interface default reports cpu-staged.
    CHECK(std::string(gl_env().renderer->last_device_path()) == "cpu-staged");
}

// ===========================================================================
// Metal run — same matrix (CPU-tensor uploads, staged) + the device paths that
// nothing had ever executed (compute / blit from a raw MTLBuffer).
// ===========================================================================
#ifdef CALIPER_HAVE_METAL
namespace {

struct MetalEnv {
    bool ok = false;
    GLFWwindow* window = nullptr;
    ImGuiContext* imgui_ctx = nullptr;
    std::unique_ptr<HostRenderer> renderer;
    std::unique_ptr<TensorBridge> bridge;
    id<MTLDevice> device = nil;

    MetalEnv() {
        if (!glfw_guard().ok) return;
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) return;                 // no Metal GPU -> skip
        renderer = make_metal_renderer();
        glfwDefaultWindowHints();                  // clear prior GL hints
        renderer->window_hints();                  // GLFW_NO_API
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(64, 64, "caliper_gfx_tests(metal)", nullptr, nullptr);
        if (!window) return;
        imgui_ctx = ImGui::CreateContext();
        ImGui::SetCurrentContext(imgui_ctx);
        if (!renderer->init(window)) {
            ImGui::DestroyContext(imgui_ctx);
            glfwDestroyWindow(window);
            return;
        }
        bridge = std::make_unique<TensorBridge>(*renderer);
        ok = true;
    }
    ~MetalEnv() {
        if (!ok) return;
        ImGui::SetCurrentContext(imgui_ctx);
        bridge.reset();
        renderer->shutdown();
        ImGui::DestroyContext(imgui_ctx);
        glfwDestroyWindow(window);
    }
};

MetalEnv& metal_env() { static MetalEnv e; return e; }

// Test-only readback: blit the RGBA8 texture into a shared MTLBuffer, wait, and
// copy its unified contents out. NOT the render path.
std::vector<uint8_t> metal_readback(HostRenderer& r, CaliperTextureId tex, int w, int h) {
    @autoreleasepool {
        void* p = (void*)(uintptr_t)r.tex_imtexture_id(tex);
        id<MTLTexture> t = (__bridge id<MTLTexture>)p;
        id<MTLDevice> dev = t.device;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSUInteger bpr = (NSUInteger)w * 4;
        id<MTLBuffer> out = [dev newBufferWithLength:bpr * (NSUInteger)h
                                             options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [q commandBuffer];
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
        [cb waitUntilCompleted];
        std::vector<uint8_t> px((size_t)w * h * 4);
        std::memcpy(px.data(), out.contents, px.size());
        return px;
    }
}

Backend metal_backend() {
    Backend b;
    b.bridge = metal_env().bridge.get();
    b.renderer = metal_env().renderer.get();
    HostRenderer* r = b.renderer;
    b.readback = [r](CaliperTextureId id, int w, int h) {
        return metal_readback(*r, id, w, h);
    };
    return b;
}

// A page-aligned MTLBuffer of `bytes` filled from `src`. Shared storage = unified
// memory, so contents is directly addressable (the C-ABI contract: the METAL
// tensor's data is a bridge-cast id<MTLBuffer>).
id<MTLBuffer> device_buffer(const void* src, size_t bytes) {
    id<MTLBuffer> b = [metal_env().device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
    if (src) std::memcpy(b.contents, src, bytes);
    return b;
}

}  // namespace

TEST_CASE("gfx/Metal: 4x4 f32 ramp mapped through viridis is pixel-exact (staged)") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend(); mat_f32_viridis(bk);
}
TEST_CASE("gfx/Metal: f32 mapped through magma and RdBu is pixel-exact (staged)") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend(); mat_f32_magma_rdbu(bk);
}
TEST_CASE("gfx/Metal: 2x3 u8 direct (C=1,3,4) expands pixel-exact (staged)") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend(); mat_u8_direct(bk);
}
TEST_CASE("gfx/Metal: update_texture changes the pixels on the GPU (staged)") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend(); mat_update(bk);
}
TEST_CASE("gfx/Metal: invalid tensors return id 0") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend(); mat_invalid(bk);
}
TEST_CASE("gfx/Metal: alloc_shared roundtrip — write buffer, update, readback") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend(); mat_alloc_shared(bk);
}

// ---- DEVICE PATHS (first-ever execution of C2's compute/blit) --------------

// f32 + LUT on a METAL-device tensor -> compute shader -> pixel-exact vs CPU
// reference. Sizes deliberately non-multiples of the 16x16 threadgroup to prove
// the dispatch edge-guard (5x3 -> 1x1 groups, 17x9 -> 2x1 groups).
TEST_CASE("gfx/Metal: device f32+LUT takes the compute path, pixel-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();

    for (auto wh : {std::pair<int,int>{4, 4}, {5, 3}, {17, 9}}) {
        const int w = wh.first, h = wh.second;
        const int n = w * h;
        std::vector<float> data(n);
        for (int i = 0; i < n; ++i) data[i] = (float)i;      // spans 0..n-1
        const float vmin = 0.0f, vmax = (float)(n - 1);      // hits idx 0 and 255

        id<MTLBuffer> buf = device_buffer(data.data(), data.size() * sizeof(float));
        CaliperTensor t{};
        t.struct_size = sizeof(t);
        t.data = (__bridge void*)buf;
        t.dtype = CALIPER_DT_F32;
        t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
        t.device = CALIPER_DEV_METAL;

        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
            &t, CALIPER_CMAP_VIRIDIS, vmin, vmax, 0);
        REQUIRE(id != 0);
        CHECK(std::string(bk.renderer->last_device_path()) == "compute");

        std::vector<uint8_t> ref((size_t)n * 4);
        map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                         vmin, vmax, ref.data());
        CHECK(bk.readback(id, w, h) == ref);
        bk.bridge->release_texture(id);
    }
}

// u8 RGBA (HWC, C=4) on a METAL-device tensor -> blit -> byte-exact vs CPU ref.
TEST_CASE("gfx/Metal: device u8 HWC takes the blit path, byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();

    const int h = 3, w = 5, c = 4;
    const int n = w * h * c;
    std::vector<uint8_t> data(n);
    for (int i = 0; i < n; ++i) data[i] = (uint8_t)((i * 37 + 11) & 0xff);

    id<MTLBuffer> buf = device_buffer(data.data(), data.size());
    CaliperTensor t{};
    t.struct_size = sizeof(t);
    t.data = (__bridge void*)buf;
    t.dtype = CALIPER_DT_U8;
    t.ndim = 3; t.shape[0] = h; t.shape[1] = w; t.shape[2] = c;
    t.strides[0] = w * c; t.strides[1] = c; t.strides[2] = 1;
    t.device = CALIPER_DEV_METAL;

    CaliperTextureId id = bk.bridge->texture_from_tensor(&t, 0);
    REQUIRE(id != 0);
    CHECK(std::string(bk.renderer->last_device_path()) == "blit");

    std::vector<uint8_t> ref((size_t)w * h * 4);
    expand_u8_to_rgba8(data.data(), w, h, 4, ref.data());
    CHECK(bk.readback(id, w, h) == ref);
    bk.bridge->release_texture(id);
}

// A deliberately-short MTLBuffer (half the declared f32 extent) must be REJECTED
// before any dispatch — the device path must bound the buffer's byte length
// against the tensor extent, so it never reads memory it can't reason about and
// never faults the GPU.
TEST_CASE("gfx/Metal: short device buffer is rejected (no GPU fault)") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();

    const int h = 4, w = 4;
    const size_t full_bytes = (size_t)h * w * sizeof(float);
    id<MTLBuffer> buf = device_buffer(nullptr, full_bytes / 2);   // half the declared size

    CaliperTensor t{};
    t.struct_size = sizeof(t);
    t.data = (__bridge void*)buf;
    t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_METAL;

    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, 15.0f, 0);
    CHECK(id == 0);   // rejected: buffer too short for the declared extent
}
#endif  // CALIPER_HAVE_METAL
