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
#include <backends/imgui_impl_opengl3.h>

#include "tensor_bridge.h"
#include "renderer/host_renderer.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>
#include <backends/imgui_impl_metal.h>
#endif

#ifdef CALIPER_HAVE_VULKAN
#include "cuda_driver.h"
#include <utility>
#ifdef _WIN32
#include <windows.h>   // CloseHandle for the VMM shareable-handle test blocks
#endif
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

// Read an RGBA8 texture back off the GL GPU. Post-fix, the bridge id's VALUE is
// the renderer's ImGui handle — for GL that is the GL texture name — so it is
// bound directly here, the same value ImGui_ImplOpenGL3 binds (§5.4).
std::vector<uint8_t> gl_readback(HostRenderer& /*r*/, CaliperTextureId id, int w, int h) {
    GLuint name = (GLuint)id;
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

// Regression test for the user SIGSEGV (integer table ids crashed ImGui on the
// first Image bind). Two guarantees: (a) the PUBLIC bridge id's VALUE is exactly
// the renderer's ImGui handle for that texture — for GL, a live GL name; and (b)
// binding it via ImGui::Image and running the REAL ImGui_ImplOpenGL3 draw path
// (RenderDrawData -> glBindTexture on the id) executes cleanly offscreen. This
// is the exact path that faulted on Metal; here it is proven on GL.
TEST_CASE("gfx/GL: bridge id is the ImGui handle and survives the real ImGui draw path") {
    REQUIRE_GL();
    Backend bk = gl_backend();

    uint8_t rgba[2 * 2 * 4]; for (int i = 0; i < 16; ++i) rgba[i] = (uint8_t)(i * 15 + 8);
    CaliperTensor t = u8_3d(rgba, 2, 2, 4);
    CaliperTextureId id = bk.bridge->texture_from_tensor(&t, 0);
    REQUIRE(id != 0);
    // (a) the id's VALUE is the renderer's ImGui handle: a live GL texture name.
    CHECK(glIsTexture((GLuint)id) == GL_TRUE);

    // (b) offscreen FBO to draw into deterministically (hidden window).
    const int W = 32, H = 32;
    GLuint color = 0, fbo = 0;
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    REQUIRE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    ImGui::SetCurrentContext(gl_env().imgui_ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)W, (float)H);
    io.DeltaTime = 1.0f / 60.0f;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();
    ImGui::GetBackgroundDrawList()->AddImage(
        (ImTextureID)id, ImVec2(0, 0), ImVec2((float)W, (float)H));
    ImGui::Render();

    glViewport(0, 0, W, H);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());   // <- exact path that crashed
    glFinish();

    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color);

    bool drew = false;   // the bridge texture actually rasterized (non-bg texel)
    for (size_t i = 0; i < px.size(); ++i) if ((i % 4) != 3 && px[i]) { drew = true; break; }
    CHECK(drew);
    bk.bridge->release_texture(id);
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

Backend metal_backend() {
    Backend b;
    b.bridge = metal_env().bridge.get();
    b.renderer = metal_env().renderer.get();
    HostRenderer* r = b.renderer;
    b.readback = [r](CaliperTextureId id, int w, int h) {
        return r->debug_readback_rgba8(id, w, h);   // renderer-queue readback (M1)
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

// Same compute path with a NONZERO vmin: values in [-1,1] mapped with
// vmin=-1, so the (v-vmin) term the shader computes is exercised away from 0
// (a zero vmin would hide a dropped/incorrect subtraction). Pixel-exact vs the
// identical CPU reference, which uses the same (v-vmin)/(vmax-vmin) math.
TEST_CASE("gfx/Metal: device f32+LUT compute path is exact with nonzero vmin") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();

    const int w = 4, h = 4;
    const int n = w * h;
    std::vector<float> data(n);
    for (int i = 0; i < n; ++i) data[i] = -1.0f + 2.0f * ((float)i / (float)(n - 1)); // -1..1
    const float vmin = -1.0f, vmax = 1.0f;

    id<MTLBuffer> buf = device_buffer(data.data(), data.size() * sizeof(float));
    CaliperTensor t{};
    t.struct_size = sizeof(t);
    t.data = (__bridge void*)buf;
    t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_METAL;

    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_RDBU, vmin, vmax, 0);
    REQUIRE(id != 0);
    CHECK(std::string(bk.renderer->last_device_path()) == "compute");

    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_RDBU),
                     vmin, vmax, ref.data());
    CHECK(bk.readback(id, w, h) == ref);
    bk.bridge->release_texture(id);
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

// M1 pipelining proof (the Vulkan burst test's twin): several device updates
// enqueued back-to-back with NO readback between them, so successive compute
// passes are in flight together, ordered only by queue commit order (D23).
// The final readback must equal the LAST write byte-for-byte. Fresh source
// buffers per generation keep CPU writes outside the contract (a NULL-stream
// caller owns producer quiescence); dropping each buffer's last strong ref
// mid-flight also exercises command-buffer resource retention (spec §3.2).
TEST_CASE("gfx/Metal: burst updates pipeline in order, final readback pixel-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();

    const int w = 17, h = 9, n = w * h;   // non-16-multiple edge sizes
    auto gen_data = [&](int gen) {
        std::vector<float> d(n);
        for (int i = 0; i < n; ++i) d[i] = (float)((i * 7 + gen * 13) % n);
        return d;
    };

    std::vector<float> d0 = gen_data(0);
    id<MTLBuffer> buf0 = device_buffer(d0.data(), (size_t)n * sizeof(float));
    CaliperTensor t{};
    t.struct_size = sizeof(t); t.data = (__bridge void*)buf0; t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_METAL;

    // NB: not named `id` — that is the Objective-C `id` type, needed by the
    // `id<MTLBuffer>` declaration inside the loop below.
    CaliperTextureId tex_id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(tex_id != 0);

    std::vector<float> last;
    for (int gen = 1; gen <= 8; ++gen) {
        last = gen_data(gen);
        id<MTLBuffer> b = device_buffer(last.data(), (size_t)n * sizeof(float));
        t.data = (__bridge void*)b;      // b's last strong ref dies each loop turn
        REQUIRE(bk.bridge->update_texture(tex_id, &t));
    }
    CHECK(std::string(bk.renderer->last_device_path()) == "compute");

    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(last.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    CHECK(bk.readback(tex_id, w, h) == ref);
    bk.bridge->release_texture(tex_id);
}

// M2b: a non-NULL t.stream (the producer's MTLCommandQueue*) must GPU-order
// the update AFTER the producer's committed work. Deterministic, no timing
// luck: the producer's payload write is gated behind an MTLSharedEvent the
// TEST only fires after update_texture returns. A renderer that ignores
// t.stream colormaps the stale bytes (fails); one that orders reads the fresh.
TEST_CASE("gfx/Metal: non-NULL stream orders the update after the producer queue") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE(bk.renderer->honors_stream_ordered_handoff());

    const int w = 8, h = 8, n = w * h;
    std::vector<float> stale(n, 0.0f);
    std::vector<float> fresh(n);
    for (int i = 0; i < n; ++i) fresh[i] = (float)i;

    id<MTLBuffer> tensor_buf = device_buffer(stale.data(), (size_t)n * sizeof(float));
    id<MTLBuffer> payload    = device_buffer(fresh.data(), (size_t)n * sizeof(float));

    CaliperTensor t{};
    t.struct_size = sizeof(t);
    t.data = (__bridge void*)tensor_buf;
    t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_METAL;

    // NB: not named `id` — that is the Objective-C `id` keyword, needed below.
    CaliperTextureId tex_id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(tex_id != 0);

    // Producer queue with pending committed work: a payload blit blocked on a
    // gate the CPU holds — the stand-in for torch's just-committed kernels.
    id<MTLDevice> dev = metal_env().device;
    id<MTLCommandQueue> producer = [dev newCommandQueue];
    id<MTLSharedEvent> gate = [dev newSharedEvent];
    id<MTLCommandBuffer> pc = [producer commandBuffer];
    [pc encodeWaitForEvent:gate value:1];
    id<MTLBlitCommandEncoder> pb = [pc blitCommandEncoder];
    [pb copyFromBuffer:payload sourceOffset:0
              toBuffer:tensor_buf destinationOffset:0
                  size:(NSUInteger)n * sizeof(float)];
    [pb endEncoding];
    [pc commit];

    // Handoff with the producer queue in t.stream and NO drain anywhere.
    t.stream = (__bridge void*)producer;
    REQUIRE(bk.bridge->update_texture(tex_id, &t));

    gate.signaledValue = 1;   // only NOW may the producer's write run

    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(fresh.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    CHECK(bk.readback(tex_id, w, h) == ref);
    bk.bridge->release_texture(tex_id);
}

// THE regression test for the reported SIGSEGV. Before the fix the bridge handed
// out a small monotonic integer id; ImGui_ImplMetal_RenderDrawData reached
// setFragmentTexture:/objc_retain on that integer-as-pointer and faulted on
// KERN_INVALID_ADDRESS 0x1 the first time any bridge texture was drawn. Post-fix
// the id's VALUE is the id<MTLTexture> pointer ImGui binds. This binds a bridge
// id via ImGui::Image and runs the ACTUAL ImGui_ImplMetal draw path offscreen —
// the exact call chain that crashed. If it returns, the crash is fixed.
TEST_CASE("gfx/Metal: bridge id is the ImGui handle and survives the real ImGui draw path") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    @autoreleasepool {
        uint8_t rgba[2 * 2 * 4]; for (int i = 0; i < 16; ++i) rgba[i] = (uint8_t)(i * 15 + 8);
        CaliperTensor t = u8_3d(rgba, 2, 2, 4);
        // NB: not named `id` — that is the Objective-C `id` keyword, needed below.
        CaliperTextureId tex_id = bk.bridge->texture_from_tensor(&t, 0);
        REQUIRE(tex_id != 0);
        // (a) the id's VALUE is the renderer's ImGui handle: a live id<MTLTexture>.
        id<MTLTexture> as_tex = (__bridge id<MTLTexture>)(void*)(uintptr_t)tex_id;
        bool tex_live = (as_tex != nil);   // keep the ObjC ptr out of doctest decomposition
        REQUIRE(tex_live);
        CHECK((int)as_tex.width == 2);
        CHECK((int)as_tex.height == 2);

        // (b) offscreen RGBA8 render target on the SAME device ImGui_ImplMetal was
        //     initialized with (== the bridge texture's device).
        const int W = 32, H = 32;
        id<MTLDevice> dev = as_tex.device;
        MTLTextureDescriptor* rtd =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:(NSUInteger)W
                                                              height:(NSUInteger)H
                                                           mipmapped:NO];
        rtd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        rtd.storageMode = MTLStorageModeShared;
        id<MTLTexture> rt = [dev newTextureWithDescriptor:rtd];
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = rt;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

        ImGui::SetCurrentContext(metal_env().imgui_ctx);
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2((float)W, (float)H);
        io.DeltaTime = 1.0f / 60.0f;
        ImGui_ImplMetal_NewFrame(pass);
        ImGui::NewFrame();
        ImGui::GetBackgroundDrawList()->AddImage(
            (ImTextureID)tex_id, ImVec2(0, 0), ImVec2((float)W, (float)H));
        ImGui::Render();

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cb, enc);  // <- exact crashing path
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // It ran without SIGSEGV. Confirm the bridge texture actually rasterized.
        NSUInteger bpr = (NSUInteger)W * 4;
        id<MTLBuffer> out = [dev newBufferWithLength:bpr * (NSUInteger)H
                                            options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb2 = [q commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb2 blitCommandEncoder];
        [blit copyFromTexture:rt sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake((NSUInteger)W, (NSUInteger)H, 1)
                     toBuffer:out destinationOffset:0
       destinationBytesPerRow:bpr destinationBytesPerImage:bpr * (NSUInteger)H];
        [blit endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
        const uint8_t* p = (const uint8_t*)out.contents;
        bool drew = false;   // a non-background (non-alpha) texel means it drew
        for (NSUInteger i = 0; i < bpr * (NSUInteger)H; ++i)
            if ((i % 4) != 3 && p[i]) { drew = true; break; }
        CHECK(drew);
        bk.bridge->release_texture(tex_id);
    }
}

// ---- v1.2 imported-allocation rows (Metal: in-process MTLBuffer import) ----

// Import an in-process MTLBuffer and colormap a texture straight from a NONZERO
// byte offset inside it — no CPU copy of the tensor data. Byte-exact vs the same
// CPU reference the staged path uses. Then a released alloc must refuse.
TEST_CASE("gfx/Metal: import in-process MTLBuffer, colormap from a nonzero offset, byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->caps() & CALIPER_BRIDGE_CAP_IMPORT_ALLOC) != 0);

    // 4×4 f32 ramp at byte offset 256 inside a 4096-byte buffer.
    const int W = 4, H = 4;
    const uint64_t off = 256;
    std::vector<uint8_t> bytes(4096, 0);
    float ramp[W * H];
    for (int i = 0; i < W * H; ++i) ramp[i] = (float)i / (float)(W * H - 1);
    std::memcpy(bytes.data() + off, ramp, sizeof(ramp));
    id<MTLBuffer> buf = device_buffer(bytes.data(), bytes.size());
    bool buf_live = (buf != nil);   // keep the ObjC ptr out of doctest decomposition
    REQUIRE(buf_live);

    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);

    // Create the W×H viridis texture the house way: a seed CPU tensor pins the
    // colormap/vmin/vmax (0..1) that the imported update reuses (§ update_alloc).
    std::vector<float> seed((size_t)W * H, 0.0f);
    CaliperTensor seed_t = f32_2d(seed.data(), H, W);
    CaliperTextureId tex = bk.bridge->texture_from_tensor_mapped(
        &seed_t, CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, 0);
    REQUIRE(tex != 0);

    CaliperTensor d{};
    d.struct_size = sizeof(CaliperTensor);
    d.dtype = CALIPER_DT_F32; d.ndim = 2;
    d.shape[0] = H; d.shape[1] = W; d.strides[0] = W; d.strides[1] = 1;
    d.device = CALIPER_DEV_METAL;    // data/stream stay null: alloc+offset IS the address
    REQUIRE(bk.bridge->update_texture_from_alloc(tex, alloc, off, &d));
    CHECK(std::string(bk.renderer->last_device_path()) == "compute-imported");

    // Byte-exact vs the identical CPU reference the staged path uses (vmin=0, vmax=1).
    std::vector<uint8_t> ref((size_t)W * H * 4);
    map_f32_to_rgba8(ramp, W, H, colormap_lut(CALIPER_CMAP_VIRIDIS), 0.0f, 1.0f, ref.data());
    CHECK(bk.readback(tex, W, H) == ref);

    bk.bridge->release_allocation(alloc);
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(tex, alloc, off, &d));  // released → refuses
    bk.bridge->release_texture(tex);
}

// Import gates fail closed: wrong handle kind, null handle, zero size, and a
// size overclaim (buf shorter than declared) all return 0; an OOB offset on a
// valid import refuses and leaves pixels untouched.
TEST_CASE("gfx/Metal: import gates fail closed") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    std::vector<uint8_t> z(1024, 0);
    id<MTLBuffer> buf = device_buffer(z.data(), z.size());

    // wrong handle kind / null handle / zero size / size overclaim
    CHECK(bk.bridge->import_allocation((__bridge void*)buf, 1024,
                                       CALIPER_ALLOC_HANDLE_OPAQUE_FD) == 0);
    CHECK(bk.bridge->import_allocation(nullptr, 1024,
                                       CALIPER_ALLOC_HANDLE_MTLBUFFER) == 0);
    CHECK(bk.bridge->import_allocation((__bridge void*)buf, 0,
                                       CALIPER_ALLOC_HANDLE_MTLBUFFER) == 0);
    CHECK(bk.bridge->import_allocation((__bridge void*)buf, 4096,
                                       CALIPER_ALLOC_HANDLE_MTLBUFFER) == 0);  // buf.length is 1024

    // OOB offset on a valid import refuses and leaves pixels untouched
    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, 1024, CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);
    std::vector<float> seed(16, 0.0f);
    CaliperTensor seed_t = f32_2d(seed.data(), 4, 4);
    CaliperTextureId tex = bk.bridge->texture_from_tensor_mapped(
        &seed_t, CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, 0);
    REQUIRE(tex != 0);
    CaliperTensor d{};
    d.struct_size = sizeof(CaliperTensor);
    d.dtype = CALIPER_DT_F32; d.ndim = 2;
    d.shape[0] = 4; d.shape[1] = 4; d.strides[0] = 4; d.strides[1] = 1;
    d.device = CALIPER_DEV_METAL;
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(tex, alloc, 1024, &d));   // offset==length
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(tex, alloc, 1000, &d));   // extent past end
    bk.bridge->release_allocation(alloc);
    bk.bridge->release_texture(tex);
}

// ---- caliper.geometry.v1 rows (Metal): byte-exact mirror of the Vulkan cases,
// alloc source = in-process shared MTLBuffer instead of CUDA VMM. The three
// helpers below are duplicated verbatim from the Vulkan section (compiled out
// on Mac) — same NDC mapping / identity camera / CPU reference image. ----
namespace {

// NDC position whose 1-px point covers exactly pixel (px,py) of a WxH view
// under the backend's GL-style (+y up) mapping.
void ndc_for_pixel(int px, int py, int w, int h, float* out3) {
    out3[0] = 2.0f * ((float)px + 0.5f) / (float)w - 1.0f;
    out3[1] = 1.0f - 2.0f * ((float)py + 0.5f) / (float)h;
    out3[2] = 0.0f;
}

CaliperGeomCamera identity_cam() {
    CaliperGeomCamera c{};
    for (int i = 0; i < 4; ++i) { c.view[i * 4 + i] = 1.f; c.proj[i * 4 + i] = 1.f; }
    return c;
}

// Expected image: clear color everywhere, LUT/flat color at the given pixels.
std::vector<uint8_t> geom_ref(int w, int h, uint32_t clear_rgba,
                              const std::vector<std::pair<int,int>>& px,
                              const std::vector<uint32_t>& color_rgba) {
    std::vector<uint8_t> ref((size_t)w * h * 4);
    for (int i = 0; i < w * h; ++i) {
        ref[(size_t)i * 4 + 0] = (uint8_t)(clear_rgba         & 0xFF);
        ref[(size_t)i * 4 + 1] = (uint8_t)((clear_rgba >> 8)  & 0xFF);
        ref[(size_t)i * 4 + 2] = (uint8_t)((clear_rgba >> 16) & 0xFF);
        ref[(size_t)i * 4 + 3] = (uint8_t)((clear_rgba >> 24) & 0xFF);
    }
    for (size_t k = 0; k < px.size(); ++k) {
        const size_t at = ((size_t)px[k].second * w + px[k].first) * 4;
        ref[at + 0] = (uint8_t)(color_rgba[k]         & 0xFF);
        ref[at + 1] = (uint8_t)((color_rgba[k] >> 8)  & 0xFF);
        ref[at + 2] = (uint8_t)((color_rgba[k] >> 16) & 0xFF);
        ref[at + 3] = (uint8_t)((color_rgba[k] >> 24) & 0xFF);
    }
    return ref;
}

uint32_t metal_geom_pixel_rgba(const std::vector<uint8_t>& px, int w, int x, int y) {
    const size_t at = ((size_t)y * (size_t)w + (size_t)x) * 4u;
    return (uint32_t)px[at + 0] |
           ((uint32_t)px[at + 1] << 8) |
           ((uint32_t)px[at + 2] << 16) |
           ((uint32_t)px[at + 3] << 24);
}

}  // namespace

TEST_CASE("gfx/metal geometry: imported points byte-exact — colormap extremes at a nonzero offset") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_IMPORTED_POINTS));

    const int W = 64, H = 64;
    const uint64_t pos_off = 512, attr_off = 2048;
    const std::vector<std::pair<int,int>> px = {{3, 5}, {40, 22}, {63, 63}};
    const float attrs[3] = {0.0f, 1.0f, 0.5f};      // LUT[0], LUT[255], LUT[128]

    std::vector<uint8_t> bytes(4096, 0);
    float pos[9];
    for (int i = 0; i < 3; ++i) ndc_for_pixel(px[i].first, px[i].second, W, H, &pos[i*3]);
    std::memcpy(bytes.data() + pos_off,  pos,   sizeof(pos));
    std::memcpy(bytes.data() + attr_off, attrs, sizeof(attrs));
    id<MTLBuffer> buf = device_buffer(bytes.data(), bytes.size());
    REQUIRE((buf != nil));

    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);
    CaliperTextureId view = bk.bridge->geom_create_view(W, H);
    REQUIRE(view != 0);

    CaliperGeomCamera cam = identity_cam();
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, pos_off, 3,
                                        alloc, attr_off, CALIPER_CMAP_VIRIDIS,
                                        0.f, 1.f, 1.f, 0xFF000000u));
    CHECK(std::string(bk.renderer->last_device_path()) == "points-imported");

    const uint32_t* lut = colormap_lut(CALIPER_CMAP_VIRIDIS);
    auto ref = geom_ref(W, H, 0xFF000000u, px, {lut[0], lut[255], lut[128]});
    auto got = bk.readback(view, W, H);
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != ref[i]) { FAIL("first diff at byte ", i, ": got ", (int)got[i], " ref ", (int)ref[i]); }
    CHECK(got == ref);

    // flat-white path: attr_alloc = 0
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, pos_off, 3,
                                        0, 0, CALIPER_CMAP_VIRIDIS,
                                        0.f, 1.f, 1.f, 0xFF000000u));
    auto ref2 = geom_ref(W, H, 0xFF000000u, px,
                         {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu});
    CHECK(bk.readback(view, W, H) == ref2);
    bk.bridge->release_allocation(alloc);
    bk.bridge->geom_release_view(view);
}

TEST_CASE("gfx/metal geometry: count 0 clears; gates keep prior pixels; released refuses") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    if ((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_IMPORTED_POINTS) == 0) return;

    const int W = 32, H = 32;
    CaliperTextureId view = bk.bridge->geom_create_view(W, H);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    // count==0 = pure clear (teal), alloc ids 0
    const uint32_t teal = 10u | (20u << 8) | (30u << 16) | (255u << 24);
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, 0, 0, 0, 0, 0,
                                        CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, teal));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, teal, {}, {}));

    // one real point, then assert every gate refuses AND pixels stay put
    std::vector<uint8_t> bytes(1024, 0);
    float p3[3]; ndc_for_pixel(7, 9, W, H, p3);
    std::memcpy(bytes.data(), p3, sizeof(p3));
    id<MTLBuffer> buf = device_buffer(bytes.data(), bytes.size());
    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1, 0, 0,
                                        CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));
    auto snap = bk.readback(view, W, H);
    std::string path = bk.renderer->last_device_path();

    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 2, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // misaligned
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1024/12 + 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // OOB count
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, 999, 0, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // unknown alloc
    CHECK_FALSE(bk.bridge->geom_draw_points(view, nullptr, alloc, 0, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // null cam
    CHECK(bk.readback(view, W, H) == snap);
    CHECK(std::string(bk.renderer->last_device_path()) == path);

    bk.bridge->release_allocation(alloc);
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // released alloc
    bk.bridge->geom_release_view(view);
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, 0, 0, 0, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // released view
}

TEST_CASE("gfx/metal geometry.v1_1: indexed triangles from imported buffers honor depth") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    const float near_z = 0.20f;
    const float far_z  = 0.80f;
    const float pos[] = {
        -1.0f, -1.0f, near_z,  3.0f, -1.0f, near_z, -1.0f,  3.0f, near_z,
        -1.0f, -1.0f, far_z,   3.0f, -1.0f, far_z,  -1.0f,  3.0f, far_z,
    };
    const uint32_t idx[] = {0, 1, 2, 0, 1, 2};

    id<MTLBuffer> pos_buf = device_buffer(pos, sizeof(pos));
    id<MTLBuffer> idx_buf = device_buffer(idx, sizeof(idx));
    REQUIRE((pos_buf != nil));
    REQUIRE((idx_buf != nil));

    CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId idx_alloc = bk.bridge->import_allocation(
        (__bridge void*)idx_buf, sizeof(idx), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pos_alloc != 0);
    REQUIRE(idx_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH);
    REQUIRE(view != 0);

    CaliperGeomCamera cam = identity_cam();
    CaliperGeomDraw near_draw{};
    near_draw.pos_alloc = pos_alloc;
    near_draw.vertex_count = 3;
    near_draw.index_alloc = idx_alloc;
    near_draw.index_count = 3;
    near_draw.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    near_draw.color_mode = CALIPER_GEOM_COLOR_FLAT;
    near_draw.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    near_draw.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    near_draw.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
    near_draw.flat_rgba = 0xFF00FF00u; // green in little-endian RGBA8
    near_draw.vmin = 0.0f;
    near_draw.vmax = 1.0f;
    near_draw.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) near_draw.model[i * 4 + i] = 1.0f;

    CaliperGeomDraw far_draw = near_draw;
    far_draw.pos_offset = 9u * sizeof(float);
    far_draw.index_offset = 3u * sizeof(uint32_t);
    far_draw.flat_rgba = 0xFF0000FFu; // red; would overwrite without depth

    CaliperGeomDraw draws[2] = {near_draw, far_draw};
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, draws, 2,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(std::string(bk.renderer->last_device_path()) == "primitives-imported");

    const auto px = bk.readback(view, W, H);
    REQUIRE(px.size() == (size_t)W * (size_t)H * 4u);
    CHECK(metal_geom_pixel_rgba(px, W, W / 2, H / 2) == 0xFF00FF00u);

    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->release_allocation(idx_alloc);
    bk.bridge->geom_release_view(view);
}
#endif  // CALIPER_HAVE_METAL

// ===========================================================================
// Vulkan run (Windows) — the same §16 matrix on the CPU-staged Vulkan path
// (portable, no CUDA), plus hardware-gated CUDA device-path + alloc_shared
// byte-exact tests when a UUID-matched CUDA device is paired.
// ===========================================================================
#ifdef CALIPER_HAVE_VULKAN
namespace {

struct VkEnv {
    bool ok = false;
    GLFWwindow* window = nullptr;
    ImGuiContext* imgui_ctx = nullptr;
    std::unique_ptr<HostRenderer> renderer;
    std::unique_ptr<TensorBridge> bridge;

    VkEnv() {
        if (!glfw_guard().ok) return;
        renderer = make_vulkan_renderer();
        glfwDefaultWindowHints();
        renderer->window_hints();                  // GLFW_NO_API
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(64, 64, "caliper_gfx_tests(vk)", nullptr, nullptr);
        if (!window) return;
        imgui_ctx = ImGui::CreateContext();
        ImGui::SetCurrentContext(imgui_ctx);
        if (!renderer->init(window)) {             // no Vulkan ICD -> skip
            ImGui::DestroyContext(imgui_ctx);
            glfwDestroyWindow(window);
            return;
        }
        bridge = std::make_unique<TensorBridge>(*renderer);
        ok = true;
    }
    ~VkEnv() {
        if (!ok) return;
        ImGui::SetCurrentContext(imgui_ctx);
        bridge.reset();
        renderer->shutdown();
        ImGui::DestroyContext(imgui_ctx);
        glfwDestroyWindow(window);
    }
};

VkEnv& vk_env() { static VkEnv e; return e; }

Backend vk_backend() {
    Backend b;
    b.bridge = vk_env().bridge.get();
    b.renderer = vk_env().renderer.get();
    HostRenderer* r = b.renderer;
    b.readback = [r](CaliperTextureId id, int w, int h) {
        return r->debug_readback_rgba8(id, w, h);   // renderer copies the VkImage out
    };
    return b;
}

// CUDA device tests need a current context to allocate source buffers. On a
// single-GPU box device 0 IS the renderer's UUID-paired device, and the primary
// context is process-global, so retaining it here yields the same context the
// renderer's interop uses. Multi-GPU -> skip (can't assume which device pairs).
bool vk_cuda_ready() {
    if (!vk_env().ok) return false;
    if (vk_env().renderer->interop_device() != CALIPER_DEV_CUDA) return false;
    const cudadrv::Api* cu = cudadrv::api();
    if (!cu || !cu->cuMemAlloc) return false;
    int n = 0;
    if (cu->cuDeviceGetCount(&n) != cudadrv::CUDA_SUCCESS || n != 1) return false;
    static cudadrv::CUcontext ctx = nullptr;
    if (!ctx && cu->cuDevicePrimaryCtxRetain(&ctx, 0) != cudadrv::CUDA_SUCCESS) return false;
    cu->cuCtxSetCurrent(ctx);
    return true;
}

}  // namespace

// ---- CPU-staged matrix (portable: runs on any Vulkan ICD, no CUDA) ----------
TEST_CASE("gfx/Vulkan: 4x4 f32 ramp mapped through viridis is pixel-exact (staged)") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    Backend bk = vk_backend(); mat_f32_viridis(bk);
}
TEST_CASE("gfx/Vulkan: f32 mapped through magma and RdBu is pixel-exact (staged)") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    Backend bk = vk_backend(); mat_f32_magma_rdbu(bk);
}
TEST_CASE("gfx/Vulkan: 2x3 u8 direct (C=1,3,4) expands pixel-exact (staged)") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    Backend bk = vk_backend(); mat_u8_direct(bk);
}
TEST_CASE("gfx/Vulkan: update_texture changes the pixels on the GPU (staged)") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    Backend bk = vk_backend(); mat_update(bk);
}
TEST_CASE("gfx/Vulkan: invalid tensors return id 0") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    Backend bk = vk_backend(); mat_invalid(bk);
}

// ---- CUDA device paths (hardware-gated: needs a UUID-paired CUDA device) -----
TEST_CASE("gfx/Vulkan+CUDA: device f32+LUT takes the compute path, pixel-exact") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    if (!vk_cuda_ready()) { MESSAGE("no CUDA interop / not single-GPU — skipping"); return; }
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();

    for (auto wh : {std::pair<int,int>{4, 4}, {5, 3}, {17, 9}}) {   // non-16-multiple edges
        const int w = wh.first, h = wh.second, n = w * h;
        std::vector<float> data(n);
        for (int i = 0; i < n; ++i) data[i] = (float)i;
        const float vmin = 0.0f, vmax = (float)(n - 1);

        cudadrv::CUdeviceptr buf = 0;
        REQUIRE(cu->cuMemAlloc(&buf, (size_t)n * sizeof(float)) == cudadrv::CUDA_SUCCESS);
        cu->cuMemcpyHtoD(buf, data.data(), (size_t)n * sizeof(float));

        CaliperTensor t{};
        t.struct_size = sizeof(t); t.data = (void*)(uintptr_t)buf; t.dtype = CALIPER_DT_F32;
        t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
        t.device = CALIPER_DEV_CUDA;

        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
            &t, CALIPER_CMAP_VIRIDIS, vmin, vmax, 0);
        REQUIRE(id != 0);
        CHECK(std::string(bk.renderer->last_device_path()) == "compute");

        std::vector<uint8_t> ref((size_t)n * 4);
        map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                         vmin, vmax, ref.data());
        CHECK(bk.readback(id, w, h) == ref);
        bk.bridge->release_texture(id);
        cu->cuMemFree(buf);
    }
}

// V4 pipelining stress: several device updates enqueued back-to-back with NO
// readback between them, so successive chains (CUDA copy -> Vulkan pass) are
// actually in flight together and ordered only by the texture's timeline
// semaphore (retire + signal/wait). The final readback must equal the LAST
// write byte-for-byte — a torn or reordered chain would surface a stale mix.
TEST_CASE("gfx/Vulkan+CUDA: burst updates pipeline in order, final frame pixel-exact") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    if (!vk_cuda_ready()) { MESSAGE("no CUDA interop / not single-GPU — skipping"); return; }
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();

    const int w = 17, h = 9, n = w * h;   // non-16-multiple edge sizes
    cudadrv::CUdeviceptr buf = 0;
    REQUIRE(cu->cuMemAlloc(&buf, (size_t)n * sizeof(float)) == cudadrv::CUDA_SUCCESS);

    CaliperTensor t{};
    t.struct_size = sizeof(t); t.data = (void*)(uintptr_t)buf; t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_CUDA;

    // Create with generation 0, then burst 8 more generations without reading.
    std::vector<float> data(n);
    auto fill = [&](int gen) {
        for (int i = 0; i < n; ++i) data[i] = (float)((i * 7 + gen * 13) % n);
        cu->cuMemcpyHtoD(buf, data.data(), (size_t)n * sizeof(float));
    };
    fill(0);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(id != 0);
    for (int gen = 1; gen <= 8; ++gen) {
        fill(gen);
        REQUIRE(bk.bridge->update_texture(id, &t));
    }
    CHECK(std::string(bk.renderer->last_device_path()) == "compute");

    // fill(8) is the last write; the readback retires the whole chain.
    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    CHECK(bk.readback(id, w, h) == ref);
    bk.bridge->release_texture(id);
    cu->cuMemFree(buf);
}

// Frame-thread hitch forensics (embed_scope "hitches every batch" report):
// the applet's per-generation refresh does release+create for all 9 textures
// (8x conv 48x48 + 1x embw 36x512). On the Vulkan+CUDA path a CREATE is the
// full interop setup — vkCreateImage + exportable memory + cuImportExternalMemory
// + timeline semaphore create/export/import — while an UPDATE is just a
// stream-ordered copy + compute pass. This case times both patterns at the
// applet's real sizes and prints µs/op so the fix (create-once + update) is
// justified by measurement, not vibes (D21).
TEST_CASE("gfx/Vulkan+CUDA: timing — create+release cycle vs update, applet-shaped") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    if (!vk_cuda_ready()) { MESSAGE("no CUDA interop / not single-GPU — skipping"); return; }
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();

    struct Shape { int w, h; const char* name; };
    const Shape shapes[2] = {{48, 48, "conv 48x48"}, {512, 36, "embw 36x512"}};
    constexpr int kIters = 200;

    for (const auto& s : shapes) {
        const int n = s.w * s.h;
        std::vector<float> data(n);
        for (int i = 0; i < n; ++i) data[i] = (float)(i % 251);
        cudadrv::CUdeviceptr buf = 0;
        REQUIRE(cu->cuMemAlloc(&buf, (size_t)n * sizeof(float)) == cudadrv::CUDA_SUCCESS);
        cu->cuMemcpyHtoD(buf, data.data(), (size_t)n * sizeof(float));

        CaliperTensor t{};
        t.struct_size = sizeof(t); t.data = (void*)(uintptr_t)buf; t.dtype = CALIPER_DT_F32;
        t.ndim = 2; t.shape[0] = s.h; t.shape[1] = s.w; t.strides[0] = s.w; t.strides[1] = 1;
        t.device = CALIPER_DEV_CUDA;

        // Pattern A: the applet today — release + create per generation.
        using clk = std::chrono::steady_clock;
        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
            &t, CALIPER_CMAP_VIRIDIS, 0.f, 250.f, 0);
        REQUIRE(id != 0);
        auto t0 = clk::now();
        for (int i = 0; i < kIters; ++i) {
            bk.bridge->release_texture(id);
            id = bk.bridge->texture_from_tensor_mapped(
                &t, CALIPER_CMAP_VIRIDIS, 0.f, 250.f, 0);
            REQUIRE(id != 0);
        }
        double us_create = std::chrono::duration<double, std::micro>(
                               clk::now() - t0).count() / kIters;

        // Pattern B: the proposed fix — create once, update per generation.
        auto t1 = clk::now();
        for (int i = 0; i < kIters; ++i)
            REQUIRE(bk.bridge->update_texture(id, &t));
        double us_update = std::chrono::duration<double, std::micro>(
                               clk::now() - t1).count() / kIters;
        bk.bridge->release_texture(id);
        cu->cuMemFree(buf);

        MESSAGE(s.name << ": release+create = " << us_create
                << " us/op, update = " << us_update << " us/op ("
                << (us_update > 0 ? us_create / us_update : 0) << "x)");
    }
}

TEST_CASE("gfx/Vulkan+CUDA: alloc_shared is device-backed and zero-copy, pixel-exact") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    if (!vk_cuda_ready()) { MESSAGE("no CUDA interop / not single-GPU — skipping"); return; }
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 8, H = 8, n = W * H;
    int64_t shape[2] = {H, W};
    CaliperTensor out{};
    CaliperTextureId tex = 0;
    REQUIRE(bk.bridge->alloc_shared(CALIPER_DT_F32, 2, shape, &out, &tex));
    REQUIRE(tex != 0);
    REQUIRE(out.data != nullptr);
    CHECK(out.device == CALIPER_DEV_CUDA);          // literal zero-copy: device-backed

    // The applet's kernels would write out.data (a CUDA device ptr) in place;
    // stand in with a host->device copy. alloc_shared f32 defaults vmin=0,vmax=1.
    std::vector<float> data(n);
    for (int i = 0; i < n; ++i) data[i] = (float)i / (float)(n - 1);
    cu->cuMemcpyHtoD((cudadrv::CUdeviceptr)(uintptr_t)out.data,
                     data.data(), (size_t)n * sizeof(float));

    REQUIRE(bk.bridge->update_texture(tex, &out));
    CHECK(std::string(bk.renderer->last_device_path()) == "compute");   // no D2D, then colormap

    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(data.data(), W, H, colormap_lut(CALIPER_CMAP_VIRIDIS), 0.0f, 1.0f, ref.data());
    CHECK(bk.readback(tex, W, H) == ref);
    bk.bridge->free_shared(tex);
}

// ===========================================================================
// Bridge v1.2 imported-allocation rows (Task 6). Hardware-gated like the CUDA
// rows above PLUS cudadrv::vmm_api() presence (VMM shareable handles need a
// CUDA 10.2+ driver; a missing table skips, never fails). Windows-only: the
// import path is OPAQUE_WIN32 (the locked T5 design).
// ===========================================================================
#ifdef _WIN32
namespace {

// A cuMemCreate'd, granularity-padded, WIN32-shareable, mapped+RW device
// allocation — the exact shape ExportablePool::alloc_block produces applet-side,
// built here through the OPTIONAL VmmApi table so the test controls every byte.
struct VmmBlock {
    const cudadrv::VmmApi* vmm = nullptr;
    cudadrv::CUdeviceptr va = 0;
    cudadrv::CUmemGenericAllocationHandle mem = 0;
    size_t size = 0;
    void* os_handle = nullptr;
    bool ok = false;

    explicit VmmBlock(size_t min_bytes) {
        vmm = cudadrv::vmm_api();
        if (!vmm) return;
        cudadrv::MemAllocationProp prop{};
        prop.type                 = cudadrv::kMemAllocationTypePinned;
        prop.requestedHandleTypes = cudadrv::kMemHandleTypeWin32;
        prop.location.type        = cudadrv::kMemLocationTypeDevice;
        prop.location.id          = 0;
        // Hardware finding (driver 596.47): a WIN32-shareable cuMemCreate with
        // null win32HandleMetaData is CUDA_ERROR_INVALID_VALUE — an exportable
        // NT handle needs SECURITY_ATTRIBUTES (same fix as ExportablePool).
        static SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, FALSE};
        prop.win32HandleMetaData  = &sa;
        size_t gran = 0;
        if (vmm->cuMemGetAllocationGranularity(&gran, &prop,
                cudadrv::kMemAllocGranularityMinimum) != cudadrv::CUDA_SUCCESS ||
            gran == 0)
            return;
        size = ((min_bytes + gran - 1) / gran) * gran;
        if (vmm->cuMemCreate(&mem, size, &prop, 0) != cudadrv::CUDA_SUCCESS) {
            mem = 0; return;
        }
        if (vmm->cuMemAddressReserve(&va, size, 0, 0, 0) != cudadrv::CUDA_SUCCESS) {
            va = 0; unwind(); return;
        }
        if (vmm->cuMemMap(va, size, 0, mem, 0) != cudadrv::CUDA_SUCCESS) {
            vmm->cuMemAddressFree(va, size); va = 0; unwind(); return;
        }
        cudadrv::MemAccessDesc acc{};
        acc.location.type = cudadrv::kMemLocationTypeDevice;
        acc.location.id   = 0;
        acc.flags         = cudadrv::kMemAccessFlagsProtReadWrite;
        if (vmm->cuMemSetAccess(va, size, &acc, 1) != cudadrv::CUDA_SUCCESS) {
            mapped_unwind(); return;
        }
        if (vmm->cuMemExportToShareableHandle(&os_handle, mem,
                cudadrv::kMemHandleTypeWin32, 0) != cudadrv::CUDA_SUCCESS) {
            os_handle = nullptr; mapped_unwind(); return;
        }
        ok = true;
    }
    void mapped_unwind() {
        vmm->cuMemUnmap(va, size);
        vmm->cuMemAddressFree(va, size);
        va = 0;
        unwind();
    }
    void unwind() {
        if (mem) { vmm->cuMemRelease(mem); mem = 0; }
    }
    ~VmmBlock() {
        if (os_handle) CloseHandle((HANDLE)os_handle);
        if (ok) mapped_unwind(); else unwind();
    }
};

// Guard set shared by every v1.2 row: Vulkan ICD + UUID-paired single CUDA GPU
// + the optional VMM driver table.
bool vmm_rows_ready() {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return false; }
    if (!vk_cuda_ready()) { MESSAGE("no CUDA interop / not single-GPU — skipping"); return false; }
    if (!cudadrv::vmm_api()) { MESSAGE("no CUDA VMM driver API — skipping"); return false; }
    return true;
}

// Row-major 2D f32 CPU tensor over `data` (the CPU seed shape the §16 matrix uses).
CaliperTensor f32_2d(float* data, int w, int h) {
    CaliperTensor t{};
    t.struct_size = sizeof(t); t.data = data; t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_CPU;
    return t;
}

// The same tensor re-phrased as an imported-allocation update desc: data is
// IGNORED by contract (alloc + offset are the address), device is the active
// CUDA backend.
CaliperTensor import_desc(const CaliperTensor& t) {
    CaliperTensor d = t;
    d.data = nullptr;
    d.device = CALIPER_DEV_CUDA;
    return d;
}

}  // namespace

// Row 1 — byte-exact at offset 0 AND at a 512-byte offset (torch pool
// sub-allocation alignment), both through "compute-imported": no D2D copy, the
// pass reads the imported buffer in place.
TEST_CASE("gfx/Vulkan+CUDA: imported allocation f32 byte-exact at offsets 0 and 512") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();

    const int wA = 17, hA = 9, nA = wA * hA;   // grid A at offset 0
    const int wB = 5,  hB = 3, nB = wB * hB;   // grid B at offset 512
    const uint64_t offB = 512;

    VmmBlock blk(offB + (size_t)nB * sizeof(float));
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine "
                            "— shareable handles unsupported by this driver?");

    std::vector<float> dataA(nA), dataB(nB);
    for (int i = 0; i < nA; ++i) dataA[i] = (float)i;
    for (int i = 0; i < nB; ++i) dataB[i] = (float)(nB - 1 - i);

    const CaliperAllocId alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(alloc != 0);

    // Grid A (612 bytes at offset 0) and grid B (offset 512) overlap by
    // construction, so each region is written, updated, and read back before
    // the next write — the update captures the bytes present at update time.
    auto row = [&](int w, int h, uint64_t off, const std::vector<float>& data) {
        const int n = w * h;
        REQUIRE(cu->cuMemcpyHtoD(blk.va + off, data.data(),
                                 (size_t)n * sizeof(float)) == cudadrv::CUDA_SUCCESS);
        std::vector<float> seed((size_t)n, 0.0f);
        CaliperTensor t = f32_2d(seed.data(), w, h);
        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
            &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
        REQUIRE(id != 0);

        CaliperTensor d = import_desc(t);
        REQUIRE(bk.bridge->update_texture_from_alloc(id, alloc, off, &d));
        CHECK(std::string(bk.renderer->last_device_path()) == "compute-imported");

        std::vector<uint8_t> ref((size_t)n * 4);
        map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                         0.0f, (float)(n - 1), ref.data());
        CAPTURE(off); CAPTURE(w); CAPTURE(h);
        const std::vector<uint8_t> got = bk.readback(id, w, h);
        if (got != ref) {
            size_t first = 0;
            while (first < ref.size() && first < got.size() && got[first] == ref[first]) ++first;
            MESSAGE("readback mismatch: off=" << off << " got.size=" << got.size()
                    << " ref.size=" << ref.size() << " first-diff byte=" << first
                    << " got=" << (first < got.size() ? (int)got[first] : -1)
                    << " ref=" << (first < ref.size() ? (int)ref[first] : -1));
        }
        CHECK(got == ref);   // byte-exact, no tolerance
        bk.bridge->release_texture(id);
    };
    row(wA, hA, 0, dataA);
    row(wB, hB, offB, dataB);
    bk.bridge->release_allocation(alloc);
}

// Row 2 — misaligned f32 offset (4 violates minStorageBufferOffsetAlignment):
// update returns false, the texture keeps its prior pixels, and the device-path
// telemetry is untouched. Fallback, never a wrong image. (Assumes the limit is
// > 4 — 16+ on NVIDIA, and these rows are UUID-gated to NVIDIA hardware; if
// the gate ever widens to an ICD with limit <= 4, this row would need a
// different misaligned offset.)
TEST_CASE("gfx/Vulkan+CUDA: imported f32 misaligned offset falls back, pixels unchanged") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();

    const int w = 4, h = 4, n = w * h;
    std::vector<float> seed((size_t)n);
    for (int i = 0; i < n; ++i) seed[i] = (float)i;
    CaliperTensor t = f32_2d(seed.data(), w, h);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(id != 0);
    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(seed.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    REQUIRE(bk.readback(id, w, h) == ref);

    VmmBlock blk(4096);
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    const CaliperAllocId alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(alloc != 0);

    CaliperTensor d = import_desc(t);
    const std::string before = bk.renderer->last_device_path();
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(id, alloc, 4, &d));
    CHECK(std::string(bk.renderer->last_device_path()) == before);
    CHECK(bk.readback(id, w, h) == ref);   // pixels unchanged

    bk.bridge->release_allocation(alloc);
    bk.bridge->release_texture(id);
}

// Row 3 — u8 RGBA at a nonzero offset takes "blit-imported" byte-exact.
// DEVIATION from the handoff row wording ("3-channel"): the locked T5 design
// mirrors blit_u8's RGBA8-only gate (c != 4 -> false; 1/3-channel expansion is
// the CPU staging path's job), so the positive row uses C=4 and the 3-channel
// desc is asserted as the fallback contract instead. Recorded in progress.md.
TEST_CASE("gfx/Vulkan+CUDA: imported u8 RGBA at offset 512 blit-imported; 3-channel falls back") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();

    const int w = 6, h = 4, n = w * h;
    const uint64_t off = 512;
    VmmBlock blk(off + (size_t)n * 4);
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");

    std::vector<uint8_t> px((size_t)n * 4);
    for (size_t i = 0; i < px.size(); ++i) px[i] = (uint8_t)((i * 37 + 11) & 0xff);
    REQUIRE(cu->cuMemcpyHtoD(blk.va + off, px.data(), px.size()) == cudadrv::CUDA_SUCCESS);

    const CaliperAllocId alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(alloc != 0);

    // 4-channel texture (CPU seed), then the imported RGBA update.
    std::vector<uint8_t> seed4((size_t)n * 4, 0);
    CaliperTensor t4{};
    t4.struct_size = sizeof(t4); t4.data = seed4.data(); t4.dtype = CALIPER_DT_U8;
    t4.ndim = 3; t4.shape[0] = h; t4.shape[1] = w; t4.shape[2] = 4;
    t4.strides[0] = (int64_t)w * 4; t4.strides[1] = 4; t4.strides[2] = 1;
    t4.device = CALIPER_DEV_CPU;
    CaliperTextureId id4 = bk.bridge->texture_from_tensor(&t4, 0);
    REQUIRE(id4 != 0);

    CaliperTensor d4 = import_desc(t4);
    REQUIRE(bk.bridge->update_texture_from_alloc(id4, alloc, off, &d4));
    CHECK(std::string(bk.renderer->last_device_path()) == "blit-imported");
    std::vector<uint8_t> ref4((size_t)n * 4);
    expand_u8_to_rgba8(px.data(), w, h, 4, ref4.data());
    CHECK(bk.readback(id4, w, h) == ref4);

    // Fallback contract: a 3-channel u8 desc must return false (RGBA8-only
    // blit), leave the texture's pixels alone, and not claim a device path.
    std::vector<uint8_t> seed3((size_t)n * 3, 200);
    CaliperTensor t3{};
    t3.struct_size = sizeof(t3); t3.data = seed3.data(); t3.dtype = CALIPER_DT_U8;
    t3.ndim = 3; t3.shape[0] = h; t3.shape[1] = w; t3.shape[2] = 3;
    t3.strides[0] = (int64_t)w * 3; t3.strides[1] = 3; t3.strides[2] = 1;
    t3.device = CALIPER_DEV_CPU;
    CaliperTextureId id3 = bk.bridge->texture_from_tensor(&t3, 0);
    REQUIRE(id3 != 0);
    std::vector<uint8_t> ref3((size_t)n * 4);
    expand_u8_to_rgba8(seed3.data(), w, h, 3, ref3.data());
    REQUIRE(bk.readback(id3, w, h) == ref3);

    CaliperTensor d3 = import_desc(t3);
    const std::string before = bk.renderer->last_device_path();
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(id3, alloc, off, &d3));
    CHECK(std::string(bk.renderer->last_device_path()) == before);
    CHECK(bk.readback(id3, w, h) == ref3);   // pixels unchanged

    bk.bridge->release_allocation(alloc);
    bk.bridge->release_texture(id4);
    bk.bridge->release_texture(id3);
}

// Row 4 — release + reuse: update-after-release is false (the fallback
// contract — the applet CPU-stages, never crashes), and a 50x import/release
// loop yields strictly increasing ids (never reused while live).
TEST_CASE("gfx/Vulkan+CUDA: imported alloc release contract + 50x import/release ids increase") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();

    const int w = 4, h = 4, n = w * h;
    VmmBlock blk((size_t)n * sizeof(float));
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");

    std::vector<float> data((size_t)n);
    for (int i = 0; i < n; ++i) data[i] = (float)((i * 5 + 3) % n);
    REQUIRE(cu->cuMemcpyHtoD(blk.va, data.data(),
                             (size_t)n * sizeof(float)) == cudadrv::CUDA_SUCCESS);

    std::vector<float> seed((size_t)n, 0.0f);
    CaliperTensor t = f32_2d(seed.data(), w, h);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(id != 0);

    CaliperAllocId alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(alloc != 0);

    CaliperTensor d = import_desc(t);
    REQUIRE(bk.bridge->update_texture_from_alloc(id, alloc, 0, &d));
    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    CHECK(bk.readback(id, w, h) == ref);

    bk.bridge->release_allocation(alloc);
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(id, alloc, 0, &d));
    CHECK(bk.readback(id, w, h) == ref);      // keeps the last good pixels

    CaliperAllocId prev = alloc;
    for (int i = 0; i < 50; ++i) {
        const CaliperAllocId a = bk.bridge->import_allocation(
            blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
        REQUIRE(a != 0);
        CHECK(a > prev);                       // strictly increasing, never reused
        prev = a;
        bk.bridge->release_allocation(a);
    }
    bk.bridge->release_texture(id);
}

// Row 5 — bounds: a window with offset + extent > imported size is rejected by
// the host-side gate (the renderer re-checks against the real allocation too),
// with pixels and telemetry untouched.
TEST_CASE("gfx/Vulkan+CUDA: imported window exceeding the allocation is rejected") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();

    const int w = 4, h = 4, n = w * h;                 // 64-byte f32 window
    std::vector<float> seed((size_t)n);
    for (int i = 0; i < n; ++i) seed[i] = (float)i;
    CaliperTensor t = f32_2d(seed.data(), w, h);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(id != 0);
    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(seed.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    REQUIRE(bk.readback(id, w, h) == ref);

    VmmBlock blk(4096);
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    const CaliperAllocId alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(alloc != 0);

    CaliperTensor d = import_desc(t);
    const std::string before = bk.renderer->last_device_path();
    // Window entirely past the end, and one straddling the end.
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(id, alloc, blk.size, &d));
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(id, alloc, blk.size - 32, &d));
    CHECK(std::string(bk.renderer->last_device_path()) == before);
    CHECK(bk.readback(id, w, h) == ref);   // untouched

    bk.bridge->release_allocation(alloc);
    bk.bridge->release_texture(id);
}
// ===========================================================================
// caliper.geometry.v1 rows (points from imported allocations). Same guards as
// the v1.2 rows; byte-exact where rasterization is deterministic: 1-px points
// at exact pixel centers land on exactly those pixels with exactly the LUT
// color (additive blend onto the cleared background), and every other pixel
// equals the clear color.
// ===========================================================================
namespace {

// NDC position whose 1-px point covers exactly pixel (px,py) of a WxH view
// under the backend's GL-style (+y up, negative-viewport) mapping.
void ndc_for_pixel(int px, int py, int w, int h, float* out3) {
    out3[0] = 2.0f * ((float)px + 0.5f) / (float)w - 1.0f;
    out3[1] = 1.0f - 2.0f * ((float)py + 0.5f) / (float)h;
    out3[2] = 0.0f;
}

CaliperGeomCamera identity_cam() {
    CaliperGeomCamera c{};
    for (int i = 0; i < 4; ++i) { c.view[i * 4 + i] = 1.f; c.proj[i * 4 + i] = 1.f; }
    return c;
}

// Expected image: clear color everywhere, LUT/flat color at the given pixels.
std::vector<uint8_t> geom_ref(int w, int h, uint32_t clear_rgba,
                              const std::vector<std::pair<int,int>>& px,
                              const std::vector<uint32_t>& color_rgba) {
    std::vector<uint8_t> ref((size_t)w * h * 4);
    for (int i = 0; i < w * h; ++i) {
        ref[(size_t)i * 4 + 0] = (uint8_t)(clear_rgba         & 0xFF);
        ref[(size_t)i * 4 + 1] = (uint8_t)((clear_rgba >> 8)  & 0xFF);
        ref[(size_t)i * 4 + 2] = (uint8_t)((clear_rgba >> 16) & 0xFF);
        ref[(size_t)i * 4 + 3] = (uint8_t)((clear_rgba >> 24) & 0xFF);
    }
    for (size_t k = 0; k < px.size(); ++k) {
        const size_t at = ((size_t)px[k].second * w + px[k].first) * 4;
        ref[at + 0] = (uint8_t)(color_rgba[k]         & 0xFF);
        ref[at + 1] = (uint8_t)((color_rgba[k] >> 8)  & 0xFF);
        ref[at + 2] = (uint8_t)((color_rgba[k] >> 16) & 0xFF);
        ref[at + 3] = (uint8_t)((color_rgba[k] >> 24) & 0xFF);
    }
    return ref;
}

}  // namespace

TEST_CASE("gfx/geometry: imported points byte-exact — colormap extremes at a nonzero offset") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    if (bk.bridge->geom_caps() == 0) { MESSAGE("no geometry path — skipping"); return; }
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 64, H = 64;
    const uint64_t pos_off = 512, attr_off = 2048;
    VmmBlock blk(4096);
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");

    // Three points at distinct pixel centers; attrs hit LUT[0], LUT[255],
    // LUT[128] (t=0.5 -> idx = 0.5*255+0.5 = 128 exactly).
    const std::vector<std::pair<int,int>> px = {{3, 5}, {40, 22}, {63, 63}};
    float pos[9];
    for (int i = 0; i < 3; ++i)
        ndc_for_pixel(px[i].first, px[i].second, W, H, pos + 3 * i);
    const float attrs[3] = {0.0f, 1.0f, 0.5f};
    REQUIRE(cu->cuMemcpyHtoD(blk.va + pos_off, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(blk.va + attr_off, attrs, sizeof(attrs)) == cudadrv::CUDA_SUCCESS);

    const CaliperAllocId alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(alloc != 0);
    CaliperTextureId view = bk.bridge->geom_create_view(W, H);
    REQUIRE(view != 0);

    CaliperGeomCamera cam = identity_cam();
    const uint32_t clear = 0xFF000000u;   // opaque black
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, pos_off, 3,
                                        alloc, attr_off, CALIPER_CMAP_VIRIDIS,
                                        0.f, 1.f, 1.f, clear));
    CHECK(std::string(bk.renderer->last_device_path()) == "points-imported");

    const uint32_t* lut = colormap_lut(CALIPER_CMAP_VIRIDIS);
    const std::vector<uint8_t> ref =
        geom_ref(W, H, clear, px, {lut[0], lut[255], lut[128]});
    const std::vector<uint8_t> got = bk.readback(view, W, H);
    if (got != ref) {
        size_t first = 0;
        while (first < ref.size() && first < got.size() && got[first] == ref[first]) ++first;
        MESSAGE("geom readback mismatch: first-diff byte=" << first
                << " got=" << (first < got.size() ? (int)got[first] : -1)
                << " ref=" << (first < ref.size() ? (int)ref[first] : -1));
    }
    CHECK(got == ref);   // byte-exact, no tolerances

    // Flat path: attr_alloc 0 -> pure white points, same geometry.
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, pos_off, 3,
                                        0, 0, 0, 0.f, 1.f, 1.f, clear));
    const std::vector<uint8_t> ref_flat =
        geom_ref(W, H, clear, px, {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu});
    CHECK(bk.readback(view, W, H) == ref_flat);

    bk.bridge->release_allocation(alloc);
    bk.bridge->geom_release_view(view);
}

TEST_CASE("gfx/geometry: count 0 clears; gates keep prior pixels; released view refuses") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    if (bk.bridge->geom_caps() == 0) { MESSAGE("no geometry path — skipping"); return; }
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    VmmBlock blk(4096);
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    float p[3];
    ndc_for_pixel(7, 9, W, H, p);
    REQUIRE(cu->cuMemcpyHtoD(blk.va, p, sizeof(p)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(alloc != 0);
    CaliperTextureId view = bk.bridge->geom_create_view(W, H);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    // count 0 = pure clear to a known non-black color (r=10 g=20 b=30 a=255).
    const uint32_t teal = 10u | (20u << 8) | (30u << 16) | (255u << 24);
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, 0, 0, 0, 0, 0, 0,
                                        0.f, 1.f, 1.f, teal));
    const std::vector<uint8_t> cleared = geom_ref(W, H, teal, {}, {});
    CHECK(bk.readback(view, W, H) == cleared);

    // A real frame, then every gate: false + pixels stay exactly that frame.
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1, 0, 0, 0,
                                        0.f, 1.f, 1.f, 0xFF000000u));
    const std::vector<uint8_t> frame = bk.readback(view, W, H);
    const std::string before = bk.renderer->last_device_path();
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 2, 1, 0, 0, 0,
                                            0.f, 1.f, 1.f, 0u));            // misaligned
    // OOB against the REAL (granularity-padded) size — 4096 requested bytes
    // became a 2 MiB block, so the count must be derived, not assumed.
    const uint64_t oob_count = blk.size / 12u + 1u;
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, oob_count, 0, 0, 0,
                                            0.f, 1.f, 1.f, 0u));            // OOB
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, 999u, 0, 1, 0, 0, 0,
                                            0.f, 1.f, 1.f, 0u));            // unknown alloc
    CHECK_FALSE(bk.bridge->geom_draw_points(view, nullptr, alloc, 0, 1, 0, 0, 0,
                                            0.f, 1.f, 1.f, 0u));            // null cam
    CHECK(std::string(bk.renderer->last_device_path()) == before);
    CHECK(bk.readback(view, W, H) == frame);

    // Released alloc, then released view: both refuse.
    bk.bridge->release_allocation(alloc);
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1, 0, 0, 0,
                                            0.f, 1.f, 1.f, 0u));
    bk.bridge->geom_release_view(view);
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, 0, 0, 0, 0, 0, 0,
                                            0.f, 1.f, 1.f, 0u));
}

#endif  // _WIN32
#endif  // CALIPER_HAVE_VULKAN
