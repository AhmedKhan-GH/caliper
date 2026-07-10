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
#include <caliper/services/geometry_v1_2.h>

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

// Row A — one non-indexed full-viewport triangle, FLAT/UNLIT/OPAQUE, no depth.
// The (-1,-1),(3,-1),(-1,3) trick covers every pixel center unambiguously, so
// the whole frame is the flat color: identical to geom_ref's clear-only image.
TEST_CASE("gfx/metal geometry.v1_1: unindexed triangle, FLAT, OPAQUE is byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 16, H = 16;
    const float pos[] = {
        -1.0f, -1.0f, 0.5f,
         3.0f, -1.0f, 0.5f,
        -1.0f,  3.0f, 0.5f,
    };
    id<MTLBuffer> pos_buf = device_buffer(pos, sizeof(pos));
    REQUIRE((pos_buf != nil));
    CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pos_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);

    CaliperGeomCamera cam = identity_cam();
    const uint32_t flat = 0xFF3377AAu;   // little-endian RGBA8
    CaliperGeomDraw d{};
    d.pos_alloc = pos_alloc;
    d.vertex_count = 3;
    d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.color_mode = CALIPER_GEOM_COLOR_FLAT;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.flat_rgba = flat;
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;

    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(std::string(bk.renderer->last_device_path()) == "primitives-imported");
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, flat, {}, {}));

    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view);
}

// Row B — indexed quad (4 verts, 6 u32 indices) with all buffers at nonzero,
// 4-byte-aligned offsets (buffer starts filled with garbage to prove offsets
// are honored). COLOR_COLORMAP with every vertex attr equal keeps the color
// flat and byte-exact: attr==1 -> LUT[255], attr==0 -> LUT[0]. The quad covers
// exactly pixel columns/rows 8..23 of a 32x32 view (NDC edges at +/-0.5, which
// land on integer pixel boundaries — never through a pixel center).
TEST_CASE("gfx/metal geometry.v1_1: indexed quad pulls u32 indices and LUT extremes at nonzero offsets") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    const float pos[12] = {
        -0.5f,  0.5f, 0.5f,   // v0 top-left     -> pixel (8,8)
         0.5f,  0.5f, 0.5f,   // v1 top-right    -> col 24, row 8
        -0.5f, -0.5f, 0.5f,   // v2 bottom-left  -> col 8, row 24
         0.5f, -0.5f, 0.5f,   // v3 bottom-right -> col 24, row 24
    };
    const uint32_t idx[6] = {0, 1, 2, 2, 1, 3};   // culling off; winding free
    const uint64_t pos_off = 256, idx_off = 128, attr_off = 64;

    std::vector<uint8_t> pos_bytes(pos_off + sizeof(pos), 0xAB);
    std::memcpy(pos_bytes.data() + pos_off, pos, sizeof(pos));
    std::vector<uint8_t> idx_bytes(idx_off + sizeof(idx), 0xCD);
    std::memcpy(idx_bytes.data() + idx_off, idx, sizeof(idx));

    id<MTLBuffer> pos_buf = device_buffer(pos_bytes.data(), pos_bytes.size());
    id<MTLBuffer> idx_buf = device_buffer(idx_bytes.data(), idx_bytes.size());
    REQUIRE((pos_buf != nil)); REQUIRE((idx_buf != nil));
    CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, pos_bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId idx_alloc = bk.bridge->import_allocation(
        (__bridge void*)idx_buf, idx_bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pos_alloc != 0); REQUIRE(idx_alloc != 0);

    std::vector<std::pair<int,int>> rect;   // cols 8..23, rows 8..23
    for (int y = 8; y < 24; ++y)
        for (int x = 8; x < 24; ++x) rect.emplace_back(x, y);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();
    const uint32_t* lut = colormap_lut(CALIPER_CMAP_VIRIDIS);

    auto make_draw = [&](CaliperAllocId attr_alloc) {
        CaliperGeomDraw d{};
        d.pos_alloc = pos_alloc;   d.pos_offset = pos_off;   d.vertex_count = 4;
        d.index_alloc = idx_alloc; d.index_offset = idx_off; d.index_count = 6;
        d.attr_alloc = attr_alloc; d.attr_offset = attr_off;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.colormap = CALIPER_CMAP_VIRIDIS;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // attr all 1.0 -> constant LUT[255]
    const float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<uint8_t> a1(attr_off + sizeof(ones), 0xEE);
    std::memcpy(a1.data() + attr_off, ones, sizeof(ones));
    id<MTLBuffer> a1_buf = device_buffer(a1.data(), a1.size());
    CaliperAllocId a1_alloc = bk.bridge->import_allocation(
        (__bridge void*)a1_buf, a1.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(a1_alloc != 0);
    CaliperGeomDraw d1 = make_draw(a1_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d1, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) ==
          geom_ref(W, H, 0xFF000000u, rect, std::vector<uint32_t>(rect.size(), lut[255])));

    // fresh clear, attr all 0.0 -> constant LUT[0]
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<uint8_t> a0(attr_off + sizeof(zeros), 0x11);
    std::memcpy(a0.data() + attr_off, zeros, sizeof(zeros));
    id<MTLBuffer> a0_buf = device_buffer(a0.data(), a0.size());
    CaliperAllocId a0_alloc = bk.bridge->import_allocation(
        (__bridge void*)a0_buf, a0.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(a0_alloc != 0);
    CaliperGeomDraw d0 = make_draw(a0_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d0, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) ==
          geom_ref(W, H, 0xFF000000u, rect, std::vector<uint32_t>(rect.size(), lut[0])));

    bk.bridge->release_allocation(a0_alloc);
    bk.bridge->release_allocation(a1_alloc);
    bk.bridge->release_allocation(idx_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view);
}

// Row C — the v1_1 primitives point path (ADDITIVE) must be byte-identical to
// the frozen v1 draw_points path given the same buffers. Over a black clear,
// additive one-pixel points equal the LUT color exactly, matching v1.
TEST_CASE("gfx/metal geometry.v1_1: additive points via draw_primitives match v1 draw_points byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    const std::pair<int,int> px[3] = {{3, 5}, {18, 20}, {31, 31}};
    const float attrs[3] = {0.0f, 0.5f, 1.0f};
    float pos[9];
    for (int i = 0; i < 3; ++i) ndc_for_pixel(px[i].first, px[i].second, W, H, &pos[i * 3]);

    id<MTLBuffer> pos_buf  = device_buffer(pos, sizeof(pos));
    id<MTLBuffer> attr_buf = device_buffer(attrs, sizeof(attrs));
    REQUIRE((pos_buf != nil)); REQUIRE((attr_buf != nil));
    CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId attr_alloc = bk.bridge->import_allocation(
        (__bridge void*)attr_buf, sizeof(attrs), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pos_alloc != 0); REQUIRE(attr_alloc != 0);

    const uint32_t clear = 0xFF000000u;
    CaliperGeomCamera cam = identity_cam();

    // frozen v1 path
    CaliperTextureId view1 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view1 != 0);
    REQUIRE(bk.bridge->geom_draw_points(view1, &cam, pos_alloc, 0, 3,
                                        attr_alloc, 0, CALIPER_CMAP_VIRIDIS,
                                        0.0f, 1.0f, 1.0f, clear));
    auto got1 = bk.readback(view1, W, H);

    // v1_1 primitives path
    CaliperTextureId view2 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view2 != 0);
    CaliperGeomDraw d{};
    d.pos_alloc = pos_alloc; d.vertex_count = 3;
    d.attr_alloc = attr_alloc; d.attr_offset = 0;
    d.topology = CALIPER_GEOM_TOPO_POINTS;
    d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_ADDITIVE;
    d.colormap = CALIPER_CMAP_VIRIDIS;
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
    REQUIRE(bk.bridge->geom_draw_primitives(view2, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), clear));
    CHECK(std::string(bk.renderer->last_device_path()) == "primitives-imported");
    auto got2 = bk.readback(view2, W, H);

    CHECK(got1 == got2);

    bk.bridge->release_allocation(attr_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view1);
    bk.bridge->geom_release_view(view2);
}

// Row D — draw_count 0 is a pure clear of BOTH color and depth. Write depth 0.2
// with a near triangle, then a count-0 clear to teal, then a far (z=0.9) DEPTH_TEST
// triangle: it only draws if the count-0 clear reset depth to 1.0.
TEST_CASE("gfx/metal geometry.v1_1: draw_count 0 is a pure clear, and clears depth to 1.0") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 16, H = 16;
    const float near_pos[9] = {-1.f, -1.f, 0.2f,  3.f, -1.f, 0.2f, -1.f, 3.f, 0.2f};
    const float far_pos[9]  = {-1.f, -1.f, 0.9f,  3.f, -1.f, 0.9f, -1.f, 3.f, 0.9f};
    id<MTLBuffer> near_buf = device_buffer(near_pos, sizeof(near_pos));
    id<MTLBuffer> far_buf  = device_buffer(far_pos,  sizeof(far_pos));
    REQUIRE((near_buf != nil)); REQUIRE((far_buf != nil));
    CaliperAllocId near_alloc = bk.bridge->import_allocation(
        (__bridge void*)near_buf, sizeof(near_pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId far_alloc = bk.bridge->import_allocation(
        (__bridge void*)far_buf, sizeof(far_pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(near_alloc != 0); REQUIRE(far_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    auto tri_draw = [&](CaliperAllocId a, uint32_t depth_flags, uint32_t color) {
        CaliperGeomDraw d{};
        d.pos_alloc = a; d.vertex_count = 3;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.depth_flags = depth_flags;
        d.flat_rgba = color;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // (1) near green, TEST|WRITE -> colors view, writes depth 0.2
    const uint32_t green = 0xFF00FF00u;
    CaliperGeomDraw d1 = tri_draw(near_alloc,
        CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE, green);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d1, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, green, {}, {}));

    // (2) draw_count 0 -> pure clear to teal (and depth cleared to 1.0)
    const uint32_t teal = 10u | (20u << 8) | (30u << 16) | (255u << 24);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, nullptr, 0,
                                            sizeof(CaliperGeomDraw), teal));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, teal, {}, {}));

    // (3) far blue, TEST only -> passes only because depth was cleared to 1.0
    const uint32_t blue = 0xFFFF0000u;
    CaliperGeomDraw d3 = tri_draw(far_alloc, CALIPER_GEOM_DEPTH_TEST, blue);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d3, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, blue, {}, {}));

    bk.bridge->release_allocation(far_alloc);
    bk.bridge->release_allocation(near_alloc);
    bk.bridge->geom_release_view(view);
}

// Row E — two overlapping OPAQUE quads with DEPTH_TEST|DEPTH_WRITE must produce
// the same frame regardless of draw order: LESS_OR_EQUAL keeps the near quad in
// the overlap either way. Quad P (near, z=0.25, green) covers pixel rect
// [8,24)x[8,24); quad Q (far, z=0.75, red) covers [16,28)x[16,28); they overlap
// in [16,24)x[16,24). Both readbacks must be byte-identical AND equal the CPU
// reference (near wins overlap; each quad's non-overlap keeps its own color).
TEST_CASE("gfx/metal geometry.v1_1: overlapping depth-tested quads are draw-order-independent") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    // Quad P: near z=0.25, cols/rows 8..23  -> NDC x,y in [-0.5, 0.5].
    const float posP[18] = {
        -0.5f,  0.5f, 0.25f,   0.5f,  0.5f, 0.25f,  -0.5f, -0.5f, 0.25f,
        -0.5f, -0.5f, 0.25f,   0.5f,  0.5f, 0.25f,   0.5f, -0.5f, 0.25f,
    };
    // Quad Q: far z=0.75, cols/rows 16..27 -> NDC x in [0,0.75], y in [-0.75,0].
    const float posQ[18] = {
         0.0f,  0.0f,  0.75f,   0.75f, 0.0f,  0.75f,   0.0f, -0.75f, 0.75f,
         0.0f, -0.75f, 0.75f,   0.75f, 0.0f,  0.75f,   0.75f,-0.75f, 0.75f,
    };
    id<MTLBuffer> pbuf = device_buffer(posP, sizeof(posP));
    id<MTLBuffer> qbuf = device_buffer(posQ, sizeof(posQ));
    REQUIRE((pbuf != nil)); REQUIRE((qbuf != nil));
    CaliperAllocId pal = bk.bridge->import_allocation(
        (__bridge void*)pbuf, sizeof(posP), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId qal = bk.bridge->import_allocation(
        (__bridge void*)qbuf, sizeof(posQ), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pal != 0); REQUIRE(qal != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    const uint32_t green = 0xFF00FF00u, red = 0xFF0000FFu;
    auto quad_draw = [&](CaliperAllocId a, uint32_t color) {
        CaliperGeomDraw d{};
        d.pos_alloc = a; d.vertex_count = 6;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
        d.flat_rgba = color;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };
    CaliperGeomDraw P = quad_draw(pal, green);
    CaliperGeomDraw Q = quad_draw(qal, red);

    // Reference: paint Q red first, then P green (near P wins the overlap since
    // geom_ref lets a later entry override an earlier one at the same pixel).
    std::vector<std::pair<int,int>> px; std::vector<uint32_t> col;
    for (int y = 16; y < 28; ++y) for (int x = 16; x < 28; ++x) { px.emplace_back(x, y); col.push_back(red); }
    for (int y = 8;  y < 24; ++y) for (int x = 8;  x < 24; ++x) { px.emplace_back(x, y); col.push_back(green); }
    auto ref = geom_ref(W, H, 0xFF000000u, px, col);

    CaliperGeomDraw pq[2] = {P, Q};
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, pq, 2,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_pq = bk.readback(view, W, H);

    CaliperGeomDraw qp[2] = {Q, P};
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, qp, 2,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_qp = bk.readback(view, W, H);

    CHECK(got_pq == got_qp);
    CHECK(got_pq == ref);

    bk.bridge->release_allocation(pal);
    bk.bridge->release_allocation(qal);
    bk.bridge->geom_release_view(view);
}

// Row F — ALPHA blend equations are byte-exact (§4.2). Over an opaque-black
// clear, one COLOR_FLAT quad with flat_rgba=0x80FFFFFF (white, alpha 128),
// BLEND_ALPHA, no depth. color = 255*(128/255) + 0 = 128 exactly per channel;
// alpha = 128*1 + 255*(127/255) = 255 exactly -> rect pixel = 0xFF808080.
TEST_CASE("gfx/metal geometry.v1_1: ALPHA blend equations are byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    // Quad over cols/rows 8..23 (NDC +/-0.5); z irrelevant, no depth attachment.
    const float pos[18] = {
        -0.5f,  0.5f, 0.5f,   0.5f,  0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,
        -0.5f, -0.5f, 0.5f,   0.5f,  0.5f, 0.5f,   0.5f, -0.5f, 0.5f,
    };
    id<MTLBuffer> pbuf = device_buffer(pos, sizeof(pos));
    REQUIRE((pbuf != nil));
    CaliperAllocId pal = bk.bridge->import_allocation(
        (__bridge void*)pbuf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pal != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    const uint32_t clear = 0xFF000000u;   // opaque black
    CaliperGeomDraw d{};
    d.pos_alloc = pal; d.vertex_count = 6;
    d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.color_mode = CALIPER_GEOM_COLOR_FLAT;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_ALPHA;
    d.flat_rgba = 0x80FFFFFFu;            // white, alpha 128
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;

    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), clear));
    auto got = bk.readback(view, W, H);

    std::vector<std::pair<int,int>> rect;
    for (int y = 8; y < 24; ++y) for (int x = 8; x < 24; ++x) rect.emplace_back(x, y);
    auto ref = geom_ref(W, H, clear, rect,
                        std::vector<uint32_t>(rect.size(), 0xFF808080u));

    if (got != ref) {
        uint32_t obs = metal_geom_pixel_rgba(got, W, 8, 8);   // interior rect pixel
        FAIL("ALPHA blend byte mismatch at (8,8): R=", (int)(obs & 0xFF),
             " G=", (int)((obs >> 8) & 0xFF), " B=", (int)((obs >> 16) & 0xFF),
             " A=", (int)((obs >> 24) & 0xFF), " expected R=128 G=128 B=128 A=255");
    }
    CHECK(got == ref);

    bk.bridge->release_allocation(pal);
    bk.bridge->geom_release_view(view);
}

// Row G — two axis-aligned 1-px LINES crossing. Horizontal along pixel row 10
// (x 4..27), vertical along pixel column 20 (y 3..28). OPAQUE white, no depth.
// The CPU reference colors every pixel of each segment (the crossing pixel once,
// OPAQUE so no double-blend). The four segment ENDPOINT pixels are masked
// (Metal/Vulkan diamond-exit endpoint rules differ) by overwriting them with the
// GPU's own bytes before the compare.
TEST_CASE("gfx/metal geometry.v1_1: axis-aligned 1-px LINES cross, endpoints masked") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    float pos[12];
    ndc_for_pixel(4,  10, W, H, &pos[0]);   // horizontal, left end   (pixel 4,10)
    ndc_for_pixel(27, 10, W, H, &pos[3]);   // horizontal, right end  (pixel 27,10)
    ndc_for_pixel(20, 3,  W, H, &pos[6]);   // vertical, top end      (pixel 20,3)
    ndc_for_pixel(20, 28, W, H, &pos[9]);   // vertical, bottom end   (pixel 20,28)
    id<MTLBuffer> pbuf = device_buffer(pos, sizeof(pos));
    REQUIRE((pbuf != nil));
    CaliperAllocId pal = bk.bridge->import_allocation(
        (__bridge void*)pbuf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pal != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    CaliperGeomDraw d{};
    d.pos_alloc = pal; d.vertex_count = 4;
    d.topology = CALIPER_GEOM_TOPO_LINES;
    d.color_mode = CALIPER_GEOM_COLOR_FLAT;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.flat_rgba = 0xFFFFFFFFu;
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;

    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got = bk.readback(view, W, H);

    std::vector<std::pair<int,int>> px;
    for (int x = 4; x <= 27; ++x) px.emplace_back(x, 10);   // horizontal row 10
    for (int y = 3; y <= 28; ++y) px.emplace_back(20, y);   // vertical col 20
    auto ref = geom_ref(W, H, 0xFF000000u, px,
                        std::vector<uint32_t>(px.size(), 0xFFFFFFFFu));

    // Mask the 4 segment endpoints: copy the GPU's own bytes into the reference.
    const std::pair<int,int> ep[4] = {{4,10},{27,10},{20,3},{20,28}};
    for (const auto& e : ep) {
        const size_t at = ((size_t)e.second * W + e.first) * 4;
        for (int c = 0; c < 4; ++c) ref[at + c] = got[at + c];
    }
    CHECK(got == ref);

    bk.bridge->release_allocation(pal);
    bk.bridge->geom_release_view(view);
}

// Row H — LAMBERT headlight shading within +/-2 LSB. Full-viewport triangle,
// COLOR_FLAT mid-gray 0xFFB4B4B4 (180), SHADE_LAMBERT, normals required.
// Case 1: normals (0,0,1) -> lit=0.30+0.70*1.0=1.0 -> 180. Case 2: normals
// (sin60,0,cos60)=(0.8660254,0,0.5) -> lit=0.30+0.70*0.5=0.65 -> round(180*0.65)
// =117. Alpha stays 255 (Lambert scales rgb only). nmat is identity here.
TEST_CASE("gfx/metal geometry.v1_1: LAMBERT headlight shading within +/-2 LSB") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 16, H = 16;
    const float pos[9] = { -1.0f, -1.0f, 0.5f,  3.0f, -1.0f, 0.5f, -1.0f, 3.0f, 0.5f };
    id<MTLBuffer> pbuf = device_buffer(pos, sizeof(pos));
    REQUIRE((pbuf != nil));
    CaliperAllocId pal = bk.bridge->import_allocation(
        (__bridge void*)pbuf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pal != 0);

    const float n1[9] = { 0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f };
    const float s = 0.8660254f;
    const float n2[9] = { s, 0.0f, 0.5f,  s, 0.0f, 0.5f,  s, 0.0f, 0.5f };
    id<MTLBuffer> n1_buf = device_buffer(n1, sizeof(n1));
    id<MTLBuffer> n2_buf = device_buffer(n2, sizeof(n2));
    REQUIRE((n1_buf != nil)); REQUIRE((n2_buf != nil));
    CaliperAllocId n1_alloc = bk.bridge->import_allocation(
        (__bridge void*)n1_buf, sizeof(n1), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId n2_alloc = bk.bridge->import_allocation(
        (__bridge void*)n2_buf, sizeof(n2), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(n1_alloc != 0); REQUIRE(n2_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    // channel-wise |got-ref| <= tol compare (float lighting rounds to nearest)
    auto within = [](const std::vector<uint8_t>& g, const std::vector<uint8_t>& r, int tol) {
        if (g.size() != r.size()) return false;
        for (size_t i = 0; i < g.size(); ++i) {
            int diff = (int)g[i] - (int)r[i];
            if (diff < 0) diff = -diff;
            if (diff > tol) return false;
        }
        return true;
    };

    auto lambert_draw = [&](CaliperAllocId n_alloc) {
        CaliperGeomDraw d{};
        d.pos_alloc = pal; d.vertex_count = 3;
        d.normal_alloc = n_alloc;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.flat_rgba = 0xFFB4B4B4u;   // mid-gray 180, opaque
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // Case 1: lit=1.0 -> 0xB4 per channel.
    CaliperGeomDraw d1 = lambert_draw(n1_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d1, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got1 = bk.readback(view, W, H);
    auto ref1 = geom_ref(W, H, 0xFFB4B4B4u, {}, {});
    if (!within(got1, ref1, 2)) {
        uint32_t o = metal_geom_pixel_rgba(got1, W, W / 2, H / 2);
        FAIL("LAMBERT case1 out of tol: got R=", (int)(o & 0xFF), " G=", (int)((o >> 8) & 0xFF),
             " B=", (int)((o >> 16) & 0xFF), " A=", (int)((o >> 24) & 0xFF), " expected 180,180,180,255");
    }
    CHECK(within(got1, ref1, 2));

    // Case 2: lit=0.65 -> 117 per RGB channel, alpha 255 (0xFF757575).
    CaliperGeomDraw d2 = lambert_draw(n2_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d2, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got2 = bk.readback(view, W, H);
    auto ref2 = geom_ref(W, H, 0xFF757575u, {}, {});
    if (!within(got2, ref2, 2)) {
        uint32_t o = metal_geom_pixel_rgba(got2, W, W / 2, H / 2);
        FAIL("LAMBERT case2 out of tol: got R=", (int)(o & 0xFF), " G=", (int)((o >> 8) & 0xFF),
             " B=", (int)((o >> 16) & 0xFF), " A=", (int)((o >> 24) & 0xFF), " expected 117,117,117,255");
    }
    CHECK(within(got2, ref2, 2));

    bk.bridge->release_allocation(n2_alloc);
    bk.bridge->release_allocation(n1_alloc);
    bk.bridge->release_allocation(pal);
    bk.bridge->geom_release_view(view);
}

// Row I — wireframe-over-mesh: a coplanar LESS_OR_EQUAL line overlay wins.
// Draw 0: full-viewport triangle at z=0.5, FLAT dark blue, DEPTH_TEST|WRITE.
// Draw 1: the Row-G cross at the SAME z=0.5, FLAT white, DEPTH_TEST only (no
// WRITE). LESS_OR_EQUAL (§4.2) lets the coplanar lines paint over the mesh.
// Reference: blue everywhere, white along the two segments; the 4 line endpoints
// are masked as in Row G.
TEST_CASE("gfx/metal geometry.v1_1: wireframe-over-mesh coplanar LESS_OR_EQUAL overlay") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    const float tri[9] = { -1.0f, -1.0f, 0.5f,  3.0f, -1.0f, 0.5f, -1.0f, 3.0f, 0.5f };
    float line[12];
    ndc_for_pixel(4,  10, W, H, &line[0]); line[2]  = 0.5f;
    ndc_for_pixel(27, 10, W, H, &line[3]); line[5]  = 0.5f;
    ndc_for_pixel(20, 3,  W, H, &line[6]); line[8]  = 0.5f;
    ndc_for_pixel(20, 28, W, H, &line[9]); line[11] = 0.5f;

    id<MTLBuffer> tri_buf  = device_buffer(tri,  sizeof(tri));
    id<MTLBuffer> line_buf = device_buffer(line, sizeof(line));
    REQUIRE((tri_buf != nil)); REQUIRE((line_buf != nil));
    CaliperAllocId tri_alloc = bk.bridge->import_allocation(
        (__bridge void*)tri_buf, sizeof(tri), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId line_alloc = bk.bridge->import_allocation(
        (__bridge void*)line_buf, sizeof(line), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(tri_alloc != 0); REQUIRE(line_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    const uint32_t blue = 0xFF800000u;   // dark blue (B=128), opaque
    CaliperGeomDraw dtri{};
    dtri.pos_alloc = tri_alloc; dtri.vertex_count = 3;
    dtri.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    dtri.color_mode = CALIPER_GEOM_COLOR_FLAT;
    dtri.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    dtri.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    dtri.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
    dtri.flat_rgba = blue;
    dtri.vmin = 0.0f; dtri.vmax = 1.0f; dtri.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) dtri.model[i * 4 + i] = 1.0f;

    CaliperGeomDraw dline{};
    dline.pos_alloc = line_alloc; dline.vertex_count = 4;
    dline.topology = CALIPER_GEOM_TOPO_LINES;
    dline.color_mode = CALIPER_GEOM_COLOR_FLAT;
    dline.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    dline.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    dline.depth_flags = CALIPER_GEOM_DEPTH_TEST;   // no WRITE; coplanar overlay
    dline.flat_rgba = 0xFFFFFFFFu;
    dline.vmin = 0.0f; dline.vmax = 1.0f; dline.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) dline.model[i * 4 + i] = 1.0f;

    CaliperGeomDraw draws[2] = {dtri, dline};
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, draws, 2,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got = bk.readback(view, W, H);

    std::vector<std::pair<int,int>> px;
    for (int x = 4; x <= 27; ++x) px.emplace_back(x, 10);
    for (int y = 3; y <= 28; ++y) px.emplace_back(20, y);
    auto ref = geom_ref(W, H, blue, px,
                        std::vector<uint32_t>(px.size(), 0xFFFFFFFFu));

    const std::pair<int,int> ep[4] = {{4,10},{27,10},{20,3},{20,28}};
    for (const auto& e : ep) {
        const size_t at = ((size_t)e.second * W + e.first) * 4;
        for (int c = 0; c < 4; ++c) ref[at + c] = got[at + c];
    }
    CHECK(got == ref);

    bk.bridge->release_allocation(line_alloc);
    bk.bridge->release_allocation(tri_alloc);
    bk.bridge->geom_release_view(view);
}

// Row J — index *values* cannot be gated host-side; the vertex shader clamps
// vi = min(index[i], vertex_count-1) (§2.3). 3 points at distinct pixel centers,
// TOPO_POINTS, FLAT so color is unambiguous. Reference indices {0,1,2} light all
// three pixels; clamp indices {0,1,999} -> 999 clamps to vertex 2, so the image
// is BYTE-IDENTICAL to the reference. Pins the "defined image, no crash" contract.
TEST_CASE("gfx/metal geometry.v1_1: out-of-range index values clamp to vertex_count-1, defined image") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    const std::pair<int,int> px[3] = {{3, 5}, {18, 20}, {31, 31}};
    float pos[9];
    for (int i = 0; i < 3; ++i) ndc_for_pixel(px[i].first, px[i].second, W, H, &pos[i * 3]);
    id<MTLBuffer> pos_buf = device_buffer(pos, sizeof(pos));
    REQUIRE((pos_buf != nil));
    CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pos_alloc != 0);

    const uint32_t idx_ref[3]   = {0, 1, 2};
    const uint32_t idx_clamp[3] = {0, 1, 999};   // 999 -> clamps to vertex 2
    id<MTLBuffer> ref_buf   = device_buffer(idx_ref,   sizeof(idx_ref));
    id<MTLBuffer> clamp_buf = device_buffer(idx_clamp, sizeof(idx_clamp));
    REQUIRE((ref_buf != nil)); REQUIRE((clamp_buf != nil));
    CaliperAllocId ref_alloc = bk.bridge->import_allocation(
        (__bridge void*)ref_buf, sizeof(idx_ref), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId clamp_alloc = bk.bridge->import_allocation(
        (__bridge void*)clamp_buf, sizeof(idx_clamp), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(ref_alloc != 0); REQUIRE(clamp_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();
    const uint32_t flat = 0xFF00FF00u;   // opaque green

    auto point_draw = [&](CaliperAllocId idx_alloc) {
        CaliperGeomDraw d{};
        d.pos_alloc = pos_alloc; d.vertex_count = 3;
        d.index_alloc = idx_alloc; d.index_offset = 0; d.index_count = 3;
        d.topology = CALIPER_GEOM_TOPO_POINTS;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.flat_rgba = flat;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // reference: {0,1,2} -> all three pixels lit
    CaliperGeomDraw d_ref = point_draw(ref_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d_ref, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_ref = bk.readback(view, W, H);
    CHECK(got_ref == geom_ref(W, H, 0xFF000000u,
                              {px[0], px[1], px[2]}, {flat, flat, flat}));

    // clamp: {0,1,999} -> 999 clamps to vertex 2 -> byte-identical image
    CaliperGeomDraw d_clamp = point_draw(clamp_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d_clamp, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_clamp = bk.readback(view, W, H);
    CHECK(got_clamp == got_ref);

    bk.bridge->release_allocation(clamp_alloc);
    bk.bridge->release_allocation(ref_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view);
}

// Row K — every §2.3 gate refuses the WHOLE frame and leaves pixels untouched.
// Draw a known good frame on a DEPTH view and a NO-depth view (distinct colors),
// snapshot both readbacks + last_device_path. Then for each §2.3 violation build
// a draws[] valid EXCEPT the one item, CHECK_FALSE the return; after the battery
// re-read BOTH views and CHECK byte-for-byte equality with the snapshots (and
// last_device_path unchanged). One valid draw = full-viewport triangle: any
// refusal that leaks would change the frame.
TEST_CASE("gfx/metal geometry.v1_1: every §2.3 gate refuses the whole frame, pixels untouched") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    const int W = 32, H = 32;
    const uint32_t stride = sizeof(CaliperGeomDraw);

    // ---- source allocations (all valid; each battery case pokes ONE hole) ----
    const float tri[9] = { -1.f, -1.f, 0.5f,  3.f, -1.f, 0.5f, -1.f, 3.f, 0.5f };
    const uint32_t idx[3]  = {0, 1, 2};
    const float    nrm[9]  = { 0.f, 0.f, 1.f,  0.f, 0.f, 1.f,  0.f, 0.f, 1.f };
    const float    attr[3] = { 0.5f, 0.5f, 0.5f };
    id<MTLBuffer> tri_buf  = device_buffer(tri,  sizeof(tri));
    id<MTLBuffer> idx_buf  = device_buffer(idx,  sizeof(idx));
    id<MTLBuffer> nrm_buf  = device_buffer(nrm,  sizeof(nrm));
    id<MTLBuffer> attr_buf = device_buffer(attr, sizeof(attr));
    REQUIRE((tri_buf != nil)); REQUIRE((idx_buf != nil));
    REQUIRE((nrm_buf != nil)); REQUIRE((attr_buf != nil));
    CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        (__bridge void*)tri_buf, sizeof(tri), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId idx_alloc = bk.bridge->import_allocation(
        (__bridge void*)idx_buf, sizeof(idx), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId nrm_alloc = bk.bridge->import_allocation(
        (__bridge void*)nrm_buf, sizeof(nrm), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId attr_alloc = bk.bridge->import_allocation(
        (__bridge void*)attr_buf, sizeof(attr), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pos_alloc != 0); REQUIRE(idx_alloc != 0);
    REQUIRE(nrm_alloc != 0); REQUIRE(attr_alloc != 0);

    CaliperTextureId ndv = bk.bridge->geom_create_view_ex(W, H, 0);                       // no depth
    CaliperTextureId dv  = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH); // depth
    REQUIRE(ndv != 0); REQUIRE(dv != 0);
    CaliperGeomCamera cam = identity_cam();

    // A fully valid full-viewport-triangle draw for the NO-depth view.
    auto make_valid = [&]() {
        CaliperGeomDraw d{};
        d.pos_alloc = pos_alloc; d.vertex_count = 3;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.flat_rgba = 0xFF00AA00u;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // ---- known-good frame on each view; snapshot both + last_device_path ----
    CaliperGeomDraw good_nd = make_valid();
    REQUIRE(bk.bridge->geom_draw_primitives(ndv, &cam, &good_nd, 1, stride, 0xFF000000u));
    CaliperGeomDraw good_d = make_valid();
    good_d.flat_rgba = 0xFFAA0000u;
    good_d.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
    REQUIRE(bk.bridge->geom_draw_primitives(dv, &cam, &good_d, 1, stride, 0xFF000000u));
    const auto snap_nd = bk.readback(ndv, W, H);
    const auto snap_d  = bk.readback(dv,  W, H);
    const std::string good_path = bk.renderer->last_device_path();
    REQUIRE(snap_nd == geom_ref(W, H, 0xFF00AA00u, {}, {}));
    REQUIRE(snap_d  == geom_ref(W, H, 0xFFAA0000u, {}, {}));

    CaliperGeomDraw d;
    // 1. topology out of range
    d = make_valid(); d.topology = 5;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 2. color_mode out of range
    d = make_valid(); d.color_mode = 3;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 3. shade_mode out of range
    d = make_valid(); d.shade_mode = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 4. blend_mode out of range
    d = make_valid(); d.blend_mode = 3;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 5. reserved must be zero
    d = make_valid(); d.reserved[0] = 1;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 6. absent position source
    d = make_valid(); d.pos_alloc = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 7. misaligned pos_offset
    d = make_valid(); d.pos_offset = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 8. vertex_count 0
    d = make_valid(); d.vertex_count = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 9. pos bounds overflow: 4 verts * 12 = 48 > 36-byte alloc
    d = make_valid(); d.vertex_count = 4;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 10. indexed with misaligned index_offset
    d = make_valid(); d.index_alloc = idx_alloc; d.index_count = 3; d.index_offset = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 11. indexed with index_count*4 out of bounds (4*4=16 > 12-byte alloc)
    d = make_valid(); d.index_alloc = idx_alloc; d.index_offset = 0; d.index_count = 4;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 12. too few consumed vertices for the topology
    d = make_valid(); d.topology = CALIPER_GEOM_TOPO_LINES; d.vertex_count = 1;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    d = make_valid(); d.topology = CALIPER_GEOM_TOPO_TRIANGLES; d.vertex_count = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 13. LAMBERT without normals
    d = make_valid(); d.shade_mode = CALIPER_GEOM_SHADE_LAMBERT; d.normal_alloc = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 14. LAMBERT normal bounds overflow (24 + 3*12 = 60 > 36-byte alloc)
    d = make_valid(); d.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
    d.normal_alloc = nrm_alloc; d.normal_offset = 24;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 15. COLORMAP without attr; COLORMAP with unknown colormap id
    d = make_valid(); d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.colormap = CALIPER_CMAP_VIRIDIS; d.attr_alloc = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    d = make_valid(); d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.attr_alloc = attr_alloc; d.attr_offset = 0; d.colormap = 999;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 16. attr bounds overflow (4 + 3*4 = 16 > 12-byte alloc)
    d = make_valid(); d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.colormap = CALIPER_CMAP_VIRIDIS; d.attr_alloc = attr_alloc; d.attr_offset = 4;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 17. depth flags on the NO-depth view (must refuse, not silently ignore)
    d = make_valid(); d.depth_flags = CALIPER_GEOM_DEPTH_TEST;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 20. null camera
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, nullptr, &d, 1, stride, 0xFF000000u));
    // 21. draw_stride below the host minimum
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, 100u, 0xFF000000u));
    // 22. frame atomicity: draws[0] valid, draws[1] invalid -> whole call refused
    {
        CaliperGeomDraw twod[2] = { make_valid(), make_valid() };
        twod[1].topology = 5;
        CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, twod, 2, stride, 0xFF000000u));
    }

    // whole battery so far touched nothing
    CHECK(bk.readback(ndv, W, H) == snap_nd);
    CHECK(bk.readback(dv,  W, H) == snap_d);
    CHECK(std::string(bk.renderer->last_device_path()) == good_path);

    // 18. dead alloc: reference a released allocation (mutates alloc table).
    {
        id<MTLBuffer> sbuf = device_buffer(tri, sizeof(tri));
        REQUIRE((sbuf != nil));
        CaliperAllocId salloc = bk.bridge->import_allocation(
            (__bridge void*)sbuf, sizeof(tri), CALIPER_ALLOC_HANDLE_MTLBUFFER);
        REQUIRE(salloc != 0);
        bk.bridge->release_allocation(salloc);
        d = make_valid(); d.pos_alloc = salloc;
        CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    }
    // 19. dead view: released view id.
    {
        CaliperTextureId sview = bk.bridge->geom_create_view_ex(W, H, 0);
        REQUIRE(sview != 0);
        bk.bridge->geom_release_view(sview);
        d = make_valid();
        CHECK_FALSE(bk.bridge->geom_draw_primitives(sview, &cam, &d, 1, stride, 0xFF000000u));
    }

    // final: both live views still byte-identical to their good frames
    CHECK(bk.readback(ndv, W, H) == snap_nd);
    CHECK(bk.readback(dv,  W, H) == snap_d);
    CHECK(std::string(bk.renderer->last_device_path()) == good_path);

    bk.bridge->release_allocation(attr_alloc);
    bk.bridge->release_allocation(nrm_alloc);
    bk.bridge->release_allocation(idx_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(dv);
    bk.bridge->geom_release_view(ndv);
}

// Row L — draw_stride forward-compat. A struct that grew by 16 tail bytes must
// draw identically when the host is told the real stride: it reads
// min(stride, its own sizeof) per descriptor and steps `stride` between them.
// Part 1: one FLAT triangle via a normal array and via a GrownDraw array both
// match the CPU reference and each other. Part 2: two GrownDraw descriptors in
// one call — correct stride addressing must step 208 bytes, not 192, so draw[1]
// (a different-colored quad) lands only when stepping is right.
TEST_CASE("gfx/metal geometry.v1_1: draw_stride forward-compat, a grown struct draws identically") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));

    struct GrownDraw { CaliperGeomDraw d; uint8_t tail[16]; };
    static_assert(sizeof(GrownDraw) == sizeof(CaliperGeomDraw) + 16,
                  "GrownDraw must be exactly 16 bytes larger with no padding");

    const int W = 32, H = 32;
    const uint32_t nstride = sizeof(CaliperGeomDraw);
    const uint32_t gstride = sizeof(GrownDraw);

    const float tri[9] = { -1.f, -1.f, 0.5f,  3.f, -1.f, 0.5f, -1.f, 3.f, 0.5f };
    id<MTLBuffer> tri_buf = device_buffer(tri, sizeof(tri));
    REQUIRE((tri_buf != nil));
    CaliperAllocId tri_alloc = bk.bridge->import_allocation(
        (__bridge void*)tri_buf, sizeof(tri), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(tri_alloc != 0);

    CaliperGeomCamera cam = identity_cam();
    const uint32_t flat = 0xFF3377AAu;

    CaliperGeomDraw base{};
    base.pos_alloc = tri_alloc; base.vertex_count = 3;
    base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    base.color_mode = CALIPER_GEOM_COLOR_FLAT;
    base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    base.flat_rgba = flat;
    base.vmin = 0.0f; base.vmax = 1.0f; base.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) base.model[i * 4 + i] = 1.0f;

    const auto ref1 = geom_ref(W, H, flat, {}, {});

    // Part 1a: normal array, normal stride.
    CaliperTextureId v1 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v1 != 0);
    CaliperGeomDraw nd = base;
    REQUIRE(bk.bridge->geom_draw_primitives(v1, &cam, &nd, 1, nstride, 0xFF000000u));
    auto got_normal = bk.readback(v1, W, H);
    CHECK(got_normal == ref1);

    // Part 1b: grown array (tail zeroed), grown stride, pointer cast to base type.
    CaliperTextureId v2 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v2 != 0);
    GrownDraw g{}; g.d = base;
    REQUIRE(bk.bridge->geom_draw_primitives(
        v2, &cam, reinterpret_cast<const CaliperGeomDraw*>(&g), 1, gstride, 0xFF000000u));
    auto got_grown = bk.readback(v2, W, H);
    CHECK(got_grown == ref1);
    CHECK(got_grown == got_normal);

    // ---- Part 2: two descriptors; wrong stepping (192 vs 208) breaks draw[1] ----
    // draw[1] = a quad covering cols/rows 8..23 (NDC +/-0.5 -> pixel boundaries),
    // a different flat color, drawn OVER the full-viewport triangle.
    const float quad[18] = {
        -0.5f,  0.5f, 0.5f,   0.5f,  0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,   // tri 1
        -0.5f, -0.5f, 0.5f,   0.5f,  0.5f, 0.5f,   0.5f, -0.5f, 0.5f,   // tri 2
    };
    id<MTLBuffer> quad_buf = device_buffer(quad, sizeof(quad));
    REQUIRE((quad_buf != nil));
    CaliperAllocId quad_alloc = bk.bridge->import_allocation(
        (__bridge void*)quad_buf, sizeof(quad), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(quad_alloc != 0);

    const uint32_t colA = 0xFF00AA00u;   // triangle background
    const uint32_t colB = 0xFF0000AAu;   // quad overlay (distinct)
    CaliperGeomDraw dA = base; dA.flat_rgba = colA;
    CaliperGeomDraw dB = base;
    dB.pos_alloc = quad_alloc; dB.vertex_count = 6; dB.flat_rgba = colB;

    std::vector<std::pair<int,int>> rect;
    for (int y = 8; y < 24; ++y)
        for (int x = 8; x < 24; ++x) rect.emplace_back(x, y);
    const auto ref2 = geom_ref(W, H, colA, rect,
                               std::vector<uint32_t>(rect.size(), colB));

    // reference via normal structs
    CaliperTextureId v3 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v3 != 0);
    CaliperGeomDraw ndraws[2] = { dA, dB };
    REQUIRE(bk.bridge->geom_draw_primitives(v3, &cam, ndraws, 2, nstride, 0xFF000000u));
    auto got_two_normal = bk.readback(v3, W, H);
    CHECK(got_two_normal == ref2);

    // via grown structs, 208-byte stride
    CaliperTextureId v4 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v4 != 0);
    GrownDraw g2[2] = {}; g2[0].d = dA; g2[1].d = dB;
    REQUIRE(bk.bridge->geom_draw_primitives(
        v4, &cam, reinterpret_cast<const CaliperGeomDraw*>(g2), 2, gstride, 0xFF000000u));
    auto got_two_grown = bk.readback(v4, W, H);
    CHECK(got_two_grown == ref2);
    CHECK(got_two_grown == got_two_normal);

    bk.bridge->release_allocation(quad_alloc);
    bk.bridge->release_allocation(tri_alloc);
    bk.bridge->geom_release_view(v4);
    bk.bridge->geom_release_view(v3);
    bk.bridge->geom_release_view(v2);
    bk.bridge->geom_release_view(v1);
}

// Row [v1.2 donor] — UV pull at a poisoned nonzero offset, exact texel-center
// red, bilinear-center gray (within one RGBA8 LSB), and Lambert x texture
// (within two RGB LSB, alpha untouched). Transcribed byte-for-byte from the
// donor's Vulkan twin below; runs on mac later.
TEST_CASE("gfx/metal geometry.v1_2: UV offset, bilinear texture, and Lambert are byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device - skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));

    const int W = 16, H = 16;
    const float pos[] = {-1.f,-1.f,0.5f, 3.f,-1.f,0.5f, -1.f,3.f,0.5f};
    const uint64_t uv_off = 64;
    const float red_uv[] = {0.25f,0.25f, 0.25f,0.25f, 0.25f,0.25f};
    std::vector<uint8_t> uv_bytes(uv_off + sizeof(red_uv), 0xA5);
    std::memcpy(uv_bytes.data() + uv_off, red_uv, sizeof(red_uv));
    const float nrm[] = {0.f,0.f,-1.f, 0.f,0.f,-1.f, 0.f,0.f,-1.f};

    id<MTLBuffer> pos_buf = device_buffer(pos, sizeof(pos));
    id<MTLBuffer> uv_buf = device_buffer(uv_bytes.data(), uv_bytes.size());
    id<MTLBuffer> nrm_buf = device_buffer(nrm, sizeof(nrm));
    REQUIRE(pos_buf != nil); REQUIRE(uv_buf != nil); REQUIRE(nrm_buf != nil);
    CaliperAllocId pa = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId ua = bk.bridge->import_allocation(
        (__bridge void*)uv_buf, uv_bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId na = bk.bridge->import_allocation(
        (__bridge void*)nrm_buf, sizeof(nrm), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pa != 0); REQUIRE(ua != 0); REQUIRE(na != 0);

    const uint8_t rgba[] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    CaliperTensor td = u8_3d(rgba, 2, 2, 4);
    CaliperTextureId texture = bk.bridge->texture_from_tensor(&td, 0);
    REQUIRE(texture != 0);
    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);

    CaliperGeomDrawV1_2 d{};
    d.base.pos_alloc = pa; d.base.vertex_count = 3;
    d.base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.base.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
    d.base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.base.size_px = 1.f;
    for (int i = 0; i < 4; ++i) d.base.model[i * 4 + i] = 1.f;
    d.uv_alloc = ua; d.uv_offset = uv_off; d.texture = texture;
    CaliperGeomCamera cam = identity_cam();

    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, 0xFF0000FFu, {}, {}));

    const float mid_uv[] = {0.5f,0.5f, 0.5f,0.5f, 0.5f,0.5f};
    std::memcpy((uint8_t*)uv_buf.contents + uv_off, mid_uv, sizeof(mid_uv));
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    {
        const auto got = bk.readback(view, W, H);
        CHECK((got == geom_ref(W, H, 0xFF7F7F7Fu, {}, {}) ||
               got == geom_ref(W, H, 0xFF808080u, {}, {})));
    }

    std::memcpy((uint8_t*)uv_buf.contents + uv_off, red_uv, sizeof(red_uv));
    d.base.normal_alloc = na;
    d.base.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    {
        const auto got = bk.readback(view, W, H);
        CHECK((got == geom_ref(W, H, 0xFF00004Cu, {}, {}) ||
               got == geom_ref(W, H, 0xFF00004Du, {}, {})));
    }

    bk.bridge->geom_release_view(view);
    bk.bridge->release_texture(texture);
    bk.bridge->release_allocation(na);
    bk.bridge->release_allocation(ua);
    bk.bridge->release_allocation(pa);
}

// Row [v1.2 clamp-to-edge] — a full-viewport quad whose per-vertex UV is
// (0.5 + x_ndc, 0.5 + y_ndc), so UV spans -0.5..1.5 across the 2x2 texture.
// FLAT (UNLIT) so nothing but the sample colors the pixel. Each read pixel sits
// deep in an out-of-range corner (|beyond [0,1]| = 0.4375 >> 0.125 = a quarter
// texel), so clamp-to-edge samples the nearest edge texel with no bilinear mix.
TEST_CASE("gfx/metal geometry.v1_2: out-of-range UVs clamp to edge texels byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device - skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));

    const int W = 16, H = 16;
    const float pos[] = {
        -1.f,-1.f,0.5f,  1.f,-1.f,0.5f,  -1.f,1.f,0.5f,
        -1.f, 1.f,0.5f,  1.f,-1.f,0.5f,   1.f,1.f,0.5f,
    };
    const uint64_t uv_off = 64;
    const float uv[] = {
        -0.5f,-0.5f,  1.5f,-0.5f,  -0.5f,1.5f,
        -0.5f, 1.5f,  1.5f,-0.5f,   1.5f,1.5f,
    };
    std::vector<uint8_t> uv_bytes(uv_off + sizeof(uv), 0xA5);
    std::memcpy(uv_bytes.data() + uv_off, uv, sizeof(uv));

    id<MTLBuffer> pos_buf = device_buffer(pos, sizeof(pos));
    id<MTLBuffer> uv_buf = device_buffer(uv_bytes.data(), uv_bytes.size());
    REQUIRE(pos_buf != nil); REQUIRE(uv_buf != nil);
    CaliperAllocId pa = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId ua = bk.bridge->import_allocation(
        (__bridge void*)uv_buf, uv_bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pa != 0); REQUIRE(ua != 0);

    const uint8_t rgba[] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    CaliperTensor td = u8_3d(rgba, 2, 2, 4);
    CaliperTextureId texture = bk.bridge->texture_from_tensor(&td, 0);
    REQUIRE(texture != 0);
    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);

    CaliperGeomDrawV1_2 d{};
    d.base.pos_alloc = pa; d.base.vertex_count = 6;
    d.base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.base.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
    d.base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.base.size_px = 1.f;
    for (int i = 0; i < 4; ++i) d.base.model[i * 4 + i] = 1.f;
    d.uv_alloc = ua; d.uv_offset = uv_off; d.texture = texture;
    CaliperGeomCamera cam = identity_cam();

    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    const auto got = bk.readback(view, W, H);
    auto at = [&](int x, int y) {
        const size_t o = ((size_t)y * W + x) * 4;
        return (uint32_t)got[o] | ((uint32_t)got[o + 1] << 8) |
               ((uint32_t)got[o + 2] << 16) | ((uint32_t)got[o + 3] << 24);
    };
    CHECK(at(0, 15)  == 0xFF0000FFu);   // u<0, v<0 -> col0,row0 red
    CHECK(at(15, 15) == 0xFF00FF00u);   // u>1, v<0 -> col1,row0 green
    CHECK(at(0, 0)   == 0xFFFF0000u);   // u<0, v>1 -> col0,row1 blue
    CHECK(at(15, 0)  == 0xFFFFFFFFu);   // u>1, v>1 -> col1,row1 white

    bk.bridge->geom_release_view(view);
    bk.bridge->release_texture(texture);
    bk.bridge->release_allocation(ua);
    bk.bridge->release_allocation(pa);
}

// Row [v1.2 compat] — the same non-textured indexed COLORMAP+LAMBERT mesh drawn
// through the frozen v1.1 entry (stride 192) into view A and through the v1.2
// entry (zeroed tail, stride 216) into view B. Full-image equality guards against
// DIVERGENCE between the two entry points — stride handling, tail defaults,
// pipeline selection — not shader correctness: a shared-shader break corrupts both
// paths identically and still compares equal. Absolute shader correctness is
// guarded by the byte-exact rows above.
TEST_CASE("gfx/metal geometry.v1_2: v1.1 and v1.2 non-textured draws are byte-identical") {
    if (!metal_env().ok) { MESSAGE("no Metal device - skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));

    const int W = 32, H = 32;
    const float pos[9] = { -1.f,-1.f,0.5f,  3.f,-1.f,0.5f,  -1.f,3.f,0.5f };
    const uint32_t idx[3] = {0, 1, 2};
    const float nrm[9] = { 0.f,0.f,-1.f,  0.f,0.f,-1.f,  0.f,0.f,-1.f };
    const float attr[3] = { 0.5f, 0.5f, 0.5f };
    id<MTLBuffer> pos_buf = device_buffer(pos, sizeof(pos));
    id<MTLBuffer> idx_buf = device_buffer(idx, sizeof(idx));
    id<MTLBuffer> nrm_buf = device_buffer(nrm, sizeof(nrm));
    id<MTLBuffer> attr_buf = device_buffer(attr, sizeof(attr));
    REQUIRE(pos_buf != nil); REQUIRE(idx_buf != nil);
    REQUIRE(nrm_buf != nil); REQUIRE(attr_buf != nil);
    CaliperAllocId pa = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId ia = bk.bridge->import_allocation(
        (__bridge void*)idx_buf, sizeof(idx), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId na = bk.bridge->import_allocation(
        (__bridge void*)nrm_buf, sizeof(nrm), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId aa = bk.bridge->import_allocation(
        (__bridge void*)attr_buf, sizeof(attr), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pa != 0); REQUIRE(ia != 0); REQUIRE(na != 0); REQUIRE(aa != 0);

    CaliperGeomDraw base{};
    base.pos_alloc = pa; base.vertex_count = 3;
    base.index_alloc = ia; base.index_count = 3;
    base.normal_alloc = na; base.attr_alloc = aa;
    base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    base.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    base.colormap = CALIPER_CMAP_VIRIDIS;
    base.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
    base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    base.vmin = 0.f; base.vmax = 1.f; base.size_px = 1.f;
    for (int i = 0; i < 4; ++i) base.model[i * 4 + i] = 1.f;
    CaliperGeomCamera cam = identity_cam();

    CaliperTextureId va = bk.bridge->geom_create_view_ex(W, H, 0);
    CaliperTextureId vb = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(va != 0); REQUIRE(vb != 0);

    REQUIRE(bk.bridge->geom_draw_primitives(
        va, &cam, &base, 1, sizeof(CaliperGeomDraw), 0xFF000000u));
    CaliperGeomDrawV1_2 d{}; d.base = base;   // zeroed UV/texture tail
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        vb, &cam, &d, 1, sizeof(CaliperGeomDrawV1_2), 0xFF000000u));

    CHECK(bk.readback(va, W, H) == bk.readback(vb, W, H));
    // Non-triviality: view A must actually rasterize the mesh, else a blank-vs-blank
    // match would pass the equality above vacuously.
    CHECK(bk.readback(va, W, H) != geom_ref(W, H, 0xFF000000u, {}, {}));

    bk.bridge->geom_release_view(vb);
    bk.bridge->geom_release_view(va);
    bk.bridge->release_allocation(aa);
    bk.bridge->release_allocation(na);
    bk.bridge->release_allocation(ia);
    bk.bridge->release_allocation(pa);
}

// Row [v1.2 refusal purity] — a valid textured draw fills the view (pre-image),
// then four COLOR_TEXTURE gate breaches are attempted in order; each must refuse
// AND leave the view byte-identical to the pre-image, cumulatively (the Phase-B
// T3 pattern): (a) uv_alloc released after import, (b) texture names a geometry
// view (the target itself), (c) texture is a released texture id, (d) a v1.2
// submission with a short (192) draw_stride.
TEST_CASE("gfx/metal geometry.v1_2: textured gate refusals leave the view untouched (cumulative)") {
    if (!metal_env().ok) { MESSAGE("no Metal device - skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));

    const int W = 16, H = 16;
    const float pos[9] = { -1.f,-1.f,0.5f,  3.f,-1.f,0.5f,  -1.f,3.f,0.5f };
    const float uv[6]  = { 0.25f,0.25f, 0.25f,0.25f, 0.25f,0.25f };   // -> red
    id<MTLBuffer> pos_buf = device_buffer(pos, sizeof(pos));
    id<MTLBuffer> uv_buf = device_buffer(uv, sizeof(uv));
    REQUIRE(pos_buf != nil); REQUIRE(uv_buf != nil);
    CaliperAllocId pa = bk.bridge->import_allocation(
        (__bridge void*)pos_buf, sizeof(pos), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperAllocId ua = bk.bridge->import_allocation(
        (__bridge void*)uv_buf, sizeof(uv), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(pa != 0); REQUIRE(ua != 0);

    const uint8_t rgba[] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    CaliperTensor td = u8_3d(rgba, 2, 2, 4);
    CaliperTextureId texture = bk.bridge->texture_from_tensor(&td, 0);
    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(texture != 0); REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    auto make_valid = [&]() {
        CaliperGeomDrawV1_2 dd{};
        dd.base.pos_alloc = pa; dd.base.vertex_count = 3;
        dd.base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        dd.base.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
        dd.base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        dd.base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        dd.base.size_px = 1.f;
        for (int i = 0; i < 4; ++i) dd.base.model[i * 4 + i] = 1.f;
        dd.uv_alloc = ua; dd.uv_offset = 0; dd.texture = texture;
        return dd;
    };

    CaliperGeomDrawV1_2 good = make_valid();
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &good, 1, sizeof(good), 0xFF000000u));
    const auto snap = bk.readback(view, W, H);
    REQUIRE(snap == geom_ref(W, H, 0xFF0000FFu, {}, {}));

    CaliperGeomDrawV1_2 d;
    // (a) uv_alloc released after import.
    {
        id<MTLBuffer> sbuf = device_buffer(uv, sizeof(uv));
        REQUIRE(sbuf != nil);
        CaliperAllocId sa = bk.bridge->import_allocation(
            (__bridge void*)sbuf, sizeof(uv), CALIPER_ALLOC_HANDLE_MTLBUFFER);
        REQUIRE(sa != 0);
        bk.bridge->release_allocation(sa);
        d = make_valid(); d.uv_alloc = sa;
        CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
            view, &cam, &d, 1, sizeof(d), 0xFF000000u));
        CHECK(bk.readback(view, W, H) == snap);
    }
    // (b) texture names a geometry view (the current target).
    d = make_valid(); d.texture = view;
    CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == snap);
    // (c) texture is a released texture id.
    {
        CaliperTensor td2 = u8_3d(rgba, 2, 2, 4);
        CaliperTextureId stex = bk.bridge->texture_from_tensor(&td2, 0);
        REQUIRE(stex != 0);
        bk.bridge->release_texture(stex);
        d = make_valid(); d.texture = stex;
        CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
            view, &cam, &d, 1, sizeof(d), 0xFF000000u));
        CHECK(bk.readback(view, W, H) == snap);
    }
    // (d) v1.2 submission with a short (192) draw_stride.
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == snap);

    // final cumulative: nothing in the battery touched the view.
    CHECK(bk.readback(view, W, H) == snap);

    bk.bridge->geom_release_view(view);
    bk.bridge->release_texture(texture);
    bk.bridge->release_allocation(ua);
    bk.bridge->release_allocation(pa);
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

// ===========================================================================
// caliper.geometry.v1_1 §9.2 byte-exact drawing mirrors (CUDA-gated). These are
// the Vulkan twins of the Metal §9.2 rows: same geometry, same reference math,
// same expected constants — the only difference is the alloc source (a live
// cuMemCreate'd VMM block imported through OPAQUE_WIN32, instead of an in-process
// shared MTLBuffer). Each row is compared against ONE CPU reference (geom_ref),
// never against the other backend's readback. Same guards as the v1 rows above.
// ===========================================================================

// Row A — one non-indexed full-viewport triangle, FLAT/UNLIT/OPAQUE, no depth.
// The (-1,-1),(3,-1),(-1,3) trick covers every pixel center unambiguously, so
// the whole frame is the flat color: identical to geom_ref's clear-only image.
TEST_CASE("gfx/geometry.v1_1: unindexed triangle, FLAT, OPAQUE is byte-exact") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 16, H = 16;
    const float pos[] = {
        -1.0f, -1.0f, 0.5f,
         3.0f, -1.0f, 0.5f,
        -1.0f,  3.0f, 0.5f,
    };
    VmmBlock blk(sizeof(pos));
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pos_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);

    CaliperGeomCamera cam = identity_cam();
    const uint32_t flat = 0xFF3377AAu;   // little-endian RGBA8
    CaliperGeomDraw d{};
    d.pos_alloc = pos_alloc;
    d.vertex_count = 3;
    d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.color_mode = CALIPER_GEOM_COLOR_FLAT;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.flat_rgba = flat;
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;

    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(std::string(bk.renderer->last_device_path()) == "primitives-imported");
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, flat, {}, {}));

    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view);
}

// Row B — indexed quad (4 verts, 6 u32 indices) with all buffers at nonzero,
// 4-byte-aligned offsets (each block's leading bytes are filled with garbage to
// prove offsets are honored). COLOR_COLORMAP with every vertex attr equal keeps
// the color flat and byte-exact: attr==1 -> LUT[255], attr==0 -> LUT[0]. The
// quad covers exactly pixel columns/rows 8..23 of a 32x32 view (NDC edges at
// +/-0.5, which land on integer pixel boundaries — never through a pixel
// center). Each imported region is its OWN VMM block (writing region B after A
// through one block would clobber A — the ledgered Task-6 lesson).
TEST_CASE("gfx/geometry.v1_1: indexed quad pulls u32 indices and LUT extremes at nonzero offsets") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    const float pos[12] = {
        -0.5f,  0.5f, 0.5f,   // v0 top-left     -> pixel (8,8)
         0.5f,  0.5f, 0.5f,   // v1 top-right    -> col 24, row 8
        -0.5f, -0.5f, 0.5f,   // v2 bottom-left  -> col 8, row 24
         0.5f, -0.5f, 0.5f,   // v3 bottom-right -> col 24, row 24
    };
    const uint32_t idx[6] = {0, 1, 2, 2, 1, 3};   // culling off; winding free
    const uint64_t pos_off = 256, idx_off = 128, attr_off = 64;

    std::vector<uint8_t> pos_bytes(pos_off + sizeof(pos), 0xAB);
    std::memcpy(pos_bytes.data() + pos_off, pos, sizeof(pos));
    std::vector<uint8_t> idx_bytes(idx_off + sizeof(idx), 0xCD);
    std::memcpy(idx_bytes.data() + idx_off, idx, sizeof(idx));

    VmmBlock pos_blk(pos_bytes.size());
    VmmBlock idx_blk(idx_bytes.size());
    REQUIRE_MESSAGE(pos_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(idx_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos_bytes.data(), pos_bytes.size()) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(idx_blk.va, idx_bytes.data(), idx_bytes.size()) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId idx_alloc = bk.bridge->import_allocation(
        idx_blk.os_handle, idx_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pos_alloc != 0); REQUIRE(idx_alloc != 0);

    std::vector<std::pair<int,int>> rect;   // cols 8..23, rows 8..23
    for (int y = 8; y < 24; ++y)
        for (int x = 8; x < 24; ++x) rect.emplace_back(x, y);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();
    const uint32_t* lut = colormap_lut(CALIPER_CMAP_VIRIDIS);

    auto make_draw = [&](CaliperAllocId attr_alloc) {
        CaliperGeomDraw d{};
        d.pos_alloc = pos_alloc;   d.pos_offset = pos_off;   d.vertex_count = 4;
        d.index_alloc = idx_alloc; d.index_offset = idx_off; d.index_count = 6;
        d.attr_alloc = attr_alloc; d.attr_offset = attr_off;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.colormap = CALIPER_CMAP_VIRIDIS;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // attr all 1.0 -> constant LUT[255]
    const float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<uint8_t> a1(attr_off + sizeof(ones), 0xEE);
    std::memcpy(a1.data() + attr_off, ones, sizeof(ones));
    VmmBlock a1_blk(a1.size());
    REQUIRE_MESSAGE(a1_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(a1_blk.va, a1.data(), a1.size()) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId a1_alloc = bk.bridge->import_allocation(
        a1_blk.os_handle, a1_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(a1_alloc != 0);
    CaliperGeomDraw d1 = make_draw(a1_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d1, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) ==
          geom_ref(W, H, 0xFF000000u, rect, std::vector<uint32_t>(rect.size(), lut[255])));

    // fresh clear, attr all 0.0 -> constant LUT[0]
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<uint8_t> a0(attr_off + sizeof(zeros), 0x11);
    std::memcpy(a0.data() + attr_off, zeros, sizeof(zeros));
    VmmBlock a0_blk(a0.size());
    REQUIRE_MESSAGE(a0_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(a0_blk.va, a0.data(), a0.size()) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId a0_alloc = bk.bridge->import_allocation(
        a0_blk.os_handle, a0_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(a0_alloc != 0);
    CaliperGeomDraw d0 = make_draw(a0_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d0, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) ==
          geom_ref(W, H, 0xFF000000u, rect, std::vector<uint32_t>(rect.size(), lut[0])));

    bk.bridge->release_allocation(a0_alloc);
    bk.bridge->release_allocation(a1_alloc);
    bk.bridge->release_allocation(idx_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view);
}

// Row C — the v1_1 primitives point path (ADDITIVE) must be byte-identical to
// the frozen v1 draw_points path given the same buffers. Over a black clear,
// additive one-pixel points equal the LUT color exactly, so BOTH readbacks equal
// the ONE CPU reference (geom_ref with the LUT colors) and therefore each other.
TEST_CASE("gfx/geometry.v1_1: additive points via draw_primitives match v1 draw_points byte-exact") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    const std::pair<int,int> px[3] = {{3, 5}, {18, 20}, {31, 31}};
    const float attrs[3] = {0.0f, 0.5f, 1.0f};
    float pos[9];
    for (int i = 0; i < 3; ++i) ndc_for_pixel(px[i].first, px[i].second, W, H, &pos[i * 3]);

    VmmBlock pos_blk(sizeof(pos));
    VmmBlock attr_blk(sizeof(attrs));
    REQUIRE_MESSAGE(pos_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(attr_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(attr_blk.va, attrs, sizeof(attrs)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId attr_alloc = bk.bridge->import_allocation(
        attr_blk.os_handle, attr_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pos_alloc != 0); REQUIRE(attr_alloc != 0);

    const uint32_t clear = 0xFF000000u;
    CaliperGeomCamera cam = identity_cam();

    // frozen v1 path
    CaliperTextureId view1 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view1 != 0);
    REQUIRE(bk.bridge->geom_draw_points(view1, &cam, pos_alloc, 0, 3,
                                        attr_alloc, 0, CALIPER_CMAP_VIRIDIS,
                                        0.0f, 1.0f, 1.0f, clear));
    const std::vector<uint8_t> got1 = bk.readback(view1, W, H);

    // v1_1 primitives path
    CaliperTextureId view2 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view2 != 0);
    CaliperGeomDraw d{};
    d.pos_alloc = pos_alloc; d.vertex_count = 3;
    d.attr_alloc = attr_alloc; d.attr_offset = 0;
    d.topology = CALIPER_GEOM_TOPO_POINTS;
    d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_ADDITIVE;
    d.colormap = CALIPER_CMAP_VIRIDIS;
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
    REQUIRE(bk.bridge->geom_draw_primitives(view2, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), clear));
    CHECK(std::string(bk.renderer->last_device_path()) == "primitives-imported");
    const std::vector<uint8_t> got2 = bk.readback(view2, W, H);

    // The ONE CPU reference: attrs {0, 0.5, 1} -> LUT[0], LUT[128], LUT[255]
    // (t=0.5 -> idx = 0.5*255 + 0.5 = 128 exactly), additive over black clear.
    const uint32_t* lut = colormap_lut(CALIPER_CMAP_VIRIDIS);
    const std::vector<uint8_t> ref = geom_ref(W, H, clear,
        {px[0], px[1], px[2]}, {lut[0], lut[128], lut[255]});
    CHECK(got1 == ref);
    CHECK(got2 == ref);
    CHECK(got1 == got2);

    bk.bridge->release_allocation(attr_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view1);
    bk.bridge->geom_release_view(view2);
}

// Row D — draw_count 0 is a pure clear of BOTH color and depth. Write depth 0.2
// with a near triangle, then a count-0 clear to teal, then a far (z=0.9)
// DEPTH_TEST triangle: it only draws if the count-0 clear reset depth to 1.0.
TEST_CASE("gfx/geometry.v1_1: draw_count 0 is a pure clear, and clears depth to 1.0") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 16, H = 16;
    const float near_pos[9] = {-1.f, -1.f, 0.2f,  3.f, -1.f, 0.2f, -1.f, 3.f, 0.2f};
    const float far_pos[9]  = {-1.f, -1.f, 0.9f,  3.f, -1.f, 0.9f, -1.f, 3.f, 0.9f};
    VmmBlock near_blk(sizeof(near_pos));
    VmmBlock far_blk(sizeof(far_pos));
    REQUIRE_MESSAGE(near_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(far_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(near_blk.va, near_pos, sizeof(near_pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(far_blk.va, far_pos, sizeof(far_pos)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId near_alloc = bk.bridge->import_allocation(
        near_blk.os_handle, near_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId far_alloc = bk.bridge->import_allocation(
        far_blk.os_handle, far_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(near_alloc != 0); REQUIRE(far_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    auto tri_draw = [&](CaliperAllocId a, uint32_t depth_flags, uint32_t color) {
        CaliperGeomDraw d{};
        d.pos_alloc = a; d.vertex_count = 3;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.depth_flags = depth_flags;
        d.flat_rgba = color;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // (1) near green, TEST|WRITE -> colors view, writes depth 0.2
    const uint32_t green = 0xFF00FF00u;
    CaliperGeomDraw d1 = tri_draw(near_alloc,
        CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE, green);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d1, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, green, {}, {}));

    // (2) draw_count 0 -> pure clear to teal (and depth cleared to 1.0)
    const uint32_t teal = 10u | (20u << 8) | (30u << 16) | (255u << 24);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, nullptr, 0,
                                            sizeof(CaliperGeomDraw), teal));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, teal, {}, {}));

    // (3) far blue, TEST only -> passes only because depth was cleared to 1.0
    const uint32_t blue = 0xFFFF0000u;
    CaliperGeomDraw d3 = tri_draw(far_alloc, CALIPER_GEOM_DEPTH_TEST, blue);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d3, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, blue, {}, {}));

    bk.bridge->release_allocation(far_alloc);
    bk.bridge->release_allocation(near_alloc);
    bk.bridge->geom_release_view(view);
}

// Row E — indexed triangles from imported buffers honor depth. Two full-viewport
// triangles share ONE imported pos block (near z=0.20 verts 0..2, far z=0.80
// verts 3..5) and ONE index block ({0,1,2,0,1,2}); the far draw reads from
// pos_offset=9*float / index_offset=3*u32. Near green is drawn first with
// TEST|WRITE; far red second with TEST|WRITE fails the LESS test everywhere and
// is rejected. If depth were ignored the later red would overwrite green, so the
// byte-exact full-frame-green readback is the depth-honor proof.
TEST_CASE("gfx/geometry.v1_1: indexed triangles from imported buffers honor depth") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    const float near_z = 0.20f;
    const float far_z  = 0.80f;
    const float pos[] = {
        -1.0f, -1.0f, near_z,  3.0f, -1.0f, near_z, -1.0f,  3.0f, near_z,
        -1.0f, -1.0f, far_z,   3.0f, -1.0f, far_z,  -1.0f,  3.0f, far_z,
    };
    const uint32_t idx[] = {0, 1, 2, 0, 1, 2};

    VmmBlock pos_blk(sizeof(pos));
    VmmBlock idx_blk(sizeof(idx));
    REQUIRE_MESSAGE(pos_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(idx_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(idx_blk.va, idx, sizeof(idx)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId idx_alloc = bk.bridge->import_allocation(
        idx_blk.os_handle, idx_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pos_alloc != 0); REQUIRE(idx_alloc != 0);

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
    // Near green covers every pixel center (the full-viewport triangle trick),
    // far red is depth-rejected everywhere -> byte-exact full-frame green.
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, 0xFF00FF00u, {}, {}));

    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->release_allocation(idx_alloc);
    bk.bridge->geom_release_view(view);
}

// Row F — overlapping depth-tested quads are draw-order-independent. Quad P
// (near z=0.25, pixels 8..23) green; quad Q (far z=0.75, pixels 16..27) red.
// Depth (LESS, TEST|WRITE) makes the near P win the overlap regardless of
// submission order, so P,Q and Q,P produce byte-identical frames — and both
// equal the ONE CPU reference (paint Q red, then P green so near overrides).
TEST_CASE("gfx/geometry.v1_1: overlapping depth-tested quads are draw-order-independent") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    // Quad P: near z=0.25, cols/rows 8..23  -> NDC x,y in [-0.5, 0.5].
    const float posP[18] = {
        -0.5f,  0.5f, 0.25f,   0.5f,  0.5f, 0.25f,  -0.5f, -0.5f, 0.25f,
        -0.5f, -0.5f, 0.25f,   0.5f,  0.5f, 0.25f,   0.5f, -0.5f, 0.25f,
    };
    // Quad Q: far z=0.75, cols/rows 16..27 -> NDC x in [0,0.75], y in [-0.75,0].
    const float posQ[18] = {
         0.0f,  0.0f,  0.75f,   0.75f, 0.0f,  0.75f,   0.0f, -0.75f, 0.75f,
         0.0f, -0.75f, 0.75f,   0.75f, 0.0f,  0.75f,   0.75f,-0.75f, 0.75f,
    };
    VmmBlock p_blk(sizeof(posP));
    VmmBlock q_blk(sizeof(posQ));
    REQUIRE_MESSAGE(p_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(q_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(p_blk.va, posP, sizeof(posP)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(q_blk.va, posQ, sizeof(posQ)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pal = bk.bridge->import_allocation(
        p_blk.os_handle, p_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId qal = bk.bridge->import_allocation(
        q_blk.os_handle, q_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pal != 0); REQUIRE(qal != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    const uint32_t green = 0xFF00FF00u, red = 0xFF0000FFu;
    auto quad_draw = [&](CaliperAllocId a, uint32_t color) {
        CaliperGeomDraw d{};
        d.pos_alloc = a; d.vertex_count = 6;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
        d.flat_rgba = color;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };
    CaliperGeomDraw P = quad_draw(pal, green);
    CaliperGeomDraw Q = quad_draw(qal, red);

    // Reference: paint Q red first, then P green (near P wins the overlap since
    // geom_ref lets a later entry override an earlier one at the same pixel).
    std::vector<std::pair<int,int>> px; std::vector<uint32_t> col;
    for (int y = 16; y < 28; ++y) for (int x = 16; x < 28; ++x) { px.emplace_back(x, y); col.push_back(red); }
    for (int y = 8;  y < 24; ++y) for (int x = 8;  x < 24; ++x) { px.emplace_back(x, y); col.push_back(green); }
    auto ref = geom_ref(W, H, 0xFF000000u, px, col);

    CaliperGeomDraw pq[2] = {P, Q};
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, pq, 2,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_pq = bk.readback(view, W, H);

    CaliperGeomDraw qp[2] = {Q, P};
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, qp, 2,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_qp = bk.readback(view, W, H);

    CHECK(got_pq == got_qp);
    CHECK(got_pq == ref);

    bk.bridge->release_allocation(pal);
    bk.bridge->release_allocation(qal);
    bk.bridge->geom_release_view(view);
}

// Row G — ALPHA blend equations are byte-exact (§4.2). Over an opaque-black
// clear, one COLOR_FLAT quad with flat_rgba=0x80FFFFFF (white, alpha 128),
// BLEND_ALPHA, no depth. color = 255*(128/255) + 0 = 128 exactly per channel;
// alpha = 128*1 + 255*(127/255) = 255 exactly -> rect pixel = 0xFF808080.
TEST_CASE("gfx/geometry.v1_1: ALPHA blend equations are byte-exact") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    // Quad over cols/rows 8..23 (NDC +/-0.5); z irrelevant, no depth attachment.
    const float pos[18] = {
        -0.5f,  0.5f, 0.5f,   0.5f,  0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,
        -0.5f, -0.5f, 0.5f,   0.5f,  0.5f, 0.5f,   0.5f, -0.5f, 0.5f,
    };
    VmmBlock blk(sizeof(pos));
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pal = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pal != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    const uint32_t clear = 0xFF000000u;   // opaque black
    CaliperGeomDraw d{};
    d.pos_alloc = pal; d.vertex_count = 6;
    d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.color_mode = CALIPER_GEOM_COLOR_FLAT;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_ALPHA;
    d.flat_rgba = 0x80FFFFFFu;            // white, alpha 128
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;

    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), clear));
    auto got = bk.readback(view, W, H);

    std::vector<std::pair<int,int>> rect;
    for (int y = 8; y < 24; ++y) for (int x = 8; x < 24; ++x) rect.emplace_back(x, y);
    auto ref = geom_ref(W, H, clear, rect,
                        std::vector<uint32_t>(rect.size(), 0xFF808080u));

    if (got != ref) {
        const size_t at = ((size_t)8 * W + 8) * 4;   // interior rect pixel (8,8)
        FAIL("ALPHA blend byte mismatch at (8,8): R=", (int)got[at],
             " G=", (int)got[at + 1], " B=", (int)got[at + 2],
             " A=", (int)got[at + 3], " expected R=128 G=128 B=128 A=255");
    }
    CHECK(got == ref);

    bk.bridge->release_allocation(pal);
    bk.bridge->geom_release_view(view);
}

// Row H — two axis-aligned 1-px LINES crossing. Horizontal along pixel row 10
// (x 4..27), vertical along pixel column 20 (y 3..28). OPAQUE white, no depth.
// The CPU reference colors every pixel of each segment (the crossing pixel once,
// OPAQUE so no double-blend). The four segment ENDPOINT pixels are masked
// (Metal/Vulkan diamond-exit endpoint rules differ) by overwriting them with the
// GPU's own bytes before the compare — the SAME mask set the Metal row uses.
TEST_CASE("gfx/geometry.v1_1: axis-aligned 1-px LINES cross, endpoints masked") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    float pos[12];
    ndc_for_pixel(4,  10, W, H, &pos[0]);   // horizontal, left end   (pixel 4,10)
    ndc_for_pixel(27, 10, W, H, &pos[3]);   // horizontal, right end  (pixel 27,10)
    ndc_for_pixel(20, 3,  W, H, &pos[6]);   // vertical, top end      (pixel 20,3)
    ndc_for_pixel(20, 28, W, H, &pos[9]);   // vertical, bottom end   (pixel 20,28)
    VmmBlock blk(sizeof(pos));
    REQUIRE_MESSAGE(blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pal = bk.bridge->import_allocation(
        blk.os_handle, blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pal != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    CaliperGeomDraw d{};
    d.pos_alloc = pal; d.vertex_count = 4;
    d.topology = CALIPER_GEOM_TOPO_LINES;
    d.color_mode = CALIPER_GEOM_COLOR_FLAT;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.flat_rgba = 0xFFFFFFFFu;
    d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;

    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got = bk.readback(view, W, H);

    std::vector<std::pair<int,int>> px;
    for (int x = 4; x <= 27; ++x) px.emplace_back(x, 10);   // horizontal row 10
    for (int y = 3; y <= 28; ++y) px.emplace_back(20, y);   // vertical col 20
    auto ref = geom_ref(W, H, 0xFF000000u, px,
                        std::vector<uint32_t>(px.size(), 0xFFFFFFFFu));

    // Mask the 4 segment endpoints: copy the GPU's own bytes into the reference.
    const std::pair<int,int> ep[4] = {{4,10},{27,10},{20,3},{20,28}};
    for (const auto& e : ep) {
        const size_t at = ((size_t)e.second * W + e.first) * 4;
        for (int c = 0; c < 4; ++c) ref[at + c] = got[at + c];
    }
    CHECK(got == ref);

    bk.bridge->release_allocation(pal);
    bk.bridge->geom_release_view(view);
}

// Row I — LAMBERT headlight shading within +/-2 LSB. Full-viewport triangle,
// COLOR_FLAT mid-gray 0xFFB4B4B4 (180), SHADE_LAMBERT, normals in an imported
// normal_alloc. Case 1: normals (0,0,1) -> lit=0.30+0.70*1.0=1.0 -> 180. Case 2:
// normals (sin60,0,cos60)=(0.8660254,0,0.5) -> lit=0.30+0.70*0.5=0.65 ->
// round(180*0.65)=117. Alpha stays 255 (Lambert scales rgb only). nmat identity.
// Per-channel |got-want| <= 2 (float lighting rounds to nearest) — the ONLY
// tolerance row in this group.
TEST_CASE("gfx/geometry.v1_1: LAMBERT headlight shading within +/-2 LSB") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 16, H = 16;
    const float pos[9] = { -1.0f, -1.0f, 0.5f,  3.0f, -1.0f, 0.5f, -1.0f, 3.0f, 0.5f };
    VmmBlock pos_blk(sizeof(pos));
    REQUIRE_MESSAGE(pos_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pal = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pal != 0);

    const float n1[9] = { 0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f };
    const float s = 0.8660254f;
    const float n2[9] = { s, 0.0f, 0.5f,  s, 0.0f, 0.5f,  s, 0.0f, 0.5f };
    VmmBlock n1_blk(sizeof(n1));
    VmmBlock n2_blk(sizeof(n2));
    REQUIRE_MESSAGE(n1_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(n2_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(n1_blk.va, n1, sizeof(n1)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(n2_blk.va, n2, sizeof(n2)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId n1_alloc = bk.bridge->import_allocation(
        n1_blk.os_handle, n1_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId n2_alloc = bk.bridge->import_allocation(
        n2_blk.os_handle, n2_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(n1_alloc != 0); REQUIRE(n2_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    // channel-wise |got-ref| <= tol compare (float lighting rounds to nearest)
    auto within = [](const std::vector<uint8_t>& g, const std::vector<uint8_t>& r, int tol) {
        if (g.size() != r.size()) return false;
        for (size_t i = 0; i < g.size(); ++i) {
            int diff = (int)g[i] - (int)r[i];
            if (diff < 0) diff = -diff;
            if (diff > tol) return false;
        }
        return true;
    };

    auto lambert_draw = [&](CaliperAllocId n_alloc) {
        CaliperGeomDraw d{};
        d.pos_alloc = pal; d.vertex_count = 3;
        d.normal_alloc = n_alloc;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.flat_rgba = 0xFFB4B4B4u;   // mid-gray 180, opaque
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // Case 1: lit=1.0 -> 0xB4 per channel.
    CaliperGeomDraw d1 = lambert_draw(n1_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d1, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got1 = bk.readback(view, W, H);
    auto ref1 = geom_ref(W, H, 0xFFB4B4B4u, {}, {});
    if (!within(got1, ref1, 2)) {
        const size_t at = ((size_t)(H / 2) * W + (W / 2)) * 4;
        FAIL("LAMBERT case1 out of tol: got R=", (int)got1[at], " G=", (int)got1[at + 1],
             " B=", (int)got1[at + 2], " A=", (int)got1[at + 3], " expected 180,180,180,255");
    }
    CHECK(within(got1, ref1, 2));

    // Case 2: lit=0.65 -> 117 per RGB channel, alpha 255 (0xFF757575).
    CaliperGeomDraw d2 = lambert_draw(n2_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d2, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got2 = bk.readback(view, W, H);
    auto ref2 = geom_ref(W, H, 0xFF757575u, {}, {});
    if (!within(got2, ref2, 2)) {
        const size_t at = ((size_t)(H / 2) * W + (W / 2)) * 4;
        FAIL("LAMBERT case2 out of tol: got R=", (int)got2[at], " G=", (int)got2[at + 1],
             " B=", (int)got2[at + 2], " A=", (int)got2[at + 3], " expected 117,117,117,255");
    }
    CHECK(within(got2, ref2, 2));

    bk.bridge->release_allocation(n2_alloc);
    bk.bridge->release_allocation(n1_alloc);
    bk.bridge->release_allocation(pal);
    bk.bridge->geom_release_view(view);
}

// Row J — wireframe-over-mesh: a coplanar LESS_OR_EQUAL line overlay wins.
// Draw 0: full-viewport triangle at z=0.5, FLAT dark blue, DEPTH_TEST|WRITE.
// Draw 1: the Row-H cross at the SAME z=0.5, FLAT white, DEPTH_TEST only (no
// WRITE). LESS_OR_EQUAL (§4.2) lets the coplanar lines paint over the mesh.
// Reference: blue everywhere, white along the two segments; the 4 line endpoints
// are masked as in Row H.
TEST_CASE("gfx/geometry.v1_1: wireframe-over-mesh coplanar LESS_OR_EQUAL overlay") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    const float tri[9] = { -1.0f, -1.0f, 0.5f,  3.0f, -1.0f, 0.5f, -1.0f, 3.0f, 0.5f };
    float line[12];
    ndc_for_pixel(4,  10, W, H, &line[0]); line[2]  = 0.5f;
    ndc_for_pixel(27, 10, W, H, &line[3]); line[5]  = 0.5f;
    ndc_for_pixel(20, 3,  W, H, &line[6]); line[8]  = 0.5f;
    ndc_for_pixel(20, 28, W, H, &line[9]); line[11] = 0.5f;

    VmmBlock tri_blk(sizeof(tri));
    VmmBlock line_blk(sizeof(line));
    REQUIRE_MESSAGE(tri_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(line_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(tri_blk.va, tri, sizeof(tri)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(line_blk.va, line, sizeof(line)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId tri_alloc = bk.bridge->import_allocation(
        tri_blk.os_handle, tri_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId line_alloc = bk.bridge->import_allocation(
        line_blk.os_handle, line_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(tri_alloc != 0); REQUIRE(line_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    const uint32_t blue = 0xFF800000u;   // dark blue (B=128), opaque
    CaliperGeomDraw dtri{};
    dtri.pos_alloc = tri_alloc; dtri.vertex_count = 3;
    dtri.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    dtri.color_mode = CALIPER_GEOM_COLOR_FLAT;
    dtri.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    dtri.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    dtri.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
    dtri.flat_rgba = blue;
    dtri.vmin = 0.0f; dtri.vmax = 1.0f; dtri.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) dtri.model[i * 4 + i] = 1.0f;

    CaliperGeomDraw dline{};
    dline.pos_alloc = line_alloc; dline.vertex_count = 4;
    dline.topology = CALIPER_GEOM_TOPO_LINES;
    dline.color_mode = CALIPER_GEOM_COLOR_FLAT;
    dline.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    dline.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    dline.depth_flags = CALIPER_GEOM_DEPTH_TEST;   // no WRITE; coplanar overlay
    dline.flat_rgba = 0xFFFFFFFFu;
    dline.vmin = 0.0f; dline.vmax = 1.0f; dline.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) dline.model[i * 4 + i] = 1.0f;

    CaliperGeomDraw draws[2] = {dtri, dline};
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, draws, 2,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got = bk.readback(view, W, H);

    std::vector<std::pair<int,int>> px;
    for (int x = 4; x <= 27; ++x) px.emplace_back(x, 10);
    for (int y = 3; y <= 28; ++y) px.emplace_back(20, y);
    auto ref = geom_ref(W, H, blue, px,
                        std::vector<uint32_t>(px.size(), 0xFFFFFFFFu));

    const std::pair<int,int> ep[4] = {{4,10},{27,10},{20,3},{20,28}};
    for (const auto& e : ep) {
        const size_t at = ((size_t)e.second * W + e.first) * 4;
        for (int c = 0; c < 4; ++c) ref[at + c] = got[at + c];
    }
    CHECK(got == ref);

    bk.bridge->release_allocation(line_alloc);
    bk.bridge->release_allocation(tri_alloc);
    bk.bridge->geom_release_view(view);
}

// Row K — out-of-range index values clamp to vertex_count-1, defined image.
// Textual twin of the Metal-section row (~line 1681): three 1-px POINTS at
// distinct pixel centers; index buffer {0,1,999} where 999 clamps to vertex 2
// in-shader. The clamped draw is byte-identical to the {0,1,2} reference draw.
TEST_CASE("gfx/geometry.v1_1: out-of-range index values clamp to vertex_count-1, defined image") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    const std::pair<int,int> px[3] = {{3, 5}, {18, 20}, {31, 31}};
    float pos[9];
    for (int i = 0; i < 3; ++i) ndc_for_pixel(px[i].first, px[i].second, W, H, &pos[i * 3]);
    VmmBlock pos_blk(sizeof(pos));
    REQUIRE_MESSAGE(pos_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pos_alloc != 0);

    const uint32_t idx_ref[3]   = {0, 1, 2};
    const uint32_t idx_clamp[3] = {0, 1, 999};   // 999 -> clamps to vertex 2
    VmmBlock ref_blk(sizeof(idx_ref));
    VmmBlock clamp_blk(sizeof(idx_clamp));
    REQUIRE_MESSAGE(ref_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(clamp_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(ref_blk.va, idx_ref, sizeof(idx_ref)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(clamp_blk.va, idx_clamp, sizeof(idx_clamp)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId ref_alloc = bk.bridge->import_allocation(
        ref_blk.os_handle, ref_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId clamp_alloc = bk.bridge->import_allocation(
        clamp_blk.os_handle, clamp_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(ref_alloc != 0); REQUIRE(clamp_alloc != 0);

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();
    const uint32_t flat = 0xFF00FF00u;   // opaque green

    auto point_draw = [&](CaliperAllocId idx_alloc) {
        CaliperGeomDraw d{};
        d.pos_alloc = pos_alloc; d.vertex_count = 3;
        d.index_alloc = idx_alloc; d.index_offset = 0; d.index_count = 3;
        d.topology = CALIPER_GEOM_TOPO_POINTS;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.flat_rgba = flat;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // reference: {0,1,2} -> all three pixels lit
    CaliperGeomDraw d_ref = point_draw(ref_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d_ref, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_ref = bk.readback(view, W, H);
    CHECK(got_ref == geom_ref(W, H, 0xFF000000u,
                              {px[0], px[1], px[2]}, {flat, flat, flat}));

    // clamp: {0,1,999} -> 999 clamps to vertex 2 -> byte-identical image
    CaliperGeomDraw d_clamp = point_draw(clamp_alloc);
    REQUIRE(bk.bridge->geom_draw_primitives(view, &cam, &d_clamp, 1,
                                            sizeof(CaliperGeomDraw), 0xFF000000u));
    auto got_clamp = bk.readback(view, W, H);
    CHECK(got_clamp == got_ref);

    bk.bridge->release_allocation(clamp_alloc);
    bk.bridge->release_allocation(ref_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(view);
}

// Row L — every §2.3 gate refuses the WHOLE frame and leaves pixels untouched.
// Full-battery textual twin of the Metal-section row (~line 1754), from LIVE
// imported allocations. Draw a known-good frame on a DEPTH view and a NO-depth
// view (distinct colors), snapshot both readbacks + last_device_path. Then for
// each §2.3 violation build a draws[] valid EXCEPT the one item, CHECK_FALSE the
// return; after the battery re-read BOTH views and CHECK byte-for-byte equality
// with the snapshots (and last_device_path unchanged). One valid draw = full-
// viewport triangle: any refusal that leaks would change the frame. SANCTIONED
// ADAPTATION (documented in the report): the Metal bounds cases (9/11/14/16) fit
// against tight 36/12-byte MTLBuffers; here every VmmBlock is granularity-padded
// (2 MiB on this box), so the OOB counts/offsets are DERIVED from blk.size to
// remain out of bounds against the PADDED size — same gate, real threshold.
TEST_CASE("gfx/geometry.v1_1: every §2.3 gate refuses the whole frame, pixels untouched") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    const uint32_t stride = sizeof(CaliperGeomDraw);

    // ---- source allocations (all valid; each battery case pokes ONE hole).
    // Each is its OWN VmmBlock (never layered through one block — the ledgered
    // Task-6 clobber lesson). Every block is granularity-padded to blk.size.
    const float tri[9] = { -1.f, -1.f, 0.5f,  3.f, -1.f, 0.5f, -1.f, 3.f, 0.5f };
    const uint32_t idx[3]  = {0, 1, 2};
    const float    nrm[9]  = { 0.f, 0.f, 1.f,  0.f, 0.f, 1.f,  0.f, 0.f, 1.f };
    const float    attr[3] = { 0.5f, 0.5f, 0.5f };
    VmmBlock pos_blk(sizeof(tri));
    VmmBlock idx_blk(sizeof(idx));
    VmmBlock nrm_blk(sizeof(nrm));
    VmmBlock attr_blk(sizeof(attr));
    REQUIRE_MESSAGE(pos_blk.ok,  "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(idx_blk.ok,  "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(nrm_blk.ok,  "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE_MESSAGE(attr_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va,  tri,  sizeof(tri))  == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(idx_blk.va,  idx,  sizeof(idx))  == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(nrm_blk.va,  nrm,  sizeof(nrm))  == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(attr_blk.va, attr, sizeof(attr)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId pos_alloc = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId idx_alloc = bk.bridge->import_allocation(
        idx_blk.os_handle, idx_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId nrm_alloc = bk.bridge->import_allocation(
        nrm_blk.os_handle, nrm_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    const CaliperAllocId attr_alloc = bk.bridge->import_allocation(
        attr_blk.os_handle, attr_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pos_alloc != 0); REQUIRE(idx_alloc != 0);
    REQUIRE(nrm_alloc != 0); REQUIRE(attr_alloc != 0);

    // Derived OOB thresholds against the PADDED block sizes (Metal used tight
    // 36/12-byte buffers; here range_ok checks against blk.size).
    const uint64_t pos_oob_vc  = pos_blk.size / 12u + 1u;   // positions overflow
    const uint64_t idx_oob_ic  = idx_blk.size / 4u  + 1u;   // indices overflow
    const uint64_t nrm_oob_off = nrm_blk.size - 12u;        // 3*12 spills past end
    const uint64_t attr_oob_off = attr_blk.size - 8u;       // 3*4 spills past end

    CaliperTextureId ndv = bk.bridge->geom_create_view_ex(W, H, 0);                       // no depth
    CaliperTextureId dv  = bk.bridge->geom_create_view_ex(W, H, CALIPER_GEOM_VIEW_DEPTH); // depth
    REQUIRE(ndv != 0); REQUIRE(dv != 0);
    CaliperGeomCamera cam = identity_cam();

    // A fully valid full-viewport-triangle draw for the NO-depth view.
    auto make_valid = [&]() {
        CaliperGeomDraw d{};
        d.pos_alloc = pos_alloc; d.vertex_count = 3;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.flat_rgba = 0xFF00AA00u;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    // ---- known-good frame on each view; snapshot both + last_device_path ----
    CaliperGeomDraw good_nd = make_valid();
    REQUIRE(bk.bridge->geom_draw_primitives(ndv, &cam, &good_nd, 1, stride, 0xFF000000u));
    CaliperGeomDraw good_d = make_valid();
    good_d.flat_rgba = 0xFFAA0000u;
    good_d.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
    REQUIRE(bk.bridge->geom_draw_primitives(dv, &cam, &good_d, 1, stride, 0xFF000000u));
    const auto snap_nd = bk.readback(ndv, W, H);
    const auto snap_d  = bk.readback(dv,  W, H);
    const std::string good_path = bk.renderer->last_device_path();
    REQUIRE(snap_nd == geom_ref(W, H, 0xFF00AA00u, {}, {}));
    REQUIRE(snap_d  == geom_ref(W, H, 0xFFAA0000u, {}, {}));

    CaliperGeomDraw d;
    // 1. topology out of range
    d = make_valid(); d.topology = 5;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 2. color_mode out of range
    d = make_valid(); d.color_mode = 3;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 3. shade_mode out of range
    d = make_valid(); d.shade_mode = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 4. blend_mode out of range
    d = make_valid(); d.blend_mode = 3;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 5. reserved must be zero
    d = make_valid(); d.reserved[0] = 1;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 6. absent position source
    d = make_valid(); d.pos_alloc = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 7. misaligned pos_offset
    d = make_valid(); d.pos_offset = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 8. vertex_count 0
    d = make_valid(); d.vertex_count = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 9. pos bounds overflow (Metal: 4 verts*12 > 36-byte alloc; here derived so
    //    vertex_count*12 exceeds the PADDED pos block).
    d = make_valid(); d.vertex_count = pos_oob_vc;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 10. indexed with misaligned index_offset
    d = make_valid(); d.index_alloc = idx_alloc; d.index_count = 3; d.index_offset = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 11. indexed with index_count*4 out of bounds (Metal: 4*4 > 12-byte alloc;
    //     here derived so index_count*4 exceeds the PADDED index block).
    d = make_valid(); d.index_alloc = idx_alloc; d.index_offset = 0; d.index_count = idx_oob_ic;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 12. too few consumed vertices for the topology
    d = make_valid(); d.topology = CALIPER_GEOM_TOPO_LINES; d.vertex_count = 1;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    d = make_valid(); d.topology = CALIPER_GEOM_TOPO_TRIANGLES; d.vertex_count = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 13. LAMBERT without normals
    d = make_valid(); d.shade_mode = CALIPER_GEOM_SHADE_LAMBERT; d.normal_alloc = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 14. LAMBERT normal bounds overflow (Metal: 24 + 3*12 > 36-byte alloc; here
    //     normal_offset derived so offset + 3*12 exceeds the PADDED normal block).
    d = make_valid(); d.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
    d.normal_alloc = nrm_alloc; d.normal_offset = nrm_oob_off;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 15. COLORMAP without attr; COLORMAP with unknown colormap id
    d = make_valid(); d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.colormap = CALIPER_CMAP_VIRIDIS; d.attr_alloc = 0;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    d = make_valid(); d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.attr_alloc = attr_alloc; d.attr_offset = 0; d.colormap = 999;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 16. attr bounds overflow (Metal: 4 + 3*4 > 12-byte alloc; here attr_offset
    //     derived so offset + 3*4 exceeds the PADDED attr block).
    d = make_valid(); d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    d.colormap = CALIPER_CMAP_VIRIDIS; d.attr_alloc = attr_alloc; d.attr_offset = attr_oob_off;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 17. depth flags on the NO-depth view (must refuse, not silently ignore)
    d = make_valid(); d.depth_flags = CALIPER_GEOM_DEPTH_TEST;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    // 20. null camera
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, nullptr, &d, 1, stride, 0xFF000000u));
    // 21. draw_stride below the host minimum
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, 100u, 0xFF000000u));
    // 22. frame atomicity: draws[0] valid, draws[1] invalid -> whole call refused
    {
        CaliperGeomDraw twod[2] = { make_valid(), make_valid() };
        twod[1].topology = 5;
        CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, twod, 2, stride, 0xFF000000u));
    }

    // whole battery so far touched nothing
    CHECK(bk.readback(ndv, W, H) == snap_nd);
    CHECK(bk.readback(dv,  W, H) == snap_d);
    CHECK(std::string(bk.renderer->last_device_path()) == good_path);

    // 18. dead alloc: reference a released allocation (mutates alloc table).
    {
        CaliperAllocId salloc = bk.bridge->import_allocation(
            pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
        REQUIRE(salloc != 0);
        bk.bridge->release_allocation(salloc);
        d = make_valid(); d.pos_alloc = salloc;
        CHECK_FALSE(bk.bridge->geom_draw_primitives(ndv, &cam, &d, 1, stride, 0xFF000000u));
    }
    // 19. dead view: released view id.
    {
        CaliperTextureId sview = bk.bridge->geom_create_view_ex(W, H, 0);
        REQUIRE(sview != 0);
        bk.bridge->geom_release_view(sview);
        d = make_valid();
        CHECK_FALSE(bk.bridge->geom_draw_primitives(sview, &cam, &d, 1, stride, 0xFF000000u));
    }

    // final: both live views still byte-identical to their good frames
    CHECK(bk.readback(ndv, W, H) == snap_nd);
    CHECK(bk.readback(dv,  W, H) == snap_d);
    CHECK(std::string(bk.renderer->last_device_path()) == good_path);

    bk.bridge->release_allocation(attr_alloc);
    bk.bridge->release_allocation(nrm_alloc);
    bk.bridge->release_allocation(idx_alloc);
    bk.bridge->release_allocation(pos_alloc);
    bk.bridge->geom_release_view(dv);
    bk.bridge->geom_release_view(ndv);
}

// Row M — draw_stride forward-compat. Textual twin of the Metal-section row
// (~line 1934): a struct that grew by 16 tail bytes must draw identically when
// the host is told the real stride — it reads min(stride, its own sizeof) per
// descriptor and steps `stride` between them. Part 1: one FLAT triangle via a
// normal array and via a GrownDraw array both match the CPU reference and each
// other. Part 2: two GrownDraw descriptors in one call — correct stride
// addressing must step 208 bytes, not 192, so draw[1] (a different-colored quad)
// lands only when stepping is right.
TEST_CASE("gfx/geometry.v1_1: draw_stride forward-compat, a grown struct draws identically") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES));
    const cudadrv::Api* cu = cudadrv::api();

    struct GrownDraw { CaliperGeomDraw d; uint8_t tail[16]; };
    static_assert(sizeof(GrownDraw) == sizeof(CaliperGeomDraw) + 16,
                  "GrownDraw must be exactly 16 bytes larger with no padding");

    const int W = 32, H = 32;
    const uint32_t nstride = sizeof(CaliperGeomDraw);
    const uint32_t gstride = sizeof(GrownDraw);

    const float tri[9] = { -1.f, -1.f, 0.5f,  3.f, -1.f, 0.5f, -1.f, 3.f, 0.5f };
    VmmBlock tri_blk(sizeof(tri));
    REQUIRE_MESSAGE(tri_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(tri_blk.va, tri, sizeof(tri)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId tri_alloc = bk.bridge->import_allocation(
        tri_blk.os_handle, tri_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(tri_alloc != 0);

    CaliperGeomCamera cam = identity_cam();
    const uint32_t flat = 0xFF3377AAu;

    CaliperGeomDraw base{};
    base.pos_alloc = tri_alloc; base.vertex_count = 3;
    base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    base.color_mode = CALIPER_GEOM_COLOR_FLAT;
    base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    base.flat_rgba = flat;
    base.vmin = 0.0f; base.vmax = 1.0f; base.size_px = 1.0f;
    for (int i = 0; i < 4; ++i) base.model[i * 4 + i] = 1.0f;

    const auto ref1 = geom_ref(W, H, flat, {}, {});

    // Part 1a: normal array, normal stride.
    CaliperTextureId v1 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v1 != 0);
    CaliperGeomDraw nd = base;
    REQUIRE(bk.bridge->geom_draw_primitives(v1, &cam, &nd, 1, nstride, 0xFF000000u));
    auto got_normal = bk.readback(v1, W, H);
    CHECK(got_normal == ref1);

    // Part 1b: grown array (tail zeroed), grown stride, pointer cast to base type.
    CaliperTextureId v2 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v2 != 0);
    GrownDraw g{}; g.d = base;
    REQUIRE(bk.bridge->geom_draw_primitives(
        v2, &cam, reinterpret_cast<const CaliperGeomDraw*>(&g), 1, gstride, 0xFF000000u));
    auto got_grown = bk.readback(v2, W, H);
    CHECK(got_grown == ref1);
    CHECK(got_grown == got_normal);

    // ---- Part 2: two descriptors; wrong stepping (192 vs 208) breaks draw[1] ----
    // draw[1] = a quad covering cols/rows 8..23 (NDC +/-0.5 -> pixel boundaries),
    // a different flat color, drawn OVER the full-viewport triangle.
    const float quad[18] = {
        -0.5f,  0.5f, 0.5f,   0.5f,  0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,   // tri 1
        -0.5f, -0.5f, 0.5f,   0.5f,  0.5f, 0.5f,   0.5f, -0.5f, 0.5f,   // tri 2
    };
    VmmBlock quad_blk(sizeof(quad));
    REQUIRE_MESSAGE(quad_blk.ok, "VMM alloc/map/export failed on a CUDA machine");
    REQUIRE(cu->cuMemcpyHtoD(quad_blk.va, quad, sizeof(quad)) == cudadrv::CUDA_SUCCESS);
    const CaliperAllocId quad_alloc = bk.bridge->import_allocation(
        quad_blk.os_handle, quad_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(quad_alloc != 0);

    const uint32_t colA = 0xFF00AA00u;   // triangle background
    const uint32_t colB = 0xFF0000AAu;   // quad overlay (distinct)
    CaliperGeomDraw dA = base; dA.flat_rgba = colA;
    CaliperGeomDraw dB = base;
    dB.pos_alloc = quad_alloc; dB.vertex_count = 6; dB.flat_rgba = colB;

    std::vector<std::pair<int,int>> rect;
    for (int y = 8; y < 24; ++y)
        for (int x = 8; x < 24; ++x) rect.emplace_back(x, y);
    const auto ref2 = geom_ref(W, H, colA, rect,
                               std::vector<uint32_t>(rect.size(), colB));

    // reference via normal structs
    CaliperTextureId v3 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v3 != 0);
    CaliperGeomDraw ndraws[2] = { dA, dB };
    REQUIRE(bk.bridge->geom_draw_primitives(v3, &cam, ndraws, 2, nstride, 0xFF000000u));
    auto got_two_normal = bk.readback(v3, W, H);
    CHECK(got_two_normal == ref2);

    // via grown structs, 208-byte stride
    CaliperTextureId v4 = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(v4 != 0);
    GrownDraw g2[2] = {}; g2[0].d = dA; g2[1].d = dB;
    REQUIRE(bk.bridge->geom_draw_primitives(
        v4, &cam, reinterpret_cast<const CaliperGeomDraw*>(g2), 2, gstride, 0xFF000000u));
    auto got_two_grown = bk.readback(v4, W, H);
    CHECK(got_two_grown == ref2);
    CHECK(got_two_grown == got_two_normal);

    bk.bridge->release_allocation(quad_alloc);
    bk.bridge->release_allocation(tri_alloc);
    bk.bridge->geom_release_view(v4);
    bk.bridge->geom_release_view(v3);
    bk.bridge->geom_release_view(v2);
    bk.bridge->geom_release_view(v1);
}

// ===========================================================================
// caliper.geometry.v1_1 portable gate-refusal rows (no CUDA required). These
// pin the §2.3 gates that need no live imported allocation, so they run on
// any Vulkan ICD — vmm_rows_ready()/vk_cuda_ready() are NOT used here. The
// byte-exact drawing mirrors (imports, index pulling, shading, blending) are
// separate CUDA-gated follow-up tasks.
// ===========================================================================

TEST_CASE("gfx/geometry.v1_1: create_view_ex refuses unknown flags and degenerate sizes (portable)") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    Backend bk = vk_backend();
    if ((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES) == 0) {
        MESSAGE("no geometry primitives path — skipping"); return;
    }

    // Unknown flag bits (only CALIPER_GEOM_VIEW_DEPTH is defined) refuse.
    CHECK(bk.bridge->geom_create_view_ex(32, 32, CALIPER_GEOM_VIEW_DEPTH | (1u << 1)) == 0);
    CHECK(bk.bridge->geom_create_view_ex(32, 32, ~0u) == 0);

    // Degenerate/out-of-range sizes refuse exactly as v1 create_view does.
    CHECK(bk.bridge->geom_create_view_ex(0, 32, 0) == 0);
    CHECK(bk.bridge->geom_create_view_ex(32, 0, 0) == 0);
    CHECK(bk.bridge->geom_create_view_ex(0, 0, 0) == 0);
    CHECK(bk.bridge->geom_create_view_ex(20000, 32, 0) == 0);

    // Sanity: a plain valid call still succeeds — the checks above aren't
    // accidentally refusing everything.
    CaliperTextureId view = bk.bridge->geom_create_view_ex(32, 32, 0);
    CHECK(view != 0);
    if (view != 0) bk.bridge->geom_release_view(view);
}

// Portable subset of the Metal "every §2.3 gate refuses the whole frame"
// battery (Row K, ~line 1754): only the gates that need no live imported
// allocation. draw_stride/reserved/enum-range/depth-flags checks all fire
// in TensorBridge::geom_draw_primitives() BEFORE pos_alloc is ever resolved
// (read tensor_bridge.cpp: topology/color/shade/blend range -> depth_flags
// unknown bits -> reserved zero -> depth_flags-vs-view -> vertex_count ->
// point size -> THEN pos_alloc lookup) — so every draw below can carry the
// same never-imported pos_alloc and still isolate the gate under test; only
// the last row (nonexistent alloc id) actually exercises the pos_alloc gate
// itself, with every earlier field left honestly valid.
TEST_CASE("gfx/geometry.v1_1: portable §2.3 gates refuse draw_primitives, pixels untouched (no CUDA)") {
    if (!vk_env().ok) { MESSAGE("no Vulkan ICD — skipping"); return; }
    Backend bk = vk_backend();
    if ((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_PRIMITIVES) == 0) {
        MESSAGE("no geometry primitives path — skipping"); return;
    }

    const int W = 16, H = 16;
    const uint32_t stride = sizeof(CaliperGeomDraw);
    // Never imported in this test — resolvable only once a real live alloc
    // exists, which portable (no-CUDA) rows cannot create.
    const CaliperAllocId kFakePosAlloc = 0x7FFFFFFFu;

    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);   // no depth
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    // Views are cleared to opaque black at creation (vulkan_renderer.cpp) —
    // that clear IS the "prior pixels" baseline; no valid draw is possible
    // here since draw_primitives requires a live pos_alloc and this row has
    // none (by design: portable-only, no CUDA import).
    const std::vector<uint8_t> snap = bk.readback(view, W, H);
    CHECK(snap == geom_ref(W, H, 0xFF000000u, {}, {}));
    const std::string before = bk.renderer->last_device_path();

    auto make_valid = [&]() {
        CaliperGeomDraw d{};
        d.pos_alloc = kFakePosAlloc; d.vertex_count = 3;
        d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        d.color_mode = CALIPER_GEOM_COLOR_FLAT;
        d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        d.flat_rgba = 0xFF00AA00u;
        d.vmin = 0.0f; d.vmax = 1.0f; d.size_px = 1.0f;
        for (int i = 0; i < 4; ++i) d.model[i * 4 + i] = 1.0f;
        return d;
    };

    CaliperGeomDraw d;
    // 3. null camera.
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, nullptr, &d, 1, stride, 0xFF000000u));
    // 4. draw_stride below the host minimum (a valid-looking draw is fine —
    // the stride gate must fire before anything touches the draw).
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, 100u, 0xFF000000u));
    // 6. out-of-range enums.
    d = make_valid(); d.topology = 5;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, stride, 0xFF000000u));
    d = make_valid(); d.color_mode = 3;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, stride, 0xFF000000u));
    d = make_valid(); d.shade_mode = 2;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, stride, 0xFF000000u));
    d = make_valid(); d.blend_mode = 3;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, stride, 0xFF000000u));
    // 5. reserved must be zero. NOTE: pos_alloc here is the same unresolvable
    // sentinel as every other row — but per the gate order above, the
    // reserved-zero check runs before pos_alloc is ever looked up, so this
    // row still isolates the reserved gate (see the comment above the case).
    d = make_valid(); d.reserved[0] = 1;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, stride, 0xFF000000u));
    // 7. depth_flags on a depthless view — must refuse, never silently ignore.
    d = make_valid(); d.depth_flags = CALIPER_GEOM_DEPTH_TEST;
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, stride, 0xFF000000u));
    // 8. nonexistent alloc id: every other field is honestly valid, so this
    // is the pos_alloc-liveness gate itself.
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives(view, &cam, &d, 1, stride, 0xFF000000u));

    // whole battery so far touched nothing
    CHECK(bk.readback(view, W, H) == snap);
    CHECK(std::string(bk.renderer->last_device_path()) == before);

    // 2. dead view: a never-created id (0 is always invalid by ABI contract)
    // and a released view id.
    {
        d = make_valid();
        CHECK_FALSE(bk.bridge->geom_draw_primitives(0, &cam, &d, 1, stride, 0xFF000000u));

        CaliperTextureId sview = bk.bridge->geom_create_view_ex(W, H, 0);
        REQUIRE(sview != 0);
        bk.bridge->geom_release_view(sview);
        d = make_valid();
        CHECK_FALSE(bk.bridge->geom_draw_primitives(sview, &cam, &d, 1, stride, 0xFF000000u));
    }

    // final: the live view is still exactly its post-creation clear.
    CHECK(bk.readback(view, W, H) == snap);
    CHECK(std::string(bk.renderer->last_device_path()) == before);

    bk.bridge->geom_release_view(view);
}

// Row [v1.2 donor] — UV pull at a poisoned nonzero offset, exact texel-center
// red, bilinear-center gray (within one RGBA8 LSB), and Lambert x texture
// (within two RGB LSB, alpha untouched). Runs byte-exact against the live
// Vulkan textured path; the Metal-section twin is transcribed from here.
TEST_CASE("gfx/geometry.v1_2: UV offset, bilinear texture, and Lambert are byte-exact") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 16, H = 16;
    const float pos[] = {-1.f,-1.f,0.5f, 3.f,-1.f,0.5f, -1.f,3.f,0.5f};
    const uint64_t uv_off = 64;
    const float red_uv[] = {0.25f,0.25f, 0.25f,0.25f, 0.25f,0.25f};
    std::vector<uint8_t> uv_bytes(uv_off + sizeof(red_uv), 0xA5);
    std::memcpy(uv_bytes.data() + uv_off, red_uv, sizeof(red_uv));
    const float nrm[] = {0.f,0.f,-1.f, 0.f,0.f,-1.f, 0.f,0.f,-1.f};

    VmmBlock pos_blk(sizeof(pos)), uv_blk(uv_bytes.size()), nrm_blk(sizeof(nrm));
    REQUIRE(pos_blk.ok); REQUIRE(uv_blk.ok); REQUIRE(nrm_blk.ok);
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(uv_blk.va, uv_bytes.data(), uv_bytes.size()) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(nrm_blk.va, nrm, sizeof(nrm)) == cudadrv::CUDA_SUCCESS);
    CaliperAllocId pa = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    CaliperAllocId ua = bk.bridge->import_allocation(
        uv_blk.os_handle, uv_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    CaliperAllocId na = bk.bridge->import_allocation(
        nrm_blk.os_handle, nrm_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pa != 0); REQUIRE(ua != 0); REQUIRE(na != 0);

    const uint8_t rgba[] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    CaliperTensor td = u8_3d(rgba, 2, 2, 4);
    CaliperTextureId texture = bk.bridge->texture_from_tensor(&td, 0);
    REQUIRE(texture != 0);
    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);

    CaliperGeomDrawV1_2 d{};
    d.base.pos_alloc = pa; d.base.vertex_count = 3;
    d.base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.base.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
    d.base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.base.size_px = 1.f;
    for (int i = 0; i < 4; ++i) d.base.model[i * 4 + i] = 1.f;
    d.uv_alloc = ua; d.uv_offset = uv_off; d.texture = texture;
    CaliperGeomCamera cam = identity_cam();

    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, 0xFF0000FFu, {}, {}));

    const float mid_uv[] = {0.5f,0.5f, 0.5f,0.5f, 0.5f,0.5f};
    REQUIRE(cu->cuMemcpyHtoD(uv_blk.va + uv_off, mid_uv, sizeof(mid_uv)) ==
            cudadrv::CUDA_SUCCESS);
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    {
        const auto got = bk.readback(view, W, H);
        CHECK((got == geom_ref(W, H, 0xFF7F7F7Fu, {}, {}) ||
               got == geom_ref(W, H, 0xFF808080u, {}, {})));
    }

    REQUIRE(cu->cuMemcpyHtoD(uv_blk.va + uv_off, red_uv, sizeof(red_uv)) ==
            cudadrv::CUDA_SUCCESS);
    d.base.normal_alloc = na;
    d.base.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    {
        const auto got = bk.readback(view, W, H);
        CHECK((got == geom_ref(W, H, 0xFF00004Cu, {}, {}) ||
               got == geom_ref(W, H, 0xFF00004Du, {}, {})));
    }

    bk.bridge->geom_release_view(view);
    bk.bridge->release_texture(texture);
    bk.bridge->release_allocation(na);
    bk.bridge->release_allocation(ua);
    bk.bridge->release_allocation(pa);
}

// Row [v1.2 clamp-to-edge] — a full-viewport quad whose per-vertex UV is
// (0.5 + x_ndc, 0.5 + y_ndc), so UV spans -0.5..1.5 across the 2x2 texture.
// FLAT (UNLIT) so nothing but the sample colors the pixel. Each read pixel sits
// deep in an out-of-range corner (|beyond [0,1]| = 0.4375 >> 0.125 = a quarter
// texel), so clamp-to-edge samples the nearest edge texel with no bilinear mix.
TEST_CASE("gfx/geometry.v1_2: out-of-range UVs clamp to edge texels byte-exact") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 16, H = 16;
    const float pos[] = {
        -1.f,-1.f,0.5f,  1.f,-1.f,0.5f,  -1.f,1.f,0.5f,
        -1.f, 1.f,0.5f,  1.f,-1.f,0.5f,   1.f,1.f,0.5f,
    };
    const uint64_t uv_off = 64;
    const float uv[] = {
        -0.5f,-0.5f,  1.5f,-0.5f,  -0.5f,1.5f,
        -0.5f, 1.5f,  1.5f,-0.5f,   1.5f,1.5f,
    };
    std::vector<uint8_t> uv_bytes(uv_off + sizeof(uv), 0xA5);
    std::memcpy(uv_bytes.data() + uv_off, uv, sizeof(uv));

    VmmBlock pos_blk(sizeof(pos)), uv_blk(uv_bytes.size());
    REQUIRE(pos_blk.ok); REQUIRE(uv_blk.ok);
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(uv_blk.va, uv_bytes.data(), uv_bytes.size()) == cudadrv::CUDA_SUCCESS);
    CaliperAllocId pa = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    CaliperAllocId ua = bk.bridge->import_allocation(
        uv_blk.os_handle, uv_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pa != 0); REQUIRE(ua != 0);

    const uint8_t rgba[] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    CaliperTensor td = u8_3d(rgba, 2, 2, 4);
    CaliperTextureId texture = bk.bridge->texture_from_tensor(&td, 0);
    REQUIRE(texture != 0);
    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(view != 0);

    CaliperGeomDrawV1_2 d{};
    d.base.pos_alloc = pa; d.base.vertex_count = 6;
    d.base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.base.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
    d.base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.base.size_px = 1.f;
    for (int i = 0; i < 4; ++i) d.base.model[i * 4 + i] = 1.f;
    d.uv_alloc = ua; d.uv_offset = uv_off; d.texture = texture;
    CaliperGeomCamera cam = identity_cam();

    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    const auto got = bk.readback(view, W, H);
    auto at = [&](int x, int y) {
        const size_t o = ((size_t)y * W + x) * 4;
        return (uint32_t)got[o] | ((uint32_t)got[o + 1] << 8) |
               ((uint32_t)got[o + 2] << 16) | ((uint32_t)got[o + 3] << 24);
    };
    CHECK(at(0, 15)  == 0xFF0000FFu);   // u<0, v<0 -> col0,row0 red
    CHECK(at(15, 15) == 0xFF00FF00u);   // u>1, v<0 -> col1,row0 green
    CHECK(at(0, 0)   == 0xFFFF0000u);   // u<0, v>1 -> col0,row1 blue
    CHECK(at(15, 0)  == 0xFFFFFFFFu);   // u>1, v>1 -> col1,row1 white

    bk.bridge->geom_release_view(view);
    bk.bridge->release_texture(texture);
    bk.bridge->release_allocation(ua);
    bk.bridge->release_allocation(pa);
}

// Row [v1.2 compat] — the same non-textured indexed COLORMAP+LAMBERT mesh drawn
// through the frozen v1.1 entry (stride 192) into view A and through the v1.2
// entry (zeroed tail, stride 216) into view B. Full-image equality guards against
// DIVERGENCE between the two entry points — stride handling, tail defaults,
// pipeline selection — not shader correctness: a shared-shader break corrupts both
// paths identically and still compares equal. Absolute shader correctness is
// guarded by the byte-exact rows above.
TEST_CASE("gfx/geometry.v1_2: v1.1 and v1.2 non-textured draws are byte-identical") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 32, H = 32;
    const float pos[9] = { -1.f,-1.f,0.5f,  3.f,-1.f,0.5f,  -1.f,3.f,0.5f };
    const uint32_t idx[3] = {0, 1, 2};
    const float nrm[9] = { 0.f,0.f,-1.f,  0.f,0.f,-1.f,  0.f,0.f,-1.f };
    const float attr[3] = { 0.5f, 0.5f, 0.5f };
    VmmBlock pos_blk(sizeof(pos)), idx_blk(sizeof(idx)),
             nrm_blk(sizeof(nrm)), attr_blk(sizeof(attr));
    REQUIRE(pos_blk.ok); REQUIRE(idx_blk.ok); REQUIRE(nrm_blk.ok); REQUIRE(attr_blk.ok);
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(idx_blk.va, idx, sizeof(idx)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(nrm_blk.va, nrm, sizeof(nrm)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(attr_blk.va, attr, sizeof(attr)) == cudadrv::CUDA_SUCCESS);
    CaliperAllocId pa = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    CaliperAllocId ia = bk.bridge->import_allocation(
        idx_blk.os_handle, idx_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    CaliperAllocId na = bk.bridge->import_allocation(
        nrm_blk.os_handle, nrm_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    CaliperAllocId aa = bk.bridge->import_allocation(
        attr_blk.os_handle, attr_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pa != 0); REQUIRE(ia != 0); REQUIRE(na != 0); REQUIRE(aa != 0);

    CaliperGeomDraw base{};
    base.pos_alloc = pa; base.vertex_count = 3;
    base.index_alloc = ia; base.index_count = 3;
    base.normal_alloc = na; base.attr_alloc = aa;
    base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    base.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
    base.colormap = CALIPER_CMAP_VIRIDIS;
    base.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
    base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    base.vmin = 0.f; base.vmax = 1.f; base.size_px = 1.f;
    for (int i = 0; i < 4; ++i) base.model[i * 4 + i] = 1.f;
    CaliperGeomCamera cam = identity_cam();

    CaliperTextureId va = bk.bridge->geom_create_view_ex(W, H, 0);
    CaliperTextureId vb = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(va != 0); REQUIRE(vb != 0);

    REQUIRE(bk.bridge->geom_draw_primitives(
        va, &cam, &base, 1, sizeof(CaliperGeomDraw), 0xFF000000u));
    CaliperGeomDrawV1_2 d{}; d.base = base;   // zeroed UV/texture tail
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        vb, &cam, &d, 1, sizeof(CaliperGeomDrawV1_2), 0xFF000000u));

    CHECK(bk.readback(va, W, H) == bk.readback(vb, W, H));
    // Non-triviality: view A must actually rasterize the mesh, else a blank-vs-blank
    // match would pass the equality above vacuously.
    CHECK(bk.readback(va, W, H) != geom_ref(W, H, 0xFF000000u, {}, {}));

    bk.bridge->geom_release_view(vb);
    bk.bridge->geom_release_view(va);
    bk.bridge->release_allocation(aa);
    bk.bridge->release_allocation(na);
    bk.bridge->release_allocation(ia);
    bk.bridge->release_allocation(pa);
}

// Row [v1.2 refusal purity] — a valid textured draw fills the view (pre-image),
// then four COLOR_TEXTURE gate breaches are attempted in order; each must refuse
// AND leave the view byte-identical to the pre-image, cumulatively (the Phase-B
// T3 pattern): (a) uv_alloc released after import, (b) texture names a geometry
// view (the target itself), (c) texture is a released texture id, (d) a v1.2
// submission with a short (192) draw_stride.
TEST_CASE("gfx/geometry.v1_2: textured gate refusals leave the view untouched (cumulative)") {
    if (!vmm_rows_ready()) return;
    Backend bk = vk_backend();
    REQUIRE((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_TEXTURED));
    const cudadrv::Api* cu = cudadrv::api();

    const int W = 16, H = 16;
    const float pos[9] = { -1.f,-1.f,0.5f,  3.f,-1.f,0.5f,  -1.f,3.f,0.5f };
    const float uv[6]  = { 0.25f,0.25f, 0.25f,0.25f, 0.25f,0.25f };   // -> red
    VmmBlock pos_blk(sizeof(pos)), uv_blk(sizeof(uv));
    REQUIRE(pos_blk.ok); REQUIRE(uv_blk.ok);
    REQUIRE(cu->cuMemcpyHtoD(pos_blk.va, pos, sizeof(pos)) == cudadrv::CUDA_SUCCESS);
    REQUIRE(cu->cuMemcpyHtoD(uv_blk.va, uv, sizeof(uv)) == cudadrv::CUDA_SUCCESS);
    CaliperAllocId pa = bk.bridge->import_allocation(
        pos_blk.os_handle, pos_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    CaliperAllocId ua = bk.bridge->import_allocation(
        uv_blk.os_handle, uv_blk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(pa != 0); REQUIRE(ua != 0);

    const uint8_t rgba[] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    CaliperTensor td = u8_3d(rgba, 2, 2, 4);
    CaliperTextureId texture = bk.bridge->texture_from_tensor(&td, 0);
    CaliperTextureId view = bk.bridge->geom_create_view_ex(W, H, 0);
    REQUIRE(texture != 0); REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    auto make_valid = [&]() {
        CaliperGeomDrawV1_2 dd{};
        dd.base.pos_alloc = pa; dd.base.vertex_count = 3;
        dd.base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
        dd.base.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
        dd.base.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        dd.base.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        dd.base.size_px = 1.f;
        for (int i = 0; i < 4; ++i) dd.base.model[i * 4 + i] = 1.f;
        dd.uv_alloc = ua; dd.uv_offset = 0; dd.texture = texture;
        return dd;
    };

    CaliperGeomDrawV1_2 good = make_valid();
    REQUIRE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &good, 1, sizeof(good), 0xFF000000u));
    const auto snap = bk.readback(view, W, H);
    REQUIRE(snap == geom_ref(W, H, 0xFF0000FFu, {}, {}));

    CaliperGeomDrawV1_2 d;
    // (a) uv_alloc released after import.
    {
        VmmBlock sblk(sizeof(uv));
        REQUIRE(sblk.ok);
        REQUIRE(cu->cuMemcpyHtoD(sblk.va, uv, sizeof(uv)) == cudadrv::CUDA_SUCCESS);
        CaliperAllocId sa = bk.bridge->import_allocation(
            sblk.os_handle, sblk.size, CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
        REQUIRE(sa != 0);
        bk.bridge->release_allocation(sa);
        d = make_valid(); d.uv_alloc = sa;
        CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
            view, &cam, &d, 1, sizeof(d), 0xFF000000u));
        CHECK(bk.readback(view, W, H) == snap);
    }
    // (b) texture names a geometry view (the current target).
    d = make_valid(); d.texture = view;
    CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(d), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == snap);
    // (c) texture is a released texture id.
    {
        CaliperTensor td2 = u8_3d(rgba, 2, 2, 4);
        CaliperTextureId stex = bk.bridge->texture_from_tensor(&td2, 0);
        REQUIRE(stex != 0);
        bk.bridge->release_texture(stex);
        d = make_valid(); d.texture = stex;
        CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
            view, &cam, &d, 1, sizeof(d), 0xFF000000u));
        CHECK(bk.readback(view, W, H) == snap);
    }
    // (d) v1.2 submission with a short (192) draw_stride.
    d = make_valid();
    CHECK_FALSE(bk.bridge->geom_draw_primitives_v1_2(
        view, &cam, &d, 1, sizeof(CaliperGeomDraw), 0xFF000000u));
    CHECK(bk.readback(view, W, H) == snap);

    // final cumulative: nothing in the battery touched the view.
    CHECK(bk.readback(view, W, H) == snap);

    bk.bridge->geom_release_view(view);
    bk.bridge->release_texture(texture);
    bk.bridge->release_allocation(ua);
    bk.bridge->release_allocation(pa);
}

#endif  // _WIN32
#endif  // CALIPER_HAVE_VULKAN
