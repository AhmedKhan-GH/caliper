// caliper_gfx_tests — windowed, pixel-exact proof of the tensor->texture path
// (docs/development/testing.md). NOT part of the headless ctest default; registered under
// the "gfx" label and requires a GUI session. If glfwInit() fails (headless
// CI), every case is reported skipped — counted, and the binary exits with the
// skip code so ctest shows "Skipped", not "Passed" (gpu_skip.h). With
// CALIPER_REQUIRE_GPU=1 those skips are failures.
//
// This runs the SAME §16 matrix on the platform's backend: Metal on Apple
// (when a Metal device is available) and Vulkan on Windows. Each compares the
// GPU readback byte-for-byte against the shared CPU reference (map_f32_to_rgba8
// / expand_u8_to_rgba8) — the single source of truth the staging path also uses.
//
// Metal additionally exercises the DEVICE paths that nothing had ever run: a
// METAL-device CaliperTensor is built from a raw MTLBuffer allocated *in the
// test* (the C-ABI contract is data = bridge-cast id<MTLBuffer>; no torch here),
// then f32+LUT must take the "compute" path and u8 HWC the "blit" path, each
// pixel-exact vs the identical CPU reference. Readback is test-only: a Metal
// blit texture->shared-buffer + waitUntilCompleted (never the render path).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#include <GLFW/glfw3.h>
#include "../glfw_background.h"
#include <imgui.h>

#include "tensor_bridge.h"
#include "renderer/host_renderer.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
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
#include "renderer/cuda_driver.h"
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
// GLFW starts without a menu bar and the process keeps the background
// activation policy, so no Dock icon appears while the suite runs
// (glfw_background.h).
struct GlfwGuard {
    bool ok;
    GlfwGuard() : ok(!caliper_test::force_no_gpu() && caliper_test::glfw_init_in_background()) {}
    ~GlfwGuard() { if (ok) glfwTerminate(); }
};
GlfwGuard& glfw_guard() { static GlfwGuard g; return g; }

// ---- CPU-tensor builders (shared by every backend) -------------------------
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

}  // namespace

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
        glfwDefaultWindowHints();                  // clear prior hints
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
        return r->debug_readback_rgba8(id, w, h);   // renderer-queue readback
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

// The Metal environment's renderer for the graphics layer's rows (gfx_layer.cpp), or nullptr
// without a Metal device.
caliper_host::HostRenderer* gfx_metal_renderer() {
    return metal_env().ok ? metal_env().renderer.get() : nullptr;
}

TEST_CASE("gfx/Metal: 4x4 f32 ramp mapped through viridis is pixel-exact (staged)") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend bk = metal_backend(); mat_f32_viridis(bk);
}
TEST_CASE("gfx/Metal: f32 mapped through magma and RdBu is pixel-exact (staged)") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend bk = metal_backend(); mat_f32_magma_rdbu(bk);
}
TEST_CASE("gfx/Metal: 2x3 u8 direct (C=1,3,4) expands pixel-exact (staged)") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend bk = metal_backend(); mat_u8_direct(bk);
}
TEST_CASE("gfx/Metal: update_texture changes the pixels on the GPU (staged)") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend bk = metal_backend(); mat_update(bk);
}
TEST_CASE("gfx/Metal: invalid tensors return id 0") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend bk = metal_backend(); mat_invalid(bk);
}
TEST_CASE("gfx/Metal: alloc_shared roundtrip — write buffer, update, readback") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend bk = metal_backend(); mat_alloc_shared(bk);
}

// ---- DEVICE PATHS (first-ever execution of C2's compute/blit) --------------

// f32 + LUT on a METAL-device tensor -> compute shader -> pixel-exact vs CPU
// reference. Sizes deliberately non-multiples of the 16x16 threadgroup to prove
// the dispatch edge-guard (5x3 -> 1x1 groups, 17x9 -> 2x1 groups).
TEST_CASE("gfx/Metal: device f32+LUT takes the compute path, pixel-exact") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
// passes are in flight together, ordered only by queue commit order.
// The final readback must equal the LAST write byte-for-byte. Fresh source
// buffers per generation keep CPU writes outside the contract (a NULL-stream
// caller owns producer quiescence); dropping each buffer's last strong ref
// mid-flight also exercises command-buffer resource retention (spec §3.2).
TEST_CASE("gfx/Metal: burst updates pipeline in order, final readback pixel-exact") {
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
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

// Keeps a test's device updates on the buffer path, the rung below direct.
struct ForceBufferPath {
    ForceBufferPath() { vk_env().renderer->debug_force_buffer_path(true); }
    ~ForceBufferPath() { vk_env().renderer->debug_force_buffer_path(false); }
};

// ---- CPU-staged matrix (portable: runs on any Vulkan ICD, no CUDA) ----------
TEST_CASE("gfx/Vulkan: 4x4 f32 ramp mapped through viridis is pixel-exact (staged)") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    Backend bk = vk_backend(); mat_f32_viridis(bk);
}
TEST_CASE("gfx/Vulkan: f32 mapped through magma and RdBu is pixel-exact (staged)") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    Backend bk = vk_backend(); mat_f32_magma_rdbu(bk);
}
TEST_CASE("gfx/Vulkan: 2x3 u8 direct (C=1,3,4) expands pixel-exact (staged)") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    Backend bk = vk_backend(); mat_u8_direct(bk);
}
TEST_CASE("gfx/Vulkan: update_texture changes the pixels on the GPU (staged)") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    Backend bk = vk_backend(); mat_update(bk);
}
TEST_CASE("gfx/Vulkan: invalid tensors return id 0") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    Backend bk = vk_backend(); mat_invalid(bk);
}

// ---- CUDA device paths (hardware-gated: needs a UUID-paired CUDA device) -----
TEST_CASE("gfx/Vulkan+CUDA: device f32+LUT takes the compute path, pixel-exact") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();
    ForceBufferPath buffer_path;
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
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();
    ForceBufferPath buffer_path;
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

namespace {
// A torch-shaped CUDA tensor over freshly uploaded host bytes.
struct CudaTensor {
    cudadrv::CUdeviceptr buf = 0;
    CaliperTensor t{};
    size_t bytes = 0;
    CudaTensor(const void* src, size_t n) : bytes(n) {
        const cudadrv::Api* cu = cudadrv::api();
        REQUIRE(cu->cuMemAlloc(&buf, n) == cudadrv::CUDA_SUCCESS);
        cu->cuMemcpyHtoD(buf, src, n);
        t.struct_size = sizeof(t);
        t.data = (void*)(uintptr_t)buf;
        t.device = CALIPER_DEV_CUDA;
    }
    ~CudaTensor() {
        vk_env().renderer->device_update_wait();
        cudadrv::api()->cuCtxSynchronize();
        cudadrv::api()->cuMemFree(buf);
    }
    void f32(int w, int h) {
        t.dtype = CALIPER_DT_F32; t.ndim = 2;
        t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    }
    void rgba8(int w, int h) {
        t.dtype = CALIPER_DT_U8; t.ndim = 3;
        t.shape[0] = h; t.shape[1] = w; t.shape[2] = 4;
        t.strides[0] = (int64_t)w * 4; t.strides[1] = 4; t.strides[2] = 1;
    }
    void write(const void* src) { cudadrv::api()->cuMemcpyHtoD(buf, src, bytes); }
};

bool on_direct_path(HostRenderer* r) {
    return std::string(r->last_device_path()).rfind("direct-", 0) == 0;
}

// Draw `id` through the real frame loop, so a frame is genuinely in flight.
void vk_draw_frame(HostRenderer* r, CaliperTextureId id) {
    r->new_frame();
    ImGui::GetBackgroundDrawList()->AddImage((ImTextureID)id, ImVec2(0, 0), ImVec2(64, 64));
    r->render(64, 64);
}
}  // namespace

// Direct path: the kernel's output against the CPU reference, on the inputs where a
// float pipeline can differ: rounding boundaries, NaN, infinities, denormals,
// signed zero, and degenerate or inverted ranges. Contraction would fail here.
TEST_CASE("gfx/Vulkan+CUDA: direct f32 is byte-exact on edge values and ranges") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();

    const int w = 257, h = 129, n = w * h;   // non-16-multiple edges, both ways
    std::vector<float> data(n);
    const float inf = std::numeric_limits<float>::infinity();
    const float specials[] = {std::numeric_limits<float>::quiet_NaN(), inf, -inf, 0.0f, -0.0f,
                              std::numeric_limits<float>::denorm_min(),
                              -std::numeric_limits<float>::denorm_min(),
                              std::numeric_limits<float>::min(),
                              std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::lowest()};
    uint32_t rng = 12345;
    for (int i = 0; i < n; ++i) {
        if (i < 10) { data[i] = specials[i]; continue; }
        rng = rng * 1664525u + 1013904223u;
        if (i % 3 == 0) {
            // Either side of each index's rounding boundary, (k+0.5)/255.
            const float b = (((float)(rng >> 24) + 0.5f) / 255.0f);
            data[i] = (i % 2) ? std::nextafter(b, 2.0f) : std::nextafter(b, -1.0f);
        } else if (i % 3 == 1) {
            data[i] = (float)(rng >> 8) / 16777216.0f;
        } else {
            std::memcpy(&data[i], &rng, 4);   // any bit pattern
        }
    }
    CudaTensor src(data.data(), data.size() * sizeof(float));
    src.f32(w, h);

    const struct { float vmin, vmax; } ranges[] = {
        {0.f, 1.f}, {-3.5f, 7.25f}, {2.f, 2.f}, {1.f, 0.f}, {0.f, 1e-30f}, {-inf, inf}, {0.f, inf}};
    for (const auto& r : ranges) {
        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
            &src.t, CALIPER_CMAP_VIRIDIS, r.vmin, r.vmax, 0);
        REQUIRE(id != 0);
        CHECK(std::string(bk.renderer->last_device_path()) == "direct-cmap");
        std::vector<uint8_t> ref((size_t)n * 4);
        map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                         r.vmin, r.vmax, ref.data());
        CHECK(bk.readback(id, w, h) == ref);
        bk.bridge->release_texture(id);
    }
}

TEST_CASE("gfx/Vulkan+CUDA: direct u8 RGBA is byte-exact; other channel counts still work") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();

    const int w = 33, h = 17;
    std::vector<uint8_t> rgba((size_t)w * h * 4);
    for (size_t i = 0; i < rgba.size(); ++i) rgba[i] = (uint8_t)(i * 31 + 7);
    CudaTensor src(rgba.data(), rgba.size());
    src.rgba8(w, h);
    CaliperTextureId id = bk.bridge->texture_from_tensor(&src.t, 0);
    REQUIRE(id != 0);
    CHECK(std::string(bk.renderer->last_device_path()) == "direct-blit");
    CHECK(bk.readback(id, w, h) == rgba);
    bk.bridge->release_texture(id);
}

// Frames really in flight while kernels write: every update is followed by a
// frame that samples the image, and the next update must wait for it. A
// deadlock between frame and stream hangs here; a lost ordering fails the bytes.
TEST_CASE("gfx/Vulkan+CUDA: direct updates interleaved with frames stay in order") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();

    const int w = 512, h = 300, n = w * h;
    std::vector<float> data(n);
    const auto fill = [&](int gen) {
        for (int i = 0; i < n; ++i) data[i] = (float)((i + gen * 97) % 251);
    };
    fill(0);
    CudaTensor src(data.data(), data.size() * sizeof(float));
    src.f32(w, h);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &src.t, CALIPER_CMAP_MAGMA, 0.f, 250.f, 0);
    REQUIRE(id != 0);
    for (int gen = 1; gen <= 120; ++gen) {
        fill(gen);
        src.write(data.data());
        REQUIRE(bk.bridge->update_texture(id, &src.t));
        REQUIRE(on_direct_path(bk.renderer));
        vk_draw_frame(bk.renderer, id);
        if (gen % 3 == 0) {   // and several updates inside one frame
            REQUIRE(bk.bridge->update_texture(id, &src.t));
            REQUIRE(bk.bridge->update_texture(id, &src.t));
        }
    }
    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_MAGMA), 0.f, 250.f, ref.data());
    CHECK(bk.readback(id, w, h) == ref);
    bk.bridge->release_texture(id);
}

// The direct path's second image: the id an applet holds never changes, and what a frame
// actually draws, read back from the draw list, is always the newest write. A
// frame left drawing the old image would show training one update behind.
TEST_CASE("gfx/Vulkan+CUDA: frames draw the newest image, under a stable id") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();

    const int w = 64, h = 48, n = w * h;
    std::vector<float> data(n);
    const auto fill = [&](int gen) {
        for (int i = 0; i < n; ++i) data[i] = (float)((i + gen * 53) % 251);
    };
    fill(0);
    CudaTensor src(data.data(), data.size() * sizeof(float));
    src.f32(w, h);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &src.t, CALIPER_CMAP_VIRIDIS, 0.f, 250.f, 0);
    REQUIRE(id != 0);
    vk_draw_frame(bk.renderer, id);

    std::set<uint64_t> drawn_sets;
    for (int gen = 1; gen <= 6; ++gen) {
        fill(gen);
        src.write(data.data());
        REQUIRE(bk.bridge->update_texture(id, &src.t));
        REQUIRE(on_direct_path(bk.renderer));
        vk_draw_frame(bk.renderer, id);

        ImDrawData* dd = ImGui::GetDrawData();
        REQUIRE(dd != nullptr);
        uint64_t drawn = 0;
        for (ImDrawList* list : dd->CmdLists)
            for (const ImDrawCmd& cmd : list->CmdBuffer)
                if (cmd.TexRef._TexData == nullptr && cmd.ElemCount > 0)
                    drawn = (uint64_t)cmd.TexRef._TexID;
        REQUIRE(drawn != 0);
        drawn_sets.insert(drawn);

        std::vector<uint8_t> ref((size_t)n * 4);
        map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                         0.f, 250.f, ref.data());
        CHECK(bk.readback(drawn, w, h) == ref);   // what the frame drew
        CHECK(bk.readback(id, w, h) == ref);      // what the applet's id names
    }
    CHECK(drawn_sets.size() == 2);   // the two images really did trade places
    bk.bridge->release_texture(id);
}

// One texture, every writer in turn: each must order after the last and leave
// the image readable, whichever layout and timeline value it finds.
TEST_CASE("gfx/Vulkan+CUDA: direct, buffer and CPU-staged updates mix on one texture") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();

    const int w = 64, h = 48, n = w * h;
    std::vector<float> data(n);
    const auto fill = [&](int gen) {
        for (int i = 0; i < n; ++i) data[i] = (float)((i * 3 + gen * 41) % 251);
    };
    fill(0);
    CudaTensor src(data.data(), data.size() * sizeof(float));
    src.f32(w, h);
    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &src.t, CALIPER_CMAP_VIRIDIS, 0.f, 250.f, 0);
    REQUIRE(id != 0);
    const auto expect_current = [&]() {
        std::vector<uint8_t> ref((size_t)n * 4);
        map_f32_to_rgba8(data.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                         0.f, 250.f, ref.data());
        CHECK(bk.readback(id, w, h) == ref);
    };
    expect_current();

    int gen = 0;
    for (int round = 0; round < 6; ++round) {
        // Direct, then buffer with no readback between them: only ordering saves it.
        fill(++gen); src.write(data.data());
        REQUIRE(bk.bridge->update_texture(id, &src.t));
        CHECK(on_direct_path(bk.renderer));
        {
            ForceBufferPath buffer_path;
            fill(++gen); src.write(data.data());
            bk.renderer->device_update_wait();
            REQUIRE(bk.bridge->update_texture(id, &src.t));
            CHECK(std::string(bk.renderer->last_device_path()) == "compute");
        }
        // Buffer, then direct: the kernel waits for the Vulkan pass.
        REQUIRE(bk.bridge->update_texture(id, &src.t));
        CHECK(on_direct_path(bk.renderer));
        vk_draw_frame(bk.renderer, id);
        expect_current();

        // CPU-staged, then direct.
        fill(++gen);
        CaliperTensor cpu = src.t;
        cpu.data = data.data();
        cpu.device = CALIPER_DEV_CPU;
        REQUIRE(bk.bridge->update_texture(id, &cpu));
        expect_current();
        fill(++gen); src.write(data.data());
        REQUIRE(bk.bridge->update_texture(id, &src.t));
        CHECK(on_direct_path(bk.renderer));
        expect_current();
    }
    bk.bridge->release_texture(id);
}

// The kernel runs in torch's context, where a bad read poisons training, so a
// tensor that does not match its texture must never reach it.
TEST_CASE("gfx/Vulkan+CUDA: direct refuses tensors it cannot read safely") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    HostRenderer* r = vk_env().renderer.get();
    const uint32_t* lut = colormap_lut(CALIPER_CMAP_VIRIDIS);
    const cudadrv::Api* cu = cudadrv::api();

    const int w = 32, h = 32, n = w * h;
    std::vector<float> data(n, 1.f);
    CudaTensor src(data.data(), data.size() * sizeof(float));
    src.f32(w, h);
    uint64_t tex = r->tex_create_rgba8(w, h);
    REQUIRE(tex != 0);

    // A sound tensor takes the direct path, so the refusals below mean something.
    REQUIRE(r->tex_update_from_device(tex, src.t, lut, 0.f, 1.f));
    CHECK(on_direct_path(r));

    const auto refused = [&](const CaliperTensor& t) {
        r->tex_update_from_device(tex, t, lut, 0.f, 1.f);
        CHECK_FALSE(on_direct_path(r));
        r->device_update_wait();
        // The context must still be healthy, or torch would be dead too.
        CHECK(cu->cuCtxSynchronize() == cudadrv::CUDA_SUCCESS);
    };
    CaliperTensor smaller = src.t;   // smaller than the texture
    smaller.shape[0] = h / 2; smaller.shape[1] = w / 2;
    smaller.strides[0] = w / 2;
    refused(smaller);
    CaliperTensor bad_stride = src.t;
    bad_stride.strides[1] = 0;
    refused(bad_stride);
    CaliperTensor negative = src.t;
    negative.strides[0] = -w;
    refused(negative);
    CaliperTensor misaligned = src.t;
    misaligned.data = (void*)(src.buf + 1);
    misaligned.shape[0] = h - 1;
    refused(misaligned);

    // A tensor running past its allocation is refused outright.
    CaliperTensor too_long = src.t;
    too_long.strides[0] = w * 4;
    CHECK_FALSE(r->tex_update_from_device(tex, too_long, lut, 0.f, 1.f));
    r->tex_release(tex);
}

// Release must wait out a kernel on CUDA's stream, which vkQueueWaitIdle cannot.
TEST_CASE("gfx/Vulkan+CUDA: releasing a texture straight after a direct update is safe") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();

    const int w = 1024, h = 1024, n = w * h;
    std::vector<float> data(n, 0.5f);
    CudaTensor src(data.data(), data.size() * sizeof(float));
    src.f32(w, h);
    for (int i = 0; i < 40; ++i) {
        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
            &src.t, CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 0);
        REQUIRE(id != 0);
        REQUIRE(bk.bridge->update_texture(id, &src.t));
        bk.bridge->release_texture(id);
    }
    CHECK(cudadrv::api()->cuCtxSynchronize() == cudadrv::CUDA_SUCCESS);
}

// The direct-image table must resolve on any driver new enough for interop, or
// the renderer silently stays on the buffer path. Struct sizes pin the hand
// transcriptions of cuda.h's ABI.
TEST_CASE("gfx/Vulkan+CUDA: direct-image driver table resolves") {
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    CHECK(cudadrv::direct_api() != nullptr);
    CHECK(sizeof(cudadrv::Array3dDescriptor) == 40);
    CHECK(sizeof(cudadrv::ExternalMemoryMipmappedArrayDesc) == 120);
    CHECK(sizeof(cudadrv::ResourceDesc) == 144);
    CHECK(sizeof(cudadrv::ExternalSemaphoreWaitParams) == 144);
}

// Update cost measurement (docs/development/vulkan-and-cuda.md): what one tensor update costs by size, on
// the path the renderer picks. "call" is the frame-thread time inside
// update_texture; "done" adds the wait for the GPU to finish it; "frame" is
// the mean and worst whole frame with the texture drawn and updated each frame.
// Prints numbers only: it proves nothing by itself, and a hidden window's
// present pacing is not the display's.
TEST_CASE("gfx/Vulkan+CUDA: timing — update cost by size") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();
    const cudadrv::Api* cu = cudadrv::api();
    using clk = std::chrono::steady_clock;
    const auto us = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    for (bool buffer : {true, false})
    for (int side : {256, 1024, 2048, 4096}) {
        vk_env().renderer->debug_force_buffer_path(buffer);
        const int n = side * side;
        const int iters = side >= 2048 ? 60 : 200;
        std::vector<float> data(n);
        for (int i = 0; i < n; ++i) data[i] = (float)(i % 251);
        cudadrv::CUdeviceptr buf = 0;
        REQUIRE(cu->cuMemAlloc(&buf, (size_t)n * sizeof(float)) == cudadrv::CUDA_SUCCESS);
        cu->cuMemcpyHtoD(buf, data.data(), (size_t)n * sizeof(float));

        CaliperTensor t{};
        t.struct_size = sizeof(t); t.data = (void*)(uintptr_t)buf; t.dtype = CALIPER_DT_F32;
        t.ndim = 2; t.shape[0] = side; t.shape[1] = side; t.strides[0] = side; t.strides[1] = 1;
        t.device = CALIPER_DEV_CUDA;

        CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
            &t, CALIPER_CMAP_VIRIDIS, 0.f, 250.f, 0);
        REQUIRE(id != 0);
        bk.renderer->device_update_wait();

        double call = 0, done = 0;
        for (int i = 0; i < iters; ++i) {
            auto t0 = clk::now();
            REQUIRE(bk.bridge->update_texture(id, &t));
            auto t1 = clk::now();
            bk.renderer->device_update_wait();
            auto t2 = clk::now();
            call += us(t0, t1);
            done += us(t0, t2);
        }

        double frame_sum = 0, frame_max = 0;
        for (int i = 0; i < iters; ++i) {
            auto t0 = clk::now();
            bk.renderer->new_frame();
            ImGui::GetBackgroundDrawList()->AddImage(
                (ImTextureID)id, ImVec2(0, 0), ImVec2(64, 64));
            REQUIRE(bk.bridge->update_texture(id, &t));
            bk.renderer->render(64, 64);
            const double f = us(t0, clk::now());
            frame_sum += f;
            frame_max = std::max(frame_max, f);
        }
        MESSAGE(side << "x" << side << " [" << std::string(bk.renderer->last_device_path())
                << "]: call " << call / iters << " us, done " << done / iters
                << " us, frame mean " << frame_sum / iters << " us, frame max "
                << frame_max << " us");

        bk.renderer->device_update_wait();
        bk.bridge->release_texture(id);
        cu->cuMemFree(buf);
        vk_env().renderer->debug_force_buffer_path(false);
    }
}

// Frame-thread hitch forensics (per-batch texture refresh report):
// the applet's per-generation refresh does release+create for all 9 textures
// (8x conv 48x48 + 1x embw 36x512). On the Vulkan+CUDA path a CREATE is the
// full interop setup — vkCreateImage + exportable memory + cuImportExternalMemory
// + timeline semaphore create/export/import — while an UPDATE is just a
// stream-ordered copy + compute pass. This case times both patterns at the
// applet's real sizes and prints µs/op so the fix (create-once + update) is
// justified by measurement, not vibes.
TEST_CASE("gfx/Vulkan+CUDA: timing — create+release cycle vs update, applet-shaped") {
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
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
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
    if (!vk_cuda_ready()) CALIPER_SKIP_NO_GPU("no CUDA interop / not single-GPU");
    Backend bk = vk_backend();
    ForceBufferPath buffer_path;
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
    if (!vk_env().ok) { CALIPER_NOTE_NO_GPU("no Vulkan ICD"); return false; }
    if (!vk_cuda_ready()) {
        CALIPER_NOTE_NO_GPU("no CUDA interop / not single-GPU"); return false;
    }
    if (!cudadrv::vmm_api()) {
        CALIPER_NOTE_NO_GPU("no CUDA VMM driver API"); return false;
    }
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

#endif  // _WIN32
#endif  // CALIPER_HAVE_VULKAN

// ---------------------------------------------------------------------------
// The suite's hidden windows give the test process no Dock icon or menu bar:
// after GLFW is up and the platform renderer's hidden window exists, the
// activation policy is still the background one (on macOS; elsewhere there is
// nothing to show).
// ---------------------------------------------------------------------------
TEST_CASE("gfx: the test process has no Dock icon or menu bar") {
#if defined(CALIPER_HAVE_METAL)
    if (!metal_env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
#elif defined(CALIPER_HAVE_VULKAN)
    if (!vk_env().ok) CALIPER_SKIP_NO_GPU("no Vulkan ICD");
#else
    CALIPER_SKIP_NOT_APPLICABLE("no renderer on this platform");
#endif
    CHECK(caliper_host::background_activation_policy_active());
}

CALIPER_GPU_TEST_MAIN()
