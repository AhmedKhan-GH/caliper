// caliper_gfx_tests — windowed, pixel-exact proof of the tensor->texture path
// (PLATFORM.md §16). NOT part of the headless ctest default; registered under
// the "gfx" label and requires a GUI session. If glfwInit() fails (headless
// CI), every case is reported skipped so the label stays green rather than red.
//
// This is the GL run of the §16 matrix. C5 runs the SAME matrix on the Metal
// backend; both compare the GPU readback byte-for-byte against the shared CPU
// reference (map_f32_to_rgba8 / expand_u8_to_rgba8) — the single source of
// truth the staging path also uses.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include "tensor_bridge.h"
#include "renderer/host_renderer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace caliper_host;

namespace {

// One hidden GLFW+GL window + ImGui context + GLRenderer + TensorBridge for the
// whole binary. Cheap to hold; created lazily so a headless machine can skip.
struct GfxEnv {
    bool ok = false;
    GLFWwindow* window = nullptr;
    std::unique_ptr<HostRenderer> renderer;
    std::unique_ptr<TensorBridge> bridge;

    GfxEnv() {
        if (!glfwInit()) return;                 // headless -> ok stays false
        renderer = make_renderer("gl");
        renderer->window_hints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(64, 64, "caliper_gfx_tests", nullptr, nullptr);
        if (!window) { glfwTerminate(); return; }
        ImGui::CreateContext();
        if (!renderer->init(window)) {
            ImGui::DestroyContext();
            glfwDestroyWindow(window);
            glfwTerminate();
            return;
        }
        bridge = std::make_unique<TensorBridge>(*renderer);
        ok = true;
    }
    ~GfxEnv() {
        if (!ok) return;
        bridge.reset();
        renderer->shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

GfxEnv& env() { static GfxEnv e; return e; }

// Read an RGBA8 texture back off the GPU. The bridge id maps (via the renderer)
// to the GL name; the raw handle never escaped the renderer (§5.4).
std::vector<uint8_t> readback(CaliperTextureId id, int w, int h) {
    GLuint name = (GLuint)env().renderer->tex_imtexture_id(id);
    std::vector<uint8_t> px((size_t)w * h * 4, 0xAB);
    glBindTexture(GL_TEXTURE_2D, name);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return px;
}

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

#define REQUIRE_GFX() do { \
    if (!env().ok) { MESSAGE("no GUI/GL context — skipping gfx case"); return; } \
} while (0)

}  // namespace

TEST_CASE("gfx/GL: 4x4 f32 ramp mapped through viridis is pixel-exact") {
    REQUIRE_GFX();
    float ramp[16];
    for (int i = 0; i < 16; ++i) ramp[i] = (float)i;   // 0..15
    CaliperTensor t = f32_2d(ramp, 4, 4);

    CaliperTextureId id = env().bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, 15.0f, 0);
    REQUIRE(id != 0);

    std::vector<uint8_t> ref(16 * 4);
    map_f32_to_rgba8(ramp, 4, 4, colormap_lut(CALIPER_CMAP_VIRIDIS), 0.0f, 15.0f, ref.data());
    CHECK(readback(id, 4, 4) == ref);
    env().bridge->release_texture(id);
}

TEST_CASE("gfx/GL: f32 mapped through magma and RdBu is pixel-exact") {
    REQUIRE_GFX();
    float v[16];
    for (int i = 0; i < 16; ++i) v[i] = -1.0f + 2.0f * (i / 15.0f);  // -1..1
    CaliperTensor t = f32_2d(v, 4, 4);

    for (int cm : {CALIPER_CMAP_MAGMA, CALIPER_CMAP_RDBU}) {
        CaliperTextureId id = env().bridge->texture_from_tensor_mapped(&t, cm, -1.0f, 1.0f, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(16 * 4);
        map_f32_to_rgba8(v, 4, 4, colormap_lut(cm), -1.0f, 1.0f, ref.data());
        CHECK(readback(id, 4, 4) == ref);
        env().bridge->release_texture(id);
    }
}

TEST_CASE("gfx/GL: 2x3 u8 direct (C=1,3,4) expands pixel-exact") {
    REQUIRE_GFX();
    const int h = 2, w = 3;

    SUBCASE("C=1 gray replicate") {
        uint8_t g[6]; for (int i = 0; i < 6; ++i) g[i] = (uint8_t)(i * 40);
        CaliperTensor t = u8_3d(g, h, w, 1);
        CaliperTextureId id = env().bridge->texture_from_tensor(&t, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(w * h * 4);
        expand_u8_to_rgba8(g, w, h, 1, ref.data());
        CHECK(readback(id, w, h) == ref);
        env().bridge->release_texture(id);
    }
    SUBCASE("C=3 rgb, alpha forced 255") {
        uint8_t rgb[18]; for (int i = 0; i < 18; ++i) rgb[i] = (uint8_t)(i * 11);
        CaliperTensor t = u8_3d(rgb, h, w, 3);
        CaliperTextureId id = env().bridge->texture_from_tensor(&t, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(w * h * 4);
        expand_u8_to_rgba8(rgb, w, h, 3, ref.data());
        CHECK(readback(id, w, h) == ref);
        env().bridge->release_texture(id);
    }
    SUBCASE("C=4 passthrough") {
        uint8_t rgba[24]; for (int i = 0; i < 24; ++i) rgba[i] = (uint8_t)(255 - i * 7);
        CaliperTensor t = u8_3d(rgba, h, w, 4);
        CaliperTextureId id = env().bridge->texture_from_tensor(&t, 0);
        REQUIRE(id != 0);
        std::vector<uint8_t> ref(w * h * 4);
        expand_u8_to_rgba8(rgba, w, h, 4, ref.data());
        CHECK(readback(id, w, h) == ref);
        env().bridge->release_texture(id);
    }
}

TEST_CASE("gfx/GL: update_texture changes the pixels on the GPU") {
    REQUIRE_GFX();
    float a[16]; for (int i = 0; i < 16; ++i) a[i] = (float)i;
    CaliperTensor ta = f32_2d(a, 4, 4);
    CaliperTextureId id = env().bridge->texture_from_tensor_mapped(
        &ta, CALIPER_CMAP_VIRIDIS, 0.0f, 15.0f, 0);
    REQUIRE(id != 0);

    float b[16]; for (int i = 0; i < 16; ++i) b[i] = (float)(15 - i);   // reversed
    CaliperTensor tb = f32_2d(b, 4, 4);
    REQUIRE(env().bridge->update_texture(id, &tb));

    std::vector<uint8_t> ref(16 * 4);
    map_f32_to_rgba8(b, 4, 4, colormap_lut(CALIPER_CMAP_VIRIDIS), 0.0f, 15.0f, ref.data());
    CHECK(readback(id, 4, 4) == ref);
    env().bridge->release_texture(id);
}

TEST_CASE("gfx/GL: invalid tensors return id 0") {
    REQUIRE_GFX();
    float f[16] = {0};

    CaliperTensor wrong_ndim = f32_2d(f, 4, 4); wrong_ndim.ndim = 3;
    CHECK(env().bridge->texture_from_tensor_mapped(&wrong_ndim, 0, 0, 1, 0) == 0);

    CaliperTensor noncontig = f32_2d(f, 4, 4); noncontig.strides[0] = 5;   // gap
    CHECK(env().bridge->texture_from_tensor_mapped(&noncontig, 0, 0, 1, 0) == 0);

    CaliperTensor f16 = f32_2d(f, 4, 4); f16.dtype = CALIPER_DT_F16;
    CHECK(env().bridge->texture_from_tensor_mapped(&f16, 0, 0, 1, 0) == 0);

    CHECK(env().bridge->texture_from_tensor(nullptr, 0) == 0);
}

TEST_CASE("gfx/GL: alloc_shared roundtrip — write buffer, update, readback") {
    REQUIRE_GFX();
    int64_t shape[3] = {2, 3, 4};   // (H,W,4) u8 unified buffer
    CaliperTensor out{};
    CaliperTextureId tex = 0;
    REQUIRE(env().bridge->alloc_shared(CALIPER_DT_U8, 3, shape, &out, &tex));
    REQUIRE(tex != 0);
    REQUIRE(out.data != nullptr);

    uint8_t* buf = (uint8_t*)out.data;
    for (int i = 0; i < 2 * 3 * 4; ++i) buf[i] = (uint8_t)(i * 5 + 1);
    REQUIRE(env().bridge->update_texture(tex, &out));

    std::vector<uint8_t> ref(2 * 3 * 4);
    expand_u8_to_rgba8(buf, 3, 2, 4, ref.data());
    CHECK(readback(tex, 3, 2) == ref);

    env().bridge->free_shared(tex);
}

TEST_CASE("gfx/GL: last_device_path is the frozen CPU-staged fallback") {
    REQUIRE_GFX();
    // GL never reads device memory; the interface default reports cpu-staged.
    CHECK(std::string(env().renderer->last_device_path()) == "cpu-staged");
}
