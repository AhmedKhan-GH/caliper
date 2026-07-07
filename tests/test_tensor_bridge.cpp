// Unit tests for the TensorBridge core — validation, colormap index math, and
// the CPU staging bytes. Headless: a StubRenderer stands in for the GL/Metal
// backend so the acceptance-rule matrix and the LUT arithmetic are exercised
// without a window. The windowed GL pixel-exact proof lives in
// tests/gfx/gfx_main.cpp (caliper_gfx_tests, ctest label "gfx").
#include <doctest/doctest.h>

#include "tensor_bridge.h"
#include "renderer/host_renderer.h"
#include <caliper/services/tensor_bridge_v1_2.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace caliper_host;

namespace {

// Minimal HostRenderer that records the staged bytes and the device-path
// calls, so we can assert the bridge's logic without any GL/Metal context.
class StubRenderer : public HostRenderer {
public:
    explicit StubRenderer(const char* backend_name) : name_(backend_name) {}

    bool init(GLFWwindow*) override { return true; }
    void new_frame() override {}
    void render(int, int) override {}
    void shutdown() override {}
    const char* name() const override { return name_; }
    // Mirror the real backends' name->interop-device mapping so a stub named
    // "metal"/"vulkan" activates the device path (the bridge reads this, not
    // name(), since spec §3.4).
    CaliperDeviceKind interop_device() const override {
        if (std::strcmp(name_, "metal") == 0)  return CALIPER_DEV_METAL;
        if (std::strcmp(name_, "vulkan") == 0) return CALIPER_DEV_CUDA;
        return CALIPER_DEV_CPU;
    }
    void window_hints() override {}

    uint64_t tex_create_rgba8(int w, int h) override {
        if (w <= 0 || h <= 0) return 0;
        uint64_t id = next_id_++;
        created_[id] = {w, h};
        return id;
    }
    bool tex_upload_rgba8(uint64_t tex, const void* data, int w, int h) override {
        if (created_.find(tex) == created_.end() || data == nullptr) return false;
        last_upload_id = tex;
        last_upload.assign((const uint8_t*)data,
                           (const uint8_t*)data + (size_t)w * h * 4);
        ++upload_count;
        return true;
    }
    void tex_release(uint64_t tex) override { created_.erase(tex); ++release_count; }
    uint64_t tex_imtexture_id(uint64_t tex) override { return tex; }

    bool tex_update_from_device(uint64_t tex, const CaliperTensor&,
                                const uint32_t* lut, float, float) override {
        if (created_.find(tex) == created_.end()) return false;
        ++device_count;
        last_device_lut = lut;
        return device_return;
    }

    // knobs / observations
    bool device_return = true;
    int  upload_count = 0;
    int  device_count = 0;
    int  release_count = 0;
    uint64_t last_upload_id = 0;
    std::vector<uint8_t> last_upload;
    const uint32_t* last_device_lut = nullptr;

private:
    const char* name_;
    uint64_t next_id_ = 1;
    struct WH { int w, h; };
    std::unordered_map<uint64_t, WH> created_;
};

// A CPU f32 (h,w) tensor over caller-owned storage, contiguous row-major.
CaliperTensor f32_2d(const float* data, int64_t h, int64_t w) {
    CaliperTensor t{};
    t.struct_size = sizeof(CaliperTensor);
    t.data = (void*)data;
    t.dtype = CALIPER_DT_F32;
    t.ndim = 2;
    t.shape[0] = h; t.shape[1] = w;
    t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_CPU;
    return t;
}

CaliperTensor u8_3d(const uint8_t* data, int64_t h, int64_t w, int64_t c) {
    CaliperTensor t{};
    t.struct_size = sizeof(CaliperTensor);
    t.data = (void*)data;
    t.dtype = CALIPER_DT_U8;
    t.ndim = 3;
    t.shape[0] = h; t.shape[1] = w; t.shape[2] = c;
    t.strides[0] = w * c; t.strides[1] = c; t.strides[2] = 1;
    t.device = CALIPER_DEV_CPU;
    return t;
}

}  // namespace

TEST_CASE("colormap_lut: valid ids resolve, out-of-range is null") {
    CHECK(colormap_lut(CALIPER_CMAP_VIRIDIS) != nullptr);
    CHECK(colormap_lut(CALIPER_CMAP_MAGMA)   != nullptr);
    CHECK(colormap_lut(CALIPER_CMAP_RDBU)    != nullptr);
    CHECK(colormap_lut(-1) == nullptr);
    CHECK(colormap_lut(3)  == nullptr);
    // All 256 entries opaque (alpha == 255) in every LUT.
    for (int cm = 0; cm <= 2; ++cm)
        for (int i = 0; i < 256; ++i)
            CHECK(((colormap_lut(cm)[i] >> 24) & 0xff) == 0xffu);
}

TEST_CASE("map_f32_to_rgba8: index math incl. .5-rounding, clamp, NaN->0") {
    // Identity LUT: entry i has R == i, so the output R channel reveals the
    // computed index exactly, independent of colormap fidelity.
    uint32_t lut[256];
    for (int i = 0; i < 256; ++i)
        lut[i] = (uint32_t)i | ((uint32_t)i << 8) | ((uint32_t)i << 16) | (0xffu << 24);

    auto idx_of = [&](float v, float vmin, float vmax) {
        uint8_t out[4];
        map_f32_to_rgba8(&v, 1, 1, lut, vmin, vmax, out);
        return (int)out[0];
    };

    CHECK(idx_of(0.0f, 0.0f, 1.0f) == 0);      // t=0 -> 0
    CHECK(idx_of(1.0f, 0.0f, 1.0f) == 255);    // t=1 -> 255
    CHECK(idx_of(-5.0f, 0.0f, 1.0f) == 0);     // clamp low
    CHECK(idx_of(5.0f, 0.0f, 1.0f) == 255);    // clamp high
    // .5 rounding: t=0.5 -> 0.5*255+0.5 = 128.0 -> 128 (rounds up at the edge)
    CHECK(idx_of(0.5f, 0.0f, 1.0f) == 128);
    // t*255 == 126.5 -> +0.5 == 127.0 -> 127
    CHECK(idx_of(126.5f, 0.0f, 255.0f) == 127);
    // t*255 == 127.5 -> +0.5 == 128.0 -> 128
    CHECK(idx_of(127.5f, 0.0f, 255.0f) == 128);
    // degenerate range: t forced to 0
    CHECK(idx_of(42.0f, 7.0f, 7.0f) == 0);
    // NaN -> index 0
    CHECK(idx_of(std::nanf(""), 0.0f, 1.0f) == 0);
}

TEST_CASE("expand_u8_to_rgba8: channel replication / alpha rules") {
    // c==1: gray replicated, alpha 255
    uint8_t g[2] = {10, 200};
    uint8_t out1[2 * 4];
    expand_u8_to_rgba8(g, 2, 1, 1, out1);
    CHECK(out1[0] == 10);  CHECK(out1[1] == 10);  CHECK(out1[2] == 10);  CHECK(out1[3] == 255);
    CHECK(out1[4] == 200); CHECK(out1[5] == 200); CHECK(out1[6] == 200); CHECK(out1[7] == 255);

    // c==3: RGB with alpha 255
    uint8_t rgb[3] = {1, 2, 3};
    uint8_t out3[4];
    expand_u8_to_rgba8(rgb, 1, 1, 3, out3);
    CHECK(out3[0] == 1); CHECK(out3[1] == 2); CHECK(out3[2] == 3); CHECK(out3[3] == 255);

    // c==4: passthrough
    uint8_t rgba[4] = {9, 8, 7, 6};
    uint8_t out4[4];
    expand_u8_to_rgba8(rgba, 1, 1, 4, out4);
    CHECK(out4[0] == 9); CHECK(out4[1] == 8); CHECK(out4[2] == 7); CHECK(out4[3] == 6);
}

TEST_CASE("acceptance matrix: only the frozen v1 shapes/dtypes are accepted") {
    StubRenderer gl("gl");   // active device == CPU
    TensorBridge b(gl);

    float f[4] = {0, 1, 2, 3};
    uint8_t u[2 * 2 * 3] = {0};

    SUBCASE("2D f32 mapped: accepted") {
        CaliperTensor t = f32_2d(f, 2, 2);
        CHECK(b.texture_from_tensor_mapped(&t, CALIPER_CMAP_VIRIDIS, 0, 3, 0) != 0);
    }
    SUBCASE("3D u8 direct: accepted") {
        CaliperTensor t = u8_3d(u, 2, 2, 3);
        CHECK(b.texture_from_tensor(&t, 0) != 0);
    }
    SUBCASE("null tensor rejected") {
        CHECK(b.texture_from_tensor(nullptr, 0) == 0);
        CHECK(b.texture_from_tensor_mapped(nullptr, 0, 0, 1, 0) == 0);
    }
    SUBCASE("wrong ndim rejected") {
        CaliperTensor t = f32_2d(f, 2, 2); t.ndim = 3;    // f32 must be 2D
        CHECK(b.texture_from_tensor_mapped(&t, 0, 0, 1, 0) == 0);
        CaliperTensor d = u8_3d(u, 2, 2, 3); d.ndim = 2;  // u8 must be 3D
        CHECK(b.texture_from_tensor(&d, 0) == 0);
    }
    SUBCASE("wrong dtype rejected") {
        CaliperTensor t = f32_2d(f, 2, 2); t.dtype = CALIPER_DT_F16;
        CHECK(b.texture_from_tensor_mapped(&t, 0, 0, 1, 0) == 0);
        CaliperTensor d = u8_3d(u, 2, 2, 3); d.dtype = CALIPER_DT_F32;
        CHECK(b.texture_from_tensor(&d, 0) == 0);
    }
    SUBCASE("too many channels rejected") {
        uint8_t big[1 * 1 * 5] = {0};
        CaliperTensor d = u8_3d(big, 1, 1, 5);
        CHECK(b.texture_from_tensor(&d, 0) == 0);
    }
    SUBCASE("bad colormap id rejected") {
        CaliperTensor t = f32_2d(f, 2, 2);
        CHECK(b.texture_from_tensor_mapped(&t, 99, 0, 3, 0) == 0);
    }
    SUBCASE("non-contiguous strides rejected") {
        CaliperTensor t = f32_2d(f, 2, 2);
        t.strides[0] = 3;   // row stride != w -> gap
        CHECK(b.texture_from_tensor_mapped(&t, 0, 0, 3, 0) == 0);
    }
    SUBCASE("foreign device rejected on GL backend") {
        CaliperTensor t = f32_2d(f, 2, 2);
        t.device = CALIPER_DEV_METAL;   // not the active (CPU) device
        CHECK(b.texture_from_tensor_mapped(&t, 0, 0, 3, 0) == 0);
    }
    SUBCASE("non-positive shape rejected by the extent guard") {
        // A dim of 0 is contiguous by construction but has no extent; the
        // (shape[i]-1)*stride term would go negative without the guard, so the
        // bridge must reject rather than address memory it can't reason about.
        CaliperTensor t = f32_2d(f, 0, 2);   // shape {0,2}, strides {2,1}
        CHECK(b.texture_from_tensor_mapped(&t, CALIPER_CMAP_VIRIDIS, 0, 3, 0) == 0);
    }
}

TEST_CASE("CPU staging bytes match the reference conversion") {
    StubRenderer gl("gl");
    TensorBridge b(gl);

    float f[4] = {0.0f, 1.0f, 2.0f, 3.0f};   // 2x2 ramp
    CaliperTensor t = f32_2d(f, 2, 2);
    CaliperTextureId id = b.texture_from_tensor_mapped(&t, CALIPER_CMAP_VIRIDIS, 0, 3, 0);
    REQUIRE(id != 0);
    REQUIRE(gl.upload_count == 1);

    std::vector<uint8_t> ref(4 * 4);
    map_f32_to_rgba8(f, 2, 2, colormap_lut(CALIPER_CMAP_VIRIDIS), 0, 3, ref.data());
    CHECK(gl.last_upload == ref);

    // u8 (2x2x1) staged via expand reference
    uint8_t g[4] = {0, 85, 170, 255};
    CaliperTensor tu = u8_3d(g, 2, 2, 1);
    CaliperTextureId idu = b.texture_from_tensor(&tu, 0);
    REQUIRE(idu != 0);
    std::vector<uint8_t> refu(4 * 4);
    expand_u8_to_rgba8(g, 2, 2, 1, refu.data());
    CHECK(gl.last_upload == refu);
}

TEST_CASE("device tensor is validated then forwarded to the device path") {
    StubRenderer metal("metal");   // active device == METAL
    TensorBridge b(metal);

    float f[4] = {0, 1, 2, 3};
    CaliperTensor t = f32_2d(f, 2, 2);
    t.device = CALIPER_DEV_METAL;

    SUBCASE("valid device tensor -> device path, LUT forwarded, no CPU stage") {
        CaliperTextureId id = b.texture_from_tensor_mapped(&t, CALIPER_CMAP_MAGMA, 0, 3, 0);
        CHECK(id != 0);
        CHECK(metal.device_count == 1);
        CHECK(metal.upload_count == 0);
        CHECK(metal.last_device_lut == colormap_lut(CALIPER_CMAP_MAGMA));
    }
    SUBCASE("non-contiguous device tensor never reaches the device path") {
        t.strides[0] = 3;
        CaliperTextureId id = b.texture_from_tensor_mapped(&t, CALIPER_CMAP_MAGMA, 0, 3, 0);
        CHECK(id == 0);
        CHECK(metal.device_count == 0);
    }
    SUBCASE("device path failure fails the op (tensor is not CPU-reachable)") {
        metal.device_return = false;
        CaliperTextureId id = b.texture_from_tensor_mapped(&t, CALIPER_CMAP_MAGMA, 0, 3, 0);
        CHECK(id == 0);
        CHECK(metal.upload_count == 0);
    }
}

TEST_CASE("release_texture / update_texture lifecycle") {
    StubRenderer gl("gl");
    TensorBridge b(gl);

    float f[4] = {0, 1, 2, 3};
    CaliperTensor t = f32_2d(f, 2, 2);
    CaliperTextureId id = b.texture_from_tensor_mapped(&t, CALIPER_CMAP_VIRIDIS, 0, 3, 0);
    REQUIRE(id != 0);

    // update re-stages (colormap/vmin/vmax remembered from creation)
    float f2[4] = {3, 2, 1, 0};
    CaliperTensor t2 = f32_2d(f2, 2, 2);
    CHECK(b.update_texture(id, &t2));
    std::vector<uint8_t> ref(16);
    map_f32_to_rgba8(f2, 2, 2, colormap_lut(CALIPER_CMAP_VIRIDIS), 0, 3, ref.data());
    CHECK(gl.last_upload == ref);

    // update rejects a shape/dtype mismatch
    uint8_t u[12] = {0};
    CaliperTensor bad = u8_3d(u, 2, 2, 3);
    CHECK_FALSE(b.update_texture(id, &bad));
    CHECK_FALSE(b.update_texture(0, &t2));      // unknown id

    b.release_texture(id);
    CHECK(gl.release_count == 1);
    CHECK_FALSE(b.update_texture(id, &t2));      // gone after release
}

TEST_CASE("alloc_shared: host buffer + texture pair, update-on-demand") {
    StubRenderer gl("gl");
    TensorBridge b(gl);

    int64_t shape[3] = {2, 2, 4};
    CaliperTensor out{};
    CaliperTextureId tex = 0;
    REQUIRE(b.alloc_shared(CALIPER_DT_U8, 3, shape, &out, &tex));
    CHECK(tex != 0);
    CHECK(out.device == CALIPER_DEV_CPU);
    CHECK(out.data != nullptr);
    CHECK(out.dtype == CALIPER_DT_U8);
    CHECK(out.ndim == 3);

    // Writer fills the unified buffer, then pushes it.
    uint8_t* buf = (uint8_t*)out.data;
    for (int i = 0; i < 2 * 2 * 4; ++i) buf[i] = (uint8_t)(i * 3);
    CHECK(b.update_texture(tex, &out));
    std::vector<uint8_t> ref(16);
    expand_u8_to_rgba8(buf, 2, 2, 4, ref.data());
    CHECK(gl.last_upload == ref);

    b.free_shared(tex);
    CHECK(gl.release_count == 1);
}

TEST_CASE("bridge caps() surfaces the renderer's stream-handoff capability (D24)") {
    // Default renderer: no stream honor -> caps 0 (adapters must drain).
    StubRenderer plain("gl");
    TensorBridge b_plain(plain);
    CHECK(b_plain.caps() == 0u);

    // A backend that honors the stream channel -> bit 0.
    class StreamStub : public StubRenderer {
    public:
        using StubRenderer::StubRenderer;
        bool honors_stream_ordered_handoff() const override { return true; }
    };
    StreamStub honored("metal");
    TensorBridge b_honored(honored);
    CHECK(b_honored.caps() == CALIPER_BRIDGE_CAP_STREAM_ORDERED);
}

// --- Bridge v1.2: imported external allocations -----------------------------

namespace {
struct ImportStub : StubRenderer {
    using StubRenderer::StubRenderer;
    uint64_t next_id = 1;
    std::vector<uint64_t> released;
    struct Update { uint64_t tex, alloc, offset; };
    std::vector<Update> updates;
    bool supports_external_import() const override { return true; }
    uint64_t import_external_allocation(void*, uint64_t size, uint32_t type) override {
        if (type != CALIPER_ALLOC_HANDLE_OPAQUE_WIN32 &&
            type != CALIPER_ALLOC_HANDLE_OPAQUE_FD) return 0;
        if (size == 0) return 0;
        return next_id++;
    }
    void release_external_allocation(uint64_t id) override { released.push_back(id); }
    bool tex_update_from_imported(uint64_t tex, uint64_t alloc, uint64_t off,
                                  const CaliperTensor&, int32_t, float, float) override {
        updates.push_back({tex, alloc, off});
        return true;
    }
};
} // namespace

TEST_CASE("caps() adds IMPORT_ALLOC only when the renderer supports it") {
    StubRenderer plain("vulkan");
    TensorBridge b1(plain);
    CHECK((b1.caps() & CALIPER_BRIDGE_CAP_IMPORT_ALLOC) == 0u);
    ImportStub imp("vulkan");
    TensorBridge b2(imp);
    CHECK((b2.caps() & CALIPER_BRIDGE_CAP_IMPORT_ALLOC) != 0u);
}

TEST_CASE("import_allocation: id lifecycle, invalid args, double release") {
    ImportStub imp("vulkan");
    TensorBridge b(imp);
    uint64_t dummy = 42;
    CaliperAllocId a = b.import_allocation(&dummy, 4096,
                                           CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(a != 0);
    CHECK(b.import_allocation(nullptr, 4096,
                              CALIPER_ALLOC_HANDLE_OPAQUE_WIN32) == 0);   // null handle
    CHECK(b.import_allocation(&dummy, 0,
                              CALIPER_ALLOC_HANDLE_OPAQUE_WIN32) == 0);   // zero size
    CHECK(b.import_allocation(&dummy, 4096, 99u) == 0);                   // bad type
    b.release_allocation(a);
    CHECK(imp.released.size() == 1);
    b.release_allocation(a);              // double release: no-op, no crash
    CHECK(imp.released.size() == 1);
    b.release_allocation(0);              // invalid id: no-op
    CHECK(imp.released.size() == 1);
}

TEST_CASE("update_texture_from_alloc: acceptance gates + bounds + fallback contract") {
    ImportStub imp("vulkan");
    TensorBridge b(imp);
    uint64_t dummy = 42;
    // 4x4 f32 mapped texture created through the normal path first
    std::vector<float> px(16, 0.5f);
    CaliperTensor t = f32_2d(px.data(), 4, 4);            // existing test helper
    CaliperTextureId tex = b.texture_from_tensor_mapped(&t, 0, 0.f, 1.f, 0);
    REQUIRE(tex != 0);
    CaliperAllocId a = b.import_allocation(&dummy, 4 * 4 * sizeof(float),
                                           CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(a != 0);
    CaliperTensor d = t; d.data = nullptr;                  // desc: data ignored
    CHECK(b.update_texture_from_alloc(tex, a, 0, &d));
    CHECK(imp.updates.size() == 1);
    // offset + extent exceeding the imported size must be rejected host-side
    CHECK_FALSE(b.update_texture_from_alloc(tex, a, 8, &d));
    // unknown alloc / unknown texture / null desc reject without renderer call
    CHECK_FALSE(b.update_texture_from_alloc(tex, 999u, 0, &d));
    CHECK_FALSE(b.update_texture_from_alloc(0, a, 0, &d));
    CHECK_FALSE(b.update_texture_from_alloc(tex, a, 0, nullptr));
    // non-contiguous desc rejected by the same frozen gate
    CaliperTensor bad = d; bad.strides[0] = 5;
    CHECK_FALSE(b.update_texture_from_alloc(tex, a, 0, &bad));
    CHECK(imp.updates.size() == 1);
}
