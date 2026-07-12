// caliper.export.v1 battery (Rung E1). Platform-neutral logic (sidecar golden,
// atomic-write refusal purity, sugar null-degradation) runs everywhere; the live
// pixel path (quad -> PNG -> decode -> byte-exact vs CPU reference; double-export
// byte-identity; filesystem refusal purity; sidecar; sequences; sugar widening)
// is Metal-gated and self-skips with no device. Compiled as ObjC++ on Apple so it
// can allocate the MTLBuffers the imported-geometry path consumes (mirrors
// gfx_main.cpp). MSVC-safe: no compound REQUIRE expressions.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "host_services.h"
#include "export_service.h"
#include "renderer/host_renderer.h"

#include <caliper/caliper.hpp>
#include <caliper/services/export_v1.h>
#include <caliper/services/geometry_v1_3.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/fixture_host.h>

#define STB_IMAGE_IMPLEMENTATION            // decode-only (TEST-ONLY, README §)
#define STBI_ONLY_PNG
#include <stb_image.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace caliper_host;

namespace {

std::string tmp_path(const char* leaf) {
    fs::path p = fs::temp_directory_path() / "caliper_export_tests";
    std::error_code ec;
    fs::create_directories(p, ec);
    return (p / leaf).string();
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Reference for a FLAT/OPAQUE quad filling the TOP-LEFT quadrant of a 16x16 view
// (cols 0..7, rows 0..7). Row 0 = top: an ASYMMETRIC image that a bottom-up
// readback would place in the WRONG corner — the row-order pin.
std::vector<uint8_t> topleft_quad_ref(uint32_t w, uint32_t h,
                                      uint32_t flat, uint32_t clear) {
    std::vector<uint8_t> ref((size_t)w * h * 4);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const uint32_t c = (x < w / 2 && y < h / 2) ? flat : clear;
            const size_t at = ((size_t)y * w + x) * 4;
            ref[at + 0] = (uint8_t)(c & 0xFF);
            ref[at + 1] = (uint8_t)((c >> 8) & 0xFF);
            ref[at + 2] = (uint8_t)((c >> 16) & 0xFF);
            ref[at + 3] = (uint8_t)((c >> 24) & 0xFF);
        }
    return ref;
}

}  // namespace

// ===========================================================================
// Platform-neutral: the provenance sidecar is a pure, deterministic function.
// ===========================================================================
TEST_CASE("export sidecar golden (fixed-injection, mirrors the C2 report pattern)") {
    const float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    ExportProvenance p;
    p.version = "0.6.0";
    p.git_commit = "abc123def456";
    p.backend = "metal";
    p.platform = "macos";
    p.timestamp_utc = "2026-07-12T00:00:00Z";
    p.width = 4;
    p.height = 4;
    p.clear_rgba = 4278190080u;   // 0xFF000000
    p.draw_count = 1;
    p.view16 = identity;
    p.proj16 = identity;
    p.colormaps = {3};
    p.state_json = R"({"step":42})";

    const std::string golden =
        "{\n"
        "  \"caliper\": {\n"
        "    \"version\": \"0.6.0\",\n"
        "    \"git_commit\": \"abc123def456\",\n"
        "    \"backend\": \"metal\",\n"
        "    \"platform\": \"macos\"\n"
        "  },\n"
        "  \"timestamp_utc\": \"2026-07-12T00:00:00Z\",\n"
        "  \"width\": 4,\n"
        "  \"height\": 4,\n"
        "  \"clear_rgba\": 4278190080,\n"
        "  \"camera\": {\n"
        "    \"view\": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],\n"
        "    \"proj\": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]\n"
        "  },\n"
        "  \"draw_count\": 1,\n"
        "  \"colormaps\": [3],\n"
        "  \"state\": {\"step\":42}\n"
        "}\n";
    CHECK(export_build_sidecar_json(p) == golden);
}

TEST_CASE("export sidecar: null state -> literal null; empty colormaps -> []") {
    ExportProvenance p;
    p.version = "0.6.0"; p.git_commit = "x"; p.backend = "gl"; p.platform = "linux";
    p.timestamp_utc = "t"; p.width = 1; p.height = 1; p.clear_rgba = 0; p.draw_count = 0;
    p.state_json = nullptr;   // -> null
    const std::string s = export_build_sidecar_json(p);
    CHECK(s.find("\"state\": null\n") != std::string::npos);
    CHECK(s.find("\"colormaps\": []") != std::string::npos);
    // Null camera pointers degrade to all-zero matrices, never a crash.
    CHECK(s.find("\"view\": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]")
          != std::string::npos);
}

TEST_CASE("export sidecar: sequence flavor emits frame_count") {
    ExportProvenance p;
    p.version = "0.6.0"; p.backend = "metal"; p.platform = "macos";
    p.timestamp_utc = "t"; p.width = 8; p.height = 8;
    p.is_sequence = true; p.frame_count = 300;
    const std::string s = export_build_sidecar_json(p);
    CHECK(s.find("\"frame_count\": 300,") != std::string::npos);
}

// ===========================================================================
// Platform-neutral: atomic writes are the filesystem half of refusal purity.
// ===========================================================================
TEST_CASE("export atomic write: unwritable path creates nothing, truncates nothing") {
    // A path whose parent directory does not exist -> encode/open fails.
    const std::string bad = "/caliper_nonexistent_dir_zzz/should_not_appear.png";
    std::vector<uint8_t> px(4 * 4 * 4, 0x7F);
    CHECK(export_write_png_atomic(bad, px.data(), 4, 4) == false);
    CHECK(fs::exists(bad) == false);

    // A pre-existing file must be left byte-for-byte intact when the write to a
    // DIFFERENT (unwritable) target fails — but also verify the temp-then-rename
    // never touches the target on a failed text write.
    const std::string txt = tmp_path("sentinel.txt");
    std::ofstream(txt, std::ios::binary) << "ORIGINAL";
    const auto before = read_file(txt);
    // A successful write DOES replace atomically (expected); prove the bytes land.
    CHECK(export_write_text_atomic(txt, "REPLACED") == true);
    const auto after = read_file(txt);
    CHECK(std::string(after.begin(), after.end()) == "REPLACED");
    CHECK(before.size() == 8);
}

// ===========================================================================
// Platform-neutral: the SDK sugar degrades inertly with no host.
// ===========================================================================
TEST_CASE("export sugar: null host is falsy and inert") {
    caliper::Export ex;                      // default -> no service
    CHECK(bool(ex) == false);
    CHECK(ex.caps() == 0u);
    CHECK(ex.has_view_png() == false);
    CaliperGeomDrawV1_3 d = caliper::geom_draw_v1_3_defaults();
    CaliperGeomCamera c{};
    CHECK(ex.view_png("/tmp/never.png", 4, 4, c, &d, 1, 0u) == false);
    CHECK(ex.begin_sequence("/tmp/never", 4, 4) == 0u);
}

// ===========================================================================
// Live Metal path.
// ===========================================================================
#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

namespace {

// Stands up a real Metal renderer and BINDS it to the process service registry,
// so services_get(export/bridge) drive the same g_bridge/g_renderer the service
// composes over. One per process (the registry is process-wide).
struct MetalHostEnv {
    bool ok = false;
    bool glfw_ok = false;
    GLFWwindow* window = nullptr;
    ImGuiContext* imgui_ctx = nullptr;
    std::unique_ptr<HostRenderer> renderer;
    id<MTLDevice> device = nil;

    MetalHostEnv() {
        glfw_ok = (glfwInit() == GLFW_TRUE);
        if (!glfw_ok) return;
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) return;
        renderer = make_metal_renderer();
        glfwDefaultWindowHints();
        renderer->window_hints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(64, 64, "caliper_export_tests", nullptr, nullptr);
        if (!window) return;
        imgui_ctx = ImGui::CreateContext();
        ImGui::SetCurrentContext(imgui_ctx);
        if (!renderer->init(window)) return;
        services_set_renderer(renderer.get());
        ok = true;
    }
    ~MetalHostEnv() {
        if (renderer) services_set_renderer(nullptr);   // drop bridge ref first
        if (ok) renderer->shutdown();
        if (imgui_ctx) { ImGui::SetCurrentContext(imgui_ctx); ImGui::DestroyContext(imgui_ctx); }
        if (window) glfwDestroyWindow(window);
        if (glfw_ok) glfwTerminate();
    }
};
MetalHostEnv& env() { static MetalHostEnv e; return e; }

const CaliperExportV1* export_svc() {
    return static_cast<const CaliperExportV1*>(services_get(CALIPER_EXPORT_V1));
}
const CaliperTensorBridgeV1_2* bridge_svc() {
    return static_cast<const CaliperTensorBridgeV1_2*>(
        services_get(CALIPER_TENSOR_BRIDGE_V1_2));
}

id<MTLBuffer> device_buffer(const void* src, size_t bytes) {
    id<MTLBuffer> b = [env().device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
    if (src) std::memcpy(b.contents, src, bytes);
    return b;
}

CaliperGeomCamera identity_cam() {
    CaliperGeomCamera c{};
    for (int i = 0; i < 4; ++i) { c.view[i*4+i] = 1.f; c.proj[i*4+i] = 1.f; }
    return c;
}

// A FLAT/UNLIT/OPAQUE indexed quad over the TOP-LEFT quadrant of a 16x16 view:
// NDC x in [-1,0], y in [0,+1] (=> cols 0..7, rows 0..7). Culling is off, so the
// {0,1,2,2,1,3} winding is free (mirrors the gfx indexed-quad case).
struct Quad {
    CaliperAllocId pos = 0, idx = 0;
    CaliperGeomDrawV1_3 draw{};
};
Quad make_topleft_quad(uint32_t flat) {
    const float pos[12] = {
        -1.0f, 1.0f, 0.5f,   // v0 top-left
         0.0f, 1.0f, 0.5f,   // v1 top-right
        -1.0f, 0.0f, 0.5f,   // v2 bottom-left
         0.0f, 0.0f, 0.5f,   // v3 bottom-right
    };
    const uint32_t idx[6] = {0, 1, 2, 2, 1, 3};
    id<MTLBuffer> pb = device_buffer(pos, sizeof(pos));
    id<MTLBuffer> ib = device_buffer(idx, sizeof(idx));
    Quad q;
    q.pos = bridge_svc()->import_allocation((__bridge void*)pb, sizeof(pos),
                                            CALIPER_ALLOC_HANDLE_MTLBUFFER);
    q.idx = bridge_svc()->import_allocation((__bridge void*)ib, sizeof(idx),
                                            CALIPER_ALLOC_HANDLE_MTLBUFFER);
    CaliperGeomDraw d{};
    d.pos_alloc = q.pos; d.vertex_count = 4;
    d.index_alloc = q.idx; d.index_count = 6;
    d.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    d.color_mode = CALIPER_GEOM_COLOR_FLAT;
    d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
    d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
    d.flat_rgba = flat;
    d.vmin = 0.f; d.vmax = 1.f; d.size_px = 1.f;
    for (int i = 0; i < 4; ++i) d.model[i*4+i] = 1.f;
    q.draw.base.base = d;
    return q;
}

// Decode a PNG file to tightly-packed RGBA8 (stb, top-down: row 0 = image top).
std::vector<uint8_t> decode_png(const std::string& path, int* w, int* h) {
    int n = 0;
    unsigned char* p = stbi_load(path.c_str(), w, h, &n, 4);
    std::vector<uint8_t> out;
    if (p) { out.assign(p, p + (size_t)(*w) * (*h) * 4); stbi_image_free(p); }
    return out;
}

}  // namespace

TEST_CASE("export live: caps track the geometry primitives cap when a renderer is bound") {
    if (!env().ok) { MESSAGE("no Metal device — skipping"); return; }
    REQUIRE(export_svc() != nullptr);
    CHECK((export_svc()->caps() & CALIPER_EXPORT_CAP_VIEW_PNG) != 0u);
}

TEST_CASE("export live: FLAT quad PNG decodes byte-exact to the CPU reference, top-down") {
    if (!env().ok) { MESSAGE("no Metal device — skipping"); return; }
    const uint32_t W = 16, H = 16, flat = 0xFF3377AAu, clear = 0xFF000000u;
    Quad q = make_topleft_quad(flat);
    REQUIRE(q.pos != 0);
    REQUIRE(q.idx != 0);

    const std::string path = tmp_path("quad.png");
    std::error_code ec; fs::remove(path, ec);
    CaliperGeomCamera cam = identity_cam();
    const uint32_t rc = export_svc()->view_png(path.c_str(), W, H, &cam, &q.draw,
                                               1, sizeof(CaliperGeomDrawV1_3),
                                               clear, nullptr);
    REQUIRE(rc == 1u);
    REQUIRE(fs::exists(path));

    int dw = 0, dh = 0;
    std::vector<uint8_t> got = decode_png(path, &dw, &dh);
    REQUIRE(dw == (int)W);
    REQUIRE(dh == (int)H);
    const std::vector<uint8_t> ref = topleft_quad_ref(W, H, flat, clear);
    CHECK(got == ref);

    // Explicit corner assertions pin BOTH axes (a flip would move the lit block).
    auto pix = [&](int x, int y) {
        const size_t at = ((size_t)y * W + x) * 4;
        return (uint32_t)got[at] | ((uint32_t)got[at+1] << 8) |
               ((uint32_t)got[at+2] << 16) | ((uint32_t)got[at+3] << 24);
    };
    CHECK(pix(2, 2) == flat);          // top-left  -> lit
    CHECK(pix(13, 2) == clear);        // top-right -> clear
    CHECK(pix(2, 13) == clear);        // bottom-left -> clear
    CHECK(pix(13, 13) == clear);       // bottom-right -> clear

    bridge_svc()->release_allocation(q.pos);
    bridge_svc()->release_allocation(q.idx);
}

TEST_CASE("export live: same draws -> byte-identical PNG across two calls (determinism)") {
    if (!env().ok) { MESSAGE("no Metal device — skipping"); return; }
    const uint32_t W = 16, H = 16;
    Quad q = make_topleft_quad(0xFF11EE55u);
    REQUIRE(q.pos != 0);
    const std::string a = tmp_path("det_a.png");
    const std::string b = tmp_path("det_b.png");
    CaliperGeomCamera cam = identity_cam();
    const uint32_t r1 = export_svc()->view_png(a.c_str(), W, H, &cam, &q.draw, 1,
                                               sizeof(CaliperGeomDrawV1_3), 0xFF000000u, nullptr);
    const uint32_t r2 = export_svc()->view_png(b.c_str(), W, H, &cam, &q.draw, 1,
                                               sizeof(CaliperGeomDrawV1_3), 0xFF000000u, nullptr);
    REQUIRE(r1 == 1u);
    REQUIRE(r2 == 1u);
    CHECK(read_file(a) == read_file(b));
    bridge_svc()->release_allocation(q.pos);
    bridge_svc()->release_allocation(q.idx);
}

TEST_CASE("export live: the sidecar lands next to the PNG with the submitted provenance") {
    if (!env().ok) { MESSAGE("no Metal device — skipping"); return; }
    const uint32_t W = 16, H = 16;
    Quad q = make_topleft_quad(0xFF3377AAu);
    REQUIRE(q.pos != 0);
    const std::string path = tmp_path("side.png");
    CaliperGeomCamera cam = identity_cam();
    const uint32_t rc = export_svc()->view_png(path.c_str(), W, H, &cam, &q.draw, 1,
                                               sizeof(CaliperGeomDrawV1_3), 0xFF102030u,
                                               R"({"seed":7})");
    REQUIRE(rc == 1u);
    const std::string json_path = path + ".json";
    REQUIRE(fs::exists(json_path));
    const auto raw = read_file(json_path);
    const std::string j(raw.begin(), raw.end());
    CHECK(j.find("\"width\": 16") != std::string::npos);
    CHECK(j.find("\"height\": 16") != std::string::npos);
    CHECK(j.find("\"clear_rgba\": 4279246896") != std::string::npos);  // 0xFF102030
    CHECK(j.find("\"backend\": \"metal\"") != std::string::npos);
    CHECK(j.find("\"platform\": \"macos\"") != std::string::npos);
    CHECK(j.find("\"state\": {\"seed\":7}") != std::string::npos);
    bridge_svc()->release_allocation(q.pos);
    bridge_svc()->release_allocation(q.idx);
}

TEST_CASE("export live: refusal purity — bad dims/path/draw write & truncate nothing") {
    if (!env().ok) { MESSAGE("no Metal device — skipping"); return; }
    const uint32_t W = 16, H = 16;
    Quad q = make_topleft_quad(0xFF3377AAu);
    REQUIRE(q.pos != 0);
    CaliperGeomCamera cam = identity_cam();

    // w == 0 -> refuse, no file.
    const std::string p0 = tmp_path("refuse_w0.png");
    std::error_code ec; fs::remove(p0, ec);
    CHECK(export_svc()->view_png(p0.c_str(), 0, H, &cam, &q.draw, 1,
          sizeof(CaliperGeomDrawV1_3), 0xFF000000u, nullptr) == 0u);
    CHECK(fs::exists(p0) == false);

    // w > CALIPER_EXPORT_MAX_DIM -> refuse, no file.
    const std::string p1 = tmp_path("refuse_huge.png");
    fs::remove(p1, ec);
    CHECK(export_svc()->view_png(p1.c_str(), 20000, H, &cam, &q.draw, 1,
          sizeof(CaliperGeomDrawV1_3), 0xFF000000u, nullptr) == 0u);
    CHECK(fs::exists(p1) == false);

    // Unwritable path (parent dir absent) -> refuse, no file.
    const std::string p2 = "/caliper_nonexistent_dir_zzz/x.png";
    CHECK(export_svc()->view_png(p2.c_str(), W, H, &cam, &q.draw, 1,
          sizeof(CaliperGeomDrawV1_3), 0xFF000000u, nullptr) == 0u);
    CHECK(fs::exists(p2) == false);

    // Gate-refused draw (pos_alloc == 0) over a PRE-EXISTING file -> rc 0 AND the
    // file is left byte-for-byte intact (nothing created, nothing truncated).
    const std::string p3 = tmp_path("sentinel.png");
    std::ofstream(p3, std::ios::binary) << "NOT_A_PNG_SENTINEL";
    const auto before = read_file(p3);
    CaliperGeomDrawV1_3 bad = caliper::geom_draw_v1_3_defaults();
    bad.base.base.pos_alloc = 0;             // no live geometry -> gate refuses
    bad.base.base.vertex_count = 3;
    bad.base.base.topology = CALIPER_GEOM_TOPO_TRIANGLES;
    CHECK(export_svc()->view_png(p3.c_str(), W, H, &cam, &bad, 1,
          sizeof(CaliperGeomDrawV1_3), 0xFF000000u, nullptr) == 0u);
    CHECK(read_file(p3) == before);

    // A bad stride (< 256) is a geometry gate refusal too -> rc 0, no file.
    const std::string p4 = tmp_path("refuse_stride.png");
    fs::remove(p4, ec);
    CHECK(export_svc()->view_png(p4.c_str(), W, H, &cam, &q.draw, 1,
          8u /* < min stride */, 0xFF000000u, nullptr) == 0u);
    CHECK(fs::exists(p4) == false);

    bridge_svc()->release_allocation(q.pos);
    bridge_svc()->release_allocation(q.idx);
}

TEST_CASE("export live: a sequence writes N numbered frames + one finalized sidecar") {
    if (!env().ok) { MESSAGE("no Metal device — skipping"); return; }
    const uint32_t W = 16, H = 16;
    Quad q = make_topleft_quad(0xFF3377AAu);
    REQUIRE(q.pos != 0);
    const std::string dir = tmp_path("seq");
    std::error_code ec; fs::remove_all(dir, ec);
    CaliperGeomCamera cam = identity_cam();

    const uint64_t h = export_svc()->begin_sequence(dir.c_str(), W, H, R"({"run":1})");
    REQUIRE(h != 0u);
    // One sequence at a time: a second begin is refused.
    CHECK(export_svc()->begin_sequence(dir.c_str(), W, H, nullptr) == 0u);
    // A frame with the wrong handle is refused.
    CHECK(export_svc()->frame(h + 999, &cam, &q.draw, 1,
          sizeof(CaliperGeomDrawV1_3), 0xFF000000u) == 0u);

    const int N = 3;
    for (int i = 0; i < N; ++i) {
        const uint32_t fr = export_svc()->frame(h, &cam, &q.draw, 1,
                            sizeof(CaliperGeomDrawV1_3), 0xFF000000u);
        CHECK(fr == 1u);
    }
    export_svc()->end_sequence(h);

    CHECK(fs::exists(fs::path(dir) / "frame_000000.png"));
    CHECK(fs::exists(fs::path(dir) / "frame_000001.png"));
    CHECK(fs::exists(fs::path(dir) / "frame_000002.png"));
    CHECK(fs::exists(fs::path(dir) / "frame_000003.png") == false);
    const std::string sc = (fs::path(dir) / "sequence.json").string();
    REQUIRE(fs::exists(sc));
    const auto raw = read_file(sc);
    const std::string j(raw.begin(), raw.end());
    CHECK(j.find("\"frame_count\": 3") != std::string::npos);
    CHECK(j.find("\"state\": {\"run\":1}") != std::string::npos);

    // After end_sequence the handle is dead: a frame on it is refused.
    CHECK(export_svc()->frame(h, &cam, &q.draw, 1,
          sizeof(CaliperGeomDrawV1_3), 0xFF000000u) == 0u);
    bridge_svc()->release_allocation(q.pos);
    bridge_svc()->release_allocation(q.idx);
}

TEST_CASE("export live: the SDK sugar widens a v1.1 draw and produces the same pixels") {
    if (!env().ok) { MESSAGE("no Metal device — skipping"); return; }
    const uint32_t W = 16, H = 16, flat = 0xFF3377AAu, clear = 0xFF000000u;
    Quad q = make_topleft_quad(flat);      // q.draw.base.base is the v1.1 record
    REQUIRE(q.pos != 0);

    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_EXPORT_V1, services_get(CALIPER_EXPORT_V1));
    caliper::Host host(fx.host());
    caliper::Export ex(host);
    REQUIRE(bool(ex) == true);
    CHECK(ex.has_view_png() == true);

    const std::string path = tmp_path("sugar.png");
    CaliperGeomCamera cam = identity_cam();
    const CaliperGeomDraw v11 = q.draw.base.base;   // hand the sugar a v1.1 record
    const bool okp = ex.view_png(path.c_str(), W, H, cam, &v11, 1, clear);
    REQUIRE(okp == true);

    int dw = 0, dh = 0;
    std::vector<uint8_t> got = decode_png(path, &dw, &dh);
    CHECK(got == topleft_quad_ref(W, H, flat, clear));
    bridge_svc()->release_allocation(q.pos);
    bridge_svc()->release_allocation(q.idx);
}
#endif  // CALIPER_HAVE_METAL
