// The host's PNG writers (export_service.h: export_write_png_atomic for the
// shot tools and the spec runtime, export_encode_png for Caliper API's capture),
// the service registry, and the Metal window capture. The PNG and registry cases
// run everywhere. The live Metal cases skip honestly with no device (gpu_skip.h:
// counted, exit code = Skipped in ctest, a failure under CALIPER_REQUIRE_GPU=1).
// Compiled as ObjC++ on Apple for the Metal device query. MSVC-safe: no compound
// REQUIRE expressions.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "gpu_skip.h"

#include "host_services.h"
#include "export_service.h"
#include "test_temp.h"
#include "renderer/host_renderer.h"

#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>

#define STB_IMAGE_IMPLEMENTATION            // decode-only (TEST-ONLY, README §)
#define STBI_ONLY_PNG
#include <stb_image.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// §2.1 (held-open NTFS handle) and the Vulkan VMM shareable-handle blocks below
// both need Win32; include it once, lean, with NOMINMAX so it never fights the
// other headers (mirrors the box-wide NOMINMAX discipline).
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;
using namespace caliper_host;

namespace {

// This run's folder for the files the cases write (test_temp.h), removed when
// the process exits.
struct ExportTmp {
    fs::path dir = caliper_test::unique_temp_path("caliper_export_tests");
    ~ExportTmp() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

std::string tmp_path(const char* leaf) {
    static ExportTmp tmp;
    std::error_code ec;
    fs::create_directories(tmp.dir, ec);
    return (tmp.dir / leaf).string();
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace

// ===========================================================================
// Platform-neutral: the PNG writers. A failed write creates and truncates
// nothing; a success replaces the file through a sibling temp file; the file
// decodes to the pixels written, top row first; the in-memory encoder writes
// the same bytes.
// ===========================================================================
TEST_CASE("png writer: an unwritable path creates nothing, a success replaces the file") {
    // A path whose parent directory does not exist -> encode/open fails.
    const std::string bad = "/caliper_nonexistent_dir_zzz/should_not_appear.png";
    std::vector<uint8_t> px(4 * 4 * 4, 0x7F);
    std::string why;
    CHECK(export_write_png_atomic(bad, px.data(), 4, 4, &why) == false);
    CHECK(fs::exists(bad) == false);
    CHECK(why.empty() == false);

    // A previous file at the path is replaced as a whole by a good write.
    const std::string png = tmp_path("replaced.png");
    std::ofstream(png, std::ios::binary) << "ORIGINAL";
    CHECK(export_write_png_atomic(png, px.data(), 4, 4) == true);
    const auto after = read_file(png);
    REQUIRE(after.size() > 8);
    CHECK(after[1] == 'P');
    CHECK(after[2] == 'N');
    CHECK(after[3] == 'G');
    CHECK(fs::exists(png + ".caliper_tmp") == false);

    // No pixels: refused, nothing written.
    CHECK(export_write_png_atomic(tmp_path("empty.png"), px.data(), 0, 4) == false);
    CHECK(fs::exists(tmp_path("empty.png")) == false);
}

TEST_CASE("png writer: a directory at the path refuses with the reason, a success keeps why") {
    const std::string path = tmp_path("why.png");
    std::error_code ec;
    fs::remove_all(path, ec);
    REQUIRE(fs::create_directory(path, ec));
    std::vector<uint8_t> px(2 * 2 * 4, 0x40);
    std::string why;
    CHECK(export_write_png_atomic(path, px.data(), 2, 2, &why) == false);
    CHECK(why.find("cannot rename") != std::string::npos);
    CHECK(why.find(path) != std::string::npos);
    CHECK(fs::exists(path + ".caliper_tmp") == false);

    fs::remove_all(path, ec);
    why = "untouched";
    CHECK(export_write_png_atomic(path, px.data(), 2, 2, &why) == true);
    CHECK(why == "untouched");
    fs::remove(path, ec);
}

TEST_CASE("png writer: the file decodes to the pixels, and the encoder gives the same bytes") {
    // An asymmetric 4x2 image: a bottom-up writer would swap the two rows.
    const uint32_t W = 4, H = 2;
    std::vector<uint8_t> px((size_t)W * H * 4);
    for (uint32_t i = 0; i < W * H; ++i) {
        px[i * 4 + 0] = (uint8_t)(i * 30);
        px[i * 4 + 1] = (uint8_t)(i < W ? 255 : 0);
        px[i * 4 + 2] = (uint8_t)(200 - i * 10);
        px[i * 4 + 3] = 255;
    }
    const std::string png = tmp_path("roundtrip.png");
    REQUIRE(export_write_png_atomic(png, px.data(), W, H));
    int w = 0, h = 0, n = 0;
    unsigned char* got = stbi_load(png.c_str(), &w, &h, &n, 4);
    REQUIRE(got != nullptr);
    CHECK(w == (int)W);
    CHECK(h == (int)H);
    CHECK(std::memcmp(got, px.data(), px.size()) == 0);
    stbi_image_free(got);

    std::string bytes;
    REQUIRE(export_encode_png(px.data(), W, H, &bytes));
    const auto file = read_file(png);
    CHECK(bytes == std::string(file.begin(), file.end()));
    CHECK(export_encode_png(px.data(), 0, H, &bytes) == false);
    CHECK(export_encode_png(px.data(), W, H, nullptr) == false);
}

// ===========================================================================
// Windows-only: the atomic replace-existing rename on NTFS. The PNG writer
// writes <name>.caliper_tmp then fs::rename over the target, which the
// MSVC STL maps to MoveFileExW(MOVEFILE_REPLACE_EXISTING). Two NTFS-sensitive
// claims are proven ON THE BOX (never from docs): (a) a target held open by a
// handle WITHOUT FILE_SHARE_DELETE cannot be supplanted -> the write refuses
// and the original bytes survive; (b) once that handle is closed, the very same
// path is replaced atomically with the new contents.
// ===========================================================================
#ifdef _WIN32
TEST_CASE("png writer on NTFS: a held-open handle refuses; replace-existing then succeeds") {
    const std::string path = tmp_path("held_open.png");
    std::ofstream(path, std::ios::binary) << "ORIGINAL_BYTES";
    const auto before = read_file(path);
    REQUIRE(std::string(before.begin(), before.end()) == "ORIGINAL_BYTES");
    std::vector<uint8_t> px(2 * 2 * 4, 0x55);

    // Open the target for reading with a share mode that EXCLUDES
    // FILE_SHARE_DELETE, so the temp-then-rename cannot replace it out from
    // under the open handle. FILE_SHARE_READ still lets our own read_file()
    // reopen it below to prove the bytes are intact.
    const std::wstring wpath = fs::path(path).wstring();
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(h != INVALID_HANDLE_VALUE);

    const bool replaced_while_held = export_write_png_atomic(path, px.data(), 2, 2);
    MESSAGE("NTFS held-open (share=READ, no DELETE): export_write_png_atomic returned "
            << std::string(replaced_while_held ? "true" : "false"));
    CHECK(replaced_while_held == false);          // rename onto a locked target fails
    const auto during = read_file(path);
    CHECK(during == before);                       // original byte-for-byte intact

    CloseHandle(h);

    // Replace-existing success path: the handle is gone, so the second write
    // supplants the file atomically with the new contents.
    const bool replaced_after_close = export_write_png_atomic(path, px.data(), 2, 2);
    MESSAGE("NTFS replace-existing (handle closed): export_write_png_atomic returned "
            << std::string(replaced_after_close ? "true" : "false"));
    CHECK(replaced_after_close == true);
    const auto after = read_file(path);
    REQUIRE(after.size() > 8);
    CHECK(after[1] == 'P');
    CHECK(after[2] == 'N');
    CHECK(after[3] == 'G');
}
#endif  // _WIN32

// ===========================================================================
// Live Metal path.
// ===========================================================================
#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "glfw_background.h"

namespace {

// Stands up a real Metal renderer and BINDS it to the process service registry
// (services_set_renderer), as the app does. One per process (the registry is
// process-wide).
struct MetalHostEnv {
    bool ok = false;
    bool glfw_ok = false;
    GLFWwindow* window = nullptr;
    ImGuiContext* imgui_ctx = nullptr;
    std::unique_ptr<HostRenderer> renderer;
    id<MTLDevice> device = nil;

    MetalHostEnv() {
        glfw_ok = !caliper_test::force_no_gpu() && caliper_test::glfw_init_in_background();
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

}  // namespace

// The hidden window gives the test process no Dock icon or menu bar: with GLFW
// up and the window created, the activation policy is still the background one
// (glfw_background.h).
TEST_CASE("metal live: the test process has no Dock icon or menu bar") {
    if (!env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    CHECK(caliper_host::background_activation_policy_active());
}

// Caliper API's capture in the app's window (api/window_captures.h): the
// Metal window path copies the frame it presents, with no extra render. The
// window is the hidden one of this binary (glfw_background.h), so the
// swapchain path really runs and nothing appears on screen.
TEST_CASE("metal live: the window's presented frame is read back for a capture (window path)") {
    if (!env().ok) CALIPER_SKIP_NO_GPU("no Metal device");
    HostRenderer& r = *env().renderer;
    REQUIRE(r.window_capture_supported());
    int fw = 0, fh = 0;
    glfwGetFramebufferSize(env().window, &fw, &fh);
    REQUIRE(fw > 0);
    REQUIRE(r.window_capture_request());
    std::vector<uint8_t> px;
    int w = 0, h = 0;
    HostRenderer::WindowCapture got = HostRenderer::WindowCapture::Pending;
    int frames = 0;
    for (; frames < 200 && got == HostRenderer::WindowCapture::Pending; ++frames) {
        r.new_frame();
        // A red square at the top left of the frame, over the clear colour.
        ImGui::GetForegroundDrawList()->AddRectFilled(ImVec2(0, 0), ImVec2(16, 16), IM_COL32(255, 0, 0, 255));
        r.render(fw, fh);
        got = r.window_capture_poll(&px, &w, &h);
        if (got == HostRenderer::WindowCapture::Pending)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    MESSAGE("the read-back was ready after " << frames << " frames");
    REQUIRE(got == HostRenderer::WindowCapture::Ready);
    CHECK(w == fw);
    CHECK(h == fh);
    REQUIRE(px.size() == (size_t)w * (size_t)h * 4);
    auto at = [&](int x, int y) { return px.data() + ((size_t)y * (size_t)w + (size_t)x) * 4; };
    // Top row first, RGBA, opaque: red at the top left, the clear colour
    // (0.05, 0.05, 0.08) at the bottom right.
    CHECK(at(1, 1)[0] == 255);
    CHECK(at(1, 1)[1] == 0);
    CHECK(at(1, 1)[2] == 0);
    CHECK(at(w - 2, h - 2)[0] < 30);
    CHECK(at(w - 2, h - 2)[2] < 40);
    CHECK(at(w - 2, h - 2)[3] == 255);
    std::unordered_set<uint32_t> colours;
    for (size_t i = 0; i < (size_t)w * (size_t)h; ++i) {
        uint32_t c;
        std::memcpy(&c, px.data() + 4 * i, 4);
        colours.insert(c);
    }
    CHECK(colours.size() > 1);
    // Nothing armed: the next poll reports none, and frames go on.
    CHECK(r.window_capture_poll(&px, &w, &h) == HostRenderer::WindowCapture::None);
    r.new_frame();
    r.render(fw, fh);
}

#endif  // CALIPER_HAVE_METAL


CALIPER_GPU_TEST_MAIN()

// The id list the loader negotiates against and the tables the host vends are
// kept in two places (negotiation.cpp, host_services.cpp); they must agree.
TEST_CASE("services: every advertised id is vended") {
    for (const std::string& id : service_ids()) {
        CAPTURE(id);
        CHECK(services_get(id.c_str()) != nullptr);
    }
    CHECK(services_get("caliper.nope.v1") == nullptr);
}

TEST_CASE("ui v1_1: the v1 prefix is ui.v1's, and dock_hint reaches the sink") {
    const auto* v1 = static_cast<const CaliperUiV1*>(services_get(CALIPER_UI_V1));
    const auto* v11 = static_cast<const CaliperUiV1_1*>(services_get(CALIPER_UI_V1_1));
    REQUIRE(v1 != nullptr);
    REQUIRE(v11 != nullptr);
    CHECK(v11->struct_size == sizeof(CaliperUiV1_1));
    CHECK(v1->struct_size == sizeof(CaliperUiV1));
    CHECK(v11->imgui_context == v1->imgui_context);
    CHECK(v11->imgui_allocators == v1->imgui_allocators);
    REQUIRE(v11->dock_hint != nullptr);

    v11->dock_hint("Window", CALIPER_DOCK_SIDE_TOP);   // no sink: nothing happens
    static std::vector<std::pair<std::string, uint32_t>> seen;
    seen.clear();
    set_dock_hint_sink(
        [](void* user, const char* title, uint32_t role) {
            CHECK(user == &seen);
            seen.emplace_back(title, role);
        },
        &seen);
    v11->dock_hint("Window", CALIPER_DOCK_SIDE_TOP);
    v11->dock_hint(nullptr, CALIPER_DOCK_CENTRAL);      // ignored
    set_dock_hint_sink(nullptr, nullptr);
    v11->dock_hint("After", CALIPER_DOCK_CENTRAL);
    REQUIRE(seen.size() == 1);
    CHECK(seen[0].first == "Window");
    CHECK(seen[0].second == (uint32_t)CALIPER_DOCK_SIDE_TOP);
}
