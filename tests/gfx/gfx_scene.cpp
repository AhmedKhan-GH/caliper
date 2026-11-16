// Rows of the graphics suite for the scene core (src/host/scene) drawn through the whole renderer
// core on Metal: the scene compiled by SceneCompiler over BackendResources, with the generated
// programs (kMetalPrograms), submitted to the graphics layer's Metal backend on the renderer's own
// device and queue, rendered into offscreen targets and read back through the layer
// (docs/explanations/rendering.md, docs/explanations/testing.md).
//
// - The one-entity row draws the fixture of tests/scene/scene_row_fixture.h, square points one
//   pixel wide on pixel centres, and compares the readback with its reference byte for byte.
//
// CALIPER_GFX_CAPTURE_DIR=<folder> writes first-scene-points.png (scaled up eight times) there.
// Compiled as Objective-C++ with ARC on Apple.
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>

#include "../scene/scene_row_fixture.h"
#include "../shaders/colormap_cases.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "renderer/host_renderer.h"
#include "scene/backend_resources.h"
#include "scene/pick.h"
#include "tensor_bridge.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// The suite's Metal environment (gfx_main.cpp): its renderer, or nullptr without a device.
caliper_host::HostRenderer* gfx_metal_renderer();

using namespace caliper_gfx;
using namespace caliper_scene;
namespace sh = caliper_gfx::shaders;

namespace {

std::unique_ptr<Backend> metal_layer() {
    caliper_host::HostRenderer* r = gfx_metal_renderer();
    if (r == nullptr) return nullptr;
    return make_metal_backend(r->native_device(), r->native_queue());
}

std::vector<uint8_t> read_back(Backend& gfx, Texture t, uint32_t w, uint32_t h) {
    Readback r = gfx.read_texture(t, 0, 0, w, h);
    REQUIRE_MESSAGE(bool(r), gfx.last_error());
    std::vector<uint8_t> out;
    REQUIRE(gfx.wait(gfx.last_use(t)));
    REQUIRE(gfx.take_readback(r, &out));
    return out;
}

// A PNG of an RGBA8 readback in CALIPER_GFX_CAPTURE_DIR, scaled up `scale` times with nearest
// sampling so that small targets stay legible.
void capture(const char* name, const std::vector<uint8_t>& rgba, int w, int h, int scale) {
    const char* dir = std::getenv("CALIPER_GFX_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    std::vector<uint8_t> big(size_t(w) * scale * h * scale * 4);
    for (int y = 0; y < h * scale; ++y)
        for (int x = 0; x < w * scale; ++x)
            std::memcpy(&big[(size_t(y) * w * scale + x) * 4], &rgba[(size_t(y / scale) * w + x / scale) * 4], 4);
    CHECK(stbi_write_png(path.c_str(), w * scale, h * scale, 4, big.data(), w * scale * 4) != 0);
    MESSAGE("capture written: ", path);
}

// The colormap blocks of these rows, cached by colormap and range as a view caches them. The keys
// come from the CPU reference map_f32_to_rgba8 through the shader tests' derivation
// (tests/shaders/colormap_cases.h): this is the rows' own key derivation, which the shared one of
// migration step 6 replaces.
BackendResources::ColormapSource row_colormaps() {
    auto cache = std::make_shared<std::map<std::tuple<uint32_t, uint32_t, uint32_t>, sh::Cmap>>();
    return [cache](uint32_t colormap, float vmin, float vmax, sh::Cmap* out) {
        uint32_t lo, hi;
        std::memcpy(&lo, &vmin, 4);
        std::memcpy(&hi, &vmax, 4);
        const auto key = std::make_tuple(colormap, lo, hi);
        auto it = cache->find(key);
        if (it == cache->end()) {
            const uint32_t* lut = caliper_host::colormap_lut(int32_t(colormap));
            const std::vector<uint32_t> table(lut, lut + 256);
            it = cache->emplace(key, colormap_cases::block({"row", vmin, vmax}, table)).first;
        }
        *out = it->second;
    };
}

// The layer on Metal with the scene core's resources over it.
struct Rig {
    std::unique_ptr<Backend> gfx;
    std::unique_ptr<BackendResources> res;
    bool init() {
        gfx = metal_layer();
        if (!gfx) return false;
        res = std::make_unique<BackendResources>(*gfx, sh::kMetalPrograms, row_colormaps());
        std::string err;
        REQUIRE_MESSAGE(res->init(&err), err);
        return true;
    }
    ~Rig() { res.reset(); }   // before the backend
};

}  // namespace

TEST_CASE("gfx/Metal scene: the one-entity scene drawn through the scene core reads back byte-exact") {
    Rig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend& gfx = *rig.gfx;
    const scene_fixtures::SceneRowFixture fx;
    const std::vector<uint8_t> bytes = fx.bytes();
    Buffer data = gfx.create_buffer({fx.kBytes, kVertex | kCopyDst, false, "scene row"});
    REQUIRE_MESSAGE(bool(data), gfx.last_error());
    REQUIRE(gfx.write_buffer(data, 0, bytes.data(), bytes.size()));
    Texture color = gfx.create_texture({uint32_t(fx.kWidth), uint32_t(fx.kHeight), Format::RGBA8Unorm,
                                        kRenderTarget | kCopySrcTex, "scene row colour"});
    REQUIRE_MESSAGE(bool(color), gfx.last_error());

    SceneTree t(Space::k2D);
    REQUIRE(t.set_batch(t.root(), fx.batch(data, bytes.data())));
    t.update();
    rig.res->update_class_table(t);
    SceneCompiler compiler;
    caliper_gfx::Frame frame;
    compiler.prepare(t, *rig.res, frame);
    RenderTarget rt;
    rt.color = color;
    rt.clear_rgba = fx.kClear;
    DrawList list(rt);
    DrawMap map;
    const CompileResult r = compiler.compile(t, fx.camera(), fx.options(), *rig.res, list, &map);
    REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
    REQUIRE(list.records().size == 1);
    // vs_points declares img_value in group 2, so a points draw binds groups 0, 1 and 2.
    const DrawRecord& d = list.records()[0];
    CHECK(bool(d.groups[0]));
    CHECK(bool(d.groups[1]));
    CHECK(bool(d.groups[2]));
    CHECK(d.instance_count == fx.pixels.size());
    frame.render(std::move(list));
    const Serial s = gfx.submit(std::move(frame));
    REQUIRE_MESSAGE(s != 0, gfx.last_error());
    CHECK_MESSAGE(rig.res->error().empty(), rig.res->error());

    const std::vector<uint8_t> got = read_back(gfx, color, fx.kWidth, fx.kHeight);
    const std::vector<uint8_t> ref = fx.reference();
    REQUIRE(got.size() == ref.size());
    size_t first = SIZE_MAX, differ = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        if (got[i] != ref[i]) {
            if (first == SIZE_MAX) first = i;
            ++differ;
        }
    CHECK_MESSAGE(differ == 0, differ, " bytes differ, the first at byte ", first);
    capture("first-scene-points.png", got, fx.kWidth, fx.kHeight, 8);
}

#endif  // CALIPER_HAVE_METAL
