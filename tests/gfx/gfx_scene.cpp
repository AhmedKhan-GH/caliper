// Rows of the graphics suite for the scene core (src/host/scene) drawn through the whole renderer
// core on Metal: the scene compiled by SceneCompiler over BackendResources, with the generated
// programs (kMetalPrograms), submitted to the graphics layer's Metal backend on the renderer's own
// device and queue, rendered into offscreen targets and read back through the layer
// (docs/explanations/rendering.md, docs/explanations/testing.md).
//
// - The one-entity row draws the fixture of tests/scene/scene_row_fixture.h, square points one
//   pixel wide on pixel centres, and compares the readback with its reference byte for byte.
// - The surface row draws the 3-D scene of tests/scene/surface_row_fixture.h (a height field
//   coloured through viridis, a line strip, a disc marker and a box of segments, with depth) and
//   compares it with a CPU rasterisation of the same compiled draws under the parity rules of
//   web geometry: a colour from a lookup is exact away from edges, a pixel within a pixel of an
//   edge (the edges between a mesh's flat faces included) may take either side's colour, and
//   flat-shaded faces may differ by 2 levels.
// - The pick row draws the same scene's pick variants into the 16 x 16 ID window around a pixel
//   on the surface, on the marker and on the path, and decodes the texels with the Picker.
// - The timing row (CALIPER_GFX_SCENE_BENCH=1) reports the scene core's compile time and the
//   layer's CPU encode time for the surface scene and for 1,000 entities.
//
// CALIPER_GFX_CAPTURE_DIR=<folder> writes first-scene-points.png (scaled up eight times) and
// first-scene-surface.png there. Compiled as Objective-C++ with ARC on Apple.
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>

#include "../scene/scene_row_fixture.h"
#include "../scene/surface_row_fixture.h"
#include "../shaders/colormap_cases.h"
#include "scene_reference.h"
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
using namespace scene_reference;
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

// The surface scene on the layer: its buffer, targets, tree and compiled colour pass.
struct SurfaceScene {
    scene_fixtures::SurfaceRowFixture fx;
    Rig rig;
    Buffer data;
    Texture color, depth;
    SceneTree tree{Space::k3D};
    SceneCompiler compiler;
    ViewCamera camera;
    bool init() {
        if (!rig.init()) return false;
        Backend& gfx = *rig.gfx;
        const std::vector<uint8_t> bytes = fx.bytes();
        data = gfx.create_buffer({fx.bytes_total, kVertex | kIndex | kStorage | kCopyDst, false, "surface row"});
        REQUIRE_MESSAGE(bool(data), gfx.last_error());
        REQUIRE(gfx.write_buffer(data, 0, bytes.data(), bytes.size()));
        const uint32_t S = fx.kSize;
        color = gfx.create_texture({S, S, Format::RGBA8Unorm, kRenderTarget | kCopySrcTex, "surface colour"});
        depth = gfx.create_texture({S, S, Format::Depth32Float, kRenderTarget, "surface depth"});
        REQUIRE_MESSAGE((color && depth), gfx.last_error());
        REQUIRE(fx.populate(tree, data));
        tree.update();
        rig.res->update_class_table(tree);
        camera = fx.camera(tree);
        return true;
    }
    RenderTarget target() const {
        RenderTarget t;
        t.color = color;
        t.depth = depth;
        t.clear_rgba = fx.kClear;
        return t;
    }
    // Compiles the index clamp and the colour pass and submits them; returns the draws.
    std::vector<DrawInfo> draw() {
        Backend& gfx = *rig.gfx;
        caliper_gfx::Frame frame;
        compiler.prepare(tree, *rig.res, frame);
        DrawList list(target());
        DrawMap map;
        const CompileResult r = compiler.compile(tree, camera, fx.options(), *rig.res, list, &map);
        REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
        CHECK(r.skipped == 0);
        CHECK_MESSAGE(rig.res->error().empty(), rig.res->error());
        std::vector<DrawInfo> draws = draws_of(list, map, tree);
        frame.render(std::move(list));
        const Serial s = gfx.submit(std::move(frame));
        REQUIRE_MESSAGE(s != 0, gfx.last_error());
        return draws;
    }
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

TEST_CASE("gfx/Metal scene: a surface, a path, a marker and a box drawn through the scene core match the CPU reference") {
    SurfaceScene sc;
    if (!sc.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend& gfx = *sc.rig.gfx;
    const std::vector<DrawInfo> draws = sc.draw();
    // In the byte order of the paths: box, marker, path, surface; all opaque, with depth.
    REQUIRE(draws.size() == 4);
    CHECK(draws[0].path == "box");
    CHECK(draws[1].path == "marker");
    CHECK(draws[2].path == "path");
    CHECK(draws[3].path == "surface");
    CHECK(draws[3].p.color_mode == 1);      // values through the colormap
    CHECK(draws[3].p.shade_mode == 0);      // flat faces, from fs_mesh_flat
    CHECK(draws[3].record.count == uint32_t(sc.fx.kTriangles * 3));
    CHECK(bool(draws[3].record.index_buffer));
    CHECK(bool(draws[1].record.groups[2]));
    CHECK(bool(draws[3].record.groups[2]));

    const int S = sc.fx.kSize;
    const std::vector<uint8_t> got = read_back(gfx, sc.color, S, S);
    capture("first-scene-surface.png", got, S, S, 1);
    const Reference ref = rasterise(draws, sc.fx);
    CHECK(ref.clipped == 0);
    Comparison c = compare(ref, got);
    MESSAGE("surface row: ", c.interior, " interior pixels (surface ", c.interior_by_draw[3], ", path ",
            c.interior_by_draw[2], ", marker ", c.interior_by_draw[1], ", box ", c.interior_by_draw[0], ", clear ",
            c.interior_by_draw[-1], "), ", c.edge, " edge pixels, largest lit difference ", c.max_interior_lit, ", ",
            c.lit_own, " of ", c.lit, " surface pixels within 2 levels of their own face, ", c.failed, " failed");
    CHECK_MESSAGE(c.failed == 0, c.failed, " pixels outside the parity rules; ", c.first);
    // Every part is checked away from its edges.
    for (int d = -1; d < 4; ++d) CHECK(c.interior_by_draw[d] > 0);
}

TEST_CASE("gfx/Metal scene: the pick pass finds the surface, the marker and the path with their items") {
    SurfaceScene sc;
    if (!sc.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend& gfx = *sc.rig.gfx;
    const std::vector<DrawInfo> draws = sc.draw();
    REQUIRE(draws.size() == 4);
    const Reference ref = rasterise(draws, sc.fx);
    const int S = sc.fx.kSize;

    // Pixels well inside each part in the reference.
    auto interior = [&](int i, int j) {
        const uint32_t v = ref.id[size_t(j) * S + i];
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di)
                if (ref.id[size_t(std::clamp(j + dj, 0, S - 1)) * S + std::clamp(i + di, 0, S - 1)] != v) return false;
        return true;
    };
    auto pixels_of = [&](int draw) {
        std::vector<std::pair<int, int>> out;
        for (int j = 1; j + 1 < S; ++j)
            for (int i = 1; i + 1 < S; ++i)
                if (ref.id[size_t(j) * S + i] && Reference::draw_of(ref.id[size_t(j) * S + i]) == draw && interior(i, j))
                    out.push_back({i, j});
        return out;
    };
    std::vector<std::pair<int, int>> on_surface = pixels_of(3), on_path = pixels_of(2), on_marker = pixels_of(1);
    REQUIRE(!on_surface.empty());
    REQUIRE(!on_path.empty());
    REQUIRE(!on_marker.empty());
    auto nearest = [&](const std::vector<std::pair<int, int>>& v, double x, double y) {
        return *std::min_element(v.begin(), v.end(), [&](auto a, auto b) {
            return std::hypot(a.first - x, a.second - y) < std::hypot(b.first - x, b.second - y);
        });
    };
    const F4 mc = mul(draws[1].p.mvp, sc.fx.marker[0], sc.fx.marker[1], sc.fx.marker[2]);
    const double mx = (mc.x / mc.w + 1) * 0.5 * S, my = (1 - mc.y / mc.w) * 0.5 * S;
    struct Probe { const char* what; std::pair<int, int> px; const char* path; Primitive primitive; };
    const Probe probes[3] = {
        {"surface", nearest(on_surface, S / 2.0, S / 2.0), "surface", Primitive::Mesh},
        {"marker", nearest(on_marker, mx, my), "marker", Primitive::Points},
        {"path", on_path[on_path.size() / 2], "path", Primitive::Lines},
    };

    Texture ids = gfx.create_texture({16, 16, Format::RG32Uint, kRenderTarget | kCopySrcTex, "pick ids"});
    Texture pick_depth = gfx.create_texture({16, 16, Format::Depth32Float, kRenderTarget | kCopySrcTex, "pick depth"});
    REQUIRE_MESSAGE((ids && pick_depth), gfx.last_error());
    Picker picker;
    for (const Probe& pr : probes) {
        const std::string what = pr.what;
        CAPTURE(what);
        const int i = pr.px.first, j = pr.px.second;
        const uint32_t expect = ref.id[size_t(j) * S + i];
        const PickWindow w = picker.request(i + 0.5f, j + 0.5f, 8, S, S);
        const int32_t window[4] = {w.x0, w.y0, w.width, w.height};
        RenderTarget pt;
        pt.color = ids;
        pt.depth = pick_depth;
        DrawList pick(pt);
        DrawMap pick_map;
        const CompileResult r = sc.compiler.compile_pick(sc.tree, sc.camera, sc.fx.options(), window, *sc.rig.res, pick, &pick_map);
        REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
        caliper_gfx::Frame f;
        f.render(std::move(pick));
        REQUIRE_MESSAGE(gfx.submit(std::move(f)) != 0, gfx.last_error());
        const std::vector<uint8_t> id_bytes = read_back(gfx, ids, 16, 16);
        const std::vector<uint8_t> depth_bytes = read_back(gfx, pick_depth, 16, 16);
        const std::optional<Hit> hit = picker.deliver(picker.current(), pick_map.revision, id_bytes, &depth_bytes, pick_map);
        REQUIRE(hit.has_value());
        CHECK(sc.tree.path(hit->entity) == pr.path);
        CHECK(hit->primitive == pr.primitive);
        CHECK(hit->texel[0] == 8);
        CHECK(hit->texel[1] == 8);
        const uint32_t item = Reference::item_of(expect);
        if (pr.primitive == Primitive::Mesh) {
            // A vertex of the triangle under the pixel.
            const Reference::Tri& tri = ref.tris[item];
            CHECK((hit->item == uint64_t(tri.index[0]) || hit->item == uint64_t(tri.index[1]) ||
                   hit->item == uint64_t(tri.index[2])));
            // The world position from the depth lies on that face.
            double q[3], z;
            bool inside;
            REQUIRE(weights(tri.v, i + 0.5, j + 0.5, q, &z, &inside));
            double o[3];
            for (int k = 0; k < 3; ++k) o[k] = q[0] * tri.v[0].o[k] + q[1] * tri.v[1].o[k] + q[2] * tri.v[2].o[k];
            REQUIRE(hit->has_position);
            const double dist = std::sqrt((hit->position.x - o[0]) * (hit->position.x - o[0]) +
                                          (hit->position.y - o[1]) * (hit->position.y - o[1]) +
                                          (hit->position.z - o[2]) * (hit->position.z - o[2]));
            MESSAGE("surface pick: item ", hit->item, ", triangle ", item, ", position off the face's point by ", dist);
            CHECK(dist < 1e-3);
        } else {
            CHECK(hit->item == item);   // the marker's point 0, or the strip's segment
        }
        CHECK(std::fabs(hit->depth - ref.depth[size_t(j) * S + i]) < 1e-5f);
        MESSAGE(what, " at (", i, ", ", j, "): entity ", sc.tree.path(hit->entity), ", item ", hit->item, ", depth ",
                hit->depth);
    }
}

TEST_CASE("gfx/Metal scene: timing — compile and encode of the surface scene and of 1,000 entities") {
    if (std::getenv("CALIPER_GFX_SCENE_BENCH") == nullptr) return;
    SurfaceScene sc;
    if (!sc.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    Backend& gfx = *sc.rig.gfx;
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto run = [&](const char* name, const SceneTree& t, const ViewCamera& cam, const CompileOptions& o, SceneCompiler& comp) {
        const int kWarm = 5, kRuns = 20;
        std::vector<double> compile_ms, submit_ms;
        size_t draws = 0;
        for (int k = 0; k < kWarm + kRuns; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            DrawList list(sc.target());
            DrawMap map;
            const CompileResult r = comp.compile(t, cam, o, *sc.rig.res, list, &map);
            const auto t1 = std::chrono::steady_clock::now();
            REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
            draws = list.records().size;
            caliper_gfx::Frame f;
            f.render(std::move(list));
            const Serial s = gfx.submit(std::move(f));
            const auto t2 = std::chrono::steady_clock::now();
            REQUIRE_MESSAGE(s != 0, gfx.last_error());
            gfx.wait(s);                     // outside the timed region
            if (k >= kWarm) {
                compile_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                submit_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
        }
        double load[3] = {};
        getloadavg(load, 3);
        std::printf("[gfx-scene-bench] %s: %zu draws; compile median %.3f ms; submit (validate + encode + commit) "
                    "median %.3f ms; %d runs; load %.2f\n",
                    name, draws, median(compile_ms), median(submit_ms), kRuns, load[0]);
        std::fflush(stdout);
    };
    sc.draw();   // the index clamp, once
    gfx.wait(gfx.last_use(sc.color));
    run("surface scene", sc.tree, sc.camera, sc.fx.options(), sc.compiler);

    // 1,000 entities on a 10 x 10 x 10 grid, a third each of points, segments and small meshes
    // coloured through the colormap, all reading arrays of the surface row's buffer.
    SceneTree many(Space::k3D);
    const scene_fixtures::SurfaceRowFixture& fx = sc.fx;
    using A = scene_fixtures::SurfaceRowFixture;
    for (int k = 0; k < 1000; ++k) {
        char name[16];
        std::snprintf(name, sizeof name, "e%04d", k);
        const EntityId e = many.ensure(name);
        REQUIRE(bool(e));
        REQUIRE(many.set_transform(e, translation({3.0 * (k % 10), 3.0 * ((k / 10) % 10), 3.0 * (k / 100)})));
        Batch b;
        if (k % 3 == 0) {
            b.primitive = Primitive::Points;
            b.arrays = {A::array(Role::Positions, sc.data, fx.off_path, 16, 12, VertexFormat::F32x3)};
            b.size_px = 4;
        } else if (k % 3 == 1) {
            b.primitive = Primitive::Lines;
            b.arrays = {A::array(Role::Positions, sc.data, fx.off_box, 16, 12, VertexFormat::F32x3)};
        } else {
            b.primitive = Primitive::Mesh;
            b.arrays = {A::array(Role::Positions, sc.data, fx.off_positions, 18, 12, VertexFormat::F32x3),
                        A::array(Role::Values, sc.data, fx.off_values, 18, 4, VertexFormat::F32)};
            b.vmin = fx.vmin;
            b.vmax = fx.vmax;
        }
        b.bounds.add({-1, -1, -1});
        b.bounds.add({1, 1, 1});
        REQUIRE(many.set_batch(e, b));
    }
    many.update();
    ViewCamera mcam = Orbit{};
    fit(mcam, many.scene_bounds(), 1.0);
    SceneCompiler mcomp;
    run("1,000 entities", many, mcam, sc.fx.options(), mcomp);
}

#endif  // CALIPER_HAVE_METAL
