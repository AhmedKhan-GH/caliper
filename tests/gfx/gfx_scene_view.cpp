// Rows of the graphics suite for the scene view of the live views (src/host/live/live_scene.h):
// scenes that arrive as decoded wire protocol 3.2 messages, applied to a SceneView, compiled by the
// scene core and drawn through the graphics layer's Metal backend on the renderer's own device and
// queue into the view's offscreen target, read back through the layer.
//
// - The golden frames of tests/fixtures/wire_v3/v3_2 decode and apply to scene views, one per
//   view they name, and the scenes' state follows: parts, layers, a recursive clear of some parts
//   that keeps static items, and the reset that opens a new state.
// - The demo's message sequence (the reset of a scene's state, a static box of segments, a mesh
//   coloured through viridis, a line strip and a point marker, encoded with the C++ codec from the
//   surface row's fixture and decoded again) draws the surface row's scene: the draw list is the
//   row's, and the pixels match the CPU rasterisation of gfx_scene.cpp under the same parity rules.
// - A late attach: a view that held another state receives a reset and the state, and holds only
//   that; a clear keeps static items and a static clear removes them.
// - An update of one batch reuses its buffers and redraws; nothing is compiled or submitted while
//   nothing changes; a larger batch gets new buffers.
// - The timing row (CALIPER_GFX_SCENE_VIEW_BENCH=1) reports the view's cost per frame for the demo
//   scene (compile, submit, GPU time) and the cost of applying one mesh update.
//
// CALIPER_GFX_CAPTURE_DIR=<folder> writes scene-view-row.png and scene-view-row-update.png there.
// Compiled as Objective-C++ with ARC on Apple.
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>

#include "../scene/surface_row_fixture.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "live/live_scene.h"
#include "net/wire.h"
#include "renderer/host_renderer.h"
#include "scene_reference.h"

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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

caliper_host::HostRenderer* gfx_metal_renderer();

namespace {

namespace wire = caliper_net::wire;
namespace fs = std::filesystem;
using caliper_net::Json;
using caliper_live_views::SceneView;
using namespace scene_reference;

std::unique_ptr<Backend> view_layer() {
    caliper_host::HostRenderer* r = gfx_metal_renderer();
    if (r == nullptr) return nullptr;
    return caliper_gfx::make_metal_backend(r->native_device(), r->native_queue());
}

void capture_png(const char* name, const std::vector<uint8_t>& rgba, int w, int h) {
    const char* dir = std::getenv("CALIPER_GFX_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4) != 0);
    MESSAGE("capture written: ", path);
}

// A scene view on its own backend, as the live views make one.
struct ViewRig {
    std::unique_ptr<Backend> gfx;
    std::unique_ptr<SceneView> view;
    bool init(const char* space = CALIPER_NET_SPACE_3D, const char* up = "+z", const char* units = "") {
        gfx = view_layer();
        if (!gfx) return false;
        view = std::make_unique<SceneView>(*gfx, sh::kMetalPrograms);
        std::string err;
        REQUIRE_MESSAGE(view->init(&err), err);
        view->describe(wire::SceneView{space, up, units});
        return true;
    }
    ~ViewRig() { view.reset(); }   // before the backend
};

// Messages through the codec: each is encoded, decoded again, and kept with its frame, as a
// source hands a scene view a decoded message and the frame its sections point into.
struct Wire {
    std::vector<std::shared_ptr<std::vector<uint8_t>>> frames;
    wire::Message roundtrip(const wire::Message& m) {
        auto bytes = std::make_shared<std::vector<uint8_t>>();
        std::string err;
        REQUIRE_MESSAGE(wire::encode(m, bytes.get(), &err, CALIPER_NET_MINOR_3_2), err);
        wire::Message out;
        const wire::DecodeResult r = wire::decode(bytes->data(), bytes->size(), &out);
        REQUIRE_MESSAGE(r.ok(), r.message);
        frames.push_back(bytes);
        return out;
    }
};

bool feed(SceneView& v, const wire::Message& m, std::string* why = nullptr) {
    std::string w;
    bool ok = true;
    if (const auto* u = std::get_if<wire::SceneUpdate>(&m)) ok = v.apply(*u, &w);
    else if (const auto* c = std::get_if<wire::SceneClear>(&m)) ok = v.apply(*c, &w);
    if (why) *why = w;
    return ok;
}

Json rgba_json(uint32_t c) {
    Json a = Json::array();
    for (int k = 0; k < 4; ++k) a.push(Json(int((c >> (8 * k)) & 0xffu)));
    return a;
}

Json stats_json(const std::vector<float>& rows, const std::vector<float>* values) {
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i + 2 < rows.size(); i += 3)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], rows[i + k]);
            hi[k] = std::max(hi[k], rows[i + k]);
        }
    Json s = Json::array();
    s.push(Json(values ? double(*std::min_element(values->begin(), values->end())) : 0.0));
    s.push(Json(values ? double(*std::max_element(values->begin(), values->end())) : 0.0));
    for (int k = 0; k < 3; ++k) {
        s.push(Json(double(lo[k])));
        s.push(Json(double(hi[k])));
    }
    return s;
}

wire::SceneUpdate update(const char* path, const char* part, uint64_t seq, bool is_static) {
    wire::SceneUpdate u;
    u.sub = 1;
    u.view = "landscape";
    u.seq = seq;
    u.stamp_ns = 1000 + seq;
    u.path = path;
    u.part = part;
    u.is_static = is_static;
    return u;
}

// The demo's scene as protocol 3.2 messages, from the surface row's fixture: the reset of a
// scene's state, the box (static segments), the surface (a mesh coloured through viridis on the
// row's range), the path (one strip) and the marker (one point).
struct DemoMessages {
    const scene_fixtures::SurfaceRowFixture& fx;
    std::vector<int32_t> strip = {scene_fixtures::SurfaceRowFixture::kPathVertices};
    std::vector<float> values;

    explicit DemoMessages(const scene_fixtures::SurfaceRowFixture& f) : fx(f), values(f.values) {}

    wire::Message reset() const {
        wire::SceneClear c;
        c.sub = 1;
        c.view = "landscape";
        c.seq = 1;
        c.path = "";
        c.reset = true;
        c.state = true;
        return c;
    }
    wire::Message box() const {
        wire::SceneUpdate u = update("box", CALIPER_NET_PART_LINES, 2, true);
        u.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
        u.params.set("color", rgba_json(fx.kBoxRgba));
        u.params.set("width", Json(double(fx.kBoxWidth)));
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {int64_t(fx.box.size() / 3), 3},
                                               fx.box.data(), fx.box.size() * 4));
        return u;
    }
    wire::Message surface(uint64_t seq = 3) const {
        wire::SceneUpdate u = update("surface", CALIPER_NET_PART_MESH, seq, false);
        u.params.set("colormap", Json(0));
        u.params.set("flags", Json(0));
        u.params.set("vmin", Json(double(fx.vmin)));
        u.params.set("vmax", Json(double(fx.vmax)));
        u.params.set("stats", stats_json(fx.positions, &values));
        const int64_t n = int64_t(fx.positions.size() / 3);
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {n, 3}, fx.positions.data(),
                                               fx.positions.size() * 4));
        u.arrays.push_back(wire::Section::make("indices", CALIPER_NET_DTYPE_I32, {int64_t(fx.indices.size() / 3), 3},
                                               fx.indices.data(), fx.indices.size() * 4));
        u.arrays.push_back(wire::Section::make("values", CALIPER_NET_DTYPE_F32, {n}, values.data(), values.size() * 4));
        return u;
    }
    wire::Message path() const {
        wire::SceneUpdate u = update("path", CALIPER_NET_PART_LINES, 4, false);
        u.params.set("mode", CALIPER_NET_LINES_STRIPS);
        u.params.set("color", rgba_json(fx.kPathRgba));
        u.params.set("width", Json(double(fx.kPathWidth)));
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {int64_t(fx.path.size() / 3), 3},
                                               fx.path.data(), fx.path.size() * 4));
        u.arrays.push_back(wire::Section::make("strip_lengths", CALIPER_NET_DTYPE_I32, {1}, strip.data(), 4));
        return u;
    }
    wire::Message marker() const {
        wire::SceneUpdate u = update("marker", CALIPER_NET_PART_POINTS, 5, false);
        u.params.set("color", rgba_json(fx.kMarkerRgba));
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {1, 3}, fx.marker, 12));
        return u;
    }
};

// The demo's scene applied to a view and drawn at the row's size; returns the draws.
std::vector<DrawInfo> draw_demo(ViewRig& rig, Wire& w, const DemoMessages& dm) {
    SceneView& v = *rig.view;
    for (const wire::Message& m : {dm.reset(), dm.box(), dm.surface(), dm.path(), dm.marker()}) {
        std::string why;
        REQUIRE_MESSAGE(feed(v, w.roundtrip(m), &why), why);
    }
    v.keep_draws(true);
    std::string why;
    const uint32_t S = scene_fixtures::SurfaceRowFixture::kSize;
    REQUIRE_MESSAGE(v.render(S, S, 1.0f, scene_fixtures::SurfaceRowFixture::kClear, &why), why);
    return draws_of(v.last_draws(), v.last_map(), v.tree());
}

std::vector<uint8_t> pixels(SceneView& v) {
    std::vector<uint8_t> px;
    std::string why;
    REQUIRE_MESSAGE(v.read_pixels(&px, &why), why);
    REQUIRE(px.size() == size_t(v.width()) * v.height() * 4);
    return px;
}

fs::path golden_dir() { return fs::path(CALIPER_TEST_SOURCE_ROOT) / "tests" / "fixtures" / "wire_v3" / "v3_2"; }

}  // namespace

TEST_CASE("gfx/Metal scene view: the golden frames of protocol 3.2 apply to scene views and their state follows") {
    // One view per scene the frames name: "landscape" as VIEWS describes it, "world" with the
    // defaults, "flat" in 2-D (its positions are (N, 2)).
    std::map<std::string, std::unique_ptr<ViewRig>> views;
    auto view_of = [&](const std::string& name) -> SceneView* {
        auto it = views.find(name);
        if (it == views.end()) {
            auto rig = std::make_unique<ViewRig>();
            const bool ok = name == "flat" ? rig->init(CALIPER_NET_SPACE_2D, "-y", "px")
                            : name == "landscape" ? rig->init(CALIPER_NET_SPACE_3D, "+z", "")
                                                  : rig->init(CALIPER_NET_SPACE_3D, "+y", "m");
            if (!ok) return nullptr;
            it = views.emplace(name, std::move(rig)).first;
        }
        return it->second->view.get();
    };
    if (!view_of("landscape")) CALIPER_SKIP_NO_GPU("no Metal device");

    // VIEWS describes the landscape: 3-D, +z up, no unit note.
    std::vector<uint8_t> views_bin;
    {
        std::ifstream f(golden_dir() / "views_scene.bin", std::ios::binary);
        views_bin.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        wire::Message m;
        REQUIRE(wire::decode(views_bin.data(), views_bin.size(), &m).ok());
        const wire::Views& vs = std::get<wire::Views>(m);
        wire::SceneView info;
        std::string err;
        REQUIRE(wire::scene_view_info(vs.views.at(0), &info, &err));
        CHECK(info.up == "+z");
        view_of("landscape")->describe(info);
    }
    CHECK(view_of("landscape")->tree().up() == caliper_scene::Axis::PosZ);

    const char* updates[] = {"scene_update_points",      "scene_update_points_stamped", "scene_update_points_color",
                             "scene_update_points_values", "scene_update_lines_segments", "scene_update_lines_strip",
                             "scene_update_lines_strips", "scene_update_mesh",          "scene_update_mesh_normals",
                             "scene_update_mesh_colors"};
    std::vector<std::vector<uint8_t>> keep;
    auto load = [&](const char* name) {
        std::ifstream f(golden_dir() / (std::string(name) + ".bin"), std::ios::binary);
        keep.emplace_back(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        wire::Message m;
        const wire::DecodeResult r = wire::decode(keep.back().data(), keep.back().size(), &m);
        REQUIRE_MESSAGE(r.ok(), name, ": ", r.message);
        return m;
    };
    for (const char* name : updates) {
        const wire::Message m = load(name);
        const auto& u = std::get<wire::SceneUpdate>(m);
        std::string why;
        CHECK_MESSAGE(view_of(u.view)->apply(u, &why), name, ": ", why);
    }
    using caliper_scene::Primitive;
    SceneView& land = *view_of("landscape");
    SceneView& world = *view_of("world");
    SceneView& flat = *view_of("flat");
    auto batch = [](SceneView& v, const char* path, Primitive p) {
        const caliper_scene::EntityId e = v.tree().find(path);
        return e ? v.tree().batch(e, p) : nullptr;
    };
    // The landscape: the root's static segments, the static markers, the path, the current point and
    // the surface (the frame with normals replaced the first mesh; its normals are not built).
    CHECK(land.counts().entities == 5);
    CHECK(land.counts().batches == 5);
    REQUIRE(batch(land, "", Primitive::Lines));
    CHECK(land.tree().batch_static(land.tree().root(), Primitive::Lines));
    CHECK(batch(land, "markers", Primitive::Points)->colormap == 1);
    CHECK(batch(land, "markers", Primitive::Points)->vmin == 0.125f);   // flags 3: AUTO ends from the stats
    CHECK(batch(land, "markers", Primitive::Points)->vmax == 0.75f);
    CHECK(batch(land, "current", Primitive::Points)->flat_rgba == (255u | 80u << 8 | 80u << 16 | 255u << 24));
    CHECK(batch(land, "path", Primitive::Lines)->line_mode == caliper_scene::LineMode::Strips);
    REQUIRE(batch(land, "surface", Primitive::Mesh));
    CHECK(batch(land, "surface", Primitive::Mesh)->array(caliper_scene::Role::Indices)->count == 2);
    // The world: a cloud and two tracks whose per-strip colours are expanded to their vertices.
    CHECK(world.counts().entities == 2);
    const caliper_scene::Batch* tracks = batch(world, "world/tracks", Primitive::Lines);
    REQUIRE(tracks);
    CHECK(tracks->array(caliper_scene::Role::Colors)->count == 5);
    CHECK(tracks->array(caliper_scene::Role::BreakMask)->count == 5);
    // The flat scene: a 2-D triangle, static.
    CHECK(flat.tree().space() == caliper_scene::Space::k2D);
    REQUIRE(batch(flat, "shapes/tri", Primitive::Mesh));

    // Every scene compiles and draws.
    for (auto& kv : views) {
        std::string why;
        CHECK_MESSAGE(kv.second->view->render(64, 48, 1.0f, 0xff000000u, &why), kv.first, ": ", why);
        const std::vector<std::string>& notes = kv.second->view->notes();
        const std::string first = notes.empty() ? std::string() : notes.front();
        CHECK_MESSAGE(notes.empty(), kv.first, ": ", first);
    }

    // A recursive clear of points and meshes under world/cloud removes the cloud.
    {
        const wire::Message m = load("scene_clear");
        REQUIRE(world.apply(std::get<wire::SceneClear>(m), nullptr));
        CHECK(batch(world, "world/cloud", Primitive::Points) == nullptr);
        CHECK(batch(world, "world/tracks", Primitive::Lines) != nullptr);
    }
    // The reset that opens a new state empties the landscape, static items included.
    {
        const wire::Message m = load("scene_clear_reset");
        REQUIRE(land.apply(std::get<wire::SceneClear>(m), nullptr));
        CHECK(land.counts().batches == 0);
        CHECK(land.counts().buffers_live == 0);
    }
}

TEST_CASE("gfx/Metal scene view: the demo's messages draw the surface row's scene and match the CPU reference") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    const scene_fixtures::SurfaceRowFixture fx;
    const DemoMessages dm(fx);
    Wire w;
    const std::vector<DrawInfo> draws = draw_demo(rig, w, dm);
    SceneView& v = *rig.view;

    // The state: four entities, the box static, the rest stamped.
    CHECK(v.counts().entities == 4);
    CHECK(v.counts().batches == 4);
    CHECK(v.tree().batch_static(v.tree().find("box"), caliper_scene::Primitive::Lines));
    CHECK(v.counts().items == uint64_t(fx.kTriangles + (fx.kPathVertices - 1) + 1 + 12));
    // Every array went up once, a copy each: positions, indices and values, the strip's positions
    // and break mask, the marker and the box.
    CHECK(v.counts().copies_total == 7);
    CHECK(v.counts().buffers_created == 7);
    CHECK(v.counts().bytes_total ==
          (fx.positions.size() + fx.indices.size() + fx.values.size() + fx.path.size() + fx.keep.size() + 3 + fx.box.size()) * 4);
    CHECK(v.fitted());

    // The draw list is the surface row's: in the byte order of the paths, all opaque, with depth.
    REQUIRE(draws.size() == 4);
    CHECK(draws[0].path == "box");
    CHECK(draws[1].path == "marker");
    CHECK(draws[2].path == "path");
    CHECK(draws[3].path == "surface");
    CHECK(draws[3].p.color_mode == 1);
    CHECK(draws[3].record.count == uint32_t(fx.kTriangles * 3));
    CHECK(bool(draws[3].record.index_buffer));
    CHECK(draws[2].record.instance_count == uint32_t(fx.kPathVertices - 1));
    CHECK(draws[0].record.instance_count == 12u);
    CHECK(draws[1].p.flat_rgba == fx.kMarkerRgba);
    CHECK(draws[0].p.flat_rgba == fx.kBoxRgba);
    CHECK(bool(v.last_draws().target().depth));

    // The pixels match the CPU rasterisation of the same draws under the parity rules.
    const int S = fx.kSize;
    const std::vector<uint8_t> got = pixels(v);
    capture_png("scene-view-row.png", got, S, S);
    const Reference ref = rasterise(draws, fx);
    CHECK(ref.clipped == 0);
    const Comparison c = compare(ref, got);
    MESSAGE("scene view row: ", c.interior, " interior pixels (surface ", c.interior_by_draw.at(3), ", path ",
            c.interior_by_draw.at(2), ", marker ", c.interior_by_draw.at(1), ", box ", c.interior_by_draw.at(0), "), ",
            c.edge, " edge pixels, largest lit difference ", c.max_interior_lit, ", ", c.failed, " failed");
    CHECK_MESSAGE(c.failed == 0, c.failed, " pixels outside the parity rules; ", c.first);
    for (int d = -1; d < 4; ++d) CHECK(c.interior_by_draw.count(d) == 1);

    // The GPU time of the frame reaches the info line once it completed.
    v.poll();
    CHECK(v.cost().gpu_ms >= 0.0);
    CHECK(v.cost().frames == 1);
    const std::string info = v.info();
    CHECK_MESSAGE(info.find("4 entities  3,252 items  copied") == 0, info);
    CHECK_MESSAGE(info.find("gpu ") != std::string::npos, info);
}

TEST_CASE("gfx/Metal scene view: a late attach receives a reset and the state; clears keep static items") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    const scene_fixtures::SurfaceRowFixture fx;
    const DemoMessages dm(fx);
    Wire w;
    SceneView& v = *rig.view;
    // What the view held before: a batch of an earlier state.
    {
        wire::SceneUpdate old = update("old/cloud", CALIPER_NET_PART_POINTS, 90, false);
        const float p[6] = {0, 0, 0, 1, 1, 1};
        old.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {2, 3}, p, sizeof p));
        REQUIRE(feed(v, w.roundtrip(old)));
        REQUIRE(v.counts().batches == 1);
    }
    // The state of a new subscription: the reset, then the static box and the newest batches.
    auto state = [&](wire::Message m) {
        if (auto* u = std::get_if<wire::SceneUpdate>(&m)) u->state = true;
        return m;
    };
    for (const wire::Message& m : {dm.reset(), state(dm.box()), state(dm.surface()), state(dm.path())}) {
        std::string why;
        REQUIRE_MESSAGE(feed(v, w.roundtrip(m), &why), why);
    }
    CHECK_FALSE(v.tree().find("old/cloud"));
    CHECK(v.counts().batches == 3);
    CHECK(v.counts().buffers_live == 6);   // the box 1, the surface 3, the path 2

    // A clear of the whole scene that is not static keeps the static box.
    wire::SceneClear all;
    all.sub = 1;
    all.view = "landscape";
    all.seq = 9;
    all.path = "";
    all.recursive = true;
    REQUIRE(feed(v, w.roundtrip(all)));
    CHECK(v.counts().batches == 1);
    CHECK(v.tree().batch(v.tree().find("box"), caliper_scene::Primitive::Lines) != nullptr);
    // A stamped batch over a static one wins, and its clear brings the static one back.
    {
        wire::Message box2 = dm.box();
        std::get<wire::SceneUpdate>(box2).is_static = false;
        std::get<wire::SceneUpdate>(box2).params.set("width", Json(7.0));
        REQUIRE(feed(v, w.roundtrip(box2)));
        CHECK(v.tree().batch(v.tree().find("box"), caliper_scene::Primitive::Lines)->width_px == 7.0f);
        wire::SceneClear c = all;
        c.path = "box";
        c.recursive = false;
        REQUIRE(feed(v, w.roundtrip(c)));
        CHECK(v.tree().batch(v.tree().find("box"), caliper_scene::Primitive::Lines)->width_px == fx.kBoxWidth);
    }
    // A static clear removes it.
    all.is_static = true;
    REQUIRE(feed(v, w.roundtrip(all)));
    CHECK(v.counts().batches == 0);
    CHECK(v.counts().buffers_live == 0);
    std::string why;
    CHECK_MESSAGE(v.render(32, 32, 1.0f, 0xff000000u, &why), why);
}

TEST_CASE("gfx/Metal scene view: an update of one batch reuses its buffers and redraws; nothing changes, nothing draws") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    const scene_fixtures::SurfaceRowFixture fx;
    DemoMessages dm(fx);
    Wire w;
    draw_demo(rig, w, dm);
    SceneView& v = *rig.view;
    const std::vector<uint8_t> before = pixels(v);
    const uint64_t frames = v.cost().frames, created = v.counts().buffers_created;
    const uint64_t reused = v.counts().buffers_reused, bytes = v.counts().bytes_total;
    const uint64_t revision = v.tree().revision();

    // Nothing changed: no compile, no submit.
    std::string why;
    REQUIRE(v.render(fx.kSize, fx.kSize, 1.0f, fx.kClear, &why));
    CHECK(v.cost().frames == frames);

    // A new surface of the same size (the values shift, as a refit recolours it): its three
    // buffers are written in place, nothing is created, and the view draws again.
    for (float& x : dm.values) x = fx.vmin + (fx.vmax - x);
    v.begin_round();
    REQUIRE_MESSAGE(feed(v, w.roundtrip(dm.surface(6)), &why), why);
    CHECK(v.counts().buffers_created == created);
    CHECK(v.counts().buffers_reused == reused + 3);
    CHECK(v.counts().bytes_last == (fx.positions.size() + fx.indices.size() + fx.values.size()) * 4);
    CHECK(v.counts().bytes_total == bytes + v.counts().bytes_last);
    CHECK(v.tree().revision() != revision);
    REQUIRE(v.render(fx.kSize, fx.kSize, 1.0f, fx.kClear, &why));
    CHECK(v.cost().frames == frames + 1);
    const std::vector<uint8_t> after = pixels(v);
    capture_png("scene-view-row-update.png", after, fx.kSize, fx.kSize);
    size_t differ = 0;
    for (size_t i = 0; i < after.size(); i += 4) differ += std::memcmp(&after[i], &before[i], 4) != 0;
    CHECK(differ > 10000);   // the surface took its new colours

    // The camera moved: one more frame, no upload.
    v.orbit(10.0f, 0.0f);
    REQUIRE(v.render(fx.kSize, fx.kSize, 1.0f, fx.kClear, &why));
    CHECK(v.cost().frames == frames + 2);
    CHECK(v.counts().buffers_reused == reused + 3);

    // A larger marker batch gets a new buffer only when it outgrows the capacity (4 KiB here).
    wire::SceneUpdate big = std::get<wire::SceneUpdate>(dm.marker());
    std::vector<float> many(3 * 400, 0.0f);
    big.arrays[0] = wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {400, 3}, many.data(), many.size() * 4);
    REQUIRE(feed(v, w.roundtrip(big)));
    CHECK(v.counts().buffers_created == created + 1);
}

TEST_CASE("gfx/Metal scene view: timing — the demo scene's cost per frame and of applying one mesh update") {
    const char* on = std::getenv("CALIPER_GFX_SCENE_VIEW_BENCH");
    if (!on || std::string(on) != "1") return;   // opt-in, like the other timing rows
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    const scene_fixtures::SurfaceRowFixture fx;
    DemoMessages dm(fx);
    Wire w;
    draw_demo(rig, w, dm);
    SceneView& v = *rig.view;
    constexpr int kRuns = 50;
    std::vector<double> compile, submit, gpu, apply_ms;
    std::string why;
    for (int i = 0; i < kRuns; ++i) {
        v.orbit(1.0f, 0.0f);   // a moved camera: the whole frame again, no upload
        REQUIRE(v.render(fx.kSize, fx.kSize, 1.0f, fx.kClear, &why));
        compile.push_back(v.cost().compile_ms);
        submit.push_back(v.cost().submit_ms);
        REQUIRE(v.backend().wait(v.backend().last_use(v.color())));
        v.poll();
        gpu.push_back(v.cost().gpu_ms);
    }
    const wire::Message m = w.roundtrip(dm.surface(7));
    for (int i = 0; i < kRuns; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        REQUIRE(feed(v, m));
        apply_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        REQUIRE(v.render(fx.kSize, fx.kSize, 1.0f, fx.kClear, &why));
        REQUIRE(v.backend().wait(v.backend().last_use(v.color())));
    }
    auto median = [](std::vector<double> x) {
        std::sort(x.begin(), x.end());
        return x[x.size() / 2];
    };
    MESSAGE("[scene-view-bench] demo scene at 512 x 512, median of ", kRuns, ": compile ", median(compile),
            " ms, submit ", median(submit), " ms, gpu ", median(gpu), " ms; one mesh update of 1,681 vertices and 3,200 "
            "triangles applied in ", median(apply_ms), " ms");
}
#endif
