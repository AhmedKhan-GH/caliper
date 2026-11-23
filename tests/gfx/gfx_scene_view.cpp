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

#include "../scene/primitive_rows.h"
#include "../scene/surface_row_fixture.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "live/live_scene.h"
#include "live/scene_piece_form.h"
#if defined(CALIPER_HAVE_SCENE_PIECES)
#include "draw/scene_pieces.h"
#endif
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
#include <set>
#include <string>
#include <tuple>
#include <vector>

caliper_host::HostRenderer* gfx_metal_renderer();

namespace {

namespace wire = caliper_net::wire;
namespace fs = std::filesystem;
using caliper_net::Json;
using caliper_live_views::SceneView;
using caliper_live_views::SceneShapeCounts;
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
    bool init(const char* space = CALIPER_NET_SPACE_3D, const char* up = "+z", const char* units = "",
              caliper_live_views::ShapeDraw shapes = caliper_live_views::ShapeDraw::builtin) {
        gfx = view_layer();
        if (!gfx) return false;
        view = std::make_unique<SceneView>(*gfx, sh::kMetalPrograms);
        // These rows check the scene core's own draw list, its pick pass and its counters against
        // their CPU references, thus they draw every shape with the built-in draw; the rows of
        // "Shapes Drawn by Library Pieces" at the end of this file draw the same fixtures with the
        // pieces and compare them with the built-in.
        view->set_shape_draw(shapes);
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
    // defaults, "flat" and "overlay" in 2-D (their positions are (N, 2)).
    std::map<std::string, std::unique_ptr<ViewRig>> views;
    auto view_of = [&](const std::string& name) -> SceneView* {
        auto it = views.find(name);
        if (it == views.end()) {
            auto rig = std::make_unique<ViewRig>();
            const bool ok = (name == "flat" || name == "overlay") ? rig->init(CALIPER_NET_SPACE_2D, "-y", "px")
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

    // Every frame but the encoded image, cameras before the images under them.
    const char* updates[] = {"scene_update_points",          "scene_update_points_stamped", "scene_update_points_color",
                             "scene_update_points_values",   "scene_update_lines_segments", "scene_update_lines_strip",
                             "scene_update_lines_strips",    "scene_update_mesh",           "scene_update_mesh_normals",
                             "scene_update_mesh_colors",     "scene_update_camera",         "scene_update_camera_rub",
                             "scene_update_arrows",          "scene_update_boxes",          "scene_update_boxes_2d",
                             "scene_update_image_color",     "scene_update_image_depth",    "scene_update_image_depth_f32",
                             "scene_update_image_segmentation", "scene_update_image_segmentation_u16",
                             "scene_update_lines_edges",     "scene_update_mesh_textured",  "scene_update_points_labels",
                             "scene_update_points_future"};
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
    for (const char* name : {"scene_transforms_static", "scene_transforms", "scene_transforms_stamps"}) {
        const wire::Message m = load(name);
        const auto& t = std::get<wire::SceneTransforms>(m);
        std::string why;
        CHECK_MESSAGE(view_of(t.view)->apply(t, &why), name, ": ", why);
    }
    for (const char* name : {"scene_classes", "scene_classes_stamped"}) {
        const wire::Message m = load(name);
        const auto& c = std::get<wire::SceneClasses>(m);
        std::string why;
        CHECK_MESSAGE(view_of(c.view)->apply(c, &why), name, ": ", why);
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
    // The world: a cloud and two tracks whose per-strip colours are expanded to their vertices, and
    // eight more entities with a batch (below).
    CHECK(world.counts().entities == 10);
    const caliper_scene::Batch* tracks = batch(world, "world/tracks", Primitive::Lines);
    REQUIRE(tracks);
    CHECK(tracks->array(caliper_scene::Role::Colors)->count == 5);
    CHECK(tracks->array(caliper_scene::Role::BreakMask)->count == 5);
    // The flat scene: a 2-D triangle, static.
    CHECK(flat.tree().space() == caliper_scene::Space::k2D);
    REQUIRE(batch(flat, "shapes/tri", Primitive::Mesh));
    // The other parts of the world: two cameras, the images under the first, arrows, a static box,
    // index-pair edges, a textured mesh, labelled points; the transforms and the class table.
    const caliper_scene::SceneTree& wt = world.tree();
    CHECK(wt.camera(wt.find("rig/left")));
    REQUIRE(wt.camera(wt.find("sim/eye")));
    CHECK(wt.camera(wt.find("sim/eye"))->convention == caliper_scene::Convention::Rub);
    CHECK(batch(world, "rig/left/rgb", Primitive::Image)->image_form == caliper_scene::ImageForm::Color);
    CHECK(batch(world, "rig/left/depth", Primitive::Image)->image_form == caliper_scene::ImageForm::Depth);
    CHECK(batch(world, "rig/left/depth", Primitive::Image)->cloud);   // the model's default under a camera
    CHECK(batch(world, "rig/left/classes", Primitive::Image)->opacity == 0.5f);
    REQUIRE(batch(world, "field", Primitive::Arrows));
    CHECK(wt.batch_static(wt.find("arm/link0"), Primitive::Boxes));
    REQUIRE(batch(world, "graph/edges", Primitive::Lines));
    CHECK(batch(world, "graph/edges", Primitive::Lines)->line_mode == caliper_scene::LineMode::Edges);
    CHECK(batch(world, "graph/edges", Primitive::Lines)->per_edge);
    CHECK(bool(batch(world, "floor", Primitive::Mesh)->texture));
    CHECK(batch(world, "graph/nodes", Primitive::Points)->labels.size() == 3);
    CHECK(wt.transform(wt.find("rig/left")));
    CHECK(wt.transform(wt.find("arm/link0/link1")));
    CHECK(wt.transform(wt.find("b")));
    // The stamped table, empty, wins over the static one: latest-at.
    CHECK(wt.classes().empty());
    SceneView& overlay = *view_of("overlay");
    CHECK(batch(overlay, "rig/left/cars", Primitive::Boxes)->fill);
    CHECK(batch(overlay, "elevation", Primitive::Image)->image_form == caliper_scene::ImageForm::Depth);
    CHECK(batch(overlay, "labels", Primitive::Image)->opacity == 0.75f);
    CHECK(overlay.counts().textures_live == 2);

    // Every scene compiles and draws.
    for (auto& kv : views) {
        std::string why;
        CHECK_MESSAGE(kv.second->view->render(64, 48, 1.0f, 0xff000000u, &why), kv.first, ": ", why);
        const std::vector<std::string>& notes = kv.second->view->notes();
        const std::string first = notes.empty() ? std::string() : notes.front();
        CHECK_MESSAGE(notes.empty(), kv.first, ": ", first);
    }

    // The encoded image is the one form a scene view does not draw yet: it refuses and says so.
    {
        const wire::Message m = load("scene_update_image_encoded");
        std::string why;
        CHECK_FALSE(world.apply(std::get<wire::SceneUpdate>(m), &why));
        CHECK_MESSAGE(why.find("encoded images are not drawn yet") != std::string::npos, why);
    }
    // A clear of the transform and the boxes under arm removes the stamped transforms and keeps the
    // static box.
    {
        const wire::Message m = load("scene_clear_stamped");
        REQUIRE(world.apply(std::get<wire::SceneClear>(m), nullptr));
        CHECK_FALSE(wt.transform(wt.find("arm/link0/link1")));
        CHECK(batch(world, "arm/link0", Primitive::Boxes) != nullptr);
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

// The rows of every primitive (tests/scene/primitive_fixtures.h): each fixture's messages applied to
// a scene view on Metal, drawn at 256 x 256, read back and compared with the CPU reference of
// tests/scene/draw_reference.h under web geometry's parity rules. The demo's scene runs through the
// same reference first, as a check of the reference against the surface row's own.
TEST_CASE("gfx/Metal scene view: every primitive drawn from protocol 3.2 messages matches the CPU reference") {
    ViewRig probe;
    if (!probe.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    probe.view.reset();
    std::vector<primitive_fixtures::Fixture> fixtures = primitive_fixtures::all();
    const scene_fixtures::SurfaceRowFixture fx;
    const DemoMessages dm(fx);
    {
        primitive_fixtures::Fixture demo;
        demo.name = "demo";
        demo.info = {CALIPER_NET_SPACE_3D, "+z", ""};
        demo.messages = {dm.reset(), dm.box(), dm.surface(), dm.path(), dm.marker()};
        fixtures.insert(fixtures.begin(), demo);
    }
    for (const primitive_fixtures::Fixture& f : fixtures) {
        CAPTURE(f.name);
        ViewRig rig;
        REQUIRE(rig.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str()));
        SceneView& v = *rig.view;
        primitive_rows::Row row;
        row.fixture = f;
        REQUIRE_MESSAGE(row.draw(v), row.error);
        CHECK_MESSAGE(v.notes().empty(), f.name, ": ", (v.notes().empty() ? std::string() : v.notes().back()));
        const std::vector<uint8_t> got = pixels(v);
        capture_png(("scene-" + f.name + ".png").c_str(), got, int(v.width()), int(v.height()));
        const primitive_rows::Row::Result r = row.compare(v, got);
        MESSAGE(r.sentence, "; ", r.draws, " draws, ", r.triangles, " triangles, ", r.c.interior, " interior and ",
                r.c.edge, " edge pixels, ", r.c.exact, " exact");
        CHECK_MESSAGE(r.problems.empty(), (r.problems.empty() ? std::string() : r.problems.front()));
        CHECK_MESSAGE(r.clipped == 0, r.clipped, " triangles would need clipping");
        CHECK_MESSAGE(r.c.failed == 0, r.c.failed, " pixels outside the parity rules; ", r.c.first);
        // Every draw shows in the picture.
        for (uint32_t d = 0; d < r.draws; ++d) CHECK_MESSAGE(r.covered.at(d) > 0, "draw ", d, " of ", f.name, " covers no pixel");

        if (f.name == "boxes") {
            // The chain's poses compose down the tree, M_arm * M_link0 * M_link1, in double.
            using caliper_scene::Mat4;
            auto m = [](const std::vector<float>& c) {
                Mat4 r;
                for (int k = 0; k < 16; ++k) r.at(k % 4, k / 4) = c[size_t(k)];
                return r;
            };
            std::vector<float> tilt = {1, 0, 0, 0, 0, std::cos(0.6f), std::sin(0.6f), 0, 0, -std::sin(0.6f), std::cos(0.6f), 0, 0, 0, 0, 1};
            std::vector<float> next = tilt;
            next[14] = 0.4f;
            const Mat4 want = m(primitive_fixtures::translate(0.9f, -0.6f, 0)) * m(tilt) * m(next);
            const Mat4& got_w = v.tree().world(v.tree().find("arm/link0/link1"));
            for (int k = 0; k < 16; ++k) CHECK(std::fabs(got_w.at(k % 4, k / 4) - want.at(k % 4, k / 4)) < 1e-12);
            // The crates' labels, anchored at their centres, nearest first.
            CHECK(v.labels().size() == 3);
            // Class colours: 1 and 2 from the table, 7 from the palette.
            CHECK(v.tree().class_label(1) == "car");
        }
        if (f.name == "through") CHECK(v.camera_text().find("through rig/left") != std::string::npos);
        if (f.name == "images") {
            CHECK(v.counts().textures_live == 4);
            CHECK(v.counts().items == 4 + 2 + 2);
        }
    }
}

namespace {

// What decode_pick would find for a pointer at the centre of pixel (px, py) in the reference's pick
// identities: the texel with an identity nearest the pointer within the radius and the view. robust:
// the hit pixel and its neighbours show one identity with no near tie in depth, and no other
// identity lies within 1.5 pixels of the nearest distance, so that a GPU's rounding cannot change it.
struct PickExpect {
    bool hit = false, robust = false;
    bool solid = false;   // the hit pixel and its eight neighbours show the same identity
    int x = 0, y = 0;
    uint32_t id = 0;
};
PickExpect expect_pick(const draw_reference::Reference& ref, int px, int py, int radius = 8) {
    PickExpect e;
    const double cx = px + 0.5, cy = py + 0.5, r2 = double(radius) * radius;
    double best = 1e300;
    for (int j = py - radius; j < py + radius; ++j)
        for (int i = px - radius; i < px + radius; ++i) {
            if (i < 0 || j < 0 || i >= ref.W || j >= ref.H) continue;
            const uint32_t id = ref.id[size_t(j) * ref.W + i];
            if (!id) continue;
            const double d2 = (i + 0.5 - cx) * (i + 0.5 - cx) + (j + 0.5 - cy) * (j + 0.5 - cy);
            if (d2 > r2 || d2 >= best) continue;
            best = d2;
            e.hit = true;
            e.x = i;
            e.y = j;
            e.id = id;
        }
    const double limit = e.hit ? std::pow(std::sqrt(best) + 1.5, 2.0) : std::pow(radius + 2.0, 2.0);
    e.robust = true;
    for (int j = py - radius - 2; j < py + radius + 2 && e.robust; ++j)
        for (int i = px - radius - 2; i < px + radius + 2 && e.robust; ++i) {
            if (i < 0 || j < 0 || i >= ref.W || j >= ref.H) continue;
            const uint32_t id = ref.id[size_t(j) * ref.W + i];
            const double d2 = (i + 0.5 - cx) * (i + 0.5 - cx) + (j + 0.5 - cy) * (j + 0.5 - cy);
            if (id && id != e.id && d2 <= limit) e.robust = false;
        }
    if (e.hit) {
        e.solid = true;
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                const int i = std::clamp(e.x + di, 0, ref.W - 1), j = std::clamp(e.y + dj, 0, ref.H - 1);
                const size_t k = size_t(j) * ref.W + i;
                if (ref.contested[k] || (ref.id[k] != e.id && ref.id[k] != 0)) e.robust = false;
                if (ref.id[k] != e.id) e.solid = false;
            }
    }
    return e;
}

// Waits for the view's pick on Metal (the test may block; a view never does) and takes it.
caliper_scene::PickPass::Result take_pick(Backend& gfx, SceneView& v, caliper_scene::Hit* hit) {
    caliper_scene::PickPass::Result r = caliper_scene::PickPass::Result::Pending;
    for (int tries = 0; tries < 100 && r == caliper_scene::PickPass::Result::Pending; ++tries) {
        gfx.wait(v.pick_pass()->last_use());
        gfx.poll();
        r = v.poll_pick(hit);
    }
    return r;
}

}  // namespace

// Picks on Metal through the scene view's pick pass, against the identities of a CPU rasterisation
// of the same compiled draws in pick mode (every draw opaque, writing depth where it tests it): at an
// interior pixel of every draw of every primitive fixture, the exact entity, primitive, part and
// item; at pointers whose nearest hit lies on the edge of the 16 x 16 window or whose window passes
// the view's edge; a miss; and a result dropped because the view drew another frame first. The
// world position from the depth is compared with the reference's depth unprojected in double.
TEST_CASE("gfx/Metal scene view: the pick pass finds every primitive, through transforms and cameras, and drops stale results") {
    ViewRig probe_rig;
    if (!probe_rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    probe_rig.view.reset();
    size_t probes = 0, solid_probes = 0, edge_probes = 0, border_probes = 0, misses = 0, positions = 0;
    size_t through_transforms = 0;
    std::set<std::string> kinds;   // primitive / part / overlay of every draw picked
    double worst_position = 0;   // as a share of the scene's bounding radius
    for (const primitive_fixtures::Fixture& f : primitive_fixtures::all()) {
        CAPTURE(f.name);
        ViewRig rig;
        REQUIRE(rig.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str()));
        SceneView& v = *rig.view;
        primitive_rows::Row row;
        row.fixture = f;
        REQUIRE_MESSAGE(row.draw(v), row.error);
        draw_reference::Rasteriser rr(v.last_draws(), v.last_map(), v.tree(), row.parts, int(primitive_fixtures::kSize),
                                      int(primitive_fixtures::kSize));
        rr.pick = true;
        rr.run();
        REQUIRE(rr.problems.empty());
        const draw_reference::Reference& ref = rr.ref;
        const caliper_scene::DrawMap& map = v.last_map();
        const double radius = v.tree().scene_bounds().valid ? v.tree().scene_bounds().radius() : 1.0;
        const int S = int(primitive_fixtures::kSize);

        auto check_hit = [&](const PickExpect& e, int px, int py, bool exact_texel) {
            caliper_scene::Hit hit;
            std::string why;
            REQUIRE_MESSAGE(v.request_pick(px + 0.5f, py + 0.5f, &why), why);
            const caliper_scene::PickPass::Result r = take_pick(*rig.gfx, v, &hit);
            if (!e.hit) {
                CHECK_MESSAGE(r == caliper_scene::PickPass::Result::Miss, "a miss at (", px, ", ", py, ")");
                return;
            }
            REQUIRE_MESSAGE(r == caliper_scene::PickPass::Result::Hit, "a hit at (", px, ", ", py, ")");
            const int draw = draw_reference::Reference::draw_of(e.id);
            const uint32_t item = (e.id & 0xffffffu) - 1;
            const caliper_scene::DrawMapEntry& me = map.entries.at(size_t(draw));
            CHECK_MESSAGE(hit.entity == me.entity, "at (", px, ", ", py, "): ", v.tree().path(hit.entity), " for ", v.tree().path(me.entity));
            CHECK(hit.primitive == me.primitive);
            CHECK(hit.part == me.part);
            CHECK(hit.overlay == me.overlay);
            const int tx = e.x - (int(std::floor(px + 0.5f)) - 8), ty = e.y - (int(std::floor(py + 0.5f)) - 8);
            if (exact_texel) {
                CHECK(hit.texel[0] == tx);
                CHECK(hit.texel[1] == ty);
            } else {
                CHECK(std::abs(hit.texel[0] - tx) <= 1);
                CHECK(std::abs(hit.texel[1] - ty) <= 1);
            }
            if (me.primitive == caliper_scene::Primitive::Mesh && me.part == caliper_scene::DrawPart::Batch) {
                const auto& tri = rr.mesh_vertices.at(draw).at(item);   // a vertex of the triangle under the pixel
                CHECK((hit.item == tri[0] || hit.item == tri[1] || hit.item == tri[2]));
            } else if (me.primitive == caliper_scene::Primitive::Image && me.part == caliper_scene::DrawPart::Batch) {
                CHECK(hit.item == 0);   // an image writes no item; its pixel comes from its quad
                // The image pixel under the hit, from the reference's quad: its UV at the pixel centre.
                if (exact_texel) {
                    REQUIRE(hit.has_pixel);
                    bool found = false;
                    for (const uint32_t ti : ref.tris_of.at(e.id)) {
                        const draw_reference::Tri& t = ref.tris[ti];
                        double q[3], z, mw;
                        bool inside;
                        if (!draw_reference::weights(t.v, e.x + 0.5, e.y + 0.5, q, &z, &inside, &mw) || !inside) continue;
                        const double u = q[0] * t.v[0].uv[0] + q[1] * t.v[1].uv[0] + q[2] * t.v[2].uv[0];
                        const double w = q[0] * t.v[0].uv[1] + q[1] * t.v[1].uv[1] + q[2] * t.v[2].uv[1];
                        const uint32_t px_u = std::min(uint32_t(u * me.image_size[0]), me.image_size[0] - 1);
                        const uint32_t px_v = std::min(uint32_t(w * me.image_size[1]), me.image_size[1] - 1);
                        CHECK_MESSAGE(std::abs(int(hit.pixel[0]) - int(px_u)) <= (item ? 0 : 1), "image column");
                        CHECK_MESSAGE(std::abs(int(hit.pixel[1]) - int(px_v)) <= (item ? 0 : 1), "image row");
                        found = true;
                        break;
                    }
                    CHECK(found);
                }
            } else {
                CHECK_MESSAGE(hit.item == item, "at (", px, ", ", py, ") in ", v.tree().path(me.entity));
            }
            // The world position from the depth, in 3-D outside a pinhole's flat pixel space.
            const bool three_d = v.tree().space() == caliper_scene::Space::k3D && !me.overlay;
            CHECK(hit.has_position == three_d);
            if (three_d && hit.has_position && exact_texel) {
                const double X = 2.0 * (e.x + 0.5) / S - 1.0, Y = 1.0 - 2.0 * (e.y + 0.5) / S;
                double q[4];
                caliper_scene::transform_h(map.inverse_view_proj, {X, Y, ref.depth_d[size_t(e.y) * S + e.x]}, q);
                const caliper_scene::Vec3 want{q[0] / q[3], q[1] / q[3], q[2] / q[3]};
                const double off = caliper_scene::length(hit.position - want) / radius;
                worst_position = std::max(worst_position, off);
                CHECK_MESSAGE(off < 1e-4, "the position at (", px, ", ", py, ") is off by ", off, " of the radius");
                ++positions;
                // Through the entity transforms: the hit lies on the entity's own box, in its own frame,
                // within the edges' 3 pixels on screen (0.04 scene units at that distance).
                if (f.name == "boxes" && v.tree().path(hit.entity) == "arm/link0/link1") {
                    caliper_scene::Mat4 inv;
                    REQUIRE(caliper_scene::inverse(v.tree().world(hit.entity), &inv));
                    const caliper_scene::Vec3 l = caliper_scene::transform_point(inv, hit.position);
                    const double k = 0.04;
                    CHECK_MESSAGE((std::fabs(l.x) <= 0.05 + k && std::fabs(l.y) <= 0.05 + k && l.z >= -k && l.z <= 0.4 + k),
                                  "link1's local hit ", l.x, " ", l.y, " ", l.z);
                    ++through_transforms;
                }
            }
        };

        // A pixel of every draw nearest its centroid, inside the draw where it is wider than a pixel
        // (texel (8, 8) exactly), else on it (a line one or two pixels wide: within a texel).
        for (size_t d = 0; d < map.entries.size(); ++d) {
            double sx = 0, sy = 0;
            size_t n = 0;
            for (int j = 0; j < S; ++j)
                for (int i = 0; i < S; ++i) {
                    const uint32_t id = ref.id[size_t(j) * S + i];
                    if (id && draw_reference::Reference::draw_of(id) == int(d)) sx += i, sy += j, ++n;
                }
            if (n == 0) {   // covered everywhere, as a pinhole's translucent image covers the 3-D content
                MESSAGE("draw ", d, " of ", f.name, " (", v.tree().path(map.entries[d].entity), ") is covered everywhere");
                continue;
            }
            int bx = -1, by = -1;
            double bd = 1e300;
            bool solid = false;
            for (int j = 2; j < S - 2; ++j)
                for (int i = 2; i < S - 2; ++i) {
                    const uint32_t id = ref.id[size_t(j) * S + i];
                    if (!id || draw_reference::Reference::draw_of(id) != int(d)) continue;
                    const PickExpect e = expect_pick(ref, i, j);
                    if (!e.robust || e.x != i || e.y != j || (solid && !e.solid)) continue;
                    const double dd = (i - sx / n) * (i - sx / n) + (j - sy / n) * (j - sy / n);
                    if ((e.solid && !solid) || dd < bd) bd = dd, bx = i, by = j, solid = e.solid;
                }
            if (bx < 0) {   // a draw seen edge-on or under others everywhere: picked in another fixture
                MESSAGE("draw ", d, " of ", f.name, " (", v.tree().path(map.entries[d].entity), ") has no pixel to pick robustly");
                continue;
            }
            check_hit(expect_pick(ref, bx, by), bx, by, solid);
            ++probes;
            solid_probes += solid ? 1 : 0;
            const caliper_scene::DrawMapEntry& me = map.entries[d];
            kinds.insert(std::string(caliper_scene::primitive_name(me.primitive)) + "/" + std::to_string(int(me.part)) +
                         (me.overlay ? "/overlay" : ""));
        }
        // Pointers whose nearest hit lies on the window's edge, and windows past the view's edge.
        int edges_here = 0, borders_here = 0;
        for (int j = 0; j < S && (edges_here < 2 || borders_here < 2); j += 3)
            for (int i = 0; i < S && (edges_here < 2 || borders_here < 2); i += 3) {
                const PickExpect e = expect_pick(ref, i, j);
                if (!e.robust || !e.hit) continue;
                const int tx = e.x - (i - 8), ty = e.y - (j - 8);
                const bool on_edge = tx == 0 || tx == 15 || ty == 0 || ty == 15;
                const bool past_view = i < 8 || j < 8 || i >= S - 8 || j >= S - 8;
                if (on_edge && edges_here < 2) {
                    check_hit(e, i, j, false);
                    ++edges_here;
                    ++edge_probes;
                } else if (past_view && borders_here < 2) {
                    check_hit(e, i, j, false);
                    ++borders_here;
                    ++border_probes;
                }
            }
        // A miss: nothing within the radius and two pixels more.
        bool missed = false;
        for (int j = 0; j < S && !missed; j += 5)
            for (int i = 0; i < S && !missed; i += 5) {
                const PickExpect e = expect_pick(ref, i, j);
                if (e.hit || !e.robust) continue;
                check_hit(e, i, j, true);
                missed = true;
                ++misses;
            }
        CHECK_MESSAGE(missed, f.name, " has no empty place");
        // A result that arrives after the view drew another frame is dropped.
        {
            std::string why;
            REQUIRE(v.request_pick(128.5f, 128.5f, &why));
            v.orbit(2.0f, 0.0f);
            REQUIRE_MESSAGE(v.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why), why);
            caliper_scene::Hit hit;
            CHECK(take_pick(*rig.gfx, v, &hit) == caliper_scene::PickPass::Result::Dropped);
        }
        // Nothing is submitted while nobody asks: the view's own frames are the only ones.
        const uint64_t submits = v.pick_pass()->submits();
        for (int k = 0; k < 5; ++k) {
            std::string why;
            REQUIRE(v.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why));
            caliper_scene::Hit hit;
            CHECK(v.poll_pick(&hit) == caliper_scene::PickPass::Result::None);
        }
        CHECK(v.pick_pass()->submits() == submits);
    }
    MESSAGE("picks: ", probes, " pixels, one of every draw (", solid_probes, " inside a draw wider than a pixel), ", edge_probes,
            " on the window's edge, ", border_probes,
            " with the window past the view's edge, ", misses, " misses; ", positions,
            " world positions within ", worst_position, " of the scene's radius of the reference's");
    CHECK(edge_probes > 0);
    CHECK(border_probes > 0);
    CHECK(through_transforms > 0);
    // Every primitive and part was picked: points, lines, arrows, wire and filled boxes, meshes, images
    // in 2-D and flat through a pinhole, depth clouds, frustums, and 2-D drawing through a pinhole.
    std::string all;
    for (const std::string& k : kinds) all += k + " ";
    MESSAGE("kinds picked: ", all);
    for (const char* k : {"points/0", "lines/0", "arrows/0", "boxes/0", "boxes/1", "mesh/0", "image/0", "image/2",
                          "lines/3", "image/0/overlay", "points/0/overlay", "boxes/0/overlay", "arrows/0/overlay"})
        CHECK_MESSAGE(kinds.count(k) == 1, "no pick of ", k);
}

// Rectangle and lasso selections project the CPU copies of the points and the boxes' centres; a
// selection dims the picture towards the background, as the points view dims, and draws the selected
// items over it in their own colours; clearing it restores the picture byte for byte.
TEST_CASE("gfx/Metal scene view: a selection picks points and boxes by projection, dims the rest and draws them over it") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    SceneView& v = *rig.view;
    primitive_rows::Row row;
    row.fixture = primitive_fixtures::boxes();
    REQUIRE_MESSAGE(row.draw(v), row.error);
    const std::vector<uint8_t> plain = pixels(v);
    const float S = float(primitive_fixtures::kSize);
    // A rectangle over the whole picture: every box, by its centre.
    const std::string all = v.select_polygon({0, 0, S, 0, S, S, 0, S});
    caliper_net::Json d;
    REQUIRE(caliper_net::Json::parse(all.data(), all.size(), &d, nullptr));
    const caliper_net::Json& ents = *d.find("entities");
    std::vector<std::string> paths;
    size_t items = 0;
    for (const caliper_net::Json& e : ents.items()) {
        paths.push_back(e.find("entity")->as_string());
        CHECK(e.find("primitive")->as_string() == "boxes");
        CHECK(e.find("indices")->size() == e.find("row_ids")->size());
        items += e.find("indices")->size();
    }
    CHECK(paths == std::vector<std::string>{"arm/link0", "arm/link0/link1", "crates", "zone"});
    CHECK(items == 6);
    CHECK(d.find("space") == nullptr);   // no batch maps an item space
    std::string why;
    REQUIRE(v.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why));
    CHECK(v.highlighted() == 6);
    const std::vector<uint8_t> lit = pixels(v);
    capture_png("scene-boxes-selected.png", lit, int(S), int(S));
    // The background stays the background (0.28 of it over 0.72 of it), the boxes keep their colours
    // where they are drawn over the dim, and the dim shows where only the translucent faces were.
    size_t same = 0, changed = 0, red = 0;
    for (size_t k = 0; k < plain.size(); k += 4) {
        uint32_t a, b;
        std::memcpy(&a, &plain[k], 4);
        std::memcpy(&b, &lit[k], 4);
        if (a == b) ++same;
        else ++changed;
        if (b == primitive_fixtures::RGBA(230, 60, 50)) ++red;   // class 1's colour, the first crate's edges
    }
    MESSAGE("selection: ", same, " pixels unchanged, ", changed, " dimmed or redrawn, ", red, " in class 1's colour");
    CHECK(red > 50);
    CHECK(changed > 0);
    // A smaller rectangle around the first crate's centre: that crate alone.
    caliper_scene::CameraMatrices cm = caliper_scene::camera_matrices(v.effective_camera(), v.tree(), v.width(), v.height());
    double h[4];
    caliper_scene::transform_h(cm.proj * cm.view * v.tree().world(v.tree().find("crates")), {-0.7, 0, 0.3}, h);
    const float cx = float((h[0] / h[3] + 1) * 0.5 * S), cy = float((1 - h[1] / h[3]) * 0.5 * S);
    const std::string one = v.select_polygon({cx - 4, cy - 4, cx + 4, cy - 4, cx + 4, cy + 4, cx - 4, cy + 4});
    REQUIRE(caliper_net::Json::parse(one.data(), one.size(), &d, nullptr));
    REQUIRE(d.find("entities")->size() == 1);
    CHECK(d.find("entities")->at(0).find("entity")->as_string() == "crates");
    CHECK(d.find("entities")->at(0).find("indices")->size() == 1);
    CHECK(d.find("entities")->at(0).find("indices")->at(0).as_double() == 0.0);
    REQUIRE(v.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why));
    CHECK(v.highlighted() == 1);
    capture_png("scene-boxes-one-selected.png", pixels(v), int(S), int(S));
    // Nothing inside: an empty detail, and no highlight.
    CHECK(v.select_polygon({1, 1, 2, 1, 2, 2}) == "{}");
    REQUIRE(v.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why));
    CHECK(v.highlighted() == 0);
    CHECK(pixels(v) == plain);
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

TEST_CASE("gfx/Metal scene view: row ids follow the payload axes, and producer values that overflow keep the index") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    SceneView& v = *rig.view;
    Wire w;
    const float p[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    uint64_t seq = 1;
    auto cloud = [&](int64_t start, int64_t step, int64_t count, int64_t size) {
        wire::SceneUpdate u = update("cloud", CALIPER_NET_PART_POINTS, ++seq, false);
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {3, 3}, p, sizeof p));
        Json ax = Json::object();
        ax.set("dim", "point");
        ax.set("start", start);
        ax.set("step", step);
        ax.set("count", count);
        ax.set("size", size);
        Json axes = Json::array();
        axes.push(ax);
        u.axes = axes;
        std::string why;
        REQUIRE_MESSAGE(feed(v, w.roundtrip(u), &why), why);
    };
    auto id = [&](int64_t i) { return v.row_id("cloud", Primitive::Points, i); };
    // Three of 30 source rows from row 4 in steps of 2: index i is source row 4 + 2 × floor(10 i).
    cloud(4, 2, 30, 3);
    CHECK(id(0) == 4);
    CHECK(id(1) == 24);
    CHECK(id(2) == 44);
    CHECK(id(-1) == -1);
    // index × count, or the row times step, past int64_t: the index stays (each was undefined).
    const int64_t big = INT64_MAX;
    cloud(0, 1, big, 3);
    CHECK(id(0) == 0);
    CHECK(id(1) == big / 3);
    CHECK(id(2) == 2);
    cloud(big - 1, 2, 3, 3);
    CHECK(id(0) == big - 1);
    CHECK(id(1) == 1);
    cloud(0, big, 3, 3);
    CHECK(id(1) == big);
    CHECK(id(2) == 2);
    // A negative start or count, a step below 1 or no size map nothing.
    const int64_t odd[5][4] = {{-1, 1, 3, 3}, {0, 0, 3, 3}, {0, -2, 3, 3}, {0, 1, -3, 3}, {0, 1, 3, 0}};
    for (const auto& a : odd) {
        cloud(a[0], a[1], a[2], a[3]);
        CHECK(id(2) == 2);
    }
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

TEST_CASE("gfx/Metal scene view: a frame dropped after the index clamp was emitted clamps again in the next frame") {
    // A mesh of 8 vertices is drawn; then one of 4 vertices with the same indices arrives, and the
    // frame that emitted its clamp is dropped (its compile fails on a camera that is not finite).
    // The next frame must clamp again, so that every index the draw reads is below 4, rather than
    // reuse the indices clamped against 8.
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    SceneView& v = *rig.view;
    Wire w;
    const int32_t tri[6] = {5, 6, 7, 4, 5, 6};
    auto mesh = [&](uint64_t seq, const std::vector<float>& pos) {
        wire::SceneUpdate u = update("surface", CALIPER_NET_PART_MESH, seq, false);
        u.params.set("color", rgba_json(0xff3080c0u));
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {int64_t(pos.size() / 3), 3},
                                               pos.data(), pos.size() * 4));
        u.arrays.push_back(wire::Section::make("indices", CALIPER_NET_DTYPE_I32, {2, 3}, tri, sizeof tri));
        return w.roundtrip(u);
    };
    std::vector<float> eight, four;
    for (int i = 0; i < 8; ++i) {
        const float p[3] = {float(i & 1), float((i >> 1) & 1), float(i >> 2)};
        eight.insert(eight.end(), p, p + 3);
        if (i < 4) four.insert(four.end(), p, p + 3);
    }
    std::string why;
    REQUIRE_MESSAGE(feed(v, mesh(1, eight), &why), why);
    v.keep_draws(true);
    REQUIRE_MESSAGE(v.render(64, 64, 1.0f, 0xff000000u, &why), why);
    REQUIRE(v.backend().wait(v.backend().last_use(v.color())));

    // The smaller mesh, and a frame whose compile fails after prepare emitted the clamp.
    REQUIRE_MESSAGE(feed(v, mesh(2, four), &why), why);
    std::get<caliper_scene::Orbit>(v.camera()).distance = NAN;
    CHECK_FALSE(v.render(64, 64, 1.0f, 0xff000000u, &why));
    CHECK(why.find("not finite") != std::string::npos);

    // A good frame: the clamp runs again in it.
    v.fit();
    v.backend().take_stats();
    REQUIRE_MESSAGE(v.render(64, 64, 1.0f, 0xff000000u, &why), why);
    CHECK(v.backend().take_stats().dispatches == 1);
    REQUIRE(v.last_draws().records().size == 1);
    const DrawRecord& r = v.last_draws().records()[0];
    REQUIRE(bool(r.index_buffer));
    REQUIRE(r.count == 6);

    // The indices the draw reads, copied out of the layer's index buffer by a small dispatch.
    static const char* kCopyMsl = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void cs_copy(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }
)metal";
    static const EntryDesc kCopyEntries[] = {{"cs_copy", kStageCompute, {1, 1, 1}}};
    static const BindingDesc kCopyBindings[] = {
        {0, 0, BindingType::Storage, kStageCompute, false, 4, Format::RGBA8Unorm, 0},
        {0, 1, BindingType::StorageReadWrite, kStageCompute, false, 4, Format::RGBA8Unorm, 1},
    };
    Backend& gfx = v.backend();
    ProgramDesc pd;
    pd.label = "copy indices";
    pd.entries = kCopyEntries;
    pd.entry_count = 1;
    pd.bindings = kCopyBindings;
    pd.binding_count = 2;
    pd.msl = kCopyMsl;
    const Program copy = gfx.create_program(pd);
    REQUIRE_MESSAGE(bool(copy), gfx.last_error());
    ComputePipelineDesc cd;
    cd.program = copy;
    cd.entry = "cs_copy";
    const Pipeline pipe = gfx.create_compute_pipeline(cd);
    const Buffer out = gfx.create_buffer({r.count * 4, kStorage, true, "indices read back"});
    REQUIRE_MESSAGE((pipe && out), gfx.last_error());
    BindGroupDesc bd;
    bd.program = copy;
    bd.entry_count = 2;
    bd.entries[0].binding = 0;
    bd.entries[0].buffer = r.index_buffer;
    bd.entries[0].offset = r.index_offset;
    bd.entries[0].size = r.count * 4;
    bd.entries[1].binding = 1;
    bd.entries[1].buffer = out;
    const BindGroup g = gfx.create_bind_group(bd);
    REQUIRE_MESSAGE(bool(g), gfx.last_error());
    Frame f;
    const BindGroup groups[1] = {g};
    f.dispatch(pipe, groups, {}, r.count, 1, 1);
    const Serial s = gfx.submit(std::move(f));
    REQUIRE_MESSAGE(s != 0, gfx.last_error());
    REQUIRE(gfx.wait(s));
    const uint32_t* got = static_cast<const uint32_t*>(gfx.mapped(out));
    REQUIRE(got != nullptr);
    std::string seen;
    for (uint32_t i = 0; i < r.count; ++i) seen += (i ? " " : "") + std::to_string(got[i]);
    for (uint32_t i = 0; i < r.count; ++i) CHECK_MESSAGE(got[i] < 4, "the draw reads indices ", seen);
    gfx.release(g);
    gfx.release(out);
    gfx.release(pipe);
    gfx.release(copy);
}

namespace {

// Where the pixels of exactly one colour lie: their count and their centre.
struct Spot {
    int count = 0;
    double x = 0, y = 0;
};
Spot spot(const std::vector<uint8_t>& px, uint32_t w, uint32_t rgba) {
    Spot s;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        uint32_t p;
        std::memcpy(&p, &px[i], 4);
        if (p != rgba) continue;
        ++s.count;
        s.x += double((i / 4) % w);
        s.y += double((i / 4) / w);
    }
    if (s.count) {
        s.x /= s.count;
        s.y /= s.count;
    }
    return s;
}

}  // namespace

// The up-axis switch (design/scenes.md, "The up-axis switch"), drawn: a scene whose up is +y holds a
// point at the origin, where the fit puts the orbit's target, and one unit along +z and -z from it.
// Switched to Z up, the point at +z is drawn straight above the target and the one at -z straight
// below; the eye, the target and the distance stay, the scene's batches, their sequence numbers and
// the bytes copied stay, and the scale bar keeps its length and its label. Through a pinhole the
// bar hides and a switch waits for the orbit; a reset returns to the scene's up. Captures go to
// CALIPER_GFX_CAPTURE_DIR as scene-view-up-y.png and scene-view-up-z.png.
TEST_CASE("gfx/Metal scene view: switching to Z up draws the point at +z above the target and changes nothing of the scene") {
    namespace cs = caliper_scene;
    ViewRig rig;
    if (!rig.init(CALIPER_NET_SPACE_3D, "+y", "m")) CALIPER_SKIP_NO_GPU("no Metal device");
    SceneView& v = *rig.view;
    Wire w;
    wire::SceneClear reset;
    reset.sub = 1;
    reset.view = "landscape";
    reset.seq = 1;
    reset.path = "";
    reset.reset = true;
    reset.state = true;
    REQUIRE(feed(v, w.roundtrip(reset)));
    const float at[3] = {0, 0, 0}, up[3] = {0, 0, 1}, down[3] = {0, 0, -1};
    const uint32_t kRed = 0xff0000ffu, kGreen = 0xff00ff00u, kBlue = 0xffff0000u, kClear = 0xff202020u;
    const struct { const char* path; const float* pos; uint32_t rgba; } pts[] = {
        {"target", at, kRed}, {"above", up, kGreen}, {"below", down, kBlue}};
    uint64_t seq = 2;
    for (const auto& p : pts) {
        wire::SceneUpdate u = update(p.path, CALIPER_NET_PART_POINTS, seq++, false);
        u.params.set("color", rgba_json(p.rgba));
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {1, 3}, p.pos, 12));
        std::string why;
        REQUIRE_MESSAGE(feed(v, w.roundtrip(u), &why), why);
    }
    const uint32_t W = 320, H = 240;
    std::string why;
    REQUIRE_MESSAGE(v.render(W, H, 1.0f, kClear, &why), why);
    CHECK(v.up_mode() == cs::Axis::PosY);
    CHECK_FALSE(v.up_chosen());
    CHECK(v.up_choices() == std::vector<cs::Axis>{cs::Axis::PosY, cs::Axis::PosZ});
    const cs::Orbit o0 = std::get<cs::Orbit>(v.camera());
    CHECK(o0.target[0] == doctest::Approx(0.0));
    const std::vector<uint8_t> y_up = pixels(v);
    capture_png("scene-view-up-y.png", y_up, W, H);
    const caliper_live_views::SceneCounts c0 = v.counts();
    const uint64_t revision = v.tree().revision();
    const uint64_t above_seq = v.tree().batch(v.tree().find("above"), cs::Primitive::Points)->seq;
    const cs::ScaleBar bar0 = v.scale_bar(W, H);
    REQUIRE(bar0.visible);
    MESSAGE("scale bar in Y up: ", bar0.label, ", ", bar0.pixels, " px");

    REQUIRE(v.switch_up(cs::Axis::PosZ));
    CHECK(v.up_chosen());
    REQUIRE_MESSAGE(v.render(W, H, 1.0f, kClear, &why), why);
    const std::vector<uint8_t> z_up = pixels(v);
    capture_png("scene-view-up-z.png", z_up, W, H);
    const cs::Orbit o1 = std::get<cs::Orbit>(v.camera());
    CHECK(o1.up == cs::Axis::PosZ);
    const cs::Vec3 e0 = cs::orbit_eye(o0), e1 = cs::orbit_eye(o1);
    CHECK(std::fabs(e1.x - e0.x) + std::fabs(e1.y - e0.y) + std::fabs(e1.z - e0.z) <= 1e-6 * o0.distance);
    CHECK(o1.distance == o0.distance);
    CHECK(o1.target[0] == o0.target[0]);
    CHECK(o1.target[1] == o0.target[1]);
    CHECK(o1.target[2] == o0.target[2]);
    // On screen: the target at the centre, +z straight above it and -z straight below.
    const Spot t = spot(z_up, W, kRed), a = spot(z_up, W, kGreen), b = spot(z_up, W, kBlue);
    REQUIRE(t.count > 0);
    REQUIRE(a.count > 0);
    REQUIRE(b.count > 0);
    MESSAGE("Z up: target at ", t.x, ",", t.y, "; +z at ", a.x, ",", a.y, "; -z at ", b.x, ",", b.y);
    CHECK(std::fabs(t.x - (W - 1) * 0.5) <= 1.0);
    CHECK(std::fabs(t.y - (H - 1) * 0.5) <= 1.0);
    CHECK(a.y < t.y - 10.0);
    CHECK(b.y > t.y + 10.0);
    CHECK(std::fabs(a.x - t.x) <= 1.0);
    CHECK(std::fabs(b.x - t.x) <= 1.0);
    // Nothing of the scene changed: no message, no upload, the same batches and sequence numbers.
    const caliper_live_views::SceneCounts c1 = v.counts();
    CHECK(c1.messages == c0.messages);
    CHECK(c1.bytes_total == c0.bytes_total);
    CHECK(c1.buffers_created == c0.buffers_created);
    CHECK(c1.batches == c0.batches);
    CHECK(v.tree().revision() == revision);
    CHECK(v.tree().batch(v.tree().find("above"), cs::Primitive::Points)->seq == above_seq);
    // The bar keeps its length and its label.
    const cs::ScaleBar bar1 = v.scale_bar(W, H);
    CHECK(bar1.label == bar0.label);
    CHECK(bar1.pixels == doctest::Approx(bar0.pixels).epsilon(1e-12));

    // And back to Y up: the first picture's points where they were.
    REQUIRE(v.switch_up(cs::Axis::PosY));
    REQUIRE_MESSAGE(v.render(W, H, 1.0f, kClear, &why), why);
    const std::vector<uint8_t> again = pixels(v);
    for (uint32_t c : {kRed, kGreen, kBlue}) {
        const Spot p = spot(y_up, W, c), q = spot(again, W, c);
        CHECK(p.count == q.count);
        CHECK(std::fabs(p.x - q.x) <= 0.5);
        CHECK(std::fabs(p.y - q.y) <= 0.5);
    }

    // Through a pinhole the bar hides; a switch made there applies when the orbit returns.
    const cs::ViewCamera orbit_y = v.camera();
    v.camera() = cs::ThroughPinhole{};
    v.render(W, H, 1.0f, kClear, &why);   // the scene holds no camera to look through
    CHECK_FALSE(v.scale_bar(W, H).visible);
    REQUIRE(v.switch_up(cs::Axis::PosZ));
    v.camera() = orbit_y;
    REQUIRE_MESSAGE(v.render(W, H, 1.0f, kClear, &why), why);
    CHECK(std::get<cs::Orbit>(v.camera()).up == cs::Axis::PosZ);
    CHECK(v.scale_bar(W, H).visible);
    // The option hides it; a reset returns to the scene's up and keeps the option.
    v.set_scale_bar(false);
    CHECK_FALSE(v.scale_bar(W, H).visible);
    v.reset();
    REQUIRE_MESSAGE(v.render(W, H, 1.0f, kClear, &why), why);
    CHECK(v.up_mode() == cs::Axis::PosY);
    CHECK_FALSE(v.up_chosen());
    CHECK(std::get<cs::Orbit>(v.camera()).up == cs::Axis::PosY);
    CHECK(std::get<cs::Orbit>(v.camera()).azimuth == doctest::Approx(cs::kDefaultAzimuth));
    CHECK_FALSE(v.scale_bar_on());
}

// The up-axis switch, the scale bar and the reset meet looking through a camera entity (the right-click
// menu's camera, kept by path): the bar hides as it does through the view's own pinhole, since such a
// picture has no depth to measure at (design/scenes.md, "A scale bar from the scene's units"); a switch
// made there applies when the orbit returns; and a reset returns to the view's own camera.
TEST_CASE("gfx/Metal scene view: looking through a camera entity hides the scale bar, keeps a switch for the orbit and ends with a reset") {
    namespace cs = caliper_scene;
    primitive_fixtures::Fixture f = primitive_fixtures::through();
    ViewRig rig;
    if (!rig.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str())) CALIPER_SKIP_NO_GPU("no Metal device");
    SceneView& v = *rig.view;
    primitive_rows::Row row;
    row.fixture = f;
    REQUIRE_MESSAGE(row.draw(v), row.error);
    constexpr uint32_t W = primitive_fixtures::kSize;
    std::string why;
    REQUIRE(v.looking_through() == "rig/left");
    REQUIRE(std::holds_alternative<cs::ThroughPinhole>(v.effective_camera()));
    CHECK(v.camera_text().find(" through rig/left") != std::string::npos);
    CHECK_FALSE(v.scale_bar(W, W).visible);
    // A switch made while looking through: the picture stays the camera's, and the orbit takes the mode.
    REQUIRE(std::get<cs::Orbit>(v.camera()).up == cs::Axis::PosZ);
    REQUIRE(v.switch_up(cs::Axis::PosY));
    REQUIRE_MESSAGE(v.render(W, W, 1.0f, primitive_fixtures::kClear, &why), why);
    CHECK(std::holds_alternative<cs::ThroughPinhole>(v.effective_camera()));
    CHECK_FALSE(v.scale_bar(W, W).visible);
    REQUIRE(v.look_through("", &why));
    REQUIRE_MESSAGE(v.render(W, W, 1.0f, primitive_fixtures::kClear, &why), why);
    CHECK(std::holds_alternative<cs::Orbit>(v.effective_camera()));
    CHECK(std::get<cs::Orbit>(v.camera()).up == cs::Axis::PosY);
    CHECK(v.up_mode() == cs::Axis::PosY);
    CHECK(v.scale_bar(W, W).visible);
    CHECK(v.camera_text().find(" through") == std::string::npos);
    // A reset while looking through returns to the view's own orbit about the scene's up.
    REQUIRE(v.look_through("rig/left", &why));
    v.reset();
    REQUIRE_MESSAGE(v.render(W, W, 1.0f, primitive_fixtures::kClear, &why), why);
    CHECK(v.looking_through().empty());
    CHECK(std::holds_alternative<cs::Orbit>(v.effective_camera()));
    CHECK(std::get<cs::Orbit>(v.camera()).up == cs::Axis::PosZ);
    CHECK(v.scale_bar(W, W).visible);
    // The view's own pinhole camera, with no entity looked through, names no path.
    v.camera() = cs::ThroughPinhole{};
    CHECK(v.camera_text().find(" through") == std::string::npos);
    CHECK_FALSE(v.scale_bar(W, W).visible);
}

TEST_CASE("gfx/Metal scene view: a 2-D scene has no up-axis switch and an exact scale bar") {
    namespace cs = caliper_scene;
    ViewRig rig;
    if (!rig.init(CALIPER_NET_SPACE_2D, "-y", "px")) CALIPER_SKIP_NO_GPU("no Metal device");
    SceneView& v = *rig.view;
    CHECK(v.up_choices().empty());
    CHECK_FALSE(v.switch_up(cs::Axis::PosZ));
    CHECK_FALSE(v.up_chosen());
    Wire w;
    const float xy[4] = {0, 0, 640, 480};
    wire::SceneUpdate u = update("pixels", CALIPER_NET_PART_POINTS, 2, false);
    u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {2, 2}, xy, 16));
    std::string why;
    REQUIRE_MESSAGE(feed(v, w.roundtrip(u), &why), why);
    REQUIRE_MESSAGE(v.render(400, 300, 1.0f, 0xff202020u, &why), why);
    const cs::ScaleBar b = v.scale_bar(400, 300);
    REQUIRE(b.visible);
    const double upp = 2.0 * std::get<cs::PanZoom>(v.camera()).half_height / 300.0;
    CHECK(std::fabs(b.pixels - b.length / upp) <= 0.5);
    CHECK(b.label.find(" px") != std::string::npos);
    CHECK(b.label.find("at target") == std::string::npos);
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
// ---- Shapes drawn by library pieces (draw/scene_pieces.h) ------------------------------------------

namespace {

int channel_gap(uint32_t a, uint32_t b) {
    int g = 0;
    for (int k = 0; k < 4; ++k) g = std::max(g, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return g;
}

// The view's programs' ID target as (pick id, item) pairs, top row first.
std::vector<uint64_t> program_ids(Backend& gfx, SceneView& v) {
    std::vector<uint64_t> ids(size_t(v.width()) * v.height(), 0);
    if (!v.id_target()) return ids;
    const caliper_gfx::Readback rb = gfx.read_texture(v.id_target(), 0, 0, v.width(), v.height());
    std::vector<uint8_t> bytes;
    REQUIRE(rb);
    REQUIRE(gfx.wait(gfx.last_use(v.id_target())));
    REQUIRE(gfx.take_readback(rb, &bytes));
    REQUIRE(bytes.size() >= ids.size() * 8);
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t pair[2];
        std::memcpy(pair, bytes.data() + i * 8, 8);
        ids[i] = pair[0] == 0 ? 0 : (uint64_t(pair[0]) << 32 | pair[1]) + 1;
    }
    return ids;
}

// The comparison of a picture drawn by the pieces with the built-in's, under the rules of the
// match tests (tests/points_match/): a pixel whose 3 x 3 neighbourhood shows one identity in the
// pieces' ID target is interior and lies within `interior` levels of the built-in pixel; every other
// pixel lies within 1 level of a built-in pixel of its 3 x 3 neighbourhood.
struct ShapeComparison {
    size_t differ = 0, interior = 0, failed = 0;
    int max_interior = 0, max_edge = 0;
    std::string first;
};
ShapeComparison compare_shapes(const std::vector<uint8_t>& piece, const std::vector<uint8_t>& builtin,
                               const std::vector<uint64_t>& ids, int S, int interior) {
    ShapeComparison c;
    auto px = [&](const std::vector<uint8_t>& img, int x, int y) {
        uint32_t v;
        std::memcpy(&v, img.data() + (size_t(y) * S + x) * 4, 4);
        return v;
    };
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            const uint32_t p = px(piece, x, y), b = px(builtin, x, y);
            if (p != b) ++c.differ;
            const uint64_t own = ids[size_t(y) * S + x];
            bool inside = own != 0;
            for (int dy = -1; dy <= 1 && inside; ++dy)
                for (int dx = -1; dx <= 1 && inside; ++dx)
                    inside = ids[size_t(std::clamp(y + dy, 0, S - 1)) * S + std::clamp(x + dx, 0, S - 1)] == own;
            int g = channel_gap(p, b);
            if (inside) {
                ++c.interior;
                c.max_interior = std::max(c.max_interior, g);
                if (g <= interior) continue;
            } else {
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        g = std::min(g, channel_gap(p, px(builtin, std::clamp(x + dx, 0, S - 1), std::clamp(y + dy, 0, S - 1))));
                c.max_edge = std::max(c.max_edge, g);
                if (g <= 1) continue;
            }
            if (c.failed++ == 0) {
                char b2[160];
                std::snprintf(b2, sizeof b2, "pixel (%d, %d): pieces %08x, built-in %08x, %s, gap %d", x, y, p, b,
                              inside ? "interior" : "edge", g);
                c.first = b2;
            }
        }
    return c;
}

// The answer to a pick that the view asked, once its readbacks are in.
caliper_scene::PickPass::Result take_any_pick(Backend& gfx, SceneView& v, caliper_scene::Hit* hit) {
    caliper_scene::PickPass::Result r = caliper_scene::PickPass::Result::Pending;
    for (int tries = 0; tries < 10000 && r == caliper_scene::PickPass::Result::Pending; ++tries) {
        gfx.wait(v.pick_pass()->last_use());
        // The pieces' pick reads the ID target and the depth: each readback is a use of its texture.
        if (v.id_target()) gfx.wait(gfx.last_use(v.id_target()));
        if (v.depth_target()) gfx.wait(gfx.last_use(v.depth_target()));
        gfx.poll();
        r = v.poll_pick(hit);
    }
    return r;
}

void capture_shapes(const std::string& name, const std::vector<uint8_t>& piece, const std::vector<uint8_t>& builtin, int S) {
    std::vector<uint8_t> diff(piece.size(), 255);
    for (size_t i = 0; i < piece.size(); i += 4)
        for (int k = 0; k < 3; ++k) diff[i + size_t(k)] = uint8_t(std::min(255, 32 * std::abs(int(piece[i + size_t(k)]) - int(builtin[i + size_t(k)]))));
    capture_png(("scene-pieces-" + name + "-builtin.png").c_str(), builtin, S, S);
    capture_png(("scene-pieces-" + name + "-pieces.png").c_str(), piece, S, S);
    capture_png(("scene-pieces-" + name + "-diff.png").c_str(), diff, S, S);
}

std::string hit_text(const SceneView& v, const caliper_scene::Hit& h) {
    return v.tree().path(h.entity) + " " + caliper_scene::primitive_name(h.primitive) + " " + std::to_string(h.item) +
           (h.has_pixel ? " px " + std::to_string(h.pixel[0]) + "," + std::to_string(h.pixel[1]) : std::string());
}

// One fixture drawn twice, by the pieces and by the built-in: the pictures under the rules above,
// the counts of the shapes, and the picks at interior pixels of the pieces' IDs on a grid, which
// give the same entity, primitive, item and pixel in both views.
void compare_fixture(const std::string& name, ViewRig& pieces, ViewRig& builtin, int S, int interior) {
    SceneView& pv = *pieces.view;
    SceneView& bv = *builtin.view;
    const std::vector<uint8_t> p = pixels(pv), b = pixels(bv);
    const std::vector<uint64_t> ids = program_ids(*pieces.gfx, pv);
    const ShapeComparison c = compare_shapes(p, b, ids, S, interior);
    const SceneShapeCounts counts = pv.shape_counts();
    std::string falls;
    for (const auto& [kind, n] : counts.fallbacks) falls += " " + kind + " " + std::to_string(n) + ";";
    capture_shapes(name, p, b, S);
    // Picks on a grid, at interior pixels of the pieces' IDs.
    size_t asked = 0, agreed = 0;
    std::string first_pick;
    for (int y = 4; y < S; y += 9)
        for (int x = 4; x < S; x += 9) {
            const uint64_t own = ids[size_t(y) * S + x];
            bool inside = own != 0;
            for (int dy = -2; dy <= 2 && inside; ++dy)
                for (int dx = -2; dx <= 2 && inside; ++dx)
                    inside = ids[size_t(std::clamp(y + dy, 0, S - 1)) * S + std::clamp(x + dx, 0, S - 1)] == own;
            if (!inside) continue;
            std::string why;
            caliper_scene::Hit hp, hb;
            REQUIRE_MESSAGE(pv.request_pick(x + 0.5f, y + 0.5f, &why), why);
            const auto rp = take_any_pick(*pieces.gfx, pv, &hp);
            REQUIRE_MESSAGE(bv.request_pick(x + 0.5f, y + 0.5f, &why), why);
            const auto rb = take_any_pick(*builtin.gfx, bv, &hb);
            ++asked;
            // The same event: entity, primitive and index (an image's pixel, row by row), as
            // pick_detail() gives it; a box's faces and edges are one item of one piece, and so are a
            // depth image's quad and its cloud.
            auto index_of = [](const SceneView& v, const caliper_scene::Hit& h) -> int64_t {
                if (h.primitive == caliper_scene::Primitive::Image && h.part == caliper_scene::DrawPart::Batch) {
                    const caliper_scene::Batch* b = v.tree().batch(h.entity, caliper_scene::Primitive::Image);
                    return h.has_pixel && b ? int64_t(h.pixel[1]) * b->image_width + h.pixel[0] : -1;
                }
                return int64_t(h.item);
            };
            const bool same = rp == rb && (rp != caliper_scene::PickPass::Result::Hit ||
                                           (pv.tree().path(hp.entity) == bv.tree().path(hb.entity) &&
                                            hp.primitive == hb.primitive && index_of(pv, hp) == index_of(bv, hb) &&
                                            pv.pick_detail(hp).empty() == bv.pick_detail(hb).empty()));
            if (same) {
                ++agreed;
            } else if (first_pick.empty()) {
                first_pick = "at (" + std::to_string(x) + ", " + std::to_string(y) + "): pieces " +
                             (rp == caliper_scene::PickPass::Result::Hit ? hit_text(pv, hp)
                                                                         : "result " + std::to_string(int(rp))) +
                             ", built-in " +
                             (rb == caliper_scene::PickPass::Result::Hit ? hit_text(bv, hb)
                                                                         : "result " + std::to_string(int(rb)));
            }
        }
    MESSAGE("[scene-pieces] ", name, ": ", counts.pieces, " shapes by pieces, ", counts.builtin, " by the built-in (",
            falls, " ); ", c.differ, " of ", S * S, " pixels differ, ", c.interior, " interior pixels with a largest gap of ",
            c.max_interior, ", largest edge gap ", c.max_edge, " to a neighbour, ", c.failed, " outside the rules; picks ",
            agreed, " of ", asked, " agree");
    CHECK_MESSAGE(c.failed == 0, name, ": ", c.first);
    CHECK_MESSAGE(agreed == asked, name, ": ", first_pick);
}

}  // namespace

// Every fixture of the primitive rows and the demo's scene, drawn by the library pieces and by the
// built-in draw through the switch of the scene view (ShapeDraw), compare under the rules of the
// pieces' match tests, and their picks agree. Images take 1 level inside, as image@1's match does
// (its bilinear filter is written in WGSL). The fixture "through" looks through a camera entity
// whose pixel space holds points of radii in scene units, which no piece draws, thus every shape
// keeps the built-in draw; "through-pieces" is the same view without them. CALIPER_GFX_CAPTURE_DIR=<folder> writes each pair and its
// difference times 32 there.
TEST_CASE("gfx/Metal scene view: shapes drawn by the library pieces match the built-in draw, with the same picks") {
    ViewRig probe;
    if (!probe.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    std::vector<primitive_fixtures::Fixture> fixtures = primitive_fixtures::all();
    // The view through a camera with its points sized in pixels, not by radii in scene units,
    // which keep the built-in draw and with it every shape of the view: the pinhole frames the
    // pieces, the camera's pixel space lies flat over the picture, and the bars keep the ground.
    primitive_fixtures::Fixture flat = primitive_fixtures::through();
    flat.name = "through-pieces";
    for (wire::Message& m : flat.messages)
        if (auto* u = std::get_if<wire::SceneUpdate>(&m)) u->params.erase("radius");
    fixtures.push_back(flat);
    // The view through a camera whose pixel space holds a textured mesh, which keeps the built-in
    // draw: it draws among the pieces of that flat layer in the scene core's order, after the image
    // and before the arrow, the marks and the box that overlap it.
    primitive_fixtures::Fixture late = primitive_fixtures::through();
    late.name = "through-late";
    {
        std::vector<float> qp = {20, 14, 46, 14, 46, 32, 20, 14, 46, 32, 20, 32};
        std::vector<float> uv = {0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1};
        std::vector<uint32_t> tex(16);
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x) tex[y * 4 + x] = primitive_fixtures::RGBA(90 + 40 * x, 200 - 30 * y, 120, 255);
        wire::SceneUpdate card = late.update("rig/left/card", CALIPER_NET_PART_MESH);
        card.arrays.push_back(late.section("positions", CALIPER_NET_DTYPE_F32, {6, 2}, qp));
        card.arrays.push_back(late.section("uvs", CALIPER_NET_DTYPE_F32, {6, 2}, uv));
        card.arrays.push_back(late.section("texture", CALIPER_NET_DTYPE_U8, {4, 4, 4}, tex));
        late.messages.push_back(card);
    }
    fixtures.push_back(late);
    for (const primitive_fixtures::Fixture& f : fixtures) {
        CAPTURE(f.name);
        ViewRig pr, br;
        REQUIRE(pr.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str(), caliper_live_views::ShapeDraw::pieces));
        REQUIRE(br.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str(), caliper_live_views::ShapeDraw::builtin));
        primitive_rows::Row rp, rb;
        rp.fixture = f;
        rb.fixture = f;
        REQUIRE_MESSAGE(rp.draw(*pr.view), rp.error);
        REQUIRE_MESSAGE(rb.draw(*br.view), rb.error);
        // Only a textured mesh keeps the built-in draw (a texture resource of other formats and
        // sizes is not built yet), alone and without the other shapes of its view.
        const SceneShapeCounts counts = pr.view->shape_counts();
        CHECK_MESSAGE(counts.pieces > 0, f.name, " draws no shape with a piece");
        const bool textured = f.name == "mesh" || f.name == "through-late";
        CHECK(counts.builtin == (textured ? 1u : 0u));
        CHECK(counts.fallbacks.size() == (textured ? 1u : 0u));
        if (textured) CHECK(counts.fallbacks.count("textured mesh") == 1);
        compare_fixture(f.name, pr, br, int(primitive_fixtures::kSize), f.name == "images" ? 1 : 0);
    }
    {
        ViewRig pr, br;
        REQUIRE(pr.init(CALIPER_NET_SPACE_3D, "+z", "", caliper_live_views::ShapeDraw::pieces));
        REQUIRE(br.init());
        const scene_fixtures::SurfaceRowFixture fx;
        const DemoMessages dm(fx);
        Wire wp, wb;
        draw_demo(pr, wp, dm);
        draw_demo(br, wb, dm);
        // Every shape of the demo, its surface's mesh of flat faces and indices too.
        const SceneShapeCounts counts = pr.view->shape_counts();
        CHECK(counts.pieces == 4);
        CHECK(counts.builtin == 0);
        CHECK(counts.fallbacks.empty());
        compare_fixture("demo", pr, br, int(scene_fixtures::SurfaceRowFixture::kSize), 0);
    }
}

// The highlight of a selection drawn by the pieces: the dim by image@1 and the selected rows by a
// second instance of points@1 or boxes@1, into the colour alone, against the built-in's dim and
// highlight, under the rules of the comparison above; the selection over the whole picture holds
// every box, the points of radii in scene units of "edges" and the points flat over the picture
// of "through". The highlight writes no ID and no depth, thus the picks are those of the shapes.
TEST_CASE("gfx/Metal scene view: the highlight of a selection drawn by the pieces matches the built-in's") {
    ViewRig probe;
    if (!probe.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    for (const primitive_fixtures::Fixture& f :
         {primitive_fixtures::boxes(), primitive_fixtures::edges(), primitive_fixtures::through()}) {
        CAPTURE(f.name);
        ViewRig pr, br;
        REQUIRE(pr.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str(), caliper_live_views::ShapeDraw::pieces));
        REQUIRE(br.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str(), caliper_live_views::ShapeDraw::builtin));
        primitive_rows::Row rp, rb;
        rp.fixture = f;
        rb.fixture = f;
        REQUIRE_MESSAGE(rp.draw(*pr.view), rp.error);
        REQUIRE_MESSAGE(rb.draw(*br.view), rb.error);
        const std::vector<uint8_t> plain = pixels(*pr.view);
        const float S = float(primitive_fixtures::kSize);
        const std::string sp = pr.view->select_polygon({0, 0, S, 0, S, S, 0, S});
        const std::string sb = br.view->select_polygon({0, 0, S, 0, S, S, 0, S});
        CHECK(sp == sb);
        std::string why;
        REQUIRE_MESSAGE(pr.view->render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why), why);
        REQUIRE_MESSAGE(br.view->render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why), why);
        CHECK(pr.view->highlighted() > 0);
        CHECK(pr.view->highlighted() == br.view->highlighted());
        CHECK(pr.view->shape_counts().marks > 0);
        CHECK(pr.view->shape_counts().fallbacks.empty());
        CHECK(pixels(*pr.view) != plain);
        compare_fixture(f.name + "-selected", pr, br, int(primitive_fixtures::kSize), 1);
        // No selection: the marks go, and the picture is the plain one again.
        CHECK(pr.view->select_polygon({1, 1, 2, 1, 2, 2}) == "{}");
        REQUIRE_MESSAGE(pr.view->render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why), why);
        CHECK(pr.view->highlighted() == 0);
        CHECK(pr.view->shape_counts().marks == 0);
        CHECK(pixels(*pr.view) == plain);
    }
}

// Every kind of fallback that the shapes' pieces count, in the order of caliper_draw::Fallback. A
// new kind fails this row until it is listed here. What remains: inputs of another dtype or shape
// (u8 and u16 elements are not taken yet), parts above a piece's capacity, a textured mesh (a
// texture resource of other formats and sizes is not built yet), a stage program (stages go with
// their hooks), params outside a piece's range, a depth cloud in a moved pixel space, a class
// above the 4,096 colors of segmentation@1, and the error paths of a part without arrays, a
// stopped piece and a build without the library.
#if defined(CALIPER_HAVE_SCENE_PIECES)
TEST_CASE("gfx/Metal scene view: the kinds of fallback of the shapes' pieces are the listed ones") {
    const std::vector<std::string> listed = {
        "points of another shape", "lines of another shape", "arrows of another shape", "boxes of another shape",
        "mesh of another shape", "image of another shape", "depth image of another shape", "mask of another shape",
        "points above the piece's capacity", "segments above the piece's capacity", "arrows above the piece's capacity",
        "boxes above the piece's capacity", "mesh vertices above the piece's capacity",
        "mesh triangles above the piece's capacity", "image pixels above the piece's capacity",
        "depth pixels above the piece's capacity", "mask pixels above the piece's capacity", "textured mesh",
        "stage program", "point size outside 1 to 64 pixels", "width above 128 pixels",
        "box edge width outside 0.5 to 64 pixels", "depth cloud in a moved pixel space", "class above 4,095 in a mask",
        "batch without arrays to convert", "piece stopped by its runtime", "no library piece"};
    CHECK(caliper_draw::fallback_kinds() == listed);
}
#endif

// The switch: one view drawn with the pieces, then with the built-in, then with the pieces again,
// gives each time the picture of a view that drew that way from the start; nothing draws while
// nothing changes; and a pick whose frame the view replaced is dropped, as the pick pass drops one.
TEST_CASE("gfx/Metal scene view: the switch between the pieces and the built-in draw, and a stale pick of a piece") {
    ViewRig pr, br;
    if (!pr.init(CALIPER_NET_SPACE_3D, "+z", "", caliper_live_views::ShapeDraw::pieces)) CALIPER_SKIP_NO_GPU("no Metal device");
    REQUIRE(br.init());
    const scene_fixtures::SurfaceRowFixture fx;
    const DemoMessages dm(fx);
    Wire wp, wb;
    draw_demo(pr, wp, dm);
    draw_demo(br, wb, dm);
    SceneView& v = *pr.view;
    const uint32_t S = scene_fixtures::SurfaceRowFixture::kSize;
    const std::vector<uint8_t> with_pieces = pixels(v), builtin = pixels(*br.view);
    CHECK(v.shape_runtimes().size() == 4);
    CHECK(v.draw_runtimes().empty());
    std::string why;
    v.set_shape_draw(caliper_live_views::ShapeDraw::builtin);
    REQUIRE_MESSAGE(v.render(S, S, 1.0f, scene_fixtures::SurfaceRowFixture::kClear, &why), why);
    CHECK(v.shape_runtimes().empty());
    CHECK(pixels(v) == builtin);
    v.set_shape_draw(caliper_live_views::ShapeDraw::pieces);
    REQUIRE_MESSAGE(v.render(S, S, 1.0f, scene_fixtures::SurfaceRowFixture::kClear, &why), why);
    CHECK(v.shape_runtimes().size() == 4);
    const std::vector<uint8_t> again = pixels(v);
    size_t changed = 0;
    for (size_t i = 0; i + 4 <= again.size(); i += 4) changed += std::memcmp(&again[i], &with_pieces[i], 4) != 0;
    CHECK_MESSAGE(changed == 0, changed, " pixels differ after the switch back");
    capture_png("scene-pieces-switch-back.png", again, int(S), int(S));
    // Nothing changes: nothing is compiled or submitted.
    const uint64_t frames = v.cost().frames;
    REQUIRE_MESSAGE(v.render(S, S, 1.0f, scene_fixtures::SurfaceRowFixture::kClear, &why), why);
    CHECK(v.cost().frames == frames);
    // A stale pick of a piece.
    REQUIRE(v.request_pick(S * 0.5f, S * 0.5f, &why));
    v.orbit(2.0f, 0.0f);
    REQUIRE_MESSAGE(v.render(S, S, 1.0f, scene_fixtures::SurfaceRowFixture::kClear, &why), why);
    caliper_scene::Hit hit;
    CHECK(take_any_pick(*pr.gfx, v, &hit) == caliper_scene::PickPass::Result::Dropped);
}
#endif

namespace {

// points@1's resource "points" of a points update in the per-shape form, as caliper._scene_pieces
// writes it (python/caliper/_scene_pieces.py, pack_points): x, y and z (0 in 2-D) and in w the radius
// (the row's, else the batch's "radius", else 1), then the bits of the color source (packed RGBA,
// the f32 of a value, a class id, else NaN bits), the row index and 8 bytes of zero.
std::vector<uint8_t> pack_points(const wire::SceneUpdate& u) {
    const wire::Section* pos = u.array("positions");
    const size_t n = size_t(pos->shape[0]), d = size_t(pos->shape[1]);
    std::vector<uint8_t> out(n * 32, 0);
    const wire::Section* colors = u.array("colors");
    const wire::Section* values = u.array("values");
    const wire::Section* ids = u.array("class_ids");
    const wire::Section* radii = u.array("radii");
    const Json* one = u.params.find("radius");
    const float radius = one && one->is_number() ? float(one->as_double()) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        uint8_t* p = out.data() + i * 32;
        std::memcpy(p, pos->data + i * d * 4, d * 4);
        if (radii) std::memcpy(p + 12, radii->data + i * 4, 4);
        else std::memcpy(p + 12, &radius, 4);
        uint32_t bits = 0x7fc00000u;
        if (colors) std::memcpy(&bits, colors->data + (colors->shape[0] == 1 ? 0 : i) * 4, 4);
        else if (values) std::memcpy(&bits, values->data + i * 4, 4);
        else if (ids) std::memcpy(&bits, ids->data + i * 4, 4);
        std::memcpy(p + 16, &bits, 4);
        const uint32_t index = uint32_t(i);
        std::memcpy(p + 20, &index, 4);
    }
    return out;
}

// image@1's resources of a color image update, as caliper._scene_pieces writes them (pack_image):
// "pixels" one packed RGBA a pixel (a grey image repeats its value, alpha 255) and "info" the width,
// the height, the channels and 0, then the identity as four columns.
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> pack_image(const wire::SceneUpdate& u) {
    const wire::Section* px = u.array("pixels");
    const size_t h = size_t(px->shape[0]), w = size_t(px->shape[1]), n = w * h;
    const bool grey = px->shape.size() == 2;
    std::vector<uint8_t> pixels(n * 4);
    for (size_t i = 0; i < n; ++i) {
        if (grey) {
            const uint32_t g = px->data[i], rgba = 0xff000000u | g | (g << 8) | (g << 16);
            std::memcpy(&pixels[i * 4], &rgba, 4);
        } else {
            std::memcpy(&pixels[i * 4], px->data + i * 4, 4);
        }
    }
    std::vector<uint8_t> info(80, 0);
    const uint32_t size[4] = {uint32_t(w), uint32_t(h), grey ? 1u : 4u, 0};
    std::memcpy(info.data(), size, 16);
    for (int k = 0; k < 4; ++k) {
        const float one = 1.0f;
        std::memcpy(&info[size_t(16 + k * 16 + k * 4)], &one, 4);
    }
    return {std::move(pixels), std::move(info)};
}

bool color_image(const wire::SceneUpdate& u) {
    const Json* form = u.params.find("form");
    return u.part == CALIPER_NET_PART_IMAGE && form && form->is_string() && form->as_string() == CALIPER_NET_FORM_COLOR &&
           u.array("pixels");
}

// A fixture's messages as a producer sends them to a viewer with "scene_pieces": each points update
// as the RESOURCE of its scene program and the update in the piece form, through the codec, paired
// by the viewer's inbox and applied, the others as they are.
bool draw_piece_form(const primitive_fixtures::Fixture& f, SceneView& view, std::string* why) {
    view.describe(f.info);
    caliper_live::ScenePieceInbox inbox;
    std::vector<std::shared_ptr<const void>> keep;
    int program = 0;
    auto roundtrip = [&](const wire::Message& m, wire::Message* out) {
        auto bytes = std::make_shared<std::vector<uint8_t>>();
        std::string err;
        if (!wire::encode(m, bytes.get(), &err, CALIPER_NET_MINOR_3_2)) return *why = "encode: " + err, false;
        const wire::DecodeResult r = wire::decode(bytes->data(), bytes->size(), out);
        if (!r.ok()) return *why = "decode: " + r.message, false;
        keep.push_back(bytes);
        return true;
    };
    for (const wire::Message& m : f.messages) {
        const auto* cu = std::get_if<wire::SceneUpdate>(&m);
        const bool points = cu && cu->part == CALIPER_NET_PART_POINTS && cu->array("positions");
        if (points || (cu && color_image(*cu))) {
            const std::string id = "scene." + std::to_string(++program);
            // (resource id, stride, bytes) of each resource, in the order they go before the update.
            std::vector<std::tuple<std::string, uint32_t, std::vector<uint8_t>>> res;
            wire::SceneUpdate pu = *cu;
            pu.program = id;
            pu.piece = Json::object();
            pu.piece.set("version", 1);
            pu.resources = Json::object();
            if (points) {
                res.emplace_back("points", 32u, pack_points(*cu));
                pu.piece.set("name", "points");
                for (const char* name : {"positions", "colors", "values", "class_ids", "radii"})
                    if (cu->array(name)) pu.names.push_back(name);
            } else {
                auto [pixels, info] = pack_image(*cu);
                res.emplace_back("pixels", 4u, std::move(pixels));
                res.emplace_back("info", 80u, std::move(info));
                pu.piece.set("name", "image");
                pu.names.push_back("pixels");
            }
            pu.arrays.clear();
            for (auto& [rid, stride, bytes] : res) {
                wire::Resource r;
                r.program = id;
                r.resource = rid;
                r.step = 0;
                r.count = uint32_t(bytes.size() / stride);
                r.stride = stride;
                r.data = wire::Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8, {int64_t(bytes.size())},
                                             bytes.data(), bytes.size());
                pu.resources.set(rid, Json(double(r.count)));
                wire::Message rd;
                if (!roundtrip(r, &rd)) return false;
                const auto& whole = std::get<wire::Resource>(rd);
                if (!inbox.take(whole, std::make_shared<const std::vector<uint8_t>>(whole.data.data, whole.data.data + whole.data.size)))
                    return *why = "the inbox took no scene program", false;
            }
            wire::Message ud;
            if (!roundtrip(pu, &ud)) return false;
            auto& piece = std::get<wire::SceneUpdate>(ud);
            std::shared_ptr<const void> owner = inbox.pair(piece, nullptr, why);
            if (!owner) return false;
            keep.push_back(owner);
            if (!view.apply(piece, why)) return false;
            continue;
        }
        wire::Message d;
        if (!roundtrip(m, &d)) return false;
        bool ok = true;
        if (const auto* u = std::get_if<wire::SceneUpdate>(&d)) ok = view.apply(*u, why);
        else if (const auto* t = std::get_if<wire::SceneTransforms>(&d)) ok = view.apply(*t, why);
        else if (const auto* c = std::get_if<wire::SceneClasses>(&d)) ok = view.apply(*c, why);
        else if (const auto* k = std::get_if<wire::SceneClear>(&d)) ok = view.apply(*k, why);
        if (!ok) return false;
    }
    if (!f.look_through.empty() && !view.look_through(f.look_through, why)) return false;
    view.keep_draws(true);
    return view.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, why);
}

}  // namespace

// The piece form of points and color images (wire protocol 3.2, capability "scene_pieces"): every fixture
// whose points and color images arrive as points@1's and image@1's resources, written as
// caliper._scene_pieces writes them, draws the same pixels as
// from the per-shape part, both with the pieces (the received bytes against the converter's) and with
// the built-in draw (the batch rebuilt from the bytes against the decoded one), and every pick on a
// grid gives the same entity, primitive and item. The packing above writes the golden frame
// resource_scene_points from the arrays of scene_update_points_piece's batch.
TEST_CASE("gfx/Metal scene view: points and color images in the piece form draw as the per-shape part, with the same picks") {
    ViewRig probe;
    if (!probe.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    {   // the packing against the golden frame: three points with values
        const fs::path dir = fs::path(CALIPER_TEST_SOURCE_ROOT) / "tests/fixtures/wire_v3/v3_2";
        std::ifstream in(dir / "resource_scene_points.bin", std::ios::binary);
        std::vector<uint8_t> frame((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        wire::Message m;
        REQUIRE(wire::decode(frame.data(), frame.size(), &m).ok());
        const auto& golden = std::get<wire::Resource>(m);
        primitive_fixtures::Fixture g;
        wire::SceneUpdate u = g.update("p", CALIPER_NET_PART_POINTS);
        u.arrays.push_back(g.section("positions", CALIPER_NET_DTYPE_F32, {3, 3}, std::vector<float>{0, 0, 0, 1, 0, 0.5f, 0, 2, 0.25f}));
        u.arrays.push_back(g.section("values", CALIPER_NET_DTYPE_F32, {3}, std::vector<float>{0.25f, 0.5f, 1.0f}));
        const std::vector<uint8_t> packed = pack_points(u);
        REQUIRE(golden.data.size == packed.size());
        CHECK(std::memcmp(golden.data.data, packed.data(), packed.size()) == 0);
    }
    std::vector<primitive_fixtures::Fixture> fixtures = primitive_fixtures::all();
    primitive_fixtures::Fixture flat = primitive_fixtures::through();
    flat.name = "through-pieces";
    for (wire::Message& m : flat.messages)
        if (auto* u = std::get_if<wire::SceneUpdate>(&m)) u->params.erase("radius");
    fixtures.push_back(flat);
    fixtures.push_back(primitive_fixtures::through());
    size_t with_points = 0;
    for (const primitive_fixtures::Fixture& f : fixtures) {
        CAPTURE(f.name);
        uint32_t points = 0;
        for (const wire::Message& m : f.messages)
            if (const auto* u = std::get_if<wire::SceneUpdate>(&m); u && (u->part == CALIPER_NET_PART_POINTS || color_image(*u))) ++points;
        if (points == 0) continue;
        ++with_points;
        for (const auto draw : {caliper_live_views::ShapeDraw::pieces, caliper_live_views::ShapeDraw::builtin}) {
            ViewRig part, piece;
            REQUIRE(part.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str(), draw));
            REQUIRE(piece.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str(), draw));
            primitive_rows::Row row;
            row.fixture = f;
            REQUIRE_MESSAGE(row.draw(*part.view), row.error);
            std::string why;
            REQUIRE_MESSAGE(draw_piece_form(f, *piece.view, &why), why);
            const bool pieces = draw == caliper_live_views::ShapeDraw::pieces;
            if (pieces) {
                const SceneShapeCounts a = part.view->shape_counts(), b = piece.view->shape_counts();
                CHECK(a.received == 0);
                CHECK(b.pieces == a.pieces);
                CHECK(b.builtin == a.builtin);
                CHECK(b.fallbacks == a.fallbacks);
                MESSAGE("[scene-piece-form] ", f.name, ": ", b.received, " of ", points, " points and color image batches drawn from the received bytes, ",
                        b.pieces, " shapes by pieces, ", b.builtin, " by the built-in");
            }
            const std::vector<uint8_t> pa = pixels(*part.view), pb = pixels(*piece.view);
            size_t differ = 0;
            for (size_t i = 0; i < pa.size(); i += 4) differ += std::memcmp(&pa[i], &pb[i], 4) != 0;
            capture_shapes(std::string("form-") + (pieces ? "pieces-" : "builtin-") + f.name, pb, pa, int(primitive_fixtures::kSize));
            MESSAGE("[scene-piece-form] ", f.name, (pieces ? " by pieces" : " by the built-in"), ": ", differ, " of ",
                    pa.size() / 4, " pixels differ");
            CHECK_MESSAGE(differ == 0, f.name);
            if (pieces) compare_fixture("form-" + f.name, piece, part, int(primitive_fixtures::kSize), 0);
        }
    }
    CHECK(with_points >= 3);
}
