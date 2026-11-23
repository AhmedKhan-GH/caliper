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
#include "gfx/recording_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "live/live_scene.h"
#include "live/scene_piece_form.h"
#if defined(CALIPER_HAVE_SCENE_PIECES)
#include "draw/scene_pieces.h"
#include "draw/draw_runtime.h"
#endif
#include "net/wire.h"
#include "renderer/host_renderer.h"
#include "../scene/surface_row_fixture.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#include "../match_reference.h"
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
using namespace caliper_gfx;
using namespace caliper_scene;

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
        view = std::make_unique<SceneView>(*gfx, caliper_gfx::shaders::kMetalPrograms);
        // These rows check the scene core's own draw list, its pick pass and its counters against
        // their CPU references, thus they draw every shape with the reference draw; the rows of
        // "Shapes Drawn by Library Pieces" at the end of this file draw the same fixtures with the
        // pieces and compare them with the reference.
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
void draw_demo(ViewRig& rig, Wire& w, const DemoMessages& dm) {
    SceneView& v = *rig.view;
    for (const wire::Message& m : {dm.reset(), dm.box(), dm.surface(), dm.path(), dm.marker()}) {
        std::string why;
        REQUIRE_MESSAGE(feed(v, w.roundtrip(m), &why), why);
    }
    std::string why;
    const uint32_t S = scene_fixtures::SurfaceRowFixture::kSize;
    REQUIRE_MESSAGE(v.render(S, S, 1.0f, scene_fixtures::SurfaceRowFixture::kClear, &why), why);
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
    CHECK(batch(world, "floor", Primitive::Mesh)->image_width > 0);
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
        CHECK(land.counts().arrays_live == 0);
    }
}


namespace {

// Waits for the view's pick on Metal (the test may block; a view never does) and takes it.
caliper_scene::PickResult take_pick(Backend& gfx, SceneView& v, caliper_scene::Hit* hit) {
    caliper_scene::PickResult r = caliper_scene::PickResult::Pending;
    for (int tries = 0; tries < 100 && r == caliper_scene::PickResult::Pending; ++tries) {
        gfx.wait(gfx.last_use(v.id_target()));
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
    CHECK(v.counts().arrays_live == 6);   // the box 1, the surface 3, the path 2

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
    CHECK(v.counts().arrays_live == 0);
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

TEST_CASE("gfx/Metal scene view: an update of one batch replaces its arrays and redraws; nothing changes, nothing draws") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    const scene_fixtures::SurfaceRowFixture fx;
    DemoMessages dm(fx);
    Wire w;
    draw_demo(rig, w, dm);
    SceneView& v = *rig.view;
    const std::vector<uint8_t> before = pixels(v);
    const uint64_t frames = v.cost().frames;
    const uint64_t bytes = v.counts().bytes_total;
    const uint64_t revision = v.tree().revision();

    // Nothing changed: no compile, no submit.
    std::string why;
    REQUIRE(v.render(fx.kSize, fx.kSize, 1.0f, fx.kClear, &why));
    CHECK(v.cost().frames == frames);

    // A new surface of the same size (the values shift, as a refit recolours it): its three
    // arrays replace the previous values, and the view draws again.
    for (float& x : dm.values) x = fx.vmin + (fx.vmax - x);
    v.begin_round();
    REQUIRE_MESSAGE(feed(v, w.roundtrip(dm.surface(6)), &why), why);
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

    // A larger marker batch replaces the previous array.
    wire::SceneUpdate big = std::get<wire::SceneUpdate>(dm.marker());
    std::vector<float> many(3 * 400, 0.0f);
    big.arrays[0] = wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {400, 3}, many.data(), many.size() * 4);
    REQUIRE(feed(v, w.roundtrip(big)));
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

// The comparison of a picture drawn by the pieces with the reference's, under the rules of the
// match tests (tests/points_match/): a pixel whose 3 x 3 neighbourhood shows one identity in the
// pieces' ID target is interior and lies within `interior` levels of the reference pixel; every other
// pixel lies within 1 level of a reference pixel of its 3 x 3 neighbourhood.
struct ShapeComparison {
    size_t differ = 0, interior = 0, failed = 0;
    int max_interior = 0, max_edge = 0;
    std::string first;
};
ShapeComparison compare_shapes(const std::vector<uint8_t>& piece, const std::vector<uint8_t>& reference,
                               const std::vector<uint64_t>& ids, int S, int interior) {
    ShapeComparison c;
    auto px = [&](const std::vector<uint8_t>& img, int x, int y) {
        uint32_t v;
        std::memcpy(&v, img.data() + (size_t(y) * S + x) * 4, 4);
        return v;
    };
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            const uint32_t p = px(piece, x, y), b = px(reference, x, y);
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
                        g = std::min(g, channel_gap(p, px(reference, std::clamp(x + dx, 0, S - 1), std::clamp(y + dy, 0, S - 1))));
                c.max_edge = std::max(c.max_edge, g);
                if (g <= 1) continue;
            }
            if (c.failed++ == 0) {
                char b2[160];
                std::snprintf(b2, sizeof b2, "pixel (%d, %d): pieces %08x, reference %08x, %s, gap %d", x, y, p, b,
                              inside ? "interior" : "edge", g);
                c.first = b2;
            }
        }
    return c;
}

// The answer to a pick that the view asked, once its readbacks are in.
caliper_scene::PickResult take_any_pick(Backend& gfx, SceneView& v, caliper_scene::Hit* hit) {
    caliper_scene::PickResult r = caliper_scene::PickResult::Pending;
    for (int tries = 0; tries < 10000 && r == caliper_scene::PickResult::Pending; ++tries) {
        gfx.wait(gfx.last_use(v.id_target()));
        // The pieces' pick reads the ID target and the depth: each readback is a use of its texture.
        if (v.id_target()) gfx.wait(gfx.last_use(v.id_target()));
        if (v.depth_target()) gfx.wait(gfx.last_use(v.depth_target()));
        gfx.poll();
        r = v.poll_pick(hit);
    }
    return r;
}

void capture_shapes(const std::string& name, const std::vector<uint8_t>& piece, const std::vector<uint8_t>& reference, int S) {
    std::vector<uint8_t> diff(piece.size(), 255);
    for (size_t i = 0; i < piece.size(); i += 4)
        for (int k = 0; k < 3; ++k) diff[i + size_t(k)] = uint8_t(std::min(255, 32 * std::abs(int(piece[i + size_t(k)]) - int(reference[i + size_t(k)]))));
    capture_png(("scene-pieces-" + name + "-reference.png").c_str(), reference, S, S);
    capture_png(("scene-pieces-" + name + "-pieces.png").c_str(), piece, S, S);
    capture_png(("scene-pieces-" + name + "-diff.png").c_str(), diff, S, S);
}

std::string hit_text(const SceneView& v, const caliper_scene::Hit& h) {
    return v.tree().path(h.entity) + " " + caliper_scene::primitive_name(h.primitive) + " " + std::to_string(h.item) +
           (h.has_pixel ? " px " + std::to_string(h.pixel[0]) + "," + std::to_string(h.pixel[1]) : std::string());
}

// One fixture drawn twice, by the pieces and by the reference: the pictures under the rules above,
// the counts of the shapes, and the picks at interior pixels of the pieces' IDs on a grid, which
// give the same entity, primitive, item and pixel in both views.
void compare_fixture(const std::string& name, ViewRig& pieces, int S, int interior) {
    SceneView& pv = *pieces.view;
    const std::vector<uint8_t> p = pixels(pv);
    const std::vector<uint8_t> b = match_reference::picture("compare", uint32_t(S), uint32_t(S), nullptr);
    const std::vector<uint64_t> ids = program_ids(*pieces.gfx, pv);
    const ShapeComparison c = compare_shapes(p, b, ids, S, interior);
    capture_shapes(name, p, b, S);
    size_t picked = 0;
    for (int y = 4; y < S; y += 9)
        for (int x = 4; x < S; x += 9) {
            const uint64_t own = ids[size_t(y) * S + x];
            bool inside = own != 0;
            for (int dy = -2; dy <= 2 && inside; ++dy)
                for (int dx = -2; dx <= 2 && inside; ++dx)
                    inside = ids[size_t(std::clamp(y + dy, 0, S - 1)) * S + std::clamp(x + dx, 0, S - 1)] == own;
            if (!inside) continue;
            std::string why;
            Hit hit;
            REQUIRE_MESSAGE(pv.request_pick(x + 0.5f, y + 0.5f, &why), why);
            REQUIRE(take_any_pick(*pieces.gfx, pv, &hit) == PickResult::Hit);
            CHECK(pv.tree().valid(hit.entity));
            uint64_t item = hit.item;
            if (hit.primitive == Primitive::Image && hit.part == DrawPart::Batch) {
                // An image hit names its quad and pixel, while the ID target names the pixel directly.
                CHECK(hit.item == 0);
                REQUIRE(hit.has_pixel);
                const Batch* image = pv.tree().batch(hit.entity, Primitive::Image);
                REQUIRE(image != nullptr);
                CHECK(hit.pixel[0] < image->image_width);
                CHECK(hit.pixel[1] < image->image_height);
                item = uint64_t(hit.pixel[1]) * image->image_width + hit.pixel[0];
            }
            CHECK(item == uint32_t(own - 1));
            ++picked;
        }
    MESSAGE(name, ": ", c.differ, " pixels differ, ", c.failed, " outside tolerance; ", picked, " picks");
    CHECK_MESSAGE(c.failed == 0, name, ": ", c.first);
}

}  // namespace

// Library pieces preserve the saved pictures and selected item IDs of each primitive fixture.
TEST_CASE("gfx/Metal scene view: library pieces match the saved scene pictures and preserve their item IDs") {
    ViewRig probe;
    if (!probe.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    std::vector<primitive_fixtures::Fixture> fixtures = primitive_fixtures::all();
    // The same camera with point sizes in pixels.
    primitive_fixtures::Fixture flat = primitive_fixtures::through();
    flat.name = "through-pieces";
    for (wire::Message& m : flat.messages)
        if (auto* u = std::get_if<wire::SceneUpdate>(&m)) u->params.erase("radius");
    fixtures.push_back(flat);
    // A textured mesh in the camera pixel space follows the same layer order.
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
        ViewRig pr;
        REQUIRE(pr.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str()));
        primitive_rows::Row rp;
        rp.fixture = f;
        REQUIRE_MESSAGE(rp.draw(*pr.view), rp.error);
        const SceneShapeCounts counts = pr.view->shape_counts();
        CHECK_MESSAGE(counts.pieces > 0, f.name, " draws no shape with a piece");
        compare_fixture(f.name, pr, int(primitive_fixtures::kSize), f.name == "images" ? 1 : 0);
    }
    {
        ViewRig pr;
        REQUIRE(pr.init(CALIPER_NET_SPACE_3D, "+z", ""));
        const scene_fixtures::SurfaceRowFixture fx;
        const DemoMessages dm(fx);
        Wire wp;
        draw_demo(pr, wp, dm);
        // Every shape of the demo, its surface's mesh of flat faces and indices too.
        const SceneShapeCounts counts = pr.view->shape_counts();
        CHECK(counts.pieces == 4);
        compare_fixture("demo", pr, int(scene_fixtures::SurfaceRowFixture::kSize), 0);
    }
}

// The highlight of a selection drawn by the pieces: the dim by image@1 and the selected rows by a
// second instance of points@1 or boxes@1, into the colour alone, against the reference's dim and
// highlight, under the rules of the comparison above; the selection over the whole picture holds
// every box, the points of radii in scene units of "edges" and the points flat over the picture
// of "through". The highlight writes no ID and no depth, thus the picks are those of the shapes.
TEST_CASE("gfx/Metal scene view: the selection highlight matches the saved scene pictures") {
    ViewRig probe;
    if (!probe.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    for (const primitive_fixtures::Fixture& f :
         {primitive_fixtures::boxes(), primitive_fixtures::edges(), primitive_fixtures::through()}) {
        CAPTURE(f.name);
        ViewRig pr;
        REQUIRE(pr.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str()));
        primitive_rows::Row rp;
        rp.fixture = f;
        REQUIRE_MESSAGE(rp.draw(*pr.view), rp.error);
        const std::vector<uint8_t> plain = pixels(*pr.view);
        const float S = float(primitive_fixtures::kSize);
        const std::string sp = pr.view->select_polygon({0, 0, S, 0, S, S, 0, S});
        CHECK(sp != "{}");
        std::string why;
        REQUIRE_MESSAGE(pr.view->render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why), why);
        CHECK(pr.view->highlighted() > 0);
        CHECK(pr.view->shape_counts().marks > 0);
        CHECK(pixels(*pr.view) != plain);
        compare_fixture(f.name + "-selected", pr, int(primitive_fixtures::kSize), 1);
        // No selection: the marks go, and the picture is the plain one again.
        CHECK(pr.view->select_polygon({1, 1, 2, 1, 2, 2}) == "{}");
        REQUIRE_MESSAGE(pr.view->render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why), why);
        CHECK(pr.view->highlighted() == 0);
        CHECK(pr.view->shape_counts().marks == 0);
        CHECK(pixels(*pr.view) == plain);
    }
}

TEST_CASE("gfx/Metal scene view: unsupported input reports the entity and reason") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    auto u = update("wide-line", CALIPER_NET_PART_LINES, 1, false);
    const float positions[] = {-1, 0, 0, 1, 0, 0};
    u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {2, 3}, positions, sizeof positions));
    u.params = Json::object();
    u.params.set("width", Json(129.0));
    std::string why;
    REQUIRE_MESSAGE(rig.view->apply(u, &why), why);
    REQUIRE_MESSAGE(rig.view->render(64, 64, 1.0f, 0xff000000u, &why), why);
    CHECK(rig.view->shape_counts().pieces == 0);
    REQUIRE(rig.view->notes().size() == 1);
    CHECK(rig.view->notes().front() == "wide-line: scene.piece_input: width above 128 pixels");
    u.params.set("width", Json(2.0));
    REQUIRE_MESSAGE(rig.view->apply(u, &why), why);
    REQUIRE_MESSAGE(rig.view->render(64, 64, 1.0f, 0xff000000u, &why), why);
    CHECK(rig.view->shape_counts().pieces == 1);
    CHECK(rig.view->notes().empty());
}

TEST_CASE("gfx/scene view: browser refusals report each entity and clear on scene reset") {
    Caps caps = RecordingBackend::default_caps();
    caps.backend = "webgpu";
    caps.bits &= ~kCapVertexStorage;
    caps.limits.max_storage_buffers_in_vertex_stage = 0;
    RecordingBackend gfx(caps);
    SceneView view(gfx, caliper_gfx::shaders::kMetalPrograms);
    std::string why;
    REQUIRE_MESSAGE(view.init(&why), why);
    view.describe(wire::SceneView{CALIPER_NET_SPACE_3D, "+z", ""});
    const float position[] = {0, 0, 0};
    for (const char* path : {"first-points", "second-points"}) {
        auto u = update(path, CALIPER_NET_PART_POINTS, 1, false);
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {1, 3}, position, sizeof position));
        REQUIRE_MESSAGE(view.apply(u, &why), why);
    }
    REQUIRE_MESSAGE(view.render(64, 64, 1.0f, 0xff000000u, &why), why);
    const auto runtimes = view.shape_runtimes();
    REQUIRE(runtimes.size() == 2);
    REQUIRE(view.notes().size() == 2);
    for (const auto* runtime : runtimes) {
        const auto status = runtime->status();
        CHECK(status.state == caliper_draw::DrawState::refused);
        CHECK(status.error.code == "program.browser_refused");
        CHECK(status.error.message.find("storage buffers in the vertex stage, the limit is 0") != std::string::npos);
        const std::string& program = runtime->instance().program;
        const std::string path = program.substr(6, program.rfind('#') - 6);
        CHECK((path == "first-points" || path == "second-points"));
        const std::string note = path + ": " + status.error.code + ": " + status.error.message;
        CHECK(std::find(view.notes().begin(), view.notes().end(), note) != view.notes().end());
    }
    const auto notes = view.notes();
    REQUIRE_MESSAGE(view.render(64, 64, 1.0f, 0xff000000u, &why), why);
    CHECK(view.notes() == notes);
    wire::SceneClear reset;
    reset.reset = true;
    REQUIRE_MESSAGE(view.apply(reset, &why), why);
    REQUIRE_MESSAGE(view.render(64, 64, 1.0f, 0xff000000u, &why), why);
    CHECK(view.notes().empty());
    CHECK(view.shape_runtimes().empty());
}

TEST_CASE("gfx/Metal scene view: an unchanged frame stays idle and a changed frame drops its pending pick") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    const scene_fixtures::SurfaceRowFixture fx;
    const DemoMessages dm(fx);
    Wire wire;
    draw_demo(rig, wire, dm);
    auto& view = *rig.view;
    const uint32_t size = fx.kSize;
    std::string why;
    const uint64_t frames = view.cost().frames;
    REQUIRE(view.render(size, size, 1, fx.kClear, &why));
    CHECK(view.cost().frames == frames);
    REQUIRE(view.request_pick(size * 0.5f, size * 0.5f, &why));
    view.orbit(2, 0);
    REQUIRE(view.render(size, size, 1, fx.kClear, &why));
    Hit hit;
    CHECK(take_any_pick(*rig.gfx, view, &hit) == PickResult::Dropped);
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
            // (resource id, stride, bytes, width, height) of each resource, in the order they go before the
            // update; width and height of image@1's pixels, an rgba8unorm texture, and 0 for a buffer.
            std::vector<std::tuple<std::string, uint32_t, std::vector<uint8_t>, uint32_t, uint32_t>> res;
            wire::SceneUpdate pu = *cu;
            pu.program = id;
            pu.piece = Json::object();
            pu.piece.set("version", 1);
            pu.resources = Json::object();
            if (points) {
                res.emplace_back("points", 32u, pack_points(*cu), 0u, 0u);
                pu.piece.set("name", "points");
                for (const char* name : {"positions", "colors", "values", "class_ids", "radii"})
                    if (cu->array(name)) pu.names.push_back(name);
            } else {
                auto [pixels, info] = pack_image(*cu);
                const wire::Section* px = cu->array("pixels");
                res.emplace_back("pixels", 4u, std::move(pixels), uint32_t(px->shape[1]), uint32_t(px->shape[0]));
                res.emplace_back("info", 80u, std::move(info), 0u, 0u);
                pu.piece.set("name", "image");
                pu.names.push_back("pixels");
            }
            pu.arrays.clear();
            for (auto& [rid, stride, bytes, width, height] : res) {
                wire::Resource r;
                r.program = id;
                r.resource = rid;
                r.step = 0;
                r.count = uint32_t(bytes.size() / stride);
                r.stride = stride;
                if (width) {
                    r.width = width;
                    r.height = height;
                }
                r.data = wire::Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8, {int64_t(bytes.size())},
                                             bytes.data(), bytes.size());
                pu.resources.set(rid, Json(uint64_t(r.count)));
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
    return view.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, why);
}

}  // namespace

// The piece form of points and color images (wire protocol 3.2, capability "scene_pieces"): every fixture
// whose points and color images arrive as points@1's and image@1's resources, written as
// caliper._scene_pieces writes them, draws the same pixels as
// from the per-shape part, both with the pieces (the received bytes against the converter's) and with
// the reference draw (the batch rebuilt from the bytes against the decoded one), and every pick on a
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
        {
            ViewRig part, piece;
            REQUIRE(part.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str()));
            REQUIRE(piece.init(f.info.space.c_str(), f.info.up.c_str(), f.info.units.c_str()));
            primitive_rows::Row row;
            row.fixture = f;
            REQUIRE_MESSAGE(row.draw(*part.view), row.error);
            std::string why;
            REQUIRE_MESSAGE(draw_piece_form(f, *piece.view, &why), why);
            {
                const SceneShapeCounts a = part.view->shape_counts(), b = piece.view->shape_counts();
                CHECK(a.received == 0);
                CHECK(b.pieces == a.pieces);
                MESSAGE("[scene-piece-form] ", f.name, ": ", b.received, " of ", points, " points and color image batches drawn from the received bytes, ",
                        b.pieces, " shapes by pieces");
            }
            const std::vector<uint8_t> pa = pixels(*part.view), pb = pixels(*piece.view);
            size_t differ = 0;
            for (size_t i = 0; i < pa.size(); i += 4) differ += std::memcmp(&pa[i], &pb[i], 4) != 0;
            capture_shapes(std::string("form-") + f.name, pb, pa, int(primitive_fixtures::kSize));
            MESSAGE("[scene-piece-form] ", f.name, " by pieces", ": ", differ,
                    " of ", pa.size() / 4, " pixels differ");
            CHECK_MESSAGE(differ == 0, f.name);
            // Picks at the pixels whose ID and its 3 x 3 neighbours are one item of a piece: at most 96, spread
            // over the picture, each giving the same entity, primitive and item in both views.
            const int S = int(primitive_fixtures::kSize);
            const std::vector<uint64_t> ids = program_ids(*piece.gfx, *piece.view);
            std::vector<std::pair<int, int>> at;
            for (int y = 1; y + 1 < S; ++y)
                for (int x = 1; x + 1 < S; ++x) {
                    const uint64_t own = ids[size_t(y) * S + x];
                    bool inside = own != 0;
                    for (int dy = -1; dy <= 1 && inside; ++dy)
                        for (int dx = -1; dx <= 1 && inside; ++dx) inside = ids[size_t(y + dy) * S + x + dx] == own;
                    if (inside) at.emplace_back(x, y);
                }
            size_t asked = 0, agreed = 0;
            const size_t step = std::max<size_t>(1, at.size() / 96);
            for (size_t i = 0; i < at.size(); i += step) {
                std::string why2;
                caliper_scene::Hit ha, hb;
                REQUIRE_MESSAGE(piece.view->request_pick(at[i].first + 0.5f, at[i].second + 0.5f, &why2), why2);
                const auto ra = take_any_pick(*piece.gfx, *piece.view, &ha);
                REQUIRE_MESSAGE(part.view->request_pick(at[i].first + 0.5f, at[i].second + 0.5f, &why2), why2);
                const auto rb = take_any_pick(*part.gfx, *part.view, &hb);
                ++asked;
                if (ra == rb && (ra != caliper_scene::PickResult::Hit ||
                                 (piece.view->tree().path(ha.entity) == part.view->tree().path(hb.entity) &&
                                  ha.primitive == hb.primitive && ha.item == hb.item && ha.has_pixel == hb.has_pixel &&
                                  (!ha.has_pixel || (ha.pixel[0] == hb.pixel[0] && ha.pixel[1] == hb.pixel[1])))))
                    ++agreed;
            }
            MESSAGE("[scene-piece-form] ", f.name, ": picks at item interiors ", agreed, " of ", asked, " agree");
            CHECK(agreed == asked);
            CHECK(asked > 0);
        }
    }
    CHECK(with_points >= 3);
}
