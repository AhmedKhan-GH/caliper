// The spec runtime tests of scene entities: offscreen captures and picks
// (design/instrument-interface.md, Scene Entities, Pick Fields and Data Box).
//
// A scene panel draws a surface with a hole, a loss line on the back wall of its data box and
// points, from resolved arrays given by hand, into the scene view that the panel owns on the
// offscreen canvas's device. The tests check what the entities build, that the wide flat box
// fills the panel's width, that a pick at the pixel of a known cell gives its row and column,
// and that the starting camera of view_camera and of a saved view applies.
// CALIPER_SPEC_TEST_DUMP=<folder> keeps each picture.
#include <doctest/doctest.h>

#include "spec_runtime/spec_entities.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"

#include "export_service.h"
#include <caliper/embed.h>
#if defined(__APPLE__)
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#endif
#include "live/live_scene.h"
#include "live/live_views_scene.h"
#include "net/wire.h"
#include "scene/camera.h"

#include "spec_test_canvas.h"
#include "spec_test_data.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/programs/pieces.h>

#include "draw/draw_runtime.h"
#include "gfx/recording_backend.h"
#include "../../../../tests/gfx/example_pieces.h"   // the attention arcs piece and its resources

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <variant>
#include <map>
#include <memory>
#include <cstdlib>
#include <string>
#include <vector>

using namespace spec_test;
using caliper_builder::EntityData;
using caliper_host::EntityScene;
using caliper_host::SpecOffscreen;
using caliper_host::SpecPanelSource;
using caliper_host::SpecLayoutPanel;
using caliper_host::SpecRuntime;

namespace {

constexpr int kRows = 128, kCols = 16;

const char* const kSpec = R"(caliper_spec = 2
[instrument]
id = "terrain"
[[source]]
id = "act"
catalog = "live:act"
[[source]]
id = "loss"
catalog = "metrics"
[[source]]
id = "marks"
catalog = "live:marks"
[[panel]]
id = "world"
primitive = "scene"
title = "Activation Terrain"
options = { up = "+z" }
[[panel.entity]]
path = "terrain"
draw = "surface"
bind = "act"
options = { colormap = "viridis" }
[[panel.entity]]
path = "loss"
draw = "lines"
bind = { x = "loss.step", z = "loss.value" }
options = { mode = "strip", y = "max", own = ["z"], width_px = 2.0 }
[[panel.entity]]
path = "marks"
draw = "points"
bind = "marks"
options = { colormap = "magma", size_px = 6.0 }
)";

double height_at(int r, int c) {
    if (r >= 84 && r < 96 && c >= 1 && c < 6) return NAN;   // the hole, on the slope that faces the camera
    return 1.2 + 0.8 * std::sin(r / 14.0) * std::cos(c / 3.0) + r / 160.0;
}

// The resolved arrays of the three entities, as the resolver gives them for a panel with entities.
PanelData world_data(int rows = kRows) {
    PanelData d;
    d.panel = "world";
    d.primitive = spec::Primitive::scene;
    std::vector<float> h(static_cast<size_t>(rows) * kCols);
    std::vector<float> x(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        x[size_t(r)] = float(r * 16);
        for (int c = 0; c < kCols; ++c) h[size_t(r) * kCols + size_t(c)] = float(height_at(r, c));
    }
    EntityData terrain;
    terrain.path = "terrain";
    terrain.generation = 1;
    terrain.slots = {f32("height", "act", {rows, kCols}, h), f32("x", "act", {rows}, x)};
    // The rows come from a window: their steps travel with the height.
    terrain.slots[0].steps = f32("steps", "act", {rows}, x).tensor;
    std::vector<double> step, loss;
    for (int r = 0; r < rows; ++r) {
        step.push_back(r * 16.0);
        loss.push_back(2.0 * std::exp(-r / 40.0) + 0.2);
    }
    EntityData line;
    line.path = "loss";
    line.generation = 1;
    line.slots = {column("x", "loss.step", step), column("z", "loss.value", loss)};
    std::vector<float> pos, val;
    for (int k = 0; k < 24; ++k) {
        const int r = (k * 37) % rows, c = (k * 5) % kCols;
        const double hh = std::isfinite(height_at(r, c)) ? height_at(r, c) : 1.0;
        pos.insert(pos.end(), {float(r * 16), float(c), float(hh + 0.15)});
        val.push_back(float(k));
    }
    EntityData marks;
    marks.path = "marks";
    marks.generation = 1;
    marks.slots = {f32("positions", "marks", {24, 3}, pos), f32("values", "marks", {24}, val)};
    d.entities = {terrain, line, marks};
    d.generation = 1;
    return d;
}

struct Source : SpecPanelSource {
    int rows = kRows;
    uint64_t rev = 1;
    PanelData resolve(const spec::Spec&, std::string_view) override { return world_data(rows); }
    uint64_t revision() const override { return rev; }
    uint64_t generation(const spec::Spec&, std::string_view) override { return 1; }
};

struct Events : caliper_builder::EventSink {
    std::vector<caliper_builder::Event> got;
    std::uint64_t push(caliper_builder::Event e) override {
        got.push_back(std::move(e));
        return got.size();
    }
};

// The runtime's scene layer on the canvas's own device and queue.
SpecRuntime::SceneLayer canvas_layer(SpecOffscreen* c) {
    return [c](std::unique_ptr<caliper_gfx::Backend>* owned, caliper_gfx::Backend** gfx,
               const caliper_gfx::ProgramDesc** programs, std::string* why) {
#if defined(__APPLE__)
        *owned = caliper_gfx::make_metal_backend(c->native_device(), c->native_queue());
        if (!*owned) {
            *why = "no Metal backend";
            return false;
        }
        *gfx = owned->get();
        *programs = caliper_gfx::shaders::kMetalPrograms;
        return true;
#else
        (void)c;
        (void)owned;
        (void)gfx;
        (void)programs;
        *why = "the scene layer of these tests is Metal's";
        return false;
#endif
    };
}

void dump(const Image& img, const char* name) {
    if (const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP"); dir && *dir && !img.px.empty())
        caliper_host::export_write_png_atomic(std::string(dir) + "/" + name, img.px.data(), uint32_t(img.w),
                                              uint32_t(img.h));
}

// Pixels of colour, not grey: the colormapped surface, line and points.
bool colourful(uint32_t p) {
    const int r = p & 0xff, g = (p >> 8) & 0xff, b = (p >> 16) & 0xff;
    return std::abs(r - g) > 24 || std::abs(g - b) > 24 || std::abs(r - b) > 24;
}

}  // namespace

TEST_CASE("entities: a surface with a hole, a line on the back wall and points build in their data box") {
    const spec::Spec s = parse(kSpec);
    const EntityScene scene = caliper_host::build_entity_scene(s.panels[0], world_data());
    REQUIRE(scene.draws.size() == 3);
    const caliper_host::EntityDraw& t = scene.draws[0];
    CHECK(t.path == "spec/terrain");
    CHECK_FALSE(t.error);
    CHECK(t.rows == kRows);
    CHECK(t.cols == kCols);
    // Two triangles a cell, without the 12 x 5 vertices of the hole and the cells that touch them.
    int holes = 0;
    for (int r = 0; r + 1 < kRows; ++r)
        for (int c = 0; c + 1 < kCols; ++c)
            holes += !std::isfinite(height_at(r, c)) || !std::isfinite(height_at(r + 1, c)) ||
                     !std::isfinite(height_at(r, c + 1)) || !std::isfinite(height_at(r + 1, c + 1));
    CHECK(t.indices.size() == size_t(((kRows - 1) * (kCols - 1) - holes) * 6));
    // The box: 2,032 steps by 15 tokens by about 2 norm units is a strip; auto keeps every side
    // at least a quarter of the longest, which is 2 scene units.
    CHECK(scene.box.half[0] == doctest::Approx(1.0));
    CHECK(scene.box.half[1] == doctest::Approx(0.25));
    CHECK(scene.box.half[2] == doctest::Approx(0.25));
    CHECK(scene.box.lo[0] == doctest::Approx(0.0));
    CHECK(scene.box.hi[0] == doctest::Approx(2032.0));
    // The loss line stands on the back wall (y = max) and spans the box's height by its own range.
    const caliper_host::EntityDraw& l = scene.draws[1];
    CHECK_FALSE(l.error);
    CHECK(l.strip);
    CHECK(l.depth_offset == 1);
    float zlo = 9, zhi = -9;
    for (size_t i = 0; i < l.positions.size(); i += 3) {
        CHECK(l.positions[i + 1] == doctest::Approx(0.25));
        zlo = std::min(zlo, l.positions[i + 2]);
        zhi = std::max(zhi, l.positions[i + 2]);
    }
    CHECK(zlo == doctest::Approx(-0.25));
    CHECK(zhi == doctest::Approx(0.25));
    CHECK(scene.draws[2].colormap == 1);   // magma
    CHECK(scene.draws[2].size_px == 6.0f);
    // Picks give the documented fields in data units.
    const auto pf = caliper_host::entity_pick_fields(scene, "spec/terrain", 20 * kCols + 3);
    REQUIRE(pf);
    CHECK(pf->entity == "terrain");
    CHECK(pf->row == 20);
    CHECK(pf->column == 3);
    CHECK(pf->step == 320);
    REQUIRE(pf->position);
    CHECK((*pf->position)[0] == doctest::Approx(320.0));
    CHECK((*pf->position)[2] == doctest::Approx(height_at(20, 3)).epsilon(1e-6));
    REQUIRE(pf->value);
    const auto lp = caliper_host::entity_pick_fields(scene, "spec/loss", 7);
    REQUIRE(lp);
    CHECK(lp->index == 7);
    CHECK((*lp->position)[0] == doctest::Approx(112.0));
    CHECK((*lp->position)[2] == doctest::Approx(2.0 * std::exp(-7 / 40.0) + 0.2).epsilon(1e-6));
    CHECK_FALSE(caliper_host::entity_pick_fields(scene, "spec/none", 0));
}

TEST_CASE("entities: a wrong shape or a length that differs shows nothing for that entity") {
    const spec::Spec s = parse(kSpec);
    PanelData d = world_data();
    d.entities[0].slots[0] = f32("height", "act", {kRows * kCols}, std::vector<float>(kRows * kCols, 1.0f));
    d.entities[1].slots[1] = column("z", "loss.value", std::vector<double>(5, 1.0));
    const EntityScene scene = caliper_host::build_entity_scene(s.panels[0], d);
    CHECK(scene.draws[0].error.code == "entity.shape");
    CHECK(scene.draws[1].error.code == "entity.length");
    CHECK_FALSE(scene.draws[2].error);
}

TEST_CASE("entities: the scene panel opens with its wide box framed, and a pick at a cell gives its row and column") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    Source source;
    Events events;
    const spec::Spec s = parse(kSpec);
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_scene_layer(canvas_layer(c));
    rt->set_source(&source);
    rt->set_instruments(&events, nullptr);
    rt->set_spec(s);
    const int W = 1200, H = 520;
    Image img = render(W, H, [&](const ImRect& area) { rt->draw(area, c->look()); }, 4);
    dump(img, "entities-terrain-framed.png");
    REQUIRE(rt->layout().panels.size() == 1);
    const SpecLayoutPanel& lp = rt->layout().panels[0];
    caliper_live_views::SceneView* view = rt->entity_view("world");
    REQUIRE(view != nullptr);
    CHECK(view->notes().empty());
    CHECK(view->counts().entities == 4);   // three entities and the walls of the box
    CHECK(view->box_fit());
    CHECK(view->fitted());
    // The picture covers most of the panel's width: the box is framed, not a strip in its middle.
    // The box and its walls (anything brighter than the ground) span most of the width, and the
    // coloured entities most of that.
    const auto n = img.count(colourful, int(lp.rect.x), int(lp.rect.y), int(lp.rect.right()), int(lp.rect.bottom()));
    REQUIRE(n.n > 2000);
    const int top = int(lp.rect.y) + 30;   // below the panel's title strip
    const uint32_t ground = img.rgb(int(lp.rect.x) + 4, int(lp.rect.bottom()) - 4);
    auto lit = [&](uint32_t p) {
        for (int k = 0; k < 24; k += 8)
            if (std::abs(int((p >> k) & 0xff) - int((ground >> k) & 0xff)) > 20) return true;
        return false;
    };
    int bx0 = W, bx1 = 0, cx0 = W, cx1 = 0, by0 = H, by1 = 0;
    for (int y = top; y < int(lp.rect.bottom()); ++y)
        for (int x = int(lp.rect.x); x < int(lp.rect.right()); ++x) {
            const uint32_t p = img.rgb(x, y);
            if (lit(p)) bx0 = std::min(bx0, x), bx1 = std::max(bx1, x), by0 = std::min(by0, y), by1 = std::max(by1, y);
            if (colourful(p)) cx0 = std::min(cx0, x), cx1 = std::max(cx1, x);
        }
    CHECK(double(bx1 - bx0) > 0.65 * lp.rect.w);
    CHECK(double(by1 - by0) > 0.75 * (lp.rect.bottom() - top));
    CHECK(double(cx1 - cx0) > 0.6 * lp.rect.w);

    // The pixel of cell (row 110, column 1), on the front edge that faces the camera, at the
    // view's camera: a pick there gives that cell.
    const EntityScene& scene = caliper_host::spec_scene_content(*rt->entity_scene("world"));
    const caliper_host::EntityDraw& t = scene.draws[0];
    const size_t vi = size_t(110 * kCols + 1);
    const caliper_scene::CameraMatrices cm =
        caliper_scene::camera_matrices(view->effective_camera(), view->tree(), view->width(), view->height());
    REQUIRE(cm.ok);
    double h[4];
    caliper_scene::transform_h(cm.proj * cm.view,
                               {t.positions[vi * 3], t.positions[vi * 3 + 1], t.positions[vi * 3 + 2]}, h);
    const double u = (h[0] / h[3] + 1) * 0.5, v = (1 - h[1] / h[3]) * 0.5;
    REQUIRE(rt->pick_scene("world", float(u), float(v)));
    for (int k = 0; k < 6 && events.got.empty(); ++k)
        render(W, H, [&](const ImRect& area) { rt->draw(area, c->look()); }, 1);
    REQUIRE_FALSE(events.got.empty());
    const nlohmann::json& f = events.got.back().fields;
    CAPTURE(f.dump());
    REQUIRE(f.contains("fields"));
    CHECK(f["fields"]["entity"] == "terrain");
    CHECK(std::abs(f["fields"]["row"].get<int>() - 110) <= 1);
    CHECK(std::abs(f["fields"]["column"].get<int>() - 1) <= 1);
    CHECK(f["fields"]["step"].get<int>() == 16 * f["fields"]["row"].get<int>());
    CHECK(f["fields"].contains("x"));
    CHECK(f["fields"].contains("value"));
    c->with_context([&] { rt.reset(); });
}

TEST_CASE("entities: view_camera and a saved view set the starting camera, and a reset returns to it") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    Source source;
    std::string text = kSpec;
    const std::string opts = R"(options = { up = "+z" })";
    text.replace(text.find(opts), opts.size(),
                 R"(options = { up = "+z", view_camera = "side", views = [{ name = "side", target = [0.0, 0.0, 0.0], distance = 3.2, azimuth = -70.0, elevation = 28.0 }] })");
    const spec::Spec s = parse(text);
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_scene_layer(canvas_layer(c));
    rt->set_source(&source);
    rt->set_spec(s);
    const Image img = render(900, 520, [&](const ImRect& area) { rt->draw(area, c->look()); }, 4);
    dump(img, "entities-saved-view.png");
    caliper_live_views::SceneView* view = rt->entity_view("world");
    REQUIRE(view != nullptr);
    const auto* o = std::get_if<caliper_scene::Orbit>(&view->camera());
    REQUIRE(o != nullptr);
    CHECK(o->distance == doctest::Approx(3.2));
    CHECK(o->azimuth == doctest::Approx(-70.0 * caliper_scene::kPi / 180.0));
    CHECK(o->elevation == doctest::Approx(28.0 * caliper_scene::kPi / 180.0));
    CHECK(o->up == caliper_scene::Axis::PosZ);
    // The person turns the view; a reset returns to the saved view.
    view->orbit(40.0f, 10.0f);
    CHECK(view->moved());
    view->reset();
    render(900, 520, [&](const ImRect& area) { rt->draw(area, c->look()); }, 2);
    o = std::get_if<caliper_scene::Orbit>(&view->camera());
    REQUIRE(o != nullptr);
    CHECK(o->azimuth == doctest::Approx(-70.0 * caliper_scene::kPi / 180.0));
    CHECK(o->distance == doctest::Approx(3.2));
    c->with_context([&] { rt.reset(); });
}

TEST_CASE("entities: a line that lies on the surface wins by its depth offset, and loses when pushed back") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    // The ridge of column 8, drawn on the surface itself: x, y and z from the terrain's own data.
    auto ridge_scene = [](int offset) {
        std::string text = kSpec;
        const std::string loss = "[[panel.entity]]\npath = \"loss\"";
        text = text.substr(0, text.find(loss));
        text += "[[panel.entity]]\npath = \"ridge\"\ndraw = \"lines\"\nbind = { x = \"loss.step\", y = \"loss.token\", z = "
                "\"loss.value\" }\noptions = { mode = \"strip\", width_px = 3.0, depth_offset = " +
                std::to_string(offset) + " }\n";
        const spec::Spec s = parse(text);
        PanelData d = world_data();
        std::vector<double> step, token, z;
        for (int r = 0; r < kRows; ++r)
            if (std::isfinite(height_at(r, 8))) {
                step.push_back(r * 16.0);
                token.push_back(8.0);
                z.push_back(height_at(r, 8));
            }
        EntityData ridge;
        ridge.path = "ridge";
        ridge.generation = 1;
        ridge.slots = {column("x", "loss.step", step), column("y", "loss.token", token), column("z", "loss.value", z)};
        d.entities = {d.entities[0], ridge};
        return caliper_host::build_entity_scene(s.panels[0], d);
    };
    auto light = [](uint32_t p) {
        const int r = p & 0xff, g = (p >> 8) & 0xff, b = (p >> 16) & 0xff;
        return r > 190 && g > 190 && b > 190;
    };
    size_t seen[2] = {0, 0};
    const int offsets[2] = {1, -8};
    for (int k = 0; k < 2; ++k) {
        caliper_host::EntityScene scene = ridge_scene(offsets[k]);
        scene.walls = false;
        REQUIRE_FALSE(scene.draws[1].error);
        CHECK(scene.draws[1].depth_offset == offsets[k]);
        std::unique_ptr<caliper_gfx::Backend> gfx;
        caliper_gfx::Backend* backend = nullptr;
        const caliper_gfx::ProgramDesc* programs = nullptr;
        std::string why;
        REQUIRE_MESSAGE(canvas_layer(c)(&gfx, &backend, &programs, &why), why);
        caliper_live_views::SceneView view(*backend, programs);
        REQUIRE_MESSAGE(view.init(&why), why);
        view.set_box_fit(true);
        std::vector<std::string> drawn_paths;
        caliper_host::apply_entity_scene(view, scene, &drawn_paths);
        REQUIRE_MESSAGE(view.render(900, 420, 1.0f, 0xff161616u, &why), why);
        Image img;
        img.w = 900;
        img.h = 420;
        REQUIRE_MESSAGE(view.read_pixels(&img.px, &why), why);
        seen[k] = img.count(light).n;
        if (k == 0) dump(img, "entities-ridge-depth-offset.png");
    }
    MESSAGE("ridge pixels with offset 1: " << seen[0] << ", with offset -8: " << seen[1]);
    CHECK(seen[0] > 400);
    CHECK(seen[0] > 3 * seen[1]);
}

// The entities drawn by the library pieces against the generic draw through the scene view's switch
// (live_scene.h): the surface with its hole and its window's steps by surface@1 over its
// grid, the strip of the loss by path@1, the points by points@1 and the walls of the box by lines@1.
// The pictures agree under the rules of the pieces' match tests (an interior pixel of the pieces'
// IDs within 1 level, as surface@1's match allows, every other pixel within 1 level of a generic
// pixel of its 3 x 3 neighbourhood), and a pick at an interior pixel gives the same entity, item and
// pick fields in both views.
TEST_CASE("entities: the library pieces draw the surface, the path, the points and the walls as generic scene pieces do") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    const spec::Spec s = parse(kSpec);
    const EntityScene scene = caliper_host::build_entity_scene(s.panels[0], world_data());
    const int W = 640, H = 400;
    struct Drawn {
        std::unique_ptr<caliper_gfx::Backend> gfx;
        caliper_gfx::Backend* backend = nullptr;
        std::unique_ptr<caliper_live_views::SceneView> view;
        Image img;
        std::vector<uint64_t> ids;
    };
    auto draw = [&](bool specialized, Drawn* d) {
        const caliper_gfx::ProgramDesc* programs = nullptr;
        std::string why;
        REQUIRE_MESSAGE(canvas_layer(c)(&d->gfx, &d->backend, &programs, &why), why);
        d->view = std::make_unique<caliper_live_views::SceneView>(*d->backend, programs);
        REQUIRE_MESSAGE(d->view->init(&why), why);
        d->view->set_box_fit(true);
        std::vector<std::string> drawn_paths;
        const std::vector<std::string> paths = {"loss"};
        caliper_host::apply_entity_scene(*d->view, scene, &drawn_paths, &paths);
        if (!specialized) {
            for (auto index : d->view->tree().path_order()) {
                const auto entity = d->view->tree().entity_at(index);
                const auto path = d->view->tree().path(entity);
                for (int part = 1; part <= caliper_scene::kPrimitiveCount; ++part)
                    d->view->set_piece_batch(path, caliper_scene::Primitive(part), nullptr);
            }
        }
        REQUIRE_MESSAGE(d->view->render(W, H, 1.0f, 0xff161616u, &why), why);
        d->img.w = W;
        d->img.h = H;
        REQUIRE_MESSAGE(d->view->read_pixels(&d->img.px, &why), why);
        d->ids.assign(size_t(W) * H, 0);
        if (d->view->id_target()) {
            const caliper_gfx::Readback rb = d->backend->read_texture(d->view->id_target(), 0, 0, W, H);
            std::vector<uint8_t> bytes;
            REQUIRE(rb);
            REQUIRE(d->backend->wait(d->backend->last_use(d->view->id_target())));
            REQUIRE(d->backend->take_readback(rb, &bytes));
            for (size_t i = 0; i < d->ids.size() && i * 8 + 8 <= bytes.size(); ++i) {
                uint32_t pair[2];
                std::memcpy(pair, bytes.data() + i * 8, 8);
                d->ids[i] = pair[0] == 0 ? 0 : (uint64_t(pair[0]) << 32 | pair[1]) + 1;
            }
        }
    };
    Drawn pieces, generic;
    draw(true, &pieces);
    draw(false, &generic);
    const caliper_live_views::SceneShapeCounts counts = pieces.view->shape_counts();
    CHECK(counts.pieces == 4);
    dump(pieces.img, "entities-pieces.png");
    dump(generic.img, "entities-generic.png");

    auto px = [&](const Image& img, int x, int y) {
        uint32_t v;
        std::memcpy(&v, img.px.data() + (size_t(y) * W + x) * 4, 4);
        return v;
    };
    auto gap = [](uint32_t a, uint32_t b) {
        int g = 0;
        for (int k = 0; k < 4; ++k) g = std::max(g, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
        return g;
    };
    auto interior = [&](int x, int y, int r) {
        const uint64_t own = pieces.ids[size_t(y) * W + x];
        if (own == 0) return false;
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
                if (pieces.ids[size_t(std::clamp(y + dy, 0, H - 1)) * W + std::clamp(x + dx, 0, W - 1)] != own) return false;
        return true;
    };
    size_t differ = 0, failed = 0, inside = 0;
    int max_in = 0, max_edge = 0;
    std::string first;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint32_t p = px(pieces.img, x, y), b = px(generic.img, x, y);
            if (p != b) ++differ;
            int g = gap(p, b);
            const bool in = interior(x, y, 1);
            if (in) {
                ++inside;
                max_in = std::max(max_in, g);
            } else {
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        g = std::min(g, gap(p, px(generic.img, std::clamp(x + dx, 0, W - 1), std::clamp(y + dy, 0, H - 1))));
                max_edge = std::max(max_edge, g);
            }
            if (g > 1 && failed++ == 0)
                first = "pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") " + (in ? "interior" : "edge") +
                        " gap " + std::to_string(g);
        }
    MESSAGE("entities on pieces: " << differ << " of " << W * H << " pixels differ, " << inside
                                   << " interior pixels with a largest gap of " << max_in << ", largest edge gap "
                                   << max_edge << ", " << failed << " outside the rules");
    CHECK_MESSAGE(failed == 0, first);
    CHECK(inside > 4000);   // the surface's IDs change from vertex to vertex

    // Picks at interior pixels on a grid: the same hit, and the same pick fields of the entity.
    const EntityScene& sc = scene;
    size_t asked = 0, agreed = 0;
    std::string first_pick;
    auto take = [](caliper_gfx::Backend& g, caliper_live_views::SceneView& v, caliper_scene::Hit* hit) {
        caliper_scene::PickResult r = caliper_scene::PickResult::Pending;
        for (int k = 0; k < 10000 && r == caliper_scene::PickResult::Pending; ++k) {
            // The pieces' pick reads the ID target and the depth: each readback is a use of its texture.
            if (v.id_target()) g.wait(g.last_use(v.id_target()));
            if (v.depth_target()) g.wait(g.last_use(v.depth_target()));
            g.poll();
            r = v.poll_pick(hit);
        }
        return r;
    };
    for (int y = 3; y < H; y += 5)
        for (int x = 3; x < W; x += 5) {
            if (!interior(x, y, 1)) continue;
            std::string why;
            caliper_scene::Hit hp, hb;
            REQUIRE(pieces.view->request_pick(x + 0.5f, y + 0.5f, &why));
            const auto rp = take(*pieces.backend, *pieces.view, &hp);
            REQUIRE(generic.view->request_pick(x + 0.5f, y + 0.5f, &why));
            const auto rb = take(*generic.backend, *generic.view, &hb);
            ++asked;
            bool same = rp == caliper_scene::PickResult::Hit && rb == rp;
            if (same) {
                const std::string path = pieces.view->tree().path(hp.entity);
                same = path == generic.view->tree().path(hb.entity) && hp.item == hb.item && hp.primitive == hb.primitive;
                const auto fp = caliper_host::entity_pick_fields(sc, path, hp.item);
                const auto fb = caliper_host::entity_pick_fields(sc, path, hb.item);
                same = same && fp.has_value() == fb.has_value();
            }
            if (same) ++agreed;
            else if (first_pick.empty())
                first_pick = "at (" + std::to_string(x) + ", " + std::to_string(y) + "): pieces " +
                             (rp == caliper_scene::PickResult::Hit
                                  ? pieces.view->tree().path(hp.entity) + " " + std::to_string(hp.item)
                                  : "result " + std::to_string(int(rp))) +
                             ", generic " +
                             (rb == caliper_scene::PickResult::Hit
                                  ? generic.view->tree().path(hb.entity) + " " + std::to_string(hb.item)
                                  : std::string("no hit"));
        }
    MESSAGE("entities on pieces: picks " << agreed << " of " << asked << " agree");
    CHECK(asked > 20);
    CHECK_MESSAGE(agreed == asked, first_pick);
    pieces.view.reset();
    generic.view.reset();
}

TEST_CASE("entities: view_camera angles open the panel at those angles with its box framed") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    Source source;
    std::string text = kSpec;
    const std::string opts = R"(options = { up = "+z" })";
    text.replace(text.find(opts), opts.size(), R"(options = { up = "+z", view_camera = { azimuth = -62.0, elevation = 32.0 } })");
    const spec::Spec s = parse(text);
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_scene_layer(canvas_layer(c));
    rt->set_source(&source);
    rt->set_spec(s);
    const Image img = render(1000, 520, [&](const ImRect& area) { rt->draw(area, c->look()); }, 4);
    dump(img, "entities-view-camera.png");
    caliper_live_views::SceneView* view = rt->entity_view("world");
    REQUIRE(view != nullptr);
    const auto* o = std::get_if<caliper_scene::Orbit>(&view->camera());
    REQUIRE(o != nullptr);
    CHECK(o->azimuth == doctest::Approx(-62.0 * caliper_scene::kPi / 180.0));
    CHECK(o->elevation == doctest::Approx(32.0 * caliper_scene::kPi / 180.0));
    // The distance and the target come from the box fit: the box fills the picture.
    const caliper_scene::CameraMatrices cm =
        caliper_scene::camera_matrices(view->effective_camera(), view->tree(), view->width(), view->height());
    REQUIRE(cm.ok);
    const caliper_host::EntityBox& box = caliper_host::spec_scene_content(*rt->entity_scene("world")).box;
    double x0 = 9, x1 = -9, y0 = 9, y1 = -9;
    for (int k = 0; k < 8; ++k) {
        double h[4];
        caliper_scene::transform_h(cm.proj * cm.view,
                                   {(k & 1) ? box.half[0] : -box.half[0], (k & 2) ? box.half[1] : -box.half[1],
                                    (k & 4) ? box.half[2] : -box.half[2]},
                                   h);
        x0 = std::min(x0, h[0] / h[3]), x1 = std::max(x1, h[0] / h[3]);
        y0 = std::min(y0, h[1] / h[3]), y1 = std::max(y1, h[1] / h[3]);
    }
    CHECK(std::max(x1 - x0, y1 - y0) > 1.8);   // one side spans the picture, within its margin
    CHECK(x1 <= 1.0);
    CHECK(x0 >= -1.0);
    CHECK(y1 <= 1.0);
    CHECK(y0 >= -1.0);
    c->with_context([&] { rt.reset(); });
}

// ---- Programs of pieces in a scene panel (design/programmable-graphics.md, Picking and Text) --------

namespace {

namespace ex = caliper_test_pieces;
namespace cp = caliper_builder::programs;

// The attention arcs piece with its hash, as the spec holds it inline.
spec::Piece hashed_arcs() {
    spec::Piece p = ex::ring_arcs();
    p.hash = cp::piece_hash(p);
    return p;
}

template <typename T>
caliper_draw::ResourceVersion version_of(const char* id, const std::vector<T>& v, std::int64_t step) {
    caliper_draw::ResourceVersion r;
    r.resource = id;
    r.step = step;
    r.count = std::uint32_t(v.size());
    r.bytes = reinterpret_cast<const std::uint8_t*>(v.data());
    r.size = v.size() * sizeof(T);
    return r;
}

template <typename T>
void keep_version(caliper_builder::StepHistory& h, const char* resource, const std::vector<T>& v, std::int64_t step) {
    caliper_builder::ResourceVersion r;
    r.step = step;
    r.count = std::uint32_t(v.size());
    r.stride = std::uint32_t(sizeof(T));
    const auto* b = reinterpret_cast<const std::uint8_t*>(v.data());
    r.bytes = std::make_shared<const std::vector<std::uint8_t>>(b, b + v.size() * sizeof(T));
    h.append_resource("tower_arcs", resource, 4, {}, std::move(r));
}

// One frame of a runtime on the recording backend, which updates the versions it shows.
struct RecordingHost {
    caliper_gfx::RecordingBackend gfx;
    caliper_draw::HostTarget target;
    RecordingHost() {
        caliper_gfx::TextureDesc c;
        c.width = 64;
        c.height = 48;
        c.format = caliper_gfx::Format::RGBA8Unorm;
        c.usage = caliper_gfx::kRenderTarget | caliper_gfx::kSampled;
        target.color = gfx.create_texture(c);
        c.format = caliper_gfx::Format::RG32Uint;
        c.usage = caliper_gfx::kRenderTarget;
        target.id = gfx.create_texture(c);
        c.format = caliper_gfx::Format::Depth32Float;
        target.depth = gfx.create_texture(c);
        target.width = 64;
        target.height = 48;
    }
    void frame(caliper_draw::DrawRuntime& rt) {
        std::vector<caliper_gfx::Frame> frames;
        if (rt.encode(&frames, {}, target))
            for (caliper_gfx::Frame& f : frames) gfx.submit(std::move(f));
    }
};

}  // namespace

TEST_CASE("programs: the item map gives an item's fields and index, and refuses an item past the count or the bytes") {
    RecordingHost h;
    caliper_draw::DrawRuntime rt(h.gfx, true);
    caliper_draw::Instance inst;
    inst.program = "tower_arcs";
    inst.panel = "world";
    const caliper_draw::Verdict v = rt.install(hashed_arcs(), inst, 1u << 24);
    REQUIRE_MESSAGE(v.ok(), std::string(v.code + ": " + v.message));
    const ex::ArcResources res = ex::arc_resources();
    REQUIRE(res.arcs.size() > 2);
    REQUIRE(rt.set_resource(version_of("heads", res.heads, 0)).ok());
    REQUIRE(rt.set_resource(version_of("arcs", res.arcs, 0)).ok());
    REQUIRE(rt.set_resource(version_of("labels", res.labels, 0)).ok());
    CHECK_FALSE(rt.pick(0));   // no frame showed a version yet
    h.frame(rt);
    const std::optional<caliper_draw::PickedItem> p = rt.pick(1);
    REQUIRE(p);
    CHECK(p->program == "tower_arcs");
    CHECK(p->item == 1);
    CHECK(p->step == 0);
    std::map<std::string, double> f(p->fields.begin(), p->fields.end());
    const ex::Arc& a = res.arcs[1];
    CHECK(f.at("layer") == a.layer);
    CHECK(f.at("head") == a.head);
    CHECK(f.at("from_token") == a.from_token);
    CHECK(f.at("to_token") == a.to_token);
    CHECK(f.at("weight") == doctest::Approx(a.weight));
    CHECK(f.at("score") == doctest::Approx(a.score));
    // The first four integer fields of the map, in its order: layer, head and the token pair.
    CHECK(p->index == std::array<std::int32_t, 4>{std::int32_t(a.layer), std::int32_t(a.head),
                                                  std::int32_t(a.from_token), std::int32_t(a.to_token)});
    // An item that a fragment wrote past the valid count, or past what the copy holds.
    CHECK_FALSE(rt.pick(std::uint32_t(res.arcs.size())));
    CHECK_FALSE(rt.pick(0xffffffffu));
    // A version whose count says one element more than its bytes hold is refused, and the copy
    // keeps the version shown; a version of two elements then holds no third.
    std::vector<ex::Arc> fewer(res.arcs.begin(), res.arcs.begin() + 2);
    caliper_draw::ResourceVersion shortv = version_of("arcs", fewer, 1);
    shortv.count = 3;
    CHECK_FALSE(rt.set_resource(shortv).ok());
    h.frame(rt);
    REQUIRE(rt.pick(2));
    CHECK(rt.pick(2)->step == 0);
    REQUIRE(rt.set_resource(version_of("arcs", fewer, 1)).ok());
    h.frame(rt);
    REQUIRE(rt.pick(1));
    CHECK(rt.pick(1)->step == 1);
    CHECK_FALSE(rt.pick(2));
}

TEST_CASE("programs: the text table gives a label at each valid anchor and refuses a string past the table and an anchor that is not finite") {
    RecordingHost h;
    caliper_draw::DrawRuntime rt(h.gfx, true);
    caliper_draw::Instance inst;
    inst.program = "tower_arcs";
    inst.panel = "world";
    const spec::Piece piece = hashed_arcs();
    REQUIRE(rt.install(piece, inst, 1u << 24).ok());
    const ex::ArcResources res = ex::arc_resources();
    REQUIRE(rt.set_resource(version_of("heads", res.heads, 0)).ok());
    REQUIRE(rt.set_resource(version_of("arcs", res.arcs, 0)).ok());
    REQUIRE(rt.set_resource(version_of("labels", res.labels, 0)).ok());
    h.frame(rt);
    const std::vector<caliper_draw::Label> all = rt.labels();
    REQUIRE(all.size() == res.labels.size());
    CHECK(all[3].text == piece.draw->text->strings[res.labels[3].string]);
    CHECK(all[3].position[0] == res.labels[3].position[0]);
    std::vector<ex::Label> bad = res.labels;
    bad[0].string = 999;                                   // past the table's strings
    bad[1].position[1] = std::numeric_limits<float>::quiet_NaN();
    bad[2].position[2] = std::numeric_limits<float>::infinity();
    REQUIRE(rt.set_resource(version_of("labels", bad, 1)).ok());
    h.frame(rt);
    const std::vector<caliper_draw::Label> kept = rt.labels();
    CHECK(kept.size() == res.labels.size() - 3);
    for (const caliper_draw::Label& l : kept) CHECK(l.text != "");
}

TEST_CASE("programs: a piece in a scene panel is picked at its pixel, gives its item map's fields, writes a select state and places its labels") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    Source source;
    Events events;
    std::string text = kSpec;
    const std::string opts = "options = { up = \"+z\" }";
    text.replace(text.find(opts), opts.size(),
                 "options = { up = \"+z\", select = { state = \"picked_head\", field = \"head\" } }");
    text += "[[state]]\nid = \"picked_head\"\ntype = \"integer\"\nmin = 0\nmax = 63\n";
    spec::Spec s = parse(text.c_str());
    const spec::Piece piece = hashed_arcs();
    s.pieces.push_back(piece);
    spec::Program prog;
    prog.id = "tower_arcs";
    spec::PieceRef pin;
    pin.name = piece.name;
    pin.version = piece.version;
    pin.hash = piece.hash;
    pin.from = spec::PieceFrom::inline_piece;
    prog.piece = pin;
    prog.panel = "world";
    prog.bind = {{"attn0", "live:blocks.0.attn.weights"}, {"attn1", "live:blocks.1.attn.weights"}};
    s.programs.push_back(prog);
    caliper_builder::StepHistory history;
    history.configure(s);   // the pins of the spec's programs, whose resources it keeps
    caliper_builder::ValueStore values;
    values.reset(s);
    const ex::ArcResources res = ex::arc_resources();
    keep_version(history, "heads", res.heads, 0);
    keep_version(history, "arcs", res.arcs, 0);
    keep_version(history, "labels", res.labels, 0);
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_scene_layer(canvas_layer(c));
    rt->set_source(&source);
    rt->set_instruments(&events, nullptr);
    rt->set_instrument_state(&values, &history);
    rt->set_spec(s);
    const int W = 1200, H = 520;
    Image img;
    caliper_live_views::SceneView* view = nullptr;
    for (int k = 0; k < 8; ++k) {
        img = render(W, H, [&](const ImRect& area) { rt->draw(area, c->look()); }, 2);
        view = rt->entity_view("world");
        if (view && !view->draw_runtimes().empty() && !view->draw_runtimes()[0]->labels().empty()) break;
    }
    dump(img, "programs-arcs-labels.png");
    REQUIRE(view != nullptr);
    REQUIRE(view->draw_runtimes().size() == 1);
    const caliper_draw::DrawRuntime* arcs = view->draw_runtimes()[0];
    CAPTURE(arcs->status().error.code);
    CAPTURE(arcs->status().error.message);
    REQUIRE(arcs->status().state == caliper_draw::DrawState::running);
    CHECK(arcs->labels().size() == res.labels.size());

    // A texel of the ID target that holds the program's pick id: its pixel picks an arc.
    caliper_gfx::Backend& gfx = view->backend();
    const caliper_gfx::Readback rb = gfx.read_texture(view->id_target(), 0, 0, view->width(), view->height());
    REQUIRE(rb);
    std::vector<std::uint8_t> ids;
    for (int k = 0; k < 200 && !gfx.take_readback(rb, &ids); ++k) gfx.wait(gfx.last_use(view->id_target()));
    REQUIRE(ids.size() == std::size_t(view->width()) * view->height() * 8);
    int px = -1, py = -1;
    for (std::uint32_t y = 0; y < view->height() && px < 0; y += 3)
        for (std::uint32_t x = 0; x < view->width() && px < 0; x += 3) {
            std::uint32_t texel[2];
            std::memcpy(texel, ids.data() + (std::size_t(y) * view->width() + x) * 8, 8);
            if (texel[0] == arcs->pick_id()) px = int(x), py = int(y);
        }
    REQUIRE(px >= 0);
    events.got.clear();
    REQUIRE(rt->pick_scene("world", (float(px) + 0.5f) / float(view->width()), (float(py) + 0.5f) / float(view->height())));
    const auto pick_event = [&]() -> const caliper_builder::Event* {
        for (const caliper_builder::Event& e : events.got)
            if (e.kind == caliper_builder::EventKind::pick) return &e;
        return nullptr;
    };
    for (int k = 0; k < 8 && !pick_event(); ++k)
        render(W, H, [&](const ImRect& area) { rt->draw(area, c->look()); }, 1);
    REQUIRE(pick_event() != nullptr);
    const nlohmann::json& f = pick_event()->fields;
    CAPTURE(f.dump());
    CHECK(f["program"] == "tower_arcs");
    const std::uint32_t item = f["item"].get<std::uint32_t>();
    REQUIRE(item < res.arcs.size());
    const ex::Arc& a = res.arcs[item];
    CHECK(f["fields"]["layer"] == a.layer);
    CHECK(f["fields"]["head"] == a.head);
    CHECK(f["fields"]["from_token"] == a.from_token);
    CHECK(f["fields"]["to_token"] == a.to_token);
    CHECK(f["index"] == nlohmann::json::array({a.layer, a.head, a.from_token, a.to_token}));
    // What selection.get returns for the panel, and the state that the panel's select wrote.
    const std::optional<caliper_builder::PickFields> last = values.last_pick("world");
    REQUIRE(last);
    const nlohmann::json got = caliper_builder::pick_fields_json(*last);
    CHECK(got["program"] == "tower_arcs");
    CHECK(got["head"] == a.head);
    CHECK(got["fields"]["to_token"] == a.to_token);
    spec::Reference state;
    state.kind = spec::Reference::Kind::state;
    state.target = "picked_head";
    CHECK(values.value(state) == a.head);
    c->with_context([&] { rt.reset(); });
}

// ---- The programs' pick across views (live_scene.h, SceneView::request_program_pick) ----------
// A panel's pick state outlives the views that it drew: a detach destroys a live scene's view and
// its backend, a new attachment makes another, and a patch that drops the scene binding draws the
// panel's own scene on another backend. Each pick reads the view that the panel draws now, with
// the version that its frame showed.

namespace {

// A source of no data: a scene panel of programs alone reads none.
struct NoData : SpecPanelSource {
    PanelData resolve(const spec::Spec&, std::string_view) override { return PanelData{}; }
    uint64_t revision() const override { return 1; }
    uint64_t generation(const spec::Spec&, std::string_view) override { return 1; }
};

// One scene panel `world` whose program tower_arcs draws the inline attention arcs: on the live
// scene `model` when `live`, else in the panel's own scene.
spec::Spec arcs_spec(bool live) {
    std::string text = "caliper_spec = 2\n[instrument]\nid = \"arcs\"\n[[panel]]\nid = \"world\"\nprimitive = \"scene\"\n";
    if (live) text += "bind = { scene = \"live:model\" }\n";
    spec::Spec s = parse(text);
    const spec::Piece piece = hashed_arcs();
    s.pieces.push_back(piece);
    spec::Program prog;
    prog.id = "tower_arcs";
    spec::PieceRef pin;
    pin.name = piece.name;
    pin.version = piece.version;
    pin.hash = piece.hash;
    pin.from = spec::PieceFrom::inline_piece;
    prog.piece = pin;
    prog.panel = "world";
    prog.bind = {{"attn0", "live:blocks.0.attn.weights"}, {"attn1", "live:blocks.1.attn.weights"}};
    s.programs.push_back(prog);
    return s;
}

// A core with an offscreen canvas and the live scene `model`, as an attachment opens them; a
// close destroys the scene's view and its backend, as a detach does.
struct LiveCore {
    CaliperCore* core = nullptr;
    bool open() {
        CaliperCoreDesc cd{};
        cd.struct_size = sizeof cd;
        cd.renderer = CALIPER_RENDERER_DEFAULT;
        core = caliper_core_create(&cd);
        if (!core) return false;
        CaliperCanvasDesc cv{};
        cv.struct_size = sizeof cv;
        cv.mode = CALIPER_CANVAS_OFFSCREEN;
        cv.width = 64;
        cv.height = 64;
        cv.content_scale = 1.0f;
        if (caliper_core_attach_canvas(core, nullptr, &cv) != 1) return false;
        std::string why;
        const bool ok = caliper_live_views::scene_open(
            core, "model", caliper_net::wire::SceneView{CALIPER_NET_SPACE_3D, "+y", ""}, &why);
        if (!ok) MESSAGE(why);
        return ok && caliper_live_views::scene_view(core, "model") != nullptr;
    }
    void close() {
        if (core) caliper_core_shutdown(core);
        core = nullptr;
    }
    ~LiveCore() { close(); }
};

// The pixel of the view whose ID texel holds `pick_id`, from 0 to 1 across the view.
bool arc_pixel(caliper_live_views::SceneView& view, std::uint32_t pick_id, float* u, float* v) {
    caliper_gfx::Backend& gfx = view.backend();
    if (!view.id_target()) return false;
    const caliper_gfx::Readback rb = gfx.read_texture(view.id_target(), 0, 0, view.width(), view.height());
    if (!rb) return false;
    std::vector<std::uint8_t> ids;
    for (int k = 0; k < 200 && !gfx.take_readback(rb, &ids); ++k) gfx.wait(gfx.last_use(view.id_target()));
    if (ids.size() != std::size_t(view.width()) * view.height() * 8) return false;
    for (std::uint32_t y = 0; y < view.height(); y += 3)
        for (std::uint32_t x = 0; x < view.width(); x += 3) {
            std::uint32_t texel[2];
            std::memcpy(texel, ids.data() + (std::size_t(y) * view.width() + x) * 8, 8);
            if (texel[0] != pick_id) continue;
            *u = (float(x) + 0.5f) / float(view.width());
            *v = (float(y) + 0.5f) / float(view.height());
            return true;
        }
    return false;
}

// The runtime of an arcs spec on the canvas, with its history of the arcs at step 0.
struct ArcsPanel {
    SpecOffscreen* c;
    NoData source;
    Events events;
    caliper_builder::StepHistory history;
    caliper_builder::ValueStore values;
    std::unique_ptr<SpecRuntime> rt;
    ex::ArcResources res = ex::arc_resources();
    ArcsPanel(SpecOffscreen* canvas, const spec::Spec& s, SpecRuntime::SceneLayer layer) : c(canvas) {
        history.configure(s);
        values.reset(s);
        keep_version(history, "heads", res.heads, 0);
        keep_version(history, "arcs", res.arcs, 0);
        keep_version(history, "labels", res.labels, 0);
        c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
        rt->set_scene_layer(std::move(layer));
        rt->set_source(&source);
        rt->set_instruments(&events, nullptr);
        rt->set_instrument_state(&values, &history);
        rt->set_spec(s);
    }
    ~ArcsPanel() {
        c->with_context([&] { rt.reset(); });
    }
    void draw(int frames = 1) {
        render(1200, 520, [&](const ImRect& area) { rt->draw(area, c->look()); }, frames);
    }
    // Draws until the view of the panel runs its program; that view, or null.
    caliper_live_views::SceneView* running() {
        for (int k = 0; k < 12; ++k) {
            draw(2);
            caliper_live_views::SceneView* v = rt->scene_view_of("world");
            if (v && !v->draw_runtimes().empty() &&
                v->draw_runtimes()[0]->status().state == caliper_draw::DrawState::running)
                return v;
        }
        return nullptr;
    }
    const caliper_builder::Event* pick_event() const {
        for (const caliper_builder::Event& e : events.got)
            if (e.kind == caliper_builder::EventKind::pick) return &e;
        return nullptr;
    }
    // Asks for a pick at an arc's pixel of the view, as a click does, and draws until the pick
    // event; its fields, or null when none came.
    nlohmann::json pick(caliper_live_views::SceneView& view, int frames_before_answer = 0,
                        const std::function<void()>& meanwhile = nullptr) {
        float u = 0, v = 0;
        REQUIRE(arc_pixel(view, view.draw_runtimes()[0]->pick_id(), &u, &v));
        events.got.clear();
        REQUIRE(rt->pick_scene("world", u, v));
        for (int k = 0; k < frames_before_answer; ++k) draw();
        if (meanwhile) meanwhile();
        for (int k = 0; k < 8 && !pick_event(); ++k) draw();
        return pick_event() ? pick_event()->fields : nlohmann::json();
    }
};

double field_of(const nlohmann::json& f, const char* name) {
    return f.contains("fields") && f["fields"].contains(name) ? f["fields"][name].get<double>() : -1.0;
}

}  // namespace

TEST_CASE("programs: a pick after a detach, a new attachment or a patch that drops the scene binding reads the view the panel draws now") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    auto first = std::make_unique<LiveCore>();
    if (!first->open()) {
        MESSAGE("skipped: no offscreen canvas or scene view for the embed core");
        return;
    }
    CaliperCore* attached = first->core;
    // The panel's own scene makes one texture first on its backend, so that its handles are not
    // those of the live scene's backend: a handle of one backend that reached another would read
    // another texture there.
    SpecRuntime::SceneLayer layer = [c](std::unique_ptr<caliper_gfx::Backend>* owned, caliper_gfx::Backend** gfx,
                                        const caliper_gfx::ProgramDesc** programs, std::string* why) {
        if (!canvas_layer(c)(owned, gfx, programs, why)) return false;
        caliper_gfx::TextureDesc d;
        d.width = d.height = 4;
        d.format = caliper_gfx::Format::RGBA8Unorm;
        d.usage = caliper_gfx::kSampled;
        (void)(*gfx)->create_texture(d);
        return true;
    };
    ArcsPanel p(c, arcs_spec(true), layer);
    p.rt->set_live_core([&] { return attached; });
    caliper_live_views::SceneView* live = p.running();
    REQUIRE(live != nullptr);
    CHECK(p.pick(*live)["program"] == "tower_arcs");

    // The run detaches and the app attaches to another, while the panel is not drawn: the view
    // and the backend of the first pick are gone.
    first.reset();
    attached = nullptr;
    auto second = std::make_unique<LiveCore>();
    REQUIRE(second->open());
    attached = second->core;
    live = p.running();
    REQUIRE(live != nullptr);
    CHECK(p.pick(*live)["program"] == "tower_arcs");

    // panel.patch drops the scene binding: the panel draws its own scene, on another backend.
    p.rt->set_spec(arcs_spec(false));
    caliper_live_views::SceneView* own = p.running();
    REQUIRE(own != nullptr);
    CHECK(&own->backend() != &live->backend());
    CHECK(p.pick(*own)["program"] == "tower_arcs");

    // The run detaches while the panel draws its own scene: the live scene's backend goes.
    second.reset();
    attached = nullptr;
    CHECK(p.pick(*own)["program"] == "tower_arcs");
}

TEST_CASE("programs: a pick gives the fields of the version its frame showed, and a version at the same step replaces it") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* own = p.running();
    REQUIRE(own != nullptr);
    // A newer version arrives after the frame that the pick reads and before the readback answers.
    std::vector<ex::Arc> newer = p.res.arcs;
    for (ex::Arc& a : newer) a.weight += 100.0f;
    nlohmann::json f = p.pick(*own, 1, [&] { keep_version(p.history, "arcs", newer, 1); });
    REQUIRE(f.is_object());
    CAPTURE(f.dump());
    REQUIRE(f["item"].get<std::uint32_t>() < p.res.arcs.size());
    CHECK(f["step"] == 0);
    CHECK(field_of(f, "weight") == doctest::Approx(p.res.arcs[f["item"].get<std::uint32_t>()].weight));
    // The next pick reads the newer version, which the frames now show.
    p.draw(2);
    f = p.pick(*own);
    REQUIRE(f.is_object());
    CHECK(f["step"] == 1);
    CHECK(field_of(f, "weight") == doctest::Approx(newer[f["item"].get<std::uint32_t>()].weight));
    // A version that replaces the one shown at the same step, as a fork from that step sends.
    std::vector<ex::Arc> same = p.res.arcs;
    for (ex::Arc& a : same) a.weight += 200.0f;
    keep_version(p.history, "arcs", same, 1);
    p.draw(2);
    f = p.pick(*own);
    REQUIRE(f.is_object());
    CHECK(f["step"] == 1);
    CHECK(field_of(f, "weight") == doctest::Approx(same[f["item"].get<std::uint32_t>()].weight));
}

TEST_CASE("programs: a runtime that goes takes its history back from the program host of the process") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    const spec::Spec s = arcs_spec(false);
    auto first = std::make_unique<ArcsPanel>(c, s, canvas_layer(c));
    REQUIRE(first->running() != nullptr);
    // The history stays, its runtime goes; a runtime that gave no history then draws nothing of it.
    caliper_builder::StepHistory kept;
    kept.configure(s);
    const ex::ArcResources res = ex::arc_resources();
    keep_version(kept, "heads", res.heads, 0);
    keep_version(kept, "arcs", res.arcs, 0);
    keep_version(kept, "labels", res.labels, 0);
    first->rt->set_instrument_state(&first->values, &kept);
    first.reset();
    NoData source;
    Events events;
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_scene_layer(canvas_layer(c));
    rt->set_source(&source);
    rt->set_instruments(&events, nullptr);
    rt->set_spec(s);
    for (int k = 0; k < 4; ++k) render(1200, 520, [&](const ImRect& area) { rt->draw(area, c->look()); }, 2);
    caliper_live_views::SceneView* view = rt->scene_view_of("world");
    REQUIRE(view != nullptr);
    REQUIRE(view->draw_runtimes().size() == 1);
    CHECK(view->draw_runtimes()[0]->status().state != caliper_draw::DrawState::running);
    CHECK(view->draw_runtimes()[0]->resource_step("arcs") == -1);
    c->with_context([&] { rt.reset(); });
}

// ---- A drag in a panel of programs (live_views_scene.cpp, draw_scene_view_in_rect) --------------
// A drag moves the camera of a scene panel at each frame, and each frame draws the panel's scene and
// its program with the camera of that frame: with no new version, as while the run is paused, and
// with versions that arrive every few frames, as a slow run sends them. The counts and the cameras
// decide; the frame time is measured beside them, against the 60 Hz budget.

namespace {

struct DragFrame {
    std::uint64_t scene = 0, programs = 0;
    std::string camera, drawn;
    caliper_scene::Mat4 program_clip;   // the clip-from-world matrix of the programs' frame
    std::vector<std::uint8_t> pixels;   // the panel's picture, on every 15th frame
    double ms = 0;
};

// The camera in the words of drawn_camera_text(), without the size of the target.
std::string camera_now(const caliper_live_views::SceneView& v) {
    const std::string t = v.camera_text();
    return t.substr(0, t.find(" target_px"));
}

constexpr int kReadBackEvery = 15;   // drag() reads the picture back after every 15th frame

// A left drag across the panel from 25 % to 75 % of its width, one move a frame, and what each frame
// drew. `before` runs before each frame of the drag with its index, from 1. The frames of the drag
// read nothing back, as the app's frames do not.
std::vector<DragFrame> drag(ArcsPanel& p, caliper_live_views::SceneView& view, int moves,
                            const std::function<void(int)>& before = nullptr) {
    float r[4];
    REQUIRE(p.rt->scene_rect("world", r));
    auto input = [&](const std::function<void(ImGuiIO&)>& fn) { p.c->with_context([&] { fn(ImGui::GetIO()); }); };
    auto frame = [&] {
        std::string why;
        REQUIRE_MESSAGE(p.c->render(1, [&](const ImRect& area) { p.rt->draw(area, p.c->look()); }, nullptr, &why),
                        why);
    };
    const float y = r[1] + 0.5f * r[3];
    auto x_at = [&](int k) { return r[0] + (0.25f + 0.5f * float(k) / float(moves)) * r[2]; };
    input([&](ImGuiIO& io) { io.AddMousePosEvent(x_at(0), y); });
    frame();
    input([&](ImGuiIO& io) { io.AddMouseButtonEvent(0, true); });
    frame();
    std::vector<DragFrame> out;
    for (int k = 1; k <= moves; ++k) {
        if (before) before(k);
        input([&](ImGuiIO& io) { io.AddMousePosEvent(x_at(k), y); });
        const auto t0 = std::chrono::steady_clock::now();
        frame();
        DragFrame f;
        f.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        f.scene = view.cost().frames;
        f.programs = view.program_frames();
        f.camera = camera_now(view);
        f.drawn = view.drawn_camera_text();
        f.program_clip = view.program_clip();
        if (k % kReadBackEvery == 0) {   // a read back after the frame's time was taken
            std::string why;
            REQUIRE_MESSAGE(view.read_pixels(&f.pixels, &why), why);
        }
        out.push_back(f);
    }
    input([&](ImGuiIO& io) { io.AddMouseButtonEvent(0, false); });
    frame();
    input([&](ImGuiIO& io) { io.AddMousePosEvent(-FLT_MAX, -FLT_MAX); });
    frame();
    return out;
}

// Each frame of the drag drew the scene and the program once, with the camera of that frame, which
// the drag moved: the scene's camera and the programs' clip matrix change at every frame.
void check_each_frame(const std::vector<DragFrame>& frames, int moves) {
    REQUIRE(frames.size() == (std::size_t)moves);   // drag() keeps one frame per move
    for (std::size_t i = 1; i < frames.size(); ++i) {
        CAPTURE(i);
        CHECK(frames[i].scene == frames[i - 1].scene + 1);
        CHECK(frames[i].programs == frames[i - 1].programs + 1);
        CHECK(frames[i].drawn == frames[i].camera);
        CHECK(frames[i].camera != frames[i - 1].camera);
        CHECK(std::memcmp(frames[i].program_clip.m, frames[i - 1].program_clip.m, sizeof frames[i].program_clip.m) != 0);
    }
    // The pictures that the GPU drew follow the camera: each one read back differs from the one before.
    const std::vector<std::uint8_t>* last = nullptr;
    int compared = 0;
    for (const DragFrame& f : frames) {
        if (f.pixels.empty()) continue;
        if (last) {
            CHECK(f.pixels.size() == last->size());
            CHECK(f.pixels != *last);
            ++compared;
        }
        last = &f.pixels;
    }
    // drag() reads back every kReadBackEvery-th frame, and each read back but the first is compared.
    CHECK(compared == moves / kReadBackEvery - 1);
}

// The 95th percentile of the frame times, printed with its target. Above the target the test warns,
// and above twice the target it fails. A machine without a usable GPU only says why it skips.
void check_frame_time(const std::vector<DragFrame>& frames, bool usable_gpu, const char* why_not) {
    if (!usable_gpu) {
        std::printf("p95 frame time skipped: %s\n", why_not);
        return;
    }
    std::vector<double> ms;
    for (const DragFrame& f : frames) ms.push_back(f.ms);
    std::sort(ms.begin(), ms.end());
    const double p95 = ms[std::min(ms.size() - 1, std::size_t(0.95 * double(ms.size())))];
    constexpr double kTarget = 1000.0 / 60.0;
    std::printf("p95 frame time %.1f ms (target %.1f ms)\n", p95, kTarget);
    if (p95 > kTarget) std::printf("warning: the p95 frame time %.1f ms is above its target %.1f ms\n", p95, kTarget);
    CHECK_MESSAGE(p95 <= 2.0 * kTarget, "the p95 frame time " << p95 << " ms is above twice its target");
}

// A hosted runner shares a slower GPU, or has none: its frame times say nothing of a person's machine.
bool usable_gpu(const SpecOffscreen* c, const char** why_not) {
    if (!c || !c->native_device()) {
        *why_not = "no Metal device";
        return false;
    }
    if (const char* ci = std::getenv("CI"); ci && *ci && std::strcmp(ci, "false") != 0) {
        *why_not = "a hosted runner (CI is set), whose GPU is shared and several times slower";
        return false;
    }
    return true;
}

}  // namespace

TEST_CASE("programs: a drag in a panel of programs draws each frame with that frame's camera, with no new version") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        std::printf("p95 frame time skipped: no Metal device\n");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const std::vector<DragFrame> frames = drag(p, *view, 90);
    check_each_frame(frames, 90);
    CHECK(view->draw_runtimes()[0]->resource_step("arcs") == 0);
    const char* why_not = "";
    check_frame_time(frames, usable_gpu(c, &why_not), why_not);
}

TEST_CASE("programs: a drag in a panel of programs draws each frame with that frame's camera while versions arrive slowly") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    // A version every 20 frames: a third of a second apart at 60 Hz.
    std::vector<ex::Arc> arcs = p.res.arcs;
    std::int64_t step = 0;
    const std::vector<DragFrame> frames = drag(p, *view, 90, [&](int k) {
        if (k % 20 != 0) return;
        for (ex::Arc& a : arcs) a.weight *= 0.99f;
        keep_version(p.history, "arcs", arcs, ++step);
    });
    check_each_frame(frames, 90);
    CHECK(step == 4);
    CHECK(view->draw_runtimes()[0]->resource_step("arcs") == step);
}

// ---- A camera press in a scene panel (live/camera_drag.h) ----------------------------------------
// The camera follows the pointer from the first point of motion after a press, for the orbit of the
// left button and the pans of the right and middle buttons, so a drag of a point a frame moves the
// camera on its first frame of motion and as far as the pointer went. The release decides the click:
// a press whose pointer stayed within 2 points of it puts the camera back exactly as it was at the
// press and picks the item under the press point, and a drag that went farther stays a drag, also
// when it comes back to its start. A double click's fit stays, a resting finger's jitter leaves no
// drift, a pick in flight survives the next drag, and a control over the picture takes its own press.

namespace {

// The numbers of the view's orbit: target x, y, z, distance, azimuth and elevation.
std::vector<double> orbit_numbers(const caliper_live_views::SceneView& view) {
    const caliper_scene::Orbit* o = std::get_if<caliper_scene::Orbit>(&view.camera());
    if (!o) return {};
    return {o->target[0], o->target[1], o->target[2], o->distance, o->azimuth, o->elevation};
}

double camera_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0;
    for (std::size_t k = 0; k < a.size() && k < b.size(); ++k) d += (a[k] - b[k]) * (a[k] - b[k]);
    return std::sqrt(d);
}

// Queues input into the canvas's context and draws one frame of the panel, with no read back.
struct PanelInput {
    ArcsPanel& p;
    void operator()(const std::function<void(ImGuiIO&)>& fn) const {
        p.c->with_context([&] { fn(ImGui::GetIO()); });
        std::string why;
        REQUIRE_MESSAGE(p.c->render(1, [&](const ImRect& area) { p.rt->draw(area, p.c->look()); }, nullptr, &why), why);
    }
    void frame() const { (*this)([](ImGuiIO&) {}); }
    void move(float x, float y) const { (*this)([&](ImGuiIO& io) { io.AddMousePosEvent(x, y); }); }
    void button(int b, bool down) const { (*this)([&](ImGuiIO& io) { io.AddMouseButtonEvent(b, down); }); }
    // The pointer leaves, and the frames pass the double-click time, so that the next press is a
    // click of its own.
    void rest() const {
        move(-FLT_MAX, -FLT_MAX);
        for (int k = 0; k < 40; ++k) frame();
    }
};

// Two program items next to each other in the view's ID target: the texel (tx, ty) holds item
// *press, and the texel `reach` texels to its right, or below it when *down, holds another item of
// the program. False when the frame has none.
bool adjacent_items(caliper_live_views::SceneView& view, std::uint32_t pick_id, std::uint32_t reach,
                    std::uint32_t* tx, std::uint32_t* ty, bool* down, std::uint32_t* press) {
    caliper_gfx::Backend& gfx = view.backend();
    if (!view.id_target()) return false;
    const caliper_gfx::Readback rb = gfx.read_texture(view.id_target(), 0, 0, view.width(), view.height());
    if (!rb) return false;
    std::vector<std::uint8_t> ids;
    for (int k = 0; k < 200 && !gfx.take_readback(rb, &ids); ++k) gfx.wait(gfx.last_use(view.id_target()));
    if (ids.size() != std::size_t(view.width()) * view.height() * 8) return false;
    auto texel = [&](std::uint32_t x, std::uint32_t y, std::uint32_t k) {
        std::uint32_t t[2];
        std::memcpy(t, ids.data() + (std::size_t(y) * view.width() + x) * 8, 8);
        return t[k];
    };
    for (const bool vertical : {false, true})
        for (std::uint32_t y = 0; y + (vertical ? reach : 0) < view.height(); ++y)
            for (std::uint32_t x = 0; x + (vertical ? 0 : reach) < view.width(); ++x) {
                const std::uint32_t x1 = vertical ? x : x + reach, y1 = vertical ? y + reach : y;
                if (texel(x, y, 0) != pick_id || texel(x1, y1, 0) != pick_id || texel(x, y, 1) == texel(x1, y1, 1))
                    continue;
                *tx = x;
                *ty = y;
                *down = vertical;
                *press = texel(x, y, 1);
                return true;
            }
    return false;
}

int pick_events(const ArcsPanel& p) {
    int n = 0;
    for (const caliper_builder::Event& e : p.events.got) n += e.kind == caliper_builder::EventKind::pick;
    return n;
}

// The panel's picture rectangle and its centre, in whole points.
struct PanelRect {
    float r[4] = {0, 0, 0, 0};
    float cx = 0, cy = 0;
    explicit PanelRect(ArcsPanel& p) {
        REQUIRE(p.rt->scene_rect("world", r));
        cx = std::round(r[0] + 0.5f * r[2]);
        cy = std::round(r[1] + 0.5f * r[3]);
    }
};

}  // namespace

TEST_CASE("programs: a press and motion in one batch move and draw the camera in that frame") {
    SpecOffscreen* c = canvas();
    REQUIRE(c != nullptr);
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    for (const int button : {0, 1, 2}) {
        CAPTURE(button);
        in.move(pr.cx, pr.cy);
        const auto before = orbit_numbers(*view);
        in([&](ImGuiIO& io) {
            io.AddMouseButtonEvent(button, true);
            io.AddMousePosEvent(pr.cx + 1, pr.cy);
        });
        CHECK(orbit_numbers(*view) != before);
        CHECK(view->drawn_camera_text() == camera_now(*view));
        const auto after = orbit_numbers(*view);
        in.frame();
        CHECK(orbit_numbers(*view) == after);
        in.button(button, false);
        CHECK(orbit_numbers(*view) == before);
        in.rest();
    }
}

TEST_CASE("programs: a press and release in one batch pick the arc under the press") {
    SpecOffscreen* c = canvas();
    REQUIRE(c != nullptr);
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    const float kx = float(view->width()) / pr.r[2], ky = float(view->height()) / pr.r[3];
    std::uint32_t tx = 0, ty = 0, item = 0;
    bool vertical = false;
    REQUIRE(adjacent_items(*view, view->draw_runtimes()[0]->pick_id(), std::uint32_t(std::round(kx)),
                           &tx, &ty, &vertical, &item));
    const float x = pr.r[0] + (float(tx) + 0.5f) / kx;
    const float y = pr.r[1] + (float(ty) + 0.5f) / ky;
    for (const bool wobble : {false, true}) {
        CAPTURE(wobble);
        p.events.got.clear();
        in.move(x, y);
        const auto before = orbit_numbers(*view);
        in([&](ImGuiIO& io) {
            io.AddMouseButtonEvent(0, true);
            if (wobble) io.AddMousePosEvent(x + (vertical ? 0 : 1), y + (vertical ? 1 : 0));
            io.AddMouseButtonEvent(0, false);
        });
        c->with_context([&] { CHECK(ImGui::IsMouseClicked(0)); });
        CHECK(pick_events(p) == 0);
        CHECK((orbit_numbers(*view) != before) == wobble);
        in.frame();
        c->with_context([&] { CHECK(ImGui::IsMouseReleased(0)); });
        CHECK(orbit_numbers(*view) == before);
        for (int k = 0; k < 8 && !p.pick_event(); ++k) in.frame();
        REQUIRE(p.pick_event() != nullptr);
        CHECK(pick_events(p) == 1);
        CHECK(p.pick_event()->fields.value("program", std::string()) == "tower_arcs");
        CHECK(p.pick_event()->fields.value("item", std::uint32_t(0xffffffffu)) == item);
        in.rest();
    }
}

TEST_CASE("programs: a slow drag in a scene panel moves the camera on its first frame of motion, by the pointer's "
          "whole way") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    constexpr int kMoves = 30;
    for (const int button : {0, 1, 2}) {
        CAPTURE(button);
        in.move(pr.cx, pr.cy);
        const caliper_scene::ViewCamera before = view->camera();
        in.button(button, true);
        std::vector<std::vector<double>> cams = {orbit_numbers(*view)};
        REQUIRE(cams[0].size() == 6);
        for (int k = 1; k <= kMoves; ++k) {
            in.move(pr.cx + float(k), pr.cy);
            cams.push_back(orbit_numbers(*view));
        }
        in.button(button, false);
        CHECK(orbit_numbers(*view) == cams.back());   // a drag: the release keeps the camera
        in.rest();
        int first = 0;
        std::vector<double> steps;
        for (int k = 1; k <= kMoves; ++k) {
            const double d = camera_distance(cams[std::size_t(k)], cams[std::size_t(k) - 1]);
            if (d > 0 && !first) first = k;
            if (d > 0) steps.push_back(d);
        }
        MESSAGE("button " << button << ": the camera first moved " << first - 1 << " frames after the first move");
        CHECK(first == 1);
        REQUIRE(steps.size() == std::size_t(kMoves));
        std::vector<double> sorted = steps;
        std::sort(sorted.begin(), sorted.end());
        CHECK(steps[0] <= 3.0 * sorted[sorted.size() / 2]);
        // The same drag as the scene view's own steps: one point at each move.
        const caliper_scene::ViewCamera after = view->camera();
        view->camera() = before;
        for (int k = 0; k < kMoves; ++k) {
            if (button == 0) view->orbit(1.0f, 0.0f);
            else view->pan(1.0f, 0.0f, pr.r[3]);
        }
        const std::vector<double> want = orbit_numbers(*view);
        view->camera() = after;
        REQUIRE(want.size() == 6);
        for (std::size_t k = 0; k < 6; ++k) {
            CAPTURE(k);
            CHECK(cams.back()[k] == doctest::Approx(want[k]).epsilon(1e-6));
        }
    }
}

TEST_CASE("programs: a click in a scene panel with a point of wobble puts the camera back exactly and picks the item "
          "under its press, at its release") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    // Two arcs next to each other: the press on the one, the wobble of one point onto the other.
    const float kx = float(view->width()) / pr.r[2], ky = float(view->height()) / pr.r[3];
    REQUIRE(std::fabs(kx - std::round(kx)) < 1e-3f);
    std::uint32_t tx = 0, ty = 0, item = 0;
    bool vertical = false;
    REQUIRE(adjacent_items(*view, view->draw_runtimes()[0]->pick_id(), std::uint32_t(std::round(kx)), &tx, &ty,
                           &vertical, &item));
    const float x = pr.r[0] + (float(tx) + 0.5f) / kx, y = pr.r[1] + (float(ty) + 0.5f) / ky;
    const float wx = vertical ? x : x + 1.0f, wy = vertical ? y + 1.0f : y;
    for (const int button : {0, 1, 2}) {
        CAPTURE(button);
        p.events.got.clear();
        in.move(x, y);
        const std::vector<double> cam = orbit_numbers(*view);
        const bool moved = view->moved();
        in.button(button, true);
        in.move(wx, wy);
        CHECK(orbit_numbers(*view) != cam);   // the wobble moved the camera on its first point
        for (int k = 0; k < 5; ++k) in.frame();
        CHECK(pick_events(p) == 0);   // no pick while the button is held
        in.button(button, false);
        CHECK(orbit_numbers(*view) == cam);   // exactly the camera of the press
        CHECK(view->moved() == moved);
        for (int k = 0; k < 8 && !p.pick_event(); ++k) in.frame();
        if (button == 0) {
            REQUIRE(p.pick_event() != nullptr);
            CHECK(p.pick_event()->fields.value("program", std::string()) == "tower_arcs");
            CHECK(p.pick_event()->fields.value("item", std::uint32_t(0xffffffffu)) == item);   // the press's item
        } else {
            CHECK(pick_events(p) == 0);
        }
        CHECK(orbit_numbers(*view) == cam);
        in.rest();
    }
}

TEST_CASE("programs: a drag in a scene panel that goes 30 points out and comes back within a point stays a drag") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    for (const int button : {0, 1}) {
        CAPTURE(button);
        p.events.got.clear();
        in.move(pr.cx, pr.cy);
        const caliper_scene::ViewCamera before = view->camera();
        const std::vector<double> cam = orbit_numbers(*view);
        in.button(button, true);
        for (int k = 1; k <= 30; ++k) in.move(pr.cx + float(k), pr.cy);
        for (int k = 29; k >= 1; --k) in.move(pr.cx + float(k), pr.cy);
        in.button(button, false);
        for (int k = 0; k < 8; ++k) in.frame();
        CHECK(pick_events(p) == 0);
        const std::vector<double> got = orbit_numbers(*view);
        CHECK(got != cam);   // not put back
        // The camera went 30 points out and 29 back: one point from the press.
        const caliper_scene::ViewCamera after = view->camera();
        view->camera() = before;
        if (button == 0) view->orbit(1.0f, 0.0f);
        else view->pan(1.0f, 0.0f, pr.r[3]);
        const std::vector<double> want = orbit_numbers(*view);
        view->camera() = after;
        for (std::size_t k = 0; k < 6; ++k) {
            CAPTURE(k);
            CHECK(got[k] == doctest::Approx(want[k]).epsilon(1e-6));
        }
        in.rest();
    }
}

TEST_CASE("programs: a double click in a scene panel fits, and the release of its wobbling second press keeps the fit") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    // A pan away first, so that the fit moves the camera.
    in.move(pr.cx, pr.cy);
    in.button(1, true);
    for (int k = 1; k <= 40; ++k) in.move(pr.cx + float(k), pr.cy);
    in.button(1, false);
    in.rest();
    const caliper_scene::ViewCamera panned = view->camera();
    view->fit();
    const std::vector<double> fitted = orbit_numbers(*view);
    view->camera() = panned;
    REQUIRE(fitted != orbit_numbers(*view));
    // Two clicks within the double-click time; the second press wobbles a point before its release.
    in.move(pr.cx, pr.cy);
    in.button(0, true);
    in.button(0, false);
    in.button(0, true);
    CHECK(orbit_numbers(*view) == fitted);   // the second press fits
    in.move(pr.cx + 1.0f, pr.cy);
    CHECK(orbit_numbers(*view) != fitted);
    in.button(0, false);
    CHECK(orbit_numbers(*view) == fitted);   // its release puts back the fit, not the camera before it
    in.rest();
    CHECK(orbit_numbers(*view) == fitted);
}

TEST_CASE("programs: a resting finger's jitter on a scene panel moves the camera only while it rests, with no drift") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    in.move(pr.cx, pr.cy);
    const std::vector<double> start = orbit_numbers(*view);
    // A trackpad's sub-point jitter, which never comes back to the press exactly.
    const float jitter[][2] = {{0.25f, 0.0f}, {0.5f, -0.25f}, {0.75f, 0.25f}, {0.25f, 0.5f}, {-0.25f, 0.25f},
                               {-0.5f, -0.5f}, {1.0f, 0.75f}, {0.5f, 1.25f}, {0.75f, 0.5f}, {0.25f, 0.25f}};
    for (const int button : {0, 1, 0, 1, 0}) {
        CAPTURE(button);
        in.move(pr.cx, pr.cy);
        in.button(button, true);
        bool wobbled = false;
        for (int k = 0; k < 20; ++k) {
            in.move(pr.cx + jitter[k % 10][0], pr.cy + jitter[k % 10][1]);
            wobbled = wobbled || orbit_numbers(*view) != start;
        }
        CHECK(wobbled);
        in.button(button, false);
        CHECK(orbit_numbers(*view) == start);
        in.rest();
    }
    CHECK(orbit_numbers(*view) == start);
}

// A change of the camera from outside during a click stays, and the click undoes only its own
// wobble: camera.set.
namespace {
struct EntityPanel {
    SpecOffscreen* c;
    Source source;
    Events events;
    std::unique_ptr<SpecRuntime> rt;
    explicit EntityPanel(SpecOffscreen* canvas) : c(canvas) {
        c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
        rt->set_scene_layer(canvas_layer(c));
        rt->set_source(&source);
        rt->set_instruments(&events, nullptr);
        rt->set_spec(parse(kSpec));
        render(1200, 520, [&](const ImRect& area) { rt->draw(area, c->look()); }, 4);
    }
    ~EntityPanel() {
        c->with_context([&] { rt.reset(); });
    }
    void input(const std::function<void(ImGuiIO&)>& fn) {
        c->with_context([&] { fn(ImGui::GetIO()); });
        std::string why;
        REQUIRE_MESSAGE(c->render(1, [&](const ImRect& area) { rt->draw(area, c->look()); }, nullptr, &why), why);
    }
    void frames(int n) {
        for (int k = 0; k < n; ++k) input([](ImGuiIO&) {});
    }
};
}  // namespace

TEST_CASE("entities: camera.set during a click stays, and the click's wobble after it goes") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    EntityPanel p(c);
    caliper_live_views::SceneView* view = p.rt->entity_view("world");
    REQUIRE(view != nullptr);
    float r[4];
    REQUIRE(p.rt->scene_rect("world", r));
    const float cx = std::round(r[0] + 0.5f * r[2]), cy = std::round(r[1] + 0.5f * r[3]);
    const caliper_scene::Orbit* now = std::get_if<caliper_scene::Orbit>(&view->camera());
    REQUIRE(now != nullptr);
    caliper_scene::Orbit set = *now;
    set.azimuth += 0.7;
    set.elevation = 0.3;
    set.distance *= 1.5;
    p.input([&](ImGuiIO& io) { io.AddMousePosEvent(cx, cy); });
    p.input([](ImGuiIO& io) { io.AddMouseButtonEvent(1, true); });
    p.input([&](ImGuiIO& io) { io.AddMousePosEvent(cx + 1.0f, cy); });
    REQUIRE(view->set_orbit(set));   // as camera.set does, between frames
    const std::vector<double> want = orbit_numbers(*view);
    p.input([&](ImGuiIO& io) { io.AddMousePosEvent(cx + 1.0f, cy + 1.0f); });
    CHECK(orbit_numbers(*view) != want);
    p.input([](ImGuiIO& io) { io.AddMouseButtonEvent(1, false); });
    CHECK(orbit_numbers(*view) == want);
    CHECK(view->moved());   // as camera.set leaves it
}

TEST_CASE("programs: a drag that starts while a click's pick is in flight moves the camera from its first point, and "
          "the click's pick still answers with the item under it") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    const float kx = float(view->width()) / pr.r[2], ky = float(view->height()) / pr.r[3];
    std::uint32_t tx = 0, ty = 0, item = 0;
    bool vertical = false;
    REQUIRE(adjacent_items(*view, view->draw_runtimes()[0]->pick_id(), std::uint32_t(std::round(kx)), &tx, &ty,
                           &vertical, &item));
    const float x = pr.r[0] + (float(tx) + 0.5f) / kx, y = pr.r[1] + (float(ty) + 0.5f) / ky;
    p.events.got.clear();
    in.move(x, y);
    in.button(0, true);
    in.button(0, false);   // the click asks its pick
    bool down = false, pending = false;
    REQUIRE(p.rt->scene_gesture("world", &down, &pending));
    MESSAGE("the click's pick is " << std::string(pending ? "in flight" : "answered") << " when the drag's press comes");
    // The middle button, so that the press is no second click of a double click.
    in.button(2, true);
    REQUIRE(p.rt->scene_gesture("world", &down, &pending));
    MESSAGE("the click's pick is " << std::string(pending ? "in flight" : "answered")
                                   << " when the drag's first move comes");
    const std::vector<double> pressed = orbit_numbers(*view);
    in.move(x + 1.0f, y);
    CHECK(orbit_numbers(*view) != pressed);   // the first point moved the camera
    for (int k = 2; k <= 30; ++k) in.move(x + float(k), y);
    in.button(2, false);
    for (int k = 0; k < 8; ++k) in.frame();
    REQUIRE(pick_events(p) == 1);   // the click's, and none of the drag
    CHECK(p.pick_event()->fields.value("item", std::uint32_t(0xffffffffu)) == item);
    CHECK(orbit_numbers(*view) != pressed);
}

TEST_CASE("programs: a press on a control in a child window over a scene panel's picture moves no camera") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    ArcsPanel p(c, arcs_spec(false), canvas_layer(c));
    caliper_live_views::SceneView* view = p.running();
    REQUIRE(view != nullptr);
    const PanelRect pr(p);
    const PanelInput in{p};
    // A button over the middle of the picture, in a child window of its own that the page's overlay
    // draws after the panel's content, as the spec's controls are drawn.
    bool active = false;
    p.rt->set_overlay([&](const caliper_host::SpecPanelFrame& f) {
        if (f.id != "world") return;
        ImGui::SetCursorScreenPos(ImVec2(pr.cx - 40.0f, pr.cy - 20.0f));
        if (ImGui::BeginChild("##overlay.child", ImVec2(80.0f, 40.0f), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
            ImGui::Button("##overlay.control", ImVec2(80.0f, 40.0f));
            active = active || ImGui::IsItemActive();
        }
        ImGui::EndChild();
    });
    in.frame();
    in.move(pr.cx, pr.cy);
    in.frame();
    const std::vector<double> cam = orbit_numbers(*view);
    in.button(0, true);
    for (int k = 1; k <= 30; ++k) in.move(pr.cx + float(k), pr.cy);
    in.button(0, false);
    in.frame();
    CHECK(active);
    CHECK(orbit_numbers(*view) == cam);
    CHECK(pick_events(p) == 0);
    in.rest();
    // Beside the control the picture orbits.
    in.move(pr.cx - 80.0f, pr.cy);
    in.frame();
    in.button(0, true);
    in.move(pr.cx - 79.0f, pr.cy);
    CHECK(orbit_numbers(*view) != cam);
    in.button(0, false);
    in.rest();
    p.rt->set_overlay({});
}

// A 3-D points panel's box, which ImPlot3D turns with a right drag from the first point: a right
// press that wobbles a point puts the turn of its press back at its release, and the page's menu
// rule takes the box for a camera's picture (camera_area); a right drag of 3 points keeps its turn.
namespace {
struct CloudSource : SpecPanelSource {
    PanelData resolve(const spec::Spec&, std::string_view) override {
        std::vector<float> pos;
        for (int i = 0; i < 64; ++i) pos.insert(pos.end(), {float(i % 4), float((i / 4) % 4), float(i / 16)});
        return data({f32("positions", "w", {64, 3}, pos)}, spec::Primitive::points);
    }
    uint64_t revision() const override { return 1; }
    uint64_t generation(const spec::Spec&, std::string_view) override { return 1; }
};
}  // namespace

TEST_CASE("points3d: a right press that wobbles a point puts the box's turn back; a right drag of 3 points keeps it") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) {
        MESSAGE("skipped: no Metal device");
        return;
    }
    CloudSource source;
    const spec::Spec s = one_panel("[[source]]\nid = \"w\"\ncatalog = \"tensor:w\"",
                                   "primitive = \"points\"\nbind = { positions = \"w\" }");
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_source(&source);
    rt->set_spec(s);
    auto draw = [&](const ImRect& area) { rt->draw(area, c->look()); };
    render(1000, 600, draw, 4);
    auto frame = [&](const std::function<void(ImGuiIO&)>& fn) {
        c->with_context([&] { fn(ImGui::GetIO()); });
        std::string why;
        REQUIRE_MESSAGE(c->render(1, draw, nullptr, &why), why);
    };
    auto turn = [&] {
        std::vector<double> q(4);
        REQUIRE(rt->points_turn("p", q.data()));
        return q;
    };
    REQUIRE(rt->layout().panels.size() == 1);
    const SpecLayoutPanel& lp = rt->layout().panels[0];
    const float cx = std::round(lp.rect.x + 0.5f * lp.rect.w), cy = std::round(lp.rect.y + 0.55f * lp.rect.h);
    CHECK(rt->camera_area("p", cx, cy));
    CHECK_FALSE(rt->camera_area("p", lp.rect.x - 10.0f, cy));
    frame([&](ImGuiIO& io) { io.AddMousePosEvent(cx, cy); });
    const std::vector<double> press = turn();
    // A right press that wobbles one point: the box turns on that point, and its release puts the
    // turn of the press back.
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, true); });
    frame([&](ImGuiIO& io) { io.AddMousePosEvent(cx + 1.0f, cy); });
    CHECK(turn() != press);
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, false); });
    CHECK(turn() == press);
    for (int k = 0; k < 40; ++k) frame([](ImGuiIO&) {});
    CHECK(turn() == press);
    // A right drag of 3 points keeps its turn.
    frame([&](ImGuiIO& io) { io.AddMousePosEvent(cx, cy); });
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, true); });
    for (int k = 1; k <= 3; ++k) frame([&](ImGuiIO& io) { io.AddMousePosEvent(cx + float(k), cy); });
    const std::vector<double> turned = turn();
    CHECK(turned != press);
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, false); });
    CHECK(turn() == turned);
    // A double right click starts the box's own animation (a reset, or a turn onto a plane), which
    // its second release keeps: the box does not jump back to the turn before the double click.
    for (int k = 0; k < 40; ++k) frame([](ImGuiIO&) {});
    const std::vector<double> before_double = turn();
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, true); });
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, false); });
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, true); });
    for (int k = 0; k < 30; ++k) frame([](ImGuiIO&) {});   // longer than the animation
    const std::vector<double> animated = turn();
    CHECK(animated != before_double);
    frame([](ImGuiIO& io) { io.AddMouseButtonEvent(1, false); });
    CHECK(turn() == animated);
    for (int k = 0; k < 10; ++k) frame([](ImGuiIO&) {});
    CHECK(turn() == animated);
    c->with_context([&] { rt.reset(); });
}

TEST_CASE("programs: a runtime that goes takes its history back also after another runtime gave the event sink") {
    SpecOffscreen* c = canvas();
    if (!c || !c->native_device()) return;
    const spec::Spec s = arcs_spec(false);
    // The history and the values outlive the runtime that gave them, thus a host that still read
    // them would draw the arcs.
    caliper_builder::StepHistory kept;
    kept.configure(s);
    caliper_builder::ValueStore values;
    values.reset(s);
    const ex::ArcResources res = ex::arc_resources();
    keep_version(kept, "heads", res.heads, 0);
    keep_version(kept, "arcs", res.arcs, 0);
    keep_version(kept, "labels", res.labels, 0);
    auto first = std::make_unique<ArcsPanel>(c, s, canvas_layer(c));
    REQUIRE(first->running() != nullptr);
    first->rt->set_instrument_state(&values, &kept);
    // A second runtime gives the host its event sink, and no values or history.
    NoData source;
    Events events;
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_scene_layer(canvas_layer(c));
    rt->set_source(&source);
    rt->set_instruments(&events, nullptr);
    first.reset();   // the first runtime goes: its history and values leave the host, the sink stays
    rt->set_spec(s);
    for (int k = 0; k < 4; ++k) render(1200, 520, [&](const ImRect& area) { rt->draw(area, c->look()); }, 2);
    caliper_live_views::SceneView* view = rt->scene_view_of("world");
    REQUIRE(view != nullptr);
    REQUIRE(view->draw_runtimes().size() == 1);
    CHECK(view->draw_runtimes()[0]->status().state != caliper_draw::DrawState::running);
    CHECK(view->draw_runtimes()[0]->resource_step("arcs") == -1);
    c->with_context([&] { rt.reset(); });
}
