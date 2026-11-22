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
#if defined(__APPLE__)
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#endif
#include "live/live_scene.h"
#include "scene/camera.h"

#include "spec_test_canvas.h"
#include "spec_test_data.h"

#include <caliper_builder/session/events.h>

#include <cmath>
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

const char* const kSpec = R"(caliper_spec = 1
[applet]
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
