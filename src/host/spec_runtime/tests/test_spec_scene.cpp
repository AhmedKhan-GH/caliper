// The instruments of the spec runtime (design/instruments.md) without a
// window: the events of a scene panel's picks and selections, and the test
// seam that asks for a pick.
#include "spec_runtime/spec_runtime.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <fstream>
#include <sstream>

using namespace caliper_host;
namespace spec = caliper_builder::spec;
using nlohmann::json;

TEST_CASE("scene panel: a pick becomes a pick event with the panel id and the scene as its view") {
    caliper_builder::Event ev;
    const std::string detail =
        R"({"entity": "modules", "primitive": "boxes", "index": 2, "row_id": 2, "position": [0.0, 2.0, 0.0]})";
    REQUIRE(spec_scene_event("pick", "world", "model", detail, &ev));
    CHECK(ev.kind == caliper_builder::EventKind::pick);
    CHECK(ev.panel == "world");
    CHECK(ev.view == "model");
    CHECK(ev.origin == "ui");
    CHECK(ev.fields == json::parse(detail));
}

TEST_CASE("scene panel: a selection, and a cleared selection, become select events") {
    caliper_builder::Event ev;
    const std::string detail =
        R"({"entities": [{"entity": "modules", "primitive": "boxes", "indices": [0, 1], "row_ids": [0, 1]}], "truncated": false})";
    REQUIRE(spec_scene_event("select", "world", "model", detail, &ev));
    CHECK(ev.kind == caliper_builder::EventKind::select);
    CHECK(ev.fields.at("entities").size() == 1);
    REQUIRE(spec_scene_event("select", "world", "model", "{}", &ev));
    CHECK(ev.fields == json{{"entities", json::array()}, {"truncated", false}});
    CHECK_FALSE(spec_scene_event("pick", "world", "model", "not json", &ev));
    CHECK_FALSE(spec_scene_event("pick", "world", "model", "[1, 2]", &ev));
}

TEST_CASE("scene panel: the test seam asks for a pick only in a scene panel of the spec") {
    std::ifstream in(std::string(CALIPER_SPEC_FIXTURES) + "/instruments.spec.toml", std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    auto parsed = spec::parse_toml(ss.str());
    REQUIRE(parsed.ok());
    SpecTextures textures;
    SpecRuntime runtime(textures);
    CHECK_FALSE(runtime.pick_scene("world", 0.5f, 0.5f));   // no spec yet
    runtime.set_spec(parsed.spec);
    CHECK(runtime.pick_scene("world", 0.5f, 0.5f));
    CHECK_FALSE(runtime.pick_scene("lr", 0.5f, 0.5f));      // a control
    CHECK_FALSE(runtime.pick_scene("nothing", 0.5f, 0.5f));
}

// ---- The scene view's starting camera and its fit as the scene arrives ---------------------------
// (design/scenes.md, Starting Camera; design/instrument-interface.md, Starting Camera and Saved
// Views). A scene view on the offscreen canvas's device takes the scene's parts as wire messages.

#include "spec_runtime/spec_offscreen.h"

#include "export_service.h"
#if defined(__APPLE__)
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#endif
#include "live/live_scene.h"
#include "live/live_views_scene.h"
#include "net/json.h"
#include "net/wire.h"
#include "scene/camera.h"

#include "spec_test_canvas.h"

#include <cmath>
#include <cstdlib>

#if defined(__APPLE__)
namespace {

namespace wire = caliper_net::wire;

struct OwnedView {
    std::unique_ptr<caliper_gfx::Backend> gfx;
    std::unique_ptr<caliper_live_views::SceneView> view;
};

OwnedView make_view(SpecOffscreen* c, const char* up) {
    OwnedView v;
    v.gfx = caliper_gfx::make_metal_backend(c->native_device(), c->native_queue());
    REQUIRE(v.gfx);
    v.view = std::make_unique<caliper_live_views::SceneView>(*v.gfx, caliper_gfx::shaders::kMetalPrograms);
    std::string why;
    REQUIRE_MESSAGE(v.view->init(&why), why);
    wire::SceneView info;
    info.space = CALIPER_NET_SPACE_3D;
    info.up = up;
    v.view->describe(info);
    return v;
}

// A cluster of points around a centre, as one points part of the entity at `path`.
bool add_cluster(caliper_live_views::SceneView& v, const std::string& path, float cx, float cy, float cz,
                 std::vector<float>* keep) {
    keep->clear();
    for (int i = 0; i < 64; ++i) {
        const float a = float(i) * 0.7f;
        keep->insert(keep->end(), {cx + 0.4f * std::cos(a), cy + 0.4f * std::sin(a), cz + 0.05f * float(i % 8)});
    }
    wire::SceneUpdate u;
    u.path = path;
    u.part = CALIPER_NET_PART_POINTS;
    u.params = caliper_net::Json::object();
    wire::Section s;
    s.name = "positions";
    s.dtype = CALIPER_NET_DTYPE_F32;
    s.shape = {64, 3};
    s.data = reinterpret_cast<const uint8_t*>(keep->data());
    s.size = keep->size() * sizeof(float);
    u.arrays.push_back(s);
    std::string why;
    const bool ok = v.apply(u, &why);
    if (!ok) MESSAGE(why);
    return ok;
}

// Whether every point of the view's scene projects inside its picture at its camera.
bool all_inside(const caliper_live_views::SceneView& v, const std::vector<std::vector<float>>& parts) {
    const caliper_scene::CameraMatrices cm =
        caliper_scene::camera_matrices(v.effective_camera(), v.tree(), v.width(), v.height());
    if (!cm.ok) return false;
    for (const auto& p : parts)
        for (size_t i = 0; i + 2 < p.size(); i += 3) {
            double h[4];
            caliper_scene::transform_h(cm.proj * cm.view, {p[i], p[i + 1], p[i + 2]}, h);
            if (!(h[3] > 0) || std::fabs(h[0] / h[3]) > 1.0 || std::fabs(h[1] / h[3]) > 1.0) return false;
        }
    return true;
}

void keep_picture(caliper_live_views::SceneView& v, const char* name) {
    const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP");
    if (!dir || !*dir) return;
    std::vector<uint8_t> rgba;
    std::string why;
    if (v.read_pixels(&rgba, &why))
        caliper_host::export_write_png_atomic(std::string(dir) + "/" + name, rgba.data(), v.width(), v.height());
}

}  // namespace

TEST_CASE("scene view: a late viewer fits the parts as they arrive until the person moves the camera") {
    SpecOffscreen* c = spec_test::canvas();
    if (!c || !c->native_device()) return;
    OwnedView o = make_view(c, "+y");
    caliper_live_views::SceneView& v = *o.view;
    std::vector<float> first, second, third;
    std::string why;
    // The first part arrives: the view fits it.
    REQUIRE(add_cluster(v, "near", 0, 0, 0, &first));
    REQUIRE_MESSAGE(v.render(480, 320, 1.0f, 0xff161616u, &why), why);
    REQUIRE(v.fitted());
    const double d1 = std::get<caliper_scene::Orbit>(v.camera()).distance;
    // The rest arrives later, far away: the view, which nobody moved, frames the whole scene.
    REQUIRE(add_cluster(v, "far", 6, 2, -3, &second));
    REQUIRE(v.render(480, 320, 1.0f, 0xff161616u, &why));
    keep_picture(v, "scene-late-attach-refit.png");
    const double d2 = std::get<caliper_scene::Orbit>(v.camera()).distance;
    CHECK(d2 > 2 * d1);
    CHECK(all_inside(v, {first, second}));
    // Once the person turns the view, new parts no longer move the camera.
    v.orbit(30.0f, 0.0f);
    REQUIRE(v.moved());
    const caliper_scene::Orbit turned = std::get<caliper_scene::Orbit>(v.camera());
    REQUIRE(add_cluster(v, "further", -20, 0, 0, &third));
    REQUIRE(v.render(480, 320, 1.0f, 0xff161616u, &why));
    CHECK(std::get<caliper_scene::Orbit>(v.camera()).distance == turned.distance);
    // A reset frames everything again.
    v.reset();
    REQUIRE(v.render(480, 320, 1.0f, 0xff161616u, &why));
    CHECK(all_inside(v, {first, second, third}));
}

TEST_CASE("scene view: view_camera from the scene's description opens the view and a reset returns to it") {
    SpecOffscreen* c = spec_test::canvas();
    if (!c || !c->native_device()) return;
    OwnedView o = make_view(c, "+z");
    caliper_live_views::SceneView& v = *o.view;
    wire::SceneView info;
    info.space = CALIPER_NET_SPACE_3D;
    info.up = "+z";
    info.view_camera = caliper_net::Json::object();
    info.view_camera.set("azimuth", -60.0);
    info.view_camera.set("elevation", 35.0);
    v.describe(info);
    std::vector<float> pts;
    REQUIRE(add_cluster(v, "cloud", 1, 1, 0, &pts));
    std::string why;
    REQUIRE(v.render(400, 300, 1.0f, 0xff161616u, &why));
    keep_picture(v, "scene-view-camera.png");
    const auto* orbit = std::get_if<caliper_scene::Orbit>(&v.camera());
    REQUIRE(orbit != nullptr);
    CHECK(orbit->azimuth == doctest::Approx(-60.0 * caliper_scene::kPi / 180.0));
    CHECK(orbit->elevation == doctest::Approx(35.0 * caliper_scene::kPi / 180.0));
    CHECK(orbit->up == caliper_scene::Axis::PosZ);
    CHECK(all_inside(v, {pts}));   // the keys left out come from the fit
    // A target and a distance are used as given, without bounds.
    caliper_scene::StartingCamera s;
    s.form = caliper_scene::StartingCamera::Form::Orbit;
    s.target = caliper_scene::Vec3{1, 1, 0};
    s.distance = 7.5;
    v.set_starting(s);
    REQUIRE(v.render(400, 300, 1.0f, 0xff161616u, &why));
    orbit = std::get_if<caliper_scene::Orbit>(&v.camera());
    REQUIRE(orbit != nullptr);
    CHECK(orbit->distance == doctest::Approx(7.5));
    CHECK(orbit->target[0] == doctest::Approx(1.0));
    // The person zooms; a new starting camera waits for the reset, which goes to it.
    v.zoom(2.0f, 0.0f, 0.0f, 400.0f, 300.0f);
    REQUIRE(v.moved());
    s.distance = 4.0;
    v.set_starting(s);
    REQUIRE(v.render(400, 300, 1.0f, 0xff161616u, &why));
    CHECK(std::get<caliper_scene::Orbit>(v.camera()).distance != doctest::Approx(4.0));
    v.reset();
    REQUIRE(v.render(400, 300, 1.0f, 0xff161616u, &why));
    CHECK(std::get<caliper_scene::Orbit>(v.camera()).distance == doctest::Approx(4.0));
}
TEST_CASE("scene panel: a panel bound to a live scene opens on its view_camera and draws its axes mark") {
    SpecOffscreen* c = spec_test::canvas();
    if (!c || !c->native_device()) return;
    OwnedView o = make_view(c, "+z");
    caliper_live_views::SceneView& v = *o.view;
    std::vector<float> pts;
    REQUIRE(add_cluster(v, "cloud", 1, 1, 0, &pts));
    caliper_live_views::SceneRectState state;
    const auto draw = [&](const caliper_live_views::SceneRectOptions& opt) {
        return spec_test::render(400, 300, [&](const ImRect& area) {
            caliper_live_views::draw_scene_view_in_rect(v, 1.0f, "##options", area.Min, area.GetSize(), opt, state,
                                                        nullptr);
        });
    };
    const auto keep = [](const spec_test::Image& img, const char* name) {
        const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP");
        if (dir && *dir && !img.px.empty())
            caliper_host::export_write_png_atomic(std::string(dir) + "/" + name, img.px.data(), (uint32_t)img.w,
                                                  (uint32_t)img.h);
    };
    // The red of the mark's x arrow in the bottom-right corner.
    const auto red = [](const spec_test::Image& img) {
        return img.count(spec_test::near(IM_COL32(222, 92, 84, 255), 40), 280, 180, 400, 300).n;
    };
    // Without the options, as a panel drew before it took them: the scene's own camera, no mark.
    caliper_live_views::SceneRectOptions plain;
    plain.scale_bar = true;
    const spec_test::Image before = draw(plain);
    keep(before, "scene-panel-options-without.png");
    CHECK(state.camera_applied.empty());

    spec::Panel panel;
    panel.options.scene.view_camera = json{{"azimuth", -60.0}, {"elevation", 35.0}};
    caliper_live_views::SceneRectOptions opt = plain;
    opt.view_camera = spec_starting_camera(panel);
    opt.view_camera_key = panel.options.scene.view_camera.dump();
    opt.axes = true;
    const spec_test::Image after = draw(opt);
    keep(after, "scene-panel-options.png");
    CHECK(state.camera_applied == opt.view_camera_key);
    const auto* orbit = std::get_if<caliper_scene::Orbit>(&v.camera());
    REQUIRE(orbit != nullptr);
    CHECK(orbit->azimuth == doctest::Approx(-60.0 * caliper_scene::kPi / 180.0));
    CHECK(orbit->elevation == doctest::Approx(35.0 * caliper_scene::kPi / 180.0));
    CHECK(all_inside(v, {pts}));
    CHECK(red(after) > red(before) + 10);
}
#endif  // __APPLE__
