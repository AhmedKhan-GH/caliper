// scene_parity.cpp: the browser's side of the scene comparison (scene_parity.h).
#include "scene_parity.h"

#include "../../../tests/draw_parity/scene_parity.h"
#include "../../../tests/scene/primitive_rows.h"

#include <memory>

namespace caliper_web {

namespace {

struct Run {
    std::unique_ptr<caliper_live_views::SceneView> view;
    primitive_rows::Row row;
    nlohmann::json facts;
    std::vector<std::array<int, 2>> points;
    bool given_points = false;
    nlohmann::json picks = nlohmann::json::array();
    std::size_t next = 0;               // the next point to pick at
    bool asked = false;                 // a pick of points[next] is in flight
    caliper_gfx::Readback color_rb, id_rb;
    bool color_in = false, id_in = false;
    SceneParityResult result;
    bool running = false;
};
Run& run() {
    static Run r;
    return r;
}

}  // namespace

bool scene_parity_start(caliper_gfx::Backend& b, const caliper_gfx::ProgramDesc* programs, const std::string& config,
                        std::string* why) {
    Run& r = run();
    if (r.running) return (*why = "a comparison is running", false);
    scene_parity_stop();
    const nlohmann::json c = nlohmann::json::parse(config, nullptr, false);
    if (c.is_discarded() || !c.is_object() || !c.contains("fixture") || !c["fixture"].is_string())
        return (*why = "the config is no JSON object with a fixture", false);
    const std::string name = c["fixture"].get<std::string>();
    r.points.clear();
    r.given_points = c.contains("points") && c["points"].is_array();
    if (r.given_points)
        for (const nlohmann::json& p : c["points"])
            if (p.is_array() && p.size() == 2 && p[0].is_number_integer() && p[1].is_number_integer())
                r.points.push_back({p[0].get<int>(), p[1].get<int>()});
    bool found = false;
    std::string names;
    for (primitive_fixtures::Fixture& f : scene_parity::fixtures()) {
        names += (names.empty() ? "" : ",") + f.name;
        if (f.name != name) continue;
        r.row = primitive_rows::Row{};
        r.row.fixture = std::move(f);
        found = true;
    }
    if (!found) return (*why = "no fixture '" + name + "' (" + names + ")", false);
    r.view = std::make_unique<caliper_live_views::SceneView>(b, programs);
    // Without "shapes", the view keeps the module's switch (caliperConfig.shapes, main.cpp).
    if (c.contains("shapes"))
        r.view->set_shape_draw(c["shapes"] == "builtin" ? caliper_live_views::ShapeDraw::builtin
                                                        : caliper_live_views::ShapeDraw::pieces);
    if (!r.view->init(why)) return false;
    if (!r.row.draw(*r.view)) return (*why = r.row.error, false);
    // The facts of the frame whose targets the readbacks below read.
    r.facts = scene_parity::frame_json(*r.view);
    const std::uint32_t S = primitive_fixtures::kSize;
    r.color_rb = b.read_texture(r.view->color(), 0, 0, S, S);
    r.id_in = !r.view->id_target();   // no piece drew: no ID target
    if (!r.id_in) r.id_rb = b.read_texture(r.view->id_target(), 0, 0, S, S);
    if (!r.color_rb || (!r.id_in && !r.id_rb)) return (*why = "no readback: " + b.last_error(), false);
    r.picks = nlohmann::json::array();
    r.next = 0;
    r.asked = false;
    r.running = true;
    return true;
}

bool scene_parity_poll(caliper_gfx::Backend& b, std::string* report) {
    Run& r = run();
    if (!r.running) return false;
    const std::uint32_t S = primitive_fixtures::kSize;
    const std::size_t px = std::size_t(S) * S;
    if (!r.color_in) r.color_in = b.take_readback(r.color_rb, &r.result.color);
    if (!r.id_in) r.id_in = b.take_readback(r.id_rb, &r.result.ids);
    if (!r.color_in || !r.id_in) return false;
    if (r.result.ids.size() != px * 8) r.result.ids.assign(px * 8, 0);
    if (!r.given_points && r.points.empty() && r.next == 0 && !r.asked)
        r.points = scene_parity::pick_points(scene_parity::pair_ids(r.result.ids, px), int(S));
    // One pick at a time: ask, then poll each frame until it answers.
    while (r.next < r.points.size()) {
        const auto& p = r.points[r.next];
        std::string why;
        if (!r.asked) {
            if (!r.view->request_pick(float(p[0]) + 0.5f, float(p[1]) + 0.5f, &why)) {
                *report = "fail " + r.row.fixture.name + ": pick at (" + std::to_string(p[0]) + ", " +
                          std::to_string(p[1]) + "): " + why;
                r.running = false;
                return true;
            }
            r.asked = true;
        }
        caliper_scene::Hit hit;
        const caliper_scene::PickPass::Result res = r.view->poll_pick(&hit);
        if (res == caliper_scene::PickPass::Result::Pending) return false;   // next frame
        nlohmann::json j = scene_parity::hit_json(*r.view, res, hit);
        j["at"] = {p[0], p[1]};
        r.picks.push_back(j);
        r.asked = false;
        ++r.next;
    }
    nlohmann::json facts = r.facts;
    facts["picks"] = r.picks;
    r.result.facts = facts.dump();
    r.running = false;
    const bool ok = r.result.color.size() == px * 4;
    *report = std::string(ok ? "ok " : "fail ") + r.row.fixture.name + ": frame " + r.facts["frame"].dump() + ", " +
              r.facts["pieces"].dump() + " shapes by pieces, " + r.facts["builtin"].dump() + " by the built-in " +
              r.facts["fallbacks"].dump() + ", " + std::to_string(r.picks.size()) + " picks";
    return true;
}

bool scene_parity_running() { return run().running; }

const SceneParityResult& scene_parity_result() { return run().result; }

void scene_parity_stop() {
    Run& r = run();
    r.running = false;
    r.view.reset();
    r.result = SceneParityResult{};
    r.color_in = r.id_in = false;
}

}  // namespace caliper_web
