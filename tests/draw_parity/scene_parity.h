// scene_parity.h: the scene comparison between the app and the browser module. Both draw the scene
// fixtures of the scene view's rows (tests/scene/primitive_fixtures.h, as the graphics suite's
// comparison of the pieces with the built-in draws them, tests/gfx/gfx_scene_view.cpp) through a scene
// view whose shapes the library pieces draw: the app on Metal (scene_parity_app.mm, ctest
// caliper_scene_parity_app, which writes <out>/scene/), the module on WebGPU in headless Chromium
// (web/wasm/src/scene_parity.cpp). web/test/draw_parity.spec.ts compares the pictures and the IDs of
// the pieces under its edge rule and tolerances, the counts of the shapes of the same frame, and the
// hits of picks at the same points.
//
// For one fixture, each side gives the facts of the first frame that the view drew, as JSON: the
// frame, the shapes that pieces drew and those that the built-in drew with the reason of each
// fallback, the program of each pick id, and the hits of picks. The pick points are pixels inside
// one ID of the pieces (pick_points), at most kPickPoints of them, one for each program first. It
// depends on no test framework.
#pragma once

#include "../scene/primitive_fixtures.h"
#include "draw/draw_runtime.h"
#include "live/live_scene.h"
#include "scene/pick.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace scene_parity {

inline constexpr std::size_t kPickPoints = 24;

// Every fixture of the primitive rows, and "through-pieces": the view through a camera with its
// points sized in pixels, which the pieces draw, as in the graphics suite's comparison.
inline std::vector<primitive_fixtures::Fixture> fixtures() {
    std::vector<primitive_fixtures::Fixture> out = primitive_fixtures::all();
    primitive_fixtures::Fixture flat = primitive_fixtures::through();
    flat.name = "through-pieces";
    for (caliper_net::wire::Message& m : flat.messages)
        if (auto* u = std::get_if<caliper_net::wire::SceneUpdate>(&m)) u->params.erase("radius");
    out.push_back(flat);
    return out;
}

// The pieces' ID target as (pick id, item) pairs, top row first, in one id each: 0 where no piece
// drew, else the pair + 1.
inline std::vector<std::uint64_t> pair_ids(const std::vector<std::uint8_t>& rg32, std::size_t pixels) {
    std::vector<std::uint64_t> ids(pixels, 0);
    for (std::size_t i = 0; i < pixels && (i + 1) * 8 <= rg32.size(); ++i) {
        std::uint32_t pair[2];
        std::memcpy(pair, rg32.data() + i * 8, 8);
        ids[i] = pair[0] == 0 ? 0 : (std::uint64_t(pair[0]) << 32 | pair[1]) + 1;
    }
    return ids;
}

// The pick points: pixels of a piece's ID whose 3 by 3 neighbourhood shows that one ID (pick id and
// item), else whose four neighbours do, else whose two neighbours on one line through it do, as a
// line one pixel wide shows it. One for each program first, then one for each item, each the first
// such pixel in row order with the larger neighbourhood preferred, at most `most`.
inline std::vector<std::array<int, 2>> pick_points(const std::vector<std::uint64_t>& ids, int size,
                                                   std::size_t most = kPickPoints) {
    struct Candidate {
        int x = 0, y = 0, tier = 0;
    };
    auto at = [&](int x, int y) { return ids[std::size_t(y) * size + x]; };
    std::map<std::uint64_t, Candidate> by_program, by_item;
    std::vector<std::uint64_t> program_order, item_order;
    for (int y = 1; y + 1 < size; ++y)
        for (int x = 1; x + 1 < size; ++x) {
            const std::uint64_t own = at(x, y);
            if (own == 0) continue;
            const bool across = at(x - 1, y) == own && at(x + 1, y) == own, down = at(x, y - 1) == own && at(x, y + 1) == own;
            const bool diagonal = (at(x - 1, y - 1) == own && at(x + 1, y + 1) == own) ||
                                  (at(x + 1, y - 1) == own && at(x - 1, y + 1) == own);
            if (!across && !down && !diagonal) continue;
            const bool square = across && down && at(x - 1, y - 1) == own && at(x + 1, y - 1) == own &&
                                at(x - 1, y + 1) == own && at(x + 1, y + 1) == own;
            const Candidate c{x, y, square ? 3 : across && down ? 2 : 1};
            auto keep = [&](std::map<std::uint64_t, Candidate>& m, std::vector<std::uint64_t>& order, std::uint64_t key) {
                auto it = m.find(key);
                if (it == m.end()) {
                    m.emplace(key, c);
                    order.push_back(key);
                } else if (c.tier > it->second.tier) {
                    it->second = c;
                }
            };
            keep(by_program, program_order, (own - 1) >> 32);
            keep(by_item, item_order, own);
        }
    std::vector<std::array<int, 2>> out;
    std::set<std::pair<int, int>> taken;
    auto add = [&](const Candidate& c) {
        if (out.size() < most && taken.insert({c.x, c.y}).second) out.push_back({c.x, c.y});
    };
    for (std::uint64_t k : program_order) add(by_program[k]);
    for (std::uint64_t k : item_order) add(by_item[k]);
    return out;
}

// The hit of a pick as the event that it gives: the entity, the primitive, the index (an image's
// pixel row by row, as pick_detail() gives it), the source row and the pixel. The position from the
// depth goes apart, because it comes from each renderer's depth.
inline nlohmann::json hit_json(const caliper_live_views::SceneView& v, caliper_scene::PickPass::Result r,
                               const caliper_scene::Hit& h) {
    nlohmann::json j = nlohmann::json::object();
    if (r != caliper_scene::PickPass::Result::Hit) {
        j["result"] = int(r);
        return j;
    }
    const std::string path = v.tree().path(h.entity);
    std::int64_t index = std::int64_t(h.item);
    if (h.primitive == caliper_scene::Primitive::Image && h.part == caliper_scene::DrawPart::Batch) {
        const caliper_scene::Batch* b = v.tree().batch(h.entity, caliper_scene::Primitive::Image);
        index = h.has_pixel && b ? std::int64_t(h.pixel[1]) * b->image_width + h.pixel[0] : -1;
    }
    std::string key = path + " " + caliper_scene::primitive_name(h.primitive) + " " + std::to_string(index);
    if (h.part == caliper_scene::DrawPart::Frustum) key += " frustum";
    else key += " row " + std::to_string(v.row_id(path, h.primitive, index));
    if (h.has_pixel) key += " px " + std::to_string(h.pixel[0]) + "," + std::to_string(h.pixel[1]);
    j["hit"] = key;
    if (h.has_position) j["position"] = {h.position.x, h.position.y, h.position.z};
    j["depth"] = h.depth;   // the depth texel that the position comes from
    return j;
}

// The facts of the frame that the view drew last: its frame count, what drew its shapes, the program
// of each pick id of the pieces, the state of each piece's runtime, and the pick ids of the programs
// whose one ID covers what it does not draw alone: a batch that draws translucent (an image below
// full opacity, a colour or colours with alpha below 255), through which the shapes beneath show,
// and boxes with fill, whose translucent faces share the ID of their edges.
inline nlohmann::json frame_json(const caliper_live_views::SceneView& v) {
    nlohmann::json j = nlohmann::json::object();
    j["frame"] = v.cost().frames;
    j["draw"] = v.shape_draw() == caliper_live_views::ShapeDraw::pieces ? "pieces" : "builtin";
    const caliper_live_views::SceneShapeCounts c = v.shape_counts();
    j["pieces"] = c.pieces;
    j["builtin"] = c.builtin;
    j["fallbacks"] = nlohmann::json::object();
    for (const auto& [kind, n] : c.fallbacks) j["fallbacks"][kind] = n;
    j["programs"] = nlohmann::json::object();
    j["translucent"] = nlohmann::json::array();
    j["states"] = nlohmann::json::object();
    for (const caliper_draw::DrawRuntime* rt : v.shape_runtimes()) {
        const std::string& id = rt->instance().program;   // scene:<path>#<primitive> or #camera
        j["programs"][std::to_string(rt->pick_id())] = id;
        // The runtime's state after the frame: running, or the code of its refusal or stop.
        const caliper_draw::DrawStatus st = rt->status();
        j["states"][std::to_string(rt->pick_id())] =
            st.state == caliper_draw::DrawState::running ? std::string("running")
                                                         : std::to_string(int(st.state)) + " " + st.error.code + " " +
                                                               st.error.where + " " + st.error.message;
        const std::size_t hash = id.rfind('#');
        if (id.rfind("scene:", 0) != 0 || hash == std::string::npos) continue;
        const caliper_scene::EntityId e = v.tree().find(id.substr(6, hash - 6));
        for (int k = 1; k <= caliper_scene::kPrimitiveCount && e; ++k) {
            if (id.compare(hash + 1, std::string::npos, caliper_scene::primitive_name(caliper_scene::Primitive(k))) != 0)
                continue;
            const caliper_scene::Batch* b = v.tree().batch(e, caliper_scene::Primitive(k));
            const bool fill = caliper_scene::Primitive(k) == caliper_scene::Primitive::Boxes && b && b->fill;
            if (b && (b->translucent || b->opacity < 1.0f || (b->flat_rgba >> 24) < 255u || fill))
                j["translucent"].push_back(rt->pick_id());
        }
    }
    return j;
}

}  // namespace scene_parity
