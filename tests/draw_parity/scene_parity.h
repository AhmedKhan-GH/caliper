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
// fallback, the program of each pick id, and the hits of picks. The pick points are interior pixels
// of the pieces' IDs on a grid, as the graphics suite's comparison picks them, at most kPickPoints of
// them, one for each program first. It depends on no test framework.
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

// The pick points: on the grid from 4 by 9 pixels, those whose 5 by 5 neighbourhood shows one ID of a
// piece; one for each program in the order of the grid, then the others in that order, at most `most`.
inline std::vector<std::array<int, 2>> pick_points(const std::vector<std::uint64_t>& ids, int size,
                                                   std::size_t most = kPickPoints) {
    std::vector<std::array<int, 2>> inside;
    for (int y = 4; y < size; y += 9)
        for (int x = 4; x < size; x += 9) {
            const std::uint64_t own = ids[std::size_t(y) * size + x];
            bool in = own != 0;
            for (int dy = -2; dy <= 2 && in; ++dy)
                for (int dx = -2; dx <= 2 && in; ++dx)
                    in = ids[std::size_t(std::clamp(y + dy, 0, size - 1)) * size + std::clamp(x + dx, 0, size - 1)] == own;
            if (in) inside.push_back({x, y});
        }
    std::vector<std::array<int, 2>> out;
    std::set<std::uint64_t> programs;
    std::vector<bool> taken(inside.size(), false);
    for (std::size_t i = 0; i < inside.size() && out.size() < most; ++i) {
        const std::uint64_t own = ids[std::size_t(inside[i][1]) * size + inside[i][0]];
        if (programs.insert((own - 1) >> 32).second) {
            out.push_back(inside[i]);
            taken[i] = true;
        }
    }
    for (std::size_t i = 0; i < inside.size() && out.size() < most; ++i)
        if (!taken[i]) out.push_back(inside[i]);
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
    return j;
}

// The facts of the frame that the view drew last: its frame count, what drew its shapes, and the
// program of each pick id of the pieces.
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
    for (const caliper_draw::DrawRuntime* rt : v.shape_runtimes())
        j["programs"][std::to_string(rt->pick_id())] = rt->instance().program;
    return j;
}

}  // namespace scene_parity
