// primitive_rows.h: the rows of primitive_fixtures.h through a scene view, shared by the graphics
// suite on Metal (tests/gfx/gfx_scene_view.cpp) and the browser module's self-test on WebGPU
// (web/wasm/src/scene_selftest.cpp). A row encodes each message of its fixture with the C++ codec,
// decodes it again and applies it to a scene view, as a source hands decoded messages over, draws
// the view at 256 x 256, and once the caller has read the target back compares it with the CPU
// reference of draw_reference.h. It depends on no test framework.
#pragma once

#include "draw_reference.h"
#include "live/live_scene.h"
#include "net/wire.h"
#include "primitive_fixtures.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace primitive_rows {

namespace wire = caliper_net::wire;

struct Row {
    primitive_fixtures::Fixture fixture;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> frames;   // the decoded messages point into these
    std::deque<wire::Message> decoded;                           // stable: the reference points into them
    draw_reference::SceneParts parts;
    std::string error;

    // Applies the fixture to the view and draws it. False with `error` when a message did not encode,
    // decode or apply, or the view did not draw.
    bool draw(caliper_live_views::SceneView& view) {
        view.describe(fixture.info);
        for (const wire::Message& m : fixture.messages) {
            auto bytes = std::make_shared<std::vector<uint8_t>>();
            std::string err;
            if (!wire::encode(m, bytes.get(), &err, CALIPER_NET_MINOR_3_2)) return fail("encode: " + err);
            wire::Message out;
            const wire::DecodeResult r = wire::decode(bytes->data(), bytes->size(), &out);
            if (!r.ok()) return fail("decode: " + r.message);
            frames.push_back(bytes);
            decoded.push_back(std::move(out));
            const wire::Message& d = decoded.back();
            bool ok = true;
            std::string why;
            if (const auto* u = std::get_if<wire::SceneUpdate>(&d)) {
                ok = view.apply(*u, &why);
                parts.add(*u);
            } else if (const auto* t = std::get_if<wire::SceneTransforms>(&d)) {
                ok = view.apply(*t, &why);
            } else if (const auto* c = std::get_if<wire::SceneClasses>(&d)) {
                ok = view.apply(*c, &why);
                parts.classes(*c);
            } else if (const auto* k = std::get_if<wire::SceneClear>(&d)) {
                ok = view.apply(*k, &why);
            }
            if (!ok) return fail("apply: " + why);
        }
        if (!fixture.look_through.empty()) {
            std::string why;
            if (!view.look_through(fixture.look_through, &why)) return fail("look through: " + why);
        }
        view.keep_draws(true);
        std::string why;
        if (!view.render(primitive_fixtures::kSize, primitive_fixtures::kSize, 1.0f, primitive_fixtures::kClear, &why))
            return fail("render: " + why);
        return true;
    }

    // The CPU reference of what the view drew, and the comparison of `pixels` (the view's target,
    // tightly packed RGBA8) with it.
    struct Result {
        draw_reference::Comparison c;
        size_t clipped = 0, triangles = 0, draws = 0;
        std::vector<size_t> covered;   // per draw: the pixels it wrote in the reference
        std::vector<std::string> problems;
        std::string sentence;
    };
    Result compare(const caliper_live_views::SceneView& view, const std::vector<uint8_t>& pixels) const {
        Result out;
        draw_reference::Rasteriser r(view.last_draws(), view.last_map(), view.tree(), parts,
                                     int(primitive_fixtures::kSize), int(primitive_fixtures::kSize));
        r.run();
        out.c = draw_reference::compare(r.ref, pixels);
        out.clipped = r.ref.clipped;
        out.triangles = r.ref.triangles;
        out.draws = view.last_draws().records().size;
        out.covered = r.ref.writes;
        out.problems = r.problems;
        out.sentence = out.c.sentence(fixture.name);
        return out;
    }

private:
    bool fail(const std::string& why) {
        error = fixture.name + ": " + why;
        return false;
    }
};

}  // namespace primitive_rows
