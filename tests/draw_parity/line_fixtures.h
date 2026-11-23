// line_fixtures.h: the library pieces lines@1, arrows@1 and path@1 in the fixture of the pixel
// comparison of the app and the browser module (make_fixture of parity_fixture.h calls
// add_line_pieces). Each program comes as the app sends it, the instance form of PROGRAM set with
// the library piece (line_pieces.h), and one version of each published resource as RESOURCE, in
// the layout that the piece's derive half writes, made here on the CPU: two strips through
// viridis, a ring of arrows of packed RGBA, and a path through 200 rows through magma. All three
// draw under the fixed camera of the arcs at a pixel ratio of 1.
#pragma once

#include "line_pieces.h"
#include "parity_draw.h"

#include "../gfx/example_pieces.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace draw_parity {

namespace line_fixture {

struct Segment {   // the element of lines@1's `segments`, 40 bytes
    float ax, ay, az;
    std::uint32_t color_a;
    float bx, by, bz;
    std::uint32_t color_b;
    std::uint32_t index_a, index_b;
};
struct Info {      // the element of `info`, 12 bytes
    float lo, hi;
    std::uint32_t flags;
};
struct Arrow {     // the element of arrows@1's `arrows`, 32 bytes
    float ox, oy, oz;
    std::uint32_t color;
    float vx, vy, vz;
    std::uint32_t index;
};
struct Vertex {    // the element of path@1's `vertices`, 20 bytes
    float x, y, z, value;
    std::uint32_t row;
};
static_assert(sizeof(Segment) == 40 && sizeof(Info) == 12 && sizeof(Arrow) == 32 && sizeof(Vertex) == 20,
              "the layouts of the pieces' resources");

inline std::uint32_t bits_of(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

// RESOURCE of one version of a resource, its elements of the piece's layout.
template <typename T>
caliper_net::wire::Resource version(const std::string& program, const std::string& id, const std::vector<T>& v,
                                    std::int64_t step) {
    caliper_net::wire::Resource r;
    r.program = program;
    r.resource = id;
    r.step = std::uint64_t(step);
    r.count = std::uint32_t(v.size());
    r.stride = std::uint32_t(sizeof(T));
    r.data.name = "resource";
    r.data.dtype = CALIPER_NET_DTYPE_U8;
    r.data.shape = {std::int64_t(v.size() * sizeof(T))};
    r.data.data = reinterpret_cast<const std::uint8_t*>(v.data());
    r.data.size = v.size() * sizeof(T);
    return r;
}

}  // namespace line_fixture

// Adds the three programs, their resources and their plans to the fixture `fx`: `add` encodes a
// frame, and `message` makes the PROGRAM set of an instance (instance_message).
template <typename Fixture, typename Add, typename Message>
void add_line_pieces(Fixture& fx, const Add& add, const Message& message) {
    namespace lf = line_fixture;
    namespace ex = caliper_test_pieces;
    using caliper_net::Json;
    auto program = [&](const caliper_builder::spec::Piece& piece, const std::string& id, const std::string& panel,
                       const Json& values, const std::string& pin) {
        caliper_net::wire::Program m;
        std::string why;
        if (!message(piece, id, panel, values, pin, &m, &why)) {
            if (fx.error.empty()) fx.error = why;
            return;
        }
        m.bind = Json::object();
        for (const caliper_builder::spec::PieceInput& in : piece.inputs) m.bind.set(in.name, "live:fixture." + in.name);
        add(m);
    };
    auto piece_of = [&](const char* name, std::string_view manifest, std::string_view hash,
                        caliper_builder::spec::Piece* piece) {
        std::string why;
        if (!line_piece(name, manifest, hash, piece, &why) && fx.error.empty()) fx.error = why;
        return std::string(hash);
    };
    ProgramPlan plan;
    const ex::Mat4 view = ex::view_matrix({0.0f, 1.9f, 2.6f}, {0.0f, 0.3f, 0.0f});
    plan.inputs.clip_from_world = ex::multiply(ex::perspective(0.8f, 1.0f, 0.1f, 20.0f), view);
    plan.inputs.view_from_world = view;
    plan.inputs.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    plan.inputs.eye = {0.0f, 1.9f, 2.6f, 1.0f};
    plan.inputs.viewport = {0, 0, float(fx.plan.size), float(fx.plan.size)};
    plan.inputs.pixel_ratio = 1.0f;

    {   // Two strips of 30 and 34 vertices, a value at an exact level at each vertex, width 3.
        caliper_builder::spec::Piece piece;
        const std::string pin = piece_of("lines", kLinesManifest, kLinesHash, &piece);
        Json values = Json::object();
        values.set("width", 3.0);
        values.set("mode", 1);
        values.set("color_mode", 1);
        values.set("range", 0);
        program(piece, "lines_strips", "lines", values, pin);
        std::vector<std::array<float, 3>> v;
        for (int i = 0; i < 64; ++i) {
            const float t = float(i) / 63.0f, r = 0.3f + 0.6f * t;
            v.push_back({r * std::cos(11.0f * t), 0.2f + 0.4f * t, r * std::sin(11.0f * t)});
        }
        std::vector<lf::Segment> seg;
        for (std::uint32_t i = 0; i + 1 < 64; ++i) {
            const std::uint32_t a = lf::bits_of(float((i * 37u) % 256u) / 255.0f);
            const std::uint32_t b = lf::bits_of(float(((i + 1) * 37u) % 256u) / 255.0f);
            seg.push_back({v[i][0], v[i][1], v[i][2], a, v[i + 1][0], v[i + 1][1], v[i + 1][2], b, i,
                           i == 29 ? 0xffffffffu : i + 1});
        }
        add(lf::version("lines_strips", "segments", seg, std::uint64_t(fx.plan.step)));
        add(lf::version("lines_strips", "info", std::vector<lf::Info>{{0.0f, 1.0f, 0u}}, std::uint64_t(fx.plan.step)));
        ProgramPlan p = plan;
        p.id = "lines_strips";
        p.pick_id = 4;
        fx.plan.programs.push_back(p);
    }
    {   // A ring of 24 arrows, a packed RGBA each, width 3.
        caliper_builder::spec::Piece piece;
        const std::string pin = piece_of("arrows", kArrowsManifest, kArrowsHash, &piece);
        Json values = Json::object();
        values.set("width", 3.0);
        values.set("color_mode", 2);
        program(piece, "arrows_ring", "arrows", values, pin);
        std::vector<lf::Arrow> arrows;
        for (std::uint32_t i = 0; i < 24; ++i) {
            const float a = 6.2831853f * float(i) / 24.0f;
            const std::uint32_t c = (60u + 8u * i) | (200u - 6u * i) << 8 | (90u + 5u * i) << 16 | 255u << 24;
            arrows.push_back({0.5f * std::cos(a), 0.1f * float(i % 4), 0.5f * std::sin(a), c, 0.35f * std::cos(a), 0.2f,
                              0.35f * std::sin(a), i});
        }
        add(lf::version("arrows_ring", "arrows", arrows, std::uint64_t(fx.plan.step)));
        add(lf::version("arrows_ring", "info", std::vector<lf::Info>{{0.0f, 1.0f, 0u}}, std::uint64_t(fx.plan.step)));
        ProgramPlan p = plan;
        p.id = "arrows_ring";
        p.pick_id = 5;
        fx.plan.programs.push_back(p);
    }
    {   // A path through 200 rows, the row through magma with the range of the rows, width 3.
        caliper_builder::spec::Piece piece;
        const std::string pin = piece_of("path", kPathManifest, kPathHash, &piece);
        Json values = Json::object();
        values.set("width", 3.0);
        values.set("colormap", 1);
        program(piece, "path_rows", "path", values, pin);
        std::vector<lf::Vertex> rows;
        for (std::uint32_t i = 0; i < 200; ++i) {
            const float t = 5.5f * float(i) / 199.0f;
            rows.push_back({0.8f * std::sin(2.0f * t), 0.3f + 0.4f * std::cos(3.0f * t), 0.6f * std::sin(5.0f * t + 0.3f),
                            float(i), i});
        }
        add(lf::version("path_rows", "vertices", rows, std::uint64_t(fx.plan.step)));
        ProgramPlan p = plan;
        p.id = "path_rows";
        p.pick_id = 6;
        fx.plan.programs.push_back(p);
    }
}

}  // namespace draw_parity
