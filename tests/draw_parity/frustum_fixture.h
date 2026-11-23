// frustum_fixture.h: the library piece frustum@1 in the fixture of the pixel comparison
// (design/programmable-graphics.md, Pixel Comparison), as one instance in the instance form of
// PROGRAM set with one version of its resource `frustum` as RESOURCE, written here as its derive
// half writes it:
//
//   library_frustum  the frustum of a camera of 32 by 18 pixels, fx 12, fy 9, off its centre, with
//                   its image plane at 1.6, turned toward the arcs' camera, 3 pixels wide at a
//                   pixel ratio of 1.5
//
// The plan draws it under the fixed camera of the arcs' row. parity_fixture.h calls
// add_frustum_piece() from make_fixture().
#pragma once

#include "boxes_mesh_fixture.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace draw_parity {

// Appends the program, the resource and the plan of frustum@1 to the fixture.
inline void add_frustum_piece(std::vector<std::uint8_t>* frames, Plan* plan, std::string* error) {
    namespace bm = boxes_mesh_detail;
    namespace ex = caliper_test_pieces;
    using caliper_net::Json;
    std::string why;
    caliper_builder::spec::Piece piece;
    caliper_net::wire::Program m;
    Json values = Json::object();
    values.set("width", 3.0);
    if (!bm::library_piece("frustum", &piece, &why) || !bm::program(piece, "library_frustum", "frustum", values, &m, &why)) {
        if (error->empty()) *error = why;
        return;
    }
    std::vector<std::uint8_t> bytes;
    std::string err;
    if (!caliper_net::wire::encode(m, &bytes, &err)) {
        if (error->empty()) *error = "the camera's PROGRAM does not encode: " + err;
        return;
    }
    append_frame(frames, bytes);
    // The frame: a turn of 0.5 about z and of -0.3 about x, at (-0.6, 0.2, -0.4), then the map of
    // the unit square onto the image plane, as the derive half composes it (column by column).
    const double fx = 12, fy = 9, cx = 3, cy = 10, w = 32, h = 18, d = 1.6;
    const double cz = std::cos(0.5), sz = std::sin(0.5), cxr = std::cos(-0.3), sxr = std::sin(-0.3);
    // R = Rz * Rx, columns.
    const double r[3][3] = {{cz, sz, 0.0}, {-sz * cxr, cz * cxr, sxr}, {sz * sxr, -cz * sxr, cxr}};
    const double t[3] = {-0.6, 0.2, -0.4};
    const double f[4][4] = {{w * d / fx, 0, 0, 0}, {0, h * d / fy, 0, 0}, {-cx * d / fx, -cy * d / fy, d, 0}, {0, 0, 0, 1}};
    float cols[4][4];
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            double v = 0;
            for (int k = 0; k < 4; ++k) {
                const double frame = k < 3 ? (row < 3 ? r[k][row] : 0.0) : (row < 3 ? t[row] : 1.0);
                v += frame * f[c][k];
            }
            cols[c][row] = float(v);
        }
    std::vector<std::uint8_t> rows(80, 0);
    std::memcpy(rows.data(), cols, 64);
    const std::uint64_t step = std::uint64_t(plan->step);
    caliper_net::wire::Resource res = bm::resource("library_frustum", "frustum", rows, 1, 80, step);
    bytes.clear();
    if (!caliper_net::wire::encode(res, &bytes, &err)) {
        if (error->empty()) *error = "the camera's RESOURCE does not encode: " + err;
        return;
    }
    append_frame(frames, bytes);
    const ex::Mat4 view = ex::view_matrix({0.0f, 1.9f, 2.6f}, {0.0f, 0.0f, 0.0f});
    const ex::Mat4 proj = ex::perspective(0.8f, 1.0f, 0.1f, 20.0f);
    ProgramPlan p;
    p.id = "library_frustum";
    p.pick_id = 103;
    p.inputs.clip_from_world = ex::multiply(proj, view);
    p.inputs.view_from_world = view;
    p.inputs.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    p.inputs.eye = {0.0f, 1.9f, 2.6f, 1.0f};
    p.inputs.viewport = {0, 0, float(plan->size), float(plan->size)};
    p.inputs.pixel_ratio = 1.5f;
    plan->programs.push_back(p);
}

}  // namespace draw_parity
