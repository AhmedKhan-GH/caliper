// surface_fixture.h: the program of the library piece surface@1 in the pixel comparison of the app
// and the browser module (design/programmable-graphics.md, Pixel Comparison). It draws a terrain of
// 48 by 64 cells with a hole, as the derive half publishes it for a tensor that needs no pooling,
// colored by height through viridis, with its box and ranges taken from the data by the three
// compute passes of the draw half, under a fixed perspective camera at a pixel ratio of 1.
//
// parity_fixture.h includes this file at its end, after its helpers (Fixture, instance_message and
// resource_message), and calls add_surface_program() from make_fixture().
#pragma once

#include "parity_fixture.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace draw_parity {

// One element of the resource `cells` of surface@1, 32 bytes.
struct SurfaceCell {
    float x, y, z, value;
    std::int32_t row, column, step;
    std::uint32_t cols;
};

// The cells of the terrain, row by row, as the derive half writes them with a pool factor of 1.
inline std::vector<SurfaceCell> surface_cells(int rows = 48, int cols = 64) {
    std::vector<SurfaceCell> out;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const double u = double(r) / (rows - 1), v = double(c) / (cols - 1);
            float h = float(0.4 * std::sin(5.0 * u) * std::cos(4.0 * v) + 0.3 * u +
                            0.5 * std::exp(-((u - 0.6) * (u - 0.6) + (v - 0.35) * (v - 0.35)) / 0.02));
            if (r >= 30 && r < 36 && c >= 44 && c < 52) h = std::numeric_limits<float>::quiet_NaN();
            out.push_back({float(r), float(c), h, h, r, c, -1, std::uint32_t(cols)});
        }
    return out;
}

// The piece surface@1 as its piece.toml declares it, built here because the parity binaries do not
// link the TOML reader, with the files and hash of the built-in library. The test
// caliper_surface_match checks that it equals the library's piece.
inline caliper_builder::spec::Piece surface_piece() {
    namespace cp = caliper_builder::programs;
    namespace cs = caliper_builder::spec;
    cs::Piece p;
    p.name = "surface";
    p.version = 1;
    p.description = "A height field of R by C cells as a lit surface in a data box, colored by height or by a value "
                    "through a colormap, with each cell that has a NaN corner as a hole";
    p.imports = {"pick@1"};
    auto tensor = [](const char* name, const char* title) {
        cs::PieceInput in;
        in.name = name;
        in.rank_min = in.rank_max = 2;
        in.title = title;
        return in;
    };
    p.inputs = {tensor("height", "Heights (R, C)"),
                tensor("values", "Values (R, C) for the colormap; a tensor of another shape colors by height")};
    auto param = [](const char* name, cs::ParamType type, double value, std::optional<double> lo,
                    std::optional<double> hi, const char* title) {
        cs::PieceParam q;
        q.name = name;
        q.type = type;
        q.default_value = value;
        q.min = lo;
        q.max = hi;
        q.title = title;
        return q;
    };
    const cs::ParamType f32 = cs::ParamType::f32, u32 = cs::ParamType::u32;
    const std::optional<double> none;
    p.params = {param("max_cells", u32, 32768, 1, 262144, "Most Cells, Pooled Above"),
                param("pool", u32, 0, 0, 2, "Pool (0 mean, 1 max, 2 min)"),
                param("colormap", u32, 0, 0, 15, "Colormap (an id of the host's colormaps, 0 viridis)"),
                param("vmin", f32, 0, none, none, "Value at the Colormap's Start"),
                param("vmax", f32, 1, none, none, "Value at the Colormap's End"),
                param("x_lo", f32, 0, none, none, "Data Box Start along the Rows"),
                param("x_hi", f32, 1, none, none, "Data Box End along the Rows"),
                param("y_lo", f32, 0, none, none, "Data Box Start along the Columns"),
                param("y_hi", f32, 1, none, none, "Data Box End along the Columns"),
                param("z_lo", f32, 0, none, none, "Data Box Start along the Heights"),
                param("z_hi", f32, 1, none, none, "Data Box End along the Heights"),
                param("box_x", f32, 0, none, none, "Box Side along the Rows (0 for the data's proportions)"),
                param("box_y", f32, 0, none, none, "Box Side along the Columns"),
                param("box_z", f32, 0, none, none, "Box Side along the Heights"),
                param("depth_offset", cs::ParamType::i32, 0, -8, 8, "Depth Offset toward the Camera"),
                param("fit", u32, 31, 0, 31, "Ranges from the Data (1 vmin, 2 vmax, 4 rows, 8 columns, 16 heights)")};
    cs::Resource cells;
    cells.id = "cells";
    cells.layout.type_name = "Cell";
    for (const char* f : {"x: f32", "y: f32", "z: f32", "value: f32", "row: i32", "column: i32", "step: i32", "cols: u32"}) {
        const std::string t(f);
        cells.layout.fields.push_back({t.substr(0, t.find(':')), t.substr(t.find(':') + 2)});
    }
    cells.count = 262144;
    cells.counted = true;
    p.resources = {cells};
    cs::Derive derive;
    cs::DerivePass pool;
    pool.entry = "pool_cells";
    pool.workgroups = {1024, 1, 1};
    pool.writes = {"cells"};
    derive.passes = {pool};
    p.derive = derive;
    cs::Draw draw;
    draw.camera = cs::CameraMode::orbit;
    draw.pick = cs::PickMap{"cells", {"x", "y", "z", "value", "row", "column", "step"}};
    cs::DrawBuffer fit;
    fit.id = "fit";
    fit.layout.element = "vec4<f32>";
    fit.count = 8322;
    draw.buffers = {fit};
    auto compute = [](const char* entry, std::uint32_t groups) {
        cs::DrawPass c;
        c.kind = cs::PassKind::compute;
        c.entry = entry;
        c.workgroups = {groups, 1, 1};
        c.writes = {"fit"};
        return c;
    };
    cs::DrawPass cells_pass;
    cells_pass.vertex = "cell_vs";
    cells_pass.fragment = "cell_fs";
    cells_pass.instances_of = "cells";
    cells_pass.vertices = 6;
    cells_pass.blend = cs::DrawBlend::opaque;
    cells_pass.depth = cs::DrawDepth::write;
    cells_pass.pick = true;
    draw.passes = {compute("fit_threads", 16), compute("fit_groups", 1), compute("fit_total", 1), cells_pass};
    p.draw = draw;
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string_view prefix = "pieces/surface/1/";
        if (f.path.substr(0, prefix.size()) == prefix && f.path.size() > 5 && f.path.substr(f.path.size() - 5) == ".wgsl")
            p.files[std::string(f.path.substr(prefix.size()))] = std::string(f.text);
    }
    p.hash = cp::piece_hash(p);
    return p;
}

inline void add_surface_program(Fixture* fx) {
    if (!fx->error.empty()) return;
    const caliper_builder::spec::Piece piece = surface_piece();
    if (piece.files.count("draw.wgsl") == 0) {
        fx->error = "the library has no piece surface@1";
        return;
    }

    caliper_net::Json values = caliper_net::Json::object();
    values.set("colormap", 0);
    values.set("fit", 31);
    caliper_net::wire::Program m;
    std::string why;
    if (!instance_message(piece, "surface_terrain", "terrain", values, piece.hash, &m, &why)) {
        fx->error = why;
        return;
    }
    m.bind = caliper_net::Json::object();
    m.bind.set("height", "live:grid.height");
    m.bind.set("values", "live:grid.height");
    const std::vector<SurfaceCell> cells = surface_cells();
    for (const caliper_net::wire::Message& message :
         {caliper_net::wire::Message(m),
          caliper_net::wire::Message(resource_message("surface_terrain", "cells", cells, std::uint64_t(fx->plan.step)))}) {
        std::vector<std::uint8_t> bytes;
        if (!caliper_net::wire::encode(message, &bytes, &why)) {
            fx->error = "a frame of the surface does not encode: " + why;
            return;
        }
        append_frame(&fx->frames, bytes);
    }

    ProgramPlan sp;
    sp.id = "surface_terrain";
    sp.pick_id = 1u << 25;
    const ex::Mat4 view = ex::view_matrix({0.35f, -1.7f, 2.3f}, {0.0f, 0.0f, 0.0f});
    const ex::Mat4 proj = ex::perspective(0.8f, 1.0f, 0.1f, 20.0f);
    sp.inputs.clip_from_world = ex::multiply(proj, view);
    sp.inputs.view_from_world = view;
    sp.inputs.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    sp.inputs.eye = {0.35f, -1.7f, 2.3f, 1.0f};
    sp.inputs.viewport = {0, 0, float(fx->plan.size), float(fx->plan.size)};
    sp.inputs.pixel_ratio = 1.0f;
    fx->plan.programs.push_back(sp);
}

}  // namespace draw_parity
