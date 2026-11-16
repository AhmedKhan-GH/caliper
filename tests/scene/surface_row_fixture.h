// surface_row_fixture.h: the 3-D scene of the graphics suite's surface row, the first scene
// drawn through the whole renderer core (the scene core, BackendResources, the generated
// programs and a backend of the graphics layer).
//
// A small loss landscape, as demo 0 draws one (design/demos.md): a height field of 41 x 41
// vertices over the square from -1 to 1, coloured by its height through viridis, with an
// optimizer's path over it as a line strip, the current point as a disc marker, and a box of
// line segments around it, under the orbit camera about +z and its fit, with depth, at 512 x 512.
// Every array sits in one buffer at its own offset; the mesh's indices are at a multiple of 256
// bytes, because the index clamp reads them through a storage binding. The path and the marker
// are lifted above the surface and the box is padded around it, so no depth test of the row
// compares two nearly equal depths.
#pragma once

#include "scene/camera.h"
#include "scene/compile.h"
#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace scene_fixtures {

struct SurfaceRowFixture {
    static constexpr int kN = 41;                         // vertices per side
    static constexpr int kTriangles = (kN - 1) * (kN - 1) * 2;
    static constexpr int kPathVertices = 40;
    static constexpr int kSize = 512;                     // the target, square
    static constexpr uint32_t kClear = 0xff1e1a16u;       // r | g << 8 | b << 16 | a << 24
    static constexpr uint32_t kPathRgba = 0xffffffffu;
    static constexpr uint32_t kMarkerRgba = 0xff2a4cffu;
    static constexpr uint32_t kBoxRgba = 0xffb4b4b4u;
    static constexpr float kPathWidth = 4.0f, kBoxWidth = 3.0f, kMarkerSize = 14.0f;
    static constexpr uint32_t kColormap = 0;              // viridis (CALIPER_CMAP_VIRIDIS)

    std::vector<float> positions;   // kN * kN rows of (x, y, z)
    std::vector<float> values;      // the height of each vertex
    std::vector<int32_t> indices;   // kTriangles rows of three vertex indices
    std::vector<float> path;        // kPathVertices rows of (x, y, z)
    std::vector<uint32_t> keep;     // the strip's break mask, 4 bytes a vertex
    float marker[3] = {0, 0, 0};
    std::vector<float> box;         // 12 segments, 24 rows of (x, y, z)
    float vmin = 0, vmax = 0;
    uint64_t off_positions = 0, off_values = 0, off_indices = 0, off_path = 0, off_keep = 0, off_marker = 0,
             off_box = 0, bytes_total = 0;

    static double height(double x, double y) {
        return 0.16 * (x * x + 0.6 * y * y) + 0.045 * std::sin(3.1 * x + 0.5) * std::cos(2.3 * y - 0.3) + 0.03 * x * y;
    }

    SurfaceRowFixture() {
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kN; ++i) {
                const double x = -1.0 + 2.0 * i / (kN - 1), y = -1.0 + 2.0 * j / (kN - 1), z = height(x, y);
                positions.insert(positions.end(), {float(x), float(y), float(z)});
                values.push_back(float(z));
            }
        vmin = *std::min_element(values.begin(), values.end());
        vmax = *std::max_element(values.begin(), values.end());
        for (int j = 0; j + 1 < kN; ++j)
            for (int i = 0; i + 1 < kN; ++i) {
                const int32_t a = j * kN + i, b = a + 1, c = a + kN, d = c + 1;
                indices.insert(indices.end(), {a, b, d, a, d, c});
            }
        // The path spirals in from a corner towards the bowl's low ground.
        for (int k = 0; k < kPathVertices; ++k) {
            const double t = double(k) / (kPathVertices - 1);
            const double r = 0.95 * (1.0 - 0.85 * t), a = -2.2 + 4.2 * t;
            const double x = r * std::cos(a), y = 0.85 * r * std::sin(a);
            path.insert(path.end(), {float(x), float(y), float(height(x, y) + 0.035)});
            keep.push_back(k + 1 < kPathVertices ? 1u : 0u);
        }
        const float* last = &path[3 * (kPathVertices - 1)];
        marker[0] = last[0];
        marker[1] = last[1];
        marker[2] = float(height(last[0], last[1]) + 0.07);
        const float lo[3] = {-1.04f, -1.04f, vmin - 0.04f}, hi[3] = {1.04f, 1.04f, vmax + 0.12f};
        auto corner = [&](int c) { return std::vector<float>{c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]}; };
        const int edges[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7}, {7, 6}, {6, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (const auto& e : edges)
            for (int c : e) {
                const std::vector<float> p = corner(c);
                box.insert(box.end(), p.begin(), p.end());
            }
        auto place = [&](uint64_t bytes) {
            const uint64_t at = (bytes_total + 255) / 256 * 256;
            bytes_total = at + bytes;
            return at;
        };
        off_positions = place(positions.size() * 4);
        off_values = place(values.size() * 4);
        off_indices = place(indices.size() * 4);
        off_path = place(path.size() * 4);
        off_keep = place(keep.size() * 4);
        off_marker = place(12);
        off_box = place(box.size() * 4);
        bytes_total = (bytes_total + 255) / 256 * 256;
    }

    // The buffer's contents, every array at its offset.
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> b(bytes_total, 0);
        std::memcpy(b.data() + off_positions, positions.data(), positions.size() * 4);
        std::memcpy(b.data() + off_values, values.data(), values.size() * 4);
        std::memcpy(b.data() + off_indices, indices.data(), indices.size() * 4);
        std::memcpy(b.data() + off_path, path.data(), path.size() * 4);
        std::memcpy(b.data() + off_keep, keep.data(), keep.size() * 4);
        std::memcpy(b.data() + off_marker, marker, 12);
        std::memcpy(b.data() + off_box, box.data(), box.size() * 4);
        return b;
    }

    static caliper_scene::BatchArray array(caliper_scene::Role role, caliper_gfx::Buffer buffer, uint64_t offset,
                                           uint64_t count, uint32_t stride, caliper_gfx::VertexFormat format) {
        caliper_scene::BatchArray a;
        a.role = role;
        a.buffer = buffer;
        a.offset = offset;
        a.count = count;
        a.stride = stride;
        a.format = format;
        return a;
    }

    static caliper_scene::Aabb bounds_of(const std::vector<float>& rows) {
        caliper_scene::Aabb b;
        for (size_t i = 0; i + 2 < rows.size(); i += 3) b.add({rows[i], rows[i + 1], rows[i + 2]});
        return b;
    }

    // The scene: four entities under the root, each with one batch in `buffer`. False when the
    // store refuses a part.
    bool populate(caliper_scene::SceneTree& t, caliper_gfx::Buffer buffer) const {
        using namespace caliper_scene;
        using caliper_gfx::VertexFormat;
        bool ok = t.set_up(Axis::PosZ);
        Batch mesh;
        mesh.primitive = Primitive::Mesh;
        mesh.arrays = {array(Role::Positions, buffer, off_positions, uint64_t(kN) * kN, 12, VertexFormat::F32x3),
                       array(Role::Values, buffer, off_values, uint64_t(kN) * kN, 4, VertexFormat::F32),
                       array(Role::Indices, buffer, off_indices, kTriangles, 12, VertexFormat::S32)};
        mesh.colormap = kColormap;
        mesh.vmin = vmin;
        mesh.vmax = vmax;
        mesh.bounds = bounds_of(positions);
        mesh.seq = 1;
        ok = t.set_batch(t.ensure("surface"), mesh) && ok;

        Batch strip;
        strip.primitive = Primitive::Lines;
        strip.line_mode = LineMode::Strips;
        strip.arrays = {array(Role::Positions, buffer, off_path, kPathVertices, 12, VertexFormat::F32x3),
                        array(Role::BreakMask, buffer, off_keep, kPathVertices, 4, VertexFormat::U8)};
        strip.flat_rgba = kPathRgba;
        strip.width_px = kPathWidth;
        strip.bounds = bounds_of(path);
        ok = t.set_batch(t.ensure("path"), strip) && ok;

        Batch point;
        point.primitive = Primitive::Points;
        point.arrays = {array(Role::Positions, buffer, off_marker, 1, 12, VertexFormat::F32x3)};
        point.flat_rgba = kMarkerRgba;
        point.size_px = kMarkerSize;
        point.point_shape = PointShape::Disc;
        point.bounds.add({marker[0], marker[1], marker[2]});
        ok = t.set_batch(t.ensure("marker"), point) && ok;

        Batch edges;
        edges.primitive = Primitive::Lines;
        edges.arrays = {array(Role::Positions, buffer, off_box, 24, 12, VertexFormat::F32x3)};
        edges.flat_rgba = kBoxRgba;
        edges.width_px = kBoxWidth;
        edges.bounds = bounds_of(box);
        ok = t.set_batch(t.ensure("box"), edges) && ok;
        return ok;
    }

    // The orbit about +z at its default angles, fitted to the scene's bounds.
    caliper_scene::ViewCamera camera(const caliper_scene::SceneTree& t) const {
        caliper_scene::Orbit o;
        o.up = caliper_scene::Axis::PosZ;
        caliper_scene::ViewCamera cam = o;
        caliper_scene::fit(cam, t.scene_bounds(), 1.0);
        return cam;
    }

    caliper_scene::CompileOptions options() const {
        caliper_scene::CompileOptions o;
        o.width = o.height = kSize;
        o.pixel_scale = 1.0f;
        o.clear_rgba = kClear;
        return o;
    }
};

}  // namespace scene_fixtures
