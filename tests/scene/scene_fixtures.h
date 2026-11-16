// scene_fixtures.h: batches and cameras for the scene core's tests. Buffers are plain handle
// values: the scene core never reads a buffer, it only describes and binds it.
#pragma once

#include "scene/scene.h"

#include <cstdint>
#include <vector>

namespace scene_fixtures {

using caliper_gfx::Buffer;
using caliper_gfx::VertexFormat;
using namespace caliper_scene;

// Distinct fake buffer handles per role, so a test can tell bindings apart.
inline Buffer buffer_for(Role r) { return Buffer{0x1000u + uint64_t(r)}; }

inline BatchArray array(Role role, uint64_t count, uint32_t stride, VertexFormat format, uint64_t offset = 0,
                        const void* cpu = nullptr) {
    BatchArray a;
    a.role = role;
    a.buffer = buffer_for(role);
    a.offset = offset;
    a.count = count;
    a.stride = stride;
    a.format = format;
    a.cpu = cpu;
    return a;
}

inline Pinhole pinhole(uint32_t w = 640, uint32_t h = 480, double f = 525.0) {
    Pinhole c;
    c.fx = c.fy = f;
    c.cx = w / 2.0;
    c.cy = h / 2.0;
    c.width = w;
    c.height = h;
    return c;
}

inline Batch points3(uint64_t n, const float* cpu = nullptr) {
    Batch b;
    b.primitive = Primitive::Points;
    b.arrays.push_back(array(Role::Positions, n, 12, VertexFormat::F32x3, 0, cpu));
    return b;
}

inline Batch points2(uint64_t n, const float* cpu = nullptr) {
    Batch b;
    b.primitive = Primitive::Points;
    b.arrays.push_back(array(Role::Positions, n, 8, VertexFormat::F32x2, 0, cpu));
    return b;
}

inline Batch lines3(uint64_t v, LineMode mode, uint64_t edges = 0, uint32_t dims = 3) {
    Batch b;
    b.primitive = Primitive::Lines;
    b.line_mode = mode;
    b.arrays.push_back(array(Role::Positions, v, dims * 4, dims == 3 ? VertexFormat::F32x3 : VertexFormat::F32x2));
    if (mode == LineMode::Strips) b.arrays.push_back(array(Role::BreakMask, v, 4, VertexFormat::U8));
    if (mode == LineMode::Edges) b.arrays.push_back(array(Role::Indices, edges, 8, VertexFormat::S32));
    return b;
}

inline Batch mesh3(uint64_t v, uint64_t triangles, uint32_t dims = 3) {
    Batch b;
    b.primitive = Primitive::Mesh;
    b.arrays.push_back(array(Role::Positions, v, dims * 4, dims == 3 ? VertexFormat::F32x3 : VertexFormat::F32x2));
    if (triangles) b.arrays.push_back(array(Role::Indices, triangles, 12, VertexFormat::S32));
    return b;
}

}  // namespace scene_fixtures
