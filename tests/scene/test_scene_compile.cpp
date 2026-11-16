// Compiled draw lists inspected as data: programs, entries, vertex layouts, buffers, offsets,
// strides, instance counts, uniform fields, groups and draw order, for every primitive the scene
// core compiles (src/host/scene/compile.h).
#include "scene/compile.h"
#include "fake_resources.h"
#include "scene_fixtures.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>

using namespace caliper_scene;
using namespace scene_fixtures;
namespace sh = caliper_gfx::shaders;

namespace {

RenderTarget target3d() {
    RenderTarget t;
    t.color = Texture{1};
    t.depth = Texture{2};
    return t;
}

struct Compiled {
    FakeResources res;
    SceneCompiler compiler;
    Frame frame;
    DrawList list{target3d()};
    DrawMap map;
    CompileResult result;
    const DrawRecord& rec(size_t i) const { return list.records()[i]; }
    const PipelineRecord& pipe(size_t i) const { return res.pipeline(rec(i).pipeline); }
    sh::PrimParams params(size_t i) const { return params_of(list, rec(i)); }
};

CompileOptions options(uint32_t w = 400, uint32_t h = 300) {
    CompileOptions o;
    o.width = w;
    o.height = h;
    o.pixel_scale = 2.0f;
    return o;
}

void run(Compiled& c, SceneTree& t, const ViewCamera& cam, const CompileOptions& o, bool depth = true) {
    t.update();
    c.compiler.prepare(t, c.res, c.frame);
    RenderTarget rt = target3d();
    if (!depth) rt.depth = Texture{};
    c.list.reset(rt);
    c.result = c.compiler.compile(t, cam, o, c.res, c.list, &c.map);
}

ViewCamera fitted_orbit(const SceneTree& t, double aspect) {
    ViewCamera v = Orbit{};
    std::get<Orbit>(v).up = t.up();
    fit(v, t.scene_bounds(), aspect);
    return v;
}

Batch bounded(Batch b, double lo = -1, double hi = 1) {
    b.bounds.add({lo, lo, lo});
    b.bounds.add({hi, hi, hi});
    return b;
}

const VertexLayout& layout(const PipelineRecord& p, int slot) { return p.desc.vertex_layouts[slot]; }

void check_stream(const Compiled& c, size_t draw, int slot, uint32_t location, VertexFormat fmt, Step step,
                  uint32_t stride, uint64_t buffer, uint64_t offset) {
    const PipelineRecord& p = c.pipe(draw);
    const VertexLayout& l = layout(p, slot);
    CHECK(l.attribute_count == 1);
    CHECK(l.attributes[0].location == location);
    CHECK(l.attributes[0].format == fmt);
    CHECK(l.step == step);
    CHECK(l.stride == stride);
    CHECK(c.rec(draw).vertex[slot].buffer.id == buffer);
    CHECK(c.rec(draw).vertex[slot].offset == offset);
}

uint64_t buf(Role r) { return buffer_for(r).id; }
constexpr uint64_t kGeo = FakeResources::kGeometry;

bool same_floats(const float* a, const Mat4& m, double tol) {
    for (int i = 0; i < 16; ++i)
        if (!(std::fabs(a[i] - m.m[i]) <= tol * std::max(1.0, std::fabs(m.m[i])))) return false;
    return true;
}

}  // namespace

TEST_CASE("points: instanced quads with per-point positions, values through a colormap, and one radius") {
    SceneTree t;
    const EntityId e = t.ensure("cloud");
    Batch b = bounded(points3(1000));
    b.arrays[0].offset = 256;
    b.arrays.push_back(array(Role::Values, 1000, 4, VertexFormat::F32, 4096));
    b.arrays.push_back(array(Role::Radii, 1, 0, VertexFormat::F32, 8192));
    b.colormap = 2;
    b.vmin = -1.0f;
    b.vmax = 3.0f;
    REQUIRE(t.set_batch(e, b));
    REQUIRE(t.set_transform(e, translation({0.5, 0, 0})));
    t.update();
    Compiled c;
    const ViewCamera cam = fitted_orbit(t, 400.0 / 300.0);
    run(c, t, cam, options());
    REQUIRE(c.result.ok);
    REQUIRE(c.list.records().size == 1);
    const PipelineRecord& p = c.pipe(0);
    CHECK(p.program == ProgramId::scene_points);
    CHECK(p.vs == "vs_points");
    CHECK(p.fs == "fs_points");
    CHECK(p.desc.vertex_layout_count == 3);
    check_stream(c, 0, 0, 0, VertexFormat::F32x3, Step::Instance, 12, buf(Role::Positions), 256);
    check_stream(c, 0, 1, 1, VertexFormat::U32, Step::Instance, 4, buf(Role::Values), 4096);
    check_stream(c, 0, 2, 2, VertexFormat::F32, Step::Instance, 0, buf(Role::Radii), 8192);
    CHECK(p.desc.topology == Topology::TriangleList);
    CHECK(p.desc.blend == Blend::Opaque);
    CHECK(p.desc.depth);
    CHECK(p.desc.depth_test);
    CHECK(p.desc.depth_write);
    CHECK(p.desc.color_format == Format::RGBA8Unorm);
    const DrawRecord& r = c.rec(0);
    CHECK(r.count == 6);
    CHECK(r.instance_count == 1000);
    CHECK_FALSE(r.index_buffer);
    CHECK(r.pick_id == 1);
    // Groups 0, 1 and 2 (the points program declares img_value for depth clouds); no group 3.
    CHECK(c.res.group(r.groups[0]).group == 0);
    CHECK(c.res.group(r.groups[1]).group == 1);
    CHECK(c.res.group(r.groups[2]).group == 2);
    CHECK_FALSE(r.groups[3]);
    // Dynamic offsets: the frame block first in the arena, then the draw block, then the colormap.
    CHECK(r.dynamic_offsets[0] == 0);
    CHECK(r.dynamic_offsets[1] % 256 == 0);
    CHECK(r.dynamic_offsets[2] % 256 == 0);
    const sh::Frame f = frame_of(c.list, r);
    CHECK(f.viewport[0] == 400.0f);
    CHECK(f.viewport[1] == 300.0f);
    CHECK(f.ndc_scale[0] == 1.0f);
    CHECK(f.ndc_offset[0] == 0.0f);
    CHECK(f.class_count == 65536);
    const sh::PrimParams q = c.params(0);
    CHECK(q.color_mode == 1);
    CHECK(q.radius_mode == 1);
    CHECK(q.point_shape == 1);
    CHECK(q.size_px == doctest::Approx(default_point_size(1000) * 2.0f));
    CHECK(q.radius_scale == doctest::Approx(1.0f));
    CHECK(q.pick_id == 1);
    CHECK(q.vmin == -1.0f);
    CHECK(q.vmax == 3.0f);
    const sh::Cmap m = cmap_of(c.list, r);
    CHECK(m.levels[0][1] == (0xC0000000u | 2u));
    CHECK(m.vmin == -1.0f);
    // proj_scale is the projection's scale, and mvp is P * V * M from the stated eye.
    const Orbit& o = std::get<Orbit>(cam);
    CHECK(f.proj_scale[1] == doctest::Approx(1.0 / std::tan(kOrbitFovY / 2)));
    const Vec3 eye = orbit_eye(o);
    const Aabb wb = t.scene_bounds();
    const double depth = dot(wb.center() - eye, normalize(Vec3{o.target[0], o.target[1], o.target[2]} - eye));
    const double r_ = wb.radius();
    const Mat4 want = perspective_z01(kOrbitFovY, 400.0 / 300.0, std::max(depth - 2 * r_, depth * 1e-3), depth + 2 * r_) *
                      look_at(eye, {o.target[0], o.target[1], o.target[2]}, {0, 1, 0}) * translation({0.5, 0, 0});
    CHECK(same_floats(q.mvp, want, 1e-6));
    // The map knows the draw.
    REQUIRE(c.map.find(1));
    CHECK(c.map.find(1)->entity == e);
    CHECK(c.map.find(1)->primitive == Primitive::Points);
    CHECK(c.map.find(1)->items == 1000);
    CHECK(c.map.revision == t.revision());
}

TEST_CASE("points: absent streams read the constant geometry at stride 0; flat colour; colours and class ids") {
    SceneTree t;
    Batch b = bounded(points3(10));
    b.flat_rgba = 0xff112233u;
    b.size_px = 4.0f;
    b.point_shape = PointShape::Square;
    REQUIRE(t.set_batch(t.ensure("a"), b));
    Batch rgba = bounded(points3(10));
    rgba.arrays.push_back(array(Role::Colors, 10, 4, VertexFormat::Unorm8x4, 64));
    REQUIRE(t.set_batch(t.ensure("b"), rgba));
    Batch cls = bounded(points3(10));
    cls.arrays.push_back(array(Role::ClassIds, 10, 4, VertexFormat::S32, 128));
    REQUIRE(t.set_batch(t.ensure("c"), cls));
    Compiled c;
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.result.ok);
    REQUIRE(c.list.records().size == 3);
    check_stream(c, 0, 1, 1, VertexFormat::U32, Step::Instance, 0, kGeo, StaticGeometry::kZero);
    check_stream(c, 0, 2, 2, VertexFormat::F32, Step::Instance, 0, kGeo, StaticGeometry::kZero);
    CHECK(c.params(0).color_mode == 0);
    CHECK(c.params(0).flat_rgba == 0xff112233u);
    CHECK(c.params(0).radius_mode == 0);
    CHECK(c.params(0).size_px == 8.0f);
    CHECK(c.params(0).point_shape == 0);
    check_stream(c, 1, 1, 1, VertexFormat::U32, Step::Instance, 4, buf(Role::Colors), 64);
    CHECK(c.params(1).color_mode == 2);
    check_stream(c, 2, 1, 1, VertexFormat::U32, Step::Instance, 4, buf(Role::ClassIds), 128);
    CHECK(c.params(2).color_mode == 4);
    // Draws without a colormap share one block.
    CHECK(c.rec(0).dynamic_offsets[2] == c.rec(1).dynamic_offsets[2]);
    CHECK(c.res.cmap_requests == 1);
    // A 2-D scene binds positions as f32x2 and draws without depth testing.
    SceneTree flat(Space::k2D);
    Batch p2 = points2(5);
    p2.bounds.add({0, 0, 0});
    p2.bounds.add({10, 5, 0});
    REQUIRE(flat.set_batch(flat.root(), p2));
    flat.update();
    Compiled d;
    ViewCamera pz = starting_camera(StartingCamera{}, flat, 400.0 / 300.0).camera;
    run(d, flat, pz, options());
    REQUIRE(d.result.ok);
    check_stream(d, 0, 0, 0, VertexFormat::F32x2, Step::Instance, 8, buf(Role::Positions), 0);
    CHECK(d.pipe(0).desc.depth);
    CHECK_FALSE(d.pipe(0).desc.depth_test);
    CHECK_FALSE(d.pipe(0).desc.depth_write);
    // proj_scale is positive with y down, so a radius never collapses to a pixel.
    CHECK(frame_of(d.list, d.rec(0)).proj_scale[1] > 0);
}

TEST_CASE("lines as segments: both ends from one positions buffer at offsets 0 and 12, stride 24") {
    SceneTree t;
    Batch b = bounded(lines3(6, LineMode::Segments));
    b.arrays[0].offset = 1024;
    b.arrays.push_back(array(Role::Colors, 6, 4, VertexFormat::Unorm8x4, 2048));
    b.width_px = 3.0f;
    REQUIRE(t.set_batch(t.ensure("segs"), b));
    Compiled c;
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.result.ok);
    REQUIRE(c.list.records().size == 1);
    const PipelineRecord& p = c.pipe(0);
    CHECK(p.program == ProgramId::scene_lines);
    CHECK(p.vs == "vs_lines");
    CHECK(p.fs == "fs_color");
    CHECK(p.desc.vertex_layout_count == 5);
    check_stream(c, 0, 0, 0, VertexFormat::F32x3, Step::Instance, 24, buf(Role::Positions), 1024);
    check_stream(c, 0, 1, 1, VertexFormat::F32x3, Step::Instance, 24, buf(Role::Positions), 1036);
    check_stream(c, 0, 2, 2, VertexFormat::U32, Step::Instance, 8, buf(Role::Colors), 2048);
    check_stream(c, 0, 3, 3, VertexFormat::U32, Step::Instance, 8, buf(Role::Colors), 2052);
    check_stream(c, 0, 4, 4, VertexFormat::U32, Step::Instance, 0, kGeo, StaticGeometry::kKeep);
    CHECK(c.rec(0).count == 6);
    CHECK(c.rec(0).instance_count == 3);
    CHECK(c.params(0).size_px == 6.0f);
    CHECK(c.params(0).line_mode == 0);
    CHECK(c.params(0).color_mode == 2);
    CHECK_FALSE(c.rec(0).groups[2]);                  // the lines program declares no group 2
}

TEST_CASE("lines as strips: positions bound twice a vertex apart, stride 12, and the break mask") {
    SceneTree t;
    Batch b = bounded(lines3(5, LineMode::Strips));
    b.arrays[1].offset = 512;
    b.arrays.push_back(array(Role::Values, 5, 4, VertexFormat::F32, 768));
    REQUIRE(t.set_batch(t.ensure("strip"), b));
    Batch two = bounded(lines3(4, LineMode::Strips, 0, 2));
    SceneTree flat(Space::k2D);
    REQUIRE(flat.set_batch(flat.root(), two));
    Compiled c;
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.result.ok);
    check_stream(c, 0, 0, 0, VertexFormat::F32x3, Step::Instance, 12, buf(Role::Positions), 0);
    check_stream(c, 0, 1, 1, VertexFormat::F32x3, Step::Instance, 12, buf(Role::Positions), 12);
    check_stream(c, 0, 2, 2, VertexFormat::U32, Step::Instance, 4, buf(Role::Values), 768);
    check_stream(c, 0, 3, 3, VertexFormat::U32, Step::Instance, 4, buf(Role::Values), 772);
    check_stream(c, 0, 4, 4, VertexFormat::U8, Step::Instance, 4, buf(Role::BreakMask), 512);
    CHECK(c.rec(0).instance_count == 4);
    CHECK(c.params(0).color_mode == 1);
    // 2-D strips: f32x2 at stride 8, offsets 0 and 8.
    Compiled d;
    run(d, flat, starting_camera(StartingCamera{}, flat, 1).camera, options());
    REQUIRE(d.result.ok);
    check_stream(d, 0, 0, 0, VertexFormat::F32x2, Step::Instance, 8, buf(Role::Positions), 0);
    check_stream(d, 0, 1, 1, VertexFormat::F32x2, Step::Instance, 8, buf(Role::Positions), 8);
}

TEST_CASE("index-pair edges: the edge gather once per update, then the segment draw over its streams") {
    SceneTree t;
    const EntityId e = t.ensure("graph/edges");
    Batch b = bounded(lines3(10, LineMode::Edges, 7));
    b.arrays[1].offset = 256;
    b.arrays.push_back(array(Role::Values, 7, 4, VertexFormat::F32, 512));
    b.per_edge = true;
    b.seq = 1;
    REQUIRE(t.set_batch(e, b));
    t.update();
    Compiled c;
    // Without prepare the batch waits.
    c.list.reset(target3d());
    c.result = c.compiler.compile(t, fitted_orbit(t, 1), options(), c.res, c.list, &c.map);
    CHECK(c.list.records().size == 0);
    CHECK(c.result.skipped == 1);
    // prepare emits one gather with its parameters in the frame's arena.
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.frame.passes().size == 1);
    const auto& d = std::get<DispatchPass>(c.frame.passes()[0]);
    CHECK(c.res.computes.back().first == ProgramId::edge_gather);
    CHECK(c.res.computes.back().second == "cs_gather_edges");
    CHECK(d.x == 1);
    sh::GatherParams gp;
    std::memcpy(&gp, c.frame.arena().data + d.dynamic_offsets[0], sizeof gp);
    CHECK(gp.count == 7);
    CHECK(gp.last == 9);
    CHECK(gp.dims == 3);
    CHECK(gp.per_vertex == 0);
    const auto& entries = c.res.compute_groups.back().second;
    REQUIRE(entries.size() == 6);
    CHECK(entries[0].buffer == kArena);
    CHECK(entries[1].buffer.id == buf(Role::Indices));
    CHECK(entries[1].offset == 256);
    CHECK(entries[1].size == 7 * 8);
    CHECK(entries[2].buffer.id == buf(Role::Positions));
    CHECK(entries[4].size == 7 * 24);
    // The draw: instance k is edge k; per-edge values bound once for both ends.
    REQUIRE(c.list.records().size == 1);
    const uint64_t seg = entries[4].buffer.id;
    check_stream(c, 0, 0, 0, VertexFormat::F32x3, Step::Instance, 24, seg, 0);
    check_stream(c, 0, 1, 1, VertexFormat::F32x3, Step::Instance, 24, seg, 12);
    check_stream(c, 0, 2, 2, VertexFormat::U32, Step::Instance, 4, buf(Role::Values), 512);
    check_stream(c, 0, 3, 3, VertexFormat::U32, Step::Instance, 4, buf(Role::Values), 512);
    CHECK(c.rec(0).instance_count == 7);
    // A second prepare with the same arrays does nothing; a new sequence number gathers again.
    Frame again;
    c.compiler.prepare(t, c.res, again);
    CHECK(again.passes().size == 0);
    b.seq = 2;
    b.per_edge = false;
    b.arrays[2] = array(Role::Colors, 10, 4, VertexFormat::Unorm8x4, 1024);
    REQUIRE(t.set_batch(e, b));
    run(c, t, fitted_orbit(t, 1), options());
    const auto& e2 = c.res.compute_groups.back().second;
    sh::GatherParams gp2;
    const auto& d2 = std::get<DispatchPass>(c.frame.passes().data[c.frame.passes().size - 1]);
    std::memcpy(&gp2, c.frame.arena().data + d2.dynamic_offsets[0], sizeof gp2);
    CHECK(gp2.per_vertex == 1);
    CHECK(e2[3].buffer.id == buf(Role::Colors));
    // Per-vertex colours come from the gathered sources, stride 8, ends at 0 and 4.
    check_stream(c, 0, 2, 2, VertexFormat::U32, Step::Instance, 8, e2[5].buffer.id, 0);
    check_stream(c, 0, 3, 3, VertexFormat::U32, Step::Instance, 8, e2[5].buffer.id, 4);
}

TEST_CASE("meshes: the index clamp, then an indexed draw over vertex buffers; normals, flat faces and textures") {
    SceneTree t;
    const EntityId e = t.ensure("surface");
    Batch b = bounded(mesh3(100, 150));
    b.arrays[1].offset = 4096;
    b.arrays.push_back(array(Role::Values, 100, 4, VertexFormat::F32, 8192));
    b.seq = 5;
    REQUIRE(t.set_batch(e, b));
    Compiled c;
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.result.ok);
    // The clamp: 450 indices, each to at most vertex 99, in ceil(450 / 64) groups.
    REQUIRE(c.frame.passes().size == 1);
    const auto& d = std::get<DispatchPass>(c.frame.passes()[0]);
    CHECK(d.x == 8);
    sh::ClampParams cp;
    std::memcpy(&cp, c.frame.arena().data + d.dynamic_offsets[0], sizeof cp);
    CHECK(cp.count == 450);
    CHECK(cp.last == 99);
    const auto& entries = c.res.compute_groups.back().second;
    CHECK(c.res.compute_groups.back().first == ProgramId::index_clamp);
    CHECK(entries[1].buffer.id == buf(Role::Indices));
    CHECK(entries[1].offset == 4096);
    CHECK(entries[1].size == 1800);
    REQUIRE(c.list.records().size == 1);
    const DrawRecord& r = c.rec(0);
    CHECK(r.index_buffer.id == entries[2].buffer.id);
    CHECK(r.index_offset == 0);
    CHECK(r.count == 450);
    CHECK(r.instance_count == 1);
    const PipelineRecord& p = c.pipe(0);
    CHECK(p.program == ProgramId::scene_mesh);
    CHECK(p.vs == "vs_mesh");
    CHECK(p.fs == "fs_mesh_flat");
    CHECK(p.desc.vertex_layout_count == 6);
    check_stream(c, 0, 0, 0, VertexFormat::F32x3, Step::Vertex, 12, buf(Role::Positions), 0);
    check_stream(c, 0, 1, 1, VertexFormat::F32x3, Step::Vertex, 0, kGeo, StaticGeometry::kZero);
    check_stream(c, 0, 2, 2, VertexFormat::U32, Step::Vertex, 4, buf(Role::Values), 8192);
    check_stream(c, 0, 3, 3, VertexFormat::F32x2, Step::Vertex, 0, kGeo, StaticGeometry::kZero);
    const VertexLayout& pose = layout(p, 4);
    CHECK(pose.attribute_count == 4);
    CHECK(pose.stride == 0);
    CHECK(pose.step == Step::Instance);
    for (uint32_t a = 0; a < 4; ++a) {
        CHECK(pose.attributes[a].location == 4 + a);
        CHECK(pose.attributes[a].offset == 16 * a);
    }
    CHECK(r.vertex[4].buffer.id == kGeo);
    CHECK(r.vertex[4].offset == StaticGeometry::kIdentityPose);
    check_stream(c, 0, 5, 8, VertexFormat::U32, Step::Instance, 0, kGeo, StaticGeometry::kZero);
    const sh::PrimParams q = c.params(0);
    CHECK(q.shade_mode == 0);
    CHECK(q.color_mode == 1);
    CHECK(q.vertex_count == 100);
    CHECK(q.use_instance == 0);
    // With normals: the headlight in the vertex stage and fs_color, the normal matrix filled.
    b.arrays.push_back(array(Role::Normals, 100, 12, VertexFormat::F32x3, 16384));
    REQUIRE(t.set_batch(e, b));
    run(c, t, fitted_orbit(t, 1), options());
    CHECK(c.pipe(0).fs == "fs_color");
    CHECK(c.params(0).shade_mode == 1);
    check_stream(c, 0, 1, 1, VertexFormat::F32x3, Step::Vertex, 12, buf(Role::Normals), 16384);
    CHECK(std::fabs(c.params(0).nmat0[0]) + std::fabs(c.params(0).nmat1[0]) + std::fabs(c.params(0).nmat2[0]) > 0);
    // Textured: white times the texture, the image in group 2 at binding 0.
    Batch tex = bounded(mesh3(4, 0));
    tex.arrays.push_back(array(Role::Uvs, 4, 8, VertexFormat::F32x2, 64));
    tex.texture = Texture{0x77};
    REQUIRE(t.set_batch(t.ensure("textured"), tex));
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.list.records().size == 2);
    CHECK(c.pipe(1).fs == "fs_mesh_tex");
    CHECK(c.params(1).color_mode == 3);
    CHECK(c.rec(1).count == 3);   // no indices: each three vertices a triangle
    CHECK_FALSE(c.rec(1).index_buffer);
    const GroupRecord& g2 = c.res.group(c.rec(1).groups[2]);
    CHECK(g2.image.id == 0x77);
    CHECK(g2.binding == kImageRgba);
}

TEST_CASE("arrows, boxes with fill, and their streams") {
    SceneTree t;
    Batch a;
    a.primitive = Primitive::Arrows;
    a.arrays.push_back(array(Role::Origins, 4, 12, VertexFormat::F32x3));
    a.arrays.push_back(array(Role::Vectors, 4, 12, VertexFormat::F32x3, 256));
    a.arrays.push_back(array(Role::Colors, 4, 4, VertexFormat::Unorm8x4, 512));
    REQUIRE(t.set_batch(t.ensure("arrows"), bounded(a)));
    Batch bx;
    bx.primitive = Primitive::Boxes;
    bx.arrays.push_back(array(Role::Centers, 3, 12, VertexFormat::F32x3));
    bx.arrays.push_back(array(Role::HalfSizes, 3, 12, VertexFormat::F32x3, 64));
    bx.arrays.push_back(array(Role::Rotations, 3, 16, VertexFormat::F32x4, 128));
    bx.fill = true;
    REQUIRE(t.set_batch(t.ensure("boxes"), bounded(bx)));
    Compiled c;
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.result.ok);
    REQUIRE(c.list.records().size == 3);
    // Opaque first in path order (arrows, box edges), then the translucent faces.
    CHECK(c.pipe(0).vs == "vs_lines");
    CHECK(c.params(0).line_mode == 1);
    CHECK(c.rec(0).count == 9);
    CHECK(c.rec(0).instance_count == 4);
    check_stream(c, 0, 0, 0, VertexFormat::F32x3, Step::Instance, 12, buf(Role::Origins), 0);
    check_stream(c, 0, 1, 1, VertexFormat::F32x3, Step::Instance, 12, buf(Role::Vectors), 256);
    check_stream(c, 0, 2, 2, VertexFormat::U32, Step::Instance, 4, buf(Role::Colors), 512);
    check_stream(c, 0, 3, 3, VertexFormat::U32, Step::Instance, 4, buf(Role::Colors), 512);
    CHECK(c.pipe(1).program == ProgramId::scene_boxes);
    CHECK(c.pipe(1).vs == "vs_box_wire");
    CHECK(c.rec(1).count == 72);
    check_stream(c, 1, 2, 2, VertexFormat::F32x4, Step::Instance, 16, buf(Role::Rotations), 128);
    check_stream(c, 1, 3, 3, VertexFormat::U32, Step::Instance, 0, kGeo, StaticGeometry::kZero);
    CHECK(c.pipe(2).vs == "vs_box_solid");
    CHECK(c.rec(2).count == 36);
    CHECK(c.pipe(2).desc.blend == Blend::Alpha);
    CHECK(c.pipe(2).desc.depth_test);
    CHECK_FALSE(c.pipe(2).desc.depth_write);
    CHECK(c.map.find(3)->part == DrawPart::BoxFaces);
    // Without rotations the identity quaternion at stride 0; 2-D boxes draw 24 and 6 vertices.
    SceneTree flat(Space::k2D);
    Batch b2;
    b2.primitive = Primitive::Boxes;
    b2.arrays.push_back(array(Role::Centers, 2, 8, VertexFormat::F32x2));
    b2.arrays.push_back(array(Role::HalfSizes, 2, 8, VertexFormat::F32x2, 64));
    b2.fill = true;
    REQUIRE(flat.set_batch(flat.root(), bounded(b2)));
    Compiled d;
    run(d, flat, starting_camera(StartingCamera{}, flat, 1).camera, options());
    REQUIRE(d.result.ok);
    check_stream(d, 0, 2, 2, VertexFormat::F32x4, Step::Instance, 0, kGeo, StaticGeometry::kIdentityQuat);
    CHECK(d.rec(0).count == 24);
    CHECK(d.rec(1).count == 6);
}

TEST_CASE("draw order: opaque first, then translucent, each in the byte order of the paths") {
    SceneTree t;
    Batch translucent = bounded(points3(3));
    translucent.flat_rgba = 0x80ffffffu;
    REQUIRE(t.set_batch(t.ensure("b"), translucent));
    REQUIRE(t.set_batch(t.ensure("c"), bounded(points3(3))));
    REQUIRE(t.set_batch(t.ensure("a/z"), bounded(points3(3))));
    REQUIRE(t.set_batch(t.ensure("a-y"), bounded(lines3(2, LineMode::Segments))));
    Batch tl = bounded(lines3(2, LineMode::Segments));
    tl.translucent = true;
    REQUIRE(t.set_batch(t.ensure("a"), tl));
    Compiled c;
    run(c, t, fitted_orbit(t, 1), options());
    REQUIRE(c.result.ok);
    std::vector<std::string> order;
    for (size_t i = 0; i < c.list.records().size; ++i) order.push_back(t.path(c.map.find(c.rec(i).pick_id)->entity));
    CHECK(order == std::vector<std::string>{"a-y", "a/z", "c", "a", "b"});
    for (size_t i = 0; i < c.list.records().size; ++i) CHECK(c.rec(i).pick_id == i + 1);
    CHECK(c.pipe(3).desc.blend == Blend::Alpha);
    CHECK(c.pipe(4).desc.blend == Blend::Alpha);
    // Hidden entities and non-finite poses are not drawn.
    t.set_visible(t.find("c"), false);
    Mat4 bad;
    bad.m[0] = NAN;
    REQUIRE(t.set_transform(t.find("a"), bad));
    run(c, t, fitted_orbit(t, 1), options());
    order.clear();
    for (size_t i = 0; i < c.list.records().size; ++i) order.push_back(t.path(c.map.find(c.rec(i).pick_id)->entity));
    CHECK(order == std::vector<std::string>{"a-y", "b"});
}

TEST_CASE("an entity transform change rewrites only that entity's parameter slots") {
    SceneTree t;
    const EntityId a = t.ensure("a");
    const EntityId b = t.ensure("b");
    REQUIRE(t.set_batch(a, bounded(points3(50))));
    REQUIRE(t.set_batch(b, bounded(lines3(6, LineMode::Segments))));
    REQUIRE(t.set_batch(t.ensure("a/child"), bounded(mesh3(9, 0))));
    // A large static setup keeps the scene's bounds, and so the near and far planes, unchanged.
    REQUIRE(t.set_batch(t.ensure("big"), bounded(points3(4), -100, 100), true));
    const ViewCamera cam = fitted_orbit(t, 1);   // the same camera both times
    Compiled first, second;
    run(first, t, cam, options());
    REQUIRE(t.set_transform(b, translation({0.25, 0, 0})));
    run(second, t, cam, options());
    REQUIRE(first.list.records().size == 4);
    REQUIRE(second.list.records().size == 4);
    CHECK(first.list.arena().size == second.list.arena().size);
    for (size_t i = 0; i < 4; ++i) {
        const DrawRecord &x = first.rec(i), &y = second.rec(i);
        CHECK(first.res.pipeline(x.pipeline).program == second.res.pipeline(y.pipeline).program);
        CHECK(std::memcmp(x.dynamic_offsets, y.dynamic_offsets, sizeof x.dynamic_offsets) == 0);
        CHECK(std::memcmp(x.vertex, y.vertex, sizeof x.vertex) == 0);
        CHECK(x.count == y.count);
        CHECK(x.instance_count == y.instance_count);
    }
    // The arenas differ only inside b's draw block.
    const uint32_t lo = first.rec(2).dynamic_offsets[1];   // a, a/child, b in path order
    CHECK(first.map.find(3)->entity == b);
    const auto* x = first.list.arena().data;
    const auto* y = second.list.arena().data;
    bool outside_same = true, inside_differs = false;
    for (size_t i = 0; i < first.list.arena().size; ++i) {
        const bool inside = i >= lo && i < lo + sizeof(sh::PrimParams);
        if (!inside && x[i] != y[i]) outside_same = false;
        if (inside && x[i] != y[i]) inside_differs = true;
    }
    CHECK(outside_same);
    CHECK(inside_differs);
}

TEST_CASE("images: in 2-D at the entity's origin, on a camera's plane in 3-D, and flat through the camera") {
    SceneTree flat(Space::k2D);
    Batch img;
    img.primitive = Primitive::Image;
    img.texture = Texture{0x55};
    img.image_width = 64;
    img.image_height = 48;
    img.opacity = 0.5f;
    REQUIRE(flat.set_batch(flat.ensure("photo"), img));
    flat.update();
    Compiled c;
    const ViewCamera cam2d = starting_camera(StartingCamera{}, flat, 1).camera;
    run(c, flat, cam2d, options());
    REQUIRE(c.result.ok);
    REQUIRE(c.list.records().size == 1);
    CHECK(c.pipe(0).program == ProgramId::scene_image);
    CHECK(c.pipe(0).fs == "fs_image_rgba");
    CHECK(c.pipe(0).desc.vertex_layout_count == 0);
    CHECK(c.pipe(0).desc.blend == Blend::Alpha);
    CHECK(c.rec(0).count == 6);
    CHECK(c.params(0).flat_rgba == ((128u << 24) | 0xffffffu));
    CHECK(c.res.group(c.rec(0).groups[2]).image.id == 0x55);
    // The unit square maps onto (0, 0)-(64, 48) in scene units: check against the pan and zoom.
    const CameraMatrices cm = camera_matrices(cam2d, flat, 400, 300);
    const Mat4 want = cm.proj * scaling({64, 48, 1});
    CHECK(same_floats(c.params(0).mvp, want, 1e-6));

    // In 3-D under a camera: depth, segmentation and colour forms; the depth cloud; the frustum.
    SceneTree t;
    const EntityId cam = t.ensure("rig/cam");
    Pinhole pin = pinhole(64, 48, 50);
    pin.plane_distance = 0.5;
    REQUIRE(t.set_camera(cam, pin));
    Batch depth = img;
    depth.image_form = ImageForm::Depth;
    depth.opacity = 1.0f;
    depth.cloud = true;
    depth.colormap = 1;
    depth.vmin = 0.5f;
    depth.vmax = 4.0f;
    REQUIRE(t.set_batch(t.ensure("rig/cam/depth"), depth));
    Batch seg = img;
    seg.image_form = ImageForm::Segmentation;
    REQUIRE(t.set_batch(t.ensure("rig/cam/seg"), seg));
    REQUIRE(t.set_batch(t.ensure("room"), bounded(points3(10), -3, 3)));
    Batch outside = img;
    REQUIRE(t.set_batch(t.ensure("loose"), outside));
    t.update();
    Compiled d;
    const ViewCamera cam3 = fitted_orbit(t, 1);
    run(d, t, cam3, options());
    REQUIRE(d.result.ok);
    CHECK(d.result.skipped == 1);   // the image outside a camera
    // Opaque: depth quad, depth cloud, frustum (rig/cam), room; translucent: seg.
    std::vector<std::pair<std::string, DrawPart>> got;
    for (size_t i = 0; i < d.list.records().size; ++i) {
        const DrawMapEntry* m = d.map.find(d.rec(i).pick_id);
        got.push_back({d.pipe(i).vs + "/" + d.pipe(i).fs, m->part});
    }
    REQUIRE(got.size() == 5);
    CHECK(got[0].first == "vs_lines/fs_color");          // rig/cam: the frustum
    CHECK(got[0].second == DrawPart::Frustum);
    CHECK(got[1].first == "vs_image/fs_image_value");    // rig/cam/depth
    CHECK(got[2].first == "vs_depth_points/fs_points");
    CHECK(got[2].second == DrawPart::DepthCloud);
    CHECK(got[3].first == "vs_points/fs_points");        // room
    CHECK(got[4].first == "vs_image/fs_image_class");    // translucent seg
    CHECK(d.res.group(d.rec(1).groups[2]).binding == kImageValue);
    CHECK(d.res.group(d.rec(4).groups[2]).binding == kImageClass);
    CHECK(d.rec(2).instance_count == 64 * 48);
    CHECK(d.params(2).color_mode == 1);
    CHECK(d.params(4).color_mode == 4);
    // The frustum's unit corner (1, 1, 1) lands on pixel (W, H) of the plane.
    const CameraMatrices m3 = camera_matrices(cam3, t, 400, 300);
    auto clip_of = [&](const float* mvp, Vec3 p) {
        double h[4];
        transform_h(Mat4::from(mvp), p, h);
        return Vec3{h[0] / h[3], h[1] / h[3], h[2] / h[3]};
    };
    const Vec3 corner = clip_of(d.params(0).mvp, {1, 1, 1});
    const Vec3 pixel = transform_point(t.world(t.find("rig/cam/depth")), {64, 48, 0});
    double h[4];
    transform_h(m3.proj * m3.view, pixel, h);
    CHECK(corner.x == doctest::Approx(h[0] / h[3]).epsilon(1e-5));
    CHECK(corner.y == doctest::Approx(h[1] / h[3]).epsilon(1e-5));
    // The depth cloud: pixel (x, y) at depth z enters as (x z, y z, z) and lands on the camera ray.
    const double z = 2.0, x = 10.5, y = 20.5;
    const Vec3 cloud = clip_of(d.params(2).mvp, {x * z, y * z, z});
    const Vec3 ray = transform_point(t.camera_frame(cam), {(x - 32) / 50 * z, (y - 24) / 50 * z, z});
    transform_h(m3.proj * m3.view, ray, h);
    CHECK(cloud.x == doctest::Approx(h[0] / h[3]).epsilon(1e-5));
    CHECK(cloud.y == doctest::Approx(h[1] / h[3]).epsilon(1e-5));

    // Through the camera: its pixel space flat over the picture, last, without depth tests;
    // its own frustum not drawn; the letterbox scissor on every draw.
    ThroughPinhole tp;
    tp.camera = cam;
    Compiled e;
    run(e, t, ViewCamera(tp), options(400, 300));
    REQUIRE(e.result.ok);
    got.clear();
    for (size_t i = 0; i < e.list.records().size; ++i) {
        const DrawMapEntry* m = e.map.find(e.rec(i).pick_id);
        got.push_back({e.pipe(i).fs, m->part});
        CHECK(e.rec(i).scissor[0] == 0);
        CHECK(e.rec(i).scissor[2] == 400);
    }
    REQUIRE(got.size() == 4);
    CHECK(got[0].first == "fs_points");          // the depth cloud, 3-D content
    CHECK(got[1].first == "fs_points");          // room
    CHECK(got[2].first == "fs_image_value");     // overlay, opaque
    CHECK(got[3].first == "fs_image_class");     // overlay, translucent
    CHECK(e.map.find(e.rec(2).pick_id)->overlay);
    CHECK_FALSE(e.pipe(2).desc.depth_test);
    CHECK_FALSE(e.pipe(2).desc.depth_write);
    // The overlay's mvp maps the unit square onto the letterboxed picture: (1, 1) to the
    // bottom-right corner, 4:3 in a 4:3 view.
    const Vec3 br = clip_of(e.params(2).mvp, {1, 1, 0});
    CHECK(br.x == doctest::Approx(1.0));
    CHECK(br.y == doctest::Approx(-1.0));
}

TEST_CASE("entity axes and frustum options; id_target is refused") {
    SceneTree t;
    const EntityId e = t.ensure("a");
    REQUIRE(t.set_transform(e, translation({1, 0, 0})));
    REQUIRE(t.set_batch(e, bounded(points3(2))));
    CompileOptions o = options();
    o.show_axes = true;
    o.axes_length = 0.5;
    Compiled c;
    run(c, t, fitted_orbit(t, 1), o);
    REQUIRE(c.result.ok);
    REQUIRE(c.list.records().size == 2);
    CHECK(c.map.find(2)->part == DrawPart::Axes);
    check_stream(c, 1, 0, 0, VertexFormat::F32x3, Step::Instance, 24, kGeo, StaticGeometry::kAxes);
    check_stream(c, 1, 1, 1, VertexFormat::F32x3, Step::Instance, 24, kGeo, StaticGeometry::kAxes + 12);
    check_stream(c, 1, 2, 2, VertexFormat::U32, Step::Instance, 8, kGeo, StaticGeometry::kAxesColors);
    CHECK(c.rec(1).instance_count == 3);
    CHECK(c.params(1).color_mode == 2);
    o.id_target = true;
    Compiled d;
    run(d, t, fitted_orbit(t, 1), o);
    CHECK_FALSE(d.result.ok);
    CHECK(d.list.records().size == 0);
    // The constant geometry: identity pose, keep, the frustum's last side and the axes' colours.
    uint8_t geo[StaticGeometry::kBytes];
    StaticGeometry::fill(geo);
    float f[16];
    std::memcpy(f, geo + StaticGeometry::kIdentityPose, 64);
    CHECK(f[0] == 1.0f);
    CHECK(f[15] == 1.0f);
    CHECK(f[12] == 0.0f);
    uint32_t keep;
    std::memcpy(&keep, geo + StaticGeometry::kKeep, 4);
    CHECK(keep == 0xffffffffu);
    float side[6];
    std::memcpy(side, geo + StaticGeometry::kFrustum + 14 * 12, 24);   // the plane's fourth side
    CHECK(side[0] == 0.0f);
    CHECK(side[1] == 1.0f);
    CHECK(side[2] == 1.0f);
    CHECK(side[3] == 0.0f);
    CHECK(side[4] == 0.0f);
    CHECK(side[5] == 1.0f);
}

TEST_CASE("every compiled pipeline matches the generated program description") {
    // check_vertex_layouts refuses what the description does not allow.
    RenderPipelineDesc d;
    d.vertex_entry = "vs_points";
    d.fragment_entry = "fs_points";
    d.color_format = Format::RGBA8Unorm;
    d.vertex_layout_count = 2;
    d.vertex_layouts[0].attribute_count = 1;
    d.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x3, 0};
    d.vertex_layouts[1].attribute_count = 1;
    d.vertex_layouts[1].attributes[0] = {1, VertexFormat::U32, 0};
    std::string why;
    CHECK_FALSE(check_vertex_layouts(ProgramId::scene_points, d, &why));   // location 2 not fed
    CHECK(why.find("location 2") != std::string::npos);
    d.vertex_layout_count = 3;
    d.vertex_layouts[2].attribute_count = 1;
    d.vertex_layouts[2].attributes[0] = {2, VertexFormat::F32, 0};
    CHECK(check_vertex_layouts(ProgramId::scene_points, d, &why));
    d.vertex_layouts[2].attributes[0] = {2, VertexFormat::U32, 0};
    CHECK_FALSE(check_vertex_layouts(ProgramId::scene_points, d, &why));
    d.vertex_layouts[2].attributes[0] = {2, VertexFormat::F32, 0};
    d.fragment_entry = "fs_points_pick";
    CHECK_FALSE(check_vertex_layouts(ProgramId::scene_points, d, &why));   // an integer output into RGBA8
    d.color_format = Format::RG32Uint;
    CHECK(check_vertex_layouts(ProgramId::scene_points, d, &why));
    d.fragment_entry = "fs_nothing";
    CHECK_FALSE(check_vertex_layouts(ProgramId::scene_points, d, &why));
}

TEST_CASE("labels: projected anchors of points and boxes, nearest first, from CPU-visible arrays") {
    SceneTree t;
    const float pts[] = {0, 0, 0, 0, 0, -5, 100, 100, 100};
    Batch b = bounded(points3(3, pts), -5, 5);
    b.labels = {"near", "far", "outside"};
    REQUIRE(t.set_batch(t.ensure("p"), b));
    t.update();
    Orbit o;
    o.target[0] = o.target[1] = o.target[2] = 0;
    o.distance = 10;
    o.azimuth = kPi / 2;   // the eye on +z
    o.elevation = 0;
    const auto labels = project_labels(t, ViewCamera(o), 400, 300);
    REQUIRE(labels.size() == 2);
    CHECK(labels[0].text == "near");
    CHECK(labels[0].x == doctest::Approx(200));
    CHECK(labels[0].y == doctest::Approx(150));
    CHECK(labels[1].text == "far");
    CHECK(labels[1].depth > labels[0].depth);
}
