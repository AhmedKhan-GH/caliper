// The scene core's draw lists through the graphics layer: every fixture scene compiled over a
// BackendResources on the recording backend, with its index clamps and edge gathers, its colour
// pass and its pick pass, must pass the layer's validation and be submitted.
#include "gfx/recording_backend.h"
#include "gfx/shaders/generated/programs_webgpu.h"
#include "scene/backend_resources.h"
#include "scene/pick.h"
#include "scene_row_fixture.h"

#include <doctest/doctest.h>

#include <cstring>
#include <string>

using namespace caliper_scene;
using namespace caliper_gfx;
namespace sh = caliper_gfx::shaders;

namespace {

struct Layer {
    RecordingBackend backend;
    BackendResources res{backend, sh::kWebgpuPrograms};
    Buffer data;          // one buffer holds every array of the fixtures, at their offsets
    Texture rgba, value, color, depth, ids, pick_depth;
    Layer() {
        std::string err;
        REQUIRE_MESSAGE(res.init(&err), err);
        BufferDesc d;
        d.size = 1 << 20;
        d.usage = kVertex | kIndex | kStorage | kCopyDst;
        data = backend.create_buffer(d);
        REQUIRE(data);
        TextureDesc t;
        t.width = 64;
        t.height = 48;
        t.usage = kSampled | kCopyDstTex;
        t.format = Format::RGBA8Unorm;
        rgba = backend.create_texture(t);
        t.format = Format::R32Uint;
        value = backend.create_texture(t);
        t.width = 400;
        t.height = 300;
        t.usage = kRenderTarget | kCopySrcTex;
        t.format = Format::RGBA8Unorm;
        color = backend.create_texture(t);
        t.format = Format::Depth32Float;
        depth = backend.create_texture(t);
        t.width = t.height = 16;
        t.format = Format::RG32Uint;
        ids = backend.create_texture(t);
        t.format = Format::Depth32Float;
        pick_depth = backend.create_texture(t);
        REQUIRE((rgba && value && color && depth && ids && pick_depth));
    }
    BatchArray array(Role role, uint64_t count, uint32_t stride, VertexFormat format, uint64_t offset) const {
        BatchArray a;
        a.role = role;
        a.buffer = data;
        a.offset = offset;
        a.count = count;
        a.stride = stride;
        a.format = format;
        return a;
    }
};

Batch bounded(Batch b, double lo = -1, double hi = 1) {
    b.bounds.add({lo, lo, lo});
    b.bounds.add({hi, hi, hi});
    return b;
}

// Compiles and submits the colour pass and a pick pass at the view's centre.
void submit(Layer& L, SceneTree& t, const ViewCamera& cam, uint32_t expect_draws) {
    t.update();
    L.res.update_class_table(t);
    SceneCompiler compiler;
    CompileOptions o;
    o.width = 400;
    o.height = 300;
    o.show_axes = true;
    Frame frame;
    compiler.prepare(t, L.res, frame);
    RenderTarget rt;
    rt.color = L.color;
    rt.depth = L.depth;
    DrawList list(rt);
    DrawMap map;
    const CompileResult r = compiler.compile(t, cam, o, L.res, list, &map);
    REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
    CHECK(r.skipped == 0);
    CHECK(list.records().size == expect_draws);
    frame.render(std::move(list));
    const Serial s = L.backend.submit(std::move(frame));
    CHECK_MESSAGE(s != 0, L.backend.last_error());
    CHECK_MESSAGE(L.res.error().empty(), L.res.error());
    // The pick pass.
    Picker picker;
    const PickWindow w = picker.request(200.0f, 150.0f, 8, 400, 300);
    const int32_t window[4] = {w.x0, w.y0, w.width, w.height};
    RenderTarget pt;
    pt.color = L.ids;
    pt.depth = L.pick_depth;
    DrawList pick(pt);
    DrawMap pick_map;
    REQUIRE(compiler.compile_pick(t, cam, o, window, L.res, pick, &pick_map).ok);
    Frame pf;
    pf.render(std::move(pick));
    CHECK_MESSAGE(L.backend.submit(std::move(pf)) != 0, L.backend.last_error());
    const Readback rb = L.backend.read_texture(L.ids, 0, 0, 16, 16);
    REQUIRE(rb);
    std::vector<uint8_t> bytes;
    REQUIRE(L.backend.take_readback(rb, &bytes));
    CHECK(bytes.size() == 16 * 16 * 8);
    // The recording backend does not rasterise: the cleared window decodes to no hit.
    CHECK_FALSE(picker.deliver(picker.current(), pick_map.revision, bytes, nullptr, pick_map));
}

}  // namespace

TEST_CASE("the scene core's draw lists pass the layer's validation: points, lines, arrows, boxes, meshes") {
    Layer L;
    SceneTree t;
    Batch pts = bounded(Batch{});
    pts.primitive = Primitive::Points;
    pts.arrays = {L.array(Role::Positions, 100, 12, VertexFormat::F32x3, 0),
                  L.array(Role::Values, 100, 4, VertexFormat::F32, 4096),
                  L.array(Role::Radii, 1, 0, VertexFormat::F32, 8192)};
    REQUIRE(t.set_batch(t.ensure("points"), pts));
    Batch segs = bounded(Batch{});
    segs.primitive = Primitive::Lines;
    segs.arrays = {L.array(Role::Positions, 6, 12, VertexFormat::F32x3, 0),
                   L.array(Role::Colors, 6, 4, VertexFormat::Unorm8x4, 4096)};
    REQUIRE(t.set_batch(t.ensure("segments"), segs));
    Batch strips = bounded(Batch{});
    strips.primitive = Primitive::Lines;
    strips.line_mode = LineMode::Strips;
    strips.arrays = {L.array(Role::Positions, 5, 12, VertexFormat::F32x3, 0),
                     L.array(Role::BreakMask, 5, 4, VertexFormat::U8, 12288)};
    REQUIRE(t.set_batch(t.ensure("strips"), strips));
    Batch edges = bounded(Batch{});
    edges.primitive = Primitive::Lines;
    edges.line_mode = LineMode::Edges;
    edges.arrays = {L.array(Role::Positions, 10, 12, VertexFormat::F32x3, 0),
                    L.array(Role::Indices, 7, 8, VertexFormat::S32, 16384),
                    L.array(Role::Colors, 10, 4, VertexFormat::Unorm8x4, 4096)};
    edges.seq = 1;
    REQUIRE(t.set_batch(t.ensure("edges"), edges));
    Batch arrows = bounded(Batch{});
    arrows.primitive = Primitive::Arrows;
    arrows.arrays = {L.array(Role::Origins, 4, 12, VertexFormat::F32x3, 0),
                     L.array(Role::Vectors, 4, 12, VertexFormat::F32x3, 256)};
    REQUIRE(t.set_batch(t.ensure("arrows"), arrows));
    Batch boxes = bounded(Batch{});
    boxes.primitive = Primitive::Boxes;
    boxes.arrays = {L.array(Role::Centers, 3, 12, VertexFormat::F32x3, 0),
                    L.array(Role::HalfSizes, 3, 12, VertexFormat::F32x3, 256),
                    L.array(Role::ClassIds, 3, 4, VertexFormat::S32, 512)};
    boxes.fill = true;
    REQUIRE(t.set_batch(t.ensure("boxes"), boxes));
    Batch mesh = bounded(Batch{});
    mesh.primitive = Primitive::Mesh;
    mesh.arrays = {L.array(Role::Positions, 100, 12, VertexFormat::F32x3, 0),
                   L.array(Role::Indices, 150, 12, VertexFormat::S32, 20480),
                   L.array(Role::Values, 100, 4, VertexFormat::F32, 4096)};
    mesh.seq = 1;
    REQUIRE(t.set_batch(t.ensure("mesh/flat"), mesh));
    mesh.arrays.push_back(L.array(Role::Normals, 100, 12, VertexFormat::F32x3, 32768));
    REQUIRE(t.set_batch(t.ensure("mesh/lit"), mesh));
    Batch tex = bounded(Batch{});
    tex.primitive = Primitive::Mesh;
    tex.arrays = {L.array(Role::Positions, 6, 12, VertexFormat::F32x3, 0),
                  L.array(Role::Uvs, 6, 8, VertexFormat::F32x2, 40960)};
    tex.texture = L.rgba;
    REQUIRE(t.set_batch(t.ensure("mesh/textured"), tex));
    REQUIRE(t.set_transform(t.find("mesh"), translation({1, 0, 0})));
    REQUIRE(t.set_classes({{1, "one", 0xff00ff00u}}));
    ViewCamera cam = Orbit{};
    t.update();
    fit(cam, t.scene_bounds(), 400.0 / 300.0);
    // Ten batch draws (boxes twice, edges and faces), and the axes of "mesh".
    submit(L, t, cam, 11);
    CHECK(L.backend.count(RecordingBackend::OpKind::Submit) == 2);
}

TEST_CASE("the scene core's draw lists pass the layer's validation: images, cameras, depth clouds, overlays, 2-D") {
    Layer L;
    SceneTree t;
    const EntityId cam = t.ensure("rig/cam");
    Pinhole pin;
    pin.fx = pin.fy = 50;
    pin.cx = 32;
    pin.cy = 24;
    pin.width = 64;
    pin.height = 48;
    REQUIRE(t.set_camera(cam, pin));
    Batch depth;
    depth.primitive = Primitive::Image;
    depth.image_form = ImageForm::Depth;
    depth.texture = L.value;
    depth.image_width = 64;
    depth.image_height = 48;
    depth.cloud = true;
    REQUIRE(t.set_batch(t.ensure("rig/cam/depth"), depth));
    Batch rgb = depth;
    rgb.image_form = ImageForm::Color;
    rgb.texture = L.rgba;
    rgb.cloud = false;
    REQUIRE(t.set_batch(t.ensure("rig/cam/rgb"), rgb));
    Batch seg = depth;
    seg.image_form = ImageForm::Segmentation;
    seg.cloud = false;
    seg.opacity = 0.5f;
    REQUIRE(t.set_batch(t.ensure("rig/cam/seg"), seg));
    Batch marks = Batch{};
    marks.primitive = Primitive::Points;
    marks.arrays = {L.array(Role::Positions, 3, 8, VertexFormat::F32x2, 0)};
    REQUIRE(t.set_batch(t.ensure("rig/cam/marks"), marks));
    Batch room = bounded(Batch{}, -3, 3);
    room.primitive = Primitive::Points;
    room.arrays = {L.array(Role::Positions, 10, 12, VertexFormat::F32x3, 0)};
    REQUIRE(t.set_batch(t.ensure("room"), room));
    ViewCamera orbit = Orbit{};
    t.update();
    fit(orbit, t.scene_bounds(), 400.0 / 300.0);
    // rig/cam's frustum and axes, depth quad, depth cloud, rgb, marks, room, and seg.
    submit(L, t, orbit, 8);
    ThroughPinhole through;
    through.camera = cam;
    // Through the camera: no frustum; depth cloud, room, then the pixel space flat over the picture.
    submit(L, t, ViewCamera(through), 6);

    // The 2-D row's scene.
    const scene_fixtures::SceneRowFixture fx;
    const std::vector<uint8_t> bytes = fx.bytes();
    REQUIRE(L.backend.write_buffer(L.data, 0, bytes.data(), bytes.size()));
    SceneTree flat(Space::k2D);
    REQUIRE(flat.set_batch(flat.root(), fx.batch(L.data)));
    Batch photo = rgb;
    REQUIRE(flat.set_batch(flat.ensure("photo"), photo));
    submit(L, flat, fx.camera(), 2);
}
