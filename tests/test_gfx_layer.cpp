// The graphics layer's backend-neutral core on the recording backend, with no GPU
// (src/host/gfx/, docs/applets/graphics-layer.md): handles, descriptor and draw validation
// against WebGPU's default limits, the uniform arena, frames, serials, deferred release,
// asynchronous readback and the routes of import, including the failure paths.
#include <doctest/doctest.h>

#include "gfx/colormap_keys.h"
#include "gfx/colormap_tables.h"
#include "gfx/recording_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "gfx/shaders/generated/uniforms.h"
#include "gfx/generated_pipelines.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace caliper_gfx;

namespace {

// The colormap block of the generated programs (shaders::Cmap), 4,112 bytes, which the arena
// pads to 4,352.
constexpr uint32_t kCmapBytes = sizeof(shaders::Cmap);
constexpr uint32_t kCmapSlice = (kCmapBytes + 255) / 256 * 256;
static_assert(kCmapBytes == 4112 && kCmapSlice == 4352, "the generated colormap block");

// A program shaped like the scene draws: group 0 the frame block, group 1 the draw block and
// a colormap block at dynamic offsets, one vertex and one fragment entry.
const EntryDesc kEntries[] = {
    {"vs_quad", kStageVertex, {1, 1, 1}},
    {"fs_quad", kStageFragment, {1, 1, 1}},
    {"cs_fill", kStageCompute, {8, 8, 1}},
};
const BindingDesc kBindings[] = {
    {0, 0, BindingType::Uniform, kStageVertex | kStageFragment, true, 48, Format::RGBA8Unorm, 0},
    {1, 0, BindingType::Uniform, kStageVertex | kStageFragment, true, 192, Format::RGBA8Unorm, 1},
    {1, 1, BindingType::Uniform, kStageVertex, true, kCmapBytes, Format::RGBA8Unorm, 2},
};
ProgramDesc quad_program(const BindingDesc* b = kBindings, uint8_t n = 3) {
    ProgramDesc d;
    d.label = "quad";
    d.entries = kEntries;
    d.entry_count = 3;
    d.bindings = b;
    d.binding_count = n;
    return d;
}

RenderPipelineDesc quad_pipeline(Program p) {
    RenderPipelineDesc d;
    d.program = p;
    d.vertex_entry = "vs_quad";
    d.fragment_entry = "fs_quad";
    d.vertex_layout_count = 2;
    d.vertex_layouts[0].stride = 8;                     // a corner, per vertex
    d.vertex_layouts[0].attribute_count = 1;
    d.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x2, 0};
    d.vertex_layouts[1].stride = 12;                    // offset and colour, per instance
    d.vertex_layouts[1].step = Step::Instance;
    d.vertex_layouts[1].attribute_count = 2;
    d.vertex_layouts[1].attributes[0] = {1, VertexFormat::F32x2, 0};
    d.vertex_layouts[1].attributes[1] = {2, VertexFormat::U32, 8};
    d.topology = Topology::TriangleList;
    d.id_target = true;
    d.depth = true;
    d.depth_test = true;
    d.depth_write = true;
    return d;
}

// A small scene: target, program, pipeline, bind groups on the arena and two buffers.
struct Scene {
    RecordingBackend gfx;
    Texture color, id, depth;
    Program program;
    Pipeline pipeline;
    BindGroup g0, g1;
    Buffer corners, instances;

    explicit Scene(const Caps& caps = RecordingBackend::default_caps()) : gfx(caps) {
        color = gfx.create_texture({16, 16, Format::RGBA8Unorm, kRenderTarget | kCopySrcTex, "color"});
        id = gfx.create_texture({16, 16, Format::RG32Uint, kRenderTarget | kCopySrcTex, "id"});
        depth = gfx.create_texture({16, 16, Format::Depth32Float, kRenderTarget | kCopySrcTex, "depth"});
        program = gfx.create_program(quad_program());
        pipeline = gfx.create_render_pipeline(quad_pipeline(program));
        BindGroupDesc b0;
        b0.program = program;
        b0.group = 0;
        b0.entry_count = 1;
        b0.entries[0].binding = 0;
        b0.entries[0].buffer = kArena;
        g0 = gfx.create_bind_group(b0);
        BindGroupDesc b1;
        b1.program = program;
        b1.group = 1;
        b1.entry_count = 2;
        b1.entries[0].binding = 0;
        b1.entries[0].buffer = kArena;
        b1.entries[1].binding = 1;
        b1.entries[1].buffer = kArena;
        g1 = gfx.create_bind_group(b1);
        const float c[12] = {0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
        corners = gfx.create_buffer({sizeof c, kVertex | kCopyDst, false, "corners"});
        gfx.write_buffer(corners, 0, c, sizeof c);
        instances = gfx.create_buffer({12 * 4, kVertex | kCopyDst, false, "instances"});
    }

    RenderTarget target(uint32_t clear = 0xff102030u) const {
        RenderTarget t;
        t.color = color;
        t.id = id;
        t.depth = depth;
        t.clear_rgba = clear;
        return t;
    }

    // One instanced draw of `n` quads with the given arena offsets.
    DrawRecord draw(DrawList& list, uint32_t n) const {
        uint8_t frame_block[48] = {}, params[192] = {}, cmap[kCmapBytes] = {};
        DrawRecord r;
        r.pipeline = pipeline;
        r.groups[0] = g0;
        r.groups[1] = g1;
        r.dynamic_offsets[0] = list.uniform(frame_block, sizeof frame_block);
        r.dynamic_offsets[1] = list.uniform(params, sizeof params);
        r.dynamic_offsets[2] = list.uniform(cmap, sizeof cmap);
        r.vertex[0] = {corners, 0};
        r.vertex[1] = {instances, 0};
        r.count = 6;
        r.instance_count = n;
        r.pick_id = 7;
        return r;
    }
};

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

}  // namespace

TEST_CASE("gfx layer: handles carry their kind, die at release and are never reused") {
    RecordingBackend gfx;
    Buffer a = gfx.create_buffer({64, kVertex | kCopyDst});
    REQUIRE(a);
    CHECK(handle_kind(a.id) == HandleKind::Buffer);
    CHECK(gfx.alive(a));
    const uint32_t v[4] = {1, 2, 3, 4};
    CHECK(gfx.write_buffer(a, 0, v, sizeof v));
    gfx.release(a);
    CHECK_FALSE(gfx.alive(a));
    CHECK_FALSE(gfx.write_buffer(a, 0, v, sizeof v));   // a stale handle
    CHECK(has(gfx.last_error(), "released"));
    Buffer b = gfx.create_buffer({64, kVertex});
    REQUIRE(b);
    CHECK(handle_slot(b.id) == handle_slot(a.id));      // the slot comes back
    CHECK(b.id != a.id);                                // with a new generation
    CHECK_FALSE(gfx.alive(a));
    Texture as_texture{b.id};                           // a handle of another kind is refused
    CHECK_FALSE(gfx.alive(as_texture));
    gfx.release(a.id);                                  // releasing twice is refused, harmlessly
    CHECK(has(gfx.last_error(), "stale"));
    CHECK(gfx.alive(b));
}

TEST_CASE("gfx layer: descriptors are validated against WebGPU's default limits") {
    RecordingBackend gfx;
    CHECK_FALSE(gfx.create_buffer({0, kVertex}));
    CHECK_FALSE(gfx.create_buffer({6, kVertex}));                       // not a multiple of 4
    CHECK_FALSE(gfx.create_buffer({268435460ull, kVertex}));            // past max_buffer_size
    CHECK(has(gfx.last_error(), "268435456"));
    CHECK_FALSE(gfx.create_buffer({64, 0}));
    CHECK_FALSE(gfx.create_texture({8193, 4, Format::RGBA8Unorm, kSampled}));
    CHECK(has(gfx.last_error(), "8192"));
    CHECK_FALSE(gfx.create_texture({4, 4, Format::Depth32Float, kRenderTarget | kStorageTex}));
    CHECK_FALSE(gfx.create_texture({4, 4, Format::R8Unorm, kStorageTex}));
    CHECK(gfx.create_texture({8192, 1, Format::RGBA8Unorm, kSampled}));

    // Programs: a uniform block past the binding limit, a fifth dynamic offset, a buffer's
    // MSL index among the vertex buffers, a dynamic storage binding.
    BindingDesc big[] = {{1, 0, BindingType::Uniform, kStageVertex, false, 65540, Format::RGBA8Unorm, 0}};
    CHECK_FALSE(gfx.create_program(quad_program(big, 1)));
    CHECK(has(gfx.last_error(), "65536"));
    BindingDesc five[5];
    for (uint8_t i = 0; i < 5; ++i) five[i] = {1, i, BindingType::Uniform, kStageVertex, true, 16, Format::RGBA8Unorm, i};
    CHECK_FALSE(gfx.create_program(quad_program(five, 5)));
    BindingDesc msl[] = {{1, 0, BindingType::Uniform, kStageVertex, true, 16, Format::RGBA8Unorm, 16}};
    CHECK_FALSE(gfx.create_program(quad_program(msl, 1)));
    BindingDesc dyn_storage[] = {{3, 0, BindingType::Storage, kStageVertex, true, 16, Format::RGBA8Unorm, 3}};
    CHECK_FALSE(gfx.create_program(quad_program(dyn_storage, 1)));
}

TEST_CASE("gfx layer: a compatibility device refuses storage buffers in the vertex stage") {
    Caps compat = RecordingBackend::default_caps();
    compat.limits.max_storage_buffers_in_vertex_stage = 0;
    compat.bits &= ~uint64_t(kCapVertexStorage);
    RecordingBackend gfx(compat);
    BindingDesc pull[] = {{3, 0, BindingType::Storage, kStageVertex, false, 16, Format::RGBA8Unorm, 3}};
    CHECK_FALSE(gfx.create_program(quad_program(pull, 1)));
    CHECK(has(gfx.last_error(), "storage buffers in the vertex stage"));
    BindingDesc frag[] = {{3, 0, BindingType::Storage, kStageFragment, false, 16, Format::RGBA8Unorm, 3}};
    CHECK(gfx.create_program(quad_program(frag, 1)));
}

TEST_CASE("gfx layer: pipelines are validated and cached by descriptor") {
    RecordingBackend gfx;
    Program p = gfx.create_program(quad_program());
    REQUIRE(p);
    RenderPipelineDesc d = quad_pipeline(p);
    Pipeline a = gfx.create_render_pipeline(d);
    REQUIRE(a);
    Pipeline b = gfx.create_render_pipeline(d);
    CHECK(a == b);                                       // the cache
    CHECK(gfx.count(RecordingBackend::OpKind::Create, HandleKind::Pipeline) == 1);
    d.blend = Blend::Alpha;
    Pipeline c = gfx.create_render_pipeline(d);
    CHECK(c);
    CHECK(c != a);
    gfx.release(a);
    CHECK(gfx.alive(b));                                 // one create, one release
    gfx.release(b);
    CHECK_FALSE(gfx.alive(a));

    RenderPipelineDesc bad = quad_pipeline(p);
    bad.vertex_layouts[1].stride = 10;                   // not a multiple of 4
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    bad = quad_pipeline(p);
    bad.vertex_layouts[1].stride = 2052;                 // past max_vertex_stride
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    bad = quad_pipeline(p);
    bad.vertex_layouts[1].attributes[1].offset = 10;     // past its element of 12 bytes
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    bad = quad_pipeline(p);
    bad.vertex_layouts[1].attributes[1].location = 0;    // used twice
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    bad = quad_pipeline(p);
    bad.vertex_layout_count = 9;
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    bad = quad_pipeline(p);
    bad.color_format = Format::RG32Uint;
    bad.blend = Blend::Additive;                         // integer formats do not blend
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    bad = quad_pipeline(p);
    bad.vertex_entry = "vs_missing";
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    bad = quad_pipeline(p);
    bad.depth = false;                                   // a depth test without depth
    CHECK_FALSE(gfx.create_render_pipeline(bad));
    // A stride of 0 is a constant stream.
    RenderPipelineDesc constant = quad_pipeline(p);
    constant.vertex_layouts[1].stride = 0;
    CHECK(gfx.create_render_pipeline(constant));
}

TEST_CASE("gfx layer: bind groups refuse a binding that is too large, an unaligned offset and a wrong layout") {
    RecordingBackend gfx;
    BindingDesc fixed[] = {{1, 0, BindingType::Uniform, kStageVertex, false, 256, Format::RGBA8Unorm, 0}};
    Program p = gfx.create_program(quad_program(fixed, 1));
    REQUIRE(p);
    Buffer u = gfx.create_buffer({1 << 20, kUniform | kCopyDst});
    BindGroupDesc d;
    d.program = p;
    d.group = 1;
    d.entry_count = 1;
    d.entries[0].binding = 0;
    d.entries[0].buffer = u;
    d.entries[0].size = 65540;                           // past max_uniform_buffer_binding_size
    CHECK_FALSE(gfx.create_bind_group(d));
    CHECK(has(gfx.last_error(), "past the limit"));
    d.entries[0].size = 256;
    d.entries[0].offset = 128;                           // not a multiple of 256
    CHECK_FALSE(gfx.create_bind_group(d));
    CHECK(has(gfx.last_error(), "multiple of 256"));
    d.entries[0].offset = (1 << 20) - 128;               // past the end of the buffer
    CHECK_FALSE(gfx.create_bind_group(d));
    d.entries[0].offset = 0;
    d.entries[0].buffer = kArena;                        // the arena binds only dynamic uniforms
    CHECK_FALSE(gfx.create_bind_group(d));
    d.entries[0].buffer = u;
    d.group = 0;                                         // a group the program does not declare
    CHECK_FALSE(gfx.create_bind_group(d));
    d.group = 1;
    d.entries[0].binding = 5;                            // a binding it does not declare
    CHECK_FALSE(gfx.create_bind_group(d));
    d.entries[0].binding = 0;
    Buffer vb = gfx.create_buffer({256, kVertex});
    d.entries[0].buffer = vb;                            // a buffer without kUniform
    CHECK_FALSE(gfx.create_bind_group(d));
    d.entries[0].buffer = u;
    CHECK(gfx.create_bind_group(d));
}

TEST_CASE("gfx layer: the draw list's arena hands out 256-byte slices") {
    DrawList list;
    uint8_t block[kCmapBytes] = {1, 2, 3};
    CHECK(list.uniform(block, 192) == 0);
    CHECK(list.uniform(block, 192) == 256);
    CHECK(list.uniform(block, kCmapBytes) == 512);
    CHECK(list.uniform(block, 48) == 512 + kCmapSlice);  // a colormap block takes 4,352 bytes
    CHECK(list.arena().size == 512 + kCmapSlice + 256);
    CHECK(list.arena()[256] == 1);
    CHECK(list.arena()[258] == 3);
    CHECK(list.error().empty());
    CHECK(list.uniform(block, 0) == kNoOffset);
    CHECK_FALSE(list.error().empty());
    DrawList big;
    std::vector<uint8_t> huge(70000);
    CHECK(big.uniform(huge.data(), 70000) == kNoOffset);
    big.reset(RenderTarget{});
    CHECK(big.error().empty());
}

TEST_CASE("gfx layer: a frame is recorded as plain data and its clears are applied") {
    Scene s;
    REQUIRE(s.pipeline);
    REQUIRE(s.g0);
    REQUIRE(s.g1);
    DrawList list(s.target());
    DrawRecord r = s.draw(list, 4);
    list.draw(r);
    DrawRecord r2 = s.draw(list, 2);
    r2.first_instance = 2;
    list.draw(r2);
    const size_t arena_bytes = list.arena().size;
    Frame f;
    f.render(std::move(list));
    const Serial serial = s.gfx.submit(std::move(f));
    REQUIRE(serial == 1);
    CHECK(s.gfx.completed(serial));
    REQUIRE(s.gfx.submitted().size() == 1);
    const Frame& got = s.gfx.submitted()[0].frame;
    REQUIRE(got.passes().size == 1);
    const auto* pass = std::get_if<RenderPass>(&got.passes()[0]);
    REQUIRE(pass != nullptr);
    REQUIRE(pass->list.records().size == 2);
    CHECK(pass->list.records()[0].dynamic_offsets[0] == 0);
    CHECK(pass->list.records()[0].dynamic_offsets[1] == 256);
    CHECK(pass->list.records()[0].dynamic_offsets[2] == 512);
    CHECK(pass->list.records()[1].first_instance == 2);
    CHECK(pass->list.records()[1].pick_id == 7);
    CHECK(pass->list.arena().size == arena_bytes);
    CHECK(s.gfx.last_use(s.instances) == serial);
    CHECK(s.gfx.last_use(s.pipeline) == serial);
    CHECK(s.gfx.last_use(s.color) == serial);

    Stats st = s.gfx.take_stats();
    CHECK(st.submits == 1);
    CHECK(st.draws == 2);
    CHECK(st.gpu_ms < 0);                                // no timestamps on this backend

    // The clear is visible through an asynchronous readback, in the target's format.
    Readback rb = s.gfx.read_texture(s.color, 2, 3, 2, 1);
    REQUIRE(rb);
    std::vector<uint8_t> px;
    REQUIRE(s.gfx.take_readback(rb, &px));
    CHECK(px == std::vector<uint8_t>{0x30, 0x20, 0x10, 0xff, 0x30, 0x20, 0x10, 0xff});
    Readback rd = s.gfx.read_texture(s.depth, 0, 0, 1, 1);
    REQUIRE(s.gfx.take_readback(rd, &px));
    float d = 0;
    std::memcpy(&d, px.data(), 4);
    CHECK(d == 1.0f);
    Readback ri = s.gfx.read_texture(s.id, 15, 15, 1, 1);
    REQUIRE(s.gfx.take_readback(ri, &px));
    CHECK(px == std::vector<uint8_t>(8, 0));
}

TEST_CASE("gfx layer: a bad draw refuses the whole frame and leaves the target untouched") {
    Scene s;
    {   // A good frame first, so the target holds a known clear.
        DrawList list(s.target(0xff000000u));
        list.draw(s.draw(list, 4));
        Frame f;
        f.render(std::move(list));
        REQUIRE(s.gfx.submit(std::move(f)) == 1);
    }
    const std::vector<uint8_t> before = s.gfx.texture_bytes(s.color);
    const size_t recorded = s.gfx.submitted().size();
    auto refuse = [&](const char* why, auto change) {
        DrawList list(s.target(0xffffffffu));
        list.draw(s.draw(list, 4));                      // a good draw first
        DrawRecord r = s.draw(list, 4);
        change(r, list);
        list.draw(r);
        Frame f;
        f.render(std::move(list));
        CHECK_MESSAGE(s.gfx.submit(std::move(f)) == 0, why);
        CHECK_MESSAGE(has(s.gfx.last_error(), why), s.gfx.last_error());
        CHECK(s.gfx.submitted().size() == recorded);    // nothing reached the backend
        CHECK(s.gfx.texture_bytes(s.color) == before);   // the clear did not run
    };
    // A draw past a limit: five instances of 12 bytes from a buffer of four.
    refuse("reads past the end of the buffer in vertex slot 1", [](DrawRecord& r, DrawList&) { r.instance_count = 5; });
    refuse("reads past the end of the buffer in vertex slot 1", [](DrawRecord& r, DrawList&) { r.first_instance = 1; });
    refuse("reads past the end of the buffer in vertex slot 0", [](DrawRecord& r, DrawList&) { r.count = 7; });
    refuse("not a multiple of 256", [](DrawRecord& r, DrawList&) { r.dynamic_offsets[1] = 128; });
    refuse("past the end of the uniform arena", [](DrawRecord& r, DrawList& l) {
        r.dynamic_offsets[2] = uint32_t(l.arena().size);
    });
    refuse("offset 6 is not a multiple of 4", [](DrawRecord& r, DrawList&) { r.vertex[1].offset = 6; });
    refuse("bind group 1 is missing", [](DrawRecord& r, DrawList&) { r.groups[1] = BindGroup{}; });
    refuse("bind group 1 was made for another group layout", [&](DrawRecord& r, DrawList&) { r.groups[1] = s.g0; });
    refuse("index buffer", [&](DrawRecord& r, DrawList&) { r.index_buffer = s.corners; });   // lacks kIndex
    // A stale handle: the instance buffer released before the frame is submitted.
    Buffer extra = s.gfx.create_buffer({48, kVertex});
    s.gfx.release(extra);
    refuse("vertex buffer slot 1 is missing, released or unknown", [&](DrawRecord& r, DrawList&) { r.vertex[1] = {extra, 0}; });
    // A target the pipeline was not made for.
    {
        RenderTarget t = s.target();
        t.depth = Texture{};
        DrawList list(t);
        list.draw(s.draw(list, 1));
        Frame f;
        f.render(std::move(list));
        CHECK(s.gfx.submit(std::move(f)) == 0);
        CHECK(has(s.gfx.last_error(), "attachments do not match"));
    }
    // A uniform the list refused refuses the frame.
    {
        DrawList list(s.target());
        list.draw(s.draw(list, 1));
        list.uniform(nullptr, 16);
        Frame f;
        f.render(std::move(list));
        CHECK(s.gfx.submit(std::move(f)) == 0);
    }
    // The next good frame takes the next serial: refused frames use none.
    DrawList list(s.target());
    list.draw(s.draw(list, 4));
    Frame f;
    f.render(std::move(list));
    CHECK(s.gfx.submit(std::move(f)) == 2);
}

TEST_CASE("gfx layer: a draw that binds a target of its own pass as a texture refuses the frame") {
    // WebGPU refuses such a pass when it is encoded and Metal leaves it undefined, so the core
    // refuses the frame before either sees it. Bound in another pass, the same texture is fine.
    RecordingBackend gfx;
    Texture color = gfx.create_texture({16, 16, Format::RGBA8Unorm, kRenderTarget | kSampled | kCopySrcTex, "color"});
    Texture ids = gfx.create_texture({16, 16, Format::RG32Uint, kRenderTarget | kSampled, "id"});
    Texture other = gfx.create_texture({16, 16, Format::RGBA8Unorm, kRenderTarget | kSampled, "other"});
    Texture other_ids = gfx.create_texture({16, 16, Format::RG32Uint, kRenderTarget | kSampled, "other id"});
    static const BindingDesc kSampling[] = {
        {0, 0, BindingType::Uniform, kStageVertex | kStageFragment, true, 48, Format::RGBA8Unorm, 0},
        {2, 0, BindingType::Texture, kStageFragment, false, 0, Format::RGBA8Unorm, 0},
        {2, 1, BindingType::UintTexture, kStageFragment, false, 0, Format::RGBA8Unorm, 1},
    };
    Program program = gfx.create_program(quad_program(kSampling, 3));
    REQUIRE_MESSAGE(bool(program), gfx.last_error());
    RenderPipelineDesc pd = quad_pipeline(program);
    pd.depth = pd.depth_test = pd.depth_write = false;
    Pipeline pipe = gfx.create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(pipe), gfx.last_error());
    BindGroupDesc b0;
    b0.program = program;
    b0.group = 0;
    b0.entry_count = 1;
    b0.entries[0].binding = 0;
    b0.entries[0].buffer = kArena;
    const BindGroup g0 = gfx.create_bind_group(b0);
    auto images = [&](Texture image, Texture uint_image) {
        BindGroupDesc b2;
        b2.program = program;
        b2.group = 2;
        b2.entry_count = 2;
        b2.entries[0].binding = 0;
        b2.entries[0].texture = image;
        b2.entries[1].binding = 1;
        b2.entries[1].texture = uint_image;
        const BindGroup g = gfx.create_bind_group(b2);
        REQUIRE_MESSAGE(bool(g), gfx.last_error());
        return g;
    };
    const float c[12] = {0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
    Buffer corners = gfx.create_buffer({sizeof c, kVertex | kCopyDst, false, "corners"});
    gfx.write_buffer(corners, 0, c, sizeof c);
    Buffer instances = gfx.create_buffer({12, kVertex | kCopyDst, false, "instances"});
    auto pass = [&](Frame& f, Texture target, Texture target_ids, BindGroup g2) {
        RenderTarget t;
        t.color = target;
        t.id = target_ids;
        DrawList list(t);
        uint8_t block[48] = {};
        DrawRecord r;
        r.pipeline = pipe;
        r.groups[0] = g0;
        r.groups[2] = g2;
        r.dynamic_offsets[0] = list.uniform(block, sizeof block);
        r.vertex[0] = {corners, 0};
        r.vertex[1] = {instances, 0};
        r.count = 6;
        list.draw(r);
        f.render(std::move(list));
    };
    const BindGroup reads_other = images(other, other_ids);
    const BindGroup reads_color = images(color, ids);
    const BindGroup reads_ids = images(other, ids);
    {   // Sampling other textures, and the target of an earlier pass.
        Frame f;
        pass(f, color, ids, reads_other);
        pass(f, other, other_ids, reads_color);
        CHECK_MESSAGE(gfx.submit(std::move(f)) != 0, gfx.last_error());
    }
    const size_t recorded = gfx.submitted().size();
    for (const auto& [g2, why] : {std::pair<BindGroup, const char*>{reads_color, "binds the pass's own colour target"},
                                  std::pair<BindGroup, const char*>{reads_ids, "binds the pass's own ID target"}}) {
        Frame f;
        pass(f, color, ids, g2);
        CHECK(gfx.submit(std::move(f)) == 0);
        CHECK_MESSAGE(has(gfx.last_error(), why), gfx.last_error());
        CHECK(gfx.submitted().size() == recorded);   // nothing reached the backend
    }
}

TEST_CASE("gfx layer: indexed draws check their index range, not the vertex range") {
    Scene s;
    Buffer idx = s.gfx.create_buffer({6 * 4, kIndex | kCopyDst});
    const uint32_t indices[6] = {0, 1, 2, 3, 4, 5};
    REQUIRE(s.gfx.write_buffer(idx, 0, indices, sizeof indices));
    DrawList list(s.target());
    DrawRecord r = s.draw(list, 1);
    r.index_buffer = idx;
    r.count = 6;
    r.first_vertex = 100;                                // a base vertex is the producer's business
    list.draw(r);
    DrawRecord past = r;
    past.index_offset = 4;                               // six indices from the second: past the end
    Frame good;
    good.render(std::move(list));
    CHECK(s.gfx.submit(std::move(good)) == 1);
    DrawList bad_list(s.target());
    bad_list.uniform(indices, 16);
    past.dynamic_offsets[0] = past.dynamic_offsets[1] = past.dynamic_offsets[2] = 0;
    bad_list.draw(past);
    Frame bad;
    bad.render(std::move(bad_list));
    CHECK(s.gfx.submit(std::move(bad)) == 0);
}

TEST_CASE("gfx layer: release is deferred until the last submit that used it completed") {
    Scene s;
    s.gfx.set_auto_complete(false);
    DrawList list(s.target());
    list.draw(s.draw(list, 4));
    Frame f;
    f.render(std::move(list));
    const Serial serial = s.gfx.submit(std::move(f));
    REQUIRE(serial == 1);
    CHECK_FALSE(s.gfx.completed(serial));
    CHECK(s.gfx.last_use(s.instances) == serial);
    const size_t destroyed = s.gfx.count(RecordingBackend::OpKind::Destroy, HandleKind::Buffer);
    s.gfx.release(s.instances);
    CHECK_FALSE(s.gfx.alive(s.instances));               // dead for new uses at once
    CHECK(s.gfx.count(RecordingBackend::OpKind::Destroy, HandleKind::Buffer) == destroyed);
    CHECK(s.gfx.last_use(s.instances) == serial);        // still answers while pending
    s.gfx.finish(serial);
    CHECK(s.gfx.completed(serial));
    CHECK(s.gfx.count(RecordingBackend::OpKind::Destroy, HandleKind::Buffer) == destroyed + 1);
    // A buffer that no submit used is destroyed at once.
    Buffer idle = s.gfx.create_buffer({16, kVertex});
    s.gfx.release(idle);
    CHECK(s.gfx.count(RecordingBackend::OpKind::Destroy, HandleKind::Buffer) == destroyed + 2);
}

TEST_CASE("gfx layer: readback is asynchronous and taken once") {
    Scene s;
    s.gfx.set_auto_complete(false);
    Texture t = s.gfx.create_texture({4, 2, Format::RGBA8Unorm, kSampled | kCopyDstTex | kCopySrcTex});
    std::vector<uint8_t> pixels(4 * 2 * 4);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t(i * 3);
    REQUIRE(s.gfx.write_texture(t, pixels.data(), 16));
    Readback rb = s.gfx.read_texture(t, 1, 1, 2, 1);
    REQUIRE(rb);
    std::vector<uint8_t> out;
    CHECK_FALSE(s.gfx.take_readback(rb, &out));          // not ready
    CHECK_FALSE(s.gfx.wait(99));                         // never submitted
    CHECK(s.gfx.wait(s.gfx.last_use(t)));                // a blocking wait completes it here
    REQUIRE(s.gfx.take_readback(rb, &out));
    CHECK(out == std::vector<uint8_t>(pixels.begin() + 20, pixels.begin() + 28));
    CHECK_FALSE(s.gfx.take_readback(rb, &out));          // taken once
    CHECK_FALSE(s.gfx.read_texture(t, 3, 0, 2, 1));      // outside the texture
    Texture no_src = s.gfx.create_texture({4, 4, Format::RGBA8Unorm, kSampled});
    CHECK_FALSE(s.gfx.read_texture(no_src, 0, 0, 1, 1));
}

TEST_CASE("gfx layer: import reports its route and the bytes it copied") {
    RecordingBackend gfx;
    const size_t page = RecordingBackend::kHostPage;
    std::vector<uint8_t> storage(2 * page);              // page-aligned memory, portably
    void* aligned = storage.data() + (page - reinterpret_cast<uintptr_t>(storage.data()) % page) % page;
    std::memset(aligned, 0x5a, page);
    ExternalBuffer ext;
    ext.kind = ExternalKind::HostMemory;
    ext.handle = aligned;
    ext.size = page;
    ext.usage = kVertex;
    ImportResult in_place = gfx.import_buffer(ext);
    REQUIRE(in_place.buffer);
    CHECK(in_place.route == Route::InPlace);
    CHECK(in_place.bytes_copied == 0);
    CHECK(std::string(in_place.path) == "in-place");
    static_cast<uint8_t*>(aligned)[7] = 0x11;            // in place: the layer sees the producer's bytes
    CHECK(gfx.buffer_bytes(in_place.buffer)[7] == 0x11);

    std::vector<uint8_t> loose(100 * 4, 0x22);
    ext.handle = loose.data() + 4;                       // not page-aligned
    ext.size = 64;
    ImportResult copied = gfx.import_buffer(ext);
    REQUIRE(copied.buffer);
    CHECK(copied.route == Route::CopiedFromHost);
    CHECK(copied.bytes_copied == 64);
    CHECK(std::string(copied.path) == "copied-from-host");
    CHECK_FALSE(copied.reason.empty());

    ext.allow_copy = false;
    ImportResult refused = gfx.import_buffer(ext);
    CHECK_FALSE(refused.buffer);
    CHECK(refused.route == Route::Refused);
    CHECK(std::string(refused.path) == "refused");
    CHECK_FALSE(refused.reason.empty());

    ExternalBuffer metal;
    metal.kind = ExternalKind::MetalBuffer;
    metal.handle = aligned;
    metal.size = page;
    metal.usage = kVertex;
    ImportResult no_metal = gfx.import_buffer(metal);
    CHECK(no_metal.route == Route::Refused);
    CHECK(has(no_metal.reason, "cannot import an MTLBuffer"));
    metal.kind = ExternalKind::HostMemory;
    metal.size = 268435456ull + 4;                       // above WebGPU's buffer limit
    CHECK(has(gfx.import_buffer(metal).reason, "above the backend's limits"));

    Stats st = gfx.take_stats();
    CHECK(st.imports_in_place == 1);
    CHECK(st.imports_copied == 1);
    CHECK(st.bytes_copied_from_host == 64);
    gfx.release(in_place.buffer);
}

TEST_CASE("gfx layer: a transient host import is always copied, and an import's memory is mapped for reading") {
    RecordingBackend gfx;
    const size_t page = RecordingBackend::kHostPage;
    std::vector<uint8_t> storage(2 * page);
    uint8_t* aligned = storage.data() + (page - reinterpret_cast<uintptr_t>(storage.data()) % page) % page;
    std::memset(aligned, 0x5a, page);
    ExternalBuffer ext;
    ext.kind = ExternalKind::HostMemory;
    ext.handle = aligned;
    ext.size = page;
    ext.usage = kVertex;
    ext.transient = true;                                // page-aligned, but valid only during the call
    ImportResult copied = gfx.import_buffer(ext);
    REQUIRE(copied.buffer);
    CHECK(copied.route == Route::CopiedFromHost);
    CHECK(copied.bytes_copied == page);
    CHECK(has(copied.reason, "only during the call"));
    aligned[7] = 0x11;                                   // the producer's bytes change; the copy does not
    CHECK(gfx.buffer_bytes(copied.buffer)[7] == 0x5a);
    const uint8_t* m = static_cast<const uint8_t*>(gfx.mapped(copied.buffer));
    REQUIRE(m != nullptr);                               // an import is mapped for reading
    CHECK(m[7] == 0x5a);
    ext.transient = false;
    ImportResult wrapped = gfx.import_buffer(ext);
    REQUIRE(wrapped.buffer);
    CHECK(wrapped.route == Route::InPlace);
    CHECK(static_cast<const uint8_t*>(gfx.mapped(wrapped.buffer))[7] == 0x11);
    BufferDesc bd;
    bd.size = 256;
    bd.usage = kVertex | kCopyDst;
    Buffer own = gfx.create_buffer(bd);
    CHECK(gfx.mapped(own) == nullptr);                   // a buffer made without host_visible is not
    gfx.release(copied.buffer);
    gfx.release(wrapped.buffer);
    gfx.release(own);
}

TEST_CASE("gfx layer: a read-only import binds for reading and is refused in a read_write binding") {
    RecordingBackend gfx;
    const size_t page = RecordingBackend::kHostPage;
    std::vector<uint8_t> storage(2 * page);
    uint8_t* aligned = storage.data() + (page - reinterpret_cast<uintptr_t>(storage.data()) % page) % page;
    ExternalBuffer ext;
    ext.kind = ExternalKind::HostMemory;
    ext.handle = aligned;
    ext.size = page;
    ext.usage = kStorage;
    ext.allow_copy = false;
    ext.read_only = true;                                // a training tensor of the producer runtime
    ImportResult ro = gfx.import_buffer(ext);
    REQUIRE(ro.buffer);
    CHECK(ro.route == Route::InPlace);
    ext.read_only = false;
    ImportResult rw = gfx.import_buffer(ext);
    REQUIRE(rw.buffer);

    const EntryDesc cs[] = {{"cs_main", kStageCompute, {64, 1, 1}}};
    BindingDesc bindings[] = {{0, 0, BindingType::Storage, kStageCompute, false, 4, Format::RGBA8Unorm, 0},
                              {0, 1, BindingType::StorageReadWrite, kStageCompute, false, 4, Format::RGBA8Unorm, 1}};
    ProgramDesc pd;
    pd.label = "read-only rule";
    pd.entries = cs;
    pd.entry_count = 1;
    pd.bindings = bindings;
    pd.binding_count = 2;
    Program p = gfx.create_program(pd);
    REQUIRE(p);
    Buffer out = gfx.create_buffer({256, kStorage});
    BindGroupDesc bg;
    bg.program = p;
    bg.entry_count = 2;
    bg.entries[0].binding = 0;
    bg.entries[0].buffer = ro.buffer;                    // var<storage, read>: allowed
    bg.entries[1].binding = 1;
    bg.entries[1].buffer = out;
    CHECK(gfx.create_bind_group(bg));
    bg.entries[1].buffer = ro.buffer;                    // var<storage, read_write>: refused
    CHECK_FALSE(gfx.create_bind_group(bg));
    CHECK(has(gfx.last_error(), "read-only import"));
    bg.entries[1].buffer = rw.buffer;                    // an import without the flag keeps the old rule
    CHECK(gfx.create_bind_group(bg));
    gfx.release(ro.buffer);
    gfx.release(rw.buffer);
    gfx.release(out);
}

TEST_CASE("gfx layer: copies and dispatches are validated like draws") {
    Scene s;
    Texture t = s.gfx.create_texture({4, 2, Format::RGBA8Unorm, kSampled | kCopyDstTex | kCopySrcTex | kStorageTex});
    Buffer src = s.gfx.create_buffer({512, kCopySrc | kCopyDst});
    std::vector<uint8_t> rows(512);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = uint8_t(i);
    REQUIRE(s.gfx.write_buffer(src, 0, rows.data(), rows.size()));
    {
        Frame f;
        f.copy_buffer_to_texture(src, 0, 128, t);        // the row pitch must be a multiple of 256
        CHECK(s.gfx.submit(std::move(f)) == 0);
        CHECK(has(s.gfx.last_error(), "multiple of 256"));
    }
    UpdateResult fill = s.gfx.fill_texture_from_buffer(t, src, 0, 256);
    REQUIRE(fill.ok);
    CHECK(fill.route == Route::CopiedOnGpu);
    const std::vector<uint8_t> got = s.gfx.texture_bytes(t);
    CHECK(std::equal(got.begin(), got.begin() + 16, rows.begin()));
    CHECK(std::equal(got.begin() + 16, got.end(), rows.begin() + 256));
    CHECK_FALSE(s.gfx.fill_texture_from_buffer(t, src, 0, 16).ok);   // the fill program reads storage
    CHECK(has(s.gfx.last_error(), "kStorage"));
    CHECK_FALSE(s.gfx.update_texture_from_tensor(t, TensorSource{}, TextureMapping{}).ok);

    ComputePipelineDesc cd;
    cd.program = s.program;
    cd.entry = "cs_fill";
    Pipeline cs = s.gfx.create_compute_pipeline(cd);
    REQUIRE(cs);
    cd.entry = "vs_quad";                                // not a compute entry
    CHECK_FALSE(s.gfx.create_compute_pipeline(cd));
    Frame f;
    uint8_t block[192] = {};
    const uint32_t offsets[3] = {f.uniform(block, 48), f.uniform(block, 192), f.uniform(block, 192)};
    const BindGroup groups[2] = {s.g0, s.g1};
    Texture writes[1] = {t};
    f.dispatch(cs, groups, offsets, 1, 1, 1, writes);
    // The colormap block reads 4,112 bytes from the frame arena's last slice of 256: refused.
    CHECK(s.gfx.submit(std::move(f)) == 0);
    CHECK(has(s.gfx.last_error(), "uniform arena"));
    Frame ok;
    uint8_t cmap[kCmapBytes] = {};
    const uint32_t ok_offsets[3] = {ok.uniform(block, 48), ok.uniform(block, 192), ok.uniform(cmap, kCmapBytes)};
    ok.dispatch(cs, groups, ok_offsets, 1, 1, 1, writes);
    CHECK(s.gfx.submit(std::move(ok)) != 0);
    Frame zero;
    zero.dispatch(cs, groups, ok_offsets, 0, 1, 1);
    CHECK(s.gfx.submit(std::move(zero)) == 0);
}

TEST_CASE("gfx layer: a bind group serves every program with the same group layout") {
    Scene s;
    // Another program with the same group 0 and group 1 but its own entries.
    const EntryDesc entries[] = {{"vs_other", kStageVertex, {1, 1, 1}}, {"fs_other", kStageFragment, {1, 1, 1}}};
    ProgramDesc other = quad_program();
    other.label = "other";
    other.entries = entries;
    other.entry_count = 2;
    Program p2 = s.gfx.create_program(other);
    REQUIRE(p2);
    RenderPipelineDesc d = quad_pipeline(p2);
    d.vertex_entry = "vs_other";
    d.fragment_entry = "fs_other";
    Pipeline pipe2 = s.gfx.create_render_pipeline(d);
    REQUIRE(pipe2);
    DrawList list(s.target());
    DrawRecord r = s.draw(list, 2);
    r.pipeline = pipe2;                                  // the first program's bind groups
    list.draw(r);
    Frame f;
    f.render(std::move(list));
    CHECK(s.gfx.submit(std::move(f)) != 0);

    // A program whose group 1 differs (a 256-byte draw block) refuses them.
    BindingDesc changed[3] = {kBindings[0], kBindings[1], kBindings[2]};
    changed[1].size = 256;
    ProgramDesc third = quad_program(changed, 3);
    Program p3 = s.gfx.create_program(third);
    Pipeline pipe3 = s.gfx.create_render_pipeline(quad_pipeline(p3));
    REQUIRE(pipe3);
    DrawList list3(s.target());
    DrawRecord r3 = s.draw(list3, 2);
    r3.pipeline = pipe3;
    list3.draw(r3);
    Frame f3;
    f3.render(std::move(list3));
    CHECK(s.gfx.submit(std::move(f3)) == 0);
    CHECK(has(s.gfx.last_error(), "another group layout"));
}

TEST_CASE("gfx layer: waits on producer events need the capability") {
    RecordingBackend gfx;
    Frame f;
    int dummy = 0;
    f.wait(ExternalSync{ExternalSync::MetalEvent, &dummy, 1});
    CHECK(gfx.submit(std::move(f)) == 0);
    CHECK(has(gfx.last_error(), "Metal event"));
    Frame drained;
    drained.wait(ExternalSync{});
    CHECK(gfx.submit(std::move(drained)) == 1);
}

TEST_CASE("gfx layer: storage bindings, the sizes index and override constants follow the generated programs") {
    RecordingBackend gfx;
    // A read-write storage buffer is not visible to the vertex stage (WebGPU).
    BindingDesc rw_vertex[] = {{3, 0, BindingType::StorageReadWrite, kStageVertex, false, 4, Format::RGBA8Unorm, 3}};
    CHECK_FALSE(gfx.create_program(quad_program(rw_vertex, 1)));
    CHECK(has(gfx.last_error(), "read-write storage"));
    // Metal buffer 15 is Naga's sizes buffer.
    BindingDesc at_sizes[] = {{1, 0, BindingType::Uniform, kStageVertex, true, 16, Format::RGBA8Unorm, 15}};
    CHECK_FALSE(gfx.create_program(quad_program(at_sizes, 1)));
    CHECK(has(gfx.last_error(), "sizes buffer"));
    // Override constants are refused.
    Program p = gfx.create_program(quad_program());
    RenderPipelineDesc d = quad_pipeline(p);
    d.constant_count = 1;
    d.constants[0] = {0, 1.0f};
    CHECK_FALSE(gfx.create_render_pipeline(d));
    CHECK(has(gfx.last_error(), "override constants"));

    // A storage binding of size 0 takes the rest of its buffer; a declared block that does
    // not fit there is refused.
    const EntryDesc cs[] = {{"cs_main", kStageCompute, {64, 1, 1}}};
    BindingDesc storage[] = {{0, 0, BindingType::Storage, kStageCompute, false, 4, Format::RGBA8Unorm, 0},
                             {0, 1, BindingType::StorageReadWrite, kStageCompute, false, 512, Format::RGBA8Unorm, 1}};
    ProgramDesc pd;
    pd.label = "storage";
    pd.entries = cs;
    pd.entry_count = 1;
    pd.bindings = storage;
    pd.binding_count = 2;
    Program sp = gfx.create_program(pd);
    REQUIRE(sp);
    Buffer b = gfx.create_buffer({1024, kStorage});
    BindGroupDesc bg;
    bg.program = sp;
    bg.entry_count = 2;
    bg.entries[0].binding = 0;
    bg.entries[0].buffer = b;
    bg.entries[0].offset = 256;
    bg.entries[1].binding = 1;
    bg.entries[1].buffer = b;
    bg.entries[1].offset = 512;                          // 512 bytes remain: the block fits
    CHECK(gfx.create_bind_group(bg));
    bg.entries[1].offset = 768;                          // 256 remain, the block needs 512
    CHECK_FALSE(gfx.create_bind_group(bg));
    bg.entries[1].offset = 300;                          // not a multiple of 256
    CHECK_FALSE(gfx.create_bind_group(bg));
    Buffer no_storage = gfx.create_buffer({1024, kVertex});
    bg.entries[1].offset = 0;
    bg.entries[1].buffer = no_storage;
    CHECK_FALSE(gfx.create_bind_group(bg));
}

TEST_CASE("gfx layer: every generated program, pipeline pair and compute entry passes the layer's checks") {
    using namespace caliper_gfx::shaders;
    Caps compat = RecordingBackend::default_caps();     // and on a compatibility device
    compat.limits.max_storage_buffers_in_vertex_stage = 0;
    compat.limits.max_texture_dimension_2d = 4096;
    compat.limits.max_color_attachments = 4;
    for (const Caps& caps : {RecordingBackend::default_caps(), compat}) {
        RecordingBackend gfx(caps);
        for (uint32_t i = 0; i < kProgramCount; ++i) {
            const ProgramDescription& desc = kProgramDescriptions[i];
            Program p = gfx.create_program(kMetalPrograms[i]);
            REQUIRE_MESSAGE(bool(p), (std::string(desc.name) + ": " + gfx.last_error()));
            for (uint32_t k = 0; k < desc.pair_count; ++k) {
                Pipeline pipe = gfx.create_render_pipeline(caliper_test::generated_pipeline(desc, p, desc.pairs[k]));
                CHECK_MESSAGE(bool(pipe), (std::string(desc.name) + " " + desc.pairs[k].vertex + "+" +
                                      desc.pairs[k].fragment + ": " + gfx.last_error()));
            }
            for (uint32_t k = 0; k < desc.entry_count; ++k) {
                if (desc.entries[k].stage != kStageCompute) continue;
                ComputePipelineDesc cd;
                cd.program = p;
                cd.entry = desc.entries[k].name;
                CHECK_MESSAGE(bool(gfx.create_compute_pipeline(cd)), (std::string(desc.name) + ": " + gfx.last_error()));
            }
        }
    }
}

// The texture path's operation (ops.cpp) on the recording backend: what update_texture_from_tensor
// records for each source, the parameters and key block it puts in the frame's arena, the bind
// groups it caches per buffer and texture and drops once either is gone, and its refusals.
TEST_CASE("gfx layer: update_texture_from_tensor records the colormap and the fill for each source") {
    namespace sh = caliper_gfx::shaders;
    RecordingBackend gfx;
    const uint32_t W = 5, H = 3;
    Texture t = gfx.create_texture({W, H, Format::RGBA8Unorm, kSampled | kStorageTex | kCopyDstTex | kCopySrcTex});
    REQUIRE(t);
    sh::Cmap block{};
    block.vmin = 0.0f;
    block.vmax = 1.0f;
    block.levels[7][2] = 0xff00ff00u;
    const TextureMapping m{&block, uint32_t(sizeof block)};
    std::vector<float> values(size_t(W) * H + 8, 0.25f);
    Buffer values_buf = gfx.create_buffer({values.size() * 4, kStorage | kCopyDst});
    REQUIRE(gfx.write_buffer(values_buf, 0, values.data(), values.size() * 4));

    // From a buffer at byte 8, rows of 6 values: one dispatch of colormap_texture over 1 x 1 groups.
    TensorSource src;
    src.buffer = values_buf;
    src.offset = 8;
    src.w = W;
    src.h = H;
    src.stride_y = 6;
    gfx.clear_log();
    UpdateResult r = gfx.update_texture_from_tensor(t, src, m);
    REQUIRE_MESSAGE(r.ok, gfx.last_error());
    CHECK(std::string(r.path) == "compute-imported");
    CHECK(r.route == Route::InPlace);
    CHECK(r.bytes_copied == 0);
    CHECK(r.serial != 0);
    REQUIRE(gfx.submitted().size() == 1);
    {
        const Frame& f = gfx.submitted()[0].frame;
        REQUIRE(f.passes().size == 1);
        const auto* d = std::get_if<DispatchPass>(&f.passes()[0]);
        REQUIRE(d != nullptr);
        CHECK(d->x == 1);
        CHECK(d->y == 1);
        sh::TexParams tp;
        std::memcpy(&tp, f.arena().data + d->dynamic_offsets[0], sizeof tp);
        CHECK(tp.width == W);
        CHECK(tp.height == H);
        CHECK(tp.base == 2);
        CHECK(tp.row == 6);
        CHECK(std::memcmp(f.arena().data + d->dynamic_offsets[1], &block, sizeof block) == 0);
        REQUIRE(d->writes.size() == 1);
        CHECK(d->writes[0] == t);
    }
    // Program, pipeline, its group 1 and the group of (buffer, texture) were made once.
    const size_t groups = gfx.count(RecordingBackend::OpKind::Create, HandleKind::BindGroup);
    CHECK(groups == 2);
    REQUIRE(gfx.update_texture_from_tensor(t, src, m).ok);
    CHECK(gfx.count(RecordingBackend::OpKind::Create, HandleKind::BindGroup) == groups);

    // From host bytes: the values go into a staging buffer the texture keeps, then the colormap.
    CaliperTensor host{};
    host.data = values.data();
    host.device = CALIPER_DEV_CPU;
    TensorSource hs;
    hs.tensor = &host;
    hs.w = W;
    hs.h = H;
    gfx.clear_log();
    r = gfx.update_texture_from_tensor(t, hs, m);
    REQUIRE_MESSAGE(r.ok, gfx.last_error());
    CHECK(std::string(r.path) == "cpu-staged");
    CHECK(r.route == Route::CopiedFromHost);
    CHECK(r.bytes_copied == W * H * 4);
    CHECK(gfx.count(RecordingBackend::OpKind::Create, HandleKind::Buffer) == 1);
    CHECK(gfx.count(RecordingBackend::OpKind::WriteBuffer) == 1);
    CHECK(gfx.submitted().size() == 1);

    // RGBA8 texels from host bytes: written into the texture, with no pass.
    std::vector<uint32_t> texels(size_t(W) * H);
    for (size_t i = 0; i < texels.size(); ++i) texels[i] = uint32_t(i * 0x01020304u);
    CaliperTensor img{};
    img.data = texels.data();
    img.device = CALIPER_DEV_CPU;
    TensorSource is;
    is.tensor = &img;
    is.w = W;
    is.h = H;
    is.dtype = CALIPER_DT_U8;
    is.channels = 4;
    is.stride_x = 4;
    gfx.clear_log();
    r = gfx.update_texture_from_tensor(t, is, TextureMapping{});
    REQUIRE_MESSAGE(r.ok, gfx.last_error());
    CHECK(std::string(r.path) == "cpu-staged");
    CHECK(r.serial == 0);
    CHECK(gfx.submitted().empty());
    const std::vector<uint8_t> got = gfx.texture_bytes(t);
    CHECK(std::memcmp(got.data(), texels.data(), got.size()) == 0);

    // RGBA8 texels from a buffer at byte 4: the fill program, whatever the row pitch.
    Buffer texel_buf = gfx.create_buffer({texels.size() * 4 + 4, kStorage | kCopyDst});
    TensorSource bs = is;
    bs.tensor = nullptr;
    bs.buffer = texel_buf;
    bs.offset = 4;
    gfx.clear_log();
    r = gfx.update_texture_from_tensor(t, bs, TextureMapping{});
    REQUIRE_MESSAGE(r.ok, gfx.last_error());
    CHECK(std::string(r.path) == "blit-imported");
    REQUIRE(gfx.submitted().size() == 1);
    {
        const auto* d = std::get_if<DispatchPass>(&gfx.submitted()[0].frame.passes()[0]);
        REQUIRE(d != nullptr);
        sh::FillParams fp;
        std::memcpy(&fp, gfx.submitted()[0].frame.arena().data + d->dynamic_offsets[0], sizeof fp);
        CHECK(fp.base == 1);
        CHECK(fp.row == W);
    }

    // A released texture's cached groups (over the value buffer, its staging buffer and the texel
    // buffer) and its staging buffer go at the next update.
    gfx.release(t);
    Texture t2 = gfx.create_texture({W, H, Format::RGBA8Unorm, kStorageTex | kCopyDstTex});
    gfx.clear_log();
    REQUIRE(gfx.update_texture_from_tensor(t2, src, m).ok);
    CHECK(gfx.count(RecordingBackend::OpKind::Destroy, HandleKind::BindGroup) == 3);
    CHECK(gfx.count(RecordingBackend::OpKind::Destroy, HandleKind::Buffer) == 1);

    // Refusals.
    CHECK_FALSE(gfx.update_texture_from_tensor(t2, src, TextureMapping{}).ok);
    CHECK(has(gfx.last_error(), "key block"));
    TensorSource past = src;
    past.offset = 28;
    CHECK_FALSE(gfx.update_texture_from_tensor(t2, past, m).ok);
    CHECK(has(gfx.last_error(), "past the end"));
    CaliperTensor metal{};
    metal.data = values.data();
    metal.device = CALIPER_DEV_METAL;
    TensorSource ms = hs;
    ms.tensor = &metal;
    CHECK_FALSE(gfx.update_texture_from_tensor(t2, ms, m).ok);
    CHECK(has(gfx.last_error(), "cannot read a tensor on this device in place"));
    Texture sampled = gfx.create_texture({W, H, Format::RGBA8Unorm, kSampled});
    CHECK_FALSE(gfx.update_texture_from_tensor(sampled, src, m).ok);
    CHECK(has(gfx.last_error(), "kStorageTex"));
}

// ---- Colormaps (src/host/gfx/colormap_tables.h) -----------------------------------------------
namespace {
struct CmapSample { const char* name; int32_t id; uint32_t at[5]; };
// Entries 0, 64, 128, 191 and 255, printed by colormap_tables_gen.py --samples from matplotlib
// 3.11.2 (ids 4 to 9) and cmcrameri 1.8 (ids 10 to 15).
const CmapSample kCmapSamples[] = {
    {"cividis", 4, {0xff4e2200u, 0xff6c4e43u, 0xff787c7du, 0xff6dadbbu, 0xff38e8feu}},
    {"inferno", 5, {0xff040000u, 0xff6e1057u, 0xff5437bcu, 0xff0a8cf9u, 0xffa4fffcu}},
    {"plasma", 6, {0xff87080du, 0xffa8037eu, 0xff7847ccu, 0xff4194f8u, 0xff21f9f0u}},
    {"turbo", 7, {0xff3b1230u, 0xffebbc28u, 0xff3cfca4u, 0xff2281fbu, 0xff03047au}},
    {"twilight", 8, {0xffe2d9e2u, 0xffba7562u, 0xff371430u, 0xff5256b2u, 0xffe2d9e2u}},
    {"coolwarm", 9, {0xffc04c3bu, 0xfffeb08du, 0xffdcdcddu, 0xff7b9af4u, 0xff2604b4u}},
    {"batlow", 10, {0xff591901u, 0xff616022u, 0xff318282u, 0xff6b9df1u, 0xfffaccfau}},
    {"vik", 11, {0xff611200u, 0xffa67d30u, 0xffe0e5ecu, 0xff4372c3u, 0xff080059u}},
    {"roma", 12, {0xff00177eu, 0xff328cb6u, 0xffc3eac0u, 0xffc79d39u, 0xff983103u}},
    {"oslo", 13, {0xff010101u, 0xff5b3915u, 0xffbc7b50u, 0xffcab0a2u, 0xffffffffu}},
    {"lajolla", 14, {0xff001919u, 0xff2a3467u, 0xff4e60d9u, 0xff54acecu, 0xffcbfeffu}},
    {"berlin", 15, {0xffffb09eu, 0xff866828u, 0xff090c19u, 0xff1c327bu, 0xffadadffu}},
};
}  // namespace
TEST_CASE("colormaps: each added colormap has the exact packed RGBA at five entries") {
    constexpr int kAt[5] = {0, 64, 128, 191, 255};
    for (const CmapSample& s : kCmapSamples) {
        CAPTURE(s.name);
        const uint32_t* lut = cmap::builtin_lut(s.id);
        REQUIRE(lut != nullptr);
        for (int k = 0; k < 5; ++k) {
            CAPTURE(kAt[k]);
            CHECK(lut[kAt[k]] == s.at[k]);
        }
        for (int k = 0; k < 256; ++k) CHECK((lut[k] >> 24) == 0xffu);
    }
}

TEST_CASE("colormaps: ids outside 0 to 15 have no table, and ids 0 to 3 are the host's and the grey ramp") {
    CHECK(cmap::kBuiltinColormaps == 16);
    CHECK(cmap::builtin_lut(16) == nullptr);
    CHECK(cmap::builtin_lut(-1) == nullptr);
    CHECK(cmap::builtin_lut(0) == caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS));
    CHECK(cmap::builtin_lut(1) == caliper_host::colormap_lut(CALIPER_CMAP_MAGMA));
    CHECK(cmap::builtin_lut(2) == caliper_host::colormap_lut(CALIPER_CMAP_RDBU));
    const uint32_t* greys = cmap::builtin_lut(3);
    REQUIRE(greys != nullptr);
    for (uint32_t i = 0; i < 256; ++i) CHECK(greys[i] == (0xff000000u | (i << 16) | (i << 8) | i));
}

TEST_CASE("colormaps: derive_block fills level k with the table's entry k for all sixteen ids") {
    for (int32_t id = 0; id < cmap::kBuiltinColormaps; ++id) {
        CAPTURE(id);
        shaders::Cmap block{};
        REQUIRE(cmap::derive_block(id, 0.0f, 1.0f, &block));
        const uint32_t* lut = cmap::builtin_lut(id);
        REQUIRE(lut != nullptr);
        for (uint32_t k = 0; k < 256; ++k) CHECK(block.levels[k][2] == lut[k]);
    }
    shaders::Cmap unknown{};
    CHECK_FALSE(cmap::derive_block(16, 0.0f, 1.0f, &unknown));
}
