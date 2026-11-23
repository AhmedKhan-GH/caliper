// Tests of the draw half of a program in a viewer (draw/draw_runtime.h,
// design/programmable-graphics.md, Render Graph, Draw State and Safety of
// Draws) on the graphics layer's recording backend, with no GPU: the two
// worked examples install and record their passes in order, everything is
// checked before anything is made, the reset rules zero the draw buffers,
// and a program over its time or memory, or with a refused frame, stops with
// its pass named while the panel's other programs go on.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "draw/draw_host.h"
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"
#include "gfx/recording_backend.h"

#include <caliper_builder/programs/sandbox.h>

#include "../../../../tests/gfx/example_draw.h"

#include <array>
#include <cstring>

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace ex = caliper_test_pieces;
namespace gfx = caliper_gfx;

namespace {

cd::DrawChecks checks() { return ex::draw_checks(); }

// The modes to test: the WGSL as it is (the browser's path), and the translation to MSL with
// caliper-wgsl where the build has it.
std::vector<bool> modes() {
    std::vector<bool> m{true};
    if (cp::compiler_available()) m.push_back(false);
    return m;
}

// The recording backend's capabilities with GPU timestamps, as Metal has them.
gfx::Caps timed_caps() {
    gfx::Caps c = gfx::RecordingBackend::default_caps();
    c.bits |= gfx::kCapTimestamps;
    return c;
}

struct Host {
    gfx::RecordingBackend gfx;
    cd::HostTarget target;
    explicit Host(std::uint32_t w = 64, std::uint32_t h = 48, const gfx::Caps& caps = gfx::RecordingBackend::default_caps())
        : gfx(caps) {
        gfx::TextureDesc c;
        c.width = w;
        c.height = h;
        c.format = gfx::Format::RGBA8Unorm;
        c.usage = gfx::kRenderTarget | gfx::kSampled;
        target.color = gfx.create_texture(c);
        c.format = gfx::Format::RG32Uint;
        c.usage = gfx::kRenderTarget;
        target.id = gfx.create_texture(c);
        c.format = gfx::Format::Depth32Float;
        target.depth = gfx.create_texture(c);
        target.width = w;
        target.height = h;
    }
};

template <typename T>
cd::ResourceVersion version(const char* id, const std::vector<T>& v, std::int64_t step) {
    cd::ResourceVersion r;
    r.resource = id;
    r.step = step;
    r.count = std::uint32_t(v.size());
    r.bytes = reinterpret_cast<const std::uint8_t*>(v.data());
    r.size = v.size() * sizeof(T);
    return r;
}

void give_arcs(cd::DrawRuntime& rt, const ex::ArcResources& res, std::int64_t step) {
    REQUIRE(rt.set_resource(version("heads", res.heads, step)).ok());
    REQUIRE(rt.set_resource(version("arcs", res.arcs, step)).ok());
    REQUIRE(rt.set_resource(version("labels", res.labels, step)).ok());
}

// Records and submits one frame of a runtime; the frames encode() appended.
std::vector<gfx::Frame> frame(Host& h, cd::DrawRuntime& rt, const cd::FrameInputs& in = {}) {
    std::vector<gfx::Frame> frames;
    if (!rt.encode(&frames, in, h.target)) return frames;
    std::vector<gfx::Frame> copies = frames;
    for (gfx::Frame& f : copies) {
        const gfx::Serial s = h.gfx.submit(std::move(f));
        CHECK_MESSAGE(s != 0, h.gfx.last_error());
    }
    return frames;
}

const gfx::DrawList& list_of(const gfx::Frame& f, std::size_t i) {
    return std::get<gfx::RenderPass>(f.passes()[i]).list;
}

}  // namespace

TEST_CASE("the field installs, waits for its resource, then draws one full-screen triangle") {
    for (bool webgpu : modes()) {
        CAPTURE(webgpu);
        Host h;
        cd::DrawRuntime rt(h.gfx, webgpu, checks());
        CHECK(rt.status().state == cd::DrawState::waiting);
        cd::Instance in;
        in.program = "head_pattern";
        in.panel = "pattern";
        const cp::Verdict v = rt.install(ex::attention_field(), in, 7);
        REQUIRE_MESSAGE(v.ok(), v.message);
        CHECK(rt.status().state == cd::DrawState::compiled);
        CHECK(rt.camera() == cs::CameraMode::none);
        std::vector<gfx::Frame> none;
        CHECK(!rt.encode(&none, {}, h.target));   // no version of `pattern` yet

        const std::vector<float> pattern = ex::field_pattern();
        REQUIRE(rt.set_resource(version("pattern", pattern, 40)).ok());
        const std::vector<gfx::Frame> frames = frame(h, rt);
        REQUIRE(frames.size() == 1);
        REQUIRE(frames[0].passes().size == 1);
        const gfx::DrawList& list = list_of(frames[0], 0);
        CHECK(list.target().color == h.target.color);
        CHECK(!list.target().id);
        CHECK(!list.target().clear);
        REQUIRE(list.records().size == 1);
        CHECK(list.records()[0].count == 3);
        CHECK(list.records()[0].instance_count == 1);
        CHECK(rt.status().state == cd::DrawState::running);
        CHECK(rt.status().resource_step == 40);
    }
}

TEST_CASE("the arcs record the instanced pass with its ID, the glow target and the bloom, in order") {
    for (bool webgpu : modes()) {
        CAPTURE(webgpu);
        Host h;
        cd::DrawRuntime rt(h.gfx, webgpu, checks());
        cd::Instance in;
        in.program = "tower_arcs";
        REQUIRE(rt.install(ex::attention_arcs(), in, 9).ok());
        const ex::ArcResources res = ex::arc_resources();
        REQUIRE(res.arcs.size() > 10);
        give_arcs(rt, res, 100);
        const std::vector<gfx::Frame> frames = frame(h, rt);
        REQUIRE(frames.size() == 1);
        REQUIRE(frames[0].passes().size == 3);
        const gfx::DrawList& arcs = list_of(frames[0], 0);
        CHECK(arcs.target().color == h.target.color);
        CHECK(arcs.target().id == h.target.id);
        CHECK(arcs.target().depth == h.target.depth);
        CHECK(arcs.records()[0].instance_count == res.arcs.size());
        CHECK(arcs.records()[0].count == 64);
        CHECK(arcs.records()[0].pick_id == 9);
        const gfx::DrawList& glow = list_of(frames[0], 1);
        const auto& m = rt.internals();
        REQUIRE(m.targets.size() == 1);
        CHECK(glow.target().color == m.targets[0].tex);
        CHECK(glow.target().clear);
        CHECK(m.targets[0].width == 32);
        CHECK(m.targets[0].height == 24);
        const gfx::DrawList& bloom = list_of(frames[0], 2);
        CHECK(bloom.target().color == h.target.color);
        CHECK(!bloom.target().id);
        CHECK(bloom.records()[0].count == 3);

        // The view of the frame: its pick id and the frame counters.
        std::vector<std::uint8_t> view = h.gfx.buffer_bytes(m.view);
        REQUIRE(view.size() >= cp::kViewBytes);
        std::uint32_t frame_x = 0, frame_y = 0, pick = 0;
        std::memcpy(&frame_x, view.data() + 272, 4);
        std::memcpy(&frame_y, view.data() + 276, 4);
        std::memcpy(&pick, view.data() + 280, 4);
        CHECK(frame_x == 0);
        CHECK(frame_y == 1);
        CHECK(pick == 9);
        frame(h, rt);
        view = h.gfx.buffer_bytes(m.view);
        std::memcpy(&frame_x, view.data() + 272, 4);
        std::memcpy(&frame_y, view.data() + 276, 4);
        CHECK(frame_x == 1);
        CHECK(frame_y == 0);
    }
}

TEST_CASE("a pass whose instance count is zero is skipped, and its target is still cleared") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    REQUIRE(rt.install(ex::attention_arcs(), {}, 1).ok());
    ex::ArcResources res = ex::arc_resources();
    res.arcs.clear();
    give_arcs(rt, res, 5);
    const std::vector<gfx::Frame> frames = frame(h, rt);
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].passes().size == 2);
    CHECK(list_of(frames[0], 0).records().size == 0);   // the glow target's clear
    CHECK(list_of(frames[0], 0).target().clear);
    CHECK(list_of(frames[0], 1).records().size == 1);   // the bloom
}

// A draw-only piece with the optional buffer input `values`: a full-screen pass that shows its
// count, then one bar for each value.
cs::Piece optional_bars() {
    cs::Piece p;
    p.name = "bars";
    cs::PieceInput in;
    in.name = "values";
    in.kind = cs::InputKind::buffer;
    in.layout.element = "f32";
    in.optional = true;
    p.inputs = {in};
    cs::Draw d;
    d.camera = cs::CameraMode::none;
    cs::DrawPass back;
    back.fragment = "back_fs";
    cs::DrawPass bars;
    bars.vertex = "bar_vs";
    bars.fragment = "bar_fs";
    bars.instances_of = "values";
    bars.vertices = 6;
    d.passes = {back, bars};
    p.draw = d;
    p.files = {{"draw.wgsl", R"(@fragment
fn back_fs(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(counts.values), in.uv, 1.0);
}
@vertex
fn bar_vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(values[i], f32(v) * 0.1, 0.0, 1.0);
}
@fragment
fn bar_fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
)"}};
    return p;
}

TEST_CASE("an optional buffer input left unbound has the count 0, and its passes run without a version") {
    for (bool webgpu : modes()) {
        CAPTURE(webgpu);
        Host h;
        cd::DrawRuntime rt(h.gfx, webgpu, checks());
        cd::Instance unbound;
        unbound.program = "bars";
        const cp::Verdict v = rt.install(optional_bars(), unbound, 3);
        REQUIRE_MESSAGE(v.ok(), v.message);
        const std::vector<gfx::Frame> frames = frame(h, rt);   // no version to wait for
        REQUIRE(frames.size() == 1);
        auto records = [](const gfx::Frame& f) {
            std::vector<gfx::DrawRecord> all;
            for (std::size_t i = 0; i < f.passes().size; ++i)
                if (const auto* rp = std::get_if<gfx::RenderPass>(&f.passes()[i]))
                    for (std::size_t k = 0; k < rp->list.records().size; ++k) all.push_back(rp->list.records()[k]);
            return all;
        };
        CHECK(records(frames[0]).size() == 1);   // the full-screen pass; the bars have 0 instances
        const std::vector<std::uint8_t> counts = h.gfx.buffer_bytes(rt.internals().counts);
        REQUIRE(counts.size() >= 4);
        std::uint32_t n = 99;
        std::memcpy(&n, counts.data(), 4);
        CHECK(n == 0u);
        const std::vector<float> values{0.5f, -0.5f};
        const cp::Verdict r = rt.set_resource(version("values", values, 1));
        CHECK(r.code == cp::kProgramUnknownResource);

        // Bound, the same input waits for its version, as any buffer input does.
        Host h2;
        cd::DrawRuntime bound(h2.gfx, webgpu, checks());
        cd::Instance with = unbound;
        with.bind["values"] = "program:source/values";
        REQUIRE(bound.install(optional_bars(), with, 4).ok());
        std::vector<gfx::Frame> none;
        CHECK(!bound.encode(&none, {}, h2.target));
        REQUIRE(bound.set_resource(version("values", values, 1)).ok());
        const std::vector<gfx::Frame> drawn = frame(h2, bound);
        REQUIRE(drawn.size() == 1);
        const std::vector<gfx::DrawRecord> both = records(drawn[0]);
        REQUIRE(both.size() == 2);
        CHECK(both[1].instance_count == 2);
    }
}

// A draw-only piece that draws the points of `pts` through the indices of `idx`.
cs::Piece indexed_points() {
    cs::Piece p;
    p.name = "indexed";
    cs::PieceInput pts;
    pts.name = "pts";
    pts.kind = cs::InputKind::buffer;
    pts.layout.element = "vec4<f32>";
    cs::PieceInput idx = pts;
    idx.name = "idx";
    idx.layout.element = "u32";
    p.inputs = {pts, idx};
    cs::Draw d;
    d.camera = cs::CameraMode::none;
    cs::DrawPass tri;
    tri.vertex = "tri_vs";
    tri.fragment = "tri_fs";
    tri.indices = "idx";
    tri.instances = 2;
    d.passes = {tri};
    p.draw = d;
    p.files = {{"draw.wgsl", R"(@vertex
fn tri_vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> { return pts[v]; }
@fragment
fn tri_fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
)"}};
    return p;
}

TEST_CASE("an indexed pass draws the valid indices of its version as an index buffer, each instance") {
    for (bool webgpu : modes()) {
        CAPTURE(webgpu);
        Host h;
        cd::DrawRuntime rt(h.gfx, webgpu, checks());
        cd::Instance in;
        in.program = "indexed";
        in.bind = {{"pts", "program:src/pts"}, {"idx", "program:src/idx"}};
        const cp::Verdict v = rt.install(indexed_points(), in, 5);
        REQUIRE_MESSAGE(v.ok(), v.message);
        const std::vector<std::array<float, 4>> pts{{0, 0, 0, 1}, {1, 0, 0, 1}, {0, 1, 0, 1}, {1, 1, 0, 1}};
        REQUIRE(rt.set_resource(version("pts", pts, 2)).ok());
        const std::vector<std::uint32_t> idx{0, 1, 2, 2, 1, 3, 3, 1, 7};   // 7 is past the four points
        REQUIRE(rt.set_resource(version("idx", idx, 2)).ok());
        std::vector<gfx::Frame> frames = frame(h, rt);
        REQUIRE(frames.size() == 1);
        const gfx::DrawList& list = list_of(frames[0], 0);
        REQUIRE(list.records().size == 1);
        const gfx::DrawRecord& r = list.records()[0];
        CHECK(r.index_buffer);
        CHECK(r.count == 9);
        CHECK(r.instance_count == 2);
        // A version without indices skips the pass.
        const std::vector<std::uint32_t> none;
        REQUIRE(rt.set_resource(version("idx", none, 3)).ok());
        frames = frame(h, rt);
        CHECK(frames.empty());
        // A version over the buffer's first capacity grows it, still with the index usage.
        std::vector<std::uint32_t> many(3000, 1u);
        REQUIRE(rt.set_resource(version("idx", many, 4)).ok());
        frames = frame(h, rt);
        REQUIRE(frames.size() == 1);
        CHECK(list_of(frames[0], 0).records()[0].count == 3000);
    }
}

// A texture of r8unorm of at most 8 by 4 texels and 20 texels, an r32uint texture, and a buffer of u16
// elements, which one full-screen pass reads: textureLoad of each texture and the checked accessor.
cs::Piece textured() {
    cs::Piece p;
    p.name = "textured";
    cs::Resource grey;
    grey.id = "grey";
    grey.kind = cs::ResourceKind::texture;
    grey.format = "r8unorm";
    grey.width = 8;
    grey.height = 4;
    grey.texels = 20;
    cs::Resource ids = grey;
    ids.id = "ids";
    ids.format = "r32uint";
    ids.texels.reset();
    cs::Resource depth;
    depth.id = "depth";
    depth.layout.element = "u16";
    depth.count = 5;
    p.resources = {grey, ids, depth};
    cs::Draw d;
    d.camera = cs::CameraMode::none;
    cs::DrawPass full;
    full.fragment = "full_fs";
    d.passes = {full};
    p.draw = d;
    p.files = {{"draw.wgsl", R"(@fragment
fn full_fs(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let g = textureLoad(grey, vec2<i32>(0, 0), 0).r;
    let k = textureLoad(ids, vec2<i32>(0, 0), 0).r;
    return vec4<f32>(g, f32(k), f32(depth_at(4u)), f32(counts.grey + counts.depth));
}
)"}};
    return p;
}

TEST_CASE("a texture takes the size and format of each version within its largest, and a narrow buffer its count") {
    for (bool webgpu : modes()) {
        CAPTURE(webgpu);
        Host h;
        cd::DrawRuntime rt(h.gfx, webgpu, checks());
        cd::Instance in;
        in.program = "textured";
        const cp::Verdict v = rt.install(textured(), in, 3);
        REQUIRE_MESSAGE(v.ok(), v.message);
        // A 5 by 3 grey version: 15 texels in 4 words. The texture takes its size.
        std::vector<std::uint8_t> g(16, 7);
        cd::ResourceVersion grey;
        grey.resource = "grey";
        grey.step = 1;
        grey.width = 5;
        grey.height = 3;
        grey.format = "r8unorm";
        grey.count = 4;
        grey.bytes = g.data();
        grey.size = g.size();
        const cp::Verdict gv = rt.set_resource(grey);
        REQUIRE_MESSAGE(gv.ok(), gv.message);
        const auto& slot = rt.internals().slots[0];
        CHECK(slot.tex_width == 5);
        CHECK(slot.tex_height == 3);
        CHECK(slot.count == 15);   // the texels, as `counts` gives them
        // An r32uint version of 2 by 2, and 5 u16 elements in 3 words.
        const std::vector<std::uint32_t> k{1, 2, 3, 4};
        cd::ResourceVersion ids = version("ids", k, 1);
        ids.width = ids.height = 2;
        ids.format = "r32uint";
        REQUIRE(rt.set_resource(ids).ok());
        const std::vector<std::uint32_t> words{0x00020001u, 0x00040003u, 0x00000005u};
        cd::ResourceVersion depth = version("depth", words, 1);
        depth.elements = 5;
        REQUIRE(rt.set_resource(depth).ok());
        CHECK(rt.internals().slots[2].count == 5);
        CHECK(frame(h, rt).size() == 1);

        // Refusals, each before anything is written: another format, a side past the largest, more
        // texels than declared, words that do not match the size, and a narrow count that disagrees.
        const std::size_t before = h.gfx.count(gfx::RecordingBackend::OpKind::WriteTexture);
        cd::ResourceVersion bad = grey;
        bad.format = "";   // rgba8unorm
        CHECK(rt.set_resource(bad).code == cp::kProgramBadResource);
        bad = grey;
        bad.width = 9;
        CHECK(rt.set_resource(bad).code == cp::kProgramResourceTooLarge);
        bad = grey;
        bad.width = 7;   // 21 texels, above the 20 of texels
        bad.count = 6;
        bad.size = 24;
        std::vector<std::uint8_t> g24(24, 1);
        bad.bytes = g24.data();
        CHECK(rt.set_resource(bad).code == cp::kProgramResourceTooLarge);
        bad = grey;
        bad.count = 5;   // 15 texels fill 4 words
        CHECK(rt.set_resource(bad).code == cp::kProgramBadResource);
        bad = grey;
        bad.height = 0;
        CHECK(rt.set_resource(bad).code == cp::kProgramResourceTooLarge);
        CHECK(h.gfx.count(gfx::RecordingBackend::OpKind::WriteTexture) == before);
        cd::ResourceVersion d2 = depth;
        d2.elements = 7;   // 7 u16 elements fill 4 words
        CHECK(rt.set_resource(d2).code == cp::kProgramResourceTooLarge);
        d2.elements = 4;   // 4 elements fill 2 words, and the version has 3
        CHECK(rt.set_resource(d2).code == cp::kProgramBadResource);
        d2.elements.reset();
        CHECK(rt.set_resource(d2).code == cp::kProgramBadResource);
        cd::ResourceVersion sized = depth;
        sized.width = 1;
        CHECK(rt.set_resource(sized).code == cp::kProgramBadResource);
        // A version without texels keeps the texture and gives the count 0.
        cd::ResourceVersion none;
        none.resource = "grey";
        none.step = 2;
        none.format = "";   // RESOURCE gives a format only with a size
        REQUIRE(rt.set_resource(none).ok());
        CHECK(rt.internals().slots[0].count == 0);
        // The same size again reuses the texture; another size makes a new one.
        const std::size_t made = h.gfx.count(gfx::RecordingBackend::OpKind::Create, gfx::HandleKind::Texture);
        grey.step = 3;
        REQUIRE(rt.set_resource(grey).ok());
        CHECK(h.gfx.count(gfx::RecordingBackend::OpKind::Create, gfx::HandleKind::Texture) == made);
        std::vector<std::uint8_t> g8(8, 9);
        grey.width = 2;
        grey.height = 4;
        grey.count = 2;
        grey.bytes = g8.data();
        grey.size = g8.size();
        REQUIRE(rt.set_resource(grey).ok());
        CHECK(h.gfx.count(gfx::RecordingBackend::OpKind::Create, gfx::HandleKind::Texture) == made + 1);
        CHECK(frame(h, rt).size() == 1);
    }
}

TEST_CASE("the checks run before anything is made") {
    Host h;
    const std::size_t before = h.gfx.ops().size();
    cd::DrawChecks refusing = checks();
    refusing.manifest = [](const cs::Piece&) {
        cp::Verdict v;
        v.code = cp::kPieceBadManifest;
        v.message = "refused";
        return v;
    };
    cd::DrawRuntime rt(h.gfx, true, refusing);
    const cp::Verdict v = rt.install(ex::attention_arcs(), {}, 1);
    CHECK(v.code == cp::kPieceBadManifest);
    CHECK(rt.status().state == cd::DrawState::refused);
    CHECK(h.gfx.ops().size() == before);   // no buffer, texture, program or pipeline

    cd::DrawRuntime pinned(h.gfx, true, checks());
    cs::Piece p = ex::attention_arcs();
    p.hash = "sha256:" + std::string(64, '0');
    CHECK(pinned.install(p, {}, 1).code == cp::kPieceHashMismatch);
    CHECK(h.gfx.ops().size() == before);

    cd::DrawRuntime pick_offscreen(h.gfx, true, checks());
    p = ex::attention_arcs();
    p.draw->passes[1].pick = true;
    const cp::Verdict off = pick_offscreen.install(p, {}, 1);
    CHECK(off.code == cp::kProgramBadPass);
    CHECK(off.where == "draw.passes[1].pick");
    CHECK(h.gfx.ops().size() == before);
}

TEST_CASE("a pass that does not compile is refused with its file and line") {
    if (!cp::compiler_available()) return;
    Host h;
    cd::DrawRuntime rt(h.gfx, false, checks());
    cs::Piece p = ex::attention_field();
    p.files["draw.wgsl"] += "\nfn broken( {\n";
    const cp::Verdict v = rt.install(p, {}, 1);
    CHECK(v.code == cp::kProgramCompileFailed);
    CHECK(v.where == "draw.passes[0]");
    CHECK(v.message.find("draw.wgsl line") != std::string::npos);
    CHECK(rt.status().state == cd::DrawState::refused);
}

TEST_CASE("a version is held to its resource's layout and count") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    REQUIRE(rt.install(ex::attention_arcs(), {}, 1).ok());
    std::vector<float> too_many(9, 0.0f);
    CHECK(rt.set_resource(version("heads", too_many, 1)).code == cp::kProgramResourceTooLarge);
    cd::ResourceVersion odd = version("heads", std::vector<float>(4, 0.0f), 1);
    odd.size = 12;
    CHECK(rt.set_resource(odd).code == cp::kProgramBadResource);
    CHECK(rt.set_resource(version("nothing", too_many, 1)).code == cp::kProgramUnknownResource);
    // A larger version grows the resource's buffer and binds it again.
    const ex::ArcResources res = ex::arc_resources();
    give_arcs(rt, res, 1);
    CHECK(rt.internals().slots[1].capacity >= res.arcs.size() * sizeof(ex::Arc));
    CHECK(frame(h, rt).size() == 1);
}

TEST_CASE("the reset rules zero the draw buffers at a new epoch, a cursor move or a new version") {
    Host h;
    cs::Piece p = ex::attention_arcs();
    cs::DrawBuffer trail, age, fixed;
    trail.id = "trail";
    trail.layout = ex::element("f32");
    trail.count = 16;
    trail.reset = cs::BufferReset::cursor;
    age.id = "age";
    age.layout = ex::element("u32");
    age.count = 16;
    age.reset = cs::BufferReset::resource;
    fixed.id = "fixed";
    fixed.layout = ex::element("u32");
    fixed.count = 16;
    p.draw->buffers = {trail, age, fixed};
    cd::DrawRuntime rt(h.gfx, true, checks());
    REQUIRE(rt.install(p, {}, 1).ok());
    const ex::ArcResources res = ex::arc_resources();
    give_arcs(rt, res, 10);
    const auto& m = rt.internals();
    auto fill = [&] {
        const std::vector<std::uint8_t> ones(64, 1);
        for (const auto& b : m.buffers) h.gfx.write_buffer(b.buffer, 0, ones.data(), ones.size());
    };
    auto zeroed = [&](std::size_t i) {
        const std::vector<std::uint8_t> bytes = h.gfx.buffer_bytes(m.buffers[i].buffer);
        return bytes[0] == 0 && bytes[63] == 0;
    };
    cd::FrameInputs in;
    frame(h, rt, in);
    fill();
    frame(h, rt, in);
    CHECK(!zeroed(0));
    CHECK(!zeroed(1));
    CHECK(!zeroed(2));

    in.cursor_moved = true;
    frame(h, rt, in);
    CHECK(zeroed(0));
    CHECK(!zeroed(1));
    CHECK(!zeroed(2));
    in.cursor_moved = false;

    fill();
    give_arcs(rt, res, 20);   // a new version of the resources
    frame(h, rt, in);
    CHECK(!zeroed(0));
    CHECK(zeroed(1));
    CHECK(!zeroed(2));

    fill();
    frame(h, rt, in);   // a parameter or nothing changed: nothing returns to zero
    CHECK(!zeroed(0));
    CHECK(!zeroed(1));

    in.epoch = 1;       // a new attachment or epoch
    frame(h, rt, in);
    CHECK(zeroed(0));
    CHECK(zeroed(1));
    CHECK(zeroed(2));
}

TEST_CASE("a program over its frame time for 30 frames stops, naming its costliest pass") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance in;
    in.program = "tower_arcs";
    REQUIRE(rt.install(ex::attention_arcs(), in, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    CHECK(rt.status().frame_limit_ms == doctest::Approx(4.0));

    // A spike, and 29 slow frames then a fast one, never stop it.
    for (int i = 0; i < 29; ++i) {
        REQUIRE(frame(h, rt).size() == 1);
        rt.frame_done(9.0);
    }
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_done(1.0);
    CHECK(rt.status().state == cd::DrawState::running);
    for (int i = 0; i < 30; ++i) {
        REQUIRE(frame(h, rt).size() == 1);
        rt.frame_done(20.0);
    }
    CHECK(rt.status().state == cd::DrawState::running);
    CHECK(rt.status().frame_ms == doctest::Approx(20.0));
    // The next frame measures each pass in a submit of its own.
    const std::vector<gfx::Frame> measured = frame(h, rt);
    REQUIRE(measured.size() == 3);
    rt.frame_done(1.0);
    rt.frame_done(1.5);
    CHECK(rt.status().state == cd::DrawState::running);
    rt.frame_done(50.0);
    const cd::DrawStatus s = rt.status();
    CHECK(s.state == cd::DrawState::stopped);
    CHECK(s.error.code == cp::kProgramDrawOverTime);
    CHECK(s.error.where == "draw.passes[2]");
    CHECK(s.pass == 2);
    CHECK(s.entry == "bloom");
    CHECK(s.error.message == "Stopped tower_arcs: draw pass 2 (bloom) took 50.0 ms a frame, above its 4 ms");
    std::vector<gfx::Frame> after;
    CHECK(!rt.encode(&after, {}, h.target));
    CHECK(rt.internals().passes.empty());   // its objects are released
}

TEST_CASE("a measured frame back under the limit keeps the program") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    REQUIRE(rt.install(ex::attention_arcs(), {}, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    for (int i = 0; i < 30; ++i) {
        frame(h, rt);
        rt.frame_done(6.0);
    }
    REQUIRE(frame(h, rt).size() == 3);
    rt.frame_done(1.0);
    rt.frame_done(1.0);
    rt.frame_done(1.0);
    CHECK(rt.status().state == cd::DrawState::running);
    CHECK(frame(h, rt).size() == 1);
}

TEST_CASE("a frame still pending after 5 s stops its program at once, and the other program of the panel goes on") {
    Host h;
    h.gfx.set_auto_complete(false);
    cd::DrawRuntime slow(h.gfx, true, checks()), other(h.gfx, true, checks());
    cd::Instance a, b;
    a.program = "slow";
    b.program = "other";
    REQUIRE(slow.install(ex::attention_arcs(), a, 1).ok());
    REQUIRE(other.install(ex::attention_field(), b, 2).ok());
    give_arcs(slow, ex::arc_resources(), 1);
    const std::vector<float> pattern = ex::field_pattern();
    REQUIRE(other.set_resource(version("pattern", pattern, 1)).ok());
    REQUIRE(frame(h, slow).size() == 1);
    REQUIRE(frame(h, other).size() == 1);
    slow.frame_pending(400.0);
    CHECK(slow.status().state == cd::DrawState::running);
    slow.frame_pending(1000.5);   // late, not stopped
    CHECK(slow.status().state == cd::DrawState::running);
    slow.frame_pending(5000.5);
    CHECK(slow.status().state == cd::DrawState::stopped);
    CHECK(slow.status().error.code == cp::kProgramDrawTimeout);
    CHECK(slow.status().error.message.find("Stopped slow:") == 0);
    CHECK(slow.status().error.message.find("5 s") != std::string::npos);
    CHECK(frame(h, other).size() == 1);
    CHECK(other.status().state == cd::DrawState::running);
}

TEST_CASE("a late frame waits for its GPU time: a busy GPU keeps the program, over 1 s of GPU time stops it") {
    Host h(64, 48, timed_caps());
    h.gfx.set_auto_complete(false);
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance a;
    a.program = "arcs";
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    REQUIRE(frame(h, rt).size() == 1);
    // A training run holds the GPU: the frame waits 1.5 s by the wall clock but took 2 ms on it.
    rt.frame_pending(1500.0);
    CHECK(rt.status().state == cd::DrawState::running);
    CHECK(frame(h, rt).empty());   // nothing more is submitted while the frame is late
    rt.frame_done(2.0, 1600.0);
    CHECK(rt.status().state == cd::DrawState::running);
    CHECK(rt.status().frame_ms == doctest::Approx(2.0));
    // Late again and again under the same load: each frame is judged by its own GPU time.
    for (int k = 0; k < 4; ++k) {
        REQUIRE(frame(h, rt).size() == 1);
        rt.frame_pending(1200.0);
        rt.frame_done(2.0, 1300.0);
    }
    CHECK(rt.status().state == cd::DrawState::running);
    // A frame whose own GPU time is over 1 s stops the program.
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_pending(1100.0);
    rt.frame_done(1250.0, 1300.0);
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.code == cp::kProgramDrawTimeout);
    CHECK(rt.status().error.message.find("of GPU time") != std::string::npos);
    // A late frame that the driver ended gives no GPU time on a device that has timestamps.
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_pending(1500.0);
    rt.frame_done(-1.0, 2500.0);
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.code == cp::kProgramDrawTimeout);
}

TEST_CASE("a frame over 1 s of GPU time stops the program also when no poll saw it late") {
    Host h(64, 48, timed_caps());
    h.gfx.set_auto_complete(false);
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance a;
    a.program = "arcs";
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_pending(900.0);         // the last poll before it completed: not late yet
    rt.frame_done(1500.0, 1600.0);
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.code == cp::kProgramDrawTimeout);
    CHECK(rt.status().error.message == "Stopped arcs: a frame of its draw passes took 1500.0 ms of GPU time, above 1 s");
}

TEST_CASE("heavy frames in a row stop the program once their GPU time passes 1 s, and a light frame ends the row") {
    Host h(64, 48, timed_caps());
    h.gfx.set_auto_complete(false);
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance a;
    a.program = "arcs";
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    // A busy GPU delays light frames by the wall clock: they never count.
    for (int k = 0; k < 10; ++k) {
        REQUIRE(frame(h, rt).size() == 1);
        rt.frame_pending(1500.0);
        rt.frame_done(3.0, 1600.0);
    }
    // Frames of 400 ms, a light one between them, then frames of 900 ms of GPU time.
    for (const double ms : {400.0, 400.0, 3.0, 400.0, 400.0}) {
        REQUIRE(frame(h, rt).size() == 1);
        rt.frame_done(ms, ms + 50.0);
    }
    CHECK(rt.status().state == cd::DrawState::running);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_done(400.0, 450.0);   // three in a row: 1,200 ms together
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.code == cp::kProgramDrawTimeout);
    CHECK(rt.status().error.message ==
          "Stopped arcs: 3 frames of its draw passes in a row took 1200.0 ms of GPU time together, above 1 s");
    // A program just under 1 s of GPU time a frame stops at its second frame.
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_pending(1100.0);
    rt.frame_done(900.0, 1200.0);
    CHECK(rt.status().state == cd::DrawState::running);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_done(900.0, 950.0);
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.code == cp::kProgramDrawTimeout);
}

TEST_CASE("a frame whose time the backend no longer keeps is done, and neither stops nor counts") {
    Host h(64, 48, timed_caps());
    h.gfx.set_auto_complete(false);
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance a;
    a.program = "arcs";
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_pending(1100.0);   // late at the last poll, then the panel was not drawn for a while
    rt.frame_done(gfx::kGpuMsForgotten, 60000.0);
    CHECK(rt.status().state == cd::DrawState::running);
    CHECK(rt.status().frame_ms < 0);   // nothing measured
    REQUIRE(frame(h, rt).size() == 1);   // it submits again
    rt.frame_done(2.0, 3.0);
    CHECK(rt.status().state == cd::DrawState::running);
    CHECK(rt.status().frame_ms == doctest::Approx(2.0));
}

TEST_CASE("without GPU time, three late frames in a row stop the program, and a frame in time ends the row") {
    Host h;
    h.gfx.set_auto_complete(false);
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance a;
    a.program = "arcs";
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    const auto late = [&] {
        REQUIRE(frame(h, rt).size() == 1);
        rt.frame_pending(1100.0);
        rt.frame_done(-1.0, 1200.0);
    };
    late();
    late();
    CHECK(rt.status().state == cd::DrawState::running);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_done(-1.0, 3.0);   // in time: the row starts again
    late();
    late();
    CHECK(rt.status().state == cd::DrawState::running);
    // A frame that waited behind a late one, and was never seen late, neither counts nor ends the row.
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_done(-1.0, 1100.0);
    CHECK(rt.status().state == cd::DrawState::running);
    late();
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.code == cp::kProgramDrawTimeout);
    CHECK(rt.status().error.message.find("3 frames") != std::string::npos);
}

TEST_CASE("a pick keeps the version of the frame it read, and a version at the same step replaces the copy") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance a;
    a.program = "tower_arcs";
    REQUIRE(rt.install(ex::attention_arcs(), a, 1).ok());
    ex::ArcResources res = ex::arc_resources();
    REQUIRE(res.arcs.size() > 2);
    REQUIRE(res.labels.size() > 1);
    give_arcs(rt, res, 1);
    REQUIRE(frame(h, rt).size() == 1);
    const cd::ShownItems first = rt.shown_items();
    REQUIRE(first);
    const float w1 = res.arcs[1].weight;
    // A newer version arrives while a pick of the first frame waits for its readback.
    std::vector<ex::Arc> arcs2 = res.arcs;
    for (ex::Arc& arc : arcs2) arc.weight += 100.0f;
    REQUIRE(rt.set_resource(version("arcs", arcs2, 2)).ok());
    REQUIRE(frame(h, rt).size() == 1);
    auto weight = [](const std::optional<cd::PickedItem>& p) {
        for (const auto& [name, value] : p->fields)
            if (name == "weight") return value;
        return -1.0;
    };
    const std::optional<cd::PickedItem> old = cd::DrawRuntime::pick_shown(1, first);
    REQUIRE(old);
    CHECK(old->step == 1);
    CHECK(weight(old) == doctest::Approx(w1));
    const std::optional<cd::PickedItem> now = rt.pick(1);
    REQUIRE(now);
    CHECK(now->step == 2);
    CHECK(weight(now) == doctest::Approx(w1 + 100.0f));
    // A version that replaces another at the same step, as after a fork from the shown step: the
    // pick and the labels read the new bytes.
    std::vector<ex::Arc> arcs3 = res.arcs;
    for (ex::Arc& arc : arcs3) arc.weight += 200.0f;
    std::vector<ex::Label> labels3 = res.labels;
    labels3[0].string = res.labels[1].string;
    labels3[0].position[0] += 1.0f;
    REQUIRE(rt.set_resource(version("arcs", arcs3, 2)).ok());
    REQUIRE(rt.set_resource(version("labels", labels3, 1)).ok());
    REQUIRE(frame(h, rt).size() == 1);
    CHECK(weight(rt.pick(1)) == doctest::Approx(w1 + 200.0f));
    const std::vector<cd::Label> labels = rt.labels();
    REQUIRE(!labels.empty());
    CHECK(labels[0].position[0] == doctest::Approx(res.labels[0].position[0] + 1.0f));
    // The pick of the first frame still reads its own version.
    CHECK(weight(cd::DrawRuntime::pick_shown(1, first)) == doctest::Approx(w1));
    CHECK_FALSE(cd::DrawRuntime::pick_shown(1, cd::ShownItems{}));
}

// A piece whose labels take their strings from the published buffer `names`, 32 bytes a string,
// pinned with the end of the text at the anchor.
cs::Piece named_tags() {
    cs::Piece p;
    p.name = "tags";
    cs::Resource anchors;
    anchors.id = "anchors";
    anchors.layout.type_name = "Tag";
    anchors.layout.fields = {{"position", "vec4<f32>"}, {"string", "u32"}};
    anchors.count = 8;
    cs::Resource names;
    names.id = "names";
    names.layout.type_name = "Name";
    names.layout.fields = {{"b0", "vec4<u32>"}, {"b1", "vec4<u32>"}};
    names.count = 8;
    p.resources = {anchors, names};
    cs::Derive dv;
    dv.passes = {cs::DerivePass{"place", {1, 1, 1}, {"anchors", "names"}}};
    p.derive = dv;
    cs::Draw d;
    d.camera = cs::CameraMode::none;
    cs::DrawPass back;
    back.fragment = "back_fs";
    d.passes = {back};
    cs::TextTable t;
    t.anchors = "anchors";
    t.strings_from = "names";
    t.align = cs::TextAlign::right;
    d.text = t;
    p.draw = d;
    p.files = {{"derive.wgsl", "@compute @workgroup_size(1)\nfn place() {}\n"},
               {"draw.wgsl", "@fragment\nfn back_fs(in: CaliperFullscreen) -> @location(0) vec4<f32> {\n"
                             "    return vec4<f32>(f32(counts.names), in.uv, 1.0);\n}\n"}};
    return p;
}

TEST_CASE("labels take their strings from a published buffer, checked as UTF-8, and keep the table's alignment") {
    REQUIRE(cp::check_piece_manifest(named_tags()).ok());
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance in;
    in.program = "tags";
    REQUIRE(rt.install(named_tags(), in, 2).ok());
    struct Tag {
        float position[4];
        std::uint32_t string, pad[3];
    };
    struct Name {
        char bytes[32];
    };
    const std::vector<Tag> tags{{{0, 0, 0, 1}, 0, {}}, {{1, 0, 0, 1}, 1, {}}, {{2, 0, 0, 1}, 2, {}},
                                {{3, 0, 0, 1}, 3, {}}};
    std::vector<Name> names(3);
    std::memcpy(names[0].bytes, "cat", 3);                               // ended by a zero byte
    std::memcpy(names[1].bytes, "d\xC3\x28g", 4);                        // not UTF-8
    std::memcpy(names[2].bytes, "abcdefghijklmnopqrstuvwxyz\xC3\xA9-\xE2\x82\xAC", 32);   // the whole stride
    REQUIRE(rt.set_resource(version("anchors", tags, 1)).ok());
    CHECK(rt.labels().empty());                                         // no frame showed a version yet
    REQUIRE(rt.set_resource(version("names", names, 1)).ok());
    REQUIRE(frame(h, rt).size() == 1);
    const std::vector<cd::Label> labels = rt.labels();
    REQUIRE(labels.size() == 2);                                        // tag 1 is not UTF-8, tag 3 has no string
    CHECK(labels[0].text == "cat");
    CHECK(labels[0].position[0] == 0.0f);
    CHECK(labels[0].align == cs::TextAlign::right);
    CHECK(labels[1].text == std::string("abcdefghijklmnopqrstuvwxyz\xC3\xA9-\xE2\x82\xAC"));
    CHECK(labels[1].position[0] == 2.0f);
    // A newer version of the strings names the labels anew at the next frame.
    std::memcpy(names[0].bytes, "cow", 3);
    REQUIRE(rt.set_resource(version("names", names, 2)).ok());
    CHECK(rt.labels()[0].text == "cat");
    REQUIRE(frame(h, rt).size() == 1);
    CHECK(rt.labels()[0].text == "cow");
}

TEST_CASE("a refused frame is an error of its program only") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance in;
    in.program = "tower_arcs";
    REQUIRE(rt.install(ex::attention_arcs(), in, 1).ok());
    rt.frame_refused("a bind group was released");
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.code == cp::kProgramLayerRefused);
}

TEST_CASE("the panel's rule stops a program after one measured frame") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    cd::Instance in;
    in.program = "tower_arcs";
    REQUIRE(rt.install(ex::attention_arcs(), in, 1).ok());
    give_arcs(rt, ex::arc_resources(), 1);
    rt.request_stop(cp::kProgramDrawOverTime, "the programs of its panel took 18.0 ms a frame, above 16.7 ms");
    REQUIRE(frame(h, rt).size() == 3);
    rt.frame_done(3.0);
    rt.frame_done(2.0);
    rt.frame_done(1.0);
    CHECK(rt.status().state == cd::DrawState::stopped);
    CHECK(rt.status().error.where == "draw.passes[0]");
    CHECK(rt.status().error.message.find("above 16.7 ms") != std::string::npos);
}

TEST_CASE("draw memory over draw.memory_mb stops the program and names the object") {
    SUBCASE("a draw buffer at the install") {
        Host h;
        cs::Piece p = ex::attention_field();
        p.draw->memory_mb = 1;
        cs::DrawBuffer big;
        big.id = "big";
        big.layout = ex::element("vec4<f32>");
        big.count = 1u << 17;   // 2 MiB
        p.draw->buffers = {big};
        cd::DrawRuntime rt(h.gfx, true, checks());
        const cp::Verdict v = rt.install(p, {}, 1);
        CHECK(v.code == cp::kProgramDrawOverMemory);
        CHECK(v.where == "draw.buffers[0]");
        CHECK(rt.status().state == cd::DrawState::stopped);
    }
    SUBCASE("a resource that grows") {
        Host h;
        cs::Piece p = ex::attention_field();
        p.draw->memory_mb = 1;
        p.resources[0].count = 1u << 20;
        cd::DrawRuntime rt(h.gfx, true, checks());
        REQUIRE(rt.install(p, {}, 1).ok());
        const std::vector<float> big(1u << 19, 0.5f);   // 2 MiB
        const cp::Verdict v = rt.set_resource(version("pattern", big, 1));
        CHECK(v.code == cp::kProgramDrawOverMemory);
        CHECK(v.where == "resources[pattern]");
        CHECK(rt.status().state == cd::DrawState::stopped);
    }
    SUBCASE("a target at the panel's size") {
        Host h(1024, 1024);
        cs::Piece p = ex::attention_arcs();
        p.draw->memory_mb = 1;
        p.draw->targets[0].scale = 1.0f;   // 4 MiB at 1,024 by 1,024
        cd::DrawRuntime rt(h.gfx, true, checks());
        REQUIRE(rt.install(p, {}, 1).ok());
        give_arcs(rt, ex::arc_resources(), 1);
        std::vector<gfx::Frame> frames;
        CHECK(!rt.encode(&frames, {}, h.target));
        CHECK(rt.status().error.code == cp::kProgramDrawOverMemory);
        CHECK(rt.status().error.where == "draw.targets[0]");
    }
}

TEST_CASE("decode reads a PROGRAM's declaration and checks the hash against the pin") {
    const cs::Piece field = ex::attention_field();
    const cd::DrawChecks c = checks();
    const cs::Piece hashed = field;
    const std::string pin = c.hash(hashed);
    const std::string manifest = R"({"name": "attention_field", "version": 1,
        "description": "The attention pattern of one head of a two-layer model, as a field with contour lines",
        "imports": ["colormap@1"],
        "inputs": [{"name": "attn0", "kind": "tensor", "rank_min": 4, "rank_max": 4},
                   {"name": "attn1", "kind": "tensor", "rank_min": 4, "rank_max": 4}],
        "params": [{"name": "layer", "type": "i32", "min": 0, "max": 1},
                   {"name": "head", "type": "i32", "min": 0, "max": 3},
                   {"name": "gain", "default": 1.0, "min": 0.1, "max": 10.0}],
        "resources": [{"id": "pattern", "kind": "buffer", "element": "f32", "count": 8192}],
        "derive": {"passes": [{"entry": "copy_rows", "workgroups": [16, 1, 1], "writes": ["pattern"]}]},
        "draw": {"camera": "none", "passes": [{"kind": "render", "fragment": "field"}]}})";
    const std::string decl = R"({"id": "head_pattern", "panel": "pattern",
        "piece": {"name": "attention_field", "version": 1, "hash": ")" + pin + R"("},
        "bind": {"attn0": "live:blocks.0.attn.weights.act"}, "manifest": )" + manifest + "}";
    cs::Piece piece;
    cd::Instance instance;
    const cp::Verdict v = cd::DrawRuntime::decode(decl, ex::kFieldCommon, ex::kFieldDraw, &piece, &instance, c,
                                                      ex::kFieldDerive);
    REQUIRE_MESSAGE(v.ok(), v.message);
    CHECK(instance.program == "head_pattern");
    CHECK(instance.panel == "pattern");
    CHECK(instance.bind.at("attn0") == "live:blocks.0.attn.weights.act");
    CHECK(piece.name == "attention_field");
    CHECK(piece.params == field.params);
    CHECK(piece.resources == field.resources);
    CHECK(piece.derive == field.derive);
    CHECK(piece.draw == field.draw);
    CHECK(piece.hash == pin);

    // Sections with the imports resolved give back the author's files.
    const cp::Resolved r = ex::resolve(hashed);
    const std::string common = r.lib_text + r.files.at("common.wgsl");
    cs::Piece again;
    REQUIRE(cd::DrawRuntime::decode(decl, common, r.files.at("draw.wgsl"), &again, &instance, c,
                                    r.files.at("derive.wgsl"))
                .ok());
    CHECK(again.files == piece.files);

    // Another file, or a broken declaration, is refused.
    std::string changed = ex::kFieldDraw;
    changed += "// changed\n";
    CHECK(cd::DrawRuntime::decode(decl, ex::kFieldCommon, changed, &piece, &instance, c, ex::kFieldDerive).code ==
          cp::kPieceHashMismatch);
    CHECK(cd::DrawRuntime::decode(decl, ex::kFieldCommon, ex::kFieldDraw, &piece, &instance, c).code ==
          cp::kPieceHashMismatch);   // without derive.wgsl the pin's hash cannot hold
    CHECK(cd::DrawRuntime::decode("{", "", "", &piece, &instance, c).code == cp::kPieceBadManifest);
    CHECK(cd::DrawRuntime::decode(R"({"id": "x", "manifest": {"name": 3}})", "", "", &piece, &instance, c).code ==
          cp::kPieceBadManifest);

    // The decoded piece installs and draws.
    Host h;
    cd::DrawRuntime rt(h.gfx, true, c);
    const cp::Verdict installed = rt.install(again, instance, 3);
    REQUIRE_MESSAGE(installed.ok(), installed.message);
}

TEST_CASE("decode reads the sample of the canonical manifest that the app sends") {
    cs::Piece field = ex::attention_field();
    if (cp::piece_hash(field).empty()) { MESSAGE("the checks of pieces are not built"); return; }
    field.sample["attn0"] = cs::SampleInput{"sample/attn0.bin", {1, 4, 32, 32}};
    field.sample["attn1"] = cs::SampleInput{"sample/attn1.bin", {1, 4, 32, 32}};
    const cd::DrawChecks c = checks();
    const std::string pin = c.hash(field);
    const std::string manifest = cp::canonical_manifest(field);
    REQUIRE(manifest.find(R"("sample":[{)") != std::string::npos);   // an array of {input, shape}
    auto declaration = [&](const std::string& m) {
        return R"({"id": "head_pattern", "panel": "pattern",
            "piece": {"name": "attention_field", "version": 1, "hash": ")" + pin + R"("}, "manifest": )" + m + "}";
    };
    cs::Piece piece;
    cd::Instance instance;
    const cp::Verdict v = cd::DrawRuntime::decode(declaration(manifest), ex::kFieldCommon, ex::kFieldDraw, &piece,
                                                  &instance, c, ex::kFieldDerive);
    REQUIRE_MESSAGE(v.ok(), v.message);
    CHECK(piece.sample == field.sample);
    CHECK(piece.hash == pin);

    // A sample in another form, or a broken one, is refused.
    for (const std::string& bad : {std::string(R"({"name": "f", "version": 1, "sample": {"x": {"shape": [1]}}})"),
                                   std::string(R"({"name": "f", "version": 1, "sample": [{"input": "x"}]})"),
                                   std::string(R"({"name": "f", "version": 1, "sample": [{"input": "x", "shape": [-1]}]})"),
                                   std::string(R"({"name": "f", "version": 1, "sample": [{"shape": [1]}]})")})
        CHECK(cd::DrawRuntime::decode(declaration(bad), ex::kFieldCommon, ex::kFieldDraw, &piece, &instance, c,
                                      ex::kFieldDerive)
                  .code == cp::kPieceBadManifest);
}

TEST_CASE("remove releases the program and returns to waiting") {
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    REQUIRE(rt.install(ex::attention_arcs(), {}, 1).ok());
    rt.remove();
    CHECK(rt.status().state == cd::DrawState::waiting);
    CHECK(rt.internals().passes.empty());
    CHECK(!rt.pick(0));
    CHECK(rt.labels().empty());
}

// ---- the app's host of draw halves ----------------------------------------------------------------

namespace {

struct Sink final : caliper_builder::EventSink {
    std::vector<caliper_builder::Event> events;
    std::uint64_t push(caliper_builder::Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
};

cs::Program instance_of(const char* id, const char* piece, const char* panel) {
    cs::Program p;
    p.id = id;
    cs::PieceRef ref;
    ref.name = piece;
    ref.version = 1;
    ref.from = cs::PieceFrom::workspace;
    p.piece = ref;
    p.panel = std::string(panel);
    return p;
}

}  // namespace

TEST_CASE("the draw host gives each panel its programs, packs params and refuses a camera conflict") {
    cs::Spec spec;
    cs::Program field = instance_of("head_pattern", "attention_field", "pattern");
    field.params["gain"] = 2.0;
    cs::Reference layer;
    layer.kind = cs::Reference::Kind::state;
    layer.target = "layer";
    field.params["layer"] = layer;
    field.params["head"] = 9.0;   // above its max of 3: clamped
    spec.programs = {field, instance_of("tower_arcs", "attention_arcs", "tower"),
                     instance_of("tower_field", "attention_field", "tower")};
    cd::DrawHost host;
    Sink sink;
    host.set_events(&sink);
    host.set_resolver([](const cs::Spec&, const cs::Program& p, cs::Piece* out) {
        *out = p.piece->name == "attention_arcs" ? ex::attention_arcs() : ex::attention_field();
        return cp::Verdict{};
    });
    host.set_evaluate([](const cs::Reference& r) { return r.target == "layer" ? nlohmann::json(1) : nlohmann::json(); });
    host.reconcile(spec);

    const std::vector<cd::PanelProgram> pattern = host.feed("pattern")->programs();
    REQUIRE(pattern.size() == 1);
    CHECK(pattern[0].instance.program == "head_pattern");
    CHECK(pattern[0].pick_id >= (1u << 24));
    const std::vector<cd::PanelProgram> tower = host.feed("tower")->programs();
    REQUIRE(tower.size() == 2);
    CHECK(tower[0].refused.ok());
    CHECK(tower[1].refused.code == cp::kProgramCameraConflict);
    CHECK(pattern[0].pick_id != tower[0].pick_id);
    REQUIRE(sink.events.size() == 1);
    CHECK(sink.events[0].fields["program"] == "tower_field");
    CHECK(sink.events[0].fields["change"] == "refused");
    CHECK(sink.events[0].fields["half"] == "draw");
    CHECK(sink.events[0].fields["code"] == cp::kProgramCameraConflict);
    CHECK(host.status().at("tower_field").state == cd::DrawState::refused);

    // The feed gives the runtime its resource at the cursor and the params in the manifest's order.
    const std::vector<float> values = ex::field_pattern();
    int asked = 0;
    host.set_resources([&](const std::string& program, const std::string& resource, std::int64_t cursor,
                           cd::ResourceVersion* out, std::uint64_t* epoch) {
        ++asked;
        CHECK(program == "head_pattern");
        CHECK(resource == "pattern");
        CHECK(cursor == 70);
        *out = version("pattern", values, 64);
        *epoch = 3;
        return true;
    });
    cd::DrawHost::Cursor cursor;
    cursor.step = 70;
    cursor.latest = 90;
    cursor.live = false;
    host.set_cursor(cursor);
    Host h;
    cd::DrawRuntime rt(h.gfx, true, checks());
    REQUIRE(rt.install(pattern[0].piece, pattern[0].instance, pattern[0].pick_id).ok());
    cd::FrameInputs in;
    host.feed("pattern")->frame(rt, &in);
    CHECK(asked == 1);
    CHECK(rt.internals().slots[0].step == 64);
    CHECK(in.epoch == 3);
    CHECK(in.cursor_step == 70);
    CHECK(in.latest_step == 90);
    CHECK(!in.live);
    REQUIRE(in.params.size() == 12);
    std::int32_t l = 0, hd = 0;
    float gain = 0;
    std::memcpy(&l, in.params.data(), 4);
    std::memcpy(&hd, in.params.data() + 4, 4);
    std::memcpy(&gain, in.params.data() + 8, 4);
    CHECK(l == 1);
    CHECK(hd == 3);
    CHECK(gain == doctest::Approx(2.0));

    // Its states reach the events: compiled once, then stopped with the pass and the entry.
    host.feed("pattern")->status(rt);
    host.feed("pattern")->status(rt);
    REQUIRE(frame(h, rt).size() == 1);
    rt.frame_pending(5000.5);
    host.feed("pattern")->status(rt);
    REQUIRE(sink.events.size() == 3);
    CHECK(sink.events[1].fields["change"] == "compiled");
    CHECK(sink.events[2].fields["change"] == "stopped");
    CHECK(sink.events[2].fields["code"] == cp::kProgramDrawTimeout);
    CHECK(sink.events[2].panel == "pattern");
    const nlohmann::json j = cd::draw_status_json(host.status().at("head_pattern"));
    CHECK(j["state"] == "stopped");
    CHECK(j["error"]["code"] == cp::kProgramDrawTimeout);
    CHECK(j["frame_limit_ms"] == 4.0);

    // A program that leaves the spec leaves its panel; a changed piece gets a new revision.
    const std::uint64_t before = host.feed("pattern")->programs()[0].revision;
    spec.programs[0].bind["attn0"] = "live:other";
    host.reconcile(spec);
    CHECK(host.feed("pattern")->programs()[0].revision != before);
    spec.programs.erase(spec.programs.begin());
    host.reconcile(spec);
    CHECK(host.feed("pattern")->programs().empty());
    CHECK(host.status().count("head_pattern") == 0);
}

TEST_CASE("the draw host holds a foreign spec's piece for a review of its hash, then draws it") {
    cs::Spec spec;
    spec.programs = {instance_of("tower_arcs", "attention_arcs", "tower"),
                     instance_of("tower_field", "attention_field", "tower")};
    spec.provenance.foreign = true;
    spec.provenance.from = std::string("a shared instrument");
    cd::DrawHost host;
    Sink sink;
    host.set_events(&sink);
    host.set_resolver([](const cs::Spec&, const cs::Program& p, cs::Piece* out) {
        *out = p.piece->name == "attention_arcs" ? ex::attention_arcs() : ex::attention_field();
        return cp::Verdict{};
    });
    host.reconcile(spec);

    // No program of the foreign spec reaches a runtime: the view installs only what the feed
    // does not refuse, and each one waits in the state review, without an error event. A
    // program in review selects no camera, thus no conflict appears either.
    std::vector<cd::PanelProgram> tower = host.feed("tower")->programs();
    REQUIRE(tower.size() == 2);
    for (const cd::PanelProgram& p : tower) CHECK(p.refused.code == cp::kProgramNeedsReview);
    CHECK(sink.events.empty());
    CHECK(host.status().at("tower_arcs").state == cd::DrawState::review);
    CHECK(host.status().at("tower_arcs").error.ok());
    CHECK(cd::draw_status_json(host.status().at("tower_arcs"))["state"] == "review");

    // A review of another hash, or of another program's target, does not count.
    const std::string arcs = cp::piece_hash(ex::attention_arcs());
    const std::string arcs_hex = arcs.substr(arcs.find(':') + 1);
    spec.provenance.reviewed = {{"program:tower_field", arcs_hex},
                                {"program:tower_arcs", std::string(64, '0')}};
    host.reconcile(spec);
    tower = host.feed("tower")->programs();
    for (const cd::PanelProgram& p : tower) CHECK(p.refused.code == cp::kProgramNeedsReview);

    // The review of the piece's hash for its program lets that program draw; the other stays.
    spec.provenance.reviewed = {{"program:tower_arcs", arcs_hex}};
    host.reconcile(spec);
    tower = host.feed("tower")->programs();
    REQUIRE(tower.size() == 2);
    CHECK(tower[0].refused.ok());
    CHECK(tower[1].refused.code == cp::kProgramNeedsReview);
    CHECK(host.status().at("tower_arcs").state == cd::DrawState::waiting);
    CHECK(host.status().at("tower_field").state == cd::DrawState::review);
    CHECK(sink.events.empty());

    // The person's own spec draws at once.
    spec.provenance = cs::Provenance{};
    host.reconcile(spec);
    tower = host.feed("tower")->programs();
    CHECK(tower[0].refused.ok());
    CHECK(tower[1].refused.code == cp::kProgramCameraConflict);
}
