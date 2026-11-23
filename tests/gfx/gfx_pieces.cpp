// The graphics suite's rows of programmable graphics (design/programmable-graphics.md, Field
// Example, Arc Example and Safety of Draws): the two worked examples of the plan drawn on Metal
// by the draw runtime (draw/draw_runtime.h) from their recorded resources, with no training run
// and no window. The field matches a CPU reference of its fragment; the arcs draw instanced with
// additive blending, write an ID for each arc, and add the bloom of their half-size glow target;
// a pass over its time stops its program after 30 frames with the pass named, while the other
// program of the panel goes on. With CALIPER_GFX_CAPTURE_DIR set, each row writes its pictures
// there, and each prints the median GPU time of its program's frame.
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#ifdef CALIPER_HAVE_METAL

#include "draw/draw_host.h"
#include "example_draw.h"
#include "gfx/colormap_tables.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "live/live_scene.h"
#include "net/wire.h"
#include "renderer/host_renderer.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

caliper_host::HostRenderer* gfx_metal_renderer();

namespace {

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace ex = caliper_test_pieces;
namespace gfx = caliper_gfx;

constexpr std::uint32_t kSize = 512;
constexpr std::uint32_t kBackground = 0xff201814u;   // r 0x14, g 0x18, b 0x20

std::unique_ptr<gfx::Backend> layer() {
    caliper_host::HostRenderer* r = gfx_metal_renderer();
    if (r == nullptr) return nullptr;
    return gfx::make_metal_backend(r->native_device(), r->native_queue());
}

void capture_png(const char* name, const std::vector<std::uint8_t>& rgba, int w, int h) {
    const char* dir = std::getenv("CALIPER_GFX_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4) != 0);
    MESSAGE("capture written: ", path);
}

// The host target of a 3-D scene panel, and the host's colormaps.
struct Panel {
    gfx::Backend& g;
    cd::HostTarget target;
    explicit Panel(gfx::Backend& backend) : g(backend) {
        gfx::TextureDesc d;
        d.width = kSize;
        d.height = kSize;
        d.format = gfx::Format::RGBA8Unorm;
        d.usage = gfx::kRenderTarget | gfx::kSampled | gfx::kCopySrcTex;
        target.color = g.create_texture(d);
        d.format = gfx::Format::RG32Uint;
        d.usage = gfx::kRenderTarget | gfx::kCopySrcTex;
        target.id = g.create_texture(d);
        d.format = gfx::Format::Depth32Float;
        d.usage = gfx::kRenderTarget;
        target.depth = g.create_texture(d);
        target.width = kSize;
        target.height = kSize;
        gfx::TextureDesc c;
        c.width = 256;
        c.height = gfx::cmap::kBuiltinColormaps;
        c.format = gfx::Format::RGBA8Unorm;
        c.usage = gfx::kSampled | gfx::kCopyDstTex;
        target.colormaps = g.create_texture(c);
        std::vector<std::uint32_t> rows(256u * gfx::cmap::kBuiltinColormaps);
        for (int id = 0; id < gfx::cmap::kBuiltinColormaps; ++id)
            std::memcpy(rows.data() + 256u * std::uint32_t(id), gfx::cmap::builtin_lut(id), 256 * 4);
        REQUIRE(g.write_texture(target.colormaps, rows.data(), 256 * 4));
    }
    ~Panel() {
        for (gfx::Texture t : {target.color, target.id, target.depth, target.colormaps})
            if (t) g.release(t);
    }
    // One frame: the scene's clear, then the program's frames, each in a submit of its own; the
    // test waits for each and gives the runtime its GPU time, as the scene view does when polled.
    // A device without timestamps gives none, and the runtime takes the wall time from the submit
    // until the wait returned; `inject_ms` >= 0 gives that time in place of the measured one, for
    // a program whose test must not depend on the speed of the runner's GPU.
    bool frame(cd::DrawRuntime& rt, const cd::FrameInputs& in, double inject_ms = -1) {
        gfx::RenderTarget clear;
        clear.color = target.color;
        clear.id = target.id;
        clear.depth = target.depth;
        clear.clear_rgba = kBackground;
        gfx::Frame f;
        f.render(gfx::DrawList(clear));
        REQUIRE(g.submit(std::move(f)) != 0);
        std::vector<gfx::Frame> frames;
        if (!rt.encode(&frames, in, target)) return false;
        for (gfx::Frame& pf : frames) {
            const auto submitted = std::chrono::steady_clock::now();
            const gfx::Serial s = g.submit(std::move(pf));
            REQUIRE_MESSAGE(s != 0, g.last_error());
            REQUIRE(g.wait(s));
            const double wall_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submitted).count();
            if (inject_ms >= 0) {
                rt.frame_done(inject_ms, wall_ms);
                continue;
            }
            // The time comes with the completion handler, which may run after the wait.
            double ms = -1;
            if (g.caps().has(gfx::kCapTimestamps)) {
                const auto start = std::chrono::steady_clock::now();
                while (!g.submit_gpu_ms(s, &ms) && std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            rt.frame_done(ms, wall_ms);
        }
        return true;
    }
    std::vector<std::uint8_t> read(gfx::Texture t) {
        std::vector<std::uint8_t> out;
        const gfx::Readback rb = g.read_texture(t, 0, 0, kSize, kSize);
        REQUIRE(rb);
        REQUIRE(g.wait(g.last_use(t)));
        REQUIRE(g.take_readback(rb, &out));
        return out;
    }
};

// The field's fragment on the CPU (the plan's draw.wgsl), at pixel (x, y), with the colormap
// magma of the host's tables. *near_contour is set where the contour test is too close to call.
std::array<int, 4> field_reference(const std::vector<float>& pattern, int x, int y, std::uint32_t layer,
                                   std::uint32_t head, float gain, bool* near_contour) {
    const std::uint32_t T = 32;
    auto value = [&](std::uint32_t i, std::uint32_t j) { return pattern[((layer * 4 + head) * T + i) * T + j]; };
    const float u = (float(x) + 0.5f) / float(kSize), v = (float(y) + 0.5f) / float(kSize);
    const float px = u * T - 0.5f, py = v * T - 0.5f;
    const std::uint32_t ax = std::uint32_t(std::min(std::max(std::floor(px), 0.0f), float(T - 1)));
    const std::uint32_t ay = std::uint32_t(std::min(std::max(std::floor(py), 0.0f), float(T - 1)));
    const std::uint32_t bx = std::min(ax + 1, T - 1), by = std::min(ay + 1, T - 1);
    const float fx = std::min(std::max(px - float(ax), 0.0f), 1.0f);
    const float fy = std::min(std::max(py - float(ay), 0.0f), 1.0f);
    const float top = value(ay, ax) + (value(ay, bx) - value(ay, ax)) * fx;
    const float bottom = value(by, ax) + (value(by, bx) - value(by, ax)) * fx;
    const float t = std::min(std::max((top + (bottom - top) * fy) * gain, 0.0f), 1.0f);
    const std::uint32_t* lut = gfx::cmap::builtin_lut(1);   // magma
    const float xf = t * 255.0f;
    const int x0 = int(std::floor(xf)), x1 = std::min(x0 + 1, 255);
    const float f = xf - float(x0);
    std::array<float, 4> c{};
    for (int k = 0; k < 4; ++k) {
        const float a = float((lut[x0] >> (8 * k)) & 0xff), b = float((lut[x1] >> (8 * k)) & 0xff);
        c[std::size_t(k)] = (a + (b - a) * f) / 255.0f;
    }
    const float contour = std::fabs((t * 8.0f - std::floor(t * 8.0f)) - 0.5f);
    *near_contour = std::fabs(contour - 0.04f) < 0.01f;
    if (contour < 0.04f)
        for (float& ch : c) ch = ch + (1.0f - ch) * 0.35f;
    std::array<int, 4> out{};
    for (int k = 0; k < 4; ++k) out[std::size_t(k)] = int(std::lround(c[std::size_t(k)] * 255.0f));
    return out;
}

cd::FrameInputs arc_camera(float pixel_ratio) {
    cd::FrameInputs in;
    const ex::Mat4 view = ex::view_matrix({0.0f, 1.9f, 2.6f}, {0.0f, 0.3f, 0.0f});
    const ex::Mat4 proj = ex::perspective(0.8f, 1.0f, 0.1f, 20.0f);
    in.clip_from_world = ex::multiply(proj, view);
    in.view_from_world = view;
    in.world_from_clip = ex::Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    in.eye = {0.0f, 1.9f, 2.6f, 1.0f};
    in.viewport = {0, 0, float(kSize), float(kSize)};
    in.pixel_ratio = pixel_ratio;
    return in;
}

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

int channel_distance(const std::uint8_t* px, std::uint32_t rgba) {
    int d = 0;
    for (int k = 0; k < 3; ++k) d = std::max(d, std::abs(int(px[k]) - int((rgba >> (8 * k)) & 0xff)));
    return d;
}

}  // namespace

TEST_CASE("gfx/Metal pieces: the field draws its recorded pattern as the CPU reference of its fragment") {
    std::unique_ptr<gfx::Backend> g = layer();
    if (!g) CALIPER_SKIP_NO_GPU("no Metal device");
    Panel panel(*g);
    const cs::Piece piece = ex::attention_field();
    cd::DrawRuntime rt(*g, false, ex::draw_checks());
    cd::Instance in;
    in.program = "head_pattern";
    in.panel = "pattern";
    const cp::Verdict v = rt.install(piece, in, 1);
    REQUIRE_MESSAGE(v.ok(), v.message);
    const std::vector<float> pattern = ex::field_pattern();
    REQUIRE(rt.set_resource(version("pattern", pattern, 100)).ok());
    cd::FrameInputs frame;
    frame.params = ex::pack({1, 2, 2.0}, piece);   // layer 1, head 2 (the induction head), gain 2
    for (int i = 0; i < 10; ++i) REQUIRE(panel.frame(rt, frame));
    const std::vector<std::uint8_t> rgba = panel.read(panel.target.color);
    capture_png("field.png", rgba, int(kSize), int(kSize));
    MESSAGE("attention_field: median GPU time of a frame ", rt.status().frame_ms, " ms at 512 by 512");
    CHECK(rt.status().state == cd::DrawState::running);
    CHECK(rt.status().frame_ms >= 0);

    int compared = 0, off = 0, worst = 0;
    for (int y = 4; y < int(kSize); y += 8)
        for (int x = 4; x < int(kSize); x += 8) {
            bool near = false;
            const std::array<int, 4> ref = field_reference(pattern, x, y, 1, 2, 2.0f, &near);
            if (near) continue;
            const std::uint8_t* px = rgba.data() + (std::size_t(y) * kSize + std::size_t(x)) * 4;
            int d = 0;
            for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int(px[k]) - ref[std::size_t(k)]));
            worst = std::max(worst, d);
            if (d > 2) ++off;
            ++compared;
        }
    MESSAGE("field: ", compared, " pixels compared, ", off, " off by more than 2 levels, worst ", worst);
    CHECK(compared > 3500);
    CHECK(off == 0);
    // The induction stripe: query 20 attends to key 9, and not to key 3.
    const std::uint8_t* stripe = rgba.data() + (std::size_t(20 * 16 + 8) * kSize + 9 * 16 + 8) * 4;
    const std::uint8_t* dark = rgba.data() + (std::size_t(20 * 16 + 8) * kSize + 3 * 16 + 8) * 4;
    CHECK(int(stripe[0]) + stripe[1] + stripe[2] > 600);
    CHECK(int(dark[0]) + dark[1] + dark[2] < 60);
}

TEST_CASE("gfx/Metal pieces: the arcs draw instanced and additive, with an ID for each arc and the bloom of their glow") {
    std::unique_ptr<gfx::Backend> g = layer();
    if (!g) CALIPER_SKIP_NO_GPU("no Metal device");
    Panel panel(*g);
    const cs::Piece piece = ex::attention_arcs();
    cd::DrawRuntime rt(*g, false, ex::draw_checks());
    cd::Instance in;
    in.program = "tower_arcs";
    in.panel = "tower";
    const std::uint32_t pick_id = 1u << 24;
    const cp::Verdict v = rt.install(piece, in, pick_id);
    REQUIRE_MESSAGE(v.ok(), v.message);
    const ex::ArcResources res = ex::arc_resources();
    REQUIRE(rt.set_resource(version("heads", res.heads, 100)).ok());
    REQUIRE(rt.set_resource(version("arcs", res.arcs, 100)).ok());
    REQUIRE(rt.set_resource(version("labels", res.labels, 100)).ok());
    cd::FrameInputs frame = arc_camera(2.0f);
    frame.params = ex::pack({0.05, 1.0}, piece);
    for (int i = 0; i < 10; ++i) REQUIRE(panel.frame(rt, frame));
    const std::vector<std::uint8_t> rgba = panel.read(panel.target.color);
    const std::vector<std::uint8_t> ids = panel.read(panel.target.id);
    capture_png("arcs.png", rgba, int(kSize), int(kSize));
    MESSAGE("attention_arcs: ", res.arcs.size(), " arcs, median GPU time of a frame ", rt.status().frame_ms,
            " ms at 512 by 512");
    CHECK(rt.status().state == cd::DrawState::running);

    int lit = 0, picked = 0, halo = 0, bad_item = 0;
    for (std::size_t p = 0; p < std::size_t(kSize) * kSize; ++p) {
        std::uint32_t id[2];
        std::memcpy(id, ids.data() + p * 8, 8);
        const bool bright = channel_distance(rgba.data() + p * 4, kBackground) > 8;
        lit += bright;
        if (id[0] == pick_id) {
            ++picked;
            if (id[1] >= res.arcs.size()) ++bad_item;
        } else {
            CHECK(id[0] == 0);
            halo += bright;
        }
    }
    MESSAGE("arcs: ", lit, " lit pixels, ", picked, " with an arc's ID, ", halo, " lit by the bloom alone");
    CHECK(lit >= 2000);
    CHECK(picked >= 1000);
    CHECK(bad_item == 0);
    CHECK(halo >= 500);

    // Without the glow the bloom adds nothing: the halo goes and the arcs stay.
    cd::FrameInputs dark = frame;
    dark.params = ex::pack({0.05, 0.0}, piece);
    REQUIRE(panel.frame(rt, dark));
    const std::vector<std::uint8_t> plain = panel.read(panel.target.color);
    capture_png("arcs_without_glow.png", plain, int(kSize), int(kSize));
    int halo_plain = 0;
    for (std::size_t p = 0; p < std::size_t(kSize) * kSize; ++p) {
        std::uint32_t id0 = 0;
        std::memcpy(&id0, ids.data() + p * 8, 4);
        if (id0 == 0 && channel_distance(plain.data() + p * 4, kBackground) > 8) ++halo_plain;
    }
    CHECK(halo_plain * 10 < halo);
}

TEST_CASE("gfx/Metal pieces: a pass over its frame time stops its program after 30 frames, named, and the other goes on") {
    std::unique_ptr<gfx::Backend> g = layer();
    if (!g) CALIPER_SKIP_NO_GPU("no Metal device");
    if (!g->caps().has(gfx::kCapTimestamps)) CALIPER_SKIP_NOT_APPLICABLE("the Metal device gives no GPU timestamps");
    Panel panel(*g);
    cs::Piece slow = ex::attention_field();
    slow.name = "slow_field";
    slow.draw->frame_ms = 0.5;
    cs::DrawPass loop;
    loop.kind = cs::PassKind::render;
    loop.fragment = "slow";
    loop.blend = cs::DrawBlend::additive;
    slow.draw->passes.push_back(loop);
    slow.files["draw.wgsl"] += R"(
@fragment
fn slow(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    var x = in.uv.x + in.uv.y;
    for (var i = 0; i < 20000; i = i + 1) {
        x = fract(sin(x * 12.9898 + f32(i)) * 43758.5453);
    }
    return vec4<f32>(0.0, 0.0, 0.0, x * 0.0001);
}
)";
    cd::DrawRuntime rt(*g, false, ex::draw_checks()), other(*g, false, ex::draw_checks());
    cd::Instance a, b;
    a.program = "slow_pattern";
    b.program = "head_pattern";
    REQUIRE_MESSAGE(rt.install(slow, a, 1).ok(), rt.status().error.message);
    REQUIRE(other.install(ex::attention_field(), b, 2).ok());
    const std::vector<float> pattern = ex::field_pattern();
    REQUIRE(rt.set_resource(version("pattern", pattern, 1)).ok());
    REQUIRE(other.set_resource(version("pattern", pattern, 1)).ok());
    cd::FrameInputs frame;
    frame.params = ex::pack({1, 2, 2.0}, slow);
    // The slow pass loops 20,000 times in each fragment, thousands of times its 0.5 ms on any GPU;
    // the other program gets 1 ms a frame, so that the speed of the runner's GPU decides nothing.
    int frames = 0;
    while (frames < 40 && panel.frame(rt, frame)) {
        ++frames;
        REQUIRE(panel.frame(other, frame, 1.0));
    }
    const cd::DrawStatus s = rt.status();
    MESSAGE("slow_pattern: stopped after ", frames, " frames: ", s.error.message);
    CHECK(s.state == cd::DrawState::stopped);
    CHECK(s.error.code == cp::kProgramDrawOverTime);
    CHECK(s.error.where == "draw.passes[1]");
    CHECK(s.entry == "slow");
    CHECK(frames == 31);   // 30 frames over its limit, then one that measures each pass
    CHECK(other.status().state == cd::DrawState::running);
    CHECK(panel.frame(other, frame, 1.0));
}

TEST_CASE("gfx/Metal pieces: a draw-half compute pass feeds an instanced draw, keeps its state and resets it, and depth tests") {
    std::unique_ptr<gfx::Backend> g = layer();
    if (!g) CALIPER_SKIP_NO_GPU("no Metal device");
    Panel panel(*g);
    cs::Piece p;
    p.name = "dots";
    p.description = "Squares that a compute pass places and counts frames into";
    cs::Draw draw;
    draw.camera = cs::CameraMode::none;
    cs::DrawBuffer state, points;
    state.id = "state";
    state.layout = ex::element("u32");
    state.count = 4;
    state.reset = cs::BufferReset::cursor;
    points.id = "points";
    points.layout = ex::fields("Pt", {{"pos", "vec4<f32>"}});
    points.count = 64;
    draw.buffers = {state, points};
    cs::DrawPass tick, dots, back;
    tick.kind = cs::PassKind::compute;
    tick.entry = "tick";
    tick.writes = {"state", "points"};
    dots.kind = cs::PassKind::render;
    dots.vertex = "dot_vs";
    dots.fragment = "dot_fs";
    dots.instances = 64;
    dots.vertices = 6;
    back.kind = cs::PassKind::render;
    back.vertex = "back_vs";
    back.fragment = "backdrop";
    back.vertices = 3;
    back.depth = cs::DrawDepth::test;
    draw.passes = {tick, dots, back};
    p.draw = draw;
    p.files["draw.wgsl"] = R"(// draw.wgsl
@compute @workgroup_size(64)
fn tick(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i == 0u) {
        state[0] = state[0] + 1u;
    }
    let col = f32(i % 8u);
    let row = f32(i / 8u);
    points[i] = Pt(vec4<f32>(-0.7 + col * 0.2, -0.7 + row * 0.2, 0.5, 1.0));
}

struct DotOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) n: u32,
}

@vertex
fn dot_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> DotOut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                                      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let p = points[i].pos;
    var out: DotOut;
    out.position = vec4<f32>(p.xy + corners[v] * 0.05, p.z, 1.0);
    out.n = state[0];
    return out;
}

@fragment
fn dot_fs(in: DotOut) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(in.n) / 255.0, 0.5, 0.25, 1.0);
}

@vertex
fn back_vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((v << 1u) & 2u), f32(v & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.9, 1.0);
}

@fragment
fn backdrop() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
)";
    cd::DrawRuntime rt(*g, false, ex::draw_checks());
    cd::Instance in;
    in.program = "dots";
    const cp::Verdict v = rt.install(p, in, 1);
    REQUIRE_MESSAGE(v.ok(), v.message);
    cd::FrameInputs frame;
    for (int i = 0; i < 3; ++i) REQUIRE(panel.frame(rt, frame));
    std::vector<std::uint8_t> rgba = panel.read(panel.target.color);
    capture_png("dots.png", rgba, int(kSize), int(kSize));
    // A square at clip (-0.7, -0.7), the bottom left: pixel (77, 435); the backdrop between squares.
    auto px = [&](int x, int y) { return rgba.data() + (std::size_t(y) * kSize + std::size_t(x)) * 4; };
    CHECK(int(px(77, 435)[0]) == 3);     // three frames counted in the draw buffer
    CHECK(int(px(77, 435)[1]) == 128);
    CHECK(int(px(77, 435)[2]) == 64);   // the backdrop behind it failed the depth test
    CHECK(int(px(102, 410)[2]) == 255);  // and drew between the squares
    frame.cursor_moved = true;           // the reset rule "cursor": the count starts again
    REQUIRE(panel.frame(rt, frame));
    rgba = panel.read(panel.target.color);
    CHECK(int(px(77, 435)[0]) == 1);
    MESSAGE("dots: median GPU time of a frame ", rt.status().frame_ms, " ms");
}

namespace {

// A feed of fixed programs over the recorded resources, as the app's draw host gives them.
struct FixedFeed final : cd::DrawFeed {
    std::vector<cd::PanelProgram> list;
    std::uint64_t rev = 1;
    std::vector<float> pattern = ex::field_pattern();
    ex::ArcResources arcs = ex::arc_resources();
    std::map<std::string, std::vector<cd::DrawState>> states;
    std::uint64_t revision() const override { return rev; }
    std::vector<cd::PanelProgram> programs() const override { return list; }
    cd::DrawChecks checks() const override { return ex::draw_checks(); }
    void frame(cd::DrawRuntime& rt, cd::FrameInputs* in) override {
        if (rt.instance().program == "tower_arcs") {
            rt.set_resource(version("heads", arcs.heads, 1));
            rt.set_resource(version("arcs", arcs.arcs, 1));
            rt.set_resource(version("labels", arcs.labels, 1));
            in->params = ex::pack({0.05, 1.0}, ex::attention_arcs());
        } else {
            rt.set_resource(version("pattern", pattern, 1));
            in->params = ex::pack({1, 2, 2.0}, ex::attention_field());
        }
    }
    void status(const cd::DrawRuntime& rt) override { states[rt.instance().program].push_back(rt.status().state); }
    void add(cs::Piece piece, const char* program, std::uint32_t pick_id) {
        cd::PanelProgram p;
        p.piece = std::move(piece);
        p.instance.program = program;
        p.instance.panel = "tower";
        p.pick_id = pick_id;
        p.revision = ++rev;
        list.push_back(std::move(p));
    }
};

struct ViewRig {
    std::unique_ptr<gfx::Backend> g;
    std::unique_ptr<caliper_live_views::SceneView> view;
    bool init() {
        g = layer();
        if (!g) return false;
        view = std::make_unique<caliper_live_views::SceneView>(*g, gfx::shaders::kMetalPrograms);
        std::string err;
        REQUIRE_MESSAGE(view->init(&err), err);
        view->describe(caliper_net::wire::SceneView{CALIPER_NET_SPACE_3D, "+y", ""});
        return true;
    }
    // One frame of the view, then the poll that a later frame makes once the GPU is done; the
    // submits' times come with the completion handlers, which may run after the wait.
    void frame() {
        std::string why;
        REQUIRE_MESSAGE(view->render(kSize, kSize, 2.0f, kBackground, &why), why);
        const gfx::Serial last = g->last_use(view->color());
        REQUIRE(g->wait(last));
        double ms = 0;
        const auto start = std::chrono::steady_clock::now();
        while (!g->submit_gpu_ms(last, &ms) && std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        view->poll();
    }
    ~ViewRig() { view.reset(); }
};

}  // namespace

TEST_CASE("gfx/Metal pieces: the scene view draws its programs after its scene, with their IDs, in submits of their own") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    FixedFeed feed;
    feed.add(ex::attention_arcs(), "tower_arcs", 1u << 24);
    rig.view->set_draw_feed(&feed);
    for (int i = 0; i < 3; ++i) rig.frame();
    REQUIRE(rig.view->draw_runtimes().size() == 1);
    const cd::DrawStatus s = rig.view->draw_runtimes()[0]->status();
    CHECK_MESSAGE(s.state == cd::DrawState::running, s.error.message);
    CHECK(s.frames == 3);
    CHECK(feed.states["tower_arcs"] == std::vector<cd::DrawState>{cd::DrawState::compiled, cd::DrawState::running});
    std::vector<std::uint8_t> rgba;
    std::string why;
    REQUIRE_MESSAGE(rig.view->read_pixels(&rgba, &why), why);
    capture_png("scene_view_tower.png", rgba, int(kSize), int(kSize));
    REQUIRE(rig.view->id_target());
    std::vector<std::uint8_t> ids;
    const gfx::Readback rb = rig.g->read_texture(rig.view->id_target(), 0, 0, kSize, kSize);
    REQUIRE(rig.g->wait(rig.g->last_use(rig.view->id_target())));
    REQUIRE(rig.g->take_readback(rb, &ids));
    int lit = 0, picked = 0;
    for (std::size_t p = 0; p < std::size_t(kSize) * kSize; ++p) {
        lit += channel_distance(rgba.data() + p * 4, kBackground) > 8;
        std::uint32_t id0 = 0;
        std::memcpy(&id0, ids.data() + p * 8, 4);
        picked += id0 == (1u << 24);
    }
    MESSAGE("scene view: ", lit, " lit pixels, ", picked, " with an arc's ID, median GPU time ", s.frame_ms, " ms");
    CHECK(lit >= 2000);
    CHECK(picked >= 1000);
    // A program that leaves the feed leaves the view, and the scene draws alone.
    feed.list.clear();
    ++feed.rev;
    rig.frame();
    CHECK(rig.view->draw_runtimes().empty());
    rig.view->set_draw_feed(nullptr);
}

TEST_CASE("gfx/Metal pieces: a scene view of programs alone opens at its starting camera over the fit of their box") {
    namespace sc = caliper_scene;
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    FixedFeed feed;
    feed.add(ex::attention_arcs(), "tower_arcs", 1u << 24);
    rig.view->set_draw_feed(&feed);
    rig.frame();
    // No starting camera: the fit of the box at the default angles.
    const sc::Orbit* o = std::get_if<sc::Orbit>(&rig.view->camera());
    REQUIRE(o);
    CHECK(o->azimuth == doctest::Approx(sc::kDefaultAzimuth));
    CHECK(o->elevation == doctest::Approx(sc::kDefaultElevation));
    const double fit_distance = o->distance;
    CHECK(fit_distance > 0.0);
    // The angles of view_camera alone: the fit of the box at those angles.
    sc::StartingCamera angles;
    angles.form = sc::StartingCamera::Form::Orbit;
    angles.azimuth_deg = 40.0;
    angles.elevation_deg = 25.0;
    rig.view->set_starting(angles);
    rig.frame();
    o = std::get_if<sc::Orbit>(&rig.view->camera());
    REQUIRE(o);
    CHECK(o->azimuth == doctest::Approx(40.0 * sc::kPi / 180.0));
    CHECK(o->elevation == doctest::Approx(25.0 * sc::kPi / 180.0));
    CHECK(o->distance == doctest::Approx(fit_distance));
    // A whole orbit: its target and distance as given.
    sc::StartingCamera whole = angles;
    whole.target = sc::Vec3{0.5, 1.0, -0.5};
    whole.distance = 7.0;
    rig.view->set_starting(whole);
    rig.frame();
    o = std::get_if<sc::Orbit>(&rig.view->camera());
    REQUIRE(o);
    CHECK(o->target[0] == 0.5);
    CHECK(o->target[1] == 1.0);
    CHECK(o->target[2] == -0.5);
    CHECK(o->distance == 7.0);
    CHECK(o->azimuth == doctest::Approx(40.0 * sc::kPi / 180.0));
    rig.view->set_draw_feed(nullptr);
}

TEST_CASE("gfx/Metal pieces: the scene view stops a program over its time and the other program draws on") {
    ViewRig rig;
    if (!rig.init()) CALIPER_SKIP_NO_GPU("no Metal device");
    cs::Piece slow = ex::attention_field();
    slow.name = "slow_field";
    slow.draw->camera = cs::CameraMode::orbit;
    slow.draw->frame_ms = 0.5;
    cs::DrawPass loop;
    loop.kind = cs::PassKind::render;
    loop.fragment = "slow";
    loop.blend = cs::DrawBlend::additive;
    slow.draw->passes.push_back(loop);
    slow.files["draw.wgsl"] += R"(
@fragment
fn slow(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    var x = in.uv.x + in.uv.y;
    for (var i = 0; i < 20000; i = i + 1) {
        x = fract(sin(x * 12.9898 + f32(i)) * 43758.5453);
    }
    return vec4<f32>(0.0, 0.0, 0.0, x * 0.0001);
}
)";
    // The view measures real GPU times: tower_arcs keeps the largest frame budget, 16 ms, thus a
    // slow or paravirtual GPU does not stop it, and slow_field is over its 0.5 ms by thousands
    // of times on any GPU.
    cs::Piece arcs = ex::attention_arcs();
    arcs.draw->frame_ms = cs::piece_limits::kFrameMsMax;
    FixedFeed feed;
    feed.add(arcs, "tower_arcs", 1u << 24);
    feed.add(slow, "slow_field", (1u << 24) + 1);
    rig.view->set_draw_feed(&feed);
    int frames = 0;
    auto slow_state = [&] {
        for (const cd::DrawRuntime* rt : rig.view->draw_runtimes())
            if (rt->instance().program == "slow_field") return rt->status();
        return cd::DrawStatus{};
    };
    while (frames < 60 && slow_state().state != cd::DrawState::stopped) {
        rig.frame();
        ++frames;
    }
    const cd::DrawStatus s = slow_state();
    MESSAGE("scene view: slow_field stopped after ", frames, " frames: ", s.error.message);
    CHECK(s.state == cd::DrawState::stopped);
    CHECK(s.error.code == cp::kProgramDrawOverTime);
    CHECK(s.error.where == "draw.passes[1]");
    CHECK(frames <= 33);
    CHECK(feed.states["slow_field"].back() == cd::DrawState::stopped);
    rig.frame();
    for (const cd::DrawRuntime* rt : rig.view->draw_runtimes())
        if (rt->instance().program == "tower_arcs") CHECK(rt->status().state == cd::DrawState::running);
    rig.view->set_draw_feed(nullptr);
}

#endif  // CALIPER_HAVE_METAL
