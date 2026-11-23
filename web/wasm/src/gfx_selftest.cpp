// gfx_selftest.cpp: the graphics layer's self-test in the page (gfx_selftest.h).
//
// One Frame holds four passes, in this order:
//   1. a dispatch of cs_fill over the left half of a 16 x 16 RGBA8 storage texture, its parameters
//      a uniform in the frame's arena at offset 0;
//   2. the same over the right half with other parameters, at the arena's second slice;
//   3. a copy of an 8 x 8 RGBA8 image from a buffer with 256-byte rows into a texture;
//   4. one draw list on a 64 x 64 target with an RG32Uint ID target and a Depth32Float depth
//      buffer: an indexed quad (u32 indices from a byte offset, with a base vertex) and an
//      instanced draw of four squares fed by five vertex streams (a vertex-step corner stream,
//      instance-step centres at a 4-byte offset, colours imported from host memory by a copy, a
//      constant half size at stride 0, and depths), depth-tested against the quad;
//   5. a second draw list that loads the same colour and depth targets, without the ID target,
//      and draws an alpha-blended instanced draw of two more squares from first_instance 4 under
//      a scissor, one of them fully transparent. A compatibility-mode device cannot blend a
//      pipeline that also writes the ID target, so blending has a list of its own.
// Every edge lies on a pixel boundary and every colour is a byte divided by 255, so each texel of
// the colour, ID and depth targets and of both textures is known exactly; the CPU computes them
// and the readbacks must equal them byte for byte. The draw program is released before the frame
// is submitted: its pipelines and bind groups hold what they need, so the frame draws as if it
// were alive.
//
// Two more frames follow the first, before the readbacks: one whose draw samples the colour target
// it renders into, which the layer's core refuses; and one whose dispatch binds one buffer as both
// read-only and read-write storage, which the core passes and WebGPU refuses when it is encoded, so
// that the backend's error scope around the submit must report it.
//
// The report is "ok" or "fail" and then: color, id, depth, fill and copy, the texels that differ
// in each readback; import, the route label and bytes copied of the colour stream's import;
// own-target, refused or submitted; submit-error, reported or lost; level, core or compatibility;
// errors, the device errors other than the expected refusal.
#include "gfx_selftest.h"

#include "gfx/shaders/generated/programs_webgpu.h"
#include "renderer/webgpu_backend.h"

#include <cstdio>
#include <cstring>

namespace caliper_web {

namespace gfx = caliper_gfx;

namespace {

constexpr const char* kDrawWgsl = R"wgsl(
struct Draw { color: u32, pick: u32, depth: f32, pad: u32 };
@group(0) @binding(0) var<uniform> d: Draw;   // dynamic: the draw's slot in the list's arena

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat, either) color: u32,
    @location(1) @interpolate(flat, either) item: u32,
};

fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu),
                     f32((c >> 24u) & 0xffu)) / 255.0;
}

@vertex
fn vs_quad(@location(0) p: vec2<f32>) -> VOut {
    return VOut(vec4<f32>(p, d.depth, 1.0), d.color, 0u);
}

@vertex
fn vs_inst(@builtin(instance_index) iid: u32, @location(0) corner: vec2<f32>, @location(1) center: vec2<f32>,
           @location(2) color: u32, @location(3) half: f32, @location(4) z: f32) -> VOut {
    return VOut(vec4<f32>(center + corner * half, z, 1.0), color, iid);
}

struct Out { @location(0) color: vec4<f32>, @location(1) id: vec2<u32> };

@fragment
fn fs_pick(v: VOut) -> Out { return Out(unpack(v.color), vec2<u32>(d.pick, v.item)); }

@fragment
fn fs_color(v: VOut) -> @location(0) vec4<f32> { return unpack(v.color); }
)wgsl";

constexpr const char* kFillWgsl = R"wgsl(
struct Fill { value: u32, x0: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> f: Fill;   // dynamic: the dispatch's slot in the frame's arena
@group(1) @binding(0) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x + f.x0;
    let c = vec4<u32>(x * 16u, gid.y * 16u, f.value, 255u);
    textureStore(dst, vec2<i32>(i32(x), i32(gid.y)), vec4<f32>(c) / 255.0);
}
)wgsl";

struct DrawBlock { uint32_t color, pick; float depth; uint32_t pad; };
struct FillBlock { uint32_t value, x0, pad0, pad1; };

constexpr gfx::EntryDesc kDrawEntries[] = {{"vs_quad", gfx::kStageVertex, {1, 1, 1}},
                                           {"vs_inst", gfx::kStageVertex, {1, 1, 1}},
                                           {"fs_pick", gfx::kStageFragment, {1, 1, 1}},
                                           {"fs_color", gfx::kStageFragment, {1, 1, 1}}};
constexpr gfx::BindingDesc kDrawBindings[] = {
    {0, 0, gfx::BindingType::Uniform, gfx::kStageVertex | gfx::kStageFragment, true, sizeof(DrawBlock)}};
constexpr gfx::ProgramDesc kDraw = {"caliper.selftest.draw", kDrawEntries, 4, kDrawBindings, 1, kDrawWgsl,
                                    nullptr, nullptr, 0};
constexpr gfx::EntryDesc kFillEntries[] = {{"cs_fill", gfx::kStageCompute, {8, 8, 1}}};
constexpr gfx::BindingDesc kFillBindings[] = {
    {0, 0, gfx::BindingType::Uniform, gfx::kStageCompute, true, sizeof(FillBlock)},
    {1, 0, gfx::BindingType::StorageTexture, gfx::kStageCompute, false, 0, gfx::Format::RGBA8Unorm}};
constexpr gfx::ProgramDesc kFill = {"caliper.selftest.fill", kFillEntries, 1, kFillBindings, 2, kFillWgsl,
                                    nullptr, nullptr, 0};

// A draw that reads texel (0, 0) of a texture, over one triangle that covers its target.
constexpr const char* kSampleWgsl = R"wgsl(
@group(0) @binding(0) var img: texture_2d<f32>;

@vertex
fn vs_cover(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(vec2<f32>(f32(i & 1u), f32(i >> 1u)) * 4.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_load() -> @location(0) vec4<f32> { return textureLoad(img, vec2<i32>(0, 0), 0); }
)wgsl";
constexpr gfx::EntryDesc kSampleEntries[] = {{"vs_cover", gfx::kStageVertex, {1, 1, 1}},
                                             {"fs_load", gfx::kStageFragment, {1, 1, 1}}};
constexpr gfx::BindingDesc kSampleBindings[] = {
    {0, 0, gfx::BindingType::Texture, gfx::kStageFragment, false, 0, gfx::Format::RGBA8Unorm}};
constexpr gfx::ProgramDesc kSample = {"caliper.selftest.sample", kSampleEntries, 2, kSampleBindings, 1, kSampleWgsl,
                                      nullptr, nullptr, 0};

// A copy from one storage buffer to another; bound to one buffer twice, WebGPU refuses it.
constexpr const char* kAliasWgsl = R"wgsl(
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(1)
fn cs_copy(@builtin(global_invocation_id) gid: vec3<u32>) { dst[gid.x] = src[gid.x]; }
)wgsl";
constexpr gfx::EntryDesc kAliasEntries[] = {{"cs_copy", gfx::kStageCompute, {1, 1, 1}}};
constexpr gfx::BindingDesc kAliasBindings[] = {
    {0, 0, gfx::BindingType::Storage, gfx::kStageCompute, false, 4},
    {0, 1, gfx::BindingType::StorageReadWrite, gfx::kStageCompute, false, 4}};
constexpr gfx::ProgramDesc kAlias = {"caliper.selftest.alias", kAliasEntries, 1, kAliasBindings, 2, kAliasWgsl,
                                     nullptr, nullptr, 0};

constexpr uint32_t W = 64, H = 64, FW = 16, FH = 16, CW = 8, CH = 8, kCopyPitch = 256;
constexpr uint32_t kClear = 0xff201008u;   // r 8, g 16, b 32, a 255
constexpr float kHalf = 3.0f / 32.0f;      // 3 pixels of 64
const DrawBlock kQuad{0xff3366ccu, 7u, 0.5f, 0u};
constexpr uint32_t kPick2 = 9, kPick3 = 11;

struct Inst { int cx, cy; float z; uint32_t rgba; };
const Inst kInst[6] = {
    {8, 52, 0.25f, 0xff0000ffu}, {24, 52, 0.25f, 0xff00ff00u}, {40, 30, 0.75f, 0xffff0000u},
    {48, 40, 0.25f, 0xff00ffffu}, {8, 52, 0.9f, 0x00ffffffu}, {56, 8, 0.9f, 0xff808080u},
};
constexpr uint16_t kScissor3[4] = {0, 0, 56, 64};

uint8_t copy_byte(uint32_t x, uint32_t y, uint32_t c) { return uint8_t((x * 31 + y * 7 + c * 50) & 0xff); }

enum { kColor, kId, kDepth, kFillTex, kCopyTex, kReads };

struct State {
    bool running = false;
    gfx::Program draw_prog, fill_prog, sample_prog, alias_prog;
    gfx::Pipeline pa, pb, pc, pf, ps, palias;
    gfx::BindGroup draw_g0, fill_g0, fill_g1, sample_g0, alias_g0;
    gfx::Buffer verts, idx, corners, centers, half, zs, colors, copy_src, alias;
    gfx::Texture color, id, depth, fill, copy;
    gfx::Readback rb[kReads];
    bool got[kReads] = {};
    std::vector<uint8_t> out[kReads];
    std::string import_path;
    uint64_t import_bytes = 0;
    bool own_target_refused = false;
    gfx::Serial alias_serial = 0;
    int waits = 0;   // polls after the readbacks, while the expected submit error has not arrived
};
State S;
std::vector<SelftestImage> g_images;

template <class H> void drop(gfx::Backend& b, H h) {
    if (h) b.release(h);
}

void release_all(gfx::Backend& b) {
    for (auto& r : S.rb) drop(b, r);
    drop(b, S.pa);
    drop(b, S.pb);
    drop(b, S.pc);
    drop(b, S.pf);
    drop(b, S.ps);
    drop(b, S.palias);
    drop(b, S.draw_g0);
    drop(b, S.fill_g0);
    drop(b, S.fill_g1);
    drop(b, S.sample_g0);
    drop(b, S.alias_g0);
    for (gfx::Buffer x : {S.verts, S.idx, S.corners, S.centers, S.half, S.zs, S.colors, S.copy_src, S.alias}) drop(b, x);
    for (gfx::Texture x : {S.color, S.id, S.depth, S.fill, S.copy}) drop(b, x);
    for (gfx::Program x : {S.draw_prog, S.fill_prog, S.sample_prog, S.alias_prog}) drop(b, x);
    S = State{};
}

gfx::Buffer make_buffer(gfx::Backend& b, const void* data, uint64_t size, uint32_t usage, const char* label) {
    gfx::BufferDesc d;
    d.size = size;
    d.usage = usage | gfx::kCopyDst;
    d.label = label;
    gfx::Buffer x = b.create_buffer(d);
    if (x && data) b.write_buffer(x, 0, data, size);
    return x;
}

gfx::Texture make_texture(gfx::Backend& b, uint32_t w, uint32_t h, gfx::Format f, uint32_t usage, const char* label) {
    gfx::TextureDesc d;
    d.width = w;
    d.height = h;
    d.format = f;
    d.usage = usage;
    d.label = label;
    return b.create_texture(d);
}

gfx::RenderPipelineDesc inst_pipeline(gfx::Program p) {
    gfx::RenderPipelineDesc d;
    d.program = p;
    d.vertex_entry = "vs_inst";
    d.fragment_entry = "fs_pick";
    d.vertex_layout_count = 5;
    const struct { uint32_t stride; gfx::Step step; gfx::VertexFormat f; } L[5] = {
        {8, gfx::Step::Vertex, gfx::VertexFormat::F32x2},     // corner
        {8, gfx::Step::Instance, gfx::VertexFormat::F32x2},   // centre
        {4, gfx::Step::Instance, gfx::VertexFormat::U32},     // colour
        {0, gfx::Step::Instance, gfx::VertexFormat::F32},     // half size: constant
        {4, gfx::Step::Instance, gfx::VertexFormat::F32},     // depth
    };
    for (uint32_t i = 0; i < 5; ++i) {
        d.vertex_layouts[i].stride = L[i].stride;
        d.vertex_layouts[i].step = L[i].step;
        d.vertex_layouts[i].attribute_count = 1;
        d.vertex_layouts[i].attributes[0] = {i, L[i].f, 0};
    }
    d.id_target = true;
    d.depth = true;
    d.depth_test = d.depth_write = true;
    d.label = "caliper.selftest.instanced";
    return d;
}

// ---- what the CPU expects --------------------------------------------------------------------------

struct Expect {
    std::vector<uint32_t> color = std::vector<uint32_t>(W * H, kClear);
    std::vector<uint32_t> pick = std::vector<uint32_t>(W * H, 0), item = std::vector<uint32_t>(W * H, 0);
    std::vector<float> depth = std::vector<float>(W * H, 1.0f);
};

void rect(Expect& e, int x0, int y0, int x1, int y1, const uint16_t sc[4], uint32_t rgba, uint32_t pick,
          uint32_t item, float z, bool test, bool write, bool alpha, bool id = true) {
    const int sx0 = sc[0], sy0 = sc[1], sx1 = sc[0] + sc[2], sy1 = sc[1] + sc[3];
    for (int y = std::max({y0, sy0, 0}); y < std::min({y1, sy1, int(H)}); ++y)
        for (int x = std::max({x0, sx0, 0}); x < std::min({x1, sx1, int(W)}); ++x) {
            const size_t i = size_t(y) * W + x;
            if (test && !(z <= e.depth[i])) continue;
            // Alpha blending of a fully transparent or fully opaque source is exact.
            if (!alpha || (rgba >> 24) == 255u) e.color[i] = rgba;
            if (id) {
                e.pick[i] = pick;
                e.item[i] = item;
            }
            if (write) e.depth[i] = z;
        }
}

Expect expected() {
    Expect e;
    const uint16_t full[4] = {0, 0, 0xffff, 0xffff};
    rect(e, 16, 8, 48, 40, full, kQuad.color, kQuad.pick, 0, kQuad.depth, true, true, false);
    for (uint32_t k = 0; k < 4; ++k)
        rect(e, kInst[k].cx - 3, kInst[k].cy - 3, kInst[k].cx + 3, kInst[k].cy + 3, full, kInst[k].rgba, kPick2, k,
             kInst[k].z, true, true, false);
    for (uint32_t k = 4; k < 6; ++k)
        rect(e, kInst[k].cx - 3, kInst[k].cy - 3, kInst[k].cx + 3, kInst[k].cy + 3, kScissor3, kInst[k].rgba, kPick3,
             k, kInst[k].z, false, false, true, false);
    return e;
}

}  // namespace

bool gfx_selftest_running() { return S.running; }
const std::vector<SelftestImage>& gfx_selftest_images() { return g_images; }

bool gfx_selftest_start(gfx::Backend& b, std::string* why) {
    if (S.running) {
        *why = "already running";
        return false;
    }
    g_images.clear();
    S = State{};
    auto fail = [&](const char* what) {
        *why = std::string(what) + ": " + b.last_error();
        release_all(b);
        return false;
    };
    S.draw_prog = b.create_program(kDraw);
    S.fill_prog = b.create_program(kFill);
    if (!S.draw_prog || !S.fill_prog) return fail("programs");

    // Targets and textures.
    S.color = make_texture(b, W, H, gfx::Format::RGBA8Unorm, gfx::kRenderTarget | gfx::kCopySrcTex | gfx::kSampled,
                           "selftest.color");
    S.id = make_texture(b, W, H, gfx::Format::RG32Uint, gfx::kRenderTarget | gfx::kCopySrcTex, "selftest.id");
    S.depth = make_texture(b, W, H, gfx::Format::Depth32Float, gfx::kRenderTarget | gfx::kCopySrcTex, "selftest.depth");
    S.fill = make_texture(b, FW, FH, gfx::Format::RGBA8Unorm, gfx::kStorageTex | gfx::kCopySrcTex, "selftest.fill");
    S.copy = make_texture(b, CW, CH, gfx::Format::RGBA8Unorm, gfx::kCopyDstTex | gfx::kCopySrcTex, "selftest.copy");
    if (!S.color || !S.id || !S.depth || !S.fill || !S.copy) return fail("textures");

    // Streams. The quad's vertices start with a dummy that the base vertex skips; its indices start
    // with two that the index offset skips.
    const float verts[10] = {9, 9, -0.5f, 0.75f, 0.5f, 0.75f, -0.5f, -0.25f, 0.5f, -0.25f};
    const uint32_t idx[8] = {77, 77, 0, 1, 2, 2, 1, 3};
    const float corners[12] = {-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1};
    float centers[13] = {0};   // 4 bytes of padding, then (x, y) per instance in clip space
    float zs[6];
    uint32_t colors[6];
    for (int k = 0; k < 6; ++k) {
        centers[1 + 2 * k] = float(kInst[k].cx) / 32.0f - 1.0f;
        centers[2 + 2 * k] = 1.0f - float(kInst[k].cy) / 32.0f;
        zs[k] = kInst[k].z;
        colors[k] = kInst[k].rgba;
    }
    const float half[4] = {kHalf, 0, 0, 0};
    S.verts = make_buffer(b, verts, sizeof verts, gfx::kVertex, "selftest.verts");
    S.idx = make_buffer(b, idx, sizeof idx, gfx::kIndex, "selftest.indices");
    S.corners = make_buffer(b, corners, sizeof corners, gfx::kVertex, "selftest.corners");
    S.centers = make_buffer(b, centers, sizeof centers, gfx::kVertex, "selftest.centers");
    S.half = make_buffer(b, half, sizeof half, gfx::kVertex, "selftest.half");
    S.zs = make_buffer(b, zs, sizeof zs, gfx::kVertex, "selftest.z");
    gfx::ExternalBuffer ext;
    ext.kind = gfx::ExternalKind::HostMemory;
    ext.handle = colors;
    ext.size = sizeof colors;
    ext.usage = gfx::kVertex;
    ext.label = "selftest.colors";
    const gfx::ImportResult imp = b.import_buffer(ext);
    S.colors = imp.buffer;
    S.import_path = imp.path;
    S.import_bytes = imp.bytes_copied;
    std::vector<uint8_t> copy_bytes(size_t(kCopyPitch) * CH, 0xee);
    for (uint32_t y = 0; y < CH; ++y)
        for (uint32_t x = 0; x < CW; ++x)
            for (uint32_t c = 0; c < 4; ++c) copy_bytes[size_t(y) * kCopyPitch + x * 4 + c] = copy_byte(x, y, c);
    S.copy_src = make_buffer(b, copy_bytes.data(), copy_bytes.size(), gfx::kCopySrc, "selftest.copy");
    if (!S.verts || !S.idx || !S.corners || !S.centers || !S.half || !S.zs || !S.colors || !S.copy_src)
        return fail("buffers");

    // Pipelines and bind groups.
    gfx::RenderPipelineDesc qa;
    qa.program = S.draw_prog;
    qa.vertex_entry = "vs_quad";
    qa.fragment_entry = "fs_pick";
    qa.vertex_layout_count = 1;
    qa.vertex_layouts[0].stride = 8;
    qa.vertex_layouts[0].attribute_count = 1;
    qa.vertex_layouts[0].attributes[0] = {0, gfx::VertexFormat::F32x2, 0};
    qa.id_target = true;
    qa.depth = qa.depth_test = qa.depth_write = true;
    qa.label = "caliper.selftest.quad";
    S.pa = b.create_render_pipeline(qa);
    gfx::RenderPipelineDesc ib = inst_pipeline(S.draw_prog);
    S.pb = b.create_render_pipeline(ib);
    gfx::RenderPipelineDesc ic = inst_pipeline(S.draw_prog);
    ic.fragment_entry = "fs_color";
    ic.id_target = false;
    ic.blend = gfx::Blend::Alpha;
    ic.depth_test = ic.depth_write = false;
    ic.label = "caliper.selftest.blended";
    S.pc = b.create_render_pipeline(ic);
    gfx::ComputePipelineDesc cd;
    cd.program = S.fill_prog;
    cd.entry = "cs_fill";
    cd.label = "caliper.selftest.fill";
    S.pf = b.create_compute_pipeline(cd);
    if (!S.pa || !S.pb || !S.pc || !S.pf) return fail("pipelines");
    gfx::BindGroupDesc g;
    g.group = 0;
    g.entry_count = 1;
    g.entries[0].binding = 0;
    g.entries[0].buffer = gfx::kArena;
    g.program = S.draw_prog;
    S.draw_g0 = b.create_bind_group(g);
    g.program = S.fill_prog;
    S.fill_g0 = b.create_bind_group(g);
    gfx::BindGroupDesc g1;
    g1.program = S.fill_prog;
    g1.group = 1;
    g1.entry_count = 1;
    g1.entries[0].binding = 0;
    g1.entries[0].texture = S.fill;
    S.fill_g1 = b.create_bind_group(g1);
    if (!S.draw_g0 || !S.fill_g0 || !S.fill_g1) return fail("bind groups");
    // The draw program goes before the frame.
    b.release(S.draw_prog);
    S.draw_prog = {};

    // The frame.
    gfx::Frame f;
    const FillBlock left{50, 0, 0, 0}, right{200, 8, 0, 0};
    const uint32_t o0 = f.uniform(&left, sizeof left), o1 = f.uniform(&right, sizeof right);
    const gfx::BindGroup fg[2] = {S.fill_g0, S.fill_g1};
    const gfx::Texture writes[1] = {S.fill};
    f.dispatch(S.pf, fg, gfx::View<uint32_t>(&o0, 1), 1, 2, 1, writes);
    f.dispatch(S.pf, fg, gfx::View<uint32_t>(&o1, 1), 1, 2, 1, writes);
    f.copy_buffer_to_texture(S.copy_src, 0, kCopyPitch, S.copy);
    gfx::RenderTarget tg;
    tg.color = S.color;
    tg.id = S.id;
    tg.depth = S.depth;
    tg.clear_rgba = kClear;
    gfx::DrawList list(tg);
    gfx::DrawRecord r1;
    r1.pipeline = S.pa;
    r1.groups[0] = S.draw_g0;
    r1.dynamic_offsets[0] = list.uniform(&kQuad, sizeof kQuad);
    r1.vertex[0] = {S.verts, 0};
    r1.index_buffer = S.idx;
    r1.index_offset = 8;
    r1.count = 6;
    r1.first_vertex = 1;
    r1.pick_id = kQuad.pick;
    list.draw(r1);
    gfx::DrawRecord r2;
    r2.pipeline = S.pb;
    r2.groups[0] = S.draw_g0;
    const DrawBlock b2{0, kPick2, 0, 0}, b3{0, kPick3, 0, 0};
    r2.dynamic_offsets[0] = list.uniform(&b2, sizeof b2);
    r2.vertex[0] = {S.corners, 0};
    r2.vertex[1] = {S.centers, 4};
    r2.vertex[2] = {S.colors, 0};
    r2.vertex[3] = {S.half, 0};
    r2.vertex[4] = {S.zs, 0};
    r2.count = 6;
    r2.instance_count = 4;
    r2.pick_id = kPick2;
    list.draw(r2);
    f.render(std::move(list));
    gfx::RenderTarget tg2 = tg;
    tg2.id = {};
    tg2.clear = false;
    gfx::DrawList blended(tg2);
    gfx::DrawRecord r3 = r2;
    r3.pipeline = S.pc;
    r3.dynamic_offsets[0] = blended.uniform(&b3, sizeof b3);
    r3.instance_count = 2;
    r3.first_instance = 4;
    r3.pick_id = kPick3;
    std::copy(kScissor3, kScissor3 + 4, r3.scissor);
    blended.draw(r3);
    f.render(std::move(blended));
    if (!b.submit(std::move(f))) return fail("submit");

    // A draw that samples the colour target it renders into: the core refuses the frame.
    S.sample_prog = b.create_program(kSample);
    gfx::RenderPipelineDesc sd;
    sd.program = S.sample_prog;
    sd.vertex_entry = "vs_cover";
    sd.fragment_entry = "fs_load";
    sd.label = "caliper.selftest.sample";
    S.ps = b.create_render_pipeline(sd);
    gfx::BindGroupDesc sg;
    sg.program = S.sample_prog;
    sg.group = 0;
    sg.entry_count = 1;
    sg.entries[0].binding = 0;
    sg.entries[0].texture = S.color;
    S.sample_g0 = b.create_bind_group(sg);
    if (!S.ps || !S.sample_g0) return fail("the sampling draw");
    {
        gfx::RenderTarget st;
        st.color = S.color;
        st.clear = false;
        gfx::DrawList own(st);
        gfx::DrawRecord r;
        r.pipeline = S.ps;
        r.groups[0] = S.sample_g0;
        r.count = 3;
        own.draw(r);
        gfx::Frame sf;
        sf.render(std::move(own));
        S.own_target_refused = b.submit(std::move(sf)) == 0 &&
                               b.last_error().find("binds the pass's own colour target") != std::string::npos;
    }
    // A dispatch that binds one buffer as read-only and read-write storage: WebGPU refuses it.
    S.alias_prog = b.create_program(kAlias);
    gfx::ComputePipelineDesc ad;
    ad.program = S.alias_prog;
    ad.entry = "cs_copy";
    ad.label = "caliper.selftest.alias";
    S.palias = b.create_compute_pipeline(ad);
    S.alias = make_buffer(b, nullptr, 256, gfx::kStorage, "selftest.alias");
    gfx::BindGroupDesc ag;
    ag.program = S.alias_prog;
    ag.group = 0;
    ag.entry_count = 2;
    ag.entries[0].binding = 0;
    ag.entries[0].buffer = S.alias;
    ag.entries[1].binding = 1;
    ag.entries[1].buffer = S.alias;
    S.alias_g0 = b.create_bind_group(ag);
    if (!S.palias || !S.alias || !S.alias_g0) return fail("the aliasing dispatch");
    {
        gfx::Frame af;
        af.dispatch(S.palias, gfx::View<gfx::BindGroup>(&S.alias_g0, 1), {}, 4, 1, 1);
        S.alias_serial = b.submit(std::move(af));
        if (!S.alias_serial) return fail("the aliasing dispatch's submit");
    }

    S.rb[kColor] = b.read_texture(S.color, 0, 0, W, H);
    S.rb[kId] = b.read_texture(S.id, 0, 0, W, H);
    S.rb[kDepth] = b.read_texture(S.depth, 0, 0, W, H);
    S.rb[kFillTex] = b.read_texture(S.fill, 0, 0, FW, FH);
    S.rb[kCopyTex] = b.read_texture(S.copy, 0, 0, CW, CH);
    for (const auto& r : S.rb)
        if (!r) return fail("readback");
    S.running = true;
    return true;
}

bool gfx_selftest_poll(gfx::Backend& b, std::string* report) {
    if (!S.running) return false;
    for (int i = 0; i < kReads; ++i)
        if (!S.got[i] && b.take_readback(S.rb[i], &S.out[i])) {
            S.got[i] = true;
            S.rb[i] = {};
        }
    for (bool g : S.got)
        if (!g) return false;
    // The error of the aliasing dispatch, from the error scope around its submit, and the others.
    const std::string alias_error = "submit " + std::to_string(S.alias_serial) + ":";
    std::string errors;
    bool reported = false, compat = false;
    if (auto* wb = dynamic_cast<gfx::WebGpuBackend*>(&b)) {
        compat = wb->compatibility();
        for (const auto& m : wb->device_errors()) {
            if (m.compare(0, alias_error.size(), alias_error) == 0) reported = true;
            else errors += (errors.empty() ? "" : " | ") + m;
        }
    }
    if (!reported && ++S.waits < 240) return false;   // a frame more for the scope's callback

    const Expect e = expected();
    auto u32 = [](const std::vector<uint8_t>& v, size_t i) {
        uint32_t x;
        std::memcpy(&x, v.data() + 4 * i, 4);
        return x;
    };
    long long bad_color = -1, bad_id = -1, bad_depth = -1, bad_fill = -1, bad_copy = -1;
    if (S.out[kColor].size() == size_t(W) * H * 4) {
        bad_color = 0;
        for (size_t i = 0; i < size_t(W) * H; ++i) bad_color += u32(S.out[kColor], i) != e.color[i];
    }
    if (S.out[kId].size() == size_t(W) * H * 8) {
        bad_id = 0;
        for (size_t i = 0; i < size_t(W) * H; ++i)
            bad_id += u32(S.out[kId], 2 * i) != e.pick[i] || u32(S.out[kId], 2 * i + 1) != e.item[i];
    }
    if (S.out[kDepth].size() == size_t(W) * H * 4) {
        bad_depth = 0;
        for (size_t i = 0; i < size_t(W) * H; ++i) {
            float z;
            std::memcpy(&z, S.out[kDepth].data() + 4 * i, 4);
            bad_depth += std::memcmp(&z, &e.depth[i], 4) != 0;
        }
    }
    if (S.out[kFillTex].size() == size_t(FW) * FH * 4) {
        bad_fill = 0;
        for (uint32_t y = 0; y < FH; ++y)
            for (uint32_t x = 0; x < FW; ++x) {
                const uint32_t want = (x * 16) | ((y * 16) << 8) | ((x < 8 ? 50u : 200u) << 16) | (255u << 24);
                bad_fill += u32(S.out[kFillTex], size_t(y) * FW + x) != want;
            }
    }
    if (S.out[kCopyTex].size() == size_t(CW) * CH * 4) {
        bad_copy = 0;
        for (uint32_t y = 0; y < CH; ++y)
            for (uint32_t x = 0; x < CW; ++x)
                for (uint32_t c = 0; c < 4; ++c)
                    bad_copy += S.out[kCopyTex][(size_t(y) * CW + x) * 4 + c] != copy_byte(x, y, c);
    }
    const bool ok = bad_color == 0 && bad_id == 0 && bad_depth == 0 && bad_fill == 0 && bad_copy == 0 &&
                    S.import_path == std::string("copied-from-host") && S.import_bytes == 24 &&
                    S.own_target_refused && reported && errors.empty();
    char buf[640];
    std::snprintf(buf, sizeof buf,
                  "%s color=%lld id=%lld depth=%lld fill=%lld copy=%lld import=%s:%llu own-target=%s submit-error=%s "
                  "level=%s",
                  ok ? "ok" : "fail", bad_color, bad_id, bad_depth, bad_fill, bad_copy, S.import_path.c_str(),
                  (unsigned long long)S.import_bytes, S.own_target_refused ? "refused" : "submitted",
                  reported ? "reported" : "lost", compat ? "compatibility" : "core");
    *report = buf;
    if (!errors.empty()) *report += " errors=" + errors;

    // The pictures, for the captures: the colour target, the ID target as colours (red the pick id,
    // green the item), and the two textures.
    g_images.push_back({"layer-draws", W, H, S.out[kColor]});
    SelftestImage ids{"layer-ids", W, H, std::vector<uint8_t>(size_t(W) * H * 4, 0)};
    if (S.out[kId].size() == size_t(W) * H * 8)
        for (size_t i = 0; i < size_t(W) * H; ++i) {
            const uint32_t pick = u32(S.out[kId], 2 * i), item = u32(S.out[kId], 2 * i + 1);
            ids.rgba[4 * i] = uint8_t(pick * 20);
            ids.rgba[4 * i + 1] = uint8_t(pick ? 60 + item * 32 : 0);
            ids.rgba[4 * i + 3] = 255;
        }
    g_images.push_back(std::move(ids));
    g_images.push_back({"layer-compute", FW, FH, S.out[kFillTex]});
    g_images.push_back({"layer-copy", CW, CH, S.out[kCopyTex]});
    release_all(b);
    return true;
}

}  // namespace caliper_web
