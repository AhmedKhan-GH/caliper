// Rows of the graphics suite for the graphics layer (src/host/gfx) on Metal, on the renderer's
// own device and queue (docs/applets/graphics-layer.md). Each row draws through the layer,
// reads the targets back through the layer's asynchronous readback, and compares them byte
// for byte with a CPU reference built the way the suite's geometry rows build theirs (the
// clear everywhere, each quad's colour on the pixels it covers, quads edged on pixel
// boundaries). The test program is hand-written MSL that lives only here; production
// programs come from the shader generator. The pipeline checks cover each generated program.
// Compiled as Objective-C++ with ARC on Apple.
//
// CALIPER_GFX_CAPTURE_DIR=<folder> writes PNG captures of the rows' readbacks there.
// CALIPER_GFX_LAYER_BENCH=1 runs the encode-time row (1,000 draws a frame, recording backend
// and Metal); without it that row returns at once.
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>

#include "gfx/metal_backend.h"
#include "gfx/recording_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "generated_pipelines.h"
#include "renderer/host_renderer.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// The suite's Metal environment (gfx_main.cpp): its renderer, or nullptr without a device.
caliper_host::HostRenderer* gfx_metal_renderer();

using namespace caliper_gfx;

namespace {

const char* kMsl = R"metal(
#include <metal_stdlib>
using namespace metal;

struct Quad { float4 rect; float z; uint rgba; uint pick_id; uint item; };   // 32 bytes

struct VIn  { float2 corner [[attribute(0)]]; };
struct VOut { float4 pos [[position]]; float4 color; uint2 pick [[flat]]; };
struct FOut { float4 color [[color(0)]]; uint2 id [[color(1)]]; };

static float4 unpack_rgba(uint c) {
    return float4(float(c & 0xffu), float((c >> 8) & 0xffu), float((c >> 16) & 0xffu), float(c >> 24)) / 255.0f;
}

vertex VOut vs_quad(VIn in [[stage_in]], constant Quad& q [[buffer(0)]]) {
    VOut o;
    o.pos = float4(q.rect.xy + (q.rect.zw - q.rect.xy) * in.corner, q.z, 1.0f);
    o.color = unpack_rgba(q.rgba);
    o.pick = uint2(q.pick_id, q.item);
    return o;
}

fragment FOut fs_quad(VOut in [[stage_in]]) {
    FOut f;
    f.color = in.color;
    f.id = in.pick;
    return f;
}

fragment float4 fs_sample(VOut in [[stage_in]], texture2d<float> img [[texture(0)]]) {
    return img.read(uint2(0, 0));
}
)metal";

struct Quad { float rect[4]; float z; uint32_t rgba, pick_id, item; };
static_assert(sizeof(Quad) == 32, "the MSL block");

const EntryDesc kEntries[] = {
    {"vs_quad", kStageVertex, {1, 1, 1}},
    {"fs_quad", kStageFragment, {1, 1, 1}},
};
const BindingDesc kBindings[] = {
    {1, 0, BindingType::Uniform, kStageVertex, true, 32, Format::RGBA8Unorm, 0},
};

ProgramDesc test_program() {
    ProgramDesc d;
    d.label = "gfx-layer-test";
    d.entries = kEntries;
    d.entry_count = 2;
    d.bindings = kBindings;
    d.binding_count = 1;
    d.msl = kMsl;
    return d;
}

// The same quad with a fragment entry that reads texel (0, 0) of the image in group 2.
const EntryDesc kSampleEntries[] = {
    {"vs_quad", kStageVertex, {1, 1, 1}},
    {"fs_sample", kStageFragment, {1, 1, 1}},
};
const BindingDesc kSampleBindings[] = {
    {1, 0, BindingType::Uniform, kStageVertex, true, 32, Format::RGBA8Unorm, 0},
    {2, 0, BindingType::Texture, kStageFragment, false, 0, Format::RGBA8Unorm, 0},
};

ProgramDesc sample_program() {
    ProgramDesc d;
    d.label = "gfx-layer-sample";
    d.entries = kSampleEntries;
    d.entry_count = 2;
    d.bindings = kSampleBindings;
    d.binding_count = 2;
    d.msl = kMsl;
    return d;
}

// Six corners of a triangle-list quad, then the four distinct corners for indexed draws.
const float kCorners[20] = {0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1};
const uint32_t kQuadIndices[6] = {0, 1, 2, 1, 3, 2};

// The NDC rectangle whose edges are the pixel boundaries x0..x1, y0..y1 of a WxH target.
void pixel_rect(int x0, int y0, int x1, int y1, int W, int H, float out[4]) {
    out[0] = 2.0f * x0 / W - 1.0f;
    out[1] = 1.0f - 2.0f * y1 / H;
    out[2] = 2.0f * x1 / W - 1.0f;
    out[3] = 1.0f - 2.0f * y0 / H;
}

// The CPU reference: colour, ID and depth targets painted rectangle by rectangle.
struct Reference {
    int W, H;
    std::vector<uint8_t> color;
    std::vector<uint32_t> id;
    std::vector<float> depth;
    Reference(int w, int h, uint32_t clear) : W(w), H(h), color(size_t(w) * h * 4), id(size_t(w) * h * 2, 0),
                                               depth(size_t(w) * h, 1.0f) {
        for (int i = 0; i < w * h; ++i) std::memcpy(&color[size_t(i) * 4], &clear, 4);
    }
    // Opaque or additive, with a less-or-equal depth test and write when z >= 0.
    void paint(int x0, int y0, int x1, int y1, uint32_t rgba, uint32_t pick, uint32_t item, float z, bool additive) {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const size_t i = size_t(y) * W + x;
                if (z >= 0 && !(z <= depth[i])) continue;
                if (z >= 0) depth[i] = z;
                for (int c = 0; c < 4; ++c) {
                    const uint32_t src = (rgba >> (8 * c)) & 0xff;
                    color[i * 4 + c] = uint8_t(additive ? std::min<uint32_t>(255, color[i * 4 + c] + src) : src);
                }
                id[i * 2] = pick;
                id[i * 2 + 1] = item;
            }
    }
};

std::vector<uint8_t> read_back(Backend& gfx, Texture t, uint32_t w, uint32_t h) {
    Readback r = gfx.read_texture(t, 0, 0, w, h);
    REQUIRE(bool(r));
    std::vector<uint8_t> out;
    REQUIRE(gfx.wait(gfx.last_use(t)));
    REQUIRE(gfx.take_readback(r, &out));
    return out;
}

size_t first_diff(const std::vector<uint8_t>& a, const void* b, size_t n) {
    if (a.size() != n) return 0;
    const auto* p = static_cast<const uint8_t*>(b);
    for (size_t i = 0; i < n; ++i)
        if (a[i] != p[i]) return i;
    return SIZE_MAX;
}

void capture(const char* name, const std::vector<uint8_t>& rgba, int w, int h) {
    const char* dir = std::getenv("CALIPER_GFX_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    // Scaled up eight times with nearest sampling, so each pixel stays a visible square.
    const int s = 8;
    std::vector<uint8_t> big(size_t(w) * s * h * s * 4);
    for (int y = 0; y < h * s; ++y)
        for (int x = 0; x < w * s; ++x)
            std::memcpy(&big[(size_t(y) * w * s + x) * 4], &rgba[(size_t(y / s) * w + x / s) * 4], 4);
    CHECK(stbi_write_png(path.c_str(), w * s, h * s, 4, big.data(), w * s * 4) != 0);
    MESSAGE("capture written: ", path);
}

std::unique_ptr<Backend> metal_layer() {
    caliper_host::HostRenderer* r = gfx_metal_renderer();
    if (r == nullptr) return nullptr;
    return make_metal_backend(r->native_device(), r->native_queue());
}

}  // namespace

TEST_CASE("gfx/Metal layer: a quad drawn through the layer reads back byte-exact") {
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    CHECK(std::string(gfx->caps().backend) == "metal");
    CHECK(gfx->caps().has(kCapImportMetalBuffer));
    const int W = 32, H = 32;
    const uint32_t clear = 0xff000000u;
    Texture color = gfx->create_texture({W, H, Format::RGBA8Unorm, kRenderTarget | kSampled | kCopySrcTex, "color"});
    Texture ids = gfx->create_texture({W, H, Format::RG32Uint, kRenderTarget | kCopySrcTex, "id"});
    Texture depth = gfx->create_texture({W, H, Format::Depth32Float, kRenderTarget | kCopySrcTex, "depth"});
    Program program = gfx->create_program(test_program());
    REQUIRE_MESSAGE(bool(program), gfx->last_error());
    RenderPipelineDesc pd;
    pd.program = program;
    pd.vertex_entry = "vs_quad";
    pd.fragment_entry = "fs_quad";
    pd.vertex_layout_count = 1;
    pd.vertex_layouts[0].stride = 8;
    pd.vertex_layouts[0].attribute_count = 1;
    pd.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x2, 0};
    pd.id_target = true;
    pd.depth = pd.depth_test = pd.depth_write = true;
    Pipeline opaque = gfx->create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(opaque), gfx->last_error());
    pd.blend = Blend::Additive;
    Pipeline additive = gfx->create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(additive), gfx->last_error());
    BindGroupDesc bd;
    bd.program = program;
    bd.group = 1;
    bd.entry_count = 1;
    bd.entries[0].binding = 0;
    bd.entries[0].buffer = kArena;
    BindGroup group = gfx->create_bind_group(bd);
    REQUIRE_MESSAGE(bool(group), gfx->last_error());
    Buffer corners = gfx->create_buffer({sizeof kCorners, kVertex | kCopyDst, false, "corners"});
    Buffer indices = gfx->create_buffer({sizeof kQuadIndices, kIndex | kCopyDst, false, "indices"});
    REQUIRE(gfx->write_buffer(corners, 0, kCorners, sizeof kCorners));
    REQUIRE(gfx->write_buffer(indices, 0, kQuadIndices, sizeof kQuadIndices));
    gfx->take_stats();

    RenderTarget tg;
    tg.color = color;
    tg.id = ids;
    tg.depth = depth;
    tg.clear_rgba = clear;
    DrawList list(tg);
    auto quad = [&](Pipeline p, int x0, int y0, int x1, int y1, float z, uint32_t rgba, uint32_t pick, uint32_t item) {
        Quad q{};
        pixel_rect(x0, y0, x1, y1, W, H, q.rect);
        q.z = z;
        q.rgba = rgba;
        q.pick_id = pick;
        q.item = item;
        DrawRecord r;
        r.pipeline = p;
        r.groups[1] = group;
        r.dynamic_offsets[0] = list.uniform(&q, sizeof q);
        r.vertex[0] = {corners, 0};
        r.count = 6;
        r.pick_id = pick;
        return r;
    };
    // A: opaque, six vertices. B: indexed with a base vertex, behind A where they overlap.
    // C: additive, in front of both.
    list.draw(quad(opaque, 4, 6, 20, 22, 0.5f, 0xffc04020u, 3, 0));
    DrawRecord b = quad(opaque, 12, 14, 28, 30, 0.75f, 0xff30a050u, 4, 1);
    b.index_buffer = indices;
    b.first_vertex = 6;
    list.draw(b);
    list.draw(quad(additive, 16, 2, 24, 10, 0.25f, 0x40101010u, 5, 2));
    Frame f;
    f.render(std::move(list));
    const Serial s = gfx->submit(std::move(f));
    REQUIRE_MESSAGE(s != 0, gfx->last_error());

    Reference ref(W, H, clear);
    ref.paint(4, 6, 20, 22, 0xffc04020u, 3, 0, 0.5f, false);
    ref.paint(12, 14, 28, 30, 0xff30a050u, 4, 1, 0.75f, false);
    ref.paint(16, 2, 24, 10, 0x40101010u, 5, 2, 0.25f, true);

    const std::vector<uint8_t> got = read_back(*gfx, color, W, H);
    const size_t dc = first_diff(got, ref.color.data(), ref.color.size());
    CHECK_MESSAGE(dc == SIZE_MAX, "colour differs at byte ", dc);
    const std::vector<uint8_t> got_id = read_back(*gfx, ids, W, H);
    const size_t di = first_diff(got_id, ref.id.data(), ref.id.size() * 4);
    CHECK_MESSAGE(di == SIZE_MAX, "ID target differs at byte ", di);
    const std::vector<uint8_t> got_depth = read_back(*gfx, depth, W, H);
    const size_t dd = first_diff(got_depth, ref.depth.data(), ref.depth.size() * 4);
    CHECK_MESSAGE(dd == SIZE_MAX, "depth differs at byte ", dd);
    CHECK(gfx->completed(s));
    CHECK(gfx->imgui_texture_id(color) != 0);
    const Stats st = gfx->take_stats();
    CHECK(st.submits == 1);
    CHECK(st.draws == 3);
    CHECK(st.copies == 0);
    CHECK(st.gpu_ms >= 0);   // GPU time from the command buffer's start and end
    capture("layer_quad.png", got, W, H);

    // The same frame again without a clear keeps what the targets hold (load, not clear).
    RenderTarget keep = tg;
    keep.clear = false;
    DrawList again(keep);
    Frame f2;
    f2.render(std::move(again));
    REQUIRE(gfx->submit(std::move(f2)) != 0);
    CHECK(read_back(*gfx, color, W, H) == got);
}

TEST_CASE("gfx/Metal layer: an instanced draw from an imported MTLBuffer is in place, with zero bytes copied") {
    // The test shader reads each imported instance without a resource copy.
    caliper_host::HostRenderer* renderer = gfx_metal_renderer();
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx || renderer == nullptr) CALIPER_SKIP_NO_GPU("no Metal device");
    const int W = 32, H = 32;
    const uint32_t clear = 0xff000000u;
    Texture color = gfx->create_texture({W, H, Format::RGBA8Unorm, kRenderTarget | kCopySrcTex, "color"});
    Texture picks = gfx->create_texture({W, H, Format::RG32Uint, kRenderTarget | kCopySrcTex, "picks"});
    const char* source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Input { float3 pos [[attribute(0)]]; uint rgba [[attribute(1)]]; float radius [[attribute(2)]]; };
struct Output { float4 pos [[position]]; float4 color; uint2 id [[flat]]; };
vertex Output vs_instance(Input in [[stage_in]], uint vid [[vertex_id]], uint instance [[instance_id]]) {
    const float2 corners[6] = {float2(-1,-1),float2(1,-1),float2(-1,1),float2(1,-1),float2(1,1),float2(-1,1)};
    Output o;
    o.pos = float4(in.pos.xy + corners[vid] * in.radius, in.pos.z, 1);
    o.color = float4(in.rgba & 255u,(in.rgba >> 8u) & 255u,(in.rgba >> 16u) & 255u,in.rgba >> 24u) / 255.0f;
    o.id = uint2(9, 10 + instance);
    return o;
}
fragment float4 fs_instance(Output in [[stage_in]]) { return in.color; }
fragment uint2 fs_instance_pick(Output in [[stage_in]]) { return in.id; }
)metal";
    const EntryDesc entries[] = {{"vs_instance", kStageVertex, {1,1,1}},
                                 {"fs_instance", kStageFragment, {1,1,1}},
                                 {"fs_instance_pick", kStageFragment, {1,1,1}}};
    ProgramDesc description;
    description.label = "imported-instance-test";
    description.entries = entries;
    description.entry_count = 3;
    description.msl = source;
    Program program = gfx->create_program(description);
    REQUIRE_MESSAGE(bool(program), gfx->last_error());
    RenderPipelineDesc pd;
    pd.program = program;
    pd.vertex_entry = "vs_instance";
    pd.fragment_entry = "fs_instance";
    pd.vertex_layout_count = 2;
    pd.vertex_layouts[0].stride = 16;                    // centre and colour, per instance
    pd.vertex_layouts[0].step = Step::Instance;
    pd.vertex_layouts[0].attribute_count = 2;
    pd.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x3, 0};
    pd.vertex_layouts[0].attributes[1] = {1, VertexFormat::U32, 12};
    pd.vertex_layouts[1].stride = 0;                     // one radius for every point
    pd.vertex_layouts[1].attribute_count = 1;
    pd.vertex_layouts[1].attributes[0] = {2, VertexFormat::F32, 0};
    Pipeline draw_pipe = gfx->create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(draw_pipe), gfx->last_error());
    pd.fragment_entry = "fs_instance_pick";
    pd.color_format = Format::RG32Uint;
    Pipeline pick_pipe = gfx->create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(pick_pipe), gfx->last_error());
    const float radius[4] = {0.25f, 0, 0, 0};            // a quarter of the view's half height: 4 pixels
    Buffer constant = gfx->create_buffer({sizeof radius, kVertex | kCopyDst, false, "radius"});
    REQUIRE(gfx->write_buffer(constant, 0, radius, sizeof radius));

    // The producer's buffer, on the renderer's device: four points at byte offset 20, a
    // 4-byte-aligned offset that a storage binding's 256-byte alignment would refuse.
    struct PointRow { float pos[3]; uint32_t rgba; };
    static_assert(sizeof(PointRow) == 16, "the instance stride");
    const uint64_t kOffset = 20, kBytes = 256;
    const PointRow rows[4] = {
        {{-0.5f, 0.5f, 0.5f}, 0xff2040e0u}, {{0.5f, 0.5f, 0.5f}, 0xff10c060u},
        {{-0.5f, -0.5f, 0.5f}, 0xffe0a000u}, {{0.5f, -0.5f, 0.5f}, 0xffff00ffu},
    };
    id<MTLDevice> device = (__bridge id<MTLDevice>)renderer->native_device();
    id<MTLBuffer> producer = [device newBufferWithLength:kBytes options:MTLResourceStorageModeShared];
    std::memset(producer.contents, 0xee, kBytes);
    std::memcpy(static_cast<uint8_t*>(producer.contents) + kOffset, rows, sizeof rows);
    gfx->take_stats();

    ExternalBuffer ext;
    ext.kind = ExternalKind::MetalBuffer;
    ext.handle = (__bridge void*)producer;
    ext.size = kBytes;
    ext.usage = kVertex;
    ext.allow_copy = false;                              // in place or nothing
    ImportResult imp = gfx->import_buffer(ext);
    REQUIRE_MESSAGE(bool(imp.buffer), imp.reason);
    CHECK(imp.route == Route::InPlace);
    CHECK(imp.bytes_copied == 0);
    CHECK(std::string(imp.path) == "in-place");

    auto draw = [&](Buffer instances) {
        caliper_gfx::Frame f;
        for (int pass = 0; pass < 2; ++pass) {           // colour, then picks
            RenderTarget tg;
            tg.color = pass == 0 ? color : picks;
            tg.clear_rgba = pass == 0 ? clear : 0;
            DrawList list(tg);
            DrawRecord r;
            r.pipeline = pass == 0 ? draw_pipe : pick_pipe;
            r.vertex[0] = {instances, kOffset};
            r.vertex[1] = {constant, 0};
            r.count = 6;
            r.instance_count = 4;
            r.pick_id = 9;
            list.draw(r);
            f.render(std::move(list));
        }
        const Serial s = gfx->submit(std::move(f));
        REQUIRE_MESSAGE(s != 0, gfx->last_error());
    };
    auto reference = [&](const PointRow* rs) {
        Reference ref(W, H, clear);
        const int cx[4] = {8, 24, 8, 24}, cy[4] = {8, 8, 24, 24};
        for (int i = 0; i < 4; ++i)
            ref.paint(cx[i] - 4, cy[i] - 4, cx[i] + 4, cy[i] + 4, rs[i].rgba, 9, 10 + uint32_t(i), -1, false);
        return ref;
    };
    draw(imp.buffer);
    const Stats st = gfx->take_stats();
    CHECK(st.imports_in_place == 1);
    CHECK(st.imports_copied == 0);
    CHECK(st.copies == 0);
    CHECK(st.bytes_copied_from_host == 0);
    CHECK(st.bytes_copied_on_gpu == 0);
    CHECK(st.draws == 2);
    MESSAGE("import route ", std::string(imp.path), ", bytes copied ", imp.bytes_copied, ", copies this frame ",
            st.copies);

    Reference ref = reference(rows);
    const std::vector<uint8_t> got = read_back(*gfx, color, W, H);
    const size_t dc = first_diff(got, ref.color.data(), ref.color.size());
    CHECK_MESSAGE(dc == SIZE_MAX, "colour differs at byte ", dc);
    const std::vector<uint8_t> got_picks = read_back(*gfx, picks, W, H);
    const size_t di = first_diff(got_picks, ref.id.data(), ref.id.size() * 4);
    CHECK_MESSAGE(di == SIZE_MAX, "pick target differs at byte ", di);
    capture("layer_instanced.png", got, W, H);

    // In place: the producer rewrites a point, and the next frame draws it with no write or
    // import through the layer.
    PointRow changed[4] = {rows[0], rows[1], rows[2], rows[3]};
    changed[3].rgba = 0xff00ffffu;
    std::memcpy(static_cast<uint8_t*>(producer.contents) + kOffset + 3 * sizeof(PointRow), &changed[3],
                sizeof(PointRow));
    draw(imp.buffer);
    Reference ref2 = reference(changed);
    CHECK(first_diff(read_back(*gfx, color, W, H), ref2.color.data(), ref2.color.size()) == SIZE_MAX);

    // The same points from host memory that is not page-aligned: a copy, and the result says so.
    std::vector<uint8_t> host(kBytes + 64, 0xee);
    uint8_t* unaligned = host.data() + 4;
    if (reinterpret_cast<uintptr_t>(unaligned) % getpagesize() == 0) unaligned += 4;
    std::memcpy(unaligned + kOffset, rows, sizeof rows);
    ExternalBuffer hext;
    hext.kind = ExternalKind::HostMemory;
    hext.handle = unaligned;
    hext.size = kBytes;
    hext.usage = kVertex;
    hext.allow_copy = false;
    ImportResult refused = gfx->import_buffer(hext);
    CHECK(refused.route == Route::Refused);
    CHECK_FALSE(refused.reason.empty());
    hext.allow_copy = true;
    ImportResult copied = gfx->import_buffer(hext);
    REQUIRE_MESSAGE(bool(copied.buffer), copied.reason);
    CHECK(copied.route == Route::CopiedFromHost);
    CHECK(copied.bytes_copied == kBytes);
    CHECK(std::string(copied.path) == "copied-from-host");
    draw(copied.buffer);
    const Stats st2 = gfx->take_stats();
    CHECK(st2.imports_copied == 1);
    CHECK(st2.bytes_copied_from_host == kBytes);
    CHECK(first_diff(read_back(*gfx, color, W, H), ref.color.data(), ref.color.size()) == SIZE_MAX);

    // A page-aligned host page is wrapped in place.
    const size_t page = size_t(getpagesize());
    void* aligned = nullptr;
    REQUIRE(posix_memalign(&aligned, page, page) == 0);
    std::memcpy(static_cast<uint8_t*>(aligned) + kOffset, rows, sizeof rows);
    hext.handle = aligned;
    hext.size = page;
    ImportResult wrapped = gfx->import_buffer(hext);
    CHECK(wrapped.route == Route::InPlace);
    CHECK(wrapped.bytes_copied == 0);
    draw(wrapped.buffer);
    CHECK(first_diff(read_back(*gfx, color, W, H), ref.color.data(), ref.color.size()) == SIZE_MAX);
    gfx->release(wrapped.buffer);
    gfx->wait(gfx->last_use(color));
    gfx->poll();

    // The submits since the last take whose GPU time is known are counted beside the sum.
    const Stats st3 = gfx->take_stats();
    CHECK(st3.gpu_submits >= 1);
    CHECK(st3.gpu_ms > 0.0);

    // A transient import of the same aligned page is copied, never wrapped, and reads like it.
    hext.transient = true;
    ImportResult transient = gfx->import_buffer(hext);
    REQUIRE_MESSAGE(bool(transient.buffer), transient.reason);
    CHECK(transient.route == Route::CopiedFromHost);
    CHECK(transient.bytes_copied == page);
    static_cast<uint8_t*>(aligned)[kOffset] ^= 0xff;    // the producer's bytes change after the call
    const uint8_t* tm = static_cast<const uint8_t*>(gfx->mapped(transient.buffer));
    REQUIRE(tm != nullptr);
    CHECK(std::memcmp(tm + kOffset, rows, sizeof rows) == 0);
    gfx->release(transient.buffer);
    hext.transient = false;
    std::free(aligned);

    // The CPU view of an imported MTLBuffer: its contents in shared storage, none in private storage.
    id<MTLBuffer> shared_buf = [device newBufferWithBytes:rows length:sizeof rows options:MTLResourceStorageModeShared];
    id<MTLBuffer> private_buf = [device newBufferWithLength:sizeof rows options:MTLResourceStorageModePrivate];
    ExternalBuffer mext;
    mext.kind = ExternalKind::MetalBuffer;
    mext.handle = (__bridge void*)shared_buf;
    mext.size = sizeof rows;
    mext.usage = kVertex;
    ImportResult ms = gfx->import_buffer(mext);
    REQUIRE(bool(ms.buffer));
    CHECK(gfx->mapped(ms.buffer) == shared_buf.contents);
    mext.handle = (__bridge void*)private_buf;
    ImportResult mp = gfx->import_buffer(mext);
    REQUIRE(bool(mp.buffer));
    CHECK(gfx->mapped(mp.buffer) == nullptr);
    gfx->release(ms.buffer);
    gfx->release(mp.buffer);

    // An imported texture of the device is used in place too.
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:4 height:4 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> mt = [device newTextureWithDescriptor:td];
    ExternalTexture et;
    et.kind = ExternalKind::MetalTexture;
    et.handle = (__bridge void*)mt;
    et.width = et.height = 4;
    et.format = Format::RGBA8Unorm;
    et.usage = kSampled;
    ImportResult ti = gfx->import_texture(et);
    CHECK(ti.route == Route::InPlace);
    CHECK(gfx->imgui_texture_id(ti.texture) == (uint64_t)(__bridge void*)mt);
    et.width = 8;                                        // a description that does not match
    CHECK(gfx->import_texture(et).route == Route::Refused);
}

TEST_CASE("gfx/Metal layer: every generated program compiles and makes its pipelines through the layer") {
    using namespace caliper_gfx::shaders;
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    size_t pipelines = 0;
    for (uint32_t i = 0; i < kProgramCount; ++i) {
        const ProgramDescription& desc = kProgramDescriptions[i];
        Program p = gfx->create_program(kMetalPrograms[i]);
        REQUIRE_MESSAGE(bool(p), (std::string(desc.name) + ": " + gfx->last_error()));
        for (uint32_t k = 0; k < desc.pair_count; ++k) {
            Pipeline pipe = gfx->create_render_pipeline(caliper_test::generated_pipeline(desc, p, desc.pairs[k]));
            CHECK_MESSAGE(bool(pipe), (std::string(desc.name) + " " + desc.pairs[k].vertex + "+" +
                                      desc.pairs[k].fragment + ": " + gfx->last_error()));
            pipelines += bool(pipe);
        }
        for (uint32_t k = 0; k < desc.entry_count; ++k) {
            if (desc.entries[k].stage != kStageCompute) continue;
            ComputePipelineDesc cd;
            cd.program = p;
            cd.entry = desc.entries[k].name;
            Pipeline cp = gfx->create_compute_pipeline(cd);
            CHECK_MESSAGE(bool(cp), (std::string(desc.name) + ": " + gfx->last_error()));
            pipelines += bool(cp);
        }
    }
    MESSAGE(kProgramCount, " programs and ", pipelines, " pipelines made on Metal through the layer");
}

TEST_CASE("gfx/Metal layer: a frame the core refuses leaves the target untouched") {
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    const int W = 8, H = 8;
    Texture color = gfx->create_texture({W, H, Format::RGBA8Unorm, kRenderTarget | kCopySrcTex});
    Program program = gfx->create_program(test_program());
    RenderPipelineDesc pd;
    pd.program = program;
    pd.vertex_entry = "vs_quad";
    pd.fragment_entry = "fs_quad";
    pd.vertex_layout_count = 1;
    pd.vertex_layouts[0].stride = 8;
    pd.vertex_layouts[0].attribute_count = 1;
    pd.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x2, 0};
    pd.id_target = false;
    Pipeline pipe = gfx->create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(pipe), gfx->last_error());
    BindGroupDesc bd;
    bd.program = program;
    bd.group = 1;
    bd.entry_count = 1;
    bd.entries[0].binding = 0;
    bd.entries[0].buffer = kArena;
    BindGroup group = gfx->create_bind_group(bd);
    Buffer corners = gfx->create_buffer({sizeof kCorners, kVertex | kCopyDst});
    REQUIRE(gfx->write_buffer(corners, 0, kCorners, sizeof kCorners));
    RenderTarget tg;
    tg.color = color;
    tg.clear_rgba = 0xff0000ffu;
    {
        DrawList list(tg);
        Frame f;
        f.render(std::move(list));
        REQUIRE(gfx->submit(std::move(f)) != 0);
    }
    const std::vector<uint8_t> before = read_back(*gfx, color, W, H);
    tg.clear_rgba = 0xff00ff00u;
    DrawList list(tg);
    Quad q{};
    pixel_rect(0, 0, 8, 8, W, H, q.rect);
    DrawRecord r;
    r.pipeline = pipe;
    r.groups[1] = group;
    r.dynamic_offsets[0] = list.uniform(&q, sizeof q);
    r.vertex[0] = {corners, 0};
    r.count = 12;                                        // past the ten corners in the buffer
    list.draw(r);
    Frame f;
    f.render(std::move(list));
    CHECK(gfx->submit(std::move(f)) == 0);
    CHECK(gfx->last_error().find("reads past the end") != std::string::npos);
    CHECK(read_back(*gfx, color, W, H) == before);      // the new clear never ran
}

TEST_CASE("gfx/Metal layer: a draw that samples its own pass's target refuses the frame and leaves it untouched") {
    // Sampling a texture while the same pass renders into it is undefined on Metal and refused by
    // WebGPU, so the core refuses the frame. Sampled in a pass that renders elsewhere, it is fine.
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    const int W = 8, H = 8;
    Texture color = gfx->create_texture({W, H, Format::RGBA8Unorm, kRenderTarget | kSampled | kCopySrcTex, "color"});
    Texture other = gfx->create_texture({W, H, Format::RGBA8Unorm, kRenderTarget | kCopySrcTex, "other"});
    Program program = gfx->create_program(sample_program());
    REQUIRE_MESSAGE(bool(program), gfx->last_error());
    RenderPipelineDesc pd;
    pd.program = program;
    pd.vertex_entry = "vs_quad";
    pd.fragment_entry = "fs_sample";
    pd.vertex_layout_count = 1;
    pd.vertex_layouts[0].stride = 8;
    pd.vertex_layouts[0].attribute_count = 1;
    pd.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x2, 0};
    Pipeline pipe = gfx->create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(pipe), gfx->last_error());
    BindGroupDesc b1;
    b1.program = program;
    b1.group = 1;
    b1.entry_count = 1;
    b1.entries[0].binding = 0;
    b1.entries[0].buffer = kArena;
    BindGroup g1 = gfx->create_bind_group(b1);
    BindGroupDesc b2;
    b2.program = program;
    b2.group = 2;
    b2.entry_count = 1;
    b2.entries[0].binding = 0;
    b2.entries[0].texture = color;
    BindGroup g2 = gfx->create_bind_group(b2);
    REQUIRE_MESSAGE((g1 && g2), gfx->last_error());
    Buffer corners = gfx->create_buffer({sizeof kCorners, kVertex | kCopyDst});
    REQUIRE(gfx->write_buffer(corners, 0, kCorners, sizeof kCorners));
    auto sample_into = [&](Texture target, uint32_t clear) {
        RenderTarget tg;
        tg.color = target;
        tg.clear_rgba = clear;
        DrawList list(tg);
        Quad q{};
        pixel_rect(0, 0, W, H, W, H, q.rect);
        DrawRecord r;
        r.pipeline = pipe;
        r.groups[1] = g1;
        r.groups[2] = g2;
        r.dynamic_offsets[0] = list.uniform(&q, sizeof q);
        r.vertex[0] = {corners, 0};
        r.count = 6;
        list.draw(r);
        Frame f;
        f.render(std::move(list));
        return gfx->submit(std::move(f));
    };
    {   // colour holds blue
        RenderTarget tg;
        tg.color = color;
        tg.clear_rgba = 0xffff0000u;
        Frame f;
        f.render(DrawList(tg));
        REQUIRE(gfx->submit(std::move(f)) != 0);
    }
    const std::vector<uint8_t> before = read_back(*gfx, color, W, H);
    // Sampled while rendering into another target: every pixel takes colour's texel (0, 0).
    REQUIRE_MESSAGE(sample_into(other, 0xff00ff00u) != 0, gfx->last_error());
    const std::vector<uint8_t> copied = read_back(*gfx, other, W, H);
    CHECK(copied == before);
    // Sampled while rendering into colour itself: refused, and colour keeps its blue.
    CHECK(sample_into(color, 0xff0000ffu) == 0);
    CHECK_MESSAGE(gfx->last_error().find("binds the pass's own colour target") != std::string::npos, gfx->last_error());
    CHECK(read_back(*gfx, color, W, H) == before);
}

TEST_CASE("gfx/Metal layer: a write after a queued write lands last, while the queued one waits behind other work") {
    // A write to a resource that an unfinished submit uses is queued as a blit. Here that blit
    // waits behind a command buffer that waits on an event, the submit that used the resource
    // completes, and a second write follows. The second write is the newer one, so a readback
    // ordered after both must see its bytes, for a buffer and for a shared texture alike.
    caliper_host::HostRenderer* renderer = gfx_metal_renderer();
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx || renderer == nullptr) CALIPER_SKIP_NO_GPU("no Metal device");
    id<MTLDevice> device = (__bridge id<MTLDevice>)renderer->native_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)renderer->native_queue();
    const uint32_t kW = 64;   // one row of RGBA8, 256 bytes
    std::vector<uint32_t> older(kW), newer(kW);
    for (uint32_t i = 0; i < kW; ++i) {
        older[i] = 0xff000000u | i;
        newer[i] = 0xff00ff00u | (i << 16);
    }
    // Both events are signalled whatever happens, so that nothing stays held on the shared queue.
    struct Gates {
        id<MTLSharedEvent> submit, queue;
        ~Gates() {
            submit.signaledValue = 1;
            queue.signaledValue = 1;
        }
    };
    for (const bool texture : {false, true}) {
        @autoreleasepool {
            CAPTURE(texture);
            Gates gates{[device newSharedEvent], [device newSharedEvent]};
            Buffer buf = gfx->create_buffer({kW * 4, kCopySrc | kCopyDst, false, "written"});
            Texture tex = gfx->create_texture({kW, 1, Format::RGBA8Unorm, kSampled | kCopyDstTex | kCopySrcTex, "written"});
            REQUIRE((buf && tex));
            // The submit that uses the resource waits on the first event, so it is unfinished
            // when the first write comes: the buffer is its copy's source, the texture its target.
            Frame f;
            f.wait(ExternalSync{ExternalSync::MetalEvent, (__bridge void*)gates.submit, 1});
            f.copy_buffer_to_texture(buf, 0, kW * 4, tex);
            const Serial s = gfx->submit(std::move(f));
            REQUIRE_MESSAGE(s != 0, gfx->last_error());
            // Other work on the queue that waits on the second event, ahead of the queued write.
            id<MTLCommandBuffer> hold = [queue commandBuffer];
            [hold encodeWaitForEvent:gates.queue value:1];
            [hold commit];
            auto write = [&](const std::vector<uint32_t>& d) {
                return texture ? gfx->write_texture(tex, d.data(), kW * 4) : gfx->write_buffer(buf, 0, d.data(), kW * 4);
            };
            REQUIRE(write(older));
            gates.submit.signaledValue = 1;
            REQUIRE(gfx->wait(s));
            CHECK(gfx->last_use(texture ? tex.id : buf.id) == s);   // the submit is over
            REQUIRE(write(newer));
            gates.queue.signaledValue = 1;
            if (!texture) {   // the buffer is read through a copy into the texture, after both writes
                Frame copy;
                copy.copy_buffer_to_texture(buf, 0, kW * 4, tex);
                REQUIRE(gfx->submit(std::move(copy)) != 0);
            }
            const std::vector<uint8_t> got = read_back(*gfx, tex, kW, 1);
            const size_t d = first_diff(got, newer.data(), kW * 4);
            const bool holds_older = first_diff(got, older.data(), kW * 4) == SIZE_MAX;
            CHECK_MESSAGE(d == SIZE_MAX, "the readback differs from the newer write at byte ", d,
                          std::string(holds_older ? ": it holds the older, queued write" : ""));
            gfx->release(buf);
            gfx->release(tex);
        }
    }
}

TEST_CASE("gfx/Metal layer: a queued texture write reads only the rows the caller gave") {
    // A 16 x 4 RGBA8 render target (private, so every write is queued) written with a row pitch
    // of 256 needs 3 * 256 + 64 bytes from the caller. The data ends right at a page that cannot
    // be read, so reading row_pitch * height bytes would fault.
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    const uint32_t W = 16, H = 4, kPitch = 256, kTight = (H - 1) * kPitch + W * 4;
    Texture t = gfx->create_texture({W, H, Format::RGBA8Unorm, kRenderTarget | kCopyDstTex | kCopySrcTex, "target"});
    REQUIRE(bool(t));
    const size_t page = size_t(getpagesize());
    void* mem = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    REQUIRE(mem != MAP_FAILED);
    REQUIRE(mprotect(static_cast<uint8_t*>(mem) + page, page, PROT_NONE) == 0);
    uint8_t* data = static_cast<uint8_t*>(mem) + page - kTight;
    std::vector<uint8_t> want(size_t(W) * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W * 4; ++x) {
            const uint8_t v = uint8_t(y * 61 + x * 7 + 1);
            data[size_t(y) * kPitch + x] = v;
            want[size_t(y) * W * 4 + x] = v;
        }
    REQUIRE(gfx->write_texture(t, data, kPitch));
    const std::vector<uint8_t> got = read_back(*gfx, t, W, H);
    const size_t d = first_diff(got, want.data(), want.size());
    CHECK_MESSAGE(d == SIZE_MAX, "the texture differs from the rows written at byte ", d);
    munmap(mem, 2 * page);
}

// The layer's CPU encode time for a draw list of 1,000 draws, on the recording backend and on
// Metal: building the list (a uniform slice and a record per draw) and submitting it
// (validation, the arena upload, encoding and commit). Opt-in, because timing needs a quiet
// machine: CALIPER_GFX_LAYER_BENCH=1.
TEST_CASE("gfx/Metal layer: timing — CPU encode time of 1,000 draws") {
    if (std::getenv("CALIPER_GFX_LAYER_BENCH") == nullptr) return;
    std::unique_ptr<Backend> metal = metal_layer();
    if (!metal) CALIPER_SKIP_NO_GPU("no Metal device");
    auto run = [](Backend& gfx, const char* name) {
        const int W = 64, H = 64, kDraws = 1000, kWarm = 50, kFrames = 400;
        Texture color = gfx.create_texture({W, H, Format::RGBA8Unorm, kRenderTarget});
        Texture ids = gfx.create_texture({W, H, Format::RG32Uint, kRenderTarget});
        Program program = gfx.create_program(test_program());
        RenderPipelineDesc pd;
        pd.program = program;
        pd.vertex_entry = "vs_quad";
        pd.fragment_entry = "fs_quad";
        pd.vertex_layout_count = 1;
        pd.vertex_layouts[0].stride = 8;
        pd.vertex_layouts[0].attribute_count = 1;
        pd.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x2, 0};
        pd.id_target = true;
        Pipeline pipe = gfx.create_render_pipeline(pd);
        BindGroupDesc bd;
        bd.program = program;
        bd.group = 1;
        bd.entry_count = 1;
        bd.entries[0].binding = 0;
        bd.entries[0].buffer = kArena;
        BindGroup group = gfx.create_bind_group(bd);
        Buffer corners = gfx.create_buffer({sizeof kCorners, kVertex | kCopyDst});
        gfx.write_buffer(corners, 0, kCorners, sizeof kCorners);
        RenderTarget tg;
        tg.color = color;
        tg.id = ids;
        std::vector<double> build_ms, submit_ms;
        DrawList list;
        for (int frame = 0; frame < kWarm + kFrames; ++frame) {
            const auto t0 = std::chrono::steady_clock::now();
            list.reset(tg);
            list.reserve(kDraws, size_t(kDraws) * 256);
            for (int i = 0; i < kDraws; ++i) {
                Quad q{};
                const int x = i % 32, y = (i / 32) % 32;
                pixel_rect(x * 2, y * 2, x * 2 + 2, y * 2 + 2, W, H, q.rect);
                q.rgba = 0xff000000u | uint32_t(i * 2654435761u >> 8);
                q.pick_id = 1;
                q.item = uint32_t(i);
                DrawRecord r;
                r.pipeline = pipe;
                r.groups[1] = group;
                r.dynamic_offsets[0] = list.uniform(&q, sizeof q);
                r.vertex[0] = {corners, 0};
                r.count = 6;
                list.draw(r);
            }
            const auto t1 = std::chrono::steady_clock::now();
            Frame f;
            f.render(std::move(list));
            const Serial s = gfx.submit(std::move(f));
            const auto t2 = std::chrono::steady_clock::now();
            REQUIRE_MESSAGE(s != 0, gfx.last_error());
            gfx.wait(s);                                 // outside the timed region
            list = DrawList();
            if (frame >= kWarm) {
                build_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                submit_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
        }
        auto pct = [](std::vector<double> v, double p) {
            std::sort(v.begin(), v.end());
            return v[size_t(p * (v.size() - 1))];
        };
        double load[3] = {};
        getloadavg(load, 3);
        std::printf("[gfx-layer-bench] backend %s, %d draws a frame, %zu frames: build median %.3f ms p95 %.3f ms; "
                    "submit (validate + encode + commit) median %.3f ms p95 %.3f ms; per draw %.1f ns; load %.2f\n",
                    name, kDraws, submit_ms.size(), pct(build_ms, 0.5), pct(build_ms, 0.95), pct(submit_ms, 0.5),
                    pct(submit_ms, 0.95), (pct(build_ms, 0.5) + pct(submit_ms, 0.5)) * 1e6 / kDraws, load[0]);
        std::fflush(stdout);
    };
    RecordingBackend recording;
    recording.set_keep_frames(false);
    run(recording, "recording");
    run(*metal, "metal");
}

#endif  // CALIPER_HAVE_METAL
