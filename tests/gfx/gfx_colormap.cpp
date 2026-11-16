// Rows of the graphics suite for the colormap by keys (renderer core, "Exact colormap output",
// migration step 6) on Metal, through the graphics layer's Metal backend and the generated
// programs, with every block's keys from the shared header src/host/gfx/colormap_keys.h:
//
// - The compute form that the texture path uses (the program colormap_texture): the boundary
//   inputs of every range of tests/shaders/colormap_cases.h as an f32 image, from a layer buffer
//   and from an imported MTLBuffer at an element offset, stored to an rgba8unorm texture.
// - The vertex-stage colormap of a points draw (scene_points with color_mode 1): one point a
//   pixel wide on each pixel centre, one per input, its value from a layer buffer and from an
//   imported MTLBuffer at a byte offset.
//
// Each target is read back through the layer and compared byte for byte with map_f32_to_rgba8
// through the test table, and each import must report the in-place route with zero bytes
// copied. A last case draws viridis over a ramp and rows of NaN and infinities, which
// CALIPER_GFX_CAPTURE_DIR=<folder> writes as colormap-keys-texture-ramp.png and
// colormap-keys-points-ramp.png, scaled up four times.
//
// CALIPER_GFX_CMAP_BENCH=1 runs the timing row: the GPU time of the integer-decision pass (the
// generated colormap_texture, compiled as the layer compiles it) against the float pass of the
// Metal renderer today (cmap_f32, copied below for this measurement only, compiled as the
// renderer compiles it), on 1,024 x 1,024 and 2,048 x 2,048 heatmaps, and the integer pass with the
// other math mode. Compiled as Objective-C++ with ARC on Apple.
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>

#include "../shaders/colormap_cases.h"
#include "gfx/colormap_keys.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

// The suite's Metal environment (gfx_main.cpp): its renderer, or nullptr without a device.
caliper_host::HostRenderer* gfx_metal_renderer();

using namespace caliper_gfx;
namespace sh = caliper_gfx::shaders;

namespace {

constexpr uint32_t kWidth = 256;   // texels or points per row of every target

std::unique_ptr<Backend> metal_layer() {
    caliper_host::HostRenderer* r = gfx_metal_renderer();
    if (r == nullptr) return nullptr;
    return make_metal_backend(r->native_device(), r->native_queue());
}

std::vector<uint8_t> read_back(Backend& gfx, Texture t, uint32_t w, uint32_t h) {
    Readback r = gfx.read_texture(t, 0, 0, w, h);
    REQUIRE_MESSAGE(bool(r), gfx.last_error());
    std::vector<uint8_t> out;
    REQUIRE(gfx.wait(gfx.last_use(t)));
    REQUIRE(gfx.take_readback(r, &out));
    return out;
}

// A PNG of a readback in CALIPER_GFX_CAPTURE_DIR, scaled up four times with nearest sampling.
void capture(const char* name, const std::vector<uint8_t>& rgba, int w, int h) {
    const char* dir = std::getenv("CALIPER_GFX_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    const int s = 4;
    std::vector<uint8_t> big(size_t(w) * s * h * s * 4);
    for (int y = 0; y < h * s; ++y)
        for (int x = 0; x < w * s; ++x)
            std::memcpy(&big[(size_t(y) * w * s + x) * 4], &rgba[(size_t(y / s) * w + x / s) * 4], 4);
    CHECK(stbi_write_png(path.c_str(), w * s, h * s, 4, big.data(), w * s * 4) != 0);
    MESSAGE("capture written: ", path);
}

// One range's case: its block with the test table, its inputs padded with zeros to whole rows of
// kWidth, and the reference's RGBA for each.
struct Case {
    colormap_cases::Range range;
    sh::Cmap block{};
    std::vector<uint32_t> in, want;
    uint32_t rows = 0;
};

std::vector<Case> cases() {
    const std::vector<uint32_t> table = colormap_cases::lut();
    const auto rs = colormap_cases::ranges();
    std::vector<Case> out;
    for (size_t i = 0; i < rs.size(); ++i) {
        Case c;
        c.range = rs[i];
        c.block = colormap_cases::block(rs[i], table);
        c.in = colormap_cases::inputs(rs[i], c.block, colormap_cases::seed(i));
        while (c.in.size() % kWidth) c.in.push_back(0);
        c.rows = uint32_t(c.in.size() / kWidth);
        c.want = colormap_cases::expected(rs[i], c.in, table);
        out.push_back(std::move(c));
    }
    // Last, a picture for the captures, compared like the others: viridis over (0, 1) on a ramp
    // from -0.25 to 1.25 along each of 24 rows, then 4 rows of NaN, 2 of +inf and 2 of -inf.
    Case ramp;
    ramp.range = {"viridis ramp", 0.0f, 1.0f};
    REQUIRE(cmap::derive_block(CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, &ramp.block));
    ramp.rows = 32;
    for (uint32_t y = 0; y < ramp.rows; ++y)
        for (uint32_t x = 0; x < kWidth; ++x) {
            const float v = y < 24 ? -0.25f + 1.5f * float(x) / float(kWidth - 1)
                          : y < 28 ? std::numeric_limits<float>::quiet_NaN()
                          : y < 30 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
            ramp.in.push_back(colormap_cases::as_bits(v));
        }
    const uint32_t* viridis = caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS);
    ramp.want = colormap_cases::expected(ramp.range, ramp.in, std::vector<uint32_t>(viridis, viridis + 256));
    out.push_back(std::move(ramp));
    return out;
}

// How many readback texels differ from the reference; the first in *first.
size_t differ(const std::vector<uint8_t>& got, const std::vector<uint32_t>& want, size_t* first) {
    if (got.size() != want.size() * 4) { *first = 0; return want.size(); }
    size_t bad = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        uint32_t g;
        std::memcpy(&g, &got[i * 4], 4);
        if (g != want[i] && bad++ == 0) *first = i;
    }
    return bad;
}

// The producer's MTLBuffer on the renderer's device, its values starting at byte `offset`.
id<MTLBuffer> producer_buffer(const std::vector<uint32_t>& values, uint64_t offset) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)gfx_metal_renderer()->native_device();
    id<MTLBuffer> b = [device newBufferWithLength:offset + values.size() * 4 options:MTLResourceStorageModeShared];
    std::memset(b.contents, 0xee, offset);
    std::memcpy(static_cast<uint8_t*>(b.contents) + offset, values.data(), values.size() * 4);
    return b;
}

// Imports a producer's buffer in place, or fails the row.
Buffer import_in_place(Backend& gfx, id<MTLBuffer> producer, uint32_t usage) {
    ExternalBuffer ext;
    ext.kind = ExternalKind::MetalBuffer;
    ext.handle = (__bridge void*)producer;
    ext.size = producer.length;
    ext.usage = usage;
    ext.allow_copy = false;   // in place or nothing
    ImportResult imp = gfx.import_buffer(ext);
    REQUIRE_MESSAGE(bool(imp.buffer), imp.reason);
    CHECK(imp.route == Route::InPlace);
    CHECK(imp.bytes_copied == 0);
    CHECK(std::string(imp.path) == "in-place");
    return imp.buffer;
}

}  // namespace

TEST_CASE("gfx/Metal colormap: the texture path's compute form matches map_f32_to_rgba8 byte for byte, "
          "from a layer buffer and from an imported buffer") {
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    const uint32_t prog = uint32_t(sh::ProgramId::colormap_texture);
    Program program = gfx->create_program(sh::kMetalPrograms[prog]);
    REQUIRE_MESSAGE(bool(program), gfx->last_error());
    ComputePipelineDesc cd;
    cd.program = program;
    cd.entry = "cs_colormap_texture";
    Pipeline pipe = gfx->create_compute_pipeline(cd);
    REQUIRE_MESSAGE(bool(pipe), gfx->last_error());
    BindGroupDesc g1;
    g1.program = program;
    g1.group = 1;
    g1.entry_count = 1;
    g1.entries[0].binding = 1;
    g1.entries[0].buffer = kArena;
    BindGroup group1 = gfx->create_bind_group(g1);
    REQUIRE_MESSAGE(bool(group1), gfx->last_error());

    size_t compared = 0;
    const std::vector<Case> all = cases();
    for (const Case& c : all) {
        Texture out = gfx->create_texture({kWidth, c.rows, Format::RGBA8Unorm, kStorageTex | kCopySrcTex, "colormap"});
        REQUIRE_MESSAGE(bool(out), gfx->last_error());
        // From a layer buffer the values start at element 0; from the producer's buffer at
        // element 5, a 20-byte offset that a storage binding's 256-byte alignment would refuse,
        // so the whole buffer is bound and the program starts at TexParams::base.
        const uint32_t kBase = 5;
        Buffer layer_buf = gfx->create_buffer({c.in.size() * 4, kStorage | kCopyDst, false, "values"});
        REQUIRE(gfx->write_buffer(layer_buf, 0, c.in.data(), c.in.size() * 4));
        id<MTLBuffer> producer = producer_buffer(c.in, kBase * 4);
        gfx->take_stats();
        const Buffer imported = import_in_place(*gfx, producer, kStorage);
        for (int from_import = 0; from_import < 2; ++from_import) {
            INFO("range " << c.range.name << (from_import ? ", imported buffer" : ", layer buffer"));
            BindGroupDesc g0;
            g0.program = program;
            g0.group = 0;
            g0.entry_count = 3;
            g0.entries[0].binding = 0;
            g0.entries[0].buffer = kArena;
            g0.entries[1].binding = 1;
            g0.entries[1].buffer = from_import ? imported : layer_buf;   // size 0: the whole buffer
            g0.entries[2].binding = 2;
            g0.entries[2].texture = out;
            BindGroup group0 = gfx->create_bind_group(g0);
            REQUIRE_MESSAGE(bool(group0), gfx->last_error());
            caliper_gfx::Frame f;
            sh::TexParams tp{kWidth, c.rows, from_import ? kBase : 0u, kWidth};
            const uint32_t offsets[2] = {f.uniform(&tp, sizeof tp), f.uniform(&c.block, sizeof c.block)};
            const BindGroup groups[2] = {group0, group1};
            const uint16_t* wg = sh::kProgramDescriptions[prog].entries[0].workgroup_size;   // 16 x 8
            f.dispatch(pipe, groups, offsets, (kWidth + wg[0] - 1) / wg[0], (c.rows + wg[1] - 1) / wg[1], 1);
            const Serial s = gfx->submit(std::move(f));
            REQUIRE_MESSAGE(s != 0, gfx->last_error());
            const std::vector<uint8_t> got = read_back(*gfx, out, kWidth, c.rows);
            size_t first = 0;
            const size_t bad = differ(got, c.want, &first);
            CHECK_MESSAGE(bad == 0, bad, " of ", c.want.size(), " texels differ, first input 0x", std::hex,
                          c.in[first < c.in.size() ? first : 0]);
            compared += c.want.size();
            gfx->release(group0);
            if (&c == &all.back() && from_import) capture("colormap-keys-texture-ramp.png", got, int(kWidth), int(c.rows));
        }
        const Stats st = gfx->take_stats();
        CHECK(st.imports_in_place == 1);
        CHECK(st.imports_copied == 0);
        CHECK(st.copies == 0);
        CHECK(st.bytes_copied_on_gpu == 0);
        CHECK(st.dispatches == 2);
        gfx->release(imported);
        gfx->release(layer_buf);
        gfx->release(out);
    }
    MESSAGE(compared, " texels compared over ", all.size(), " cases, from a layer buffer and an imported one");
}

namespace {

CaliperTensor tensor2d(void* data, CaliperDType dt, int64_t h, int64_t w, int64_t c, CaliperDeviceKind dev) {
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = data;
    t.dtype = dt;
    t.ndim = c ? 3 : 2;
    t.shape[0] = h;
    t.shape[1] = w;
    t.shape[2] = c;
    t.device = dev;
    return t;
}

}  // namespace

TEST_CASE("gfx/Metal texture path: update_texture_from_tensor matches map_f32_to_rgba8 and the image's bytes, "
          "from a layer buffer, an imported MTLBuffer at an offset, a METAL tensor and a CPU tensor") {
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    const std::vector<uint32_t> table = colormap_cases::lut();
    const auto rs = colormap_cases::ranges();
    const uint32_t usage = kSampled | kStorageTex | kCopyDstTex | kCopySrcTex;
    size_t compared = 0;
    for (size_t i = 0; i < rs.size(); ++i) {
        const sh::Cmap block = colormap_cases::block(rs[i], table);
        std::vector<uint32_t> in = colormap_cases::inputs(rs[i], block, colormap_cases::seed(i));
        // An odd width, so that rows end inside a 16 x 8 workgroup and the pitch is no multiple of 256.
        const uint32_t W = 251, H = uint32_t((in.size() + W - 1) / W);
        in.resize(size_t(W) * H, 0);
        const std::vector<uint32_t> want = colormap_cases::expected(rs[i], in, table);
        Texture out = gfx->create_texture({W, H, Format::RGBA8Unorm, usage, "texture"});
        REQUIRE_MESSAGE(bool(out), gfx->last_error());
        const TextureMapping m{&block, uint32_t(sizeof block)};
        Buffer layer_buf = gfx->create_buffer({in.size() * 4, kStorage | kCopyDst, false, "values"});
        REQUIRE(gfx->write_buffer(layer_buf, 0, in.data(), in.size() * 4));
        const uint64_t kOffset = 20;
        id<MTLBuffer> producer = producer_buffer(in, kOffset);
        const Buffer imported = import_in_place(*gfx, producer, kStorage);
        for (int from = 0; from < 4; ++from) {
            INFO("range " << rs[i].name << ", source " << from);
            TensorSource src;
            src.w = W;
            src.h = H;
            CaliperTensor t{};
            const char* path = "";
            Route route = Route::InPlace;
            if (from == 0) {
                src.buffer = layer_buf;
                path = "compute-imported";
            } else if (from == 1) {
                src.buffer = imported;
                src.offset = kOffset;
                path = "compute-imported";
            } else if (from == 2) {
                t = tensor2d((__bridge void*)producer, CALIPER_DT_F32, H, W, 0, CALIPER_DEV_METAL);
                src.tensor = &t;
                src.offset = kOffset;
                path = "compute";
            } else {
                t = tensor2d(in.data(), CALIPER_DT_F32, H, W, 0, CALIPER_DEV_CPU);
                src.tensor = &t;
                path = "cpu-staged";
                route = Route::CopiedFromHost;
            }
            gfx->take_stats();
            const UpdateResult r = gfx->update_texture_from_tensor(out, src, m);
            REQUIRE_MESSAGE(r.ok, gfx->last_error());
            CHECK(std::string(r.path) == path);
            CHECK(r.route == route);
            CHECK(r.bytes_copied == (route == Route::CopiedFromHost ? in.size() * 4 : 0u));
            const Stats st = gfx->take_stats();
            CHECK(st.dispatches == 1);
            CHECK(st.imports_copied == 0);
            CHECK(st.bytes_copied_from_host == (route == Route::CopiedFromHost ? in.size() * 4 : 0u));
            const std::vector<uint8_t> got = read_back(*gfx, out, W, H);
            size_t first = 0;
            const size_t bad = differ(got, want, &first);
            CHECK_MESSAGE(bad == 0, bad, " of ", want.size(), " texels differ, first input 0x", std::hex, in[first]);
            compared += want.size();
        }
        gfx->release(imported);
        gfx->release(layer_buf);
        gfx->release(out);
    }

    // RGBA8 images: 17 x 9 texels, a row pitch of 68 bytes, from a buffer at byte 20, a METAL tensor,
    // and a CPU tensor, which is uploaded into the texture with no pass.
    const uint32_t W = 17, H = 9;
    std::vector<uint32_t> texels(size_t(W) * H);
    for (size_t i = 0; i < texels.size(); ++i) texels[i] = uint32_t(i * 2654435761u);
    Texture img = gfx->create_texture({W, H, Format::RGBA8Unorm, usage, "image"});
    id<MTLBuffer> producer = producer_buffer(texels, 20);
    const Buffer imported = import_in_place(*gfx, producer, kStorage);
    for (int from = 0; from < 3; ++from) {
        INFO("image source " << from);
        TensorSource src;
        src.w = W;
        src.h = H;
        src.dtype = CALIPER_DT_U8;
        src.channels = 4;
        src.stride_x = 4;
        CaliperTensor t{};
        if (from == 0) {
            src.buffer = imported;
            src.offset = 20;
        } else if (from == 1) {
            t = tensor2d((__bridge void*)producer, CALIPER_DT_U8, H, W, 4, CALIPER_DEV_METAL);
            src.tensor = &t;
            src.offset = 20;
        } else {
            t = tensor2d(texels.data(), CALIPER_DT_U8, H, W, 4, CALIPER_DEV_CPU);
            src.tensor = &t;
        }
        const UpdateResult r = gfx->update_texture_from_tensor(img, src, TextureMapping{});
        REQUIRE_MESSAGE(r.ok, gfx->last_error());
        CHECK(std::string(r.path) == (from == 0 ? "blit-imported" : from == 1 ? "blit" : "cpu-staged"));
        const std::vector<uint8_t> got = read_back(*gfx, img, W, H);
        size_t first = 0;
        CHECK(differ(got, texels, &first) == 0);
        // Clear it between sources, so that each one is seen to write every texel.
        std::vector<uint32_t> zeros(texels.size(), 0);
        REQUIRE(gfx->write_texture(img, zeros.data(), W * 4));
    }
    gfx->release(imported);
    gfx->release(img);
    MESSAGE(compared, " f32 texels compared over ", rs.size(), " ranges and 4 sources, and an image from 3 sources");
}

TEST_CASE("gfx/Metal texture path: a METAL tensor's stream orders the update after its producer's queue, "
          "and the update's refusals name their reason") {
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    id<MTLDevice> device = (__bridge id<MTLDevice>)gfx_metal_renderer()->native_device();
    const uint32_t W = 64, H = 8;
    std::vector<uint32_t> before(size_t(W) * H, 0x11223344u), after(size_t(W) * H);
    for (size_t i = 0; i < after.size(); ++i) after[i] = uint32_t(i * 2654435761u) | 0xff000000u;
    id<MTLBuffer> data = producer_buffer(before, 0);
    id<MTLBuffer> fresh = producer_buffer(after, 0);
    Texture img = gfx->create_texture({W, H, Format::RGBA8Unorm, kStorageTex | kCopySrcTex, "image"});
    // The producer's queue holds a write of the fresh texels behind an event the test signals
    // after the update was submitted: the update must wait for it on the GPU.
    id<MTLCommandQueue> producer = [device newCommandQueue];
    id<MTLSharedEvent> gate = [device newSharedEvent];
    id<MTLCommandBuffer> cb = [producer commandBuffer];
    [cb encodeWaitForEvent:gate value:1];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:fresh sourceOffset:0 toBuffer:data destinationOffset:0 size:after.size() * 4];
    [blit endEncoding];
    [cb commit];
    CaliperTensor t = tensor2d((__bridge void*)data, CALIPER_DT_U8, H, W, 4, CALIPER_DEV_METAL);
    t.stream = (__bridge void*)producer;
    TensorSource src;
    src.w = W;
    src.h = H;
    src.dtype = CALIPER_DT_U8;
    src.channels = 4;
    src.stride_x = 4;
    src.tensor = &t;
    const UpdateResult r = gfx->update_texture_from_tensor(img, src, TextureMapping{});
    REQUIRE_MESSAGE(r.ok, gfx->last_error());
    CHECK_FALSE(gfx->completed(r.serial));   // held behind the producer's write
    gate.signaledValue = 1;
    const std::vector<uint8_t> got = read_back(*gfx, img, W, H);
    size_t first = 0;
    CHECK(differ(got, after, &first) == 0);

    // Refusals: each leaves the reason and changes nothing.
    TensorSource bad = src;
    bad.tensor = nullptr;
    CHECK_FALSE(gfx->update_texture_from_tensor(img, bad, TextureMapping{}).ok);
    CHECK(gfx->last_error().find("neither a buffer nor a tensor") != std::string::npos);
    bad = src;
    bad.w = W - 1;
    CHECK_FALSE(gfx->update_texture_from_tensor(img, bad, TextureMapping{}).ok);
    CHECK(gfx->last_error().find("the texture 64x8") != std::string::npos);
    bad = src;
    bad.channels = 3;
    bad.stride_x = 3;
    CHECK_FALSE(gfx->update_texture_from_tensor(img, bad, TextureMapping{}).ok);
    CHECK(gfx->last_error().find("u8 with four") != std::string::npos);
    bad = src;
    bad.offset = 2;
    CHECK_FALSE(gfx->update_texture_from_tensor(img, bad, TextureMapping{}).ok);
    CHECK(gfx->last_error().find("multiples of 4") != std::string::npos);
    bad = src;
    bad.dtype = CALIPER_DT_F32;
    bad.channels = 1;
    bad.stride_x = 1;
    CHECK_FALSE(gfx->update_texture_from_tensor(img, bad, TextureMapping{}).ok);
    CHECK(gfx->last_error().find("key block") != std::string::npos);
    bad = src;
    bad.offset = 4;   // the image no longer fits the MTLBuffer
    CHECK_FALSE(gfx->update_texture_from_tensor(img, bad, TextureMapping{}).ok);
    CHECK(gfx->last_error().find("cannot be read in place") != std::string::npos);
    gfx->release(img);
}

TEST_CASE("gfx/Metal colormap: the vertex-stage colormap of a points draw matches map_f32_to_rgba8 byte for byte, "
          "from a layer buffer and from an imported buffer") {
    using namespace caliper_gfx::shaders;
    std::unique_ptr<Backend> gfx = metal_layer();
    if (!gfx) CALIPER_SKIP_NO_GPU("no Metal device");
    Program program = gfx->create_program(kMetalPrograms[uint32_t(ProgramId::scene_points)]);
    REQUIRE_MESSAGE(bool(program), gfx->last_error());
    RenderPipelineDesc pd;
    pd.program = program;
    pd.vertex_entry = "vs_points";
    pd.fragment_entry = "fs_points";
    pd.vertex_layout_count = 3;
    pd.vertex_layouts[0].stride = 12;                   // the centre, per instance
    pd.vertex_layouts[0].step = Step::Instance;
    pd.vertex_layouts[0].attribute_count = 1;
    pd.vertex_layouts[0].attributes[0] = {0, VertexFormat::F32x3, 0};
    pd.vertex_layouts[1].stride = 4;                    // the value's bits, per instance
    pd.vertex_layouts[1].step = Step::Instance;
    pd.vertex_layouts[1].attribute_count = 1;
    pd.vertex_layouts[1].attributes[0] = {1, VertexFormat::U32, 0};
    pd.vertex_layouts[2].stride = 0;                    // one radius, unused with size_px
    pd.vertex_layouts[2].attribute_count = 1;
    pd.vertex_layouts[2].attributes[0] = {2, VertexFormat::F32, 0};
    Pipeline pipe = gfx->create_render_pipeline(pd);
    REQUIRE_MESSAGE(bool(pipe), gfx->last_error());
    Texture class_table = gfx->create_texture({1, 1, Format::RGBA8Unorm, kSampled, "class table"});
    Texture no_image = gfx->create_texture({1, 1, Format::R32Uint, kSampled, "no image"});
    BindGroupDesc g0;
    g0.program = program;
    g0.group = 0;
    g0.entry_count = 2;
    g0.entries[0].binding = 0;
    g0.entries[0].buffer = kArena;
    g0.entries[1].binding = 1;
    g0.entries[1].texture = class_table;
    BindGroupDesc g1;
    g1.program = program;
    g1.group = 1;
    g1.entry_count = 2;
    g1.entries[0].binding = 0;
    g1.entries[0].buffer = kArena;
    g1.entries[1].binding = 1;
    g1.entries[1].buffer = kArena;
    BindGroupDesc g2;
    g2.program = program;
    g2.group = 2;
    g2.entry_count = 1;
    g2.entries[0].binding = 3;
    g2.entries[0].texture = no_image;
    BindGroup group0 = gfx->create_bind_group(g0), group1 = gfx->create_bind_group(g1), group2 = gfx->create_bind_group(g2);
    REQUIRE_MESSAGE((group0 && group1 && group2), gfx->last_error());
    const float radius[4] = {1.0f, 0, 0, 0};
    Buffer constant = gfx->create_buffer({sizeof radius, kVertex | kCopyDst, false, "radius"});
    REQUIRE(gfx->write_buffer(constant, 0, radius, sizeof radius));

    size_t compared = 0;
    const std::vector<Case> all = cases();
    for (const Case& c : all) {
        const uint32_t n = uint32_t(c.in.size()), H = c.rows;
        // Point i on the centre of pixel (i % kWidth, i / kWidth), one pixel wide.
        std::vector<float> centres(size_t(n) * 3);
        for (uint32_t i = 0; i < n; ++i) {
            centres[3 * i + 0] = 2.0f * (float(i % kWidth) + 0.5f) / float(kWidth) - 1.0f;
            centres[3 * i + 1] = 1.0f - 2.0f * (float(i / kWidth) + 0.5f) / float(H);
            centres[3 * i + 2] = 0.5f;
        }
        Buffer pos = gfx->create_buffer({centres.size() * 4, kVertex | kCopyDst, false, "centres"});
        REQUIRE(gfx->write_buffer(pos, 0, centres.data(), centres.size() * 4));
        Buffer layer_vals = gfx->create_buffer({c.in.size() * 4, kVertex | kCopyDst, false, "values"});
        REQUIRE(gfx->write_buffer(layer_vals, 0, c.in.data(), c.in.size() * 4));
        const uint64_t kOffset = 20;   // the producer's values start at byte 20
        id<MTLBuffer> producer = producer_buffer(c.in, kOffset);
        gfx->take_stats();
        const Buffer imported = import_in_place(*gfx, producer, kVertex);
        Texture color = gfx->create_texture({kWidth, H, Format::RGBA8Unorm, kRenderTarget | kCopySrcTex, "points"});
        REQUIRE(bool(color));
        for (int from_import = 0; from_import < 2; ++from_import) {
            INFO("range " << c.range.name << (from_import ? ", imported buffer" : ", layer buffer"));
            RenderTarget tg;
            tg.color = color;
            tg.clear_rgba = 0;   // no reference colour is 0: every alpha of the test table is at least 128
            DrawList list(tg);
            sh::Frame fb{};
            fb.viewport[0] = float(kWidth);
            fb.viewport[1] = float(H);
            fb.proj_scale[0] = fb.proj_scale[1] = 1.0f;
            fb.ndc_scale[0] = fb.ndc_scale[1] = 1.0f;
            PrimParams pp{};
            pp.mvp[0] = pp.mvp[5] = pp.mvp[10] = pp.mvp[15] = 1.0f;
            pp.color_mode = 1;                          // values through the colormap
            pp.size_px = 1.0f;
            pp.vmin = c.block.vmin;
            pp.vmax = c.block.vmax;
            DrawRecord r;
            r.pipeline = pipe;
            r.groups[0] = group0;
            r.groups[1] = group1;
            r.groups[2] = group2;
            r.dynamic_offsets[0] = list.uniform(&fb, sizeof fb);
            r.dynamic_offsets[1] = list.uniform(&pp, sizeof pp);
            r.dynamic_offsets[2] = list.uniform(&c.block, sizeof c.block);
            r.vertex[0] = {pos, 0};
            r.vertex[1] = from_import ? VertexBinding{imported, kOffset} : VertexBinding{layer_vals, 0};
            r.vertex[2] = {constant, 0};
            r.count = 6;
            r.instance_count = n;
            list.draw(r);
            caliper_gfx::Frame f;
            f.render(std::move(list));
            const Serial s = gfx->submit(std::move(f));
            REQUIRE_MESSAGE(s != 0, gfx->last_error());
            const std::vector<uint8_t> got = read_back(*gfx, color, kWidth, H);
            size_t first = 0;
            const size_t bad = differ(got, c.want, &first);
            CHECK_MESSAGE(bad == 0, bad, " of ", c.want.size(), " pixels differ, first input 0x", std::hex,
                          c.in[first < c.in.size() ? first : 0]);
            compared += c.want.size();
            if (&c == &all.back() && from_import) capture("colormap-keys-points-ramp.png", got, int(kWidth), int(H));
        }
        const Stats st = gfx->take_stats();
        CHECK(st.imports_in_place == 1);
        CHECK(st.copies == 0);
        CHECK(st.bytes_copied_on_gpu == 0);
        CHECK(st.draws == 2);
        gfx->release(color);
        gfx->release(imported);
        gfx->release(layer_vals);
        gfx->release(pos);
    }
    MESSAGE(compared, " points compared over ", all.size(), " cases, from a layer buffer and an imported one");
}

// ---- The timing row --------------------------------------------------------------------------

namespace {

// The float pass of the Metal renderer today (src/host/renderer/metal_renderer.mm, kColormapShaderSrc),
// copied for this measurement only: step 8 replaces it with the generated colormap_texture.
const char* kFloatPassMsl = R"metal(
#include <metal_stdlib>
using namespace metal;

struct CmapParams {
    uint  w;
    uint  h;
    uint  sx;    // element stride along x
    uint  sy;    // element stride along y
    float vmin;
    float vmax;
};

kernel void cmap_f32(device const float*  src  [[buffer(0)]],
                     device const uint*   lut  [[buffer(1)]],
                     constant CmapParams& p    [[buffer(2)]],
                     texture2d<float, access::write> dst [[texture(0)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.w || gid.y >= p.h) return;
    float v = src[gid.y * p.sy + gid.x * p.sx];
    float denom = p.vmax - p.vmin;
    float t = denom != 0.0f ? (v - p.vmin) / denom : 0.0f;
    t = clamp(t, 0.0f, 1.0f);
    uint idx = (uint)(t * 255.0f + 0.5f);
    uint packed = lut[idx];
    float4 c = float4(float(packed        & 0xffu),
                      float((packed >> 8)  & 0xffu),
                      float((packed >> 16) & 0xffu),
                      float((packed >> 24) & 0xffu)) / 255.0f;
    dst.write(c, gid);
}
)metal";

struct FloatParams { uint32_t w, h, sx, sy; float vmin, vmax; };

int binding_msl(const sh::ProgramDescription& p, uint8_t group, uint8_t binding) {
    for (uint8_t i = 0; i < p.binding_count; ++i)
        if (p.bindings[i].group == group && p.bindings[i].binding == binding) return p.bindings[i].msl_index;
    return -1;
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[size_t(p * double(v.size() - 1))];
}

}  // namespace

TEST_CASE("gfx/Metal colormap: timing — GPU time of the integer-decision pass against the float pass") {
    if (std::getenv("CALIPER_GFX_CMAP_BENCH") == nullptr) return;
    caliper_host::HostRenderer* renderer = gfx_metal_renderer();
    if (renderer == nullptr) CALIPER_SKIP_NO_GPU("no Metal device");
    id<MTLDevice> device = (__bridge id<MTLDevice>)renderer->native_device();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    auto compile = [&](const char* msl, bool fast, const char* entry) -> id<MTLComputePipelineState> {
        MTLCompileOptions* o = [MTLCompileOptions new];   // only the math mode, as the layer sets it
        if (@available(macOS 15.0, *)) {
            o.mathMode = fast ? MTLMathModeFast : MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            o.fastMathEnabled = fast;
#pragma clang diagnostic pop
        }
        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@(msl) options:o error:&err];
        if (lib == nil) return nil;
        return [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@(entry)] error:&err];
    };
    // The integer pass is the generated colormap_texture compiled as the layer compiles it (fast math
    // when its description says so) and dispatched with its own workgroup; the float pass is the
    // Metal renderer's cmap_f32, copied into this test, compiled as the renderer compiles it (default
    // options, which are fast math) with its 16 x 16 groups. A third pass, the integer one with safe
    // math, shows what the compile option is worth.
    const uint32_t prog = uint32_t(sh::ProgramId::colormap_texture);
    const sh::ProgramDescription& d = sh::kProgramDescriptions[prog];
    const bool fast = sh::kMetalPrograms[prog].fast_math;
    const uint32_t wx = d.entries[0].workgroup_size[0], wy = d.entries[0].workgroup_size[1];
    struct Pass { std::string name; id<MTLComputePipelineState> ps; bool integer; uint32_t tx, ty; };
    std::vector<Pass> passes = {
        {std::string("integer decision, as the layer compiles it (") + (fast ? "fast" : "safe") + " math, " +
             std::to_string(wx) + " x " + std::to_string(wy) + ")",
         compile(sh::kMetalPrograms[prog].msl, fast, "cs_colormap_texture"), true, wx, wy},
        {"float pass (renderer: default fast math, 16 x 16)", compile(kFloatPassMsl, true, "cmap_f32"), false, 16, 16},
        {std::string("integer decision with ") + (fast ? "safe" : "fast") + " math",
         compile(sh::kMetalPrograms[prog].msl, !fast, "cs_colormap_texture"), true, wx, wy},
    };
    for (const Pass& p : passes) REQUIRE_MESSAGE(bool(p.ps), p.name);
    const size_t kFloat = 1;
    const int at_tp = binding_msl(d, 0, 0), at_values = binding_msl(d, 0, 1), at_out = binding_msl(d, 0, 2),
              at_cmap = binding_msl(d, 1, 1);
    REQUIRE((at_tp >= 0 && at_values >= 0 && at_out >= 0 && at_cmap >= 0));
    const uint32_t* lut = caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS);
    sh::Cmap block{};
    REQUIRE(cmap::derive_block(CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, &block));
    id<MTLBuffer> cmap_buf = [device newBufferWithBytes:&block length:sizeof block options:MTLResourceStorageModeShared];
    id<MTLBuffer> lut_buf = [device newBufferWithBytes:lut length:1024 options:MTLResourceStorageModeShared];

    // Each sample is one command buffer of kPerBuffer dispatches of one pass, so that the GPU
    // stays busy and its clock settles; the passes take turns, each first in turn.
    const int kPerBuffer = 16, kWarm = 5;
    const char* samples_env = std::getenv("CALIPER_GFX_CMAP_BENCH_SAMPLES");
    const int kSamples = samples_env ? std::max(1, std::atoi(samples_env)) : 40;
    const char* fields[2] = {"smooth field", "uniform random values"};
    for (uint32_t size : {1024u, 2048u}) {
        for (int field = 0; field < 2; ++field) {
            std::vector<float> v(size_t(size) * size);
            uint64_t s = 7;
            for (uint32_t y = 0; y < size; ++y)
                for (uint32_t x = 0; x < size; ++x) {
                    float val;
                    if (field == 0) {
                        val = 0.5f + 0.5f * std::sin(float(x) * 0.0123f) * std::cos(float(y) * 0.0071f);
                    } else {
                        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                        val = float((s * 2685821657736338717ull) >> 40) / float(1u << 24);
                    }
                    v[size_t(y) * size + x] = val;
                }
            id<MTLBuffer> src = [device newBufferWithBytes:v.data() length:v.size() * 4 options:MTLResourceStorageModeShared];
            MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                          width:size height:size mipmapped:NO];
            td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
            td.storageMode = MTLStorageModeShared;
            id<MTLTexture> dst = [device newTextureWithDescriptor:td];
            const sh::TexParams tp{size, size, 0, size};
            const uint32_t sizes[1] = {uint32_t(v.size() * 4)};
            const FloatParams fp{size, size, 1, size, 0.0f, 1.0f};
            auto run = [&](const Pass& p, int n) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:p.ps];
                if (p.integer) {
                    [enc setBytes:&tp length:sizeof tp atIndex:NSUInteger(at_tp)];
                    [enc setBuffer:src offset:0 atIndex:NSUInteger(at_values)];
                    [enc setBuffer:cmap_buf offset:0 atIndex:NSUInteger(at_cmap)];
                    [enc setBytes:sizes length:sizeof sizes atIndex:sh::kMetalSizesBuffer];
                    [enc setTexture:dst atIndex:NSUInteger(at_out)];
                } else {
                    [enc setBuffer:src offset:0 atIndex:0];
                    [enc setBuffer:lut_buf offset:0 atIndex:1];
                    [enc setBytes:&fp length:sizeof fp atIndex:2];
                    [enc setTexture:dst atIndex:0];
                }
                for (int i = 0; i < n; ++i)
                    [enc dispatchThreadgroups:MTLSizeMake((size + p.tx - 1) / p.tx, (size + p.ty - 1) / p.ty, 1)
                        threadsPerThreadgroup:MTLSizeMake(p.tx, p.ty, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / n;
            };
            // Every pass gives the reference's bytes before it is timed.
            std::vector<uint8_t> ref(v.size() * 4), got(v.size() * 4);
            caliper_host::map_f32_to_rgba8(v.data(), int(size), int(size), lut, 0.0f, 1.0f, ref.data());
            for (const Pass& p : passes) {
                run(p, 1);
                [dst getBytes:got.data() bytesPerRow:size * 4 fromRegion:MTLRegionMake2D(0, 0, size, size) mipmapLevel:0];
                CHECK_MESSAGE(got == ref, p.name);
            }
            const size_t n = passes.size();
            std::vector<std::vector<double>> t(n);
            for (int i = 0; i < kWarm; ++i)
                for (const Pass& p : passes) run(p, kPerBuffer);
            for (int i = 0; i < kSamples; ++i)
                for (size_t k = 0; k < n; ++k) {
                    const size_t j = (k + size_t(i)) % n;   // each pass first in turn
                    t[j].push_back(run(passes[j], kPerBuffer));
                }
            double load[3] = {};
            getloadavg(load, 3);
            const double base = pct(t[kFloat], 0.5);
            for (size_t k = 0; k < n; ++k) {
                const double m = pct(t[k], 0.5);
                std::printf("[gfx-cmap-bench] %u x %u, %s: %-56s median %7.1f us, p95 %7.1f us per pass, %5.2fx the float "
                            "pass (%+.1f us); %d samples of %d dispatches; load %.2f\n", size, size, fields[field],
                            passes[k].name.c_str(), m, pct(t[k], 0.95), m / base, m - base, kSamples, kPerBuffer, load[0]);
            }
            std::fflush(stdout);
        }
    }
}

#endif  // CALIPER_HAVE_METAL
