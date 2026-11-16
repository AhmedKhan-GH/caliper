// caliper_shader_tests: the generated shader programs on Metal, headless (no window).
//
// 1. Every generated MSL program compiles on the default device, and every entry becomes a
//    pipeline: each compute entry a compute pipeline state, and each vertex and fragment pair
//    of the manifest a render pipeline state, with a vertex descriptor from the program's
//    vertex inputs (slot i at Metal buffer kMetalVertexBufferBase + i), its fragment output's
//    format, and once more with a depth attachment.
// 2. The colormap by keys, through the generated MSL of the colormap probe, matches the CPU
//    reference map_f32_to_rgba8 byte for byte on the boundary cases of colormap_cases.h, with
//    the compiler's fast math on and off.
//
// CALIPER_COLORMAP_CASES_OUT=<file> also writes the cases (block, inputs, expected RGBA per
// range) for the browser test of the same comparison (web/test/shaders.spec.ts).
#define DOCTEST_CONFIG_IMPLEMENT
#include "../gpu_skip.h"
#include "colormap_cases.h"
#include "gfx/shaders/generated/programs_metal.h"

#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace caliper_gfx;
using namespace caliper_gfx::shaders;

namespace {

id<MTLDevice> device() {
    static id<MTLDevice> d = caliper_test::force_no_gpu() ? nil : MTLCreateSystemDefaultDevice();
    return d;
}

id<MTLLibrary> compile(uint32_t program, bool fast_math, std::string* error) {
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion2_4;   // the generator's MSL version
    if (@available(macOS 15.0, *)) {
        o.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        o.fastMathEnabled = fast_math;
#pragma clang diagnostic pop
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [device() newLibraryWithSource:@(kMetalPrograms[program].msl) options:o error:&err];
    if (!lib && error) *error = err ? [[err localizedDescription] UTF8String] : "unknown error";
    return lib;
}

MTLVertexFormat metal_format(VertexFormat f) {
    switch (f) {
        case VertexFormat::F32: return MTLVertexFormatFloat;
        case VertexFormat::F32x2: return MTLVertexFormatFloat2;
        case VertexFormat::F32x3: return MTLVertexFormatFloat3;
        case VertexFormat::F32x4: return MTLVertexFormatFloat4;
        case VertexFormat::U32: return MTLVertexFormatUInt;
        case VertexFormat::S32: return MTLVertexFormatInt;
        case VertexFormat::U32x2: return MTLVertexFormatUInt2;
        default: return MTLVertexFormatInvalid;
    }
}

uint32_t format_bytes(VertexFormat f) {   // the test's own; gfx.h's needs the layer's library
    switch (f) {
        case VertexFormat::F32x2: case VertexFormat::U32x2: return 8;
        case VertexFormat::F32x3: return 12;
        case VertexFormat::F32x4: return 16;
        default: return 4;
    }
}

MTLPixelFormat metal_format(Format f) {
    switch (f) {
        case Format::RGBA8Unorm: return MTLPixelFormatRGBA8Unorm;
        case Format::RG32Uint: return MTLPixelFormatRG32Uint;
        case Format::R32Uint: return MTLPixelFormatR32Uint;
        case Format::R32Float: return MTLPixelFormatR32Float;
        default: return MTLPixelFormatInvalid;
    }
}

int binding_index(const ProgramDescription& p, uint8_t group, uint8_t binding) {
    for (uint8_t i = 0; i < p.binding_count; ++i)
        if (p.bindings[i].group == group && p.bindings[i].binding == binding) return p.bindings[i].msl_index;
    return -1;
}

}  // namespace

TEST_CASE("every generated MSL program compiles and every entry and pair becomes a pipeline on Metal") {
    if (!device()) { CALIPER_SKIP_NO_GPU("no Metal device"); }
    int libraries = 0, computes = 0, renders = 0;
    for (uint32_t i = 0; i < kProgramCount; ++i) {
        const ProgramDescription& p = kProgramDescriptions[i];
        INFO("program " << p.name);
        std::string error;
        id<MTLLibrary> lib = compile(i, false, &error);
        REQUIRE_MESSAGE(bool(lib), error);
        ++libraries;
        for (uint8_t e = 0; e < p.entry_count; ++e) {
            id<MTLFunction> fn = [lib newFunctionWithName:@(p.entries[e].name)];
            CHECK_MESSAGE(bool(fn), p.entries[e].name);
            if (fn && p.entries[e].stage == kStageCompute) {
                NSError* err = nil;
                id<MTLComputePipelineState> ps = [device() newComputePipelineStateWithFunction:fn error:&err];
                CHECK_MESSAGE(bool(ps), p.entries[e].name << ": " << (err ? [[err localizedDescription] UTF8String] : ""));
                computes += ps != nil;
            }
        }
        for (uint8_t k = 0; k < p.pair_count; ++k) {
            const PairDesc& pair = p.pairs[k];
            MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
            for (uint8_t v = 0; v < p.vertex_input_count; ++v) {
                const VertexInputDesc& in = p.vertex_inputs[v];
                if (std::strcmp(in.entry, pair.vertex) != 0) continue;
                const NSUInteger slot = kMetalVertexBufferBase + in.location;
                vd.attributes[in.location].format = metal_format(in.format);
                vd.attributes[in.location].offset = 0;
                vd.attributes[in.location].bufferIndex = slot;
                vd.layouts[slot].stride = format_bytes(in.format);
                vd.layouts[slot].stepFunction = MTLVertexStepFunctionPerInstance;
            }
            MTLPixelFormat color = MTLPixelFormatInvalid;
            for (uint8_t o = 0; o < p.fragment_output_count; ++o)
                if (std::strcmp(p.fragment_outputs[o].entry, pair.fragment) == 0) color = metal_format(p.fragment_outputs[o].format);
            for (int depth = 0; depth < 2; ++depth) {
                MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
                d.vertexFunction = [lib newFunctionWithName:@(pair.vertex)];
                d.fragmentFunction = [lib newFunctionWithName:@(pair.fragment)];
                d.vertexDescriptor = vd;
                d.colorAttachments[0].pixelFormat = color;
                if (depth) d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
                NSError* err = nil;
                id<MTLRenderPipelineState> ps = [device() newRenderPipelineStateWithDescriptor:d error:&err];
                CHECK_MESSAGE(bool(ps), pair.vertex << " + " << pair.fragment << ": "
                                             << (err ? [[err localizedDescription] UTF8String] : ""));
                renders += ps != nil;
            }
        }
    }
    MESSAGE(libraries << " libraries, " << computes << " compute pipelines, " << renders << " render pipelines");
    CHECK(libraries == int(kProgramCount));
}

TEST_CASE("the colormap by keys matches map_f32_to_rgba8 byte for byte through the generated MSL") {
    if (!device()) { CALIPER_SKIP_NO_GPU("no Metal device"); }
    const uint32_t prog = uint32_t(ProgramId::colormap_probe);
    const ProgramDescription& p = kProgramDescriptions[prog];
    const int cmap_at = binding_index(p, 1, 1), in_at = binding_index(p, 3, 0), out_at = binding_index(p, 3, 1);
    REQUIRE(cmap_at >= 0); REQUIRE(in_at >= 0); REQUIRE(out_at >= 0);
    const std::vector<uint32_t> table = colormap_cases::lut();
    id<MTLCommandQueue> queue = [device() newCommandQueue];

    FILE* out = nullptr;
    if (const char* path = std::getenv("CALIPER_COLORMAP_CASES_OUT")) {
        out = std::fopen(path, "wb");
        REQUIRE(out != nullptr);
        const auto n = uint32_t(colormap_cases::ranges().size());
        std::fwrite("CMAPCASE", 1, 8, out);
        std::fwrite(&n, 4, 1, out);
    }
    size_t total = 0, keys_seen = 0;
    for (int fast = 0; fast < 2; ++fast) {
        std::string error;
        id<MTLLibrary> lib = compile(prog, fast != 0, &error);
        REQUIRE_MESSAGE(bool(lib), error);
        NSError* err = nil;
        id<MTLComputePipelineState> ps =
            [device() newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cs_colormap_probe"] error:&err];
        REQUIRE(bool(ps));
        uint64_t seed = 0x9e3779b97f4a7c15ull;
        for (const auto& r : colormap_cases::ranges()) {
            INFO("range " << r.name << ", fast math " << (fast ? "on" : "off"));
            const Cmap block = colormap_cases::block(r, table);
            const std::vector<uint32_t> in = colormap_cases::inputs(r, block, seed++);
            const std::vector<uint32_t> want = colormap_cases::expected(r, in, table);
            const uint32_t sizes[2] = {uint32_t(in.size() * 4), uint32_t(in.size() * 4)};
            id<MTLBuffer> bin = [device() newBufferWithBytes:in.data() length:in.size() * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bout = [device() newBufferWithLength:in.size() * 4 options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBytes:&block length:sizeof(block) atIndex:NSUInteger(cmap_at)];
            [enc setBuffer:bin offset:0 atIndex:NSUInteger(in_at)];
            [enc setBuffer:bout offset:0 atIndex:NSUInteger(out_at)];
            [enc setBytes:sizes length:sizeof(sizes) atIndex:kMetalSizesBuffer];
            [enc dispatchThreadgroups:MTLSizeMake(in.size() / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
            const auto* got = static_cast<const uint32_t*>(bout.contents);
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < in.size(); ++i)
                if (got[i] != want[i] && bad++ == 0) first = i;
            CHECK_MESSAGE(bad == 0, bad << " of " << in.size() << " differ, first input 0x" << std::hex << in[first]
                                        << ": got 0x" << got[first] << ", want 0x" << want[first]);
            total += in.size();
            if (fast == 0) {
                for (int k = 1; k < 256; ++k) keys_seen += (&block.keys[0][0])[k] != 0xffffffffu;
                if (out) {
                    const auto n = uint32_t(in.size());
                    std::fwrite(&block, sizeof(block), 1, out);
                    std::fwrite(&n, 4, 1, out);
                    std::fwrite(in.data(), 4, in.size(), out);
                    std::fwrite(want.data(), 4, want.size(), out);
                }
            }
        }
    }
    if (out) std::fclose(out);
    MESSAGE(total << " inputs compared over " << colormap_cases::ranges().size() << " ranges, twice; "
                  << keys_seen << " reachable key edges");
}

CALIPER_GPU_TEST_MAIN()
