// caliper_wgsl_tests: the runtime WGSL compiler (tools/shaders/caliper-wgsl) through its C
// interface, include/caliper/wgsl.h. A valid compute program validates, reports its bindings and
// translates to MSL and SPIR-V; built-in programs with one entry point translate byte for byte
// as the developer-time driver translated them; invalid, oversized and non-UTF-8 sources fail
// with a message and a place; and, in a build with the test hooks, a panic inside the library
// comes back as an error. No GPU.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "caliper/wgsl.h"

#include "gfx/shaders/generated/edge_gather_msl.h"
#include "gfx/shaders/generated/edge_gather_spv.h"
#include "gfx/shaders/generated/edge_gather_wgsl.h"
#include "gfx/shaders/generated/fill_texture_msl.h"
#include "gfx/shaders/generated/fill_texture_spv.h"
#include "gfx/shaders/generated/fill_texture_wgsl.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char kCompute[] = R"(struct Params { scale: f32, count: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(1) @binding(0) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn scale_values(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.count) {
        dst[id.x] = src[id.x] * params.scale;
    }
}
)";

// Line 3 assigns a float to a u32.
const char kInvalid[] = "@compute @workgroup_size(1)\nfn main() {\n    let x: u32 = 1.5;\n}\n";

const char kRender[] = R"(@group(0) @binding(0) var<uniform> tint: vec4<f32>;
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;
@group(1) @binding(0) var<storage, read> offsets: array<vec2<f32>>;

struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex
fn vs(@location(0) corner: vec2<f32>, @builtin(instance_index) i: u32) -> Out {
    return Out(vec4<f32>(corner + offsets[i], 0.0, 1.0), corner);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, in.uv) * tint;
}
)";

// One module, freed on scope exit.
struct Module {
    caliper_wgsl_module* m = nullptr;
    caliper_wgsl_error error{};
    caliper_wgsl_status status;
    Module(const char* source, size_t size, const caliper_wgsl_options* options = nullptr)
        : status(caliper_wgsl_module_create(source, size, options, &m, &error)) {}
    explicit Module(const char* source) : Module(source, std::strlen(source)) {}
    ~Module() {
        caliper_wgsl_module_destroy(m);
        caliper_wgsl_error_free(&error);
    }
};

std::string msl_of(const caliper_wgsl_module* m, const char* name, caliper_wgsl_stage stage) {
    caliper_wgsl_buffer b{};
    caliper_wgsl_error e{};
    caliper_wgsl_status s = caliper_wgsl_translate_msl(m, name, stage, &b, &e);
    INFO((e.report ? e.report : ""));
    REQUIRE(s == CALIPER_WGSL_OK);
    REQUIRE(b.data != nullptr);
    CHECK(b.data[b.size] == 0);
    std::string text(reinterpret_cast<const char*>(b.data), b.size);
    caliper_wgsl_buffer_free(&b);
    CHECK(b.data == nullptr);
    caliper_wgsl_error_free(&e);
    return text;
}

std::vector<uint32_t> spirv_of(const caliper_wgsl_module* m, const char* name, caliper_wgsl_stage stage) {
    caliper_wgsl_buffer b{};
    caliper_wgsl_error e{};
    caliper_wgsl_status s = caliper_wgsl_translate_spirv(m, name, stage, &b, &e);
    INFO((e.report ? e.report : ""));
    REQUIRE(s == CALIPER_WGSL_OK);
    REQUIRE(b.size % 4 == 0);
    std::vector<uint32_t> words(b.size / 4);
    std::memcpy(words.data(), b.data, b.size);
    caliper_wgsl_buffer_free(&b);
    caliper_wgsl_error_free(&e);
    return words;
}

}  // namespace

TEST_CASE("versions") {
    CHECK(caliper_wgsl_abi_version() == CALIPER_WGSL_ABI_VERSION);
    CHECK(std::string(caliper_wgsl_naga_version()) == "30.0.2");
    caliper_wgsl_options o{};
    caliper_wgsl_options_init(&o);
    CHECK(o.max_source_bytes == (1u << 20));
    CHECK(o.msl_version_major == 2);
    CHECK(o.msl_version_minor == 4);
    CHECK(o.spirv_version_major == 1);
    CHECK(o.spirv_version_minor == 3);
    CHECK(o.metal_vertex_buffer_base == 16);
}

TEST_CASE("a compute program validates, reports its bindings and translates") {
    Module mod(kCompute);
    INFO((mod.error.report ? mod.error.report : ""));
    REQUIRE(mod.status == CALIPER_WGSL_OK);
    CHECK(mod.error.message == nullptr);

    REQUIRE(caliper_wgsl_module_entry_point_count(mod.m) == 1);
    caliper_wgsl_entry_point ep{};
    REQUIRE(caliper_wgsl_module_entry_point(mod.m, 0, &ep) == CALIPER_WGSL_OK);
    CHECK(std::string(ep.name) == "scale_values");
    CHECK(ep.stage == CALIPER_WGSL_STAGE_COMPUTE);
    CHECK(ep.workgroup_size[0] == 64);
    CHECK(ep.workgroup_size[1] == 1);
    CHECK(ep.workgroup_size[2] == 1);

    REQUIRE(caliper_wgsl_module_binding_count(mod.m) == 3);
    caliper_wgsl_binding b[3]{};
    for (size_t i = 0; i < 3; ++i) REQUIRE(caliper_wgsl_module_binding(mod.m, i, &b[i]) == CALIPER_WGSL_OK);
    CHECK(std::string(b[0].name) == "params");
    CHECK(b[0].group == 0);
    CHECK(b[0].binding == 0);
    CHECK(b[0].kind == CALIPER_WGSL_BINDING_UNIFORM_BUFFER);
    CHECK(b[0].access == CALIPER_WGSL_ACCESS_READ);
    CHECK(b[0].size == 8);
    CHECK(std::string(b[1].name) == "src");
    CHECK(b[1].kind == CALIPER_WGSL_BINDING_STORAGE_BUFFER);
    CHECK(b[1].access == CALIPER_WGSL_ACCESS_READ);
    CHECK(std::string(b[2].name) == "dst");
    CHECK(b[2].group == 1);
    CHECK(b[2].binding == 0);
    CHECK(b[2].access == (CALIPER_WGSL_ACCESS_READ | CALIPER_WGSL_ACCESS_WRITE));
    for (size_t i = 0; i < 3; ++i) {
        CHECK(b[i].stages == (1u << CALIPER_WGSL_STAGE_COMPUTE));
        CHECK(b[i].metal_index == i);
        CHECK(b[i].storage_format == nullptr);
    }
    caliper_wgsl_binding past{};
    CHECK(caliper_wgsl_module_binding(mod.m, 3, &past) == CALIPER_WGSL_ERROR_ARGUMENT);

    std::string msl = msl_of(mod.m, "scale_values", CALIPER_WGSL_STAGE_COMPUTE);
    CHECK(msl.rfind("// language: metal2.4", 0) == 0);
    CHECK(msl.find("kernel void scale_values(") != std::string::npos);
    CHECK(msl.find("[[buffer(0)]]") != std::string::npos);
    CHECK(msl.find("[[buffer(2)]]") != std::string::npos);

    std::vector<uint32_t> spv = spirv_of(mod.m, "scale_values", CALIPER_WGSL_STAGE_COMPUTE);
    REQUIRE(spv.size() > 5);
    CHECK(spv[0] == 0x07230203u);  // the SPIR-V magic number
    CHECK(spv[1] == 0x00010300u);  // version 1.3

    caliper_wgsl_buffer none{};
    caliper_wgsl_error e{};
    CHECK(caliper_wgsl_translate_msl(mod.m, "scale_values", CALIPER_WGSL_STAGE_VERTEX, &none, &e) ==
          CALIPER_WGSL_ERROR_ARGUMENT);
    CHECK(std::string(e.message) == "the module has no vertex entry point named scale_values");
    CHECK(none.data == nullptr);
    caliper_wgsl_error_free(&e);
    CHECK(caliper_wgsl_translate_spirv(mod.m, "missing", CALIPER_WGSL_STAGE_COMPUTE, &none, &e) ==
          CALIPER_WGSL_ERROR_ARGUMENT);
    caliper_wgsl_error_free(&e);
    CHECK(caliper_wgsl_translate_spirv(mod.m, "scale_values", static_cast<caliper_wgsl_stage>(9), &none, &e) ==
          CALIPER_WGSL_ERROR_ARGUMENT);
    caliper_wgsl_error_free(&e);
}

TEST_CASE("a render program numbers Metal bindings per kind across its stages") {
    Module mod(kRender);
    INFO((mod.error.report ? mod.error.report : ""));
    REQUIRE(mod.status == CALIPER_WGSL_OK);
    REQUIRE(caliper_wgsl_module_binding_count(mod.m) == 4);
    caliper_wgsl_binding b[4]{};
    for (size_t i = 0; i < 4; ++i) REQUIRE(caliper_wgsl_module_binding(mod.m, i, &b[i]) == CALIPER_WGSL_OK);
    CHECK(b[0].kind == CALIPER_WGSL_BINDING_UNIFORM_BUFFER);
    CHECK(b[0].metal_index == 0);
    CHECK(b[0].stages == (1u << CALIPER_WGSL_STAGE_FRAGMENT));
    CHECK(b[1].kind == CALIPER_WGSL_BINDING_TEXTURE);
    CHECK(b[1].metal_index == 0);
    CHECK(b[2].kind == CALIPER_WGSL_BINDING_SAMPLER);
    CHECK(b[2].metal_index == 0);
    CHECK(b[3].kind == CALIPER_WGSL_BINDING_STORAGE_BUFFER);
    CHECK(b[3].metal_index == 1);
    CHECK(b[3].stages == (1u << CALIPER_WGSL_STAGE_VERTEX));

    std::string vs = msl_of(mod.m, "vs", CALIPER_WGSL_STAGE_VERTEX);
    CHECK(vs.find("vertex ") != std::string::npos);
    CHECK(vs.find(" vs(") != std::string::npos);
    CHECK(vs.find("fragment") == std::string::npos);
    std::string fs = msl_of(mod.m, "fs", CALIPER_WGSL_STAGE_FRAGMENT);
    CHECK(fs.find("[[texture(0)]]") != std::string::npos);
    CHECK(fs.find("[[sampler(0)]]") != std::string::npos);
    CHECK(spirv_of(mod.m, "fs", CALIPER_WGSL_STAGE_FRAGMENT).size() > 5);
}

TEST_CASE("built-in programs translate as the developer-time driver translated them") {
    struct Builtin {
        const char* wgsl;
        const char* msl;
        const uint32_t* spirv;
        size_t spirv_words;
    };
    using namespace caliper_gfx::shaders;
    const Builtin programs[] = {
        {kFillTextureWgsl, kFillTextureMsl, kFillTextureSpirv, sizeof(kFillTextureSpirv) / 4},
        {kEdgeGatherWgsl, kEdgeGatherMsl, kEdgeGatherSpirv, sizeof(kEdgeGatherSpirv) / 4},
    };
    for (const Builtin& p : programs) {
        Module mod(p.wgsl);
        INFO((mod.error.report ? mod.error.report : ""));
        REQUIRE(mod.status == CALIPER_WGSL_OK);
        REQUIRE(caliper_wgsl_module_entry_point_count(mod.m) == 1);
        caliper_wgsl_entry_point ep{};
        REQUIRE(caliper_wgsl_module_entry_point(mod.m, 0, &ep) == CALIPER_WGSL_OK);
        INFO(ep.name);
        auto stage = static_cast<caliper_wgsl_stage>(ep.stage);
        CHECK(msl_of(mod.m, ep.name, stage) == std::string(p.msl));
        CHECK(spirv_of(mod.m, ep.name, stage) == std::vector<uint32_t>(p.spirv, p.spirv + p.spirv_words));
    }
}

TEST_CASE("an invalid program fails with its message, line and column") {
    Module mod(kInvalid);
    CHECK(mod.status == CALIPER_WGSL_ERROR_PARSE);
    CHECK(mod.m == nullptr);
    CHECK(mod.error.status == CALIPER_WGSL_ERROR_PARSE);
    CHECK(mod.error.line == 3);
    CHECK(mod.error.column == 9);  // the name x
    REQUIRE(mod.error.message != nullptr);
    REQUIRE(mod.error.report != nullptr);
    MESSAGE(std::string(mod.error.report));
    CHECK(std::string(mod.error.message).find("u32") != std::string::npos);
    CHECK(std::string(mod.error.report).find("let x: u32 = 1.5;") != std::string::npos);

    Module syntax("fn main( {\n");
    CHECK(syntax.status == CALIPER_WGSL_ERROR_PARSE);
    CHECK(syntax.error.line == 1);
    CHECK(syntax.error.column > 0);

    // Deep nesting stops at Naga's parser limit instead of exhausting the stack.
    std::string deep = "fn f() -> f32 { return " + std::string(20000, '(') + "1.0" + std::string(20000, ')') + "; }\n";
    Module nested(deep.c_str());
    CHECK(nested.status == CALIPER_WGSL_ERROR_PARSE);
}

TEST_CASE("limits and arguments are checked") {
    caliper_wgsl_options o{};
    caliper_wgsl_options_init(&o);
    o.max_source_bytes = 64;
    Module big(kCompute, sizeof(kCompute) - 1, &o);
    CHECK(big.status == CALIPER_WGSL_ERROR_TOO_LARGE);
    CHECK(big.m == nullptr);
    CHECK(std::string(big.error.message).find("more than the limit of 64") != std::string::npos);

    const char bad_utf8[] = "fn f() {}\n// \xff\n";
    Module text(bad_utf8, sizeof(bad_utf8) - 1);
    CHECK(text.status == CALIPER_WGSL_ERROR_PARSE);
    CHECK(text.error.line == 2);
    CHECK(text.error.column == 4);

    caliper_wgsl_options_init(&o);
    o.metal_vertex_buffer_base = 40;
    Module options(kCompute, sizeof(kCompute) - 1, &o);
    CHECK(options.status == CALIPER_WGSL_ERROR_ARGUMENT);

    caliper_wgsl_options_init(&o);
    o.max_source_bytes = static_cast<size_t>(CALIPER_WGSL_MAX_SOURCE_BYTES_LIMIT) + 1;
    Module above(kCompute, sizeof(kCompute) - 1, &o);
    CHECK(above.status == CALIPER_WGSL_ERROR_ARGUMENT);
    CHECK(above.m == nullptr);

    // A claimed size of SIZE_MAX is refused before the library reads the one byte behind source.
    const char one = 'a';
    Module huge(&one, SIZE_MAX);
    CHECK(huge.status == CALIPER_WGSL_ERROR_TOO_LARGE);
    CHECK(huge.m == nullptr);

    Module overrides("override k: f32 = 1.0;\n@compute @workgroup_size(1) fn main() { _ = k; }\n");
    CHECK(overrides.status == CALIPER_WGSL_ERROR_UNSUPPORTED);

    Module cube("@group(0) @binding(0) var t: texture_cube<f32>;\n"
                "@group(0) @binding(1) var s: sampler;\n"
                "@fragment fn fs() -> @location(0) vec4<f32> { return textureSample(t, s, vec3<f32>(1.0)); }\n");
    CHECK(cube.status == CALIPER_WGSL_ERROR_UNSUPPORTED);

    caliper_wgsl_error e{};
    CHECK(caliper_wgsl_module_create(kCompute, sizeof(kCompute) - 1, nullptr, nullptr, &e) == CALIPER_WGSL_ERROR_ARGUMENT);
    caliper_wgsl_error_free(&e);
    caliper_wgsl_error_free(&e);  // a second free is a no-op
    caliper_wgsl_error_free(nullptr);
    caliper_wgsl_buffer_free(nullptr);
    caliper_wgsl_module_destroy(nullptr);
    CHECK(caliper_wgsl_module_binding_count(nullptr) == 0);
    caliper_wgsl_buffer b{};
    CHECK(caliper_wgsl_translate_msl(nullptr, "main", CALIPER_WGSL_STAGE_COMPUTE, &b, nullptr) == CALIPER_WGSL_ERROR_ARGUMENT);
}

#ifdef CALIPER_WGSL_TEST_HOOKS
TEST_CASE("a panic inside the library returns an error and the library keeps working") {
    caliper_wgsl_error e{};
    CHECK(caliper_wgsl_test_panic(&e) == CALIPER_WGSL_ERROR_INTERNAL);
    CHECK(e.status == CALIPER_WGSL_ERROR_INTERNAL);
    REQUIRE(e.message != nullptr);
    CHECK(std::string(e.message).find("caliper_wgsl_test_panic") != std::string::npos);
    caliper_wgsl_error_free(&e);
    CHECK(caliper_wgsl_test_panic(nullptr) == CALIPER_WGSL_ERROR_INTERNAL);

    Module after(kCompute);
    CHECK(after.status == CALIPER_WGSL_OK);
}
#endif  // CALIPER_WGSL_TEST_HOOKS
