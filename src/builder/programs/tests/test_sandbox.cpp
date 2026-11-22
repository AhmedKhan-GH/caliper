// Tests of the sandbox of GPU programs and stages (sandbox.h,
// design/gpu-programs.md, Sandbox): the value types and the stage contract,
// the declaration limits, provenance, the source scan, the binding checks
// against a reflection made by hand (which need no compiler), and, when the
// build has caliper-wgsl, the same checks on reflections of real sources.
#include <caliper_builder/programs/sandbox.h>
#include <array>
#include <cstring>

#include <doctest/doctest.h>

#include <string>

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;

namespace {

// The command table's example of program.create: one point for each row of a
// weight matrix.
const char* const kRows =
    "@group(0) @binding(0) var<storage, read> w: array<f32>;\n"
    "@group(0) @binding(1) var<storage, read_write> positions: array<f32>;\n"
    "@group(0) @binding(2) var<storage, read_write> count: atomic<u32>;\n"
    "@compute @workgroup_size(64) fn fill(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "  let i = id.x;\n"
    "  if (i >= 256u) { return; }\n"
    "  positions[3u * i] = w[3u * i];\n"
    "  positions[3u * i + 1u] = w[3u * i + 1u];\n"
    "  positions[3u * i + 2u] = w[3u * i + 2u];\n"
    "  atomicAdd(&count, 1u);\n"
    "}\n";

cs::Program geometry_program() {
    cs::Program p;
    p.id = "rows";
    p.wgsl = kRows;
    p.inputs = {{"w", "live:layers.0.weight"}};
    p.output.kind = cs::OutputKind::geometry;
    p.output.panel = "world";
    p.output.max_count = 256;
    p.output.arrays = {cs::GeometryArray::positions};
    return p;
}

cs::Program pixels_program() {
    cs::Program p;
    p.id = "image";
    p.wgsl = "@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;\n"
             "@compute @workgroup_size(8, 8) fn fill(@builtin(global_invocation_id) id: vec3<u32>) {\n"
             "  if (id.x >= 64u || id.y >= 32u) { return; }\n"
             "  pixels[id.y * 64u + id.x] = pack4x8unorm(vec4<f32>(1.0, 0.0, 0.0, 1.0));\n"
             "}\n";
    p.output.kind = cs::OutputKind::pixels;
    p.output.width = 64;
    p.output.height = 32;
    return p;
}

cp::ReflectedBinding binding(std::string name, cp::BindingKind kind, bool write, std::uint64_t size = 0) {
    cp::ReflectedBinding b;
    b.name = std::move(name);
    b.kind = kind;
    b.read = true;
    b.write = write;
    b.size = size;
    return b;
}

// The reflection of kRows, as caliper-wgsl gives it.
cp::Reflection rows_reflection() {
    cp::Reflection r;
    r.bindings = {binding("w", cp::BindingKind::storage_buffer, false),
                  binding("positions", cp::BindingKind::storage_buffer, true),
                  binding("count", cp::BindingKind::storage_buffer, true, 4)};
    r.entries = {{"fill", cp::EntryStage::compute, {64, 1, 1}}};
    return r;
}

cp::Reflection pixels_reflection() {
    cp::Reflection r;
    r.bindings = {binding("pixels", cp::BindingKind::storage_buffer, true)};
    r.entries = {{"fill", cp::EntryStage::compute, {8, 8, 1}}};
    return r;
}

}  // namespace

TEST_CASE("the program value types name their kinds and fixed bindings") {
    CHECK(cs::output_kind_from_name(cs::output_kind_name(cs::OutputKind::pixels)) == cs::OutputKind::pixels);
    CHECK(cs::geometry_part_from_name("lines") == cs::GeometryPart::lines);
    CHECK_FALSE(cs::geometry_part_from_name("mesh"));
    CHECK(cs::geometry_array_from_name("radii") == cs::GeometryArray::radii);
    CHECK(cs::geometry_array_words(cs::GeometryArray::positions) == 3);
    CHECK(cs::stage_kind_from_name("vertex") == cs::StageKind::vertex);
    CHECK_FALSE(cs::stage_kind_from_name("compute"));
    CHECK(cs::default_entity("rows") == "programs/rows");
    CHECK(cs::program_view_name("rows") == "program.rows");
    cs::Stage s;
    s.panel = "world";
    s.entity = "programs/rows";
    CHECK(cs::stage_id(s) == "world/programs/rows/material");
    CHECK(cs::stage_target(s) == "stage:world/programs/rows/material");
    cs::ProgramLimits limits;
    CHECK(limits.time_ms_or_default() == cs::program_limits::kTimeMsDefault);
    CHECK(limits.memory_bytes_or_default() == (std::uint64_t(64) << 20));
    CHECK(cs::Provenance{}.empty());
}

TEST_CASE("the stage prelude declares the three structs of the contract on its stated lines") {
    const std::string_view p = cp::stage_prelude();
    std::uint32_t lines = 0;
    for (char c : p) lines += c == '\n';
    CHECK(lines == cp::stage_prelude_lines());
    for (const char* name : {"struct MaterialIn", "struct VertexIn", "struct VertexOut"})
        CHECK(p.find(name) != std::string_view::npos);
}

TEST_CASE("sha256_hex gives the FIPS 180-4 digests") {
    CHECK(cp::sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(cp::sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(cp::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(cp::sha256_hex(std::string(1000, 'a')) ==
          "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

TEST_CASE("the declaration check passes a geometry and a pixels program and refuses each limit") {
    CHECK(cp::check_declaration(geometry_program()).ok());
    CHECK(cp::check_declaration(pixels_program()).ok());

    struct Case {
        const char* name;
        void (*change)(cs::Program&);
        const char* code;
        const char* where;
    };
    const Case cases[] = {
        {"id", [](cs::Program& p) { p.id = "Rows"; }, cp::kProgramOverLimits, "id"},
        {"source size", [](cs::Program& p) { p.wgsl = std::string(64 * 1024 + 1, ' '); }, cp::kProgramTooLarge, "wgsl"},
        {"nine inputs", [](cs::Program& p) { p.inputs.assign(9, {"w", "live:w"}); }, cp::kProgramOverLimits, "inputs"},
        {"input source", [](cs::Program& p) { p.inputs[0].source = "layers.0.weight"; }, cp::kProgramOverLimits,
         "inputs[0].source"},
        {"input var", [](cs::Program& p) { p.inputs[0].var = "0w"; }, cp::kProgramOverLimits, "inputs[0].var"},
        {"input named count", [](cs::Program& p) { p.inputs[0].var = "count"; }, cp::kProgramOutputMismatch,
         "inputs[0].var"},
        {"two inputs, one var", [](cs::Program& p) { p.inputs.push_back({"w", "live:b"}); }, cp::kProgramOverLimits,
         "inputs[1].var"},
        {"no panel", [](cs::Program& p) { p.output.panel.clear(); }, cp::kProgramBadTarget, "output.panel"},
        {"entity", [](cs::Program& p) { p.output.entity = "a//b"; }, cp::kProgramBadTarget, "output.entity"},
        {"max_count 0", [](cs::Program& p) { p.output.max_count = 0; }, cp::kProgramOverLimits, "output.max_count"},
        {"max_count above", [](cs::Program& p) { p.output.max_count = 1000001; }, cp::kProgramOverLimits,
         "output.max_count"},
        {"no positions", [](cs::Program& p) { p.output.arrays = {cs::GeometryArray::colors}; },
         cp::kProgramOutputMismatch, "output.arrays"},
        {"radii on lines",
         [](cs::Program& p) {
             p.output.part = cs::GeometryPart::lines;
             p.output.arrays.push_back(cs::GeometryArray::radii);
         },
         cp::kProgramOutputMismatch, "output.arrays"},
        {"width on geometry", [](cs::Program& p) { p.output.width = 4; }, cp::kProgramOutputMismatch, "output.width"},
        {"time_ms", [](cs::Program& p) { p.limits.time_ms = 51; }, cp::kProgramOverLimits, "limits.time_ms"},
        {"memory_mb", [](cs::Program& p) { p.limits.memory_mb = 513; }, cp::kProgramOverLimits, "limits.memory_mb"},
        {"every 0", [](cs::Program& p) { p.limits.every = 0u; }, cp::kProgramOverLimits, "limits.every"},
        {"every above", [](cs::Program& p) { p.limits.every = 10001u; }, cp::kProgramOverLimits, "limits.every"},
        {"workgroups", [](cs::Program& p) { p.workgroups = std::array<std::uint32_t, 3>{70000, 1, 1}; },
         cp::kProgramOverLimits, "workgroups"},
        {"outputs above the budget",
         [](cs::Program& p) {
             p.output.max_count = 1000000;
             p.limits.memory_mb = 1u;
         },
         cp::kProgramOverLimits, "limits.memory_mb"},
    };
    for (const Case& c : cases) {
        CAPTURE(c.name);
        cs::Program p = geometry_program();
        c.change(p);
        const cp::Verdict v = cp::check_declaration(p);
        CHECK(v.code == c.code);
        CHECK(v.where == c.where);
        CHECK_FALSE(v.message.empty());
    }

    cs::Program big = pixels_program();
    big.output.width = 2049;
    CHECK(cp::check_declaration(big).code == cp::kProgramOverLimits);
    cs::Program mixed = pixels_program();
    mixed.output.panel = "world";
    CHECK(cp::check_declaration(mixed).code == cp::kProgramOutputMismatch);
}

TEST_CASE("the declaration of a stage checks its panel, its entity and its size") {
    cs::Stage s;
    s.panel = "world";
    s.entity = "programs/rows";
    s.wgsl = "fn material(in: MaterialIn) -> vec4<f32> { return in.color; }\n";
    CHECK(cp::check_stage_declaration(s).ok());
    cs::Stage bad = s;
    bad.panel = "World";
    CHECK(cp::check_stage_declaration(bad).code == cp::kStageNotScenePanel);
    bad = s;
    bad.entity = "programs/";
    CHECK(cp::check_stage_declaration(bad).code == cp::kProgramBadTarget);
    bad = s;
    bad.wgsl = std::string(16 * 1024 + 1, ' ');
    CHECK(cp::check_stage_declaration(bad).code == cp::kProgramTooLarge);
}

TEST_CASE("a foreign spec runs a source only when its hash is reviewed") {
    const std::string hash = cp::sha256_hex(kRows);
    cs::Provenance own;
    CHECK(cp::check_provenance(own, "program:rows", hash).ok());
    cs::Provenance foreign;
    foreign.foreign = true;
    foreign.from = "a shared spec";
    CHECK(cp::check_provenance(foreign, "program:rows", hash).code == cp::kProgramNeedsReview);
    foreign.reviewed.push_back({"program:rows", hash});
    CHECK(cp::check_provenance(foreign, "program:rows", hash).ok());
    CHECK(cp::check_provenance(foreign, "program:rows", cp::sha256_hex("changed")).code == cp::kProgramNeedsReview);
    CHECK(cp::check_provenance(foreign, "program:other", hash).code == cp::kProgramNeedsReview);
}

TEST_CASE("the source check refuses a directive with its place and the size limit") {
    CHECK(cp::check_source(kRows, cs::program_limits::kProgramSourceBytes).ok());
    const cp::Verdict enable = cp::check_source("// half floats\n  enable f16;\nfn f() {}\n", 4096);
    CHECK(enable.code == cp::kProgramDirectiveNotAllowed);
    CHECK(enable.line == 2);
    CHECK(enable.column == 3);
    CHECK(cp::check_source("/* a /* nested */ comment */ requires readonly_and_readwrite_storage_textures;", 4096).code ==
          cp::kProgramDirectiveNotAllowed);
    CHECK(cp::check_source("diagnostic(off, derivative_uniformity);", 4096).code == cp::kProgramDirectiveNotAllowed);
    // The attribute of a function is not a directive.
    CHECK(cp::check_source("@diagnostic(off, derivative_uniformity) fn f() {}", 4096).ok());
    CHECK(cp::check_source(std::string(4097, ' '), 4096).code == cp::kProgramTooLarge);
}

TEST_CASE("the binding check of a program refuses a write to an input and every undeclared binding") {
    const cs::Program rows = geometry_program();
    CHECK(cp::check_program(rows, rows_reflection()).ok());
    CHECK(cp::check_program(pixels_program(), pixels_reflection()).ok());

    // A read_write input: refused, at the declaration of its variable.
    cs::Program writes = rows;
    writes.wgsl = std::string(kRows).replace(std::string(kRows).find("read> w"), 7, "read_write> w");
    cp::Reflection r = rows_reflection();
    r.bindings[0].write = true;
    cp::Verdict v = cp::check_program(writes, r);
    CHECK(v.code == cp::kProgramWritesInput);
    CHECK(v.where == "w");
    CHECK(v.line == 1);
    CHECK(v.column == 48);
    // An input bound as a uniform is no storage read either.
    r = rows_reflection();
    r.bindings[0].kind = cp::BindingKind::uniform_buffer;
    CHECK(cp::check_program(rows, r).code == cp::kProgramWritesInput);

    r = rows_reflection();
    r.bindings.push_back(binding("secret", cp::BindingKind::storage_buffer, false));
    CHECK(cp::check_program(rows, r).code == cp::kProgramBindingUndeclared);
    r = rows_reflection();
    r.bindings.push_back(binding("colors", cp::BindingKind::storage_buffer, true));
    CHECK(cp::check_program(rows, r).code == cp::kProgramBindingUndeclared);
    r = rows_reflection();
    r.bindings.push_back(binding("pixels", cp::BindingKind::storage_buffer, true));
    CHECK(cp::check_program(rows, r).code == cp::kProgramBindingUndeclared);

    r = rows_reflection();
    r.bindings.pop_back();
    CHECK(cp::check_program(rows, r).code == cp::kProgramOutputMismatch);   // count is necessary
    r = rows_reflection();
    r.bindings[2].size = 8;
    CHECK(cp::check_program(rows, r).code == cp::kProgramOutputMismatch);
    r = rows_reflection();
    r.bindings[1].write = false;
    CHECK(cp::check_program(rows, r).code == cp::kProgramOutputMismatch);
    cs::Program unused = rows;
    unused.output.arrays.push_back(cs::GeometryArray::colors);
    CHECK(cp::check_program(unused, rows_reflection()).code == cp::kProgramOutputMismatch);
    r = rows_reflection();
    r.bindings.push_back(binding("caliper", cp::BindingKind::uniform_buffer, false, 32));
    CHECK(cp::check_program(rows, r).ok());
    r.bindings.back().size = 16;
    CHECK(cp::check_program(rows, r).code == cp::kProgramOutputMismatch);
    r = pixels_reflection();
    r.bindings.clear();
    CHECK(cp::check_program(pixels_program(), r).code == cp::kProgramOutputMismatch);   // pixels is necessary
}

TEST_CASE("the entry point and the workgroup of a program are checked") {
    const cs::Program rows = geometry_program();
    cp::Reflection r = rows_reflection();
    r.entries.push_back({"other", cp::EntryStage::compute, {64, 1, 1}});
    CHECK(cp::check_program(rows, r).code == cp::kProgramNoEntry);
    cs::Program named = rows;
    named.entry = "other";
    CHECK(cp::check_program(named, r).ok());
    named.entry = "missing";
    CHECK(cp::check_program(named, r).code == cp::kProgramNoEntry);
    // Metal reserves main, thus an entry point of that name is refused with its cause.
    r = rows_reflection();
    r.entries[0].name = "main";
    const cp::Verdict reserved = cp::check_program(rows, r);
    CHECK(reserved.code == cp::kProgramCompileFailed);
    CHECK(reserved.message.find("main") != std::string::npos);
    r = rows_reflection();
    r.entries[0].workgroup_size = {16, 16, 2};
    CHECK(cp::check_program(rows, r).code == cp::kProgramOverLimits);
    r = rows_reflection();
    r.entries.clear();
    CHECK(cp::check_program(rows, r).code == cp::kProgramNoEntry);
    // A dispatch derived from max_count above 65535 workgroups in one dimension.
    cs::Program wide = rows;
    wide.output.max_count = 1000000;
    r = rows_reflection();
    r.entries[0].workgroup_size = {1, 1, 1};
    CHECK(cp::check_program(wide, r).code == cp::kProgramOverLimits);
    wide.workgroups = std::array<std::uint32_t, 3>{1000, 1000, 1};
    CHECK(cp::check_program(wide, r).ok());
}

TEST_CASE("the dispatch size and the output bytes follow the declaration") {
    CHECK(cp::dispatch_size(geometry_program(), rows_reflection()) == std::array<std::uint32_t, 3>{4, 1, 1});
    CHECK(cp::dispatch_size(pixels_program(), pixels_reflection()) == std::array<std::uint32_t, 3>{8, 4, 1});
    cs::Program fixed = geometry_program();
    fixed.workgroups = std::array<std::uint32_t, 3>{2, 3, 4};
    CHECK(cp::dispatch_size(fixed, rows_reflection()) == std::array<std::uint32_t, 3>{2, 3, 4});
    CHECK(cp::output_bytes(geometry_program()) == 256u * 12u + 4u);
    cs::Program colored = geometry_program();
    colored.output.arrays.push_back(cs::GeometryArray::colors);
    CHECK(cp::output_bytes(colored) == 256u * 16u + 4u);
    CHECK(cp::output_bytes(pixels_program()) == 64u * 32u * 4u);
}

TEST_CASE("the binding check of a stage refuses a binding and an entry point") {
    cs::Stage s;
    s.panel = "world";
    s.entity = "programs/rows";
    s.wgsl = "const k = 0.5;\nfn material(in: MaterialIn) -> vec4<f32> { return in.color * k; }\n";
    CHECK(cp::check_stage(s, {}).ok());
    cp::Reflection r;
    r.bindings = {binding("tex", cp::BindingKind::texture, false)};
    CHECK(cp::check_stage(s, r).code == cp::kStageBindingNotAllowed);
    // A binding no entry point uses leaves the reflection; the source shows it.
    cs::Stage declared = s;
    declared.wgsl = "fn material(in: MaterialIn) -> vec4<f32> { return in.color; }\n"
                    "@group(0) @binding(0) var<uniform> u: vec4<f32>;\n";
    cp::Verdict v = cp::check_stage(declared, {});
    CHECK(v.code == cp::kStageBindingNotAllowed);
    CHECK(v.line == 2);
    CHECK(v.column == 1);
    r = {};
    r.entries = {{"fs", cp::EntryStage::fragment, {0, 0, 0}}};
    CHECK(cp::check_stage(s, r).code == cp::kStageEntryNotAllowed);
    cs::Stage entry = s;
    entry.wgsl = "// no entry here: @fragment\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n";
    v = cp::check_stage(entry, {});
    CHECK(v.code == cp::kStageEntryNotAllowed);
    CHECK(v.line == 2);
}

TEST_CASE("without the compiler library the compile answers program.compiler_unavailable") {
    if (cp::compiler_available()) return;
    CHECK(cp::compile_and_reflect("fn f() {}").verdict.code == cp::kProgramCompilerUnavailable);
}

TEST_CASE("with the compiler library the sandbox checks real sources") {
    if (!cp::compiler_available()) {
        MESSAGE("skipped: this build has no caliper-wgsl library");
        return;
    }
    const cp::Compiled c = cp::compile_and_reflect(kRows);
    REQUIRE(c.verdict.ok());
    REQUIRE(c.reflection.bindings.size() == 3);
    CHECK(c.reflection.bindings[0].name == "w");
    CHECK(c.reflection.bindings[0].read);
    CHECK_FALSE(c.reflection.bindings[0].write);
    CHECK(c.reflection.bindings[2].name == "count");
    CHECK(c.reflection.bindings[2].size == 4);
    REQUIRE(c.reflection.entries.size() == 1);
    CHECK(c.reflection.entries[0].workgroup_size == std::array<std::uint32_t, 3>{64, 1, 1});
    const cp::Verdict ok = cp::check_program(geometry_program(), c.reflection);
    CHECK_MESSAGE(ok.ok(), ok.code << ": " << ok.message);

    // An input declared read_write is refused by the binding check.
    std::string rw = kRows;
    rw.replace(rw.find("read> w"), 7, "read_write> w");
    cs::Program writes = geometry_program();
    writes.wgsl = rw;
    const cp::Compiled cw = cp::compile_and_reflect(rw);
    REQUIRE(cw.verdict.ok());
    CHECK(cp::check_program(writes, cw.reflection).code == cp::kProgramWritesInput);

    // A store to a var<storage, read> does not compile.
    std::string store = kRows;
    store.replace(store.find("atomicAdd"), 0, "w[0] = 1.0;\n  ");
    const cp::Compiled cs_ = cp::compile_and_reflect(store);
    CHECK(cs_.verdict.code == cp::kProgramCompileFailed);
    CHECK(cs_.verdict.line > 0);

    // A syntax error has its line and column.
    const cp::Compiled bad = cp::compile_and_reflect("fn f() {\n  let x = ;\n}\n");
    CHECK(bad.verdict.code == cp::kProgramCompileFailed);
    CHECK(bad.verdict.line == 2);
    CHECK(bad.verdict.column > 0);
    CHECK_FALSE(bad.verdict.message.empty());

    // A pixels program and its dispatch.
    const cp::Compiled px = cp::compile_and_reflect(pixels_program().wgsl);
    REQUIRE(px.verdict.ok());
    const cp::Verdict pok = cp::check_program(pixels_program(), px.reflection);
    CHECK_MESSAGE(pok.ok(), pok.code << ": " << pok.message);
    CHECK(cp::dispatch_size(pixels_program(), px.reflection) == std::array<std::uint32_t, 3>{8, 4, 1});

    // A stage compiles with the prelude before it and has no entry point.
    cs::Stage s;
    s.panel = "world";
    s.entity = "programs/rows";
    s.wgsl = "fn material(in: MaterialIn) -> vec4<f32> { return vec4<f32>(in.color.rgb * in.value, 1.0); }\n";
    const cp::Compiled st = cp::compile_and_reflect(std::string(cp::stage_prelude()) + s.wgsl);
    REQUIRE(st.verdict.ok());
    CHECK(cp::check_stage(s, st.reflection).ok());
}

TEST_CASE("sandbox: params of a program follow its struct Params, and pack into the uniform by name") {
    caliper_builder::spec::Program p;
    p.id = "scaled";
    p.wgsl = "struct Params { gain: f32, rows: u32, shift: i32 }\n"
             "@group(0) @binding(0) var<uniform> params: Params;\n";
    p.params = {{"gain", 0.5}, {"rows", 4.0}, {"shift", -2.0}};
    CHECK(caliper_builder::programs::check_params(p).ok());
    p.params.erase("shift");
    caliper_builder::programs::Verdict v = caliper_builder::programs::check_params(p);
    CHECK(v.code == "program.params_mismatch");
    CHECK(v.line == 1);
    CHECK(v.message.find("shift") != std::string::npos);

    std::vector<caliper_builder::programs::ParamField> fields;
    caliper_builder::programs::Verdict why;
    REQUIRE(caliper_builder::programs::params_fields(p.wgsl, &fields, &why));
    REQUIRE(fields.size() == 3);
    CHECK(fields[1].type == caliper_builder::programs::ParamType::u32);
    const std::vector<std::uint8_t> bytes =
        caliper_builder::programs::pack_params(fields, {{"shift", -2.4}, {"gain", 0.5}, {"rows", 7.6}});
    REQUIRE(bytes.size() == 16);
    float gain = 0;
    std::uint32_t rows = 0;
    std::int32_t shift = 0;
    std::memcpy(&gain, bytes.data(), 4);
    std::memcpy(&rows, bytes.data() + 4, 4);
    std::memcpy(&shift, bytes.data() + 8, 4);
    CHECK(gain == 0.5f);
    CHECK(rows == 8u);
    CHECK(shift == -2);

    // A field of another type, and a params variable without params.
    REQUIRE(caliper_builder::programs::params_fields("struct Params { v: vec2<f32> }", &fields, &why));
    CHECK(why.code == "program.params_mismatch");
    caliper_builder::spec::Program bare;
    bare.wgsl = "@group(0) @binding(0) var<uniform> params: vec4<f32>;\n";
    CHECK(caliper_builder::programs::check_params(bare).code == "program.params_mismatch");
}

TEST_CASE("sandbox: a stage reads its params by name, and the viewer declares them after its source") {
    caliper_builder::spec::Stage s;
    s.panel = "world";
    s.entity = "spec/terrain";
    s.params = {{"level", 0.25}, {"band", 0.01}};
    s.wgsl = "fn material(in: MaterialIn) -> vec4<f32> {\n"
             "  if (abs(in.value - params.level) < params.band) { return vec4<f32>(1.0); }\n"
             "  return in.color;\n}\n";
    CHECK(caliper_builder::programs::check_stage_params(s).ok());
    const std::string uniform = caliper_builder::programs::stage_params_declaration(s, true);
    CHECK(uniform == "\nstruct Params { band: f32, level: f32 }\n@group(3) @binding(0) var<uniform> params: Params;\n");
    CHECK(caliper_builder::programs::stage_params_declaration(s, false).find("var<private> params") != std::string::npos);
    if (caliper_builder::programs::compiler_available()) {
        const std::string composed = std::string(caliper_builder::programs::stage_prelude()) + s.wgsl +
                                     caliper_builder::programs::stage_params_declaration(s, false);
        const caliper_builder::programs::Compiled c = caliper_builder::programs::compile_and_reflect(composed);
        INFO(c.verdict.message);
        CHECK(c.verdict.ok());
        CHECK(caliper_builder::programs::check_stage(s, c.reflection).ok());
    }
    s.wgsl = "fn material(in: MaterialIn) -> vec4<f32> {\n  return in.color * params.gain;\n}\n";
    caliper_builder::programs::Verdict v = caliper_builder::programs::check_stage_params(s);
    CHECK(v.code == "stage.params_mismatch");
    CHECK(v.line == 2);
    s.wgsl = "struct Params { level: f32 }\nfn material(in: MaterialIn) -> vec4<f32> { return in.color; }\n";
    CHECK(caliper_builder::programs::check_stage_params(s).code == "stage.params_mismatch");
}

TEST_CASE("sandbox: a tensor output writes out, and an operator's id holds dots") {
    caliper_builder::spec::Program p;
    p.id = "operator.terrain.0";
    p.wgsl = "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n"
             "@compute @workgroup_size(64)\nfn run(@builtin(global_invocation_id) id: vec3<u32>) {\n"
             "  if (id.x < 16u) { out[id.x] = f32(id.x); }\n}\n";
    p.output.kind = caliper_builder::spec::OutputKind::tensor;
    p.output.shape = {4, 4};
    CHECK(caliper_builder::programs::check_declaration(p).ok());
    CHECK(caliper_builder::programs::output_bytes(p) == 64);
    p.output.shape = {1001, 1000};
    CHECK(caliper_builder::programs::check_declaration(p).code == "program.over_limits");
    p.output.shape = {1, 1, 1, 1, 1};
    CHECK(caliper_builder::programs::check_declaration(p).code == "program.over_limits");
    p.output.shape = {16};
    p.output.max_count = 3;
    CHECK(caliper_builder::programs::check_declaration(p).code == "program.output_mismatch");
    p.output.max_count = 0;
    for (const char* bad : {"operator.Terrain.0", "operator.terrain.", "operator..0", "op.terrain.0"}) {
        p.id = bad;
        CHECK(caliper_builder::programs::check_declaration(p).code == "program.over_limits");
    }
    p.id = "operator.terrain.12";
    if (caliper_builder::programs::compiler_available()) {
        const caliper_builder::programs::Compiled c = caliper_builder::programs::compile_and_reflect(p.wgsl);
        REQUIRE(c.verdict.ok());
        CHECK(caliper_builder::programs::check_program(p, c.reflection).ok());
        CHECK(caliper_builder::programs::dispatch_size(p, c.reflection) == std::array<std::uint32_t, 3>{1, 1, 1});
        p.wgsl.replace(p.wgsl.find("out:"), 4, "res:");
        p.wgsl.replace(p.wgsl.find("out["), 4, "res[");
        const caliper_builder::programs::Compiled d = caliper_builder::programs::compile_and_reflect(p.wgsl);
        REQUIRE(d.verdict.ok());
        CHECK(caliper_builder::programs::check_program(p, d.reflection).code == "program.binding_undeclared");
    }
}
