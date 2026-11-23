// Tests of the composer of a piece's derive half (programs/pieces.h,
// design/programmable-graphics.md, Generated Header): the header of the plan
// for the arcs piece, the segments that map a line back to its file, and each
// refusal of a name, a type or a count that would reach the WGSL text.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;

namespace {

cs::Resource buffer(const std::string& id, const std::string& element, std::uint32_t count) {
    cs::Resource r;
    r.id = id;
    r.layout.element = element;
    r.count = count;
    return r;
}

cs::Resource record(const std::string& id, const std::string& name, std::vector<cs::Field> fields,
                    std::uint32_t count) {
    cs::Resource r;
    r.id = id;
    r.layout.type_name = name;
    r.layout.fields = std::move(fields);
    r.count = count;
    return r;
}

cs::PieceInput tensor(const std::string& name) {
    cs::PieceInput in;
    in.name = name;
    in.rank_min = in.rank_max = 4;
    return in;
}

// The manifest of attention_arcs in the plan, without its draw half.
cs::Piece arcs() {
    cs::Piece p;
    p.name = "attention_arcs";
    p.inputs = {tensor("attn0"), tensor("attn1")};
    cs::PieceParam threshold;
    threshold.name = "threshold";
    cs::PieceParam glow;
    glow.name = "glow";
    p.params = {threshold, glow};
    p.resources.push_back(buffer("heads", "f32", 8));
    p.resources.push_back(record("arcs", "Arc",
                                 {{"from_token", "u32"}, {"to_token", "u32"}, {"layer", "u32"}, {"head", "u32"},
                                  {"weight", "f32"}, {"score", "f32"}},
                                 4096));
    p.resources.back().counted = true;
    p.resources.push_back(record("labels", "Label", {{"position", "vec4<f32>"}, {"string", "u32"}}, 32));
    cs::Derive d;
    d.passes = {{"scores", {1, 1, 1}, {"heads"}}, {"build", {128, 1, 1}, {"arcs"}}, {"place_labels", {1, 1, 1}, {}}};
    p.derive = d;
    return p;
}

cp::Resolved files(const std::string& common, const std::string& derive) {
    cp::Resolved r;
    if (!common.empty()) r.files["common.wgsl"] = common;
    r.files["derive.wgsl"] = derive;
    return r;
}

std::string code_of(const cs::Piece& p) {
    return cp::compose_derive(p, files("", "fn f() {}\n")).verdict.code;
}

}  // namespace

TEST_CASE("the derive header of the arcs piece binds every name at its place") {
    const cp::Composed c = cp::compose_derive(arcs(), files("const T: u32 = 32u;\n", "fn f() {}"));
    REQUIRE(c.verdict.ok());
    const char* kHeader =
        "struct CaliperFrame { step: u32, pass_index: u32, max_count: u32, width: u32, height: u32, time_s: f32, "
        "every: u32, pad: u32 }\n"
        "@group(0) @binding(0) var<uniform> caliper: CaliperFrame;\n"
        "struct Params { threshold: f32, glow: f32 }\n"
        "@group(0) @binding(1) var<uniform> params: Params;\n"
        "struct CaliperShapes { attn0: vec4<u32>, attn1: vec4<u32> }\n"
        "@group(0) @binding(2) var<uniform> shapes: CaliperShapes;\n"
        "@group(1) @binding(0) var<storage, read> attn0: array<f32>;\n"
        "@group(1) @binding(1) var<storage, read> attn1: array<f32>;\n"
        "struct Arc { from_token: u32, to_token: u32, layer: u32, head: u32, weight: f32, score: f32 }\n"
        "struct Label { position: vec4<f32>, string: u32 }\n"
        "@group(2) @binding(0) var<storage, read_write> heads: array<f32>;\n"
        "@group(2) @binding(2) var<storage, read_write> arcs: array<Arc>;\n"
        "@group(2) @binding(3) var<storage, read_write> arcs_count: atomic<u32>;\n"
        "@group(2) @binding(4) var<storage, read_write> labels: array<Label>;\n";
    CHECK(c.text == std::string(kHeader) + "const T: u32 = 32u;\nfn f() {}\n");
    REQUIRE(c.segments.size() == 3);
    CHECK(c.segments[0].part == "header");
    CHECK(c.segments[0].first_line == 1);
    CHECK(c.segments[0].lines == 14);
    CHECK(c.segments[1].part == "common.wgsl");
    CHECK(c.segments[1].first_line == 15);
    CHECK(c.segments[2].part == "derive.wgsl");
    CHECK(c.segments[2].first_line == 16);
    CHECK(c.segments[2].lines == 1);
}

TEST_CASE("the lib modules come after the header with their own segments") {
    cp::Resolved r = files("", "fn f() {}\n");
    r.lib_text = "fn colormap__one() -> f32 { return 1.0; }\nfn easing__two() -> f32 {\n return 2.0;\n}\n";
    r.lib_segments = {{"lib/colormap@1", 1, 1}, {"lib/easing@1", 2, 3}};
    cs::Piece p;
    p.name = "plain";
    p.resources.push_back(buffer("a", "vec4<u32>", 2));
    p.derive = cs::Derive{{{"run", {1, 1, 1}, {"a"}}}, {}, {}, {}};
    const cp::Composed c = cp::compose_derive(p, r);
    REQUIRE(c.verdict.ok());
    // No params and no tensor input: neither Params nor shapes.
    CHECK(c.text.find("Params") == std::string::npos);
    CHECK(c.text.find("shapes") == std::string::npos);
    CHECK(c.text.find("@group(2) @binding(0) var<storage, read_write> a: array<vec4<u32>>;\n") != std::string::npos);
    REQUIRE(c.segments.size() == 4);
    CHECK(c.segments[1].part == "lib/colormap@1");
    CHECK(c.segments[1].first_line == c.segments[0].lines + 1);
    CHECK(c.segments[2].part == "lib/easing@1");
    CHECK(c.segments[2].first_line == c.segments[0].lines + 2);
    CHECK(c.segments[3].part == "derive.wgsl");
    CHECK(c.segments[3].first_line == c.segments[0].lines + 5);
}

TEST_CASE("a buffer input keeps its index, a texture is a packed array and an integer tensor keeps its type") {
    cs::Piece p;
    p.name = "mixed";
    cs::PieceInput b;
    b.name = "other";
    b.kind = cs::InputKind::buffer;
    p.inputs = {b, tensor("ids")};
    p.inputs[1].dtype = "u32";
    cs::Resource t;
    t.id = "img";
    t.kind = cs::ResourceKind::texture;
    t.width = 4;
    t.height = 2;
    p.resources = {t};
    cs::PieceParam n;
    n.name = "n";
    n.type = cs::ParamType::u32;
    p.params = {n};
    p.derive = cs::Derive{{{"run", {1, 1, 1}, {"img"}}}, {}, {}, {}};
    const cp::Composed c = cp::compose_derive(p, files("", "fn f() {}\n"));
    REQUIRE(c.verdict.ok());
    CHECK(c.text.find("@group(1) @binding(1) var<storage, read> ids: array<u32>;\n") != std::string::npos);
    CHECK(c.text.find("other") == std::string::npos);   // the draw half reads a buffer input
    CHECK(c.text.find("struct CaliperShapes { ids: vec4<u32> }\n") != std::string::npos);
    CHECK(c.text.find("struct Params { n: u32 }\n") != std::string::npos);
    CHECK(c.text.find("@group(2) @binding(0) var<storage, read_write> img: array<u32>;\n") != std::string::npos);
}

TEST_CASE("the composer refuses every name and type that would change the header") {
    cs::Piece p = arcs();
    p.resources[1].layout.fields[0].type = "vec3<f32>";
    CHECK(code_of(p) == cp::kProgramBadLayout);
    p = arcs();
    p.resources[1].layout.fields[0].name = "x: f32 } @group(3) @binding(0) var<storage, read_write> y: array<f32>;";
    CHECK(code_of(p) == cp::kProgramBadLayout);
    p = arcs();
    p.resources[1].layout.fields[1].name = "from_token";
    CHECK(code_of(p) == cp::kProgramBadLayout);
    p = arcs();
    p.resources[2].layout.type_name = "Arc";   // the same name with other fields
    CHECK(code_of(p) == cp::kProgramBadLayout);
    p = arcs();
    p.resources[0].layout.type_name = "Both";
    CHECK(code_of(p) == cp::kProgramBadLayout);
    p = arcs();
    p.resources[0].id = "caliper_frame";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs();
    p.resources[0].id = "attn0";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs();
    p.resources[0].id = "arcs_count";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs();
    p.resources[0].id = "a__b";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs();
    p.resources[0].count = cs::piece_limits::kCountMax + 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs();
    p.resources[2].kind = cs::ResourceKind::texture;
    p.resources[2].width = 8193;
    p.resources[2].height = 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p.resources[2].width = 8192;
    p.resources[2].format = "bgra8unorm";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs();
    p.inputs[0].dtype = "f16";
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs();
    p.params[1].name = "threshold";
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs();
    p.params[1].name = "glow;";
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs();
    p.derive.reset();
    CHECK(code_of(p) == cp::kProgramBadPass);
    p = arcs();
    p.derive->passes.resize(9, p.derive->passes[0]);
    CHECK(code_of(p) == cp::kProgramBadPass);
    // Without derive.wgsl, and with a resolution that failed.
    CHECK(cp::compose_derive(arcs(), cp::Resolved{}).verdict.code == cp::kPieceBadManifest);
    cp::Resolved bad = files("", "fn f() {}\n");
    bad.verdict.code = cp::kPieceUnknownImport;
    const cp::Composed c = cp::compose_derive(arcs(), bad);
    CHECK(c.verdict.code == cp::kPieceUnknownImport);
    CHECK(c.text.empty());
}

TEST_CASE("the lib modules that read the draw prelude stay blank in the derive module, with their lines") {
    cs::Piece p;
    p.name = "plain";
    p.imports = {"colormap@1", "easing@1", "pick@1"};
    p.resources.push_back(buffer("a", "f32", 4));
    p.derive = cs::Derive{{{"run", {1, 1, 1}, {"a"}}}, {}, {}, {}};
    p.files["derive.wgsl"] = "@compute @workgroup_size(1) fn run() { a[0] = easing::smooth(0.5); }\n";
    const cp::Resolved r = cp::resolve_imports(p);
    REQUIRE(r.verdict.ok());
    const cp::Composed c = cp::compose_derive(p, r);
    REQUIRE(c.verdict.ok());
    // No prelude name reaches the module; easing@1 keeps its text.
    CHECK(c.text.find("caliper_") == std::string::npos);
    CHECK(c.text.find("fn easing__smooth") != std::string::npos);
    REQUIRE(c.segments.size() == 5);
    CHECK(c.segments[1].part == "lib/colormap@1");
    CHECK(c.segments[3].part == "lib/pick@1");
    // Every segment keeps its lines, thus a message still maps to its file.
    std::uint32_t lines = 0;
    for (const cp::Segment& s : c.segments) lines += s.lines;
    CHECK(lines == std::uint32_t(std::count(c.text.begin(), c.text.end(), '\n')));
    CHECK(c.segments[4].first_line == c.segments[3].first_line + c.segments[3].lines);
    if (cp::compiler_available()) CHECK(cp::compile_and_reflect(c.text).verdict.ok());

    // The script gets the resolved text in one section: split, it composes the same module.
    std::string common = r.lib_text;
    const cp::Resolved split = cp::split_common(common + "const K: u32 = 2u;\n");
    CHECK(split.lib_text == r.lib_text);
    REQUIRE(split.lib_segments.size() == 3);
    CHECK(split.lib_segments[2].part == "lib/pick@1");
    CHECK(split.lib_segments[2].first_line == r.lib_segments[2].first_line);
    CHECK(split.files.at("common.wgsl") == "const K: u32 = 2u;\n");
    cp::Resolved again = split;
    again.files.erase("common.wgsl");
    again.files["derive.wgsl"] = r.files.at("derive.wgsl");
    CHECK(cp::compose_derive(p, again).text == c.text);
    CHECK(cp::split_common("fn f() {}\n").lib_text.empty());

    // A call of such a module from the derive half is refused with a reason.
    p.files["derive.wgsl"] = "@compute @workgroup_size(1) fn run() { a[0] = colormap::magma(0.5).x; }\n";
    CHECK(cp::compose_derive(p, cp::resolve_imports(p)).verdict.code == cp::kPieceUnknownImport);
}

TEST_CASE("a typed tensor input binds its bytes as words with a checked accessor of the bound type") {
    cs::Piece p;
    p.name = "typed";
    cs::PieceInput image = tensor("image");
    image.dtype = "f32|u8";
    cs::PieceInput ids = tensor("ids");
    ids.dtype = "i32|u8|u16";
    cs::PieceInput depth = tensor("depth");
    depth.dtype = "u16";
    p.inputs = {image, tensor("pose"), ids, depth};
    p.resources.push_back(buffer("out", "f32", 4));
    cs::Derive d;
    d.passes = {{"run", {1, 1, 1}, {"out"}}};
    p.derive = d;
    const std::string derive =
        "fn run() { out[0] = image_at(0u) + f32(ids_at(1u)) + f32(depth_at(2u)) + f32(image_type + ids_type + "
        "depth_type) + pose[0]; }\n";

    // Without a bound type, each input takes the first of its dtype.
    const cp::Composed first = cp::compose_derive(p, files("", derive));
    REQUIRE(first.verdict.ok());
    const std::string& h = first.text;
    CHECK(h.find("struct CaliperShapes { image: vec4<u32>, pose: vec4<u32>, ids: vec4<u32>, depth: vec4<u32>, "
                 "image__len: u32, ids__len: u32, depth__len: u32 }") != std::string::npos);
    CHECK(h.find("@group(1) @binding(0) var<storage, read> image: array<u32>;") != std::string::npos);
    CHECK(h.find("@group(1) @binding(1) var<storage, read> pose: array<f32>;") != std::string::npos);
    CHECK(h.find("@group(1) @binding(2) var<storage, read> ids: array<u32>;") != std::string::npos);
    CHECK(h.find("const image_type: u32 = CALIPER_F32;") != std::string::npos);
    CHECK(h.find("const ids_type: u32 = CALIPER_I32;") != std::string::npos);
    CHECK(h.find("const depth_type: u32 = CALIPER_U16;") != std::string::npos);
    CHECK(h.find("fn image_at(i: u32) -> f32 {") != std::string::npos);
    CHECK(h.find("return bitcast<f32>(image[i]);") != std::string::npos);
    CHECK(h.find("fn ids_at(i: u32) -> i32 {") != std::string::npos);
    CHECK(h.find("return bitcast<i32>(ids[i]);") != std::string::npos);
    CHECK(h.find("fn depth_at(i: u32) -> u32 {") != std::string::npos);
    CHECK(h.find("return (depth[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;") != std::string::npos);
    CHECK(h.find("if (i >= shapes.depth__len) {") != std::string::npos);
    if (cp::compiler_available()) CHECK(cp::compile_and_reflect(first.text).verdict.ok());

    // Each bound type gives its own accessor, with the same signature, thus the same piece compiles.
    const std::vector<std::pair<std::map<std::string, std::string>, std::vector<std::string>>> cases = {
        {{{"image", "u8"}, {"ids", "u8"}},
         {"const image_type: u32 = CALIPER_U8;", "return f32((image[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);",
          "return i32((ids[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);"}},
        {{{"ids", "u16"}}, {"const ids_type: u32 = CALIPER_U16;", "return i32((ids[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu);"}},
    };
    for (const auto& [bound, wants] : cases) {
        const cp::Composed c = cp::compose_derive(p, files("", derive), bound);
        REQUIRE(c.verdict.ok());
        for (const std::string& w : wants) CHECK_MESSAGE(c.text.find(w) != std::string::npos, w);
        if (cp::compiler_available()) CHECK(cp::compile_and_reflect(c.text).verdict.ok());
    }

    // A single 32-bit type keeps its plain array and needs no accessor; the header is the one of before.
    cs::Piece plain = p;
    plain.inputs = {tensor("pose")};
    const cp::Composed pc = cp::compose_derive(plain, files("", "fn run() { out[0] = pose[0]; }\n"));
    REQUIRE(pc.verdict.ok());
    CHECK(pc.text.find("CALIPER_U8") == std::string::npos);
    CHECK(pc.text.find("__len") == std::string::npos);

    // A bound type that the dtype does not name, a dtype that is not one, and a name the accessor takes.
    CHECK(cp::compose_derive(p, files("", derive), {{"image", "u16"}}).verdict.code == std::string(cp::kPieceBadBinding));
    for (const char* bad : {"f64", "u8|u8", "", "|", "f32|", "u8|u16|i32|u32|f32|f32", "U8", "u8 | u16"}) {
        cs::Piece q = p;
        q.inputs[0].dtype = bad;
        CHECK_MESSAGE(code_of(q) == std::string(cp::kPieceBadManifest), bad);
    }
    cs::Piece clash = p;
    clash.resources.push_back(buffer("image_at", "f32", 4));
    CHECK(code_of(clash) == std::string(cp::kProgramBadResource));
    cs::Piece clash2 = p;
    clash2.inputs.push_back(tensor("depth_type"));
    CHECK(code_of(clash2) == std::string(cp::kPieceBadManifest));
}

TEST_CASE("the dtype of a tensor input names one to five distinct types joined by a bar") {
    CHECK(cs::tensor_dtypes("f32") == std::vector<std::string>{"f32"});
    CHECK(cs::tensor_dtypes("f32|u8") == std::vector<std::string>{"f32", "u8"});
    CHECK(cs::tensor_dtypes("f32|i32|u32|u8|u16").size() == 5);
    CHECK(cs::tensor_dtypes("f32|f32").empty());
    CHECK(cs::tensor_dtypes("i8").empty());
    CHECK(!cs::typed_tensor_input("f32"));
    CHECK(!cs::typed_tensor_input("u32"));
    CHECK(cs::typed_tensor_input("u8"));
    CHECK(cs::typed_tensor_input("f32|i32"));
    CHECK(std::string(cs::tensor_read_type("u8|u16")) == "u32");
    CHECK(std::string(cs::tensor_read_type("u16|i32")) == "i32");
    CHECK(std::string(cs::tensor_read_type("u8|f32")) == "f32");
}
