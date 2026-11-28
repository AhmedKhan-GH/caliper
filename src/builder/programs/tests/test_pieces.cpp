// Tests of the interface of pieces (programs/pieces.h,
// design/programmable-graphics.md): the draw prelude and the lib modules are
// the embedded files and compile together the way a resolved piece does, and
// a program pins a piece; then the names, the files, the imports, the hash
// and the checks of a piece alone, with a test for each refusal.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/spec/spec.h>

#include <doctest/doctest.h>

#include <fstream>
#include <iterator>

#include <set>
#include <string>

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;

TEST_CASE("the draw prelude is the embedded file of version 1 with the five bindings of group 0") {
    const std::string_view text = cp::draw_prelude();
    REQUIRE(!text.empty());
    CHECK(text.rfind("// Caliper draw prelude, version 1", 0) == 0);
    for (const char* name : {"caliper_view", "caliper_selection", "caliper_linear", "caliper_nearest",
                             "caliper_colormaps", "fn caliper_fullscreen", "struct CaliperPicked"})
        CHECK(text.find(name) != std::string_view::npos);
    CHECK(text.find("@group(1)") == std::string_view::npos);
    CHECK(text.find("enable ") == std::string_view::npos);
}

TEST_CASE("the lib modules of version 1 are embedded and declare only names with their prefix") {
    const auto names = cp::lib_modules();
    CHECK(std::set<std::string_view>(names.begin(), names.end()) ==
          std::set<std::string_view>{"camera@1", "colormap@1", "easing@1", "pick@1"});
    for (auto name : names) {
        CAPTURE(std::string(name));
        const std::string_view text = cp::lib_module(name);
        REQUIRE(!text.empty());
        const std::string prefix = std::string(name.substr(0, name.find('@'))) + "__";
        for (const char* decl : {"\nfn ", "\nconst "}) {
            for (std::size_t at = text.find(decl); at != std::string_view::npos; at = text.find(decl, at + 1))
                CHECK(text.substr(at + std::char_traits<char>::length(decl), prefix.size()) == prefix);
        }
        CHECK(text.find("@group") == std::string_view::npos);
    }
    CHECK(cp::lib_module("colormap@2").empty());
}

TEST_CASE("the prelude and every lib module compile with a fragment that calls them as a resolved piece does") {
    if (!cp::compiler_available()) return;
    std::string module(cp::draw_prelude());
    for (auto name : cp::lib_modules()) module += std::string(cp::lib_module(name));
    // The author wrote colormap::magma and pick::id; resolution rewrites "::"
    // to "__", the same length, so the columns of the author's file stay.
    std::string author = R"wgsl(
@fragment
fn fs_test(in: CaliperFullscreen) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = colormap::magma(easing::smooth(in.uv.x));
    if (pick::selected(1u) || pick::selected_index(vec4<i32>(0, -1, -1, -1))) {
        out.color = vec4<f32>(1.0);
    }
    let c = camera::offset_pixels(camera::clip(vec3<f32>(in.uv, 0.0)), camera::pixel_direction(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(1.0, 0.0, 0.0, 1.0)));
    out.color = out.color + 0.5 * textureSampleLevel(caliper_colormaps, caliper_linear, in.uv, 0.0) +
                0.5 * textureSampleLevel(caliper_colormaps, caliper_nearest, c.xy, 0.0);
    out.id = pick::id(7u);
    return out;
}
)wgsl";
    for (std::size_t at = author.find("::"); at != std::string::npos; at = author.find("::", at)) author.replace(at, 2, "__");
    module += author;
    const cp::Compiled c = cp::compile_and_reflect(module);
    INFO(c.verdict.message);
    REQUIRE(c.verdict.ok());
    std::set<std::uint32_t> group0;
    for (const auto& b : c.reflection.bindings)
        if (b.group == cp::kGroupPrelude) group0.insert(b.binding);
    CHECK(group0 == std::set<std::uint32_t>{cp::kBindingView, cp::kBindingSelection, cp::kBindingLinear,
                                            cp::kBindingNearest, cp::kBindingColormaps});
    for (const auto& b : c.reflection.bindings)
        if (b.name == "caliper_view") CHECK(b.size == cp::kViewBytes);
}

TEST_CASE("a program that pins a piece is an instance with its pin, bindings and panel") {
    cs::Program a;
    a.id = "tower_arcs";
    CHECK(!cs::is_piece_instance(a));
    a.piece = cs::PieceRef{"ring_arcs", 1, "sha256:00", cs::PieceFrom::workspace};
    a.bind["attn0"] = "live:blocks.0.attn.weights.act";
    a.panel = "tower";
    CHECK(cs::is_piece_instance(a));
    const cs::Program b = a;
    CHECK(b.piece == a.piece);
    CHECK(b.bind == a.bind);
    CHECK(!(cs::PieceRef{"ring_arcs", 2, "sha256:00", cs::PieceFrom::workspace} == *a.piece));
}

TEST_CASE("the names of the piece values round-trip") {
    for (auto t : {cs::DrawTopology::triangle_list, cs::DrawTopology::triangle_strip, cs::DrawTopology::line_list,
                   cs::DrawTopology::line_strip, cs::DrawTopology::point_list})
        CHECK(cs::draw_topology_from_name(cs::draw_topology_name(t)) == t);
    for (auto b : {cs::DrawBlend::opaque, cs::DrawBlend::alpha, cs::DrawBlend::additive})
        CHECK(cs::draw_blend_from_name(cs::draw_blend_name(b)) == b);
    for (auto d : {cs::DrawDepth::none, cs::DrawDepth::test, cs::DrawDepth::write})
        CHECK(cs::draw_depth_from_name(cs::draw_depth_name(d)) == d);
    for (auto c : {cs::CameraMode::orbit, cs::CameraMode::pan_zoom, cs::CameraMode::none})
        CHECK(cs::camera_mode_from_name(cs::camera_mode_name(c)) == c);
    for (auto r : {cs::BufferReset::install, cs::BufferReset::cursor, cs::BufferReset::resource})
        CHECK(cs::buffer_reset_from_name(cs::buffer_reset_name(r)) == r);
    for (auto f : {cs::PieceFrom::library, cs::PieceFrom::workspace, cs::PieceFrom::inline_piece})
        CHECK(cs::piece_from_from_name(cs::piece_from_name(f)) == f);
    CHECK(cs::param_type_from_name("i32") == cs::ParamType::i32);
    CHECK(cs::input_kind_from_name("buffer") == cs::InputKind::buffer);
    CHECK(cs::resource_kind_from_name("texture") == cs::ResourceKind::texture);
    CHECK(cs::pass_kind_from_name("compute") == cs::PassKind::compute);
    CHECK(!cs::draw_blend_from_name("max"));
}

TEST_CASE("the codes of pieces are distinct") {
    const char* codes[] = {cp::kProgramNotBuilt,       cp::kPieceBadName,           cp::kPieceBadManifest,
                           cp::kPieceBadPath,          cp::kPieceUnknownImport,     cp::kPieceNotFound,
                           cp::kPieceMissing,          cp::kPieceHashMismatch,      cp::kPieceExists,
                           cp::kPieceReadOnly,         cp::kPieceInUse,             cp::kPieceBadBinding,
                           cp::kPieceCycle,            cp::kProgramBadResource,     cp::kProgramBadLayout,
                           cp::kProgramUnknownResource, cp::kProgramBadPass,        cp::kProgramTargetCycle,
                           cp::kProgramBadPick,        cp::kProgramBadText,         cp::kProgramCameraConflict,
                           cp::kProgramBindingNotAllowed, cp::kProgramReservedName, cp::kProgramEntryStage,
                           cp::kProgramInterfaceMismatch, cp::kProgramOverWebLimits, cp::kProgramResourcesUnavailable,
                           cp::kProgramResourceTooLarge, cp::kProgramDrawOverTime,  cp::kProgramDrawTimeout,
                           cp::kProgramDrawOverMemory, cp::kProgramBrowserRefused};
    std::set<std::string> seen(std::begin(codes), std::end(codes));
    CHECK(seen.size() == std::size(codes));
    CHECK(seen.count(cp::kProgramCompileFailed) == 0);
}

// ---- names, files, imports, hash and checks of a piece alone

namespace {

cs::Layout struct_layout(const char* name, std::vector<cs::Field> fields) {
    cs::Layout l;
    l.type_name = name;
    l.fields = std::move(fields);
    return l;
}

// A piece shaped like ring_arcs of the plan: a derive half that
// publishes scores, counted arcs and label anchors, and a draw half with an
// instanced pick pass, a pass into a half-size target and a bloom that reads it.
cs::Piece arcs_piece() {
    cs::Piece p;
    p.name = "ring_arcs";
    p.description = "Arcs between the tokens of one sentence";
    p.imports = {"colormap@1", "pick@1", "camera@1"};
    cs::PieceInput a0;
    a0.name = "attn0";
    a0.rank_min = a0.rank_max = 4;
    p.inputs = {a0};
    p.params = {cs::PieceParam{"threshold", cs::ParamType::f32, 0.05, 0.0, 0.5, std::string("Weight Threshold")}};
    cs::Resource heads;
    heads.id = "heads";
    heads.layout.element = "f32";
    heads.count = 8;
    cs::Resource arcs;
    arcs.id = "arcs";
    arcs.layout = struct_layout("Arc", {{"from_token", "u32"}, {"to_token", "u32"}, {"layer", "u32"}, {"head", "u32"},
                                        {"weight", "f32"}, {"score", "f32"}});
    arcs.count = 4096;
    arcs.counted = true;
    cs::Resource labels;
    labels.id = "labels";
    labels.layout = struct_layout("Label", {{"position", "vec4<f32>"}, {"string", "u32"}});
    labels.count = 32;
    p.resources = {heads, arcs, labels};
    cs::Derive d;
    d.passes = {cs::DerivePass{"scores", {1, 1, 1}, {"heads"}}, cs::DerivePass{"build", {128, 1, 1}, {"arcs"}},
                cs::DerivePass{"place_labels", {1, 1, 1}, {}}};
    p.derive = d;
    cs::Draw w;
    w.targets = {cs::DrawTarget{"glow", 0.5f, {0, 0, 0, 0}}};
    cs::DrawPass arc;
    arc.vertex = "arc_vs";
    arc.fragment = "arc_fs";
    arc.instances_of = "arcs";
    arc.vertices = 64;
    arc.topology = cs::DrawTopology::triangle_strip;
    arc.blend = cs::DrawBlend::additive;
    arc.depth = cs::DrawDepth::none;
    arc.pick = true;
    cs::DrawPass glow = arc;
    glow.fragment = "arc_glow";
    glow.pick = false;
    glow.blend.reset();
    glow.depth.reset();
    glow.target = "glow";
    cs::DrawPass bloom;
    bloom.fragment = "bloom";
    bloom.reads = {"glow"};
    bloom.blend = cs::DrawBlend::additive;
    w.passes = {arc, glow, bloom};
    w.pick = cs::PickMap{"arcs", {"layer", "head", "from_token", "to_token", "weight", "score"}};
    w.text = cs::TextTable{"labels", "position", "string", std::nullopt, {"the", "old", "map"}};
    p.draw = w;
    p.files["common.wgsl"] = "const T: u32 = 32u;\n";
    p.files["derive.wgsl"] = "@compute @workgroup_size(64) fn build(@builtin(global_invocation_id) id: vec3<u32>) {}\n";
    p.files["draw.wgsl"] = "fn tint(t: f32) -> vec4<f32> {\n    return colormap::magma(t); // colormap::magma\n}\n";
    return p;
}

std::string code_of(const cs::Piece& p) { return cp::check_piece_manifest(p).code; }

}  // namespace

TEST_CASE("published buffers hold 128 MiB, textures 256 MiB, and all resources 512 MiB") {
    // What the script sends at a pass (python/caliper/_programs.py, PUBLISHED_BYTES_MAX and PUBLISHED_RESOURCE_MAX).
    CHECK(cs::piece_limits::kPublishedBytesMax == 512u << 20);
    CHECK(cs::piece_limits::kPublishedResourceBytesMax == 128u << 20);   // CALIPER_NET_RESOURCE_VERSION_MAX_BYTES
    CHECK(cs::piece_limits::kPublishedTextureBytesMax == 256u << 20);
    CHECK(cs::piece_limits::kCountMax == 33554432u);
    cs::Piece p = arcs_piece();
    std::uint64_t others = 0;   // the arcs and the labels, which stay published
    for (std::size_t i = 1; i < p.resources.size(); ++i) {
        cp::ElementLayout l;
        REQUIRE(cp::element_layout(p.resources[i].layout, &l).ok());
        others += std::uint64_t(p.resources[i].count) * l.stride;
    }
    REQUIRE(others % 16 == 0);
    // 4,194,304 elements of 16 bytes: exactly 64 MiB, the most of a private resource, and one more is refused.
    p.resources[0].layout.element = "vec4<f32>";
    p.resources[0].count = 4194304;
    p.resources[0].publish = false;
    CHECK(code_of(p).empty());
    p.resources[0].count += 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    // Published: 128 MiB, 8,388,608 elements of 16 bytes, and one element more is refused at the resource.
    p.resources[0].publish = true;
    p.resources[0].count = 8388608;
    CHECK(code_of(p).empty());
    p.resources[0].count += 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    CHECK(cp::check_piece_manifest(p).where != "resources");
    // Five buffers reach 512 MiB while each remains below its 128 MiB binding limit.
    p.resources[1].count = 3145728;
    const std::uint64_t arcs_bytes = std::uint64_t(p.resources[1].count) * 24;
    p.resources[0].count = 7340032;
    p.resources[2].count = std::uint32_t((cs::piece_limits::kPublishedBytesMax - (320u << 20) - arcs_bytes) / 32);
    for (const char* id : {"extra_a", "extra_b"}) {
        cs::Resource extra = p.resources[0];
        extra.id = id;
        extra.count = 6815744;   // 104 MiB
        p.resources.push_back(extra);
    }
    CHECK_MESSAGE(code_of(p).empty(), cp::check_piece_manifest(p).message);
    p.resources[2].count += 1;
    CHECK(cp::check_piece_manifest(p).code == cp::kProgramBadResource);
    CHECK(cp::check_piece_manifest(p).where == "resources");
    // A count past 33,554,432 is refused before its bytes are taken.
    p = arcs_piece();
    p.resources[0].layout.element = "u32";
    p.resources[0].count = cs::piece_limits::kCountMax;
    CHECK(code_of(p).empty());
    p.resources[0].count += 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    // A texture of 4,096 by 4,096 texels holds 64 MiB, published or private.
    p = arcs_piece();
    p.resources[0].kind = cs::ResourceKind::texture;
    p.resources[0].layout = {};
    p.resources[0].count = 0;
    p.resources[0].width = p.resources[0].height = 2048;
    CHECK(code_of(p).empty());
    p.resources[0].width = p.resources[0].height = 1024;
    CHECK(code_of(p).empty());
    p.resources[0].publish = false;
    p.resources[0].width = p.resources[0].height = 4096;
    CHECK(code_of(p).empty());
    p.resources[0].width = 4097;   // a private resource holds at most 64 MiB
    CHECK(code_of(p) == cp::kProgramBadResource);
    // Full 8,192 by 8,192 textures fit at 256 MiB. Storage buffers keep the separate 128 MiB bound.
    p.resources[0].publish = true;
    p.resources[0].width = p.resources[0].height = cs::piece_limits::kTextureSideMax;
    CHECK(code_of(p).empty());
    p.resources[0].texels = 33554432u;
    CHECK(code_of(p).empty());
    p.resources[0].texels = 67108864u;
    CHECK(code_of(p).empty());
    p.resources[0].texels = 67108865u;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p.resources[0].texels.reset();
    p.resources[0].format = "r8unorm";
    CHECK(code_of(p).empty());
    p.resources[0].format = "r32uint";
    CHECK(code_of(p).empty());
    p.resources[0].format = "rgba16float";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p.resources[0].format.reset();
    p.resources[0].width = p.resources[0].height = 2;
    p.resources[0].texels = 5u;   // more than width times height
    CHECK(code_of(p) == cp::kProgramBadResource);
    p.resources[0].texels = 0u;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p.resources[0].texels.reset();
    p.resources[0].width = p.resources[0].height = cs::piece_limits::kTextureSideMax + 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    // A narrow buffer counts its words: 134,217,728 u8 elements or 67,108,864 u16 elements fill 128 MiB.
    p = arcs_piece();
    p.resources[0].layout = {};
    p.resources[0].layout.element = "u8";
    p.resources[0].count = cs::piece_limits::kCountMax * 4;
    CHECK(code_of(p).empty());
    p.resources[0].count = cs::piece_limits::kCountMax * 4 + 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p.resources[0].layout.element = "u16";
    p.resources[0].count = cs::piece_limits::kCountMax * 2;
    CHECK(code_of(p).empty());
    p.resources[0].count = cs::piece_limits::kCountMax * 2 + 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p.resources[0].layout.element = "u4";
    CHECK(code_of(p) == cp::kProgramBadLayout);
}

TEST_CASE("a piece name is a lowercase name of at most 32 bytes and a version from 1 to 1000000") {
    CHECK(cp::check_piece_name("ring_arcs", 1).ok());
    CHECK(cp::check_piece_name("a", 1000000).ok());
    for (const char* bad : {"", "Arcs", "_arcs", "1arcs", "arcs-2", "arcs/../x", "a b", "..", "arcs.toml",
                            "abcdefghijabcdefghijabcdefghijabc"})
        CHECK(cp::check_piece_name(bad, 1).code == cp::kPieceBadName);
    CHECK(cp::check_piece_name("arcs", 0).code == cp::kPieceBadName);
    CHECK(cp::check_piece_name("arcs", 1000001).code == cp::kPieceBadName);
}

TEST_CASE("a piece folder holds only the manifest, the three WGSL files and the samples it names") {
    cs::Piece p = arcs_piece();
    p.sample["attn0"] = cs::SampleInput{"sample/attn0.bin", {1, 4, 32, 32}};
    for (const char* ok : {"piece.toml", "common.wgsl", "derive.wgsl", "draw.wgsl", "sample/attn0.bin"})
        CHECK(cp::check_piece_file(p, ok).ok());
    for (const char* bad : {"", "/etc/passwd", "../piece.toml", "./draw.wgsl", "sample/../draw.wgsl",
                            "sample/attn1.bin", "sample/attn0.bin/x", "sample/../../x.bin", "sub/draw.wgsl",
                            "draw.wgsl ", "Draw.wgsl", "draw.wgsl\n", ".DS_Store", "sample\\attn0.bin"})
        CHECK(cp::check_piece_file(p, bad).code == cp::kPieceBadPath);
    p.sample.clear();
    CHECK(cp::check_piece_file(p, "sample/attn0.bin").code == cp::kPieceBadPath);
}

TEST_CASE("resolution rewrites each module::item to the same length and puts the modules first") {
    const cs::Piece p = arcs_piece();
    const cp::Resolved r = cp::resolve_imports(p);
    INFO(r.verdict.message);
    REQUIRE(r.verdict.ok());
    CHECK(r.modules == std::vector<std::string>{"colormap@1", "pick@1", "camera@1"});
    REQUIRE(r.lib_segments.size() == 3);
    CHECK(r.lib_segments[0].part == "lib/colormap@1");
    CHECK(r.lib_segments[0].first_line == 1);
    CHECK(r.lib_segments[1].first_line == r.lib_segments[0].first_line + r.lib_segments[0].lines);
    CHECK(r.lib_text.find("fn colormap__magma") != std::string::npos);
    CHECK(r.lib_text.find("fn pick__id") < r.lib_text.find("fn camera__clip"));
    const std::string& draw = r.files.at("draw.wgsl");
    CHECK(draw.size() == p.files.at("draw.wgsl").size());
    CHECK(draw.find("return colormap__magma(t);") != std::string::npos);
    CHECK(draw.find("// colormap::magma") != std::string::npos);  // comments stay as written
    CHECK(r.files.at("common.wgsl") == p.files.at("common.wgsl"));
}

TEST_CASE("resolution refuses a module that is not imported, an unknown item and an unknown module") {
    cs::Piece p = arcs_piece();
    p.files["draw.wgsl"] = "fn f() -> f32 {\n    return easing::smooth(0.5);\n}\n";
    cp::Resolved r = cp::resolve_imports(p);
    CHECK(r.verdict.code == cp::kPieceUnknownImport);
    CHECK(r.verdict.where == "draw.wgsl");
    CHECK(r.verdict.line == 2);
    CHECK(r.verdict.column == 12);
    p.files["draw.wgsl"] = "fn f() -> vec4<f32> { return colormap::plasma(0.5); }\n";
    r = cp::resolve_imports(p);
    CHECK(r.verdict.code == cp::kPieceUnknownImport);
    CHECK(r.verdict.message.find("plasma") != std::string::npos);
    p.files["draw.wgsl"] = "fn f() { let a = ::magma; }\n";
    CHECK(cp::resolve_imports(p).verdict.code == cp::kPieceUnknownImport);
    p.files["draw.wgsl"] = "fn f() {}\n";
    p.imports = {"colormap@2"};
    CHECK(cp::resolve_imports(p).verdict.code == cp::kPieceUnknownImport);
    p.imports = {"../colormap@1"};
    CHECK(cp::resolve_imports(p).verdict.code == cp::kPieceUnknownImport);
    p.imports = {"colormap@1", "colormap@1"};
    CHECK(!cp::resolve_imports(p).verdict.ok());
}

#if defined(CALIPER_PROGRAMS_LIBRARY_DIR)
TEST_CASE("the embedded library equals its files on disk, thus a changed file reached the build") {
    std::size_t n = 0;
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string path = std::string(CALIPER_PROGRAMS_LIBRARY_DIR) + "/" + std::string(f.path);
        std::ifstream in(path, std::ios::binary);
        REQUIRE_MESSAGE(in.good(), path);
        const std::string disk((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK_MESSAGE(disk == f.text, f.path, " differs from its embedded copy");
        ++n;
    }
    CHECK(n > 0);
}
#endif

TEST_CASE("the canonical manifest is RFC 8785 JSON of the manifest without hash, files and defaults") {
    cs::Piece p;
    p.name = "field";
    p.description = "Quote \" back\\slash \x01 tab\t";
    p.params = {cs::PieceParam{"gain", cs::ParamType::f32, 1.0, 0.1, 1e21, std::nullopt},
                cs::PieceParam{"tiny", cs::ParamType::f32, 1e-7, -0.0, 0.05, std::nullopt}};
    cs::Draw d;
    d.camera = cs::CameraMode::none;
    cs::DrawPass pass;
    pass.fragment = "field";
    d.passes = {pass};
    p.draw = d;
    p.hash = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    p.files["draw.wgsl"] = "x";
    CHECK(cp::canonical_manifest(p) ==
          R"({"description":"Quote \" back\\slash \u0001 tab\t","draw":{"camera":"none","passes":[{"fragment":"field","kind":"render"}]},"name":"field","params":[{"default":1,"max":1e+21,"min":0.1,"name":"gain"},{"default":1e-7,"max":0.05,"min":0,"name":"tiny"}],"version":1})");
}

TEST_CASE("an optional input reaches the canonical manifest only when it is optional") {
    cs::Piece p;
    p.name = "loose";
    cs::PieceInput in;
    in.name = "v";
    p.inputs = {in};
    CHECK(cp::canonical_manifest(p) == R"({"inputs":[{"kind":"tensor","name":"v"}],"name":"loose","version":1})");
    const std::string before = cp::piece_hash(p);
    p.inputs[0].optional = true;
    CHECK(cp::canonical_manifest(p) ==
          R"({"inputs":[{"kind":"tensor","name":"v","optional":true}],"name":"loose","version":1})");
    CHECK(cp::piece_hash(p) != before);
}

TEST_CASE("the hash covers the canonical manifest, the imported modules and the files with LF line ends") {
    cs::Piece p;
    p.name = "uv";
    p.imports = {"easing@1"};
    cs::Draw d;
    cs::DrawPass pass;
    pass.fragment = "fs";
    d.passes = {pass};
    d.camera = cs::CameraMode::none;
    p.draw = d;
    p.files["draw.wgsl"] = "@fragment fn fs() {}\r\n";
    const std::string lib(cp::lib_module("easing@1"));
    const std::string bytes = "caliper-piece/1\n" + cp::canonical_manifest(p) + "\n--- lib/easing@1 " +
                              std::to_string(lib.size()) + "\n" + lib + "--- draw.wgsl 21\n@fragment fn fs() {}\n";
    const std::string h = cp::piece_hash(p);
    CHECK(h == "sha256:" + cp::sha256_hex(bytes));
    cs::Piece q = p;
    q.files["draw.wgsl"] = "@fragment fn fs() {}\n";
    q.hash = "sha256:ffff";
    q.sample["x"] = cs::SampleInput{};
    q.sample.clear();
    CHECK(cp::piece_hash(q) == h);  // LF line ends, and the hash key itself, change nothing
    q.files["draw.wgsl"] = "@fragment fn fs() { }\n";
    CHECK(cp::piece_hash(q) != h);
    q = p;
    q.version = 2;
    CHECK(cp::piece_hash(q) != h);
    CHECK(h.size() == 71);
}

TEST_CASE("the source check refuses directives, bindings and the names of the host and the modules") {
    const std::size_t max = cs::piece_limits::kFileBytes;
    CHECK(cp::check_piece_source("fn f(a: f32) -> f32 { let b = a; return b; }", max).ok());
    CHECK(cp::check_piece_source("// @group(0) in a comment\nfn f() {}", max).ok());
    CHECK(cp::check_piece_source("enable f16;\nfn f() {}", max).code == "program.directive_not_allowed");
    const cp::Verdict b = cp::check_piece_source("fn f() {}\n@group(1) @binding(0) var<storage> x: array<f32>;", max);
    CHECK(b.code == cp::kProgramBindingNotAllowed);
    CHECK(b.line == 2);
    for (const char* bad : {"fn caliper_fullscreen() {}", "struct CaliperView { a: f32 }", "const CALIPER_X = 1;",
                            "fn colormap__magma() {}", "fn f() { let a__b = 1; }", "var<private> caliper: f32;",
                            "alias Caliper2 = f32;"})
        CHECK(cp::check_piece_source(bad, max).code == cp::kProgramReservedName);
    CHECK(cp::check_piece_source(std::string(max + 1, ' '), max).code == cp::kPieceBadPath);
}

TEST_CASE("element layouts follow the storage rules of WGSL") {
    cp::ElementLayout l;
    REQUIRE(cp::element_layout(arcs_piece().resources[1].layout, &l).ok());
    CHECK(l.stride == 24);
    CHECK(l.fields[4].offset == 16);
    REQUIRE(cp::element_layout(struct_layout("Label", {{"position", "vec4<f32>"}, {"string", "u32"}}), &l).ok());
    CHECK(l.stride == 32);
    CHECK(l.align == 16);
    REQUIRE(cp::element_layout(struct_layout("P", {{"a", "u32"}, {"b", "vec2<f32>"}, {"c", "f32"}}), &l).ok());
    CHECK(l.fields[1].offset == 8);
    CHECK(l.stride == 24);
    cs::Layout plain;
    plain.element = "vec4<u32>";
    REQUIRE(cp::element_layout(plain, &l).ok());
    CHECK(l.stride == 16);
    plain.element = "vec3<f32>";
    CHECK(cp::element_layout(plain, &l).code == cp::kProgramBadLayout);
    CHECK(cp::element_layout(cs::Layout{}, &l).code == cp::kProgramBadLayout);
    CHECK(cp::element_layout(struct_layout("Params", {{"a", "f32"}}), &l).code == cp::kProgramBadLayout);
    CHECK(cp::element_layout(struct_layout("CaliperArc", {{"a", "f32"}}), &l).code == cp::kProgramBadLayout);
    CHECK(cp::element_layout(struct_layout("Arc", {{"a", "f32"}, {"a", "u32"}}), &l).code == cp::kProgramBadLayout);
    CHECK(cp::element_layout(struct_layout("Arc", {{"a", "mat4x4<f32>"}}), &l).code == cp::kProgramBadLayout);
    CHECK(cp::element_layout(struct_layout("Arc", {}), &l).code == cp::kProgramBadLayout);
}

TEST_CASE("the manifest check takes the arcs piece of the plan") {
    const cp::Verdict v = cp::check_piece_manifest(arcs_piece());
    INFO(v.message, " at ", v.where);
    CHECK(v.ok());
}

TEST_CASE("an indexed pass draws a u32 buffer from a vertex entry as a list, and every other form is refused") {
    cs::Piece p = arcs_piece();
    cs::Resource idx;
    idx.id = "idx";
    idx.layout.element = "u32";
    idx.count = 96;
    p.resources.push_back(idx);
    cs::DrawPass& arc = p.draw->passes[0];
    arc.indices = "idx";
    arc.topology = cs::DrawTopology::triangle_list;
    arc.vertices.reset();   // the index count comes from the version
    CHECK(cp::check_piece_manifest(p).ok());
    CHECK(cp::canonical_manifest(p).find(R"("indices":"idx")") != std::string::npos);
    cs::Piece bad = p;
    bad.draw->passes[0].topology = cs::DrawTopology::triangle_strip;
    CHECK(code_of(bad) == cp::kProgramBadPass);
    bad = p;
    bad.draw->passes[0].indices = "missing";
    CHECK(code_of(bad) == cp::kProgramUnknownResource);
    bad = p;
    bad.draw->passes[0].indices = "heads";   // f32 elements
    CHECK(code_of(bad) == cp::kProgramBadPass);
    bad = p;
    bad.draw->passes[0].indices = "labels";  // a struct
    CHECK(code_of(bad) == cp::kProgramBadPass);
    bad = p;
    bad.draw->passes[2].indices = "idx";     // the full-screen bloom has no vertex entry
    CHECK(code_of(bad) == cp::kProgramBadPass);
    bad = p;
    bad.resources.back().publish = false;     // a private resource never reaches the viewer
    CHECK(code_of(bad) == cp::kProgramUnknownResource);
}

TEST_CASE("a text table takes its strings from a published buffer of at most 8224 strings of 64 bytes") {
    cs::Piece p = arcs_piece();
    cs::Resource names;
    names.id = "names";
    names.layout = struct_layout("Name", {{"b0", "vec4<u32>"}, {"b1", "vec4<u32>"}, {"b2", "vec4<u32>"}, {"b3", "vec4<u32>"}});
    names.count = 8224;
    p.resources.push_back(names);
    p.draw->text->strings.clear();
    p.draw->text->strings_from = "names";
    p.draw->text->align = cs::TextAlign::center;
    CHECK(cp::check_piece_manifest(p).ok());
    const std::string canon = cp::canonical_manifest(p);
    CHECK(canon.find(R"("strings_from":"names")") != std::string::npos);
    CHECK(canon.find(R"("align":"center")") != std::string::npos);
    CHECK(cp::canonical_manifest(arcs_piece()).find("align") == std::string::npos);
    cs::Piece bad = p;
    bad.draw->text->strings = {"x"};   // two sources
    CHECK(code_of(bad) == cp::kProgramBadText);
    bad = p;
    bad.draw->text->strings_from = "missing";
    CHECK(code_of(bad) == cp::kProgramUnknownResource);
    bad = p;
    bad.resources.back().count = 8225;
    CHECK(code_of(bad) == cp::kProgramBadText);
    bad = p;
    bad.resources.back().layout.fields.push_back({"b4", "vec4<u32>"});   // 80 bytes a string
    CHECK(code_of(bad) == cp::kProgramBadText);
    bad = p;
    bad.draw->text->strings_from = "heads";   // f32 elements
    CHECK(code_of(bad) == cp::kProgramBadText);
    bad = p;
    bad.resources.back().publish = false;
    CHECK(code_of(bad) == cp::kProgramUnknownResource);
}

TEST_CASE("the manifest check refuses each breach with its code") {
    cs::Piece p = arcs_piece();
    p.name = "Arcs";
    CHECK(code_of(p) == cp::kPieceBadName);
    p = arcs_piece();
    p.derive.reset();
    p.draw.reset();
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.files["extra.wgsl"] = "";
    CHECK(code_of(p) == cp::kPieceBadPath);
    p = arcs_piece();
    p.files.erase("draw.wgsl");
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.resources[1].id = "heads";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs_piece();
    p.resources[0].id = "params";
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs_piece();
    p.inputs[0].name = "caliper_view";
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.resources[0].count = cs::piece_limits::kCountMax + 1;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs_piece();
    p.resources[1].count = cs::piece_limits::kCountMax;  // 33,554,432 arcs of 24 bytes: 768 MiB
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs_piece();
    p.resources[0].kind = cs::ResourceKind::texture;
    CHECK(code_of(p) == cp::kProgramBadResource);
    p = arcs_piece();
    p.resources[2].layout.fields[1].type = "vec3<f32>";
    CHECK(code_of(p) == cp::kProgramBadLayout);
    p = arcs_piece();
    p.imports.push_back("noise@1");
    CHECK(code_of(p) == cp::kPieceUnknownImport);
    p = arcs_piece();
    p.derive->passes[0].writes = {"nothing"};
    CHECK(code_of(p) == cp::kProgramUnknownResource);
    p = arcs_piece();
    p.derive->passes[0].entry = "main";
    CHECK(code_of(p) == cp::kProgramBadPass);
    p = arcs_piece();
    p.derive->passes[1].workgroups = {70000, 1, 1};
    CHECK(code_of(p) == cp::kProgramBadPass);
    p = arcs_piece();
    p.draw->passes[0].instances_of = "missing";
    CHECK(code_of(p) == cp::kProgramUnknownResource);
    p = arcs_piece();
    p.draw->passes[0].vertices.reset();
    CHECK(code_of(p) == cp::kProgramBadPass);
    p = arcs_piece();
    p.draw->passes[1].target = "bloom";
    CHECK(code_of(p) == cp::kProgramBadPass);
    p = arcs_piece();
    p.draw->passes[1].depth = cs::DrawDepth::write;
    CHECK(code_of(p) == cp::kProgramBadPass);
    p = arcs_piece();
    p.draw->passes[1].reads = {"glow"};  // reads its own target
    CHECK(code_of(p) == cp::kProgramTargetCycle);
    p = arcs_piece();
    std::swap(p.draw->passes[1], p.draw->passes[2]);  // the bloom reads glow before a pass draws it
    CHECK(code_of(p) == cp::kProgramTargetCycle);
    p = arcs_piece();
    p.draw->pick->fields.push_back("missing");
    CHECK(code_of(p) == cp::kProgramBadPick);
    p = arcs_piece();
    p.draw->passes[0].pick = false;
    CHECK(code_of(p) == cp::kProgramBadPick);
    p = arcs_piece();
    p.draw->pick.reset();
    CHECK(code_of(p) == cp::kProgramBadPick);
    p = arcs_piece();
    p.draw->text->position = "string";
    CHECK(code_of(p) == cp::kProgramBadText);
    p = arcs_piece();
    p.draw->text->strings.push_back(std::string(65, 'a'));
    CHECK(code_of(p) == cp::kProgramBadText);
    p = arcs_piece();
    p.draw->text->anchors = "heads_x";
    CHECK(code_of(p) == cp::kProgramUnknownResource);
    p = arcs_piece();
    p.params[0].default_value = 0.9;
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.params.push_back(cs::PieceParam{"layer", cs::ParamType::u32, -1, std::nullopt, std::nullopt, std::nullopt});
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.draw->targets[0].scale = 0.3f;
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.draw->frame_ms = 20.0;
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.hash = "sha256:XYZ";
    CHECK(code_of(p) == cp::kPieceBadManifest);
    p = arcs_piece();
    p.depends = {cs::PieceDependency{"ring_arcs", 1}};
    CHECK(code_of(p) == cp::kPieceCycle);
    p = arcs_piece();
    p.sample["attn0"] = cs::SampleInput{"sample/attn0.bin", {1, 4, 512, 512}};  // 4 MiB
    CHECK(code_of(p) == cp::kPieceBadPath);
    p = arcs_piece();
    p.sample["attn0"] = cs::SampleInput{"sample/../../etc.bin", {1, 4, 32, 32}};
    CHECK(code_of(p) == cp::kPieceBadPath);
}

TEST_CASE("the web limits hold the draw half to eight storage buffers in a stage") {
    cs::Piece p = arcs_piece();
    for (int i = 0; i < 5; ++i) {
        cs::Resource r;
        r.id = "extra" + std::to_string(i);
        r.layout.element = "f32";
        r.count = 4;
        p.resources.push_back(r);
    }
    CHECK(p.resources.size() == 8);
    CHECK(cp::check_web_limits(p).code == cp::kProgramOverWebLimits);  // 8 published buffers and the selection
    p.resources.back().publish = false;
    CHECK(cp::check_web_limits(p).ok());
    CHECK(cp::check_piece_manifest(p).ok());
}

TEST_CASE("a binding checks the tensor's dtype and rank, or the published buffer's layout") {
    const cs::Piece p = arcs_piece();
    CHECK(cp::check_tensor_binding(p.inputs[0], "f32", 4).ok());
    CHECK(cp::check_tensor_binding(p.inputs[0], "i32", 4).code == cp::kPieceBadBinding);
    CHECK(cp::check_tensor_binding(p.inputs[0], "f32", 3).code == cp::kPieceBadBinding);
    cs::PieceInput in;
    in.name = "arcs_in";
    in.kind = cs::InputKind::buffer;
    in.layout = struct_layout("ArcIn", p.resources[1].layout.fields);
    CHECK(cp::check_buffer_binding(in, p.resources[1].layout).ok());
    CHECK(cp::check_buffer_binding(in, p.resources[2].layout).code == cp::kPieceBadBinding);
    CHECK(cp::check_tensor_binding(in, "f32", 1).code == cp::kPieceBadBinding);
    CHECK(cp::check_buffer_binding(p.inputs[0], p.resources[1].layout).code == cp::kPieceBadBinding);
}


TEST_CASE("the lines and mesh library layouts retain full scene capacities within WebGPU limits") {
    constexpr std::uint32_t count = 4194304;
    for (const std::string name : {"lines", "mesh"}) {
        CAPTURE(name);
        cs::Piece piece;
        const std::string prefix = "pieces/" + name + "/1/";
        for (const auto& file : cp::library_files()) {
            if (file.path.substr(0, prefix.size()) != prefix) continue;
            const std::string leaf(file.path.substr(prefix.size()));
            if (leaf == "piece.toml") {
                const auto parsed = cs::piece_from_toml(file.text);
                REQUIRE(parsed.ok());
                piece = parsed.piece;
            }
        }
        for (const auto& file : cp::library_files())
            if (file.path.substr(0, prefix.size()) == prefix && file.path.substr(prefix.size()) != "piece.toml")
                piece.files[std::string(file.path.substr(prefix.size()))] = file.text;
        REQUIRE(piece.name == name);
        const auto verdict = cp::check_piece_manifest(piece);
        REQUIRE_MESSAGE(verdict.ok(), verdict.message);
        std::uint64_t bytes = 0;
        std::map<std::string, std::uint32_t> strides;
        std::map<std::string, std::uint32_t> counts;
        for (const auto& resource : piece.resources) {
            if (!resource.publish) continue;
            std::uint64_t size = 0;
            if (resource.kind == cs::ResourceKind::buffer) {
                cp::ElementLayout layout;
                REQUIRE(cp::element_layout(resource.layout, &layout).ok());
                strides[resource.id] = layout.stride;
                counts[resource.id] = resource.count;
                size = cp::layout_bytes(layout, resource.count);
                CHECK(size <= cs::piece_limits::kPublishedResourceBytesMax);
            } else {
                size = cs::texture_bytes(resource);
            }
            bytes += size;
        }
        if (name == "lines") {
            CHECK(strides["segments"] == 32);
            CHECK(strides["picks"] == 8);
            CHECK(counts["segments"] == count);
            CHECK(counts["picks"] == count);
            CHECK(bytes == (std::uint64_t(160) << 20) + 12);
            CHECK(piece.draw->pick->resource == "picks");
        } else {
            CHECK(strides["vertices"] == 32);
            CHECK(strides["attributes"] == 12);
            CHECK(counts["vertices"] == count);
            CHECK(counts["attributes"] == count);
            CHECK(counts["faces"] == 3 * count);
            CHECK(strides["info"] == 4);
            CHECK(counts["info"] == 1);
            CHECK(bytes == (std::uint64_t(480) << 20) + 4);
            CHECK(piece.draw->pick->resource == "attributes");
        }
        CHECK(bytes <= cs::piece_limits::kPublishedBytesMax);
        CHECK(bytes <= std::uint64_t(piece.draw->memory_mb.value()) << 20);
        if (!cp::compiler_available()) continue;
        const auto resolved = cp::resolve_imports(piece);
        REQUIRE(resolved.verdict.ok());
        const auto composed = cp::compose_derive(piece, resolved);
        REQUIRE(composed.verdict.ok());
        for (const auto& pass : piece.derive->passes) {
            CAPTURE(pass.entry);
            const auto compiled = cp::compile_and_reflect_entry(composed.text, pass.entry);
            REQUIRE_MESSAGE(compiled.verdict.ok(), compiled.verdict.message);
            std::size_t storage = 0;
            for (const auto& binding : compiled.reflection.bindings)
                storage += binding.kind == cp::BindingKind::storage_buffer;
            CHECK(storage <= 8);
            if (name == "mesh" && pass.entry == "copy_vertices") CHECK(storage == 8);
        }
    }
    cs::Piece p = arcs_piece();
    p.draw->memory_mb = 512;
    CHECK(cp::check_piece_manifest(p).ok());
    p.draw->memory_mb = 513;
    CHECK_FALSE(cp::check_piece_manifest(p).ok());
}


TEST_CASE("an r32uint texture declares at most one scalar item field") {
    cs::Piece p;
    p.name = "texture_fields";
    cs::Resource r;
    r.id = "pixels";
    r.kind = cs::ResourceKind::texture;
    r.format = "r32uint";
    r.width = r.height = 2;
    p.resources = {r};
    cs::Draw draw;
    cs::DrawPass pass;
    pass.fragment = "color_fs";
    draw.passes = {pass};
    p.draw = draw;
    p.files["draw.wgsl"] = "@fragment fn color_fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }";
    for (const char* type : {"f32", "i32", "u32"}) {
        p.resources[0].layout.fields = {{"sample", type}};
        CHECK_MESSAGE(cp::check_piece_manifest(p).ok(), cp::check_piece_manifest(p).message);
    }
    for (const char* type : {"u16", "vec2<f32>", "vec4<u32>"}) {
        p.resources[0].layout.fields = {{"sample", type}};
        CHECK(cp::check_piece_manifest(p).code == cp::kProgramBadResource);
    }
    p.resources[0].layout.fields = {{"first", "f32"}, {"second", "f32"}};
    CHECK(cp::check_piece_manifest(p).code == cp::kProgramBadResource);
    p.resources[0].layout.fields = {{"bad__field", "f32"}};
    CHECK(cp::check_piece_manifest(p).code == cp::kProgramBadResource);
    p.resources[0].layout.fields = {{"sample", "f32"}};
    p.resources[0].format = "rgba8unorm";
    CHECK(cp::check_piece_manifest(p).code == cp::kProgramBadResource);
    p.resources[0].format = "r8unorm";
    CHECK(cp::check_piece_manifest(p).code == cp::kProgramBadResource);
}
