// Tests of the interface of pieces (programs/pieces.h,
// design/programmable-graphics.md): the draw prelude and the lib modules are
// the embedded files and compile together the way a resolved piece does, the
// stubs answer program.not_built, and a program pins a piece.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>

#include <doctest/doctest.h>

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

TEST_CASE("the functions of pieces answer program.not_built") {
    cs::Piece piece;
    piece.name = "arcs";
    CHECK(cp::check_piece_name("arcs", 1).code == cp::kProgramNotBuilt);
    CHECK(cp::check_piece_file(piece, "draw.wgsl").code == cp::kProgramNotBuilt);
    CHECK(cp::resolve_imports(piece).verdict.code == cp::kProgramNotBuilt);
    CHECK(cp::piece_hash(piece).empty());
    CHECK(cp::check_piece_manifest(piece).code == cp::kProgramNotBuilt);
    CHECK(cp::check_piece_source("fn f() {}", cs::piece_limits::kFileBytes).code == cp::kProgramNotBuilt);
    CHECK(cp::check_piece_reflection(piece, cp::Half::draw, 0, {}).code == cp::kProgramNotBuilt);
    CHECK(cp::check_web_limits(piece).code == cp::kProgramNotBuilt);
    const cp::Resolved resolved;
    CHECK(cp::compose_derive(piece, resolved).verdict.code == cp::kProgramNotBuilt);
    CHECK(cp::compose_draw(piece, resolved, 0).verdict.code == cp::kProgramNotBuilt);
    cp::ElementLayout layout;
    CHECK(cp::element_layout(cs::Layout{}, &layout).code == cp::kProgramNotBuilt);
    CHECK(cp::check_tensor_binding(cs::PieceInput{}, "f32", 4).code == cp::kProgramNotBuilt);
}

TEST_CASE("a program that pins a piece is an instance with its pin, bindings and panel") {
    cs::Program a;
    a.id = "tower_arcs";
    CHECK(!cs::is_piece_instance(a));
    a.piece = cs::PieceRef{"attention_arcs", 1, "sha256:00", cs::PieceFrom::workspace};
    a.bind["attn0"] = "live:blocks.0.attn.weights.act";
    a.panel = "tower";
    CHECK(cs::is_piece_instance(a));
    const cs::Program b = a;
    CHECK(b.piece == a.piece);
    CHECK(b.bind == a.bind);
    CHECK(!(cs::PieceRef{"attention_arcs", 2, "sha256:00", cs::PieceFrom::workspace} == *a.piece));
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
