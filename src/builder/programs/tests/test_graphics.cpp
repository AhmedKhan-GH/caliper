// Tests of the interface of the graphics form (programs/graphics.h,
// design/programmable-graphics.md): the draw prelude is the embedded file
// and compiles, the stubs answer program.not_built, and a program's view
// for the composer keeps its id, its params and its graphics form.
#include <caliper_builder/programs/graphics.h>
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
                             "caliper_colormaps", "fn caliper_fullscreen", "fn caliper_id", "struct CaliperPicked"})
        CHECK(text.find(name) != std::string_view::npos);
    CHECK(text.find("@group(1)") == std::string_view::npos);
    CHECK(text.find("enable ") == std::string_view::npos);
}

TEST_CASE("the draw prelude compiles with a fragment entry that uses its functions and bindings") {
    // The reflection lists the bindings that an entry point uses, so the
    // fragment uses all five.
    if (!cp::compiler_available()) return;
    std::string module(cp::draw_prelude());
    module += R"wgsl(
@fragment
fn fs_test(in: CaliperFullscreen) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = caliper_colormap(CALIPER_VIRIDIS, in.uv.x);
    if (caliper_selected(1u) || caliper_selected_index(vec4<i32>(0, -1, -1, -1))) {
        out.color = vec4<f32>(1.0);
    }
    out.color = out.color + 0.5 * textureSampleLevel(caliper_colormaps, caliper_linear, in.uv, 0.0) +
                0.5 * textureSampleLevel(caliper_colormaps, caliper_nearest, in.uv, 0.0);
    out.id = caliper_id(7u);
    return out;
}
)wgsl";
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

TEST_CASE("the composer and the checks of the graphics form answer program.not_built") {
    cp::GraphicsProgram g;
    g.id = "arcs";
    CHECK(cp::check_graphics_declaration(g, {}).code == cp::kProgramNotBuilt);
    CHECK(cp::check_graphics_source("fn f() {}", cs::graphics_limits::kSourceBytes).code == cp::kProgramNotBuilt);
    CHECK(cp::check_graphics_reflection(g, cp::Half::draw, 0, {}).code == cp::kProgramNotBuilt);
    CHECK(cp::check_web_limits(g).code == cp::kProgramNotBuilt);
    CHECK(cp::compose_derive(g).verdict.code == cp::kProgramNotBuilt);
    CHECK(cp::compose_draw(g, 0, {}).verdict.code == cp::kProgramNotBuilt);
    cp::ElementLayout layout;
    CHECK(cp::element_layout(cs::Layout{}, &layout).code == cp::kProgramNotBuilt);
}

TEST_CASE("graphics_program keeps the id, the params' names and the graphics form") {
    cs::Program p;
    p.id = "field";
    p.params["gain"] = 1.0;
    p.params["bias"] = 0.0;
    cs::Draw draw;
    draw.panel = "world";
    p.graphics.draw = draw;
    CHECK(cs::is_graphics(p));
    const cp::GraphicsProgram g = cp::graphics_program(p);
    CHECK(g.id == "field");
    CHECK(g.params == std::vector<std::string>{"bias", "gain"});
    CHECK(g.graphics == p.graphics);
}

TEST_CASE("the names of the graphics form's values round-trip") {
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
    CHECK(cs::resource_kind_from_name("texture") == cs::ResourceKind::texture);
    CHECK(cs::pass_kind_from_name("compute") == cs::PassKind::compute);
    CHECK(!cs::draw_blend_from_name("max"));
}

TEST_CASE("the codes of the graphics form are distinct") {
    const char* codes[] = {cp::kProgramNotBuilt,       cp::kProgramFormMixed,         cp::kProgramBadResource,
                           cp::kProgramBadLayout,      cp::kProgramUnknownResource,   cp::kProgramBadPass,
                           cp::kProgramTargetCycle,    cp::kProgramBadPick,           cp::kProgramBadText,
                           cp::kProgramUnknownState,   cp::kProgramCameraConflict,    cp::kProgramBindingNotAllowed, cp::kProgramReservedName,
                           cp::kProgramEntryStage,     cp::kProgramInterfaceMismatch, cp::kProgramOverWebLimits,
                           cp::kProgramResourcesUnavailable, cp::kProgramResourceTooLarge,
                           cp::kProgramDrawOverTime,   cp::kProgramDrawTimeout,       cp::kProgramDrawOverMemory,
                           cp::kProgramBrowserRefused};
    std::set<std::string> seen(std::begin(codes), std::end(codes));
    CHECK(seen.size() == std::size(codes));
    CHECK(seen.count(cp::kProgramCompileFailed) == 0);
}
