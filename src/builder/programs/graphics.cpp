// The composer and the checks of the graphics form (programs/graphics.h).
// Only draw_prelude() is built, in the generated caliper_draw_prelude.cpp;
// every other function answers program.not_built until the work of
// design/programmable-graphics.md fills it. This file needs only the
// standard library and spec/graphics.h, because the browser module compiles
// it too.
#include <caliper_builder/programs/graphics.h>

namespace caliper_builder::programs {

namespace {

Verdict not_built(const char* what) {
    Verdict v;
    v.code = kProgramNotBuilt;
    v.message = std::string(what) + " of programmable graphics is not built yet.";
    return v;
}

}  // namespace

Verdict element_layout(const spec::Layout&, ElementLayout* out) {
    if (out) *out = ElementLayout{};
    return not_built("The layout of a resource");
}

SourcePlace source_place(const Composed&, Half, std::uint32_t line) { return SourcePlace{"header", line}; }

Composed compose_derive(const GraphicsProgram&) {
    Composed c;
    c.verdict = not_built("The composer of the derive half");
    return c;
}

Composed compose_draw(const GraphicsProgram&, std::size_t, const std::vector<StateType>&) {
    Composed c;
    c.verdict = not_built("The composer of the draw half");
    return c;
}

Verdict check_graphics_declaration(const GraphicsProgram&, const std::vector<StateType>&) {
    return not_built("The declaration check");
}

Verdict check_graphics_source(std::string_view, std::size_t) { return not_built("The source check"); }

Verdict check_graphics_reflection(const GraphicsProgram&, Half, std::size_t, const Reflection&) {
    return not_built("The reflection check");
}

Verdict check_web_limits(const GraphicsProgram&) { return not_built("The check of the WebGPU limits"); }

}  // namespace caliper_builder::programs
