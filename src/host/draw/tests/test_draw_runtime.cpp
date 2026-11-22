// Tests of the interface of the draw half in a viewer (draw/draw_runtime.h,
// design/programmable-graphics.md) on the graphics layer's recording
// backend, with no GPU: until the render graph is built, a program is
// refused with program.not_built and the runtime records nothing.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "draw/draw_runtime.h"
#include "gfx/recording_backend.h"

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;

TEST_CASE("a draw runtime refuses a program as not built and draws nothing") {
    caliper_gfx::RecordingBackend gfx;
    cd::DrawRuntime runtime(gfx, /*webgpu=*/false);
    CHECK(runtime.status().state == cd::DrawState::waiting);

    caliper_builder::spec::Piece piece;
    cd::Instance instance;
    CHECK(cd::DrawRuntime::decode("{}", "", "", &piece, &instance).code == cp::kProgramNotBuilt);
    piece.name = "attention_arcs";
    instance.program = "tower_arcs";
    const cp::Verdict v = runtime.install(piece, instance, 1);
    CHECK(v.code == cp::kProgramNotBuilt);
    CHECK(runtime.status().state == cd::DrawState::refused);
    CHECK(runtime.status().error.code == cp::kProgramNotBuilt);

    caliper_gfx::Frame frame;
    CHECK(!runtime.encode(frame, cd::FrameInputs{}, cd::HostTarget{}));
    CHECK(!runtime.pick(0));
    CHECK(runtime.labels().empty());
    runtime.remove();
    CHECK(runtime.status().state == cd::DrawState::waiting);
}
