// The graphics suite's scene-core row, on the CPU: the one-entity scene of scene_row_fixture.h
// compiled into a draw list, whose draw is then rasterised the way vs_points and fs_points
// place a square point (the quad of size_px around the projected centre, pixel centres inside
// it covered), from the bytes its vertex bindings name. The result must equal the row's
// reference byte for byte. The GPU row draws the same list on Metal once the layer's core and
// Metal backend are on the branch.
#include "scene_row_fixture.h"
#include "fake_resources.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>

using namespace caliper_scene;
using namespace scene_fixtures;
namespace sh = caliper_gfx::shaders;

TEST_CASE("the scene row's fixture: its compiled draw covers exactly the reference's pixels") {
    const SceneRowFixture fx;
    const std::vector<uint8_t> bytes = fx.bytes();
    const caliper_gfx::Buffer buffer{0x4242};
    SceneTree t(Space::k2D);
    REQUIRE(t.set_batch(t.root(), fx.batch(buffer, bytes.data())));
    t.update();
    FakeResources res;
    SceneCompiler compiler;
    DrawList list{RenderTarget{Texture{1}, {}, {}, fx.kClear, true}};
    DrawMap map;
    const CompileResult r = compiler.compile(t, fx.camera(), fx.options(), res, list, &map);
    REQUIRE(r.ok);
    REQUIRE(list.records().size == 1);
    const DrawRecord& d = list.records()[0];
    const PipelineRecord& p = res.pipeline(d.pipeline);
    CHECK(p.program == ProgramId::scene_points);
    CHECK(p.vs == "vs_points");
    CHECK(p.fs == "fs_points");
    CHECK_FALSE(p.desc.depth);
    REQUIRE(d.count == 6);
    REQUIRE(d.instance_count == 3);
    const sh::PrimParams q = params_of(list, d);
    const sh::Frame f = frame_of(list, d);
    CHECK(q.color_mode == 2);
    CHECK(q.point_shape == 0);
    CHECK(q.size_px == 1.0f);

    // Rasterise the draw from the buffer bytes its bindings name.
    std::vector<uint8_t> image(size_t(fx.kWidth) * fx.kHeight * 4);
    for (int i = 0; i < fx.kWidth * fx.kHeight; ++i) std::memcpy(image.data() + 4 * i, &fx.kClear, 4);
    REQUIRE(d.vertex[0].buffer == buffer);
    REQUIRE(d.vertex[1].buffer == buffer);
    for (uint32_t inst = 0; inst < d.instance_count; ++inst) {
        float pos[2];
        std::memcpy(pos, bytes.data() + d.vertex[0].offset + inst * p.desc.vertex_layouts[0].stride, 8);
        uint32_t src;
        std::memcpy(&src, bytes.data() + d.vertex[1].offset + inst * p.desc.vertex_layouts[1].stride, 4);
        // c = mvp * (x, y, 0, 1) in f32, as the vertex stage computes it.
        float c[4];
        for (int row = 0; row < 4; ++row) c[row] = q.mvp[row] * pos[0] + q.mvp[4 + row] * pos[1] + q.mvp[12 + row];
        const float kx = q.size_px / f.viewport[0] * c[3], ky = q.size_px / f.viewport[1] * c[3];
        const float x0 = ((c[0] - kx) / c[3] + 1) * 0.5f * f.viewport[0];
        const float x1 = ((c[0] + kx) / c[3] + 1) * 0.5f * f.viewport[0];
        const float ya = (1 - (c[1] - ky) / c[3]) * 0.5f * f.viewport[1];
        const float yb = (1 - (c[1] + ky) / c[3]) * 0.5f * f.viewport[1];
        const float y0 = std::min(ya, yb), y1 = std::max(ya, yb);
        int covered = 0;
        for (int j = 0; j < fx.kHeight; ++j)
            for (int i = 0; i < fx.kWidth; ++i) {
                const float cx = i + 0.5f, cy = j + 0.5f;
                if (cx > x0 && cx < x1 && cy > y0 && cy < y1) {
                    std::memcpy(image.data() + (size_t(j) * fx.kWidth + i) * 4, &src, 4);
                    ++covered;
                }
            }
        CHECK(covered == 1);
    }
    const std::vector<uint8_t> ref = fx.reference();
    REQUIRE(image.size() == ref.size());
    CHECK(image == ref);
}
