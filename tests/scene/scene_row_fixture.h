// scene_row_fixture.h: the one-entity scene of the graphics suite's scene-core row
// (design/renderer-core.md, Migration step 3: "a gfx row of a one-entity scene").
//
// A 2-D scene whose pan and zoom maps scene units onto pixels (x right, y down, the origin at the
// top-left), holding one points batch: three square points one pixel wide at the centres of
// chosen pixels, each with its own RGBA colour from a colour stream, in one buffer at nonzero
// offsets, as the existing geometry rows place their points. The reference is the clear colour
// everywhere and each point's colour at its pixel. The GPU row (tests/gfx/gfx_scene.cpp) draws the
// compiled list on the layer's Metal backend with the generated programs and compares the
// readback with reference() byte for byte; caliper_scene_tests checks the compiled draw against
// the reference on the CPU.
#pragma once

#include "scene/camera.h"
#include "scene/compile.h"
#include "scene/scene.h"

#include <cstring>
#include <utility>
#include <vector>

namespace scene_fixtures {

struct SceneRowFixture {
    static constexpr int kWidth = 64, kHeight = 64;
    static constexpr uint64_t kPositionsOffset = 512, kColorsOffset = 2048, kBytes = 4096;
    static constexpr uint32_t kClear = 0xff000000u;
    std::vector<std::pair<int, int>> pixels{{3, 5}, {40, 22}, {63, 63}};
    std::vector<uint32_t> colors{0xff2040ffu, 0xff10e070u, 0xffffffffu};   // r | g << 8 | b << 16 | a << 24

    // The buffer's contents: f32 (x, y) at pixel centres, then packed RGBA.
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> b(kBytes, 0);
        for (size_t i = 0; i < pixels.size(); ++i) {
            const float xy[2] = {float(pixels[i].first) + 0.5f, float(pixels[i].second) + 0.5f};
            std::memcpy(b.data() + kPositionsOffset + 8 * i, xy, 8);
            std::memcpy(b.data() + kColorsOffset + 4 * i, &colors[i], 4);
        }
        return b;
    }

    // The scene: the root of a 2-D scene with y down holds the batch.
    caliper_scene::Batch batch(caliper_gfx::Buffer buffer, const uint8_t* cpu = nullptr) const {
        using namespace caliper_scene;
        Batch b;
        b.primitive = Primitive::Points;
        BatchArray pos;
        pos.role = Role::Positions;
        pos.buffer = buffer;
        pos.offset = kPositionsOffset;
        pos.count = pixels.size();
        pos.stride = 8;
        pos.format = caliper_gfx::VertexFormat::F32x2;
        pos.cpu = cpu ? cpu + kPositionsOffset : nullptr;
        BatchArray col = pos;
        col.role = Role::Colors;
        col.offset = kColorsOffset;
        col.stride = 4;
        col.format = caliper_gfx::VertexFormat::Unorm8x4;
        col.cpu = cpu ? cpu + kColorsOffset : nullptr;
        b.arrays = {pos, col};
        b.size_px = 1.0f;
        b.point_shape = PointShape::Square;
        b.bounds.add({0, 0, 0});
        b.bounds.add({double(kWidth), double(kHeight), 0});
        return b;
    }

    // Scene units are pixels: the centre of the target, half its height.
    caliper_scene::ViewCamera camera() const {
        caliper_scene::PanZoom p;
        p.center[0] = kWidth / 2.0;
        p.center[1] = kHeight / 2.0;
        p.half_height = kHeight / 2.0;
        p.y_down = true;
        return p;
    }

    caliper_scene::CompileOptions options() const {
        caliper_scene::CompileOptions o;
        o.width = kWidth;
        o.height = kHeight;
        o.pixel_scale = 1.0f;
        o.clear_rgba = kClear;
        return o;
    }

    // The expected readback, RGBA8 top row first.
    std::vector<uint8_t> reference() const {
        std::vector<uint8_t> ref(size_t(kWidth) * kHeight * 4);
        for (int i = 0; i < kWidth * kHeight; ++i) std::memcpy(ref.data() + 4 * i, &kClear, 4);
        for (size_t k = 0; k < pixels.size(); ++k)
            std::memcpy(ref.data() + (size_t(pixels[k].second) * kWidth + pixels[k].first) * 4, &colors[k], 4);
        return ref;
    }
};

}  // namespace scene_fixtures
