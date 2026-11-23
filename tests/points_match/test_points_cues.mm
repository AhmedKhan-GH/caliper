// The points piece retains the original depth cues, colors and pick identities.
// Both cue fixtures use saved reference images and the same camera bounds.
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#include "../boxes_match/match_rig.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace {

using namespace match_rig;

constexpr std::uint32_t kSize = 256;
constexpr int kCount = 256;
constexpr float kSizePx = 9.0f;

// points@1's params by name, in the manifest's order; the others keep their defaults.
std::vector<std::uint8_t> params_of(const cs::Piece& piece, const std::vector<std::pair<std::string, double>>& values) {
    std::vector<std::uint8_t> out(piece.params.size() * 4, 0);
    for (std::size_t i = 0; i < piece.params.size(); ++i) {
        const cs::PieceParam& p = piece.params[i];
        double v = p.default_value;
        for (const auto& [name, given] : values)
            if (name == p.name) v = given;
        std::uint32_t bits = 0;
        if (p.type == cs::ParamType::f32) {
            const float f = float(v);
            std::memcpy(&bits, &f, 4);
        } else {
            bits = std::uint32_t(v);
        }
        std::memcpy(out.data() + 4 * i, &bits, 4);
    }
    return out;
}

void run_cues(bool packed) {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    gfx::Backend& g = *m.backend;
    const std::string coloring = packed ? "rgba" : "viridis";
    CAPTURE(coloring);

    // The cloud: positions from a fixed LCG in [-1, 1]^3, values at exact colormap levels, and colors
    // of whole levels.
    std::vector<float> positions, values;
    std::vector<std::uint32_t> colors;
    std::uint32_t state = 777u;
    auto next = [&]() {
        state = state * 1664525u + 1013904223u;
        return float(double(state >> 8) / double(1u << 24)) * 2.0f - 1.0f;
    };
    for (int i = 0; i < kCount; ++i) {
        for (int k = 0; k < 3; ++k) positions.push_back(next());
        values.push_back(float((i * 37) % 256) / 255.0f);
        colors.push_back(std::uint32_t((i * 53 + 40) % 256) | std::uint32_t((i * 97 + 90) % 256) << 8 |
                         std::uint32_t((i * 31 + 200) % 256) << 16 | 0xff000000u);
    }

    // ---- The built-in draw, with the cues of its fitted camera ----------------------------------
    ReferenceScene b(g, kSize, sc::Space::k3D);
    sc::Batch batch;
    batch.primitive = sc::Primitive::Points;
    batch.size_px = kSizePx;
    batch.point_shape = sc::PointShape::Square;
    Upload up;
    up.add(sc::Role::Positions, positions, kCount, 12, gfx::VertexFormat::F32x3);
    if (packed) {
        up.add(sc::Role::Colors, colors, kCount, 4, gfx::VertexFormat::Unorm8x4);
    } else {
        up.add(sc::Role::Values, values, kCount, 4, gfx::VertexFormat::F32);
        batch.colormap = 0;
        batch.vmin = 0.0f;
        batch.vmax = 1.0f;
    }
    for (int i = 0; i < kCount; ++i) batch.bounds.add({positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]});
    b.set("cloud", batch, up);
    b.tree.update();
    sc::fit(b.camera, b.tree.scene_bounds(), 1.0);
    const sc::Orbit* orbit = std::get_if<sc::Orbit>(&b.camera);
    REQUIRE(orbit != nullptr);
    const double radius = sc::length(b.tree.scene_bounds().half_extent());
    // As PointsScene::cue_values gives them: the size at the target's distance, the fade from the
    // target to the far side of the fitted sphere.
    const float att_ref = float(orbit->distance), fade_w0 = float(orbit->distance), fade_w1 = float(orbit->distance + radius);
    b.load();

    // ---- The piece's draw, with the same cues as params ----------------------------------------
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("points", &piece, &why), why);
    PieceDraw p(g, kSize);
    p.install(piece, "cloud");
    std::vector<std::uint8_t> rows(std::size_t(kCount) * 32, 0);
    for (int i = 0; i < kCount; ++i) {
        std::uint8_t* row = rows.data() + std::size_t(i) * 32;
        const float pos[4] = {positions[3 * i], positions[3 * i + 1], positions[3 * i + 2], 0.0f};
        std::uint32_t bits = colors[std::size_t(i)];
        if (!packed) std::memcpy(&bits, &values[std::size_t(i)], 4);
        const std::uint32_t index = std::uint32_t(i);
        std::memcpy(row, pos, 16);
        std::memcpy(row + 16, &bits, 4);
        std::memcpy(row + 20, &index, 4);
    }
    p.set("points", rows, kCount);
    cd::FrameInputs in = b.frame_inputs();
    in.params = params_of(piece, {{"size", kSizePx},
                                  {"shape", 0},
                                  {"color_mode", packed ? 2 : 1},
                                  {"colormap", 0},
                                  {"vmin", 0.0},
                                  {"vmax", 1.0},
                                  {"att_ref", att_ref},
                                  {"fade_w0", fade_w0},
                                  {"fade_w1", fade_w1}});
    REQUIRE(p.draw(in));

    // ---- The comparison and the picks -----------------------------------------------------------
    const Comparison c = compare(p.color, p.ids, b.image, kSize);
    MESSAGE("points cues (", coloring, "): ", c.differ, " differing pixels, ", c.interior, " interior pixels (",
            c.lit_interior, " lit, largest gap ", c.max_interior, "), ", c.edge, " edge pixels (largest gap ", c.max_edge,
            "), ", c.failed, " failed");
    CHECK_MESSAGE(c.failed == 0, c.failed, " pixels outside the tolerances; ", c.first);
    CHECK(c.lit_interior >= 2000);
    // 16 points that show an interior pixel in the piece's ID image, spread over the cloud: the
    // built-in's pick there names each one.
    std::vector<std::uint64_t> shown;
    for (int i = 0; i < kCount; ++i)
        if (pixel_of_item(p.ids, kSize, std::uint64_t(i))) shown.push_back(std::uint64_t(i));
    REQUIRE(shown.size() >= 16);
    std::vector<std::uint64_t> items;
    for (const auto& probe : match_reference::metadata(b.reference).at("picks"))
        items.push_back(probe.at("item").get<std::uint64_t>());
    REQUIRE(items.size() == 16);
    const int agree = check_picks(b, p, items, sc::Primitive::Points);
    MESSAGE("points cues (", coloring, "): ", shown.size(), " points show an interior pixel; ", agree,
            " of 16 picks name the piece's item");
    CHECK(agree == 16);

    match_rig::capture("CALIPER_POINTS_MATCH_CAPTURE_DIR", "points-cues-" + coloring + "-builtin.png", kSize, b.image);
    match_rig::capture("CALIPER_POINTS_MATCH_CAPTURE_DIR", "points-cues-" + coloring + "-piece.png", kSize, p.color);
    match_rig::capture("CALIPER_POINTS_MATCH_CAPTURE_DIR", "points-cues-" + coloring + "-diff.png", kSize,
                       difference(p.color, b.image));
}

}  // namespace

TEST_CASE("points_match: points@1 with the depth cues draws as the built-in's cues, packed colors") { run_cues(true); }

TEST_CASE("points_match: points@1 with the depth cues draws as the built-in's cues, a colormap") { run_cues(false); }
