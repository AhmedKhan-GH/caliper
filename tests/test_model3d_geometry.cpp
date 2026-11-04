#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/model3d_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace caliper_live::model3d;
constexpr double pi = 3.14159265358979323846;

Tensor tensor(std::string name, std::vector<int64_t> shape, int layer = -1,
              std::string type = {}, std::string role = {}) {
    Tensor result;
    result.name = std::move(name); result.shape = std::move(shape);
    result.layer_index = layer; result.layer_type = std::move(type); result.role = std::move(role);
    if (result.shape.empty()) result.rows = result.cols = 1;
    else if (result.shape.size() == 1) { result.rows = result.shape[0]; result.cols = 1; }
    else {
        result.rows = 1;
        for (size_t i = 0; i + 1 < result.shape.size(); ++i) result.rows *= result.shape[i];
        result.cols = result.shape.back();
    }
    return result;
}

std::vector<Tensor> qwen_shapes() {
    std::vector<Tensor> result;
    Tensor embedding = tensor("language_model.model.embed_tokens.weight", {248320, 1024}, -1, {}, "embedding");
    embedding.tied = true;
    result.push_back(embedding);
    for (int layer = 0; layer < 24; ++layer) {
        const bool full = layer % 4 == 3;
        const std::string type = full ? "full_attention" : "linear_attention";
        const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
        auto add = [&](const char* name, std::vector<int64_t> shape) {
            result.push_back(tensor(prefix + name, std::move(shape), layer, type));
        };
        add("input_layernorm.weight", {1024});
        add("post_attention_layernorm.weight", {1024});
        add("mlp.down_proj.weight", {1024, 3584});
        add("mlp.gate_proj.weight", {3584, 1024});
        add("mlp.up_proj.weight", {3584, 1024});
        if (full) {
            add("self_attn.k_norm.weight", {256});
            add("self_attn.q_norm.weight", {256});
            add("self_attn.k_proj.weight", {512, 1024});
            add("self_attn.v_proj.weight", {512, 1024});
            add("self_attn.o_proj.weight", {1024, 2048});
            add("self_attn.q_proj.weight", {4096, 1024});
        } else {
            add("linear_attn.A_log", {16});
            add("linear_attn.norm.weight", {128});
            add("linear_attn.conv1d.weight", {6144, 4, 1});
            add("linear_attn.dt_bias", {16});
            add("linear_attn.in_proj_a.weight", {16, 1024});
            add("linear_attn.in_proj_b.weight", {16, 1024});
            add("linear_attn.in_proj_qkv.weight", {6144, 1024});
            add("linear_attn.in_proj_z.weight", {2048, 1024});
            add("linear_attn.out_proj.weight", {1024, 2048});
        }
    }
    result.push_back(tensor("language_model.model.norm.weight", {1024}, -1, {}, "final_norm"));
    return result;
}

Camera front_camera() {
    Camera camera;
    camera.distance = 5; camera.radius = 5;
    camera.azimuth = pi * 0.5; camera.elevation = 0;
    return camera;
}

Scene overlapping_scene() {
    Scene scene;
    scene.panels.push_back({0, "back", {0, 0, 0}, 2, 2, 0, false});
    scene.panels.push_back({1, "front", {0, 0, 1}, 2, 2, 1, false});
    scene.minimum = {-1, -1, 0}; scene.maximum = {1, 1, 1};
    return scene;
}

std::array<double, 4> transform(const std::array<float, 16>& m, std::array<double, 4> v) {
    std::array<double, 4> out{};
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) out[row] += m[col * 4 + row] * v[col];
    return out;
}
} // namespace

TEST_CASE("model3d uses one element scale and horizontal vector strips") {
    const auto matrix = panel_size(tensor("matrix", {3584, 1024}));
    CHECK(matrix[0] == doctest::Approx(1));
    CHECK(matrix[1] == doctest::Approx(3.5));
    CHECK(matrix[1] / matrix[0] == doctest::Approx(3584.0 / 1024));
    const auto embedding = panel_size(tensor("embedding", {248320, 1024}));
    CHECK(embedding[0] == matrix[0]);
    CHECK(embedding[1] == doctest::Approx(242.5));
    const auto norm = panel_size(tensor("norm", {1024}));
    CHECK(norm[0] == matrix[0]);
    CHECK(norm[1] == doctest::Approx(element_unit));
    const auto bias = panel_size(tensor("bias", {16}));
    CHECK(bias[0] == doctest::Approx(16 * element_unit));
    CHECK(bias[1] == norm[1]);
    const auto conv = panel_size(tensor("conv", {6144, 4, 1}));
    CHECK(conv[0] == doctest::Approx(element_unit));
    CHECK(conv[1] == doctest::Approx(24));
    CHECK(panel_size(tensor("invalid", {-1}))[0] == 0);
    CHECK(panel_size(Tensor{})[0] == 0);
}

TEST_CASE("model3d packs all320 real Qwen shapes without changing dimensions") {
    const auto tensors = qwen_shapes();
    REQUIRE(tensors.size() == 320);
    const Scene scene = layout(tensors);
    REQUIRE(scene.panels.size() == 321); // one linked output alias; no second weight
    REQUIRE(scene.groups.size() == 27);
    CHECK(scene.groups.front().label == "Embedding");
    CHECK(scene.groups[25].label == "Final norm");
    CHECK(scene.groups.back().label == "Output (tied)");
    int linear = 0, full = 0;
    for (size_t i = 1; i <= 24; ++i) {
        const Group& group = scene.groups[i];
        CHECK(group.layer_index == static_cast<int>(i - 1));
        CHECK(group.z > scene.groups[i - 1].z);
        if (group.layer_type == "linear_attention") ++linear;
        if (group.layer_type == "full_attention") ++full;
    }
    CHECK(linear == 18); CHECK(full == 6);
    REQUIRE(scene.panels.back().output_alias);
    CHECK(scene.panels.back().tensor == 0);
    CHECK(scene.panels.back().width == scene.panels.front().width);
    CHECK(scene.panels.back().height == scene.panels.front().height);
    for (const Panel& panel : scene.panels) {
        const auto expected = panel_size(tensors[panel.tensor]);
        CHECK(panel.width == expected[0]); CHECK(panel.height == expected[1]);
        const Group& group = scene.groups[panel.group];
        for (Vec3 p : corners(panel)) {
            CHECK(p.x >= group.bounds.x - 1e-9);
            CHECK(p.x <= group.bounds.x + group.bounds.width + 1e-9);
            CHECK(p.y >= group.bounds.y - 1e-9);
            CHECK(p.y <= group.bounds.y + group.bounds.height + 1e-9);
            CHECK(p.z == group.z);
        }
    }
    for (size_t a = 0; a < scene.panels.size(); ++a) {
        for (size_t b = a + 1; b < scene.panels.size(); ++b) {
            const Panel& p = scene.panels[a]; const Panel& q = scene.panels[b];
            if (p.group != q.group) continue;
            const bool overlap_x = std::fabs(p.center.x - q.center.x) < (p.width + q.width) * 0.5 - 1e-10;
            const bool overlap_y = std::fabs(p.center.y - q.center.y) < (p.height + q.height) * 0.5 - 1e-10;
            CHECK_FALSE((overlap_x && overlap_y));
        }
    }
}

TEST_CASE("model3d layer order and legacy metadata inference are deterministic") {
    std::vector<Tensor> tensors{
        tensor("language_model.model.norm.weight", {1024}),
        tensor("language_model.model.layers.10.self_attn.q_proj.weight", {4096, 1024}),
        tensor("language_model.model.embed_tokens.weight", {248320, 1024}),
        tensor("language_model.model.layers.2.linear_attn.in_proj_a.weight", {16, 1024}),
    };
    const Scene scene = layout(tensors);
    REQUIRE(scene.groups.size() == 4);
    CHECK(scene.groups[0].label == "Embedding");
    CHECK(scene.groups[1].layer_index == 2);
    CHECK(scene.groups[1].layer_type == "linear_attention");
    CHECK(scene.groups[2].layer_index == 10);
    CHECK(scene.groups[2].layer_type == "full_attention");
    CHECK(scene.groups[3].label == "Final norm");
}

TEST_CASE("model3d fit projects every bounding corner at wide and narrow aspects") {
    const Scene scene = layout(qwen_shapes());
    for (double aspect : {0.4, 1.0, 2.5}) {
        const Rect viewport{17, 29, 800 * aspect, 800};
        const Camera camera = fit(scene, aspect);
        CHECK(camera.distance > camera.radius);
        for (const Panel& panel : scene.panels) {
            for (Vec3 corner : corners(panel)) {
                const auto p = project(corner, camera, aspect, viewport);
                CHECK(p.visible);
                CHECK(p.x >= viewport.x); CHECK(p.x <= viewport.x + viewport.width);
                CHECK(p.y >= viewport.y); CHECK(p.y <= viewport.y + viewport.height);
                CHECK(p.depth >= 0); CHECK(p.depth <= 1);
            }
        }
    }
    const Camera empty = fit(Scene{}, 0);
    CHECK(std::isfinite(empty.distance)); CHECK(empty.distance > 0);
}

TEST_CASE("model3d projection agrees with renderer matrices and top-left UV") {
    const Camera camera = front_camera();
    const Rect viewport{20, 40, 800, 600};
    const double aspect = viewport.width / viewport.height;
    const Matrices m = matrices(camera, aspect);
    const Vec3 point{0.7, 0.4, 0.5};
    const auto p = project(point, camera, aspect, viewport);
    const auto clip = transform(m.projection, transform(m.view, {point.x, point.y, point.z, 1}));
    CHECK(p.x == doctest::Approx(viewport.x + (clip[0] / clip[3] * 0.5 + 0.5) * viewport.width));
    CHECK(p.y == doctest::Approx(viewport.y + (0.5 - clip[1] / clip[3] * 0.5) * viewport.height));
    CHECK(p.depth == doctest::Approx(clip[2] / clip[3]));
    Scene scene;
    scene.panels.push_back({0, "panel", {0, 0, 0}, 2, 2});
    const Vec3 cell{-0.5, 0.5, 0};
    const auto pixel = project(cell, camera, aspect, viewport);
    const Hit hit = hit_test(scene, camera, viewport, pixel.x, pixel.y);
    REQUIRE(hit.panel == 0);
    CHECK(hit.u == doctest::Approx(0.25)); CHECK(hit.v == doctest::Approx(0.25));
    const auto c = corners(scene.panels.front());
    CHECK(c[0].x == -1); CHECK(c[0].y == 1);
    CHECK(c[2].x == 1); CHECK(c[2].y == -1);
}

TEST_CASE("model3d ray hits choose nearest visible panel and reject outside pixels") {
    const Camera camera = front_camera();
    const Rect viewport{20, 40, 800, 600};
    const Scene scene = overlapping_scene();
    const Hit hit = hit_test(scene, camera, viewport, 420, 340);
    REQUIRE(hit.panel == 1);
    CHECK(hit.u == doctest::Approx(0.5)); CHECK(hit.v == doctest::Approx(0.5));
    CHECK_FALSE(hit_test(scene, camera, viewport, 19, 340).panel >= 0);
    CHECK_FALSE(hit_test(scene, camera, viewport, 820, 340).panel >= 0);
    CHECK_FALSE(hit_test(scene, camera, viewport, 420, 640).panel >= 0);
    CHECK_FALSE(hit_test(scene, camera, viewport, 21, 41).panel >= 0);
    Scene behind;
    behind.panels.push_back({0, "behind", {0, 0, 10}, 20, 20});
    CHECK_FALSE(hit_test(behind, camera, viewport, 420, 340).panel >= 0);
    CHECK_FALSE(project({0, 0, 10}, camera, 4.0 / 3, viewport).visible);
}

TEST_CASE("model3d thin strip pointer tolerance preserves dimensions") {
    const Camera camera = front_camera();
    const Rect viewport{0, 0, 800, 600};
    Scene scene;
    scene.panels.push_back({0, "norm", {0, 0, 0}, 1, element_unit});
    const auto center = project({0, 0, 0}, camera, 4.0 / 3, viewport);
    CHECK(hit_test(scene, camera, viewport, center.x, center.y + 2).panel == 0);
    CHECK(hit_test(scene, camera, viewport, center.x, center.y + 6).panel == -1);
    CHECK(scene.panels[0].height == element_unit);
}

TEST_CASE("model3d orbit pan zoom and focus remain finite and predictable") {
    const Scene scene = layout(qwen_shapes());
    const Camera original = fit(scene, 1.5);
    const Camera spun = orbit(original, 25, -15);
    CHECK(spun.azimuth == doctest::Approx(original.azimuth + 0.25));
    CHECK(spun.elevation == doctest::Approx(original.elevation - 0.15));
    CHECK(orbit(original, 0, 1e9).elevation <= 1.55);
    const Camera closer = zoom(original, 2);
    CHECK(closer.distance < original.distance);
    CHECK(zoom(original, -2).distance > original.distance);
    const Camera moved = pan(original, 30, -10, 600);
    CHECK(std::isfinite(moved.target.x));
    CHECK(std::hypot(moved.target.x - original.target.x, moved.target.y - original.target.y) > 0);
    CHECK(pan(original, 30, -10, 0).target.x == original.target.x);
    const size_t norm = scene.panels.size() - 2;
    const Camera selected = focus(scene, norm, 1.5, original);
    CHECK(selected.distance < original.distance);
    CHECK(selected.azimuth == original.azimuth);
    CHECK(selected.target.z == scene.panels[norm].center.z);
    for (Vec3 corner : corners(scene.panels[norm])) CHECK(project(corner, selected, 1.5, {0, 0, 900, 600}).visible);
    Camera invalid;
    invalid.distance = std::numeric_limits<double>::quiet_NaN();
    invalid.target.x = std::numeric_limits<double>::infinity();
    const Camera recovered = zoom(invalid, 1);
    CHECK(std::isfinite(recovered.distance)); CHECK(std::isfinite(recovered.target.x));
}
