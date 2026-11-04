#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/model3d_presentation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace caliper_live::model3d;
constexpr double pi = 3.14159265358979323846;

Tensor tensor(std::string name, int64_t rows, int64_t cols, int layer = -1,
              std::string type = {}, std::string role = {}) {
    Tensor result;
    result.name = std::move(name); result.rows = rows; result.cols = cols;
    result.shape = {rows, cols}; result.layer_index = layer;
    result.layer_type = std::move(type); result.role = std::move(role);
    return result;
}

Scene model_stack() {
    std::vector<Tensor> tensors;
    Tensor embedding = tensor("language_model.model.embed_tokens.weight", 248320, 1024, -1, {}, "embedding");
    embedding.tied = true;
    tensors.push_back(embedding);
    for (int layer = 0; layer < 24; ++layer) {
        const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
        const std::string type = layer % 4 == 3 ? "full_attention" : "linear_attention";
        tensors.push_back(tensor(prefix + "mlp.down_proj.weight", 1024, 3584, layer, type));
        tensors.push_back(tensor(prefix + "mlp.gate_proj.weight", 3584, 1024, layer, type));
        tensors.push_back(tensor(prefix + "mlp.up_proj.weight", 3584, 1024, layer, type));
    }
    Tensor norm = tensor("language_model.model.norm.weight", 1024, 1, -1, {}, "final_norm");
    norm.shape = {1024};
    tensors.push_back(norm);
    return layout(tensors);
}

Camera front_camera() {
    Camera result;
    result.azimuth = pi / 2; result.elevation = 0;
    result.distance = 5; result.radius = 5;
    return result;
}

void check_labels(const std::vector<LayerLabel>& labels, const Scene& scene,
                  const Camera& camera, Rect viewport, Rect column, double line_height, double gap) {
    double previous_bottom = column.y - gap;
    for (const auto& label : labels) {
        REQUIRE(label.group < scene.groups.size());
        CHECK(label.bounds.x == column.x);
        CHECK(label.bounds.width == column.width);
        CHECK(label.bounds.height == line_height);
        CHECK(label.bounds.y >= column.y);
        CHECK(label.bounds.y + label.bounds.height <= column.y + column.height);
        CHECK(previous_bottom + gap <= label.bounds.y + 1e-9);
        previous_bottom = label.bounds.y + label.bounds.height;
        const Group& group = scene.groups[label.group];
        const auto anchor = project({group.bounds.x + group.bounds.width,
                                     group.bounds.y + group.bounds.height * 0.5, group.z},
                                    camera, viewport.width / viewport.height, viewport);
        CHECK(anchor.visible);
        CHECK(label.anchor_x == doctest::Approx(anchor.x));
        CHECK(label.anchor_y == doctest::Approx(anchor.y));
        CHECK(label.bounds.y + line_height * 0.5 == doctest::Approx(anchor.y));
    }
}
} // namespace

TEST_CASE("model3d viewport reserves one label column without changing its camera rectangle") {
    const auto wide = viewport_layout({20, 30, 1000, 700}, 220);
    CHECK(wide.scene.x == 20); CHECK(wide.scene.y == 30);
    CHECK(wide.scene.width == 768); CHECK(wide.scene.height == 700);
    CHECK(wide.labels.x == 800); CHECK(wide.labels.y == 30);
    CHECK(wide.labels.width == 220); CHECK(wide.labels.height == 700);
    CHECK(wide.labels.x + wide.labels.width == 1020);

    const auto narrow = viewport_layout({10, 15, 100, 80}, 1000);
    CHECK(narrow.labels.width == 35);
    CHECK(narrow.scene.width >= 60);
    CHECK(narrow.scene.x + narrow.scene.width < narrow.labels.x);
    CHECK(narrow.labels.x + narrow.labels.width == 110);

    const auto no_labels = viewport_layout({2, 3, 100, 80}, 0);
    CHECK(no_labels.scene.width == 100);
    CHECK(no_labels.labels.width == 0);
    const auto invalid = viewport_layout({0, 0, -10, 0}, std::numeric_limits<double>::infinity());
    CHECK(invalid.scene.width == 0); CHECK(invalid.labels.width == 0);
}

TEST_CASE("model3d default stack keeps full-attention markers and never overlaps label rows") {
    const Scene scene = model_stack();
    REQUIRE(scene.groups.size() == 26);
    for (Rect available : {Rect{0, 0, 800, 500}, Rect{25, 40, 1600, 900}, Rect{0, 0, 420, 300}}) {
        const auto viewports = viewport_layout(available, 210);
        const Camera camera = fit(scene, viewports.scene.width / viewports.scene.height);
        const auto labels = label_layout(scene, camera, viewports.scene, viewports.labels, 14);
        REQUIRE_FALSE(labels.empty());
        CHECK(labels.size() <= scene.groups.size());
        check_labels(labels, scene, camera, viewports.scene, viewports.labels, 14, 3);
        int full = 0;
        for (const auto& label : labels)
            if (scene.groups[label.group].layer_type == "full_attention") ++full;
        CHECK(full == 6);
    }
}

TEST_CASE("model3d label thinning prefers full attention and keeps exact projected alignment") {
    Scene scene;
    scene.groups = {{"Layer 00", "linear_attention", 0, 0, {-1, -1, 2, 2}},
                    {"Layer 03", "full_attention", 3, 0, {-1, -1, 2, 2}},
                    {"Final norm", {}, -1, 0, {-1, -1, 2, 2}}};
    const Rect viewport{0, 0, 800, 600}, column{820, 0, 200, 600};
    const Camera camera = front_camera();
    const auto labels = label_layout(scene, camera, viewport, column, 18, 4);
    REQUIRE(labels.size() == 1);
    CHECK(labels.front().group == 1);
    check_labels(labels, scene, camera, viewport, column, 18, 4);

    scene.groups.erase(scene.groups.begin() + 1);
    const auto special = label_layout(scene, camera, viewport, column, 18, 4);
    REQUIRE(special.size() == 1);
    CHECK(special.front().group == 1); // Final norm wins over a linear layer.

    scene.groups.front().bounds.y = 100;
    scene.groups.back().bounds.y = -100;
    CHECK(label_layout(scene, camera, viewport, column, 18).empty());
}

TEST_CASE("model3d label column remains collision-free while orbiting panning and resizing") {
    const Scene scene = model_stack();
    for (double width : {140.0, 400.0, 1100.0}) for (double height : {45.0, 160.0, 700.0}) {
        const auto viewports = viewport_layout({31, 47, width, height}, 205);
        Camera camera = fit(scene, viewports.scene.width / viewports.scene.height);
        for (int step = 0; step < 5; ++step) {
            camera = orbit(camera, 57, -19);
            camera = pan(camera, step * 3, step * 2, height);
            const auto labels = label_layout(scene, camera, viewports.scene, viewports.labels, 17, 4);
            check_labels(labels, scene, camera, viewports.scene, viewports.labels, 17, 4);
            const auto repeated = label_layout(scene, camera, viewports.scene, viewports.labels, 17, 4);
            REQUIRE(repeated.size() == labels.size());
            for (size_t i = 0; i < labels.size(); ++i) {
                CHECK(repeated[i].group == labels[i].group);
                CHECK(repeated[i].bounds.y == labels[i].bounds.y);
            }
        }
    }
}

TEST_CASE("model3d label layout rejects empty or nonfinite extents") {
    const Scene scene = model_stack();
    const Camera camera = front_camera();
    const Rect viewport{0, 0, 600, 400}, column{620, 0, 180, 400};
    CHECK(label_layout({}, camera, viewport, column, 14).empty());
    CHECK(label_layout(scene, camera, {}, column, 14).empty());
    CHECK(label_layout(scene, camera, viewport, {}, 14).empty());
    CHECK(label_layout(scene, camera, viewport, column, 0).empty());
    CHECK(label_layout(scene, camera, viewport, column, 401).empty());
    CHECK(label_layout(scene, camera, viewport, column, std::numeric_limits<double>::quiet_NaN()).empty());
}

TEST_CASE("model3d label column moves beside the visible stack and follows camera changes") {
    Scene scene;
    scene.groups = {{"Layer 00", "linear_attention", 0, 0, {-1, -1, 2, 2}}};
    const Rect viewport{40, 60, 600, 400}, reserved{652, 60, 200, 400};
    const Camera camera = front_camera();
    const Rect column = column_beside_stack(scene, camera, viewport, reserved);
    const auto right = project({1, 0, 0}, camera, viewport.width / viewport.height, viewport);
    REQUIRE(right.visible);
    CHECK(column.x == doctest::Approx(right.x + 12));
    CHECK(column.x < reserved.x);
    CHECK(column.x >= viewport.x);
    CHECK(column.y == reserved.y); CHECK(column.width == reserved.width); CHECK(column.height == reserved.height);
    CHECK(column.x + column.width <= reserved.x + reserved.width);

    const Camera panned = pan(camera, 400, 0, viewport.height);
    const Rect after_pan = column_beside_stack(scene, panned, viewport, reserved);
    CHECK(after_pan.x == reserved.x);
    CHECK(after_pan.x >= viewport.x + viewport.width + 12);
    const Rect after_return = column_beside_stack(scene, pan(panned, -400, 0, viewport.height), viewport, reserved);
    CHECK(after_return.x == doctest::Approx(column.x));
    const Rect after_zoom = column_beside_stack(scene, zoom(camera, 2), viewport, reserved);
    CHECK(after_zoom.x > column.x);
    CHECK(after_zoom.x <= reserved.x);
}

TEST_CASE("model3d fitted label column clears every group frame without changing projection") {
    const Scene scene = model_stack();
    const auto viewports = viewport_layout({20, 40, 1600, 900}, 210);
    const double aspect = viewports.scene.width / viewports.scene.height;
    const Camera camera = fit(scene, aspect);
    const Rect column = column_beside_stack(scene, camera, viewports.scene, viewports.labels);
    CHECK(column.x < viewports.labels.x - 30);
    for (const Group& group : scene.groups) {
        for (double x : {group.bounds.x, group.bounds.x + group.bounds.width})
            for (double y : {group.bounds.y, group.bounds.y + group.bounds.height}) {
                const auto corner = project({x, y, group.z}, camera, aspect, viewports.scene);
                REQUIRE(corner.visible);
                CHECK(column.x >= corner.x + 12 - 1e-9);
            }
    }
    const auto labels = label_layout(scene, camera, viewports.scene, column, 14);
    REQUIRE_FALSE(labels.empty());
    check_labels(labels, scene, camera, viewports.scene, column, 14, 3);
}

TEST_CASE("model3d label column keeps its reserved position for clipped or invalid frames") {
    Scene scene;
    scene.groups = {{"Layer 00", "linear_attention", 0, 10, {-1, -1, 2, 2}}};
    const Rect viewport{40, 60, 600, 400}, reserved{652, 60, 200, 400};
    const Camera camera = front_camera();
    CHECK(column_beside_stack(scene, camera, viewport, reserved).x == reserved.x); // Behind camera.
    scene.groups.front().z = 5;
    CHECK(column_beside_stack(scene, camera, viewport, reserved).x == reserved.x); // Camera plane.
    scene.groups.front().z = -50;
    CHECK(column_beside_stack(scene, camera, viewport, reserved).x == reserved.x); // Beyond far plane.
    scene.groups.front().z = 0;
    scene.groups.front().bounds = {-100, -1, 200, 2};
    Camera oblique = camera;
    oblique.azimuth = pi / 4;
    CHECK(column_beside_stack(scene, oblique, viewport, reserved).x == reserved.x); // Edge crosses camera plane.
    scene.groups.front().bounds.width = std::numeric_limits<double>::quiet_NaN();
    CHECK(column_beside_stack(scene, camera, viewport, reserved).x == reserved.x);
    CHECK(column_beside_stack({}, camera, viewport, reserved).x == reserved.x);
    CHECK(column_beside_stack(scene, camera, {}, reserved).x == reserved.x);
}
