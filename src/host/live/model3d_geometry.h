#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_live::model3d {

// Pure geometry: no renderer, ImGui, MLX or process access. One world-space
// unit is 1024 elements for every panel; the constant never varies by tensor.
constexpr double element_unit = 1.0 / 1024.0;
struct Vec3 { double x = 0, y = 0, z = 0; };
struct Rect { double x = 0, y = 0, width = 0, height = 0; };
struct Tensor {
    std::string name, shape_label, group, layer_type, role;
    std::vector<int64_t> shape;
    int64_t rows = 0, cols = 0;
    int layer_index = -1;
    bool tied = false, edited = false;
    uint64_t revision = 0;
};
struct Panel {
    size_t tensor = 0;
    std::string label;
    Vec3 center;
    double width = 0, height = 0;
    int group = 0;
    bool output_alias = false;
    int row_blocks = 1; // equal row blocks laid out from left to right
};
struct Group {
    std::string label, layer_type;
    int layer_index = -1;
    double z = 0;
    Rect bounds;
};
struct Scene {
    std::vector<Panel> panels;
    std::vector<Group> groups;
    Vec3 minimum, maximum;
};
struct Camera {
    Vec3 target;
    double distance = 10, azimuth = 1.5, elevation = 1.15;
    double radius = 1;
};
struct Matrices { std::array<float, 16> view{}, projection{}; };
struct Projection { double x = 0, y = 0, depth = 0; bool visible = false; };
struct Hit { int panel = -1; double distance = 0, u = 0, v = 0; };

// Rank-one arrays are horizontal strips (N by one element), regardless of the
// row/column convention used by the request protocol. Tall higher-rank arrays
// fold into equal row blocks without changing their total element area.
int row_blocks(const Tensor& tensor);
// Wire sampling rows/columns. Folded samples repack into an atlas no larger
// than 128 x 128; side may be raised to the fold count so each block is sampled.
// Requests stay within the host's 2048-per-side limit.
std::array<int, 2> thumbnail_shape(const Tensor& tensor, int side = 128);
std::array<double, 2> panel_size(const Tensor& tensor);
Scene layout(const std::vector<Tensor>& tensors, const std::string& output_label = "Output (tied)");
Camera fit(const Scene& scene, double aspect, double azimuth = 1.5, double elevation = 1.15);
Camera focus(const Scene& scene, size_t panel, double aspect, const Camera& previous);
Matrices matrices(const Camera& camera, double aspect);
Projection project(Vec3 point, const Camera& camera, double aspect, Rect viewport);
Hit hit_test(const Scene& scene, const Camera& camera, Rect viewport, double x, double y);
Camera orbit(Camera camera, double dx, double dy);
Camera pan(Camera camera, double dx, double dy, double viewport_height);
Camera zoom(Camera camera, double wheel);
// Top-left, top-right, bottom-right, bottom-left; UV origin is top-left.
std::array<Vec3, 4> corners(const Panel& panel);

} // namespace caliper_live::model3d
