#pragma once
#include <array>
#include <cstdint>

namespace caliper_live::tensor_geometry {
// Tensor coordinates keep the wire convention: vectors are N rows x 1 column.
// Display coordinates rotate vectors into a horizontal strip.
struct Rect { double x = 0, y = 0, width = 0, height = 0; };
struct Region {
    int64_t row = 0, col = 0, rows = 1, cols = 1;
    bool operator==(const Region& b) const {
        return row == b.row && col == b.col && rows == b.rows && cols == b.cols;
    }
    bool operator!=(const Region& b) const { return !(*this == b); }
};
struct Cell { int64_t row = -1, col = -1; bool valid = false; };
struct Target { int width = 1, height = 1; };
struct ViewLayout { Rect overview, detail, plot, strip; bool elongated = false; };
struct Windows { Rect inference, tree, model3d, tensor; };
constexpr int max_target_side = 2048;
constexpr int max_samples = 131072;
constexpr double strip_height = 24;

// All drawing/hit-testing code consumes these same pure geometry decisions.
Rect fit(double rows, double cols, Rect bounds);
Region clamp(Region region, int64_t rows, int64_t cols);
Region initial_region(int64_t rows, int64_t cols, Rect bounds, bool vector);
// factor below one zooms in; the anchor cell stays at the same relative point.
Region zoom(Region region, int64_t rows, int64_t cols, double factor, Cell anchor);
// Pass total drag distance and the region at drag start so subcell motion accumulates.
Region pan(Region region, int64_t rows, int64_t cols, double dx, double dy,
           Rect picture, bool vector);
Cell hit_test(Region region, Rect picture, double x, double y, bool vector);
Region overview_region(Region current, int64_t rows, int64_t cols,
                       Rect overview, double x, double y);
Rect selection_rect(Region region, int64_t rows, int64_t cols, Rect overview);
// Target uses display orientation. Swap width/height for an Nx1 vector wire request.
Target sampling_target(Region region, Rect picture, double framebuffer_scale, bool vector);
ViewLayout view_layout(int64_t rows, int64_t cols, bool vector, Rect bounds);
Windows window_layout(Rect work);
} // namespace caliper_live::tensor_geometry
