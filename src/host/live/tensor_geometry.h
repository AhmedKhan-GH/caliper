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
struct AxisLabels { Rect first_row, last_row, columns; };
struct Windows { Rect inference, tree, model3d, tensor; };
// A window onto a tensor in tensor coordinates, in fractional cells.
struct Window { double row = 0, col = 0, rows = 1, cols = 1; };
// What the Tensor pane looks at: the display cell at the centre of its bounds
// and the side of a cell in points. Display space runs a vector's cells along
// x, like the strip it is drawn as; a matrix's rows run down.
struct Camera { double x = 0, y = 0, cell = 0; };
struct Shot { Window window; Rect picture; };
constexpr double max_cell = 160;
constexpr int max_target_side = 2048;
constexpr int max_samples = 131072;
constexpr double strip_height = 24;

// All drawing/hit-testing code consumes these same pure geometry decisions.
Rect fit(double rows, double cols, Rect bounds);
// Text is measured by the caller; labels follow the fitted picture, including
// the empty space introduced by fitting a matrix into its allocated rectangle.
AxisLabels axis_labels(Rect picture, double first_row_width, double last_row_width,
                       double column_width, double line_height, double gap = 4);
Region clamp(Region region, int64_t rows, int64_t cols);
Cell hit_test(Region region, Rect picture, double x, double y, bool vector);

// The cell size at which the whole tensor fits the bounds: the furthest out.
double fit_cell(int64_t rows, int64_t cols, bool vector, Rect bounds);
// Ordinary matrices start whole; elongated ones fill the bounds' shorter side
// from their first cell; vectors start with strip-high cells.
Camera initial_camera(int64_t rows, int64_t cols, bool vector, Rect bounds);
// Cells from fit_cell to max_cell, and the window kept inside the tensor.
Camera clamp(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds);
// The window a camera sees and the picture it fills. Square cells; the picture
// takes the whole bounds along every side where the tensor reaches past them.
Shot shoot(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds);
// factor above one zooms in; the point under (x, y) stays under it.
Camera zoom(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds,
            double factor, double x, double y);
// Pass the camera at drag start and the total drag distance in points.
Camera pan(Camera start, int64_t rows, int64_t cols, bool vector, Rect bounds, double dx, double dy);
// The camera that shows a window as large as fits, centred on it.
Camera frame(Window window, int64_t rows, int64_t cols, bool vector, Rect bounds);
// The camera moved so that a cell is at the centre, at the same zoom.
Camera centre_on(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds, Cell cell);
// The whole cells a window touches: what to sample and what to label.
Region cover(Window window, int64_t rows, int64_t cols);
Cell hit_test(Window window, Rect picture, double x, double y, bool vector);
Rect selection_rect(Window window, int64_t rows, int64_t cols, Rect overview);
// Target uses display orientation. Swap width/height for an Nx1 vector wire request.
Target sampling_target(Region region, Rect picture, double framebuffer_scale, bool vector);
ViewLayout view_layout(int64_t rows, int64_t cols, bool vector, Rect bounds);
Windows window_layout(Rect work);
} // namespace caliper_live::tensor_geometry
