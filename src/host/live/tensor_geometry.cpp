#include "live/tensor_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace caliper_live::tensor_geometry {
namespace {

Rect valid_rect(Rect r) {
    if (!std::isfinite(r.x)) r.x = 0;
    if (!std::isfinite(r.y)) r.y = 0;
    if (!std::isfinite(r.width) || r.width < 0) r.width = 0;
    if (!std::isfinite(r.height) || r.height < 0) r.height = 0;
    return r;
}

int64_t dimension(int64_t value) { return std::max<int64_t>(1, value); }

int64_t rounded(long double value, int64_t low, int64_t high) {
    if (!std::isfinite(value)) return value > 0 ? high : low;
    value = std::round(value);
    if (value <= static_cast<long double>(low)) return low;
    if (value >= static_cast<long double>(high)) return high;
    return static_cast<int64_t>(value);
}

bool inside(Rect r, double x, double y) {
    return std::isfinite(x) && std::isfinite(y) && r.width > 0 && r.height > 0 &&
           x >= r.x && y >= r.y && x - r.x < r.width && y - r.y < r.height;
}

int sample_extent(double pixels, double scale, int64_t cells) {
    if (!std::isfinite(scale) || scale <= 0) scale = 1;
    if (!std::isfinite(pixels) || pixels <= 0) pixels = 1;
    const double extent = std::ceil(pixels * scale);
    if (!std::isfinite(extent) || extent >= max_target_side)
        return static_cast<int>(std::min<int64_t>(dimension(cells), max_target_side));
    return static_cast<int>(std::min<int64_t>(dimension(cells), std::max(1, static_cast<int>(extent))));
}

} // namespace

Rect fit(double rows, double cols, Rect bounds) {
    bounds = valid_rect(bounds);
    if (!std::isfinite(rows) || !std::isfinite(cols) || rows <= 0 || cols <= 0 ||
        bounds.width <= 0 || bounds.height <= 0)
        return {bounds.x, bounds.y, 0, 0};
    const double scale = std::min(bounds.width / cols, bounds.height / rows);
    const double width = cols * scale, height = rows * scale;
    return {bounds.x + (bounds.width - width) * 0.5,
            bounds.y + (bounds.height - height) * 0.5, width, height};
}

Region clamp(Region region, int64_t rows, int64_t cols) {
    rows = dimension(rows); cols = dimension(cols);
    region.rows = std::clamp<int64_t>(region.rows, 1, rows);
    region.cols = std::clamp<int64_t>(region.cols, 1, cols);
    region.row = std::clamp<int64_t>(region.row, 0, rows - region.rows);
    region.col = std::clamp<int64_t>(region.col, 0, cols - region.cols);
    return region;
}

Region initial_region(int64_t rows, int64_t cols, Rect bounds, bool vector) {
    rows = dimension(rows); cols = dimension(cols);
    bounds = valid_rect(bounds);
    Region result{0, 0, rows, cols};
    if (vector) {
        // Start with a local strip whose cells can occupy the modest strip
        // height. The value plot supplies context over the whole vector.
        const double cell_size = std::min(strip_height, bounds.height);
        result.rows = cell_size > 0
            ? rounded(std::floor(bounds.width / cell_size), 1, rows) : 1;
        return result;
    }
    const long double ratio = static_cast<long double>(rows) / cols;
    const double viewport_ratio = bounds.width > 0 && bounds.height > 0
        ? bounds.height / bounds.width : 1;
    // Ordinary matrices start with their whole shape fitted. Extreme shapes
    // start at a readable local region; their overview still fits all cells.
    if (ratio > 8)
        result.rows = rounded(static_cast<long double>(cols) * viewport_ratio, 1, rows);
    else if (ratio < 0.125L)
        result.cols = rounded(static_cast<long double>(rows) / viewport_ratio, 1, cols);
    return result;
}

Region zoom(Region region, int64_t rows, int64_t cols, double factor, Cell anchor) {
    region = clamp(region, rows, cols);
    if (!std::isfinite(factor) || factor <= 0) return region;
    const long double ar = anchor.valid
        ? std::clamp<long double>(anchor.row + 0.5L, region.row, static_cast<long double>(region.row) + region.rows)
        : region.row + region.rows * 0.5L;
    const long double ac = anchor.valid
        ? std::clamp<long double>(anchor.col + 0.5L, region.col, static_cast<long double>(region.col) + region.cols)
        : region.col + region.cols * 0.5L;
    const long double fr = (ar - region.row) / region.rows;
    const long double fc = (ac - region.col) / region.cols;
    Region result = region;
    result.rows = rounded(region.rows * static_cast<long double>(factor), 1, dimension(rows));
    result.cols = rounded(region.cols * static_cast<long double>(factor), 1, dimension(cols));
    result.row = rounded(ar - fr * result.rows, 0, dimension(rows) - result.rows);
    result.col = rounded(ac - fc * result.cols, 0, dimension(cols) - result.cols);
    return result;
}

Region pan(Region region, int64_t rows, int64_t cols, double dx, double dy,
           Rect picture, bool vector) {
    region = clamp(region, rows, cols);
    picture = valid_rect(picture);
    if (picture.width <= 0 || picture.height <= 0) return region;
    if (!std::isfinite(dx)) dx = 0;
    if (!std::isfinite(dy)) dy = 0;
    const long double dr = vector ? -dx * static_cast<long double>(region.rows) / picture.width
                                  : -dy * static_cast<long double>(region.rows) / picture.height;
    const long double dc = vector ? -dy * static_cast<long double>(region.cols) / picture.height
                                  : -dx * static_cast<long double>(region.cols) / picture.width;
    region.row = rounded(region.row + dr, 0, dimension(rows) - region.rows);
    region.col = rounded(region.col + dc, 0, dimension(cols) - region.cols);
    return region;
}

Cell hit_test(Region region, Rect picture, double x, double y, bool vector) {
    picture = valid_rect(picture);
    if (region.row < 0 || region.col < 0 || region.rows <= 0 || region.cols <= 0 ||
        !inside(picture, x, y)) return {};
    const long double fx = (x - picture.x) / picture.width;
    const long double fy = (y - picture.y) / picture.height;
    const int64_t dr = rounded(std::floor((vector ? fx : fy) * region.rows), 0, region.rows - 1);
    const int64_t dc = rounded(std::floor((vector ? fy : fx) * region.cols), 0, region.cols - 1);
    if (region.row > std::numeric_limits<int64_t>::max() - dr ||
        region.col > std::numeric_limits<int64_t>::max() - dc) return {};
    return {region.row + dr, region.col + dc, true};
}

Region overview_region(Region current, int64_t rows, int64_t cols,
                       Rect overview, double x, double y) {
    current = clamp(current, rows, cols);
    const Cell picked = hit_test({0, 0, dimension(rows), dimension(cols)}, overview, x, y, false);
    if (!picked.valid) return current;
    current.row = rounded(picked.row + 0.5L - current.rows * 0.5L, 0, dimension(rows) - current.rows);
    current.col = rounded(picked.col + 0.5L - current.cols * 0.5L, 0, dimension(cols) - current.cols);
    return current;
}

Rect selection_rect(Region region, int64_t rows, int64_t cols, Rect overview) {
    region = clamp(region, rows, cols);
    overview = valid_rect(overview);
    const double x_scale = overview.width / dimension(cols);
    const double y_scale = overview.height / dimension(rows);
    return {overview.x + region.col * x_scale, overview.y + region.row * y_scale,
            region.cols * x_scale, region.rows * y_scale};
}

Target sampling_target(Region region, Rect picture, double framebuffer_scale, bool vector) {
    picture = valid_rect(picture);
    Target target{sample_extent(picture.width, framebuffer_scale, vector ? region.rows : region.cols),
                  sample_extent(picture.height, framebuffer_scale, vector ? region.cols : region.rows)};
    if (target.width * target.height > max_samples) {
        const double reduction = std::sqrt(static_cast<double>(max_samples) / (target.width * target.height));
        target.width = std::max(1, static_cast<int>(std::floor(target.width * reduction)));
        target.height = std::max(1, static_cast<int>(std::floor(target.height * reduction)));
    }
    return target;
}

ViewLayout view_layout(int64_t rows, int64_t cols, bool vector, Rect bounds) {
    rows = dimension(rows); cols = dimension(cols);
    bounds = valid_rect(bounds);
    ViewLayout result;
    result.detail = bounds;
    if (vector) {
        const double height = std::min(strip_height, bounds.height);
        const double gap = std::min(8.0, std::max(0.0, bounds.height - height));
        result.strip = {bounds.x, bounds.y + bounds.height - height, bounds.width, height};
        result.plot = {bounds.x, bounds.y, bounds.width, std::max(0.0, bounds.height - height - gap)};
        result.detail = result.strip;
        return result;
    }
    const long double ratio = static_cast<long double>(rows) / cols;
    result.elongated = ratio > 8 || ratio < 0.125L;
    if (!result.elongated) return result;
    if (ratio > 8) {
        const double width = std::min(72.0, bounds.width * 0.18);
        const double gap = std::min(8.0, std::max(0.0, bounds.width - width));
        result.overview = fit(static_cast<double>(rows), static_cast<double>(cols),
                              {bounds.x, bounds.y, width, bounds.height});
        result.detail = {bounds.x + width + gap, bounds.y,
                         std::max(0.0, bounds.width - width - gap), bounds.height};
    } else {
        const double height = std::min(56.0, bounds.height * 0.18);
        const double gap = std::min(8.0, std::max(0.0, bounds.height - height));
        result.overview = fit(static_cast<double>(rows), static_cast<double>(cols),
                              {bounds.x, bounds.y, bounds.width, height});
        result.detail = {bounds.x, bounds.y + height + gap, bounds.width,
                         std::max(0.0, bounds.height - height - gap)};
    }
    return result;
}

Windows window_layout(Rect work) {
    work = valid_rect(work);
    const double gap = std::min({8.0, work.width * 0.01, work.height * 0.01});
    const double left_width = std::max(0.0, work.width - gap) * 0.30;
    const double right_width = std::max(0.0, work.width - left_width - gap);
    const double top_height = std::max(0.0, work.height - gap) * 0.45;
    const double tree_width = std::max(0.0, right_width - gap) * 0.40;
    const double right_x = work.x + left_width + gap;
    const double lower_y = work.y + top_height + gap;
    return {{work.x, work.y, left_width, work.height},
            {right_x, work.y, tree_width, top_height},
            {right_x + tree_width + gap, work.y, std::max(0.0, right_width - tree_width - gap), top_height},
            {right_x, lower_y, right_width, std::max(0.0, work.height - top_height - gap)}};
}

} // namespace caliper_live::tensor_geometry
