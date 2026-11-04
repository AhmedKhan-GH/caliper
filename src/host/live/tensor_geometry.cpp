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

AxisLabels axis_labels(Rect picture, double first_row_width, double last_row_width,
                       double column_width, double line_height, double gap) {
    picture = valid_rect(picture);
    const auto extent = [](double value) {
        return std::isfinite(value) && value > 0 ? value : 0;
    };
    first_row_width = extent(first_row_width); last_row_width = extent(last_row_width);
    column_width = extent(column_width); line_height = extent(line_height); gap = extent(gap);
    if (picture.width <= 0 || picture.height <= 0 || line_height <= 0) return {};
    AxisLabels labels;
    labels.first_row = {picture.x - gap - first_row_width, picture.y, first_row_width, line_height};
    labels.last_row = {picture.x - gap - last_row_width,
                       picture.y + picture.height - line_height, last_row_width, line_height};
    // Tiny one-row matrices cannot accommodate both endpoint labels.
    if (picture.height < line_height * 2) labels.last_row.height = 0;
    labels.columns = {picture.x, picture.y + picture.height + gap, column_width, line_height};
    return labels;
}

Region clamp(Region region, int64_t rows, int64_t cols) {
    rows = dimension(rows); cols = dimension(cols);
    region.rows = std::clamp<int64_t>(region.rows, 1, rows);
    region.cols = std::clamp<int64_t>(region.cols, 1, cols);
    region.row = std::clamp<int64_t>(region.row, 0, rows - region.rows);
    region.col = std::clamp<int64_t>(region.col, 0, cols - region.cols);
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

namespace {
// Display extents: a vector is one row of N cells.
double display_width(int64_t rows, int64_t cols, bool vector) { return static_cast<double>(dimension(vector ? rows : cols)); }
double display_height(int64_t rows, int64_t cols, bool vector) { return vector ? 1.0 : static_cast<double>(dimension(rows)); }
bool usable(Rect bounds) { return bounds.width > 0 && bounds.height > 0; }
} // namespace

double fit_cell(int64_t rows, int64_t cols, bool vector, Rect bounds) {
    bounds = valid_rect(bounds);
    if (!usable(bounds)) return 1;
    return std::min(bounds.width / display_width(rows, cols, vector),
                    bounds.height / display_height(rows, cols, vector));
}

double min_cell(int64_t rows, int64_t cols, bool vector, Rect bounds) {
    bounds = valid_rect(bounds);
    const double fit = fit_cell(rows, cols, vector, bounds);
    if (vector || !usable(bounds) || static_cast<long double>(dimension(rows)) <= 8.0L * dimension(cols)) return fit;
    return std::max(fit, std::min(bounds.width / dimension(cols), strip_height));
}

Camera initial_camera(int64_t rows, int64_t cols, bool vector, Rect bounds) {
    bounds = valid_rect(bounds);
    const double width = display_width(rows, cols, vector), height = display_height(rows, cols, vector);
    Camera camera{0, 0, min_cell(rows, cols, vector, bounds)};
    if (usable(bounds)) {
        if (vector) camera.cell = std::max(camera.cell, std::min(strip_height, bounds.height));
        else if (width > height * 8) camera.cell = bounds.height / height;
    }
    // The first cell at the top left; clamp moves the centre in from there.
    return clamp(camera, rows, cols, vector, bounds);
}

Camera clamp(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds) {
    bounds = valid_rect(bounds);
    const double width = display_width(rows, cols, vector), height = display_height(rows, cols, vector);
    const double low = min_cell(rows, cols, vector, bounds);
    if (!std::isfinite(camera.cell) || camera.cell <= 0) camera.cell = low;
    camera.cell = std::clamp(camera.cell, low, std::max(low, max_cell));
    const double seen_x = usable(bounds) ? std::min(width, bounds.width / camera.cell) : width;
    const double seen_y = usable(bounds) ? std::min(height, bounds.height / camera.cell) : height;
    if (!std::isfinite(camera.x)) camera.x = 0;
    if (!std::isfinite(camera.y)) camera.y = 0;
    camera.x = std::clamp(camera.x, seen_x * 0.5, width - seen_x * 0.5);
    camera.y = std::clamp(camera.y, seen_y * 0.5, height - seen_y * 0.5);
    return camera;
}

Shot shoot(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds) {
    bounds = valid_rect(bounds);
    camera = clamp(camera, rows, cols, vector, bounds);
    const double width = display_width(rows, cols, vector), height = display_height(rows, cols, vector);
    const double seen_x = usable(bounds) ? std::min(width, bounds.width / camera.cell) : width;
    const double seen_y = usable(bounds) ? std::min(height, bounds.height / camera.cell) : height;
    const double left = camera.x - seen_x * 0.5, top = camera.y - seen_y * 0.5;
    Shot shot;
    shot.picture = {bounds.x + (bounds.width - seen_x * camera.cell) * 0.5,
                    bounds.y + (bounds.height - seen_y * camera.cell) * 0.5,
                    seen_x * camera.cell, seen_y * camera.cell};
    if (!usable(bounds)) shot.picture = {bounds.x, bounds.y, 0, 0};
    shot.window = vector ? Window{left, 0, seen_x, 1} : Window{top, left, seen_y, seen_x};
    return shot;
}

Camera zoom(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds,
            double factor, double x, double y) {
    bounds = valid_rect(bounds);
    camera = clamp(camera, rows, cols, vector, bounds);
    if (!std::isfinite(factor) || factor <= 0 || !usable(bounds)) return camera;
    const Shot before = shoot(camera, rows, cols, vector, bounds);
    const Rect p = before.picture;
    if (!std::isfinite(x)) x = p.x + p.width * 0.5;
    if (!std::isfinite(y)) y = p.y + p.height * 0.5;
    x = std::clamp(x, p.x, p.x + p.width); y = std::clamp(y, p.y, p.y + p.height);
    // The display cell under the pointer, before and after.
    const double left = camera.x - p.width / camera.cell * 0.5, top = camera.y - p.height / camera.cell * 0.5;
    const double under_x = left + (x - p.x) / camera.cell, under_y = top + (y - p.y) / camera.cell;
    Camera next = camera;
    next.cell = clamp({camera.x, camera.y, camera.cell * factor}, rows, cols, vector, bounds).cell;
    const Shot after = shoot({0, 0, next.cell}, rows, cols, vector, bounds);
    next.x = under_x - (x - after.picture.x) / next.cell + after.picture.width / next.cell * 0.5;
    next.y = under_y - (y - after.picture.y) / next.cell + after.picture.height / next.cell * 0.5;
    return clamp(next, rows, cols, vector, bounds);
}

Camera pan(Camera start, int64_t rows, int64_t cols, bool vector, Rect bounds, double dx, double dy) {
    start = clamp(start, rows, cols, vector, bounds);
    if (std::isfinite(dx)) start.x -= dx / start.cell;
    if (std::isfinite(dy)) start.y -= dy / start.cell;
    return clamp(start, rows, cols, vector, bounds);
}

Camera frame(Window window, int64_t rows, int64_t cols, bool vector, Rect bounds) {
    bounds = valid_rect(bounds);
    const double left = vector ? window.row : window.col, top = vector ? 0 : window.row;
    const double width = std::max(1e-9, vector ? window.rows : window.cols);
    const double height = std::max(1e-9, vector ? 1 : window.rows);
    Camera camera{left + width * 0.5, top + height * 0.5, 0};
    if (usable(bounds)) camera.cell = std::min(bounds.width / width, bounds.height / height);
    return clamp(camera, rows, cols, vector, bounds);
}

Camera centre_on(Camera camera, int64_t rows, int64_t cols, bool vector, Rect bounds, Cell cell) {
    if (!cell.valid) return clamp(camera, rows, cols, vector, bounds);
    camera.x = (vector ? cell.row : cell.col) + 0.5;
    camera.y = vector ? 0.5 : cell.row + 0.5;
    return clamp(camera, rows, cols, vector, bounds);
}

Region cover(Window window, int64_t rows, int64_t cols) {
    // A hair of tolerance, so a window that ends on a cell edge does not
    // touch the next cell through rounding.
    constexpr long double slack = 1e-6L;
    const int64_t row0 = rounded(std::floor(window.row + slack), 0, dimension(rows) - 1);
    const int64_t col0 = rounded(std::floor(window.col + slack), 0, dimension(cols) - 1);
    const int64_t row1 = rounded(std::ceil(static_cast<long double>(window.row) + window.rows - slack), row0 + 1, dimension(rows));
    const int64_t col1 = rounded(std::ceil(static_cast<long double>(window.col) + window.cols - slack), col0 + 1, dimension(cols));
    return {row0, col0, row1 - row0, col1 - col0};
}

Cell hit_test(Window window, Rect picture, double x, double y, bool vector) {
    picture = valid_rect(picture);
    if (!inside(picture, x, y) || !(window.rows > 0) || !(window.cols > 0)) return {};
    const long double fx = (x - picture.x) / picture.width, fy = (y - picture.y) / picture.height;
    const long double row = window.row + (vector ? fx : fy) * window.rows;
    const long double col = window.col + (vector ? fy : fx) * window.cols;
    if (!std::isfinite(row) || !std::isfinite(col) || row < 0 || col < 0) return {};
    return {static_cast<int64_t>(std::floor(row)), static_cast<int64_t>(std::floor(col)), true};
}

Fold fold(int64_t rows, int64_t cols, int blocks, Rect bounds, double gap) {
    rows = dimension(rows); cols = dimension(cols);
    bounds = valid_rect(bounds);
    Fold result;
    result.blocks = static_cast<int>(std::clamp<int64_t>(blocks, 1, rows));
    result.block_rows = (rows + result.blocks - 1) / result.blocks;
    result.gap = result.blocks > 1 && std::isfinite(gap) ? std::max(0.0, gap) : 0;
    const double gaps = result.gap * (result.blocks - 1);
    if (!usable(bounds) || bounds.width <= gaps) { result.area = {bounds.x, bounds.y, 0, 0}; return result; }
    result.scale = std::min((bounds.width - gaps) / (static_cast<double>(cols) * result.blocks),
                            bounds.height / static_cast<double>(result.block_rows));
    result.area = {bounds.x, bounds.y, static_cast<double>(cols) * result.blocks * result.scale + gaps,
                   static_cast<double>(result.block_rows) * result.scale};
    return result;
}

Rect fold_block(const Fold& f, int64_t cols, int block) {
    const double width = static_cast<double>(dimension(cols)) * f.scale;
    return {f.area.x + block * (width + f.gap), f.area.y, width, f.area.height};
}

Cell fold_hit(const Fold& f, int64_t rows, int64_t cols, double x, double y) {
    rows = dimension(rows); cols = dimension(cols);
    if (!(f.scale > 0) || !std::isfinite(x) || !std::isfinite(y)) return {};
    const double pitch = cols * f.scale + f.gap;
    const int block = static_cast<int>(std::clamp(std::floor((x - f.area.x + f.gap * 0.5) / pitch), 0.0, f.blocks - 1.0));
    const Rect b = fold_block(f, cols, block);
    const int64_t col = rounded(std::floor(std::clamp((x - b.x) / f.scale, 0.0, cols - 0.5)), 0, cols - 1);
    const int64_t row = rounded(block * f.block_rows + std::floor(std::clamp((y - b.y) / f.scale, 0.0, f.block_rows - 0.5)), 0, rows - 1);
    return {row, col, true};
}

std::vector<Rect> fold_selection(const Fold& f, Window window, int64_t rows, int64_t cols) {
    std::vector<Rect> result;
    rows = dimension(rows); cols = dimension(cols);
    if (!(f.scale > 0)) return result;
    const double first = std::clamp(window.row, 0.0, static_cast<double>(rows));
    const double last = std::clamp(window.row + window.rows, first, static_cast<double>(rows));
    for (int block = 0; block < f.blocks; ++block) {
        const double top = static_cast<double>(block) * f.block_rows, bottom = top + f.block_rows;
        const double r0 = std::max(first, top), r1 = std::min(last, bottom);
        if (r1 <= r0) continue;
        const Rect b = fold_block(f, cols, block);
        result.push_back({b.x + window.col * f.scale, b.y + (r0 - top) * f.scale, window.cols * f.scale, (r1 - r0) * f.scale});
    }
    return result;
}

Target fold_target(const Fold& f, int64_t rows, int64_t cols, double framebuffer_scale) {
    rows = dimension(rows); cols = dimension(cols);
    const Rect block = fold_block(f, cols, 0);
    const int per_block = std::max(1, std::min(sample_extent(block.height, framebuffer_scale, f.block_rows),
                                               max_target_side / std::max(1, f.blocks)));
    Target target{sample_extent(block.width, framebuffer_scale, cols), per_block * f.blocks};
    if (target.height > rows) target.height = static_cast<int>(rows);
    if (static_cast<int64_t>(target.width) * target.height > max_samples)
        target.width = std::max(1, max_samples / std::max(1, target.height));
    return target;
}

Rect scroll_thumb(Rect track, Window window, int64_t rows, double min_thumb) {
    track = valid_rect(track);
    const double total = static_cast<double>(dimension(rows));
    const double height = std::min(track.height, std::max(min_thumb, track.height * std::clamp(window.rows / total, 0.0, 1.0)));
    const double travel = total - std::clamp(window.rows, 0.0, total);
    const double at = travel > 0 ? std::clamp(window.row / travel, 0.0, 1.0) : 0;
    return {track.x, track.y + (track.height - height) * at, track.width, height};
}

double scroll_centre(Rect track, Window window, int64_t rows, double y, double min_thumb) {
    track = valid_rect(track);
    const double total = static_cast<double>(dimension(rows));
    const Rect thumb = scroll_thumb(track, window, rows, min_thumb);
    const double free = track.height - thumb.height;
    const double at = free > 0 && std::isfinite(y) ? std::clamp((y - thumb.height * 0.5 - track.y) / free, 0.0, 1.0) : 0;
    const double seen = std::clamp(window.rows, 0.0, total);
    return at * (total - seen) + seen * 0.5;
}

Rect selection_rect(Window window, int64_t rows, int64_t cols, Rect overview) {
    overview = valid_rect(overview);
    const double x_scale = overview.width / dimension(cols);
    const double y_scale = overview.height / dimension(rows);
    return {overview.x + window.col * x_scale, overview.y + window.row * y_scale,
            window.cols * x_scale, window.rows * y_scale};
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
    result.tall = ratio > 8;
    if (result.tall) {
        const double track = std::min(scroll_width, bounds.width * 0.1);
        const double gap = std::min(6.0, bounds.width * 0.02);
        result.scroll = {bounds.x + bounds.width - track, bounds.y, track, bounds.height};
        result.detail = {bounds.x, bounds.y, std::max(0.0, bounds.width - track - gap), bounds.height};
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
