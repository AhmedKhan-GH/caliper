#pragma once
// caliper_builder::programs::operator_kernels: the arithmetic of the built-in
// operators `surface` and `path` (design/instrument-interface.md, Operators),
// header-only and with no dependency beyond the standard library.
//
// The CPU references of caliper_builder/operators/operators.h run these
// functions, and so do the producer tests, which compare them with the result
// of the WGSL sources src/builder/operators/wgsl/*.wgsl on the device. The
// WGSL loops visit the cells in the same order and accumulate in f32, thus
// the two agree up to the rounding of the device.
//
// The layout of the uniform `params` of each operator program is fixed here
// too: one u32 field for each name, in the order of the WGSL struct.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace caliper_builder::programs::operator_kernels {

// The defaults and bounds of the operator parameters.
inline constexpr std::int64_t kMaxCellsDefault = 32768;
inline constexpr std::int64_t kMaxCellsMax = 262144;
inline constexpr std::int64_t kMaxPointsDefault = 65536;
inline constexpr std::int64_t kMaxPointsMax = 1048576;
// The workgroup size of both WGSL entry points.
inline constexpr std::uint32_t kWorkgroupSize = 64;

// The pools of a surface, as the WGSL's `pool` parameter numbers them.
inline constexpr std::uint32_t kPoolMean = 0, kPoolMax = 1, kPoolMin = 2;

inline std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return b <= 0 ? 0 : (a + b - 1) / b; }

// The smallest pool factor f for which ceil(rows / f) * ceil(cols / f) is at
// most max_cells; 1 when the grid fits. max_cells below 1 counts as 1.
inline std::int64_t pool_factor(std::int64_t rows, std::int64_t cols, std::int64_t max_cells) {
    if (rows <= 0 || cols <= 0) return 1;
    max_cells = std::max<std::int64_t>(1, max_cells);
    if (rows * cols <= max_cells) return 1;
    // f grows from a lower bound: ceil(r/f) * ceil(c/f) >= r * c / f^2.
    std::int64_t f = std::max<std::int64_t>(
        1, static_cast<std::int64_t>(std::floor(std::sqrt(double(rows) * double(cols) / double(max_cells)))));
    while (ceil_div(rows, f) * ceil_div(cols, f) > max_cells) ++f;
    return f;
}

// The stride of a path: the smallest s for which ceil(rows / s) is at most
// max_points; 1 when the rows fit.
inline std::int64_t path_stride(std::int64_t rows, std::int64_t max_points) {
    max_points = std::max<std::int64_t>(1, max_points);
    return rows <= max_points ? 1 : ceil_div(rows, max_points);
}

// A quiet NaN, made from its bits so that a compiler's fast math keeps it.
inline float quiet_nan() {
    const std::uint32_t bits = 0x7fc00000u;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// Whether `v` is a NaN, read from its bits as the WGSL does.
inline bool is_nan_bits(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, sizeof b);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

// The pooled heights of a row-major (rows, cols) tensor with factor f: the
// cell (i, j) pools the rows [i f, min((i+1) f, rows)) and the columns
// [j f, min((j+1) f, cols)), visiting rows then columns, skipping NaN values;
// a block of NaN values gives NaN. `height` holds ceil(rows/f) * ceil(cols/f)
// values on return.
inline void surface_heights(const float* data, std::int64_t rows, std::int64_t cols, std::int64_t f, std::uint32_t pool,
                            std::vector<float>* height) {
    const std::int64_t out_rows = ceil_div(rows, f), out_cols = ceil_div(cols, f);
    height->assign(static_cast<std::size_t>(out_rows * out_cols), 0.0f);
    for (std::int64_t i = 0; i < out_rows; ++i) {
        for (std::int64_t j = 0; j < out_cols; ++j) {
            const std::int64_t r1 = std::min(rows, (i + 1) * f), c1 = std::min(cols, (j + 1) * f);
            // The first value starts the pool, as in the WGSL, which holds no
            // constant infinity.
            float acc = 0.0f;
            std::uint32_t n = 0;
            for (std::int64_t r = i * f; r < r1; ++r) {
                for (std::int64_t c = j * f; c < c1; ++c) {
                    const float v = data[r * cols + c];
                    if (is_nan_bits(v)) continue;
                    if (n == 0) acc = v;
                    else if (pool == kPoolMax) acc = std::max(acc, v);
                    else if (pool == kPoolMin) acc = std::min(acc, v);
                    else acc += v;
                    ++n;
                }
            }
            float h = quiet_nan();
            if (n > 0) h = pool == kPoolMean ? acc / float(n) : acc;
            (*height)[static_cast<std::size_t>(i * out_cols + j)] = h;
        }
    }
}

// The mean coordinate of each block of f items out of `count`: the mean of
// `coords` over the block when given, else the mean index.
inline std::vector<float> block_means(std::int64_t count, std::int64_t f, const std::vector<float>* coords) {
    const std::int64_t blocks = ceil_div(count, f);
    std::vector<float> out(static_cast<std::size_t>(blocks), 0.0f);
    for (std::int64_t b = 0; b < blocks; ++b) {
        const std::int64_t lo = b * f, hi = std::min(count, (b + 1) * f);
        if (coords && std::int64_t(coords->size()) >= count) {
            double sum = 0;
            for (std::int64_t k = lo; k < hi; ++k) sum += (*coords)[static_cast<std::size_t>(k)];
            out[static_cast<std::size_t>(b)] = float(sum / double(hi - lo));
        } else {
            out[static_cast<std::size_t>(b)] = float(double(lo + hi - 1) / 2.0);
        }
    }
    return out;
}

// The kept rows of a path with stride s, as (P, 3) positions: the columns
// `columns[0..ncolumns)` of rows 0, s, 2s and so on; two columns give a z of 0.
inline void path_positions(const float* data, std::int64_t rows, std::int64_t cols, const std::array<int, 3>& columns,
                           int ncolumns, std::int64_t s, std::vector<float>* positions) {
    const std::int64_t points = ceil_div(rows, s);
    positions->assign(static_cast<std::size_t>(points * 3), 0.0f);
    for (std::int64_t p = 0; p < points; ++p) {
        const float* row = data + (p * s) * cols;
        for (int k = 0; k < ncolumns && k < 3; ++k)
            (*positions)[static_cast<std::size_t>(p * 3 + k)] = row[columns[static_cast<std::size_t>(k)]];
    }
}

// The kept rows of a path in the script's tensor output, (P, width) with
// width = the largest column index + 1: the first `width` columns of rows 0,
// s, 2s and so on, so that the app's reference over it with the same columns
// gives the same positions.
inline void path_rows(const float* data, std::int64_t rows, std::int64_t cols, std::int64_t width, std::int64_t s,
                      std::vector<float>* out) {
    const std::int64_t points = ceil_div(rows, s);
    out->assign(static_cast<std::size_t>(points * width), 0.0f);
    for (std::int64_t p = 0; p < points; ++p)
        for (std::int64_t c = 0; c < width && c < cols; ++c)
            (*out)[static_cast<std::size_t>(p * width + c)] = data[(p * s) * cols + c];
}

}  // namespace caliper_builder::programs::operator_kernels
