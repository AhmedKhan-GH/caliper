#pragma once
// caliper_builder::operators: the built-in operators `surface` and `path`
// (design/instrument-interface.md, section Operators).
//
// An operator is a step of a source that turns a tensor into geometry. Each
// one has one WGSL source, which runs in the script as a built-in program on
// the producer runtime, and one CPU reference, which runs in the app and
// which the tests compare with the device result. The app turns a placed
// operator into a program declaration with operator_program() and sends it
// in the message PROGRAM.
//
// The functions are defined in src/builder/operators/operators.cpp, and the
// WGSL sources are src/builder/operators/wgsl/*.wgsl, compiled in. Nothing
// here throws.
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <caliper_builder/spec/spec.h>

namespace caliper_builder::operators {

// The pool of a surface's blocks.
enum class Pool { mean, max, min };

// A height field of R' by C' cells: `height` row-major, `x` (R') the mean
// coordinate of the pooled rows, `y` (C') the mean column index.
struct Grid {
    std::int64_t rows = 0, cols = 0;
    std::vector<float> height, x, y;
};

// The CPU reference of `surface` over a row-major (rows, cols) tensor. When
// rows times cols is above max_cells, it pools blocks of f by f cells with
// the smallest f that fits; a pooled cell skips NaN values, and a block of
// NaN values gives NaN. `row_coords` (rows), when given, holds the step of
// each row after a window; else the row index is the coordinate.
Grid surface_reference(const float* data, std::int64_t rows, std::int64_t cols, const std::vector<float>* row_coords,
                       std::int64_t max_cells, Pool pool);

// A polyline: `positions` (P, 3) and `values` (P).
struct Polyline {
    std::vector<float> positions, values;
};

// The CPU reference of `path` over a row-major (rows, cols) tensor: one
// vertex for each kept row, from the columns `columns` (the first `ncolumns`
// of them; two give a z of 0), keeping every s-th row with the smallest s
// for which at most max_points remain. The value of a row is its step from
// `row_coords`, else its row index.
Polyline path_reference(const float* data, std::int64_t rows, std::int64_t cols, std::array<int, 3> columns,
                        int ncolumns, std::int64_t max_points, const std::vector<float>* row_coords);

// The WGSL source of an operator ("surface" or "path"), empty for another
// name, and its version, which the program's `builtin` names
// ("surface@1").
std::string_view builtin_wgsl(std::string_view op);
int builtin_version(std::string_view op);
// The SHA-256 of builtin_wgsl(op) in lower-case hexadecimal, which the app
// knows, so that an operator's program needs no review.
std::string builtin_sha256(std::string_view op);

// The program declaration of step `step` of `source`, an operator over a
// live tensor of `input_shape`: the id "operator.<source>.<step>", the
// built-in WGSL, the input, a `tensor` output of the pooled shape, the
// workgroups and `builtin`.
spec::Program operator_program(const spec::Source& source, std::size_t step,
                               const std::vector<std::int64_t>& input_shape);

}  // namespace caliper_builder::operators
