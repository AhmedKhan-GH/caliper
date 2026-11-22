// The built-in operators surface and path
// (caliper_builder/operators/operators.h). The arithmetic is in
// caliper_builder/programs/operator_kernels.h, which the producer tests share
// to compare the device result with these references.
#include <caliper_builder/operators/operators.h>

#include <caliper_builder/programs/operator_kernels.h>
#include <caliper_builder/programs/sandbox.h>

#include <algorithm>

namespace caliper_builder::operators {

namespace detail {
std::string_view embedded_wgsl(std::string_view op);  // operators_wgsl_embed.cpp, generated
}

namespace k = caliper_builder::programs::operator_kernels;

Grid surface_reference(const float* data, std::int64_t rows, std::int64_t cols, const std::vector<float>* row_coords,
                       std::int64_t max_cells, Pool pool) {
    Grid g;
    if (!data || rows <= 0 || cols <= 0) return g;
    const std::int64_t f = k::pool_factor(rows, cols, max_cells);
    const std::uint32_t p = pool == Pool::max ? k::kPoolMax : pool == Pool::min ? k::kPoolMin : k::kPoolMean;
    k::surface_heights(data, rows, cols, f, p, &g.height);
    g.rows = k::ceil_div(rows, f);
    g.cols = k::ceil_div(cols, f);
    g.x = k::block_means(rows, f, row_coords);
    g.y = k::block_means(cols, f, nullptr);
    return g;
}

Polyline path_reference(const float* data, std::int64_t rows, std::int64_t cols, std::array<int, 3> columns,
                        int ncolumns, std::int64_t max_points, const std::vector<float>* row_coords) {
    Polyline out;
    if (!data || rows <= 0 || cols <= 0 || ncolumns < 2 || ncolumns > 3) return out;
    for (int c = 0; c < ncolumns; ++c)
        if (columns[static_cast<std::size_t>(c)] < 0 || columns[static_cast<std::size_t>(c)] >= cols) return out;
    const std::int64_t s = k::path_stride(rows, max_points);
    k::path_positions(data, rows, cols, columns, ncolumns, s, &out.positions);
    const std::int64_t points = k::ceil_div(rows, s);
    out.values.resize(static_cast<std::size_t>(points));
    const bool steps = row_coords && std::int64_t(row_coords->size()) >= rows;
    for (std::int64_t p = 0; p < points; ++p)
        out.values[static_cast<std::size_t>(p)] = steps ? (*row_coords)[static_cast<std::size_t>(p * s)] : float(p * s);
    return out;
}

std::string_view builtin_wgsl(std::string_view op) { return detail::embedded_wgsl(op); }

int builtin_version(std::string_view op) { return builtin_wgsl(op).empty() ? 0 : 1; }

std::string builtin_sha256(std::string_view op) {
    const std::string_view wgsl = builtin_wgsl(op);
    return wgsl.empty() ? std::string() : programs::sha256_hex(wgsl);
}

namespace {

// The columns of a path step: its `columns`, else the first three (or two
// when the input has two columns).
std::vector<std::int64_t> path_columns(const spec::Transform& t, std::int64_t cols) {
    if (!t.columns.empty()) return t.columns;
    std::vector<std::int64_t> out = {0, 1, 2};
    if (cols == 2) out.pop_back();
    return out;
}

std::uint32_t pool_number(const std::optional<std::string>& pool) {
    if (pool && *pool == "max") return k::kPoolMax;
    if (pool && *pool == "min") return k::kPoolMin;
    return k::kPoolMean;
}

}  // namespace

spec::Program operator_program(const spec::Source& source, std::size_t step,
                               const std::vector<std::int64_t>& input_shape) {
    spec::Program p;
    p.id = "operator." + source.id + "." + std::to_string(step);
    p.output.kind = spec::OutputKind::tensor;
    if (step >= source.transforms.size()) return p;
    const spec::Transform& t = source.transforms[step];
    const char* op = t.op == spec::TransformOp::surface ? "surface" : t.op == spec::TransformOp::path ? "path" : "";
    if (!*op) return p;
    p.title = std::string(op) + " of " + source.id;
    p.wgsl = std::string(builtin_wgsl(op));
    p.entry = "run";
    p.builtin = std::string(op) + "@" + std::to_string(builtin_version(op));
    if (source.catalog) p.inputs.push_back(spec::ProgramInput{"input", *source.catalog});
    if (input_shape.size() != 2 || input_shape[0] < 1 || input_shape[1] < 1) return p;
    const std::int64_t rows = input_shape[0], cols = input_shape[1];
    std::uint64_t elements = 0;
    if (t.op == spec::TransformOp::surface) {
        const std::int64_t f = k::pool_factor(rows, cols, t.max_cells.value_or(k::kMaxCellsDefault));
        const std::int64_t out_rows = k::ceil_div(rows, f), out_cols = k::ceil_div(cols, f);
        p.output.shape = {out_rows, out_cols};
        p.params = {{"rows", double(rows)},         {"cols", double(cols)},         {"factor", double(f)},
                    {"out_rows", double(out_rows)}, {"out_cols", double(out_cols)}, {"pool", double(pool_number(t.pool))}};
        elements = std::uint64_t(out_rows * out_cols);
    } else {
        const std::vector<std::int64_t> columns = path_columns(t, cols);
        const std::int64_t width = *std::max_element(columns.begin(), columns.end()) + 1;
        const std::int64_t s = k::path_stride(rows, t.max_points.value_or(k::kMaxPointsDefault));
        const std::int64_t points = k::ceil_div(rows, s);
        p.output.shape = {points, width};
        p.params = {{"rows", double(rows)},     {"cols", double(cols)},       {"stride", double(s)},
                    {"points", double(points)}, {"out_cols", double(width)}};
        elements = std::uint64_t(points);
    }
    const std::uint64_t groups = (elements + k::kWorkgroupSize - 1) / k::kWorkgroupSize;
    p.workgroups = std::array<std::uint32_t, 3>{static_cast<std::uint32_t>(std::max<std::uint64_t>(1, groups)), 1, 1};
    // The budget holds the bound input and the output, rounded up to MiB.
    const std::uint64_t bytes = std::uint64_t(rows) * std::uint64_t(cols) * 4 + programs::output_bytes(p);
    const std::uint64_t mb = (bytes + (1u << 20) - 1) >> 20;
    p.limits.memory_mb = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>(std::max<std::uint64_t>(mb + 1, spec::program_limits::kMemoryMbDefault), 1,
                                  spec::program_limits::kMemoryMbMax));
    return p;
}

}  // namespace caliper_builder::operators
