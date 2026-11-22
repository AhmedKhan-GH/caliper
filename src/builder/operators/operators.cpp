// The built-in operators surface and path
// (caliper_builder/operators/operators.h). Not built yet: the references
// give empty geometry, and the validator refuses the operators
// (spec.not_built).
#include <caliper_builder/operators/operators.h>

namespace caliper_builder::operators {

namespace detail {
std::string_view embedded_wgsl(std::string_view op);  // operators_wgsl_embed.cpp, generated
}

Grid surface_reference(const float* data, std::int64_t rows, std::int64_t cols, const std::vector<float>* row_coords,
                       std::int64_t max_cells, Pool pool) {
    (void)data;
    (void)rows;
    (void)cols;
    (void)row_coords;
    (void)max_cells;
    (void)pool;
    return {};
}

Polyline path_reference(const float* data, std::int64_t rows, std::int64_t cols, std::array<int, 3> columns,
                        int ncolumns, std::int64_t max_points, const std::vector<float>* row_coords) {
    (void)data;
    (void)rows;
    (void)cols;
    (void)columns;
    (void)ncolumns;
    (void)max_points;
    (void)row_coords;
    return {};
}

std::string_view builtin_wgsl(std::string_view op) { return detail::embedded_wgsl(op); }

int builtin_version(std::string_view op) { return builtin_wgsl(op).empty() ? 0 : 1; }

std::string builtin_sha256(std::string_view op) {
    (void)op;
    return {};
}

spec::Program operator_program(const spec::Source& source, std::size_t step,
                               const std::vector<std::int64_t>& input_shape) {
    (void)input_shape;
    spec::Program p;
    p.id = "operator." + source.id + "." + std::to_string(step);
    p.output.kind = spec::OutputKind::tensor;
    return p;
}

}  // namespace caliper_builder::operators
