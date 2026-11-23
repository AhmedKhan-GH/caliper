// The validator's checks of the operators `surface` and `path`, of `params`
// and of the `tensor` output (design/instrument-interface.md, Operators,
// Tensor Output and Program Parameters).
//
// The declaration check of the sandbox, which validate.cpp runs on each
// program, already holds a tensor output's shape and the names and count of
// a program's params to their limits. This file adds the parameters of the
// operator steps, the steps after an operator, the shape of an operator's
// input when the catalog gives it (spec.operator_shape), and the agreement of the params with the WGSL
// (program.params_mismatch).
#include "validate_additions.h"

#include <caliper_builder/programs/operator_kernels.h>
#include <caliper_builder/programs/sandbox.h>

namespace caliper_builder::spec::detail {

namespace {

namespace k = programs::operator_kernels;

bool is_operator(TransformOp op) { return op == TransformOp::surface || op == TransformOp::path; }

// The literal parameters of one operator step.
// The codec reports a key of another op (spec.unknown_key).
void operator_step(const Transform& t, const std::string& path, Sink& sink) {
    if (t.op == TransformOp::surface) {
        if (t.max_cells && (*t.max_cells < 1 || *t.max_cells > k::kMaxCellsMax))
            sink.error("spec.out_of_range",
                       "max_cells is from 1 to 262,144, not " + std::to_string(*t.max_cells) + ".",
                       path + ".max_cells");
        if (t.pool && *t.pool != "mean" && *t.pool != "max" && *t.pool != "min")
            sink.error("spec.bad_value", "pool is mean, max or min, not \"" + *t.pool + "\".", path + ".pool");
        return;
    }
    if (!t.columns.empty() && (t.columns.size() < 2 || t.columns.size() > 3))
        sink.error("spec.bad_value",
                   "columns names two or three column indices, not " + std::to_string(t.columns.size()) + ".",
                   path + ".columns");
    for (std::size_t c = 0; c < t.columns.size(); ++c)
        if (t.columns[c] < 0)
            sink.error("spec.out_of_range", "A column index is at least 0.", index_path(path + ".columns", c));
    if (t.max_points && (*t.max_points < 2 || *t.max_points > k::kMaxPointsMax))
        sink.error("spec.out_of_range",
                   "max_points is from 2 to 1,048,576, not " + std::to_string(*t.max_points) + ".",
                   path + ".max_points");
}

void verdict(const programs::Verdict& v, const std::string& path, Sink& sink) {
    if (v.ok()) return;
    sink.error(v.code, v.message, v.where.empty() ? path : path + "." + v.where);
}

}  // namespace

void validate_operators(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.sources.size(); ++i) {
        const Source& s = c.spec.sources[i];
        const std::string base = index_path("source", i) + ".transform";
        // A source whose `from` ends in an operator holds geometry: no step
        // reads it.
        bool geometry = false;
        if (s.from)
            for (const Source& up : c.spec.sources)
                if (up.id == *s.from && !up.transforms.empty() && is_operator(up.transforms.back().op)) geometry = true;
        for (std::size_t j = 0; j < s.transforms.size(); ++j) {
            const Transform& t = s.transforms[j];
            const std::string path = index_path(base, j);
            if (geometry) {
                c.sink.error("spec.bad_transform",
                             std::string("The step ") + to_string(t.op) +
                                 " follows an operator, whose result is geometry and not a tensor; an operator is the "
                                 "last step of a source.",
                             path);
                continue;
            }
            if (is_operator(t.op)) {
                operator_step(t, path, c.sink);
                geometry = true;
            }
        }
    }
    for (std::size_t i = 0; i < c.spec.programs.size(); ++i) {
        const Program& p = c.spec.programs[i];
        if (p.wgsl.empty()) continue;
        verdict(programs::check_params(p), index_path("program", i), c.sink);
    }
}

bool apply_operator_shape(const Transform& t, SourceInfo& info, const std::string& path, Sink& sink) {
    if (info.shape) {
        const std::vector<std::int64_t>& shape = *info.shape;
        if (shape.size() != 2) {
            sink.error("spec.operator_shape",
                       std::string("The operator ") + to_string(t.op) + " takes a tensor of rank 2, and its input has rank " +
                           std::to_string(shape.size()) + ".",
                       path);
            info.shape.reset();
            return false;
        }
        if (t.op == TransformOp::path && shape[1] >= 0) {
            if (shape[1] < 2) {
                sink.error("spec.operator_shape",
                           "The operator path takes a tensor (R, D) with D at least 2, and its input has D = " +
                               std::to_string(shape[1]) + ".",
                           path);
                info.shape.reset();
                return false;
            }
            for (std::size_t c = 0; c < t.columns.size(); ++c)
                if (t.columns[c] >= shape[1]) {
                    sink.error("spec.operator_shape",
                               "The path reads the column " + std::to_string(t.columns[c]) + ", and its input has " +
                                   std::to_string(shape[1]) + " columns.",
                               index_path(path + ".columns", c));
                    info.shape.reset();
                    return false;
                }
        }
    }
    // The result is geometry, named arrays and not one tensor.
    info.shape.reset();
    info.dtype = "f32";
    return true;
}

}  // namespace caliper_builder::spec::detail
