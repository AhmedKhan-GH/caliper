// The operator steps of a source chain in the app (chain_hooks.h): `surface`
// and `path` through their CPU references (caliper_builder/operators/
// operators.h), and the rule that places the first operator of a live tensor
// source in the script (design/instrument-interface.md, Operator Placement).
#include <caliper_builder/session/chain_hooks.h>

#include <caliper_builder/operators/operators.h>
#include <caliper_builder/programs/operator_kernels.h>
#include <caliper_builder/spec/instrument.h>

#include <cmath>
#include <cstring>

namespace caliper_builder {

namespace {

namespace k = programs::operator_kernels;

// A float tensor of `shape` with `values`.
TensorData float_tensor(std::vector<int64_t> shape, const std::vector<float>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    TensorResult r = TensorData::from_bytes(DType::f32, std::move(shape), std::move(bytes));
    return r.error ? TensorData() : std::move(r.tensor);
}

// The elements of `t` as floats in row-major order.
std::vector<float> floats_of(const TensorData& t) {
    std::vector<float> out(static_cast<std::size_t>(t.numel()));
    if (t.dtype() == DType::f32 && t.data()) {
        std::memcpy(out.data(), t.data(), out.size() * sizeof(float));
    } else {
        for (uint64_t i = 0; i < t.numel(); ++i) out[static_cast<std::size_t>(i)] = static_cast<float>(t.element(i));
    }
    return out;
}

std::string shape_text(const std::vector<int64_t>& shape) {
    std::string s = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + (shape.size() == 1 ? ",)" : ")");
}

bool is_operator(spec::TransformOp op) { return op == spec::TransformOp::surface || op == spec::TransformOp::path; }

}  // namespace

Error apply_operator(const spec::Transform& step, const TensorData& input, const TensorData* row_steps,
                     GeometryArrays* out) {
    const char* name = spec::to_string(step.op);
    if (!is_operator(step.op))
        return Error("resolve.bad_transform", std::string("The step ") + name + " is not an operator.");
    if (!out) return Error("resolve.bad_transform", "The operator has nowhere to write its geometry.");
    if (!input.valid() || input.rank() != 2)
        return Error("resolve.operator_shape", std::string("The operator ") + name +
                                                   " takes a tensor of rank 2, and its input has the shape " +
                                                   shape_text(input.valid() ? input.shape() : std::vector<int64_t>{}) +
                                                   ".");
    const int64_t rows = input.shape()[0], cols = input.shape()[1];
    const std::vector<float> data = floats_of(input);
    std::vector<float> steps;
    const std::vector<float>* coords = nullptr;
    if (row_steps && row_steps->valid() && int64_t(row_steps->numel()) == rows) {
        steps = floats_of(*row_steps);
        coords = &steps;
    }
    out->clear();
    if (step.op == spec::TransformOp::surface) {
        if (rows < 1 || cols < 1) {
            (*out)["height"] = float_tensor({0, 0}, {});
            (*out)["x"] = float_tensor({0}, {});
            (*out)["y"] = float_tensor({0}, {});
            return {};
        }
        operators::Pool pool = operators::Pool::mean;
        if (step.pool && *step.pool == "max") pool = operators::Pool::max;
        if (step.pool && *step.pool == "min") pool = operators::Pool::min;
        const operators::Grid g = operators::surface_reference(data.data(), rows, cols, coords,
                                                               step.max_cells.value_or(k::kMaxCellsDefault), pool);
        (*out)["height"] = float_tensor({g.rows, g.cols}, g.height);
        (*out)["x"] = float_tensor({g.rows}, g.x);
        (*out)["y"] = float_tensor({g.cols}, g.y);
        return {};
    }
    // path: the columns are checked as the spec gives them, before they narrow to int
    std::array<int64_t, 3> wanted{0, 1, 2};
    int ncolumns = cols == 2 ? 2 : 3;
    if (!step.columns.empty()) {
        ncolumns = static_cast<int>(std::min<std::size_t>(step.columns.size(), 3));
        for (int c = 0; c < ncolumns; ++c) wanted[static_cast<std::size_t>(c)] = step.columns[c];
    }
    if (cols < 2)
        return Error("resolve.operator_shape", "The operator path takes a tensor (R, D) with D at least 2, and its "
                                               "input has the shape " + shape_text(input.shape()) + ".");
    std::array<int, 3> columns{0, 1, 2};
    for (int c = 0; c < ncolumns; ++c) {
        const int64_t col = wanted[static_cast<std::size_t>(c)];
        if (col < 0 || col >= cols)
            return Error("resolve.operator_shape", "The path reads the column " + std::to_string(col) +
                                                       ", and its input has " + std::to_string(cols) + " columns.");
        columns[static_cast<std::size_t>(c)] = static_cast<int>(col);
    }
    if (rows < 1) {
        (*out)["positions"] = float_tensor({0, 3}, {});
        (*out)["values"] = float_tensor({0}, {});
        return {};
    }
    const operators::Polyline line = operators::path_reference(
        data.data(), rows, cols, columns, ncolumns, step.max_points.value_or(k::kMaxPointsDefault), coords);
    const int64_t points = static_cast<int64_t>(line.values.size());
    (*out)["positions"] = float_tensor({points, 3}, line.positions);
    (*out)["values"] = float_tensor({points}, line.values);
    return {};
}

bool operator_runs_in_script(const spec::Spec& spec, const spec::Source& source, std::size_t index) {
    if (index != 0 || source.transforms.empty() || !is_operator(source.transforms[0].op)) return false;
    // The operator is the first step of the chain: every source between the
    // root and this one has no step.
    const spec::Source* s = &source;
    for (std::size_t depth = 0; depth <= spec.sources.size(); ++depth) {
        if (s->catalog) {
            const std::string prefix = spec::kLiveScenePrefix;
            return s->catalog->compare(0, prefix.size(), prefix) == 0 && s->catalog->size() > prefix.size();
        }
        if (!s->from) return false;
        const spec::Source* up = nullptr;
        for (const spec::Source& c : spec.sources)
            if (c.id == *s->from) up = &c;
        if (!up || !up->transforms.empty()) return false;
        s = up;
    }
    return false;
}

}  // namespace caliper_builder
