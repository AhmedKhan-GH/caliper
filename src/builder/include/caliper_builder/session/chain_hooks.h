#pragma once
// caliper_builder: the hooks of a source chain for references, kept steps
// and operators (design/instrument-interface.md).
//
// resolve.cpp resolves a panel's slots through the chain of each source and
// calls these hooks: it replaces the references of each step with their
// values before the step runs, adds the values that the chain reads to its
// cache key, and hands the steps `window`, `surface` and `path` to their
// owners. Each hook is defined in the file named beside it.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <caliper_builder/error.h>
#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/spec/spec.h>

namespace caliper_builder {

class ValueStore;   // values.h
class StepHistory;  // history.h

// What the hooks read. A null store gives each reference its default, and a
// null history keeps no steps.
struct ChainContext {
    const ValueStore* values = nullptr;
    const StepHistory* history = nullptr;
    std::int64_t cursor_step = 0;
    bool cursor_live = true;
};

// The named arrays of geometry that an operator gives: "height", "x", "y"
// for `surface`, "positions" and "values" for `path`.
using GeometryArrays = std::map<std::string, TensorData>;

// references.cpp: `step` with each reference replaced by the literal it
// reads now. resolve.waiting when a reference has no value and no default.
Error substitute_references(const spec::Transform& step, const ChainContext& ctx, spec::Transform* out);

// references.cpp: the values that the references of the chain's steps read
// now, as text for the chain's cache key; empty when the chain reads none.
std::string references_key(const std::vector<const spec::Source*>& chain, const ChainContext& ctx);

// steps_window.cpp: the step `window` of `source`, which stacks the kept
// steps of the source that `from` names into a tensor (W, ...) and gives the
// step of each row in `steps` (W).
Error apply_window(const spec::Transform& step, const spec::Source& source, const ChainContext& ctx,
                   TensorData* out, TensorData* steps);

// operator_steps.cpp: an operator step in the app, through its CPU
// reference (operators.h).
Error apply_operator(const spec::Transform& step, const TensorData& input, const TensorData* row_steps,
                     GeometryArrays* out);

// operator_steps.cpp: whether step `index` of `source` runs in the script:
// the first step of a source whose root is a live tensor, when it is an
// operator.
bool operator_runs_in_script(const spec::Spec& spec, const spec::Source& source, std::size_t index);

}  // namespace caliper_builder
