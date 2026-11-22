// The step `window` over kept steps (chain_hooks.h,
// design/instrument-interface.md, Window Step). It stacks the values that
// the history keeps for the source that `from` names, oldest first, and
// never holds a step after the cursor.
#include <caliper_builder/session/chain_hooks.h>

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resolve.h>

#include <cstring>
#include <limits>

namespace caliper_builder {

Error apply_window(const spec::Transform& step, const spec::Source& source, const ChainContext& ctx, TensorData* out,
                   TensorData* steps) {
    const std::string where = "source " + source.id;
    if (!source.from)
        return Error("resolve.unknown_source", "The step window reads the source that `from` names, and " + source.id +
                                              " has no `from`.", where);
    const std::string& kept_by = *source.from;
    if (!ctx.history)
        return Error("resolve.unavailable", "The app keeps no steps here, so the window over " + kept_by +
                                                " has no values.", where);
    // The window never holds a step after the cursor; a live cursor stands at
    // the newest step.
    const std::int64_t cursor = ctx.cursor_live ? ctx.history->cursor().latest : ctx.cursor_step;
    std::int64_t lo = step.start.value_or(std::numeric_limits<std::int64_t>::min());
    std::int64_t hi = step.stop ? std::min(*step.stop, cursor) : cursor;
    std::vector<std::pair<std::int64_t, TensorData>> values = ctx.history->window(kept_by, lo, hi);
    if (step.steps && (std::int64_t)values.size() > *step.steps)
        values.erase(values.begin(), values.end() - (std::ptrdiff_t)*step.steps);
    if (values.empty())
        return Error("resolve.unavailable",
                     "The source " + kept_by + " keeps no step in the window yet (up to step " + std::to_string(hi) +
                         ").",
                     where);
    const TensorData& first = values.front().second;
    const std::uint64_t one = first.nbytes();
    const std::uint64_t limit = ResolveLimits{}.max_tensor_bytes;
    if (one != 0 && values.size() > limit / one)
        return Error("resolve.too_large",
                     "The window over " + kept_by + " would make a tensor of " + std::to_string(one * values.size()) +
                         " bytes, past the limit of " + std::to_string(limit) + ".",
                     where);
    std::vector<std::int64_t> shape{(std::int64_t)values.size()};
    for (std::int64_t d : first.shape()) shape.push_back(d);
    TensorResult stacked = TensorData::allocate(first.dtype(), shape);
    if (stacked.error) return stacked.error;
    TensorResult rows = TensorData::allocate(DType::i64, {(std::int64_t)values.size()});
    if (rows.error) return rows.error;
    std::uint8_t* dst = stacked.tensor.mutable_data();
    auto* row_steps = reinterpret_cast<std::int64_t*>(rows.tensor.mutable_data());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const TensorData& v = values[i].second;
        if (v.shape() != first.shape() || v.dtype() != first.dtype())
            return Error("resolve.shape", "The kept steps of " + kept_by + " changed shape inside the window.", where);
        if (one) std::memcpy(dst + i * one, v.data(), one);
        row_steps[i] = values[i].first;
    }
    *out = std::move(stacked.tensor);
    if (steps) *steps = std::move(rows.tensor);
    return {};
}

}  // namespace caliper_builder
