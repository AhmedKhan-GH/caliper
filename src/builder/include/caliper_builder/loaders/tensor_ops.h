#pragma once
// The CPU reference implementation of the tensor transforms (slice, reduce,
// normalise) and the statistics that catalog.sample gives for a tensor.
//
// A transform is a record of the contracts' fixed vocabulary, the same in a
// loader and in a spec. Its JSON form is {"op": "slice", "axis": 1, "start": 0,
// "stop": 4}, {"op": "reduce", "axis": [2, 3], "fn": "mean", "keepdims": false}
// or {"op": "normalise", "mode": "softmax", "axis": -1}. The rules, which a GPU
// version must match, are in docs/design/loaders.md ("Tensor outputs and
// transforms"):
//   - axes may be negative and count from the end; a list may not repeat one;
//   - slice keeps the dtype; reduce and normalise compute in double precision
//     and return f32 (f64 for f64 input); argmax and argmin return i64 and take
//     exactly one axis;
//   - std and var are population statistics; entropy divides by the sum and is
//     in nats, skipping values <= 0; NaN propagates everywhere else.
// Every result is a new owned tensor. Errors have codes "transform.*", and
// apply_transforms puts the failing transform's position in `where`.
//
// tensor_stats and tensor_sample read every element once and allocate only the
// sample. No function here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/tensor.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

enum class TransformOp : uint8_t { slice, reduce, normalise };
enum class ReduceFn : uint8_t { mean, sum, min, max, std, var, norm, argmax, argmin, entropy };
enum class NormaliseMode : uint8_t { minmax, zscore, softmax, l2 };

struct TensorTransform {
    TransformOp op = TransformOp::slice;
    // slice: exactly one axis; reduce: one or more; normalise: none means the whole tensor.
    std::vector<int64_t> axes;
    // Whether `axis` was written as a list, so that the JSON form round-trips.
    bool axis_is_list = false;
    // slice: `index` drops the axis; otherwise start, stop and step select along it.
    std::optional<int64_t> index;
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step;
    // reduce
    ReduceFn fn = ReduceFn::mean;
    bool keepdims = false;
    // normalise
    NormaliseMode mode = NormaliseMode::minmax;
};

const char* reduce_fn_name(ReduceFn fn);
const char* normalise_mode_name(NormaliseMode mode);

// Parses one transform record. Unknown keys, wrong types and missing fields are
// errors; `where` names the field ("axis", "fn").
Error transform_from_json(const nlohmann::json& record, TensorTransform* out);
nlohmann::json transform_to_json(const TensorTransform& transform);

// The dtype and shape a transform gives for an input, without computing it.
Error transform_result_shape(const std::vector<int64_t>& shape, DType dtype, const TensorTransform& transform,
                             std::vector<int64_t>* out_shape, DType* out_dtype);

TensorResult apply_transform(const TensorData& input, const TensorTransform& transform);
TensorResult apply_transforms(const TensorData& input, const std::vector<TensorTransform>& transforms);

struct TensorStats {
    uint64_t count = 0;      // every element
    uint64_t nonfinite = 0;  // NaN and infinities
    // Over the finite elements; empty when there are none.
    std::optional<double> min;
    std::optional<double> max;
    std::optional<double> mean;
    std::optional<double> std;  // population standard deviation

    // {"count": n, "nonfinite": n, "min": x|null, "max": ..., "mean": ..., "std": ...}
    nlohmann::json to_json() const;
};

TensorStats tensor_stats(const TensorData& tensor);

// A slice from the start of the tensor: at most `per_axis` values along each
// axis and at most `max_values` in all; when that would be exceeded, leading
// axes are cut to one index each until it fits. The result is
// {"start": [0, ...], "shape": [...], "values": nested arrays}, with a
// non-finite value written as "NaN", "Infinity" or "-Infinity".
nlohmann::json tensor_sample(const TensorData& tensor, int64_t per_axis = 64, uint64_t max_values = 4096);

// The JSON form of one element: a number, a boolean, or a string for a
// non-finite float.
nlohmann::json tensor_element_json(const TensorData& tensor, uint64_t index);

}  // namespace caliper_builder
