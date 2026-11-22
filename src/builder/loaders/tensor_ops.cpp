// The CPU reference implementation of the tensor transforms and statistics.
#include <caliper_builder/loaders/tensor_ops.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace caliper_builder {
namespace {

using json = nlohmann::json;

const ReduceFn kReduceFns[] = {ReduceFn::mean, ReduceFn::sum,    ReduceFn::min,    ReduceFn::max,
                               ReduceFn::std,  ReduceFn::var,    ReduceFn::norm,   ReduceFn::argmax,
                               ReduceFn::argmin, ReduceFn::entropy};
const NormaliseMode kModes[] = {NormaliseMode::minmax, NormaliseMode::zscore, NormaliseMode::softmax,
                                NormaliseMode::l2};

bool json_int(const json& v, int64_t* out) {
    if (v.is_number_integer() && !v.is_number_unsigned()) {
        *out = v.get<int64_t>();
        return true;
    }
    if (v.is_number_unsigned()) {
        const uint64_t u = v.get<uint64_t>();
        if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return false;
        *out = static_cast<int64_t>(u);
        return true;
    }
    return false;
}

Error bad(const std::string& code, const std::string& message, const std::string& where) {
    return Error("transform." + code, message, where);
}

// Resolves axes against a rank: negative axes count from the end, none may repeat.
Error resolve_axes(const std::vector<int64_t>& axes, int rank, std::vector<int>* out) {
    out->clear();
    for (int64_t a : axes) {
        const int64_t r = a < 0 ? a + rank : a;
        if (r < 0 || r >= rank) {
            return bad("axis_out_of_range",
                       "Axis " + std::to_string(a) + " does not exist in a tensor of rank " + std::to_string(rank) + ".",
                       "axis");
        }
        if (std::find(out->begin(), out->end(), static_cast<int>(r)) != out->end()) {
            return bad("duplicate_axis", "Axis " + std::to_string(a) + " is listed twice.", "axis");
        }
        out->push_back(static_cast<int>(r));
    }
    return {};
}

DType float_result(DType in) { return in == DType::f64 ? DType::f64 : DType::f32; }

// The Python slice bounds of one axis of size n: start, and the number of elements selected.
void slice_bounds(const TensorTransform& t, int64_t n, int64_t* start, int64_t* count) {
    const int64_t step = t.step.value_or(1);
    auto clamp = [n](int64_t v) {
        if (v < 0) v += n;
        if (v < 0) v = 0;
        if (v > n) v = n;
        return v;
    };
    const int64_t s = t.start ? clamp(*t.start) : 0;
    const int64_t e = t.stop ? clamp(*t.stop) : n;
    *start = s;
    *count = e > s ? (e - s + step - 1) / step : 0;
}

std::vector<uint64_t> row_major_strides(const std::vector<int64_t>& shape) {
    std::vector<uint64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * static_cast<uint64_t>(shape[i + 1]);
    }
    return strides;
}

// The split of a tensor's axes into kept axes (one group per combination) and
// reduced axes (the elements of a group).
struct AxisSplit {
    std::vector<uint64_t> kept_dims, kept_strides, red_dims, red_strides;
    uint64_t groups = 1;
    uint64_t group_size = 1;
};

AxisSplit split_axes(const std::vector<int64_t>& shape, const std::vector<int>& reduced) {
    AxisSplit s;
    const std::vector<uint64_t> strides = row_major_strides(shape);
    for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
        const uint64_t d = static_cast<uint64_t>(shape[i]);
        if (std::find(reduced.begin(), reduced.end(), i) != reduced.end()) {
            s.red_dims.push_back(d);
            s.red_strides.push_back(strides[i]);
            s.group_size *= d;
        } else {
            s.kept_dims.push_back(d);
            s.kept_strides.push_back(strides[i]);
            s.groups *= d;
        }
    }
    return s;
}

uint64_t group_base(const AxisSplit& s, uint64_t g) {
    uint64_t base = 0;
    for (int i = static_cast<int>(s.kept_dims.size()) - 1; i >= 0; --i) {
        const uint64_t d = s.kept_dims[i];
        base += (g % d) * s.kept_strides[i];
        g /= d;
    }
    return base;
}

// Calls f(offset) for every element of a group, in row-major order of the reduced axes.
template <class F>
void for_each_in_group(const AxisSplit& s, uint64_t base, F&& f) {
    if (s.group_size == 0) return;
    const std::size_t n = s.red_dims.size();
    uint64_t idx[kMaxTensorRank] = {};
    uint64_t offset = base;
    for (uint64_t k = 0; k < s.group_size; ++k) {
        f(offset);
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            if (++idx[i] < s.red_dims[i]) {
                offset += s.red_strides[i];
                break;
            }
            offset -= (s.red_dims[i] - 1) * s.red_strides[i];
            idx[i] = 0;
        }
    }
}

void store(uint8_t* out, DType dtype, uint64_t i, double v) {
    if (dtype == DType::f64) {
        std::memcpy(out + i * 8, &v, 8);
    } else if (dtype == DType::i64) {
        const int64_t x = static_cast<int64_t>(v);
        std::memcpy(out + i * 8, &x, 8);
    } else {
        const float f = static_cast<float>(v);
        std::memcpy(out + i * 4, &f, 4);
    }
}

double reduce_group(const TensorData& in, const AxisSplit& s, uint64_t base, ReduceFn fn) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const uint64_t n = s.group_size;
    switch (fn) {
        case ReduceFn::sum:
        case ReduceFn::mean: {
            double sum = 0;
            for_each_in_group(s, base, [&](uint64_t o) { sum += in.element(o); });
            if (fn == ReduceFn::sum) return sum;
            return n ? sum / static_cast<double>(n) : nan;
        }
        case ReduceFn::min:
        case ReduceFn::max: {
            if (n == 0) return nan;
            bool first = true;
            double best = 0;
            bool saw_nan = false;
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                if (std::isnan(v)) saw_nan = true;
                if (first || (fn == ReduceFn::min ? v < best : v > best)) best = v;
                first = false;
            });
            return saw_nan ? nan : best;
        }
        case ReduceFn::var:
        case ReduceFn::std: {
            if (n == 0) return nan;
            double mean = 0, m2 = 0;
            uint64_t k = 0;
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                ++k;
                const double d = v - mean;
                mean += d / static_cast<double>(k);
                m2 += d * (v - mean);
            });
            const double var = m2 / static_cast<double>(n);
            return fn == ReduceFn::var ? var : std::sqrt(var);
        }
        case ReduceFn::norm: {
            double sq = 0;
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                sq += v * v;
            });
            return std::sqrt(sq);
        }
        case ReduceFn::argmax:
        case ReduceFn::argmin: {
            uint64_t k = 0, best_k = 0;
            double best = 0;
            bool found_nan = false;
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                if (!found_nan) {
                    if (std::isnan(v)) {
                        found_nan = true;
                        best_k = k;
                    } else if (k == 0 || (fn == ReduceFn::argmax ? v > best : v < best)) {
                        best = v;
                        best_k = k;
                    }
                }
                ++k;
            });
            return static_cast<double>(best_k);
        }
        case ReduceFn::entropy: {
            double sum = 0;
            bool saw_nan = false;
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                if (std::isnan(v)) saw_nan = true;
                else if (v > 0) sum += v;
            });
            if (saw_nan || !(sum > 0)) return nan;
            double h = 0;
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                if (v > 0) {
                    const double p = v / sum;
                    h -= p * std::log(p);
                }
            });
            return h;
        }
    }
    return nan;
}

void normalise_group(const TensorData& in, const AxisSplit& s, uint64_t base, NormaliseMode mode, uint8_t* out,
                     DType out_dtype) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    switch (mode) {
        case NormaliseMode::minmax: {
            const double lo = reduce_group(in, s, base, ReduceFn::min);
            const double hi = reduce_group(in, s, base, ReduceFn::max);
            const double range = hi - lo;
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                double r;
                if (std::isnan(range)) r = nan;
                else if (range == 0) r = std::isnan(v) ? nan : 0.0;
                else r = (v - lo) / range;
                store(out, out_dtype, o, r);
            });
            return;
        }
        case NormaliseMode::zscore: {
            const double mean = reduce_group(in, s, base, ReduceFn::mean);
            const double sd = reduce_group(in, s, base, ReduceFn::std);
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                double r;
                if (std::isnan(sd)) r = nan;
                else if (sd == 0) r = std::isnan(v) ? nan : 0.0;
                else r = (v - mean) / sd;
                store(out, out_dtype, o, r);
            });
            return;
        }
        case NormaliseMode::softmax: {
            const double hi = reduce_group(in, s, base, ReduceFn::max);
            double sum = 0;
            for_each_in_group(s, base, [&](uint64_t o) { sum += std::exp(in.element(o) - hi); });
            for_each_in_group(s, base, [&](uint64_t o) {
                store(out, out_dtype, o, std::exp(in.element(o) - hi) / sum);
            });
            return;
        }
        case NormaliseMode::l2: {
            const double norm = reduce_group(in, s, base, ReduceFn::norm);
            for_each_in_group(s, base, [&](uint64_t o) {
                const double v = in.element(o);
                double r;
                if (std::isnan(norm)) r = nan;
                else if (norm == 0) r = 0.0;
                else r = v / norm;
                store(out, out_dtype, o, r);
            });
            return;
        }
    }
}

std::vector<int> all_axes(int rank) {
    std::vector<int> a(static_cast<std::size_t>(rank));
    for (int i = 0; i < rank; ++i) a[i] = i;
    return a;
}

json float_json(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    return v;
}

}  // namespace

const char* reduce_fn_name(ReduceFn fn) {
    switch (fn) {
        case ReduceFn::mean: return "mean";
        case ReduceFn::sum: return "sum";
        case ReduceFn::min: return "min";
        case ReduceFn::max: return "max";
        case ReduceFn::std: return "std";
        case ReduceFn::var: return "var";
        case ReduceFn::norm: return "norm";
        case ReduceFn::argmax: return "argmax";
        case ReduceFn::argmin: return "argmin";
        case ReduceFn::entropy: return "entropy";
    }
    return "mean";
}

const char* normalise_mode_name(NormaliseMode mode) {
    switch (mode) {
        case NormaliseMode::minmax: return "minmax";
        case NormaliseMode::zscore: return "zscore";
        case NormaliseMode::softmax: return "softmax";
        case NormaliseMode::l2: return "l2";
    }
    return "minmax";
}

Error transform_from_json(const json& record, TensorTransform* out) {
    if (!record.is_object()) return bad("bad_record", "A transform is a table with an `op` key.", "");
    auto op_it = record.find("op");
    if (op_it == record.end()) return bad("missing_key", "A transform needs `op`.", "op");
    if (!op_it->is_string()) return bad("bad_value", "`op` is a string.", "op");
    const std::string op = op_it->get<std::string>();
    TensorTransform t;
    std::vector<std::string> allowed;
    if (op == "slice") {
        t.op = TransformOp::slice;
        allowed = {"op", "axis", "index", "start", "stop", "step"};
    } else if (op == "reduce") {
        t.op = TransformOp::reduce;
        allowed = {"op", "axis", "fn", "keepdims"};
    } else if (op == "normalise") {
        t.op = TransformOp::normalise;
        allowed = {"op", "axis", "mode"};
    } else {
        return bad("unknown_op", "`op` is one of slice, reduce and normalise, not \"" + op + "\".", "op");
    }
    for (auto it = record.begin(); it != record.end(); ++it) {
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end()) {
            return bad("unknown_key", "A " + op + " transform has no key `" + it.key() + "`.", it.key());
        }
    }
    auto axis_it = record.find("axis");
    if (axis_it != record.end()) {
        int64_t a = 0;
        if (json_int(*axis_it, &a)) {
            t.axes = {a};
        } else if (axis_it->is_array() && t.op != TransformOp::slice) {
            if (axis_it->empty()) return bad("bad_value", "`axis` lists at least one axis.", "axis");
            if (axis_it->size() > static_cast<std::size_t>(kMaxTensorRank)) {
                return bad("bad_value", "`axis` lists at most 8 axes.", "axis");
            }
            for (const json& v : *axis_it) {
                if (!json_int(v, &a)) return bad("bad_value", "`axis` lists integers.", "axis");
                t.axes.push_back(a);
            }
            t.axis_is_list = true;
        } else {
            return bad("bad_value",
                       t.op == TransformOp::slice ? "`axis` is an integer." : "`axis` is an integer or a list of integers.",
                       "axis");
        }
    } else if (t.op != TransformOp::normalise) {
        return bad("missing_key", "A " + op + " transform needs `axis`.", "axis");
    }
    if (t.op == TransformOp::slice) {
        auto get = [&](const char* key, std::optional<int64_t>* dst) -> Error {
            auto it = record.find(key);
            if (it == record.end()) return {};
            int64_t v = 0;
            if (!json_int(*it, &v)) return bad("bad_value", std::string("`") + key + "` is an integer.", key);
            *dst = v;
            return {};
        };
        if (Error e = get("index", &t.index)) return e;
        if (Error e = get("start", &t.start)) return e;
        if (Error e = get("stop", &t.stop)) return e;
        if (Error e = get("step", &t.step)) return e;
        if (t.index && (t.start || t.stop || t.step)) {
            return bad("bad_value", "A slice takes either `index` or `start`, `stop` and `step`, not both.", "index");
        }
        if (t.step && *t.step < 1) return bad("bad_value", "`step` is at least 1.", "step");
    } else if (t.op == TransformOp::reduce) {
        auto fn_it = record.find("fn");
        if (fn_it == record.end()) return bad("missing_key", "A reduce transform needs `fn`.", "fn");
        if (!fn_it->is_string()) return bad("bad_value", "`fn` is a string.", "fn");
        const std::string fn = fn_it->get<std::string>();
        bool found = false;
        for (ReduceFn f : kReduceFns) {
            if (fn == reduce_fn_name(f)) {
                t.fn = f;
                found = true;
            }
        }
        if (!found) {
            return bad("bad_value",
                       "`fn` is one of mean, sum, min, max, std, var, norm, argmax, argmin and entropy, not \"" + fn +
                           "\".",
                       "fn");
        }
        auto kd = record.find("keepdims");
        if (kd != record.end()) {
            if (!kd->is_boolean()) return bad("bad_value", "`keepdims` is true or false.", "keepdims");
            t.keepdims = kd->get<bool>();
        }
        if ((t.fn == ReduceFn::argmax || t.fn == ReduceFn::argmin) && t.axes.size() != 1) {
            return bad("bad_value", std::string("`") + reduce_fn_name(t.fn) + "` takes exactly one axis.", "axis");
        }
    } else {
        auto mode_it = record.find("mode");
        if (mode_it == record.end()) return bad("missing_key", "A normalise transform needs `mode`.", "mode");
        if (!mode_it->is_string()) return bad("bad_value", "`mode` is a string.", "mode");
        const std::string mode = mode_it->get<std::string>();
        bool found = false;
        for (NormaliseMode m : kModes) {
            if (mode == normalise_mode_name(m)) {
                t.mode = m;
                found = true;
            }
        }
        if (!found) {
            return bad("bad_value", "`mode` is one of minmax, zscore, softmax and l2, not \"" + mode + "\".", "mode");
        }
    }
    *out = std::move(t);
    return {};
}

json transform_to_json(const TensorTransform& t) {
    json j = json::object();
    auto put_axis = [&] {
        if (t.axes.empty()) return;
        if (t.axis_is_list || t.axes.size() != 1) j["axis"] = t.axes;
        else j["axis"] = t.axes[0];
    };
    switch (t.op) {
        case TransformOp::slice:
            j["op"] = "slice";
            put_axis();
            if (t.index) j["index"] = *t.index;
            if (t.start) j["start"] = *t.start;
            if (t.stop) j["stop"] = *t.stop;
            if (t.step) j["step"] = *t.step;
            break;
        case TransformOp::reduce:
            j["op"] = "reduce";
            put_axis();
            j["fn"] = reduce_fn_name(t.fn);
            if (t.keepdims) j["keepdims"] = true;
            break;
        case TransformOp::normalise:
            j["op"] = "normalise";
            j["mode"] = normalise_mode_name(t.mode);
            put_axis();
            break;
    }
    return j;
}

Error transform_result_shape(const std::vector<int64_t>& shape, DType dtype, const TensorTransform& t,
                             std::vector<int64_t>* out_shape, DType* out_dtype) {
    const int rank = static_cast<int>(shape.size());
    std::vector<int> axes;
    if (Error e = resolve_axes(t.axes, rank, &axes)) return e;
    switch (t.op) {
        case TransformOp::slice: {
            if (axes.size() != 1) return bad("bad_value", "A slice takes exactly one axis.", "axis");
            const int a = axes[0];
            const int64_t n = shape[a];
            std::vector<int64_t> s = shape;
            if (t.index) {
                const int64_t i = *t.index < 0 ? *t.index + n : *t.index;
                if (i < 0 || i >= n) {
                    return bad("index_out_of_range",
                               "Index " + std::to_string(*t.index) + " is outside an axis of size " + std::to_string(n) +
                                   ".",
                               "index");
                }
                s.erase(s.begin() + a);
            } else {
                int64_t start = 0, count = 0;
                slice_bounds(t, n, &start, &count);
                s[a] = count;
            }
            *out_shape = std::move(s);
            *out_dtype = dtype;
            return {};
        }
        case TransformOp::reduce: {
            if (axes.empty()) return bad("missing_key", "A reduce transform needs `axis`.", "axis");
            if (t.fn == ReduceFn::argmax || t.fn == ReduceFn::argmin) {
                if (axes.size() != 1) {
                    return bad("bad_value", std::string("`") + reduce_fn_name(t.fn) + "` takes exactly one axis.",
                               "axis");
                }
                if (shape[axes[0]] == 0) {
                    return bad("empty_reduce", std::string("`") + reduce_fn_name(t.fn) + "` over an empty axis has no answer.",
                               "axis");
                }
            }
            std::vector<int64_t> s;
            for (int i = 0; i < rank; ++i) {
                const bool reduced = std::find(axes.begin(), axes.end(), i) != axes.end();
                if (!reduced) s.push_back(shape[i]);
                else if (t.keepdims) s.push_back(1);
            }
            *out_shape = std::move(s);
            *out_dtype = (t.fn == ReduceFn::argmax || t.fn == ReduceFn::argmin) ? DType::i64 : float_result(dtype);
            return {};
        }
        case TransformOp::normalise:
            *out_shape = shape;
            *out_dtype = float_result(dtype);
            return {};
    }
    return {};
}

TensorResult apply_transform(const TensorData& in, const TensorTransform& t) {
    if (!in.valid()) return {bad("no_input", "The transform has no input tensor.", ""), {}};
    std::vector<int64_t> out_shape;
    DType out_dtype = in.dtype();
    if (Error e = transform_result_shape(in.shape(), in.dtype(), t, &out_shape, &out_dtype)) return {e, {}};
    TensorResult r = TensorData::allocate(out_dtype, out_shape);
    if (r.error) return r;
    uint8_t* out = r.tensor.mutable_data();
    std::vector<int> axes;
    resolve_axes(t.axes, in.rank(), &axes);
    switch (t.op) {
        case TransformOp::slice: {
            const int a = axes[0];
            const int64_t n = in.shape()[a];
            int64_t start = 0, count = 1, step = 1;
            if (t.index) {
                start = *t.index < 0 ? *t.index + n : *t.index;
            } else {
                slice_bounds(t, n, &start, &count);
                step = t.step.value_or(1);
            }
            uint64_t outer = 1, inner = dtype_size(in.dtype());
            for (int i = 0; i < a; ++i) outer *= static_cast<uint64_t>(in.shape()[i]);
            for (int i = a + 1; i < in.rank(); ++i) inner *= static_cast<uint64_t>(in.shape()[i]);
            if (!out || inner == 0 || count == 0) return r;
            const uint8_t* src = in.data();
            uint8_t* dst = out;
            for (uint64_t o = 0; o < outer; ++o) {
                for (int64_t k = 0; k < count; ++k) {
                    const uint64_t i = static_cast<uint64_t>(start + k * step);
                    std::memcpy(dst, src + (o * static_cast<uint64_t>(n) + i) * inner, inner);
                    dst += inner;
                }
            }
            return r;
        }
        case TransformOp::reduce: {
            const AxisSplit s = split_axes(in.shape(), axes);
            if (!out) return r;
            for (uint64_t g = 0; g < s.groups; ++g) {
                store(out, out_dtype, g, reduce_group(in, s, group_base(s, g), t.fn));
            }
            return r;
        }
        case TransformOp::normalise: {
            const std::vector<int> norm_axes = axes.empty() ? all_axes(in.rank()) : axes;
            const AxisSplit s = split_axes(in.shape(), norm_axes);
            if (!out) return r;
            for (uint64_t g = 0; g < s.groups; ++g) {
                normalise_group(in, s, group_base(s, g), t.mode, out, out_dtype);
            }
            return r;
        }
    }
    return r;
}

TensorResult apply_transforms(const TensorData& input, const std::vector<TensorTransform>& transforms) {
    TensorData current = input;
    for (std::size_t i = 0; i < transforms.size(); ++i) {
        TensorResult r = apply_transform(current, transforms[i]);
        if (r.error) {
            const std::string field = r.error.where;
            r.error.where = "transform[" + std::to_string(i) + "]" + (field.empty() ? "" : "." + field);
            return r;
        }
        current = std::move(r.tensor);
    }
    return {{}, current};
}

json TensorStats::to_json() const {
    auto opt = [](const std::optional<double>& v) -> json { return v ? json(*v) : json(nullptr); };
    return json{{"count", count}, {"nonfinite", nonfinite}, {"min", opt(min)},
                {"max", opt(max)},   {"mean", opt(mean)},       {"std", opt(std)}};
}

TensorStats tensor_stats(const TensorData& t) {
    TensorStats s;
    if (!t.valid()) return s;
    s.count = t.numel();
    double lo = 0, hi = 0, mean = 0, m2 = 0;
    uint64_t k = 0;
    for (uint64_t i = 0; i < t.numel(); ++i) {
        const double v = t.element(i);
        if (!std::isfinite(v)) {
            ++s.nonfinite;
            continue;
        }
        if (k == 0 || v < lo) lo = v;
        if (k == 0 || v > hi) hi = v;
        ++k;
        const double d = v - mean;
        mean += d / static_cast<double>(k);
        m2 += d * (v - mean);
    }
    if (k > 0) {
        s.min = lo;
        s.max = hi;
        s.mean = mean;
        s.std = std::sqrt(m2 / static_cast<double>(k));
    }
    return s;
}

json tensor_element_json(const TensorData& t, uint64_t i) {
    const uint8_t* p = t.data() + i * dtype_size(t.dtype());
    switch (t.dtype()) {
        case DType::boolean: return *p != 0;
        case DType::u64: { uint64_t v; std::memcpy(&v, p, 8); return v; }
        case DType::i64: { int64_t v; std::memcpy(&v, p, 8); return v; }
        case DType::f16:
        case DType::f32:
        case DType::f64: return float_json(t.element(i));
        default: return static_cast<int64_t>(t.element(i));
    }
}

json tensor_sample(const TensorData& t, int64_t per_axis, uint64_t max_values) {
    json out = json::object();
    if (!t.valid()) return out;
    if (per_axis < 1) per_axis = 1;
    if (max_values < 1) max_values = 1;
    const int rank = t.rank();
    std::vector<int64_t> k(static_cast<std::size_t>(rank));
    uint64_t total = 1;
    for (int i = 0; i < rank; ++i) {
        k[i] = std::min<int64_t>(t.shape()[i], per_axis);
        total *= static_cast<uint64_t>(k[i]);
    }
    for (int i = 0; i < rank && total > max_values; ++i) {
        if (k[i] <= 1) continue;
        if (i == rank - 1) {
            const uint64_t others = total / static_cast<uint64_t>(k[i]);
            k[i] = static_cast<int64_t>(std::max<uint64_t>(1, max_values / std::max<uint64_t>(1, others)));
        } else {
            k[i] = 1;
        }
        total = 1;
        for (int j = 0; j < rank; ++j) total *= static_cast<uint64_t>(k[j]);
    }
    const std::vector<uint64_t> strides = row_major_strides(t.shape());
    out["start"] = std::vector<int64_t>(static_cast<std::size_t>(rank), 0);
    out["shape"] = k;
    if (rank == 0) {
        out["values"] = tensor_element_json(t, 0);
        return out;
    }
    // Builds the nested arrays one axis at a time; the depth is at most kMaxTensorRank.
    struct Builder {
        const TensorData& t;
        const std::vector<int64_t>& k;
        const std::vector<uint64_t>& strides;
        json build(int axis, uint64_t offset) const {
            json arr = json::array();
            for (int64_t i = 0; i < k[axis]; ++i) {
                const uint64_t o = offset + static_cast<uint64_t>(i) * strides[axis];
                if (axis + 1 == static_cast<int>(k.size())) arr.push_back(tensor_element_json(t, o));
                else arr.push_back(build(axis + 1, o));
            }
            return arr;
        }
    };
    out["values"] = Builder{t, k, strides}.build(0, 0);
    return out;
}

}  // namespace caliper_builder
