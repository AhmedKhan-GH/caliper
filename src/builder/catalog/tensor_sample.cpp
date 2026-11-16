#include <caliper_builder/catalog/tensor_sample.h>

#include <cmath>
#include <cstring>
#include <limits>

namespace caliper_builder {

namespace {

using nlohmann::json;

std::uint64_t load_le(const unsigned char* p, std::size_t n) {
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

// One element as a double, for the statistics.
double element_as_double(CatalogDtype dtype, const unsigned char* p) {
    switch (dtype) {
        case CatalogDtype::u8: return p[0];
        case CatalogDtype::i8: return static_cast<std::int8_t>(p[0]);
        case CatalogDtype::u16: return static_cast<std::uint16_t>(load_le(p, 2));
        case CatalogDtype::i16: return static_cast<std::int16_t>(static_cast<std::uint16_t>(load_le(p, 2)));
        case CatalogDtype::u32: return static_cast<std::uint32_t>(load_le(p, 4));
        case CatalogDtype::i32: return static_cast<std::int32_t>(static_cast<std::uint32_t>(load_le(p, 4)));
        case CatalogDtype::u64: return static_cast<double>(load_le(p, 8));
        case CatalogDtype::i64: return static_cast<double>(static_cast<std::int64_t>(load_le(p, 8)));
        case CatalogDtype::f16: return catalog_half_to_float(static_cast<std::uint16_t>(load_le(p, 2)));
        case CatalogDtype::f32: {
            const auto bits = static_cast<std::uint32_t>(load_le(p, 4));
            float f;
            std::memcpy(&f, &bits, sizeof f);
            return f;
        }
        case CatalogDtype::f64: {
            const std::uint64_t bits = load_le(p, 8);
            double d;
            std::memcpy(&d, &bits, sizeof d);
            return d;
        }
        case CatalogDtype::boolean: return p[0] != 0 ? 1.0 : 0.0;
    }
    return 0.0;
}

json float_value(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
    return d;
}

// One element as JSON, exact for every integer type.
json element_as_json(CatalogDtype dtype, const unsigned char* p) {
    switch (dtype) {
        case CatalogDtype::u64: return load_le(p, 8);
        case CatalogDtype::i64: return static_cast<std::int64_t>(load_le(p, 8));
        case CatalogDtype::boolean: return p[0] != 0;
        case CatalogDtype::f16:
        case CatalogDtype::f32:
        case CatalogDtype::f64: return float_value(element_as_double(dtype, p));
        default: return static_cast<std::int64_t>(element_as_double(dtype, p));
    }
}

}  // namespace

float catalog_half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {  // subnormal: normalise it
            int e = -1;
            do {
                ++e;
                mantissa <<= 1;
            } while ((mantissa & 0x400u) == 0);
            mantissa &= 0x3FFu;
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

std::vector<std::int64_t> default_slice_count(const std::vector<std::int64_t>& shape) {
    static constexpr std::int64_t kPerRank[] = {1, 64, 64, 16, 8, 5, 4, 3, 2};
    const std::size_t rank = shape.size();
    const std::int64_t per_axis = rank < sizeof kPerRank / sizeof kPerRank[0] ? kPerRank[rank] : 1;
    std::vector<std::int64_t> count(rank);
    for (std::size_t i = 0; i < rank; ++i) count[i] = shape[i] < per_axis ? shape[i] : per_axis;
    return count;
}

TensorSampleResult sample_tensor(CatalogDtype dtype, const std::vector<std::int64_t>& shape, const void* data,
                                 std::size_t size, const std::vector<std::int64_t>& start_in,
                                 const std::vector<std::int64_t>& count_in) {
    TensorSampleResult out;
    const std::size_t dsize = catalog_dtype_size(dtype);
    if (dsize == 0) {
        out.error = Error("catalog.bad_tensor", "The dtype is not a known dtype.", "dtype");
        return out;
    }
    if (shape.size() > CatalogLimits::max_rank) {
        out.error = Error("catalog.bad_tensor", "A tensor has at most 8 dimensions.", "shape");
        return out;
    }
    // The element count, refusing a negative size and an overflow.
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            out.error = Error("catalog.bad_tensor", "A size of the shape is negative.", "shape");
            return out;
        }
        const auto d = static_cast<std::uint64_t>(shape[i]);
        if (d != 0 && elements > static_cast<std::uint64_t>(CatalogLimits::max_elements) / d) {
            out.error = Error("catalog.bad_tensor", "The shape holds more than 2^62 elements.", "shape");
            return out;
        }
        elements *= d;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / dsize || elements * dsize != size) {
        out.error = Error("catalog.bad_tensor",
                          "The buffer holds " + std::to_string(size) + " bytes, and the shape and dtype need " +
                              std::to_string(elements) + " elements of " + std::to_string(dsize) + " bytes.",
                          "bytes");
        return out;
    }
    if (size != 0 && data == nullptr) {
        out.error = Error("catalog.bad_tensor", "The buffer is null.", "bytes");
        return out;
    }

    CatalogSampleRequest request;
    request.start = start_in;
    request.count = count_in;
    if (Error err = check_catalog_sample_request(request, &shape)) {
        out.error = std::move(err);
        return out;
    }
    const std::size_t rank = shape.size();
    std::vector<std::int64_t> start = start_in.empty() ? std::vector<std::int64_t>(rank, 0) : start_in;
    std::vector<std::int64_t> count = count_in.empty() ? default_slice_count(shape) : count_in;

    const auto* bytes = static_cast<const unsigned char*>(data);

    // Statistics, with a stride beyond 2^26 elements.
    CatalogTensorStats& st = out.stats;
    st.count = elements;
    const std::uint64_t cap = CatalogSampleLimits::max_stats_elements;
    const std::uint64_t stride = elements <= cap ? 1 : (elements + cap - 1) / cap;
    double mean = 0, m2 = 0, lo = 0, hi = 0;
    std::uint64_t finite = 0;
    for (std::uint64_t i = 0; i < elements; i += stride) {
        ++st.sampled;
        const double v = element_as_double(dtype, bytes + i * dsize);
        if (!std::isfinite(v)) {
            ++st.nonfinite;
            continue;
        }
        ++finite;
        if (finite == 1) {
            lo = hi = v;
        } else {
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        const double delta = v - mean;
        mean += delta / static_cast<double>(finite);
        m2 += delta * (v - mean);
    }
    st.exact = st.sampled == st.count;
    if (finite > 0) {
        st.min = lo;
        st.max = hi;
        st.mean = mean;
        st.std = std::sqrt(m2 / static_cast<double>(finite));
    }

    // The slice, in row-major order.
    CatalogSlice& sl = out.slice;
    sl.start = start;
    sl.shape = count;
    sl.values = json::array();
    std::int64_t box = 1;
    for (const auto c : count) box *= c;
    if (box == 0 || elements == 0) return out;
    std::vector<std::uint64_t> strides(rank, 1);
    for (std::size_t i = rank; i-- > 1;) strides[i - 1] = strides[i] * static_cast<std::uint64_t>(shape[i]);
    std::vector<std::int64_t> idx(rank, 0);
    for (std::int64_t n = 0; n < box; ++n) {
        std::uint64_t offset = 0;
        for (std::size_t a = 0; a < rank; ++a) offset += static_cast<std::uint64_t>(start[a] + idx[a]) * strides[a];
        sl.values.push_back(element_as_json(dtype, bytes + offset * dsize));
        for (std::size_t a = rank; a-- > 0;) {
            if (++idx[a] < count[a]) break;
            idx[a] = 0;
        }
    }
    return out;
}

}  // namespace caliper_builder
