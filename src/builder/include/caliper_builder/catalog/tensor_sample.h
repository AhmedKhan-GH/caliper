#pragma once
// sample_tensor(): the statistics and the slice of a tensor for
// catalog.sample, computed from its dtype, its shape and its bytes
// (little-endian, row-major, contiguous), by the rules of docs/design/catalog.md:
// exact statistics up to 2^26 elements and a stride beyond; a slice of at most
// 64 elements per axis and 4096 in all; non-finite values as "nan", "inf" and
// "-inf"; f16 decoded exactly.
//
// The buffer's size must be exactly the element count times the dtype's size;
// otherwise the answer is catalog.bad_tensor and nothing is read, so a
// provider can hand it bytes mapped from a file. A bad start or count is
// catalog.bad_request. No function here throws.
#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/error.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace caliper_builder {

struct TensorSampleResult {
    CatalogTensorStats stats;
    CatalogSlice slice;
    Error error;
};

// The default size of the slice on each axis: as many elements as fit the
// limits (64 for rank 1 or 2, 16, 8, 5, 4, 3 and 2 for ranks 3 to 8), or the
// axis's size when that is smaller.
std::vector<std::int64_t> default_slice_count(const std::vector<std::int64_t>& shape);

TensorSampleResult sample_tensor(CatalogDtype dtype, const std::vector<std::int64_t>& shape, const void* data,
                                 std::size_t size, const std::vector<std::int64_t>& start = {},
                                 const std::vector<std::int64_t>& count = {});

// Decodes an IEEE 754 half-precision value.
float catalog_half_to_float(std::uint16_t half);

}  // namespace caliper_builder
