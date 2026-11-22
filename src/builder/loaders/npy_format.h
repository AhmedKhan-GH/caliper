#pragma once
// The .npy format, shared by the npy and npz readers and by write_npy: the
// header (magic, version, a Python dict literal), the dtype string, and the
// conversion of big-endian or Fortran-order data into the catalog's layout.
// Internal to the loaders module.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/loaders/tensor_readers.h>

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_builder {
namespace detail {

struct NpyHeader {
    DType dtype = DType::f32;
    bool big_endian = false;
    bool fortran_order = false;
    std::vector<int64_t> shape;
    uint64_t data_offset = 0;  // from the start of the .npy data
    uint64_t nbytes = 0;       // of the array data
    // Whether the data can be used as it lies (little-endian, C order).
    bool in_catalog_layout() const { return !fortran_order && (!big_endian || dtype_size(dtype) == 1); }
};

// The length of the magic, version and header of an .npy file, from at least its
// first 12 bytes. Fails unless the magic and version are right and the length
// fits `available` and `limits.max_npy_header`.
Error npy_header_length(const uint8_t* prefix, uint64_t prefix_size, uint64_t available, const ReaderLimits& limits,
                        uint64_t* total, const std::string& where);

// Parses a complete header (magic through the dict). An object dtype is a pickle.
Error parse_npy_header(const uint8_t* bytes, uint64_t size, NpyHeader* out, const std::string& where);

// Converts array data (big-endian or Fortran order) into an owned tensor in the
// catalog's layout. `data` holds header.nbytes bytes.
TensorResult npy_convert(const uint8_t* data, const NpyHeader& header, const ReaderLimits& limits,
                         const std::string& where);

// The npy dtype string of a catalog dtype ("<f4", "|b1").
std::string npy_descr(DType dtype);

}  // namespace detail
}  // namespace caliper_builder
