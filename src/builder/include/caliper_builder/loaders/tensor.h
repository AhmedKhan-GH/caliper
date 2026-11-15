#pragma once
// TensorData: a dtype, a shape and the bytes of a tensor, contiguous, row-major
// and little-endian, either owned in memory or mapped from a file at an offset.
// It is the value that the tensor readers produce, that the tensor transforms
// take and return, and that a tensor source of the catalog points to.
//
// A TensorData is cheap to copy: copies share one immutable block of bytes,
// which a mapping keeps mapped until the last copy goes. A tensor has at most
// kMaxTensorRank dimensions, every dimension is >= 0, and a tensor of rank 0 is
// a scalar with one element. Element bytes may be unaligned (a safetensors
// buffer gives no alignment), so code that reads them uses element() or memcpy.
//
// On Windows the file is read into memory instead of mapped; the tensor still
// records its file and offset. No function here throws.
#include <caliper_builder/error.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_builder {

// The dtypes of the catalog: u8 i8 u16 i16 u32 i32 u64 i64 f16 f32 f64 bool.
enum class DType : uint8_t { u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, boolean };

constexpr int kMaxTensorRank = 8;

// The catalog name of a dtype ("f32", "bool").
const char* dtype_name(DType dtype);
// Parses a catalog name; returns false for anything else.
bool dtype_from_name(const std::string& name, DType* out);
// The size of one element in bytes.
std::size_t dtype_size(DType dtype);
bool dtype_is_float(DType dtype);

// IEEE 754 half precision to single precision, exact for every input.
float half_to_float(uint16_t bits);
// bfloat16 to single precision, exact for every input.
float bfloat16_to_float(uint16_t bits);

// Checks a shape: rank at most kMaxTensorRank, no negative dimension, and no
// overflow of the element count or the byte count. Fills `numel` and `nbytes`.
Error check_shape(const std::vector<int64_t>& shape, DType dtype, uint64_t* numel, uint64_t* nbytes);

struct TensorResult;

class TensorData {
public:
    TensorData();

    // An owned tensor that takes `bytes`; fails unless their size matches the shape.
    static TensorResult from_bytes(DType dtype, std::vector<int64_t> shape, std::vector<uint8_t> bytes);
    // An owned tensor of zero bytes, which the caller fills through mutable_data().
    static TensorResult allocate(DType dtype, std::vector<int64_t> shape);
    // A tensor mapped from `path` at `offset`; fails unless the file holds all of its bytes.
    static TensorResult map_file(const std::string& path, uint64_t offset, DType dtype, std::vector<int64_t> shape);

    DType dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    int rank() const { return static_cast<int>(shape_.size()); }
    uint64_t numel() const { return numel_; }
    uint64_t nbytes() const { return nbytes_; }
    // False for a default-constructed TensorData, which has no storage.
    bool valid() const { return valid_; }

    // The bytes; nullptr when the tensor has no elements.
    const uint8_t* data() const { return data_; }
    // The bytes of an owned tensor, for its maker to fill; nullptr for a mapped one.
    // Copies share storage, so fill a tensor before handing out copies of it.
    uint8_t* mutable_data();

    bool is_mapped() const { return mapped_; }
    // The file and offset of a mapped tensor (also kept on Windows, where it is read).
    const std::string& file() const { return file_; }
    uint64_t file_offset() const { return file_offset_; }

    // Element `i` (row-major) as a double; bool is 0 or 1. No bounds check.
    double element(uint64_t i) const;

    struct Storage;

private:
    std::shared_ptr<Storage> storage_;
    const uint8_t* data_ = nullptr;
    DType dtype_ = DType::f32;
    std::vector<int64_t> shape_;
    uint64_t numel_ = 0;
    uint64_t nbytes_ = 0;
    bool valid_ = false;
    bool mapped_ = false;
    std::string file_;
    uint64_t file_offset_ = 0;
};

struct TensorResult {
    Error error;
    TensorData tensor;
};

}  // namespace caliper_builder
