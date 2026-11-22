// TensorData: owned and mapped storage, dtype helpers and shape checks.
#include <caliper_builder/loaders/tensor.h>

#include <cstring>
#include <limits>

#ifdef _WIN32
#include <cstdio>
#include <fstream>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace caliper_builder {

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::u8: return "u8";
        case DType::i8: return "i8";
        case DType::u16: return "u16";
        case DType::i16: return "i16";
        case DType::u32: return "u32";
        case DType::i32: return "i32";
        case DType::u64: return "u64";
        case DType::i64: return "i64";
        case DType::f16: return "f16";
        case DType::f32: return "f32";
        case DType::f64: return "f64";
        case DType::boolean: return "bool";
    }
    return "f32";
}

bool dtype_from_name(const std::string& name, DType* out) {
    static const DType all[] = {DType::u8,  DType::i8,  DType::u16, DType::i16, DType::u32, DType::i32,
                                DType::u64, DType::i64, DType::f16, DType::f32, DType::f64, DType::boolean};
    for (DType d : all) {
        if (name == dtype_name(d)) {
            *out = d;
            return true;
        }
    }
    return false;
}

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::u8:
        case DType::i8:
        case DType::boolean: return 1;
        case DType::u16:
        case DType::i16:
        case DType::f16: return 2;
        case DType::u32:
        case DType::i32:
        case DType::f32: return 4;
        case DType::u64:
        case DType::i64:
        case DType::f64: return 8;
    }
    return 1;
}

bool dtype_is_float(DType dtype) {
    return dtype == DType::f16 || dtype == DType::f32 || dtype == DType::f64;
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // A subnormal half is a normal float: shift the mantissa up until its leading bit leaves.
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            bits = sign | (exponent << 23) | (mantissa << 13);
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

float bfloat16_to_float(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

Error check_shape(const std::vector<int64_t>& shape, DType dtype, uint64_t* numel, uint64_t* nbytes) {
    if (shape.size() > static_cast<std::size_t>(kMaxTensorRank)) {
        return Error("tensor.rank", "A tensor has at most " + std::to_string(kMaxTensorRank) +
                                        " dimensions; this one has " + std::to_string(shape.size()) + ".");
    }
    uint64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) return Error("tensor.shape", "A tensor dimension is negative.");
        const uint64_t ud = static_cast<uint64_t>(d);
        if (ud != 0 && n > std::numeric_limits<uint64_t>::max() / ud) {
            return Error("tensor.too_large", "The tensor's element count overflows.");
        }
        n *= ud;
    }
    const uint64_t size = dtype_size(dtype);
    if (n > std::numeric_limits<uint64_t>::max() / size) {
        return Error("tensor.too_large", "The tensor's byte count overflows.");
    }
    if (numel) *numel = n;
    if (nbytes) *nbytes = n * size;
    return {};
}

struct TensorData::Storage {
    std::vector<uint8_t> bytes;
    void* map_base = nullptr;
    std::size_t map_length = 0;
    ~Storage() {
#ifndef _WIN32
        if (map_base) munmap(map_base, map_length);
#endif
    }
};

TensorData::TensorData() = default;

uint8_t* TensorData::mutable_data() {
    if (!storage_ || mapped_ || storage_->bytes.empty()) return nullptr;
    return storage_->bytes.data();
}

TensorResult TensorData::from_bytes(DType dtype, std::vector<int64_t> shape, std::vector<uint8_t> bytes) {
    TensorResult r;
    uint64_t numel = 0, nbytes = 0;
    if (Error e = check_shape(shape, dtype, &numel, &nbytes)) {
        r.error = e;
        return r;
    }
    if (bytes.size() != nbytes) {
        r.error = Error("tensor.size", "The byte count does not match the dtype and the shape.");
        return r;
    }
    TensorData t;
    t.storage_ = std::make_shared<Storage>();
    t.storage_->bytes = std::move(bytes);
    t.data_ = t.storage_->bytes.empty() ? nullptr : t.storage_->bytes.data();
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.numel_ = numel;
    t.nbytes_ = nbytes;
    t.valid_ = true;
    r.tensor = std::move(t);
    return r;
}

TensorResult TensorData::allocate(DType dtype, std::vector<int64_t> shape) {
    uint64_t numel = 0, nbytes = 0;
    if (Error e = check_shape(shape, dtype, &numel, &nbytes)) return {e, {}};
    if (nbytes > std::numeric_limits<std::size_t>::max()) {
        return {Error("tensor.too_large", "The tensor does not fit in memory."), {}};
    }
    std::vector<uint8_t> bytes;
    try {
        bytes.assign(static_cast<std::size_t>(nbytes), 0);
    } catch (...) {
        return {Error("tensor.too_large", "Memory for the tensor could not be allocated."), {}};
    }
    return from_bytes(dtype, std::move(shape), std::move(bytes));
}

TensorResult TensorData::map_file(const std::string& path, uint64_t offset, DType dtype, std::vector<int64_t> shape) {
    TensorResult r;
    uint64_t numel = 0, nbytes = 0;
    if (Error e = check_shape(shape, dtype, &numel, &nbytes)) {
        r.error = e;
        return r;
    }
    TensorData t;
    t.storage_ = std::make_shared<Storage>();
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.numel_ = numel;
    t.nbytes_ = nbytes;
    t.valid_ = true;
    t.mapped_ = true;
    t.file_ = path;
    t.file_offset_ = offset;
#ifdef _WIN32
    // Windows reads the bytes instead of mapping them; the file and offset are kept.
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        r.error = Error("loader.io", "The file could not be opened.", path);
        return r;
    }
    const uint64_t size = static_cast<uint64_t>(in.tellg());
    if (offset > size || nbytes > size - offset) {
        r.error = Error("loader.truncated", "The file is shorter than the tensor it should hold.", path);
        return r;
    }
    try {
        t.storage_->bytes.resize(static_cast<std::size_t>(nbytes));
    } catch (...) {
        r.error = Error("tensor.too_large", "Memory for the tensor could not be allocated.", path);
        return r;
    }
    in.seekg(static_cast<std::streamoff>(offset));
    if (nbytes && !in.read(reinterpret_cast<char*>(t.storage_->bytes.data()), static_cast<std::streamsize>(nbytes))) {
        r.error = Error("loader.io", "The file could not be read.", path);
        return r;
    }
    t.data_ = nbytes ? t.storage_->bytes.data() : nullptr;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        r.error = Error("loader.io", "The file could not be opened.", path);
        return r;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        r.error = Error("loader.io", "The path is not a regular file.", path);
        return r;
    }
    const uint64_t size = static_cast<uint64_t>(st.st_size);
    if (offset > size || nbytes > size - offset) {
        ::close(fd);
        r.error = Error("loader.truncated", "The file is shorter than the tensor it should hold.", path);
        return r;
    }
    if (nbytes > 0) {
        const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
        const uint64_t aligned = offset - offset % page;
        const uint64_t length = nbytes + (offset - aligned);
        if (length > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            r.error = Error("tensor.too_large", "The tensor is too large to map.", path);
            return r;
        }
        void* base = mmap(nullptr, static_cast<std::size_t>(length), PROT_READ, MAP_PRIVATE, fd,
                          static_cast<off_t>(aligned));
        if (base == MAP_FAILED) {
            ::close(fd);
            r.error = Error("loader.io", "The file could not be mapped.", path);
            return r;
        }
        t.storage_->map_base = base;
        t.storage_->map_length = static_cast<std::size_t>(length);
        t.data_ = static_cast<const uint8_t*>(base) + (offset - aligned);
    }
    ::close(fd);
#endif
    r.tensor = std::move(t);
    return r;
}

std::optional<double> TensorData::element_at(int64_t i) const {
    if (i < 0 || static_cast<uint64_t>(i) >= numel_ || !data_) return std::nullopt;
    return element(static_cast<uint64_t>(i));
}

std::optional<double> TensorData::element_at(int64_t row, int64_t column) const {
    if (shape_.size() != 2 || row < 0 || column < 0 || row >= shape_[0] || column >= shape_[1]) return std::nullopt;
    return element_at(row * shape_[1] + column);
}

double TensorData::element(uint64_t i) const {
    const uint8_t* p = data_ + i * dtype_size(dtype_);
    switch (dtype_) {
        case DType::u8: return static_cast<double>(*p);
        case DType::i8: return static_cast<double>(static_cast<int8_t>(*p));
        case DType::boolean: return *p ? 1.0 : 0.0;
        case DType::u16: { uint16_t v; std::memcpy(&v, p, 2); return v; }
        case DType::i16: { int16_t v; std::memcpy(&v, p, 2); return v; }
        case DType::f16: { uint16_t v; std::memcpy(&v, p, 2); return half_to_float(v); }
        case DType::u32: { uint32_t v; std::memcpy(&v, p, 4); return v; }
        case DType::i32: { int32_t v; std::memcpy(&v, p, 4); return v; }
        case DType::f32: { float v; std::memcpy(&v, p, 4); return v; }
        case DType::u64: { uint64_t v; std::memcpy(&v, p, 8); return static_cast<double>(v); }
        case DType::i64: { int64_t v; std::memcpy(&v, p, 8); return static_cast<double>(v); }
        case DType::f64: { double v; std::memcpy(&v, p, 8); return v; }
    }
    return 0.0;
}

}  // namespace caliper_builder
