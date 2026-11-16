// The .npy reader and writer.
#include <caliper_builder/loaders/tensor_readers.h>

#include "file_io.h"
#include "npy_format.h"
#include "refusals.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace caliper_builder {
namespace detail {
namespace {

const uint8_t kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

Error malformed(const std::string& where, const std::string& what) {
    return Error("loader.malformed", "The .npy header is malformed: " + what + ".", where);
}

// A reader of the restricted Python literal that numpy writes as the header.
class Literal {
public:
    Literal(const char* p, const char* end) : p_(p), end_(end) {}

    void ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }
    bool eat(char c) {
        ws();
        if (p_ < end_ && *p_ == c) {
            ++p_;
            return true;
        }
        return false;
    }
    bool peek(char c) {
        ws();
        return p_ < end_ && *p_ == c;
    }
    bool at_end() {
        ws();
        return p_ == end_;
    }
    bool string(std::string* out) {
        ws();
        if (p_ >= end_ || (*p_ != '\'' && *p_ != '"')) return false;
        const char quote = *p_++;
        const char* start = p_;
        while (p_ < end_ && *p_ != quote) {
            if (*p_ == '\\') return false;  // numpy never escapes; refuse rather than interpret
            ++p_;
        }
        if (p_ >= end_) return false;
        out->assign(start, p_);
        ++p_;
        return true;
    }
    bool word(const char* w) {
        ws();
        const std::size_t n = std::strlen(w);
        if (static_cast<std::size_t>(end_ - p_) < n || std::strncmp(p_, w, n) != 0) return false;
        p_ += n;
        return true;
    }
    bool integer(int64_t* out) {
        ws();
        if (p_ >= end_ || *p_ < '0' || *p_ > '9') return false;
        uint64_t v = 0;
        while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
            const uint64_t d = static_cast<uint64_t>(*p_ - '0');
            if (v > (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - d) / 10) return false;
            v = v * 10 + d;
            ++p_;
        }
        if (p_ < end_ && *p_ == 'L') ++p_;  // Python 2 long literals
        *out = static_cast<int64_t>(v);
        return true;
    }

private:
    const char* p_;
    const char* end_;
};

Error parse_descr(const std::string& d, NpyHeader* h, const std::string& where) {
    if (d.empty()) return malformed(where, "the dtype is empty");
    std::size_t i = 0;
    char order = '=';
    if (d[0] == '<' || d[0] == '>' || d[0] == '|' || d[0] == '=') {
        order = d[0];
        i = 1;
    }
    if (i >= d.size()) return malformed(where, "the dtype has no kind");
    const char kind = d[i++];
    if (kind == 'O') return pickle_refused(where, "an .npy array with the object dtype");
    const std::string size_text = d.substr(i);
    int size = 0;
    if (size_text.empty() || size_text.size() > 3) {
        return Error("loader.unsupported", "The .npy dtype '" + d + "' is not supported.", where);
    }
    for (char c : size_text) {
        if (c < '0' || c > '9') return Error("loader.unsupported", "The .npy dtype '" + d + "' is not supported.", where);
        size = size * 10 + (c - '0');
    }
    DType dtype;
    bool ok = true;
    switch (kind) {
        case 'b': ok = size == 1; dtype = DType::boolean; break;
        case 'i':
            dtype = size == 1 ? DType::i8 : size == 2 ? DType::i16 : size == 4 ? DType::i32 : DType::i64;
            ok = size == 1 || size == 2 || size == 4 || size == 8;
            break;
        case 'u':
            dtype = size == 1 ? DType::u8 : size == 2 ? DType::u16 : size == 4 ? DType::u32 : DType::u64;
            ok = size == 1 || size == 2 || size == 4 || size == 8;
            break;
        case 'f':
            dtype = size == 2 ? DType::f16 : size == 4 ? DType::f32 : DType::f64;
            ok = size == 2 || size == 4 || size == 8;
            break;
        default: ok = false; dtype = DType::u8; break;
    }
    if (!ok) return Error("loader.unsupported", "The .npy dtype '" + d + "' is not supported.", where);
    h->dtype = dtype;
    h->big_endian = order == '>';
    return {};
}

}  // namespace

Error npy_header_length(const uint8_t* prefix, uint64_t prefix_size, uint64_t available, const ReaderLimits& limits,
                        uint64_t* total, const std::string& where) {
    if (prefix_size < 10 || std::memcmp(prefix, kMagic, 6) != 0) {
        return Error("loader.malformed", "The data does not start with the .npy magic bytes.", where);
    }
    const uint8_t major = prefix[6];
    uint64_t header_len = 0, start = 0;
    if (major == 1) {
        header_len = le16(prefix + 8);
        start = 10;
    } else if (major == 2 || major == 3) {
        if (prefix_size < 12) return Error("loader.truncated", "The .npy header is cut short.", where);
        header_len = le32(prefix + 8);
        start = 12;
    } else {
        return Error("loader.unsupported", "The .npy format version " + std::to_string(major) + " is not supported.",
                     where);
    }
    if (header_len > limits.max_npy_header) {
        return Error("loader.too_large", "The .npy header is larger than the reader's limit.", where);
    }
    if (!fits(start, header_len, available)) {
        return Error("loader.truncated", "The .npy header runs past the end of the data.", where);
    }
    *total = start + header_len;
    return {};
}

Error parse_npy_header(const uint8_t* bytes, uint64_t size, NpyHeader* out, const std::string& where) {
    const uint64_t start = bytes[6] == 1 ? 10 : 12;
    Literal lit(reinterpret_cast<const char*>(bytes + start), reinterpret_cast<const char*>(bytes + size));
    NpyHeader h;
    bool have_descr = false, have_order = false, have_shape = false;
    if (!lit.eat('{')) return malformed(where, "it is not a dict");
    while (!lit.eat('}')) {
        std::string key;
        if (!lit.string(&key)) return malformed(where, "a key is not a string");
        if (!lit.eat(':')) return malformed(where, "a key has no value");
        if (key == "descr") {
            if (have_descr) return malformed(where, "'descr' appears twice");
            if (lit.peek('[')) return Error("loader.unsupported", "Structured .npy dtypes are not supported.", where);
            std::string descr;
            if (!lit.string(&descr)) return malformed(where, "'descr' is not a string");
            if (Error e = parse_descr(descr, &h, where)) return e;
            have_descr = true;
        } else if (key == "fortran_order") {
            if (have_order) return malformed(where, "'fortran_order' appears twice");
            if (lit.word("True")) h.fortran_order = true;
            else if (lit.word("False")) h.fortran_order = false;
            else return malformed(where, "'fortran_order' is not True or False");
            have_order = true;
        } else if (key == "shape") {
            if (have_shape) return malformed(where, "'shape' appears twice");
            if (!lit.eat('(')) return malformed(where, "'shape' is not a tuple");
            while (!lit.eat(')')) {
                int64_t d = 0;
                if (!lit.integer(&d)) return malformed(where, "'shape' holds something other than sizes");
                if (h.shape.size() >= static_cast<std::size_t>(kMaxTensorRank)) {
                    return Error("tensor.rank", "A tensor has at most 8 dimensions.", where);
                }
                h.shape.push_back(d);
                if (!lit.eat(',') && !lit.peek(')')) return malformed(where, "'shape' is not a tuple of sizes");
            }
            have_shape = true;
        } else {
            return malformed(where, "the key '" + key + "' is unknown");
        }
        if (!lit.eat(',') && !lit.peek('}')) return malformed(where, "the dict is not closed");
    }
    if (!lit.at_end()) return malformed(where, "text follows the dict");
    if (!have_descr || !have_order || !have_shape) return malformed(where, "it lacks descr, fortran_order or shape");
    uint64_t numel = 0;
    if (Error e = check_shape(h.shape, h.dtype, &numel, &h.nbytes)) {
        e.where = where;
        return e;
    }
    h.data_offset = size;
    *out = std::move(h);
    return {};
}

TensorResult npy_convert(const uint8_t* data, const NpyHeader& h, const ReaderLimits& limits,
                         const std::string& where) {
    if (h.nbytes > limits.max_derived_bytes) {
        return {Error("loader.too_large", "The converted tensor would exceed the reader's limit.", where), {}};
    }
    TensorResult r = TensorData::allocate(h.dtype, h.shape);
    if (r.error) {
        r.error.where = where;
        return r;
    }
    uint8_t* out = r.tensor.mutable_data();
    if (!out) return r;
    const std::size_t item = dtype_size(h.dtype);
    const bool swap = h.big_endian && item > 1;
    auto copy_item = [&](uint8_t* dst, const uint8_t* src) {
        if (swap) {
            for (std::size_t b = 0; b < item; ++b) dst[b] = src[item - 1 - b];
        } else {
            std::memcpy(dst, src, item);
        }
    };
    const uint64_t numel = r.tensor.numel();
    if (!h.fortran_order) {
        for (uint64_t i = 0; i < numel; ++i) copy_item(out + i * item, data + i * item);
        return r;
    }
    // Fortran order: the source runs with the first axis fastest. Walk it in that
    // order and place each element at its row-major position.
    const std::size_t rank = h.shape.size();
    std::vector<uint64_t> row_stride(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        row_stride[i] = row_stride[i + 1] * static_cast<uint64_t>(h.shape[i + 1]);
    }
    std::vector<uint64_t> idx(rank, 0);
    uint64_t dst = 0;
    for (uint64_t i = 0; i < numel; ++i) {
        copy_item(out + dst * item, data + i * item);
        for (std::size_t a = 0; a < rank; ++a) {
            if (++idx[a] < static_cast<uint64_t>(h.shape[a])) {
                dst += row_stride[a];
                break;
            }
            dst -= (idx[a] - 1) * row_stride[a];
            idx[a] = 0;
        }
    }
    return r;
}

std::string npy_descr(DType dtype) {
    switch (dtype) {
        case DType::boolean: return "|b1";
        case DType::u8: return "|u1";
        case DType::i8: return "|i1";
        case DType::u16: return "<u2";
        case DType::i16: return "<i2";
        case DType::f16: return "<f2";
        case DType::u32: return "<u4";
        case DType::i32: return "<i4";
        case DType::f32: return "<f4";
        case DType::u64: return "<u8";
        case DType::i64: return "<i8";
        case DType::f64: return "<f8";
    }
    return "|u1";
}

}  // namespace detail

using namespace detail;

TensorReadResult read_npy(const std::string& path, const ReaderLimits& limits) {
    TensorReadResult result;
    InputFile f;
    if ((result.error = f.open(path))) return result;
    uint8_t prefix[12] = {};
    const uint64_t prefix_size = f.size() < 12 ? f.size() : 12;
    if ((result.error = f.read(0, prefix_size, prefix))) return result;
    uint64_t header_total = 0;
    if ((result.error = npy_header_length(prefix, prefix_size, f.size(), limits, &header_total, path))) return result;
    std::vector<uint8_t> header;
    if ((result.error = f.read_vec(0, header_total, &header, limits.max_npy_header + 12))) return result;
    NpyHeader h;
    if ((result.error = parse_npy_header(header.data(), header.size(), &h, path))) return result;
    if (!fits(h.data_offset, h.nbytes, f.size())) {
        result.error = Error("loader.truncated", "The .npy file is shorter than the array its header describes.", path);
        return result;
    }
    ReadTensor t;
    t.name = file_stem(path);
    const uint64_t trailing = f.size() - h.data_offset - h.nbytes;
    if (trailing > 0) {
        result.warnings.push_back(std::to_string(trailing) + " bytes after the array were ignored.");
    }
    if (h.in_catalog_layout()) {
        TensorResult m = TensorData::map_file(path, h.data_offset, h.dtype, h.shape);
        if ((result.error = m.error)) return result;
        t.data = std::move(m.tensor);
    } else {
        std::vector<uint8_t> bytes;
        if ((result.error = f.read_vec(h.data_offset, h.nbytes, &bytes, limits.max_derived_bytes))) return result;
        TensorResult c = npy_convert(bytes.data(), h, limits, path);
        if ((result.error = c.error)) return result;
        t.data = std::move(c.tensor);
        t.derived = true;
        t.warnings.push_back(h.fortran_order ? "converted from Fortran order to row-major order"
                                             : "converted from big-endian to little-endian bytes");
    }
    result.tensors.push_back(std::move(t));
    return result;
}

Error write_npy(const std::string& path, const TensorData& tensor, uint64_t* data_offset) {
    if (!tensor.valid()) return Error("loader.io", "There is no tensor to write.", path);
    std::string dict = "{'descr': '" + npy_descr(tensor.dtype()) + "', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < tensor.shape().size(); ++i) {
        if (i) dict += ", ";
        dict += std::to_string(tensor.shape()[i]);
    }
    if (tensor.shape().size() == 1) dict += ",";
    dict += "), }";
    std::size_t total = 10 + dict.size() + 1;
    const std::size_t padded = (total + 63) / 64 * 64;
    dict.append(padded - total, ' ');
    dict += '\n';
    std::vector<uint8_t> head(kMagic, kMagic + 6);
    head.push_back(1);
    head.push_back(0);
    const uint16_t len = static_cast<uint16_t>(dict.size());
    head.push_back(static_cast<uint8_t>(len & 0xFF));
    head.push_back(static_cast<uint8_t>(len >> 8));
    head.insert(head.end(), dict.begin(), dict.end());
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
    if (!out) return Error("loader.io", "The tensor file could not be created.", path);
    out.write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
    if (tensor.nbytes() > 0) {
        out.write(reinterpret_cast<const char*>(tensor.data()), static_cast<std::streamsize>(tensor.nbytes()));
    }
    out.close();
    if (!out) return Error("loader.io", "The tensor file could not be written.", path);
    if (data_offset) *data_offset = head.size();
    return {};
}

}  // namespace caliper_builder
