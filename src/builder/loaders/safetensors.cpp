// The safetensors reader: an 8-byte header length, a JSON header and a byte
// buffer that the tensors index without holes or overlaps.
#include <caliper_builder/loaders/tensor_readers.h>

#include "file_io.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <limits>

namespace caliper_builder {

using namespace detail;
using json = nlohmann::json;

namespace {

Error malformed(const std::string& where, const std::string& what) {
    return Error("loader.malformed", "The safetensors header is malformed: " + what + ".", where);
}

struct Entry {
    std::string name;
    std::string dtype_text;
    bool known = false;
    bool bf16 = false;
    DType dtype = DType::u8;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;
};

bool map_dtype(const std::string& s, Entry* e) {
    static const std::pair<const char*, DType> table[] = {
        {"BOOL", DType::boolean}, {"U8", DType::u8},   {"I8", DType::i8},   {"U16", DType::u16},
        {"I16", DType::i16},      {"U32", DType::u32}, {"I32", DType::i32}, {"U64", DType::u64},
        {"I64", DType::i64},      {"F16", DType::f16}, {"F32", DType::f32}, {"F64", DType::f64}};
    for (const auto& p : table) {
        if (s == p.first) {
            e->dtype = p.second;
            e->known = true;
            return true;
        }
    }
    if (s == "BF16") {
        e->bf16 = true;
        return true;
    }
    return false;
}

// The element size of dtypes the reader can check but does not load.
uint64_t other_item_size(const std::string& s) {
    if (s.rfind("F8_", 0) == 0) return 1;
    return 0;
}

bool json_u64(const json& v, uint64_t* out) {
    if (v.is_number_unsigned()) {
        *out = v.get<uint64_t>();
        return true;
    }
    if (v.is_number_integer()) {
        const int64_t i = v.get<int64_t>();
        if (i < 0) return false;
        *out = static_cast<uint64_t>(i);
        return true;
    }
    return false;
}

}  // namespace

TensorReadResult read_safetensors(const std::string& path, const TensorFilter& filter, const ReaderLimits& limits) {
    TensorReadResult result;
    InputFile f;
    if ((result.error = f.open(path))) return result;
    uint8_t len_bytes[8];
    if (f.size() < 8) {
        result.error = Error("loader.truncated", "The file is too short to be a safetensors file.", path);
        return result;
    }
    if ((result.error = f.read(0, 8, len_bytes))) return result;
    const uint64_t header_len = le64(len_bytes);
    if (header_len > limits.max_safetensors_header) {
        result.error = Error("loader.too_large", "The safetensors header is larger than the reader's limit.", path);
        return result;
    }
    if (!fits(8, header_len, f.size())) {
        result.error = Error("loader.truncated", "The safetensors header runs past the end of the file.", path);
        return result;
    }
    std::vector<uint8_t> header;
    if ((result.error = f.read_vec(8, header_len, &header, limits.max_safetensors_header))) return result;
    if (header.empty() || header[0] != '{') {
        result.error = malformed(path, "it does not start with '{'");
        return result;
    }
    if (json_depth(header.data(), header.size(), limits.max_json_depth) > limits.max_json_depth) {
        result.error = malformed(path, "it is nested too deeply");
        return result;
    }
    const json doc = json::parse(header.begin(), header.end(), nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        result.error = malformed(path, "it is not a JSON object");
        return result;
    }
    const uint64_t data_start = 8 + header_len;
    const uint64_t data_size = f.size() - data_start;
    std::vector<Entry> entries;
    for (auto it = doc.begin(); it != doc.end(); ++it) {
        const std::string& key = it.key();
        const json& v = it.value();
        if (key == "__metadata__") {
            if (!v.is_object()) {
                result.error = malformed(path, "__metadata__ is not an object");
                return result;
            }
            for (auto m = v.begin(); m != v.end(); ++m) {
                if (!m.value().is_string()) {
                    result.error = malformed(path, "__metadata__ holds a value that is not a string");
                    return result;
                }
            }
            continue;
        }
        if (entries.size() >= limits.max_tensors) {
            result.error = Error("loader.too_large", "The file holds more tensors than the reader's limit.", path);
            return result;
        }
        if (!v.is_object() || v.size() != 3 || !v.contains("dtype") || !v.contains("shape") ||
            !v.contains("data_offsets")) {
            result.error = malformed(path, "the entry of " + key + " is not {dtype, shape, data_offsets}");
            return result;
        }
        Entry e;
        e.name = key;
        const json& dt = v["dtype"];
        const json& shape = v["shape"];
        const json& offsets = v["data_offsets"];
        if (!dt.is_string() || !shape.is_array() || !offsets.is_array() || offsets.size() != 2) {
            result.error = malformed(path, "the entry of " + key + " has values of the wrong type");
            return result;
        }
        e.dtype_text = dt.get<std::string>();
        if (shape.size() > static_cast<std::size_t>(kMaxTensorRank)) {
            result.error = Error("tensor.rank", "The tensor " + key + " has more than 8 dimensions.", path);
            return result;
        }
        for (const json& d : shape) {
            uint64_t u = 0;
            if (!json_u64(d, &u) || u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                result.error = malformed(path, "the shape of " + key + " is not a list of sizes");
                return result;
            }
            e.shape.push_back(static_cast<int64_t>(u));
        }
        if (!json_u64(offsets[0], &e.begin) || !json_u64(offsets[1], &e.end) || e.begin > e.end) {
            result.error = malformed(path, "the data offsets of " + key + " are invalid");
            return result;
        }
        if (e.end > data_size) {
            result.error = Error("loader.truncated", "The tensor " + key + " lies past the end of the file.", path);
            return result;
        }
        map_dtype(e.dtype_text, &e);
        uint64_t item = e.known ? dtype_size(e.dtype) : e.bf16 ? 2 : other_item_size(e.dtype_text);
        uint64_t numel = 0;
        if (Error err = check_shape(e.shape, DType::u8, &numel, nullptr)) {
            err.where = path;
            result.error = err;
            return result;
        }
        if (item > 0) {
            if (numel > std::numeric_limits<uint64_t>::max() / item || numel * item != e.end - e.begin) {
                result.error = malformed(path, "the byte range of " + key + " does not match its dtype and shape");
                return result;
            }
        }
        entries.push_back(std::move(e));
    }
    // The tensors must index the buffer exactly: no holes, no overlaps, nothing left over.
    std::vector<const Entry*> sorted;
    for (const Entry& e : entries) sorted.push_back(&e);
    std::sort(sorted.begin(), sorted.end(), [](const Entry* a, const Entry* b) {
        return a->begin != b->begin ? a->begin < b->begin : a->end < b->end;
    });
    uint64_t expect = 0;
    for (const Entry* e : sorted) {
        if (e->begin != expect) {
            result.error = malformed(path, e->begin < expect ? "two tensors overlap" : "the buffer has a hole");
            return result;
        }
        expect = e->end;
    }
    if (expect != data_size) {
        result.error = malformed(path, "the buffer holds bytes that no tensor uses");
        return result;
    }
    for (const Entry& e : entries) {
        if (filter && !filter(e.name)) continue;
        if (!e.known && !e.bf16) {
            result.warnings.push_back("Skipped the tensor " + e.name + ", whose dtype " + e.dtype_text +
                                      " is not supported.");
            continue;
        }
        ReadTensor t;
        t.name = e.name;
        if (e.known) {
            TensorResult m = TensorData::map_file(path, data_start + e.begin, e.dtype, e.shape);
            if ((result.error = m.error)) return result;
            t.data = std::move(m.tensor);
        } else {
            const uint64_t n = (e.end - e.begin) / 2;
            if (n > limits.max_derived_bytes / 4) {
                result.error = Error("loader.too_large", "The converted tensor " + e.name + " exceeds the reader's limit.", path);
                return result;
            }
            std::vector<uint8_t> raw;
            if ((result.error = f.read_vec(data_start + e.begin, e.end - e.begin, &raw, limits.max_derived_bytes))) {
                return result;
            }
            TensorResult o = TensorData::allocate(DType::f32, e.shape);
            if ((result.error = o.error)) return result;
            uint8_t* out = o.tensor.mutable_data();
            for (uint64_t i = 0; i < n; ++i) {
                const float v = bfloat16_to_float(le16(raw.data() + 2 * i));
                std::memcpy(out + 4 * i, &v, 4);
            }
            t.data = std::move(o.tensor);
            t.derived = true;
            t.warnings.push_back("converted from bf16 to f32");
        }
        result.tensors.push_back(std::move(t));
    }
    return result;
}

}  // namespace caliper_builder
