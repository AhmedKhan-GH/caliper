// The .npz reader: a zip archive of .npy members.
#include <caliper_builder/loaders/tensor_readers.h>

#include "file_io.h"
#include "npy_format.h"
#include "refusals.h"
#include "zip_reader.h"

#include <set>

namespace caliper_builder {

using namespace detail;

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

TensorReadResult read_npz(const std::string& path, const TensorFilter& filter, const ReaderLimits& limits) {
    TensorReadResult result;
    InputFile f;
    if ((result.error = f.open(path))) return result;
    std::vector<ZipEntry> entries;
    if ((result.error = read_zip_directory(f, limits, &entries))) return result;

    // A torch archive or a pickled member refuses the whole file, before any tensor is read.
    for (const ZipEntry& e : entries) {
        if (e.name == "data.pkl" || ends_with(e.name, "/data.pkl")) {
            result.error = pickle_refused(path, "a torch archive with " + e.name);
            return result;
        }
        if (ends_with(e.name, ".pkl") || ends_with(e.name, ".pickle")) {
            result.error = pickle_refused(path, "the member " + e.name);
            return result;
        }
    }

    std::set<std::string> seen;
    for (const ZipEntry& e : entries) {
        if (!e.name.empty() && e.name.back() == '/') continue;  // a folder entry
        std::vector<uint8_t> prefix;
        if (!ends_with(e.name, ".npy")) {
            if ((result.error = read_zip_member_prefix(f, e, 2, &prefix))) return result;
            if (looks_like_pickle(prefix.data(), prefix.size())) {
                result.error = pickle_refused(path, "the member " + e.name);
                return result;
            }
            result.warnings.push_back("Skipped the member " + e.name + ", which is not an .npy file.");
            continue;
        }
        const std::string name = e.name.substr(0, e.name.size() - 4);
        if (!seen.insert(name).second) {
            result.error = Error("loader.malformed", "The archive has two members named " + e.name + ".", path);
            return result;
        }
        // Every member's header is read, so an object dtype anywhere refuses the file.
        if ((result.error = read_zip_member_prefix(f, e, 12, &prefix))) return result;
        uint64_t header_total = 0;
        const std::string where = path + ":" + e.name;
        if ((result.error = npy_header_length(prefix.data(), prefix.size(), e.size, limits, &header_total, where))) {
            return result;
        }
        if ((result.error = read_zip_member_prefix(f, e, header_total, &prefix))) return result;
        if (prefix.size() != header_total) {
            result.error = Error("loader.truncated", "The member's .npy header is cut short.", where);
            return result;
        }
        NpyHeader h;
        if ((result.error = parse_npy_header(prefix.data(), prefix.size(), &h, where))) {
            if (result.error.code == "loader.pickle_refused") result.error = pickle_refused(path, "the member " + e.name + " has the object dtype");
            return result;
        }
        if (filter && !filter(name)) continue;
        if (result.tensors.size() >= limits.max_tensors) {
            result.error = Error("loader.too_large", "The archive holds more tensors than the reader's limit.", path);
            return result;
        }
        if (!fits(h.data_offset, h.nbytes, e.size)) {
            result.error = Error("loader.truncated", "The member is shorter than the array its header describes.", where);
            return result;
        }
        ReadTensor t;
        t.name = name;
        if (e.method == 0 && !(e.flags & 1) && h.in_catalog_layout()) {
            TensorResult m = TensorData::map_file(path, e.data_offset + h.data_offset, h.dtype, h.shape);
            if ((result.error = m.error)) return result;
            t.data = std::move(m.tensor);
        } else {
            std::vector<uint8_t> bytes;
            const uint64_t max = limits.max_derived_bytes + limits.max_npy_header + 12;
            if ((result.error = read_zip_member(f, e, max, &bytes))) return result;
            if (h.in_catalog_layout()) {
                if (h.nbytes > limits.max_derived_bytes) {
                    result.error = Error("loader.too_large", "The member is larger than the reader's limit.", where);
                    return result;
                }
                std::vector<uint8_t> data(bytes.begin() + static_cast<std::ptrdiff_t>(h.data_offset),
                                          bytes.begin() + static_cast<std::ptrdiff_t>(h.data_offset + h.nbytes));
                TensorResult o = TensorData::from_bytes(h.dtype, h.shape, std::move(data));
                if ((result.error = o.error)) return result;
                t.data = std::move(o.tensor);
            } else {
                TensorResult c = npy_convert(bytes.data() + h.data_offset, h, limits, where);
                if ((result.error = c.error)) return result;
                t.data = std::move(c.tensor);
                t.warnings.push_back(h.fortran_order ? "converted from Fortran order to row-major order"
                                                     : "converted from big-endian to little-endian bytes");
            }
            t.derived = true;
        }
        result.tensors.push_back(std::move(t));
    }
    return result;
}

}  // namespace caliper_builder
