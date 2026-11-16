// The pickle refusal: by extension, by the protocol marker, by the torch zip
// layout and by the object dtype of .npy data.
#include <caliper_builder/loaders/pickle.h>
#include <caliper_builder/loaders/readers.h>

#include "file_io.h"
#include "npy_format.h"
#include "refusals.h"
#include "zip_reader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <functional>

namespace caliper_builder {

using namespace detail;

namespace {

std::string lower_extension(const std::string& path) {
    std::string ext = std::filesystem::u8path(path).extension().u8string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Refuses an .npy header with the object dtype. `prefix` holds its first bytes,
// `available` bounds its length, and `read_prefix` reads its first n bytes.
Error check_npy_header(const std::string& path, const std::string& label, std::vector<uint8_t> prefix,
                       uint64_t available, const ReaderLimits& limits,
                       const std::function<Error(uint64_t, std::vector<uint8_t>*)>& read_prefix) {
    uint64_t total = 0;
    if (npy_header_length(prefix.data(), prefix.size(), available, limits, &total, path)) return {};
    if (read_prefix(total, &prefix) || prefix.size() != total) return {};
    NpyHeader h;
    Error e = parse_npy_header(prefix.data(), prefix.size(), &h, path);
    if (e.code == "loader.pickle_refused") return pickle_refused(path, label + " has the object dtype");
    return {};
}

}  // namespace

bool has_pickle_extension(const std::string& path) {
    const std::string ext = lower_extension(path);
    for (const char* p : {".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".joblib"}) {
        if (ext == p) return true;
    }
    return false;
}

Error check_not_pickle(const std::string& path, const ReaderLimits& limits) {
    if (has_pickle_extension(path)) {
        return pickle_refused(path, "the extension " + lower_extension(path) + " is used for pickle and torch files");
    }
    InputFile f;
    if (f.open(path)) return {};
    uint8_t head[16] = {};
    const uint64_t n = std::min<uint64_t>(f.size(), sizeof head);
    if (f.read(0, n, head)) return {};
    const std::string format = sniff_format(head, static_cast<std::size_t>(n), f.size());
    if (format == "pickle") return pickle_refused(path, "it starts with the pickle protocol marker");

    if (format == "npy") {
        std::vector<uint8_t> prefix(head, head + n);
        return check_npy_header(path, "the array", prefix, f.size(), limits,
                                [&](uint64_t total, std::vector<uint8_t>* out) {
                                    return f.read_vec(0, total, out, limits.max_npy_header + 12);
                                });
    }

    if (format == "zip") {
        std::vector<ZipEntry> entries;
        if (read_zip_directory(f, limits, &entries)) return {};
        for (const ZipEntry& e : entries) {
            if (e.name == "data.pkl" || ends_with(e.name, "/data.pkl")) {
                return pickle_refused(path, "a torch archive with " + e.name);
            }
            if (ends_with(e.name, ".pkl") || ends_with(e.name, ".pickle")) {
                return pickle_refused(path, "the member " + e.name);
            }
        }
        for (const ZipEntry& e : entries) {
            if (!e.name.empty() && e.name.back() == '/') continue;
            std::vector<uint8_t> prefix;
            if (read_zip_member_prefix(f, e, 12, &prefix)) continue;
            if (looks_like_pickle(prefix.data(), prefix.size())) return pickle_refused(path, "the member " + e.name);
            if (ends_with(e.name, ".npy")) {
                if (Error err = check_npy_header(path, "the member " + e.name, prefix, e.size, limits,
                                                 [&](uint64_t total, std::vector<uint8_t>* out) {
                                                     return read_zip_member_prefix(f, e, total, out);
                                                 })) {
                    return err;
                }
            }
        }
        return {};
    }

    // Protocols 0 and 1 have no marker. Only a file whose extension has no reader
    // is judged by its first opcode and its final STOP opcode, so a text file of a
    // known format is never mistaken for one.
    ReaderKind reader;
    if (format.empty() && n >= 2 && !reader_for_extension(path, &reader)) {
        const uint8_t first = head[0];
        uint8_t last = 0;
        if (!f.read(f.size() - 1, 1, &last) && last == '.' &&
            (first == '(' || first == ']' || first == '}' || first == 'c')) {
            return pickle_refused(path, "it looks like a pickle of protocol 0 or 1");
        }
    }
    return {};
}

}  // namespace caliper_builder
