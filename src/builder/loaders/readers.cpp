// Reader names, extensions and magic bytes.
#include <caliper_builder/loaders/readers.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>

namespace caliper_builder {
namespace {

const ReaderKind kAll[] = {ReaderKind::csv,  ReaderKind::parquet, ReaderKind::json,        ReaderKind::jsonl,
                           ReaderKind::arrow, ReaderKind::npy,    ReaderKind::npz,         ReaderKind::safetensors,
                           ReaderKind::hdf5, ReaderKind::image,   ReaderKind::wav,         ReaderKind::python};

struct Ext {
    const char* ext;
    ReaderKind reader;
};
const Ext kExtensions[] = {
    {".csv", ReaderKind::csv},          {".tsv", ReaderKind::csv},          {".parquet", ReaderKind::parquet},
    {".pq", ReaderKind::parquet},       {".json", ReaderKind::json},        {".jsonl", ReaderKind::jsonl},
    {".ndjson", ReaderKind::jsonl},     {".arrow", ReaderKind::arrow},      {".feather", ReaderKind::arrow},
    {".npy", ReaderKind::npy},          {".npz", ReaderKind::npz},          {".safetensors", ReaderKind::safetensors},
    {".h5", ReaderKind::hdf5},          {".hdf5", ReaderKind::hdf5},        {".png", ReaderKind::image},
    {".jpg", ReaderKind::image},        {".jpeg", ReaderKind::image},       {".wav", ReaderKind::wav},
};

}  // namespace

const char* reader_name(ReaderKind reader) {
    switch (reader) {
        case ReaderKind::csv: return "csv";
        case ReaderKind::parquet: return "parquet";
        case ReaderKind::json: return "json";
        case ReaderKind::jsonl: return "jsonl";
        case ReaderKind::arrow: return "arrow";
        case ReaderKind::npy: return "npy";
        case ReaderKind::npz: return "npz";
        case ReaderKind::safetensors: return "safetensors";
        case ReaderKind::hdf5: return "hdf5";
        case ReaderKind::image: return "image";
        case ReaderKind::wav: return "wav";
        case ReaderKind::python: return "python";
    }
    return "csv";
}

bool reader_from_name(const std::string& name, ReaderKind* out) {
    for (ReaderKind r : kAll) {
        if (name == reader_name(r)) {
            *out = r;
            return true;
        }
    }
    return false;
}

std::vector<std::string> reader_names() {
    std::vector<std::string> names;
    for (ReaderKind r : kAll) names.push_back(reader_name(r));
    return names;
}

bool reader_is_tabular(ReaderKind r) {
    return r == ReaderKind::csv || r == ReaderKind::parquet || r == ReaderKind::json || r == ReaderKind::jsonl ||
           r == ReaderKind::arrow;
}

bool reader_is_single_tensor(ReaderKind r) {
    return r == ReaderKind::npy || r == ReaderKind::image || r == ReaderKind::wav;
}

bool reader_for_extension(const std::string& path, ReaderKind* out) {
    std::string ext = std::filesystem::u8path(path).extension().u8string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    for (const Ext& e : kExtensions) {
        if (ext == e.ext) {
            *out = e.reader;
            return true;
        }
    }
    return false;
}

std::string sniff_format(const uint8_t* b, std::size_t n, uint64_t file_size) {
    auto starts = [&](const char* sig, std::size_t len) { return n >= len && std::memcmp(b, sig, len) == 0; };
    if (starts("\x93NUMPY", 6)) return "npy";
    if (starts("PK\x03\x04", 4) || starts("PK\x05\x06", 4)) return "zip";
    if (starts("PAR1", 4)) return "parquet";
    if (starts("ARROW1", 6)) return "arrow";
    if (starts("\x89HDF\r\n\x1a\n", 8)) return "hdf5";
    if (starts("\x89PNG\r\n\x1a\n", 8)) return "png";
    if (starts("\xFF\xD8\xFF", 3)) return "jpeg";
    if (n >= 12 && std::memcmp(b, "RIFF", 4) == 0 && std::memcmp(b + 8, "WAVE", 4) == 0) return "wav";
    if (n >= 2 && b[0] == 0x80 && b[1] >= 2 && b[1] <= 5) return "pickle";
    if (n >= 9 && b[8] == '{') {
        uint64_t len = 0;
        for (int i = 7; i >= 0; --i) len = (len << 8) | b[i];
        if (len >= 2 && len <= file_size && 8 + len <= file_size) return "safetensors";
    }
    return "";
}

bool reader_for_format(const std::string& format, ReaderKind* out) {
    static const std::pair<const char*, ReaderKind> table[] = {
        {"npy", ReaderKind::npy},     {"zip", ReaderKind::npz},     {"parquet", ReaderKind::parquet},
        {"arrow", ReaderKind::arrow}, {"hdf5", ReaderKind::hdf5},   {"png", ReaderKind::image},
        {"jpeg", ReaderKind::image},  {"wav", ReaderKind::wav},     {"safetensors", ReaderKind::safetensors}};
    for (const auto& p : table) {
        if (format == p.first) {
            *out = p.second;
            return true;
        }
    }
    return false;
}

namespace {

Error bad_value(const std::string& key, const std::string& message) {
    return Error("loader.bad_value", message, "reader." + key);
}

bool printable(const std::string& s, bool allow_tab) {
    for (unsigned char c : s) {
        if (c < 0x20 && !(allow_tab && c == '\t')) return false;
        if (c == 0x7F) return false;
    }
    return true;
}

Error one_char(const std::string& key, const nlohmann::json& v, bool allow_tab) {
    if (!v.is_string()) return bad_value(key, "`" + key + "` is a string of one character.");
    const std::string s = v.get<std::string>();
    if (s.size() != 1 || !printable(s, allow_tab)) {
        return bad_value(key, "`" + key + "` is one printable character" + (allow_tab ? " or a tab." : "."));
    }
    return {};
}

}  // namespace

Error validate_reader_options(ReaderKind reader, const nlohmann::json& options) {
    if (options.is_null()) return {};
    if (!options.is_object()) return Error("loader.bad_value", "The reader options are a table.", "reader");
    for (auto it = options.begin(); it != options.end(); ++it) {
        const std::string& key = it.key();
        const nlohmann::json& v = it.value();
        const bool tabular_union = (reader == ReaderKind::csv || reader == ReaderKind::parquet ||
                                    reader == ReaderKind::json || reader == ReaderKind::jsonl) &&
                                   key == "union_by_name";
        if (tabular_union || (reader == ReaderKind::csv && key == "header")) {
            if (!v.is_boolean()) return bad_value(key, "`" + key + "` is true or false.");
        } else if (reader == ReaderKind::csv && key == "delimiter") {
            if (Error e = one_char(key, v, true)) return e;
        } else if (reader == ReaderKind::csv && (key == "quote" || key == "escape")) {
            if (Error e = one_char(key, v, false)) return e;
        } else if (reader == ReaderKind::csv && key == "skip") {
            if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > 1000000) {
                return bad_value(key, "`skip` is an integer from 0 to 1,000,000.");
            }
        } else if (reader == ReaderKind::csv && key == "null") {
            if (!v.is_string() || v.get<std::string>().size() > 256 || !printable(v.get<std::string>(), false)) {
                return bad_value(key, "`null` is a string of at most 256 printable characters.");
            }
        } else if (reader == ReaderKind::json && key == "format") {
            const std::string f = v.is_string() ? v.get<std::string>() : "";
            if (f != "auto" && f != "array" && f != "unstructured") {
                return bad_value(key, "`format` is \"auto\", \"array\" or \"unstructured\".");
            }
        } else if (reader == ReaderKind::hdf5 && key == "dataset") {
            if (!v.is_string() || v.get<std::string>().empty() || v.get<std::string>().size() > 1024 ||
                !printable(v.get<std::string>(), false)) {
                return bad_value(key, "`dataset` is the path of a dataset in the file.");
            }
        } else if (reader == ReaderKind::image && key == "channels") {
            if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > 4) {
                return bad_value(key, "`channels` is 0, 1, 2, 3 or 4.");
            }
        } else {
            return Error("loader.unknown_key",
                         std::string("The ") + reader_name(reader) + " reader has no option `" + key + "`.",
                         "reader." + key);
        }
    }
    return {};
}

}  // namespace caliper_builder
