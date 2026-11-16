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

}  // namespace caliper_builder
