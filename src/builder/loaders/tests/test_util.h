#pragma once
// Helpers for the loaders tests: a fresh folder under the build tree for each
// test, and small file writers. Nothing here touches a path outside the
// build folder.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace loaders_test {

namespace fs = std::filesystem;

// A new, empty folder <build>/test-tmp/<name>, returned as a canonical path.
inline std::string fresh_dir(const std::string& name) {
    fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return fs::canonical(dir).string();
}

inline void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

inline void write_text(const std::string& path, const std::string& text) {
    write_bytes(path, std::vector<uint8_t>(text.begin(), text.end()));
}

inline std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline void append(std::vector<uint8_t>& out, const void* p, std::size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    out.insert(out.end(), b, b + n);
}
inline void append(std::vector<uint8_t>& out, const std::string& s) { append(out, s.data(), s.size()); }
inline void append(std::vector<uint8_t>& out, const std::vector<uint8_t>& b) { out.insert(out.end(), b.begin(), b.end()); }

template <class T>
inline void put(std::vector<uint8_t>& out, T v) {
    append(out, &v, sizeof v);
}

template <class T>
inline std::vector<uint8_t> raw(const std::vector<T>& values) {
    std::vector<uint8_t> out(values.size() * sizeof(T));
    if (!values.empty()) std::memcpy(out.data(), values.data(), out.size());
    return out;
}

}  // namespace loaders_test
