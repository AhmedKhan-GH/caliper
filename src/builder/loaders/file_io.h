#pragma once
// InputFile: bounds-checked random reads of one regular file, for the readers.
// Every read is checked against the file's size as measured when it opened,
// so a header's offsets can never send a read past the end. Internal to the
// loaders module.
#include <caliper_builder/error.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace caliper_builder {
namespace detail {

class InputFile {
public:
    Error open(const std::string& path);
    uint64_t size() const { return size_; }
    const std::string& path() const { return path_; }

    // Reads exactly n bytes at offset; fails with loader.truncated past the end.
    Error read(uint64_t offset, uint64_t n, void* out);
    // Reads n bytes at offset into `out`, refusing more than `max` bytes.
    Error read_vec(uint64_t offset, uint64_t n, std::vector<uint8_t>* out, uint64_t max);

private:
    std::ifstream in_;
    std::string path_;
    uint64_t size_ = 0;
};

inline uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
inline uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t le64(const uint8_t* p) { return static_cast<uint64_t>(le32(p)) | (static_cast<uint64_t>(le32(p + 4)) << 32); }

// a + b <= limit without overflow.
inline bool fits(uint64_t a, uint64_t b, uint64_t limit) { return a <= limit && b <= limit - a; }

// The file name without folders and without its last extension.
std::string file_stem(const std::string& path);

// The nesting depth of JSON text, counted outside strings, so that a hostile
// document cannot exhaust the parser's or the destructor's stack. Counting
// stops once the depth exceeds `stop_at`.
int json_depth(const uint8_t* text, std::size_t size, int stop_at);

}  // namespace detail
}  // namespace caliper_builder
