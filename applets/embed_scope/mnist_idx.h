#pragma once
// Minimal MNIST IDX reader (+gunzip). Header-only so the test suite can
// exercise it without linking the applet. IDX: big-endian magic + dims,
// then raw bytes. Images magic 2051, labels magic 2049.
#include <zlib.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace mnist_idx {

inline std::optional<std::vector<uint8_t>> gunzip(
    const std::vector<uint8_t>& gz) {
    z_stream s{};
    if (inflateInit2(&s, 15 + 32) != Z_OK) return std::nullopt;  // auto gzip/zlib
    std::vector<uint8_t> out;
    out.resize(gz.size() * 4 + 1024);
    s.next_in = const_cast<uint8_t*>(gz.data());
    s.avail_in = (uInt)gz.size();
    size_t written = 0;
    int rc;
    do {
        if (written == out.size()) out.resize(out.size() * 2);
        s.next_out = out.data() + written;
        s.avail_out = (uInt)(out.size() - written);
        rc = inflate(&s, Z_NO_FLUSH);
        written = out.size() - s.avail_out;
        if (rc != Z_OK && rc != Z_STREAM_END) { inflateEnd(&s); return std::nullopt; }
    } while (rc != Z_STREAM_END);
    inflateEnd(&s);
    out.resize(written);
    return out;
}

namespace detail {
inline std::optional<uint32_t> be32_at(const std::vector<uint8_t>& b, size_t i) {
    if (i + 4 > b.size()) return std::nullopt;
    return ((uint32_t)b[i] << 24) | ((uint32_t)b[i + 1] << 16) |
           ((uint32_t)b[i + 2] << 8) | (uint32_t)b[i + 3];
}
} // namespace detail

struct Images {
    int n = 0, rows = 0, cols = 0;
    std::vector<uint8_t> pixels;   // n*rows*cols, row-major
};

inline std::optional<Images> parse_images(const std::vector<uint8_t>& raw) {
    auto magic = detail::be32_at(raw, 0);
    if (!magic || *magic != 2051) return std::nullopt;
    auto n = detail::be32_at(raw, 4), r = detail::be32_at(raw, 8),
         c = detail::be32_at(raw, 12);
    if (!n || !r || !c) return std::nullopt;
    size_t need = (size_t)*n * *r * *c;
    if (raw.size() < 16 + need) return std::nullopt;
    Images out;
    out.n = (int)*n; out.rows = (int)*r; out.cols = (int)*c;
    out.pixels.assign(raw.begin() + 16, raw.begin() + 16 + need);
    return out;
}

inline std::optional<std::vector<uint8_t>> parse_labels(
    const std::vector<uint8_t>& raw) {
    auto magic = detail::be32_at(raw, 0);
    if (!magic || *magic != 2049) return std::nullopt;
    auto n = detail::be32_at(raw, 4);
    if (!n || raw.size() < 8 + *n) return std::nullopt;
    return std::vector<uint8_t>(raw.begin() + 8, raw.begin() + 8 + *n);
}

} // namespace mnist_idx
