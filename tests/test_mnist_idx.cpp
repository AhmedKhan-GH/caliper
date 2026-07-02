#include <doctest/doctest.h>
#include "mnist_idx.h"
#include <zlib.h>
#include <cstring>
using namespace mnist_idx;

namespace {
void be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}
std::vector<uint8_t> gzip_compress(const std::vector<uint8_t>& in) {
    z_stream s{};
    deflateInit2(&s, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8,
                 Z_DEFAULT_STRATEGY);                     // gzip wrapper
    std::vector<uint8_t> out(deflateBound(&s, in.size()));
    s.next_in = const_cast<uint8_t*>(in.data());
    s.avail_in = (uInt)in.size();
    s.next_out = out.data();
    s.avail_out = (uInt)out.size();
    deflate(&s, Z_FINISH);
    out.resize(out.size() - s.avail_out);
    deflateEnd(&s);
    return out;
}
} // namespace

TEST_CASE("mnist_idx: images parse (magic, dims, pixel order)") {
    std::vector<uint8_t> raw;
    be32(raw, 2051); be32(raw, 2); be32(raw, 2); be32(raw, 3);  // 2 imgs, 2x3
    for (int i = 0; i < 12; i++) raw.push_back((uint8_t)(i * 10));
    auto img = parse_images(raw);
    REQUIRE(img.has_value());
    CHECK(img->n == 2); CHECK(img->rows == 2); CHECK(img->cols == 3);
    REQUIRE(img->pixels.size() == 12);
    CHECK(img->pixels[0] == 0); CHECK(img->pixels[11] == 110);
}

TEST_CASE("mnist_idx: labels parse; wrong magic refused") {
    std::vector<uint8_t> raw;
    be32(raw, 2049); be32(raw, 3);
    raw.push_back(7); raw.push_back(0); raw.push_back(9);
    auto lab = parse_labels(raw);
    REQUIRE(lab.has_value());
    CHECK((*lab)[0] == 7); CHECK((*lab)[2] == 9);
    // images magic fed to labels parser (and vice versa) must refuse
    std::vector<uint8_t> wrong; be32(wrong, 2051); be32(wrong, 1);
    CHECK_FALSE(parse_labels(wrong).has_value());
}

TEST_CASE("mnist_idx: truncated payload refused") {
    std::vector<uint8_t> raw;
    be32(raw, 2051); be32(raw, 2); be32(raw, 28); be32(raw, 28);  // promises 1568
    raw.push_back(1);                                              // delivers 1
    CHECK_FALSE(parse_images(raw).has_value());
}

TEST_CASE("mnist_idx: gunzip round-trips") {
    std::vector<uint8_t> original;
    for (int i = 0; i < 5000; i++) original.push_back((uint8_t)(i % 251));
    auto un = gunzip(gzip_compress(original));
    REQUIRE(un.has_value());
    CHECK(*un == original);
    CHECK_FALSE(gunzip({0x00, 0x01, 0x02}).has_value());  // not gzip
}
