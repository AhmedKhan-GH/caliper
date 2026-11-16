#include "net/sha1.h"

#include <cstring>

namespace caliper_net {
namespace {

inline uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

void compress(uint32_t h[5], const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
               (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
    for (int i = 16; i < 80; i++) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
        else { f = b ^ c ^ d; k = 0xca62c1d6; }
        const uint32_t t = rotl(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rotl(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}

constexpr char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

}  // namespace

void sha1(const void* data, size_t len, uint8_t out[20]) {
    uint32_t h[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const size_t full = len / 64;
    for (size_t i = 0; i < full; i++) compress(h, p + 64 * i);

    // Final block(s): the remainder, 0x80, zeros, then the bit length (big-endian).
    uint8_t tail[128] = {};
    const size_t rem = len - full * 64;
    if (rem) std::memcpy(tail, p + full * 64, rem);
    tail[rem] = 0x80;
    const size_t tail_len = rem < 56 ? 64 : 128;
    const uint64_t bits = uint64_t(len) * 8;
    for (int i = 0; i < 8; i++) tail[tail_len - 1 - i] = uint8_t(bits >> (8 * i));
    compress(h, tail);
    if (tail_len == 128) compress(h, tail + 64);

    for (int i = 0; i < 5; i++) {
        out[4 * i] = uint8_t(h[i] >> 24);
        out[4 * i + 1] = uint8_t(h[i] >> 16);
        out[4 * i + 2] = uint8_t(h[i] >> 8);
        out[4 * i + 3] = uint8_t(h[i]);
    }
}

std::string base64_encode(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const uint32_t v = (uint32_t(p[i]) << 16) | (uint32_t(p[i + 1]) << 8) | p[i + 2];
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];
        out += kB64[v & 63];
    }
    if (len - i == 1) {
        const uint32_t v = uint32_t(p[i]) << 16;
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += "==";
    } else if (len - i == 2) {
        const uint32_t v = (uint32_t(p[i]) << 16) | (uint32_t(p[i + 1]) << 8);
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

bool base64_decode(const std::string& in, std::string* out) {
    out->clear();
    if (in.size() % 4 != 0) return false;
    for (size_t i = 0; i < in.size(); i += 4) {
        const bool last = i + 4 == in.size();
        int v[4];
        int pad = 0;
        for (int j = 0; j < 4; j++) {
            const char c = in[i + j];
            if (c == '=' && last && j >= 2) {
                v[j] = 0;
                ++pad;
                continue;
            }
            if (pad) return false;   // a character after '='
            v[j] = b64_value(c);
            if (v[j] < 0) return false;
        }
        const uint32_t w = (uint32_t(v[0]) << 18) | (uint32_t(v[1]) << 12) |
                           (uint32_t(v[2]) << 6) | uint32_t(v[3]);
        out->push_back(char(w >> 16));
        if (pad < 2) out->push_back(char((w >> 8) & 0xff));
        if (pad < 1) out->push_back(char(w & 0xff));
    }
    return true;
}

}  // namespace caliper_net
