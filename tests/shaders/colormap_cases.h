#pragma once
// The boundary cases of the colormap by keys (design: renderer-core, "Exact colormap output"):
// for each range, the keys derived from the CPU reference map_f32_to_rgba8, and the inputs
// the design lists (each key and its two neighbours, signed zeros, infinities, quiet,
// signalling and negative NaNs, the range's ends, the extreme floats and 65,536 random bit
// patterns), with the reference's RGBA for each. The key derivation here is the test's own;
// the shared C++ header of migration step 6 replaces it.
#include "gfx/shaders/generated/uniforms.h"
#include "tensor_bridge.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace colormap_cases {

struct Range { const char* name; float vmin, vmax; };

inline std::vector<Range> ranges() {
    const float inf = std::numeric_limits<float>::infinity();
    return {
        {"increasing", 0.0f, 1.0f},
        {"increasing, offset", -3.5f, 7.25f},
        {"decreasing", 2.0f, -1.0f},
        {"decreasing, wide", 1e30f, -1e30f},
        {"degenerate", 0.5f, 0.5f},
        {"infinite end above", 0.0f, inf},
        {"infinite end below", -inf, 1.0f},
        {"subnormal", -1e-39f, 1e-39f},
        {"largest floats", 0.0f, FLT_MAX},
        {"overflowing span", -FLT_MAX, FLT_MAX},
    };
}

// An unsigned key that orders floats numerically: bits | 0x80000000 when positive, ~bits when negative.
inline uint32_t key_of(uint32_t bits) { return (bits >> 31) ? ~bits : (bits | 0x80000000u); }
inline uint32_t bits_of(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key; }
inline float as_float(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t as_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// The table: entry k has red k, so equal bytes mean an equal index; every channel varies.
inline std::vector<uint32_t> lut() {
    std::vector<uint32_t> t(256);
    for (uint32_t k = 0; k < 256; ++k)
        t[k] = k | ((255u - k) << 8) | (((k * 37u) & 255u) << 16) | ((255u - ((k * 11u) & 127u)) << 24);
    return t;
}

// The reference's index of one value, read from map_f32_to_rgba8 with an identity table.
inline uint32_t ref_index(uint32_t bits, float vmin, float vmax) {
    static const std::vector<uint32_t> identity = [] { std::vector<uint32_t> t(256); for (uint32_t k = 0; k < 256; ++k) t[k] = k; return t; }();
    const float v = as_float(bits);
    uint8_t px[4];
    caliper_host::map_f32_to_rgba8(&v, 1, 1, identity.data(), vmin, vmax, px);
    return px[0];
}

// The block of one range: keys[k] is the first (flipped) key whose reference index is at
// least k, or 0xffffffff when no float reaches k; keys[0] = 0, keys[256] = keys[257] = 0xffffffff.
inline caliper_gfx::shaders::Cmap block(const Range& r, const std::vector<uint32_t>& table) {
    caliper_gfx::shaders::Cmap c{};
    c.vmin = r.vmin; c.vmax = r.vmax;
    const uint32_t kmin = key_of(0xff800000u), kmax = key_of(0x7f800000u);   // -inf, +inf
    const uint32_t flip = ref_index(0xff800000u, r.vmin, r.vmax) > ref_index(0x7f800000u, r.vmin, r.vmax) ? 0xffffffffu : 0u;
    c.flip = flip;
    const uint32_t lo = flip ? ~kmax : kmin, hi = flip ? ~kmin : kmax;
    auto f = [&](uint32_t fk) { return ref_index(bits_of(fk ^ flip), r.vmin, r.vmax); };
    uint32_t keys[260];
    keys[0] = 0;
    for (uint32_t k = 1; k < 256; ++k) {
        if (f(hi) < k) { keys[k] = 0xffffffffu; continue; }
        uint32_t a = lo, b = hi;   // the first fk in [a, b] with f(fk) >= k
        while (a < b) { const uint32_t mid = a + (b - a) / 2; if (f(mid) >= k) b = mid; else a = mid + 1; }
        keys[k] = a;
    }
    keys[256] = keys[257] = keys[258] = keys[259] = 0xffffffffu;
    std::memcpy(c.keys, keys, sizeof(c.keys));
    std::memcpy(c.lut, table.data(), sizeof(c.lut));
    return c;
}

// The inputs of one range, padded with zeros to a multiple of 64.
inline std::vector<uint32_t> inputs(const Range& r, const caliper_gfx::shaders::Cmap& c, uint64_t seed) {
    std::vector<uint32_t> in = {
        0x00000000u, 0x80000000u,                         // +0, -0
        0x7f800000u, 0xff800000u,                         // +inf, -inf
        0x7fc00000u, 0x7f800001u, 0x7fbfffffu,            // quiet and signalling NaNs
        0xffc00000u, 0xff800001u, 0x7fffffffu, 0xffffffffu,   // negative NaNs, the extremes
        as_bits(r.vmin), as_bits(r.vmax), as_bits(FLT_MAX), as_bits(-FLT_MAX),
        as_bits(FLT_MIN), as_bits(-FLT_MIN), 0x00000001u, 0x80000001u, as_bits(1.0f), as_bits(-1.0f),
    };
    const uint32_t* keys = &c.keys[0][0];
    for (uint32_t k = 1; k < 256; ++k) {
        if (keys[k] == 0xffffffffu) continue;
        for (int d = -1; d <= 1; ++d) in.push_back(bits_of((keys[k] + uint32_t(d)) ^ c.flip));
    }
    uint64_t s = seed;
    for (int i = 0; i < 65536; ++i) {   // xorshift64*
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        in.push_back(uint32_t((s * 2685821657736338717ull) >> 32));
    }
    while (in.size() % 64) in.push_back(0);
    return in;
}

// The reference's packed RGBA, r | g << 8 | b << 16 | a << 24, for each input.
inline std::vector<uint32_t> expected(const Range& r, const std::vector<uint32_t>& in, const std::vector<uint32_t>& table) {
    std::vector<float> v(in.size());
    std::memcpy(v.data(), in.data(), in.size() * 4);
    std::vector<uint8_t> px(in.size() * 4);
    caliper_host::map_f32_to_rgba8(v.data(), int(in.size()), 1, table.data(), r.vmin, r.vmax, px.data());
    std::vector<uint32_t> out(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = px[4 * i] | (px[4 * i + 1] << 8) | (px[4 * i + 2] << 16) | (uint32_t(px[4 * i + 3]) << 24);
    return out;
}

}  // namespace colormap_cases
