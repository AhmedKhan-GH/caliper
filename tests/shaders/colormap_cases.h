#pragma once
// The boundary cases of the colormap by keys (design: renderer-core, "Exact colormap output"):
// the ranges, a test table, and for each range the inputs the design lists (each key and its two
// neighbours, the range's ends and their neighbours, signed zeros, infinities, quiet, signalling
// and negative NaNs, subnormals, the largest floats and 65,536 random bit patterns), with the
// reference's RGBA for each. The keys come from the shared header src/host/gfx/colormap_keys.h.
// The fast CPU test, the Metal rows, the shader tests and the case file of the browser rows
// (caliper_colormap_cases) all take their cases from here.
#include "gfx/colormap_keys.h"
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
    const float nan = std::numeric_limits<float>::quiet_NaN();
    return {
        {"increasing", 0.0f, 1.0f},
        {"increasing, offset", -3.5f, 7.25f},
        {"decreasing", 2.0f, -1.0f},
        {"decreasing, wide", 1e30f, -1e30f},
        {"degenerate", 0.5f, 0.5f},
        {"signed zeros", -0.0f, 0.0f},
        {"one step", 1.0f, std::nextafter(1.0f, 2.0f)},
        {"infinite end above", 0.0f, inf},
        {"infinite end below", -inf, 1.0f},
        {"infinite end, decreasing", 1.0f, -inf},
        {"subnormal", -1e-39f, 1e-39f},
        {"largest floats", 0.0f, FLT_MAX},
        {"overflowing span", -FLT_MAX, FLT_MAX},
        {"NaN end", nan, 1.0f},
    };
}

inline float as_float(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t as_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// The table: entry k has red k, so equal bytes mean an equal index; every channel varies.
inline std::vector<uint32_t> lut() {
    std::vector<uint32_t> t(256);
    for (uint32_t k = 0; k < 256; ++k)
        t[k] = k | ((255u - k) << 8) | (((k * 37u) & 255u) << 16) | ((255u - ((k * 11u) & 127u)) << 24);
    return t;
}

// The block of one range with the test table, from the shared derivation.
inline caliper_gfx::shaders::Cmap block(const Range& r, const std::vector<uint32_t>& table) {
    caliper_gfx::shaders::Cmap c{};
    caliper_gfx::cmap::fill_block(caliper_gfx::cmap::derive_keys(r.vmin, r.vmax), r.vmin, r.vmax, table.data(), &c);
    return c;
}

// The seed of range i's random bit patterns, the same in every test and in the case file.
inline uint64_t seed(size_t i) { return 0x9e3779b97f4a7c15ull + i; }

// The inputs of one range, padded with zeros to a multiple of 64.
inline std::vector<uint32_t> inputs(const Range& r, const caliper_gfx::shaders::Cmap& c, uint64_t seed) {
    using caliper_gfx::cmap::bits_of;
    std::vector<uint32_t> in = {
        0x00000000u, 0x80000000u,                             // +0, -0
        0x7f800000u, 0xff800000u,                             // +inf, -inf
        0x7fc00000u, 0x7f800001u, 0x7fbfffffu,                // quiet and signalling NaNs
        0xffc00000u, 0xff800001u, 0x7fffffffu, 0xffffffffu,   // negative NaNs, the extremes
        as_bits(FLT_MAX), as_bits(-FLT_MAX), 0x7f7ffffeu, 0xff7ffffeu,   // the largest floats
        as_bits(FLT_MIN), as_bits(-FLT_MIN),                  // the smallest normals
        0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu,   // the smallest and largest subnormals
        as_bits(1.0f), as_bits(-1.0f),
    };
    for (const float end : {r.vmin, r.vmax}) {                // the ends and their neighbours
        const uint32_t b = as_bits(end);
        in.push_back(b);
        in.push_back(b + 1);
        in.push_back(b - 1);
    }
    for (uint32_t k = 1; k < 256; ++k) {
        const uint32_t key = caliper_gfx::cmap::key_at(c, k);
        if (key == caliper_gfx::cmap::kSentinel) continue;
        for (int d = -1; d <= 1; ++d) in.push_back(bits_of((key + uint32_t(d)) ^ c.flip));
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
