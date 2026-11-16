// scene_colormap_keys.h: TEMPORARY. The colormap blocks of the scene view, until migration step 6.
//
// A colormapped value of a scene is decided by integer comparisons against 257 unsigned keys that
// the CPU derives from the reference map_f32_to_rgba8 for each colormap and range (renderer core,
// "Exact colormap output"). Step 6 puts that derivation in a shared C++ header with its exhaustive
// CPU test, and deletes this file. Until then the scene view derives the keys here, exactly as the
// graphics suite's integration rows do (tests/shaders/colormap_cases.h, block()), and owns the
// cache of blocks by colormap and the bits of the range, which nothing else owned. This is the one
// place in src/ that derives keys; nothing else may include it.
#pragma once

#include "gfx/shaders/generated/uniforms.h"
#include "tensor_bridge.h"   // map_f32_to_rgba8, colormap_lut

#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>

namespace caliper_live_views {

class SceneColormapKeys {
public:
    static constexpr size_t kMaxBlocks = 64;   // ranges kept; the cache starts again beyond

    // The block of colormap `colormap` over [vmin, vmax]: its keys, its flip and its table.
    void block(uint32_t colormap, float vmin, float vmax, caliper_gfx::shaders::Cmap* out) {
        uint32_t lo, hi;
        std::memcpy(&lo, &vmin, 4);
        std::memcpy(&hi, &vmax, 4);
        const auto key = std::make_tuple(colormap, lo, hi);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            if (cache_.size() >= kMaxBlocks) cache_.clear();
            it = cache_.emplace(key, derive(colormap, vmin, vmax)).first;
            ++derived_;
        }
        *out = it->second;
    }
    uint64_t derived() const { return derived_; }   // blocks derived so far (a cache miss each)

private:
    static uint32_t key_of(uint32_t bits) { return (bits >> 31) ? ~bits : (bits | 0x80000000u); }
    static uint32_t bits_of(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key; }

    // The reference's index of one value, read through an identity table.
    static uint32_t ref_index(uint32_t bits, float vmin, float vmax) {
        static const struct Identity {
            uint32_t t[256];
            Identity() { for (uint32_t k = 0; k < 256; ++k) t[k] = k; }
        } identity;
        float v;
        std::memcpy(&v, &bits, 4);
        uint8_t px[4];
        caliper_host::map_f32_to_rgba8(&v, 1, 1, identity.t, vmin, vmax, px);
        return px[0];
    }

    // keys[k] is the first (flipped) key whose reference index is at least k, or 0xffffffff when
    // no float reaches k; keys[0] = 0 and keys[256..259] = 0xffffffff.
    static caliper_gfx::shaders::Cmap derive(uint32_t colormap, float vmin, float vmax) {
        caliper_gfx::shaders::Cmap c{};
        c.vmin = vmin;
        c.vmax = vmax;
        const uint32_t kmin = key_of(0xff800000u), kmax = key_of(0x7f800000u);   // -inf, +inf
        const uint32_t flip = ref_index(0xff800000u, vmin, vmax) > ref_index(0x7f800000u, vmin, vmax) ? 0xffffffffu : 0u;
        c.flip = flip;
        const uint32_t lo = flip ? ~kmax : kmin, hi = flip ? ~kmin : kmax;
        auto f = [&](uint32_t fk) { return ref_index(bits_of(fk ^ flip), vmin, vmax); };
        uint32_t keys[260];
        keys[0] = 0;
        const uint32_t top = f(hi);
        for (uint32_t k = 1; k < 256; ++k) {
            if (top < k) { keys[k] = 0xffffffffu; continue; }
            uint32_t a = lo, b = hi;   // the first fk in [a, b] with f(fk) >= k
            while (a < b) {
                const uint32_t mid = a + (b - a) / 2;
                if (f(mid) >= k) b = mid;
                else a = mid + 1;
            }
            keys[k] = a;
        }
        keys[256] = keys[257] = keys[258] = keys[259] = 0xffffffffu;
        static_assert(sizeof(c.keys) == sizeof(keys), "the block's keys");
        std::memcpy(c.keys, keys, sizeof(c.keys));
        const uint32_t* lut = caliper_host::colormap_lut(int32_t(colormap));
        if (!lut) lut = caliper_host::colormap_lut(0);
        std::memcpy(c.lut, lut, sizeof(c.lut));
        return c;
    }

    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, caliper_gfx::shaders::Cmap> cache_;
    uint64_t derived_ = 0;
};

}  // namespace caliper_live_views
