// colormap_keys.h: the colormap by keys on the CPU (renderer core, "Exact colormap output").
//
// A colormap's level is decided by integer comparisons against keys that the CPU derives from
// the reference map_f32_to_rgba8 itself, so that no compiler's or adapter's float arithmetic can
// move a value to another level. The algorithm has two sources: this header and the WGSL of
// src/host/gfx/shaders/common/colormap.wgsl, which take the same steps.
//
//   key_of            the key of a float's bits, an unsigned integer in the floats' numeric order
//   derive_keys       the 255 keys of a range, each found by binary search over keys
//   fill_block        a Cmap block (the generated uniform struct) from the keys and a table: one
//                     16-byte entry per level, its first key, the next level's first key, its colour
//   colormap_index    the integer decision, as the shader takes it; colormap_rgba its colour
//   resolve_range     the range of a mapping whose ends may come from statistics
//   KeyBlockCache     the key blocks of a view or of the layer, by colormap and range
//
// Plain C++17 with no GPU. The derivation evaluates map_f32_to_rgba8, so a target that derives
// keys links src/host/tensor_bridge.cpp, as the host, the tests and the WebAssembly views do.
// The functions the decision needs are marked CALIPER_CMAP_HD, which is __host__ __device__
// under nvcc, so that a CUDA kernel can include this header and take the same steps.
//
// What the keys rest on: the reference is monotone in the value, because every IEEE operation
// in it is, so it is monotone in the key. For a range, key k (1 to 255) is the first key whose
// reference level is at least k, or kSentinel when no float reaches k. A decreasing range flips
// every key (XOR with 0xffffffff), so that the level only ever grows with the key and the
// shader only ever tests key >= keys[k]. A degenerate range, and a range with an infinite end
// or an infinite span, maps every value to level 0, so all its keys are the sentinel. NaN is
// level 0, decided on the bits. The exhaustive test (tests/test_colormap_exhaustive.cpp)
// compares colormap_index with map_f32_to_rgba8 for all 2^32 inputs on a set of ranges.
#pragma once

#include "gfx/shaders/generated/uniforms.h"
#include "tensor_bridge.h"   // map_f32_to_rgba8 (the reference) and colormap_lut (the tables)

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <unordered_map>

#if defined(__CUDACC__)
#define CALIPER_CMAP_HD __host__ __device__
#else
#define CALIPER_CMAP_HD
#endif

namespace caliper_gfx {
namespace cmap {

inline constexpr uint32_t kSentinel = 0xffffffffu;   // a key that no float reaches
inline constexpr uint32_t kKeyCount = 257;           // key[0..256]: each level's first key, and the end

// ---- Keys --------------------------------------------------------------------------------------

// The key of a float's bits: bits | 0x80000000 for a positive float, ~bits for a negative one.
// Keys order floats numerically, from -inf (0x007fffff) to +inf (0xff800000), with -0 just below
// +0; NaN bits fall outside that span.
CALIPER_CMAP_HD inline uint32_t key_of(uint32_t bits) { return (bits >> 31) ? ~bits : (bits | 0x80000000u); }
// The bits of a key: key_of's inverse.
CALIPER_CMAP_HD inline uint32_t bits_of(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key; }
CALIPER_CMAP_HD inline bool nan_bits(uint32_t bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }

struct Keys {
    uint32_t flip = 0;               // 0xffffffff for a decreasing range, whose keys are flipped
    uint32_t key[kKeyCount] = {};    // key[0] = 0; key[k] for k in 1..255; key[256] = kSentinel
};

// The reference's level of one value: map_f32_to_rgba8 itself, read through an identity table.
inline uint32_t reference_index(uint32_t bits, float vmin, float vmax) {
    struct Identity {
        uint32_t t[256];
        Identity() { for (uint32_t k = 0; k < 256; ++k) t[k] = k; }
    };
    static const Identity identity;
    float v;
    std::memcpy(&v, &bits, 4);
    uint8_t px[4];
    caliper_host::map_f32_to_rgba8(&v, 1, 1, identity.t, vmin, vmax, px);
    return px[0];
}

// The keys of the range (vmin, vmax). Each key is a binary search over the (flipped) keys of
// -inf to +inf, about 32 evaluations of the reference, starting at the key below it.
inline Keys derive_keys(float vmin, float vmax) {
    Keys k;
    const uint32_t neg_inf = 0xff800000u, pos_inf = 0x7f800000u;
    const uint32_t at_neg = reference_index(neg_inf, vmin, vmax), at_pos = reference_index(pos_inf, vmin, vmax);
    k.flip = at_neg > at_pos ? 0xffffffffu : 0u;
    const uint32_t lo = key_of(k.flip ? pos_inf : neg_inf) ^ k.flip;   // the lowest flipped key
    const uint32_t hi = key_of(k.flip ? neg_inf : pos_inf) ^ k.flip;   // the highest
    const uint32_t top = k.flip ? at_neg : at_pos;                     // the highest level reached
    auto level = [&](uint32_t fk) { return reference_index(bits_of(fk ^ k.flip), vmin, vmax); };
    k.key[0] = 0;
    uint32_t from = lo;
    for (uint32_t l = 1; l < 256; ++l) {
        if (top < l) { k.key[l] = kSentinel; continue; }
        uint32_t a = from, b = hi;   // the first flipped key in [a, b] whose level is at least l
        while (a < b) {
            const uint32_t mid = a + (b - a) / 2;
            if (level(mid) >= l) b = mid; else a = mid + 1;
        }
        k.key[l] = from = a;
    }
    k.key[256] = kSentinel;
    return k;
}

// ---- The block ---------------------------------------------------------------------------------

static_assert(sizeof(shaders::Cmap::levels) == 256 * 16, "the block's levels: one 16-byte entry each");

// The Cmap block of a range: the range itself (for the shader's guess), the flip, and for each
// level k the entry (key[k], key[k + 1], the table's colour k packed r | g << 8 | b << 16 | a << 24,
// 0); the colours are zero when lut256 is null. A level test and the level's colour are then one
// load in the shader, from a uniform array of vec4, whose 16-byte stride WGSL requires.
inline void fill_block(const Keys& k, float vmin, float vmax, const uint32_t* lut256, shaders::Cmap* out) {
    std::memset(out, 0, sizeof *out);
    out->vmin = vmin;
    out->vmax = vmax;
    out->flip = k.flip;
    for (uint32_t l = 0; l < 256; ++l) {
        out->levels[l][0] = k.key[l];
        out->levels[l][1] = k.key[l + 1];
        out->levels[l][2] = lut256 != nullptr ? lut256[l] : 0u;
    }
}

// The block of a built-in colormap (CALIPER_CMAP_*) and a range. False for an unknown colormap,
// whose block has its keys and zero colours.
inline bool derive_block(int32_t colormap, float vmin, float vmax, shaders::Cmap* out) {
    const uint32_t* lut = caliper_host::colormap_lut(colormap);
    fill_block(derive_keys(vmin, vmax), vmin, vmax, lut, out);
    return lut != nullptr;
}

// ---- The decision, as the shader takes it ------------------------------------------------------

// The first key of level k, 0 to 256 (256: the end, kSentinel).
CALIPER_CMAP_HD inline uint32_t key_at(const shaders::Cmap& m, uint32_t k) {
    return k < 256 ? m.levels[k][0] : m.levels[255][1];
}
// Whether a key lies in level k, from that level's entry alone.
CALIPER_CMAP_HD inline bool level_ok(const shaders::Cmap& m, uint32_t key, uint32_t k) {
    return key >= m.levels[k][0] && key < m.levels[k][1];
}
// The packed colour of level k.
CALIPER_CMAP_HD inline uint32_t level_rgba(const shaders::Cmap& m, uint32_t k) { return m.levels[k][2]; }

// The usual float formula, which only chooses which comparisons run. WGSL's clamp and u32()
// saturate; here NaN and out-of-range values are clamped first, so the conversion is defined.
CALIPER_CMAP_HD inline uint32_t guess_level(float v, float vmin, float vmax) {
    float t = (v - vmin) / (vmax - vmin);
    if (!(t >= 0.0f)) t = 0.0f;
    else if (t > 1.0f) t = 1.0f;
    const uint32_t g = uint32_t(t * 255.0f + 0.5f);
    return g < 255u ? g : 255u;
}

// The level of a value's bits: NaN from the bits, then the guess if two integer comparisons
// confirm it, then its two neighbours, then a binary search of eight integer steps.
CALIPER_CMAP_HD inline uint32_t colormap_index(uint32_t bits, const shaders::Cmap& m) {
    if (nan_bits(bits)) return 0;
    const uint32_t key = key_of(bits) ^ m.flip;
    float v;
    std::memcpy(&v, &bits, 4);
    const uint32_t g = guess_level(v, m.vmin, m.vmax);
    if (level_ok(m, key, g)) return g;
    if (g > 0 && level_ok(m, key, g - 1)) return g - 1;
    if (g < 255 && level_ok(m, key, g + 1)) return g + 1;
    uint32_t lo = 0, hi = 256;   // level lo starts at or below key, level hi above it
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        if (key >= m.levels[mid][0]) lo = mid; else hi = mid;
    }
    return lo;
}

// The packed colour of a value's bits, r | g << 8 | b << 16 | a << 24.
CALIPER_CMAP_HD inline uint32_t colormap_rgba(uint32_t bits, const shaders::Cmap& m) {
    return level_rgba(m, colormap_index(bits, m));
}

// ---- Mappings ----------------------------------------------------------------------------------

// The flags of a mapping that take a range end from statistics, with the bits of the snapshot
// flags (CALIPER_LIVE_SNAP_AUTO_VMIN and _VMAX, CALIPER_NET_SNAP_AUTO_VMIN and _VMAX).
enum : uint32_t { kAutoVmin = 1u << 0, kAutoVmax = 1u << 1 };

struct Mapping {
    int32_t colormap = 0;            // CALIPER_CMAP_*
    float vmin = 0.0f, vmax = 1.0f;  // the fixed ends
    uint32_t flags = 0;              // kAutoVmin | kAutoVmax
};
struct Range { float vmin = 0.0f, vmax = 1.0f; };

// The range a mapping draws with: an end whose flag is set from stats[0] (vmin) or stats[1]
// (vmax), the other from the mapping. False when a flag names a statistic that stats lacks.
// The keys follow the reference for any range, so a decreasing, degenerate, infinite or NaN end
// needs no special case here; a view that widens a flat range or keeps its last finite one
// does so before it asks for the block.
inline bool resolve_range(const Mapping& m, const float* stats, int32_t stats_count, Range* out) {
    Range r{m.vmin, m.vmax};
    if (m.flags & kAutoVmin) {
        if (stats == nullptr || stats_count < 1) return false;
        r.vmin = stats[0];
    }
    if (m.flags & kAutoVmax) {
        if (stats == nullptr || stats_count < 2) return false;
        r.vmax = stats[1];
    }
    *out = r;
    return true;
}

// ---- The cache ---------------------------------------------------------------------------------

// The key blocks of one owner (a view, or the layer's texture path), by colormap and the exact
// bits of the range, the least recently used dropped beyond the capacity. A miss costs one
// derivation (about 8,000 evaluations of the reference); a hit costs a hash lookup. Not thread
// safe: one owner uses it from one thread.
class KeyBlockCache {
public:
    struct Stats { uint64_t hits = 0, misses = 0, evictions = 0; };

    explicit KeyBlockCache(size_t capacity = 64) : capacity_(capacity == 0 ? 1 : capacity) {}

    // The block of a colormap and a range, or null for an unknown colormap. The pointer stays
    // valid until the block is evicted or the cache is cleared.
    const shaders::Cmap* block(int32_t colormap, float vmin, float vmax) {
        const Key key{colormap, bits(vmin), bits(vmax)};
        auto it = index_.find(key);
        if (it != index_.end()) {
            ++stats_.hits;
            entries_.splice(entries_.begin(), entries_, it->second);
            return it->second->known ? &it->second->block : nullptr;
        }
        ++stats_.misses;
        if (entries_.size() >= capacity_) {
            index_.erase(entries_.back().key);
            entries_.pop_back();
            ++stats_.evictions;
        }
        entries_.emplace_front();
        Entry& e = entries_.front();
        e.key = key;
        e.known = derive_block(colormap, vmin, vmax, &e.block);
        index_[key] = entries_.begin();
        return e.known ? &e.block : nullptr;
    }

    // The block of a mapping, its ends resolved from stats as resolve_range does. Null when a
    // statistic is missing or the colormap is unknown.
    const shaders::Cmap* block(const Mapping& m, const float* stats = nullptr, int32_t stats_count = 0) {
        Range r;
        if (!resolve_range(m, stats, stats_count, &r)) return nullptr;
        return block(m.colormap, r.vmin, r.vmax);
    }

    // Copies the block into out; for an unknown colormap, a block with the range and zero
    // levels, which draws transparent black.
    void fill(int32_t colormap, float vmin, float vmax, shaders::Cmap* out) {
        if (const shaders::Cmap* b = block(colormap, vmin, vmax)) { *out = *b; return; }
        std::memset(out, 0, sizeof *out);
        out->vmin = vmin;
        out->vmax = vmax;
    }

    void clear() { index_.clear(); entries_.clear(); }
    size_t size() const { return entries_.size(); }
    size_t capacity() const { return capacity_; }
    const Stats& stats() const { return stats_; }

private:
    struct Key {
        int32_t colormap;
        uint32_t vmin_bits, vmax_bits;
        bool operator==(const Key& o) const {
            return colormap == o.colormap && vmin_bits == o.vmin_bits && vmax_bits == o.vmax_bits;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            uint64_t h = (uint64_t(k.vmin_bits) << 32 | k.vmax_bits) * 0x9e3779b97f4a7c15ull;
            return size_t(h ^ (h >> 29) ^ uint64_t(uint32_t(k.colormap)) * 0xc2b2ae3d27d4eb4full);
        }
    };
    struct Entry { Key key{}; bool known = false; shaders::Cmap block{}; };
    static uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

    size_t capacity_;
    std::list<Entry> entries_;   // most recently used first
    std::unordered_map<Key, std::list<Entry>::iterator, KeyHash> index_;
    Stats stats_;
};

// A colormap source for the scene core's BackendResources (src/host/scene/backend_resources.h),
// which a view builds with the cache it owns: each block from the cache.
inline std::function<void(uint32_t, float, float, shaders::Cmap*)> colormap_source(std::shared_ptr<KeyBlockCache> cache) {
    return [cache = std::move(cache)](uint32_t colormap, float vmin, float vmax, shaders::Cmap* out) {
        cache->fill(int32_t(colormap), vmin, vmax, out);
    };
}

}  // namespace cmap
}  // namespace caliper_gfx
