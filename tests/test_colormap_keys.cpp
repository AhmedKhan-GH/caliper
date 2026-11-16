// The colormap by keys on the CPU (src/host/gfx/colormap_keys.h): the key of a float, the keys
// of each boundary range, the integer decision against the reference map_f32_to_rgba8 byte for
// byte on the boundary sets of tests/shaders/colormap_cases.h, the mapping's AUTO ends and the
// cache of key blocks. The same decision for all 2^32 inputs is caliper_colormap_exhaustive
// (label slow). CALIPER_CMAP_BENCH=1 runs a row that times a derivation and a cache hit.
#include <doctest/doctest.h>

#include "gfx/colormap_keys.h"
#include "shaders/colormap_cases.h"
#include "tensor_bridge.h"
#include <caliper/live_net.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

using namespace caliper_gfx;
using colormap_cases::as_bits;
using colormap_cases::as_float;

namespace {

uint64_t xorshift(uint64_t& s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 2685821657736338717ull;
}

const uint32_t* keys_of(const shaders::Cmap& c) { return &c.keys[0][0]; }

}  // namespace

static_assert(cmap::kAutoVmin == CALIPER_NET_SNAP_AUTO_VMIN && cmap::kAutoVmax == CALIPER_NET_SNAP_AUTO_VMAX,
              "a mapping's flags take the snapshot flags' bits");

TEST_CASE("colormap keys: the key orders floats numerically and bits_of inverts it") {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> sorted = {-inf, -FLT_MAX, -1e30f, -1.0f, -FLT_MIN, -1e-45f, -0.0f, 0.0f, 1e-45f,
                                       FLT_MIN, 1.0f, 1e30f, FLT_MAX, inf};
    for (size_t i = 1; i < sorted.size(); ++i)
        CHECK(cmap::key_of(as_bits(sorted[i - 1])) < cmap::key_of(as_bits(sorted[i])));
    CHECK(cmap::key_of(0xff800000u) == 0x007fffffu);   // -inf, the lowest
    CHECK(cmap::key_of(0x7f800000u) == 0xff800000u);   // +inf, the highest
    CHECK(cmap::key_of(0x80000000u) + 1 == cmap::key_of(0x00000000u));   // -0 just below +0
    uint64_t s = 1;
    int disordered = 0, uninverted = 0;
    for (int i = 0; i < 200000; ++i) {
        const uint32_t a = uint32_t(xorshift(s) >> 32), b = uint32_t(xorshift(s) >> 32);
        uninverted += cmap::bits_of(cmap::key_of(a)) != a;
        if (cmap::nan_bits(a) || cmap::nan_bits(b)) continue;
        const float fa = as_float(a), fb = as_float(b);
        if (fa < fb) disordered += !(cmap::key_of(a) < cmap::key_of(b));
        if (fa > fb) disordered += !(cmap::key_of(a) > cmap::key_of(b));
    }
    CHECK(disordered == 0);
    CHECK(uninverted == 0);
}

TEST_CASE("colormap keys: each range's keys start at 0, never fall, end in sentinels and flip only a decreasing range") {
    const std::vector<uint32_t> table = colormap_cases::lut();
    for (const auto& r : colormap_cases::ranges()) {
        INFO("range " << r.name);
        const shaders::Cmap c = colormap_cases::block(r, table);
        const uint32_t* k = keys_of(c);
        CHECK(k[0] == 0);
        for (uint32_t l = 1; l < cmap::kKeyCount; ++l) CHECK(k[l] >= k[l - 1]);
        for (uint32_t l = 256; l < cmap::kKeyCount; ++l) CHECK(k[l] == cmap::kSentinel);
        CHECK(as_bits(c.vmin) == as_bits(r.vmin));
        CHECK(as_bits(c.vmax) == as_bits(r.vmax));
        const bool decreasing = r.vmax < r.vmin && std::isfinite(r.vmin) && std::isfinite(r.vmax) &&
                                std::isfinite(r.vmax - r.vmin);
        CHECK(c.flip == (decreasing ? 0xffffffffu : 0u));
        // A degenerate range, and an infinite or NaN end or span, maps every value to level 0.
        const bool constant = !(r.vmax != r.vmin) || !std::isfinite(r.vmax - r.vmin);
        if (constant) {
            for (uint32_t l = 1; l < 256; ++l) CHECK(k[l] == cmap::kSentinel);
        } else {
            CHECK(k[255] != cmap::kSentinel);   // a finite range reaches the top level
        }
        CHECK(std::memcmp(c.lut, table.data(), sizeof c.lut) == 0);
    }
}

TEST_CASE("colormap keys: the integer decision matches map_f32_to_rgba8 byte for byte on the boundary sets") {
    const std::vector<uint32_t> table = colormap_cases::lut();
    const auto rs = colormap_cases::ranges();
    size_t total = 0, edges = 0;
    for (size_t i = 0; i < rs.size(); ++i) {
        const auto& r = rs[i];
        INFO("range " << r.name);
        const shaders::Cmap c = colormap_cases::block(r, table);
        const std::vector<uint32_t> in = colormap_cases::inputs(r, c, colormap_cases::seed(i));
        const std::vector<uint32_t> want = colormap_cases::expected(r, in, table);
        size_t bad = 0, first = 0;
        for (size_t j = 0; j < in.size(); ++j)
            if (cmap::colormap_rgba(in[j], c) != want[j] && bad++ == 0) first = j;
        CHECK_MESSAGE(bad == 0, bad << " of " << in.size() << " differ, first input 0x" << std::hex << in[first]);
        // The index alone, through an identity read of the reference.
        size_t bad_index = 0;
        for (size_t j = 0; j < in.size(); ++j)
            bad_index += cmap::colormap_index(in[j], c) != cmap::reference_index(in[j], r.vmin, r.vmax);
        CHECK(bad_index == 0);
        total += in.size();
        for (uint32_t l = 1; l < 256; ++l) edges += keys_of(c)[l] != cmap::kSentinel;
    }
    MESSAGE(total << " inputs over " << rs.size() << " ranges, " << edges << " keys below the sentinel");
}

TEST_CASE("colormap keys: the float guess chooses comparisons but never the level") {
    // The same blocks with a guess that is wrong everywhere: the range the guess reads is
    // replaced, and the decision still matches the reference.
    const std::vector<uint32_t> table = colormap_cases::lut();
    const auto rs = colormap_cases::ranges();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float wrong[][2] = {{1.0f, 0.0f}, {nan, nan}, {-1e38f, 1e38f}, {0.25f, 0.2500001f}};
    for (size_t i = 0; i < rs.size(); ++i) {
        const auto& r = rs[i];
        const shaders::Cmap good = colormap_cases::block(r, table);
        const std::vector<uint32_t> in = colormap_cases::inputs(r, good, colormap_cases::seed(i));
        for (const auto& w : wrong) {
            INFO("range " << r.name << ", guess from (" << w[0] << ", " << w[1] << ")");
            shaders::Cmap c = good;
            c.vmin = w[0];
            c.vmax = w[1];
            size_t bad = 0;
            for (uint32_t b : in) bad += cmap::colormap_index(b, c) != cmap::colormap_index(b, good);
            CHECK(bad == 0);
        }
    }
}

TEST_CASE("colormap keys: a built-in colormap's block holds its table, and an unknown colormap is refused") {
    shaders::Cmap c{};
    CHECK(cmap::derive_block(CALIPER_CMAP_MAGMA, -1.0f, 3.0f, &c));
    CHECK(std::memcmp(c.lut, caliper_host::colormap_lut(CALIPER_CMAP_MAGMA), sizeof c.lut) == 0);
    CHECK(c.vmin == -1.0f);
    CHECK(c.vmax == 3.0f);
    float v[3] = {-1.0f, 1.0f, 3.0f};
    uint8_t px[12];
    caliper_host::map_f32_to_rgba8(v, 3, 1, caliper_host::colormap_lut(CALIPER_CMAP_MAGMA), -1.0f, 3.0f, px);
    for (int i = 0; i < 3; ++i) {
        uint32_t want;
        std::memcpy(&want, px + 4 * i, 4);
        CHECK(cmap::colormap_rgba(as_bits(v[i]), c) == want);
    }
    shaders::Cmap u{};
    CHECK_FALSE(cmap::derive_block(97, 0.0f, 1.0f, &u));
    CHECK(keys_of(u)[255] != cmap::kSentinel);   // the keys are there; the table is zero
    for (uint32_t k = 0; k < 256; ++k) CHECK((&u.lut[0][0])[k] == 0);
}

TEST_CASE("colormap keys: a mapping takes an AUTO end from the statistics as the snapshot flags do") {
    const float stats[2] = {-2.0f, 6.0f};
    cmap::Range r;
    cmap::Mapping m{CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, 0};
    REQUIRE(cmap::resolve_range(m, nullptr, 0, &r));
    CHECK((r.vmin == 0.0f && r.vmax == 1.0f));
    m.flags = cmap::kAutoVmin;
    REQUIRE(cmap::resolve_range(m, stats, 1, &r));
    CHECK((r.vmin == -2.0f && r.vmax == 1.0f));
    m.flags = cmap::kAutoVmax;
    CHECK_FALSE(cmap::resolve_range(m, stats, 1, &r));   // stats[1] is missing
    REQUIRE(cmap::resolve_range(m, stats, 2, &r));
    CHECK((r.vmin == 0.0f && r.vmax == 6.0f));
    m.flags = cmap::kAutoVmin | cmap::kAutoVmax;
    REQUIRE(cmap::resolve_range(m, stats, 2, &r));
    CHECK((r.vmin == -2.0f && r.vmax == 6.0f));
    CHECK_FALSE(cmap::resolve_range(m, nullptr, 2, &r));
    cmap::KeyBlockCache cache;
    const shaders::Cmap* b = cache.block(m, stats, 2);
    REQUIRE(b != nullptr);
    CHECK((b->vmin == -2.0f && b->vmax == 6.0f));
    CHECK(cache.block(m, stats, 1) == nullptr);
}

TEST_CASE("colormap keys: the cache derives each mapping once, keeps exact bits apart and drops the least recently used") {
    cmap::KeyBlockCache cache(3);
    const shaders::Cmap* a = cache.block(CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f);
    REQUIRE(a != nullptr);
    shaders::Cmap fresh{};
    REQUIRE(cmap::derive_block(CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, &fresh));
    CHECK(std::memcmp(a, &fresh, sizeof fresh) == 0);
    CHECK(cache.block(CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f) == a);   // a hit, the same block
    CHECK(cache.stats().hits == 1);
    CHECK(cache.stats().misses == 1);
    // -0 and +0 are different bits, and so different entries; another colormap is another entry.
    const shaders::Cmap* z = cache.block(CALIPER_CMAP_VIRIDIS, -0.0f, 1.0f);
    REQUIRE(z != nullptr);
    CHECK(z != a);
    const shaders::Cmap* m = cache.block(CALIPER_CMAP_MAGMA, 0.0f, 1.0f);
    REQUIRE(m != nullptr);
    CHECK(std::memcmp(m->lut, caliper_host::colormap_lut(CALIPER_CMAP_MAGMA), sizeof m->lut) == 0);
    CHECK(cache.size() == 3);
    CHECK(cache.stats().misses == 3);
    // Use a again, then add a fourth: the least recently used (-0, 1) goes.
    CHECK(cache.block(CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f) == a);
    CHECK(cache.block(CALIPER_CMAP_RDBU, 0.0f, 1.0f) != nullptr);
    CHECK(cache.size() == 3);
    CHECK(cache.stats().evictions == 1);
    const uint64_t misses = cache.stats().misses;
    CHECK(cache.block(CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f) == a);    // still there
    CHECK(cache.block(CALIPER_CMAP_MAGMA, 0.0f, 1.0f) == m);      // still there
    CHECK(cache.stats().misses == misses);
    cache.block(CALIPER_CMAP_VIRIDIS, -0.0f, 1.0f);               // derived again
    CHECK(cache.stats().misses == misses + 1);
    // An unknown colormap: null, and fill gives a block with the range and nothing else.
    CHECK(cache.block(97, 0.0f, 1.0f) == nullptr);
    CHECK(cache.block(97, 0.0f, 1.0f) == nullptr);
    shaders::Cmap u;
    std::memset(&u, 0xab, sizeof u);
    cache.fill(97, 2.0f, 3.0f, &u);
    CHECK((u.vmin == 2.0f && u.vmax == 3.0f && u.flip == 0));
    CHECK(keys_of(u)[255] == 0);
    // The scene core's colormap source, over a shared cache.
    auto shared = std::make_shared<cmap::KeyBlockCache>();
    auto source = cmap::colormap_source(shared);
    shaders::Cmap s{};
    source(uint32_t(CALIPER_CMAP_VIRIDIS), 0.0f, 1.0f, &s);
    CHECK(std::memcmp(&s, &fresh, sizeof s) == 0);
    source(uint32_t(CALIPER_CMAP_VIRIDIS), 0.0f, 1.0f, &s);
    CHECK(shared->stats().hits == 1);
    cache.clear();
    CHECK(cache.size() == 0);
}

TEST_CASE("colormap keys: timing — one derivation and one cache hit") {
    if (std::getenv("CALIPER_CMAP_BENCH") == nullptr) return;
    using clock = std::chrono::steady_clock;
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto p95 = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[size_t(0.95 * (v.size() - 1))]; };
    double load[3] = {};
    getloadavg(load, 3);
    // The evaluations of the reference that a derivation makes, counted by repeating its searches
    // over the (flipped) keys of -inf to +inf, 0x007fffff to 0xff800000 whether flipped or not,
    // with the derived keys standing in for the reference.
    auto evaluations = [](float vmin, float vmax) {
        size_t n = 2;
        const shaders::Cmap c = colormap_cases::block({"", vmin, vmax}, colormap_cases::lut());
        uint32_t from = 0x007fffffu;
        for (uint32_t l = 1; l < 256; ++l) {
            if (keys_of(c)[l] == cmap::kSentinel) continue;
            uint32_t a = from, b = 0xff800000u;
            while (a < b) { const uint32_t mid = a + (b - a) / 2; ++n; if (mid >= keys_of(c)[l]) b = mid; else a = mid + 1; }
            from = a;
        }
        return n;
    };
    const float ranges[][2] = {{0.0f, 1.0f}, {-3.5f, 7.25f}, {2.0f, -1.0f}, {0.0f, FLT_MAX}};
    for (const auto& r : ranges) {
        std::vector<double> us;
        volatile uint32_t sink = 0;
        for (int i = 0; i < 400; ++i) {
            const auto t0 = clock::now();
            const cmap::Keys k = cmap::derive_keys(r[0], r[1]);
            const auto t1 = clock::now();
            sink = sink + k.key[128];
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::printf("[cmap-bench] derive_keys(%g, %g): median %.1f us, p95 %.1f us, %zu evaluations of the reference; load %.2f\n",
                    double(r[0]), double(r[1]), median(us), p95(us), evaluations(r[0], r[1]), load[0]);
    }
    // The hit path: the block's pointer, and the copy a view makes into its arena.
    cmap::KeyBlockCache cache;
    const float lo[4] = {0.0f, -1.0f, 2.0f, -3.5f};
    for (float l : lo) cache.block(CALIPER_CMAP_VIRIDIS, l, l + 1.0f);
    const int kHits = 1000000;
    shaders::Cmap out;
    std::vector<double> ns_block, ns_fill;
    for (int rep = 0; rep < 9; ++rep) {
        const auto t0 = clock::now();
        uintptr_t acc = 0;
        for (int i = 0; i < kHits; ++i) acc += uintptr_t(cache.block(CALIPER_CMAP_VIRIDIS, lo[i & 3], lo[i & 3] + 1.0f));
        const auto t1 = clock::now();
        uint32_t seen = 0;   // reads each copy, so that the copies are not elided
        for (int i = 0; i < kHits; ++i) {
            cache.fill(CALIPER_CMAP_VIRIDIS, lo[i & 3], lo[i & 3] + 1.0f, &out);
            seen += out.keys[(i * 7) & 63][i & 3] + out.lut[(i * 13) & 63][(i >> 2) & 3];
        }
        const auto t2 = clock::now();
        CHECK(acc != 0);
        CHECK(seen != 1);
        ns_block.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kHits);
        ns_fill.push_back(std::chrono::duration<double, std::nano>(t2 - t1).count() / kHits);
    }
    std::printf("[cmap-bench] cache hit: block() median %.1f ns, fill() with the 2,080-byte copy median %.1f ns; "
                "%llu hits, %llu misses; load %.2f\n", median(ns_block), median(ns_fill),
                (unsigned long long)cache.stats().hits, (unsigned long long)cache.stats().misses, load[0]);
    std::fflush(stdout);
}
