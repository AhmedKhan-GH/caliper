// caliper_colormap_exhaustive: the colormap by keys (src/host/gfx/colormap_keys.h) against the
// reference map_f32_to_rgba8 for all 2^32 inputs on each range of tests/shaders/colormap_cases.h
// (design: renderer-core, "Exact colormap output"). It compares the level, so it proves the
// decision for every table. It is the ctest entry of the same name, label slow, which exits 77
// (skipped) unless CALIPER_SLOW_TESTS=1, so that the default run passes over it:
//
//     CALIPER_SLOW_TESTS=1 ctest -L slow --output-on-failure
//
// CALIPER_TEST_THREADS sets the threads, 6 by default and at most 6, the machine's share of a
// heavy job.
#include "gfx/colormap_keys.h"
#include "shaders/colormap_cases.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace caliper_gfx;

namespace {

struct Result { uint64_t compared = 0, bad = 0; uint32_t first_bad = 0; bool any_bad = false; };

// Inputs [begin, end) of one range: the reference through an identity table, a block at a time,
// against the integer decision.
void compare(const colormap_cases::Range& r, const shaders::Cmap& c, uint64_t begin, uint64_t end, Result* out) {
    constexpr uint32_t kBlock = 1u << 16;
    static const struct Identity {
        uint32_t t[256];
        Identity() { for (uint32_t k = 0; k < 256; ++k) t[k] = k; }
    } identity;
    std::vector<float> v(kBlock);
    std::vector<uint8_t> px(size_t(kBlock) * 4);
    for (uint64_t at = begin; at < end; at += kBlock) {
        const uint32_t n = uint32_t(std::min<uint64_t>(kBlock, end - at));
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t bits = uint32_t(at + i);
            std::memcpy(&v[i], &bits, 4);
        }
        caliper_host::map_f32_to_rgba8(v.data(), int(n), 1, identity.t, r.vmin, r.vmax, px.data());
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t bits = uint32_t(at + i);
            if (cmap::colormap_index(bits, c) != px[size_t(i) * 4]) {
                if (!out->any_bad) { out->any_bad = true; out->first_bad = bits; }
                ++out->bad;
            }
        }
        out->compared += n;
    }
}

}  // namespace

int main() {
    const char* slow = std::getenv("CALIPER_SLOW_TESTS");
    if (slow == nullptr || std::strcmp(slow, "1") != 0) {
        std::printf("skipped: the exhaustive colormap test runs with CALIPER_SLOW_TESTS=1 (label slow)\n");
        return 77;
    }
    int threads = 6;
    if (const char* t = std::getenv("CALIPER_TEST_THREADS")) threads = std::max(1, std::min(6, std::atoi(t)));
    double load[3] = {};
    getloadavg(load, 3);
    std::printf("caliper_colormap_exhaustive: all 2^32 inputs per range, %d threads, load %.2f at the start\n", threads,
                load[0]);
    std::fflush(stdout);
    const auto all = std::chrono::steady_clock::now();
    int failed = 0;
    uint64_t total = 0;
    for (const auto& r : colormap_cases::ranges()) {
        shaders::Cmap c{};
        cmap::fill_block(cmap::derive_keys(r.vmin, r.vmax), r.vmin, r.vmax, nullptr, &c);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Result> results(static_cast<size_t>(threads));
        std::vector<std::thread> pool;
        std::atomic<uint64_t> next{0};
        constexpr uint64_t kSlice = 1ull << 24;   // 256 slices, taken in turn
        for (int t = 0; t < threads; ++t)
            pool.emplace_back([&, t] {
                for (;;) {
                    const uint64_t s = next.fetch_add(kSlice);
                    if (s >= (1ull << 32)) break;
                    compare(r, c, s, s + kSlice, &results[size_t(t)]);
                }
            });
        for (auto& th : pool) th.join();
        Result sum;
        for (const Result& x : results) {
            sum.compared += x.compared;
            sum.bad += x.bad;
            if (x.any_bad && (!sum.any_bad || x.first_bad < sum.first_bad)) { sum.any_bad = true; sum.first_bad = x.first_bad; }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint32_t edges = 0;
        for (uint32_t l = 1; l < 256; ++l) edges += (&c.keys[0][0])[l] != cmap::kSentinel;
        std::printf("  %-26s (%.9g, %.9g): %llu inputs, %llu differ, %u keys below the sentinel, flip %s, %.1f s\n", r.name,
                    double(r.vmin), double(r.vmax), (unsigned long long)sum.compared, (unsigned long long)sum.bad,
                    edges, c.flip ? "yes" : "no", s);
        if (sum.any_bad) {
            std::printf("    first differing input 0x%08x: decision %u, reference %u\n", sum.first_bad,
                        cmap::colormap_index(sum.first_bad, c), cmap::reference_index(sum.first_bad, r.vmin, r.vmax));
        }
        std::fflush(stdout);
        failed += sum.bad != 0 || sum.compared != (1ull << 32);
        total += sum.compared;
    }
    getloadavg(load, 3);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - all).count();
    std::printf("%s: %llu inputs over %zu ranges in %.1f s with %d threads, load %.2f at the end\n",
                failed ? "FAILED" : "passed", (unsigned long long)total, colormap_cases::ranges().size(), s, threads, load[0]);
    return failed ? 1 : 0;
}
