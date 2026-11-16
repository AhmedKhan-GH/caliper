// caliper_colormap_gpu_exhaustive: the generated colormap_texture on Metal, created through the
// graphics layer's Metal backend (so compiled exactly as the layer compiles it, fast math included,
// and dispatched with its own 16 x 8 workgroup), against the reference map_f32_to_rgba8 for all
// 2^32 input bit patterns on each range of tests/shaders/colormap_cases.h, with the test table
// (every channel varies, so equal bytes mean an equal level). It is the ctest entry of the same
// name, label slow, which exits 77 (skipped) unless CALIPER_SLOW_TESTS=1:
//
//     CALIPER_SLOW_TESTS=1 ctest -L slow --output-on-failure
//
// The method: the inputs go in 256 chunks of 2^24 (chunk c holds the bit patterns c << 24 to
// (c << 24) + 2^24 - 1, as a 4,096 x 4,096 image in a host-visible layer buffer, written once per
// chunk); for each chunk and each range, one dispatch writes a 4,096 x 4,096 rgba8unorm texture,
// the layer reads it back, and CPU threads compare every texel with the reference's bytes, computed
// slice by slice from the same buffer. Each dispatch runs in its own autorelease pool, so the
// Metal objects of a submit and its 64 MB readback are freed at once; peak resident memory stays
// under 500 MB (without the pool it grew past 30 GB). CALIPER_TEST_THREADS sets
// the CPU threads, 6 by default and at most 6. The first differing inputs are printed; the test
// is never adjusted to pass.
#include "gfx/colormap_keys.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "shaders/colormap_cases.h"
#include "tensor_bridge.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

using namespace caliper_gfx;
namespace sh = caliper_gfx::shaders;

namespace {

constexpr uint32_t kSide = 4096;                  // a chunk is kSide x kSide = 2^24 inputs
constexpr uint64_t kChunk = uint64_t(kSide) * kSide;

struct Mismatch { uint32_t bits, got, want; };

double peak_mb() {
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return double(u.ru_maxrss) / (1024.0 * 1024.0);   // bytes on macOS
}

}  // namespace

int main() {
    const char* slow = std::getenv("CALIPER_SLOW_TESTS");
    if (slow == nullptr || std::strcmp(slow, "1") != 0) {
        std::printf("skipped: the exhaustive GPU colormap test runs with CALIPER_SLOW_TESTS=1 (label slow)\n");
        return 77;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        std::printf("skipped: no Metal device\n");
        return 77;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::unique_ptr<Backend> gfx = make_metal_backend((__bridge void*)device, (__bridge void*)queue);
    int threads = 6;
    if (const char* t = std::getenv("CALIPER_TEST_THREADS")) threads = std::max(1, std::min(6, std::atoi(t)));

    const uint32_t prog = uint32_t(sh::ProgramId::colormap_texture);
    const sh::ProgramDescription& desc = sh::kProgramDescriptions[prog];
    Program program = gfx->create_program(sh::kMetalPrograms[prog]);
    ComputePipelineDesc cd;
    cd.program = program;
    cd.entry = "cs_colormap_texture";
    Pipeline pipe = gfx->create_compute_pipeline(cd);
    Buffer values = gfx->create_buffer({kChunk * 4, kStorage, true, "inputs"});
    Texture out = gfx->create_texture({kSide, kSide, Format::RGBA8Unorm, kStorageTex | kCopySrcTex, "colours"});
    if (!program || !pipe || !values || !out || gfx->mapped(values) == nullptr) {
        std::printf("FAILED: the layer refused the setup: %s\n", gfx->last_error().c_str());
        return 1;
    }
    BindGroupDesc g0;
    g0.program = program;
    g0.group = 0;
    g0.entry_count = 3;
    g0.entries[0].binding = 0;
    g0.entries[0].buffer = kArena;
    g0.entries[1].binding = 1;
    g0.entries[1].buffer = values;
    g0.entries[2].binding = 2;
    g0.entries[2].texture = out;
    BindGroupDesc g1;
    g1.program = program;
    g1.group = 1;
    g1.entry_count = 1;
    g1.entries[0].binding = 1;
    g1.entries[0].buffer = kArena;
    const BindGroup groups[2] = {gfx->create_bind_group(g0), gfx->create_bind_group(g1)};
    if (!groups[0] || !groups[1]) {
        std::printf("FAILED: %s\n", gfx->last_error().c_str());
        return 1;
    }
    const uint16_t* wg = desc.entries[0].workgroup_size;
    uint32_t* inputs = static_cast<uint32_t*>(gfx->mapped(values));

    const auto ranges = colormap_cases::ranges();
    const std::vector<uint32_t> table = colormap_cases::lut();
    std::vector<sh::Cmap> blocks;
    for (const auto& r : ranges) blocks.push_back(colormap_cases::block(r, table));
    std::vector<uint64_t> bad(ranges.size(), 0);
    std::vector<std::vector<Mismatch>> first(ranges.size());
    std::mutex first_mutex;

    double load[3] = {};
    getloadavg(load, 3);
    std::printf("caliper_colormap_gpu_exhaustive: colormap_texture through the layer (%s math, %u x %u groups), "
                "all 2^32 inputs on %zu ranges in 256 chunks of 4,096 x 4,096, %d CPU threads, load %.2f at the start\n",
                sh::kMetalPrograms[prog].fast_math ? "fast" : "safe", wg[0], wg[1], ranges.size(), threads, load[0]);
    std::fflush(stdout);
    const auto start = std::chrono::steady_clock::now();
    std::vector<uint8_t> got;
    // CALIPER_GPU_EXHAUSTIVE_CHUNKS limits the chunks, for a quick look; the test fails then.
    uint32_t chunks = 256;
    if (const char* c = std::getenv("CALIPER_GPU_EXHAUSTIVE_CHUNKS")) chunks = uint32_t(std::max(1, std::min(256, std::atoi(c))));
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        {   // the chunk's bit patterns, in parallel
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t)
                pool.emplace_back([&, t] {
                    const uint64_t a = kChunk * uint64_t(t) / uint64_t(threads), b = kChunk * uint64_t(t + 1) / uint64_t(threads);
                    for (uint64_t i = a; i < b; ++i) inputs[i] = (chunk << 24) | uint32_t(i);
                });
            for (auto& th : pool) th.join();
        }
        for (size_t r = 0; r < ranges.size(); ++r) @autoreleasepool {   // drains each submit's Metal objects
            caliper_gfx::Frame f;
            const sh::TexParams tp{kSide, kSide, 0, kSide};
            const uint32_t offsets[2] = {f.uniform(&tp, sizeof tp), f.uniform(&blocks[r], sizeof blocks[r])};
            f.dispatch(pipe, groups, offsets, (kSide + wg[0] - 1) / wg[0], (kSide + wg[1] - 1) / wg[1], 1);
            const Serial s = gfx->submit(std::move(f));
            Readback rb = gfx->read_texture(out, 0, 0, kSide, kSide);
            if (s == 0 || !rb || !gfx->wait(gfx->last_use(out)) || !gfx->take_readback(rb, &got) ||
                got.size() != kChunk * 4) {
                std::printf("FAILED: chunk %u, range %s: the layer did not run or read back: %s\n", chunk,
                            ranges[r].name, gfx->last_error().c_str());
                return 1;
            }
            std::atomic<uint64_t> range_bad{0};
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t)
                pool.emplace_back([&, t] {
                    constexpr uint32_t kSlice = 1u << 16;
                    std::vector<uint8_t> want(size_t(kSlice) * 4);
                    const uint64_t a = kChunk * uint64_t(t) / uint64_t(threads), b = kChunk * uint64_t(t + 1) / uint64_t(threads);
                    uint64_t n_bad = 0;
                    for (uint64_t at = a; at < b; at += kSlice) {
                        const uint32_t n = uint32_t(std::min<uint64_t>(kSlice, b - at));
                        caliper_host::map_f32_to_rgba8(reinterpret_cast<const float*>(inputs + at), int(n), 1, table.data(),
                                                       ranges[r].vmin, ranges[r].vmax, want.data());
                        if (std::memcmp(want.data(), got.data() + at * 4, size_t(n) * 4) == 0) continue;
                        for (uint32_t i = 0; i < n; ++i) {
                            uint32_t g, w;
                            std::memcpy(&g, got.data() + (at + i) * 4, 4);
                            std::memcpy(&w, want.data() + size_t(i) * 4, 4);
                            if (g == w) continue;
                            ++n_bad;
                            std::lock_guard<std::mutex> lock(first_mutex);
                            if (first[r].size() < 5) first[r].push_back({inputs[at + i], g, w});
                        }
                    }
                    range_bad += n_bad;
                });
            for (auto& th : pool) th.join();
            bad[r] += range_bad;
        }
        if ((chunk + 1) % 32 == 0 || chunk + 1 == chunks) {
            uint64_t total_bad = 0;
            for (uint64_t b : bad) total_bad += b;
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("  %u of 256 chunks: %llu differ so far, %.0f s, peak memory %.0f MB\n", chunk + 1,
                        (unsigned long long)total_bad, sec, peak_mb());
            std::fflush(stdout);
        }
    }
    int failed = 0;
    for (size_t r = 0; r < ranges.size(); ++r) {
        std::printf("  %-26s (%.9g, %.9g): %llu inputs, %llu differ\n", ranges[r].name, double(ranges[r].vmin),
                    double(ranges[r].vmax), (unsigned long long)(chunks * kChunk), (unsigned long long)bad[r]);
        for (const Mismatch& m : first[r])
            std::printf("    input 0x%08x: GPU 0x%08x, reference 0x%08x\n", m.bits, m.got, m.want);
        failed += bad[r] != 0;
    }
    if (chunks != 256) {
        std::printf("FAILED: only %u of 256 chunks ran (CALIPER_GPU_EXHAUSTIVE_CHUNKS)\n", chunks);
        failed = 1;
    }
    getloadavg(load, 3);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%s: %llu inputs over %zu ranges in %.0f s with %d CPU threads, peak memory %.0f MB, load %.2f at the end\n",
                failed ? "FAILED" : "passed", (unsigned long long)(uint64_t(ranges.size()) * chunks * kChunk), ranges.size(), sec,
                threads, peak_mb(), load[0]);
    return failed ? 1 : 0;
}
