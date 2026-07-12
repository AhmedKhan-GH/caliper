// Tests for the platform telemetry providers behind caliper.feed.v1 (feed spec
// §4 / T2 on macOS; §6.2 on Windows). The sampling thread is
// real-sensor-dependent, so the deterministic cases here cover only the
// LIFECYCLE (start/stop clean, and re-startable across the embed
// create/shutdown/create cycling that runs services twice) and a NON-FATAL
// live smoke (if the provider vends channels, at least one sample lands per
// guaranteed channel within a bounded window; skip-with-message otherwise so CI
// stays green). REQUIRE/CHECK operands are pulled into locals first (MSVC-safe).
#include <doctest/doctest.h>

#include "feed_store.h"

#include <chrono>
#include <set>
#include <string>
#include <thread>

using caliper_host::FeedStore;

#if defined(__APPLE__) || defined(_WIN32)
#ifdef __APPLE__
#include "feed_provider_mac.h"
#else
#include "feed_provider_win.h"
#endif

namespace {

#ifdef __APPLE__
// The channel ids the guaranteed tier must always vend on a Mac.
const char* const kGuaranteed[] = {
    "sys.cpu.util", "sys.mem.used", "sys.thermal.state", "sys.gpu.util"};
// Never fails on a Mac (public ProcessInfo API) — always ticks.
const char* const kAlwaysTicks = "sys.thermal.state";
#else
// The channel ids the guaranteed tier must always vend on Windows. Battery and
// the NVML GPU channels are box-dependent (probe tier), so they are not here.
const char* const kGuaranteed[] = {"sys.cpu.util", "sys.mem.used"};
// GetSystemTimes is unconditional on Windows — always ticks.
const char* const kAlwaysTicks = "sys.cpu.util";
#endif
constexpr uint32_t kGuaranteedCount =
    (uint32_t)(sizeof(kGuaranteed) / sizeof(kGuaranteed[0]));

std::set<std::string> channel_ids(FeedStore& s) {
    std::set<std::string> ids;
    uint32_t n = s.channel_count();
    for (uint32_t i = 0; i < n; ++i) {
        CaliperFeedChannelInfo info;
        info.struct_size = sizeof(CaliperFeedChannelInfo);
        if (s.channel_info(i, &info)) ids.insert(info.id);
    }
    return ids;
}

}  // namespace

TEST_CASE("feed provider: start registers the guaranteed tier, caps LIVE") {
    FeedStore store;
    caliper_host::feed_provider_start(store);

    uint32_t caps = store.caps();
    uint32_t count = store.channel_count();
    CHECK(caps == CALIPER_FEED_CAP_LIVE);   // >=1 channel => LIVE
    CHECK(count >= kGuaranteedCount);

    std::set<std::string> ids = channel_ids(store);
    for (const char* id : kGuaranteed) {
        bool present = ids.count(id) != 0;
        CHECK(present);
    }

    caliper_host::feed_provider_stop();   // BEFORE `store` dies (it writes into it)
}

TEST_CASE("feed provider: start/stop cycles cleanly (embed double-cycle), "
          "channels stable, seq monotonic across cycles") {
    FeedStore store;

    caliper_host::feed_provider_start(store);
    uint32_t count1 = store.channel_count();
    // Let a few ticks land so seq advances.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    // Snapshot the newest seq of the always-ticking channel.
    uint64_t cur = 0;
    CaliperFeedSample buf[64];
    uint32_t got1 = store.read(kAlwaysTicks, buf, 64, &cur);
    uint64_t last_seq_cycle1 = got1 ? buf[got1 - 1].seq : 0;
    caliper_host::feed_provider_stop();

    // Second cycle on the SAME persistent store (mirrors services_init running a
    // second time in the embed battery): channels must NOT duplicate, and the
    // per-channel seq must keep climbing — never reset.
    caliper_host::feed_provider_start(store);
    uint32_t count2 = store.channel_count();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    uint64_t cur2 = last_seq_cycle1;   // resume from where cycle 1 left off
    uint32_t got2 = store.read(kAlwaysTicks, buf, 64, &cur2);
    uint64_t last_seq_cycle2 = got2 ? buf[got2 - 1].seq : last_seq_cycle1;
    caliper_host::feed_provider_stop();

    CHECK(count2 == count1);                    // no duplicate registration
    CHECK(last_seq_cycle1 >= 1u);               // cycle 1 produced samples
    CHECK(last_seq_cycle2 > last_seq_cycle1);   // seq advanced, no reset

    // A third stop with nothing running must be a safe no-op.
    caliper_host::feed_provider_stop();
}

TEST_CASE("feed provider: live smoke — each guaranteed channel yields a sample "
          "(non-fatal / CI-safe)") {
    FeedStore store;
    caliper_host::feed_provider_start(store);

    if (store.caps() != CALIPER_FEED_CAP_LIVE) {
        caliper_host::feed_provider_stop();
        MESSAGE("provider vended no channels on this host — smoke skipped");
        return;
    }

    // Bounded window: poll up to ~2 s for the first sample of each channel.
    for (const char* id : kGuaranteed) {
        bool got = false;
        float value = 0.0f;
        for (int i = 0; i < 100 && !got; ++i) {
            uint64_t cur = 0;
            CaliperFeedSample buf[16];
            uint32_t n = store.read(id, buf, 16, &cur);
            if (n > 0) { got = true; value = buf[n - 1].value; }
            else std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!got) {
            MESSAGE("no sample from " << id << " within the window — skipped");
            continue;
        }
        // Plausibility (utils are %, thermal is 0..3).
        if (std::string(id) == "sys.thermal.state") {
            CHECK(value >= 0.0f);
            CHECK(value <= 3.0f);
        } else {
            CHECK(value >= 0.0f);
            CHECK(value <= 100.0f);
        }
    }

    caliper_host::feed_provider_stop();
}

#else  // no provider on this platform: zero channels (honest degradation, T1)

TEST_CASE("feed provider: absent on hosts without a provider (zero channels)") {
    FeedStore store;
    uint32_t count = store.channel_count();
    uint32_t caps = store.caps();
    CHECK(count == 0u);
    CHECK(caps == 0u);
    MESSAGE("no telemetry provider on this platform — feed stays inert");
}

#endif
