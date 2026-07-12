// Tests for caliper_host::FeedStore — the per-channel ring-buffer store behind
// caliper.feed.v1 (feed spec §4). TDD: written before the implementation.
//
// The load-bearing guarantees exercised here are the cursor/read contract
// (tail-start, oldest-first, cursor advance), the honest-loss GAP contract
// (overflow a small ring and observe the seq jump), and thread safety (a writer
// injecting samples while a reader loops). REQUIRE/CHECK operands are kept
// simple (values pulled into locals first) so MSVC re-runs them cleanly.
#include <doctest/doctest.h>

#include "feed_store.h"

#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using caliper_host::FeedStore;

// A CaliperFeedChannelInfo the caller pre-fills with sentinel bytes so a
// refused channel_info() (return 0) can be proven to leave it UNTOUCHED.
static CaliperFeedChannelInfo poisoned_info() {
    CaliperFeedChannelInfo info;
    std::memset(&info, 0xA5, sizeof info);
    info.struct_size = sizeof(CaliperFeedChannelInfo);
    return info;
}

TEST_CASE("feed: a fresh store is inert — no channels, caps 0") {
    FeedStore store;
    uint32_t count = store.channel_count();
    uint32_t caps = store.caps();
    CHECK(count == 0u);
    CHECK(caps == 0u);   // CALIPER_FEED_CAP_LIVE unset with no channels
}

TEST_CASE("feed: add_channel registers, caps flips to LIVE, info round-trips") {
    FeedStore store;
    bool added = store.add_channel("sys.cpu.util", "CPU Utilization", "%",
                                   10.0f, 4096);
    CHECK(added);
    uint32_t count = store.channel_count();
    uint32_t caps = store.caps();
    CHECK(count == 1u);
    CHECK(caps == CALIPER_FEED_CAP_LIVE);

    CaliperFeedChannelInfo info = poisoned_info();
    uint32_t ok = store.channel_info(0, &info);
    CHECK(ok == 1u);
    CHECK(std::string(info.id) == "sys.cpu.util");
    CHECK(std::string(info.name) == "CPU Utilization");
    CHECK(std::string(info.units) == "%");
    CHECK(info.nominal_hz == doctest::Approx(10.0f));
}

TEST_CASE("feed: add_channel rejects empty id, zero capacity, and duplicates") {
    FeedStore store;
    bool empty_id = store.add_channel("", "n", "u", 1.0f, 16);
    bool zero_cap = store.add_channel("c", "n", "u", 1.0f, 0);
    CHECK_FALSE(empty_id);
    CHECK_FALSE(zero_cap);

    bool first = store.add_channel("dup", "n", "u", 1.0f, 16);
    bool second = store.add_channel("dup", "n2", "u2", 2.0f, 32);
    CHECK(first);
    CHECK_FALSE(second);
    uint32_t count = store.channel_count();
    CHECK(count == 1u);   // the duplicate never registered
}

TEST_CASE("feed: channel_info refuses bad index and bad size, leaving info untouched") {
    FeedStore store;
    store.add_channel("c", "name", "u", 5.0f, 16);

    // Bad index: only index 0 exists.
    CaliperFeedChannelInfo bad_index = poisoned_info();
    uint32_t r_index = store.channel_info(1, &bad_index);
    CHECK(r_index == 0u);
    // Untouched: the sentinel 0xA5 bytes survive in id[0] and name[0].
    CHECK(static_cast<unsigned char>(bad_index.id[0]) == 0xA5u);
    CHECK(static_cast<unsigned char>(bad_index.name[0]) == 0xA5u);

    // Bad size: struct_size too small for the host to fill safely.
    CaliperFeedChannelInfo bad_size = poisoned_info();
    bad_size.struct_size = 4;   // < sizeof(CaliperFeedChannelInfo)
    uint32_t r_size = store.channel_info(0, &bad_size);
    CHECK(r_size == 0u);
    CHECK(static_cast<unsigned char>(bad_size.id[0]) == 0xA5u);
    CHECK(static_cast<unsigned char>(bad_size.name[0]) == 0xA5u);
}

TEST_CASE("feed: push assigns per-channel seq monotonic from 1; unknown -> 0") {
    FeedStore store;
    store.add_channel("c", "n", "u", 1.0f, 16);
    uint64_t s1 = store.push("c", 100, 1.0f);
    uint64_t s2 = store.push("c", 200, 2.0f);
    uint64_t s3 = store.push("c", 300, 3.0f);
    CHECK(s1 == 1u);
    CHECK(s2 == 2u);
    CHECK(s3 == 3u);

    uint64_t ghost = store.push("nope", 400, 4.0f);
    CHECK(ghost == 0u);   // unknown channel
}

TEST_CASE("feed: unknown channel_id read returns 0 and leaves the cursor untouched") {
    FeedStore store;
    store.add_channel("c", "n", "u", 1.0f, 16);
    store.push("c", 1, 1.0f);

    CaliperFeedSample buf[8];
    uint64_t cursor = 42;   // an arbitrary caller cursor
    uint32_t n = store.read("does.not.exist", buf, 8, &cursor);
    CHECK(n == 0u);
    CHECK(cursor == 42u);   // untouched
}

TEST_CASE("feed: reading an empty channel returns 0, cursor untouched") {
    FeedStore store;
    store.add_channel("c", "n", "u", 1.0f, 16);

    CaliperFeedSample buf[8];
    uint64_t cursor = 0;
    uint32_t n = store.read("c", buf, 8, &cursor);
    CHECK(n == 0u);
    CHECK(cursor == 0u);
}

TEST_CASE("feed: tail read (cursor==0) returns at most `max` NEWEST, oldest-first, cursor advances") {
    FeedStore store;
    store.add_channel("c", "n", "u", 1.0f, 4096);
    for (int i = 1; i <= 10; ++i)
        store.push("c", i * 10, static_cast<float>(i));

    CaliperFeedSample buf[8];
    uint64_t cursor = 0;
    uint32_t n = store.read("c", buf, 3, &cursor);   // want the 3 newest
    CHECK(n == 3u);
    // Newest three are seqs 8,9,10, delivered oldest-first.
    CHECK(buf[0].seq == 8u);
    CHECK(buf[1].seq == 9u);
    CHECK(buf[2].seq == 10u);
    // Oldest-first ordering, strictly increasing.
    bool ordered = buf[0].seq < buf[1].seq && buf[1].seq < buf[2].seq;
    CHECK(ordered);
    // Value fidelity: value(seq) = seq.
    CHECK(buf[2].value == doctest::Approx(10.0f));
    // Cursor advanced to the last copied seq.
    CHECK(cursor == 10u);

    // Immediately polling again with the advanced cursor: caught up -> 0,
    // cursor untouched.
    uint32_t again = store.read("c", buf, 8, &cursor);
    CHECK(again == 0u);
    CHECK(cursor == 10u);
}

TEST_CASE("feed: incremental read from a cursor is oldest-first, bounded by max, no overflow") {
    FeedStore store;
    store.add_channel("c", "n", "u", 1.0f, 100);   // roomy: no eviction
    for (int i = 1; i <= 10; ++i)
        store.push("c", i, static_cast<float>(i));

    // A reader that has consumed up to seq 2 and asks for the next 3.
    CaliperFeedSample buf[8];
    uint64_t cursor = 2;
    uint32_t n = store.read("c", buf, 3, &cursor);
    CHECK(n == 3u);
    CHECK(buf[0].seq == 3u);
    CHECK(buf[1].seq == 4u);
    CHECK(buf[2].seq == 5u);
    CHECK(cursor == 5u);   // advanced to last copied, not to newest

    // Next window continues contiguously.
    uint32_t n2 = store.read("c", buf, 3, &cursor);
    CHECK(n2 == 3u);
    CHECK(buf[0].seq == 6u);
    CHECK(cursor == 8u);
}

TEST_CASE("feed: GAP contract — overflowing a small ring makes the seq jump observable") {
    FeedStore store;
    store.add_channel("c", "n", "u", 1.0f, 4);   // tiny ring: 4 newest only
    for (int i = 1; i <= 4; ++i)
        store.push("c", i, static_cast<float>(i));

    // A reader catches up to seq 4.
    CaliperFeedSample buf[16];
    uint64_t cursor = 0;
    uint32_t first = store.read("c", buf, 16, &cursor);
    CHECK(first == 4u);
    CHECK(cursor == 4u);

    // Six more samples arrive (seqs 5..10). The ring holds only the newest 4
    // (seqs 7,8,9,10); seqs 5 and 6 were overwritten.
    for (int i = 5; i <= 10; ++i)
        store.push("c", i, static_cast<float>(i));

    uint32_t n = store.read("c", buf, 16, &cursor);
    CHECK(n == 4u);                 // documented capacity: at most 4 buffered
    // The GAP: the reader's cursor was 4, so it expected seq 5 next — but the
    // oldest surviving sample is seq 7. The returned seqs JUMP past 5,6.
    CHECK(buf[0].seq == 7u);
    uint64_t expected_next = 5u;    // cursor(4) + 1
    bool gap_observed = buf[0].seq > expected_next;
    CHECK(gap_observed);
    CHECK(buf[3].seq == 10u);
    CHECK(cursor == 10u);

    // A fresh tail reader also sees exactly the newest 4 — the ring never keeps
    // more than its capacity.
    uint64_t fresh = 0;
    uint32_t m = store.read("c", buf, 100, &fresh);
    CHECK(m == 4u);
    CHECK(buf[0].seq == 7u);
    CHECK(buf[3].seq == 10u);
}

TEST_CASE("feed: two readers with independent cursors progress independently") {
    FeedStore store;
    store.add_channel("c", "n", "u", 1.0f, 100);
    for (int i = 1; i <= 10; ++i)
        store.push("c", i, static_cast<float>(i));

    CaliperFeedSample buf[128];
    uint64_t a = 0, b = 0;

    // Reader A drains everything.
    uint32_t na = store.read("c", buf, 100, &a);
    CHECK(na == 10u);
    CHECK(a == 10u);

    // Reader B, on its OWN cursor, takes only 4 (tail read of the newest 4).
    uint32_t nb = store.read("c", buf, 4, &b);
    CHECK(nb == 4u);
    CHECK(b == 10u);
    CHECK(buf[0].seq == 7u);

    // More samples arrive; each reader continues from where IT left off.
    for (int i = 11; i <= 15; ++i)
        store.push("c", i, static_cast<float>(i));

    uint32_t na2 = store.read("c", buf, 100, &a);
    CHECK(na2 == 5u);
    CHECK(buf[0].seq == 11u);
    CHECK(a == 15u);

    uint32_t nb2 = store.read("c", buf, 2, &b);
    CHECK(nb2 == 2u);
    CHECK(buf[0].seq == 11u);
    CHECK(b == 12u);       // B lags A, entirely independently
}

TEST_CASE("feed: thread race — a writer injects while a reader loops (bounded, deterministic)") {
    FeedStore store;
    constexpr int N = 4000;
    // Capacity == N so nothing is ever evicted while the writer produces N
    // samples: the run is loss-free, so the reader must observe seqs 1..N
    // exactly and contiguously. max == N so a tail read never skips ahead.
    store.add_channel("race", "n", "u", 10.0f, N);

    std::atomic<bool> go{false};

    std::thread writer([&store, &go]() {
        while (!go.load()) { /* start barrier */ }
        for (int i = 1; i <= N; ++i)
            store.push("race", i, static_cast<float>(i));
    });

    std::vector<uint64_t> seen;
    seen.reserve(N);
    std::thread reader([&store, &go, &seen]() {
        while (!go.load()) { /* start barrier */ }
        std::vector<CaliperFeedSample> buf(N);
        uint64_t cursor = 0;
        // Bounded: exits the moment N are collected; the guard only trips on a
        // bug (a stuck cursor), turning a hang into a loud failure.
        for (long guard = 0; guard < 50'000'000 && seen.size() < (size_t)N; ++guard) {
            uint32_t n = store.read("race", buf.data(),
                                    static_cast<uint32_t>(buf.size()), &cursor);
            for (uint32_t i = 0; i < n; ++i)
                seen.push_back(buf[i].seq);
        }
    });

    go.store(true);
    writer.join();
    reader.join();

    // Join-before-assert: the reader collected every seq, once, in order.
    REQUIRE(seen.size() == static_cast<size_t>(N));
    bool contiguous = true;
    for (int i = 0; i < N; ++i)
        if (seen[i] != static_cast<uint64_t>(i + 1)) { contiguous = false; break; }
    CHECK(contiguous);   // 1,2,3,...,N with no gap, no duplicate, no reorder
}
