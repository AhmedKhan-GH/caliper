// Tests for pulsescope::PulseRing — the pure ring-copy + honest-loss GAP logic
// behind the PulseScope applet (feed spec §5). Rides the fast caliper_tests
// suite (no torch, no UI). REQUIRE/CHECK operands are pulled into locals first
// so MSVC re-evaluates them cleanly (house rule).
#include <doctest/doctest.h>

#include "pulse_ring.h"

#include <cstdint>
#include <string>
#include <vector>

using pulsescope::PulseRing;

// Build a sample as the feed vends them: monotonic seq, host-clock t_ns, value.
static CaliperFeedSample mk(uint64_t seq, int64_t t_ns, float value) {
    CaliperFeedSample s;
    s.seq = seq;
    s.t_ns = t_ns;
    s.value = value;
    s.reserved0 = 0.0f;
    return s;
}

TEST_CASE("pulse_ring: a fresh ring is empty, no gaps, no last value") {
    PulseRing ring;
    bool empty = ring.empty();
    std::size_t size = ring.size();
    uint64_t gaps = ring.gap_total();
    CHECK(empty);
    CHECK(size == 0u);
    CHECK(gaps == 0u);
}

TEST_CASE("pulse_ring: ingest copies oldest-first values and computes relative-time x") {
    PulseRing ring;
    const int64_t epoch = 1'000'000'000;  // 1e9 ns origin
    std::vector<CaliperFeedSample> batch = {
        mk(1, epoch,                     10.0f),
        mk(2, epoch +   500'000'000,     20.0f),  // +0.5 s
        mk(3, epoch + 1'000'000'000,     30.0f),  // +1.0 s
    };
    uint64_t lost = ring.ingest(batch.data(), 3, epoch);
    CHECK(lost == 0u);

    std::size_t size = ring.size();
    CHECK(size == 3u);
    // x is seconds relative to the epoch.
    CHECK(ring.xs()[0] == doctest::Approx(0.0));
    CHECK(ring.xs()[1] == doctest::Approx(0.5));
    CHECK(ring.xs()[2] == doctest::Approx(1.0));
    // y is the raw channel value, oldest-first.
    CHECK(ring.ys()[0] == doctest::Approx(10.0));
    CHECK(ring.ys()[2] == doctest::Approx(30.0));
    float last = ring.last_value();
    uint64_t last_seq = ring.last_seq();
    CHECK(last == doctest::Approx(30.0f));
    CHECK(last_seq == 3u);
}

TEST_CASE("pulse_ring: the FIRST batch never counts as loss even if seq starts high") {
    // A fresh tail read legitimately starts at newest-minus-max (e.g. seq 97),
    // NOT at seq 1 — that is not a gap. last_seq_==0 guards against a false gap.
    PulseRing ring;
    const int64_t epoch = 0;
    std::vector<CaliperFeedSample> batch = {
        mk(97, 0,               1.0f),
        mk(98, 100'000'000,     2.0f),
        mk(99, 200'000'000,     3.0f),
    };
    uint64_t lost = ring.ingest(batch.data(), 3, epoch);
    uint64_t gaps = ring.gap_total();
    CHECK(lost == 0u);   // NOT 96
    CHECK(gaps == 0u);
}

TEST_CASE("pulse_ring: contiguous batches across polls report no gap") {
    PulseRing ring;
    const int64_t epoch = 0;
    std::vector<CaliperFeedSample> b1 = {mk(1, 0, 1.0f), mk(2, 1, 2.0f)};
    std::vector<CaliperFeedSample> b2 = {mk(3, 2, 3.0f), mk(4, 3, 4.0f)};
    ring.ingest(b1.data(), 2, epoch);
    uint64_t lost = ring.ingest(b2.data(), 2, epoch);
    uint64_t gaps = ring.gap_total();
    std::size_t size = ring.size();
    CHECK(lost == 0u);
    CHECK(gaps == 0u);
    CHECK(size == 4u);
}

TEST_CASE("pulse_ring: a seq jump between polls is counted as lost samples (honest loss)") {
    PulseRing ring;
    const int64_t epoch = 0;
    std::vector<CaliperFeedSample> b1 = {mk(1, 0, 1.0f), mk(2, 1, 2.0f), mk(3, 2, 3.0f)};
    ring.ingest(b1.data(), 3, epoch);
    // Next poll: the host ring overwrote seqs 4,5 — read resumes at seq 6.
    std::vector<CaliperFeedSample> b2 = {mk(6, 5, 6.0f), mk(7, 6, 7.0f)};
    uint64_t lost = ring.ingest(b2.data(), 2, epoch);
    uint64_t gaps = ring.gap_total();
    CHECK(lost == 2u);          // seqs 4 and 5 were lost
    CHECK(gaps == 2u);

    // A second gap accumulates onto the total.
    std::vector<CaliperFeedSample> b3 = {mk(10, 9, 10.0f)};  // lost 8,9
    uint64_t lost2 = ring.ingest(b3.data(), 1, epoch);
    uint64_t gaps2 = ring.gap_total();
    CHECK(lost2 == 2u);
    CHECK(gaps2 == 4u);         // cumulative
}

TEST_CASE("pulse_ring: capacity trims the oldest points, keeping the newest window") {
    PulseRing ring(4);   // tiny scrolling window
    const int64_t epoch = 0;
    for (uint64_t i = 1; i <= 10; ++i) {
        CaliperFeedSample s = mk(i, static_cast<int64_t>(i), static_cast<float>(i));
        ring.ingest(&s, 1, epoch);
    }
    std::size_t size = ring.size();
    CHECK(size == 4u);                       // never more than capacity
    // The newest 4 survive, oldest-first: values 7,8,9,10.
    CHECK(ring.ys()[0] == doctest::Approx(7.0));
    CHECK(ring.ys()[3] == doctest::Approx(10.0));
    float last = ring.last_value();
    CHECK(last == doctest::Approx(10.0f));
    // Trimming points does NOT lose the gap accounting (no gaps here).
    uint64_t gaps = ring.gap_total();
    CHECK(gaps == 0u);
}

TEST_CASE("pulse_ring: ingest of an empty/null batch is a no-op") {
    PulseRing ring;
    uint64_t a = ring.ingest(nullptr, 0, 0);
    CaliperFeedSample s = mk(1, 0, 1.0f);
    uint64_t b = ring.ingest(&s, 0, 0);   // n==0
    std::size_t size = ring.size();
    CHECK(a == 0u);
    CHECK(b == 0u);
    CHECK(size == 0u);
}

TEST_CASE("pulse_ring: thermal_word maps the 0..3 state code to its label") {
    CHECK(std::string(pulsescope::thermal_word(0)) == "nominal");
    CHECK(std::string(pulsescope::thermal_word(1)) == "fair");
    CHECK(std::string(pulsescope::thermal_word(2)) == "serious");
    CHECK(std::string(pulsescope::thermal_word(3)) == "critical");
    CHECK(std::string(pulsescope::thermal_word(9)) == "unknown");
    CHECK(std::string(pulsescope::thermal_word(-1)) == "unknown");
}
