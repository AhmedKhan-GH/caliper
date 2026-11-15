// live_clock.h: maps a producer's snapshot stamps into the viewer's clock
// (host-internal, header-only, portable).
//
// The live views compute a snapshot's latency as (apply time - stamp_ns) on
// the viewer's own clock (live_views.cpp now_ns()). A producer on the same
// machine stamps with that clock, so the shared-memory session needs nothing.
// A producer reached over the network stamps with ITS monotonic clock
// (Python's time.monotonic_ns()), whose zero is unrelated to the viewer's, so
// a network Source maps every stamp through this class before it hands a
// snapshot over.
//
// Wire protocol v3 carries no clock exchange, so the offset is estimated from
// the stamps alone: for each message, stamp - arrival = (producer clock -
// viewer clock) - one-way delay, and the largest value seen is the message with
// the least delay. The mapping therefore sets that fastest message's latency to
// zero: the latency shown is the delay ABOVE the fastest recent message (the
// queueing, rate limiting and frame delays a viewer sees), not including a
// constant transit time. On loopback that transit time is well under a
// millisecond. The estimate covers the last kWindow seconds in one-second
// buckets, so a clock that drifts or a session that restarts is followed.
#pragma once

#include <cstdint>

namespace caliper_live {

class StampClock {
public:
    static constexpr int kWindow = 16;   // seconds the estimate looks back

    // Note one message: its producer stamp and the viewer time it arrived (ns).
    void observe(uint64_t stamp_ns, uint64_t arrival_ns) {
        if (!stamp_ns || !arrival_ns) return;
        const int64_t d = (int64_t)(stamp_ns - arrival_ns);   // wraps like the clocks
        const uint64_t sec = arrival_ns / 1000000000ull;
        Bucket& b = buckets_[sec % kWindow];
        if (b.sec != sec || !b.used) {
            b.sec = sec;
            b.used = true;
            b.max_d = d;
        } else if (d > b.max_d) {
            b.max_d = d;
        }
        // The largest value among the buckets of the last kWindow seconds.
        bool any = false;
        int64_t best = 0;
        for (const Bucket& x : buckets_) {
            if (!x.used || sec - x.sec >= (uint64_t)kWindow) continue;
            if (!any || x.max_d > best) best = x.max_d;
            any = true;
        }
        offset_ = best;
        have_ = any;
    }

    bool ready() const { return have_; }
    // The producer's clock minus the viewer's, plus the least one-way delay.
    int64_t offset_ns() const { return offset_; }

    // A producer stamp in the viewer's clock (0 stays 0: unknown).
    uint64_t to_viewer(uint64_t stamp_ns) const {
        if (!stamp_ns || !have_) return 0;
        return (uint64_t)((int64_t)stamp_ns - offset_);
    }

private:
    struct Bucket {
        uint64_t sec = 0;
        int64_t max_d = 0;
        bool used = false;
    };
    Bucket buckets_[kWindow];
    int64_t offset_ = 0;
    bool have_ = false;
};

}  // namespace caliper_live
