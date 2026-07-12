#pragma once
// ============================================================================
// pulse_ring.h — the pure (torch-free, std::-only) logic behind PulseScope.
//
// PulseScope is CPU + ImPlot only: no tensors, no bridge, no geometry. The load-
// bearing logic that CAN go wrong lives here as std::-only code so the fast unit
// suite (caliper_tests) checks it WITHOUT a UI or a live sensor:
//   * PulseRing — the applet-side ring copy of one channel's newest samples,
//     with the honest-loss GAP accounting (a seq jump between polls == host-ring
//     overwrite == data lost, counted) and the shared relative-time axis
//     (seconds from a single epoch, so every channel aligns — spec §2/§5).
//   * thermal_word — the sys.thermal.state code (0..3) -> its word. The feed ABI
//     carries a bare float; the unitless nominal->critical mapping cannot cross
//     it, so the UI must label it (T3 review-carry).
// ============================================================================
#include <caliper/services/feed_v1.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pulsescope {

// An ordered ring of the newest samples of ONE channel, copied applet-side from
// caliper.feed.v1 read() on the poller thread. Oldest-first; the oldest points
// drop once `capacity` is exceeded (the applet keeps a scrolling window, the
// host keeps the authoritative ring). Pure/std-only — no locks here; the applet
// owns the mutex that guards a PulseRing across its poller/frame threads.
class PulseRing {
public:
    explicit PulseRing(std::size_t capacity = 2000)
        : cap_(capacity ? capacity : 1) {}

    // Ingest ONE read() batch: `s[0..n)` are oldest-first with contiguous seqs
    // (the read() contract). `epoch_ns` is the shared host-clock origin (the
    // first t_ns the applet ever saw, across ALL channels) so the x-axis of
    // every channel is the same timeline.
    //
    // Honest-loss GAP: read() resumes at the oldest still-buffered sample, so if
    // the host ring overwrote samples between our polls the first seq JUMPS past
    // our last_seq+1. That jump is exactly the count of lost samples; we add it
    // to the cumulative gap total and return it. The VERY FIRST batch never
    // counts as loss (last_seq_==0): a fresh tail read legitimately starts at a
    // high seq (newest-minus-max), which is not a gap.
    uint64_t ingest(const CaliperFeedSample* s, uint32_t n, int64_t epoch_ns) {
        if (!s || n == 0) return 0;
        uint64_t lost = 0;
        if (last_seq_ != 0 && s[0].seq > last_seq_ + 1)
            lost = s[0].seq - last_seq_ - 1;
        gap_total_ += lost;
        for (uint32_t i = 0; i < n; ++i) {
            xs_.push_back(static_cast<double>(s[i].t_ns - epoch_ns) * 1e-9);
            ys_.push_back(static_cast<double>(s[i].value));
            last_seq_   = s[i].seq;
            last_t_ns_  = s[i].t_ns;
            last_value_ = s[i].value;
        }
        trim();
        return lost;
    }

    bool              empty()      const { return xs_.empty(); }
    std::size_t       size()       const { return xs_.size(); }
    const std::vector<double>& xs() const { return xs_; }
    const std::vector<double>& ys() const { return ys_; }
    uint64_t          gap_total()  const { return gap_total_; }
    uint64_t          last_seq()   const { return last_seq_; }
    int64_t           last_t_ns()  const { return last_t_ns_; }
    float             last_value() const { return last_value_; }

private:
    void trim() {
        if (xs_.size() <= cap_) return;
        const std::size_t drop = xs_.size() - cap_;
        xs_.erase(xs_.begin(), xs_.begin() + static_cast<std::ptrdiff_t>(drop));
        ys_.erase(ys_.begin(), ys_.begin() + static_cast<std::ptrdiff_t>(drop));
    }

    std::size_t         cap_;
    std::vector<double> xs_;          // relative seconds from the shared epoch
    std::vector<double> ys_;          // channel value
    uint64_t            gap_total_ = 0;
    uint64_t            last_seq_   = 0;
    int64_t             last_t_ns_  = 0;
    float               last_value_ = 0.0f;
};

// The unitless sys.thermal.state code (NSProcessInfo thermalState, 0..3) -> its
// word. The feed ABI vends a bare float; this mapping cannot cross it, so the UI
// labels the number (T3 review-carry). Out-of-range -> "unknown".
inline const char* thermal_word(int state) {
    switch (state) {
        case 0: return "nominal";
        case 1: return "fair";
        case 2: return "serious";
        case 3: return "critical";
        default: return "unknown";
    }
}

}  // namespace pulsescope
