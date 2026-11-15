// live_series_ring.h: how a viewer applies wire protocol v3 SERIES deltas to
// the ring a SERIES live view reads (host-internal, header-only, portable).
// Shared by the native NetSource (live_net_source.mm) and the browser's
// (web/wasm/src/net_source.cpp).
//
// The ring holds M values; absolute value index i lives at ring[i % M], and
// `total` is one past the newest index held, which is the series_total a
// SERIES snapshot hands the live view (caliper/live_views.h). A delta carries
// the values start .. start + n - 1:
//   - a delta that starts past the end held leaves a gap, which reads as NaN
//     (ImPlot breaks the line there); only the gap's last M values matter;
//   - values a delta overlaps are replaced, but never ones older than the
//     window of the newest M, whose ring positions now hold newer values.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace caliper_live {

inline void series_ring_apply(float* ring, uint64_t M, uint64_t* total, uint64_t start,
                              const float* values, uint64_t n) {
    const uint64_t end = start + n;
    if (start > *total)
        for (uint64_t i = std::max(*total, start > M ? start - M : 0); i < start; ++i) ring[i % M] = NAN;
    const uint64_t newest = std::max(*total, end);
    const uint64_t lo = std::max<uint64_t>(start, newest > M ? newest - M : 0);
    for (uint64_t i = lo; i < end; ++i) ring[i % M] = values[i - start];
    *total = newest;
}

}  // namespace caliper_live
