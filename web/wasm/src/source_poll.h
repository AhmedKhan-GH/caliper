// source_poll.h: one frame's poll of a caliper_live::Source into the typed
// live views, for host-memory snapshots (the browser's NetSource).
//
// The per-poll order is live_core.cpp's (the shared-memory viewer):
// heartbeat(), producer_alive(), sync() and a reconcile of the view list,
// then per view at most one take() and an apply through
// caliper_core_live_apply, then drew() or unclaim(). The differences are only
// where a snapshot lives: here mtl_buffer is null and host_base holds the
// bytes, so the payload is a CPU tensor, which the live views copy when they
// apply it; no claim outlives the apply and nothing waits on the GPU. After
// an apply the snapshot's payload map goes to the view, which reports events
// in source indices with it.
#pragma once

#include "live/live_source.h"

#include <cstdint>
#include <string>
#include <vector>

struct CaliperCore;

namespace caliper_web {

struct PolledView {
    uint64_t key = 0;
    std::string name;
    uint32_t kind = 0;
    bool shown = false;      // a live view of this name exists in the core
    uint64_t seq = 0;        // the last applied seq
    uint64_t updates = 0;
    std::string error;       // the last apply's refusal
};

struct PollState {
    std::vector<PolledView> views;
    uint64_t polls = 0, updates = 0, errors = 0;
    bool producer_alive = true;
    // The source maps producer stamps from another clock (a network source):
    // every apply carries CALIPER_LIVE_APPLY_STAMP_RELATIVE.
    bool relative_stamps = false;
};

// Applies what is new; returns how many snapshots were applied.
int poll_source(CaliperCore* core, caliper_live::Source& src, PollState& st);

// Removes every live view this poll created (detach, shutdown).
void remove_polled_views(CaliperCore* core, PollState& st);

}  // namespace caliper_web
