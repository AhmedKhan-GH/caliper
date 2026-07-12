#pragma once
// caliper_host::FeedStore — the per-channel ring-buffer store behind
// caliper.feed.v1 (feed spec §4). A channel is one named f32 time series; each
// keeps a fixed-capacity ring of the newest samples, timestamped and sequenced.
//
// Threading: ONE MUTEX PER CHANNEL. A channel's writer (the provider thread, or
// a test injecting samples) and its readers (any thread, any cadence) serialize
// on that channel's own mutex — the same one-lock simplicity the metrics store
// proved race-safe, but sharded per channel so independent channels never
// contend. The channel registry is append-only and guarded by a separate
// registry mutex; in v0 all channels are registered at setup (before any
// provider thread or reader), so registry reads are effectively uncontended.
//
// The read surface mirrors the caliper.feed.v1 vtable slot-for-slot
// (caps/channel_count/channel_info/read), so the host thunks in
// host_services.cpp are trivial forwarders and this class carries the tested
// logic. The ABI types (CaliperFeedSample, CaliperFeedChannelInfo) are the only
// non-primitive types here — they cross the boundary unchanged.
//
// Providers (T2 macOS sensors) and the deterministic test provider both drive
// the store through add_channel()/push(); the store itself owns no thread.
#include <caliper/services/feed_v1.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace caliper_host {

class FeedStore {
public:
    FeedStore();
    ~FeedStore();

    FeedStore(const FeedStore&) = delete;
    FeedStore& operator=(const FeedStore&) = delete;

    // --- provider / test-provider write path ---

    // Register a channel with a ring `capacity` (max newest samples retained).
    // Returns false if `id` is empty, already registered, or capacity == 0.
    // A setup-time call: register channels before starting readers/writers.
    bool add_channel(const std::string& id, const std::string& name,
                     const std::string& units, float nominal_hz,
                     uint32_t capacity);

    // Append one sample to `channel_id`, assigning the next per-channel seq
    // (monotonic from 1) and stamping t_ns. Overflow drops the oldest sample.
    // Returns the assigned seq, or 0 if the channel is unknown.
    uint64_t push(const std::string& channel_id, int64_t t_ns, float value);

    // --- read surface backing caliper.feed.v1 (mirrors the vtable) ---

    // Capability bits: CALIPER_FEED_CAP_LIVE iff channel_count() > 0.
    uint32_t caps() const;
    uint32_t channel_count() const;
    // Fill *info for channel `index`. 1 on success; 0 on bad index or bad size
    // (info->struct_size < sizeof(CaliperFeedChannelInfo)), *info UNTOUCHED.
    uint32_t channel_info(uint32_t index, CaliperFeedChannelInfo* info) const;
    // Copy up to `max` samples with seq > *cursor into buf (oldest-first),
    // advance *cursor to the last copied seq, return the count. See feed_v1.h
    // read() for the full cursor / tail-start / gap contract.
    uint32_t read(const char* channel_id, CaliperFeedSample* buf,
                  uint32_t max, uint64_t* cursor);

private:
    struct Channel {
        std::string id;
        std::string name;
        std::string units;
        float       nominal_hz = 0.0f;
        uint32_t    capacity = 0;
        std::vector<CaliperFeedSample> ring;  // size == capacity, indexed by seq
        uint64_t    next_seq = 1;             // seq to assign the next push
        mutable std::mutex mu;                // guards ring + next_seq
    };

    // Registry mutex guards the channel vector (append-only). Channel objects
    // are heap-stable (unique_ptr), so a pointer obtained under reg_mu_ stays
    // valid for the lifetime of the store.
    mutable std::mutex reg_mu_;
    std::vector<std::unique_ptr<Channel>> channels_;

    Channel* find(const std::string& id) const;  // nullptr if unknown
};

}  // namespace caliper_host
