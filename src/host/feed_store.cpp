#include "feed_store.h"

#include <cstdio>   // std::snprintf (bounded, null-terminating field copy)

namespace caliper_host {

FeedStore::FeedStore() = default;
FeedStore::~FeedStore() = default;

FeedStore::Channel* FeedStore::find(const std::string& id) const {
    // Registry is append-only; the returned Channel* stays valid because the
    // objects are heap-stable (unique_ptr) and never removed in v0.
    std::lock_guard<std::mutex> lk(reg_mu_);
    for (const auto& c : channels_)
        if (c->id == id) return c.get();
    return nullptr;
}

bool FeedStore::add_channel(const std::string& id, const std::string& name,
                            const std::string& units, float nominal_hz,
                            uint32_t capacity) {
    if (id.empty() || capacity == 0) return false;
    std::lock_guard<std::mutex> lk(reg_mu_);
    for (const auto& c : channels_)
        if (c->id == id) return false;  // no duplicate ids
    auto ch = std::make_unique<Channel>();
    ch->id = id;
    ch->name = name;
    ch->units = units;
    ch->nominal_hz = nominal_hz;
    ch->capacity = capacity;
    ch->ring.resize(capacity);
    ch->next_seq = 1;
    channels_.push_back(std::move(ch));
    return true;
}

uint64_t FeedStore::push(const std::string& channel_id, int64_t t_ns,
                         float value) {
    Channel* ch = find(channel_id);
    if (!ch) return 0;
    std::lock_guard<std::mutex> lk(ch->mu);
    const uint64_t seq = ch->next_seq++;
    // seq is contiguous from 1, so the sample with seq S lives at (S-1)%capacity;
    // writing there overwrites the oldest sample once the ring is full.
    CaliperFeedSample& slot = ch->ring[(seq - 1) % ch->capacity];
    slot.seq = seq;
    slot.t_ns = t_ns;
    slot.value = value;
    slot.reserved0 = 0.0f;
    return seq;
}

uint32_t FeedStore::caps() const {
    return channel_count() > 0 ? CALIPER_FEED_CAP_LIVE : 0u;
}

uint32_t FeedStore::channel_count() const {
    std::lock_guard<std::mutex> lk(reg_mu_);
    return static_cast<uint32_t>(channels_.size());
}

uint32_t FeedStore::channel_info(uint32_t index,
                                 CaliperFeedChannelInfo* info) const {
    // Bad size / null: refuse BEFORE touching *info (the contract: untouched).
    if (!info || info->struct_size < sizeof(CaliperFeedChannelInfo)) return 0;
    std::lock_guard<std::mutex> lk(reg_mu_);
    if (index >= channels_.size()) return 0;   // bad index: *info untouched
    const Channel& ch = *channels_[index];
    // Bounded, always-null-terminating copies into the fixed ABI fields.
    std::snprintf(info->id, sizeof info->id, "%s", ch.id.c_str());
    std::snprintf(info->name, sizeof info->name, "%s", ch.name.c_str());
    std::snprintf(info->units, sizeof info->units, "%s", ch.units.c_str());
    info->nominal_hz = ch.nominal_hz;
    return 1;
}

uint32_t FeedStore::read(const char* channel_id, CaliperFeedSample* buf,
                         uint32_t max, uint64_t* cursor) {
    if (!channel_id || !buf || !cursor || max == 0) return 0;
    Channel* ch = find(channel_id);
    if (!ch) return 0;   // unknown channel: *cursor untouched

    std::lock_guard<std::mutex> lk(ch->mu);
    const uint64_t newest = ch->next_seq - 1;   // 0 when nothing pushed yet
    if (newest == 0) return 0;                   // empty: *cursor untouched
    // Oldest still-buffered seq: the ring holds at most `capacity` newest.
    const uint64_t oldest =
        (newest > ch->capacity) ? (newest - ch->capacity + 1) : 1;

    // start_excl: copy samples with seq > start_excl.
    uint64_t start_excl;
    if (*cursor == 0) {
        // Tail read: newest minus max (at most `max` newest samples).
        start_excl = (newest > max) ? (newest - max) : 0;
    } else {
        start_excl = *cursor;
    }
    // Clamp up to the oldest available (oldest-1 exclusive). If the caller's
    // cursor sat below this, the ring overwrote those samples: the copy resumes
    // at `oldest`, so the returned seqs JUMP past *cursor+1 — the honest,
    // observable loss contract.
    if (start_excl + 1 < oldest) start_excl = oldest - 1;

    if (start_excl >= newest) return 0;   // caught up: *cursor untouched

    const uint64_t first = start_excl + 1;
    uint64_t n = newest - start_excl;     // available new samples
    if (n > max) n = max;                 // bounded by the caller's buffer

    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t seq = first + i;
        buf[i] = ch->ring[(seq - 1) % ch->capacity];
    }
    *cursor = first + n - 1;              // advance to the last copied seq
    return static_cast<uint32_t>(n);
}

}  // namespace caliper_host
