// feed_catalog.h: the channels of caliper.feed.v1 as sources of the builder's
// catalog (design/catalog.md, the kind `feed`), and their data for the
// session's resolve (caliper_builder::CatalogDataSource). Host-internal;
// built only with CALIPER_WITH_BUILDER.
//
// One source per channel of the FeedStore: the id is `feed:<channel id>`
// (sys.cpu.util), the title the channel's display name, the columns
// `seq BIGINT`, `t_ns BIGINT` and `value FLOAT`, `rows` the samples the ring
// holds now, `rate_hz` the channel's nominal rate (null when it is 0, an
// irregular channel), and the origin {service "feed", channel, units}.
// `updated_at` is the newest sample's time: its t_ns on the host's steady
// clock, turned into wall-clock time by the offset between the two clocks
// now. A channel id that breaks the catalog's name rules is not listed.
//
// COSTS. Every call reads the store through its own surface: the registry
// mutex to find a channel (held for a scan of the channel list) and the
// channel's own mutex for its ring, which a push holds for one store of 24
// bytes. Nothing waits on a provider thread, so every call may run on the
// frame thread. list() and describe() read each channel's ring state and its
// newest sample (O(channels)); sample() copies at most 1000 samples, read()
// at most the request's max_rows (the ring holds 4096 by default).
// generation() is the channel's newest seq, so it changes with every sample.
// revision() grows with the channel count; channels are never removed in v0,
// so a channel id that is not registered is simply not found.
//
// Thread-safe; no function throws.
#pragma once

#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/session/resolve.h>

#include <cstdint>
#include <string>

namespace caliper_host {

class FeedStore;

class FeedCatalog : public caliper_builder::CatalogProvider, public caliper_builder::CatalogDataSource {
public:
    explicit FeedCatalog(FeedStore& store);

    // CatalogProvider
    caliper_builder::CatalogListResult list(const caliper_builder::CatalogFilter& filter) override;
    caliper_builder::CatalogDescribeResult describe(const std::string& id) override;
    caliper_builder::CatalogSampleResult sample(const std::string& id,
                                                const caliper_builder::CatalogSampleRequest& request) override;
    std::uint64_t revision() const override;

    // CatalogDataSource
    std::uint64_t generation(const std::string& id) override;
    caliper_builder::ServedData read(const std::string& id, const caliper_builder::ServedRequest& request) override;

    // The host's steady clock in nanoseconds, the clock of a sample's t_ns
    // (mach_absolute_time on macOS).
    static std::int64_t steady_now_ns();

private:
    // The channel of a feed id, by its registry index; false when unknown.
    bool channel_of(const std::string& id, std::string* channel, std::uint32_t* index) const;
    bool entry_of(std::uint32_t index, caliper_builder::CatalogEntry* out) const;

    FeedStore& store_;
};

}  // namespace caliper_host
