#pragma once
/* caliper.feed.v1 — live telemetry ingestion (docs/applets/services/caliper-feed-v1.md).
 *
 * The host owns the telemetry providers, buffers their samples in a ring buffer for each channel, and vends a
 * pull-based read surface. Every entry point is callable from any thread and never blocks. The caller holds the
 * read cursor, so no callbacks cross the ABI and the host keeps no state for each reader. A loss of samples shows
 * as a gap in the sequence numbers and is never silent. Immutable once published: additive growth is feed.v1_1. */
#include <stdint.h>
#include <stddef.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_FEED_V1 "caliper.feed.v1"

/**
 * Bit 0 of `caps()`, set if and only if the host has at least one registered channel. Without a provider or a
 * channel, `caps()` returns 0 and every read yields nothing.
 */
#define CALIPER_FEED_CAP_LIVE (1u << 0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * One measurement of one channel.
 *
 * The struct is 24 bytes and 8-aligned.
 */
typedef struct CaliperFeedSample {
    /** The sequence number of the sample. It counts up from 1 for each channel. */
    uint64_t seq;
    /** The time of the sample in nanoseconds of the host steady clock. All channels share one epoch. */
    int64_t  t_ns;
    /** The measured value, in the units of the channel. */
    float    value;
    /** Reserved. */
    float    reserved0;
} CaliperFeedSample;

/**
 * The descriptor of one channel.
 *
 * The caller sets `struct_size` to `sizeof(CaliperFeedChannelInfo)`, and the host fills the other fields.
 */
typedef struct CaliperFeedChannelInfo {
    /** `sizeof(CaliperFeedChannelInfo)`, which the caller sets. */
    uint32_t struct_size;
    /** The stable channel id as a NUL-terminated string, for example "sys.cpu.util". */
    char     id[64];
    /** The display name of the channel as a NUL-terminated string. */
    char     name[64];
    /** A short label for the units as a NUL-terminated string, for example "%", "degC" or "W". */
    char     units[16];
    /** A hint of the sample rate in hertz. 0 means irregular. */
    float    nominal_hz;
} CaliperFeedChannelInfo;

/**
 * The live telemetry service.
 *
 * Every entry is callable from any thread and never blocks. A channel keeps a fixed number of the newest samples,
 * and an overflow drops the oldest sample.
 */
typedef struct CaliperFeedV1 {
    /** `sizeof(CaliperFeedV1)` as the host built it. */
    uint32_t struct_size;
    /** Return the capability bits. Bit 0 is `CALIPER_FEED_CAP_LIVE`. */
    uint32_t (*caps)(void);
    /** Return the number of registered channels. */
    uint32_t (*channel_count)(void);
    /**
     * Fill `*info` for the channel at `index` and return 1. Return 0 and leave `*info` untouched if `index` is not
     * less than `channel_count()` or if `info->struct_size` is smaller than `sizeof(CaliperFeedChannelInfo)`.
     */
    uint32_t (*channel_info)(uint32_t index, CaliperFeedChannelInfo* info);
    /**
     * Copy up to `max` samples with `seq` greater than `*cursor` into `buf`, oldest first, advance `*cursor` to the
     * last copied `seq` and return the count. A cursor of 0 is a tail read that starts at the newest sample minus
     * `max`. An unknown channel id, a `NULL` buffer, a `max` of 0 or a reader that has caught up returns 0 and
     * leaves the cursor untouched. If the ring overwrote the sample after `*cursor`, the copy resumes at the oldest
     * sample that remains, so the first returned `seq` is greater than `*cursor + 1`. The jump is the number of
     * lost samples.
     */
    uint32_t (*read)(const char* channel_id, CaliperFeedSample* buf,
                     uint32_t max, uint64_t* cursor);
    /** Reserved for sources that applets register. Always `NULL`. */
    void*    reserved0;
} CaliperFeedV1;

#ifdef __cplusplus
}
/* --- ABI freeze: sizes + offsets pinned (only C types cross the boundary) --- */
static_assert(sizeof(CaliperFeedSample) == 24,
              "CaliperFeedSample ABI size is frozen (24 B, 8-aligned)");
static_assert(offsetof(CaliperFeedSample, seq) == 0);
static_assert(offsetof(CaliperFeedSample, t_ns) == 8);
static_assert(offsetof(CaliperFeedSample, value) == 16);
static_assert(offsetof(CaliperFeedSample, reserved0) == 20);

static_assert(sizeof(CaliperFeedChannelInfo) == 152,
              "CaliperFeedChannelInfo ABI size is frozen");
static_assert(offsetof(CaliperFeedChannelInfo, struct_size) == 0);
static_assert(offsetof(CaliperFeedChannelInfo, id) == 4);
static_assert(offsetof(CaliperFeedChannelInfo, name) == 68);
static_assert(offsetof(CaliperFeedChannelInfo, units) == 132);
static_assert(offsetof(CaliperFeedChannelInfo, nominal_hz) == 148);

static_assert(offsetof(CaliperFeedV1, struct_size) == 0);
static_assert(offsetof(CaliperFeedV1, caps) == sizeof(void*));
static_assert(offsetof(CaliperFeedV1, channel_count) ==
              offsetof(CaliperFeedV1, caps) + sizeof(void*));
static_assert(offsetof(CaliperFeedV1, channel_info) ==
              offsetof(CaliperFeedV1, channel_count) + sizeof(void*));
static_assert(offsetof(CaliperFeedV1, read) ==
              offsetof(CaliperFeedV1, channel_info) + sizeof(void*));
static_assert(offsetof(CaliperFeedV1, reserved0) ==
              offsetof(CaliperFeedV1, read) + sizeof(void*));
static_assert(sizeof(CaliperFeedV1) ==
              offsetof(CaliperFeedV1, reserved0) + sizeof(void*),
              "CaliperFeedV1 vtable layout is frozen");
#endif
