#pragma once
/* caliper.feed.v1 — live telemetry ingestion (physical twins, rung one).
 *
 * The HOST owns telemetry providers (v0: the Mac's own sensors), buffers their
 * samples in per-channel ring buffers with timestamps + sequence numbers, and
 * vends this tiny PULL-BASED, ANY-THREAD, NON-BLOCKING read surface. Applets
 * copy samples into their own tensors on their job threads (tensors -> pixels
 * stays downstream); embedders reach the same surface through
 * caliper_core_get_service. Sources are enumerable, reads never block, and data
 * loss is VISIBLE (sequence gaps), never silent.
 *
 * Four entry points, all any-thread and non-blocking:
 *   caps()          — capability bits; bit 0 (CALIPER_FEED_CAP_LIVE) is set iff
 *                     at least one channel is registered (channel_count() > 0),
 *                     i.e. live telemetry is available on THIS host. A host with
 *                     no provider (Windows v0) or no channels reports 0 and every
 *                     read yields nothing — honest degradation, never fake data.
 *   channel_count() — number of registered channels.
 *   channel_info()  — fill a caller-owned CaliperFeedChannelInfo for an index.
 *   read()          — copy new samples for a channel past a caller-held cursor.
 *
 * Rationale for pull + caller cursors: no callbacks across the ABI (no
 * reentrancy/threading contract to defend), no per-subscriber host state (any
 * number of readers, including embedder UI threads, at any cadence). Push can
 * be layered later without breaking this surface (reserved0 holds the slot).
 *
 * Invariants (feed spec): only C types cross the ABI; timestamps are host-clock
 * ns (the same epoch across channels, so series align); loss is visible via seq
 * gaps, never silent; absent capability => inert entry points + honest status.
 *
 * IMMUTABLE once published; additive growth lands as feed.v1_1. */
#include <stdint.h>
#include <stddef.h>

#define CALIPER_FEED_V1 "caliper.feed.v1"

/* caps() bit 0: live telemetry is available (>= 1 channel registered). */
#define CALIPER_FEED_CAP_LIVE (1u << 0)

#ifdef __cplusplus
extern "C" {
#endif

/* One measurement of one channel. seq is per-channel monotonic from 1; t_ns is
 * the host steady clock (mach_absolute-derived ns), same epoch across channels.
 * 24 bytes, 8-aligned, static_asserted. */
typedef struct CaliperFeedSample {
    uint64_t seq;
    int64_t  t_ns;
    float    value;
    float    reserved0;
} CaliperFeedSample;

/* One channel's descriptor. The CALLER sets struct_size = sizeof(this); the host
 * fills the rest. id is a stable string (e.g. "sys.cpu.util"); units is a short
 * label ("%", "degC", "W"); nominal_hz is a rate hint (0 = irregular). */
typedef struct CaliperFeedChannelInfo {
    uint32_t struct_size;   /* caller sets; host fills the rest */
    char     id[64];
    char     name[64];
    char     units[16];
    float    nominal_hz;    /* 0 = irregular */
} CaliperFeedChannelInfo;

typedef struct CaliperFeedV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);                      /* bit 0: live (see header) */
    uint32_t (*channel_count)(void);
    /* Fill *info for channel `index`. Returns 1 on success, 0 on a bad index
     * (>= channel_count) or a bad size (info->struct_size <
     * sizeof(CaliperFeedChannelInfo)); on 0 the *info is left UNTOUCHED. */
    uint32_t (*channel_info)(uint32_t index, CaliperFeedChannelInfo* info);
    /* Copy up to `max` samples with seq > *cursor into buf (OLDEST-FIRST),
     * advance *cursor to the last copied seq, and return the count copied.
     *   - cursor==0 is a TAIL read: start at the newest sample minus `max`, so a
     *     fresh reader gets at most `max` newest samples (seq starts at 1, so 0
     *     is never a real sample — it is the sentinel for "start fresh").
     *   - Unknown channel_id (or null buf, or max==0) -> 0, *cursor UNTOUCHED.
     *   - Caught up (no seq > *cursor) -> 0, *cursor UNTOUCHED.
     *   - A GAP: if *cursor points below the oldest still-buffered sample (the
     *     ring overwrote it), the copy resumes at the oldest available sample,
     *     so the returned seqs JUMP past *cursor+1 — data was lost, honestly
     *     observable by the caller (returned seq > previous *cursor + 1).
     * Capacity: each channel keeps at most a fixed number of newest samples;
     * older samples are dropped (overflow drops oldest). Non-blocking. */
    uint32_t (*read)(const char* channel_id, CaliperFeedSample* buf,
                     uint32_t max, uint64_t* cursor);
    void*    reserved0;     /* future: applet-registered sources */
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
