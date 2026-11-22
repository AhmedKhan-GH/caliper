/* ===========================================================================
 * caliper/live.h: live tensor streaming between a training process (the client)
 * and a viewer process, protocol v2, macOS and Metal only.
 *
 * The client opens a local live session with a session file and POSIX shared
 * memory. The viewer lists the session files, attaches to one, maps the same
 * memory and draws the newest completed snapshot of every view in place. The
 * client never waits for the viewer.
 *
 * The session file, the shared memory layout, the slot protocol and the
 * versioning are described in docs/development/live-session-layout.md. The
 * client and viewer calls are described in docs/web/embedding-api.md.
 * ===========================================================================*/
#ifndef CALIPER_LIVE_H
#define CALIPER_LIVE_H

#include <stddef.h>
#include <stdint.h>

#include <caliper/embed.h>   /* CaliperCore (viewer side) */

#ifdef __cplusplus
extern "C" {
#endif

/** The version of the shared memory protocol. A reader refuses any other version. */
#define CALIPER_LIVE_PROTOCOL_VERSION 2u
/** The magic number of the control segment, the ASCII text "CLIVECTL" in little-endian order. */
#define CALIPER_LIVE_CTL_MAGIC  0x4c54434556494c43ull
/** The magic number of a view segment, the ASCII text "CLIVESEG" in little-endian order. */
#define CALIPER_LIVE_SEG_MAGIC  0x4745534556494c43ull

/** The number of rows in the control table, and thus the largest number of views in one session. */
#define CALIPER_LIVE_MAX_VIEWS    64
/** The number of slots of every view kind except a series, which has 1 slot. */
#define CALIPER_LIVE_SLOTS        3
/** The size in bytes of a view name, with the terminating NUL. */
#define CALIPER_LIVE_NAME_MAX     64
/**
 * The size in bytes of a shared memory name, with the terminating NUL. It fits the 31 characters that macOS allows.
 */
#define CALIPER_LIVE_SEGNAME_MAX  32
/** The largest number of characters of a session token. The characters are `A-Z`, `a-z`, `0-9`, `.`, `_` and `-`. */
#define CALIPER_LIVE_SESSION_MAX  20
/** The size in bytes of a script name, with the terminating NUL. Added in protocol v2. */
#define CALIPER_LIVE_SCRIPT_MAX   128
/** The size in bytes of the stats region of one slot. It holds 64 `float` values. */
#define CALIPER_LIVE_STATS_BYTES  256
/** The slot index that means no slot. A claim or a published slot has this value when it is free or empty. */
#define CALIPER_LIVE_SLOT_NONE    0xffffffffu
/**
 * The age in nanoseconds below which a viewer heartbeat counts as attached. `caliper_live_client_attached` uses it.
 */
#define CALIPER_LIVE_ATTACH_WINDOW_NS 1000000000ull

/**
 * The kind of a view, which fixes the shape and the type of its payload.
 *
 * The values are the same as the kinds of `caliper/live_views.h`.
 */
typedef enum CaliperLiveKind {
    /** No kind. */
    CALIPER_LIVE_KIND_NONE      = 0,
    /** A heatmap, an `(H,W)` array of `float` that the viewer colormaps. */
    CALIPER_LIVE_KIND_HEATMAP   = 1,
    /** An image, an `(H,W,4)` array of `uint8` in RGBA order. */
    CALIPER_LIVE_KIND_IMAGE     = 2,
    /**
     * A point cloud, an `(N,3)` array of `float` positions with an optional `(N,)` array of `float` in the aux
     * region.
     */
    CALIPER_LIVE_KIND_POINTS    = 3,
    /** A series, an `(M,)` array of `float` that is an append-only ring in one slot. */
    CALIPER_LIVE_KIND_SERIES    = 4,
    /** A histogram, a `(B,)` array of `float` bin counts over the range from `stats[0]` to `stats[1]`. */
    CALIPER_LIVE_KIND_HISTOGRAM = 5
} CaliperLiveKind;

/** The state of one row of the control table. */
typedef enum CaliperLiveEntryState {
    /** The row is free. */
    CALIPER_LIVE_ENTRY_FREE   = 0,
    /** The row describes a live view. */
    CALIPER_LIVE_ENTRY_ACTIVE = 1
} CaliperLiveEntryState;

/** The state of a session, which the client keeps in the control segment. */
typedef enum CaliperLiveSessionState {
    /** The session is open. */
    CALIPER_LIVE_SESSION_OPEN   = 1,
    /** The client closed the session cleanly and unlinked the shared memory names. */
    CALIPER_LIVE_SESSION_CLOSED = 2
} CaliperLiveSessionState;

/**
 * The mapping flags of one snapshot, which the `flags` field of `CaliperLiveSlotMeta` holds.
 *
 * The viewer maps them onto the `CALIPER_LIVE_APPLY_*` flags of `caliper/live_views.h`.
 */
enum {
    /**
     * Bit flag for a lower end of the range that comes from `stats[0]`, the GPU minimum. Without it, the range uses
     * `vmin`.
     */
    CALIPER_LIVE_SNAP_AUTO_VMIN = 1u << 0,
    /**
     * Bit flag for an upper end of the range that comes from `stats[1]`, the GPU maximum. Without it, the range
     * uses `vmax`.
     */
    CALIPER_LIVE_SNAP_AUTO_VMAX = 1u << 1,
    /**
     * Bit flag for a heatmap or an image that the viewer draws 1:1 at the origin. Tests use it. Added in protocol
     * v2.
     */
    CALIPER_LIVE_SNAP_RAW       = 1u << 2
};

/* --- shared structs (protocol v2) ---------------------------------------- */

/**
 * One row of the control table.
 *
 * The client writes a row while the `generation` of the control segment is odd. The viewer reads a row as a snapshot
 * between two equal even generations.
 */
typedef struct CaliperLiveViewEntry {
    /** `sizeof(CaliperLiveViewEntry)` as the client built it. */
    uint32_t struct_size;
    /** The state of the row, a `CaliperLiveEntryState` value. */
    uint32_t state;
    /** The kind of the view, a `CaliperLiveKind` value. */
    uint32_t kind;
    /** The `CaliperDType` of the payload. */
    int32_t  dtype;
    /** The number of dimensions of the payload. */
    uint32_t ndim;
    /** The number of slots of the view, `CALIPER_LIVE_SLOTS`, or 1 for a series. */
    uint32_t slot_count;
    /** The extent of each dimension of the payload. */
    int64_t  shape[4];
    /** The id of the view. It is unique in the session, the client never reuses it, and it names the view segment. */
    uint64_t view_id;
    /** The size in bytes of the payload of one snapshot, as an exact extent. */
    uint64_t slot_bytes;
    /** The size in bytes of the stats region of one slot, `CALIPER_LIVE_STATS_BYTES`. */
    uint64_t stats_bytes;
    /** The length in bytes of the whole mapping of the view segment. */
    uint64_t segment_bytes;
    /** The name of the view, NUL-terminated. */
    char     name[CALIPER_LIVE_NAME_MAX];
    /** The shared memory name of the view segment, NUL-terminated. */
    char     segment[CALIPER_LIVE_SEGNAME_MAX];
    /**
     * The size in bytes of the aux region of one slot, which holds `float` values, or 0 for no aux region. Added in
     * protocol v2.
     */
    uint64_t aux_bytes;
    /** The number of `float` elements in the aux region, or 0 for none. Added in protocol v2. */
    int64_t  aux_count;
} CaliperLiveViewEntry;

/**
 * The header of the control segment `/<session>-ctl`, which holds the session header and a table of up to
 * `CALIPER_LIVE_MAX_VIEWS` views.
 *
 * The client stores `magic` last, with release order, when it opens the session. Thus a `magic` of 0 means that the
 * session is still starting. The segment has one viewer at most, because the viewer fields have one writer.
 */
typedef struct CaliperLiveControl {
    /** `CALIPER_LIVE_CTL_MAGIC`. The client stores it last, with release order. */
    uint64_t magic;
    /** `CALIPER_LIVE_PROTOCOL_VERSION`. */
    uint32_t version;
    /** `sizeof(CaliperLiveControl)` as the client built it. */
    uint32_t struct_size;
    /** `sizeof(CaliperLiveViewEntry)` as the client built it. */
    uint32_t entry_size;
    /** The number of rows in `views`, `CALIPER_LIVE_MAX_VIEWS`. */
    uint32_t max_views;
    /** The process id of the client. */
    int32_t  client_pid;
    /** The state of the session, a `CaliperLiveSessionState` value. Access it with atomic operations. */
    uint32_t state;
    /** A seqlock over `views`. The client makes it odd while it edits the table. Access it with atomic operations. */
    uint64_t generation;
    /**
     * The process id of the attached viewer, or 0 for no viewer.
     * The viewer writes it with atomic operations, and a detaching viewer sets it and `viewer_heartbeat_ns` to 0.
     */
    int32_t  viewer_pid;
    /** Reserved. */
    uint32_t reserved0;
    /** The time of the last poll of the viewer, in nanoseconds of `CLOCK_UPTIME_RAW`, or 0 for no viewer. */
    uint64_t viewer_heartbeat_ns;
    /** Reserved for later versions. */
    uint64_t reserved[6];
    /** The table of views. The viewer reads it again when `generation` changes. */
    CaliperLiveViewEntry views[CALIPER_LIVE_MAX_VIEWS];
    /**
     * The wall clock time of the opening of the session, in milliseconds since the Unix epoch. The client writes it
     * before `magic`.
     */
    uint64_t start_unix_ms;
    /** The session token, NUL-terminated. The client writes it before `magic`. */
    char     session[CALIPER_LIVE_SESSION_MAX + 4];
    /** The name of the script, for example `train.py`, NUL-terminated. The client writes it before `magic`. */
    char     script[CALIPER_LIVE_SCRIPT_MAX];
} CaliperLiveControl;

/** The metadata of one slot. The client writes it before it publishes the slot. */
typedef struct CaliperLiveSlotMeta {
    /** The publish sequence number that this slot carries. */
    uint64_t seq;
    /** The `CLOCK_UPTIME_RAW` time of the client at the `show()` call, in nanoseconds. */
    uint64_t stamp_ns;
    /** The `CaliperColormap` of a heatmap or of continuous points. */
    int32_t  colormap;
    /** The mapping flags, a combination of the `CALIPER_LIVE_SNAP_*` bits. */
    uint32_t flags;
    /**
     * The fixed range, `vmin` for the lower end and `vmax` for the upper end, for the ends that are not automatic.
     */
    float    vmin, vmax;
    /** Reserved for later versions. */
    uint64_t reserved[2];
    /** For a series, the number of values appended so far. Added in protocol v2. */
    uint64_t series_total;
    /** For points, the number of dimensions of a point, 2 or 3. Added in protocol v2. */
    int32_t  points_dims;
    /**
     * For points, the number of categorical classes in the aux region, or 0 when the aux region holds values. Added
     * in protocol v2.
     */
    int32_t  label_count;
    /** The number of meaningful `float` values in the stats region of the slot. Added in protocol v2. */
    int32_t  stats_count;
    /** Reserved. */
    uint32_t reserved1;
} CaliperLiveSlotMeta;

/**
 * The header page of a view segment `/<session>-v<id>`.
 *
 * The header page is followed by the slots. The sync words are accessed with atomic operations on the shared mapping.
 */
typedef struct CaliperLiveSegmentHeader {
    /** `CALIPER_LIVE_SEG_MAGIC`. */
    uint64_t magic;
    /** `CALIPER_LIVE_PROTOCOL_VERSION`. */
    uint32_t version;
    /** `sizeof(CaliperLiveSegmentHeader)` as the writer built it. */
    uint32_t struct_size;
    /** The process id of the client that writes the segment. */
    int32_t  writer_pid;
    /** The number of slots, `CALIPER_LIVE_SLOTS`, or 1 for a series. */
    uint32_t slot_count;
    /** The id of the view, the same as `view_id` in its table row. */
    uint64_t view_id;
    /** The size in bytes of the payload of one snapshot, as an exact extent. */
    uint64_t payload_bytes;
    /** The size in bytes of the stats region of one slot. */
    uint64_t stats_bytes;
    /** The length in bytes of the whole mapping of the segment. */
    uint64_t segment_bytes;
    /** The byte offset of the payload of each slot. Each payload is page-aligned and row-major. */
    uint64_t slot_offset[CALIPER_LIVE_SLOTS];
    /**
     * The byte offset of the stats region of each slot. It holds `float` values, the minimum first and the maximum
     * second, then values of the kind.
     */
    uint64_t stats_offset[CALIPER_LIVE_SLOTS];

    /* --- sync words (atomic) --------------------------------------------- */
    /** The number of publishes so far. The client stores it last, with release order. */
    uint64_t seq;
    /** The index of the newest completed snapshot, or `CALIPER_LIVE_SLOT_NONE`. */
    uint32_t published_slot;
    /** The slot that the viewer applied last, which is its first claim, or `CALIPER_LIVE_SLOT_NONE`. */
    uint32_t viewer_reading_slot;
    /** The `stamp_ns` of the latest publish. */
    uint64_t stamp_ns;
    /** The number of snapshots that the client dropped because no slot was free. */
    uint64_t dropped_client;
    /** The last `seq` that the viewer drew. The viewer writes it. */
    uint64_t viewer_seq;
    /** The number of publishes that the viewer never drew. The viewer writes it. */
    uint64_t skipped_viewer;
    /** Reserved for later versions. */
    uint64_t reserved[4];

    /** The metadata of each slot. */
    CaliperLiveSlotMeta slot[CALIPER_LIVE_SLOTS];
    /** The kind of the view, a `CaliperLiveKind` value. Added in protocol v2. */
    uint32_t kind;
    /**
     * The slot that the viewer applied before `viewer_reading_slot`, which is its second claim, or
     * `CALIPER_LIVE_SLOT_NONE`. Access it with atomic operations. Added in protocol v2.
     */
    uint32_t viewer_prev_slot;
    /** The size in bytes of the aux region of one slot, or 0 for none. Added in protocol v2. */
    uint64_t aux_bytes;
    /**
     * The byte offset of the aux region of each slot. The region holds point labels or values as `float`. Added in
     * protocol v2.
     */
    uint64_t aux_offset[CALIPER_LIVE_SLOTS];
} CaliperLiveSegmentHeader;

/* --- client API (training process) ----------------------------------------
 * Implemented in src/host/live/live_client.mm. Each function returns 0/NULL on
 * refusal with a reason in caliper_live_last_error() (thread-local). */

/** The opaque client of a live session. It owns the control segment and the session file. */
typedef struct CaliperLiveClient  CaliperLiveClient;
/**
 * The opaque segment of one view, with its shared mapping. The client and every holder of a reference keep it
 * alive.
 */
typedef struct CaliperLiveSegment CaliperLiveSegment;

/** The flags of `CaliperLiveClientDesc.flags`. */
enum {
    /** Bit flag to use shared memory only and write no session file. Added in protocol v2. */
    CALIPER_LIVE_CLIENT_NO_SESSION_FILE = 1u << 0
};

/** The description of a client, which `caliper_live_client_open` takes. */
typedef struct CaliperLiveClientDesc {
    /** `sizeof(CaliperLiveClientDesc)`. */
    size_t      struct_size;
    /**
     * The session token, 1 to 20 characters of `A-Z`, `a-z`, `0-9`, `.`, `_` and `-`. With `NULL`, the client uses
     * `clp-<pid>-<6 hex>`.
     */
    const char* session;
    /**
     * The `id<MTLDevice>` for the segment buffers, which is the device of the producer, for example the device of a
     * torch buffer. With `NULL`, the client uses `MTLCreateSystemDefaultDevice()`.
     */
    void*       mtl_device;
    /**
     * The name of the script that viewers show. With `NULL`, the client uses the process name. Added in protocol
     * v2.
     */
    const char* script;
    /**
     * The root folder of the session file. With `NULL`, the client uses the default data root, as
     * `caliper_live_data_root` returns it. Added in protocol v2.
     */
    const char* data_dir;
    /** A combination of the `CALIPER_LIVE_CLIENT_*` flags. Added in protocol v2. */
    uint32_t    flags;
} CaliperLiveClientDesc;

/**
 * The description of a view, which `caliper_live_client_add_view` takes.
 *
 * The kinds have these payloads. A heatmap is `(H,W)` of `float`, an image is `(H,W,4)` of `uint8`, points are `(N,3)` of
 * `float` with an optional aux array of `N` values, a series is `(M,)` of `float` in 1 slot, and a histogram is `(B,)` of `float`.
 */
typedef struct CaliperLiveViewDesc {
    /** `sizeof(CaliperLiveViewDesc)`. */
    size_t      struct_size;
    /** The name of the view, 1 to 63 bytes. It is unique in the session. */
    const char* name;
    /** The kind of the view, a `CaliperLiveKind` value. */
    uint32_t    kind;
    /** The `CaliperDType` of the payload. It must match the kind. */
    int32_t     dtype;
    /** The number of dimensions of the payload. */
    int32_t     ndim;
    /** The extent of each dimension of the payload. */
    int64_t     shape[4];
    /**
     * The number of slots. With 0, the view uses the number of its kind, which is 3, or 1 for a series. Added in
     * protocol v2.
     */
    uint32_t    slots;
    /**
     * For points, the number `N` of aux values that every slot gets as an `(N,)` array of `float`. Added in
     * protocol v2.
     */
    int64_t     aux_count;
} CaliperLiveViewDesc;

/** The live description of a view segment, which `caliper_live_segment_info` fills. */
typedef struct CaliperLiveSegmentInfo {
    /** `sizeof(CaliperLiveSegmentInfo)`. */
    size_t   struct_size;
    /**
     * The `id<MTLBuffer>` over the whole mapping. The segment owns it, and it is `NULL` on a machine without a
     * device.
     */
    void*    mtl_buffer;
    /** The CPU address of the mapping. */
    void*    host_ptr;
    /** The length in bytes of the whole mapping. */
    uint64_t segment_bytes;
    /** The size in bytes of the payload of one snapshot. */
    uint64_t payload_bytes;
    /** The size in bytes of the stats region of one slot. */
    uint64_t stats_bytes;
    /** The byte offset of the payload of each slot. */
    uint64_t slot_offset[CALIPER_LIVE_SLOTS];
    /** The byte offset of the stats region of each slot. */
    uint64_t stats_offset[CALIPER_LIVE_SLOTS];
    /** The number of slots. */
    uint32_t slot_count;
    /** The row of the view in the control table, or -1 after the client removed the view. */
    int32_t  table_index;
    /** The shared memory name of the segment. It is valid while the segment lives. */
    const char* segment_name;
    /** The `seq` of the shared header, which counts the publishes. */
    uint64_t seq;
    /** The `published_slot` of the shared header. */
    uint32_t published_slot;
    /** The `viewer_reading_slot` of the shared header. */
    uint32_t viewer_reading_slot;
    /** The `dropped_client` counter of the shared header. */
    uint64_t dropped_client;
    /** The `viewer_seq` of the shared header. */
    uint64_t viewer_seq;
    /** The `skipped_viewer` counter of the shared header. */
    uint64_t skipped_viewer;
    /** The kind of the view, a `CaliperLiveKind` value. Added in protocol v2. */
    uint32_t kind;
    /** The `viewer_prev_slot` of the shared header. Added in protocol v2. */
    uint32_t viewer_prev_slot;
    /** The size in bytes of the aux region of one slot, or 0 for none. Added in protocol v2. */
    uint64_t aux_bytes;
    /** The byte offset of the aux region of each slot. Added in protocol v2. */
    uint64_t aux_offset[CALIPER_LIVE_SLOTS];
} CaliperLiveSegmentInfo;

/** The metadata of one published snapshot, which the publish calls copy into the slot metadata. */
typedef struct CaliperLivePublish {
    /** `sizeof(CaliperLivePublish)`. */
    size_t   struct_size;
    /**
     * The `CLOCK_UPTIME_RAW` time of the `show()` call, in nanoseconds. It is the same clock as the Python
     * `time.monotonic_ns`.
     */
    uint64_t stamp_ns;
    /** The `CaliperColormap` of a heatmap or of continuous points. */
    int32_t  colormap;
    /** A combination of the `CALIPER_LIVE_SNAP_*` bits. */
    uint32_t flags;
    /**
     * The fixed range, `vmin` for the lower end and `vmax` for the upper end, for the ends that are not automatic.
     */
    float    vmin, vmax;
    /** For a series, the number of values appended so far. Added in protocol v2. */
    uint64_t series_total;
    /** For points, the number of dimensions of a point, 2 or 3. Added in protocol v2. */
    int32_t  points_dims;
    /**
     * For points, the number of categorical classes in the aux region, or 0 when the aux region holds values. Added
     * in protocol v2.
     */
    int32_t  label_count;
    /** The number of meaningful `float` values in the stats region. With 0, the call uses 2. Added in protocol v2. */
    int32_t  stats_count;
} CaliperLivePublish;

/**
 * Return the reason for the most recent refusal on this thread, as a string that is valid until the next live call
 * on this thread.
 */
const char*         caliper_live_last_error(void);
/** Return the `CLOCK_UPTIME_RAW` time in nanoseconds. */
uint64_t            caliper_live_now_ns(void);

/**
 * Create the session, with the control segment first and then, unless `CALIPER_LIVE_CLIENT_NO_SESSION_FILE` is set, the
 * session file.
 * Return the client, or `NULL` with a reason in `caliper_live_last_error`. Call the client functions from one thread at a time per client.
 */
CaliperLiveClient*  caliper_live_client_open(const CaliperLiveClientDesc* desc);
/** Return the session token of the client. */
const char*         caliper_live_client_session(const CaliperLiveClient* c);
/** Return the path of the session file, or an empty string when the client has none. Added in protocol v2. */
const char*         caliper_live_client_session_file(const CaliperLiveClient* c);
/**
 * Return 1 while a viewer heartbeat is younger than `CALIPER_LIVE_ATTACH_WINDOW_NS`, and 0 otherwise or for `NULL`.
 * The call is one atomic load and one clock read, and any thread may call it. The Python `show()` takes no snapshot
 * while it returns 0, and thus an unwatched session costs almost nothing. Added in protocol v2.
 */
int                 caliper_live_client_attached(const CaliperLiveClient* c);
/**
 * Return 1 and store the viewer process id in `viewer_pid`, with 0 for none, and the time since its last poll in
 * `age_ns`, with `UINT64_MAX` for none.
 */
int                 caliper_live_client_viewer(const CaliperLiveClient* c,
                                               int32_t* viewer_pid, uint64_t* age_ns);
/**
 * Create a view and its segment, and rewrite the session file.
 * The client keeps one reference to the segment. The returned pointer is borrowed, so retain it to keep it beyond
 * `caliper_live_client_remove_view` or `caliper_live_client_close`.
 */
CaliperLiveSegment* caliper_live_client_add_view(CaliperLiveClient* c,
                                                 const CaliperLiveViewDesc* desc);
/**
 * Drop the view from the table and unlink its shared memory name. The mapping lives on while references remain, for
 * example the DLPack capsules that retain the segment.
 */
int                 caliper_live_client_remove_view(CaliperLiveClient* c,
                                                    CaliperLiveSegment* s);
/**
 * Mark the session closed, unlink every name, remove the session file and drop the references of the client. The
 * call never waits for a viewer and is safe on `NULL`. The mappings live on while references remain.
 */
void                caliper_live_client_close(CaliperLiveClient* c);

/** Add one reference to the segment. */
void                caliper_live_segment_retain(CaliperLiveSegment* s);
/** Drop one reference to the segment. The last reference unmaps it. */
void                caliper_live_segment_release(CaliperLiveSegment* s);
/** Fill `out` with the layout and the live counters of the segment. Return 1 on success. */
int                 caliper_live_segment_info(CaliperLiveSegment* s,
                                              CaliperLiveSegmentInfo* out);
/**
 * Choose a slot to write, for tests and C producers. The Python ring chooses its own slot.
 * A free slot is not `published_slot`, not a viewer claim and not a slot of this client that still holds an unpublished
 * snapshot. Return the slot index. Return -1 with `*busy` set to 1 when every candidate is in flight, and then the
 * caller drops this snapshot, which counts in `dropped_client`. Return -1 with `*busy` set to 0 when the segment was
 * removed. A series always gets slot 0.
 */
int                 caliper_live_segment_acquire(CaliperLiveSegment* s, int* busy);
/**
 * Publish an acquired `slot` with the metadata `meta`. The GPU writes to the slot must be complete. Return 1 on
 * success.
 */
int                 caliper_live_segment_publish(CaliperLiveSegment* s, int slot,
                                                 const CaliperLivePublish* meta);
/**
 * Publish a slot that the caller chose itself, by the rule of the Python ring that the slot is not
 * `published_slot`, not a claim and not pending. The other rules are those of `caliper_live_segment_publish`. Added
 * in protocol v2.
 */
int                 caliper_live_segment_publish_slot(CaliperLiveSegment* s, int slot,
                                                      const CaliperLivePublish* meta);
/**
 * Return 1 and store the two claims of the viewer in `reading` and `prev`, each `CALIPER_LIVE_SLOT_NONE` when free.
 * The loads use seq_cst order. Added in protocol v2.
 */
int                 caliper_live_segment_claims(CaliperLiveSegment* s, uint32_t* reading,
                                                uint32_t* prev);
/** Count one snapshot that the caller dropped because no slot was free. Added in protocol v2. */
void                caliper_live_segment_count_drop(CaliperLiveSegment* s);
/** Give back an acquired slot without a publish. */
void                caliper_live_segment_abandon(CaliperLiveSegment* s, int slot);

/* --- session discovery (any process; src/host/live/live_session.mm) -------- */

/** One live session that the listing found. */
typedef struct CaliperLiveSessionInfo {
    /** `sizeof(CaliperLiveSessionInfo)`. */
    size_t   struct_size;
    /** The process id of the client. */
    int32_t  pid;
    /** 1 when the client runs and its session is open. */
    int32_t  alive;
    /** The number of views in the control table now. */
    int32_t  view_count;
    /** The protocol version of the session file. */
    int32_t  protocol;
    /** The wall clock time of the opening of the session, in milliseconds since the Unix epoch. */
    uint64_t start_unix_ms;
    /** The process id of a viewer with a fresh heartbeat, or 0. */
    int32_t  viewer_pid;
    /** 1 when `protocol` equals `CALIPER_LIVE_PROTOCOL_VERSION`. */
    int32_t  compatible;
    /** The session token, NUL-terminated. */
    char     session[CALIPER_LIVE_SESSION_MAX + 4];
    /** The name of the script, NUL-terminated. */
    char     script[CALIPER_LIVE_SCRIPT_MAX];
    /** The path of the session file, NUL-terminated. */
    char     path[1024];
} CaliperLiveSessionInfo;

/**
 * Return the default data root, which is `$CALIPER_DATA_DIR`, else `caliper::app_data_dir()` with the `data_dir`
 * override of an embed core. Sessions live in `<root>/live`. The string is thread-local and valid until the next
 * call on this thread.
 */
const char* caliper_live_data_root(void);
/**
 * List the live sessions under `<data_root>/live` in the default root, newest first.
 * The call reclaims stale files, which are files of a dead or reused process id. It unlinks their shared memory names
 * and removes the file, and it does not list them. Thus a trainer that a signal killed leaks nothing past the next
 * listing. The call fills `min(n, cap)` entries and returns the number `n` of sessions, or -1 with a reason in
 * `caliper_live_last_error`. The argument `out` may be `NULL` when `cap` is 0. Any thread may call it.
 */
int caliper_live_list_sessions(CaliperLiveSessionInfo* out, int cap);
/**
 * List the live sessions as `caliper_live_list_sessions` does, under `<data_root>/live`, with `NULL` for the
 * default root.
 */
int caliper_live_list_sessions_in(const char* data_root, CaliperLiveSessionInfo* out, int cap);

/* --- viewer API (embed core; src/host/live/live_core.cpp) -----------------
 * Frame thread only. The host calls caliper_core_live_poll once per frame
 * before caliper_core_frame. Each live view becomes a typed live view of the
 * same name (caliper/live_views.h), which the poll updates in place from the
 * snapshot's MTLBuffer and byte offset, with no host copy. */
/** Defined as 1 when the header declares the live viewer calls. */
#define CALIPER_EMBED_HAVE_LIVE 1
/**
 * Defined as 1 when the header declares the session list, the attach by process id and the picker. Added in
 * protocol v2.
 */
#define CALIPER_EMBED_HAVE_LIVE_SESSIONS 1

/**
 * Attach to the session with the token `session`.
 * The call accepts a session that is still starting, when the control segment is absent or not yet initialized and a
 * session file names the token, or when `magic` is 0. It returns 1, and the poll keeps trying with `status.pending`
 * set to 1. It refuses an unknown token, a protocol mismatch and a session that another live viewer holds, because a
 * session has one viewer. Return 1 on success and 0 on refusal.
 */
int  caliper_core_live_attach(CaliperCore* core, const char* session);
/**
 * Attach to the session of the process `pid`, found by its session file under `data_root`, with `NULL` for the
 * default root. Added in protocol v2.
 */
int  caliper_core_live_attach_pid(CaliperCore* core, int32_t pid, const char* data_root);
/** Defined as 1 when the header declares `caliper_core_live_attach_url`. */
#define CALIPER_EMBED_HAVE_LIVE_NET 1
/**
 * Attach to a wire protocol v3 stream on this machine. The stream comes from a producer, which is `caliper.serve()`, or from a hub, which is `caliper serve`.
 * The `url` is the stream link `ws://127.0.0.1:<port>/v3/stream?token=<token>` or the page link
 * `http://127.0.0.1:<port>/?token=<token>`, and `localhost` works for `127.0.0.1`. An optional `session=<id>` parameter
 * names the session of a hub. The call waits up to 2 seconds for the server. It returns 0 with a reason in the last
 * error in four cases. The link is not a loopback stream link, nothing listens, the server refuses with HTTP 403 for
 * the token, the Host header or the Origin header, or the server breaks the protocol. It returns 1 once attached,
 * or while the server holds no session yet, with `status.pending` set to 1. The call subscribes every view of the
 * session with its default data view. The session of the status is the session id and never the token. The field
 * `client_alive` becomes 0 when the connection or the session ends, and the views keep their last pixels. The stream
 * reconnects after a drop. The views, the claims, the status, the detach call and the `caliper_core_live_view_*`
 * probes behave as for an attach to shared memory. The protocol is in `caliper/live_net.h`.
 */
int  caliper_core_live_attach_url(CaliperCore* core, const char* url);
/**
 * Detach from the session, drop the live views, release the claims and clear the heartbeat, and thus
 * `caliper_live_client_attached` turns false at once. Training is not affected.
 */
void caliper_core_live_detach(CaliperCore* core);
/**
 * Map and unmap the segments when the table changes, apply the newest completed snapshot of every view and write
 * the heartbeat. Return the number of views updated by this call, or -1 when the core is not attached, picker mode
 * included. A Detach that the user clicked in the status window happens in this call, and then it returns -1.
 */
int  caliper_core_live_poll(CaliperCore* core);
/**
 * Enable or disable the session picker, which is the landing of "Attach to process". While it is enabled and the
 * core is not attached, the poll refreshes the session list of `data_root`, with `NULL` for the default root, once
 * a second, and the frame draws it with an Attach button for each session. Detach in the status window returns to
 * the picker. Return 1 on success. Added in protocol v2.
 */
int  caliper_core_live_picker(CaliperCore* core, int enable, const char* data_root);

/** The counters and latency of the attached session, which `caliper_core_live_status` fills. */
typedef struct CaliperLiveStatus {
    /** `sizeof(CaliperLiveStatus)`. */
    size_t   struct_size;
    /** 1 when the core is attached to a session. */
    int32_t  attached;
    /** The process id of the client. */
    int32_t  client_pid;
    /** 0 once the client exited or closed the session. */
    int32_t  client_alive;
    /** The number of live views. */
    int32_t  views;
    /** The number of frames drawn since the attach. */
    uint64_t frames;
    /** The number of polls since the attach. */
    uint64_t polls;
    /** The number of view updates applied since the attach. */
    uint64_t updates;
    /** The number of publishes that the viewer never drew because a newer one arrived first. */
    uint64_t skipped;
    /** The sum of the snapshots that the client dropped because no slot was free. */
    uint64_t client_dropped;
    /** The number of updates that the viewer refused. */
    uint64_t errors;
    /** The number of latency samples in the current window, which starts at the last reset. */
    uint64_t latency_samples;
    /**
     * The median, the 95th percentile and the largest step-to-screen latency in milliseconds over the current
     * window. The latency is the viewer `CLOCK_UPTIME_RAW` time when the update is encoded minus the `show()` stamp
     * of the client. The frame that draws the update is committed in the same tick.
     */
    double   latency_ms_median, latency_ms_p95, latency_ms_max;
    /** The frames per second over the current window. */
    double   fps;
    /** The number of frames in the current window. */
    uint64_t window_frames;
    /** The number of view updates in the current window. */
    uint64_t window_updates;
    /**
     * The session token, or an empty string when the core is not attached. The string is valid until the next live
     * call. Added in protocol v2.
     */
    const char* session;
    /** The name of the script of the session. The string is valid until the next live call. Added in protocol v2. */
    const char* script;
    /**
     * The wall clock time of the opening of the session, in milliseconds since the Unix epoch. Added in protocol
     * v2.
     */
    uint64_t start_unix_ms;
    /** 1 when the core is attached and waits for a session that is still starting. Added in protocol v2. */
    int32_t  pending;
    /** 1 when the session picker is enabled. Added in protocol v2. */
    int32_t  picker;
    /** The number of sessions in the last listing of the picker. Added in protocol v2. */
    int32_t  sessions_listed;
    /** Reserved. */
    int32_t  reserved0;
} CaliperLiveStatus;

/** The state of one live view, which `caliper_core_live_view_status` fills. */
typedef struct CaliperLiveViewStatus {
    /** `sizeof(CaliperLiveViewStatus)`. */
    size_t      struct_size;
    /** The name of the view. The string is valid until the next live call. */
    const char* name;
    /** The kind of the view, a `CaliperLiveKind` value. */
    uint32_t    kind;
    /** The `seq` of the last publish that the viewer drew. */
    uint64_t    seq;
    /** The number of updates applied since the attach. */
    uint64_t    updates;
    /** The number of updates in the current window. */
    uint64_t    window_updates;
    /** The number of publishes that the viewer never drew. */
    uint64_t    skipped;
    /** The number of snapshots that the client dropped for this view. */
    uint64_t    client_dropped;
    /** The step-to-screen latency of the last update, in milliseconds. */
    double      latency_ms_last;
    /** The reason for the failure of the last update, or an empty string when it succeeded. */
    const char* error;
    /** For a series, the number of values appended so far in the last applied snapshot. Added in protocol v2. */
    uint64_t    series_total;
    /** For points, the number of dimensions of a point in the last applied snapshot. Added in protocol v2. */
    int32_t     points_dims;
    /** For points, the number of categorical classes in the last applied snapshot. Added in protocol v2. */
    int32_t     label_count;
    /** The number of meaningful stats values in the last applied snapshot. Added in protocol v2. */
    int32_t     stats_count;
    /** The slot that the last applied snapshot came from. Added in protocol v2. */
    int32_t     slot;
    /**
     * The path that applied the last snapshot, `live_views` or `v1.2` for the heatmap and image path. Added in
     * protocol v2.
     */
    const char* apply;
} CaliperLiveViewStatus;

/**
 * Fill `out` with the status of the session and return 1. A `reset_window` other than 0 starts a new latency and
 * fps window after the call fills `out`.
 */
int  caliper_core_live_status(CaliperCore* core, CaliperLiveStatus* out, int reset_window);
/** Fill `out` with the status of the live view at `index` and return 1, or return 0 when there is no such view. */
int  caliper_core_live_view_status(CaliperCore* core, int index, CaliperLiveViewStatus* out);
/**
 * Copy the raw samples of the current latency window, so call it before a reset. Each sample is a pair of the
 * viewer `CLOCK_UPTIME_RAW` time in nanoseconds when the update was encoded and the step-to-screen latency in
 * milliseconds. Copy up to `cap` pairs and return the number copied. Either array may be `NULL` to count only.
 */
size_t caliper_core_live_window_latencies(CaliperCore* core, uint64_t* t_ns, double* lat_ms, size_t cap);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_LIVE_H */
