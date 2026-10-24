/* ===========================================================================
 * caliper/live.h: live tensor streaming between a training process and a
 * viewer process (protocol v2, macOS / Metal). "Attach to app".
 *
 * THE MODEL
 *   The training process (the CLIENT: caliper.live.SessionTransport, driven by
 *   the snapshot ring in caliper/_ring.py) opens a LOCAL LIVE SESSION on its
 *   first show: POSIX shared memory for the GPU snapshots plus a SESSION FILE
 *   <data_root>/live/<pid>.json naming it. A viewer process (caliper-viewer, the
 *   Caliper app's "Attach to process", any embed host) LISTS the session files,
 *   ATTACHES to one, maps the same memory, wraps it as an MTLBuffer and draws
 *   the newest COMPLETED snapshot of every view in place through the typed live
 *   views (caliper/live_views.h). Training never waits for the viewer: a
 *   snapshot is ONE GPU-to-GPU copy on torch's own stream into a free slot, and
 *   the viewer samples at its own frame rate (skipping snapshots is by design).
 *   Local only: same user (names and files are 0600), no network.
 *
 * NEAR-FREE WHEN NOBODY WATCHES
 *   An attached viewer stores CLOCK_UPTIME_RAW ns into the control segment's
 *   viewer_heartbeat_ns on every poll. caliper_live_client_attached() is one
 *   atomic load and a clock read: "a heartbeat younger than
 *   CALIPER_LIVE_ATTACH_WINDOW_NS". With no viewer the client takes no
 *   snapshot at all (show() checks this and returns).
 *
 * SESSION FILES (<data_root>/live/<pid>.json; data_root = the desc's data_dir,
 * else $CALIPER_DATA_DIR, else caliper::app_data_dir(), which honours the embed
 * data_dir override; macOS default ~/Library/Application Support/Caliper)
 *   One flat JSON object written atomically (tmp + rename) by the client at
 *   open and on every view add/remove, removed at close:
 *     protocol, pid, proc_start_us (the pid's start time: guards pid reuse),
 *     session (the token), script, start_unix_ms, views [names],
 *     segments [shm names, the control segment first].
 *   caliper_live_list_sessions reads them; a file whose pid is gone (or reused)
 *   is STALE: its shm names (only names under its own "/<token>-" prefix) are
 *   unlinked and the file removed, so a SIGKILLed trainer leaks nothing past
 *   the next listing.
 *
 * SHARED MEMORY LAYOUT (all segments page-aligned; the client creates and
 * unlinks every name, a lister unlinks a dead client's names, the viewer only
 * maps):
 *   "/<session>-ctl"   one CaliperLiveControl: session header + a table of up
 *                      to 64 views. `generation` is a seqlock (odd while the
 *                      client edits the table); the viewer re-reads the table
 *                      when it changes. `magic` is stored LAST (release) at
 *                      open, so magic == 0 means "still starting".
 *   "/<session>-v<id>" one segment per view: a header page
 *                      (CaliperLiveSegmentHeader) then slot_count slots (3, or
 *                      1 for SERIES). Slot i = a page-aligned payload at
 *                      slot_offset[i] (row-major), an optional aux region at
 *                      aux_offset[i] (POINTS labels / per-point values, f32),
 *                      and a stats region at stats_offset[i]
 *                      (CALIPER_LIVE_STATS_BYTES of f32: vmin, vmax, then kind-
 *                      specific, written on the GPU by torch.aminmax(out=)...).
 *   Each process wraps ITS mapping of a segment as one MTLBuffer
 *   (newBufferWithBytesNoCopy, shared storage); a slot is that buffer plus a
 *   byte offset, so the same physical pages are the training process's torch
 *   tensor and the viewer's draw source. No host copy anywhere for MPS.
 *
 * SLOT PROTOCOL (lock-free, per view; sync words are accessed with the
 * compiler's __atomic builtins on the shared mapping):
 *   The viewer holds at most TWO claims per view: viewer_reading_slot (the
 *   snapshot it applied last; a points view draws it every frame) and
 *   viewer_prev_slot (the one before, until the GPU read of it retired:
 *   caliper_core_live_view_read_done). A slot the client may write is one that
 *   is neither published_slot nor either claim nor one of its own pending
 *   (written, unpublished) slots; with none left the client DROPS the new
 *   snapshot (dropped_client) rather than overwrite one in flight, so the
 *   oldest pending snapshot always completes and publishes.
 *   client publish : after the snapshot's GPU work COMPLETED (torch.mps.Event
 *                    query), fill slot[s] metadata, store published_slot = s
 *                    (seq_cst), then seq += 1 (release).
 *   viewer take    : on seq change, store viewer_prev_slot = the current claim,
 *                    then s = published_slot; store viewer_reading_slot = s
 *                    (seq_cst); re-load published_slot and retry if it moved.
 *                    With seq_cst on both sides (a Dekker pair) either the
 *                    client sees the claim and avoids s, or the viewer sees the
 *                    newer publish and moves on. It takes a new snapshot of a
 *                    view only once viewer_prev_slot is free again (polled,
 *                    never waited on).
 *   SERIES views have ONE slot that is never claimed: the payload is an
 *   append-only ring of capacity M that the client keeps writing; each publish
 *   carries series_total and the viewer copies the window it draws.
 *
 * VERSIONING: magic + version + struct_size on every shared struct; fields are
 * append-only (v2 appended the aux region, the second claim, the slot meta for
 * points/series/histogram and the session identity). A reader refuses another
 * magic or protocol version and a smaller struct_size than its own.
 * CaliperDType values (caliper/tensor.h) are the dtypes.
 *
 * THREADING: the client functions are safe to call from any one thread at a
 * time per client (the Python glue serialises them with the GIL).
 * caliper_live_client_attached may be called from any thread. The listing
 * functions are thread-safe. The viewer functions (caliper_core_live_*) are
 * FRAME-THREAD ONLY, like the embed views they drive.
 * ===========================================================================*/
#ifndef CALIPER_LIVE_H
#define CALIPER_LIVE_H

#include <stddef.h>
#include <stdint.h>

#include <caliper/embed.h>   /* CaliperCore (viewer side) */

#ifdef __cplusplus
extern "C" {
#endif

#define CALIPER_LIVE_PROTOCOL_VERSION 2u
#define CALIPER_LIVE_CTL_MAGIC  0x4c54434556494c43ull   /* "CLIVECTL" little-endian */
#define CALIPER_LIVE_SEG_MAGIC  0x4745534556494c43ull   /* "CLIVESEG" little-endian */

#define CALIPER_LIVE_MAX_VIEWS    64
#define CALIPER_LIVE_SLOTS        3    /* slots of every kind but SERIES (1)        */
#define CALIPER_LIVE_NAME_MAX     64   /* view name incl. NUL */
#define CALIPER_LIVE_SEGNAME_MAX  32   /* shm name incl. NUL (macOS PSHMNAMLEN 31) */
#define CALIPER_LIVE_SESSION_MAX  20   /* session token chars, [A-Za-z0-9._-] */
#define CALIPER_LIVE_SCRIPT_MAX   128  /* script name incl. NUL (v2)                */
#define CALIPER_LIVE_STATS_BYTES  256  /* per-slot stats region (64 f32)            */
#define CALIPER_LIVE_SLOT_NONE    0xffffffffu
/* caliper_live_client_attached(): a viewer heartbeat younger than this. */
#define CALIPER_LIVE_ATTACH_WINDOW_NS 1000000000ull

/* View kinds (values shared with caliper/live_views.h). */
typedef enum CaliperLiveKind {
    CALIPER_LIVE_KIND_NONE      = 0,
    CALIPER_LIVE_KIND_HEATMAP   = 1,   /* (H,W) f32, colormapped                    */
    CALIPER_LIVE_KIND_IMAGE     = 2,   /* (H,W,4) u8 RGBA                           */
    CALIPER_LIVE_KIND_POINTS    = 3,   /* (N,3) f32 positions, aux (N,) f32 optional */
    CALIPER_LIVE_KIND_SERIES    = 4,   /* (M,) f32 append-only ring, ONE slot        */
    CALIPER_LIVE_KIND_HISTOGRAM = 5    /* (B,) f32 bin counts over [stats0, stats1]  */
} CaliperLiveKind;

typedef enum CaliperLiveEntryState {
    CALIPER_LIVE_ENTRY_FREE   = 0,
    CALIPER_LIVE_ENTRY_ACTIVE = 1
} CaliperLiveEntryState;

typedef enum CaliperLiveSessionState {
    CALIPER_LIVE_SESSION_OPEN   = 1,
    CALIPER_LIVE_SESSION_CLOSED = 2    /* client closed cleanly; names unlinked */
} CaliperLiveSessionState;

/* Per-snapshot mapping flags (CaliperLiveSlotMeta.flags). The viewer maps them
 * onto CALIPER_LIVE_APPLY_* (caliper/live_views.h). */
enum {
    CALIPER_LIVE_SNAP_AUTO_VMIN = 1u << 0,  /* vmin = stats[0] (GPU aminmax), else meta.vmin */
    CALIPER_LIVE_SNAP_AUTO_VMAX = 1u << 1,  /* vmax = stats[1] (GPU aminmax), else meta.vmax */
    CALIPER_LIVE_SNAP_RAW       = 1u << 2   /* v2: heatmap/image drawn 1:1 at the origin (tests) */
};

/* --- shared structs (protocol v2) ---------------------------------------- */

/* One row of the control table. Written by the client inside an odd
 * generation; read by the viewer as a snapshot between two equal even
 * generations. */
typedef struct CaliperLiveViewEntry {
    uint32_t struct_size;       /* sizeof(CaliperLiveViewEntry)                   */
    uint32_t state;             /* CaliperLiveEntryState                          */
    uint32_t kind;              /* CaliperLiveKind                                */
    int32_t  dtype;             /* CaliperDType of the payload                    */
    uint32_t ndim;
    uint32_t slot_count;        /* CALIPER_LIVE_SLOTS, 1 for SERIES               */
    int64_t  shape[4];
    uint64_t view_id;           /* per-session, never reused; names the segment   */
    uint64_t slot_bytes;        /* payload bytes of one snapshot (exact extent)   */
    uint64_t stats_bytes;       /* CALIPER_LIVE_STATS_BYTES                       */
    uint64_t segment_bytes;     /* total mapping length of the view segment       */
    char     name[CALIPER_LIVE_NAME_MAX];
    char     segment[CALIPER_LIVE_SEGNAME_MAX];
    /* v2 */
    uint64_t aux_bytes;         /* per-slot aux region (f32), 0 = none            */
    int64_t  aux_count;         /* aux f32 elements, 0 = none                     */
} CaliperLiveViewEntry;

typedef struct CaliperLiveControl {
    uint64_t magic;             /* CALIPER_LIVE_CTL_MAGIC, stored last (release)  */
    uint32_t version;           /* CALIPER_LIVE_PROTOCOL_VERSION                  */
    uint32_t struct_size;       /* sizeof(CaliperLiveControl)                     */
    uint32_t entry_size;        /* sizeof(CaliperLiveViewEntry)                   */
    uint32_t max_views;         /* CALIPER_LIVE_MAX_VIEWS                         */
    int32_t  client_pid;
    uint32_t state;             /* CaliperLiveSessionState (atomic)               */
    uint64_t generation;        /* seqlock over views[] (atomic; odd = editing)   */
    /* Written by the viewer (atomic): who is attached and when it last polled,
     * CLOCK_UPTIME_RAW ns. 0 = no viewer (a detaching viewer zeroes both). */
    int32_t  viewer_pid;
    uint32_t reserved0;
    uint64_t viewer_heartbeat_ns;
    uint64_t reserved[6];
    CaliperLiveViewEntry views[CALIPER_LIVE_MAX_VIEWS];
    /* v2: session identity, written once before magic */
    uint64_t start_unix_ms;     /* wall clock at open                             */
    char     session[CALIPER_LIVE_SESSION_MAX + 4];   /* the token, NUL-terminated */
    char     script[CALIPER_LIVE_SCRIPT_MAX];          /* e.g. "train.py"          */
} CaliperLiveControl;

/* Per-slot metadata, written by the client BEFORE it publishes the slot. */
typedef struct CaliperLiveSlotMeta {
    uint64_t seq;               /* the publish sequence number this slot carries  */
    uint64_t stamp_ns;          /* client CLOCK_UPTIME_RAW at the show() call     */
    int32_t  colormap;          /* CaliperColormap (heatmaps, continuous points)  */
    uint32_t flags;             /* CALIPER_LIVE_SNAP_*                            */
    float    vmin, vmax;        /* fixed range for the non-AUTO ends              */
    uint64_t reserved[2];
    /* v2 */
    uint64_t series_total;      /* SERIES: values appended so far                 */
    int32_t  points_dims;       /* POINTS: 2 or 3                                 */
    int32_t  label_count;       /* POINTS: categorical classes in aux, 0 = values */
    int32_t  stats_count;       /* meaningful f32 in the slot's stats region      */
    uint32_t reserved1;
} CaliperLiveSlotMeta;

/* The header page of a view segment. */
typedef struct CaliperLiveSegmentHeader {
    uint64_t magic;             /* CALIPER_LIVE_SEG_MAGIC                         */
    uint32_t version;           /* CALIPER_LIVE_PROTOCOL_VERSION                  */
    uint32_t struct_size;       /* sizeof(CaliperLiveSegmentHeader)               */
    int32_t  writer_pid;
    uint32_t slot_count;        /* CALIPER_LIVE_SLOTS, 1 for SERIES               */
    uint64_t view_id;
    uint64_t payload_bytes;     /* exact snapshot extent                          */
    uint64_t stats_bytes;
    uint64_t segment_bytes;
    uint64_t slot_offset[CALIPER_LIVE_SLOTS];   /* payload, page-aligned          */
    uint64_t stats_offset[CALIPER_LIVE_SLOTS];  /* stats f32[0]=vmin, [1]=vmax... */

    /* --- sync words (atomic) --------------------------------------------- */
    uint64_t seq;                   /* publishes so far; release-stored last     */
    uint32_t published_slot;        /* newest completed snapshot or SLOT_NONE    */
    uint32_t viewer_reading_slot;   /* viewer claim (last applied) or SLOT_NONE  */
    uint64_t stamp_ns;              /* stamp of the latest publish               */
    uint64_t dropped_client;        /* snapshots dropped: no free slot (busy)    */
    uint64_t viewer_seq;            /* last seq the viewer drew (viewer-written) */
    uint64_t skipped_viewer;        /* publishes never drawn (viewer-written)    */
    uint64_t reserved[4];

    CaliperLiveSlotMeta slot[CALIPER_LIVE_SLOTS];
    /* v2 */
    uint32_t kind;                  /* CaliperLiveKind                           */
    uint32_t viewer_prev_slot;      /* second viewer claim (atomic) or SLOT_NONE */
    uint64_t aux_bytes;             /* per-slot aux region, 0 = none             */
    uint64_t aux_offset[CALIPER_LIVE_SLOTS];
} CaliperLiveSegmentHeader;

/* --- client API (training process) ----------------------------------------
 * Implemented in src/host/live/live_client.mm. Each function returns 0/NULL on
 * refusal with a reason in caliper_live_last_error() (thread-local). */
typedef struct CaliperLiveClient  CaliperLiveClient;
typedef struct CaliperLiveSegment CaliperLiveSegment;

enum {
    CALIPER_LIVE_CLIENT_NO_SESSION_FILE = 1u << 0   /* v2: shm only, no session file */
};

typedef struct CaliperLiveClientDesc {
    size_t      struct_size;    /* sizeof(CaliperLiveClientDesc)                  */
    const char* session;        /* NULL -> "clp-<pid>-<6 hex>"; 1..20 of [A-Za-z0-9._-] */
    void*       mtl_device;     /* id<MTLDevice> for the segment MTLBuffers (the
                                 * producer's device, e.g. [torchbuffer device]);
                                 * NULL -> MTLCreateSystemDefaultDevice()         */
    /* v2 */
    const char* script;         /* shown by viewers; NULL -> the process name     */
    const char* data_dir;       /* session-file root; NULL -> the default (above) */
    uint32_t    flags;          /* CALIPER_LIVE_CLIENT_*                          */
} CaliperLiveClientDesc;

typedef struct CaliperLiveViewDesc {
    size_t      struct_size;    /* sizeof(CaliperLiveViewDesc)                    */
    const char* name;           /* 1..63 bytes, unique in the session             */
    uint32_t    kind;           /* CaliperLiveKind                                */
    int32_t     dtype;          /* CaliperDType; must match the kind              */
    int32_t     ndim;
    int64_t     shape[4];
    /* v2 */
    uint32_t    slots;          /* 0 -> the kind's count (3; SERIES 1)            */
    int64_t     aux_count;      /* POINTS: N to give every slot an (N,) f32 aux   */
} CaliperLiveViewDesc;
/* Kinds: HEATMAP (H,W) f32; IMAGE (H,W,4) u8; POINTS (N,3) f32 (+ aux N);
 * SERIES (M,) f32, 1 slot; HISTOGRAM (B,) f32. */

typedef struct CaliperLiveSegmentInfo {
    size_t   struct_size;       /* sizeof(CaliperLiveSegmentInfo)                 */
    void*    mtl_buffer;        /* id<MTLBuffer> over the whole mapping (borrowed;
                                 * NULL on a device-less machine)                 */
    void*    host_ptr;          /* CPU address of the mapping                     */
    uint64_t segment_bytes;
    uint64_t payload_bytes;
    uint64_t stats_bytes;
    uint64_t slot_offset[CALIPER_LIVE_SLOTS];
    uint64_t stats_offset[CALIPER_LIVE_SLOTS];
    uint32_t slot_count;
    int32_t  table_index;       /* row in the control table, -1 once removed      */
    const char* segment_name;   /* valid while the segment lives                  */
    /* live counters from the shared header */
    uint64_t seq;
    uint32_t published_slot;
    uint32_t viewer_reading_slot;
    uint64_t dropped_client;
    uint64_t viewer_seq;
    uint64_t skipped_viewer;
    /* v2 */
    uint32_t kind;
    uint32_t viewer_prev_slot;
    uint64_t aux_bytes;
    uint64_t aux_offset[CALIPER_LIVE_SLOTS];
} CaliperLiveSegmentInfo;

typedef struct CaliperLivePublish {
    size_t   struct_size;       /* sizeof(CaliperLivePublish)                     */
    uint64_t stamp_ns;          /* CLOCK_UPTIME_RAW (== Python time.monotonic_ns) */
    int32_t  colormap;
    uint32_t flags;             /* CALIPER_LIVE_SNAP_*                            */
    float    vmin, vmax;
    /* v2 */
    uint64_t series_total;
    int32_t  points_dims;
    int32_t  label_count;
    int32_t  stats_count;       /* 0 -> 2                                         */
} CaliperLivePublish;

const char*         caliper_live_last_error(void);
uint64_t            caliper_live_now_ns(void);   /* CLOCK_UPTIME_RAW in ns */

/* Create the session: the control segment, then (unless NO_SESSION_FILE) the
 * session file. */
CaliperLiveClient*  caliper_live_client_open(const CaliperLiveClientDesc* desc);
const char*         caliper_live_client_session(const CaliperLiveClient* c);
/* v2: the session file's path, "" without one. */
const char*         caliper_live_client_session_file(const CaliperLiveClient* c);
/* v2: 1 while a viewer heartbeat is younger than CALIPER_LIVE_ATTACH_WINDOW_NS.
 * One atomic load + one clock read; safe from any thread; 0 on NULL. */
int                 caliper_live_client_attached(const CaliperLiveClient* c);
/* Viewer heartbeat as seen by the client: *viewer_pid (0 = none) and ns since
 * its last poll (UINT64_MAX when none). Returns 1. */
int                 caliper_live_client_viewer(const CaliperLiveClient* c,
                                               int32_t* viewer_pid, uint64_t* age_ns);
/* Create a view and its segment (and rewrite the session file). The client
 * keeps one reference; the returned pointer is borrowed (retain it to outlive
 * remove_view/close). */
CaliperLiveSegment* caliper_live_client_add_view(CaliperLiveClient* c,
                                                 const CaliperLiveViewDesc* desc);
/* Drop the view from the table and unlink its name; the mapping lives on while
 * references remain (DLPack capsules retain the segment). */
int                 caliper_live_client_remove_view(CaliperLiveClient* c,
                                                    CaliperLiveSegment* s);
/* Mark the session closed, unlink every name, remove the session file, drop
 * the client's references. Never waits for a viewer. Safe on NULL. Mappings
 * live on while references remain. */
void                caliper_live_client_close(CaliperLiveClient* c);

void                caliper_live_segment_retain(CaliperLiveSegment* s);
void                caliper_live_segment_release(CaliperLiveSegment* s);
int                 caliper_live_segment_info(CaliperLiveSegment* s,
                                              CaliperLiveSegmentInfo* out);
/* C-side slot choice (tests, C producers; the Python ring chooses its own): a
 * slot that is neither published_slot nor a viewer claim nor a slot of this
 * client still holding an unpublished snapshot. Returns the slot index, or -1
 * with *busy = 1 when every candidate is in flight: the caller drops THIS
 * snapshot (counted in dropped_client). -1 with *busy = 0: segment removed.
 * SERIES: always slot 0. */
int                 caliper_live_segment_acquire(CaliperLiveSegment* s, int* busy);
/* Publish an ACQUIRED `slot` (its GPU writes must have COMPLETED). 1 on success. */
int                 caliper_live_segment_publish(CaliperLiveSegment* s, int slot,
                                                 const CaliperLivePublish* meta);
/* v2: publish a slot the caller chose itself (caliper/_ring.py's rule: never
 * published_slot nor a claim nor pending). Same rules otherwise. */
int                 caliper_live_segment_publish_slot(CaliperLiveSegment* s, int slot,
                                                      const CaliperLivePublish* meta);
/* v2: the viewer's two claims (seq_cst loads; SLOT_NONE when free). Returns 1. */
int                 caliper_live_segment_claims(CaliperLiveSegment* s, uint32_t* reading,
                                                uint32_t* prev);
/* v2: count a snapshot the caller dropped for want of a free slot. */
void                caliper_live_segment_count_drop(CaliperLiveSegment* s);
/* Give back an acquired slot without publishing it. */
void                caliper_live_segment_abandon(CaliperLiveSegment* s, int slot);

/* --- session discovery (any process; src/host/live/live_session.mm) -------- */
typedef struct CaliperLiveSessionInfo {
    size_t   struct_size;       /* sizeof(CaliperLiveSessionInfo)                 */
    int32_t  pid;
    int32_t  alive;             /* 1: the client runs and its session is open     */
    int32_t  view_count;        /* views in the control table now                 */
    int32_t  protocol;          /* the file's protocol version                    */
    uint64_t start_unix_ms;
    int32_t  viewer_pid;        /* a viewer with a fresh heartbeat, else 0        */
    int32_t  compatible;        /* 1: protocol == CALIPER_LIVE_PROTOCOL_VERSION   */
    char     session[CALIPER_LIVE_SESSION_MAX + 4];
    char     script[CALIPER_LIVE_SCRIPT_MAX];
    char     path[1024];        /* the session file                               */
} CaliperLiveSessionInfo;

/* The resolved default data root ($CALIPER_DATA_DIR, else caliper::app_data_dir(),
 * which honours an embed core's data_dir override): sessions live in
 * <root>/live. Thread-local string, valid until the next call on this thread. */
const char* caliper_live_data_root(void);
/* List the live sessions under <data_root>/live (NULL -> the default root),
 * newest first. Stale files (dead or reused pid) are reclaimed: their shm names
 * unlinked, the file removed; they are not listed. Fills min(n, cap) entries
 * (out may be NULL with cap 0) and returns n, or -1 with last_error. */
int caliper_live_list_sessions(CaliperLiveSessionInfo* out, int cap);
int caliper_live_list_sessions_in(const char* data_root, CaliperLiveSessionInfo* out, int cap);

/* --- viewer API (embed core; src/host/live/live_core.cpp) -----------------
 * FRAME-THREAD ONLY. The host calls caliper_core_live_poll once per frame
 * BEFORE caliper_core_frame; each live view becomes a typed live view of the
 * same name (caliper/live_views.h) updated in place from the snapshot's
 * MTLBuffer + byte offset (no host copy; heatmaps path "compute-imported"). */
#define CALIPER_EMBED_HAVE_LIVE 1
#define CALIPER_EMBED_HAVE_LIVE_SESSIONS 1   /* v2: list / attach-by-pid / picker */

/* Attach to the session `session` (token). Tolerant of a session that is still
 * starting (control segment absent or not yet initialised while a session file
 * names the token, or present with magic 0): returns 1 and the poll keeps
 * retrying (status.pending = 1). Refuses an unknown token, a protocol mismatch
 * and a session another live viewer is attached to (one viewer per session). */
int  caliper_core_live_attach(CaliperCore* core, const char* session);  /* 1 ok */
/* v2: attach to the session of process `pid` (its session file under
 * data_root/live, NULL = the default root). */
int  caliper_core_live_attach_pid(CaliperCore* core, int32_t pid, const char* data_root);
/* Detach: drop the live views, release the claims, clear the heartbeat (the
 * client's attached() turns false at once). Training is untouched. */
void caliper_core_live_detach(CaliperCore* core);
/* Map/unmap on table changes, apply the newest completed snapshot of every
 * view, write the heartbeat. Returns the number of views updated this call,
 * -1 when not attached (incl. picker mode). Performs a Detach clicked in the
 * status window (then returns -1). */
int  caliper_core_live_poll(CaliperCore* core);
/* v2: the session picker ("attach to process" landing). While enabled and not
 * attached, the poll refreshes the session list of data_root (NULL = default)
 * once a second and the frame draws it with an Attach button per session;
 * Detach in the status window returns to the picker. 1 ok. */
int  caliper_core_live_picker(CaliperCore* core, int enable, const char* data_root);

typedef struct CaliperLiveStatus {
    size_t   struct_size;       /* sizeof(CaliperLiveStatus)                      */
    int32_t  attached;
    int32_t  client_pid;
    int32_t  client_alive;      /* 0 once the client exited or closed            */
    int32_t  views;
    uint64_t frames;            /* frames drawn since attach                      */
    uint64_t polls;
    uint64_t updates;           /* view updates applied since attach              */
    uint64_t skipped;           /* publishes never drawn (newer arrived first)    */
    uint64_t client_dropped;    /* client-side busy-dropped snapshots (sum)       */
    uint64_t errors;            /* refused updates                                */
    /* step-to-screen latency over the current window (since the last reset):
     * viewer CLOCK_UPTIME_RAW when the update is encoded minus the client's
     * show() stamp; the frame drawing it is committed in the same tick.        */
    uint64_t latency_samples;
    double   latency_ms_median, latency_ms_p95, latency_ms_max;
    double   fps;               /* frames/s over the current window               */
    uint64_t window_frames;
    uint64_t window_updates;
    /* v2 (strings valid until the next live call) */
    const char* session;        /* token ("" when not attached)                   */
    const char* script;
    uint64_t start_unix_ms;
    int32_t  pending;           /* attached, waiting for a starting session       */
    int32_t  picker;            /* the session picker is enabled                  */
    int32_t  sessions_listed;   /* picker: sessions in the last listing           */
    int32_t  reserved0;
} CaliperLiveStatus;

typedef struct CaliperLiveViewStatus {
    size_t      struct_size;    /* sizeof(CaliperLiveViewStatus)                  */
    const char* name;           /* valid until the next live call                 */
    uint32_t    kind;
    uint64_t    seq;            /* last drawn publish seq                         */
    uint64_t    updates;
    uint64_t    window_updates;
    uint64_t    skipped;
    uint64_t    client_dropped;
    double      latency_ms_last;
    const char* error;          /* "" when the last update succeeded             */
    /* v2: what the last applied snapshot carried */
    uint64_t    series_total;
    int32_t     points_dims;
    int32_t     label_count;
    int32_t     stats_count;
    int32_t     slot;           /* slot it came from                              */
    const char* apply;          /* "live_views" or "v1.2" (the heatmap/image path) */
} CaliperLiveViewStatus;

/* reset_window != 0 starts a new latency/fps window after filling `out`. */
int  caliper_core_live_status(CaliperCore* core, CaliperLiveStatus* out, int reset_window);
int  caliper_core_live_view_status(CaliperCore* core, int index, CaliperLiveViewStatus* out);
/* Raw samples of the current latency window (call before a reset): up to `cap`
 * pairs of (viewer CLOCK_UPTIME_RAW ns when the update was encoded, step-to-
 * screen ms). Returns the number copied. Either array may be NULL to count. */
size_t caliper_core_live_window_latencies(CaliperCore* core, uint64_t* t_ns, double* lat_ms, size_t cap);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_LIVE_H */
