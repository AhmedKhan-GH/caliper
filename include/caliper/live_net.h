/* ===========================================================================
 * caliper/live_net.h: wire protocol v3, live data streaming over a WebSocket.
 *
 * This header defines the constants and the frame header of protocol v3. The
 * reference page docs/references/wire-protocol.md specifies every message and
 * embeds this file verbatim; tests/fixtures/wire_v3 holds the golden frames
 * that every implementation (python/caliper/_wire.py, the native NetSource,
 * the hub, the TypeScript viewer) decodes in its tests. Plain C99 and C++, no
 * platform headers, no functions to link: other languages port it by hand.
 *
 * Protocol v2 (caliper/live.h, shared memory) is unchanged. v2 stays the
 * zero-copy path between processes on one machine; v3 sends copies of reduced
 * snapshots over a socket to browsers, notebooks, the hub and the native app.
 *
 * ROLES
 *   producer  the training process (caliper.serve() in Python). It owns the
 *             tensors and computes every data view on its own device.
 *   viewer    a browser element, the Jupyter widget or the native app's
 *             NetSource. It draws what it receives and computes nothing.
 *   hub       `caliper serve`: producers connect out to it, viewers connect to
 *             it, and it forwards between them. It holds several sessions.
 *   A connection has a CLIENT (the side that opened it: a viewer, or a producer
 *   connecting to a hub) and a SERVER (caliper.serve() or the hub).
 *
 * ENDPOINT AND HANDSHAKE
 *   One WebSocket endpoint, CALIPER_NET_PATH, on 127.0.0.1 only. The session
 *   token travels as the query parameter CALIPER_NET_TOKEN_PARAM of the upgrade
 *   request. Before the upgrade the server answers HTTP 404 for another path,
 *   and HTTP 403 when the Host header is not "127.0.0.1:<port>" or
 *   "localhost:<port>", when the Origin header is not accepted (the page
 *   states the Origin rule exactly), or when the token is missing or wrong
 *   (compared in constant time). No subprotocol, no permessage-deflate. Then
 *   the client sends HELLO, the server answers WELCOME, and nothing but BYE
 *   may precede them.
 *
 * FRAMES
 *   Every WebSocket message is ONE binary frame (a text message closes the
 *   connection with CALIPER_NET_CLOSE_UNSUPPORTED_DATA). All integers are
 *   little-endian. A frame is, in this order:
 *     1. CaliperNetFrameHeader, CALIPER_NET_HEADER_BYTES (24) bytes;
 *     2. section_count uint32 section lengths in bytes;
 *     3. meta_len bytes of metadata: one UTF-8 JSON object (RFC 8259), at
 *        least 2 bytes, no byte order mark, nesting depth at most
 *        CALIPER_NET_MAX_JSON_DEPTH;
 *     4. section_count payload sections. Section i starts at the first offset
 *        that is a multiple of CALIPER_NET_SECTION_ALIGN (8), counted from the
 *        first byte of the frame, at or after the end of the part before it
 *        (the metadata for i = 0). The sender fills the gaps with zero bytes;
 *        the receiver ignores them.
 *   frame_len is the total: the end of the last section, or the end of the
 *   metadata when there is none. No bytes follow.
 *   When section_count > 0 the metadata key "sections" is an array of
 *   section_count descriptors {"name": string, "dtype": CaliperNetDType,
 *   "shape": [int...], "encoding": CaliperNetEncoding (optional, 0)}; names are
 *   unique in the frame. A RAW section holds prod(shape) * item size bytes,
 *   row-major, little-endian. "sections" is reserved in every message.
 *
 * DECODING ORDER (every implementation applies these checks in this order, so
 * all of them reject a bad frame with the same close code):
 *   1. fewer than 8 bytes                                -> PROTOCOL_ERROR
 *   2. magic != CALIPER_NET_MAGIC                        -> PROTOCOL_ERROR
 *   3. major != CALIPER_NET_PROTOCOL_MAJOR               -> UNSUPPORTED_MAJOR
 *   4. fewer than 24 bytes                               -> PROTOCOL_ERROR
 *   5. frame_len > CALIPER_NET_MAX_FRAME_BYTES, meta_len > CALIPER_NET_MAX_META_BYTES
 *      or section_count > CALIPER_NET_MAX_SECTIONS      -> TOO_BIG
 *   6. message length != frame_len                       -> PROTOCOL_ERROR
 *   7. the layout above does not end exactly at frame_len -> PROTOCOL_ERROR
 *   8. metadata not a JSON object as stated above        -> PROTOCOL_ERROR
 *   9. "sections" missing, with another count, a malformed descriptor, or a
 *      RAW section of a known dtype whose length disagrees with its shape
 *                                                        -> PROTOCOL_ERROR
 *   minor, flags, reserved and the padding bytes are never grounds to reject.
 *   A frame that passes is well formed. A well-formed frame whose message is
 *   invalid (a missing field, an unknown view, a bad data view) is answered
 *   with an ERROR message and the connection stays open.
 *
 * VERSIONING
 *   Major 3. A peer closes with CALIPER_NET_CLOSE_UNSUPPORTED_MAJOR on another
 *   major. Within major 3 a receiver ignores message types, metadata keys,
 *   capabilities, frame flags and sections it does not know, and a minor
 *   version only adds those. A sender uses an addition only when the peer's
 *   minor (the header of its HELLO or WELCOME) or one of its capabilities
 *   says the peer knows it.
 *   3.1 adds controls (capability "controls": the run state, parameters and
 *   actions a script registers, CONTROLS, CONTROL, CONTROL_REPLY and the
 *   WELCOME key "control") and linked selection (capability "selection":
 *   item spaces in VIEWS, "space" and "items" in a select event, SELECTION).
 *   Every 3.1 addition, new keys of 3.0 messages included, is used only with
 *   a peer that announced its capability; the minor alone never enables one.
 *   3.0 messages keep their meaning and encoding.
 *
 * FLOW CONTROL
 *   Nothing in the protocol lets a slow viewer block a producer. Per connection
 *   a sender keeps at most ONE undelivered snapshot per subscription (a view and
 *   its data view) and replaces it with a newer one; series values are
 *   coalesced into one pending SERIES message per subscription and never
 *   dropped; control messages go in order and first. A producer takes no
 *   snapshot of a view that no subscription names. Series are the exception
 *   that makes backfill possible: a producer retains appended values whether or
 *   not anyone subscribes, without synchronising with the device.
 *
 * CONTROLS (3.1)
 *   A script registers parameters and actions and calls a checkpoint in its
 *   training loop. A viewer sends CONTROL (pause, resume, step, set a
 *   parameter, run an action); the producer validates it, applies it on the
 *   training thread at the next checkpoint, answers the sender with
 *   CONTROL_REPLY and announces the new state to every viewer with CONTROLS.
 *   A hub forwards CONTROL to the session's producer only when it was started
 *   to allow control; WELCOME's "control" tells a viewer whether it may.
 *
 * SECURITY
 *   Loopback (127.0.0.1) and the session token only, until a dedicated security
 *   review. The stream carries data, viewer events (selections, picks, hovers
 *   that Python code can read) and, from 3.1, control requests that name a
 *   parameter or an action the script registered. It never carries host
 *   commands or the command layer. No message carries code, and the only code
 *   a message makes a producer run is an action the script registered, by its
 *   exact name and with no argument from the viewer.
 * ===========================================================================*/
#ifndef CALIPER_LIVE_NET_H
#define CALIPER_LIVE_NET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- version, endpoint ---------------------------------------------------- */
#define CALIPER_NET_PROTOCOL_MAJOR 3u
#define CALIPER_NET_PROTOCOL_MINOR 1u
#define CALIPER_NET_MAGIC          0x574c4143u   /* bytes "CALW" as a little-endian uint32 */
#define CALIPER_NET_PATH           "/v3/stream"
#define CALIPER_NET_TOKEN_PARAM    "token"

/* --- frame layout ---------------------------------------------------------- */
#define CALIPER_NET_HEADER_BYTES   24u
#define CALIPER_NET_SECTION_ALIGN  8u

/* The fixed header at offset 0 of every frame. The struct matches the bytes on
 * a little-endian machine; read it with memcpy (a received buffer need not be
 * aligned). It is followed by uint32_t section_len[section_count]. */
typedef struct CaliperNetFrameHeader {
    uint32_t magic;          /* offset  0: CALIPER_NET_MAGIC                        */
    uint16_t major;          /* offset  4: CALIPER_NET_PROTOCOL_MAJOR               */
    uint16_t minor;          /* offset  6: the sender's minor version               */
    uint16_t type;           /* offset  8: CaliperNetMessageType                    */
    uint16_t flags;          /* offset 10: CALIPER_NET_FRAME_*; none in 3.1, so 0   */
    uint32_t frame_len;      /* offset 12: bytes of the whole frame, header included */
    uint32_t meta_len;       /* offset 16: bytes of the JSON metadata               */
    uint16_t section_count;  /* offset 20: 0..CALIPER_NET_MAX_SECTIONS              */
    uint16_t reserved;       /* offset 22: 0 when sent, ignored when received       */
} CaliperNetFrameHeader;

/* C99 has no static_assert: an array of negative size fails the build. */
typedef char caliper_net_header_size_check[sizeof(CaliperNetFrameHeader) == 24 ? 1 : -1];

/* Frame flags: none are defined in 3.1. A receiver ignores bits it does not
 * know; a flag that changes how a frame decodes is only sent to a peer that
 * announced the capability for it. */
#define CALIPER_NET_FRAME_FLAGS_NONE 0x0000u

/* --- message types (header.type) -------------------------------------------
 * Direction: C = the client of the connection, S = its server; V viewer,
 * P producer, H hub. Type 0 is never sent. */
typedef enum CaliperNetMessageType {
    CALIPER_NET_MSG_HELLO        = 0x0001,  /* C -> S   role, capabilities, producer identity */
    CALIPER_NET_MSG_WELCOME      = 0x0002,  /* S -> C   role, capabilities, session identity  */
    CALIPER_NET_MSG_BYE          = 0x0003,  /* either   reason, then the sender closes        */
    CALIPER_NET_MSG_ERROR        = 0x0004,  /* either   code, message, fatal                  */
    CALIPER_NET_MSG_PING         = 0x0005,  /* either   nonce                                 */
    CALIPER_NET_MSG_PONG         = 0x0006,  /* either   the ping's nonce                      */
    CALIPER_NET_MSG_SESSIONS     = 0x0010,  /* S -> V   the sessions the server holds         */
    CALIPER_NET_MSG_ATTACH       = 0x0011,  /* V -> S   pick a session (null detaches)        */
    CALIPER_NET_MSG_VIEWS        = 0x0020,  /* P -> V/H, H -> V   every view, resent on change */
    CALIPER_NET_MSG_SUBSCRIBE    = 0x0021,  /* V -> P/H, H -> P   sub, view, rate, data view   */
    CALIPER_NET_MSG_UNSUBSCRIBE  = 0x0022,  /* V -> P/H, H -> P   sub                          */
    CALIPER_NET_MSG_VIEW_REQUEST = 0x0023,  /* V -> P/H, H -> P   a new data view for a sub    */
    CALIPER_NET_MSG_VIEW_REPLY   = 0x0024,  /* P -> V/H, H -> V   resolved data view or error  */
    CALIPER_NET_MSG_SNAPSHOT     = 0x0030,  /* P -> V/H, H -> V   sections "data" (+ "aux")    */
    CALIPER_NET_MSG_SERIES       = 0x0031,  /* P -> V/H, H -> V   section "values"             */
    CALIPER_NET_MSG_EVENT        = 0x0040,  /* V -> P/H, H -> P   select, pick, hover          */
    /* 3.1, capability "controls" */
    CALIPER_NET_MSG_CONTROLS     = 0x0050,  /* P -> V/H, H -> V   run state, parameters, actions */
    CALIPER_NET_MSG_CONTROL      = 0x0051,  /* V -> P/H, H -> P   pause, resume, step, set, run */
    CALIPER_NET_MSG_CONTROL_REPLY = 0x0052, /* P -> V/H, H -> V   the answer to one CONTROL     */
    /* 3.1, capability "selection" */
    CALIPER_NET_MSG_SELECTION    = 0x0060   /* P -> V/H, H -> V   the selection of an item space */
} CaliperNetMessageType;

/* --- live view kinds (values of CaliperLiveKind, caliper/live.h) ----------- */
typedef enum CaliperNetKind {
    CALIPER_NET_KIND_HEATMAP   = 1,   /* data (H,W) f32                                  */
    CALIPER_NET_KIND_IMAGE     = 2,   /* data (H,W,4) u8 RGBA                            */
    CALIPER_NET_KIND_POINTS    = 3,   /* data (N,D) f32, D = points_dims 2|3; aux (N,) f32 */
    CALIPER_NET_KIND_SERIES    = 4,   /* SERIES messages only: values (M,) f32           */
    CALIPER_NET_KIND_HISTOGRAM = 5    /* data (B,) f32 bin counts over [stats0, stats1]  */
} CaliperNetKind;

/* --- section dtypes (values of CaliperDType, caliper/tensor.h) -------------
 * 3.1 producers send F32 and U8 payloads only (I64 nowhere yet); the rest are
 * defined so that a receiver can name what it refuses. */
typedef enum CaliperNetDType {
    CALIPER_NET_DTYPE_F32  = 0,   /* 4 bytes */
    CALIPER_NET_DTYPE_F16  = 1,   /* 2 bytes */
    CALIPER_NET_DTYPE_BF16 = 2,   /* 2 bytes */
    CALIPER_NET_DTYPE_I64  = 3,   /* 8 bytes */
    CALIPER_NET_DTYPE_I32  = 4,   /* 4 bytes */
    CALIPER_NET_DTYPE_U8   = 5    /* 1 byte  */
} CaliperNetDType;

/* --- section encodings ---------------------------------------------------- */
typedef enum CaliperNetEncoding {
    CALIPER_NET_ENCODING_RAW       = 0,   /* a raw little-endian array (dtype, shape)  */
    CALIPER_NET_ENCODING_ARROW_IPC = 1    /* reserved for a future table kind; unused  */
} CaliperNetEncoding;

/* --- colormaps (values of CaliperColormap, caliper/embed.h) --------------- */
#define CALIPER_NET_COLORMAP_VIRIDIS 0
#define CALIPER_NET_COLORMAP_MAGMA   1
#define CALIPER_NET_COLORMAP_RDBU    2

/* --- snapshot flags (metadata "flags"; values of CALIPER_LIVE_SNAP_*) ------ */
#define CALIPER_NET_SNAP_AUTO_VMIN 0x1u   /* vmin = stats[0], else the fixed "vmin" */
#define CALIPER_NET_SNAP_AUTO_VMAX 0x2u   /* vmax = stats[1], else the fixed "vmax" */
#define CALIPER_NET_SNAP_RAW       0x4u   /* heatmap/image drawn 1:1 (tests)         */

/* --- WebSocket close codes -------------------------------------------------- */
#define CALIPER_NET_CLOSE_NORMAL            1000  /* after a BYE                              */
#define CALIPER_NET_CLOSE_GOING_AWAY        1001  /* the process is exiting                   */
#define CALIPER_NET_CLOSE_UNSUPPORTED_DATA  1003  /* a text message                           */
#define CALIPER_NET_CLOSE_TOO_BIG           1009  /* decoding step 5                          */
#define CALIPER_NET_CLOSE_PROTOCOL_ERROR    4000  /* a frame or message breaking this spec    */
#define CALIPER_NET_CLOSE_UNSUPPORTED_MAJOR 4001  /* decoding step 3                          */
#define CALIPER_NET_CLOSE_ROLE_REFUSED      4002  /* the server does not take this role       */
#define CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT 4003  /* no HELLO (or WELCOME) in time            */
#define CALIPER_NET_CLOSE_IDLE_TIMEOUT      4004  /* nothing received for IDLE_TIMEOUT_MS     */
#define CALIPER_NET_CLOSE_SERVER_FULL       4005  /* the server's connection limit            */

/* --- strings used as JSON values ------------------------------------------ */
#define CALIPER_NET_ROLE_VIEWER   "viewer"
#define CALIPER_NET_ROLE_PRODUCER "producer"
#define CALIPER_NET_ROLE_HUB      "hub"

#define CALIPER_NET_SESSION_LIVE  "live"
#define CALIPER_NET_SESSION_ENDED "ended"

#define CALIPER_NET_BYE_NORMAL    "normal"     /* done; closes with 1000                */
#define CALIPER_NET_BYE_SHUTDOWN  "shutdown"   /* the process is exiting; 1001          */
#define CALIPER_NET_BYE_REPLACED  "replaced"   /* the producer reconnected to the hub   */
#define CALIPER_NET_BYE_ERROR     "error"      /* after a fatal ERROR                   */

#define CALIPER_NET_ERR_BAD_REQUEST          "bad_request"           /* missing or invalid field */
#define CALIPER_NET_ERR_UNKNOWN_SESSION      "unknown_session"
#define CALIPER_NET_ERR_UNKNOWN_VIEW         "unknown_view"
#define CALIPER_NET_ERR_UNKNOWN_SUBSCRIPTION "unknown_subscription"
#define CALIPER_NET_ERR_BAD_DATA_VIEW        "bad_data_view"
#define CALIPER_NET_ERR_NOT_ATTACHED         "not_attached"
#define CALIPER_NET_ERR_UNSUPPORTED          "unsupported"
#define CALIPER_NET_ERR_LIMIT                "limit"
#define CALIPER_NET_ERR_INTERNAL             "internal"
/* 3.1: CONTROL_REPLY errors */
#define CALIPER_NET_ERR_FORBIDDEN            "forbidden"       /* this connection may not control */
#define CALIPER_NET_ERR_UNKNOWN_CONTROL      "unknown_control" /* no parameter or action of that name */
#define CALIPER_NET_ERR_BAD_VALUE            "bad_value"       /* a value of the wrong type or choice */
#define CALIPER_NET_ERR_TIMEOUT              "timeout"         /* a queued action did not start in time */
#define CALIPER_NET_ERR_ACTION_FAILED        "action_failed"   /* the action raised; message says why */

#define CALIPER_NET_EVENT_SELECT "select"
#define CALIPER_NET_EVENT_PICK   "pick"
#define CALIPER_NET_EVENT_HOVER  "hover"

#define CALIPER_NET_REDUCE_MEAN "mean"
#define CALIPER_NET_REDUCE_SUM  "sum"
#define CALIPER_NET_REDUCE_MAX  "max"
#define CALIPER_NET_REDUCE_MIN  "min"
#define CALIPER_NET_REDUCE_NORM "norm"         /* sqrt(sum(x * x)) */

#define CALIPER_NET_POOL_MEAN "mean"           /* budget "pool": heatmaps; images always mean */
#define CALIPER_NET_POOL_MAX  "max"
#define CALIPER_NET_POOL_MIN  "min"

#define CALIPER_NET_CAP_EVENTS    "events"     /* producer: Python code reads viewer events */
#define CALIPER_NET_CAP_ARROW_IPC "arrow_ipc"  /* reserved                                  */
#define CALIPER_NET_CAP_CONTROLS  "controls"   /* 3.1: run state, parameters and actions    */
#define CALIPER_NET_CAP_SELECTION "selection"  /* 3.1: item spaces and linked selection     */

/* 3.1: CONTROLS "run.state" */
#define CALIPER_NET_RUN_NONE      "none"       /* the script has called no checkpoint yet   */
#define CALIPER_NET_RUN_RUNNING   "running"
#define CALIPER_NET_RUN_PAUSING   "pausing"    /* a pause waits for the next checkpoint     */
#define CALIPER_NET_RUN_PAUSED    "paused"     /* the training thread waits in a checkpoint */
#define CALIPER_NET_RUN_STEPPING  "stepping"   /* runs steps_left steps, then pauses        */

/* 3.1: CONTROL "op" */
#define CALIPER_NET_OP_PAUSE      "pause"
#define CALIPER_NET_OP_RESUME     "resume"
#define CALIPER_NET_OP_STEP       "step"       /* "count" steps (default 1), then pause     */
#define CALIPER_NET_OP_SET        "set"        /* parameter "name" to "value"               */
#define CALIPER_NET_OP_RUN        "run"        /* action "name", no argument                */

/* 3.1: parameter "type" */
#define CALIPER_NET_PARAM_NUMBER  "number"     /* finite, clamped to [min, max]             */
#define CALIPER_NET_PARAM_INTEGER "integer"    /* whole, clamped to [min, max]              */
#define CALIPER_NET_PARAM_BOOLEAN "boolean"
#define CALIPER_NET_PARAM_CHOICE  "choice"     /* one of "choices"                          */

/* 3.1: CONTROL_REPLY "status" */
#define CALIPER_NET_STATUS_OK     "ok"         /* done (an action: finished without error)  */
#define CALIPER_NET_STATUS_QUEUED "queued"     /* an action waits; one more reply follows   */
#define CALIPER_NET_STATUS_ERROR  "error"      /* refused or failed: see "error"            */

/* --- limits and defaults ---------------------------------------------------- */
#define CALIPER_NET_MAX_FRAME_BYTES     (64u << 20)  /* any frame                          */
#define CALIPER_NET_MAX_META_BYTES      (1u << 20)   /* the metadata of any frame          */
#define CALIPER_NET_MAX_SECTIONS        8u
#define CALIPER_NET_MAX_JSON_DEPTH      16u          /* nested objects and arrays          */
#define CALIPER_NET_NAME_MAX            63u          /* bytes: view, source, dim names     */
#define CALIPER_NET_SESSION_ID_MAX      63u          /* bytes of [A-Za-z0-9._-]            */
#define CALIPER_NET_TOKEN_MIN           22u          /* chars of [A-Za-z0-9_-], >= 128 bits */
#define CALIPER_NET_TOKEN_MAX           128u
#define CALIPER_NET_MAX_VIEWS           256u         /* per session                        */
#define CALIPER_NET_MAX_SOURCES         64u          /* per view                           */
#define CALIPER_NET_MAX_DIMS            8u           /* per source                         */
#define CALIPER_NET_MAX_SUBSCRIPTIONS   256u         /* per connection                     */
#define CALIPER_NET_MAX_STATS           64u          /* "stats" entries of a snapshot      */
#define CALIPER_NET_DEFAULT_MAX_EXTENT  512u         /* heatmap/image rows and columns     */
#define CALIPER_NET_LIMIT_MAX_EXTENT    4096u
#define CALIPER_NET_DEFAULT_MAX_POINTS  100000u
#define CALIPER_NET_LIMIT_MAX_POINTS    1000000u
#define CALIPER_NET_MAX_BINS            65536u
#define CALIPER_NET_SERIES_RETAIN_MIN   4096u        /* values a producer retains, at least */
#define CALIPER_NET_SERIES_MAX_VALUES   262144u      /* values in one SERIES message       */
#define CALIPER_NET_MAX_EVENT_INDICES   32768u       /* entries of an event index list     */
#define CALIPER_NET_DEFAULT_RATE_HZ     60u          /* subscribe "max_rate" default       */
#define CALIPER_NET_LIMIT_RATE_HZ       240u         /* larger rates are clamped           */
#define CALIPER_NET_MAX_QUEUED_EVENTS   256u         /* a producer drops the oldest beyond */
#define CALIPER_NET_HANDSHAKE_TIMEOUT_MS 10000u
#define CALIPER_NET_PING_INTERVAL_MS    10000u       /* ping after this long without input */
#define CALIPER_NET_IDLE_TIMEOUT_MS     30000u       /* close after this long without input */
/* 3.1 */
#define CALIPER_NET_MAX_PARAMS          64u          /* parameters of a session            */
#define CALIPER_NET_MAX_ACTIONS         64u          /* actions of a session               */
#define CALIPER_NET_MAX_CHOICES         64u          /* choices of a choice parameter      */
#define CALIPER_NET_TEXT_MAX            1024u        /* bytes: help texts, reply messages  */
#define CALIPER_NET_CONTROL_RATE_HZ     30u          /* "set" requests per parameter, at most */
#define CALIPER_NET_ACTION_WAIT_MS      600000u      /* a queued action not started by then fails */
#define CALIPER_NET_MAX_ITEM_SPACES     16u          /* item spaces of a session           */
#define CALIPER_NET_MAX_DETAIL_ITEMS    32u          /* items in a selection's details     */
#define CALIPER_NET_MAX_DETAIL_FIELDS   16u          /* fields in a selection's details    */
#define CALIPER_NET_DETAIL_TEXT_MAX     256u         /* bytes of a string in the details   */
#define CALIPER_NET_DETAILS_TIMEOUT_MS  2000u        /* the script's details callback, at most */
#define CALIPER_NET_REPLY_WAIT_MS       60000u       /* a hub answers an unanswered CONTROL timeout */
#define CALIPER_NET_RELEASE_GRACE_MS    10000u       /* a paused run with no controller resumes */
#define CALIPER_NET_ORDER_WAIT_MS       2000u        /* a run state waits this long for data, at most */

/* Item size in bytes of a CaliperNetDType, 0 for an unknown one. */
static inline uint32_t caliper_net_dtype_size(int32_t dtype) {
    switch (dtype) {
        case CALIPER_NET_DTYPE_F32:  return 4u;
        case CALIPER_NET_DTYPE_F16:  return 2u;
        case CALIPER_NET_DTYPE_BF16: return 2u;
        case CALIPER_NET_DTYPE_I64:  return 8u;
        case CALIPER_NET_DTYPE_I32:  return 4u;
        case CALIPER_NET_DTYPE_U8:   return 1u;
        default:                     return 0u;
    }
}

/* n rounded up to CALIPER_NET_SECTION_ALIGN (n <= CALIPER_NET_MAX_FRAME_BYTES). */
static inline uint32_t caliper_net_align(uint32_t n) {
    return (n + (CALIPER_NET_SECTION_ALIGN - 1u)) & ~(CALIPER_NET_SECTION_ALIGN - 1u);
}

/* ===========================================================================
 * PROTOCOL 3.2 DRAFT: THE SCENES SUBSET
 *
 * Proposals that sprint 2 has not confirmed: the minor version 2, the
 * capability "scenes", the kind scene (6), SCENE_UPDATE (0x0070) and
 * SCENE_CLEAR (0x0072), and the limits below. Each is one line here, so a
 * renumbering changes one line. docs/references/wire-protocol-3-2.md states
 * what is built of them: the parts points, lines (segments and strips) and
 * mesh, SCENE_CLEAR, the VIEWS key "scene" and the default scene data view.
 * Senders keep writing CALIPER_NET_PROTOCOL_MINOR in their frames until 3.2 is
 * confirmed; the 3.2 golden frames (tests/fixtures/wire_v3/v3_2) carry
 * CALIPER_NET_MINOR_3_2. A peer decides by the capability, never the minor.
 * Every 3.1 message keeps its meaning and encoding.
 * ===========================================================================*/
#define CALIPER_NET_MINOR_3_2            2u
#define CALIPER_NET_CAP_SCENES           "scenes"     /* kind scene and its messages      */
#define CALIPER_NET_KIND_SCENE           6            /* SCENE_UPDATE and SCENE_CLEAR      */
#define CALIPER_NET_MSG_SCENE_UPDATE     0x0070u      /* P -> V/H, H -> V   one part of one entity */
#define CALIPER_NET_MSG_SCENE_CLEAR      0x0072u      /* P -> V/H, H -> V   clear an entity or reset */

/* SCENE_UPDATE "part"; arrows, boxes, image and camera are drafted, not built */
#define CALIPER_NET_PART_POINTS          "points"
#define CALIPER_NET_PART_LINES           "lines"
#define CALIPER_NET_PART_MESH            "mesh"
/* lines "mode" ("edges" is drafted, not built) and "color_per" */
#define CALIPER_NET_LINES_SEGMENTS       "segments"   /* vertices 2k and 2k + 1: segment k  */
#define CALIPER_NET_LINES_STRIPS         "strips"     /* section strip_lengths i32 (S,)     */
#define CALIPER_NET_COLOR_PER_VERTEX     "vertex"
#define CALIPER_NET_COLOR_PER_STRIP      "strip"
/* VIEWS "scene" {"space", "up", "units"} */
#define CALIPER_NET_SPACE_3D             "3d"
#define CALIPER_NET_SPACE_2D             "2d"

#define CALIPER_NET_PATH_MAX             255u         /* bytes of an entity path            */
#define CALIPER_NET_PATH_SEGMENTS_MAX    16u          /* segments of an entity path         */
#define CALIPER_NET_UNITS_MAX            15u          /* bytes of a scene's unit note       */
#define CALIPER_NET_MAX_ENTITIES         1024u        /* entities of a scene                */
#define CALIPER_NET_MAX_BATCHES          2048u        /* batches of a scene                 */
#define CALIPER_NET_MAX_BATCH_POINTS     4194304u     /* points, line or mesh vertices, triangles */
#define CALIPER_NET_MAX_BATCH_ITEMS      1048576u     /* line strips of a batch             */
/* the scene data view's budget: default and largest value of each key
 * (max_points shares CALIPER_NET_DEFAULT_MAX_POINTS and _LIMIT_MAX_POINTS) */
#define CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS    100000u
#define CALIPER_NET_SCENE_LIMIT_MAX_ITEMS      1048576u
#define CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES 1000000u
#define CALIPER_NET_SCENE_LIMIT_MAX_VERTICES   4194304u
#define CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT   1024u
#define CALIPER_NET_SCENE_LIMIT_MAX_EXTENT     8192u

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_LIVE_NET_H */
