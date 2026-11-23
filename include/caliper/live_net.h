/* ===========================================================================
 * caliper/live_net.h: wire protocol v3, live data streaming over a WebSocket.
 *
 * This header defines the constants and the frame header of protocol v3. The
 * reference page docs/reference/wire-protocol.md specifies every message and
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
 *   exact name and with no argument from the viewer. One exception: PROGRAM
 *   (3.2 draft, capability "programs") carries the WGSL source of a GPU
 *   program. A producer that announced "programs" runs it only on the GPU,
 *   after the sandbox checks, with read-only bindings of its tensors, and
 *   never on the CPU.
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
        case 6:                      return 2u;   /* CALIPER_NET_DTYPE_U16, the 3.2 block */
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
 * Provisional proposals: the minor version 2, the
 * capabilities, the kinds scene (6), text (7) and bars (8), the messages
 * 0x0070 to 0x0074, 0x0078 to 0x007B and 0x0080 to 0x0082, the dtype u16 (6), the error codes
 * and the limits below. Each is one line here, so
 * a renumbering changes one line (the dtype also has one line in
 * caliper_net_dtype_size above). docs/reference/wire-protocol-3-2.md states
 * what is built of them: the codecs build the whole draft (every part of
 * SCENE_UPDATE, split updates, SCENE_TRANSFORMS, SCENE_CLEAR, SCENE_CLASSES,
 * stamps, TEXT, bar charts and the recording messages); producers, the hub
 * and viewers build the scenes. Senders keep writing
 * CALIPER_NET_PROTOCOL_MINOR in their frames until 3.2 is confirmed; the 3.2
 * golden frames (tests/fixtures/wire_v3/v3_2) carry CALIPER_NET_MINOR_3_2. A
 * peer decides by the capability, never the minor. Every 3.1 message keeps
 * its meaning and encoding. CaliperDType (sdk/include/caliper/tensor.h) has
 * no CALIPER_DT_U16 yet: the SDK owns it, so the draft depends on a later SDK addition.
 * ===========================================================================*/
#define CALIPER_NET_MINOR_3_2            2u
#define CALIPER_NET_CAP_SCENES           "scenes"     /* kind scene and its messages      */
#define CALIPER_NET_KIND_SCENE           6            /* the scene messages                */
#define CALIPER_NET_MSG_SCENE_UPDATE     0x0070u      /* P -> V/H, H -> V   one part of one entity */
#define CALIPER_NET_MSG_SCENE_TRANSFORMS 0x0071u      /* P -> V/H, H -> V   entity transforms    */
#define CALIPER_NET_MSG_SCENE_CLEAR      0x0072u      /* P -> V/H, H -> V   clear an entity or reset */
#define CALIPER_NET_MSG_SCENE_CLASSES    0x0073u      /* P -> V/H, H -> V   the class table      */
#define CALIPER_NET_DTYPE_U16            6            /* 2 bytes: depth and segmentation images */

/* SCENE_UPDATE "part" */
#define CALIPER_NET_PART_POINTS          "points"
#define CALIPER_NET_PART_LINES           "lines"
#define CALIPER_NET_PART_ARROWS          "arrows"
#define CALIPER_NET_PART_BOXES           "boxes"
#define CALIPER_NET_PART_MESH            "mesh"
#define CALIPER_NET_PART_IMAGE           "image"
#define CALIPER_NET_PART_CAMERA          "camera"
/* lines "mode" and "color_per" */
#define CALIPER_NET_LINES_SEGMENTS       "segments"   /* vertices 2k and 2k + 1: segment k  */
#define CALIPER_NET_LINES_STRIPS         "strips"     /* section strip_lengths i32 (S,)     */
#define CALIPER_NET_LINES_EDGES          "edges"      /* section indices i32 (E, 2)         */
#define CALIPER_NET_COLOR_PER_VERTEX     "vertex"
#define CALIPER_NET_COLOR_PER_STRIP      "strip"
#define CALIPER_NET_COLOR_PER_EDGE       "edge"
/* image "form" and "format" */
#define CALIPER_NET_FORM_COLOR           "color"        /* pixels u8 (H, W, 4) or (H, W)  */
#define CALIPER_NET_FORM_DEPTH           "depth"        /* pixels f32 or u16 (H, W)       */
#define CALIPER_NET_FORM_SEGMENTATION    "segmentation" /* pixels u16 or u8 (H, W)        */
#define CALIPER_NET_FORM_ENCODED         "encoded"      /* pixels u8 (B,): one JPEG or PNG */
#define CALIPER_NET_FORMAT_JPEG          "jpeg"
#define CALIPER_NET_FORMAT_PNG           "png"
/* camera "convention" */
#define CALIPER_NET_CONVENTION_RDF       "rdf"        /* x right, y down, z forward         */
#define CALIPER_NET_CONVENTION_RUB       "rub"        /* x right, y up, looking along -z    */
/* SCENE_CLEAR "parts" also takes "transform" */
#define CALIPER_NET_PART_TRANSFORM       "transform"
/* VIEWS "scene" {"space", "up", "units", "view_camera"} */
#define CALIPER_NET_SPACE_3D             "3d"
#define CALIPER_NET_SPACE_2D             "2d"

#define CALIPER_NET_PATH_MAX             255u         /* bytes of an entity path            */
#define CALIPER_NET_PATH_SEGMENTS_MAX    16u          /* segments of an entity path         */
#define CALIPER_NET_UNITS_MAX            15u          /* bytes of a scene's unit note       */
#define CALIPER_NET_MAX_ENTITIES         1024u        /* entities of a scene                */
#define CALIPER_NET_MAX_BATCHES          2048u        /* batches of a scene                 */
#define CALIPER_NET_MAX_BATCH_POINTS     4194304u     /* points, line or mesh vertices, edges, triangles */
#define CALIPER_NET_MAX_BATCH_ITEMS      1048576u     /* boxes, arrows or line strips of a batch */
#define CALIPER_NET_MAX_LABELS           4096u        /* labels of a batch, of NAME_MAX bytes */
#define CALIPER_NET_MAX_CLASSES          4096u        /* entries of a class table           */
#define CALIPER_NET_MAX_CLASS_ID         65535u       /* the largest class id               */
#define CALIPER_NET_MAX_IMAGE_SIDE       8192u        /* pixels on a side of an image or texture */
#define CALIPER_NET_MAX_ENCODED_BYTES    16777216u    /* bytes of an encoded frame          */
#define CALIPER_NET_MAX_TRANSFORMS       1024u        /* entity transforms in one SCENE_TRANSFORMS */
#define CALIPER_NET_MAX_TIMELINES        8u           /* timelines of a session: values of a stamp */
#define CALIPER_NET_STAMP_ABSENT         (-0x7fffffffffffffffLL - 1)   /* a stamp value: absent on that timeline */
/* the scene data view's budget: default and largest value of each key
 * (max_points shares CALIPER_NET_DEFAULT_MAX_POINTS and _LIMIT_MAX_POINTS) */
#define CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS    100000u
#define CALIPER_NET_SCENE_LIMIT_MAX_ITEMS      1048576u
#define CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES 1000000u
#define CALIPER_NET_SCENE_LIMIT_MAX_VERTICES   4194304u
#define CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT   1024u
#define CALIPER_NET_SCENE_LIMIT_MAX_EXTENT     8192u

/* --- timelines, text logs, bar charts, split updates and recordings -------- */
#define CALIPER_NET_CAP_TIMELINES        "timelines"  /* VIEWS "timelines", stamps, "static"  */
#define CALIPER_NET_CAP_TEXT             "text"       /* kind text and TEXT                 */
#define CALIPER_NET_CAP_BARS             "bars"       /* kind bars, a SNAPSHOT of kind 8    */
#define CALIPER_NET_CAP_RECORDINGS       "recordings" /* the recording messages and keys    */
#define CALIPER_NET_KIND_TEXT            7            /* TEXT messages only                 */
#define CALIPER_NET_KIND_BARS            8            /* data (B,) f32, colors (B, 4) u8    */
#define CALIPER_NET_MSG_TEXT             0x0074u      /* P -> V/H, H -> V   new lines of a text log */
#define CALIPER_NET_MSG_RECORDINGS       0x0078u      /* S -> V   the recordings a server can open */
#define CALIPER_NET_MSG_RECORDING_OPEN   0x0079u      /* V -> S   open one; answered by VIEWS or ERROR */
#define CALIPER_NET_MSG_CURSOR           0x007Au      /* V -> S   move the viewer's cursor      */
#define CALIPER_NET_MSG_TIME_STATE       0x007Bu      /* S -> V   the state of the cursor       */
#define CALIPER_NET_ERR_UNKNOWN_RECORDING "unknown_recording"
#define CALIPER_NET_ERR_BAD_RECORDING    "bad_recording"
#define CALIPER_NET_ERR_NOT_SEEKABLE     "not_seekable"
#define CALIPER_NET_ERR_BAD_CURSOR       "bad_cursor"
#define CALIPER_NET_TIMELINE_SEQUENCE    "sequence"
#define CALIPER_NET_TIMELINE_TIME        "time"
#define CALIPER_NET_LEVEL_DEBUG          0            /* TEXT "levels"                      */
#define CALIPER_NET_LEVEL_INFO           1
#define CALIPER_NET_LEVEL_WARNING        2
#define CALIPER_NET_LEVEL_ERROR          3
#define CALIPER_NET_CURSOR_SEEK          "seek"       /* CURSOR "op"                        */
#define CALIPER_NET_CURSOR_PLAY          "play"
#define CALIPER_NET_CURSOR_PAUSE         "pause"
#define CALIPER_NET_CURSOR_STEP          "step"
#define CALIPER_NET_CURSOR_FOLLOW        "follow"
#define CALIPER_NET_PACE_RECORDED        "recorded"
#define CALIPER_NET_PACE_EVEN            "even"
#define CALIPER_NET_TIME_FOLLOWING       "following"  /* TIME_STATE "state"                 */
#define CALIPER_NET_TIME_PAUSED          "paused"
#define CALIPER_NET_TIME_PLAYING         "playing"
#define CALIPER_NET_STAMP_TEXT_MAX       20u          /* characters of stamp text           */
#define CALIPER_NET_MAX_PIECES           16u          /* pieces of a split update           */
#define CALIPER_NET_MAX_UPDATE_BYTES     536870912u   /* bytes of a whole split update      */
#define CALIPER_NET_MAX_INCOMPLETE_UPDATES 2u         /* split updates a receiver holds per subscription */
#define CALIPER_NET_TEXT_LINE_MAX        4096u        /* bytes of a text line               */
#define CALIPER_NET_TEXT_MAX_LINES       65536u       /* lines in one TEXT                  */
#define CALIPER_NET_TEXT_RETAIN_MIN      4096u        /* lines a producer retains per text log */
#define CALIPER_NET_MAX_BARS             4096u        /* bars of a snapshot                 */
#define CALIPER_NET_MAX_RECORDINGS       1024u        /* recordings in RECORDINGS           */
#define CALIPER_NET_MAX_RECORDED         64u          /* VIEWS "recorded": data views of a view */
#define CALIPER_NET_MAX_CURSOR_STEP      1000u        /* CURSOR "count", either way         */
#define CALIPER_NET_CURSOR_RATE_HZ       30u          /* CURSOR messages a viewer sends a second */
#define CALIPER_NET_TIME_STATE_RATE_HZ   30u          /* TIME_STATE a server sends a second while playing */
#define CALIPER_NET_MIN_SPEED            0.0625       /* playback speed, 1/16                */
#define CALIPER_NET_MAX_SPEED            64u
#define CALIPER_NET_RECORDING_LINGER_MS  600000u      /* an opened recording without viewers stays listed */

/* --- GPU programs and interventions (design/gpu-programs.md) ---------------
 * "programs": a viewer sends PROGRAM (set or remove one program) only to a
 * producer that announced it, and the producer answers each PROGRAM and
 * reports state changes with PROGRAM_STATE. "interventions": the producer's
 * CONTROLS holds the key "interventions" ({points, active}), and a viewer
 * sends CONTROL with op "intervene" and the key "intervene" ({set: {point,
 * op, from_step, until_step}}, {clear: point} or {clear_all: true}). A
 * producer announces both only under caliper.watch(model, optimizer), beside
 * "controls". A hub forwards PROGRAM, as CONTROL, only when it allows control.
 * A PROGRAM with op "params" carries the keys request_id, op, id and "params"
 * (a JSON object of 1 to 16 names, each a WGSL field name of 1 to 63 bytes,
 * [A-Za-z_][A-Za-z0-9_]*, not starting with "__", each value a finite
 * number), no "declaration" and no section. The producer applies the values
 * at the next pass of that program and answers with PROGRAM_STATE carrying
 * the request_id. A "set" declaration may also hold "params" of the same form,
 * an output {"kind": "tensor", "shape": [..]} (1 to 4 integers, each at least
 * 1, product at most 1,000,000; no other output keys for that kind) and
 * "builtin" (a string of 1 to 32 bytes such as "surface@1", for a program the
 * app makes for an operator). A program id is either 1 to 55 bytes of
 * [a-z0-9_-] starting with a letter, or the operator form
 * "operator.<id>.<n>": the prefix "operator.", a spec id (a lower-case letter
 * then lower-case letters, digits, _ or -), a dot and 1 to 5 decimal digits,
 * the whole at most 55 bytes.
 * docs/reference/wire-protocol-3-2.md states the keys and checks. */
#define CALIPER_NET_CAP_PROGRAMS         "programs"      /* PROGRAM and PROGRAM_STATE        */
#define CALIPER_NET_CAP_INTERVENTIONS    "interventions" /* CONTROLS "interventions", CONTROL op "intervene" */
#define CALIPER_NET_MSG_PROGRAM          0x0080u      /* V -> P/H, H -> P   set or remove one program; section "wgsl" */
#define CALIPER_NET_MSG_PROGRAM_STATE    0x0081u      /* P -> V/H, H -> V   the status items of the programs */
#define CALIPER_NET_PROGRAM_SET          "set"        /* PROGRAM "op": install or replace   */
#define CALIPER_NET_PROGRAM_REMOVE       "remove"     /* PROGRAM "op"                       */
#define CALIPER_NET_PROGRAM_PARAMS       "params"     /* PROGRAM "op": new parameter values, no source */
#define CALIPER_NET_SECTION_WGSL         "wgsl"       /* u8 (B,): the WGSL source as UTF-8  */
#define CALIPER_NET_OP_INTERVENE         "intervene"  /* CONTROL "op", key "intervene"      */
#define CALIPER_NET_INTERVENE_ABLATE     "ablate"     /* the only op; scale, steer, patch reserved */
#define CALIPER_NET_MAX_PROGRAMS         16u          /* programs of a producer; items of PROGRAM_STATE */
#define CALIPER_NET_PROGRAM_ID_MAX       55u          /* bytes of an id, [a-z0-9_-] from a letter, or operator.<id>.<n>: "program.<id>" fits NAME_MAX */
#define CALIPER_NET_PROGRAM_SOURCE_MAX   65536u       /* bytes of the section "wgsl"        */
#define CALIPER_NET_PROGRAM_STATE_RATE_HZ 2u          /* unrequested PROGRAM_STATE a second, at most */
#define CALIPER_NET_POINT_MAX            255u         /* bytes of an intervention point     */
#define CALIPER_NET_MAX_POINTS_LISTED    4096u        /* CONTROLS "interventions" "points"  */
#define CALIPER_NET_MAX_INTERVENTIONS    64u          /* CONTROLS "interventions" "active"  */
/* the PROGRAM set declaration's bounds: each count that sizes a buffer */
#define CALIPER_NET_PROGRAM_MAX_INPUTS   8u           /* "inputs"                           */
#define CALIPER_NET_PROGRAM_MAX_COUNT    1000000u     /* geometry "max_count"               */
#define CALIPER_NET_PROGRAM_SIDE_MAX     2048u        /* pixels "width" and "height"        */
#define CALIPER_NET_PROGRAM_TIME_MS_MAX  50u          /* "limits" "time_ms"                 */
#define CALIPER_NET_PROGRAM_MEMORY_MB_MAX 512u        /* "limits" "memory_mb"               */
#define CALIPER_NET_PROGRAM_EVERY_MAX    10000u       /* "limits" "every", and the cadence N */
#define CALIPER_NET_PROGRAM_WORKGROUPS_MAX 65535u     /* each of the three "workgroups"     */
/* 3.2 draft, programmable graphics (design/programmable-graphics.md, Resource Message).
 * "resources": a producer under caliper.watch(model, optimizer) announces it
 * beside "programs", a viewer that draws pieces announces it in HELLO, and a
 * hub announces it to both. A PROGRAM "set" of an instance of a piece holds
 * "piece" ({name, version, hash}) and "manifest" (a JSON object), and may
 * hold "bind", "params", "panel" and "segments", in place of "declaration",
 * "scene" and "sha256"; it carries the sections "common", "derive" and
 * "draw" (each UTF-8, with the imports resolved) in place of "wgsl", with
 * "derive" or "draw" or both. A viewer sends one only to a producer that
 * announced "resources". After a derive pass, the producer sends each
 * published resource that the pass wrote as one RESOURCE with the section
 * "resource" to each viewer or hub that announced "resources". It sends the
 * draw half of each installed program to such a peer as PROGRAM "set" with
 * request_id 0 and all three sections, when the peer attaches and after each
 * install (the pin's hash covers derive.wgsl, thus a viewer checks it only
 * with that section), and PROGRAM "remove" with request_id 0 after a
 * removal. RESOURCE holds "program", "resource", "step" (the run's step at
 * the pass), "count" (valid elements), "stride" (bytes of one element, a
 * multiple of 4) and, for a texture, "width" and "height" (1 to
 * CALIPER_NET_RESOURCE_SIDE_MAX each) with an optional "format" (rgba8unorm when
 * absent), stride 4 and count the 32-bit words of its texels in row order:
 * width * height for rgba8unorm and r32uint, (width * height + 3) / 4 for
 * r8unorm. A buffer of narrow elements holds "element" ("u8" or "u16") and
 * "elements", its exact element count, with stride 4 and count the words,
 * (elements * bytes + 3) / 4. Its section holds exactly count * stride bytes.
 * A version larger than one frame travels as 2 to CALIPER_NET_MAX_PIECES
 * RESOURCE parts, in order, each with the whole version's keys and the key
 * "split" ({version, part, parts, first}): the sender's counter of versions
 * in parts, this part's number from 0, the number of parts, and the first
 * element of the version that the part's section holds. A part holds whole
 * elements; the parts hold the version's elements in order, without gap or
 * overlap. A receiver puts the parts together, holds at most
 * CALIPER_NET_MAX_INCOMPLETE_UPDATES incomplete versions per connection, and
 * drops one whose next part has not come after CALIPER_NET_RESOURCE_PART_WAIT
 * further RESOURCE frames, or when part 0 of a newer version of the same
 * resource comes.
 * PROGRAM_STATE gives an instance the status kind "piece".
 * docs/reference/wire-protocol-3-2.md states the keys and checks. */
#define CALIPER_NET_CAP_RESOURCES        "resources"  /* RESOURCE, and PROGRAM of a piece          */
/* 3.2 draft, the piece form of SCENE_UPDATE: an entity part's arrays travel as the RESOURCE
 * versions of a scene program "scene.<n>" (1 to 9 digits), an instance of the part's library
 * piece, sent just before the update that names them with the keys program, piece, resources
 * and arrays. A producer sends the form only to a peer that announced the capability. */
#define CALIPER_NET_CAP_SCENE_PIECES     "scene_pieces" /* the piece form of SCENE_UPDATE   */
#define CALIPER_NET_SCENE_PROGRAM_PREFIX "scene."     /* the id of a scene program: scene.<n> */
#define CALIPER_NET_MSG_RESOURCE         0x0082u      /* P -> V/H, H -> V   one version of one published resource */
#define CALIPER_NET_SECTION_DERIVE       "derive"     /* u8 (B,): the derive half's WGSL    */
#define CALIPER_NET_SECTION_DRAW         "draw"       /* u8 (B,): the draw half's WGSL      */
#define CALIPER_NET_SECTION_COMMON       "common"     /* u8 (B,): the lib modules and common.wgsl */
#define CALIPER_NET_SECTION_RESOURCE     "resource"   /* u8 (B,): the valid bytes of one resource */
#define CALIPER_NET_STATUS_PIECE         "piece"      /* PROGRAM_STATE "kind" of an instance */
#define CALIPER_NET_MAX_RESOURCES        8u           /* resources of one program           */
/* A resource of 64 MiB less 64 KiB fits one frame of CALIPER_NET_MAX_FRAME_BYTES with its metadata. */
#define CALIPER_NET_RESOURCE_MAX_BYTES   67043328u    /* bytes of the section "resource"    */
/* A buffer version stays within the default WebGPU storage binding limit. */
#define CALIPER_NET_RESOURCE_VERSION_MAX_BYTES 134217728u /* bytes of a buffer version in parts */
#define CALIPER_NET_RESOURCE_TEXTURE_VERSION_MAX_BYTES 268435456u /* bytes of a texture version in parts */
#define CALIPER_NET_RESOURCE_MAX_PARTS   32u          /* 8 MiB chunks of a full RGBA8 texture */
#define CALIPER_NET_RESOURCE_PART_WAIT   256u         /* RESOURCE frames a receiver waits for the next part */
#define CALIPER_NET_RESOURCE_STRIDE_MAX  256u         /* "stride": the bytes of one element */
#define CALIPER_NET_RESOURCE_SIDE_MAX    8192u        /* a texture's "width" and "height": WebGPU's default largest 2-D texture */
#define CALIPER_NET_TEXTURE_RGBA8        "rgba8unorm" /* "format": four 8-bit channels, one word a texel */
#define CALIPER_NET_TEXTURE_R8           "r8unorm"    /* "format": one 8-bit channel, four texels a word */
#define CALIPER_NET_TEXTURE_R32UINT      "r32uint"    /* "format": one 32-bit unsigned integer a texel */
#define CALIPER_NET_ELEMENT_U8           "u8"         /* "element": four elements a word     */
#define CALIPER_NET_ELEMENT_U16          "u16"        /* "element": two elements a word      */
#define CALIPER_NET_PIECE_NAME_MAX       32u          /* bytes of "piece" "name", [a-z][a-z0-9_]* */
#define CALIPER_NET_PIECE_VERSION_MAX    1000000u     /* "piece" "version", from 1          */
#define CALIPER_NET_PIECE_COMMON_MAX     589824u      /* bytes of "common": 8 lib modules and common.wgsl of 64 KiB each */
#define CALIPER_NET_PIECE_SEGMENTS_MAX   16u          /* items of "segments"                */
#define CALIPER_NET_TENSOR_NAME_MAX      255u         /* bytes of an input's "source"       */

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_LIVE_NET_H */
