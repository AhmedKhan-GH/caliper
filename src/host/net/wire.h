#pragma once
// wire.h: the C++ codec of wire protocol v3 (caliper/live_net.h,
// docs/reference/wire-protocol.md), the native twin of python/caliper/
// _wire.py. Host-internal C++17; no sockets (ws.h carries the bytes).
//
// FRAMES. decode_frame() applies the protocol's decoding steps 1 to 9 in their
// order and reports the first failure with its WebSocket close code and the
// reason word of the reference codec ("truncated", "bad_magic",
// "unsupported_major", "too_big", "malformed"). encode_frame() writes the
// canonical form: the header, the section lengths, the canonical JSON
// metadata (json.h) with "sections" written from the sections, then each
// section at the next multiple of 8, the gaps zero-filled.
//
// MESSAGES. decode() turns one frame into one of the message structs below
// (Unknown for a type this version does not know: a receiver ignores it). A
// well-formed frame whose message is invalid is a message error: the receiver
// answers with an ERROR message carrying `error_code` (bad_request,
// unsupported) and keeps the connection. The field rules are those of
// _wire.py's from_frame(); encode() writes the keys its to_meta() writes, so a
// message decoded from a canonical frame encodes back to the same bytes.
//
// Objects the protocol passes through (software, identities, data views, view
// entries, event details, axes, tile, the view_reply error) are kept as Json,
// so a forwarder loses no key it does not know. Sessions and Views also carry
// a parsed copy (`list`) for readers; encode() writes the Json.
//
// 3.1. The messages of controls (CONTROLS, CONTROL, CONTROL_REPLY) and linked
// selection (SELECTION) decode and encode here with the field rules of
// _wire.py, as do the keys 3.1 adds to 3.0 messages (WELCOME's `control`,
// `items` on a source dim and on a series view, `space` and `items` in an
// event's detail). Which peer may receive them is the caller's business: the
// protocol sends each only to a peer that announced its capability. encode()
// writes the codec's minor (CALIPER_NET_PROTOCOL_MINOR, 1) unless told
// otherwise; the 3.0 golden frames encode back to their bytes at minor 0.
//
// 3.2 draft. The scenes subset (SCENE_UPDATE, SCENE_CLEAR, the scene view
// helpers) sits in one block before the Message variant; the rules are those of
// the block at the end of _wire.py and of docs/reference/wire-protocol-3-2.md.
//
// LIFETIME. A decoded Section points into the buffer given to decode(): keep
// that buffer alive while the message is used.
#include "caliper/live_net.h"
#include "net/json.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace caliper_net {

struct HttpRequest;   // ws.h

namespace wire {

// ---- the upgrade request (the reference page, "The upgrade request") --------------
// The server's checks before it upgrades, in the page's order: the path is
// exactly CALIPER_NET_PATH (else 404), the Host header names 127.0.0.1 or
// localhost with the server's port (else 403), the Origin header is absent or
// accepted (else 403), and the token parameter equals the session token,
// compared in constant time (else 403). Returns 0 to accept, else the status.
// `path` is CALIPER_NET_PATH except for a server that mounts the endpoint
// elsewhere: Caliper API serves the same stream at /v0/stream (api/api_server.h).
int upgrade_status(const HttpRequest& r, const std::string& session_token, uint16_t port,
                   const std::string& path = CALIPER_NET_PATH);
// The same checks, in the same order, for a server that accepts more than one
// token (Caliper API: api/api_server.h): the token parameter passes when
// `token_ok` returns true for it, and `token_ok` compares in constant time.
int upgrade_status(const HttpRequest& r, const std::function<bool(const std::string&)>& token_ok, uint16_t port,
                   const std::string& path = CALIPER_NET_PATH);
// The single rules, for a server that serves plain HTTP on the same port (the
// Host check applies to those requests too).
bool host_allowed(const std::string& host, uint16_t port);
bool origin_allowed(const std::string& origin);
bool token_equal(const std::string& a, const std::string& b);   // constant time in b's length

// ---- frames -------------------------------------------------------------------------

struct Section {
    std::string name;
    int64_t dtype = CALIPER_NET_DTYPE_F32;        // CaliperNetDType (unknown ones are kept)
    std::vector<int64_t> shape;
    int64_t encoding = CALIPER_NET_ENCODING_RAW;
    const uint8_t* data = nullptr;   // decoded: inside the frame; encode: the caller's bytes
    size_t size = 0;                 // bytes
    size_t offset = 0;               // decoded: offset in the frame

    // prod(shape) (1 for a scalar); -1 on overflow.
    int64_t count() const;
    // A RAW section of this dtype and rank.
    bool is(int64_t dt, size_t rank) const {
        return encoding == CALIPER_NET_ENCODING_RAW && dtype == dt && shape.size() == rank;
    }
    // A section over caller-owned bytes (for encode()).
    static Section make(const std::string& name, int64_t dtype, std::vector<int64_t> shape,
                        const void* data, size_t size);
};

struct Frame {
    uint16_t type = 0;
    uint16_t flags = 0;
    uint16_t major = CALIPER_NET_PROTOCOL_MAJOR;
    uint16_t minor = CALIPER_NET_PROTOCOL_MINOR;
    Json meta;                        // the metadata object, "sections" included
    std::vector<Section> sections;
    const Section* section(const std::string& name) const;
};

// Why a frame is not well formed: the close code and the reference codec's
// reason word.
struct FrameError {
    int close_code = 0;
    std::string reason;
    std::string message;
};

bool decode_frame(const uint8_t* buf, size_t n, Frame* out, FrameError* err);

// `meta` must not hold "sections" (it is written from `sections`). False with
// *err when the metadata or the frame would pass the protocol's limits.
bool encode_frame(uint16_t type, const Json& meta, const std::vector<Section>& sections,
                  std::vector<uint8_t>* out, FrameError* err = nullptr,
                  uint16_t minor = CALIPER_NET_PROTOCOL_MINOR, uint16_t flags = 0);

// The head of a frame whose sections already exist as one block of bytes: the
// header, the section lengths, `meta` (which holds the "sections" descriptors
// unchanged) and the zero padding up to the next multiple of 8. A decoded
// frame's bytes from its first section's offset to its end are such a block,
// because every section keeps its distance from the first one, so a forwarder
// can rewrite a field such as "sub" and send the new head followed by the
// original sections without copying them. tail_len is the block's length. The
// header carries `minor` and `flags`.
bool encode_head(uint16_t type, const Json& meta, const std::vector<uint32_t>& section_lens, size_t tail_len,
                 std::vector<uint8_t>* head, FrameError* err = nullptr, uint16_t minor = CALIPER_NET_PROTOCOL_MINOR,
                 uint16_t flags = 0);

// ---- messages -------------------------------------------------------------------------

struct Identity {   // a session identity: {"id", "script", "pid", "start_unix_ms"}
    std::string id;
    std::string script;
    uint64_t pid = 0;
    uint64_t start_unix_ms = 0;
    Json to_json() const;
    static bool parse(const Json& j, Identity* out);   // the _identity() rule
};

struct SessionInfo {   // an entry of SESSIONS: an identity plus "state" and "views"
    Identity identity;
    std::string state;   // CALIPER_NET_SESSION_LIVE or _ENDED
    uint64_t views = 0;
    Json to_json() const;
};

struct DimInfo {
    std::string name;
    uint64_t size = 0;
    std::string items;            // 3.1: the item space the dim indexes, "" none
};
struct SourceInfo {
    std::string name;
    std::string dtype;            // as given ("f32"), "" when absent or not a string
    std::vector<DimInfo> dims;
};
struct ViewInfo {   // an entry of VIEWS
    uint64_t id = 0;
    std::string name;
    int64_t kind = 0;
    std::vector<SourceInfo> sources;
    Json default_view;            // the view's default data view, null when absent
    std::string items;            // 3.1, series: the item space of its index, "" none
};

struct Hello {
    static constexpr uint16_t kType = CALIPER_NET_MSG_HELLO;
    std::string role;                      // viewer or producer
    std::vector<std::string> capabilities;
    Json software;                         // null when absent
    Json session;                          // an identity (producer), null when absent
};
struct Welcome {
    static constexpr uint16_t kType = CALIPER_NET_MSG_WELCOME;
    std::string role;                      // producer or hub
    std::vector<std::string> capabilities;
    Json software;
    Json session;
    std::optional<bool> control;           // 3.1: whether CONTROL is accepted (forwarded)
};
struct Bye {
    static constexpr uint16_t kType = CALIPER_NET_MSG_BYE;
    std::string reason;
    std::string message;
};
struct Error {
    static constexpr uint16_t kType = CALIPER_NET_MSG_ERROR;
    std::string code;
    std::string message;
    bool fatal = false;
    std::optional<uint32_t> in_reply_to, request_id, sub;
    std::optional<std::string> view, session;
};
struct Ping {
    static constexpr uint16_t kType = CALIPER_NET_MSG_PING;
    uint32_t nonce = 0;
};
struct Pong {
    static constexpr uint16_t kType = CALIPER_NET_MSG_PONG;
    uint32_t nonce = 0;
};
struct Sessions {
    static constexpr uint16_t kType = CALIPER_NET_MSG_SESSIONS;
    Json sessions = Json::array();         // what encode() writes
    std::vector<SessionInfo> list;         // decoded copy
};
struct Attach {
    static constexpr uint16_t kType = CALIPER_NET_MSG_ATTACH;
    std::optional<std::string> session;    // nullopt: detach (null on the wire)
};
struct Views {
    static constexpr uint16_t kType = CALIPER_NET_MSG_VIEWS;
    Json views = Json::array();            // what encode() writes
    std::optional<std::string> session;    // written as null when absent
    std::vector<ViewInfo> list;            // decoded copy
    Json timelines;                        // 3.2 draft: the timelines, null when absent
};
struct Subscribe {
    static constexpr uint16_t kType = CALIPER_NET_MSG_SUBSCRIBE;
    uint32_t sub = 0;
    std::string view;
    uint32_t request_id = 0;
    std::optional<double> max_rate;        // default CALIPER_NET_DEFAULT_RATE_HZ
    Json data_view;                        // null: the view's default
};
struct Unsubscribe {
    static constexpr uint16_t kType = CALIPER_NET_MSG_UNSUBSCRIBE;
    uint32_t sub = 0;
};
struct ViewRequest {
    static constexpr uint16_t kType = CALIPER_NET_MSG_VIEW_REQUEST;
    uint32_t request_id = 0;
    uint32_t sub = 0;
    std::string view;
    Json data_view = Json::object();
};
struct ViewReply {
    static constexpr uint16_t kType = CALIPER_NET_MSG_VIEW_REPLY;
    uint32_t request_id = 0;
    uint32_t sub = 0;
    std::string view;
    Json data_view;                        // exactly one of data_view and error is set
    Json error;                            // {"code", "message"}
};
struct Snapshot {
    static constexpr uint16_t kType = CALIPER_NET_MSG_SNAPSHOT;
    uint32_t sub = 0;
    std::string view;
    uint64_t seq = 0;
    uint64_t stamp_ns = 0;
    int64_t kind = 0;                      // CaliperNetKind (unknown kinds are kept)
    Json data_view = Json::object();
    Section data;                          // "data"
    std::optional<Section> aux;            // "aux" (points)
    int64_t colormap = CALIPER_NET_COLORMAP_VIRIDIS;
    uint64_t flags = 0;                    // CALIPER_NET_SNAP_*
    double vmin = 0.0, vmax = 1.0;
    std::vector<double> stats;             // NaN for a null entry
    Json axes = Json::array();
    Json tile;                             // null when absent
    std::optional<int64_t> points_dims;    // points: set from the payload on decode
    std::optional<uint64_t> label_count;
    // 3.2 draft (the block below): a bar chart's colours (B, 4) u8, labels and direction, a static
    // snapshot, its stamp i64 (T,), and the cursor and emptiness of a recorded one (no section).
    std::optional<Section> colors;
    Json labels;                           // null when absent
    bool horizontal = false, is_static = false, empty = false;
    std::optional<Section> stamp;
    std::optional<uint32_t> cursor;
    // 3.2 draft, capability "programs": the shared cadence of the views' snapshots (an object:
    // every, passes_per_s, step_ms, pass_ms, share, budget, limit); null when absent.
    Json cadence;
};
struct Series {
    static constexpr uint16_t kType = CALIPER_NET_MSG_SERIES;
    uint32_t sub = 0;
    std::string view;
    uint64_t start = 0;                    // absolute index of the first value
    uint64_t total = 0;                    // values appended so far
    uint64_t stamp_ns = 0;
    Section values;                        // (M,) f32
    // 3.2 draft: a stamp (T,) for all or stamps (M, T); a decimated recorded series' absolute
    // indices (M,) i64 with its bucket size; the cursor generation.
    std::optional<Section> stamp, stamps, indices;
    std::optional<uint64_t> decimated;
    std::optional<uint32_t> cursor;
};
struct Event {
    static constexpr uint16_t kType = CALIPER_NET_MSG_EVENT;
    std::string view;
    std::string type;                      // select, pick, hover
    Json detail = Json::object();
    std::optional<uint32_t> sub;
    std::optional<uint64_t> seq;
    Json data_view;                        // null when absent
};
// ---- 3.1: controls ("controls") and linked selection ("selection") ---------
struct Controls {   // the run state, parameters and actions of a script
    static constexpr uint16_t kType = CALIPER_NET_MSG_CONTROLS;
    Json run = Json::object();             // {"state", "step", "steps_left"}
    Json params = Json::array();           // parameter entries, validated
    Json actions = Json::array();          // action entries, validated
    std::optional<Section> stamp;          // 3.2 draft: when the run state or a value changed
    Json interventions;                    // 3.2 draft: {"points", "active"} as given; null when absent
};
struct Control {    // pause, resume, step, set, run, intervene (3.2 draft)
    static constexpr uint16_t kType = CALIPER_NET_MSG_CONTROL;
    uint32_t request_id = 0;
    std::string op;
    std::optional<std::string> name;       // set, run
    Json value;                            // set: a number, a boolean or a choice; null otherwise
    std::optional<uint32_t> count;         // step: at least 1; absent means 1
    Json intervene;                        // intervene: exactly one of set, clear and clear_all, as given; null otherwise
};
struct ControlReply {
    static constexpr uint16_t kType = CALIPER_NET_MSG_CONTROL_REPLY;
    uint32_t request_id = 0;
    std::string op;
    std::string status;                    // ok, queued, error
    std::optional<std::string> name;
    Json value;                            // set with ok: the accepted value; null when absent
    std::optional<std::string> message;    // run with ok: the action's result text
    Json error;                            // status error: {"code", "message"}; null otherwise
};
struct Selection {  // the selection of one item space
    static constexpr uint16_t kType = CALIPER_NET_MSG_SELECTION;
    std::string space;
    Json items = Json::array();            // ascending item ids, without repeats
    std::optional<std::string> view;
    bool truncated = false;
    Json details;                          // {"fields", "items", "values"}; null when absent
    std::optional<std::string> details_error;
};

struct Unknown {   // a type this version does not know; receivers ignore it
    uint16_t type = 0;
    Json meta;
    std::vector<Section> sections;
};

// ---- protocol 3.2 draft: the scenes subset (proposals: minor 2, capability "scenes", kind 6, 0x0070 to 0x0073, dtype u16 6; the 3.2 block of caliper/live_net.h) ----
// The rules of the block at the end of _wire.py, checked in the same order. A
// Whether `id` is the id of a scene program, "scene.<n>" with 1 to 9 digits.
bool scene_program_id(const std::string& id);

// SceneUpdate's `arrays` hold the sections of its part in the part's order
// (PART_SECTIONS); decoded, only the known ones, pointing into the frame like
// every decoded Section. Every scene message may carry a stamp section, i64
// (T,) with T from 1 to CALIPER_NET_MAX_TIMELINES, and the key `cursor`.
struct SceneUpdate {                 // 3.2 draft: one part of one entity of a scene
    static constexpr uint16_t kType = CALIPER_NET_MSG_SCENE_UPDATE;
    uint32_t sub = 0;
    std::string view;
    uint64_t seq = 0, stamp_ns = 0;
    std::string path;                // an entity path ("" is the root)
    std::string part;                // CALIPER_NET_PART_*: points, lines, arrows, boxes, mesh, image, camera
    std::vector<Section> arrays;     // the part's sections, in the part's order (decoded: only the known ones)
    Json params = Json::object();    // the part's keys as given (a null stat decodes to NaN); a camera's
                                     // fx, fy, cx, cy, width, height, convention, plane_distance
    bool is_static = false;          // "static"
    bool state = false;
    Json axes;                       // null when absent
    std::optional<std::string> items;
    std::optional<Section> stamp;    // i64 (T,)
    std::optional<uint32_t> cursor;  // the TIME_STATE generation a recorded message answers
    Json split;                      // a piece of a split update: {update, piece, pieces, first, rows}; null
                                     // when whole. A piece's arrays are its rows, unchecked against the part
                                     // until SplitUpdates puts the update together; its params are empty
                                     // but in piece 0
    // The piece form (capability "scene_pieces"): the scene program "scene.<n>" whose RESOURCE versions,
    // sent just before, hold the batch's arrays as the resources of the library piece `piece` (the pin:
    // name, version, hash); `resources` maps each resource id to the count of that version, and `names`
    // lists the part's arrays the batch held (the key "arrays"). Such an update carries no array section.
    std::optional<std::string> program;
    Json piece;                      // null unless the piece form
    Json resources;                  // null unless the piece form
    std::vector<std::string> names;
    // Receiver side, never on the wire: the RESOURCE versions that the receiver paired with a piece-form
    // update (caliper_live::ScenePieceInbox), one section each, named by its resource id, u8 of the shape
    // (count, stride), over bytes that the queued item keeps alive.
    std::vector<Section> paired;
    const Section* array(const std::string& name) const;
};
struct SceneTransforms {             // 3.2 draft: entity transforms, (K, 16) f32 column-major matrices
    static constexpr uint16_t kType = CALIPER_NET_MSG_SCENE_TRANSFORMS;
    uint32_t sub = 0;
    std::string view;
    uint64_t seq = 0, stamp_ns = 0;
    std::vector<std::string> paths;  // K distinct entity paths, never the root
    Section matrices;                // (K, 16) f32; row k belongs to paths[k]
    bool is_static = false, state = false;
    std::optional<Section> stamp;    // (T,): one stamp for all
    std::optional<Section> stamps;   // (K, T): a stamp each
    std::optional<uint32_t> cursor;
};
struct SceneClear {
    static constexpr uint16_t kType = CALIPER_NET_MSG_SCENE_CLEAR;
    uint32_t sub = 0;
    std::string view;
    uint64_t seq = 0, stamp_ns = 0;
    std::string path;
    bool recursive = false;
    std::optional<std::vector<std::string>> parts;
    bool reset = false, is_static = false, state = false;
    std::optional<Section> stamp;
    std::optional<uint32_t> cursor;
};
struct SceneClasses {                // 3.2 draft: the class table, replaced whole
    static constexpr uint16_t kType = CALIPER_NET_MSG_SCENE_CLASSES;
    uint32_t sub = 0;
    std::string view;
    uint64_t seq = 0, stamp_ns = 0;
    Json classes = Json::array();    // [{"id", "label", "color"}] in ascending id order
    bool is_static = false, state = false;
    std::optional<Section> stamp;
    std::optional<uint32_t> cursor;
};
// ---- 3.2 draft: text logs, recordings and split updates ----
struct Text {                        // new lines of a text log (kind 7)
    static constexpr uint16_t kType = CALIPER_NET_MSG_TEXT;
    uint32_t sub = 0;
    std::string view;
    uint64_t start = 0, total = 0, stamp_ns = 0;
    std::optional<uint64_t> dropped;
    Section text;                    // (L,) u8, the lines' UTF-8
    Section offsets;                 // (M + 1,) i32
    Section levels;                  // (M,) u8: 0 debug, 1 info, 2 warning, 3 error
    std::optional<Section> stamp, stamps;
    std::optional<uint32_t> cursor;
    std::string line(size_t i) const;   // line i's text
};
struct Recordings {                  // server to viewer: the recordings it can open
    static constexpr uint16_t kType = CALIPER_NET_MSG_RECORDINGS;
    Json recordings = Json::array();
};
struct RecordingOpen {
    static constexpr uint16_t kType = CALIPER_NET_MSG_RECORDING_OPEN;
    uint32_t request_id = 0;
    std::string recording;
};
struct Cursor {                      // viewer to server; only the fields of `op` are kept
    static constexpr uint16_t kType = CALIPER_NET_MSG_CURSOR;
    uint32_t request_id = 0;
    std::string op;                  // seek, play, pause, step, follow
    std::optional<std::string> timeline, value, pace;   // value: stamp text
    std::optional<int64_t> count;
    std::optional<double> speed;
    Json loop;                       // [start, end] as stamp text; null when absent
};
struct TimeState {                   // server to viewer
    static constexpr uint16_t kType = CALIPER_NET_MSG_TIME_STATE;
    std::optional<uint32_t> request_id;
    std::string session;
    uint32_t generation = 0;
    std::string state;               // following, paused, playing
    std::string timeline;
    std::optional<std::string> cursor;   // stamp text; nullopt (null on the wire) while following
    std::optional<double> speed;
    std::optional<std::string> pace;
    Json loop;                       // null when absent
    Json ranges = Json::object();    // timeline name -> [min, max] stamp text
    bool live = false;
};
// ---- 3.2 draft: GPU programs (capability "programs"; design/gpu-programs.md, Wire Path) ----
// The rules of the block "GPU programs and interventions" at the end of _wire.py.
struct Program {                     // viewer to producer: install or replace (op set) or remove (op remove) one program
    static constexpr uint16_t kType = CALIPER_NET_MSG_PROGRAM;
    uint32_t request_id = 0;
    std::string op;                  // CALIPER_NET_PROGRAM_SET, _REMOVE or _PARAMS
    std::string id;                  // 1 to CALIPER_NET_PROGRAM_ID_MAX bytes of [a-z0-9_-], first a letter, or operator.<id>.<n>
    Json declaration;                // set: the declaration as given (checked on decode); null otherwise
    Json params;  // op params: the values by name; null otherwise
    std::optional<std::string> wgsl;     // set: the section "wgsl", the source as UTF-8
    std::optional<std::string> sha256;   // set: of the source, 64 lowercase hex digits
    std::optional<std::string> scene;    // set: the scene of a geometry output (needed for geometry)
    // A set of an instance of a piece (capability "resources"; design/programmable-graphics.md, Resource
    // Message), in place of declaration, scene, sha256 and wgsl: the pin {name, version, hash}, the
    // manifest, the bindings and the segments as given (checked on decode, null when absent), the
    // instance's values in `params`, the panel, and the sections with the imports resolved, of which
    // derive or draw is present. A producer sends the draw half alone, with request_id 0, to a viewer.
    Json piece, manifest, bind, segments;
    std::optional<std::string> panel;
    std::optional<std::string> common, derive, draw;
};
struct ProgramState {                // producer to viewer: the status item of each program, in installation order
    static constexpr uint16_t kType = CALIPER_NET_MSG_PROGRAM_STATE;
    Json programs = Json::array();   // the status items as given (checked on decode), kept verbatim
    std::optional<uint32_t> request_id;   // the PROGRAM it answers; absent for an unrequested one
    std::optional<uint64_t> step;         // the script's step
    std::optional<double> step_ms;        // the base step time b of the cadence
};
// 3.2 draft, capability "resources" (design/programmable-graphics.md, Resource Message): one version of
// one published resource of a program, as a derive pass wrote it at a step of the run's cadence.
struct Resource {                    // producer to viewer or hub, hub to viewer
    static constexpr uint16_t kType = CALIPER_NET_MSG_RESOURCE;
    std::string program;             // the program's id
    std::string resource;            // the resource's id, a WGSL name of 1 to 63 bytes
    uint64_t step = 0;               // the run's step at the pass: the version's step stamp
    uint32_t count = 0;              // valid elements
    uint32_t stride = 0;             // bytes of one element: a multiple of 4 from 4 to CALIPER_NET_RESOURCE_STRIDE_MAX
    std::optional<uint32_t> width, height;   // a texture: count == width * height and stride 4
    // A part of a version in parts: the sender's counter of such versions, this part's number from 0, the
    // number of parts (2 to CALIPER_NET_MAX_PIECES) and the first element that `data` holds.
    struct Split {
        uint64_t version = 0;
        uint32_t part = 0, parts = 0, first = 0;
    };
    std::optional<Split> split;
    Section data;                    // "resource": u8 (count * stride,), or a part's elements; inside the frame on decode
};
// Stamp text: decimal digits with an optional leading minus, at most 20 characters, a signed 64-bit value.
bool stamp_text_ok(const std::string& v);

// A whole update put together from the pieces of a split update; its sections point into `keep`.
struct WholeUpdate {
    SceneUpdate update;
    std::shared_ptr<const void> keep;
};
// A receiver's split updates of one subscription (_wire.py's SplitUpdates). add() copies each piece's
// rows; once every piece and row has arrived it checks the whole update as an unsplit one. A newer
// update of the same entity part discards an incomplete one, at most CALIPER_NET_MAX_INCOMPLETE_UPDATES
// are held (the oldest goes first), and a piece of an update not held is dropped and counted.
class SplitUpdates {
public:
    enum class Result { Pending, Whole, Dropped, Refused };
    // Refused: the piece contradicts its update, or the whole update breaks the protocol; *err says why
    // (answered with bad_request) and the update is discarded.
    Result add(const SceneUpdate& piece, WholeUpdate* out, std::string* err);
    size_t held() const { return open_.size(); }
    uint64_t dropped() const { return dropped_; }
    uint64_t discarded() const { return discarded_; }

private:
    struct Array {
        int64_t dtype = 0;
        std::vector<int64_t> trailing;
        std::vector<uint8_t> bytes;
        std::vector<std::pair<uint64_t, uint64_t>> ranges;   // (first row, rows), within the rows
    };
    struct Assembly {
        uint64_t update = 0;
        SceneUpdate p0;              // piece 0 without its arrays; its stamp in stamp_bytes
        std::vector<uint8_t> stamp_bytes;
        int64_t pieces = 0;
        Json rows;
        std::vector<int64_t> seen;
        std::vector<std::pair<std::string, Array>> arrays;
        uint64_t bytes = 0;
    };
    std::vector<Assembly> open_;     // oldest first
    uint64_t dropped_ = 0, discarded_ = 0;
};

// A whole version put together from the parts of a version in parts; its data points into `keep`.
struct WholeResource {
    Resource resource;
    std::shared_ptr<const std::vector<uint8_t>> keep;
};
// A receiver's versions in parts of one connection (_wire.py's SplitResources). add() takes every RESOURCE
// of the connection in order. A version in one frame is Whole at once, with its data in the frame. Part 0
// allocates the version's count * stride bytes (at most CALIPER_NET_RESOURCE_VERSION_MAX_BYTES, which the
// decoder checked) and each part copies its elements in; the last part makes it Whole. A part whose
// version is not held is dropped and counted. A part out of order, or one whose keys differ from part 0,
// or a last part that does not end at the count, is Refused and discards its version. Part 0 of a newer
// version, or a version in one frame, of the same resource discards an incomplete one; at most
// CALIPER_NET_MAX_INCOMPLETE_UPDATES are held (the oldest goes first); and one whose next part has not come
// after CALIPER_NET_RESOURCE_PART_WAIT further RESOURCE frames is dropped (expired). With `copy` false the
// receiver keeps no bytes and checks the order alone, as the hub does, which keeps the frames themselves.
class SplitResources {
public:
    enum class Result { Pending, Whole, Dropped, Refused };
    explicit SplitResources(bool copy = true) : copy_(copy) {}
    Result add(const Resource& part, WholeResource* out, std::string* err);
    size_t held() const { return open_.size(); }
    bool holds(const std::string& program, const std::string& resource) const;
    uint64_t held_bytes() const;
    uint64_t dropped() const { return dropped_; }
    uint64_t discarded() const { return discarded_; }
    uint64_t expired() const { return expired_; }

private:
    struct Assembly {
        Resource head;               // part 0's keys, without data
        uint32_t next_part = 1;
        uint64_t next_first = 0;
        uint64_t last_frame = 0;     // frames_ at its last part
        std::shared_ptr<std::vector<uint8_t>> bytes;
    };
    bool copy_;
    std::vector<Assembly> open_;     // oldest first
    uint64_t frames_ = 0, dropped_ = 0, discarded_ = 0, expired_ = 0;
};

// The key "scene" of a VIEWS entry. view_camera: null when absent, unreadable or in a 2-D scene; else the
// orbit object (target, distance, azimuth, elevation) or the entity path of a camera.
struct SceneView { std::string space, up, units; Json view_camera; };
// An entity path: 0 to 16 segments of [A-Za-z0-9_.-]{1,63} joined by '/', at most 255 bytes, never "." or "..".
bool entity_path_ok(const std::string& path);
// A part's index in the order points, lines, arrows, boxes, mesh, image, camera (-1: unknown), and its
// array sections in the order a sender writes them (null-terminated).
int scene_part_index(const std::string& part);
const char* const* scene_part_sections(int part);
// The key "scene" of a VIEWS entry of kind scene with its defaults filled in (bad_request on false).
bool scene_view_info(const Json& entry, SceneView* out, std::string* err);
// The resolved default scene data view.
Json scene_data_view();
// The resolved data view of a scene subscription: only the default is built (bad_data_view on false;
// `request` null means none).
bool resolve_scene_data_view(const Json& view, const Json& request, Json* out, std::string* err);
// ---- end of the protocol 3.2 draft block ----

using Message = std::variant<Hello, Welcome, Bye, Error, Ping, Pong, Sessions, Attach, Views, Subscribe,
                             Unsubscribe, ViewRequest, ViewReply, Snapshot, Series, Event, Controls, Control,
                             ControlReply, Selection, SceneUpdate, SceneTransforms, SceneClear, SceneClasses,
                             Text, Recordings, RecordingOpen, Cursor, TimeState, Program, ProgramState, Resource,
                             Unknown>;

struct DecodeResult {
    enum class Status { Ok, BadFrame, BadMessage };
    Status status = Status::Ok;
    uint16_t type = 0;          // the frame's message type (0 for a bad frame)
    int close_code = 0;         // BadFrame: close the connection with this code
    std::string reason;         // BadFrame: truncated, bad_magic, unsupported_major, too_big, malformed
    std::string error_code;     // BadMessage: the ERROR code to answer with
    std::string message;        // what was wrong
    Frame frame;                // Ok and BadMessage: the decoded frame
    bool ok() const { return status == Status::Ok; }
};

// One frame's bytes to a message. *out is set only when the result is Ok.
DecodeResult decode(const uint8_t* buf, size_t n, Message* out);

// The frame bytes of a message. False with *err when it would break a limit.
bool encode(const Message& m, std::vector<uint8_t>* out, std::string* err = nullptr,
            uint16_t minor = CALIPER_NET_PROTOCOL_MINOR);

// The message type of a variant alternative, and a type's name ("snapshot", "?").
uint16_t message_type(const Message& m);
const char* type_name(uint16_t type);

}  // namespace wire
}  // namespace caliper_net
