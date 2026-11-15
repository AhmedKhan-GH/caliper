#pragma once
// wire.h: the C++ codec of wire protocol v3 (caliper/live_net.h,
// docs/references/wire-protocol.md), the native twin of python/caliper/
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
// LIFETIME. A decoded Section points into the buffer given to decode(): keep
// that buffer alive while the message is used.
#include "caliper/live_net.h"
#include "net/json.h"

#include <cstddef>
#include <cstdint>
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
};
struct Series {
    static constexpr uint16_t kType = CALIPER_NET_MSG_SERIES;
    uint32_t sub = 0;
    std::string view;
    uint64_t start = 0;                    // absolute index of the first value
    uint64_t total = 0;                    // values appended so far
    uint64_t stamp_ns = 0;
    Section values;                        // (M,) f32
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
};
struct Control {    // pause, resume, step, set, run
    static constexpr uint16_t kType = CALIPER_NET_MSG_CONTROL;
    uint32_t request_id = 0;
    std::string op;
    std::optional<std::string> name;       // set, run
    Json value;                            // set: a number, a boolean or a choice; null otherwise
    std::optional<uint32_t> count;         // step: at least 1; absent means 1
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

using Message = std::variant<Hello, Welcome, Bye, Error, Ping, Pong, Sessions, Attach, Views, Subscribe,
                             Unsubscribe, ViewRequest, ViewReply, Snapshot, Series, Event, Controls, Control,
                             ControlReply, Selection, Unknown>;

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
