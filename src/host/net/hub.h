#pragma once
// hub.h: the hub of wire protocol v3 (`caliper serve`): producers connect out
// to it and each becomes a session; viewers connect to it, list the sessions,
// attach to one, subscribe to its views and send data view requests and
// events, which the hub forwards to that session's producer. The reference
// page docs/reference/wire-protocol.md states the protocol ("The producer
// connection to a hub", "Choosing a session", "Flow control", "Limits"); the
// guide docs/python/hub-setup.md states what this hub keeps and for how long.
// Host-internal C++17 on net/ws.h, net/wire.h and net/http_door.h; no window,
// renderer or GPU.
//
// THE MODEL. One thread runs the hub: run() loops around one poll(2) of the
// WebSocket server until stop(). That is enough because the hub only moves
// bytes: it decodes metadata, never copies a payload per viewer (a snapshot's
// sections are shared by every viewer connection that sends it), and every
// socket is non-blocking, so no connection can hold the loop. A test runs a
// Hub on a thread of its own; `caliper serve` (host_cli.h) is a thin wrapper.
//
// SUBSCRIPTIONS. On the producer connection the hub is the producer's one
// viewer. It numbers its own subscriptions there (`sub` and `request_id` are
// rewritten in both directions), and viewers that ask for the same view with
// equal resolved data views share one producer subscription, at the largest
// max_rate among them. The hub keeps the latest snapshot of every producer
// subscription and the retained values of every series view (it subscribes
// to each series view as soon as VIEWS names it), so a late viewer gets them
// at once. Events go to the producer of the viewer's session.
//
// FLOW CONTROL. Per viewer connection the hub hands one data message (SNAPSHOT
// or SERIES) to the socket at a time and chooses the next once it has
// drained; it keeps at most one pending snapshot per subscription, replaced by
// a newer one, and serves series values from its retained history by index,
// coalesced and never dropped while retained. Scene bundles also send one
// frame at a time. Their resources and final update stay together before
// other data, even when the bundle exceeds the send queue. Control messages go first.
// Each subscription gets its max_rate messages per second on average, in
// bursts of at most two, so that a producer sending at that very rate with
// some jitter passes through without waiting. A slow viewer therefore delays
// only itself, and the hub keeps reading every producer.
//
// SESSIONS THAT END. When a producer disconnects, its session stays listed as
// `ended` with its last snapshots and series, and viewers can still attach and
// subscribe to what it retained. A producer that reconnects with the same id,
// pid and start time resumes it (BYE `replaced` to the old connection). Ended
// sessions go after ended_ttl_ms, beyond max_ended_sessions (the oldest
// first), or when retained data passes max_retained_bytes.
//
// PROTOCOL 3.2 DRAFT, SCENES. The hub announces `scenes`, leaves the scene views
// out of the VIEWS of a viewer that did not announce it, forwards SCENE_UPDATE
// and SCENE_CLEAR to the viewers of the producer subscription (in-order clears
// and states, the newest batch per entity part, each part at max_rate), and
// keeps the newest batch of each entity part, a clear applied, which a late
// viewer receives as the scene's state after a reset of the hub's own.
//
// PROTOCOL 3.1. The hub announces `controls` and `selection`. It keeps the
// latest CONTROLS of each session and the latest SELECTION of each item space
// (at most CALIPER_NET_MAX_ITEM_SPACES), also after the session ends, passes
// them to every viewer of the session that announced the capability, and
// sends them to a viewer right after the VIEWS of an attachment; a newer one
// replaces one still waiting for a slow viewer. A CONTROLS reaches a viewer
// after the data the hub received before it, or after 2 s
// (CALIPER_NET_ORDER_WAIT_MS), so that a paused view shows the step the run
// stopped at. VIEWS reach a viewer without
// `selection` with the item spaces removed, and a select event reaches a
// producer without `selection` without them. A viewer's CONTROL goes to the
// producer of its own session only with allow_control (--allow-control, off
// by default), under a request_id of the hub's, and the CONTROL_REPLY comes
// back to that viewer under its own; without allow_control, unattached, or
// for an ended session or a producer without `controls`, the hub answers
// itself (forbidden, not_attached, unsupported). WELCOME's `control` tells a
// viewer, and a producer, which it is. Waiting controls are bounded per viewer
// and per session, and answered `unsupported` when the producer goes and
// `timeout` when no answer comes.
//
// SECURITY. 127.0.0.1 only, the session token, the Host and Origin checks of
// http_door.h. The hub forwards data, events and, when allowed, controls that
// name what a script registered; it never runs anything a message names and
// carries no host command.
//
// EMBEDDED. A server with endpoints of its own (Caliper API, api/
// api_server.h) runs a hub on its port: it mounts the stream at another path
// and takes every other request through HubOptions::on_request, answering it
// with Hub::respond() from any thread.
#include "caliper/live_net.h"
#include "net/ws.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caliper_net {

struct HubOptions {
    std::string host = "127.0.0.1";
    uint16_t port = 0;                 // 0: any free port (see Hub::port())
    std::string token;                 // the session token (valid_token())
    std::string assets_dir;            // serve the viewer page from here; "" serves a note
    std::string software_version;      // WELCOME software.version
    // 3.1: forward viewers' CONTROL to the producer of their session. Off by
    // default: every CONTROL is then answered `forbidden` (--allow-control).
    bool allow_control = false;

    // Bounds. Each is documented in docs/python/hub-setup.md.
    size_t max_viewers = 256;          // viewer connections
    size_t max_producers = 64;         // live sessions
    size_t max_ended_sessions = 32;    // ended sessions kept, the oldest go first
    uint64_t ended_ttl_ms = 24ull * 3600 * 1000;   // how long an ended session is kept
    size_t max_retained_bytes = size_t(1) << 30;   // latest snapshots plus series values
    size_t series_retain = size_t(1) << 20;        // values kept per series view
    size_t max_idle_channels = 64;     // per session: data views nobody watches, kept with their
                                       // last snapshot for a late viewer
    size_t max_queued_bytes = size_t(160) << 20;   // per connection: the send queue
    size_t max_tcp_connections = 512;  // every TCP connection, plain HTTP included

    int handshake_timeout_ms = 10000;  // CALIPER_NET_HANDSHAKE_TIMEOUT_MS
    int ping_interval_ms = 10000;      // CALIPER_NET_PING_INTERVAL_MS
    int idle_timeout_ms = 30000;       // CALIPER_NET_IDLE_TIMEOUT_MS
    int shutdown_timeout_ms = 2000;    // wait this long for peers to answer the close

    // One line per session that starts, resumes or ends (the CLI prints them);
    // null for silence. Never given the token. Called on the hub's thread.
    std::function<void(const std::string&)> log;

    // For a server that embeds the hub. The stream is upgraded at
    // stream_path, and also at stream_alias when that is not empty, with the
    // same checks. When on_request is set, every request on another path goes
    // to it, on the hub's thread after the Host check, and the hub serves no
    // files and no note; the request is answered through Hub::respond().
    // on_tick runs on the hub's thread once per loop round, and at least
    // every tick_ms when that is above 0.
    std::string stream_path = CALIPER_NET_PATH;
    std::string stream_alias;          // a second path for the same stream, "" none
    std::function<void(ConnId, const HttpRequest&)> on_request;
    std::function<void()> on_tick;
    int tick_ms = 0;
    // For an embedded server whose stream takes more than one token (Caliper
    // API: its session token, and the tokens of the agents it serves). When
    // set, an upgrade's token is checked with it in place of `token` (which
    // must still be valid, and which viewer_url() gives): kRefused answers
    // 403 as a wrong token does, kWatch opens a connection that receives
    // everything but may not control (its WELCOME sets `control` false and a
    // CONTROL is answered `forbidden`, as without allow_control) or publish
    // (a producer's HELLO is answered ERROR `unsupported` and closed with
    // 4002, as caliper.serve() refuses producers), and kFull one that may
    // publish, and control when allow_control is on. Called on the hub's
    // thread at each upgrade, so a token issued or revoked later counts from
    // the next upgrade; it compares in constant time.
    enum class StreamAccess { kRefused, kWatch, kFull };
    std::function<StreamAccess(const std::string& token)> stream_access;
    // For an embedded server with an endpoint that takes a body (Caliper
    // API's command endpoint): the most body bytes to read for a request head,
    // 0 for none, as net/ws.h's Options::body_limit. A body must arrive within
    // handshake_timeout_ms. Called on the hub's thread, before the Host check.
    std::function<size_t(const HttpRequest&)> body_limit;
};

// Counters and sizes, as of the end of the hub's last loop round.
struct HubStats {
    size_t connections = 0;            // upgraded connections
    size_t viewers = 0, producers = 0;
    size_t sessions_live = 0, sessions_ended = 0;
    size_t channels = 0;               // producer subscriptions the hub keeps (open or idle)
    size_t channels_open = 0;          // of which subscribed at the producer
    size_t viewer_subscriptions = 0;
    uint64_t retained_snapshot_bytes = 0;   // latest snapshots kept
    uint64_t retained_series_bytes = 0;     // series values kept
    uint64_t retained_control_bytes = 0;    // 3.1: CONTROLS and SELECTIONs kept
    uint64_t queued_bytes = 0;              // unsent bytes over every connection
    uint64_t snapshots_in = 0, snapshots_out = 0, snapshots_superseded = 0;
    uint64_t series_in = 0, series_out = 0, series_values_out = 0;
    uint64_t events_forwarded = 0, events_dropped = 0;
    // 3.1
    uint64_t controls_forwarded = 0, controls_refused = 0, control_replies = 0;
    uint64_t controls_out = 0, selections_out = 0, selections_dropped = 0;
    // 3.2 draft: PROGRAMs forwarded and refused, PROGRAM_STATEs sent to viewers
    uint64_t programs_forwarded = 0, programs_refused = 0, program_states_out = 0;
    uint64_t controls_held = 0;             // CONTROLS that waited for the data before them
    uint64_t controls_hold_expired = 0;     // of which went after 2 s with that data still unsent
    // 3.2 draft: scene messages in, out and replaced by a newer batch of the
    // same entity part while waiting, and the bytes of the batches kept
    uint64_t scenes_in = 0, scenes_out = 0, scenes_superseded = 0;
    uint64_t scene_pieces_dropped = 0;   // pieces of a split update whose piece 0 the hub did not hold, and updates
                                         // in the piece form whose RESOURCE versions did not come before them
    uint64_t retained_scene_bytes = 0;
    // 3.2 draft, capability "resources": RESOURCE versions in, out and replaced by a newer version of
    // the same resource while waiting, the draw halves that producers relayed and the hub sent to
    // viewers, and the bytes of the newest versions kept for late viewers
    uint64_t resources_in = 0, resources_out = 0, resources_superseded = 0, resources_dropped = 0;
    // parts of a version in parts that the hub dropped or refused (its part 0 not held, out of order, a
    // part that differs from part 0, a total that disagrees), and incomplete versions it let go
    // (a newer version, the bound of two, or no next part within CALIPER_NET_RESOURCE_PART_WAIT frames)
    uint64_t resource_parts_dropped = 0, resource_versions_incomplete = 0;
    uint64_t program_relays_in = 0, program_relays_out = 0;
    uint64_t retained_resource_bytes = 0;
    size_t controls_waiting = 0;            // CONTROLs waiting for the producer's answer
    uint64_t refused_host = 0, refused_origin = 0, refused_token = 0, refused_full = 0;
    uint64_t http_files = 0, http_not_found = 0;
    uint64_t sessions_evicted = 0;
    uint64_t loops = 0;
};

class Hub {
public:
    explicit Hub(HubOptions opt);
    ~Hub();
    Hub(const Hub&) = delete;
    Hub& operator=(const Hub&) = delete;

    // Binds 127.0.0.1:port and opens the assets folder. False with *err.
    bool listen(std::string* err);
    uint16_t port() const;
    const std::string& token() const;
    std::string viewer_url() const;    // http://127.0.0.1:<port>/?token=<token>
    std::string stream_url() const;    // ws://127.0.0.1:<port><stream_path> (no token)

    // Any thread: answers a request that on_request took, with a plain HTTP
    // response after which the connection closes. Ignored once run() ended.
    void respond(ConnId c, int status, std::string content_type, std::string body,
                 std::vector<std::pair<std::string, std::string>> headers = {});

    // Runs until stop(), then sends BYE `shutdown` on every connection, closes
    // them with 1001 and stops listening.
    void run();
    // Any thread, and async-signal-safe: makes run() finish.
    void stop();
    bool stopping() const { return stop_.load(); }

    HubStats stats() const;            // any thread

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> stop_{false};
};

// ---- the session token and the hub file (caliper serve) ----------------------------

// A session token by the protocol's rule: 22 to 128 of [A-Za-z0-9_-].
bool valid_token(const std::string& t);
// 32 characters of base64url from 24 bytes of the system's secure random
// source (the strength of Python's secrets.token_urlsafe(24)).
std::string random_token();
// The token on the first line of `path`. False with *err when the file is
// missing, holds no valid token, or (POSIX) can be read by other users.
bool read_token_file(const std::string& path, std::string* token, std::string* err);

// The hub file tells producers on this machine where the hub is: a JSON object
// {"pid", "port", "stream_url", "token", "url", "version": 1}, written with
// mode 0600 (owner only) through a temporary file renamed over `path`.
std::string hub_file_text(uint16_t port, const std::string& token, long pid);
bool write_hub_file(const std::string& path, const std::string& text, std::string* err);
// Removes `path` if it still holds `text` (a newer hub's file is left alone).
void remove_hub_file(const std::string& path, const std::string& text);

}  // namespace caliper_net
