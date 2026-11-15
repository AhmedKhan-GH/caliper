// live_net_source.h: a caliper_live::Source over wire protocol v3 (caliper/
// live_net.h): the native app as a viewer of a producer (caliper.serve()) or
// a hub (`caliper serve`) on the loopback interface. macOS only so far (the
// slots are MTLBuffers); host-internal.
//
// THE CONNECTION. A network thread, created by open_net_source() and joined by
// the destructor, owns the WebSocket (net/ws.h). It sends HELLO as a viewer,
// expects WELCOME, follows SESSIONS (the server attaches it to its one live
// session; with `session` set it sends ATTACH for that one), subscribes to
// every view of the VIEWS list with the view's default data view, answers
// PING, pings after 10 s of silence and closes after 30 s (4004). Frames that
// break the protocol close the connection with the decoder's code; an invalid
// message is answered with ERROR and the connection stays open. After a
// connection that was attached ends without a refusal, the thread reconnects
// (every 0.25 s growing to 2 s): views of the same session keep their keys and
// last pixels, series continue from the backfill, and a new session id starts
// new views. A failure before the first WELCOME is final.
//
// THE HAND-OVER (the claim rules of live_source.h). Every method of Source is
// called on the frame thread. Each view of a texture or points kind owns
// kSlots shared-storage MTLBuffers on the renderer's device. The network thread
// copies a received payload into a slot that is neither claimed (current or
// previous) nor the newest completed one, then publishes it; take() claims the
// newest completed slot. With two claims and one published slot at most, a
// fourth slot is always free, so a snapshot is never dropped for want of a slot
// and the newest one always reaches the view. One mutex guards the slot words
// and is held only for those few loads and stores, never across a copy or an
// allocation. Points travel as (N, D) and are written as the (N, 3) the points
// view draws, z = 0 when D = 2. A series view keeps an append-only ring of
// kSeriesRing values that only the frame thread writes: the network thread
// queues the SERIES deltas, take() applies them (gaps become NaN, overlaps
// replace values) and hands the ring over, never claimed.
#pragma once

#include "live/live_source.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caliper_live {

// A stream URL, checked and normalised. Accepted forms:
//   ws://127.0.0.1:<port>/v3/stream?token=<token>     what a viewer connects to
//   http://127.0.0.1:<port>/?token=<token>            the page link caliper.serve() prints
// with "localhost" for 127.0.0.1, and an optional `session=<id>` parameter that
// names the session to attach to. The host must be loopback and the token
// 22 to 128 characters of [A-Za-z0-9_-] (the protocol's rules).
struct StreamUrl {
    std::string host;        // "127.0.0.1" or "localhost"
    uint16_t    port = 0;
    std::string token;
    std::string session;     // "" = the server's default
    std::string ws_url;      // ws://<host>:<port>/v3/stream?token=<token>
    std::string display;     // ws://<host>:<port>/v3/stream (no token: safe to show and log)
};
bool parse_stream_url(const std::string& url, StreamUrl* out, std::string* err);

class NetSource : public Source {
public:
    static constexpr uint32_t kSlots = 4;
    static constexpr uint32_t kSeriesRing = 65536;   // series values a view holds

    enum class Ready { Attached, Pending, Failed };
    // Waits up to timeout_ms for the first attachment (VIEWS of a session) or a
    // final failure. Pending: connected, but no session yet, or still
    // connecting. Failed: *err says why (unreachable, refused with an HTTP
    // status, a protocol error).
    virtual Ready wait_ready(int timeout_ms, std::string* err) = 0;
    // True until the first attachment or a final failure.
    virtual bool pending() const = 0;
    // What the connection is doing, for a status line ("" when all is well).
    virtual std::string problem() const = 0;
};

// Starts the network thread. NULL with *err for a bad URL or a missing device.
std::unique_ptr<NetSource> open_net_source(const std::string& url, void* mtl_device, std::string* err);

}  // namespace caliper_live
