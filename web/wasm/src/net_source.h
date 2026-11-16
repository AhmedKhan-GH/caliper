// net_source.h: the browser's caliper_live::Source over wire protocol v3
// (caliper/live_net.h, docs/references/wire-protocol.md).
//
// A viewer connection on the browser's own WebSocket (Emscripten's websocket
// API): HELLO (with the capability scenes of the protocol 3.2 draft), WELCOME,
// SESSIONS (the server's default attach), VIEWS,
// SUBSCRIBE / VIEW_REPLY for the views this instance shows, SNAPSHOT (the
// latest wins), SERIES deltas with their backfill, VIEW_REQUEST for a new
// data view, EVENT for what a person did in a view, PING / PONG both ways and
// BYE, and the SCENE_UPDATE and SCENE_CLEAR messages of a scene view, which
// are kept in order for the scene view with a copy of their frame
// (live/scene_queue.h). Frames go through the shared C++ codec
// (src/host/net/wire.h).
//
// Everything runs on the browser's main thread: the socket callbacks run
// between frames, and the frame loop polls the Source exactly as live_core.cpp
// polls the shared-memory session. A snapshot lands in host memory, so
// Snapshot::mtl_buffer is null and host_base points at the bytes; the live
// views copy CPU payloads when they apply them, so a claim ends at the apply.
//
// Memory is bounded: per view one buffer that receives snapshots and one that
// the last take() handed out (swapped, never reallocated once large enough),
// and a series ring of fixed capacity indexed by absolute value index.
#pragma once

#include "live/live_clock.h"
#include "live/live_source.h"
#include "live/scene_queue.h"
#include "net/json.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace caliper_web {

class NetSource final : public caliper_live::Source {
public:
    // Values a series view keeps (the newest kSeriesCapacity indices).
    static constexpr uint64_t kSeriesCapacity = 65536;

    struct Options {
        std::string url;    // ws://127.0.0.1:<port>/v3/stream?token=<token>
        std::string view;   // the one view to show; "" = every view of the session
        int32_t kind = 0;   // the expected kind of `view`, 0 = any
        double max_rate = 60.0;
    };
    // What the page host hears about (all on the main thread).
    struct Callbacks {
        std::function<void(const std::string& view, const std::string& data_view_json)> data_view;
        std::function<void(const std::string& text)> status;   // connection state changes
    };

    NetSource(Options opt, Callbacks cb);
    ~NetSource() override;

    bool connect();                  // opens the socket; HELLO follows on open
    void close(const char* reason);  // BYE, then close (1000)

    // A new data view for the shown view (VIEW_REQUEST). False when there is
    // no single subscribed view yet or the JSON is not an object.
    bool request_data_view(const std::string& data_view_json, std::string* err);
    // The view key of a view name, 0 when it is not shown.
    uint64_t key_of(const std::string& name) const;

    // Connection state for the page host's status line.
    const std::string& state_text() const { return state_; }
    bool connected() const { return welcomed_ && !closed_; }
    uint64_t messages() const { return messages_; }
    uint64_t bytes() const { return bytes_; }
    // Snapshots are handed over with their stamps in this page's clock
    // (live_clock.h): the producer's clock minus ours, as estimated so far.
    int64_t clock_offset_ns() const { return clock_.offset_ns(); }

    // The snapshot the last take() of view `name` handed out, with host_base
    // set (valid until its next take), or null. For the colormap check.
    const caliper_live::Snapshot* shown(const std::string& name) const;
    // A short line per view (name, subscription state, error) for the status.
    std::string views_text() const;

    // ---- caliper_live::Source ------------------------------------------------
    std::string label() const override { return session_id_; }
    std::string script() const override { return script_; }
    uint64_t start_unix_ms() const override { return start_unix_ms_; }
    int32_t producer_pid() const override { return pid_; }
    bool producer_alive() override { return !closed_ && !ended_; }
    bool sync(std::vector<caliper_live::SourceView>* views) override;
    bool take(uint64_t key, uint64_t after_seq, caliper_live::Snapshot* out) override;
    void release_prev(uint64_t) override {}
    void unclaim(uint64_t) override {}
    void drew(uint64_t, uint64_t, uint64_t) override {}
    uint64_t producer_dropped(uint64_t) override { return 0; }
    void heartbeat() override;
    // EVENT with the subscription, seq and data view of what the view shows.
    bool send_event(uint64_t key, const std::string& type, const std::string& detail_json) override;
    bool request_data_view(uint64_t key, const std::string& data_view_json, std::string* err) override;
    // The view's VIEWS entry (a scene's description); the resolved data view is not kept here.
    bool data_view_of(uint64_t key, caliper_net::Json* resolved, caliper_net::Json* offer) override;
    bool take_scene(uint64_t key, std::vector<caliper_live::SceneItem>* out) override;
    void stop_scene(uint64_t key) override;

    // Socket callbacks (public for the C trampolines in net_source.cpp).
    void on_open();
    void on_message(const uint8_t* data, uint32_t n, bool text);
    void on_close(unsigned short code, const char* reason);

private:
    struct Payload {
        std::vector<uint8_t> bytes;   // data, then aux at aux_offset (8-aligned)
        std::vector<float> stats;
        caliper_live::Snapshot meta;  // host_base / stats / map are set at take()
        caliper_live::PayloadMap map; // the snapshot's axes and tile
        caliper_net::Json data_view;  // the snapshot's resolved data view
    };
    struct View {
        uint64_t id = 0;              // the producer's view id = the Source key
        std::string name;
        int32_t kind = 0;
        uint32_t sub = 0;             // 0 = not subscribed
        uint32_t request_id = 0;      // of the pending SUBSCRIBE / VIEW_REQUEST
        bool subscribed = false;      // VIEW_REPLY said yes
        std::string error;
        std::string data_view;        // the resolved data view (JSON text)
        // snapshots
        Payload incoming, shown;
        bool fresh = false;           // incoming holds a snapshot not yet taken
        // series
        std::vector<float> ring;      // kSeriesCapacity values, index i at i % capacity
        uint64_t series_end = 0;      // one past the newest index held
        uint64_t series_seq = 0;      // SERIES messages received
        uint64_t series_taken = 0;
        uint64_t series_stamp = 0;
        float no_stats[1] = {0.0f};
        // scene (kind 6)
        caliper_net::Json offer;      // its VIEWS entry
        caliper_live::SceneQueue scene;
        caliper_net::wire::SplitUpdates splits;   // split scene updates being put together
    };

    View* by_sub(uint32_t sub);
    View* by_id(uint64_t id);
    void send(const std::vector<uint8_t>& frame);
    template <class M> void send_msg(const M& m);
    void subscribe(View& v);
    void resubscribe(View& v);
    void set_state(const std::string& s);

    Options opt_;
    Callbacks cb_;
    int socket_ = 0;
    bool welcomed_ = false, closed_ = false, ended_ = false;
    std::string state_ = "connecting";
    std::string session_id_, script_;
    uint64_t start_unix_ms_ = 0;
    int32_t pid_ = 0;
    std::vector<View> views_;
    bool views_changed_ = false;
    uint32_t next_sub_ = 1, next_request_ = 1, next_nonce_ = 1;
    double last_rx_ms_ = 0.0, last_ping_ms_ = 0.0;
    uint64_t messages_ = 0, bytes_ = 0;
    caliper_live::StampClock clock_;
    std::vector<uint8_t> out_;        // reused encode buffer
};

}  // namespace caliper_web
