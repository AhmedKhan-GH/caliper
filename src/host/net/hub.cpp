// hub.cpp: the wire protocol v3 hub of hub.h: public Hub lifecycle, event
// loop, HTTP/WebSocket accept and connection routing, low-level sending,
// timers, stats publishing, and token/hub-file helpers.
//
// The state, all owned by the thread in run():
//   Peer      one upgraded connection: a viewer, a producer, or a producer
//             connection that a resumed session replaced (retired).
//   Session   one producer process: its identity, its VIEWS, its channels and
//             its retained series. Kept after the producer leaves (ended).
//   Channel   one subscription of the hub at a producer (`up_sub`): a view
//             under one resolved data view. Viewer subscriptions whose
//             resolved data views are equal share it. It keeps the latest
//             snapshot, and stays (idle, with that snapshot) after its last
//             viewer leaves, so a late viewer is served at once.
//   History   the retained values of one series view, by absolute index.
//   VSub      one subscription of a viewer: its own `sub`, its channel, its
//             rate, its pending snapshot (latest wins) or its series cursor.
//   Waiter    a viewer's SUBSCRIBE or VIEW_REQUEST waiting for the producer's
//             VIEW_REPLY on a channel.
#include "net/hub.h"
#include "net/hub_internal.h"

#include "caliper/live_net.h"
#include "net/http_door.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#  include <process.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif
#ifdef __APPLE__
#  include <stdlib.h>   // arc4random_buf
#endif

namespace caliper_net {

using namespace hub_detail;

namespace {

namespace fs = std::filesystem;

std::string base64url(const uint8_t* p, size_t n) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < n; ++i) {
        acc = (acc << 8) | p[i];
        bits += 8;
        while (bits >= 6) {
            bits -= 6;
            out += k[(acc >> bits) & 63];
        }
    }
    if (bits > 0) out += k[(acc << (6 - bits)) & 63];
    return out;
}

void secure_random(uint8_t* p, size_t n) {
#if defined(__APPLE__)
    arc4random_buf(p, n);
#elif defined(_WIN32)
    std::random_device rd;   // rand_s, the system's cryptographic generator, on MSVC
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rd();
#else
    std::ifstream f("/dev/urandom", std::ios::binary);
    f.read((char*)p, (std::streamsize)n);
    if (!f) {
        std::random_device rd;
        for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rd();
    }
#endif
}

long this_pid() {
#ifdef _WIN32
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

}  // namespace

Options Hub::Impl::ws_options(const HubOptions& h) {
    Options o;
    o.max_queued_bytes = h.max_queued_bytes;
    o.max_connections = h.max_tcp_connections;
    o.header_timeout_ms = h.handshake_timeout_ms;
    o.body_limit = h.body_limit;
    o.body_timeout_ms = h.handshake_timeout_ms;
    return o;
}

void Hub::Impl::write_replies() {
    std::vector<Reply> r;
    {
        std::lock_guard<std::mutex> g(replies_mu);
        r.swap(replies);
    }
    for (Reply& x : r) server.respond(x.conn, x.status, x.type, x.body, x.headers);
}

// ---- sending ------------------------------------------------------------------
bool Hub::Impl::send_bytes(ConnId conn, std::vector<uint8_t>&& b) {
    auto shared = std::make_shared<const Bytes>(std::move(b));
    if (server.send(conn, shared)) return true;
    overflow(conn);
    return false;
}

bool Hub::Impl::send(ConnId conn, const wire::Message& m) {
    std::vector<uint8_t> b;
    if (!wire::encode(m, &b)) return false;
    return send_bytes(conn, std::move(b));
}

// A send queue past max_queued_bytes: the peer reads far too slowly.
void Hub::Impl::overflow(ConnId conn) {
    auto it = peers.find(conn);
    if (it == peers.end() || it->second.closing) return;
    close(conn, kCloseOverflow, "send queue full");
}

void Hub::Impl::close(ConnId conn, uint16_t code, const std::string& reason) {
    auto it = peers.find(conn);
    if (it != peers.end()) it->second.closing = true;
    server.close(conn, code, reason);
}

void Hub::Impl::bye(ConnId conn, const char* reason, uint16_t code, const std::string& text) {
    wire::Bye b;
    b.reason = reason;
    b.message = text;
    send(conn, b);
    close(conn, code, reason);
}

void Hub::Impl::error(ConnId conn, const std::string& code, const std::string& text,
                      std::optional<uint32_t> in_reply_to, bool fatal) {
    wire::Error e;
    e.code = code;
    e.message = text;
    e.fatal = fatal;
    e.in_reply_to = in_reply_to;
    send(conn, e);
}

void Hub::Impl::fatal(ConnId conn, uint16_t close_code, const std::string& code, const std::string& text) {
    error(conn, code, text, {}, true);
    close(conn, close_code, text.substr(0, 120));
}

void Hub::Impl::reply_error(ConnId conn, uint32_t request_id, uint32_t sub, const std::string& view,
                            const std::string& code, const std::string& text) {
    wire::ViewReply r;
    r.request_id = request_id;
    r.sub = sub;
    r.view = view;
    Json e = Json::object();
    e.set("code", code);
    e.set("message", text);
    r.error = e;
    send(conn, r);
}

void Hub::Impl::reply_ok(ConnId conn, uint32_t request_id, uint32_t sub, const std::string& view,
                         const Json& resolved) {
    wire::ViewReply r;
    r.request_id = request_id;
    r.sub = sub;
    r.view = view;
    r.data_view = resolved.is_object() ? resolved : Json::object();
    send(conn, r);
}

bool Hub::Impl::has_cap(const std::vector<std::string>& caps, const char* cap) {
    return std::find(caps.begin(), caps.end(), cap) != caps.end();
}

// The hub's WELCOME. 3.1: it announces `controls` and `selection`, and to a
// client that announced `controls` it says whether it takes (to a viewer)
// or forwards (to a producer) controls: only with allow_control.
wire::Welcome Hub::Impl::hub_welcome(bool client_has_controls) {
    wire::Welcome w;
    w.role = CALIPER_NET_ROLE_HUB;
    w.capabilities = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_SELECTION,
                      CALIPER_NET_CAP_PROGRAMS, CALIPER_NET_CAP_INTERVENTIONS, CALIPER_NET_CAP_INTERPRETATION, CALIPER_NET_CAP_TEXT_PARAMS,
                      CALIPER_NET_CAP_RESOURCES,       // 3.2 draft: forwarded
                      CALIPER_NET_CAP_SCENE_PIECES};   // 3.2 draft: bundled per entity part
    Json sw = Json::object();
    sw.set("name", "caliper-hub");
    sw.set("version", opt.software_version);
    w.software = sw;
    if (client_has_controls) w.control = opt.allow_control;
    return w;
}

// ---- connections ---------------------------------------------------------------------
void Hub::Impl::on_hello(Peer& p, const wire::Hello& h) {
    if (h.role == CALIPER_NET_ROLE_PRODUCER) {
        on_producer_hello(p, h);
        return;
    }
    size_t viewers = 0;
    for (auto& kv : peers) viewers += kv.second.role == Peer::Role::Viewer ? 1 : 0;
    if (viewers >= opt.max_viewers) {
        ++c.refused_full;
        fatal(p.conn, CALIPER_NET_CLOSE_SERVER_FULL, CALIPER_NET_ERR_LIMIT,
              "the hub holds " + std::to_string(opt.max_viewers) + " viewer connections");
        return;
    }
    p.role = Peer::Role::Viewer;
    p.cap_controls = has_cap(h.capabilities, CALIPER_NET_CAP_CONTROLS);
    p.cap_selection = has_cap(h.capabilities, CALIPER_NET_CAP_SELECTION);
    p.cap_scenes = has_cap(h.capabilities, CALIPER_NET_CAP_SCENES);
    p.cap_programs = has_cap(h.capabilities, CALIPER_NET_CAP_PROGRAMS);
    p.cap_interventions = has_cap(h.capabilities, CALIPER_NET_CAP_INTERVENTIONS);
    p.cap_interpretation = has_cap(h.capabilities, CALIPER_NET_CAP_INTERPRETATION);
    p.cap_text_params = has_cap(h.capabilities, CALIPER_NET_CAP_TEXT_PARAMS);
    p.cap_resources = has_cap(h.capabilities, CALIPER_NET_CAP_RESOURCES);
    p.cap_scene_pieces = p.cap_scenes && has_cap(h.capabilities, CALIPER_NET_CAP_SCENE_PIECES);
    wire::Welcome w = hub_welcome(p.cap_controls);
    if (p.cap_controls && p.watch_only) w.control = false;
    send(p.conn, w);
    send(p.conn, sessions_msg());
    reconsider_defaults();
}

void Hub::Impl::on_message(ConnId conn, Bytes&& data, bool binary) {
    auto pt = peers.find(conn);
    if (pt == peers.end() || pt->second.closing) return;
    Peer& p = pt->second;
    if (!binary) {
        close(conn, CALIPER_NET_CLOSE_UNSUPPORTED_DATA, "binary frames only");
        return;
    }
    if (p.role == Peer::Role::Retired) return;
    auto buf = std::make_shared<const Bytes>(std::move(data));
    wire::Message m;
    wire::DecodeResult r = wire::decode(buf->data(), buf->size(), &m);
    if (r.status == wire::DecodeResult::Status::BadFrame) {
        close(conn, (uint16_t)r.close_code, r.reason);
        return;
    }
    if (r.status == wire::DecodeResult::Status::BadMessage) {
        if (p.role == Peer::Role::None) {
            fatal(conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, r.error_code, r.message);
            return;
        }
        error(conn, r.error_code, r.message, r.type);
        return;
    }
    switch (p.role) {
        case Peer::Role::None:
            if (auto* h = std::get_if<wire::Hello>(&m)) on_hello(p, *h);
            else if (std::holds_alternative<wire::Bye>(m)) close(conn, CALIPER_NET_CLOSE_NORMAL, "bye");
            else
                fatal(conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST,
                      "the first message is hello");
            break;
        case Peer::Role::Viewer: on_viewer_message(p, r, m); break;
        case Peer::Role::Producer: on_producer_message(p, r, m, std::move(buf)); break;
        case Peer::Role::Retired: break;
    }
}

void Hub::Impl::on_closed(ConnId conn) {
    watch_only.erase(conn);
    auto pt = peers.find(conn);
    if (pt == peers.end()) return;
    Peer& p = pt->second;
    if (p.role == Peer::Role::Viewer) {
        detach_all(p);
        // Its waiting controls get no answer: nobody would read it.
        for (auto& kv : sessions)
            for (auto it = kv.second.routes.begin(); it != kv.second.routes.end();)
                it = it->second.viewer == conn ? kv.second.routes.erase(it) : std::next(it);
        for (auto& kv : sessions)
            for (auto it = kv.second.program_routes.begin(); it != kv.second.program_routes.end();)
                it = it->second.viewer == conn ? kv.second.program_routes.erase(it) : std::next(it);
    } else if (p.role == Peer::Role::Producer) {
        auto it = sessions.find(p.session);
        if (it != sessions.end() && it->second.producer == conn) end_session(it->second);
    }
    peers.erase(conn);
}

void Hub::Impl::handle(Event& e) {
    switch (e.type) {
        case Event::Type::Request:
            if (door->answer(server, e.conn, e.request) && opt.stream_access) {
                std::string t;
                e.request.query_param(CALIPER_NET_TOKEN_PARAM, &t);
                if (opt.stream_access(t) != HubOptions::StreamAccess::kFull) watch_only.insert(e.conn);
            }
            break;
        case Event::Type::Open: {
            Peer p;
            p.conn = e.conn;
            p.opened_ms = monotonic_ms();
            p.watch_only = watch_only.erase(e.conn) > 0;
            peers[e.conn] = std::move(p);
            break;
        }
        case Event::Type::Message:
            on_message(e.conn, std::move(e.data), e.binary);
            break;
        case Event::Type::Writable: {
            auto pt = peers.find(e.conn);
            if (pt != peers.end() && pt->second.in_flight && server.queued_bytes(e.conn) == 0) {
                pt->second.in_flight = false;
                pt->second.dirty = true;
            }
            // CONTROLS and SELECTIONs that waited for room (3.1).
            if (pt != peers.end() && (pt->second.pending_controls || !pt->second.pending_selections.empty()))
                pt->second.dirty = true;
            break;
        }
        case Event::Type::Pong:
            break;
        case Event::Type::Closed:
            on_closed(e.conn);
            break;
    }
}

// ---- timers ------------------------------------------------------------------------------
void Hub::Impl::tick() {
    const uint64_t ms = monotonic_ms();
    for (auto& kv : peers) {
        Peer& p = kv.second;
        if (p.closing) continue;
        if (p.role == Peer::Role::None) {
            if (ms - p.opened_ms >= (uint64_t)opt.handshake_timeout_ms)
                close(p.conn, CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT, "no hello");
            continue;
        }
        if (p.role == Peer::Role::Retired) continue;
        const uint64_t last = server.last_input_ms(p.conn);
        if (ms - last >= (uint64_t)opt.idle_timeout_ms) {
            close(p.conn, CALIPER_NET_CLOSE_IDLE_TIMEOUT, "idle");
        } else if (ms - last >= (uint64_t)opt.ping_interval_ms && p.pinged_at != last) {
            p.pinged_at = last;
            send(p.conn, wire::Ping{++p.nonce});
        }
    }
    // CONTROL requests whose answer never came (3.1).
    for (auto& kv : sessions)
        for (auto it = kv.second.routes.begin(); it != kv.second.routes.end();) {
            if (it->second.deadline_ms > ms) {
                ++it;
                continue;
            }
            answer_route(it->second, CALIPER_NET_ERR_TIMEOUT, "the producer did not answer in time");
            it = kv.second.routes.erase(it);
        }
    for (auto& kv : sessions)   // 3.2 draft: PROGRAMs whose PROGRAM_STATE never came
        for (auto it = kv.second.program_routes.begin(); it != kv.second.program_routes.end();) {
            if (it->second.deadline_ms > ms) {
                ++it;
                continue;
            }
            answer_program_route(it->second, CALIPER_NET_ERR_TIMEOUT, "the producer did not answer in time");
            it = kv.second.program_routes.erase(it);
        }
    if (ms - last_expire_ms >= 100) {
        last_expire_ms = ms;
        for (;;) {
            std::string old;
            for (auto& kv : sessions)
                if (!kv.second.live && ms - kv.second.ended_ms >= opt.ended_ttl_ms) old = kv.first;
            if (old.empty()) break;
            ++c.sessions_evicted;
            remove_session(old, true);
            log("session " + old + " forgotten");
            broadcast_sessions();
        }
    }
}

int Hub::Impl::timeout_ms(uint64_t now) {
    int t = 250;
    for (auto& kv : peers) {
        const Peer& p = kv.second;
        if (p.dirty) return 0;
        if (p.due_us) {
            const uint64_t d = p.due_us > now ? (p.due_us - now + 999) / 1000 : 0;
            t = std::min<int>(t, (int)d);
        }
    }
    return t;
}

// Stats for other threads: every 20 ms while busy, and before each wait.
void Hub::Impl::publish(bool force) {
    const uint64_t ms = monotonic_ms();
    if (!force && ms - last_publish_ms < 20) return;
    last_publish_ms = ms;
    HubStats st = c;
    st.connections = peers.size();
    for (auto& kv : peers) {
        st.viewers += kv.second.role == Peer::Role::Viewer ? 1 : 0;
        st.producers += kv.second.role == Peer::Role::Producer ? 1 : 0;
        st.viewer_subscriptions += kv.second.subs.size();
        st.queued_bytes += server.queued_bytes(kv.first);
    }
    for (auto& kv : sessions) {
        (kv.second.live ? st.sessions_live : st.sessions_ended) += 1;
        st.channels += kv.second.channels.size();
        st.channels_open += kv.second.by_up_sub.size();
    }
    st.retained_snapshot_bytes = snap_bytes;
    st.retained_series_bytes = series_bytes;
    st.retained_control_bytes = meta_bytes;
    st.retained_scene_bytes = scene_bytes;
    st.retained_resource_bytes = resource_bytes;
    for (auto& kv : sessions) st.controls_waiting += kv.second.routes.size();
    const DoorStats& d = door->stats();
    st.refused_host = d.refused_host;
    st.refused_origin = d.refused_origin;
    st.refused_token = d.refused_token;
    st.http_files = d.files;
    st.http_not_found = d.not_found;
    std::lock_guard<std::mutex> g(stats_mu);
    published = st;
}

void Hub::Impl::run(const std::atomic<bool>& stop) {
    {
        std::lock_guard<std::mutex> g(replies_mu);
        running = true;
    }
    while (!stop.load()) {
        uint64_t now = now_us();
        int wait = timeout_ms(now);
        if (opt.tick_ms > 0) wait = std::min(wait, opt.tick_ms);
        publish(wait > 0);   // current whenever the loop is about to wait
        std::vector<Event> evs = server.poll(wait);
        for (Event& e : evs) handle(e);
        write_replies();
        if (opt.on_tick) opt.on_tick();
        tick();
        now = now_us();
        for (auto& kv : peers) {
            Peer& p = kv.second;
            if (p.role != Peer::Role::Viewer) continue;
            if (p.dirty || (p.due_us && p.due_us <= now)) {
                p.dirty = false;
                pump(p, now);
            }
        }
        ++c.loops;
    }
    {
        std::lock_guard<std::mutex> g(replies_mu);
        running = false;
        replies.clear();
    }
    // Shut down: BYE to every connection, then the close handshake.
    for (auto& kv : peers)
        if (!kv.second.closing && server.is_open(kv.first)) {
            wire::Bye b;
            b.reason = CALIPER_NET_BYE_SHUTDOWN;
            b.message = "the hub is shutting down";
            send(kv.first, b);
        }
    server.shutdown(CALIPER_NET_CLOSE_GOING_AWAY, opt.shutdown_timeout_ms);
    peers.clear();
    publish(true);
}

// ---- Hub ---------------------------------------------------------------------------------------

Hub::Hub(HubOptions opt) : impl_(std::make_unique<Impl>(std::move(opt))) {}
Hub::~Hub() = default;

bool Hub::listen(std::string* err) {
    Impl& m = *impl_;
    if (!valid_token(m.opt.token)) {
        *err = "the session token must be 22 to 128 characters of A-Z, a-z, 0-9, '_' and '-'";
        return false;
    }
    if (!m.opt.assets_dir.empty() && !m.files.open(m.opt.assets_dir, err)) return false;
    if (!m.server.listen(m.opt.port, err, m.opt.host)) return false;
    const std::string note =
        "This is a Caliper hub. It serves no viewer page because it was started without --assets.\n"
        "The Python package ships the viewer page in caliper/_web/; start the hub with\n"
        "--assets <that folder> to open the stream in a browser at this address.\n";
    m.door = std::make_unique<HttpDoor>(m.opt.token, m.server.port(), &m.files, note,
                                        m.opt.stream_path, m.opt.on_request, m.opt.stream_alias);
    if (m.opt.stream_access) {
        auto access = m.opt.stream_access;
        m.door->set_token_check(
            [access](const std::string& t) { return access(t) != HubOptions::StreamAccess::kRefused; });
    }
    m.publish(true);
    return true;
}

uint16_t Hub::port() const { return impl_->server.port(); }
const std::string& Hub::token() const { return impl_->opt.token; }

std::string Hub::viewer_url() const {
    return "http://127.0.0.1:" + std::to_string(port()) + "/?" CALIPER_NET_TOKEN_PARAM "=" + impl_->opt.token;
}

std::string Hub::stream_url() const {
    return "ws://127.0.0.1:" + std::to_string(port()) + impl_->opt.stream_path;
}

void Hub::respond(ConnId c, int status, std::string content_type, std::string body,
                  std::vector<std::pair<std::string, std::string>> headers) {
    // Woken under the lock: once run() has cleared `running`, nothing touches
    // the server from another thread.
    std::lock_guard<std::mutex> g(impl_->replies_mu);
    if (!impl_->running) return;
    impl_->replies.push_back({c, status, std::move(content_type), std::move(body), std::move(headers)});
    impl_->server.wake();
}

void Hub::run() {
    if (!impl_->door) return;
    impl_->run(stop_);
}

void Hub::stop() {
    stop_.store(true);
    impl_->server.wake();
}

HubStats Hub::stats() const {
    std::lock_guard<std::mutex> g(impl_->stats_mu);
    return impl_->published;
}

// ---- tokens and the hub file ------------------------------------------------------------------

bool valid_token(const std::string& t) {
    if (t.size() < CALIPER_NET_TOKEN_MIN || t.size() > CALIPER_NET_TOKEN_MAX) return false;
    for (char ch : t)
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' ||
              ch == '-'))
            return false;
    return true;
}

std::string random_token() {
    uint8_t b[24];
    secure_random(b, sizeof b);
    return base64url(b, sizeof b);
}

bool read_token_file(const std::string& path, std::string* token, std::string* err) {
#ifndef _WIN32
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        *err = "cannot read the token file " + path;
        return false;
    }
    if (st.st_mode & 077) {
        *err = "the token file " + path + " can be read or written by other users; make it private (chmod 600)";
        return false;
    }
#endif
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *err = "cannot read the token file " + path;
        return false;
    }
    std::string line;
    std::getline(f, line);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
    size_t a = 0;
    while (a < line.size() && (line[a] == ' ' || line[a] == '\t')) ++a;
    line = line.substr(a);
    if (!valid_token(line)) {
        *err = "the token file " + path +
               " does not start with a session token (22 to 128 characters of A-Z, a-z, 0-9, '_' and '-')";
        return false;
    }
    *token = line;
    return true;
}

std::string hub_file_text(uint16_t port, const std::string& token, long pid) {
    Json j = Json::object();
    j.set("version", 1);
    j.set("pid", (int64_t)pid);
    j.set("port", (uint64_t)port);
    j.set("token", token);
    j.set("stream_url", "ws://127.0.0.1:" + std::to_string(port) + CALIPER_NET_PATH);
    j.set("url", "http://127.0.0.1:" + std::to_string(port) + "/?" CALIPER_NET_TOKEN_PARAM "=" + token);
    return j.dump() + "\n";
}

bool write_hub_file(const std::string& path, const std::string& text, std::string* err) {
    std::error_code ec;
    const fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    const std::string tmp = path + ".tmp" + std::to_string(this_pid());
#ifndef _WIN32
    ::unlink(tmp.c_str());
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_TRUNC, 0600);
    if (fd < 0) {
        *err = "cannot write " + tmp;
        return false;
    }
    size_t off = 0;
    while (off < text.size()) {
        const ssize_t w = ::write(fd, text.data() + off, text.size() - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
    ::close(fd);
    if (off != text.size()) {
        ::unlink(tmp.c_str());
        *err = "cannot write " + tmp;
        return false;
    }
#else
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text;
        if (!f) {
            *err = "cannot write " + tmp;
            return false;
        }
    }
#endif
    fs::rename(tmp, p, ec);
    if (ec) {
        fs::remove(tmp, ec);
        *err = "cannot write " + path;
        return false;
    }
    return true;
}

void remove_hub_file(const std::string& path, const std::string& text) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    if (ss.str() != text) return;
    std::error_code ec;
    fs::remove(path, ec);
}

}  // namespace caliper_net
