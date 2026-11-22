// net_source.cpp: the browser's Source over wire protocol v3 (net_source.h).
#include "net_source.h"

#include "live/live_series_ring.h"
#include "live/scene_queue.h"

#include "caliper/live.h"       // CALIPER_LIVE_KIND_*, CALIPER_LIVE_SNAP_*
#include "caliper/live_net.h"
#include "net/wire.h"

#include <emscripten/emscripten.h>
#include <emscripten/websocket.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <variant>

namespace caliper_web {

namespace wire = caliper_net::wire;
using caliper_net::Json;

namespace {

double now_ms() { return emscripten_get_now(); }

// The live views' clock off Apple (live_views.cpp now_ns()), which the stamps
// are mapped into.
uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool on_open_cb(int, const EmscriptenWebSocketOpenEvent*, void* ud) {
    static_cast<NetSource*>(ud)->on_open();
    return true;
}
bool on_message_cb(int, const EmscriptenWebSocketMessageEvent* e, void* ud) {
    static_cast<NetSource*>(ud)->on_message(e->data, e->numBytes, e->isText);
    return true;
}
bool on_error_cb(int, const EmscriptenWebSocketErrorEvent*, void*) { return true; }
bool on_close_cb(int, const EmscriptenWebSocketCloseEvent* e, void* ud) {
    static_cast<NetSource*>(ud)->on_close(e->code, e->reason);
    return true;
}

std::string str_or(const Json* j, const char* fallback) {
    return j && j->is_string() ? j->as_string() : std::string(fallback);
}

int64_t int_or(const Json* j, int64_t fallback) {
    int64_t v = 0;
    return j && j->get_i64(&v) ? v : fallback;
}

// A snapshot's axes and tile (wire protocol v3, Payload axes).
caliper_live::PayloadMap payload_map(const Json& axes, const Json& tile) {
    caliper_live::PayloadMap m;
    if (axes.is_array())
        for (size_t i = 0; i < axes.size(); ++i) {
            const Json& a = axes.at(i);
            if (!a.is_object()) continue;
            caliper_live::PayloadAxis ax;
            ax.dim = str_or(a.find("dim"), "");
            ax.start = int_or(a.find("start"), 0);
            ax.step = int_or(a.find("step"), 1);
            ax.count = int_or(a.find("count"), 0);
            ax.size = int_or(a.find("size"), 0);
            m.axes.push_back(std::move(ax));
        }
    if (tile.is_object()) {
        m.tile_rows = int_or(tile.find("rows"), 0);
        m.tile_cols = int_or(tile.find("cols"), 0);
    }
    return m;
}

}  // namespace

NetSource::NetSource(Options opt, Callbacks cb) : opt_(std::move(opt)), cb_(std::move(cb)) {}

NetSource::~NetSource() {
    if (socket_ > 0) {
        if (!closed_) emscripten_websocket_close(socket_, CALIPER_NET_CLOSE_NORMAL, "");
        emscripten_websocket_delete(socket_);
    }
}

void NetSource::set_state(const std::string& s) {
    if (s == state_) return;
    state_ = s;
    if (cb_.status) cb_.status(s);
}

bool NetSource::connect() {
    if (!emscripten_websocket_is_supported()) {
        set_state("this browser has no WebSocket");
        return false;
    }
    EmscriptenWebSocketCreateAttributes a;
    emscripten_websocket_init_create_attributes(&a);   // no subprotocol
    a.url = opt_.url.c_str();
    a.protocols = nullptr;
    a.createOnMainThread = true;
    socket_ = emscripten_websocket_new(&a);
    if (socket_ <= 0) {
        set_state("could not open the WebSocket");
        return false;
    }
    emscripten_websocket_set_onopen_callback(socket_, this, on_open_cb);
    emscripten_websocket_set_onmessage_callback(socket_, this, on_message_cb);
    emscripten_websocket_set_onerror_callback(socket_, this, on_error_cb);
    emscripten_websocket_set_onclose_callback(socket_, this, on_close_cb);
    last_rx_ms_ = now_ms();
    set_state("connecting");
    return true;
}

void NetSource::send(const std::vector<uint8_t>& frame) {
    if (socket_ <= 0 || closed_) return;
    emscripten_websocket_send_binary(socket_, (void*)frame.data(), (uint32_t)frame.size());
}

template <class M>
void NetSource::send_msg(const M& m) {
    std::string err;
    if (!wire::encode(wire::Message(m), &out_, &err)) {
        std::fprintf(stderr, "[caliper-views] encode %s: %s\n", wire::type_name(M::kType), err.c_str());
        return;
    }
    send(out_);
}

void NetSource::close(const char* reason) {
    if (socket_ <= 0 || closed_) return;
    wire::Bye bye;
    bye.reason = reason ? reason : CALIPER_NET_BYE_NORMAL;
    send_msg(bye);
    emscripten_websocket_close(socket_, CALIPER_NET_CLOSE_NORMAL, "");
    closed_ = true;
    set_state("closed");
}

void NetSource::on_open() {
    wire::Hello h;
    h.role = CALIPER_NET_ROLE_VIEWER;
    h.capabilities = {CALIPER_NET_CAP_SCENES};   // the 3.2 draft: this viewer draws scenes
    Json sw = Json::object();
    sw.set("name", "caliper-views");
    sw.set("version", "0.1");
    h.software = sw;
    send_msg(h);
    set_state("waiting for the producer");
}

void NetSource::on_close(unsigned short code, const char* reason) {
    closed_ = true;
    char buf[160];
    std::snprintf(buf, sizeof buf, "disconnected (%u%s%s)", (unsigned)code, reason && reason[0] ? " " : "",
                  reason ? reason : "");
    set_state(ended_ ? "session ended" : buf);
}

NetSource::View* NetSource::by_sub(uint32_t sub) {
    for (auto& v : views_)
        if (v.sub == sub && sub != 0) return &v;
    return nullptr;
}

NetSource::View* NetSource::by_id(uint64_t id) {
    for (auto& v : views_)
        if (v.id == id) return &v;
    return nullptr;
}

void NetSource::subscribe(View& v) {
    v.sub = next_sub_++;
    v.request_id = next_request_++;
    v.subscribed = false;
    wire::Subscribe s;
    s.sub = v.sub;
    s.view = v.name;
    s.request_id = v.request_id;
    s.max_rate = opt_.max_rate;
    send_msg(s);
}

// The scene's queue passed its bound while the display was paused and was emptied: a SUBSCRIBE
// on the same sub replaces the subscription, and the sender answers with the scene's state,
// whose reset the queue waits for (live/scene_queue.h).
void NetSource::resubscribe(View& v) {
    v.request_id = next_request_++;
    wire::Subscribe s;
    s.sub = v.sub;
    s.view = v.name;
    s.request_id = v.request_id;
    s.max_rate = opt_.max_rate;
    send_msg(s);
}

void NetSource::on_message(const uint8_t* data, uint32_t n, bool text) {
    last_rx_ms_ = now_ms();
    ++messages_;
    bytes_ += n;
    if (text) {   // the protocol is binary only
        emscripten_websocket_close(socket_, CALIPER_NET_CLOSE_UNSUPPORTED_DATA, "text message");
        closed_ = true;
        set_state("closed: the server sent a text message");
        return;
    }
    // A scene message is kept for the frame's poll, so it is decoded from a copy of its
    // frame that lives as long as the message (the callback's bytes do not).
    std::shared_ptr<std::vector<uint8_t>> keep;
    uint16_t type = 0;
    if (n >= 10) std::memcpy(&type, data + 8, 2);
    if (type == CALIPER_NET_MSG_SCENE_UPDATE || type == CALIPER_NET_MSG_SCENE_CLEAR) {
        keep = std::make_shared<std::vector<uint8_t>>(data, data + n);
        data = keep->data();
    }
    wire::Message msg;
    const wire::DecodeResult r = wire::decode(data, n, &msg);
    if (r.status == wire::DecodeResult::Status::BadFrame) {
        emscripten_websocket_close(socket_, (unsigned short)r.close_code, r.reason.c_str());
        closed_ = true;
        set_state("closed: bad frame (" + r.reason + ")");
        return;
    }
    if (r.status == wire::DecodeResult::Status::BadMessage) {
        wire::Error e;
        e.code = r.error_code;
        e.message = r.message;
        e.in_reply_to = r.type;
        send_msg(e);
        // A scene message this viewer does not build is ignored, and its view says so.
        if (r.type == CALIPER_NET_MSG_SCENE_UPDATE && r.error_code == CALIPER_NET_ERR_UNSUPPORTED) {
            uint64_t sub = 0;
            const Json* s = r.frame.meta.find("sub");
            if (View* v = s && s->get_u64(&sub) ? by_sub((uint32_t)sub) : nullptr)
                if (caliper_live::scene_queue_push(v->scene, caliper_live::SceneItem{nullptr, "ignored: " + r.message}))
                    resubscribe(*v);
        }
        return;
    }

    std::visit([&](auto& m) {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, wire::Welcome>) {
            welcomed_ = true;
            if (const Json* id = m.session.is_object() ? m.session.find("id") : nullptr)
                session_id_ = str_or(id, "");
            set_state("connected");
        } else if constexpr (std::is_same_v<T, wire::Sessions>) {
            for (const auto& s : m.list) {
                if (s.identity.id == session_id_ || m.list.size() == 1) {
                    session_id_ = s.identity.id;
                    script_ = s.identity.script;
                    pid_ = (int32_t)s.identity.pid;
                    start_unix_ms_ = s.identity.start_unix_ms;
                    if (s.state == CALIPER_NET_SESSION_ENDED) {
                        ended_ = true;
                        set_state("session ended");
                    }
                }
            }
        } else if constexpr (std::is_same_v<T, wire::Views>) {
            // Keep the views that still exist with the same id; subscribe to
            // new ones this instance shows. The producer ended the
            // subscriptions of views that left or changed id.
            std::vector<View> next;
            for (const auto& vi : m.list) {
                const bool wanted = opt_.view.empty() || vi.name == opt_.view;
                if (!wanted) continue;
                auto it = std::find_if(views_.begin(), views_.end(),
                                       [&](const View& v) { return v.id == vi.id && v.name == vi.name; });
                if (it != views_.end()) {
                    next.push_back(std::move(*it));
                    continue;
                }
                View v;
                v.id = vi.id;
                v.name = vi.name;
                v.kind = (int32_t)vi.kind;
                for (size_t i = 0; i < m.list.size() && i < m.views.size(); ++i)
                    if (m.list[i].name == vi.name) v.offer = m.views.at(i);
                if (opt_.kind && v.kind != opt_.kind) {
                    v.error = "the view's kind is " + std::to_string(v.kind) + ", not " +
                              std::to_string(opt_.kind);
                } else {
                    if (v.kind == CALIPER_LIVE_KIND_SERIES) v.ring.assign(kSeriesCapacity, NAN);
                    next.push_back(std::move(v));
                    subscribe(next.back());
                    continue;
                }
                next.push_back(std::move(v));
            }
            views_ = std::move(next);
            views_changed_ = true;
            if (views_.empty() && !opt_.view.empty()) set_state("waiting for the view '" + opt_.view + "'");
            else if (connected()) set_state("connected");
        } else if constexpr (std::is_same_v<T, wire::ViewReply>) {
            View* v = by_sub(m.sub);
            if (!v || m.request_id != v->request_id) return;
            if (m.error.is_object()) {
                v->error = str_or(m.error.find("code"), "error") + ": " + str_or(m.error.find("message"), "");
                return;
            }
            v->subscribed = true;
            v->error.clear();
            v->data_view = m.data_view.dump();
            if (cb_.data_view) cb_.data_view(v->name, v->data_view);
        } else if constexpr (std::is_same_v<T, wire::Snapshot>) {
            View* v = by_sub(m.sub);
            if (!v || m.kind != v->kind) return;
            clock_.observe(m.stamp_ns, now_ns());
            Payload& p = v->incoming;
            const wire::Section& d = m.data;
            // Points travel as (N, 2) or (N, 3); the live views take (N, 3),
            // 2-D padded with z = 0 (the v2 convention).
            const bool pad = m.kind == CALIPER_LIVE_KIND_POINTS && d.shape.size() == 2 && d.shape[1] == 2;
            const size_t data_bytes = pad ? (size_t)d.shape[0] * 12 : d.size;
            const size_t aux_off = (data_bytes + 7) & ~size_t(7);
            const size_t aux_bytes = m.aux ? m.aux->size : 0;
            p.bytes.resize(aux_off + aux_bytes);   // capacity is kept between snapshots
            if (pad) {
                const float* src = (const float*)d.data;
                float* dst = (float*)p.bytes.data();
                for (int64_t i = 0; i < d.shape[0]; ++i) {
                    dst[3 * i] = src[2 * i];
                    dst[3 * i + 1] = src[2 * i + 1];
                    dst[3 * i + 2] = 0.0f;
                }
            } else {
                std::memcpy(p.bytes.data(), d.data, d.size);
            }
            if (aux_bytes) std::memcpy(p.bytes.data() + aux_off, m.aux->data, aux_bytes);
            p.stats.assign(m.stats.begin(), m.stats.end());
            caliper_live::Snapshot& s = p.meta;
            s = caliper_live::Snapshot{};
            s.kind = (uint32_t)m.kind;
            s.dtype = (int32_t)d.dtype;
            s.ndim = (int32_t)std::min<size_t>(d.shape.size(), 4);
            for (int i = 0; i < s.ndim; ++i) s.shape[i] = d.shape[i];
            if (pad) s.shape[1] = 3;
            s.aux_offset = aux_off;
            s.aux_count = m.aux ? m.aux->count() : 0;
            s.colormap = (int32_t)m.colormap;
            s.flags = (uint32_t)m.flags;
            s.vmin = (float)m.vmin;
            s.vmax = (float)m.vmax;
            s.stats_count = (int32_t)p.stats.size();
            s.points_dims = m.points_dims ? (int32_t)*m.points_dims : 0;
            s.label_count = m.label_count ? (int32_t)*m.label_count : 0;
            s.seq = m.seq;
            s.stamp_ns = m.stamp_ns;
            p.map = payload_map(m.axes, m.tile);
            p.data_view = m.data_view;
            v->fresh = true;
        } else if constexpr (std::is_same_v<T, wire::Series>) {
            View* v = by_sub(m.sub);
            if (!v || v->kind != CALIPER_LIVE_KIND_SERIES || v->ring.empty()) return;
            clock_.observe(m.stamp_ns, now_ns());
            // Gaps read as NaN, overlaps replace values: the native source's rule.
            caliper_live::series_ring_apply(v->ring.data(), kSeriesCapacity, &v->series_end, m.start,
                                            (const float*)m.values.data, (uint64_t)m.values.count());
            v->series_stamp = m.stamp_ns;
            ++v->series_seq;
        } else if constexpr (std::is_same_v<T, wire::SceneUpdate> || std::is_same_v<T, wire::SceneClear> ||
                             std::is_same_v<T, wire::SceneTransforms> || std::is_same_v<T, wire::SceneClasses>) {
            View* v = by_sub(m.sub);
            if (!v || v->kind != CALIPER_NET_KIND_SCENE) return;
            std::shared_ptr<const void> hold = keep;
            std::vector<wire::Section> sections = r.frame.sections;
            if constexpr (std::is_same_v<T, wire::SceneUpdate>) {
                // A piece of a split update is kept until the update is whole, then queued as any update.
                if (!m.split.is_null()) {
                    wire::WholeUpdate whole;
                    std::string err;
                    const auto res = v->splits.add(m, &whole, &err);
                    if (res == wire::SplitUpdates::Result::Pending) return;
                    if (res != wire::SplitUpdates::Result::Whole) {
                        if (res == wire::SplitUpdates::Result::Refused) {
                            wire::Error e;
                            e.code = CALIPER_NET_ERR_BAD_REQUEST;
                            e.message = err;
                            e.in_reply_to = r.type;
                            e.sub = m.sub;
                            e.view = m.view;
                            send_msg(e);
                        }
                        const std::string note = res == wire::SplitUpdates::Result::Refused
                                                     ? "ignored: " + err
                                                     : std::string("ignored: a piece of a split update whose first piece is not held");
                        if (caliper_live::scene_queue_push(v->scene, caliper_live::SceneItem{nullptr, note})) resubscribe(*v);
                        return;
                    }
                    m = std::move(whole.update);
                    hold = whole.keep;
                    sections = m.arrays;
                }
                // A form the scene view does not draw yet (an encoded image) is answered ERROR
                // unsupported and noted, not queued; a section it does not use (a stamp) is noted.
                const std::string why = caliper_live::scene_not_drawn(m);
                if (!why.empty()) {
                    wire::Error e;
                    e.code = CALIPER_NET_ERR_UNSUPPORTED;
                    e.message = why;
                    e.in_reply_to = r.type;
                    e.sub = m.sub;
                    e.view = m.view;
                    send_msg(e);
                    if (caliper_live::scene_queue_push(v->scene, caliper_live::SceneItem{nullptr, "ignored: " + why}))
                        resubscribe(*v);
                    return;
                }
                const std::string ignored = caliper_live::scene_ignored(m, sections);
                if (!ignored.empty() && caliper_live::scene_queue_push(v->scene, caliper_live::SceneItem{nullptr, ignored}))
                    resubscribe(*v);
            }
            caliper_live::SceneItem item;
            item.keep = hold;
            item.msg = std::move(m);
            if (caliper_live::scene_queue_push(v->scene, std::move(item))) resubscribe(*v);
        } else if constexpr (std::is_same_v<T, wire::Ping>) {
            wire::Pong pong;
            pong.nonce = m.nonce;
            send_msg(pong);
        } else if constexpr (std::is_same_v<T, wire::Bye>) {
            ended_ = m.reason == CALIPER_NET_BYE_SHUTDOWN || m.reason == CALIPER_NET_BYE_NORMAL;
            set_state(ended_ ? "session ended" : "closed: " + m.reason);
        } else if constexpr (std::is_same_v<T, wire::Error>) {
            std::fprintf(stderr, "[caliper-views] producer error %s: %s\n", m.code.c_str(), m.message.c_str());
            if (m.view) {
                for (auto& v : views_)
                    if (v.name == *m.view) v.error = m.code + ": " + m.message;
            }
        }
    }, msg);
}

void NetSource::heartbeat() {
    if (socket_ <= 0 || closed_) return;
    const double t = now_ms();
    if (t - last_rx_ms_ > CALIPER_NET_IDLE_TIMEOUT_MS) {
        emscripten_websocket_close(socket_, CALIPER_NET_CLOSE_IDLE_TIMEOUT, "idle");
        closed_ = true;
        set_state("closed: the producer went quiet");
        return;
    }
    if (welcomed_ && t - last_rx_ms_ > CALIPER_NET_PING_INTERVAL_MS &&
        t - last_ping_ms_ > CALIPER_NET_PING_INTERVAL_MS) {
        wire::Ping p;
        p.nonce = next_nonce_++;
        send_msg(p);
        last_ping_ms_ = t;
    }
}

bool NetSource::request_data_view(const std::string& json, std::string* err) {
    if (views_.size() != 1 || !views_[0].sub) {
        if (err) *err = "no single subscribed view yet";
        return false;
    }
    return request_data_view(views_[0].id, json, err);
}

uint64_t NetSource::key_of(const std::string& name) const {
    for (const auto& v : views_)
        if (v.name == name && v.sub) return v.id;
    return 0;
}

bool NetSource::send_event(uint64_t key, const std::string& type, const std::string& detail_json) {
    View* v = by_id(key);
    if (!v || !v->sub || closed_) return false;
    Json detail;
    std::string perr;
    if (!Json::parse(detail_json.data(), detail_json.size(), &detail, &perr) || !detail.is_object()) {
        std::fprintf(stderr, "[caliper-views] event detail is not a JSON object: %s\n", perr.c_str());
        return false;
    }
    wire::Event e;
    e.view = v->name;
    e.type = type;
    e.detail = std::move(detail);
    e.sub = v->sub;
    // What the viewer was showing: the snapshot's seq and data view, or for a
    // series (no snapshots) the data view its subscription resolved.
    if (v->shown.meta.host_base) {
        e.seq = v->shown.meta.seq;
        e.data_view = v->shown.data_view;
    } else if (!v->data_view.empty()) {
        Json dv;
        if (Json::parse(v->data_view.data(), v->data_view.size(), &dv, nullptr)) e.data_view = dv;
    }
    send_msg(e);
    return true;
}

bool NetSource::request_data_view(uint64_t key, const std::string& json, std::string* err) {
    View* v = by_id(key);
    if (!v || !v->sub) {
        if (err) *err = "the view is not subscribed";
        return false;
    }
    Json dv;
    std::string perr;
    if (!Json::parse(json.data(), json.size(), &dv, &perr) || !dv.is_object()) {
        if (err) *err = "the data view must be a JSON object" + (perr.empty() ? "" : " (" + perr + ")");
        return false;
    }
    wire::ViewRequest r;
    r.request_id = v->request_id = next_request_++;
    r.sub = v->sub;
    r.view = v->name;
    r.data_view = dv;
    send_msg(r);
    return true;
}

bool NetSource::sync(std::vector<caliper_live::SourceView>* out) {
    if (!views_changed_) return false;
    views_changed_ = false;
    out->clear();
    for (const auto& v : views_)
        if (v.sub) out->push_back({v.id, v.name, (uint32_t)v.kind});
    return true;
}

bool NetSource::take(uint64_t key, uint64_t after_seq, caliper_live::Snapshot* out) {
    View* v = by_id(key);
    if (!v) return false;
    if (v->kind == CALIPER_LIVE_KIND_SERIES) {
        if (v->series_seq == v->series_taken || v->series_seq <= after_seq) return false;
        v->series_taken = v->series_seq;
        caliper_live::Snapshot s;
        s.kind = CALIPER_LIVE_KIND_SERIES;
        s.dtype = CALIPER_NET_DTYPE_F32;
        s.ndim = 1;
        s.shape[0] = (int64_t)kSeriesCapacity;
        s.host_base = v->ring.data();
        s.series_total = v->series_end;
        s.seq = v->series_seq;
        s.stamp_ns = clock_.to_viewer(v->series_stamp);
        *out = s;
        return true;
    }
    if (!v->fresh || v->incoming.meta.seq <= after_seq) return false;
    std::swap(v->incoming, v->shown);   // the buffers trade places; no copy, no allocation
    v->fresh = false;
    caliper_live::Snapshot& s = v->shown.meta;
    s.host_base = v->shown.bytes.data();
    s.byte_offset = 0;
    s.stats = v->shown.stats.empty() ? nullptr : v->shown.stats.data();
    s.map = &v->shown.map;
    *out = s;
    out->stamp_ns = clock_.to_viewer(s.stamp_ns);
    return true;
}

bool NetSource::data_view_of(uint64_t key, Json* resolved, Json* offer) {
    View* v = by_id(key);
    if (!v) return false;
    if (resolved && !v->data_view.empty()) Json::parse(v->data_view.data(), v->data_view.size(), resolved, nullptr);
    if (offer) *offer = v->offer;
    return true;
}

bool NetSource::take_scene(uint64_t key, std::vector<caliper_live::SceneItem>* out) {
    View* v = by_id(key);
    if (!v || v->kind != CALIPER_NET_KIND_SCENE || v->scene.items.empty()) return false;
    for (auto& it : v->scene.items) out->push_back(std::move(it));
    v->scene.items.clear();
    return true;
}

void NetSource::stop_scene(uint64_t key) {
    View* v = by_id(key);
    if (!v || v->kind != CALIPER_NET_KIND_SCENE || v->scene.stopped) return;
    v->scene.items.clear();
    v->scene.stopped = true;   // what is still in flight is dropped
    if (v->sub) {
        wire::Unsubscribe u;
        u.sub = v->sub;
        send_msg(u);
    }
}

const caliper_live::Snapshot* NetSource::shown(const std::string& name) const {
    for (const auto& v : views_)
        if (v.name == name && v.shown.meta.host_base) return &v.shown.meta;
    return nullptr;
}

std::string NetSource::views_text() const {
    std::string t;
    for (const auto& v : views_) {
        if (!t.empty()) t += "\n";
        t += v.name + (v.subscribed ? "" : " (subscribing)");
        if (!v.error.empty()) t += ": " + v.error;
    }
    return t;
}

}  // namespace caliper_web
