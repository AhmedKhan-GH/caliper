// hub_viewer.cpp: session directory, viewer attachment, channel and
// subscription deduplication, and viewer requests, events, controls, and
// programs for the wire protocol v3 hub.
#include "net/hub_internal.h"

#include "caliper/live_net.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace caliper_net {

using namespace hub_detail;

// ---- sessions list --------------------------------------------------------------
std::vector<Session*> Hub::Impl::ordered_sessions() {
    std::vector<Session*> v;
    for (auto& kv : sessions) v.push_back(&kv.second);
    std::sort(v.begin(), v.end(), [](Session* a, Session* b) { return a->order < b->order; });
    return v;
}

wire::Sessions Hub::Impl::sessions_msg() {
    wire::Sessions m;
    Json list = Json::array();
    for (Session* s : ordered_sessions()) {
        Json e = s->identity.to_json();
        e.set("state", s->live ? CALIPER_NET_SESSION_LIVE : CALIPER_NET_SESSION_ENDED);
        e.set("views", (uint64_t)s->views.size());
        Json caps = Json::array();
        for (const auto& x : s->capabilities) caps.push(x);
        e.set("capabilities", caps);
        list.push(e);
    }
    m.sessions = list;
    return m;
}

void Hub::Impl::broadcast_sessions() {
    std::vector<uint8_t> b;
    if (!wire::encode(sessions_msg(), &b)) return;
    auto shared = std::make_shared<const Bytes>(std::move(b));
    for (auto& kv : peers)
        if (kv.second.role == Peer::Role::Viewer && !kv.second.closing && !server.send(kv.first, shared))
            overflow(kv.first);
}

// VIEWS for one viewer: the item spaces of 3.1 only to a viewer that
// announced `selection`.
// 3.2 draft: the scenes only to a viewer that announced `scenes`.
wire::Views Hub::Impl::views_msg(const Session* s, bool with_items, bool with_scenes) {
    wire::Views v;
    if (s) {
        v.session = s->id;
        v.views = with_items ? s->views : s->views_plain;
        if (!with_scenes) {
            Json out = Json::array();
            for (const Json& e : v.views.items()) {
                const Json* k = e.find("kind");
                int64_t kind = 0;
                if (!(k && k->get_i64(&kind) && kind == CALIPER_NET_KIND_SCENE)) out.push(e);
            }
            v.views = out;
        }
    } else {
        v.views = Json::array();
    }
    return v;
}

size_t Hub::Impl::live_count() const {
    size_t n = 0;
    for (const auto& kv : sessions) n += kv.second.live ? 1 : 0;
    return n;
}

// ---- viewers: attaching ------------------------------------------------------------
void Hub::Impl::attach(Peer& p, const std::string& sid) {
    if (p.attached == sid) {   // already there: confirm
        Session* s = sid.empty() ? nullptr : &sessions.at(sid);
        send(p.conn, views_msg(s, p.cap_selection, p.cap_scenes));
        if (s) send_state(p, *s);
        return;
    }
    detach_all(p);
    p.attached = sid;
    Session* s = nullptr;
    if (!sid.empty()) {
        s = &sessions.at(sid);
        s->viewers.insert(p.conn);
    }
    send(p.conn, views_msg(s, p.cap_selection, p.cap_scenes));
    if (s) send_state(p, *s);
}

// Ends every subscription of a viewer and leaves its session (no message).
void Hub::Impl::detach_all(Peer& p) {
    std::vector<uint32_t> subs;
    for (auto& kv : p.subs) subs.push_back(kv.first);
    for (uint32_t s : subs) end_vsub(p, s);
    p.pending_sub.clear();
    // The controls and selections of the session it leaves are not sent.
    p.pending_controls.reset();
    p.pending_selections.clear();
    p.pending_resources.clear();   // 3.2 draft
    stop_sending_resource(p);
    drop_held_controls(p);
    auto it = sessions.find(p.attached);
    if (it != sessions.end()) it->second.viewers.erase(p.conn);
    p.attached.clear();
}

// The server's default: a viewer that never sent ATTACH and is not attached
// to a live session follows the one live session, when there is one.
void Hub::Impl::reconsider_defaults() {
    if (live_count() != 1) return;
    std::string live;
    for (auto& kv : sessions)
        if (kv.second.live) live = kv.first;
    for (auto& kv : peers) {
        Peer& p = kv.second;
        if (p.role != Peer::Role::Viewer || p.closing || !p.follows_default || p.attached == live) continue;
        auto it = sessions.find(p.attached);
        if (it != sessions.end() && it->second.live) continue;
        attach(p, live);
    }
}

// ---- channels ----------------------------------------------------------------------
Session* Hub::Impl::session_of(const Peer& p) {
    auto it = sessions.find(p.attached);
    return it == sessions.end() ? nullptr : &it->second;
}

Channel* Hub::Impl::channel(Session& s, uint64_t id) {
    auto it = s.channels.find(id);
    return it == s.channels.end() ? nullptr : &it->second;
}

void Hub::Impl::mark_dirty(ConnId conn) {
    auto it = peers.find(conn);
    if (it != peers.end()) it->second.dirty = true;
}

VSub* Hub::Impl::vsub(const SubKey& k) {
    auto it = peers.find(k.conn);
    if (it == peers.end()) return nullptr;
    auto jt = it->second.subs.find(k.sub);
    return jt == it->second.subs.end() ? nullptr : &jt->second;
}

void Hub::Impl::set_latest(Channel& ch, SnapPtr snap) {
    if (ch.latest) snap_bytes -= ch.latest->bytes();
    ch.latest = std::move(snap);
    if (ch.latest) snap_bytes += ch.latest->bytes();
}

double Hub::Impl::want_rate(const Channel& ch) {
    double r = 0;
    for (const SubKey& k : ch.members)
        if (VSub* v = vsub(k)) r = std::max(r, v->rate);
    for (const Waiter& w : ch.waiters) r = std::max(r, w.rate);
    if (ch.keeper || ch.kind == CALIPER_NET_KIND_SERIES) r = std::max(r, kDefaultRate);
    return r > 0 ? r : kDefaultRate;
}

void Hub::Impl::send_subscribe(Session& s, Channel& ch) {
    ch.up_request = s.next_request++;
    if (s.next_request == 0) s.next_request = 1;
    s.by_up_request[ch.up_request] = ch.id;
    ch.up_rate = want_rate(ch);
    wire::Subscribe m;
    m.sub = ch.up_sub;
    m.view = ch.view;
    m.request_id = ch.up_request;
    m.max_rate = ch.up_rate;
    m.data_view = ch.requested;
    send(s.producer, m);
}

// A new producer subscription for a channel that has none. False at the
// protocol's limit of subscriptions on one connection.
bool Hub::Impl::subscribe_upstream(Session& s, Channel& ch) {
    if (!s.producer || ch.up_sub) return false;
    if (s.by_up_sub.size() >= CALIPER_NET_MAX_SUBSCRIPTIONS) return false;
    ch.up_sub = s.next_sub++;
    if (s.next_sub == 0) s.next_sub = 1;
    s.by_up_sub[ch.up_sub] = ch.id;
    ch.scene_seen = false;       // 3.2 draft: the producer sends the scene's state again
    ch.scene_forgotten = false;
    send_subscribe(s, ch);
    return true;
}

void Hub::Impl::unsubscribe_upstream(Session& s, Channel& ch) {
    if (!ch.up_sub) return;
    if (s.producer) {
        wire::Unsubscribe u;
        u.sub = ch.up_sub;
        send(s.producer, u);
    }
    s.by_up_sub.erase(ch.up_sub);
    if (ch.up_request) s.by_up_request.erase(ch.up_request);
    ch.up_sub = 0;
    ch.up_request = 0;
    ch.up_rate = 0;
}

// The producer subscription follows the largest max_rate of the channel's
// viewers (series only ever go up, since a new SUBSCRIBE resends backfill).
void Hub::Impl::adjust_rate(Session& s, Channel& ch) {
    if (!ch.up_sub || ch.up_request || !s.producer) return;
    const double want = want_rate(ch);
    const bool series = ch.kind == CALIPER_NET_KIND_SERIES;
    if (series ? want > ch.up_rate + 1e-9 : std::fabs(want - ch.up_rate) > 1e-9) send_subscribe(s, ch);
}

// After a change of membership: a channel nobody watches gives up its
// producer subscription and stays idle with its last snapshot.
void Hub::Impl::settle(Session& s, uint64_t id) {
    Channel* ch = channel(s, id);
    if (!ch) return;
    if (ch->idle()) {
        unsubscribe_upstream(s, *ch);
        ch->used_ms = monotonic_ms();
        const bool keep = ch->resolved_ok && (ch->latest || s.series.count(ch->view) || !ch->scene_held.empty());
        if (!keep) {
            destroy_channel(s, id);
            return;
        }
        trim_idle(s);
    } else {
        adjust_rate(s, *ch);
    }
}

// At most max_idle_channels idle channels per session, the least recently used go.
void Hub::Impl::trim_idle(Session& s) {
    for (;;) {
        size_t n = 0;
        Channel* lru = nullptr;
        for (auto& kv : s.channels) {
            Channel& ch = kv.second;
            if (!ch.idle()) continue;
            ++n;
            if (!lru || ch.used_ms < lru->used_ms) lru = &ch;
        }
        if (n <= opt.max_idle_channels || !lru) return;
        destroy_channel(s, lru->id);
    }
}

// Removes a channel: its viewer subscriptions end without a message (the
// VIEWS that caused it tells the viewers), its waiters are answered.
void Hub::Impl::destroy_channel(Session& s, uint64_t id, const char* code, const std::string& text) {
    auto it = s.channels.find(id);
    if (it == s.channels.end()) return;
    Channel ch = std::move(it->second);
    s.channels.erase(it);
    if (ch.up_sub) {
        if (s.producer) {
            wire::Unsubscribe u;
            u.sub = ch.up_sub;
            send(s.producer, u);
        }
        s.by_up_sub.erase(ch.up_sub);
    }
    if (ch.up_request) s.by_up_request.erase(ch.up_request);
    if (ch.latest) snap_bytes -= ch.latest->bytes();
    release_scene(ch);
    for (const SubKey& k : ch.members) {
        auto pt = peers.find(k.conn);
        if (pt == peers.end()) continue;
        auto st = pt->second.subs.find(k.sub);
        if (st != pt->second.subs.end() && st->second.channel == id) pt->second.subs.erase(st);
    }
    for (const Waiter& w : ch.waiters) fail_waiter(s, ch.view, w, code, text);
}

bool Hub::Impl::waiter_valid(Session& s, const Waiter& w, Peer** out) {
    auto pt = peers.find(w.conn);
    if (pt == peers.end() || pt->second.closing || pt->second.attached != s.id) return false;
    Peer& p = pt->second;
    if (w.view_request) {
        auto st = p.subs.find(w.sub);
        if (st == p.subs.end() || st->second.ticket != w.ticket) return false;
    } else {
        auto pi = p.pending_sub.find(w.sub);
        if (pi == p.pending_sub.end() || pi->second != w.ticket) return false;
    }
    *out = &p;
    return true;
}

void Hub::Impl::fail_waiter(Session& s, const std::string& view, const Waiter& w, const std::string& code,
                            const std::string& text) {
    Peer* p = nullptr;
    if (!waiter_valid(s, w, &p)) return;
    if (!w.view_request) p->pending_sub.erase(w.sub);
    reply_error(p->conn, w.request_id, w.sub, view, code, text);
}

// Puts a viewer subscription on a channel.
void Hub::Impl::join(Session& s, Peer& p, VSub& v, Channel& ch) {
    v.channel = ch.id;
    v.series = ch.kind == CALIPER_NET_KIND_SERIES;
    v.pending.reset();
    ch.members.insert({p.conn, v.sub});
    ch.used_ms = monotonic_ms();
    if (ch.kind == CALIPER_NET_KIND_SCENE) {   // 3.2 draft
        v.scene_pieces = p.cap_scene_pieces;
        // The hub forgot the scene's state: it subscribes again for a fresh one, which
        // reaches this viewer as the producer's own state.
        if (ch.scene_forgotten && ch.up_sub && !ch.up_request && s.producer) {
            ch.scene_forgotten = false;
            send_subscribe(s, ch);
        }
        join_scene(v, ch);
        p.dirty = true;
        return;
    }
    if (v.series) {
        v.backfill = backfill_of(ch.resolved);
        v.next = 0;
        v.sent_any = false;
    } else {
        v.pending = ch.latest;
        ++v.pending_n;
    }
    p.dirty = true;
    (void)s;
}

void Hub::Impl::leave(Session& s, Peer& p, VSub& v) {
    v.pending.reset();
    v.scene_first.clear();
    v.scene_pending.clear();
    v.transforms_pending.clear();
    Channel* ch = channel(s, v.channel);
    v.channel = 0;
    if (!ch) return;
    ch->members.erase({p.conn, v.sub});
    settle(s, ch->id);
}

void Hub::Impl::end_vsub(Peer& p, uint32_t sub) {
    auto it = p.subs.find(sub);
    if (it == p.subs.end()) return;
    if (Session* s = session_of(p)) leave(*s, p, it->second);
    if (p.sending_scene_sub == sub) p.sending_scene.clear();
    p.subs.erase(sub);
    if (p.held_controls) p.dirty = true;   // it may have waited for this one
}

// The viewer's request is answered by a channel that has resolved.
void Hub::Impl::serve_waiter(Session& s, Channel& ch, const Waiter& w) {
    Peer* p = nullptr;
    if (!waiter_valid(s, w, &p)) return;
    if (!w.view_request) {
        p->pending_sub.erase(w.sub);
        VSub v;
        v.sub = w.sub;
        v.view = ch.view;
        v.ticket = w.ticket;
        v.rate = w.rate;
        reply_ok(p->conn, w.request_id, w.sub, ch.view, ch.resolved);
        VSub& in = p->subs[w.sub] = v;
        join(s, *p, in, ch);
    } else {
        VSub& v = p->subs.at(w.sub);
        reply_ok(p->conn, w.request_id, w.sub, ch.view, ch.resolved);
        if (v.channel != ch.id) {
            const uint64_t old = v.channel;
            Channel* oc = channel(s, old);
            if (oc) oc->members.erase({p->conn, v.sub});
            const uint64_t keep_next = v.next;
            const bool was_sent = v.sent_any;
            join(s, *p, v, ch);
            if (v.series) {   // the same series: carry on from where it was
                v.next = keep_next;
                v.sent_any = was_sent;
            }
            if (oc) settle(s, old);
        }
    }
    adjust_rate(s, ch);
}

// What an ended session can still show for (view, data view): the channel
// that resolved this request, else the most recently used one of the view
// that retained something.
Channel* Hub::Impl::ended_lookup(Session& s, const std::string& view, const std::string& key, uint64_t exclude) {
    auto vi = s.view_index.find(view);
    if (vi == s.view_index.end()) return nullptr;
    Channel* best = nullptr;
    for (auto& kv : s.channels) {
        Channel& ch = kv.second;
        if (ch.id == exclude || ch.view != view || ch.view_id != vi->second.id || !ch.resolved_ok) continue;
        const bool has = ch.latest || s.series.count(view) || !ch.scene_held.empty();
        if (!has) continue;
        if (ch.aliases.count(key) || ch.resolved_key == key) return &ch;
        if (!best || ch.used_ms > best->used_ms) best = &ch;
    }
    return best;
}

// A SUBSCRIBE or VIEW_REQUEST of a viewer for (view, data view).
void Hub::Impl::request(Session& s, const std::string& view, const Json& requested, const Waiter& w) {
    auto vi = s.view_index.find(view);
    if (vi == s.view_index.end()) {
        fail_waiter(s, view, w, CALIPER_NET_ERR_UNKNOWN_VIEW, "no view '" + view + "' in session " + s.id);
        return;
    }
    const std::string key = canon(requested);
    if (!s.live) {
        Channel* ch = ended_lookup(s, view, key);
        if (ch) serve_waiter(s, *ch, w);
        else
            fail_waiter(s, view, w, CALIPER_NET_ERR_UNKNOWN_VIEW,
                        "session " + s.id + " has ended and the hub kept nothing of view '" + view + "'");
        return;
    }
    Channel* found = nullptr;
    for (auto& kv : s.channels) {
        Channel& ch = kv.second;
        if (ch.view != view || ch.view_id != vi->second.id) continue;
        if (ch.aliases.count(key) || (ch.resolved_ok && ch.resolved_key == key) ||
            (!ch.resolved_ok && canon(ch.requested) == key)) {
            found = &ch;
            if (ch.up_sub) break;   // prefer one that is subscribed
        }
    }
    if (found) {
        if (found->up_sub && !found->up_request && found->resolved_ok) {
            serve_waiter(s, *found, w);
            return;
        }
        found->waiters.push_back(w);
        if (!found->up_sub && !subscribe_upstream(s, *found)) {
            found->waiters.pop_back();
            fail_waiter(s, view, w, CALIPER_NET_ERR_LIMIT,
                        "the hub holds " + std::to_string(CALIPER_NET_MAX_SUBSCRIPTIONS) +
                            " subscriptions at this producer");
            settle(s, found->id);
        }
        return;
    }
    if (s.by_up_sub.size() >= CALIPER_NET_MAX_SUBSCRIPTIONS) {
        fail_waiter(s, view, w, CALIPER_NET_ERR_LIMIT,
                    "the hub holds " + std::to_string(CALIPER_NET_MAX_SUBSCRIPTIONS) +
                        " subscriptions at this producer");
        return;
    }
    Channel ch;
    ch.id = next_channel++;
    ch.view = view;
    ch.view_id = vi->second.id;
    ch.kind = vi->second.kind;
    ch.requested = requested.is_object() ? requested : Json::object();
    ch.used_ms = monotonic_ms();
    ch.waiters.push_back(w);
    Channel& in = s.channels[ch.id] = std::move(ch);
    subscribe_upstream(s, in);
}

// ---- viewers: messages ---------------------------------------------------------------
void Hub::Impl::viewer_subscribe(Peer& p, const wire::Subscribe& m) {
    Session* s = session_of(p);
    if (!s) {
        reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_NOT_ATTACHED,
                    "attach to a session first");
        return;
    }
    const bool known = p.subs.count(m.sub) || p.pending_sub.count(m.sub);
    if (!known && p.subs.size() + p.pending_sub.size() >= CALIPER_NET_MAX_SUBSCRIPTIONS) {
        reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_LIMIT,
                    "at most " + std::to_string(CALIPER_NET_MAX_SUBSCRIPTIONS) + " subscriptions per connection");
        return;
    }
    if (auto vi = s->view_index.find(m.view);   // 3.2 draft: no scene for a viewer without `scenes`
        vi != s->view_index.end() && vi->second.kind == CALIPER_NET_KIND_SCENE && !p.cap_scenes) {
        reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_UNKNOWN_VIEW,
                    "no view '" + m.view + "' in session " + s->id);
        return;
    }
    end_vsub(p, m.sub);
    Waiter w;
    w.conn = p.conn;
    w.sub = m.sub;
    w.request_id = m.request_id;
    w.ticket = ++p.tickets;
    w.rate = clamp_rate(m.max_rate);
    w.requested = m.data_view.is_object() ? m.data_view : Json::object();
    p.pending_sub[m.sub] = w.ticket;
    request(*s, m.view, w.requested, w);
}

void Hub::Impl::viewer_view_request(Peer& p, const wire::ViewRequest& m) {
    Session* s = session_of(p);
    auto it = p.subs.find(m.sub);
    if (!s || it == p.subs.end()) {
        reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_UNKNOWN_SUBSCRIPTION,
                    "no subscription " + std::to_string(m.sub));
        return;
    }
    if (it->second.view != m.view) {
        reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_BAD_REQUEST,
                    "subscription " + std::to_string(m.sub) + " is to view '" + it->second.view + "'");
        return;
    }
    Waiter w;
    w.conn = p.conn;
    w.sub = m.sub;
    w.request_id = m.request_id;
    w.ticket = it->second.ticket;
    w.view_request = true;
    w.rate = it->second.rate;
    w.requested = m.data_view;
    request(*s, m.view, m.data_view, w);
}

void Hub::Impl::viewer_event(Peer& p, const wire::DecodeResult& r, const wire::Event& e) {
    Session* s = session_of(p);
    if (!s || !s->live || !s->producer) {
        ++c.events_dropped;
        return;
    }
    if (!(s->events_cap || s->selection_cap) || server.queued_bytes(s->producer) > kEventQueueLimit) {
        ++c.events_dropped;
        return;
    }
    Json meta = r.frame.meta;
    meta.erase("sections");
    meta.erase("sub");
    // 3.1: a selection's item space and ids only to a producer that keeps
    // linked selections; it reads the rest of the event as before.
    if (!s->selection_cap) {
        if (const Json* d = meta.find("detail"); d && d->is_object() && (d->has("space") || d->has("items"))) {
            Json detail = *d;
            detail.erase("space");
            detail.erase("items");
            meta.set("detail", detail);
        }
    }
    if (e.sub) {
        auto it = p.subs.find(*e.sub);
        if (it != p.subs.end())
            if (Channel* ch = channel(*s, it->second.channel); ch && ch->up_sub) meta.set("sub", ch->up_sub);
    }
    std::vector<uint8_t> b;
    if (!wire::encode_frame(CALIPER_NET_MSG_EVENT, meta, {}, &b)) {
        ++c.events_dropped;
        return;
    }
    send_bytes(s->producer, std::move(b));
    ++c.events_forwarded;
}

// ---- 3.1: controls ----------------------------------------------------------------
// A viewer's CONTROL: forwarded to the producer of its session only when
// the hub allows control, else answered here, as "Controls through a hub"
// says. Selecting is not a control: select events go on as before.
void Hub::Impl::viewer_control(Peer& p, const wire::Control& ctl) {
    if (!p.cap_controls) {
        // CONTROL_REPLY goes only to a peer that announced `controls`.
        wire::Error e;
        e.code = CALIPER_NET_ERR_BAD_REQUEST;
        e.message = "a viewer announces the capability 'controls' before it sends CONTROL";
        e.in_reply_to = CALIPER_NET_MSG_CONTROL;
        e.request_id = ctl.request_id;
        send(p.conn, e);
        return;
    }
    auto refuse = [&](const char* code, const std::string& text) {
        ++c.controls_refused;
        control_error(p, ctl.request_id, ctl.op, ctl.name, code, text);
    };
    const auto* method = ctl.interpretation.find("method");
    const bool read = ctl.op == CALIPER_NET_OP_INTERPRET && method &&
        (method->as_string().rfind("feature.",0) == 0 || method->as_string() == "trace.read" || method->as_string() == "trace.status");
    if (!opt.allow_control && !read) {
        refuse(CALIPER_NET_ERR_FORBIDDEN, "this hub does not allow control; start it with --allow-control");
        return;
    }
    if (p.watch_only && !read) {
        refuse(CALIPER_NET_ERR_FORBIDDEN, "this connection's token may watch the stream, not control a run");
        return;
    }
    Session* s = session_of(p);
    if (!s) {
        refuse(CALIPER_NET_ERR_NOT_ATTACHED, "attach to a session first");
        return;
    }
    if (!s->live || !s->producer) {
        refuse(CALIPER_NET_ERR_UNSUPPORTED, "session " + s->id + " has ended");
        return;
    }
    if (ctl.op == CALIPER_NET_OP_INTERPRET && !(p.cap_interpretation && s->interpretation_cap)) {
        refuse("interpret.unsupported", "Both peers must announce interpretation.");
        return;
    }
    if (!s->controls_cap) {
        refuse(CALIPER_NET_ERR_UNSUPPORTED, "the producer of session " + s->id + " does not take controls");
        return;
    }
    if (ctl.op == CALIPER_NET_OP_SET && !(p.cap_text_params && s->text_params_cap)) {
        const bool text_parameter = ctl.name && s->text_params.count(*ctl.name);
        if (text_parameter || (ctl.value.is_string() &&
            (ctl.value.as_string().empty() || ctl.value.as_string().size() > CALIPER_NET_NAME_MAX))) {
            refuse(CALIPER_NET_ERR_UNSUPPORTED, "Both peers must announce text_params for text parameters.");
            return;
        }
    }
    if (ctl.op == "intervene" && !(p.cap_interventions && s->interventions_cap)) {
        refuse(CALIPER_NET_ERR_UNSUPPORTED, p.cap_interventions
                   ? "the producer of session " + s->id + " does not take interventions"
                   : "a viewer announces the capability 'interventions' before it sends intervene");
        return;
    }
    if (p.controls_waiting >= kMaxControlsWaiting || s->routes.size() >= kMaxRoutes) {
        refuse(CALIPER_NET_ERR_LIMIT, "too many controls wait for the producer's answer");
        return;
    }
    // The hub's own request_id at the producer: viewers' ids may collide.
    uint32_t up = 0;
    do {
        up = s->next_control++;
    } while (up == 0 || s->routes.count(up));
    ControlRoute route;
    route.viewer = p.conn;
    route.request_id = ctl.request_id;
    route.op = ctl.op;
    route.name = ctl.name;
    route.deadline_ms = monotonic_ms() + kReplyWaitMs;
    s->routes[up] = route;
    ++p.controls_waiting;
    wire::Control fwd = ctl;   // only the keys of its op; never `args`
    fwd.request_id = up;
    send(s->producer, fwd);
    ++c.controls_forwarded;
}

// 3.2 draft: a viewer's PROGRAM, forwarded to the producer of its session
// under a request_id of the hub's, when the hub allows control. Every
// refusal is a non-fatal ERROR in reply to the PROGRAM.
void Hub::Impl::viewer_program(Peer& p, const wire::Program& m) {
    auto refuse = [&](const char* code, const std::string& text) {
        ++c.programs_refused;
        wire::Error e;
        e.code = code;
        e.message = text.substr(0, CALIPER_NET_TEXT_MAX);
        e.in_reply_to = CALIPER_NET_MSG_PROGRAM;
        e.request_id = m.request_id;
        send(p.conn, e);
    };
    if (!p.cap_programs) {
        refuse(CALIPER_NET_ERR_UNSUPPORTED, "a viewer announces the capability 'programs' before it sends PROGRAM");
        return;
    }
    if (!opt.allow_control) {
        refuse(CALIPER_NET_ERR_FORBIDDEN, "this hub does not allow control; start it with --allow-control");
        return;
    }
    if (p.watch_only) {
        refuse(CALIPER_NET_ERR_FORBIDDEN, "this connection's token may watch the stream, not control a run");
        return;
    }
    Session* s = session_of(p);
    if (!s) {
        refuse(CALIPER_NET_ERR_NOT_ATTACHED, "attach to a session first");
        return;
    }
    if (!s->live || !s->producer) {
        refuse(CALIPER_NET_ERR_UNSUPPORTED, "session " + s->id + " has ended");
        return;
    }
    if (!s->programs_cap) {
        refuse(CALIPER_NET_ERR_UNSUPPORTED, "the producer of session " + s->id + " does not take programs");
        return;
    }
    if (!m.piece.is_null() && !(p.cap_resources && s->resources_cap)) {   // an instance of a piece
        refuse(CALIPER_NET_ERR_UNSUPPORTED, p.cap_resources ? "the producer of session " + s->id +
                                                                  " does not take pieces"
                                                            : std::string("a viewer announces the capability "
                                                                          "'resources' before it sends a piece"));
        return;
    }
    if (s->program_routes.size() >= kMaxRoutes) {
        refuse(CALIPER_NET_ERR_LIMIT, "too many programs wait for the producer's answer");
        return;
    }
    uint32_t up = 0;
    do {
        up = s->next_program++;
    } while (up == 0 || s->program_routes.count(up));
    ProgramRoute route;
    route.viewer = p.conn;
    route.request_id = m.request_id;
    route.deadline_ms = monotonic_ms() + kReplyWaitMs;
    s->program_routes[up] = route;
    wire::Program fwd = m;
    fwd.request_id = up;
    send(s->producer, fwd);
    ++c.programs_forwarded;
}

// A non-fatal ERROR in reply to the PROGRAM of a route (3.2 draft).
void Hub::Impl::answer_program_route(const ProgramRoute& r, const char* code, const std::string& text) {
    auto pt = peers.find(r.viewer);
    if (pt == peers.end() || pt->second.closing) return;
    wire::Error e;
    e.code = code;
    e.message = text.substr(0, CALIPER_NET_TEXT_MAX);
    e.in_reply_to = CALIPER_NET_MSG_PROGRAM;
    e.request_id = r.request_id;
    send(pt->second.conn, e);
}

// A CONTROL_REPLY with an error to the viewer of a route (3.1).
void Hub::Impl::answer_route(const ControlRoute& r, const char* code, const std::string& text) {
    auto pt = peers.find(r.viewer);
    if (pt == peers.end()) return;
    if (pt->second.controls_waiting) --pt->second.controls_waiting;
    if (pt->second.closing) return;
    control_error(pt->second, r.request_id, r.op, r.name, code, text);
}

void Hub::Impl::control_error(Peer& p, uint32_t request_id, const std::string& op,
                              const std::optional<std::string>& name, const char* code, const std::string& text) {
    wire::ControlReply r;
    r.request_id = request_id;
    r.op = op;
    r.status = CALIPER_NET_STATUS_ERROR;
    r.name = name;
    Json e = Json::object();
    e.set("code", code);
    e.set("message", text.substr(0, CALIPER_NET_TEXT_MAX));
    r.error = e;
    send(p.conn, r);
}

void Hub::Impl::on_viewer_message(Peer& p, const wire::DecodeResult& r, wire::Message& m) {
    if (auto* x = std::get_if<wire::Ping>(&m)) {
        send(p.conn, wire::Pong{x->nonce});
    } else if (std::holds_alternative<wire::Pong>(m)) {
    } else if (auto* a = std::get_if<wire::Attach>(&m)) {
        p.follows_default = false;
        if (a->session && !sessions.count(*a->session)) {
            wire::Error e;
            e.code = CALIPER_NET_ERR_UNKNOWN_SESSION;
            e.message = "no session '" + *a->session + "'";
            e.in_reply_to = CALIPER_NET_MSG_ATTACH;
            e.session = *a->session;
            send(p.conn, e);
            return;
        }
        attach(p, a->session ? *a->session : std::string());
    } else if (auto* s = std::get_if<wire::Subscribe>(&m)) {
        viewer_subscribe(p, *s);
    } else if (auto* u = std::get_if<wire::Unsubscribe>(&m)) {
        end_vsub(p, u->sub);
        p.pending_sub.erase(u->sub);
    } else if (auto* q = std::get_if<wire::ViewRequest>(&m)) {
        viewer_view_request(p, *q);
    } else if (auto* e = std::get_if<wire::Event>(&m)) {
        viewer_event(p, r, *e);
    } else if (auto* ctl = std::get_if<wire::Control>(&m)) {
        viewer_control(p, *ctl);
    } else if (auto* pg = std::get_if<wire::Program>(&m)) {
        viewer_program(p, *pg);   // 3.2 draft
    } else if (std::holds_alternative<wire::Bye>(m)) {
        close(p.conn, CALIPER_NET_CLOSE_NORMAL, "bye");
    } else if (auto* er = std::get_if<wire::Error>(&m)) {
        if (er->fatal) close(p.conn, CALIPER_NET_CLOSE_NORMAL, "peer error");
    } else if (std::holds_alternative<wire::Unknown>(m)) {
    } else {
        fatal(p.conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST,
              std::string("a viewer does not send ") + wire::type_name(r.type));
    }
}

}  // namespace caliper_net
