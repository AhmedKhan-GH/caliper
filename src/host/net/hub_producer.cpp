// hub_producer.cpp: producer sessions, resume/end/evict/retention, budget
// enforcement, and incoming producer messages for the wire protocol v3 hub.
#include "net/hub_internal.h"

#include "caliper/live_net.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace caliper_net {

using namespace hub_detail;

// ---- producers ---------------------------------------------------------------------
bool Hub::Impl::valid_id_char(char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '.' ||
           ch == '_' || ch == '-';
}

void Hub::Impl::on_producer_hello(Peer& p, const wire::Hello& h) {
    // A connection whose token may only watch does not take the producer
    // role, as caliper.serve() refuses producers (HubOptions::stream_access).
    if (p.watch_only) {
        fatal(p.conn, CALIPER_NET_CLOSE_ROLE_REFUSED, CALIPER_NET_ERR_UNSUPPORTED,
              "this connection's token may watch the stream, not publish to it");
        return;
    }
    wire::Identity id;
    if (!wire::Identity::parse(h.session, &id)) {
        fatal(p.conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST, "a producer's hello needs 'session'");
        return;
    }
    const std::string proposed = id.id;
    // Resume: the same process (id, pid, start time) on a new connection.
    Session* s = nullptr;
    for (auto& kv : sessions) {
        Session& x = kv.second;
        if ((x.id == proposed || x.proposed == proposed) && x.identity.pid == id.pid &&
            x.identity.start_unix_ms == id.start_unix_ms)
            s = &x;
    }
    if (s) {
        const bool was_live = s->live;
        if (s->producer) {
            auto old = peers.find(s->producer);
            if (old != peers.end()) {
                old->second.role = Peer::Role::Retired;
                bye(old->first, CALIPER_NET_BYE_REPLACED, CALIPER_NET_CLOSE_NORMAL,
                    "the producer resumed its session on a new connection");
            }
        }
        if (!was_live && live_count() >= opt.max_producers) {
            fatal(p.conn, CALIPER_NET_CLOSE_SERVER_FULL, CALIPER_NET_ERR_LIMIT,
                  "the hub holds " + std::to_string(opt.max_producers) + " live sessions");
            return;
        }
        drop_upstream(*s);
        s->producer = p.conn;
        s->live = true;
        s->ended_ms = 0;
        s->resubscribe = true;
        s->identity.script = id.script;
        set_producer_caps(*s, h.capabilities);
        forget_programs(*s);   // 3.2 draft: the producer relays its draw halves again
        p.role = Peer::Role::Producer;
        p.session = s->id;
        welcome_producer(p, *s);
        log("session " + s->id + " resumed (" + s->identity.script + ", pid " + std::to_string(id.pid) + ")");
        if (!was_live) {
            broadcast_sessions();
            reconsider_defaults();
        }
        return;
    }
    if (live_count() >= opt.max_producers) {
        fatal(p.conn, CALIPER_NET_CLOSE_SERVER_FULL, CALIPER_NET_ERR_LIMIT,
              "the hub holds " + std::to_string(opt.max_producers) + " live sessions");
        return;
    }
    // The id: kept unless a live session of another process holds it.
    std::string chosen = proposed;
    for (int n = 2; sessions.count(chosen) && sessions.at(chosen).live; ++n) {
        const std::string suffix = "-" + std::to_string(n);
        chosen = proposed.substr(0, CALIPER_NET_SESSION_ID_MAX - suffix.size()) + suffix;
    }
    // An ended session of another process under that id gives way: its
    // viewers move to the new session, as after a notebook kernel restart.
    std::vector<ConnId> movers;
    if (auto it = sessions.find(chosen); it != sessions.end()) {
        movers.assign(it->second.viewers.begin(), it->second.viewers.end());
        remove_session(chosen, /*reattach=*/false);
    }
    Session ns;
    ns.id = chosen;
    ns.proposed = proposed;
    ns.identity = id;
    ns.identity.id = chosen;
    set_producer_caps(ns, h.capabilities);
    ns.producer = p.conn;
    ns.order = next_order++;
    Session& s2 = sessions[chosen] = std::move(ns);
    p.role = Peer::Role::Producer;
    p.session = chosen;
    welcome_producer(p, s2);
    log("session " + chosen + " started (" + id.script + ", pid " + std::to_string(id.pid) + ")");
    broadcast_sessions();
    for (ConnId c2 : movers) {
        auto pt = peers.find(c2);
        if (pt != peers.end() && pt->second.role == Peer::Role::Viewer && !pt->second.closing)
            attach(pt->second, chosen);
    }
    reconsider_defaults();
}

void Hub::Impl::set_producer_caps(Session& s, const std::vector<std::string>& caps) {
    s.capabilities = caps;
    s.controls_cap = has_cap(caps, CALIPER_NET_CAP_CONTROLS);
    s.selection_cap = has_cap(caps, CALIPER_NET_CAP_SELECTION);
    s.events_cap = has_cap(caps, CALIPER_NET_CAP_EVENTS);
    s.programs_cap = has_cap(caps, CALIPER_NET_CAP_PROGRAMS);
    s.interventions_cap = has_cap(caps, CALIPER_NET_CAP_INTERVENTIONS);
    s.interpretation_cap = has_cap(caps, CALIPER_NET_CAP_INTERPRETATION);
    s.text_params_cap = has_cap(caps, CALIPER_NET_CAP_TEXT_PARAMS);
    s.resources_cap = has_cap(caps, CALIPER_NET_CAP_RESOURCES);
    s.resource_parts = wire::SplitResources(false);   // a new producer connection starts no version
    s.resource_building.clear();
    s.scene_versions.clear();
    s.scene_parts = wire::SplitResources(false);
    s.scene_building.clear();
}

void Hub::Impl::welcome_producer(Peer& p, Session& s) {
    wire::Welcome w = hub_welcome(s.controls_cap);
    w.session = s.identity.to_json();
    send(p.conn, w);
}

// The producer connection is gone or replaced: no producer subscription remains.
// The producer connection is gone or replaced: no producer subscription
// remains, and the CONTROL requests that waited for it are answered.
void Hub::Impl::drop_upstream(Session& s) {
    for (auto& kv : s.channels) {
        kv.second.up_sub = 0;
        kv.second.up_request = 0;
        kv.second.up_rate = 0;
    }
    s.by_up_sub.clear();
    s.by_up_request.clear();
    s.producer = 0;
    std::map<uint32_t, ControlRoute> routes;
    routes.swap(s.routes);
    for (auto& kv : routes)
        answer_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED,
                     "the session's producer disconnected before it answered");
    std::map<uint32_t, ProgramRoute> programs;
    programs.swap(s.program_routes);
    for (auto& kv : programs)
        answer_program_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED,
                             "the session's producer disconnected before it answered");
}

void Hub::Impl::end_session(Session& s) {
    drop_upstream(s);
    s.live = false;
    s.ended_ms = monotonic_ms();
    s.resubscribe = true;
    // Requests that waited for the producer get what the session kept.
    std::vector<uint64_t> ids;
    for (auto& kv : s.channels) ids.push_back(kv.first);
    for (uint64_t id : ids) {
        Channel* ch = channel(s, id);
        if (!ch || ch->waiters.empty()) continue;
        std::vector<Waiter> ws;
        ws.swap(ch->waiters);
        const std::string view = ch->view;
        for (const Waiter& w : ws) {
            Channel* alt = ended_lookup(s, view, canon(w.requested), ch->resolved_ok ? 0 : id);
            if (alt) serve_waiter(s, *alt, w);
            else
                fail_waiter(s, view, w, CALIPER_NET_ERR_UNKNOWN_VIEW,
                            "session " + s.id + " ended before its producer answered");
        }
    }
    for (uint64_t id : ids) {
        Channel* ch = channel(s, id);
        if (ch && ch->members.empty() && ch->waiters.empty() && !ch->resolved_ok) destroy_channel(s, id);
    }
    log("session " + s.id + " ended");
    broadcast_sessions();
    reconsider_defaults();
    while (ended_count() > opt.max_ended_sessions) evict_oldest_ended();
}

size_t Hub::Impl::ended_count() const { return sessions.size() - live_count(); }

void Hub::Impl::evict_oldest_ended() {
    std::string oldest;
    uint64_t t = 0;
    for (auto& kv : sessions)
        if (!kv.second.live && (oldest.empty() || kv.second.ended_ms < t)) {
            oldest = kv.first;
            t = kv.second.ended_ms;
        }
    if (oldest.empty()) return;
    ++c.sessions_evicted;
    remove_session(oldest, /*reattach=*/true);
    log("session " + oldest + " forgotten");
    broadcast_sessions();
}

// Forgets a session. Its viewers are detached; with `reattach` they follow
// the default or hear that they are attached to nothing.
void Hub::Impl::remove_session(const std::string& id, bool reattach) {
    auto it = sessions.find(id);
    if (it == sessions.end()) return;
    Session& s = it->second;
    std::vector<ConnId> viewers(s.viewers.begin(), s.viewers.end());
    for (ConnId v : viewers) {
        auto pt = peers.find(v);
        if (pt != peers.end()) detach_all(pt->second);
    }
    std::vector<uint64_t> ids;
    for (auto& kv : s.channels) ids.push_back(kv.first);
    for (uint64_t cid : ids) destroy_channel(s, cid);
    for (auto& kv : s.series) series_bytes -= kv.second.v.size() * sizeof(float);
    forget_meta(s);
    for (auto& kv : s.routes) answer_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED, "the session is gone");
    s.routes.clear();
    for (auto& kv : s.program_routes) answer_program_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED, "the session is gone");
    s.program_routes.clear();
    if (s.producer) {
        auto pt = peers.find(s.producer);
        if (pt != peers.end()) pt->second.role = Peer::Role::Retired;
    }
    sessions.erase(it);
    if (!reattach) return;
    for (ConnId v : viewers) {
        auto pt = peers.find(v);
        if (pt == peers.end() || pt->second.closing) continue;
        send(v, views_msg(nullptr, pt->second.cap_selection, pt->second.cap_scenes));
    }
    reconsider_defaults();
}

// VIEWS without the item spaces of 3.1 (`items` on a view and on a dim),
// for a viewer that did not announce `selection`.
Json Hub::Impl::strip_items(const Json& views) {
    Json out = Json::array();
    for (const Json& v : views.items()) {
        Json e = v;
        e.erase("items");
        if (const Json* srcs = v.find("sources"); srcs && srcs->is_array()) {
            Json ns = Json::array();
            for (const Json& src : srcs->items()) {
                Json sc = src;
                if (const Json* dims = src.find("dims"); dims && dims->is_array()) {
                    Json nd = Json::array();
                    for (const Json& d : dims->items()) {
                        Json dc = d;
                        dc.erase("items");
                        nd.push(dc);
                    }
                    sc.set("dims", nd);
                }
                ns.push(sc);
            }
            e.set("sources", ns);
        }
        out.push(e);
    }
    return out;
}

void Hub::Impl::forget_meta(Session& s) {
    if (s.controls) meta_bytes -= s.controls->size();
    s.controls.reset();
    s.text_params.clear();
    if (s.controls_plain) meta_bytes -= s.controls_plain->size();
    s.controls_plain.reset();
    if (s.program_state) meta_bytes -= s.program_state->size();
    s.program_state.reset();
    forget_programs(s);   // 3.2 draft
    for (auto& kv : s.selections)
        if (kv.second.frame) meta_bytes -= kv.second.frame->size();
    s.selections.clear();
}

// Forgets what the session keeps of one program: its draw half and its resources' newest
// versions; and drops the versions of them that wait at each viewer, which belong to the
// program's earlier install.
void Hub::Impl::forget_program(Session& s, const std::string& id) {
    if (auto it = s.draw_halves.find(id); it != s.draw_halves.end()) {
        meta_bytes -= it->second.frame->size();
        s.draw_halves.erase(it);
    }
    const std::string prefix = id + "/";
    for (auto it = s.kept_resources.lower_bound(prefix);
         it != s.kept_resources.end() && it->first.compare(0, prefix.size(), prefix) == 0;) {
        resource_bytes -= version_bytes(it->second);
        it = s.kept_resources.erase(it);
    }
    for (ConnId v : s.viewers) {
        auto pt = peers.find(v);
        if (pt == peers.end()) continue;
        auto& pending = pt->second.pending_resources;
        for (auto it = pending.lower_bound(prefix);
             it != pending.end() && it->first.compare(0, prefix.size(), prefix) == 0;)
            it = pending.erase(it);
        if (pt->second.sending_key.compare(0, prefix.size(), prefix) == 0) stop_sending_resource(pt->second);
    }
}

void Hub::Impl::forget_programs(Session& s) {
    std::vector<std::string> ids;
    for (const auto& kv : s.draw_halves) ids.push_back(kv.first);
    for (const auto& kv : s.kept_resources) ids.push_back(kv.first.substr(0, kv.first.find('/')));
    for (const std::string& id : ids) forget_program(s, id);
}

// The producer's relay of a draw half (a PROGRAM set of an instance of a piece) or of a removal.
// The hub keeps the last draw half of each program, at most CALIPER_NET_MAX_PROGRAMS, and sends
// the relay to every attached viewer that announced "resources". A removal, or the draw half of
// another piece, also drops the program's kept resources; the same piece relayed again (after an
// attachment) keeps them, so that a static resource, which the script publishes once, stays. A
// producer without the capability, and any other PROGRAM from a producer, are ignored.
void Hub::Impl::producer_program(Session& s, const wire::Program& m, SharedBytes buf) {
    const bool set = m.op == CALIPER_NET_PROGRAM_SET && !m.piece.is_null();
    if (!s.resources_cap || !(set || m.op == CALIPER_NET_PROGRAM_REMOVE)) return;
    ++c.program_relays_in;
    const Json* h = set ? m.piece.find("hash") : nullptr;
    const std::string hash = h && h->is_string() ? h->as_string() : std::string();
    auto old = s.draw_halves.find(m.id);
    if (!set || (old != s.draw_halves.end() && old->second.hash != hash)) {
        forget_program(s, m.id);
    } else if (old != s.draw_halves.end()) {
        meta_bytes -= old->second.frame->size();
        s.draw_halves.erase(old);
    }
    if (set && s.draw_halves.size() < CALIPER_NET_MAX_PROGRAMS) {
        s.draw_halves[m.id] = Session::DrawHalf{hash, buf};
        meta_bytes += buf->size();
    }
    for (ConnId v : s.viewers) {
        auto pt = peers.find(v);
        if (pt == peers.end() || pt->second.closing || !pt->second.cap_resources) continue;
        if (!server.send(pt->second.conn, buf)) {
            overflow(pt->second.conn);
            continue;
        }
        ++c.program_relays_out;
    }
    if (retained() > opt.max_retained_bytes) enforce_budget();
}

// The producer's RESOURCE: offered to every attached viewer that announced "resources", with one
// waiting version per resource and viewer, and kept as the newest version for a later viewer
// while it fits the budget of retained bytes. A version over the budget is forwarded, not kept.
// The parts of a version in parts wait until the version is whole, which the order checks of
// wire::SplitResources decide; a part they drop or refuse is counted and goes nowhere.
void Hub::Impl::producer_resource(Session& s, const wire::Resource& m, SharedBytes buf) {
    if (wire::scene_program_id(m.program)) return scene_resource(s, m, std::move(buf));
    if (!s.resources_cap) return;
    const std::string key = m.program + "/" + m.resource;
    std::string err;
    const uint64_t let_go = s.resource_parts.discarded() + s.resource_parts.expired();
    const wire::SplitResources::Result got = s.resource_parts.add(m, nullptr, &err);
    std::vector<SharedBytes> frames;
    if (!m.split) {
        frames.push_back(std::move(buf));
    } else if (got == wire::SplitResources::Result::Pending) {
        if (m.split->part == 0) s.resource_building[key].clear();
        s.resource_building[key].push_back(std::move(buf));
    } else if (got == wire::SplitResources::Result::Whole) {
        frames = std::move(s.resource_building[key]);
        frames.push_back(std::move(buf));
        s.resource_building.erase(key);
    } else {
        ++c.resource_parts_dropped;
    }
    // The frames of the versions that the checks no longer hold: refused, replaced, or expired.
    for (auto it = s.resource_building.begin(); it != s.resource_building.end();) {
        const size_t slash = it->first.find('/');
        if (!s.resource_parts.holds(it->first.substr(0, slash), it->first.substr(slash + 1)))
            it = s.resource_building.erase(it);
        else
            ++it;
    }
    c.resource_versions_incomplete += s.resource_parts.discarded() + s.resource_parts.expired() - let_go -
                                      (got == wire::SplitResources::Result::Refused ? 1 : 0);
    if (frames.empty()) return;
    ++c.resources_in;
    auto version = std::make_shared<const std::vector<SharedBytes>>(std::move(frames));
    const uint64_t bytes = version_bytes(version);
    if (auto it = s.kept_resources.find(key); it != s.kept_resources.end()) {
        resource_bytes -= version_bytes(it->second);
        s.kept_resources.erase(it);
    }
    if (s.kept_resources.size() < kResourcesMax && retained() + bytes <= opt.max_retained_bytes) {
        s.kept_resources[key] = version;
        resource_bytes += bytes;
    }
    for (ConnId v : s.viewers) {
        auto pt = peers.find(v);
        if (pt == peers.end() || pt->second.closing || !pt->second.cap_resources) continue;
        offer_resource(pt->second, key, version);
    }
}

// The producer's answer, routed to the viewer that asked under its own
// request_id; the route of a `queued` run stays until the final answer.
void Hub::Impl::producer_control_reply(Session& s, const wire::ControlReply& m) {
    auto it = s.routes.find(m.request_id);
    if (it == s.routes.end()) return;
    ControlRoute route = it->second;
    const bool queued = m.status == CALIPER_NET_STATUS_QUEUED;
    if (queued) {
        it->second.deadline_ms = monotonic_ms() + kQueuedWaitMs;
    } else {
        s.routes.erase(it);
    }
    auto pt = peers.find(route.viewer);
    if (pt == peers.end()) return;
    Peer& v = pt->second;
    if (!queued && v.controls_waiting) --v.controls_waiting;
    if (v.closing) return;
    wire::ControlReply r = m;
    r.request_id = route.request_id;
    send(v.conn, r);
    ++c.control_replies;
}

// The producer's CONTROLS: kept (also after the session ends) and passed
// to every viewer of the session that announced `controls`, after the data
// the hub received before it (deliver_controls).
// 3.2 draft: a viewer without `interventions` gets the CONTROLS without that key.
void Hub::Impl::producer_controls(Session& s, const wire::Controls& m, SharedBytes buf) {
    s.text_params.clear();
    for (const auto& param : m.params.items())
        if (param.find("type")->as_string() == CALIPER_NET_PARAM_TEXT)
            s.text_params.insert(param.find("name")->as_string());
    if (s.controls) meta_bytes -= s.controls->size();
    if (s.controls_plain) meta_bytes -= s.controls_plain->size();
    s.controls = buf;
    s.controls_plain.reset();
    meta_bytes += buf->size();
    if (!m.interventions.is_null()) {
        wire::Controls plain = m;
        plain.interventions = Json();
        std::vector<uint8_t> b;
        if (wire::encode(wire::Message(plain), &b)) {
            s.controls_plain = std::make_shared<const Bytes>(std::move(b));
            meta_bytes += s.controls_plain->size();
        }
    }
    for (ConnId v : s.viewers) {
        auto pt = peers.find(v);
        if (pt != peers.end() && !pt->second.closing) deliver_controls(pt->second, controls_for(pt->second, s));
    }
    if (retained() > opt.max_retained_bytes) enforce_budget();
}

// 3.2 draft: the producer's PROGRAM_STATE. With a request_id it goes to the
// viewer that asked, under its own id. Without one it goes to every viewer
// of the session that announced `programs`, and the latest is kept for a
// viewer that attaches later.
void Hub::Impl::producer_program_state(Session& s, const wire::ProgramState& m, SharedBytes buf) {
    if (m.request_id) {
        auto it = s.program_routes.find(*m.request_id);
        if (it == s.program_routes.end()) return;
        ProgramRoute route = it->second;
        s.program_routes.erase(it);
        auto pt = peers.find(route.viewer);
        if (pt == peers.end() || pt->second.closing || !pt->second.cap_programs) return;
        wire::ProgramState r = m;
        r.request_id = route.request_id;
        send(pt->second.conn, r);
        ++c.program_states_out;
        return;
    }
    if (s.program_state) meta_bytes -= s.program_state->size();
    s.program_state = buf;
    meta_bytes += buf->size();
    for (ConnId v : s.viewers) {
        auto pt = peers.find(v);
        if (pt == peers.end() || pt->second.closing || !pt->second.cap_programs) continue;
        if (!server.send(pt->second.conn, buf)) overflow(pt->second.conn);
        ++c.program_states_out;
    }
    if (retained() > opt.max_retained_bytes) enforce_budget();
}

// The producer's SELECTION of one item space: kept per space, at most
// CALIPER_NET_MAX_ITEM_SPACES spaces per session, and passed to every
// viewer of the session that announced `selection`.
void Hub::Impl::producer_selection(Session& s, const wire::Selection& m, SharedBytes buf) {
    auto it = s.selections.find(m.space);
    if (it == s.selections.end()) {
        if (s.selections.size() >= CALIPER_NET_MAX_ITEM_SPACES) {
            ++c.selections_dropped;
            return;
        }
        it = s.selections.emplace(m.space, SelectionState()).first;
    }
    if (it->second.frame) meta_bytes -= it->second.frame->size();
    it->second.frame = buf;
    it->second.empty = m.items.size() == 0;
    meta_bytes += buf->size();
    for (ConnId v : s.viewers) {
        auto pt = peers.find(v);
        if (pt != peers.end() && !pt->second.closing) offer_selection(pt->second, m.space, buf);
    }
    if (retained() > opt.max_retained_bytes) enforce_budget();
}

void Hub::Impl::producer_views(Session& s, const wire::Views& m) {
    Json list = Json::array();
    std::map<std::string, ViewInfo> index;
    for (size_t i = 0; i < m.list.size() && i < CALIPER_NET_MAX_VIEWS; ++i) {
        list.push(m.views.at(i));
        index[m.list[i].name] = ViewInfo{m.list[i].id, m.list[i].kind};
    }
    const size_t before = s.views.size();
    // Channels and series of views that went away or were re-created.
    std::vector<uint64_t> gone;
    for (auto& kv : s.channels) {
        auto vi = index.find(kv.second.view);
        if (vi == index.end() || vi->second.id != kv.second.view_id) gone.push_back(kv.first);
    }
    for (uint64_t id : gone) destroy_channel(s, id);
    for (auto it = s.series.begin(); it != s.series.end();) {
        auto vi = index.find(it->first);
        if (vi == index.end() || vi->second.id != it->second.view_id) {
            series_bytes -= it->second.v.size() * sizeof(float);
            it = s.series.erase(it);
        } else {
            ++it;
        }
    }
    s.views = list;
    s.views_plain = strip_items(list);
    s.view_index = index;
    s.have_views = true;
    // After a resume: subscribe again for every channel someone still wants.
    if (s.resubscribe) {
        s.resubscribe = false;
        std::vector<uint64_t> ids;
        for (auto& kv : s.channels) ids.push_back(kv.first);
        for (uint64_t id : ids) {
            Channel* ch = channel(s, id);
            if (ch && !ch->up_sub && (!ch->members.empty() || !ch->waiters.empty() || ch->keeper))
                subscribe_upstream(s, *ch);
        }
    }
    // Keep every series view from the start, for late viewers.
    for (const auto& kv : index) {
        if (kv.second.kind != CALIPER_NET_KIND_SERIES) continue;
        bool have = false;
        for (auto& ck : s.channels)
            if (ck.second.keeper && ck.second.view == kv.first && ck.second.view_id == kv.second.id) have = true;
        if (have) continue;
        Channel ch;
        ch.id = next_channel++;
        ch.view = kv.first;
        ch.view_id = kv.second.id;
        ch.kind = kv.second.kind;
        ch.keeper = true;
        ch.used_ms = monotonic_ms();
        const uint64_t cid = ch.id;
        Channel& in = s.channels[cid] = std::move(ch);
        if (!subscribe_upstream(s, in)) s.channels.erase(cid);
    }
    // The viewers attached to the session hear the new list, with the item
    // spaces only when they announced `selection`.
    for (bool with_items : {true, false})
        for (bool with_scenes : {true, false}) {
            std::vector<uint8_t> b;
            if (!wire::encode(views_msg(&s, with_items, with_scenes), &b)) continue;
            auto shared = std::make_shared<const Bytes>(std::move(b));
            for (ConnId v : s.viewers) {
                auto pt = peers.find(v);
                if (pt == peers.end() || pt->second.cap_selection != with_items ||
                    pt->second.cap_scenes != with_scenes)
                    continue;
                if (!server.send(v, shared)) overflow(v);
            }
        }
    if (s.views.size() != before) broadcast_sessions();
}

void Hub::Impl::producer_reply(Session& s, const wire::ViewReply& m) {
    auto it = s.by_up_request.find(m.request_id);
    if (it == s.by_up_request.end()) return;
    const uint64_t id = it->second;
    s.by_up_request.erase(it);
    Channel* ch = channel(s, id);
    if (!ch || ch->up_request != m.request_id) return;
    ch->up_request = 0;
    std::vector<Waiter> ws;
    ws.swap(ch->waiters);
    if (!m.error.is_null()) {
        // The producer holds no subscription now.
        s.by_up_sub.erase(ch->up_sub);
        ch->up_sub = 0;
        const Json* code = m.error.find("code");
        const Json* text = m.error.find("message");
        const std::string cs = code && code->is_string() ? code->as_string() : CALIPER_NET_ERR_INTERNAL;
        const std::string ts = text && text->is_string() ? text->as_string() : std::string();
        for (const Waiter& w : ws) fail_waiter(s, ch->view, w, cs, ts);
        if (ch->members.empty() && !ch->resolved_ok) destroy_channel(s, id);
        else settle(s, id);
        return;
    }
    const std::string key = canon(m.data_view);
    // Equal resolved data views share one producer subscription.
    Channel* other = nullptr;
    for (auto& kv : s.channels) {
        Channel& o = kv.second;
        if (o.id == id || o.view != ch->view || o.view_id != ch->view_id || !o.resolved_ok ||
            o.resolved_key != key || o.up_request)
            continue;
        if (!other || (o.up_sub && !other->up_sub)) other = &o;
    }
    if (other && other->up_sub) {
        // Merge this channel into the subscribed one.
        other->aliases.insert(ch->aliases.begin(), ch->aliases.end());
        other->aliases.insert(canon(ch->requested));
        other->keeper = other->keeper || ch->keeper;
        for (const SubKey& k : ch->members)
            if (VSub* v = vsub(k)) {
                v->channel = other->id;
                other->members.insert(k);
            }
        ch->members.clear();
        if (!other->latest && ch->latest) set_latest(*other, ch->latest);
        if (other->scene_held.empty()) other->scene_held.swap(ch->scene_held);   // 3.2 draft
        const uint64_t oid = other->id;
        destroy_channel(s, id);
        Channel* o = channel(s, oid);
        for (const Waiter& w : ws)
            if (o) serve_waiter(s, *o, w);
        if (o) adjust_rate(s, *o);
        return;
    }
    ch->resolved = m.data_view;
    ch->resolved_key = key;
    ch->resolved_ok = true;
    ch->aliases.insert(canon(ch->requested));
    if (other && !other->up_sub) {
        // An idle channel with the same data view gives this one its aliases.
        ch->aliases.insert(other->aliases.begin(), other->aliases.end());
        if (!ch->latest && other->latest) set_latest(*ch, other->latest);
        if (ch->scene_held.empty()) ch->scene_held.swap(other->scene_held);      // 3.2 draft
        for (const SubKey& k : other->members)
            if (VSub* v = vsub(k)) {
                v->channel = ch->id;
                ch->members.insert(k);
            }
        other->members.clear();
        const uint64_t oid = other->id;
        destroy_channel(s, oid);
        ch = channel(s, id);
        if (!ch) return;
    }
    for (const Waiter& w : ws) {
        Channel* x = channel(s, id);
        if (x) serve_waiter(s, *x, w);
    }
    settle(s, id);
}

void Hub::Impl::producer_snapshot(Session& s, wire::DecodeResult& r, const wire::Snapshot& m, SharedBytes buf) {
    auto it = s.by_up_sub.find(m.sub);
    if (it == s.by_up_sub.end()) return;
    Channel* ch = channel(s, it->second);
    if (!ch || r.frame.sections.empty()) return;
    auto snap = std::make_shared<Snap>();
    snap->frame = std::move(buf);
    snap->meta = std::move(r.frame.meta);
    for (const auto& sec : r.frame.sections) snap->lens.push_back((uint32_t)sec.size);
    snap->tail = r.frame.sections.front().offset;
    snap->flags = r.frame.flags;
    set_latest(*ch, snap);
    ch->used_ms = monotonic_ms();
    for (const SubKey& k : ch->members) {
        VSub* v = vsub(k);
        if (!v) continue;
        if (v->pending) ++c.snapshots_superseded;
        v->pending = snap;
        ++v->pending_n;
        mark_dirty(k.conn);
    }
    if (retained() > opt.max_retained_bytes) enforce_budget();
}

void Hub::Impl::producer_series(Session& s, const wire::Series& m) {
    auto it = s.by_up_sub.find(m.sub);
    if (it == s.by_up_sub.end()) return;
    Channel* ch = channel(s, it->second);
    if (!ch || ch->kind != CALIPER_NET_KIND_SERIES) return;
    History& h = s.series[ch->view];
    if (h.view_id != ch->view_id) {
        series_bytes -= h.v.size() * sizeof(float);
        h = History();
        h.view_id = ch->view_id;
    }
    const size_t before = h.v.size();
    const uint64_t n = (uint64_t)m.values.size / sizeof(float);
    const uint8_t* p = m.values.data;
    auto val = [&](uint64_t i) {
        float f;
        std::memcpy(&f, p + i * sizeof(float), sizeof f);
        return f;
    };
    if (h.v.empty()) h.first = m.start;
    if (m.start > h.end()) {   // a gap the hub cannot fill: start again from here
        h.v.clear();
        h.first = m.start;
    }
    const uint64_t a = std::max(m.start, h.first), b = std::min(m.start + n, h.end());
    for (uint64_t i = a; i < b; ++i) h.v[(size_t)(i - h.first)] = val(i - m.start);
    for (uint64_t i = std::max(m.start, h.end()); i < m.start + n; ++i) h.v.push_back(val(i - m.start));
    h.total = std::max({h.total, m.total, h.end()});
    h.stamp_ns = std::max(h.stamp_ns, m.stamp_ns);
    // Bounded: series_retain values, and no growth past the byte budget.
    size_t cap = opt.series_retain;
    if (retained() + (h.v.size() - before) * sizeof(float) > opt.max_retained_bytes)
        cap = std::max(kSeriesKeep, std::min(cap, before));
    while (h.v.size() > cap) {
        h.v.pop_front();
        ++h.first;
    }
    series_bytes = series_bytes - before * sizeof(float) + h.v.size() * sizeof(float);
    for (auto& kv : s.channels) {
        if (kv.second.view != ch->view) continue;
        for (const SubKey& k : kv.second.members) mark_dirty(k.conn);
    }
    if (retained() > opt.max_retained_bytes) enforce_budget();
}

// Over the byte budget: ended sessions go first (the oldest), then the
// last snapshots of channels nobody watches (the least recently used), then
// the states of watched scenes (the least recently used), which the hub
// forgets (3.2 draft).
void Hub::Impl::enforce_budget() {
    while (retained() > opt.max_retained_bytes) {
        if (ended_count() > 0) {
            evict_oldest_ended();
            continue;
        }
        Session* ls = nullptr;
        Channel* lru = nullptr;
        for (auto& kv : sessions)
            for (auto& ck : kv.second.channels) {
                Channel& ch = ck.second;
                if ((!ch.latest && ch.scene_held.empty()) || !ch.members.empty() || !ch.waiters.empty()) continue;
                if (!lru || ch.used_ms < lru->used_ms) {
                    lru = &ch;
                    ls = &kv.second;
                }
            }
        if (!lru) {
            Channel* scene = nullptr;
            for (auto& kv : sessions)
                for (auto& ck : kv.second.channels)
                    if (!ck.second.scene_held.empty() && (!scene || ck.second.used_ms < scene->used_ms))
                        scene = &ck.second;
            if (!scene) return;
            forget_scene(*scene);
            continue;
        }
        set_latest(*lru, nullptr);
        release_scene(*lru);   // 3.2 draft
        if (!lru->up_sub && lru->idle()) destroy_channel(*ls, lru->id);
    }
}

void Hub::Impl::on_producer_message(Peer& p, wire::DecodeResult& r, wire::Message& m, SharedBytes buf) {
    auto it = sessions.find(p.session);
    if (it == sessions.end()) return;
    Session& s = it->second;
    if (std::holds_alternative<wire::Snapshot>(m)) {
        ++c.snapshots_in;
        producer_snapshot(s, r, std::get<wire::Snapshot>(m), std::move(buf));
    } else if (auto* x = std::get_if<wire::Series>(&m)) {
        ++c.series_in;
        producer_series(s, *x);
    } else if (auto* v = std::get_if<wire::Views>(&m)) {
        producer_views(s, *v);
    } else if (std::holds_alternative<wire::SceneUpdate>(m) || std::holds_alternative<wire::SceneClear>(m) ||
               std::holds_alternative<wire::SceneTransforms>(m) || std::holds_alternative<wire::SceneClasses>(m)) {
        producer_scene(s, r, m, std::move(buf));                 // 3.2 draft
    } else if (auto* q = std::get_if<wire::ViewReply>(&m)) {
        producer_reply(s, *q);
    } else if (auto* cs = std::get_if<wire::Controls>(&m)) {
        producer_controls(s, *cs, std::move(buf));
    } else if (auto* ps = std::get_if<wire::ProgramState>(&m)) {
        producer_program_state(s, *ps, std::move(buf));   // 3.2 draft
    } else if (auto* rs = std::get_if<wire::Resource>(&m)) {
        producer_resource(s, *rs, std::move(buf));        // 3.2 draft
    } else if (auto* pg = std::get_if<wire::Program>(&m)) {
        producer_program(s, *pg, std::move(buf));         // 3.2 draft: a relayed draw half
    } else if (auto* sel = std::get_if<wire::Selection>(&m)) {
        producer_selection(s, *sel, std::move(buf));
    } else if (auto* cr = std::get_if<wire::ControlReply>(&m)) {
        producer_control_reply(s, *cr);
    } else if (auto* pi = std::get_if<wire::Ping>(&m)) {
        send(p.conn, wire::Pong{pi->nonce});
    } else if (std::holds_alternative<wire::Pong>(m)) {
    } else if (std::holds_alternative<wire::Bye>(m)) {
        close(p.conn, CALIPER_NET_CLOSE_NORMAL, "bye");
    } else if (auto* er = std::get_if<wire::Error>(&m)) {
        if (er->fatal) close(p.conn, CALIPER_NET_CLOSE_NORMAL, "peer error");
    } else if (std::holds_alternative<wire::Unknown>(m)) {
    } else {
        fatal(p.conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST,
              std::string("a producer does not send ") + wire::type_name(r.type) + " to a hub");
    }
}

}  // namespace caliper_net
