// hub_flow.cpp: per-viewer flow control, CONTROLS/SELECTION coalescing and
// ordering holds, RESOURCE send queues, rate limiting, and snapshot/series/
// scene draining for the wire protocol v3 hub.
#include "net/hub_internal.h"

#include "caliper/live_net.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace caliper_net {

using namespace hub_detail;

// ---- 3.1: CONTROLS and SELECTION to viewers ----------------------------------------
// Control messages, sent at once in order; when a viewer's send queue is
// full they wait, and a newer CONTROLS (or SELECTION of the same item
// space) replaces the waiting one, as the protocol coalesces them.
void Hub::Impl::offer_controls(Peer& p, const SharedBytes& frame) {
    if (!p.cap_controls || !frame) return;
    if (!p.pending_controls && server.queued_bytes(p.conn) < kMetaQueueLimit) {
        if (!server.send(p.conn, frame)) overflow(p.conn);
        ++c.controls_out;
    } else {
        p.pending_controls = frame;
    }
}

void Hub::Impl::offer_selection(Peer& p, const std::string& space, const SharedBytes& frame) {
    if (!p.cap_selection || !frame) return;
    if (!p.pending_selections.count(space) && server.queued_bytes(p.conn) < kMetaQueueLimit) {
        if (!server.send(p.conn, frame)) overflow(p.conn);
        ++c.selections_out;
    } else {
        p.pending_selections[space] = frame;
    }
}

// In pump(), before data: what waited, once the queue has room.
void Hub::Impl::flush_meta(Peer& p) {
    if (server.queued_bytes(p.conn) >= kMetaQueueLimit) return;
    if (p.pending_controls) {
        if (!server.send(p.conn, p.pending_controls)) overflow(p.conn);
        p.pending_controls.reset();
        ++c.controls_out;
    }
    for (auto& kv : p.pending_selections) {
        if (!server.send(p.conn, kv.second)) overflow(p.conn);
        ++c.selections_out;
    }
    p.pending_selections.clear();
}

// A hub delivers a run state after the data it received before it, or
// after 2 s, so that a paused view shows the step the run stopped at. A
// CONTROLS for a viewer waits until each of its subscriptions has sent, or
// replaced with a newer one, the snapshot that was pending when the
// CONTROLS came, and has sent its series up to where they then ended. It
// never waits for the viewer to drain: while data streams, an echo waits
// for at most one message per subscription. kOrderWaitUs after the first
// CONTROLS held for a viewer it goes anyway, with any newer one in its place.
void Hub::Impl::deliver_controls(Peer& p, const SharedBytes& frame) {
    if (!p.cap_controls || !frame) return;
    if (Session* s = session_of(p))
        for (const auto& kv : p.subs) {
            const VSub& v = kv.second;
            if (v.series) {
                auto h = s->series.find(v.view);
                if (h != s->series.end() && std::max(v.next, h->second.first) < h->second.end()) {
                    uint64_t& need = p.held_series[kv.first];
                    need = std::max(need, h->second.end());
                }
            } else if (v.pending) {
                p.held_snaps[kv.first] = v.pending_n;
            }
        }
    if (!p.held_controls && p.held_snaps.empty() && p.held_series.empty()) {
        offer_controls(p, frame);
        return;
    }
    if (!p.held_controls) {
        p.held_since_us = now_us();
        ++c.controls_held;
    }
    p.held_controls = frame;
    p.dirty = true;
}

// In pump(): sends a held CONTROLS once what it waits for has gone out or
// been replaced, or its time is up.
void Hub::Impl::release_controls(Peer& p, uint64_t now) {
    if (!p.held_controls) return;
    Session* s = session_of(p);
    for (auto it = p.held_snaps.begin(); it != p.held_snaps.end();) {
        auto v = p.subs.find(it->first);
        const bool waits = v != p.subs.end() && v->second.pending && v->second.pending_n == it->second;
        it = waits ? std::next(it) : p.held_snaps.erase(it);
    }
    for (auto it = p.held_series.begin(); it != p.held_series.end();) {
        auto v = p.subs.find(it->first);
        bool waits = false;
        if (s && v != p.subs.end() && v->second.series) {
            auto h = s->series.find(v->second.view);
            if (h != s->series.end())
                waits = std::max(v->second.next, h->second.first) < std::min(it->second, h->second.end());
        }
        it = waits ? std::next(it) : p.held_series.erase(it);
    }
    const bool waits = !p.held_snaps.empty() || !p.held_series.empty();
    if (waits && now < p.held_since_us + kOrderWaitUs) return;
    if (waits) ++c.controls_hold_expired;
    SharedBytes frame = std::move(p.held_controls);
    drop_held_controls(p);
    offer_controls(p, frame);
}

void Hub::Impl::drop_held_controls(Peer& p) {
    p.held_controls.reset();
    p.held_snaps.clear();
    p.held_series.clear();
    p.held_since_us = 0;
}

// The session's latest CONTROLS as one viewer may see it: without the key
// "interventions" unless the viewer announced that capability (3.2 draft).
SharedBytes Hub::Impl::controls_for(const Peer& p, const Session& s) {
    const auto frame = (!p.cap_interventions && s.controls_plain) ? s.controls_plain : s.controls;
    if (p.cap_text_params || !frame) return frame;
    wire::Message message;
    if (wire::decode(frame->data(), frame->size(), &message).status != wire::DecodeResult::Status::Ok) return {};
    auto* controls = std::get_if<wire::Controls>(&message);
    if (!controls) return {};
    Json params = Json::array();
    for (const auto& param : controls->params.items())
        if (param.find("type")->as_string() != CALIPER_NET_PARAM_TEXT) params.push(param);
    controls->params = std::move(params);
    Bytes bytes;
    if (!wire::encode(message, &bytes)) return {};
    return std::make_shared<const Bytes>(std::move(bytes));
}

// Right after the VIEWS of an attachment: the session's latest CONTROLS and
// every item space's selection that is not empty. A new attachment has no
// data pending, so its CONTROLS goes at once, before any SUBSCRIBE.
void Hub::Impl::send_state(Peer& p, const Session& s) {
    if (p.cap_controls && s.controls) deliver_controls(p, controls_for(p, s));
    if (p.cap_programs && s.program_state) {
        if (!server.send(p.conn, s.program_state)) overflow(p.conn);
        ++c.program_states_out;
    }
    if (p.cap_selection)
        for (const auto& kv : s.selections)
            if (!kv.second.empty) offer_selection(p, kv.first, kv.second.frame);
    if (p.cap_resources) {   // 3.2 draft: the draw halves first, then their resources as data
        for (const auto& kv : s.draw_halves) {
            if (!server.send(p.conn, kv.second.frame)) {
                overflow(p.conn);
                return;
            }
            ++c.program_relays_out;
        }
        for (const auto& kv : s.kept_resources) offer_resource(p, kv.first, kv.second);
    }
}

// ---- 3.2 draft: resources (capability "resources") ----------------------------------------
// One RESOURCE for a viewer: it replaces a waiting version of the same resource and keeps that
// version's place; a new resource past kResourcesMax waiting ones is dropped.
void Hub::Impl::offer_resource(Peer& p, const std::string& key, const ResourceVersion& buf) {
    auto it = p.pending_resources.find(key);
    if (it != p.pending_resources.end()) {
        ++c.resources_superseded;
        it->second.second = buf;
    } else if (p.pending_resources.size() >= kResourcesMax) {
        ++c.resources_dropped;
        return;
    } else {
        p.pending_resources[key] = {++p.resource_order, buf};
    }
    p.dirty = true;
}

void Hub::Impl::stop_sending_resource(Peer& p) {
    p.sending_resource.reset();
    p.sending_key.clear();
    p.sending_next = 0;
    p.sent_part = false;
}

// ---- delivery --------------------------------------------------------------------------
// A held CONTROLS goes right behind the data it waited for, in the same
// send queue; until then the viewer is due again when its wait runs out.
void Hub::Impl::pump(Peer& p, uint64_t now) {
    p.due_us = 0;
    if (p.closing) return;
    release_controls(p, now);
    pump_data(p, now);
    if (!p.held_controls) return;
    release_controls(p, now);
    if (!p.held_controls) return;
    const uint64_t until = p.held_since_us + kOrderWaitUs;
    if (!p.due_us || until < p.due_us) p.due_us = until;
}

// Hands the next data message of one viewer to its socket, if the last one
// has drained: series before snapshots, the subscription that waited
// longest first, each at most at its max_rate.
void Hub::Impl::pump_data(Peer& p, uint64_t now) {
    if (p.pending_controls || !p.pending_selections.empty()) flush_meta(p);   // control messages first
    if (p.in_flight) return;
    Session* s = session_of(p);
    if (!s) return;
    if (!p.sending_scene.empty()) {
        auto it = p.subs.find(p.sending_scene_sub);
        if (it != p.subs.end() && it->second.ticket == p.sending_scene_ticket) {
            p.in_flight = send_scene_part(p, it->second);
            return;
        }
        p.sending_scene.clear();
    }
    VSub* best = nullptr;
    bool best_series = false;
    uint64_t due_min = 0;
    VSub* scene_v = nullptr;           // 3.2 draft: the scene message to send next, if any
    bool scene_head = false;           // from the in-order queue (else a pending batch)
    SceneKey scene_key;
    for (auto& kv : p.subs) {
        VSub& v = kv.second;
        if (v.scene) {
            // In-order messages and pending entity transforms go now; a pending batch at most
            // at `rate` per entity part, a camera and a class table at once.
            if (!v.scene_first.empty() || !v.transforms_pending.empty()) {
                if (!scene_v || !scene_head || v.last_us < scene_v->last_us) {
                    scene_v = &v;
                    scene_head = true;
                }
            } else if (!scene_head) {
                const uint64_t iv = (uint64_t)(1e6 / v.rate);
                for (const auto& q : v.scene_pending) {
                    auto lt = v.scene_last_us.find(q.first);
                    const bool unlimited = q.first.second < 0 || q.first.second == kCameraOrder;
                    const uint64_t due = lt == v.scene_last_us.end() || unlimited ? 0 : lt->second + iv;
                    if (due <= now) {
                        if (!scene_v || v.last_us < scene_v->last_us) {
                            scene_v = &v;
                            scene_key = q.first;
                        }
                        break;
                    }
                    if (!due_min || due < due_min) due_min = due;
                }
            }
            continue;
        }
        bool has = false;
        if (v.series) {
            auto h = s->series.find(v.view);
            has = h != s->series.end() && (std::max(v.next, h->second.first) < h->second.end());
        } else {
            has = (bool)v.pending;
        }
        if (!has) continue;
        if (!v.refill_us) v.tokens = kBurst;
        else v.tokens = std::min(kBurst, v.tokens + (double)(now - v.refill_us) * v.rate / 1e6);
        v.refill_us = now;
        if (v.tokens < 1.0) {
            const uint64_t due = now + (uint64_t)std::ceil((1.0 - v.tokens) * 1e6 / v.rate);
            if (!due_min || due < due_min) due_min = due;
            continue;
        }
        if (!best || (v.series && !best_series) ||
            (v.series == best_series && v.last_us < best->last_us)) {
            best = &v;
            best_series = v.series;
        }
    }
    p.due_us = due_min;
    if (scene_v && !(best && best->series)) {   // 3.2 draft: scenes after series, before snapshots
        bool ok;
        if (scene_head && !scene_v->scene_first.empty()) {
            auto item = scene_v->scene_first.front();
            scene_v->scene_first.pop_front();
            ok = send_scene(p, *scene_v, item.first, item.second);
        } else if (scene_head) {   // the pending entity transforms, merged
            auto set = std::make_shared<SceneMsg>();
            set->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
            set->rows = std::make_shared<const TransformRows>(std::move(scene_v->transforms_pending));
            scene_v->transforms_pending.clear();
            ok = send_scene(p, *scene_v, set, false);
        } else {
            auto q = scene_v->scene_pending.find(scene_key);
            ScenePtr msg = q->second;
            scene_v->scene_pending.erase(q);
            scene_v->scene_last_us[scene_key] = now;
            ok = send_scene(p, *scene_v, msg, false);
        }
        if (!ok) return;
        scene_v->last_us = now;
        scene_v->sent_any = true;
        p.in_flight = true;
        return;
    }
    // 3.2 draft: resources after series and scenes, before snapshots; the one that waited longest.
    // The parts of a version in parts go in order, one at a time; between two of them at most one
    // waiting version in one frame goes, and another version in parts waits for the end of this one.
    if ((p.sending_resource || !p.pending_resources.empty()) && !(best && best->series)) {
        auto oldest = p.pending_resources.end(), single = p.pending_resources.end();
        for (auto it = p.pending_resources.begin(); it != p.pending_resources.end(); ++it) {
            if (oldest == p.pending_resources.end() || it->second.first < oldest->second.first) oldest = it;
            if (it->second.second->size() == 1 &&
                (single == p.pending_resources.end() || it->second.first < single->second.first))
                single = it;
        }
        SharedBytes buf;
        if (p.sending_resource && (!p.sent_part || single == p.pending_resources.end())) {
            buf = (*p.sending_resource)[p.sending_next++];
            p.sent_part = true;
            if (p.sending_next == p.sending_resource->size()) stop_sending_resource(p);
        } else {
            auto next = p.sending_resource ? single : oldest;
            if (next == p.pending_resources.end()) return;
            ResourceVersion v = std::move(next->second.second);
            const std::string key = next->first;
            p.pending_resources.erase(next);
            buf = (*v)[0];
            p.sent_part = false;
            if (v->size() > 1) {
                p.sending_resource = std::move(v);
                p.sending_key = key;
                p.sending_next = 1;
                p.sent_part = true;
            }
        }
        if (!server.send(p.conn, buf)) {
            overflow(p.conn);
            return;
        }
        ++c.resources_out;
        p.in_flight = true;
        return;
    }
    if (!best) return;
    bool ok;
    if (best->series) ok = send_series(p, *s, *best);
    else ok = send_snapshot(p, *best);
    if (!ok) return;
    best->last_us = now;
    best->sent_any = true;
    best->tokens -= 1.0;
    p.in_flight = true;
}

bool Hub::Impl::send_series(Peer& p, Session& s, VSub& v) {
    History& h = s.series.at(v.view);
    uint64_t start = std::max(v.next, h.first);
    if (!v.sent_any && v.backfill != std::numeric_limits<uint64_t>::max())
        start = std::max(start, h.end() > v.backfill ? h.end() - v.backfill : 0);
    const uint64_t stop = std::min<uint64_t>(h.end(), start + CALIPER_NET_SERIES_MAX_VALUES);
    if (stop <= start) return false;
    std::vector<float> vals((size_t)(stop - start));
    for (uint64_t i = start; i < stop; ++i) vals[(size_t)(i - start)] = h.v[(size_t)(i - h.first)];
    wire::Series m;
    m.sub = v.sub;
    m.view = v.view;
    m.start = start;
    m.total = std::max(h.total, stop);
    m.stamp_ns = h.stamp_ns;
    m.values = wire::Section::make("values", CALIPER_NET_DTYPE_F32, {(int64_t)vals.size()}, vals.data(),
                                   vals.size() * sizeof(float));
    v.next = stop;
    if (!send(p.conn, m)) return false;
    ++c.series_out;
    c.series_values_out += stop - start;
    return true;
}

bool Hub::Impl::send_snapshot(Peer& p, VSub& v) {
    SnapPtr snap = std::move(v.pending);
    v.pending.reset();
    Json meta = snap->meta;
    meta.set("sub", v.sub);
    std::vector<uint8_t> head;
    if (!wire::encode_head(CALIPER_NET_MSG_SNAPSHOT, meta, snap->lens, snap->tail_len(), &head, nullptr,
                           CALIPER_NET_PROTOCOL_MINOR, snap->flags))
        return false;
    if (!server.send(p.conn, std::move(head), snap->frame, snap->tail, snap->tail_len())) {
        overflow(p.conn);
        return false;
    }
    ++c.snapshots_out;
    return true;
}

}  // namespace caliper_net
