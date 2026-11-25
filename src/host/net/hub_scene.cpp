// hub_scene.cpp: wire protocol 3.2 draft scene handling (scenes and
// scene_pieces): split updates, piece-form bundles, retained scene state,
// late-viewer sync, and per-frame bundle draining for the hub.
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
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace caliper_net {

using namespace hub_detail;

namespace {

// The messages a viewer's scene subscription may have waiting in order (a state, a reset and at
// most CALIPER_NET_MAX_BATCHES batches, and as many clears); past it the hub's own state replaces them.
constexpr size_t kSceneFirstMax = 2 * CALIPER_NET_MAX_BATCHES;
// A viewer's pending batches (one per entity part) and pending entity transforms (one per entity)
// wait while its send is in flight and count in no budget, so each holds at most this many; past
// it the hub's own state replaces them, as past kSceneFirstMax.
constexpr size_t kScenePendingMax = CALIPER_NET_MAX_BATCHES;
constexpr int kClassesOrder = -2, kTransformsOrder = -1;

int scene_part_order(const std::string& part) {
    const int i = wire::scene_part_index(part);
    return i < 0 ? 7 : i;
}

bool scene_reaches(const SceneMsg& c, const std::string& p) {
    return p == c.path || (c.recursive && (c.path.empty() || (p.size() > c.path.size() &&
                                                              p.compare(0, c.path.size(), c.path) == 0 &&
                                                              p[c.path.size()] == '/')));
}

// Whether clear `c` removes the batch of `part` at `path`: a reset removes
// everything; otherwise static items stay unless the clear is static too.
// Only a reset removes the class table, which belongs to no entity.
bool scene_clears(const SceneMsg& c, const SceneMsg& b) {
    if (c.reset) return true;
    if (b.type == CALIPER_NET_MSG_SCENE_CLASSES) return false;
    if (b.is_static && !c.is_static) return false;
    return scene_reaches(c, b.path) &&
           (c.parts.empty() || std::find(c.parts.begin(), c.parts.end(), b.part) != c.parts.end());
}

// Whether clear `c` removes the entity transform of `path`.
bool scene_clears_transform(const SceneMsg& c, const std::string& path, const TransformRow& t) {
    if (c.reset) return true;
    if (t.is_static && !c.is_static) return false;
    return scene_reaches(c, path) && (c.parts.empty() || std::find(c.parts.begin(), c.parts.end(),
                                                                    CALIPER_NET_PART_TRANSFORM) != c.parts.end());
}

// SCENE_TRANSFORMS frames of transform rows: the static ones first, then the others, grouped by the
// length of their stamps, at most CALIPER_NET_MAX_TRANSFORMS in a message.
std::vector<ScenePtr> transform_messages(const std::string& view, const TransformRows& rows) {
    std::vector<ScenePtr> out;
    std::map<std::pair<int, size_t>, std::vector<const std::pair<const std::string, TransformRow>*>> groups;
    for (const auto& kv : rows) groups[{kv.second.is_static ? 0 : 1, kv.second.stamp.size()}].push_back(&kv);
    for (const auto& g : groups) {
        const auto& items = g.second;
        for (size_t at = 0; at < items.size(); at += CALIPER_NET_MAX_TRANSFORMS) {
            const size_t k = std::min<size_t>(CALIPER_NET_MAX_TRANSFORMS, items.size() - at);
            wire::SceneTransforms t;
            t.view = view;
            t.is_static = g.first.first == 0;
            std::vector<float> mats(k * 16);
            std::vector<int64_t> stamps(k * g.first.second);
            for (size_t i = 0; i < k; ++i) {
                const auto& kv = *items[at + i];
                t.paths.push_back(kv.first);
                t.seq = std::max(t.seq, kv.second.seq);
                t.stamp_ns = std::max(t.stamp_ns, kv.second.stamp_ns);
                std::memcpy(mats.data() + 16 * i, kv.second.m, 64);
                std::copy(kv.second.stamp.begin(), kv.second.stamp.end(), stamps.begin() + (ptrdiff_t)(i * g.first.second));
            }
            t.matrices = wire::Section::make("matrices", CALIPER_NET_DTYPE_F32, {(int64_t)k, 16}, mats.data(), mats.size() * 4);
            if (g.first.second)
                t.stamps = wire::Section::make("stamps", CALIPER_NET_DTYPE_I64, {(int64_t)k, (int64_t)g.first.second},
                                               stamps.data(), stamps.size() * 8);
            std::vector<uint8_t> bytes;
            if (!wire::encode(wire::Message(t), &bytes)) continue;
            auto msg = std::make_shared<SceneMsg>();
            wire::Frame f;
            wire::FrameError fe;
            if (!wire::decode_frame(bytes.data(), bytes.size(), &f, &fe)) continue;
            msg->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
            msg->meta = std::move(f.meta);
            msg->flags = f.flags;
            for (const auto& sec : f.sections) msg->lens.push_back((uint32_t)sec.size);
            msg->tail = f.sections.front().offset;
            msg->is_static = t.is_static;
            msg->frame = std::make_shared<const Bytes>(std::move(bytes));
            out.push_back(std::move(msg));
        }
    }
    return out;
}

}  // namespace

// A scene program's RESOURCE (the piece form of SCENE_UPDATE): its whole version waits for the update
// that names it, in the place of an older one, and never takes the program route. At most every batch
// of a scene times the resources of a piece wait, so that updates that never come cannot grow it.
void Hub::Impl::scene_resource(Session& s, const wire::Resource& m, SharedBytes buf) {
    const std::string key = m.program + "/" + m.resource;
    std::string err;
    const wire::SplitResources::Result got = s.scene_parts.add(m, nullptr, &err);
    std::vector<SharedBytes> frames;
    if (!m.split) {
        frames.push_back(std::move(buf));
    } else if (got == wire::SplitResources::Result::Pending) {
        if (m.split->part == 0) s.scene_building[key].clear();
        s.scene_building[key].push_back(std::move(buf));
    } else if (got == wire::SplitResources::Result::Whole) {
        frames = std::move(s.scene_building[key]);
        frames.push_back(std::move(buf));
        s.scene_building.erase(key);
    } else {
        ++c.resource_parts_dropped;
    }
    for (auto it = s.scene_building.begin(); it != s.scene_building.end();) {
        const size_t slash = it->first.find('/');
        if (!s.scene_parts.holds(it->first.substr(0, slash), it->first.substr(slash + 1)))
            it = s.scene_building.erase(it);
        else
            ++it;
    }
    if (frames.empty()) return;
    if (s.scene_versions.size() >= size_t(CALIPER_NET_MAX_BATCHES) * CALIPER_NET_MAX_RESOURCES &&
        !s.scene_versions.count(key))
        s.scene_versions.clear();
    Session::SceneVersion& v = s.scene_versions[key];
    v.count = m.count;
    v.frames = std::move(frames);
}

// The update in the piece form takes the versions it names into one bundle: their frames as they came,
// then the update. Null when a version did not come or has another count, which drops the update.
std::shared_ptr<SceneMsg> Hub::Impl::scene_bundle(Session& s, const wire::SceneUpdate& up, const ScenePtr& update) {
    auto group = std::make_shared<SceneMsg>();
    group->type = CALIPER_NET_MSG_SCENE_UPDATE;
    group->path = up.path;
    group->part = up.part;
    group->is_static = up.is_static;
    group->piece_form = true;
    group->meta = Json::object();
    if (up.state) group->meta.set("state", true);
    for (const auto& kv : up.resources.members()) {
        auto it = s.scene_versions.find(*up.program + "/" + kv.first);
        uint64_t count = 0;
        if (it == s.scene_versions.end() || !kv.second.get_u64(&count) || it->second.count != count) return nullptr;
        for (const SharedBytes& f : it->second.frames) {
            auto raw = std::make_shared<SceneMsg>();
            raw->type = CALIPER_NET_MSG_RESOURCE;
            raw->raw = true;
            raw->frame = f;
            group->pieces.push_back(std::move(raw));
        }
    }
    for (const auto& kv : up.resources.members()) s.scene_versions.erase(*up.program + "/" + kv.first);
    group->pieces.push_back(update);
    return group;
}

// ---- 3.2 draft: scenes ---------------------------------------------------------------
// A scene message of a producer: the hub keeps, for a late viewer, the newest batch of each
// entity part of the channel (a camera is one), the class table and the newest entity transform
// of each entity, applies a clear to what it keeps, and offers the message to the channel's
// viewers. Clears and the producer's state go to a viewer in order and are never dropped (a clear
// first drops the viewer's pending batches and transforms it clears); any other batch replaces
// the viewer's pending one of the same entity part, a class table the pending one, and entity
// transforms merge into the viewer's pending ones, the newest per entity. The pieces of a split
// update are put together first and then go as one update, replaced only as a whole.
void Hub::Impl::producer_scene(Session& s, wire::DecodeResult& r, const wire::Message& m, SharedBytes buf) {
    const auto* up = std::get_if<wire::SceneUpdate>(&m);
    const auto* cl = std::get_if<wire::SceneClear>(&m);
    const auto* tr = std::get_if<wire::SceneTransforms>(&m);
    const auto* kl = std::get_if<wire::SceneClasses>(&m);
    const uint32_t sub = up ? up->sub : cl ? cl->sub : tr ? tr->sub : kl->sub;
    auto it = s.by_up_sub.find(sub);
    if (it == s.by_up_sub.end()) return;
    Channel* ch = channel(s, it->second);
    if (!ch || ch->kind != CALIPER_NET_KIND_SCENE) return;
    ++c.scenes_in;
    auto msg = std::make_shared<SceneMsg>();
    msg->type = r.type;
    msg->meta = std::move(r.frame.meta);
    msg->flags = r.frame.flags;
    for (const auto& sec : r.frame.sections) msg->lens.push_back((uint32_t)sec.size);
    msg->tail = r.frame.sections.empty() ? buf->size() : r.frame.sections.front().offset;
    msg->frame = std::move(buf);
    bool state = false;
    if (up) {
        msg->path = up->path;
        msg->part = up->part;
        msg->is_static = up->is_static;
        state = up->state;
        ch->scene_seq = std::max(ch->scene_seq, up->seq);
    } else if (cl) {
        msg->path = cl->path;
        msg->is_static = cl->is_static;
        msg->recursive = cl->recursive;
        msg->reset = cl->reset;
        if (cl->parts) msg->parts = *cl->parts;
        state = cl->state;
        ch->scene_seq = std::max(ch->scene_seq, cl->seq);
        if (cl->reset) {   // the producer's state begins: the hub keeps it again
            ch->scene_seen = true;
            ch->scene_forgotten = false;
        }
    } else if (tr) {
        msg->is_static = tr->is_static;
        state = tr->state;
        ch->scene_seq = std::max(ch->scene_seq, tr->seq);
    } else {
        msg->is_static = kl->is_static;
        state = kl->state;
        ch->scene_seq = std::max(ch->scene_seq, kl->seq);
    }
    ch->used_ms = monotonic_ms();
    if (up && !up->split.is_null()) {
        std::shared_ptr<SceneMsg> group = scene_piece(*ch, *up, msg);
        if (!group) return;   // more pieces to come, or a piece of an update not held
        msg = group;
        state = group->meta.find("state") && group->meta.find("state")->is_bool() &&
                group->meta.find("state")->as_bool();
    } else if (up && up->program) {
        std::shared_ptr<SceneMsg> group = scene_bundle(s, *up, msg);
        if (!group) {   // its versions did not come just before it
            ++c.scene_pieces_dropped;
            return;
        }
        msg = group;
    }
    // The entity transforms of the message, as rows.
    TransformRows rows;
    if (tr) {
        const int64_t t = tr->stamps ? tr->stamps->shape[1] : tr->stamp ? tr->stamp->shape[0] : 0;
        for (size_t k = 0; k < tr->paths.size(); ++k) {
            TransformRow row;
            std::memcpy(row.m, tr->matrices.data + 64 * k, 64);
            row.is_static = tr->is_static;
            row.seq = tr->seq;
            row.stamp_ns = tr->stamp_ns;
            row.stamp.resize((size_t)t);
            if (tr->stamps) std::memcpy(row.stamp.data(), tr->stamps->data + 8 * t * k, 8 * (size_t)t);
            else if (tr->stamp) std::memcpy(row.stamp.data(), tr->stamp->data, 8 * (size_t)t);
            rows[tr->paths[k]] = std::move(row);
        }
    }
    const SceneKey key = kl ? SceneKey{"", kClassesOrder} : SceneKey{msg->path, scene_part_order(msg->part)};
    if (cl) {
        for (auto h = ch->scene_held.begin(); h != ch->scene_held.end();) {
            if (h->first.second == kTransformsOrder) {
                auto kept = std::make_shared<TransformRows>(*h->second->rows);
                for (auto q = kept->begin(); q != kept->end();)
                    q = scene_clears_transform(*msg, q->first, q->second) ? kept->erase(q) : std::next(q);
                scene_bytes -= h->second->bytes();
                if (kept->empty()) {
                    h = ch->scene_held.erase(h);
                    continue;
                }
                auto set = std::make_shared<SceneMsg>();
                set->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
                set->rows = kept;
                scene_bytes += set->bytes();
                h->second = set;
                ++h;
            } else if (scene_clears(*msg, *h->second)) {
                scene_bytes -= h->second->bytes();
                h = ch->scene_held.erase(h);
            } else {
                ++h;
            }
        }
    } else if (!ch->scene_forgotten) {
        const SceneKey hold = tr ? SceneKey{"", kTransformsOrder} : key;
        auto h = ch->scene_held.find(hold);
        if (h == ch->scene_held.end() && ch->scene_held.size() >= CALIPER_NET_MAX_BATCHES) {
            forget_scene(*ch);   // more batches than a scene holds: the producer's state serves a late viewer
        } else if (tr) {
            auto kept = std::make_shared<TransformRows>(h == ch->scene_held.end() ? TransformRows()
                                                                                   : *h->second->rows);
            for (const auto& kv : rows) (*kept)[kv.first] = kv.second;
            auto set = std::make_shared<SceneMsg>();
            set->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
            set->rows = kept;
            ScenePtr& slot = ch->scene_held[hold];
            if (slot) scene_bytes -= slot->bytes();
            slot = set;
            scene_bytes += set->bytes();
        } else {
            ScenePtr& slot = ch->scene_held[hold];
            if (slot) scene_bytes -= slot->bytes();
            slot = msg;
            scene_bytes += msg->bytes();
        }
    }
    for (const SubKey& k : ch->members) {
        VSub* v = vsub(k);
        if (!v || !v->scene) continue;
        if (msg->piece_form && !v->scene_pieces) continue;   // a viewer of the per-shape parts: the structure alone
        if (cl) {
            if (cl->reset) {   // a new state: nothing queued before it matters
                v->scene_first.clear();
                v->transforms_pending.clear();
            }
            for (auto q = v->scene_pending.begin(); q != v->scene_pending.end();)
                q = scene_clears(*msg, *q->second) ? v->scene_pending.erase(q) : std::next(q);
            for (auto q = v->transforms_pending.begin(); q != v->transforms_pending.end();)
                q = scene_clears_transform(*msg, q->first, q->second) ? v->transforms_pending.erase(q) : std::next(q);
            v->scene_first.emplace_back(msg, false);
            if (v->scene_first.size() > kSceneFirstMax) resync_scene(s, *ch, *v);
        } else if (state) {
            v->scene_first.emplace_back(msg, false);
            if (v->scene_first.size() > kSceneFirstMax) resync_scene(s, *ch, *v);
        } else if (tr) {
            for (const auto& kv : rows) {
                if (v->transforms_pending.count(kv.first)) ++c.scenes_superseded;
                v->transforms_pending[kv.first] = kv.second;
            }
            if (v->transforms_pending.size() > kScenePendingMax) resync_scene(s, *ch, *v);
        } else {
            ScenePtr& slot = v->scene_pending[key];
            if (slot) ++c.scenes_superseded;
            slot = msg;
            if (v->scene_pending.size() > kScenePendingMax) resync_scene(s, *ch, *v);
        }
        mark_dirty(k.conn);
    }
    if (retained() > opt.max_retained_bytes) enforce_budget();
}

// A piece of a split update (the draft's "Splitting a large update"): the channel puts the
// pieces of an update together, piece 0 first, and returns the whole update once its last piece
// has arrived, else null. A newer piece 0 of the same entity part discards an incomplete update,
// at most CALIPER_NET_MAX_INCOMPLETE_UPDATES are put together at once (the oldest goes), and a
// piece of an update not held is dropped. The viewers check the pieces as they put them together.
std::shared_ptr<SceneMsg> Hub::Impl::scene_piece(Channel& ch, const wire::SceneUpdate& up, const ScenePtr& piece) {
    const Json& sp = up.split;
    uint64_t update = 0, num = 0, pieces = 0;
    sp.find("update")->get_u64(&update);
    sp.find("piece")->get_u64(&num);
    sp.find("pieces")->get_u64(&pieces);
    auto& building = ch.scene_building;
    if (num == 0) {
        for (auto b = building.begin(); b != building.end();) {
            if (b->first == update || (b->second->path == up.path && b->second->part == up.part)) {
                scene_bytes -= b->second->bytes();
                b = building.erase(b);
            } else {
                ++b;
            }
        }
        auto group = std::make_shared<SceneMsg>();
        group->type = CALIPER_NET_MSG_SCENE_UPDATE;
        group->path = up.path;
        group->part = up.part;
        group->is_static = up.is_static;
        group->meta = Json::object();
        if (up.state) group->meta.set("state", true);
        group->pieces_expected = (size_t)pieces;
        building.emplace_back(update, group);
        while (building.size() > CALIPER_NET_MAX_INCOMPLETE_UPDATES) {
            scene_bytes -= building.front().second->bytes();
            building.erase(building.begin());
        }
    }
    auto b = std::find_if(building.begin(), building.end(), [&](const auto& x) { return x.first == update; });
    if (b == building.end()) {
        ++c.scene_pieces_dropped;
        return nullptr;
    }
    std::shared_ptr<SceneMsg> group = b->second;
    group->pieces.push_back(piece);
    scene_bytes += piece->bytes();
    if (group->pieces.size() < group->pieces_expected) return nullptr;
    scene_bytes -= group->bytes();   // counted again where it is kept
    building.erase(b);
    return group;
}

// The hub forgets what it keeps of a scene. The viewers that watch it keep
// receiving its messages; a later viewer makes the hub subscribe again (join),
// and the producer's fresh state reaches it.
void Hub::Impl::forget_scene(Channel& ch) {
    release_scene(ch);
    ch.scene_seen = false;
    ch.scene_forgotten = true;
}

// What a channel keeps of a scene, and the split updates it puts together, leave the budget.
void Hub::Impl::release_scene(Channel& ch) {
    for (const auto& kv : ch.scene_held) scene_bytes -= kv.second->bytes();
    ch.scene_held.clear();
    for (const auto& b : ch.scene_building) scene_bytes -= b.second->bytes();
    ch.scene_building.clear();
}

// A viewer joins a scene channel. Once the producer's state has passed on
// it, the hub sends its own: a reset, then every batch it keeps, as state.
void Hub::Impl::join_scene(VSub& v, const Channel& ch) {
    v.scene = true;
    v.scene_first.clear();
    v.scene_pending.clear();
    v.scene_last_us.clear();
    if (!ch.scene_seen) return;   // the producer's own state is on its way
    auto reset = std::make_shared<SceneMsg>();
    reset->type = CALIPER_NET_MSG_SCENE_CLEAR;
    reset->reset = true;
    Json meta = Json::object();
    meta.set("view", ch.view);
    meta.set("seq", ch.scene_seq);
    meta.set("stamp_ns", (uint64_t)now_us() * 1000u);
    meta.set("path", "");
    meta.set("reset", true);
    reset->meta = std::move(meta);
    v.scene_first.emplace_back(reset, true);
    v.transforms_pending.clear();
    // In the order of the keys: the class table, the entity transforms, then every camera and
    // batch in the byte order of their paths.
    for (const auto& kv : ch.scene_held)
        if (!kv.second->piece_form || v.scene_pieces) v.scene_first.emplace_back(kv.second, true);
}

// A viewer's in-order queue passed kSceneFirstMax, or its pending batches or entity transforms
// passed kScenePendingMax (it stopped reading while the producer clears or writes new entities):
// the hub's own state replaces them, since it holds the effect of every message so far.
// While the hub holds no complete state, the producer's next state does instead: it opens with a
// reset, which drops what was queued before it, so the queue is dropped now.
void Hub::Impl::resync_scene(Session& s, Channel& ch, VSub& v) {
    if (ch.scene_seen) {
        join_scene(v, ch);
        return;
    }
    v.scene_first.clear();
    v.scene_pending.clear();
    v.transforms_pending.clear();
    if (ch.scene_forgotten && ch.up_sub && !ch.up_request && s.producer) {
        ch.scene_forgotten = false;
        send_subscribe(s, ch);
    }
}

bool Hub::Impl::send_scene_frame(Peer& p, VSub& v, const ScenePtr& msg, bool state) {
    if (msg->raw) {   // a scene program's RESOURCE, as it came
        if (!server.send(p.conn, msg->frame)) {
            overflow(p.conn);
            return false;
        }
        return true;
    }
    Json meta = msg->meta;
    meta.set("sub", v.sub);
    if (state) meta.set("state", true);
    if (!msg->lens.empty()) {
        std::vector<uint8_t> head;
        if (!wire::encode_head(msg->type, meta, msg->lens, msg->tail_len(), &head, nullptr,
                               CALIPER_NET_PROTOCOL_MINOR, msg->flags))
            return false;
        if (!server.send(p.conn, std::move(head), msg->frame, msg->tail, msg->tail_len())) {
            overflow(p.conn);
            return false;
        }
    } else {
        meta.erase("sections");
        std::vector<uint8_t> b;
        if (!wire::encode_frame(msg->type, meta, {}, &b)) return false;
        if (!server.send(p.conn, std::make_shared<const Bytes>(std::move(b)))) {
            overflow(p.conn);
            return false;
        }
    }
    ++c.scenes_out;
    return true;
}

// Expand only the next scene message, then send one frame. The socket's
// Writable event resumes this bundle after that frame drains. Pending
// resets and newer batches wait until the active bundle is complete.
bool Hub::Impl::send_scene_part(Peer& p, VSub& v) {
    while (!p.sending_scene.empty()) {
        ScenePtr msg = std::move(p.sending_scene.front());
        p.sending_scene.pop_front();
        if (msg->rows) {
            const auto messages = transform_messages(v.view, *msg->rows);
            for (auto it = messages.rbegin(); it != messages.rend(); ++it)
                p.sending_scene.push_front(*it);
        } else if (!msg->pieces.empty()) {
            for (auto it = msg->pieces.rbegin(); it != msg->pieces.rend(); ++it)
                p.sending_scene.push_front(*it);
        } else {
            return send_scene_frame(p, v, msg, p.sending_scene_state);
        }
    }
    p.dirty = true;   // An empty transform set leaves the next data message ready.
    return false;
}

bool Hub::Impl::send_scene(Peer& p, VSub& v, const ScenePtr& msg, bool state) {
    p.sending_scene = {msg};
    p.sending_scene_sub = v.sub;
    p.sending_scene_ticket = v.ticket;
    p.sending_scene_state = state;
    return send_scene_part(p, v);
}

}  // namespace caliper_net
