// scene_queue.h: the scene messages a source holds for the frame thread (live_source.h,
// Source::take_scene), shared by the app's NetSource and the browser's.
//
// A scene message is applied in order, so a source queues them, but it keeps only what still
// matters while the frame thread has not taken them, as SceneView::apply would leave it: a reset
// drops every message before it; a clear drops the queued updates it removes and the queued
// clears it covers; an update of an entity part replaces an older update of the same part and
// layer (static or not). The queue therefore holds at most one update per entity part and layer
// and one clear per target, and it never drops a reset or a clear. Past kSceneQueueMax messages
// (a sender that clears ever new paths while the display is paused) it empties and waits for the
// scene's next state: the source asks for it again, and everything before its reset is dropped.
#pragma once

#include "live/live_source.h"

#include <algorithm>
#include <deque>
#include <string>

namespace caliper_live {

inline constexpr size_t kSceneQueueMax = 8192;

// ---- what the scene view draws (docs/references/wire-protocol-3-2.md, "Versioning") ----------
// The codec decodes every part of the 3.2 draft; the scene view draws points, lines of mode
// segments and strips, and meshes. For another part or mode this is why a source does not queue
// the update: it answers ERROR unsupported and notes it in the view. "" when the view draws it.
inline std::string scene_not_drawn(const caliper_net::wire::SceneUpdate& u) {
    if (u.part == CALIPER_NET_PART_POINTS || u.part == CALIPER_NET_PART_MESH) return {};
    if (u.part == CALIPER_NET_PART_LINES) {
        const caliper_net::Json* mode = u.params.find("mode");
        if (mode && mode->is_string() && mode->as_string() == CALIPER_NET_LINES_EDGES)
            return "lines of mode 'edges' (index-pair edges) are not drawn yet";
        return {};
    }
    return "the part '" + u.part + "' is not drawn yet";
}

// The sections and keys of an update the view draws that it does not use (a stamp, class ids,
// normals, a texture, labels or a section no part has), named for a note; "" when none.
inline std::string scene_ignored(const caliper_net::wire::SceneUpdate& u, const caliper_net::wire::Frame& f) {
    static const char* const points[] = {"positions", "colors", "values", "radii", nullptr};
    static const char* const lines[] = {"positions", "strip_lengths", "colors", "values", nullptr};
    static const char* const mesh[] = {"positions", "indices", "colors", "values", nullptr};
    const char* const* used = u.part == CALIPER_NET_PART_POINTS ? points : u.part == CALIPER_NET_PART_LINES ? lines : mesh;
    std::string out;
    for (const caliper_net::wire::Section& s : f.sections) {
        bool drawn = false;
        for (const char* const* k = used; *k && !drawn; ++k) drawn = s.name == *k;
        if (!drawn) out += (out.empty() ? "" : ", ") + s.name;
    }
    if (u.params.find("labels")) out += std::string(out.empty() ? "" : ", ") + "labels";
    if (out.empty()) return out;
    return "ignored in " + (u.path.empty() ? std::string("the root") : u.path) + " " + u.part + ": " + out;
}

// A scene view's queue, whether it waits for the scene's next state (a reset), and whether the
// view will never be shown (Source::stop_scene), so that nothing is queued for it.
struct SceneQueue {
    std::deque<SceneItem> items;
    bool resync = false;
    bool stopped = false;
};

// Whether a clear of `c.path` reaches the entity at `path`.
inline bool scene_clear_reaches(const caliper_net::wire::SceneClear& c, const std::string& path) {
    if (c.reset || c.path == path) return true;
    if (!c.recursive) return false;
    return c.path.empty() || (path.size() > c.path.size() && path.compare(0, c.path.size(), c.path) == 0 &&
                              path[c.path.size()] == '/');
}

// Whether clear `c` removes the part that update `u` holds: a reset removes everything;
// otherwise the clear reaches the entity, names the part (or every part), and removes a static
// item only if it is static itself.
inline bool scene_clear_removes(const caliper_net::wire::SceneClear& c, const caliper_net::wire::SceneUpdate& u) {
    if (c.reset) return true;
    if (!scene_clear_reaches(c, u.path) || (u.is_static && !c.is_static)) return false;
    return !c.parts || std::find(c.parts->begin(), c.parts->end(), u.part) != c.parts->end();
}

// Whether clear `a` removes everything that clear `b` removes.
inline bool scene_clear_covers(const caliper_net::wire::SceneClear& a, const caliper_net::wire::SceneClear& b) {
    if (a.reset) return true;
    if (b.reset || (b.is_static && !a.is_static)) return false;
    const bool path = a.path == b.path ? a.recursive || !b.recursive : a.recursive && scene_clear_reaches(a, b.path);
    if (!path || !a.parts) return path;
    if (!b.parts) return false;
    for (const std::string& p : *b.parts)
        if (std::find(a.parts->begin(), a.parts->end(), p) == a.parts->end()) return false;
    return true;
}

// Queues `item`. True when the queue passed kSceneQueueMax and was emptied: the source then asks
// the sender for the scene's state again (a SUBSCRIBE on the same sub), and the queue takes
// nothing until that state's reset.
inline bool scene_queue_push(SceneQueue& sq, SceneItem item) {
    namespace wire = caliper_net::wire;
    if (sq.stopped) return false;
    std::deque<SceneItem>& q = sq.items;
    if (const auto* c = std::get_if<wire::SceneClear>(&item.msg)) {
        if (c->reset) {
            q.clear();
            sq.resync = false;
        } else if (!sq.resync) {
            q.erase(std::remove_if(q.begin(), q.end(),
                                   [&](const SceneItem& it) {
                                       if (const auto* qu = std::get_if<wire::SceneUpdate>(&it.msg))
                                           return scene_clear_removes(*c, *qu);
                                       const auto* qc = std::get_if<wire::SceneClear>(&it.msg);
                                       return qc && scene_clear_covers(*c, *qc);
                                   }),
                    q.end());
        }
    } else if (const auto* u = std::get_if<wire::SceneUpdate>(&item.msg)) {
        if (!sq.resync)
            for (auto it = q.begin(); it != q.end(); ++it) {
                const auto* qu = std::get_if<wire::SceneUpdate>(&it->msg);
                if (qu && qu->path == u->path && qu->part == u->part && qu->is_static == u->is_static) {
                    q.erase(it);   // at most one: the queue holds one update per part and layer
                    break;
                }
            }
    } else if (!q.empty()) {
        const auto* last = std::get_if<std::string>(&q.back().msg);
        if (last && *last == std::get<std::string>(item.msg)) return false;
    }
    if (sq.resync) return false;   // waiting for the scene's next state
    q.push_back(std::move(item));
    if (q.size() <= kSceneQueueMax) return false;
    q.clear();
    sq.resync = true;
    return true;
}

}  // namespace caliper_live
