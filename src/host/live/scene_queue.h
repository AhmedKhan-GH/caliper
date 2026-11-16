// scene_queue.h: the scene messages a source holds for the frame thread (live_source.h,
// Source::take_scene), shared by the app's NetSource and the browser's.
//
// A scene message is applied in order, so a source queues them, but it keeps only what still
// matters while the frame thread has not taken them: a reset drops every message before it, and
// an update of an entity part replaces an older update of the same part and layer (static or not)
// queued after the last clear of that entity. The queue therefore holds at most one update per
// entity part and layer between clears, as the flow control of protocol 3.2 asks of a sender.
#pragma once

#include "live/live_source.h"

#include <deque>
#include <string>

namespace caliper_live {

inline constexpr size_t kSceneQueueMax = 8192;   // past this the oldest items go (a producer that only clears)

// Whether a clear of `c.path` reaches the entity at `path`.
inline bool scene_clear_reaches(const caliper_net::wire::SceneClear& c, const std::string& path) {
    if (c.reset || c.path == path) return true;
    if (!c.recursive) return false;
    return c.path.empty() || (path.size() > c.path.size() && path.compare(0, c.path.size(), c.path) == 0 &&
                              path[c.path.size()] == '/');
}

inline void scene_queue_push(std::deque<SceneItem>& q, SceneItem item) {
    namespace wire = caliper_net::wire;
    if (const auto* c = std::get_if<wire::SceneClear>(&item.msg)) {
        if (c->reset) q.clear();
    } else if (const auto* u = std::get_if<wire::SceneUpdate>(&item.msg)) {
        for (auto it = q.end(); it != q.begin();) {
            --it;
            if (const auto* qc = std::get_if<wire::SceneClear>(&it->msg)) {
                if (scene_clear_reaches(*qc, u->path)) break;
                continue;
            }
            const auto* qu = std::get_if<wire::SceneUpdate>(&it->msg);
            if (qu && qu->path == u->path && qu->part == u->part && qu->is_static == u->is_static) {
                q.erase(it);
                break;
            }
        }
    } else if (!q.empty()) {
        const auto* last = std::get_if<std::string>(&q.back().msg);
        if (last && *last == std::get<std::string>(item.msg)) return;
    }
    q.push_back(std::move(item));
    while (q.size() > kSceneQueueMax) q.pop_front();
}

}  // namespace caliper_live
