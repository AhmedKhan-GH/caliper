// source_poll.cpp: see source_poll.h.
#include "source_poll.h"

#include "caliper/live.h"
#include "caliper/live_net.h"
#include "caliper/live_views.h"
#include "live/live_views_scene.h"   // scene views (kind 6) and their messages
#include "live/live_views_state.h"   // set_payload_map
#include "net/wire.h"

#include <algorithm>

namespace caliper_web {

namespace {

uint32_t apply_flags(uint32_t snap) {
    uint32_t f = 0;
    if (snap & CALIPER_LIVE_SNAP_AUTO_VMIN) f |= CALIPER_LIVE_APPLY_AUTO_VMIN;
    if (snap & CALIPER_LIVE_SNAP_AUTO_VMAX) f |= CALIPER_LIVE_APPLY_AUTO_VMAX;
    if (snap & CALIPER_LIVE_SNAP_RAW) f |= CALIPER_LIVE_APPLY_RAW;
    return f;
}

// The snapshot as a CPU tensor at host_base (+ byte_offset as the data offset).
bool apply(CaliperCore* core, PolledView& v, const caliper_live::Snapshot& s, bool relative) {
    CaliperTensor data{};
    data.struct_size = sizeof data;
    data.data = s.host_base;
    data.dtype = (CaliperDType)s.dtype;
    data.ndim = s.ndim;
    int64_t st = 1;
    for (int i = s.ndim - 1; i >= 0; --i) {
        data.shape[i] = s.shape[i];
        data.strides[i] = st;
        st *= s.shape[i];
    }
    data.device = CALIPER_DEV_CPU;

    CaliperTensor aux{};
    if (s.aux_count > 0) {
        aux.struct_size = sizeof aux;
        aux.data = s.host_base;
        aux.dtype = CALIPER_DT_F32;
        aux.ndim = 1;
        aux.shape[0] = s.aux_count;
        aux.strides[0] = 1;
        aux.device = CALIPER_DEV_CPU;
    }

    CaliperLiveSnapshot ls{};
    ls.struct_size = sizeof ls;
    ls.name = v.name.c_str();
    ls.kind = (int32_t)s.kind;
    ls.data = &data;
    ls.data_offset = s.byte_offset;
    ls.aux = s.aux_count > 0 ? &aux : nullptr;
    ls.aux_offset = s.aux_count > 0 ? s.aux_offset : 0;
    ls.stats = s.stats;
    ls.stats_count = s.stats ? s.stats_count : 0;
    ls.colormap = s.colormap;
    ls.vmin = s.vmin;
    ls.vmax = s.vmax;
    ls.flags = apply_flags(s.flags) | (relative ? CALIPER_LIVE_APPLY_STAMP_RELATIVE : 0u);
    ls.series_total = s.series_total;
    ls.points_dims = s.points_dims;
    ls.label_count = s.label_count;
    ls.seq = s.seq;
    ls.stamp_ns = s.stamp_ns;   // already in the viewer's clock (live_clock.h)
    if (!caliper_core_live_apply(core, &ls)) {
        v.error = caliper_core_last_error(core);
        return false;
    }
    v.error.clear();
    // How the payload maps onto the source, for the view's events.
    caliper_live_views::set_payload_map(core, v.name.c_str(), s.map ? *s.map : caliper_live::PayloadMap{});
    return true;
}

void reconcile(CaliperCore* core, PollState& st, const std::vector<caliper_live::SourceView>& now) {
    std::vector<PolledView> next;
    for (const auto& sv : now) {
        auto it = std::find_if(st.views.begin(), st.views.end(),
                               [&](const PolledView& v) { return v.key == sv.key; });
        if (it != st.views.end()) {
            next.push_back(std::move(*it));
            it->key = 0;   // consumed
            continue;
        }
        PolledView v;
        v.key = sv.key;
        v.name = sv.name;
        v.kind = sv.kind;
        next.push_back(std::move(v));
    }
    for (auto& v : st.views)
        if (v.key != 0 && v.shown) caliper_core_live_view_remove(core, v.name.c_str());
    st.views = std::move(next);
}

// A scene view: opened with the scene's description from VIEWS, then fed the
// source's messages in order, as live_core.cpp does for the app. While the
// display is paused the source holds them.
int poll_scene(CaliperCore* core, caliper_live::Source& src, PollState& st, PolledView& v) {
    namespace wire = caliper_net::wire;
    if (!v.shown) {
        if (!v.error.empty()) return 0;
        wire::SceneView info{CALIPER_NET_SPACE_3D, "+y", "m"};
        caliper_net::Json offer;
        std::string why;
        if (src.data_view_of(v.key, nullptr, &offer)) wire::scene_view_info(offer, &info, &why);
        if (!caliper_live_views::scene_open(core, v.name, info, &why)) {
            v.error = why;
            ++st.errors;
            return 0;
        }
        v.shown = true;
    }
    if (caliper_core_live_paused(core)) return 0;
    std::vector<caliper_live::SceneItem> items;
    if (!src.take_scene(v.key, &items)) return 0;
    caliper_live_views::scene_begin_round(core, v.name);
    int applied = 0;
    for (const caliper_live::SceneItem& it : items) {
        std::string why;
        bool ok = true;
        if (const auto* u = std::get_if<wire::SceneUpdate>(&it.msg)) {
            ok = caliper_live_views::scene_apply(core, v.name, *u, &why);
            v.seq = u->seq;
            ++applied;
        } else if (const auto* c = std::get_if<wire::SceneClear>(&it.msg)) {
            ok = caliper_live_views::scene_apply(core, v.name, *c, &why);
            ++applied;
        } else {
            caliper_live_views::scene_note(core, v.name, std::get<std::string>(it.msg));
        }
        if (!ok) {
            ++st.errors;
            v.error = why;
        }
    }
    v.updates += (uint64_t)applied;
    st.updates += (uint64_t)applied;
    return applied;
}

}  // namespace

int poll_source(CaliperCore* core, caliper_live::Source& src, PollState& st) {
    ++st.polls;
    src.heartbeat();
    st.producer_alive = src.producer_alive();
    std::vector<caliper_live::SourceView> now;
    if (src.sync(&now)) reconcile(core, st, now);

    int applied = 0;
    for (auto& v : st.views) {
        if (v.kind == (uint32_t)caliper_live_views::kKindScene) {
            applied += poll_scene(core, src, st, v);
            continue;
        }
        caliper_live::Snapshot s;
        if (!src.take(v.key, v.seq, &s)) continue;
        if (!apply(core, v, s, st.relative_stamps)) {
            src.unclaim(v.key);
            ++st.errors;
            v.seq = s.seq;   // do not retry the same snapshot every frame
            continue;
        }
        src.release_prev(v.key);   // a CPU payload was copied by the apply
        const uint64_t skipped = v.seq && s.seq > v.seq + 1 ? s.seq - v.seq - 1 : 0;
        src.drew(v.key, s.seq, skipped);
        v.shown = true;
        v.seq = s.seq;
        ++v.updates;
        ++st.updates;
        ++applied;
    }
    return applied;
}

void remove_polled_views(CaliperCore* core, PollState& st) {
    for (auto& v : st.views)
        if (v.shown) caliper_core_live_view_remove(core, v.name.c_str());
    st.views.clear();
}

}  // namespace caliper_web
