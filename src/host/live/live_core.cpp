// live_core.cpp: the viewer side of live streaming inside the embed core
// (caliper/live.h, caliper_core_live_*), plus the two hooks embed_core.cpp
// calls (caliper_live_frame, caliper_live_shutdown).
//
// TRANSPORT-AGNOSTIC. Everything here works on caliper_live::Snapshot ("an
// MTLBuffer + byte offsets + stats + seq/stamp") from a caliper_live::Source
// (live_source.h). The shared-memory session of another process is one Source
// (live_session.mm). This file only:
//   1. reconciles the source's view list with typed live views (same names),
//   2. per frame, for each view whose previous GPU read has retired, claims the
//      newest completed snapshot and applies it through caliper_core_live_apply
//      (caliper/live_views.h; METAL tensors at the snapshot's byte offsets, so
//      the view layer reads the producer's pages in place, never waiting),
//   3. keeps the counters (updates, skipped, step-to-screen latency, fps),
//      writes the heartbeat that makes the client's attached() true, and draws
//      the status window (session, pid, script, "session ended", per-view
//      seq/latency/dropped, Detach) and, when enabled, the session picker.
// Nothing here blocks: a view whose previous read is still on the GPU simply
// waits for a later frame (every completion probe is a poll).
//
// CLAIMS PER KIND (see live_source.h): HEATMAP / IMAGE / HISTOGRAM are read
// once per apply, so the next snapshot is claimed only once read_done says the
// last apply's read retired (one claim). POINTS are drawn from the payload every
// frame: the next snapshot is claimed while the current one stays claimed as
// "previous" until the view layer's read_done says its last frame retired (two
// claims). SERIES are never claimed.
#include "caliper/live.h"
#include "caliper/live_views.h"

#include "embed/embed_core.h"
#include "live/live_shm.h"
#include "live/live_source.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using caliper_live::SessionEntry;
using caliper_live::Snapshot;
using caliper_live::Source;
using caliper_live::SourceView;

namespace {

// THE SWITCH. false: HEATMAP / IMAGE go through B's embed v1.2 path
// (caliper_core_view_tensor + the renderer's completion handle); true: every
// kind goes through caliper_core_live_apply (caliper/live_views.h). POINTS /
// SERIES / HISTOGRAM always use caliper_core_live_apply.
constexpr bool kTextureKindsThroughLiveViews = true;

constexpr uint64_t kPendingRetryNs = 50ull * 1000 * 1000;     // a starting session: retry every 50 ms
constexpr uint64_t kPickerRefreshNs = 1000ull * 1000 * 1000;  // the picker relists once a second

}  // namespace

struct CaliperLiveState {
    // ---- picker ----------------------------------------------------------
    bool        picker_enabled = false;
    std::string picker_root;                 // "" = the default data root
    std::vector<SessionEntry> sessions;
    std::string picker_error;
    uint64_t    sessions_t = 0;              // last listing (CLOCK_UPTIME_RAW ns)
    std::string attach_request;              // token clicked in the picker

    // ---- attached --------------------------------------------------------
    bool                    attached = false;
    std::unique_ptr<Source> source;          // null while pending
    std::string             session;
    std::string             script;
    uint64_t                start_ms = 0;
    int32_t                 pid = 0;         // from the session file while pending
    bool                    pending = false;
    uint64_t                retry_ns = 0;
    bool                    detach_requested = false;
    std::string             pending_why;

    struct View {
        uint64_t    key = 0;
        std::string name;
        uint32_t    kind = 0;
        bool        shown = false;          // a view of this name exists in the core
        bool        typed = false;          // shown through caliper_core_live_apply
        bool        prev_claimed = false;   // POINTS: the source holds a previous claim
        uint64_t    seq = 0;                // last drawn seq
        uint64_t    updates = 0, window_updates = 0, skipped = 0;
        double      latency_ms_last = 0.0;
        // what the last applied snapshot carried
        uint64_t    series_total = 0;
        int32_t     points_dims = 0, label_count = 0, stats_count = 0;
        uint32_t    slot = CALIPER_LIVE_SLOT_NONE;
        // v1.2 path
        float       vmin = 0.0f, vmax = 1.0f;
        bool        have_range = false;     // last finite range (non-finite stats fallback)
        void*       inflight = nullptr;     // +1 handle on this view's last GPU update
        std::string error;
    };
    std::vector<View> views;

    uint64_t frames = 0, polls = 0, updates = 0, skipped = 0, errors = 0;
    bool     client_alive = true;

    // Current stats window (reset by caliper_core_live_status(reset=1)).
    std::vector<double> window_lat;          // ms
    std::vector<uint64_t> window_lat_t;      // encode time (viewer CLOCK_UPTIME_RAW ns)
    uint64_t window_frames = 0, window_updates = 0;
    uint64_t window_t0 = 0;

    // Status-window smoothing (independent of the stats window).
    std::vector<double> recent_lat;          // last 240 samples
    uint64_t fps_t0 = 0, fps_frames = 0;
    double   fps_shown = 0.0;

    std::string name_buf, err_buf;           // string storage for *_status out-params
};

namespace {

using View = CaliperLiveState::View;

bool uses_live_views(uint32_t kind) {
    return kTextureKindsThroughLiveViews ||
           (kind != CALIPER_LIVE_KIND_HEATMAP && kind != CALIPER_LIVE_KIND_IMAGE);
}

void drop_inflight(CaliperCore* core, View& v) {
    if (v.inflight) core->renderer->device_update_release(v.inflight);
    v.inflight = nullptr;
}

void remove_view(CaliperCore* core, View& v) {
    drop_inflight(core, v);
    if (v.shown) {
        if (v.typed) caliper_core_live_view_remove(core, v.name.c_str());
        else caliper_core_view_remove(core, v.name.c_str());
    }
    v.shown = false;
}

// Back to "not attached": views, source (unmaps; clears the heartbeat and the
// claims), counters. The picker settings survive.
void teardown_attached(CaliperCore* core, CaliperLiveState* L) {
    for (auto& v : L->views) remove_view(core, v);   // releases the imports, bridge still bound
    L->views.clear();
    L->source.reset();   // MTLBuffer deallocators unmap once the GPU let go
    const bool picker = L->picker_enabled;
    const std::string root = L->picker_root;
    *L = CaliperLiveState{};
    L->picker_enabled = picker;
    L->picker_root = root;
}

void destroy_state(CaliperCore* core) {
    CaliperLiveState* L = core->live;
    if (!L) return;
    teardown_attached(core, L);
    delete L;
    core->live = nullptr;
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * (double)(v.size() - 1);
    const size_t i = (size_t)pos;
    const double f = pos - (double)i;
    return i + 1 < v.size() ? v[i] * (1.0 - f) + v[i + 1] * f : v[i];
}

std::string root_of(const CaliperLiveState* L) {
    return caliper_live::data_root(L && !L->picker_root.empty() ? L->picker_root.c_str() : nullptr);
}

void reconcile(CaliperCore* core, CaliperLiveState* L, const std::vector<SourceView>& now) {
    std::vector<View> next;
    for (const auto& sv : now) {
        auto it = std::find_if(L->views.begin(), L->views.end(),
                               [&](const View& v) { return v.key == sv.key; });
        if (it != L->views.end()) {
            next.push_back(std::move(*it));
            it->key = 0;   // consumed
            it->inflight = nullptr;
            it->shown = false;
            continue;
        }
        View v;
        v.key = sv.key;
        v.name = sv.name;
        v.kind = sv.kind;
        next.push_back(std::move(v));
    }
    for (auto& v : L->views)
        if (v.key != 0) remove_view(core, v);   // gone from the source
    L->views = std::move(next);
}

// ---- the v1.2 path (HEATMAP / IMAGE while the switch is off) ----------------

// Resolve the mapping range of a heatmap snapshot: AUTO ends from the GPU
// aminmax, fixed ends from the publish metadata; non-finite stats keep the
// previous finite range; a flat range widens by +-0.5 (like _convert.py).
void resolve_range(View& v, const Snapshot& s, float* lo, float* hi) {
    const float st0 = s.stats && s.stats_count > 0 ? s.stats[0] : 0.0f;
    const float st1 = s.stats && s.stats_count > 1 ? s.stats[1] : 0.0f;
    float a = (s.flags & CALIPER_LIVE_SNAP_AUTO_VMIN) ? st0 : s.vmin;
    float b = (s.flags & CALIPER_LIVE_SNAP_AUTO_VMAX) ? st1 : s.vmax;
    if (!std::isfinite(a) || !std::isfinite(b)) {
        a = v.have_range ? v.vmin : 0.0f;
        b = v.have_range ? v.vmax : 1.0f;
    } else {
        if (a == b) { a -= 0.5f; b += 0.5f; }
        v.vmin = a; v.vmax = b; v.have_range = true;
    }
    *lo = a; *hi = b;
}

bool apply_v12(CaliperCore* core, View& v, const Snapshot& s) {
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = s.mtl_buffer;
    t.dtype = (CaliperDType)s.dtype;
    t.ndim = s.ndim;
    int64_t st = 1;
    for (int i = s.ndim - 1; i >= 0; --i) { t.shape[i] = s.shape[i]; t.strides[i] = st; st *= s.shape[i]; }
    t.device = CALIPER_DEV_METAL;
    t.stream = nullptr;

    CaliperViewDesc d{};
    d.struct_size = sizeof d;
    d.name = v.name.c_str();
    d.tensor = &t;
    d.byte_offset = s.byte_offset;
    d.flags = (s.flags & CALIPER_LIVE_SNAP_RAW) ? CALIPER_VIEW_FLAG_RAW : 0;   // never WAIT_GPU
    if (s.kind == CALIPER_LIVE_KIND_HEATMAP) {
        d.colormap = s.colormap;
        resolve_range(v, s, &d.vmin, &d.vmax);
    } else {
        d.colormap = 0; d.vmin = 0.0f; d.vmax = 1.0f;
    }
    if (!caliper_core_view_tensor(core, &d)) {
        v.error = caliper_core_last_error(core);
        return false;
    }
    drop_inflight(core, v);
    v.inflight = core->renderer->device_update_retain_last();
    return true;
}

// ---- the typed path (caliper/live_views.h) -------------------------------------

uint32_t apply_flags(uint32_t snap) {
    uint32_t f = 0;
    if (snap & CALIPER_LIVE_SNAP_AUTO_VMIN) f |= CALIPER_LIVE_APPLY_AUTO_VMIN;
    if (snap & CALIPER_LIVE_SNAP_AUTO_VMAX) f |= CALIPER_LIVE_APPLY_AUTO_VMAX;
    if (snap & CALIPER_LIVE_SNAP_RAW) f |= CALIPER_LIVE_APPLY_RAW;
    return f;
}

bool apply_typed(CaliperCore* core, View& v, const Snapshot& s) {
    CaliperTensor data{};
    data.struct_size = sizeof data;
    data.data = s.mtl_buffer;
    data.dtype = (CaliperDType)s.dtype;
    data.ndim = s.ndim;
    int64_t st = 1;
    for (int i = s.ndim - 1; i >= 0; --i) { data.shape[i] = s.shape[i]; data.strides[i] = st; st *= s.shape[i]; }
    data.device = CALIPER_DEV_METAL;
    data.stream = nullptr;

    CaliperTensor aux{};
    if (s.aux_count > 0) {
        aux.struct_size = sizeof aux;
        aux.data = s.mtl_buffer;
        aux.dtype = CALIPER_DT_F32;
        aux.ndim = 1;
        aux.shape[0] = s.aux_count;
        aux.strides[0] = 1;
        aux.device = CALIPER_DEV_METAL;
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
    ls.flags = apply_flags(s.flags);
    ls.series_total = s.series_total;
    ls.points_dims = s.points_dims;
    ls.label_count = s.label_count;
    ls.seq = s.seq;
    ls.stamp_ns = s.stamp_ns;
    if (!caliper_core_live_apply(core, &ls)) {
        v.error = caliper_core_last_error(core);
        return false;
    }
    return true;
}

// The one place that decides how a snapshot is drawn.
bool apply_snapshot(CaliperCore* core, View& v, const Snapshot& s) {
    const bool typed = uses_live_views(s.kind);
    if (v.shown && v.typed != typed) remove_view(core, v);
    const bool ok = typed ? apply_typed(core, v, s) : apply_v12(core, v, s);
    if (!ok) return false;
    v.error.clear();
    v.shown = true;
    v.typed = typed;
    return true;
}

// "Has the GPU read of what this view was given retired?" (never waits):
// texture/copy kinds: the last apply's read; POINTS: the superseded payload's
// last frame (the view layer answers).
bool read_done(CaliperCore* core, View& v) {
    if (!v.shown) return true;
    if (v.typed) return caliper_core_live_view_read_done(core, v.name.c_str()) == 1;
    if (!v.inflight) return true;
    if (!core->renderer->device_update_is_done(v.inflight)) return false;
    drop_inflight(core, v);
    return true;
}

// ---- attach -----------------------------------------------------------------

// A session file under the root naming `token` (alive or not), if any.
bool find_session(const std::string& root, const std::string& token, SessionEntry* out) {
    std::vector<SessionEntry> all;
    std::string err;
    if (!caliper_live::list_sessions(root, &all, &err)) return false;
    for (auto& e : all)
        if (e.session == token) { *out = e; return true; }
    return false;
}

// Try to open the source of L->session. true: open (or still pending).
bool try_open(CaliperCore* core, CaliperLiveState* L, std::string* err,
              caliper_live::OpenFail* out_fail = nullptr) {
    caliper_live::OpenFail fail = caliper_live::OpenFail::None;
    std::unique_ptr<Source> src =
        caliper_live::open_shm_source(L->session, core->renderer->native_device(), err, &fail);
    if (src) {
        L->source = std::move(src);
        L->pending = false;
        L->pending_why.clear();
        L->pid = L->source->producer_pid();
        L->script = L->source->script();
        L->start_ms = L->source->start_unix_ms();
        L->client_alive = L->source->producer_alive();
        L->window_t0 = L->fps_t0 = caliper_live::now_ns();
        core->log(1, "live: attached to session '" + L->session + "' (" + L->script + ", client pid " +
                         std::to_string(L->pid) + ")");
        return true;
    }
    if (fail == caliper_live::OpenFail::Starting || (L->pending && fail == caliper_live::OpenFail::Missing)) {
        L->pending = true;   // (a pending attach keeps waiting while the name is absent)
        L->pending_why = *err;
        return true;
    }
    if (out_fail) *out_fail = fail;
    return false;
}

int attach_impl(CaliperCore* core, const std::string& token, const char* root = nullptr) {
    if (!caliper_live::valid_session(token.c_str())) {
        core->fail("live_attach: session must be 1..20 chars of [A-Za-z0-9._-]");
        return 0;
    }
    if (!core->canvas_ready || !core->renderer) {
        core->fail("live_attach: attach a canvas first (the renderer's device exists only after attach_canvas)");
        return 0;
    }
    if (core->live && core->live->attached) {
        core->fail("live_attach: already attached to '" + core->live->session + "'; detach first");
        return 0;
    }
    if (!core->renderer->native_device()) {
        core->fail("live_attach: the active renderer has no Metal device (live views need Metal)");
        return 0;
    }
    if (!core->live) core->live = new CaliperLiveState();
    CaliperLiveState* L = core->live;
    L->session = token;
    L->attached = true;
    std::string err;
    caliper_live::OpenFail fail = caliper_live::OpenFail::None;
    if (try_open(core, L, &err, &fail)) {
        if (L->pending) L->retry_ns = caliper_live::now_ns() + kPendingRetryNs;
        core->last_error.clear();
        return 1;
    }
    // No control segment yet: a session file naming the token with a running
    // writer means the session is starting; wait for it.
    SessionEntry e;
    if (fail == caliper_live::OpenFail::Missing &&
        find_session(root && root[0] ? caliper_live::data_root(root) : root_of(L), token, &e) &&
        (e.alive || e.starting)) {
        L->pending = true;
        L->pending_why = err;
        L->pid = e.pid;
        L->script = e.script;
        L->start_ms = e.start_unix_ms;
        L->retry_ns = caliper_live::now_ns() + kPendingRetryNs;
        core->last_error.clear();
        return 1;
    }
    teardown_attached(core, L);
    if (!L->picker_enabled) destroy_state(core);   // a picker stays up
    core->fail("live_attach: " + err);
    return 0;
}

// ---- per-poll work ----------------------------------------------------------

void refresh_picker(CaliperLiveState* L, bool force) {
    const uint64_t t = caliper_live::now_ns();
    if (!force && L->sessions_t && t - L->sessions_t < kPickerRefreshNs) return;
    L->sessions_t = t;
    std::string err;
    if (!caliper_live::list_sessions(root_of(L), &L->sessions, &err)) L->picker_error = err;
    else L->picker_error.clear();
}

int poll_attached(CaliperCore* core, CaliperLiveState* L) {
    ++L->polls;
    if (!L->source) {
        // Pending: retry the open; a writer that died while starting ends it.
        const uint64_t t = caliper_live::now_ns();
        if (t >= L->retry_ns) {
            L->retry_ns = t + kPendingRetryNs;
            std::string err;
            if (!try_open(core, L, &err)) L->pending_why = err;
            if (!L->source && L->pid && !caliper_live::pid_alive(L->pid)) L->client_alive = false;
        }
        if (!L->source) return 0;
    }
    L->source->heartbeat();
    L->client_alive = L->source->producer_alive();

    std::vector<SourceView> now;
    if (L->source->sync(&now)) reconcile(core, L, now);

    int applied = 0;
    for (auto& v : L->views) {
        const bool series = v.kind == CALIPER_LIVE_KIND_SERIES;
        const bool points = v.kind == CALIPER_LIVE_KIND_POINTS;
        if (points) {
            // The previous claim stays until its last frame retired; at most
            // two claims, so nothing new until then.
            if (v.prev_claimed) {
                if (!read_done(core, v)) continue;
                L->source->release_prev(v.key);
                v.prev_claimed = false;
            }
        } else if (!series && !read_done(core, v)) {
            continue;   // the last apply's GPU read is still running: next frame
        }
        Snapshot s;
        if (!L->source->take(v.key, v.seq, &s)) continue;
        if (!points && !series) L->source->release_prev(v.key);   // one claim for read-once kinds
        const uint64_t skipped = v.seq && s.seq > v.seq + 1 ? s.seq - v.seq - 1 : 0;
        if (!apply_snapshot(core, v, s)) {
            L->source->unclaim(v.key);
            ++L->errors;
            v.seq = s.seq;   // do not retry the same snapshot every frame
            continue;
        }
        if (points) {
            v.prev_claimed = true;
            if (read_done(core, v)) {   // nothing superseded still drawing
                L->source->release_prev(v.key);
                v.prev_claimed = false;
            }
        }
        const uint64_t t = caliper_live::now_ns();
        const double lat = t > s.stamp_ns ? (double)(t - s.stamp_ns) * 1e-6 : 0.0;
        v.latency_ms_last = lat;
        v.seq = s.seq;
        v.series_total = s.series_total;
        v.points_dims = s.points_dims;
        v.label_count = s.label_count;
        v.stats_count = s.stats_count;
        v.slot = s.slot;
        ++v.updates; ++v.window_updates;
        v.skipped += skipped;
        ++L->updates; ++L->window_updates;
        L->skipped += skipped;
        L->window_lat.push_back(lat);
        L->window_lat_t.push_back(t);
        L->recent_lat.push_back(lat);
        if (L->recent_lat.size() > 240) L->recent_lat.erase(L->recent_lat.begin(), L->recent_lat.begin() + 120);
        L->source->drew(v.key, s.seq, skipped);
        ++applied;
    }
    return applied;
}

const char* kind_name(uint32_t k) {
    switch (k) {
        case CALIPER_LIVE_KIND_HEATMAP: return "heatmap";
        case CALIPER_LIVE_KIND_IMAGE: return "image";
        case CALIPER_LIVE_KIND_POINTS: return "points";
        case CALIPER_LIVE_KIND_SERIES: return "series";
        case CALIPER_LIVE_KIND_HISTOGRAM: return "histogram";
        default: return "?";
    }
}

std::string clock_of(uint64_t unix_ms) {
    if (!unix_ms) return "-";
    const time_t s = (time_t)(unix_ms / 1000);
    struct tm tmv{};
    localtime_r(&s, &tmv);
    char b[32];
    std::strftime(b, sizeof b, "%H:%M:%S", &tmv);
    return b;
}

std::string age_of(uint64_t unix_ms) {
    if (!unix_ms) return "";
    const uint64_t now = caliper_live::unix_ms();
    const uint64_t s = now > unix_ms ? (now - unix_ms) / 1000 : 0;
    char b[32];
    if (s < 60) std::snprintf(b, sizeof b, "%llus ago", (unsigned long long)s);
    else if (s < 3600) std::snprintf(b, sizeof b, "%llum ago", (unsigned long long)(s / 60));
    else std::snprintf(b, sizeof b, "%lluh ago", (unsigned long long)(s / 3600));
    return b;
}

void draw_picker(CaliperCore* core, CaliperLiveState* L, float lw, float lh) {
    ImGui::SetNextWindowPos(ImVec2(lw * 0.5f, lh * 0.4f), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(std::min(720.0f, lw - 16.0f), 320.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Live sessions###caliper.live.picker", nullptr, ImGuiWindowFlags_NoCollapse)) {
        ImGui::TextDisabled("%s/live  (refreshed every second)", root_of(L).c_str());
        if (!L->picker_error.empty()) ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "%s", L->picker_error.c_str());
        if (L->sessions.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("No live sessions. A script opens one on its first caliper.show(); "
                               "it appears here while it runs.");
        } else if (ImGui::BeginTable("##sessions", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                                                           ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Script", ImGuiTableColumnFlags_WidthStretch, 3.0f);
            ImGui::TableSetupColumn("PID", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("Started", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("Views", ImGuiTableColumnFlags_WidthStretch, 0.8f);
            ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableHeadersRow();
            for (const auto& e : L->sessions) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(e.script.c_str());
                ImGui::TableNextColumn(); ImGui::Text("%d", e.pid);
                ImGui::TableNextColumn(); ImGui::Text("%s (%s)", clock_of(e.start_unix_ms).c_str(),
                                                      age_of(e.start_unix_ms).c_str());
                ImGui::TableNextColumn(); ImGui::Text("%d", e.view_count);
                ImGui::TableNextColumn();
                const bool compatible = e.protocol == (int32_t)CALIPER_LIVE_PROTOCOL_VERSION;
                if (!compatible) ImGui::TextDisabled("protocol v%d", e.protocol);
                else if (e.starting) ImGui::TextUnformatted("starting");
                else if (!e.alive) ImGui::TextDisabled("closing");
                else if (e.viewer_pid) ImGui::Text("viewed by %d", e.viewer_pid);
                else ImGui::TextUnformatted("running");
                ImGui::TableNextColumn();
                const bool can = compatible && (e.alive || e.starting) && !e.viewer_pid;
                ImGui::BeginDisabled(!can);
                ImGui::PushID(e.session.c_str());
                if (ImGui::SmallButton("Attach")) L->attach_request = e.session;
                ImGui::PopID();
                ImGui::EndDisabled();
            }
            ImGui::EndTable();
        }
    }
    ImGui::End();
    (void)core;
}

void draw_status(CaliperCore* core, CaliperLiveState* L, float lw, float lh) {
    // Fixed first size, anchored at the bottom-right corner (the host's input
    // probe aims at it: kStatusW x kStatusH points, 8 pt from the corner).
    constexpr float kStatusW = 520.0f, kStatusH = 250.0f;
    ImGui::SetNextWindowPos(ImVec2(lw - 8.0f, lh - 8.0f), ImGuiCond_FirstUseEver, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowSize(ImVec2(kStatusW, kStatusH), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.85f);
    if (ImGui::Begin("live###caliper.live.status", nullptr, ImGuiWindowFlags_NoFocusOnAppearing)) {
        static const char kSpin[] = "|/-\\";
        ImGui::Text("%s  pid %d  session %s", L->script.empty() ? "?" : L->script.c_str(), L->pid,
                    L->session.c_str());
        if (L->pending && L->client_alive) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "waiting for the session to start");
        } else if (!L->client_alive) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f),
                               "SESSION ENDED: the script is gone; showing the last frames received");
        } else {
            ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f), "running");
        }
        ImGui::Text("frame %llu  %c  fps %.1f  views %d", (unsigned long long)L->frames,
                    kSpin[(L->frames / 8) % 4], L->fps_shown, (int)L->views.size());
        ImGui::Text("updates %llu  skipped %llu  errors %llu", (unsigned long long)L->updates,
                    (unsigned long long)L->skipped, (unsigned long long)L->errors);
        ImGui::Text("step-to-screen ms  median %.2f  p95 %.2f  (last %zu)",
                    percentile(L->recent_lat, 0.5), percentile(L->recent_lat, 0.95), L->recent_lat.size());
        for (const auto& v : L->views) {
            ImGui::Text("  %s (%s): seq %llu  lat %.1f ms  dropped@client %llu%s%s", v.name.c_str(),
                        kind_name(v.kind), (unsigned long long)v.seq, v.latency_ms_last,
                        (unsigned long long)(L->source ? L->source->producer_dropped(v.key) : 0),
                        v.error.empty() ? "" : "  ERROR: ", v.error.c_str());
        }
        if (ImGui::Button("Detach")) L->detach_requested = true;
        ImGui::SameLine();
        // Pause freezes the display only: applies keep being accepted (and
        // dropped), so claims progress and training is never slowed.
        const bool paused = caliper_core_live_paused(core) != 0;
        if (ImGui::Button(paused ? "Resume" : "Pause")) caliper_core_live_set_paused(core, paused ? 0 : 1);
        ImGui::SameLine();
        ImGui::TextDisabled("hovered: %s  focused: %s", ImGui::IsWindowHovered() ? "yes" : "no",
                            ImGui::IsWindowFocused() ? "yes" : "no");
    }
    ImGui::End();
    (void)core;
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
extern "C" int caliper_core_live_attach(CaliperCore* core, const char* session) {
    if (!core) return 0;
    if (!session) {
        core->fail("live_attach: session must be 1..20 chars of [A-Za-z0-9._-]");
        return 0;
    }
    return attach_impl(core, session);
}

extern "C" int caliper_core_live_attach_pid(CaliperCore* core, int32_t pid, const char* data_root) {
    if (!core) return 0;
    std::vector<SessionEntry> all;
    std::string err;
    if (!caliper_live::list_sessions(caliper_live::data_root(data_root), &all, &err)) {
        core->fail("live_attach_pid: " + err);
        return 0;
    }
    for (const auto& e : all)
        if (e.pid == pid) return attach_impl(core, e.session, data_root);
    core->fail("live_attach_pid: no live session of pid " + std::to_string(pid) + " under " +
               caliper_live::sessions_dir(caliper_live::data_root(data_root)));
    return 0;
}

extern "C" void caliper_core_live_detach(CaliperCore* core) {
    if (core) destroy_state(core);
}

extern "C" int caliper_core_live_picker(CaliperCore* core, int enable, const char* data_root) {
    if (!core) return 0;
    if (enable) {
        if (!core->live) core->live = new CaliperLiveState();
        core->live->picker_enabled = true;
        core->live->picker_root = data_root ? data_root : "";
        refresh_picker(core->live, /*force=*/true);
        return 1;
    }
    if (core->live) {
        core->live->picker_enabled = false;
        if (!core->live->attached) destroy_state(core);
    }
    return 1;
}

extern "C" int caliper_core_live_poll(CaliperCore* core) {
    if (!core || !core->live) return -1;
    CaliperLiveState* L = core->live;
    if (L->attached && L->detach_requested) {
        // Detach clicked in the status window (performed here, between frames,
        // never inside the ImGui frame that drew the views).
        core->log(1, "live: detached from session '" + L->session + "'");
        if (L->picker_enabled) {
            teardown_attached(core, L);
            refresh_picker(L, true);
        } else {
            destroy_state(core);
        }
        return -1;
    }
    if (!L->attached) {
        if (!L->picker_enabled) return -1;
        refresh_picker(L, false);
        if (L->attach_request.empty()) return -1;
        const std::string token = std::move(L->attach_request);
        L->attach_request.clear();
        if (!attach_impl(core, token)) {
            L->picker_error = caliper_core_last_error(core);
            return -1;
        }
    }
    return poll_attached(core, L);
}

extern "C" int caliper_core_live_status(CaliperCore* core, CaliperLiveStatus* o, int reset) {
    using S = CaliperLiveStatus;
    if (!core) return 0;
    if (!o || o->struct_size < sizeof(S)) {
        core->fail("live_status: null/short CaliperLiveStatus");
        return 0;
    }
    const size_t sz = o->struct_size;
    const bool v2 = true;
    *o = S{};
    o->struct_size = sz;
    CaliperLiveState* L = core->live;
    if (v2) {
        o->session = "";
        o->script = "";
    }
    if (!L) return 1;   // attached = 0
    if (v2) {
        o->picker = L->picker_enabled ? 1 : 0;
        o->sessions_listed = (int32_t)L->sessions.size();
    }
    if (!L->attached) return 1;
    o->attached = 1;
    o->client_pid = L->pid;
    o->client_alive = L->client_alive ? 1 : 0;
    o->views = (int32_t)L->views.size();
    o->frames = L->frames;
    o->polls = L->polls;
    o->updates = L->updates;
    o->skipped = L->skipped;
    o->errors = L->errors;
    if (L->source)
        for (const auto& v : L->views) o->client_dropped += L->source->producer_dropped(v.key);
    o->latency_samples = L->window_lat.size();
    o->latency_ms_median = percentile(L->window_lat, 0.5);
    o->latency_ms_p95 = percentile(L->window_lat, 0.95);
    o->latency_ms_max = L->window_lat.empty() ? 0.0
                        : *std::max_element(L->window_lat.begin(), L->window_lat.end());
    const uint64_t t = caliper_live::now_ns();
    const double secs = L->window_t0 ? (double)(t - L->window_t0) * 1e-9 : 0.0;
    o->fps = secs > 0 ? (double)L->window_frames / secs : 0.0;
    o->window_frames = L->window_frames;
    o->window_updates = L->window_updates;
    if (v2) {
        o->session = L->session.c_str();
        o->script = L->script.c_str();
        o->start_unix_ms = L->start_ms;
        o->pending = L->pending ? 1 : 0;
    }
    if (reset) {
        L->window_lat.clear();
        L->window_lat_t.clear();
        L->window_frames = L->window_updates = 0;
        L->window_t0 = t;
        for (auto& v : L->views) v.window_updates = 0;
    }
    return 1;
}

extern "C" size_t caliper_core_live_window_latencies(CaliperCore* core, uint64_t* t_ns,
                                                    double* lat_ms, size_t cap) {
    if (!core || !core->live) return 0;
    const CaliperLiveState* L = core->live;
    const size_t n = (t_ns || lat_ms) ? std::min(cap, L->window_lat.size()) : L->window_lat.size();
    for (size_t i = 0; i < n && (t_ns || lat_ms); ++i) {
        if (t_ns) t_ns[i] = L->window_lat_t[i];
        if (lat_ms) lat_ms[i] = L->window_lat[i];
    }
    return n;
}

extern "C" int caliper_core_live_view_status(CaliperCore* core, int index, CaliperLiveViewStatus* o) {
    using S = CaliperLiveViewStatus;
    if (!core) return 0;
    if (!o || o->struct_size < sizeof(S)) {
        core->fail("live_view_status: null/short CaliperLiveViewStatus");
        return 0;
    }
    CaliperLiveState* L = core->live;
    if (!L || !L->attached || index < 0 || index >= (int)L->views.size()) return 0;
    const auto& v = L->views[(size_t)index];
    L->name_buf = v.name;
    L->err_buf = v.error;
    o->name = L->name_buf.c_str();
    o->kind = v.kind;
    o->seq = v.seq;
    o->updates = v.updates;
    o->window_updates = v.window_updates;
    o->skipped = v.skipped;
    o->client_dropped = L->source ? L->source->producer_dropped(v.key) : 0;
    o->latency_ms_last = v.latency_ms_last;
    o->error = L->err_buf.c_str();
    {
        o->series_total = v.series_total;
        o->points_dims = v.points_dims;
        o->label_count = v.label_count;
        o->stats_count = v.stats_count;
        o->slot = v.slot == CALIPER_LIVE_SLOT_NONE ? -1 : (int32_t)v.slot;
        o->apply = !v.shown ? "" : (v.typed ? "live_views" : "v1.2");
    }
    return 1;
}

// ---------------------------------------------------------------------------
// embed_core.cpp hooks
// ---------------------------------------------------------------------------

// Inside the ImGui frame, after the v1.2 views (which include the live views).
void caliper_live_frame(CaliperCore* core) {
    if (!core || !core->live) return;
    CaliperLiveState* L = core->live;
    if (L->attached) {
        ++L->frames;
        ++L->window_frames;
        ++L->fps_frames;
        const uint64_t t = caliper_live::now_ns();
        if (L->fps_t0 == 0) L->fps_t0 = t;
        if (t - L->fps_t0 >= 500000000ull) {   // refresh the shown fps twice a second
            L->fps_shown = (double)L->fps_frames / ((double)(t - L->fps_t0) * 1e-9);
            L->fps_t0 = t;
            L->fps_frames = 0;
        }
    }
    if (core->canvas_mode == caliper_host::HostRenderer::CANVAS_OFFSCREEN) return;   // tests: no chrome

    const float lw = (float)core->canvas_w / core->scale;
    const float lh = (float)core->canvas_h / core->scale;
    if (L->attached) draw_status(core, L, lw, lh);
    else if (L->picker_enabled) draw_picker(core, L, lw, lh);
}

// Called by caliper_core_shutdown before the v1.2 views are destroyed (the
// bridge still has its renderer, so imports release cleanly).
void caliper_live_shutdown(CaliperCore* core) {
    if (core) destroy_state(core);
}
