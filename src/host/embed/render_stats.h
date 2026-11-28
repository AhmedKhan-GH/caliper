// render_stats.h: the copy counters and frame statistics of a core
// (host-internal; docs/instruments/rendering.md, "Copy Counters and Frame
// Statistics"). Step 0 of the renderer core's migration adds them, so that the
// copies each route makes on the way to pixels, the frame times and the GPU
// times can be compared before and after every later step.
//
// What counts as a copy is defined once, at HostRenderer::note_copy
// (renderer/host_renderer.h): an upload of host bytes, a host copy into a
// staging buffer, a copy inside the GPU, or a readback of a payload. The
// renderer keeps the core-wide total; this file keeps the per-view counters
// (keyed by view name, so a heatmap's typed live view and its embed v1.2 view
// share one entry) and the per-frame window. Frame thread only, like every
// view call. The viewer's --stats line, the Python views' info and the tests
// read them through the functions below.
#pragma once

#include "embed/embed_core.h"
#include "renderer/host_renderer.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace caliper_stats {

using CopyCount = caliper_host::HostRenderer::CopyCount;

// One view's copies (CaliperCore::CopyStats::View): `total` since the first
// view of that name on this core, `last` since the start of its last update (an
// apply or a view_tensor call). Copies a view makes while drawing, such as
// staging a selection, add to both.
using ViewCopies = CaliperCore::CopyStats::View;

// The core's copies per frame: what the renderer counted between the ends of
// consecutive frames (caliper_core_frame), and over the window since the last
// reset of frame_copies().
struct FrameCopies {
    uint64_t  frames = 0;          // frames ended since the canvas came up
    CopyCount last_frame;          // copies made during the last frame
    uint64_t  window_frames = 0;   // frames ended in the window
    CopyCount window;              // copies made in the window
};

namespace detail {
inline CopyCount minus(CopyCount a, CopyCount b) { return {a.copies - b.copies, a.bytes - b.bytes}; }
inline void add(CopyCount* a, CopyCount b) {
    a->copies += b.copies;
    a->bytes += b.bytes;
}
inline CopyCount renderer_total(const CaliperCore* core) {
    return core && core->renderer ? core->renderer->copy_count() : CopyCount{};
}
}  // namespace detail

// ---- write side (the views) ----------------------------------------------------

// A view update starts: its `last` count restarts at zero.
inline void begin_update(CaliperCore* core, const std::string& name) {
    if (core) core->copy_stats.views[name].last = {};
}
// The renderer's copies since `before` (a renderer_total() taken at the start
// of the update) belong to view `name`.
inline void add_since(CaliperCore* core, const std::string& name, CopyCount before) {
    if (!core) return;
    const CopyCount d = detail::minus(detail::renderer_total(core), before);
    if (d.copies == 0 && d.bytes == 0) return;
    ViewCopies& v = core->copy_stats.views[name];
    detail::add(&v.total, d);
    detail::add(&v.last, d);
}
// Count one copy of `bytes` made for view `name`, in the renderer's total and
// in the view's.
inline void note_copy(CaliperCore* core, const std::string& name, uint64_t bytes) {
    if (!core || !core->renderer) return;
    core->renderer->note_copy(bytes);
    ViewCopies& v = core->copy_stats.views[name];
    detail::add(&v.total, CopyCount{1, bytes});
    detail::add(&v.last, CopyCount{1, bytes});
}
inline CopyCount renderer_total(const CaliperCore* core) { return detail::renderer_total(core); }

// The end of a frame (caliper_core_frame, after the frame was committed).
inline void end_frame(CaliperCore* core) {
    if (!core || !core->renderer) return;
    CaliperCore::CopyStats& s = core->copy_stats;
    const CopyCount now = core->renderer->copy_count();
    s.last_frame = detail::minus(now, s.at_frame_end);
    s.at_frame_end = now;
    ++s.frames;
    ++s.window_frames;
}

// ---- read side ----------------------------------------------------------------------

// The counters of view `name`; false when no view of that name ever counted
// or updated on this core.
inline bool view_copies(const CaliperCore* core, const char* name, ViewCopies* out) {
    if (!core || !name || !out) return false;
    const auto it = core->copy_stats.views.find(name);
    if (it == core->copy_stats.views.end()) return false;
    *out = it->second;
    return true;
}

// The core's copies per frame. reset_window starts a new window after
// filling `out`.
inline bool frame_copies(CaliperCore* core, FrameCopies* out, bool reset_window) {
    if (!core || !core->renderer || !out) return false;
    CaliperCore::CopyStats& s = core->copy_stats;
    out->frames = s.frames;
    out->last_frame = s.last_frame;
    out->window_frames = s.window_frames;
    out->window = detail::minus(s.at_frame_end, s.window_start);
    if (reset_window) {
        s.window_start = s.at_frame_end;
        s.window_frames = 0;
    }
    return true;
}

// The renderer's frame-time and GPU-time statistics (CALIPER_METAL_STATS on
// Metal); false when the backend records none or they are off.
inline bool frame_stats(CaliperCore* core, caliper_host::HostRenderer::FrameStats* out, bool reset_window) {
    return core && core->renderer && out && core->renderer->frame_stats(out, reset_window);
}

}  // namespace caliper_stats
