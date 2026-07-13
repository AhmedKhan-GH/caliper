#pragma once
// Shared, torch-free / UI-free logic behind the "Export figure (4K)" + "Record
// 10 s" affordances (export.v1 E2). Both mesh_scope and twin_scope drive the
// export ABI the SAME way: figure = one view_png; record = a wall-clock-paced
// begin_sequence/frame loop. The pure parts — the frame budget, the pacing
// predicate, and the on-disk path assembly — live here so they are unit-tested
// once (rides the fast caliper_tests suite) rather than twice inside the two
// UI translation units.
//
// Threading note (why the applets run the export INLINE on the frame thread and
// not on a background caliper.jobs worker): the host renderer's texture/import
// tables (metal_renderer.mm: textures_, imported_, next_id_) are UNSYNCHRONIZED,
// and caliper.jobs runs fns on real std::threads. A background export would
// race those tables against the frame thread's own draw + the host's ImGui
// present. Doing the capture inline — after the applet has already built its
// draw arrays for this frame — reuses the EXACT immediate-mode arrays and the
// same worker snapshot the frame drew with, races nothing, and keeps the
// live view watchable while recording.
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string>

namespace caliper::exportui {

// Filename-safe local timestamp "YYYYmmddTHHMMSS" for the artifact stem. Not
// pure (reads the wall clock) and not unit-tested; the path-assembly helpers
// below take the stamp as an argument so THEY stay testable.
inline std::string now_stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02dT%02d%02d%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

// Number of frames in a `seconds`-long clip at `fps` (10 s @ 30 fps = 300).
// round-to-nearest; any positive duration yields at least one frame; a
// non-positive duration or fps yields zero (nothing to record).
inline uint32_t frame_budget(double seconds, double fps) {
    if (seconds <= 0.0 || fps <= 0.0) return 0u;
    const double n = seconds * fps + 0.5;
    return n < 1.0 ? 1u : static_cast<uint32_t>(n);
}

// Wall-clock pacing for the record loop: a capture is due when at least one
// frame interval (1000/fps ms) has elapsed since the previous capture. The
// first frame (`first == true`) is always due so a record starts immediately.
inline bool capture_due(int64_t now_ms, int64_t last_ms, double fps, bool first) {
    if (first) return true;
    if (fps <= 0.0) return false;
    const double interval_ms = 1000.0 / fps;
    return static_cast<double>(now_ms - last_ms) >= interval_ms;
}

// Artifacts live under "<root>/exports/". An empty root (host vends no
// data_dir) degrades to "." so the path stays writable-relative rather than
// rooting at "/exports/...".
inline std::string exports_dir(const std::string& root) {
    const std::string base = root.empty() ? std::string(".") : root;
    return base + "/exports";
}

// "<root>/exports/<stem>_<stamp>.png" — the figure PNG (its .json sidecar lands
// alongside, written by the export service).
inline std::string figure_png_path(const std::string& root,
                                    const std::string& stem,
                                    const std::string& stamp) {
    return exports_dir(root) + "/" + stem + "_" + stamp + ".png";
}

// "<root>/exports/<stem>_<stamp>" — the directory a record fills with
// frame_%06u.png + sequence.json.
inline std::string record_dir_path(const std::string& root,
                                   const std::string& stem,
                                   const std::string& stamp) {
    return exports_dir(root) + "/" + stem + "_" + stamp;
}

}  // namespace caliper::exportui
