#pragma once
// caliper.export.v1 helpers (Rung E). The service THUNKS + sequence bookkeeping
// live in host_services.cpp next to bridge()/g_renderer (the composition reaches
// geometry.v1_3's create_view_ex + draw_primitives + the renderer readback); the
// heavy, isolated pieces — deterministic PNG encode (stb_image_write, its sole
// STB_IMAGE_WRITE_IMPLEMENTATION TU), atomic file writes, and the provenance
// sidecar builder — live here so stb stays in one TU and the sidecar format is
// white-box testable with an injected timestamp (the golden test).
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_host {

// Every field the §3 sidecar carries, ALL injectable so the golden test can pin
// the exact JSON with a fixed timestamp/commit/backend (mirrors the C2 report
// pattern). view16/proj16 are 16-float column-major camera matrices (as
// submitted). state_json is copied VERBATIM as the "state" value; nullptr → the
// JSON literal null. colormaps are the distinct colormap ids used by the draws,
// in first-seen order.
struct ExportProvenance {
    std::string version;
    std::string git_commit;
    std::string backend;
    std::string platform;
    std::string timestamp_utc;
    uint32_t    width = 0;
    uint32_t    height = 0;
    uint32_t    clear_rgba = 0;
    uint32_t    draw_count = 0;
    const float* view16 = nullptr;
    const float* proj16 = nullptr;
    std::vector<int32_t> colormaps;
    const char* state_json = nullptr;   // nullable, verbatim
    uint32_t    frame_count = 0;        // sequence sidecar only (0 => omitted)
    bool        is_sequence = false;    // emit "frame_count" instead of "state"?
};

// Build the sidecar JSON (deterministic layout, trailing newline). Floats print
// with %.9g (round-trips float32). Pure — no clock, no fs.
std::string export_build_sidecar_json(const ExportProvenance& p);

// ISO-8601 UTC timestamp of "now" (e.g. "2026-07-12T18:04:22Z").
std::string export_utc_timestamp();

// Compile-time platform tag ("macos" / "windows" / "linux" / "unknown").
const char* export_platform_string();

// Encode tightly-packed top-down RGBA8 to a PNG, ATOMICALLY: write a sibling
// temp file then rename onto `path`. Returns false (leaving `path` untouched,
// creating nothing at `path`) if the encode or rename fails — the filesystem
// half of refusal purity. Deterministic (stb PNG's fixed filter/zlib).
bool export_write_png_atomic(const std::string& path,
                             const uint8_t* rgba, uint32_t w, uint32_t h);

// Write UTF-8 text atomically (temp + rename); same refusal-purity guarantee.
bool export_write_text_atomic(const std::string& path, const std::string& text);

}  // namespace caliper_host
