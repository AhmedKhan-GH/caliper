#pragma once
// The host's PNG writers (export_service.cpp, the single stb_image_write TU):
// an atomic PNG file write for the shot tools and the spec runtime, and an
// in-memory encode for Caliper API's capture. Not an applet service.
#include <cstdint>
#include <string>

namespace caliper_host {

// The writer below takes an optional `why`: on a false return it says what
// the filesystem refused (e.g. "cannot rename <tmp> onto <path>: Is a
// directory"). Untouched on success.

// Encode tightly-packed top-down RGBA8 to a PNG, ATOMICALLY: write a sibling
// temp file then rename onto `path`. Returns false (leaving `path` untouched,
// creating nothing at `path`) if the encode or rename fails — the filesystem
// half of refusal purity. Deterministic (stb PNG's fixed filter/zlib).
bool export_write_png_atomic(const std::string& path,
                             const uint8_t* rgba, uint32_t w, uint32_t h,
                             std::string* why = nullptr);

// Encode tightly-packed top-down RGBA8 to PNG bytes in memory, with the
// encoder of export_write_png_atomic (Caliper API's capture). False when there
// are no pixels or the encoder fails; `out` is replaced.
bool export_encode_png(const uint8_t* rgba, uint32_t w, uint32_t h, std::string* out);

}  // namespace caliper_host
