#pragma once
// Minimal SHA-256 (FIPS 180-4) for content-addressing artifacts. Vendored
// implementation written from the public specification — no external
// dependency, no license encumbrance. One-shot API only; host-internal.
#include <cstddef>
#include <cstdint>
#include <string>

namespace caliper_host {

// Returns the lowercase 64-char hex digest of `len` bytes at `data`.
std::string sha256_hex(const void* data, size_t len);

}  // namespace caliper_host
