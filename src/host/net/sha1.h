#pragma once
// Minimal SHA-1 (FIPS 180-4) and base64 (RFC 4648) for the WebSocket opening
// handshake (RFC 6455 section 4.2.2: Sec-WebSocket-Accept is the base64 of the
// SHA-1 of the client's key and a fixed GUID). Written from the public
// specifications, like sha256.h; one-shot API only; host-internal. SHA-1 is
// used here only because RFC 6455 names it, never for security.
#include <cstddef>
#include <cstdint>
#include <string>

namespace caliper_net {

// The 20-byte SHA-1 digest of `len` bytes at `data`.
void sha1(const void* data, size_t len, uint8_t out[20]);

// Standard base64 with padding.
std::string base64_encode(const void* data, size_t len);
// Decodes standard base64 with padding; false on any other character, a wrong
// length or bad padding.
bool base64_decode(const std::string& in, std::string* out);

}  // namespace caliper_net
