#pragma once
// SHA-256 (FIPS 180-4) for the command layer: the journal names a redacted
// argument by its digest, and an undo record can keep the digest of the text
// its command wrote, to notice a change made outside the command layer. The
// result is 64 lower-case hexadecimal characters. Does not throw.
#include <string>
#include <string_view>

namespace caliper_builder {

std::string command_sha256_hex(std::string_view data);

}  // namespace caliper_builder
