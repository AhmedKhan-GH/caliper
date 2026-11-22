#pragma once
// Internal to the mcp module: small codecs and checks that several of its
// files share. Base64 (RFC 4648, with padding), the PNG signature and size, a
// nesting check that runs before a JSON text is parsed, and a bounded file
// read. No function here throws.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace caliper_builder::mcp::detail {

std::string base64_encode(std::string_view bytes);
// False on a character outside the alphabet, bad padding or a bad length.
bool base64_decode(std::string_view text, std::string* out);

// True when `bytes` starts with the PNG signature and an IHDR chunk; fills
// the image's width and height.
bool png_size(std::string_view bytes, std::uint32_t* width, std::uint32_t* height);

// True when no array or object in the JSON text nests deeper than
// `max_depth`. Brackets inside strings are skipped. It does not check that the
// text is JSON; the parser does that after.
bool json_depth_ok(std::string_view text, int max_depth);

// Reads a whole file of at most `max_bytes`. False with `why` when it cannot
// be opened, is not a regular file or is larger.
bool read_small_file(const std::string& path, std::size_t max_bytes, std::string* out, std::string* why);

// Replaces each occurrence of `secret` in `text` by "<redacted>". A secret
// shorter than 8 characters is left alone, as it cannot be told from text.
void redact(std::string* text, const std::string& secret);

}  // namespace caliper_builder::mcp::detail
