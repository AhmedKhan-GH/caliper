#pragma once
// Text helpers of the catalog library, internal to it: UTF-8 checks and a cut
// at a character boundary. Not a public header.
#include <cstddef>
#include <string>
#include <string_view>

namespace caliper_builder {
namespace catalog_detail {

// True when `text` is well-formed UTF-8 (no overlong forms, no surrogates,
// nothing above U+10FFFF).
bool valid_utf8(std::string_view text);

// True when `text` holds a C0 control character or DEL.
bool has_control(std::string_view text);

// Cuts `text` to at most `max_bytes` bytes without splitting a character.
// Returns true when it cut anything. `text` must be valid UTF-8.
bool cut_utf8(std::string& text, std::size_t max_bytes);

// Replaces every byte that is not part of a well-formed UTF-8 character by
// U+FFFD, so that the text can be written as JSON.
std::string repair_utf8(std::string_view text);

}  // namespace catalog_detail
}  // namespace caliper_builder
