#include "text_util.h"

namespace caliper_builder {
namespace catalog_detail {

namespace {

// The length of the well-formed character at `i`, or 0 when it is malformed.
std::size_t char_length(const unsigned char* s, std::size_t n, std::size_t i) {
    const unsigned char c = s[i];
    if (c < 0x80) return 1;
    std::size_t len = 0;
    unsigned int cp = 0;
    if (c >= 0xC2 && c <= 0xDF) {
        len = 2;
        cp = c & 0x1F;
    } else if (c >= 0xE0 && c <= 0xEF) {
        len = 3;
        cp = c & 0x0F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        len = 4;
        cp = c & 0x07;
    } else {
        return 0;
    }
    if (n - i < len) return 0;
    for (std::size_t k = 1; k < len; ++k) {
        const unsigned char cc = s[i + k];
        if ((cc & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (cc & 0x3F);
    }
    if ((len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000)) return 0;  // overlong
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0;                            // surrogate
    if (cp > 0x10FFFF) return 0;
    return len;
}

}  // namespace

bool valid_utf8(std::string_view text) {
    const auto* s = reinterpret_cast<const unsigned char*>(text.data());
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n;) {
        const std::size_t len = char_length(s, n, i);
        if (len == 0) return false;
        i += len;
    }
    return true;
}

std::string repair_utf8(std::string_view text) {
    const auto* s = reinterpret_cast<const unsigned char*>(text.data());
    const std::size_t n = text.size();
    std::string out;
    out.reserve(n);
    for (std::size_t i = 0; i < n;) {
        const std::size_t len = char_length(s, n, i);
        if (len == 0) {
            out += "\xEF\xBF\xBD";
            ++i;
        } else {
            out.append(text.data() + i, len);
            i += len;
        }
    }
    return out;
}

bool has_control(std::string_view text) {
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        if (c < 0x20 || c == 0x7F) return true;
    }
    return false;
}

bool cut_utf8(std::string& text, std::size_t max_bytes) {
    if (text.size() <= max_bytes) return false;
    std::size_t end = max_bytes;
    // Step back over continuation bytes to the start of the cut character.
    while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) --end;
    text.resize(end);
    return true;
}

}  // namespace catalog_detail
}  // namespace caliper_builder
