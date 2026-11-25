#pragma once
// side_chat_internal.h: private string helpers shared by the side chat
// translation units (src/host/builder/side_chat*.cpp). Not part of the public
// interface; external callers include builder/side_chat.h.
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace caliper_host {
namespace side_chat_detail {

// The lines of a text, without their line feeds.
inline std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        out.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        start = end == std::string::npos ? text.size() : end + 1;
    }
    return out;
}

inline std::string trimmed(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline std::string upper(std::string s) {
    for (char& c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

}  // namespace side_chat_detail
}  // namespace caliper_host
