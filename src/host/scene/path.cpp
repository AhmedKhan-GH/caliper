// path.cpp: the entity path grammar (path.h).
#include "scene/path.h"

namespace caliper_scene {

namespace {

bool segment_char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
           c == '.' || c == '-';
}

bool fail(std::string* why, std::string text) {
    if (why) *why = std::move(text);
    return false;
}

}  // namespace

bool valid_path(std::string_view path, std::string* why) {
    if (path.empty()) return true;   // the root
    if (path.size() > kMaxPathBytes)
        return fail(why, "an entity path is at most 255 bytes, and this one has " +
                             std::to_string(path.size()));
    uint32_t segments = 0;
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find('/', start);
        if (end == std::string_view::npos) end = path.size();
        const std::string_view seg = path.substr(start, end - start);
        if (seg.empty())
            return fail(why, "an entity path has no empty segment, no leading or trailing '/' and no '//'");
        if (seg.size() > kMaxSegmentBytes)
            return fail(why, "a segment of an entity path is at most 63 bytes, and '" + std::string(seg) +
                                 "' has " + std::to_string(seg.size()));
        if (seg == "." || seg == "..")
            return fail(why, "the segments '.' and '..' are refused, because they would read as relative steps");
        for (char c : seg)
            if (!segment_char(c))
                return fail(why, "a segment of an entity path holds only letters, digits, '_', '.' and '-', "
                                 "and '" + std::string(seg) + "' does not");
        if (++segments > kMaxPathSegments)
            return fail(why, "an entity path has at most 16 segments");
        start = end + 1;
        if (end == path.size()) break;
    }
    return true;
}

std::string_view parent_path(std::string_view path) {
    const size_t slash = path.rfind('/');
    return slash == std::string_view::npos ? std::string_view() : path.substr(0, slash);
}

uint32_t path_depth(std::string_view path) {
    if (path.empty()) return 0;
    uint32_t n = 1;
    for (char c : path)
        if (c == '/') ++n;
    return n;
}

}  // namespace caliper_scene
