// path.h: the entity path grammar of a scene (design/scenes.md, "Entities and entity paths").
//
// An entity path is 1 to 16 segments, each matching [A-Za-z0-9_.-]{1,63}, joined by '/', at
// most 255 bytes in all; the segments "." and ".." are refused. The empty path is the root, the
// scene's own space.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace caliper_scene {

inline constexpr uint32_t kMaxPathBytes = 255;
inline constexpr uint32_t kMaxPathSegments = 16;
inline constexpr uint32_t kMaxSegmentBytes = 63;

// True for the empty path (the root) and for every path of the grammar. On false, *why (when
// given) says which rule the path breaks.
bool valid_path(std::string_view path, std::string* why = nullptr);

// The parent of a valid non-empty path: everything before its last '/', or "" for one segment.
std::string_view parent_path(std::string_view path);

// The number of segments of a valid path (0 for the root).
uint32_t path_depth(std::string_view path);

}  // namespace caliper_scene
