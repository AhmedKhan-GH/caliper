#pragma once
// Path confinement and glob expansion for loader inputs.
//
// A loader reads only files inside the workspace, inside a folder the user
// granted, or a file the user granted (AllowedRoots, all canonical paths). A
// path with a ".." segment is refused outright; every match is resolved to its
// canonical path, with every symbolic link followed, and must then lie inside
// one of the roots, so a symbolic link that points out of the workspace is
// refused too.
//
// A glob uses `*` (any characters within one path segment, not matching a
// leading dot), `?` (one character) and a segment that is exactly `**` (zero or
// more folders); the rest is literal. A relative pattern resolves against the
// workspace. The walk does not follow symbolic links to folders, goes at most
// 32 folders deep, visits at most 100,000 entries and returns at most 10,000
// files, sorted. Errors: "loader.path_not_allowed", "loader.no_match",
// "loader.too_many_files", "loader.bad_value". No function here throws.
#include <caliper_builder/error.h>

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_builder {

struct AllowedRoots {
    std::string workspace;             // canonical
    std::vector<std::string> folders;  // canonical, granted by the user
    std::vector<std::string> files;    // canonical, granted by the user
};

struct GlobLimits {
    int max_depth = 32;
    uint64_t max_visited = 100000;
    uint64_t max_matches = 10000;
};

struct ExpandResult {
    Error error;
    std::vector<std::string> files;  // canonical, sorted, unique
};

// Whether `canonical` lies inside `folder` (both canonical), or is it.
bool path_within(const std::string& canonical, const std::string& folder);
// Whether a canonical path is allowed by the roots.
bool path_allowed(const std::string& canonical, const AllowedRoots& roots);
// Matches one path segment against a pattern of `*` and `?`.
bool glob_match(const std::string& pattern, const std::string& name);

// Expands a file path or a glob into allowed, existing regular files.
ExpandResult expand_input(const std::string& pattern, const AllowedRoots& roots, const GlobLimits& limits = {});

// The path as the catalog shows it: relative to the workspace when inside it
// (with forward slashes), otherwise the canonical path.
std::string display_path(const std::string& canonical, const std::string& workspace);

// Protected files: the person's settings that set what an agent may do, such
// as agent-permissions.json, which no command may write, replace or remove.
// The list is process-wide; a session adds its files as it opens, and the
// list only grows.
void protect_file(const std::string& path);
// Whether writing, replacing or removing `path` would change a protected
// file. Both paths are compared resolved: symbolic links and ".." followed,
// and the case of ASCII letters ignored, since the file systems of macOS and
// Windows ignore it; and an existing `path` that is the same file as a
// protected one (a hard link) counts. A folder that holds a protected file
// counts too, for a removal of the folder.
bool touches_protected_file(const std::string& path);

}  // namespace caliper_builder
