// Path confinement and glob expansion for loader inputs.
#include <caliper_builder/loaders/paths.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <mutex>

namespace caliper_builder {

namespace fs = std::filesystem;

namespace {

bool has_wildcard(const std::string& s) { return s.find_first_of("*?") != std::string::npos; }

struct Walk {
    const AllowedRoots& roots;
    const GlobLimits& limits;
    std::vector<std::string> segments;
    std::vector<std::string> matches;
    uint64_t visited = 0;
    Error error;

    void add(const fs::path& p) {
        if (error) return;
        std::error_code ec;
        const fs::path canonical = fs::canonical(p, ec);
        if (ec || !fs::is_regular_file(canonical, ec)) return;
        const std::string c = canonical.u8string();
        if (!path_allowed(c, roots)) {
            error = Error("loader.path_not_allowed",
                          "The path lies outside the workspace and the files and folders the user granted.", c);
            return;
        }
        matches.push_back(c);
        if (matches.size() > limits.max_matches) {
            error = Error("loader.too_many_files", "The glob matches more than " +
                                                       std::to_string(limits.max_matches) + " files.");
        }
    }

    // Lists a folder's entries without following symbolic links to folders.
    std::vector<fs::directory_entry> list(const fs::path& dir) {
        std::vector<fs::directory_entry> out;
        std::error_code ec;
        fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
        if (ec) return out;
        for (; it != fs::directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (++visited > limits.max_visited) {
                error = Error("loader.too_many_files", "The glob visits more than " +
                                                           std::to_string(limits.max_visited) + " entries.");
                return {};
            }
            out.push_back(*it);
        }
        return out;
    }

    void match(const fs::path& dir, std::size_t i, int depth) {
        if (error) return;
        if (depth > limits.max_depth) {
            error = Error("loader.too_many_files", "The glob goes more than " + std::to_string(limits.max_depth) +
                                                       " folders deep.");
            return;
        }
        const std::string& seg = segments[i];
        const bool last = i + 1 == segments.size();
        std::error_code ec;
        if (seg == "**") {
            match(dir, i + 1, depth);
            for (const fs::directory_entry& e : list(dir)) {
                if (error) return;
                if (e.is_directory(ec) && !e.is_symlink(ec)) match(e.path(), i, depth + 1);
            }
        } else if (has_wildcard(seg)) {
            for (const fs::directory_entry& e : list(dir)) {
                if (error) return;
                const std::string name = e.path().filename().u8string();
                if (!glob_match(seg, name)) continue;
                if (last) {
                    if (e.is_regular_file(ec)) add(e.path());
                } else if (e.is_directory(ec) && !e.is_symlink(ec)) {
                    match(e.path(), i + 1, depth + 1);
                }
            }
        } else {
            const fs::path p = dir / fs::u8path(seg);
            if (last) {
                if (fs::is_regular_file(p, ec)) add(p);
            } else if (fs::is_directory(p, ec) && !fs::is_symlink(p, ec)) {
                match(p, i + 1, depth + 1);
            }
        }
    }
};

}  // namespace

bool path_within(const std::string& canonical, const std::string& folder) {
    const fs::path c = fs::u8path(canonical), f = fs::u8path(folder);
    auto ci = c.begin();
    for (auto fi = f.begin(); fi != f.end(); ++fi) {
        if (fi->empty()) continue;  // the empty element after a trailing separator
        if (ci == c.end() || *ci != *fi) return false;
        ++ci;
    }
    return true;
}

bool path_allowed(const std::string& canonical, const AllowedRoots& roots) {
    if (!roots.workspace.empty() && path_within(canonical, roots.workspace)) return true;
    for (const std::string& f : roots.folders) {
        if (path_within(canonical, f)) return true;
    }
    for (const std::string& f : roots.files) {
        if (canonical == f) return true;
    }
    return false;
}

bool glob_match(const std::string& pattern, const std::string& name) {
    if (!name.empty() && name[0] == '.' && (pattern.empty() || pattern[0] != '.')) return false;
    std::size_t p = 0, n = 0, star = std::string::npos, mark = 0;
    while (n < name.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) {
            ++p;
            ++n;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            mark = n;
        } else if (star != std::string::npos) {
            p = star + 1;
            n = ++mark;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}

ExpandResult expand_input(const std::string& pattern, const AllowedRoots& roots, const GlobLimits& limits) {
    ExpandResult r;
    if (pattern.empty() || pattern.size() > 4096 || pattern.find('\0') != std::string::npos) {
        r.error = Error("loader.bad_value", "The input path is empty, too long or holds a NUL.");
        return r;
    }
    for (unsigned char c : pattern) {
        if (c < 0x20) {
            r.error = Error("loader.bad_value", "The input path holds a control character.");
            return r;
        }
    }
    const fs::path pat = fs::u8path(pattern);
    fs::path base;
    std::vector<std::string> segments;
    bool started = false;
    for (const fs::path& part : pat.is_absolute() ? pat.relative_path() : pat) {
        const std::string s = part.u8string();
        if (s.empty() || s == ".") continue;
        if (s == "..") {
            r.error = Error("loader.path_not_allowed", "The input path has a \"..\" segment.", pattern);
            return r;
        }
        if (!started && !has_wildcard(s)) {
            base /= part;
            continue;
        }
        started = true;
        segments.push_back(s);
    }
    const fs::path root = pat.is_absolute() ? pat.root_path() : fs::u8path(roots.workspace);
    if (roots.workspace.empty() && !pat.is_absolute()) {
        r.error = Error("loader.bad_value", "A relative input path needs a workspace.", pattern);
        return r;
    }
    const fs::path fixed = root / base;
    if (segments.empty()) {
        // A plain file: it must exist and be allowed.
        std::error_code ec;
        const fs::path canonical = fs::canonical(fixed, ec);
        if (ec || !fs::is_regular_file(canonical, ec)) {
            r.error = Error("loader.no_match", "The input file does not exist or is not a regular file.", pattern);
            return r;
        }
        const std::string c = canonical.u8string();
        if (!path_allowed(c, roots)) {
            r.error = Error("loader.path_not_allowed",
                            "The path lies outside the workspace and the files and folders the user granted.", c);
            return r;
        }
        r.files.push_back(c);
        return r;
    }
    if (segments.back() == "**") {
        r.error = Error("loader.bad_value", "A glob ends with a file name pattern, not with **.", pattern);
        return r;
    }
    std::error_code ec;
    if (!fs::is_directory(fixed, ec)) {
        r.error = Error("loader.no_match", "The glob's folder does not exist.", pattern);
        return r;
    }
    Walk walk{roots, limits, segments, {}, 0, {}};
    walk.match(fixed, 0, 0);
    if (walk.error) {
        r.error = walk.error;
        return r;
    }
    std::sort(walk.matches.begin(), walk.matches.end());
    walk.matches.erase(std::unique(walk.matches.begin(), walk.matches.end()), walk.matches.end());
    if (walk.matches.empty()) {
        r.error = Error("loader.no_match", "The glob matches no file.", pattern);
        return r;
    }
    r.files = std::move(walk.matches);
    return r;
}

std::string display_path(const std::string& canonical, const std::string& workspace) {
    if (!workspace.empty() && path_within(canonical, workspace)) {
        return fs::u8path(canonical).lexically_relative(fs::u8path(workspace)).generic_u8string();
    }
    return canonical;
}

namespace {

std::mutex g_protected_mutex;
std::vector<fs::path> g_protected;

// The absolute path with the links of its last name followed, also a link
// whose target does not exist yet.
fs::path follow_links(const fs::path& p) {
    std::error_code ec;
    fs::path abs = fs::absolute(p, ec);
    if (ec) abs = p;
    for (int i = 0; i < 40; ++i) {
        std::error_code lec;
        if (!fs::is_symlink(fs::symlink_status(abs, lec))) break;
        const fs::path target = fs::read_symlink(abs, lec);
        if (lec) break;
        abs = target.is_absolute() ? target : abs.parent_path() / target;
    }
    return abs;
}

std::string lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

// Whether both exist and are the same file by device and inode.
bool same_file(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    return fs::exists(a, ec) && fs::exists(b, ec) && fs::equivalent(a, b, ec) && !ec;
}

// The path resolved as far as it exists (links and ".." followed, also a
// link whose target does not exist yet), with the rest lexically normal, as a
// string with ASCII letters in lower case.
std::string resolved_key(const fs::path& p) {
    std::error_code ec;
    const fs::path abs = follow_links(p);
    fs::path r = fs::weakly_canonical(abs, ec);
    if (ec) r = abs.lexically_normal();
    std::string s = r.lexically_normal().u8string();
    while (s.size() > 1 && (s.back() == '/' || s.back() == '\\')) s.pop_back();
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (c == '\\') c = '/';
    }
    return s;
}

}  // namespace

void protect_file(const std::string& path) {
    if (path.empty()) return;
    std::error_code ec;
    fs::path p = fs::absolute(fs::u8path(path), ec);
    if (ec) return;
    std::lock_guard<std::mutex> lock(g_protected_mutex);
    if (std::find(g_protected.begin(), g_protected.end(), p) == g_protected.end()) g_protected.push_back(p);
}

bool touches_protected_file(const std::string& path) {
    std::vector<fs::path> files;
    {
        std::lock_guard<std::mutex> lock(g_protected_mutex);
        files = g_protected;
    }
    if (files.empty() || path.empty()) return false;
    const fs::path p = fs::u8path(path);
    const std::string key = resolved_key(p);
    const fs::path target = follow_links(p);
    std::error_code ec;
    const bool folder = fs::is_directory(p, ec);
    for (const fs::path& f : files) {
        const std::string fkey = resolved_key(f);
        // The same path, or a folder that holds the file, by their resolved names.
        if (fkey == key || (fkey.size() > key.size() && fkey.compare(0, key.size(), key) == 0 && fkey[key.size()] == '/'))
            return true;
        // The same file by another name: a hard link, a case variant, or a
        // second spelling of a folder above it (on macOS /System/Volumes/Data
        // and / hold the same /Users), by device and inode.
        if (same_file(p, f)) return true;
        if (folder)
            for (fs::path a = f.parent_path(); !a.empty(); a = a.parent_path()) {
                if (same_file(p, a)) return true;
                if (a == a.parent_path()) break;
            }
        // A file not made yet: the same folder by device and inode, and the
        // same name but for the case of its letters.
        if (lower_ascii(target.filename().u8string()) == lower_ascii(f.filename().u8string()) &&
            same_file(target.parent_path(), f.parent_path()))
            return true;
    }
    return false;
}

}  // namespace caliper_builder

