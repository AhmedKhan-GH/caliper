// gfx_reference.h: the graphics suite's reference mode (design/renderer-core.md, "The rule of
// every step"; docs/explanations/testing.md).
//
// A migration step that promises unchanged pixels records, at its first commit, what the old path
// draws for every case of its rows, before any code moves; the commits that replace the path must
// then reproduce it. The record is a committed text file with one line per case:
//
//   <case> <width>x<height> fnv1a64=<16 hex digits> <key>=<value> ... info=<text to the line's end>
//
// The hash is FNV-1a over the readback's tightly packed RGBA8 bytes. The keys are what else the
// row promises to keep (a path label, a copy count); info is the view's header text. The images
// themselves are not committed: a run writes them as PNG files where CALIPER_GFX_CAPTURE_DIR says,
// and a later run compares with such a folder pixel by pixel.
//
//   CALIPER_GFX_RECORD=<file>            writes the measured line of every case there, sorted by
//                                        case (the step's first commit records with it and commits
//                                        the file under tests/gfx/references/)
//   CALIPER_GFX_REFERENCES=<file>        reads that file instead of the committed one
//   CALIPER_GFX_CAPTURE_DIR=<folder>     writes each case's readback as <prefix>-<case>.png there
//   CALIPER_GFX_REFERENCE_IMAGES=<folder> a folder of those PNG files from an earlier build: each case
//                                        is compared with its image there, and the differing pixels
//                                        are counted and listed
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace gfx_reference {

inline uint64_t fnv1a64(const std::vector<uint8_t>& bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : bytes) {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h;
}

inline std::string hex16(uint64_t v) {
    char t[17];
    std::snprintf(t, sizeof t, "%016llx", (unsigned long long)v);
    return t;
}

inline const char* env(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? v : nullptr;
}

// One case's line.
struct Line {
    std::string name;
    int width = 0, height = 0;
    std::string hash;                                // 16 hex digits
    std::vector<std::pair<std::string, std::string>> fields;   // in the order written
    std::string info;

    std::string field(const std::string& key) const {
        for (const auto& kv : fields)
            if (kv.first == key) return kv.second;
        return {};
    }
    std::string text() const {
        std::string s = name + " " + std::to_string(width) + "x" + std::to_string(height) + " fnv1a64=" + hash;
        for (const auto& kv : fields) s += " " + kv.first + "=" + kv.second;
        return s + " info=" + info;
    }
    static bool parse(const std::string& text, Line* out) {
        std::istringstream in(text);
        Line l;
        std::string size, tok;
        if (!(in >> l.name >> size)) return false;
        const size_t x = size.find('x');
        if (x == std::string::npos) return false;
        l.width = std::atoi(size.c_str());
        l.height = std::atoi(size.c_str() + x + 1);
        const size_t at = text.find(" info=");
        const std::string head = at == std::string::npos ? text : text.substr(0, at);
        if (at != std::string::npos) l.info = text.substr(at + 6);
        std::istringstream hin(head);
        hin >> tok >> tok;   // name, size
        while (hin >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "fnv1a64") l.hash = v;
            else l.fields.emplace_back(k, v);
        }
        *out = l;
        return true;
    }
};

// The references of one row file: read for comparing, collected for recording.
class Set {
public:
    // committed: the file in the source tree, read unless CALIPER_GFX_REFERENCES names another.
    explicit Set(const std::string& committed) {
        const char* other = env("CALIPER_GFX_REFERENCES");
        path_ = other ? other : committed;
        std::ifstream in(path_);
        std::string text;
        while (std::getline(in, text)) {
            if (text.empty() || text[0] == '#') continue;
            Line l;
            if (Line::parse(text, &l)) lines_[l.name] = l;
        }
    }
    ~Set() { flush(); }
    const std::string& path() const { return path_; }
    bool recording() const { return env("CALIPER_GFX_RECORD") != nullptr; }
    const Line* find(const std::string& name) const {
        const auto it = lines_.find(name);
        return it == lines_.end() ? nullptr : &it->second;
    }
    void record(const Line& l) { recorded_[l.name] = l; }
    // Writes the recorded lines (record mode only), with a header that says how they were made.
    void flush() {
        const char* out = env("CALIPER_GFX_RECORD");
        if (!out || recorded_.empty()) return;
        std::ofstream f(out);
        f << "# The graphics suite's references (tests/gfx/gfx_reference.h): one line per case, the readback's\n"
             "# FNV-1a hash, what else the row keeps, and the view's header text.\n";
        for (const auto& kv : recorded_) f << kv.second.text() << "\n";
        std::fprintf(stderr, "[gfx-reference] %zu lines written to %s\n", recorded_.size(), out);
        recorded_.clear();
    }

private:
    std::string path_;
    std::map<std::string, Line> lines_;
    std::map<std::string, Line> recorded_;
};

}  // namespace gfx_reference
