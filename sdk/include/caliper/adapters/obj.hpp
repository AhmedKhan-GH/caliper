#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace caliper::obj {

struct Mesh {
    std::vector<float> positions;  // (vertex_count, 3)
    std::vector<float> normals;    // (vertex_count, 3), zero when absent
    std::vector<float> uvs;        // (vertex_count, 2), zero when absent
    std::vector<int32_t> indices;  // triangulated
    bool has_normals = false;
    bool has_uvs = false;

    size_t vertex_count() const { return positions.size() / 3; }
    size_t triangle_count() const { return indices.size() / 3; }
};

namespace detail {

struct Key {
    int v = -1;
    int vt = -1;
    int vn = -1;
    bool operator==(const Key& other) const {
        return v == other.v && vt == other.vt && vn == other.vn;
    }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = static_cast<size_t>(k.v + 1);
        h = h * 16777619u ^ static_cast<size_t>(k.vt + 2);
        h = h * 16777619u ^ static_cast<size_t>(k.vn + 2);
        return h;
    }
};

inline bool parse_int(std::string_view text, int& out) {
    if (text.empty()) return false;
    const char* first = text.data();
    const char* last = first + text.size();
    auto result = std::from_chars(first, last, out);
    return result.ec == std::errc{} && result.ptr == last && out != 0;
}

inline bool resolve_index(int raw, size_t count, int& out) {
    const int64_t resolved = raw > 0
        ? static_cast<int64_t>(raw) - 1
        : static_cast<int64_t>(count) + raw;
    if (resolved < 0 || resolved >= static_cast<int64_t>(count) ||
        resolved > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(resolved);
    return true;
}

inline bool parse_face_vertex(std::string_view token,
                              size_t position_count, size_t uv_count,
                              size_t normal_count, Key& key) {
    const size_t slash1 = token.find('/');
    const size_t slash2 = slash1 == std::string_view::npos
        ? std::string_view::npos : token.find('/', slash1 + 1);
    if (slash2 != std::string_view::npos &&
        token.find('/', slash2 + 1) != std::string_view::npos) {
        return false;
    }

    const std::string_view vpart = token.substr(0, slash1);
    const std::string_view vtpart = slash1 == std::string_view::npos
        ? std::string_view{} : token.substr(
              slash1 + 1, slash2 == std::string_view::npos
                  ? std::string_view::npos : slash2 - slash1 - 1);
    const std::string_view vnpart = slash2 == std::string_view::npos
        ? std::string_view{} : token.substr(slash2 + 1);

    int raw = 0;
    if (!parse_int(vpart, raw) || !resolve_index(raw, position_count, key.v))
        return false;
    if (!vtpart.empty()) {
        if (!parse_int(vtpart, raw) || !resolve_index(raw, uv_count, key.vt))
            return false;
    }
    if (!vnpart.empty()) {
        if (!parse_int(vnpart, raw) || !resolve_index(raw, normal_count, key.vn))
            return false;
    }
    return true;
}

inline bool finite(float value) { return std::isfinite(value); }

}  // namespace detail

inline bool load(std::istream& input, Mesh& output, std::string* error = nullptr) {
    std::vector<std::array<float, 3>> src_positions;
    std::vector<std::array<float, 2>> src_uvs;
    std::vector<std::array<float, 3>> src_normals;
    std::unordered_map<detail::Key, int32_t, detail::KeyHash> dedup;
    Mesh mesh;
    bool all_uvs = true;
    bool all_normals = true;
    size_t line_no = 0;

    auto fail = [&](const std::string& reason) {
        if (error) *error = "OBJ line " + std::to_string(line_no) + ": " + reason;
        output = {};
        return false;
    };

    std::string line;
    while (std::getline(input, line)) {
        ++line_no;
        std::istringstream row(line);
        std::string tag;
        if (!(row >> tag) || tag[0] == '#') continue;

        if (tag == "v") {
            std::array<float, 3> value{};
            if (!(row >> value[0] >> value[1] >> value[2]) ||
                !detail::finite(value[0]) || !detail::finite(value[1]) ||
                !detail::finite(value[2])) {
                return fail("malformed position");
            }
            src_positions.push_back(value);
        } else if (tag == "vt") {
            std::array<float, 2> value{};
            if (!(row >> value[0] >> value[1]) ||
                !detail::finite(value[0]) || !detail::finite(value[1])) {
                return fail("malformed texture coordinate");
            }
            src_uvs.push_back(value);
        } else if (tag == "vn") {
            std::array<float, 3> value{};
            if (!(row >> value[0] >> value[1] >> value[2]) ||
                !detail::finite(value[0]) || !detail::finite(value[1]) ||
                !detail::finite(value[2])) {
                return fail("malformed normal");
            }
            src_normals.push_back(value);
        } else if (tag == "f") {
            std::vector<detail::Key> face;
            std::string token;
            while (row >> token) {
                if (!token.empty() && token[0] == '#') break;
                detail::Key key;
                if (!detail::parse_face_vertex(token, src_positions.size(),
                                               src_uvs.size(), src_normals.size(),
                                               key)) {
                    return fail("malformed or out-of-range face index");
                }
                face.push_back(key);
            }
            if (face.size() < 3) return fail("face has fewer than three vertices");

            auto emit = [&](const detail::Key& key, int32_t& index) -> bool {
                auto found = dedup.find(key);
                if (found != dedup.end()) {
                    index = found->second;
                    return true;
                }
                if (mesh.vertex_count() >=
                    static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    return false;
                }
                index = static_cast<int32_t>(mesh.vertex_count());
                dedup.emplace(key, index);
                const auto& p = src_positions[static_cast<size_t>(key.v)];
                mesh.positions.insert(mesh.positions.end(), p.begin(), p.end());
                if (key.vn >= 0) {
                    const auto& n = src_normals[static_cast<size_t>(key.vn)];
                    mesh.normals.insert(mesh.normals.end(), n.begin(), n.end());
                } else {
                    mesh.normals.insert(mesh.normals.end(), {0.f, 0.f, 0.f});
                    all_normals = false;
                }
                if (key.vt >= 0) {
                    const auto& uv = src_uvs[static_cast<size_t>(key.vt)];
                    mesh.uvs.insert(mesh.uvs.end(), uv.begin(), uv.end());
                } else {
                    mesh.uvs.insert(mesh.uvs.end(), {0.f, 0.f});
                    all_uvs = false;
                }
                return true;
            };

            for (size_t i = 1; i + 1 < face.size(); ++i) {
                for (const detail::Key& key : {face[0], face[i], face[i + 1]}) {
                    int32_t index = 0;
                    if (!emit(key, index)) return fail("too many deduplicated vertices");
                    mesh.indices.push_back(index);
                }
            }
        }
    }

    if (!input.eof() && input.fail()) return fail("input read failed");
    if (mesh.indices.empty()) return fail("contains no faces");
    mesh.has_uvs = all_uvs;
    mesh.has_normals = all_normals;
    output = std::move(mesh);
    if (error) error->clear();
    return true;
}

inline bool load_file(const std::string& path, Mesh& output,
                      std::string* error = nullptr) {
    std::ifstream file(path);
    if (!file) {
        output = {};
        if (error) *error = "could not open OBJ: " + path;
        return false;
    }
    return load(file, output, error);
}

}  // namespace caliper::obj

