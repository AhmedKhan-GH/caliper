// The shapes of a scene view drawn by the library pieces (scene_pieces.h).
//
// Each writer below writes the published resources of its piece as that piece's derive half
// writes them from the same arrays (src/builder/programs/library/pieces/<name>/1/derive.wgsl),
// with three differences that the host decides: the colormap's range is the batch's resolved
// vmin and vmax (range 0, as the scene view resolves the producer's stats), the class colors are
// the scene's class table, and the pose and the depth offset are in the program's matrix, so that
// a piece draws the batch where the scene core draws it, composed in double and rounded to f32
// once, as the scene core composes its matrices.
#include "draw/scene_pieces.h"

#include "scene/scene_math.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>

namespace caliper_draw {

namespace cs = caliper_builder::spec;
namespace cp = caliper_builder::programs;
namespace wire = caliper_net::wire;
using caliper_scene::Batch;
using caliper_scene::EntityId;
using caliper_scene::Mat4;
using caliper_scene::Primitive;

namespace {

constexpr std::uint32_t kNanBits = 0x7fc00000u;
constexpr std::uint32_t kNone = 0xffffffffu;

// The capacities of the library pieces' resources (their piece.toml).
constexpr std::uint64_t kPointsMax = 262144, kSegmentsMax = 209714, kArrowsMax = 262143, kBoxesMax = 131072,
                        kMeshVerticesMax = 98304, kMeshTrianglesMax = 196608, kImagePixelsMax = 2097120,
                        kMaskPixelsMax = 2093024, kPaletteSize = 4096;

std::uint32_t bits_of(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

float float_of(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void put(std::vector<std::uint8_t>& out, std::size_t at, const void* p, std::size_t n) { std::memcpy(out.data() + at, p, n); }

// A section of the part with `cols` values a row (0: one value a row) of `elem` bytes: its rows,
// or -1 when the section is absent, is not raw, has another dtype or rank, or holds fewer bytes
// than its shape needs.
int64_t rows_of(const wire::Section* s, int64_t dtype, std::size_t elem, int64_t cols) {
    if (!s || s->encoding != CALIPER_NET_ENCODING_RAW || s->dtype != dtype || !s->data) return -1;
    if (cols == 0 ? s->shape.size() != 1 : (s->shape.size() != 2 || s->shape[1] != cols)) return -1;
    const int64_t n = s->shape[0];
    if (n < 0) return -1;
    const uint64_t need = uint64_t(n) * uint64_t(cols == 0 ? 1 : cols) * elem;
    if (uint64_t(n) > (uint64_t(1) << 40) || s->size < need) return -1;
    return n;
}

// The (N, D) f32 rows of a section as positions: x, then y and z where D has them.
struct Rows {
    const float* p = nullptr;
    int64_t n = 0;
    int64_t d = 0;
    float at(int64_t i, int a) const { return a < d ? p[i * d + a] : 0.0f; }
};
bool rows(const wire::Section* s, Rows* out) {
    if (!s || s->shape.size() != 2) return false;
    const int64_t d = s->shape[1];
    if (d < 1 || d > 4) return false;
    const int64_t n = rows_of(s, CALIPER_NET_DTYPE_F32, 4, d);
    if (n < 0) return false;
    out->p = reinterpret_cast<const float*>(s->data);
    out->n = n;
    out->d = d;
    return true;
}

// The color source of a part: packed RGBA from `colors` (mode 2), the bits of `values` (mode 1),
// the class ids (written later from the class table), or none (mode 0); k rows, 1 for all.
struct Source {
    std::uint32_t mode = 0;
    std::vector<std::uint32_t> bits;   // per row
    std::vector<std::int32_t> ids;     // class ids, per row
    bool alpha = false;                // a packed alpha below 255
    std::uint32_t at(uint64_t i) const {
        if (bits.empty()) return kNanBits;
        if (bits.size() == 1) return bits[0];
        return i < bits.size() ? bits[std::size_t(i)] : kNanBits;
    }
};
bool source_of(const wire::SceneUpdate& u, Source* out, std::string* why) {
    if (const wire::Section* c = u.array("colors")) {
        const int64_t k = rows_of(c, CALIPER_NET_DTYPE_U8, 1, 4);
        if (k < 0) return *why = "colors of another shape", false;
        out->mode = 2;
        out->bits.resize(std::size_t(k));
        std::memcpy(out->bits.data(), c->data, std::size_t(k) * 4);   // r | g << 8 | b << 16 | a << 24
        for (std::uint32_t b : out->bits) out->alpha = out->alpha || (b >> 24) < 255u;
    } else if (const wire::Section* v = u.array("values")) {
        const int64_t k = rows_of(v, CALIPER_NET_DTYPE_F32, 4, 0);
        if (k < 0) return *why = "values of another shape", false;
        out->mode = 1;
        out->bits.resize(std::size_t(k));
        std::memcpy(out->bits.data(), v->data, std::size_t(k) * 4);
    } else if (const wire::Section* ids = u.array("class_ids")) {
        const int64_t k = rows_of(ids, CALIPER_NET_DTYPE_I32, 4, 0);
        if (k < 0) return *why = "class ids of another shape", false;
        out->mode = 3;
        out->ids.resize(std::size_t(k));
        std::memcpy(out->ids.data(), ids->data, std::size_t(k) * 4);
    }
    return true;
}

std::uint32_t class_color(const std::vector<std::uint32_t>& texels, std::int32_t id) {
    if (id < 0 || id > 65535 || texels.size() < 65536) return caliper_scene::kDefaultRgba;
    return texels[std::size_t(id)];
}

// The range and flags of `info` for lines@1 and arrows@1: the finite range of the values, bit 0
// for a NaN or no value, bit 1 for a packed alpha below 255.
void info_of(const Source& s, std::uint64_t rows, std::vector<std::uint8_t>* out) {
    float lo = 0, hi = 0;
    std::uint32_t flags = 0;
    bool seen = false;
    for (std::uint32_t b : s.bits) {
        if ((b >> 24) < 255u) flags |= 2u;
        const float v = float_of(b);
        if (std::isnan(v)) {
            flags |= 1u;
            continue;
        }
        lo = seen ? std::min(lo, v) : v;
        hi = seen ? std::max(hi, v) : v;
        seen = true;
    }
    if (s.bits.size() > 1 && s.bits.size() < rows) flags |= 1u;   // rows past the end read NaN
    if (!seen) flags |= 1u;
    out->assign(12, 0);
    put(*out, 0, &lo, 4);
    put(*out, 4, &hi, 4);
    put(*out, 8, &flags, 4);
}

}  // namespace

// What one part of one layer became: the piece, its resources and what its params need beyond
// the batch, or the kind of fallback when no piece draws it.
struct ScenePieces::Converted {
    std::string piece;
    std::string fallback;
    std::vector<PieceBatch::Resource> resources;
    std::vector<std::pair<std::string, double>> params;   // an override's
    bool is_override = false;
    std::uint64_t generation = 0;
    std::uint64_t count = 0;          // items, for the default point size
    std::uint32_t color_mode = 0;
    std::uint32_t mode = 0, color_per = 0, shading = 0, indexed = 0;
    std::uint32_t width = 0, height = 0;   // an image's pixels
    // Class ids, written into the resource from the class table at each change of the table.
    std::vector<std::int32_t> class_ids;
    std::uint64_t classes_revision = ~0ull;
    std::size_t class_resource = 0;   // the resource whose class colors follow the table
};

// One program of the frame: the batch it draws and what each frame gives it.
struct ScenePieces::Program {
    std::string id;
    std::string path;
    Primitive primitive = Primitive::Points;
    std::shared_ptr<Converted> conv;
    const cs::Piece* piece = nullptr;
    std::uint32_t pick_id = 0;
    std::vector<std::uint8_t> params;
    Mat4 model;
    bool overlay = false;
    int depth_offset = 0;
    // The scene core's draw groups: opaque (0) then translucent (1); on a camera's image plane the
    // images first (sub 0) and then that camera's 2-D entities (sub 1).
    int group = 0, sub = 0;
    bool translucent = false;
    bool is_image = false;
    bool frustum = false;       // frustum@1 over a camera entity, not a batch
    std::uint32_t camera = 0;   // the index of the camera of the batch's pixel space, 0 for none
    bool given = false;                 // the runtime holds the resources of given_generation
    std::uint64_t given_generation = 0;
};

const cs::Piece* ScenePieces::library_piece(const std::string& name) {
    static const std::map<std::string, cs::Piece> pieces = [] {
        std::map<std::string, std::map<std::string, std::string>> folders;
        for (const cp::LibraryFile& f : cp::library_files()) {
            const std::string path(f.path);
            if (path.rfind("pieces/", 0) != 0) continue;
            const std::size_t cut = path.rfind('/');
            folders[path.substr(0, cut)][path.substr(cut + 1)] = std::string(f.text);
        }
        std::map<std::string, cs::Piece> out;
        for (auto& [folder, texts] : folders) {
            auto manifest = texts.find("piece.toml");
            if (manifest == texts.end()) continue;
            cs::PieceParse parsed = cs::piece_from_toml(manifest->second);
            if (!parsed.ok() || parsed.piece.version != 1) continue;
            cs::Piece p = std::move(parsed.piece);
            if ("pieces/" + p.name + "/1" != folder) continue;
            for (const char* file : cs::kPieceFiles)
                if (auto it = texts.find(file); it != texts.end()) p.files[file] = it->second;
            p.hash = cp::piece_hash(p);
            out.emplace(p.name, std::move(p));
        }
        return out;
    }();
    auto it = pieces.find(name);
    return it == pieces.end() ? nullptr : &it->second;
}

ScenePieces::ScenePieces() = default;
ScenePieces::~ScenePieces() = default;

namespace {

using Converted = ScenePieces::Converted;

bool too_many(Converted& c, uint64_t n, uint64_t most, const char* what) {
    if (n <= most) return false;
    c.fallback = std::string(what) + " above the piece's capacity";
    return true;
}

void points_of(const wire::SceneUpdate& u, const Batch& b, Converted& c) {
    c.piece = "points";
    Rows pos;
    std::string why;
    Source src;
    if (!rows(u.array("positions"), &pos) || !source_of(u, &src, &why)) return void(c.fallback = "points of another shape");
    if (b.array(caliper_scene::Role::Radii)) return void(c.fallback = "points with radii in scene units");
    if (too_many(c, uint64_t(pos.n), kPointsMax, "points")) return;
    c.count = uint64_t(pos.n);
    c.color_mode = src.mode == 3 ? 2 : src.mode;
    std::vector<std::uint8_t> out(std::size_t(pos.n) * 32, 0);
    for (int64_t i = 0; i < pos.n; ++i) {
        const float p[4] = {pos.at(i, 0), pos.at(i, 1), pos.at(i, 2), 1.0f};
        const std::uint32_t bits = src.mode == 3 ? 0u : src.at(uint64_t(i)), index = std::uint32_t(i);
        put(out, std::size_t(i) * 32, p, 16);
        put(out, std::size_t(i) * 32 + 16, &bits, 4);
        put(out, std::size_t(i) * 32 + 20, &index, 4);
    }
    if (src.mode == 3) {
        c.class_ids = src.ids;
        c.class_resource = 0;
    }
    c.resources.push_back({"points", std::uint32_t(pos.n), std::move(out)});
}

void lines_of(const wire::SceneUpdate& u, const Batch& b, Converted& c) {
    c.piece = "lines";
    Rows pos;
    std::string why;
    Source src;
    if (!rows(u.array("positions"), &pos) || !source_of(u, &src, &why) || src.mode == 3)
        return void(c.fallback = "lines of another shape");
    const uint64_t n = uint64_t(pos.n);
    c.mode = b.line_mode == caliper_scene::LineMode::Strips ? 1u : b.line_mode == caliper_scene::LineMode::Edges ? 2u : 0u;
    c.color_mode = src.mode;
    // Strips: a vertex joins the next unless it ends its strip, as the scene view's break mask.
    std::vector<std::uint8_t> joins;
    std::vector<std::uint32_t> strip_of;
    const std::int32_t* pairs = nullptr;
    uint64_t count = n / 2;
    if (c.mode == 1) {
        const wire::Section* sl = u.array("strip_lengths");
        const int64_t s = rows_of(sl, CALIPER_NET_DTYPE_I32, 4, 0);
        if (s < 0) return void(c.fallback = "lines of another shape");
        const std::int32_t* len = reinterpret_cast<const std::int32_t*>(sl->data);
        joins.assign(std::size_t(n), 0);
        strip_of.assign(std::size_t(n), 0);
        uint64_t v = 0;
        for (int64_t k = 0; k < s && v < n; ++k) {
            const uint64_t l = uint64_t(std::max<std::int32_t>(len[k], 0));
            for (uint64_t j = 0; j < l && v < n; ++j, ++v) {
                joins[std::size_t(v)] = j + 1 < l ? 1 : 0;
                strip_of[std::size_t(v)] = std::uint32_t(k);
            }
        }
        count = n > 0 ? n - 1 : 0;
        // A per-strip color source stays per strip: the batch's per-vertex colors repeat it.
        const caliper_net::Json* per = u.params.find("color_per");
        c.color_per = per && per->is_string() && per->as_string() == "strip" ? 1u : 0u;
    } else if (c.mode == 2) {
        const wire::Section* ix = u.array("indices");
        const int64_t e = rows_of(ix, CALIPER_NET_DTYPE_I32, 4, 2);
        if (e < 0) return void(c.fallback = "lines of another shape");
        pairs = reinterpret_cast<const std::int32_t*>(ix->data);
        count = n > 0 ? uint64_t(e) : 0;
        c.color_per = b.per_edge ? 1u : 0u;
    }
    if (too_many(c, count, kSegmentsMax, "segments")) return;
    c.count = count;
    std::vector<std::uint8_t> out(std::size_t(count) * 40, 0);
    for (uint64_t i = 0; i < count; ++i) {
        std::uint32_t a = std::uint32_t(2 * i), bb = a + 1, ca = src.at(a), cb = src.at(bb);
        bool keep = true;
        if (c.mode == 1) {
            a = std::uint32_t(i);
            bb = a + 1;
            keep = joins[std::size_t(i)] != 0;
            if (c.color_per == 1) ca = cb = src.at(strip_of[std::size_t(i)]);
            else ca = src.at(a), cb = src.at(bb);
        } else if (c.mode == 2) {
            a = std::min(std::uint32_t(pairs[2 * i]), std::uint32_t(n - 1));
            bb = std::min(std::uint32_t(pairs[2 * i + 1]), std::uint32_t(n - 1));
            if (c.color_per == 1) ca = cb = src.at(i);
            else ca = src.at(a), cb = src.at(bb);
        }
        const float pa[3] = {pos.at(a, 0), pos.at(a, 1), pos.at(a, 2)}, pb[3] = {pos.at(bb, 0), pos.at(bb, 1), pos.at(bb, 2)};
        const std::uint32_t ib = keep ? bb : kNone;
        const std::size_t o = std::size_t(i) * 40;
        put(out, o, pa, 12);
        put(out, o + 12, &ca, 4);
        put(out, o + 16, pb, 12);
        put(out, o + 28, &cb, 4);
        put(out, o + 32, &a, 4);
        put(out, o + 36, &ib, 4);
    }
    std::vector<std::uint8_t> info;
    info_of(src, c.mode == 1 && c.color_per == 1 ? src.bits.size() : n, &info);
    c.resources.push_back({"segments", std::uint32_t(count), std::move(out)});
    c.resources.push_back({"info", 1, std::move(info)});
}

void arrows_of(const wire::SceneUpdate& u, Converted& c) {
    c.piece = "arrows";
    Rows o, v;
    std::string why;
    Source src;
    if (!rows(u.array("origins"), &o) || !rows(u.array("vectors"), &v) || !source_of(u, &src, &why) || src.mode == 3)
        return void(c.fallback = "arrows of another shape");
    if (too_many(c, uint64_t(o.n), kArrowsMax, "arrows")) return;
    c.count = uint64_t(o.n);
    c.color_mode = src.mode;
    std::vector<std::uint8_t> out(std::size_t(o.n) * 32, 0);
    for (int64_t i = 0; i < o.n; ++i) {
        const float po[3] = {o.at(i, 0), o.at(i, 1), o.at(i, 2)};
        float pv[3] = {0, 0, 0};
        std::uint32_t index = kNone;
        if (i < v.n) {
            for (int a = 0; a < 3; ++a) pv[a] = v.at(i, a);
            index = std::uint32_t(i);
        }
        const std::uint32_t color = src.at(uint64_t(i));
        const std::size_t at = std::size_t(i) * 32;
        put(out, at, po, 12);
        put(out, at + 12, &color, 4);
        put(out, at + 16, pv, 12);
        put(out, at + 28, &index, 4);
    }
    std::vector<std::uint8_t> info;
    info_of(src, uint64_t(o.n), &info);
    c.resources.push_back({"arrows", std::uint32_t(o.n), std::move(out)});
    c.resources.push_back({"info", 1, std::move(info)});
}

void boxes_of(const wire::SceneUpdate& u, Converted& c) {
    c.piece = "boxes";
    Rows ctr, half, rot;
    std::string why;
    Source src;
    if (!rows(u.array("centers"), &ctr) || !rows(u.array("half_sizes"), &half) || !source_of(u, &src, &why) ||
        src.mode == 1)
        return void(c.fallback = "boxes of another shape");
    const bool has_rot = u.array("rotations") != nullptr;
    if (has_rot && (!rows(u.array("rotations"), &rot) || rot.d != 4)) return void(c.fallback = "boxes of another shape");
    int64_t n = ctr.n;
    if (half.n != 1) n = std::min(n, half.n);
    if (too_many(c, uint64_t(n), kBoxesMax, "boxes")) return;
    c.count = uint64_t(n);
    c.color_mode = src.mode == 2 ? 1u : src.mode == 3 ? 2u : 0u;
    const float dims = ctr.d >= 3 ? 3.0f : 2.0f;
    bool alpha = false;
    std::vector<std::uint8_t> out(std::size_t(n) * 64, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t h = half.n == 1 ? 0 : i;
        const float center[4] = {ctr.at(i, 0), ctr.at(i, 1), dims == 3.0f ? ctr.at(i, 2) : 0.0f, dims};
        const float hs[4] = {half.at(h, 0), half.at(h, 1), dims == 3.0f ? half.at(h, 2) : 0.0f, 0.0f};
        float q[4] = {0, 0, 0, 1};
        if (has_rot && dims == 3.0f && (rot.n == 1 || i < rot.n))
            for (int a = 0; a < 4; ++a) q[a] = rot.at(rot.n == 1 ? 0 : i, a);
        const std::uint32_t rgba = src.mode == 2 ? src.at(uint64_t(i)) : 0u, index = std::uint32_t(i), klass = 0;
        alpha = alpha || (src.mode == 2 && (rgba >> 24) < 255u);
        const std::size_t at = std::size_t(i) * 64;
        put(out, at, center, 16);
        put(out, at + 16, hs, 16);
        put(out, at + 32, q, 16);
        put(out, at + 48, &rgba, 4);
        put(out, at + 52, &klass, 4);
        put(out, at + 56, &index, 4);
    }
    if (n > 0 && alpha) {
        const float one = 1.0f;
        put(out, 28, &one, 4);   // box 0's half_size.w: an alpha below 255
    }
    if (src.mode == 3) {
        c.class_ids = src.ids;
        c.class_resource = 0;
    }
    c.resources.push_back({"boxes", std::uint32_t(n), std::move(out)});
}

void mesh_of(const wire::SceneUpdate& u, const Batch& b, Converted& c) {
    c.piece = "mesh";
    Rows pos, nrm;
    std::string why;
    Source src;
    if (b.texture || u.array("uvs") || u.array("texture")) return void(c.fallback = "textured mesh");
    if (!rows(u.array("positions"), &pos) || !source_of(u, &src, &why) || src.mode == 3)
        return void(c.fallback = "mesh of another shape");
    const bool normals = u.array("normals") != nullptr;
    if (normals && (!rows(u.array("normals"), &nrm) || nrm.d < 3)) return void(c.fallback = "mesh of another shape");
    const wire::Section* ix = u.array("indices");
    const bool indexed = ix != nullptr;
    int64_t t = pos.n / 3;
    if (indexed) {
        t = rows_of(ix, CALIPER_NET_DTYPE_I32, 4, 3);
        if (t < 0) return void(c.fallback = "mesh of another shape");
    }
    if (!normals && indexed) return void(c.fallback = "indexed mesh with flat faces");
    if (too_many(c, uint64_t(pos.n), kMeshVerticesMax, "mesh vertices") ||
        too_many(c, uint64_t(t), kMeshTrianglesMax, "mesh triangles"))
        return;
    c.count = uint64_t(pos.n);
    c.color_mode = src.mode;
    c.shading = normals ? 1u : 0u;
    c.indexed = indexed ? 1u : 0u;
    std::vector<std::uint8_t> verts(std::size_t(pos.n) * 48, 0);
    bool alpha = false;
    for (int64_t i = 0; i < pos.n; ++i) {
        const std::uint32_t s = src.at(uint64_t(i));
        const float p[4] = {pos.at(i, 0), pos.at(i, 1), pos.at(i, 2), src.mode == 1 ? float_of(s) : float_of(kNanBits)};
        float nv[4] = {0, 0, 0, 0};
        if (normals && i < nrm.n)
            for (int a = 0; a < 3; ++a) nv[a] = nrm.at(i, a);
        const std::uint32_t rgba = src.mode == 2 ? s : 0u, index = std::uint32_t(i);
        alpha = alpha || (src.mode == 2 && (rgba >> 24) < 255u);
        const std::size_t at = std::size_t(i) * 48;
        put(verts, at, p, 16);
        put(verts, at + 16, nv, 16);
        put(verts, at + 32, &rgba, 4);
        put(verts, at + 36, &index, 4);
    }
    if (pos.n > 0 && alpha) {
        const float one = 1.0f;
        put(verts, 28, &one, 4);   // vertex 0's normal.w: an alpha below 255
    }
    std::vector<std::uint8_t> tris(std::size_t(t) * 16, 0);
    const std::uint32_t last = pos.n > 0 ? std::uint32_t(pos.n - 1) : 0;
    const std::int32_t* ids = indexed ? reinterpret_cast<const std::int32_t*>(ix->data) : nullptr;
    for (int64_t k = 0; k < t; ++k) {
        std::uint32_t v[4] = {std::uint32_t(3 * k), std::uint32_t(3 * k + 1), std::uint32_t(3 * k + 2), 0};
        if (ids)
            for (int a = 0; a < 3; ++a) v[a] = std::min(std::uint32_t(ids[3 * k + a]), last);
        put(tris, std::size_t(k) * 16, v, 16);
    }
    c.resources.push_back({"vertices", std::uint32_t(pos.n), std::move(verts)});
    c.resources.push_back({"triangles", std::uint32_t(t), std::move(tris)});
}

// The info of an image piece: its size and the identity pose (the pose is in the matrix).
std::vector<std::uint8_t> image_info(std::uint32_t w, std::uint32_t h, std::uint32_t channels, std::size_t bytes) {
    std::vector<std::uint8_t> out(bytes, 0);
    const std::uint32_t size[4] = {w, h, channels, 0};
    put(out, 0, size, 16);
    for (int k = 0; k < 4; ++k) {
        const float one = 1.0f;
        put(out, std::size_t(16 + k * 16 + k * 4), &one, 4);
    }
    return out;
}

void image_of(const wire::SceneUpdate& u, const Batch& b, Converted& c) {
    const wire::Section* px = u.array("pixels");
    const uint64_t w = b.image_width, h = b.image_height, n = w * h;
    c.width = std::uint32_t(w);
    c.height = std::uint32_t(h);
    c.count = n;
    if (b.image_form == caliper_scene::ImageForm::Color || b.image_form == caliper_scene::ImageForm::Grey) {
        c.piece = "image";
        const bool grey = b.image_form == caliper_scene::ImageForm::Grey;
        if (!px || px->dtype != CALIPER_NET_DTYPE_U8 || px->size < n * (grey ? 1 : 4))
            return void(c.fallback = "image of another shape");
        if (too_many(c, n, kImagePixelsMax, "image pixels")) return;
        std::vector<std::uint8_t> out(std::size_t(n) * 4);
        if (grey) {
            for (uint64_t i = 0; i < n; ++i) {
                const std::uint32_t g = px->data[i], rgba = 0xff000000u | g | (g << 8) | (g << 16);
                put(out, std::size_t(i) * 4, &rgba, 4);
            }
        } else {
            std::memcpy(out.data(), px->data, std::size_t(n) * 4);
        }
        c.resources.push_back({"pixels", std::uint32_t(n), std::move(out)});
        c.resources.push_back({"info", 1, image_info(c.width, c.height, grey ? 1 : 4, 80)});
    } else if (b.image_form == caliper_scene::ImageForm::Depth) {
        c.piece = "depth";
        const bool u16 = px && px->dtype == CALIPER_NET_DTYPE_U16;
        if (!px || (!u16 && px->dtype != CALIPER_NET_DTYPE_F32) || px->size < n * (u16 ? 2 : 4))
            return void(c.fallback = "depth image of another shape");
        if (too_many(c, n, kImagePixelsMax, "depth pixels")) return;
        const caliper_net::Json* sp = u.params.find("scale");
        const float scale = float(sp && sp->is_number() ? sp->as_double() : (u16 ? 0.001 : 1.0));
        std::vector<std::uint8_t> out(std::size_t(n) * 4);
        float lo = 0, hi = 0, seen = 0;
        for (uint64_t i = 0; i < n; ++i) {
            float d;
            if (u16) {
                std::uint16_t raw;
                std::memcpy(&raw, px->data + i * 2, 2);
                d = float(raw) * scale;
            } else {
                float raw;
                std::memcpy(&raw, px->data + i * 4, 4);
                d = scale == 1.0f ? raw : raw * scale;
            }
            put(out, std::size_t(i) * 4, &d, 4);
            if (std::isfinite(d) && d != 0.0f) {
                lo = seen != 0 ? std::min(lo, d) : d;
                hi = seen != 0 ? std::max(hi, d) : d;
                seen = 1;
            }
        }
        std::vector<std::uint8_t> info = image_info(c.width, c.height, 1, 96);
        const float range[4] = {lo, hi, seen, 0};
        put(info, 80, range, 16);
        c.resources.push_back({"pixels", std::uint32_t(n), std::move(out)});
        c.resources.push_back({"info", 1, std::move(info)});
    } else {
        c.piece = "segmentation";
        const bool u16 = px && px->dtype == CALIPER_NET_DTYPE_U16;
        if (!px || px->size < n * (u16 ? 2 : 1)) return void(c.fallback = "mask of another shape");
        if (too_many(c, n, kMaskPixelsMax, "mask pixels")) return;
        std::vector<std::uint8_t> out(std::size_t(n) * 4);
        for (uint64_t i = 0; i < n; ++i) {
            std::uint32_t id = 0;
            if (u16) {
                std::uint16_t raw;
                std::memcpy(&raw, px->data + i * 2, 2);
                id = raw;
            } else {
                id = px->data[i];
            }
            put(out, std::size_t(i) * 4, &id, 4);
        }
        c.resources.push_back({"pixels", std::uint32_t(n), std::move(out)});
        c.resources.push_back({"palette", std::uint32_t(kPaletteSize), std::vector<std::uint8_t>(kPaletteSize * 4, 0)});
        c.resources.push_back({"info", 1, image_info(c.width, c.height, 1, 80)});
        c.class_resource = 1;
        c.class_ids = {0};   // marks the palette as following the class table
    }
}

// The params of a piece, packed in the manifest's order as the struct Params holds them: the
// given values, else the defaults, each clamped to its range.
std::vector<std::uint8_t> pack(const cs::Piece& piece, const std::vector<std::pair<std::string, double>>& values) {
    std::vector<std::uint8_t> out(piece.params.size() * 4, 0);
    for (std::size_t i = 0; i < piece.params.size(); ++i) {
        const cs::PieceParam& p = piece.params[i];
        double value = p.default_value;
        for (const auto& [name, v] : values)
            if (name == p.name) value = v;
        if (p.min && value < *p.min) value = *p.min;
        if (p.max && value > *p.max) value = *p.max;
        std::uint32_t bits = 0;
        if (p.type == cs::ParamType::f32) {
            bits = bits_of(float(value));
        } else if (p.type == cs::ParamType::i32) {
            const double c = std::isfinite(value) ? std::min(2147483647.0, std::max(-2147483648.0, value)) : 0.0;
            bits = std::uint32_t(std::int32_t(c));
        } else {
            const double c = std::isfinite(value) ? std::min(4294967295.0, std::max(0.0, value)) : 0.0;
            bits = std::uint32_t(c);
        }
        std::memcpy(out.data() + i * 4, &bits, 4);
    }
    return out;
}

bool is_identity(const Mat4& m) {
    const Mat4 id;
    for (int i = 0; i < 16; ++i)
        if (m.m[i] != id.m[i]) return false;
    return true;
}

}  // namespace

void ScenePieces::take(const std::string& path, Primitive p, bool fixed, const wire::SceneUpdate& u, const Batch& b) {
    auto c = std::make_shared<Converted>();
    c->generation = ++generation_;
    switch (p) {
        case Primitive::Points: points_of(u, b, *c); break;
        case Primitive::Lines: lines_of(u, b, *c); break;
        case Primitive::Arrows: arrows_of(u, *c); break;
        case Primitive::Boxes: boxes_of(u, *c); break;
        case Primitive::Mesh: mesh_of(u, b, *c); break;
        case Primitive::Image: image_of(u, b, *c); break;
    }
    if (c->fallback.empty() && !library_piece(c->piece)) c->fallback = "no library piece " + c->piece;
    if (!c->fallback.empty()) c->resources.clear();
    parts_[{{path, int(p)}, fixed}] = std::move(c);
}

void ScenePieces::drop(const std::string& path, Primitive p, bool fixed) { parts_.erase({{path, int(p)}, fixed}); }

void ScenePieces::clear() {
    parts_.clear();
    refused_.clear();
}

void ScenePieces::set_override(const std::string& path, Primitive p, PieceBatch batch) {
    auto c = std::make_shared<Converted>();
    c->generation = ++generation_;
    c->is_override = true;
    c->piece = batch.piece;
    c->params = std::move(batch.params);
    c->resources = std::move(batch.resources);
    if (!library_piece(c->piece)) {
        c->fallback = "no library piece " + c->piece;
        c->resources.clear();
    }
    overrides_[{path, int(p)}] = std::move(c);
}

void ScenePieces::clear_override(const std::string& path, Primitive p) { overrides_.erase({path, int(p)}); }

void ScenePieces::sync(const caliper_scene::SceneTree& tree, const SyncState& state) {
    tree_ = &tree;
    counts_ = ShapeCounts{};
    std::vector<std::unique_ptr<Program>> next;
    bool changed = false;
    std::vector<std::uint32_t> texels;
    const bool through = bool(state.through);
    if (!frustum_) {
        frustum_ = std::make_shared<Converted>();
        frustum_->piece = "frustum";
        frustum_->generation = ++generation_;
        frustum_->count = 1;
        std::vector<std::uint8_t> bytes(80, 0);
        for (int k = 0; k < 4; ++k) {
            const float one = 1.0f;
            put(bytes, std::size_t(k * 16 + k * 4), &one, 4);
        }
        frustum_->resources.push_back({"frustum", 1, std::move(bytes)});
        if (!library_piece("frustum")) frustum_->fallback = "no library piece frustum";
    }
    for (std::uint32_t idx : tree.path_order()) {
        const EntityId e = tree.entity_at(idx);
        if (!tree.drawable(e)) continue;
        const std::string& path = tree.path(e);
        // A camera entity's frustum: frustum@1, with the camera's frame and image plane in its
        // matrix as the scene core composes them; none in 2-D, as the scene core draws none.
        if (const caliper_scene::Pinhole* pin = tree.camera(e); pin && state.three_d) {
            if (through) {
                ++counts_.builtin;
                ++counts_.fallbacks["view through a camera"];
            } else if (!frustum_->fallback.empty()) {
                ++counts_.builtin;
                ++counts_.fallbacks[frustum_->fallback];
            } else {
                auto pr = std::make_unique<Program>();
                pr->id = "scene:" + path + "#camera";
                pr->path = path;
                pr->primitive = Primitive::Lines;
                pr->frustum = true;
                pr->conv = frustum_;
                pr->piece = library_piece("frustum");
                const double d = tree.plane_distance(e);
                Mat4 f;
                f.at(0, 0) = pin->width * d / pin->fx;
                f.at(0, 2) = -pin->cx * d / pin->fx;
                f.at(1, 1) = pin->height * d / pin->fy;
                f.at(1, 2) = -pin->cy * d / pin->fy;
                f.at(2, 2) = d;
                pr->model = tree.camera_frame(e) * f;
                pr->params = pack(*pr->piece, {{"width", 2.0}, {"color", double(caliper_scene::CompileOptions{}.camera_rgba)}});
                if (refused_.count(pr->id)) {
                    ++counts_.builtin;
                    ++counts_.fallbacks["piece stopped by its runtime"];
                } else {
                    auto pid = pick_ids_.find(pr->id);
                    if (pid == pick_ids_.end()) pid = pick_ids_.emplace(pr->id, next_pick_++).first;
                    pr->pick_id = pid->second;
                    ++counts_.pieces;
                    next.push_back(std::move(pr));
                }
            }
        }
        for (int k = 1; k <= caliper_scene::kPrimitiveCount; ++k) {
            const Primitive prim = Primitive(k);
            const Batch* b = tree.batch(e, prim);
            if (!b || b->count() == 0) continue;
            auto fall = [&](const std::string& kind) {
                ++counts_.builtin;
                ++counts_.fallbacks[kind];
            };
            std::shared_ptr<Converted> c;
            if (auto o = overrides_.find({path, k}); o != overrides_.end()) c = o->second;
            else if (auto q = parts_.find({{path, k}, tree.batch_static(e, prim)}); q != parts_.end()) c = q->second;
            if (!c) {
                fall("batch without arrays to convert");
                continue;
            }
            if (!c->fallback.empty()) {
                fall(c->fallback);
                continue;
            }
            const EntityId pcam = tree.pixel_space_of(e);
            if (through) {
                fall("view through a camera");
                continue;
            }
            if (state.stage && state.stage(path, prim)) {
                fall("stage program");
                continue;
            }
            const std::string id = "scene:" + path + "#" + caliper_scene::primitive_name(prim);
            if (auto r = refused_.find(id); r != refused_.end()) {
                fall("piece stopped by its runtime");
                continue;
            }
            const cs::Piece* piece = library_piece(c->piece);
            auto pr = std::make_unique<Program>();
            pr->id = id;
            pr->path = path;
            pr->primitive = prim;
            pr->conv = c;
            pr->piece = piece;
            pr->model = tree.world(e);
            pr->depth_offset = state.three_d ? b->depth_offset : 0;
            pr->is_image = prim == Primitive::Image;
            pr->camera = pcam ? pcam.index + 1 : 0;
            pr->translucent = b->translucent || (prim == Primitive::Image && b->opacity < 1.0f) ||
                              (prim != Primitive::Image && c->color_mode == 0 && (b->flat_rgba >> 24) < 255u);
            // The params from the batch as the scene view resolved it, with its styles.
            std::vector<std::pair<std::string, double>> v;
            const double flat = double(b->flat_rgba);
            switch (prim) {
                case Primitive::Points: {
                    const float size = b->size_px > 0 ? b->size_px : caliper_scene::default_point_size(c->count);
                    if (!(size >= 1.0f && size <= 64.0f)) {
                        fall("point size outside 1 to 64 pixels");
                        continue;
                    }
                    v = {{"size", size},
                         {"shape", b->point_shape == caliper_scene::PointShape::Disc ? 1.0 : 0.0},
                         {"color_mode", double(c->color_mode)},
                         {"color", flat},
                         {"colormap", double(b->colormap)},
                         {"vmin", b->vmin},
                         {"vmax", b->vmax}};
                    break;
                }
                case Primitive::Lines:
                case Primitive::Arrows:
                    if (b->width_px > 128.0f) {
                        fall("width above 128 pixels");
                        continue;
                    }
                    v = {{"width", b->width_px},   {"mode", double(c->mode)},         {"color_mode", double(c->color_mode)},
                         {"color", flat},          {"color_per", double(c->color_per)}, {"colormap", double(b->colormap)},
                         {"range", 0.0},           {"vmin", b->vmin},                 {"vmax", b->vmax},
                         {"depth_offset", 0.0}};
                    break;
                case Primitive::Boxes:
                    if (b->width_px < 0.5f || b->width_px > 64.0f) {
                        fall("box edge width outside 0.5 to 64 pixels");
                        continue;
                    }
                    v = {{"width", b->width_px},
                         {"fill", b->fill ? 1.0 : 0.0},
                         {"color_mode", double(c->color_mode)},
                         {"color", flat},
                         {"depth_offset", 0.0}};
                    break;
                case Primitive::Mesh:
                    v = {{"color_mode", double(c->color_mode)}, {"color", flat},
                         {"colormap", double(b->colormap)},     {"vmin", b->vmin},
                         {"vmax", b->vmax},                     {"shading", double(c->shading)},
                         {"indexed", double(c->indexed)},       {"depth_offset", 0.0}};
                    break;
                case Primitive::Image: {
                    v = {{"opacity", b->opacity}, {"scale", 1.0}, {"placement", 0.0}};
                    if (c->piece == "depth") {
                        const bool cloud = b->cloud && bool(pcam);
                        v.insert(v.end(), {{"colormap", double(b->colormap)}, {"vmin", b->vmin}, {"vmax", b->vmax},
                                           {"auto_range", 0.0}, {"cloud", cloud ? 1.0 : 0.0}});
                        if (cloud) {
                            // The cloud back-projects on the camera's image plane: the camera's pose,
                            // its pinhole and its plane, with the image at the pixel space's origin.
                            const caliper_scene::Pinhole* pin = tree.camera(pcam);
                            if (!pin || !is_identity(tree.pixel_local(e))) {
                                fall("depth cloud in a moved pixel space");
                                continue;
                            }
                            const Mat4* pose = tree.transform(pcam);
                            pr->model = tree.world(pcam);
                            (void)pose;
                            v.insert(v.end(), {{"placement", 1.0},
                                               {"fx", pin->fx},
                                               {"fy", pin->fy},
                                               {"cx", pin->cx},
                                               {"cy", pin->cy},
                                               {"plane", tree.plane_distance(pcam)},
                                               {"convention", pin->convention == caliper_scene::Convention::Rub ? 1.0 : 0.0}});
                        }
                    }
                    break;
                }
            }
            if (c->is_override) v = c->params;
            if (!c->class_ids.empty() && c->classes_revision != tree.classes_revision()) {
                if (texels.empty()) {
                    texels.resize(65536);
                    tree.class_texels(texels.data());
                }
                PieceBatch::Resource& r = c->resources[c->class_resource];
                if (c->piece == "segmentation") {
                    bool high = false;
                    for (const caliper_scene::ClassEntry& ce : tree.classes()) high = high || ce.id >= kPaletteSize;
                    if (high) c->fallback = "class above 4,095 in a mask";
                    for (std::size_t i = 0; i < kPaletteSize; ++i) put(r.bytes, i * 4, &texels[i], 4);
                } else {
                    const std::size_t stride = c->piece == "points" ? 32 : 64, at = c->piece == "points" ? 16 : 52;
                    for (std::size_t i = 0; i < r.count; ++i) {
                        const std::int32_t id = c->class_ids.size() == 1 ? c->class_ids[0]
                                                : i < c->class_ids.size() ? c->class_ids[i]
                                                                          : -1;
                        const std::uint32_t rgba = class_color(texels, id);
                        put(r.bytes, i * stride + at, &rgba, 4);
                    }
                }
                c->classes_revision = tree.classes_revision();
                c->generation = ++generation_;
                if (!c->fallback.empty()) {
                    fall(c->fallback);
                    continue;
                }
            }
            pr->params = pack(*piece, v);
            auto pid = pick_ids_.find(id);
            if (pid == pick_ids_.end()) pid = pick_ids_.emplace(id, next_pick_++).first;
            pr->pick_id = pid->second;
            ++counts_.pieces;
            next.push_back(std::move(pr));
        }
    }
    // The scene core's order: opaque, then translucent, in path order within each; a camera's
    // plane with a translucent image draws with the translucent group, its images first.
    std::vector<std::uint32_t> translucent_planes;
    for (const auto& pr : next)
        if (pr->is_image && pr->camera && pr->translucent) translucent_planes.push_back(pr->camera);
    for (auto& pr : next) {
        const bool plane = pr->camera && std::find(translucent_planes.begin(), translucent_planes.end(), pr->camera) !=
                                             translucent_planes.end();
        pr->group = plane || pr->translucent ? 1 : 0;
        pr->sub = pr->camera && !pr->is_image ? 1 : 0;
    }
    std::stable_sort(next.begin(), next.end(), [](const std::unique_ptr<Program>& a, const std::unique_ptr<Program>& b) {
        return a->group != b->group ? a->group < b->group : a->sub < b->sub;
    });
    // A program keeps what it gave its runtime while it stays; the list changes when a program
    // comes, goes or changes its piece.
    if (next.size() != programs_.size()) changed = true;
    for (std::size_t i = 0; i < next.size() && !changed; ++i)
        changed = next[i]->id != programs_[i]->id || next[i]->conv->piece != programs_[i]->conv->piece;
    for (auto& n : next)
        for (auto& o : programs_)
            if (o->id == n->id && o->conv == n->conv) {
                n->given = o->given;
                n->given_generation = o->given_generation;
            }
    programs_ = std::move(next);
    if (changed) ++revision_;
    // Pick ids of programs that are gone.
    for (auto it = pick_ids_.begin(); it != pick_ids_.end();) {
        bool live = false;
        for (const auto& pr : programs_) live = live || pr->id == it->first;
        it = live ? std::next(it) : pick_ids_.erase(it);
    }
}

bool ScenePieces::draws(EntityId e, Primitive p) const {
    if (!tree_ || programs_.empty()) return false;
    const std::string& path = tree_->path(e);
    for (const auto& pr : programs_)
        if (!pr->frustum && pr->primitive == p && pr->path == path) return refused_.find(pr->id) == refused_.end();
    return false;
}

bool ScenePieces::draws_frustum(EntityId e) const {
    if (!tree_ || programs_.empty()) return false;
    const std::string& path = tree_->path(e);
    for (const auto& pr : programs_)
        if (pr->frustum && pr->path == path) return refused_.find(pr->id) == refused_.end();
    return false;
}

bool ScenePieces::any() const { return !programs_.empty(); }

void ScenePieces::set_camera(const caliper_scene::CameraMatrices& cm) { cm_ = cm; }

bool ScenePieces::owns(std::uint32_t pick_id) const {
    for (const auto& pr : programs_)
        if (pr->pick_id == pick_id) return true;
    return false;
}

bool ScenePieces::hit(const caliper_scene::FrameHit& fh, const caliper_scene::PickWindow& w, caliper_scene::Hit* out) const {
    if (!tree_) return false;
    for (const auto& pr : programs_) {
        if (pr->pick_id != fh.pick_id) continue;
        const Converted& c = *pr->conv;
        if (c.count > 0 && fh.item >= c.count && pr->primitive != Primitive::Lines) return false;
        caliper_scene::Hit h;
        h.entity = tree_->find(pr->path);
        if (!h.entity) return false;
        h.primitive = pr->primitive;
        h.part = pr->frustum ? caliper_scene::DrawPart::Frustum : caliper_scene::DrawPart::Batch;
        h.item = fh.item;
        h.texel[0] = fh.texel[0];
        h.texel[1] = fh.texel[1];
        if (pr->primitive == Primitive::Image && c.width > 0) {
            // The scene core's image is one quad, item 0, and its pick gives the pixel under the pointer.
            h.item = 0;
            h.has_pixel = true;
            h.pixel[0] = fh.item % c.width;
            h.pixel[1] = fh.item / c.width;
        }
        h.depth = fh.depth;
        Mat4 inv;
        if (h.depth < 1.0f && w.view_width > 0 && w.view_height > 0 && caliper_scene::inverse(cm_.proj * cm_.view, &inv)) {
            const double cx = w.x0 + fh.texel[0] + 0.5, cy = w.y0 + fh.texel[1] + 0.5;
            const double X = 2.0 * cx / w.view_width - 1.0, Y = 1.0 - 2.0 * cy / w.view_height, Z = h.depth;
            double p[4];
            for (int r = 0; r < 4; ++r) p[r] = inv.at(r, 0) * X + inv.at(r, 1) * Y + inv.at(r, 2) * Z + inv.at(r, 3);
            if (p[3] != 0) {
                h.position = {p[0] / p[3], p[1] / p[3], p[2] / p[3]};
                h.has_position = std::isfinite(h.position.x) && std::isfinite(h.position.y) && std::isfinite(h.position.z);
            }
        }
        *out = h;
        return true;
    }
    return false;
}

ShapeCounts ScenePieces::counts() const { return counts_; }

void ScenePieces::installed(const std::string& program) {
    for (auto& pr : programs_)
        if (pr->id == program) pr->given = false;
}

std::vector<PanelProgram> ScenePieces::programs() const {
    std::vector<PanelProgram> out;
    for (const auto& pr : programs_) {
        PanelProgram p;
        p.piece = *pr->piece;
        p.instance.program = pr->id;
        p.instance.panel = "scene";
        p.pick_id = pr->pick_id;
        p.revision = std::hash<std::string>{}(pr->conv->piece);
        out.push_back(std::move(p));
    }
    return out;
}

void ScenePieces::frame(DrawRuntime& runtime, FrameInputs* in) {
    Program* pr = nullptr;
    for (auto& p : programs_)
        if (p->id == runtime.instance().program) pr = p.get();
    if (!pr) return;
    const Converted& c = *pr->conv;
    if (!pr->given || pr->given_generation != c.generation) {
        for (const PieceBatch::Resource& r : c.resources) {
            ResourceVersion v;
            v.resource = r.id;
            v.step = std::int64_t(c.generation);
            v.serial = c.generation;
            v.count = r.count;
            v.bytes = r.bytes.data();
            v.size = r.bytes.size();
            if (!runtime.set_resource(v).ok()) break;
        }
        pr->given = true;
        pr->given_generation = c.generation;
    }
    in->params = pr->params;
    in->cursor_step = in->latest_step = std::int64_t(c.generation);
    in->live = true;
    // The batch's matrix as the scene core composes it: in double, the depth offset in clip space,
    // rounded to f32 once.
    const Mat4 vm = cm_.view * pr->model;
    Mat4 mvp = cm_.proj * vm;
    if (pr->depth_offset != 0) {
        Mat4 d;
        d.m[14] = -double(pr->depth_offset) * caliper_scene::kDepthOffsetUnit;
        mvp = d * mvp;
    }
    Mat4 inv;
    if (!caliper_scene::inverse(mvp, &inv)) inv = Mat4{};
    float n0[4], n1[4], n2[4];
    caliper_scene::normal_matrix_columns(vm, n0, n1, n2);
    for (int i = 0; i < 16; ++i) {
        in->clip_from_world[std::size_t(i)] = float(mvp.m[i]);
        in->world_from_clip[std::size_t(i)] = float(inv.m[i]);
    }
    const float* cols[3] = {n0, n1, n2};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            in->view_from_world[std::size_t(col * 4 + row)] = col < 3 ? (row < 3 ? cols[col][row] : 0.0f) : (row == 3 ? 1.0f : 0.0f);
}

void ScenePieces::status(const DrawRuntime& runtime) {
    const DrawStatus st = runtime.status();
    if (st.state != DrawState::refused && st.state != DrawState::stopped) return;
    const std::string& id = runtime.instance().program;
    if (refused_.emplace(id, 0).second) ++revision_;
}

}  // namespace caliper_draw
