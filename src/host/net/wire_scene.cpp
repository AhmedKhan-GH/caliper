// wire_scene.cpp: protocol 3.2 scenes subset (SceneUpdate, SceneTransforms,
// SceneClear, SceneClasses, scene view metadata, and SplitUpdates reassembly).
#include "net/wire_internal.h"

#include <cstring>
#include <set>

namespace caliper_net {
namespace wire {

namespace {

using namespace detail;

constexpr int kPartCount = 7;
const char* const kParts[kPartCount] = {CALIPER_NET_PART_POINTS, CALIPER_NET_PART_LINES, CALIPER_NET_PART_ARROWS,
                                        CALIPER_NET_PART_BOXES,  CALIPER_NET_PART_MESH,  CALIPER_NET_PART_IMAGE,
                                        CALIPER_NET_PART_CAMERA};
enum { kPoints, kLines, kArrows, kBoxes, kMesh, kImage, kCamera };
// The array sections of each part in the order a sender writes them (PART_SECTIONS).
const char* const kPartSections[kPartCount][8] = {
    {"positions", "colors", "values", "class_ids", "radii", nullptr},
    {"positions", "strip_lengths", "indices", "colors", "values", nullptr},
    {"origins", "vectors", "colors", "values", nullptr},
    {"centers", "half_sizes", "rotations", "colors", "class_ids", nullptr},
    {"positions", "indices", "normals", "colors", "values", "uvs", "texture", nullptr},
    {"pixels", nullptr},
    {nullptr}};
// The keys of each part in the order their types are checked (PART_KEYS).
const char* const kPartKeys[kPartCount][11] = {
    {"color", "radius", "labels", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"mode", "color_per", "color", "width", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"color", "width", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"color", "labels", "fill", "width", "stats", nullptr},
    {"color", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"form", "opacity", "scale", "cloud", "colormap", "flags", "vmin", "vmax", "format", "stats", nullptr},
    {"fx", "fy", "cx", "cy", "width", "height", "convention", "plane_distance", nullptr}};
const char* const kCameraRequired[] = {"fx", "fy", "cx", "cy", "width", "height", "convention", nullptr};

// A VIEWS scene's "view_camera" as a viewer reads it (_view_camera): null for none, for a 2-D scene and
// for a value it cannot read.
Json view_camera_of(const Json& sc, bool is3d) {
    const Json* vc = sc.find("view_camera");
    if (!vc || vc->is_null() || !is3d) return Json();
    if (vc->is_string()) return !vc->as_string().empty() && entity_path_ok(vc->as_string()) ? *vc : Json();
    if (!vc->is_object()) return Json();
    Json out = Json::object();
    for (const auto& kv : vc->members()) {
        const std::string& k = kv.first;
        const Json& v = kv.second;
        if (k != "target" && k != "distance" && k != "azimuth" && k != "elevation") return Json();
        if (v.is_null()) continue;
        if (k == "target") {
            if (!v.is_array() || v.size() != 3) return Json();
            for (const Json& x : v.items())
                if (!is_finite(x)) return Json();
        } else if (k == "distance") {
            if (!is_positive_num(v)) return Json();
        } else if (k == "azimuth") {
            if (!is_finite(v)) return Json();
        } else if (!is_finite(v) || v.as_double() < -89 || v.as_double() > 89) {
            return Json();
        }
        out.set(k, v);
    }
    return out;
}

bool is_path(const Json& v) { return v.is_string() && entity_path_ok(v.as_string()); }
bool is_rgba(const Json& v) {
    if (!v.is_array() || v.size() != 4) return false;
    for (const Json& x : v.items()) {
        int64_t i = 0;
        if (!x.get_i64(&i) || i < 0 || i > 255) return false;
    }
    return true;
}
bool is_stats(const Json& v) {
    if (!v.is_array() || v.size() > CALIPER_NET_MAX_STATS) return false;
    for (const Json& x : v.items())
        if (!(x.is_null() || x.is_number())) return false;
    return true;
}
bool is_labels(const Json& v) {
    if (!v.is_array() || v.size() < 1 || v.size() > CALIPER_NET_MAX_LABELS) return false;
    for (const Json& x : v.items())
        if (!x.is_string() || x.as_string().size() > CALIPER_NET_NAME_MAX) return false;
    return true;
}
bool is_side(const Json& v) {
    int64_t i = 0;
    return v.get_i64(&i) && i >= 1 && i <= (int64_t)CALIPER_NET_MAX_IMAGE_SIDE;
}

// The check of a part key and what it expects, as _KEY_CHECKS (a camera's width is its image width).
bool key_ok(int part, const std::string& key, const Json& v, const char** what) {
    if (key == "color") { *what = "[r, g, b, a] of integers from 0 to 255"; return is_rgba(v); }
    if (part == kCamera && key == "width") { *what = "an integer from 1 to 8192"; return is_side(v); }
    if (key == "radius" || key == "width" || key == "scale" || key == "fx" || key == "fy" ||
        key == "plane_distance") {
        *what = "a positive number";
        return is_positive_num(v);
    }
    if (key == "labels") { *what = "a list of 1 to 4096 strings of at most 63 bytes"; return is_labels(v); }
    if (key == "colormap") { *what = "an integer"; return is_int(v); }
    if (key == "flags") { *what = "an unsigned integer"; return is_uint(v); }
    if (key == "vmin" || key == "vmax") { *what = "a number"; return is_num(v); }
    if (key == "stats") { *what = "at most 64 numbers or nulls"; return is_stats(v); }
    if (key == "mode") {
        *what = "segments, strips or edges";
        return one_of(v, {CALIPER_NET_LINES_SEGMENTS, CALIPER_NET_LINES_STRIPS, CALIPER_NET_LINES_EDGES});
    }
    if (key == "color_per") {
        *what = "vertex, strip or edge";
        return one_of(v, {CALIPER_NET_COLOR_PER_VERTEX, CALIPER_NET_COLOR_PER_STRIP, CALIPER_NET_COLOR_PER_EDGE});
    }
    if (key == "fill" || key == "cloud") { *what = "a boolean"; return v.is_bool(); }
    if (key == "form") {
        *what = "color, depth, segmentation or encoded";
        return one_of(v, {CALIPER_NET_FORM_COLOR, CALIPER_NET_FORM_DEPTH, CALIPER_NET_FORM_SEGMENTATION,
                          CALIPER_NET_FORM_ENCODED});
    }
    if (key == "opacity") {
        *what = "a number from 0 to 1";
        return is_finite(v) && v.as_double() >= 0 && v.as_double() <= 1;
    }
    if (key == "format") { *what = "jpeg or png"; return one_of(v, {CALIPER_NET_FORMAT_JPEG, CALIPER_NET_FORMAT_PNG}); }
    if (key == "cx" || key == "cy") { *what = "a finite number"; return is_finite(v); }
    if (key == "height") { *what = "an integer from 1 to 8192"; return is_side(v); }
    *what = "rdf or rub";   // convention
    return one_of(v, {CALIPER_NET_CONVENTION_RDF, CALIPER_NET_CONVENTION_RUB});
}

Json scene_part_keys(const Json& m, int part) {
    Json out = Json::object();
    for (const char* const* k = kPartKeys[part]; *k; ++k) {
        const Json* v = m.find(*k);
        if (!v || v->is_null()) continue;
        const char* what = "";
        if (!key_ok(part, *k, *v, &what)) bad(std::string("'") + *k + "' must be " + what + ", got " + show(*v));
        if (std::string(*k) == "stats") {
            Json a = Json::array();
            for (const Json& x : v->items()) a.push(x.is_null() ? Json(std::nan("")) : Json(x.as_double()));
            out.set(*k, a);
        } else {
            out.set(*k, *v);
        }
    }
    return out;
}

std::string str_key(const Json& params, const char* key) {
    const Json* v = params.find(key);
    return v ? v->as_string() : std::string();
}

void required(const Frame& f, const char* part, const char* name) {
    if (!f.section(name)) bad(std::string("a ") + part + " part needs a '" + name + "' section");
}

// A required f32 (N, D) section, D 2 or 3, N at most `limit` (_rows_section).
const Section* rows_section(const Frame& f, const char* part, const char* name, uint64_t limit, const char* what,
                            int64_t* n, int64_t* d) {
    required(f, part, name);
    const Section* s = part_section(f, name, CALIPER_NET_DTYPE_F32, {-1, -1});
    *n = s->shape[0];
    *d = s->shape[1];
    if (*d != 2 && *d != 3) bad(std::string("'") + name + "' is (N, 2) or (N, 3)");
    if (*n > (int64_t)limit)
        bad(std::to_string(*n) + " " + what + ", more than " + std::to_string(limit) + " in a batch");
    return s;
}

bool has_array(const std::vector<Section>& arrays, const char* name) {
    for (const Section& s : arrays)
        if (s.name == name) return true;
    return false;
}

void scene_colours(const Frame& f, int64_t rows, std::vector<Section>* arrays) {
    const Section* c = part_section(f, "colors", CALIPER_NET_DTYPE_U8, {rows, 4});
    const Section* v = part_section(f, "values", CALIPER_NET_DTYPE_F32, {rows});
    if (c && v) bad("'colors' and 'values' together");
    if (c) arrays->push_back(*c);
    if (v) arrays->push_back(*v);
}

void label_count(const Json& params, int64_t n) {
    const Json* l = params.find("labels");
    if (l && l->size() != 1 && (int64_t)l->size() != n)
        bad(std::to_string(l->size()) + " labels for " + std::to_string(n) +
            " items: give one label for all or one per item");
}

void class_ids(const Frame& f, int64_t n, std::vector<Section>* arrays) {
    if (const Section* s = part_section(f, "class_ids", CALIPER_NET_DTYPE_I32, {n})) arrays->push_back(*s);
}

void decode_points(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(*rows_section(f, CALIPER_NET_PART_POINTS, "positions", CALIPER_NET_MAX_BATCH_POINTS,
                                    "positions", &n, &d));
    scene_colours(f, n, arrays);
    class_ids(f, n, arrays);
    if (const Section* r = part_section(f, "radii", CALIPER_NET_DTYPE_F32, {n})) arrays->push_back(*r);
    label_count(params, n);
}

void decode_lines(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    if (!params.find("mode")) bad("a lines part needs 'mode'");
    const std::string md = str_key(params, "mode");
    const std::string color_per = str_key(params, "color_per");
    const bool has_cp = params.find("color_per") != nullptr;
    int64_t n = 0, d = 0;
    arrays->push_back(*rows_section(f, CALIPER_NET_PART_LINES, "positions", CALIPER_NET_MAX_BATCH_POINTS,
                                    "positions", &n, &d));
    if (md != CALIPER_NET_LINES_EDGES && f.section("indices")) bad("lines of mode '" + md + "' take no 'indices'");
    int64_t rows = n;
    if (md == CALIPER_NET_LINES_SEGMENTS) {
        if (n % 2) bad("segments need an even number of vertices, got " + std::to_string(n));
        if (f.section("strip_lengths")) bad("segments take no 'strip_lengths'");
        if (has_cp && color_per != CALIPER_NET_COLOR_PER_VERTEX) bad("segments take color_per 'vertex'");
    } else if (md == CALIPER_NET_LINES_STRIPS) {
        const Section* sl = part_section(f, "strip_lengths", CALIPER_NET_DTYPE_I32, {-1});
        if (!sl) bad("strips need a 'strip_lengths' section");
        const int64_t cnt = sl->shape[0];
        if (cnt > (int64_t)CALIPER_NET_MAX_BATCH_ITEMS)
            bad(std::to_string(cnt) + " strips, more than " + std::to_string(CALIPER_NET_MAX_BATCH_ITEMS) +
                " in a batch");
        int64_t total = 0;
        for (int64_t i = 0; i < cnt; ++i) {
            int32_t len;
            std::memcpy(&len, sl->data + 4 * i, 4);
            if (len < 1) bad("every strip length is at least 1");
            total += len;
        }
        if (total != n)
            bad("the strip lengths add up to " + std::to_string(total) + ", not to the " + std::to_string(n) +
                " vertices");
        arrays->push_back(*sl);
        if (color_per == CALIPER_NET_COLOR_PER_EDGE) bad("strips take color_per 'vertex' or 'strip'");
        if (color_per == CALIPER_NET_COLOR_PER_STRIP) rows = cnt;
    } else {
        if (f.section("strip_lengths")) bad("edges take no 'strip_lengths'");
        const Section* idx = part_section(f, "indices", CALIPER_NET_DTYPE_I32, {-1, 2});
        if (!idx) bad("edges need an 'indices' section, (E, 2) i32");
        if (idx->shape[0] > (int64_t)CALIPER_NET_MAX_BATCH_POINTS)
            bad(std::to_string(idx->shape[0]) + " edges, more than " + std::to_string(CALIPER_NET_MAX_BATCH_POINTS) +
                " in a batch");
        arrays->push_back(*idx);
        if (color_per == CALIPER_NET_COLOR_PER_STRIP) bad("edges take color_per 'vertex' or 'edge'");
        if (color_per == CALIPER_NET_COLOR_PER_EDGE) rows = idx->shape[0];
    }
    scene_colours(f, rows, arrays);
}

void decode_arrows(const Frame& f, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(
        *rows_section(f, CALIPER_NET_PART_ARROWS, "origins", CALIPER_NET_MAX_BATCH_ITEMS, "arrows", &n, &d));
    required(f, CALIPER_NET_PART_ARROWS, "vectors");
    arrays->push_back(*part_section(f, "vectors", CALIPER_NET_DTYPE_F32, {n, d}));
    scene_colours(f, n, arrays);
}

void decode_boxes(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(
        *rows_section(f, CALIPER_NET_PART_BOXES, "centers", CALIPER_NET_MAX_BATCH_ITEMS, "boxes", &n, &d));
    required(f, CALIPER_NET_PART_BOXES, "half_sizes");
    arrays->push_back(*part_section(f, "half_sizes", CALIPER_NET_DTYPE_F32, {n, d}));
    if (const Section* r = part_section(f, "rotations", CALIPER_NET_DTYPE_F32, {n, 4})) {
        if (d != 3) bad("'rotations' are for 3-D boxes only");
        arrays->push_back(*r);
    }
    if (const Section* c = part_section(f, "colors", CALIPER_NET_DTYPE_U8, {n, 4})) arrays->push_back(*c);
    class_ids(f, n, arrays);
    label_count(params, n);
}

void decode_mesh(const Frame& f, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(*rows_section(f, CALIPER_NET_PART_MESH, "positions", CALIPER_NET_MAX_BATCH_POINTS,
                                    "positions", &n, &d));
    if (const Section* idx = part_section(f, "indices", CALIPER_NET_DTYPE_I32, {-1, 3})) {
        if (idx->shape[0] > (int64_t)CALIPER_NET_MAX_BATCH_POINTS)
            bad(std::to_string(idx->shape[0]) + " triangles, more than " +
                std::to_string(CALIPER_NET_MAX_BATCH_POINTS) + " in a batch");
        arrays->push_back(*idx);
    } else if (n % 3) {
        bad("a mesh without 'indices' needs a multiple of 3 vertices, got " + std::to_string(n));
    }
    if (const Section* nrm = part_section(f, "normals", CALIPER_NET_DTYPE_F32, {n, 3})) arrays->push_back(*nrm);
    scene_colours(f, n, arrays);
    const Section* uv = part_section(f, "uvs", CALIPER_NET_DTYPE_F32, {n, 2});
    const Section* tex = part_section(f, "texture", CALIPER_NET_DTYPE_U8, {-1, -1, 4});
    if (uv && (has_array(*arrays, "colors") || has_array(*arrays, "values")))
        bad("'uvs' and 'colors' or 'values' together");
    if (!uv != !tex) bad("'uvs' and 'texture' go together");
    if (tex) {
        const int64_t side = CALIPER_NET_MAX_IMAGE_SIDE;
        if (tex->shape[0] < 1 || tex->shape[0] > side || tex->shape[1] < 1 || tex->shape[1] > side)
            bad("a texture is 1 to 8192 pixels on a side");
        arrays->push_back(*uv);
        arrays->push_back(*tex);
    }
}

// The pixels each image form takes (_FORM_PIXELS).
bool pixels_fit(const std::string& form, const Section& p) {
    const int64_t dt = p.dtype;
    const size_t r = p.shape.size();
    if (form == CALIPER_NET_FORM_COLOR) return dt == CALIPER_NET_DTYPE_U8 && (r == 3 || r == 2);
    if (form == CALIPER_NET_FORM_DEPTH) return (dt == CALIPER_NET_DTYPE_F32 || dt == CALIPER_NET_DTYPE_U16) && r == 2;
    if (form == CALIPER_NET_FORM_SEGMENTATION)
        return (dt == CALIPER_NET_DTYPE_U16 || dt == CALIPER_NET_DTYPE_U8) && r == 2;
    return dt == CALIPER_NET_DTYPE_U8 && r == 1;   // encoded
}

void decode_image(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    if (!params.find("form")) bad("an image part needs 'form'");
    const std::string form = str_key(params, "form");
    required(f, CALIPER_NET_PART_IMAGE, "pixels");
    const Section* p = f.section("pixels");
    if (p->encoding != CALIPER_NET_ENCODING_RAW || !pixels_fit(form, *p))
        bad("the pixels of a " + form + " image cannot be dtype " + std::to_string(p->dtype) + " of rank " +
            std::to_string(p->shape.size()));
    if (form == CALIPER_NET_FORM_COLOR && p->shape.size() == 3 && p->shape[2] != 4)
        bad("colour pixels are (H, W, 4) or (H, W)");
    if (form == CALIPER_NET_FORM_ENCODED) {
        if (p->shape[0] > (int64_t)CALIPER_NET_MAX_ENCODED_BYTES)
            bad("an encoded frame of " + std::to_string(p->shape[0]) + " bytes, more than 16777216");
        if (!params.find("format")) bad("an encoded image needs 'format'");
    } else if (p->shape[0] > (int64_t)CALIPER_NET_MAX_IMAGE_SIDE || p->shape[1] > (int64_t)CALIPER_NET_MAX_IMAGE_SIDE) {
        bad("an image is at most 8192 pixels on a side");
    }
    arrays->push_back(*p);
}

void decode_camera(const Json& params) {
    for (const char* const* k = kCameraRequired; *k; ++k)
        if (!params.find(*k)) bad(std::string("a camera part needs '") + *k + "'");
}

bool is_rows_map(const Json& v) {
    if (!v.is_object()) return false;
    for (const auto& kv : v.members())
        if (!is_uint(kv.second)) return false;
    return true;
}

bool in_part(int part, const std::string& name) {
    for (const char* const* k = scene_part_sections(part); *k; ++k)
        if (name == *k) return true;
    return false;
}

// The checks of a piece of a split update (_piece), in this order: the keys of `split` (update, pieces,
// piece, first, then rows in piece 0 only), the part's keys in piece 0, and that `first` names exactly
// the piece's array sections, each raw with at least one dimension, within `rows` in piece 0.
void scene_piece(const Frame& f, const Json& m, int part, SceneUpdate& u) {
    const Json& sp = u.split;
    auto need = [&](const char* key, auto ok, const char* what) -> const Json& {
        const Json* v = sp.find(key);
        if (!v || v->is_null() || !ok(*v)) bad(std::string("'split.") + key + "' must be " + what);
        return *v;
    };
    need("update", is_uint, "an unsigned integer");
    int64_t pieces = 0, piece = 0;
    need("pieces", [&](const Json& v) {
        return v.is_int() && v.get_i64(&pieces) && pieces >= 2 && pieces <= (int64_t)CALIPER_NET_MAX_PIECES;
    }, "an integer from 2 to 16");
    need("piece", [&](const Json& v) { return v.is_int() && v.get_i64(&piece) && piece >= 0 && piece < pieces; },
         "a piece number below 'pieces'");
    const Json& first = need("first", is_rows_map, "an object of row numbers");
    const Json* rows = nullptr;
    if (piece == 0) {
        rows = &need("rows", is_rows_map, "an object of row counts");
    } else if (const Json* r = sp.find("rows"); r && !r->is_null()) {
        bad("only piece 0 carries 'split.rows'");
    }
    if (piece == 0) u.params = scene_part_keys(m, part);
    for (const Section& s : f.sections) {
        if (s.name == "stamp" || !in_part(part, s.name)) continue;   // the stamp, or a section of no array
        if (s.encoding != CALIPER_NET_ENCODING_RAW || s.shape.empty())
            bad("a piece of '" + s.name + "' is a raw array of at least one dimension");
        u.arrays.push_back(s);
    }
    bool same = first.size() == u.arrays.size();
    for (const Section& s : u.arrays) same = same && first.find(s.name) != nullptr;
    if (!same) bad("'split.first' does not name exactly the arrays of the piece");
    if (rows) {
        for (const auto& kv : rows->members())
            if (!in_part(part, kv.first)) bad("'split.rows' names '" + kv.first + "', not an array of the part");
        // Unsigned, with no sum that can wrap: a first row near 2^64 must not come back below the count.
        for (const Section& s : u.arrays) {
            const Json* r = rows->find(s.name);
            const uint64_t f0 = u64(*first.find(s.name)), n = (uint64_t)s.shape[0];
            if (!r || s.shape[0] < 0 || f0 > u64(*r) || n > u64(*r) - f0)
                bad("the rows of '" + s.name + "' in piece 0 pass 'split.rows'");
        }
    }
}

template <typename M>
void scene_common(const Json& m, M& out) {
    out.sub = u32(req(m, "sub", is_u32, "a uint32"));
    out.view = req(m, "view", is_name, "a view name").as_string();
    out.seq = u64(req(m, "seq", is_uint, "an unsigned integer"));
    out.stamp_ns = u64(req(m, "stamp_ns", is_uint, "an unsigned integer"));
}

template <typename M>
void scene_flags(const Json& m, M& out) {
    if (const Json* v = opt(m, "static", is_bool, "a boolean")) out.is_static = v->as_bool();
    if (const Json* v = opt(m, "state", is_bool, "a boolean")) out.state = v->as_bool();
    if (const Json* v = opt(m, "cursor", is_u32, "a uint32")) out.cursor = u32(*v);
}

// The library piece of each part, and of each form of an image (the piece form of SCENE_UPDATE).
bool scene_piece_fits(int part, const std::string& name, const Json& params) {
    switch (part) {
        case kPoints: return name == "points";
        case kLines: return name == "lines";
        case kArrows: return name == "arrows";
        case kBoxes: return name == "boxes";
        case kMesh: return name == "mesh";
        case kImage: {
            const Json* form = params.find("form");
            const std::string f = form && form->is_string() ? form->as_string() : "";
            return (f == CALIPER_NET_FORM_COLOR && name == "image") || (f == CALIPER_NET_FORM_DEPTH && name == "depth") ||
                   (f == CALIPER_NET_FORM_SEGMENTATION && name == "segmentation");
        }
        default: return false;
    }
}

// The keys of the piece form, in the order program, piece, resources, arrays; then the part's keys, and no
// array section.
void scene_piece_form(const Frame& f, const Json& m, int part, SceneUpdate& u) {
    u.program = req(m, "program", [](const Json& v) { return v.is_string() && scene_program_id(v.as_string()); },
                    "a scene program id scene.<n> of 1 to 9 digits")
                    .as_string();
    u.piece = req(
        m, "piece",
        [](const Json& v) {
            // A library pin: name and version as a pin's, and the hash optional.
            if (!v.is_object()) return false;
            Json p = v;
            if (!p.find("hash")) p.set("hash", Json(std::string("sha256:") + std::string(64, '0')));
            return piece_pin_ok(p);
        },
        "a library pin {name, version} with an optional hash");
    const std::string name = u.piece.find("name")->as_string();
    const bool image_piece = name == "image" || name == "depth" || name == "segmentation";
    if (part == kCamera || (part == kImage ? !image_piece : !scene_piece_fits(part, name, Json::object())))
        bad("the part '" + u.part + "' has no library piece '" + name + "'");
    u.resources = req(
        m, "resources",
        [](const Json& v) {
            if (!v.is_object() || v.size() < 1 || v.size() > CALIPER_NET_MAX_RESOURCES) return false;
            for (const auto& kv : v.members())
                if (!param_name_ok(kv.first) || !is_u32(kv.second)) return false;
            return true;
        },
        "an object of 1 to 8 resource ids, each with a uint32 count");
    const Json& names = req(
        m, "arrays",
        [](const Json& v) {
            if (!v.is_array()) return false;
            for (const Json& x : v.items())
                if (!x.is_string()) return false;
            return true;
        },
        "a list of array names");
    const char* const* table = scene_part_sections(part);
    int last = -1;
    bool ok = names.size() > 0;
    for (const Json& x : names.items()) {
        int at = -1;
        for (int k = 0; table[k]; ++k)
            if (x.as_string() == table[k]) at = k;
        ok = ok && at > last && (last >= 0 || at == 0);
        last = at;
        u.names.push_back(x.as_string());
    }
    if (!ok) bad("'arrays' must list the arrays of '" + u.part + "' in their order from its first, each once");
    if (!u.split.is_null()) bad("an update in the piece form is never split");
    u.params = scene_part_keys(m, part);
    if (part == kImage && !scene_piece_fits(part, name, u.params))
        bad("this form of an image has no piece '" + name + "'");
    for (int k = 0; table[k]; ++k)
        if (f.section(table[k])) bad(std::string("an update in the piece form carries no array section, got '") +
                                     table[k] + "'");
}

bool is_transform_paths(const Json& v) {
    if (!v.is_array() || v.size() < 1 || v.size() > CALIPER_NET_MAX_TRANSFORMS) return false;
    for (const Json& p : v.items())
        if (!p.is_string() || p.as_string().empty() || !entity_path_ok(p.as_string())) return false;
    return true;
}

bool is_clear_part(const Json& v) {
    return one_of(v, {CALIPER_NET_PART_POINTS, CALIPER_NET_PART_LINES, CALIPER_NET_PART_ARROWS, CALIPER_NET_PART_BOXES,
                      CALIPER_NET_PART_MESH, CALIPER_NET_PART_IMAGE, CALIPER_NET_PART_CAMERA,
                      CALIPER_NET_PART_TRANSFORM});
}

bool is_class_entry(const Json& e) {
    if (!e.is_object()) return false;
    const Json* id = e.find("id");
    const Json* label = e.find("label");
    const Json* color = e.find("color");
    int64_t i = 0;
    return id && id->is_int() && id->get_i64(&i) && i >= 0 && i <= (int64_t)CALIPER_NET_MAX_CLASS_ID && label &&
           label->is_string() && label->as_string().size() <= CALIPER_NET_NAME_MAX && color && is_rgba(*color);
}

template <typename M>
void scene_write_flags(Json& m, const M& x) {
    if (x.is_static) m.set("static", true);
    if (x.state) m.set("state", true);
    if (x.cursor) m.set("cursor", *x.cursor);
}

template <typename M>
void scene_write_common(Json& m, const M& x) {
    m.set("sub", x.sub);
    m.set("view", x.view);
    m.set("seq", x.seq);
    m.set("stamp_ns", x.stamp_ns);
}

}  // namespace

namespace detail {

// The section `name` of a part, or nullptr; `shape` holds the rows it must have (-1: any) and its rank.
const Section* part_section(const Frame& f, const char* name, int64_t dtype, std::vector<int64_t> shape) {
    const Section* s = f.section(name);
    if (!s) return nullptr;
    bool ok = s->encoding == CALIPER_NET_ENCODING_RAW && s->dtype == dtype && s->shape.size() == shape.size();
    for (size_t i = 0; ok && i < shape.size(); ++i)
        if (shape[i] >= 0 && shape[i] != s->shape[i]) ok = false;
    if (!ok)
        bad(std::string("'") + name + "' has the wrong dtype or shape (dtype " + std::to_string(s->dtype) + ", rank " +
            std::to_string(s->shape.size()) + ")");
    return s;
}

SceneUpdate decode_scene_update(const Frame& f) {
    const Json& m = f.meta;
    SceneUpdate u;
    scene_common(m, u);
    u.path = req(m, "path", is_path, "an entity path").as_string();
    u.part = req(m, "part", is_str, "a string").as_string();
    const int part = scene_part_index(u.part);
    if (part < 0) bad("unknown part '" + u.part + "'");
    scene_flags(m, u);
    if (const Json* v = opt(m, "axes", is_list, "a list")) u.axes = *v;
    if (const Json* v = opt(m, "items", is_name, "an item space name")) u.items = v->as_string();
    const Json* sp = opt(m, "split", is_obj, "an object");
    if (sp) u.split = *sp;
    if (m.find("program")) {
        scene_piece_form(f, m, part, u);
    } else if (sp) {
        scene_piece(f, m, part, u);
    } else {
        u.params = scene_part_keys(m, part);
        switch (part) {
            case kPoints: decode_points(f, u.params, &u.arrays); break;
            case kLines: decode_lines(f, u.params, &u.arrays); break;
            case kArrows: decode_arrows(f, &u.arrays); break;
            case kBoxes: decode_boxes(f, u.params, &u.arrays); break;
            case kMesh: decode_mesh(f, &u.arrays); break;
            case kImage: decode_image(f, u.params, &u.arrays); break;
            default: decode_camera(u.params); break;
        }
    }
    scene_stamp(f, u.is_static, -1, &u.stamp);
    return u;
}

SceneTransforms decode_scene_transforms(const Frame& f) {
    const Json& m = f.meta;
    SceneTransforms t;
    scene_common(m, t);
    scene_flags(m, t);
    t.paths = strs(req(m, "paths", is_transform_paths, "a list of 1 to 1024 entity paths other than the root"));
    if (std::set<std::string>(t.paths.begin(), t.paths.end()).size() != t.paths.size())
        bad("'paths' names an entity twice");
    if (!f.section("matrices")) bad("a scene_transforms needs a 'matrices' section");
    t.matrices = *part_section(f, "matrices", CALIPER_NET_DTYPE_F32, {(int64_t)t.paths.size(), 16});
    scene_stamp(f, t.is_static, (int64_t)t.paths.size(), &t.stamp, &t.stamps);
    return t;
}

SceneClear decode_scene_clear(const Frame& f) {
    const Json& m = f.meta;
    SceneClear c;
    scene_common(m, c);
    c.path = req(m, "path", is_path, "an entity path").as_string();
    if (const Json* v = opt(m, "recursive", is_bool, "a boolean")) c.recursive = v->as_bool();
    if (const Json* v = opt(m, "parts", [](const Json& x) {
            if (!x.is_array() || x.size() == 0) return false;
            for (const Json& e : x.items())
                if (!is_clear_part(e)) return false;
            return true;
        }, "a list of parts"))
        c.parts = strs(*v);
    if (const Json* v = opt(m, "reset", is_bool, "a boolean")) c.reset = v->as_bool();
    scene_flags(m, c);
    if (c.reset && !c.path.empty()) bad("a reset clears the root: its 'path' is \"\"");
    scene_stamp(f, c.is_static, -1, &c.stamp);
    return c;
}

SceneClasses decode_scene_classes(const Frame& f) {
    const Json& m = f.meta;
    SceneClasses k;
    scene_common(m, k);
    scene_flags(m, k);
    k.classes = req(m, "classes", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_MAX_CLASSES) return false;
        for (const Json& e : v.items())
            if (!is_class_entry(e)) return false;
        return true;
    }, "a list of at most 4096 {id, label, color} entries");
    int64_t prev = -1;
    for (const Json& e : k.classes.items()) {
        int64_t i = 0;
        e.find("id")->get_i64(&i);
        if (i <= prev) bad("'classes' must be in ascending id order without repeats");
        prev = i;
    }
    scene_stamp(f, k.is_static, -1, &k.stamp);
    return k;
}

// The metadata and sections that SceneUpdate.to_meta() and to_sections() write.
void scene_update_to_meta(const SceneUpdate& u, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, u);
    m.set("path", u.path);
    m.set("part", u.part);
    if (u.program) {
        m.set("program", *u.program);
        m.set("piece", u.piece);
        m.set("resources", u.resources);
        Json names = Json::array();
        for (const std::string& n : u.names) names.push(Json(n));
        m.set("arrays", names);
    }
    const bool camera = u.part == CALIPER_NET_PART_CAMERA;
    for (const auto& kv : u.params.members()) {
        const std::string& k = kv.first;
        const Json& v = kv.second;
        if (v.is_null()) continue;
        if (k == "stats") {
            Json a = Json::array();
            for (const Json& x : v.items())
                a.push(x.is_number() && std::isfinite(x.as_double()) ? Json(x.as_double()) : Json());
            m.set("stats", a);
        } else if (camera && (k == "width" || k == "height")) {
            m.set(k, v);
        } else if (v.is_number() && (k == "vmin" || k == "vmax" || k == "radius" || k == "width" || k == "opacity" ||
                                     k == "scale" || k == "fx" || k == "fy" || k == "cx" || k == "cy" ||
                                     k == "plane_distance")) {
            m.set(k, v.as_double());
        } else {
            m.set(k, v);
        }
    }
    scene_write_flags(m, u);
    if (!u.axes.is_null()) m.set("axes", u.axes);
    if (u.items) m.set("items", *u.items);
    if (!u.split.is_null()) m.set("split", u.split);
    for (const char* const* k = scene_part_sections(scene_part_index(u.part)); *k; ++k)
        if (const Section* s = u.array(*k)) sections.push_back(*s);
    if (u.stamp) sections.push_back(named(*u.stamp, "stamp"));
}

void scene_transforms_to_meta(const SceneTransforms& t, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, t);
    m.set("paths", str_list(t.paths));
    scene_write_flags(m, t);
    sections.push_back(named(t.matrices, "matrices"));
    if (t.stamps) sections.push_back(named(*t.stamps, "stamps"));
    else if (t.stamp) sections.push_back(named(*t.stamp, "stamp"));
}

void scene_clear_to_meta(const SceneClear& c, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, c);
    m.set("path", c.path);
    if (c.recursive) m.set("recursive", true);
    if (c.reset) m.set("reset", true);
    scene_write_flags(m, c);
    if (c.parts) m.set("parts", str_list(*c.parts));
    if (c.stamp) sections.push_back(named(*c.stamp, "stamp"));
}

void scene_classes_to_meta(const SceneClasses& k, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, k);
    m.set("classes", k.classes);
    scene_write_flags(m, k);
    if (k.stamp) sections.push_back(named(*k.stamp, "stamp"));
}

}  // namespace detail

// ---- public scene helpers and SplitUpdates ---------------------------------------------------

const Section* SceneUpdate::array(const std::string& name) const {
    for (const Section& s : arrays)
        if (s.name == name) return &s;
    return nullptr;
}

int scene_part_index(const std::string& p) {
    for (int i = 0; i < kPartCount; ++i)
        if (p == kParts[i]) return i;
    return -1;
}

const char* const* scene_part_sections(int part) {
    static const char* const none[1] = {nullptr};
    return part >= 0 && part < kPartCount ? kPartSections[part] : none;
}

bool entity_path_ok(const std::string& v) {
    if (v.empty()) return true;
    if (v.size() > CALIPER_NET_PATH_MAX) return false;
    size_t segs = 1, len = 0, start = 0;
    for (size_t i = 0; i <= v.size(); ++i) {
        if (i == v.size() || v[i] == '/') {
            const std::string seg = v.substr(start, i - start);
            if (len < 1 || len > 63 || seg == "." || seg == "..") return false;
            if (i < v.size()) ++segs;
            start = i + 1;
            len = 0;
            continue;
        }
        const unsigned char c = (unsigned char)v[i];
        if (!(std::isalnum(c) || c == '_' || c == '.' || c == '-') || c >= 0x80) return false;
        ++len;
    }
    return segs <= CALIPER_NET_PATH_SEGMENTS_MAX;
}

bool scene_view_info(const Json& entry, SceneView* out, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    const Json* sc = entry.find("scene");
    const Json empty = Json::object();
    if (!sc || sc->is_null()) sc = &empty;
    if (!sc->is_object()) return fail("'scene' must be an object, got " + show(*sc));
    SceneView v;
    const Json* space = sc->find("space");   // a null key means the same as an absent one
    if (!space || space->is_null()) v.space = CALIPER_NET_SPACE_3D;
    else if (space->is_string() &&
             (space->as_string() == CALIPER_NET_SPACE_3D || space->as_string() == CALIPER_NET_SPACE_2D))
        v.space = space->as_string();
    else return fail("scene space " + show(*space) + "; expected '3d' or '2d'");
    const bool is3d = v.space == CALIPER_NET_SPACE_3D;
    const Json* up = sc->find("up");
    if (!up || up->is_null()) {
        v.up = is3d ? "+y" : "-y";
    } else {
        bool ok = false;
        if (up->is_string()) {
            const std::string& u = up->as_string();
            ok = u == "+y" || u == "-y" || (is3d && (u == "+x" || u == "-x" || u == "+z" || u == "-z"));
        }
        if (!ok) return fail("scene up " + show(*up) + " is not one of the axes of a " + v.space + " scene");
        v.up = up->as_string();
    }
    const Json* units = sc->find("units");
    if (!units || units->is_null()) v.units = "m";
    else if (units->is_string() && units->as_string().size() <= CALIPER_NET_UNITS_MAX) v.units = units->as_string();
    else return fail("scene units must be a string of at most 15 bytes, got " + show(*units));
    v.view_camera = view_camera_of(*sc, is3d);
    if (out) *out = std::move(v);
    return true;
}

Json scene_data_view() {
    Json inc = Json::array();
    inc.push("**");
    Json ent = Json::object();
    ent.set("include", inc);
    ent.set("exclude", Json::array());
    Json b = Json::object();
    b.set("max_points", (uint64_t)CALIPER_NET_DEFAULT_MAX_POINTS);
    b.set("max_items", (uint64_t)CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS);
    b.set("max_vertices", (uint64_t)CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES);
    b.set("max_extent", (uint64_t)CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT);
    Json dv = Json::object();
    dv.set("entities", ent);
    dv.set("budget", b);
    return dv;
}

bool resolve_scene_data_view(const Json& view, const Json& request, Json* out, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    const Json dv = scene_data_view();
    auto is_default = [&](const Json& v) { return v.is_null() || (v.is_object() && v.size() == 0) || v == dv; };
    if (!request.is_null() && !request.is_object()) return fail("a data view is a JSON object");
    const Json* def = view.find("default");
    if (def && !is_default(*def)) return fail("a scene's own default data view is not built yet");
    if (!is_default(request))
        return fail("only the default data view of a scene is built yet: send {} or none; entity filters and "
                    "budgets come in a later increment");
    if (out) *out = dv;
    return true;
}

SplitUpdates::Result SplitUpdates::add(const SceneUpdate& piece, WholeUpdate* out, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return Result::Refused;
    };
    const Json& sp = piece.split;
    const uint64_t uid = u64(*sp.find("update"));
    const int64_t num = i64(*sp.find("piece"));
    auto stamp_bytes = [](const SceneUpdate& u) {
        return u.stamp ? std::vector<uint8_t>(u.stamp->data, u.stamp->data + u.stamp->size) : std::vector<uint8_t>();
    };
    if (num == 0) {
        for (auto it = open_.begin(); it != open_.end();) {
            if (it->update == uid || (it->p0.path == piece.path && it->p0.part == piece.part)) {
                it = open_.erase(it);
                ++discarded_;
            } else {
                ++it;
            }
        }
        Assembly a;
        a.update = uid;
        a.p0 = piece;
        a.p0.arrays.clear();
        a.stamp_bytes = stamp_bytes(piece);
        a.p0.stamp.reset();
        a.pieces = i64(*sp.find("pieces"));
        a.rows = *sp.find("rows");
        open_.push_back(std::move(a));
        while (open_.size() > CALIPER_NET_MAX_INCOMPLETE_UPDATES) {
            open_.erase(open_.begin());
            ++discarded_;
        }
    }
    auto it = std::find_if(open_.begin(), open_.end(), [&](const Assembly& a) { return a.update == uid; });
    if (it == open_.end()) {
        ++dropped_;
        return Result::Dropped;
    }
    Assembly& a = *it;
    auto refuse = [&](const std::string& m) {
        open_.erase(std::find_if(open_.begin(), open_.end(), [&](const Assembly& x) { return x.update == uid; }));
        ++discarded_;
        return fail(m);
    };
    // The checks of _Assembly.add, in its order.
    if (piece.path != a.p0.path || piece.part != a.p0.part || piece.seq != a.p0.seq ||
        i64(*sp.find("pieces")) != a.pieces || stamp_bytes(piece) != a.stamp_bytes)
        return refuse("a piece's path, part, seq, pieces or stamp differ from its update's piece 0");
    if (std::find(a.seen.begin(), a.seen.end(), num) != a.seen.end())
        return refuse("piece " + std::to_string(num) + " arrived twice");
    a.seen.push_back(num);
    const Json& first = *sp.find("first");
    for (const Section& s : piece.arrays) {
        const Json* rows_j = a.rows.find(s.name);
        if (!rows_j) return refuse("a piece holds '" + s.name + "', which 'split.rows' does not name");
        const uint32_t item = s.dtype >= 0 && s.dtype <= 0x7fffffff ? caliper_net_dtype_size((int32_t)s.dtype) : 0u;
        if (item == 0) return refuse("a piece of '" + s.name + "' has an unknown dtype");
        if (s.shape.empty() || s.shape[0] < 0) return refuse("a piece of '" + s.name + "' has no rows");
        const std::vector<int64_t> trailing(s.shape.begin() + 1, s.shape.end());
        // The bytes of one row, held at most at one past the update limit so that no product wraps: a
        // larger row refuses any count of rows but 0 below.
        uint64_t rowbytes = item;
        for (int64_t d : trailing) {
            if (d < 0) return refuse("a piece of '" + s.name + "' has a negative dimension");
            rowbytes = d && rowbytes > (CALIPER_NET_MAX_UPDATE_BYTES + 1) / (uint64_t)d
                           ? CALIPER_NET_MAX_UPDATE_BYTES + 1
                           : rowbytes * (uint64_t)d;
        }
        const uint64_t rows = u64(*rows_j);
        if (rows > (uint64_t)INT64_MAX) return refuse("'split.rows' of '" + s.name + "' is too large");
        auto ai = std::find_if(a.arrays.begin(), a.arrays.end(), [&](const auto& x) { return x.first == s.name; });
        if (ai == a.arrays.end()) {
            const uint64_t total = rows * rowbytes;
            if ((rowbytes && rows > CALIPER_NET_MAX_UPDATE_BYTES / rowbytes) ||
                a.bytes + total > CALIPER_NET_MAX_UPDATE_BYTES)
                return refuse("a split update of more than 536870912 bytes");
            a.bytes += total;
            Array arr;
            arr.dtype = s.dtype;
            arr.trailing = trailing;
            arr.bytes.resize((size_t)total);
            a.arrays.emplace_back(s.name, std::move(arr));
            ai = a.arrays.end() - 1;
        } else if (ai->second.dtype != s.dtype || ai->second.trailing != trailing) {
            return refuse("the pieces of '" + s.name + "' differ in dtype or shape");
        }
        // Unsigned, with no sum that can wrap: f0 + n <= rows holds after this check, so the overlap test
        // and the copy below stay inside the array.
        const Json* fj = first.find(s.name);
        if (!fj || !is_uint(*fj)) return refuse("'split.first' does not name '" + s.name + "'");
        const uint64_t f0 = u64(*fj), n = (uint64_t)s.shape[0];
        if (f0 > rows || n > rows - f0) return refuse("rows of '" + s.name + "' pass its rows");
        if ((uint64_t)s.size != n * rowbytes)
            return refuse("a piece of '" + s.name + "' holds bytes that disagree with its shape");
        for (const auto& r : ai->second.ranges)
            if (f0 < r.first + r.second && r.first < f0 + n)
                return refuse("two pieces hold the same rows of '" + s.name + "'");
        ai->second.ranges.emplace_back(f0, n);
        if (s.size) std::memcpy(ai->second.bytes.data() + (size_t)(f0 * rowbytes), s.data, s.size);
    }
    if ((int64_t)a.seen.size() < a.pieces) return Result::Pending;
    for (const auto& kv : a.rows.members()) {
        auto ai = std::find_if(a.arrays.begin(), a.arrays.end(), [&](const auto& x) { return x.first == kv.first; });
        uint64_t got = 0;
        if (ai != a.arrays.end())
            for (const auto& r : ai->second.ranges) got += r.second;
        if (got != u64(kv.second)) return refuse("the pieces ended before every row of '" + kv.first + "' arrived");
    }
    // The whole update: piece 0's keys with the whole arrays and the stamp, checked as an unsplit update.
    struct Keep {
        std::vector<std::pair<std::string, Array>> arrays;
        std::vector<uint8_t> stamp;
    };
    auto keep = std::make_shared<Keep>();
    keep->arrays = std::move(a.arrays);
    keep->stamp = std::move(a.stamp_bytes);
    SceneUpdate p0 = std::move(a.p0);
    const Json rows = a.rows;
    open_.erase(std::find_if(open_.begin(), open_.end(), [&](const Assembly& x) { return x.update == uid; }));
    Frame f;
    f.type = CALIPER_NET_MSG_SCENE_UPDATE;
    p0.split = Json();
    {
        Json meta = Json::object();
        std::vector<Section> unused;
        scene_update_to_meta(p0, meta, unused);
        f.meta = std::move(meta);
    }
    const int part = scene_part_index(p0.part);
    for (const char* const* k = scene_part_sections(part); *k; ++k) {
        for (const auto& kv : keep->arrays) {
            if (kv.first != *k) continue;
            std::vector<int64_t> shape{(int64_t)u64(*rows.find(kv.first))};
            shape.insert(shape.end(), kv.second.trailing.begin(), kv.second.trailing.end());
            f.sections.push_back(Section::make(kv.first, kv.second.dtype, shape, kv.second.bytes.data(),
                                               kv.second.bytes.size()));
        }
    }
    if (!keep->stamp.empty())
        f.sections.push_back(Section::make("stamp", CALIPER_NET_DTYPE_I64, {(int64_t)keep->stamp.size() / 8},
                                           keep->stamp.data(), keep->stamp.size()));
    try {
        SceneUpdate whole = decode_scene_update(f);
        if (out) {
            out->update = std::move(whole);
            out->keep = keep;
        }
        return Result::Whole;
    } catch (const MsgError& e) {
        ++discarded_;
        return fail(e.message);
    }
}

bool scene_program_id(const std::string& id) {
    static const std::string kPrefix = CALIPER_NET_SCENE_PROGRAM_PREFIX;
    if (id.size() <= kPrefix.size() || id.size() > kPrefix.size() + 9 || id.compare(0, kPrefix.size(), kPrefix) != 0)
        return false;
    for (size_t i = kPrefix.size(); i < id.size(); ++i)
        if (id[i] < '0' || id[i] > '9') return false;
    return true;
}

}  // namespace wire
}  // namespace caliper_net
