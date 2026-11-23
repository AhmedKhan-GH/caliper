// Read-only scene array decoder for saved streams and received piece metadata.
#include "live/live_scene.h"
#include "gfx/colormap_tables.h"
#if defined(CALIPER_HAVE_DRAW) && defined(CALIPER_HAVE_SCENE_PIECES)
#define CALIPER_SCENE_PIECES 1
#include "draw/scene_pieces.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
namespace caliper_live_views {
using namespace caliper_scene;
using caliper_gfx::VertexFormat;
namespace wire = caliper_net::wire;
namespace {
double num(const caliper_net::Json& params, const char* key, double fallback) {
    const caliper_net::Json* v = params.find(key);
    return v && v->is_number() ? v->as_double() : fallback;
}

// The integer keys, read as the integers they are (the decoder takes any integer): a colormap
// id outside 0 to INT32_MAX is no colormap, and flags keep every bit of a uint64.
uint32_t colormap_id(const caliper_net::Json& params) {
    const caliper_net::Json* v = params.find("colormap");
    int64_t id = 0;
    return v && v->get_i64(&id) && id >= 0 && id <= INT32_MAX ? uint32_t(id) : 0u;
}
uint64_t flag_bits(const caliper_net::Json& params) {
    const caliper_net::Json* v = params.find("flags");
    uint64_t f = 0;
    return v && v->get_u64(&f) ? f : 0u;
}

// The `color` key, [r, g, b, a] from 0 to 255, packed r | g << 8 | b << 16 | a << 24.
bool flat_color(const caliper_net::Json& params, uint32_t* out) {
    const caliper_net::Json* c = params.find("color");
    if (!c || !c->is_array() || c->size() != 4) return false;
    uint32_t rgba = 0;
    for (size_t k = 0; k < 4; ++k) {
        const double v = c->at(k).as_double();
        rgba |= uint32_t(std::clamp(v, 0.0, 255.0)) << (8 * k);
    }
    *out = rgba;
    return true;
}


}
bool SceneView::decode_scene_update(const wire::SceneUpdate& u, const std::vector<wire::Section>* received,
                             std::string* why) {
    ++counts_.messages;
    const caliper_net::Json& params = u.params;
    const int part = wire::scene_part_index(u.part);
    const std::string where = (u.path.empty() ? std::string("the root") : u.path) + " " + u.part;
    auto refuse = [&](const std::string& text) {
        const std::string t = where + ": " + text;
        note(t);
        if (why) *why = t;
        return false;
    };
    if (part < 0) return refuse("an unknown part");
    std::string err;
    const EntityId e = tree_.ensure(u.path, &err);
    if (!e) return refuse(err);

    // A camera is no batch: it goes to the store's camera layers, which a clear of the part
    // "camera" empties.
    if (u.part == CALIPER_NET_PART_CAMERA) {
        Pinhole c;
        c.fx = num(params, "fx", 0.0);
        c.fy = num(params, "fy", 0.0);
        c.cx = num(params, "cx", 0.0);
        c.cy = num(params, "cy", 0.0);
        c.width = uint32_t(std::clamp(num(params, "width", 0.0), 0.0, double(kMaxImageSide)));
        c.height = uint32_t(std::clamp(num(params, "height", 0.0), 0.0, double(kMaxImageSide)));
        const caliper_net::Json* conv = params.find("convention");
        c.convention = conv && conv->is_string() && conv->as_string() == CALIPER_NET_CONVENTION_RUB ? Convention::Rub
                                                                                                     : Convention::Rdf;
        c.plane_distance = num(params, "plane_distance", 0.0);
        if (!tree_.set_camera(e, c, u.is_static, &err)) return refuse(err);
        changed();
        return true;
    }

    const Primitive prim = Primitive(part + 1);   // points, lines, arrows, boxes, mesh, image
    const char* first = u.part == CALIPER_NET_PART_ARROWS  ? "origins"
                        : u.part == CALIPER_NET_PART_BOXES ? "centers"
                        : u.part == CALIPER_NET_PART_IMAGE ? "pixels"
                                                           : "positions";
    const wire::Section* lead = u.array(first);
    if (!lead || lead->shape.empty()) return refuse(std::string("no ") + first);
    const bool image = prim == Primitive::Image;
    // Rows and their dimension: 2 in a 2-D scene and under a camera, 3 elsewhere.
    const uint64_t n = image ? 0 : uint64_t(lead->shape[0]);
    const uint32_t dims = image ? 0 : (lead->shape.size() == 2 ? uint32_t(lead->shape[1]) : 0);
    if (!image) {
        const uint32_t want = (tree_.space() == Space::k2D || in_pixel_space(e)) ? 2u : 3u;
        if (dims != want)
            return refuse(std::to_string(dims) + "-D " + first + " where " + std::to_string(want) + "-D data is needed");
    }

    Slot& slot = slots_[Key{u.path, int(prim), u.is_static}];
    const bool empty = image ? lead->count() == 0 : n == 0;
    if (empty) {   // an empty batch removes the part
        release(slot);
#if defined(CALIPER_SCENE_PIECES)
        if (shapes_) shapes_->drop(u.path, prim, u.is_static);
#endif
        rebuild(u.path);
        return true;
    }

    Batch b;
    b.primitive = prim;
    b.seq = ++seq_;
    uint32_t rgba = 0;
    const bool has_flat = flat_color(params, &rgba);
    if (has_flat) b.flat_rgba = rgba;
    b.colormap = colormap_id(params);
    if (!caliper_gfx::cmap::builtin_lut(int32_t(b.colormap))) b.colormap = 0;

    // The colormap's range: the fixed ends, or the producer's stats for the AUTO ones, as in
    // SNAPSHOT; a flat AUTO range widens by 0.5 on each side, as a heatmap's does.
    std::vector<double> stats;
    if (const caliper_net::Json* st = params.find("stats"); st && st->is_array())
        for (const caliper_net::Json& x : st->items()) stats.push_back(x.as_double());
    const uint64_t flags = flag_bits(params);
    double vmin = num(params, "vmin", 0.0), vmax = num(params, "vmax", 1.0);
    const bool amin = (flags & CALIPER_NET_SNAP_AUTO_VMIN) != 0, amax = (flags & CALIPER_NET_SNAP_AUTO_VMAX) != 0;
    if (amin) vmin = stats.size() > 0 ? stats[0] : NAN;
    if (amax) vmax = stats.size() > 1 ? stats[1] : NAN;
    if (amin || amax) {
        if (!std::isfinite(vmin) || !std::isfinite(vmax)) {
            vmin = 0.0;
            vmax = 1.0;
        } else if (vmin == vmax) {
            vmin -= 0.5;
            vmax += 0.5;
        }
    }
    b.vmin = float(vmin);
    b.vmax = float(vmax);
    if (const caliper_net::Json* w = params.find("width"); w && w->is_number() && w->as_double() > 0)
        b.width_px = float(w->as_double());
    if (const caliper_net::Json* f = params.find("fill"); f && f->is_bool()) b.fill = f->as_bool();
    if (const caliper_net::Json* l = params.find("labels"); l && l->is_array())
        for (const caliper_net::Json& s : l->items()) b.labels.push_back(s.as_string());

    // Bounds: the producer's for points, lines and meshes, else a scan of the rows as they arrived;
    // arrows add their tips and boxes their half sizes (the store's rule for its CPU views).
    auto rows_of = [&](const char* name) -> const float* {
        const wire::Section* s = u.array(name);
        return s ? reinterpret_cast<const float*>(s->data) : nullptr;
    };
    auto add_row = [&](const float* p, uint64_t i, double k, const float* d, const float* h) {
        double v[3] = {p[i * dims], p[i * dims + 1], dims == 3 ? p[i * dims + 2] : 0.0};
        if (d)
            for (uint32_t c = 0; c < dims; ++c) v[c] += k * d[i * dims + c];
        if (h) {
            double r[3] = {std::fabs(h[i * dims]), std::fabs(h[i * dims + 1]), dims == 3 ? std::fabs(h[i * dims + 2]) : 0.0};
            if (u.array("rotations")) {
                const double l = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                r[0] = r[1] = l;
                r[2] = dims == 3 ? l : 0.0;
            }
            for (int c = 0; c < 3; ++c) v[c] += k * r[c];
        }
        if (std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2])) b.bounds.add({v[0], v[1], v[2]});
    };
    bool have = !image && prim != Primitive::Arrows && prim != Primitive::Boxes && stats.size() >= 8;
    for (size_t k = 2; have && k < 8; ++k) have = std::isfinite(stats[k]);
    if (have) {
        b.bounds.add({stats[2], stats[4], dims == 3 ? stats[6] : 0.0});
        b.bounds.add({stats[3], stats[5], dims == 3 ? stats[7] : 0.0});
    } else if (!image) {
        const float* p = rows_of(first);
        const float* vec = prim == Primitive::Arrows ? rows_of("vectors") : nullptr;
        const float* half = prim == Primitive::Boxes ? rows_of("half_sizes") : nullptr;
        for (uint64_t i = 0; i < n; ++i) {
            if (half) {
                add_row(p, i, -1.0, nullptr, half);
                add_row(p, i, 1.0, nullptr, half);
            } else {
                add_row(p, i, 0.0, nullptr, nullptr);
                if (vec) add_row(p, i, 1.0, vec, nullptr);
            }
        }
    }

    std::string problem;
    std::map<int, std::vector<uint8_t>> cpu;   // points and boxes: every array, for selections
    auto add = [&](Role role, const void* bytes, uint64_t rows, uint32_t stride, VertexFormat fmt) {
        const uint64_t size = rows * (stride ? stride : vertex_format_bytes(fmt));
        if (size && !bytes) { problem = "an array has no bytes"; return false; }
        auto& data = cpu[int(role)];
        if (size) data.assign(static_cast<const uint8_t*>(bytes), static_cast<const uint8_t*>(bytes) + size);
        counts_.bytes_last += size;
        counts_.bytes_total += size;
        ++counts_.copies_total;
        BatchArray a;
        a.role = role;
        a.cpu = data.data();
        a.count = rows;
        a.stride = stride;
        a.format = fmt;
        b.arrays.push_back(a);
        return true;
    };
    const VertexFormat rowf = dims == 2 ? VertexFormat::F32x2 : VertexFormat::F32x3;
    auto add_rows = [&](Role role, const char* name) {
        const wire::Section* s = u.array(name);
        return !s || add(role, s->data, uint64_t(s->shape[0]), dims * 4, rowf);
    };
    // A refusal after the first upload drops the part: its buffers no longer hold what the store's
    // batch describes.
    auto failed = [&]() {
        note(where + ": " + problem);
        if (why) *why = where + ": " + problem;
        release(slot);
        rebuild(u.path);
        return false;
    };
    // CPU copies for the labels: the rows the anchors are projected from and the class ids that
    // name the items without a label of their own.
    std::vector<uint8_t> cpu_rows, cpu_ids;
    auto keep_cpu = [&](const char* rows_name) {
        const bool ids = u.array("class_ids") != nullptr;
        if (b.labels.empty() && !ids) return;
        const wire::Section* s = u.array(rows_name);
        cpu_rows.assign(s->data, s->data + s->size);
        if (const wire::Section* c = u.array("class_ids")) cpu_ids.assign(c->data, c->data + c->size);
    };

    const wire::Section* colors = u.array("colors");
    const wire::Section* values = u.array("values");
    bool translucent = has_flat && (rgba >> 24) < 255;
    auto add_colours = [&]() {
        if (colors) {
            const uint32_t* rows = reinterpret_cast<const uint32_t*>(colors->data);
            const uint64_t m = uint64_t(colors->shape[0]);
            for (uint64_t i = 0; i < m && !translucent; ++i) translucent = (rows[i] >> 24) < 255;
            return add(Role::Colors, colors->data, m, 4, VertexFormat::Unorm8x4);
        }
        if (values) return add(Role::Values, values->data, uint64_t(values->shape[0]), 4, VertexFormat::F32);
        return true;
    };
    auto add_class_ids = [&]() {
        const wire::Section* c = u.array("class_ids");
        return !c || add(Role::ClassIds, c->data, uint64_t(c->shape[0]), 4, VertexFormat::S32);
    };

    switch (prim) {
        case Primitive::Points: {
            if (!add_rows(Role::Positions, "positions") || !add_colours() || !add_class_ids()) return failed();
            if (const wire::Section* radii = u.array("radii")) {
                if (!add(Role::Radii, radii->data, n, 4, VertexFormat::F32)) return failed();
            } else if (const caliper_net::Json* r = params.find("radius"); r && r->is_number()) {
                const float one = float(r->as_double());
                if (!add(Role::Radii, &one, 1, 0, VertexFormat::F32)) return failed();
            }
            keep_cpu("positions");
            break;
        }
        case Primitive::Lines: {
            if (!add_rows(Role::Positions, "positions")) return failed();
            const caliper_net::Json* mode = params.find("mode");
            const std::string md = mode && mode->is_string() ? mode->as_string() : std::string(CALIPER_NET_LINES_SEGMENTS);
            const caliper_net::Json* cp = params.find("color_per");
            const std::string per = cp && cp->is_string() ? cp->as_string() : std::string(CALIPER_NET_COLOR_PER_VERTEX);
            if (md == CALIPER_NET_LINES_STRIPS) {
                b.line_mode = LineMode::Strips;
                // The break mask from the strip lengths, and per-strip colours or values expanded to
                // their vertices: the scene core reads both per vertex.
                const wire::Section* sl = u.array("strip_lengths");
                if (!sl) {
                    problem = "strips without strip_lengths";
                    return failed();
                }
                const int32_t* len = reinterpret_cast<const int32_t*>(sl->data);
                const uint64_t strips_n = uint64_t(sl->shape[0]);
                const bool per_strip = per == CALIPER_NET_COLOR_PER_STRIP;
                scratch_.assign(n * 4, 0);
                uint32_t* mask = reinterpret_cast<uint32_t*>(scratch_.data());
                std::vector<uint32_t> strip_of;
                if (per_strip) strip_of.resize(n);
                uint64_t v = 0;
                for (uint64_t s = 0; s < strips_n && v < n; ++s) {
                    const uint64_t l = uint64_t(std::max(len[s], 0));
                    for (uint64_t k = 0; k < l && v < n; ++k, ++v) {
                        mask[v] = k + 1 < l ? 1u : 0u;
                        if (per_strip) strip_of[v] = uint32_t(s);
                    }
                }
                if (!add(Role::BreakMask, mask, n, 4, VertexFormat::U8)) return failed();
                const wire::Section* src = colors ? colors : values;
                if (src && per_strip) {
                    std::vector<uint32_t> expanded(n);
                    const uint32_t* rows = reinterpret_cast<const uint32_t*>(src->data);
                    for (uint64_t i = 0; i < n; ++i) expanded[i] = rows[strip_of[i]];
                    if (colors) {
                        for (uint64_t s = 0; s < strips_n; ++s) translucent = translucent || (rows[s] >> 24) < 255;
                        if (!add(Role::Colors, expanded.data(), n, 4, VertexFormat::Unorm8x4)) return failed();
                    } else if (!add(Role::Values, expanded.data(), n, 4, VertexFormat::F32)) {
                        return failed();
                    }
                    colors = values = nullptr;
                }
            } else if (md == CALIPER_NET_LINES_EDGES) {
                // Index-pair edges: the scene core's edge gather clamps each pair and gathers its two
                // ends; colours and values belong to the vertices or, with color_per "edge", to the edges.
                b.line_mode = LineMode::Edges;
                b.per_edge = per == CALIPER_NET_COLOR_PER_EDGE;
                const wire::Section* idx = u.array("indices");
                if (!idx) {
                    problem = "edges without indices";
                    return failed();
                }
                if (!add(Role::Indices, idx->data, uint64_t(idx->shape[0]), 8, VertexFormat::S32)) return failed();
            } else {
                b.line_mode = LineMode::Segments;
            }
            if (!add_colours()) return failed();
            break;
        }
        case Primitive::Arrows:
            if (!add_rows(Role::Origins, "origins") || !add_rows(Role::Vectors, "vectors") || !add_colours())
                return failed();
            break;
        case Primitive::Boxes: {
            if (!add_rows(Role::Centers, "centers") || !add_rows(Role::HalfSizes, "half_sizes")) return failed();
            if (const wire::Section* r = u.array("rotations"))
                if (!add(Role::Rotations, r->data, n, 16, VertexFormat::F32x4)) return failed();
            if (!add_colours() || !add_class_ids()) return failed();
            keep_cpu("centers");
            break;
        }
        case Primitive::Mesh: {
            if (!add_rows(Role::Positions, "positions")) return failed();
            if (const wire::Section* idx = u.array("indices"))
                if (!add(Role::Indices, idx->data, uint64_t(idx->shape[0]), 12, VertexFormat::S32)) return failed();
            if (const wire::Section* nrm = u.array("normals"))
                if (!add(Role::Normals, nrm->data, n, 12, VertexFormat::F32x3)) return failed();
            if (!add_colours()) return failed();
            const wire::Section* uvs = u.array("uvs");
            const wire::Section* tex = u.array("texture");
            if (uvs && tex) {
                if (!add(Role::Uvs, uvs->data, n, 8, VertexFormat::F32x2)) return failed();
                const uint32_t tw = uint32_t(tex->shape[1]), th = uint32_t(tex->shape[0]);
                b.image_width = tw;
                b.image_height = th;
                const uint32_t* px = reinterpret_cast<const uint32_t*>(tex->data);
                for (uint64_t i = 0, m = uint64_t(tw) * th; i < m && !translucent; ++i) translucent = (px[i] >> 24) < 255;
            }
            break;
        }
        case Primitive::Image: {
            const caliper_net::Json* fm = params.find("form");
            const std::string form = fm && fm->is_string() ? fm->as_string() : std::string();
            const uint32_t h = uint32_t(lead->shape[0]);
            const uint32_t w = lead->shape.size() >= 2 ? uint32_t(lead->shape[1]) : 0;
            const uint32_t side = gfx_.caps().limits.max_texture_dimension_2d;
            if (form == CALIPER_NET_FORM_ENCODED) return refuse("encoded images are not drawn yet");
            if (w == 0 || h == 0) return refuse("an image without pixels");
            if (w > side || h > side)
                return refuse(std::to_string(w) + " x " + std::to_string(h) + " pixels, past this device's " +
                              std::to_string(side) + " on a side");
            b.image_width = w;
            b.image_height = h;
            if (const caliper_net::Json* o = params.find("opacity"); o && o->is_number())
                b.opacity = float(std::clamp(o->as_double(), 0.0, 1.0));
            else
                b.opacity = form == CALIPER_NET_FORM_SEGMENTATION ? 0.5f : 1.0f;
            if (form == CALIPER_NET_FORM_COLOR) {
                const bool grey = lead->shape.size() == 2;
                b.image_form = grey ? ImageForm::Grey : ImageForm::Color;
                if (!grey) {
                    const uint32_t* px = reinterpret_cast<const uint32_t*>(lead->data);
                    for (uint64_t i = 0, m = uint64_t(w) * h; i < m && !translucent; ++i) translucent = (px[i] >> 24) < 255;
                }
            } else if (form == CALIPER_NET_FORM_DEPTH || form == CALIPER_NET_FORM_SEGMENTATION) {
                if (form == CALIPER_NET_FORM_DEPTH) {
                    b.image_form = ImageForm::Depth;
                    const auto* cloud = params.find("cloud");
                    b.cloud = cloud && cloud->is_bool() ? cloud->as_bool() : in_pixel_space(e);
                } else {
                    b.image_form = ImageForm::Segmentation;
                }
            } else {
                return refuse("an image of form '" + form + "'");
            }
            b.translucent = translucent;
            break;
        }
    }
    b.translucent = b.translucent || translucent;
    for (BatchArray& a : b.arrays) {   // the CPU views of the labels' rows and ids
        if ((a.role == Role::Positions || a.role == Role::Centers) && !cpu_rows.empty()) a.cpu = cpu_rows.data();
        if (a.role == Role::ClassIds && !cpu_ids.empty()) a.cpu = cpu_ids.data();
    }

    // Every upload above is in the slot's buffers; the store takes the batch now.
    if (!tree_.set_batch(e, b, u.is_static, &problem)) return failed();
    slot.batch = b;
    slot.items = u.items ? *u.items : std::string();
    slot.ax_start = 0, slot.ax_step = 1, slot.ax_count = 0, slot.ax_size = 0;
    if (u.axes.is_array() && u.axes.size() > 0 && u.axes.at(u.axes.size() - 1).is_object()) {
        const caliper_net::Json& ax = u.axes.at(u.axes.size() - 1);   // the item dim's
        auto get = [&](const char* k, int64_t d) {
            int64_t v = d;
            const caliper_net::Json* j = ax.find(k);
            return j && j->get_i64(&v) ? v : d;
        };
        // The producer's values: an axis with a negative start or count, a step below 1 or no size maps
        // nothing (row_id gives the batch index), and row_id checks its products against these.
        const int64_t start = get("start", 0), step = get("step", 1), count = get("count", 0), size = get("size", 0);
        if (start >= 0 && step >= 1 && count >= 0 && size >= 1) {
            slot.ax_start = start;
            slot.ax_step = step;
            slot.ax_count = count;
            slot.ax_size = size;
        }
    }
    slot.cpu_rows = std::move(cpu_rows);   // the batch's CPU views point into these buffers
    slot.cpu_ids = std::move(cpu_ids);
    slot.cpu = std::move(cpu);
    slot.present = true;
#if defined(CALIPER_SCENE_PIECES)
    // The part's arrays as the piece's resources, while the message's sections live.
    if (shapes_) shapes_->take(u.path, prim, u.is_static, u, slot.batch, received);
#endif
    changed();
    return true;
}

}
