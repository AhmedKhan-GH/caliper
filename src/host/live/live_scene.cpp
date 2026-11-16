// live_scene.cpp: the scene view of the shared view layer (live_scene.h).
#include "live/live_scene.h"

#include "gfx/colormap_keys.h"
#include "scene/backend_resources.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace caliper_live_views {

using namespace caliper_scene;
using caliper_gfx::Buffer;
using caliper_gfx::VertexFormat;
namespace wire = caliper_net::wire;

namespace {

constexpr size_t kMaxNotes = 8;
constexpr uint64_t kMinCapacity = 4096;
constexpr uint32_t kUsage = caliper_gfx::kVertex | caliper_gfx::kStorage | caliper_gfx::kCopyDst;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

uint64_t capacity_for(uint64_t bytes) {
    uint64_t c = kMinCapacity;
    while (c < bytes) c <<= 1;
    return c;
}

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

std::string bytes_text(uint64_t b) {
    char t[32];
    if (b < 1024) std::snprintf(t, sizeof t, "%llu B", (unsigned long long)b);
    else if (b < 1024ull * 1024) std::snprintf(t, sizeof t, "%.1f KB", double(b) / 1024.0);
    else std::snprintf(t, sizeof t, "%.1f MB", double(b) / (1024.0 * 1024.0));
    return t;
}

std::string count_text(uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(size_t(i), ",");
    return s;
}

}  // namespace

SceneView::SceneView(caliper_gfx::Backend& backend, const caliper_gfx::ProgramDesc* programs)
    : gfx_(backend), programs_(programs), colormaps_(std::make_shared<caliper_gfx::cmap::KeyBlockCache>()) {}

SceneView::~SceneView() {
    for (auto& kv : slots_) release(kv.second);
    slots_.clear();
    for (const Buffer b : hl_buffers_) gfx_.release(b);
    hl_buffers_.clear();
    if (dim_texture_) gfx_.release(dim_texture_);
    pick_.reset();
    res_.reset();
    if (color_) gfx_.release(color_);
    if (depth_) gfx_.release(depth_);
}

bool SceneView::init(std::string* why) {
    res_ = std::make_unique<BackendResources>(gfx_, programs_, caliper_gfx::cmap::colormap_source(colormaps_));
    pick_ = std::make_unique<PickPass>(gfx_);
    return res_->init(why);
}

void SceneView::describe(const wire::SceneView& info) {
    const Space space = info.space == CALIPER_NET_SPACE_2D ? Space::k2D : Space::k3D;
    if (space != tree_.space()) {
        reset_scene();
        tree_ = SceneTree(space);
        camera_ = space == Space::k2D ? ViewCamera(PanZoom{}) : ViewCamera(Orbit{});
        fitted_ = moved_ = false;
    }
    Axis up = tree_.up();
    if (parse_axis(info.up, &up) && up != tree_.up()) {
        std::string err;
        if (tree_.set_up(up, &err)) {
            if (Orbit* o = std::get_if<Orbit>(&camera_)) {
                if (!moved_) o->up = up;   // the scene's up is where an unmoved view starts
            } else if (PanZoom* p = std::get_if<PanZoom>(&camera_)) {
                p->y_down = up == Axis::NegY;
            }
        } else {
            note("up '" + info.up + "': " + err);
        }
    }
    if (Orbit* o = std::get_if<Orbit>(&camera_); o && !moved_) o->up = tree_.up();
    if (PanZoom* p = std::get_if<PanZoom>(&camera_)) p->y_down = tree_.up() == Axis::NegY;
    tree_.set_units(info.units);
    changed();
}

void SceneView::note(const std::string& text) {
    if (std::find(notes_.begin(), notes_.end(), text) != notes_.end()) return;
    if (notes_.size() >= kMaxNotes) notes_.erase(notes_.begin());
    notes_.push_back(text);
}

void SceneView::begin_round() { counts_.bytes_last = 0; }

// ---- buffers ---------------------------------------------------------------------------------

bool SceneView::upload(Slot& s, Role role, const void* bytes, uint64_t size, Buffer* out, std::string* why) {
    Upload& u = s.buffers[int(role)];
    if (!u.buffer || u.capacity < size) {
        if (u.buffer) gfx_.release(u.buffer);   // the layer keeps it until its last frame completed
        caliper_gfx::BufferDesc d;
        d.size = capacity_for(size);
        d.usage = kUsage;
        d.label = role_name(role);
        u.buffer = gfx_.create_buffer(d);
        u.capacity = u.buffer ? d.size : 0;
        if (!u.buffer) {
            if (why) *why = std::string(role_name(role)) + ": no buffer of " + std::to_string(d.size) +
                            " bytes (" + gfx_.last_error() + ")";
            return false;
        }
        ++counts_.buffers_created;
    } else {
        ++counts_.buffers_reused;
    }
    if (!gfx_.write_buffer(u.buffer, 0, bytes, size)) {
        if (why) *why = std::string(role_name(role)) + ": " + gfx_.last_error();
        return false;
    }
    counts_.bytes_last += size;
    counts_.bytes_total += size;
    ++counts_.copies_total;
    *out = u.buffer;
    return true;
}

// An image's or a textured mesh's pixels into the slot's texture, which is kept while the size
// and format stay: one copy from the host, counted with the buffers' uploads.
bool SceneView::upload_texture(Slot& s, caliper_gfx::Format f, uint32_t w, uint32_t h, const void* data,
                               uint32_t row_pitch, std::string* why) {
    if (!s.texture || s.tex_w != w || s.tex_h != h || s.tex_format != f) {
        release_texture(s);
        caliper_gfx::TextureDesc d;
        d.width = w;
        d.height = h;
        d.format = f;
        d.usage = caliper_gfx::kSampled | caliper_gfx::kCopyDstTex;
        d.label = "scene image";
        s.texture = gfx_.create_texture(d);
        if (!s.texture) {
            if (why) *why = "no texture of " + std::to_string(w) + " x " + std::to_string(h) + " (" + gfx_.last_error() + ")";
            return false;
        }
        s.tex_w = w;
        s.tex_h = h;
        s.tex_format = f;
        ++counts_.textures_created;
    } else {
        ++counts_.textures_reused;
    }
    if (!gfx_.write_texture(s.texture, data, row_pitch)) {
        if (why) *why = "the image: " + gfx_.last_error();
        return false;
    }
    const uint64_t bytes = uint64_t(w) * h * caliper_gfx::format_bytes(f);
    counts_.bytes_last += bytes;
    counts_.bytes_total += bytes;
    ++counts_.copies_total;
    return true;
}

void SceneView::release_texture(Slot& s) {
    if (s.texture) gfx_.release(s.texture);   // the layer keeps it until its last frame completed
    s.texture = {};
    s.tex_w = s.tex_h = 0;
}

void SceneView::release(Slot& s) {
    for (auto& kv : s.buffers)
        if (kv.second.buffer) gfx_.release(kv.second.buffer);
    s.buffers.clear();
    release_texture(s);
    s.cpu_rows.clear();
    s.cpu_ids.clear();
    s.cpu.clear();
    s.present = false;
}

void SceneView::reset_scene() {
    for (auto& kv : slots_) release(kv.second);
    slots_.clear();
    tree_.reset();
    changed();
}

// The entity's batches from its slots: both layers of the entity's batches are cleared in the
// store, then the static layer of each part is set again, then the stamped one, which wins where
// both exist. Its transform and camera live in the store's own layers and stay.
void SceneView::rebuild(const std::string& path) {
    const EntityId e = tree_.find(path);
    if (!e) return;
    tree_.clear(e, false, true, kClearBatches);
    for (const bool fixed : {true, false})
        for (int p = 1; p <= kPrimitiveCount; ++p) {
            const auto it = slots_.find(Key{path, p, fixed});
            if (it == slots_.end() || !it->second.present) continue;
            std::string err;
            if (!tree_.set_batch(e, it->second.batch, fixed, &err)) note((path.empty() ? "the root" : path) + ": " + err);
        }
    changed();
}

// ---- messages ------------------------------------------------------------------------------------

bool SceneView::apply(const wire::SceneUpdate& u, std::string* why) {
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
    if (!caliper_host::colormap_lut(int32_t(b.colormap))) b.colormap = 0;

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
        Buffer buf;
        const uint64_t size = rows * (stride ? stride : vertex_format_bytes(fmt));
        if (!upload(slot, role, bytes, size, &buf, &problem)) return false;
        if (prim == Primitive::Points || prim == Primitive::Boxes)
            cpu[int(role)].assign(static_cast<const uint8_t*>(bytes), static_cast<const uint8_t*>(bytes) + size);
        BatchArray a;
        a.role = role;
        a.buffer = buf;
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
                if (!upload_texture(slot, caliper_gfx::Format::RGBA8Unorm, tw, th, tex->data, tw * 4, &problem))
                    return failed();
                b.texture = slot.texture;
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
            bool ok = false;
            if (form == CALIPER_NET_FORM_COLOR) {
                const bool grey = lead->shape.size() == 2;
                b.image_form = grey ? ImageForm::Grey : ImageForm::Color;
                ok = upload_texture(slot, grey ? caliper_gfx::Format::R8Unorm : caliper_gfx::Format::RGBA8Unorm, w, h,
                                    lead->data, w * (grey ? 1 : 4), &problem);
                if (!grey) {
                    const uint32_t* px = reinterpret_cast<const uint32_t*>(lead->data);
                    for (uint64_t i = 0, m = uint64_t(w) * h; i < m && !translucent; ++i) translucent = (px[i] >> 24) < 255;
                }
            } else if (form == CALIPER_NET_FORM_DEPTH || form == CALIPER_NET_FORM_SEGMENTATION) {
                // Bits in an r32uint texture: f32 depths in scene units (a u16 depth scaled to them), or
                // class ids, so that the colormap and the class table decide on the values as they are.
                std::vector<uint32_t> bits(size_t(w) * h);
                if (form == CALIPER_NET_FORM_DEPTH) {
                    b.image_form = ImageForm::Depth;
                    const bool u16 = lead->dtype == CALIPER_NET_DTYPE_U16;
                    const float scale = float(num(params, "scale", u16 ? 0.001 : 1.0));
                    if (u16) {
                        const uint16_t* src = reinterpret_cast<const uint16_t*>(lead->data);
                        for (size_t i = 0; i < bits.size(); ++i) {
                            const float d = float(src[i]) * scale;
                            std::memcpy(&bits[i], &d, 4);
                        }
                    } else {
                        const float* src = reinterpret_cast<const float*>(lead->data);
                        for (size_t i = 0; i < bits.size(); ++i) {
                            const float d = scale == 1.0f ? src[i] : src[i] * scale;
                            std::memcpy(&bits[i], &d, 4);
                        }
                    }
                    const caliper_net::Json* cl = params.find("cloud");
                    b.cloud = cl && cl->is_bool() ? cl->as_bool() : in_pixel_space(e);   // the model's default
                } else {
                    b.image_form = ImageForm::Segmentation;
                    if (lead->dtype == CALIPER_NET_DTYPE_U16) {
                        const uint16_t* src = reinterpret_cast<const uint16_t*>(lead->data);
                        for (size_t i = 0; i < bits.size(); ++i) bits[i] = src[i];
                    } else {
                        for (size_t i = 0; i < bits.size(); ++i) bits[i] = lead->data[i];
                    }
                }
                ok = upload_texture(slot, caliper_gfx::Format::R32Uint, w, h, bits.data(), w * 4, &problem);
            } else {
                return refuse("an image of form '" + form + "'");
            }
            if (!ok) return failed();
            b.texture = slot.texture;
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
        slot.ax_start = get("start", 0);
        slot.ax_step = get("step", 1);
        slot.ax_count = get("count", 0);
        slot.ax_size = get("size", 0);
    }
    slot.cpu_rows = std::move(cpu_rows);   // the batch's CPU views point into these buffers
    slot.cpu_ids = std::move(cpu_ids);
    slot.cpu = std::move(cpu);
    slot.present = true;
    // Buffers of roles this batch no longer has, and a texture it no longer has, go back to the layer.
    for (auto it = slot.buffers.begin(); it != slot.buffers.end();) {
        if (!b.array(Role(it->first))) {
            if (it->second.buffer) gfx_.release(it->second.buffer);
            it = slot.buffers.erase(it);
        } else {
            ++it;
        }
    }
    if (slot.texture && !b.texture) release_texture(slot);
    changed();
    return true;
}

bool SceneView::apply(const wire::SceneTransforms& t, std::string* why) {
    ++counts_.messages;
    const uint64_t k = t.paths.size();
    if (t.matrices.size < k * 64) {
        note("entity transforms: the matrices are short");
        if (why) *why = "entity transforms: the matrices are short";
        return false;
    }
    bool ok = true;
    for (uint64_t i = 0; i < k; ++i) {
        float m[16];
        std::memcpy(m, t.matrices.data + i * 64, 64);
        std::string err;
        const EntityId e = tree_.ensure(t.paths[i], &err);
        if (!e || !tree_.set_transform(e, m, t.is_static, &err)) {
            const std::string text = t.paths[i] + " transform: " + err;
            note(text);
            if (why && ok) *why = text;
            ok = false;
        }
    }
    changed();
    return ok;
}

bool SceneView::apply(const wire::SceneClasses& c, std::string* why) {
    ++counts_.messages;
    std::vector<ClassEntry> table;
    for (const caliper_net::Json& x : c.classes.items()) {
        ClassEntry ce;
        const caliper_net::Json* id = x.find("id");
        int64_t v = 0;
        if (!id || !id->get_i64(&v) || v < 0 || v > int64_t(kMaxClassId)) continue;
        ce.id = uint32_t(v);
        if (const caliper_net::Json* l = x.find("label"); l && l->is_string()) ce.label = l->as_string();
        if (!flat_color(x, &ce.rgba)) ce.rgba = kPalette[ce.id % 16];
        table.push_back(std::move(ce));
    }
    std::string err;
    if (!tree_.set_classes(table, c.is_static, &err)) {
        note("class table: " + err);
        if (why) *why = "class table: " + err;
        return false;
    }
    changed();
    return true;
}

bool SceneView::apply(const wire::SceneClear& c, std::string* why) {
    ++counts_.messages;
    if (c.reset) {   // a new state: everything goes, static items included
        reset_scene();
        return true;
    }
    if (!tree_.find(c.path) && c.path != "") return true;   // nothing there
    std::vector<int> parts;
    uint32_t pose = 0;   // the transform and camera bits, which the store's layers hold
    if (c.parts) {
        for (const std::string& p : *c.parts) {
            if (p == CALIPER_NET_PART_TRANSFORM) pose |= kClearTransform;
            else if (p == CALIPER_NET_PART_CAMERA) pose |= kClearCamera;
            else if (const int k = wire::scene_part_index(p); k >= 0 && k < kPrimitiveCount) parts.push_back(k + 1);
        }
    } else {
        for (int p = 1; p <= kPrimitiveCount; ++p) parts.push_back(p);
        pose = kClearTransform | kClearCamera;
    }
    std::vector<std::string> touched;
    const std::string prefix = c.path.empty() ? std::string() : c.path + "/";
    for (auto& kv : slots_) {
        const Key& k = kv.first;
        const bool inside = k.path == c.path || (c.recursive && (c.path.empty() || k.path.compare(0, prefix.size(), prefix) == 0));
        if (!inside || !kv.second.present) continue;
        if (std::find(parts.begin(), parts.end(), k.primitive) == parts.end()) continue;
        if (k.fixed && !c.is_static) continue;   // a clear keeps static items unless it is static itself
        release(kv.second);
        if (std::find(touched.begin(), touched.end(), k.path) == touched.end()) touched.push_back(k.path);
    }
    for (const std::string& p : touched) rebuild(p);
    if (pose) {
        tree_.clear(c.path.empty() ? tree_.root() : tree_.find(c.path), c.recursive, c.is_static, pose);
        changed();
    }
    (void)why;
    return true;
}

bool SceneView::look_through(const std::string& path, std::string* why) {
    if (path.empty()) {
        if (!through_.empty()) changed();
        through_.clear();
        return true;
    }
    const EntityId e = tree_.find(path);
    if (!e || !tree_.camera(e)) {
        if (why) *why = path + " holds no camera";
        return false;
    }
    through_ = path;
    changed();
    return true;
}

ViewCamera SceneView::effective_camera() const {
    if (!through_.empty()) {
        const EntityId c = tree_.find(through_);
        if (c && tree_.camera(c)) {
            ThroughPinhole t;
            t.camera = c;
            return t;
        }
    }
    return camera_;
}

std::vector<std::string> SceneView::cameras() const {
    std::vector<std::string> out;
    for (uint32_t i : tree_.path_order()) {
        const EntityId e = tree_.entity_at(i);
        if (tree_.camera(e)) out.push_back(tree_.path(e));
    }
    return out;
}

std::vector<LabelAnchor> SceneView::labels() const {
    if (!width_ || !height_) return {};
    return project_labels(tree_, effective_camera(), width_, height_);
}

// ---- picking ---------------------------------------------------------------------------------------

bool SceneView::request_pick(float x, float y, std::string* why) {
    if (!pick_ || !res_ || !width_ || !height_) {
        if (why) *why = "the scene view has not drawn yet";
        return false;
    }
    CompileOptions opt;
    opt.width = width_;
    opt.height = height_;
    opt.pixel_scale = scale_;
    opt.clear_rgba = clear_;
    return pick_->request(tree_, effective_camera(), opt, compiler_, *res_, x, y, generation_, why);
}

PickPass::Result SceneView::poll_pick(Hit* out) {
    return pick_ ? pick_->poll(generation_, out) : PickPass::Result::None;
}

int64_t SceneView::row_id(const std::string& path, Primitive p, int64_t index) const {
    const EntityId e = tree_.find(path);
    const bool fixed = e && tree_.batch_static(e, p);
    const auto it = slots_.find(Key{path, int(p), fixed});
    if (it == slots_.end() || it->second.ax_size <= 0) return index;
    const Slot& s = it->second;
    return s.ax_start + (index * s.ax_count) / s.ax_size * s.ax_step;
}

std::string SceneView::pick_detail(const Hit& h) const {
    if (h.part == DrawPart::Frustum || h.part == DrawPart::Axes) return {};
    const std::string path = tree_.path(h.entity);
    // A depth cloud's point is a pixel of its depth image; an image's item is its pixel too.
    int64_t index = int64_t(h.item);
    if (h.primitive == Primitive::Image && h.part == DrawPart::Batch) {
        const Batch* b = tree_.batch(h.entity, Primitive::Image);
        index = h.has_pixel && b ? int64_t(h.pixel[1]) * b->image_width + h.pixel[0] : 0;
    }
    caliper_net::Json d = caliper_net::Json::object();
    d.set("entity", caliper_net::Json(path));
    d.set("primitive", caliper_net::Json(primitive_name(h.primitive)));
    d.set("index", caliper_net::Json(index));
    d.set("row_id", caliper_net::Json(row_id(path, h.primitive, index)));
    if (h.primitive == Primitive::Image && h.has_pixel) {
        caliper_net::Json px = caliper_net::Json::array();
        px.push(caliper_net::Json(int64_t(h.pixel[0])));
        px.push(caliper_net::Json(int64_t(h.pixel[1])));
        d.set("pixel", px);
    }
    if (h.has_position) {
        caliper_net::Json pos = caliper_net::Json::array();
        for (double c : {h.position.x, h.position.y, h.position.z}) pos.push(caliper_net::Json(c));
        d.set("position", pos);
    }
    return d.dump();
}

// ---- selections --------------------------------------------------------------------------------------

std::string SceneView::select_polygon(const std::vector<float>& poly) {
    selected_.clear();
    const size_t m = poly.size() / 2;
    if (m >= 3 && width_ && height_) {
        float xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
        for (size_t k = 0; k < m; ++k) {
            xmin = std::min(xmin, poly[2 * k]);
            xmax = std::max(xmax, poly[2 * k]);
            ymin = std::min(ymin, poly[2 * k + 1]);
            ymax = std::max(ymax, poly[2 * k + 1]);
        }
        auto inside = [&](double px, double py) {
            if (!(px >= xmin && px <= xmax && py >= ymin && py <= ymax)) return false;
            bool in = false;
            for (size_t a = 0, b = m - 1; a < m; b = a++) {
                const double ax = poly[2 * a], ay = poly[2 * a + 1], bx = poly[2 * b], by = poly[2 * b + 1];
                if ((ay > py) != (by > py) && px < (bx - ax) * (py - ay) / (by - ay) + ax) in = !in;
            }
            return in;
        };
        tree_.update();
        const CameraMatrices cm = camera_matrices(effective_camera(), tree_, width_, height_);
        if (cm.ok)
            for (uint32_t idx : tree_.path_order()) {
                const EntityId e = tree_.entity_at(idx);
                if (!tree_.drawable(e)) continue;
                const EntityId pcam = tree_.pixel_space_of(e);
                const bool overlay = pcam && pcam == cm.through;
                const Mat4 mvp = overlay ? cm.pixel_to_clip * tree_.pixel_local(e) : cm.proj * cm.view * tree_.world(e);
                for (const Primitive p : {Primitive::Points, Primitive::Boxes}) {
                    const Batch* b = tree_.batch(e, p);
                    if (!b) continue;
                    const auto it = slots_.find(Key{tree_.path(e), int(p), tree_.batch_static(e, p)});
                    if (it == slots_.end()) continue;
                    const auto rows = it->second.cpu.find(int(p == Primitive::Points ? Role::Positions : Role::Centers));
                    if (rows == it->second.cpu.end()) continue;
                    const uint32_t comps = b->dims();
                    const uint64_t n = b->count();
                    const float* f = reinterpret_cast<const float*>(rows->second.data());
                    std::vector<int64_t> idxs;
                    for (uint64_t i = 0; i < n; ++i) {
                        const Vec3 q{f[i * comps], f[i * comps + 1], comps == 3 ? f[i * comps + 2] : 0.0};
                        double h[4];
                        transform_h(mvp, q, h);
                        if (!(h[3] > 0)) continue;
                        const double z = h[2] / h[3];
                        if (!(z >= 0 && z <= 1)) continue;
                        if (inside((h[0] / h[3] + 1) * 0.5 * width_, (1 - h[1] / h[3]) * 0.5 * height_)) idxs.push_back(int64_t(i));
                    }
                    if (!idxs.empty()) selected_[{tree_.path(e), int(p)}] = std::move(idxs);
                }
            }
    }
    changed();
    if (selected_.empty()) return "{}";
    // The detail: at most 32,768 indices in all, in path order.
    caliper_net::Json ents = caliper_net::Json::array();
    size_t total = 0;
    bool truncated = false;
    std::string space;
    std::vector<int64_t> items;
    for (const auto& kv : selected_) {
        const Primitive p = Primitive(kv.first.second);
        caliper_net::Json ent = caliper_net::Json::object();
        caliper_net::Json idx = caliper_net::Json::array(), rows = caliper_net::Json::array();
        const auto it = slots_.find(Key{kv.first.first, kv.first.second, tree_.batch_static(tree_.find(kv.first.first), p)});
        const std::string items_space = it != slots_.end() ? it->second.items : std::string();
        for (const int64_t i : kv.second) {
            if (total >= CALIPER_NET_MAX_EVENT_INDICES) {
                truncated = true;
                break;
            }
            const int64_t r = row_id(kv.first.first, p, i);
            idx.push(caliper_net::Json(i));
            rows.push(caliper_net::Json(r));
            ++total;
            if (!items_space.empty() && (space.empty() || space == items_space)) {
                space = items_space;
                items.push_back(r);
            }
        }
        ent.set("entity", caliper_net::Json(kv.first.first));
        ent.set("primitive", caliper_net::Json(primitive_name(p)));
        ent.set("indices", idx);
        ent.set("row_ids", rows);
        ents.push(ent);
        if (truncated) break;
    }
    caliper_net::Json d = caliper_net::Json::object();
    d.set("entities", ents);
    if (truncated) d.set("truncated", caliper_net::Json(true));
    if (!space.empty()) {
        std::sort(items.begin(), items.end());
        items.erase(std::unique(items.begin(), items.end()), items.end());
        caliper_net::Json it = caliper_net::Json::array();
        for (const int64_t x : items) it.push(caliper_net::Json(x));
        d.set("space", caliper_net::Json(space));
        d.set("items", it);
    }
    return d.dump();
}

bool SceneView::clear_selection() {
    const bool had = !selected_.empty();
    selected_.clear();
    if (had) changed();
    return had;
}

void SceneView::set_linked(const std::string& space, std::vector<int64_t> items) {
    std::sort(items.begin(), items.end());
    items.erase(std::unique(items.begin(), items.end()), items.end());
    const auto it = linked_.find(space);
    if (items.empty()) {
        if (it == linked_.end()) return;
        linked_.erase(it);
    } else {
        if (it != linked_.end() && it->second == items) return;
        linked_[space] = std::move(items);
    }
    changed();
}

void SceneView::set_all_linked(const std::map<std::string, std::vector<int64_t>>& by_space) {
    std::vector<std::string> gone;
    for (const auto& kv : linked_)
        if (!by_space.count(kv.first)) gone.push_back(kv.first);
    for (const std::string& sp : gone) set_linked(sp, {});
    for (const auto& kv : by_space) set_linked(kv.first, kv.second);
}

uint64_t SceneView::selected_count() const {
    uint64_t n = 0;
    for (const auto& kv : selected_) n += kv.second.size();
    return n;
}

// The highlight: a scene of the selected points and boxes alone, each entity with the poses and
// cameras of its path in the view's scene, so that the view's camera draws them where they are. Their
// arrays are the selected rows of the CPU copies, uploaded into buffers the view owns (copies, counted).
bool SceneView::build_highlight(SceneTree* hl, ViewCamera* cam, std::string* why) {
    for (const Buffer b : hl_buffers_) gfx_.release(b);
    hl_buffers_.clear();
    highlighted_ = 0;
    *hl = SceneTree(tree_.space(), tree_.up(), tree_.units());
    *cam = camera_;
    // The entity and the poses and cameras of its ancestors, copied.
    auto chain = [&](EntityId e) {
        std::vector<EntityId> up;
        for (EntityId a = e; a && a.index != 0; a = tree_.parent(a)) up.push_back(a);
        EntityId out;
        for (auto it = up.rbegin(); it != up.rend(); ++it) {
            out = hl->ensure(tree_.path(*it));
            if (const Mat4* m = tree_.transform(*it)) hl->set_transform(out, *m);
            if (const Pinhole* c = tree_.camera(*it)) {
                Pinhole pin = *c;
                pin.plane_distance = tree_.plane_distance(*it);   // the view's, which follows its bounds
                hl->set_camera(out, pin);
            }
        }
        return e.index == 0 ? hl->root() : out;
    };
    if (!through_.empty()) {
        const EntityId c = tree_.find(through_);
        if (c && tree_.camera(c)) {
            ThroughPinhole t;
            t.camera = chain(c);
            *cam = t;
        }
    }
    for (uint32_t idx : tree_.path_order()) {
        const EntityId e = tree_.entity_at(idx);
        if (!tree_.drawable(e)) continue;
        for (const Primitive p : {Primitive::Points, Primitive::Boxes}) {
            const Batch* b = tree_.batch(e, p);
            if (!b) continue;
            const std::string& path = tree_.path(e);
            const auto it = slots_.find(Key{path, int(p), tree_.batch_static(e, p)});
            if (it == slots_.end()) continue;
            const Slot& slot = it->second;
            const uint64_t n = b->count();
            std::vector<int64_t> pick;
            if (const auto sel = selected_.find({path, int(p)}); sel != selected_.end()) pick = sel->second;
            if (const auto lk = linked_.find(slot.items); !slot.items.empty() && lk != linked_.end())
                for (uint64_t i = 0; i < n; ++i)
                    if (std::binary_search(lk->second.begin(), lk->second.end(), row_id(path, p, int64_t(i))))
                        pick.push_back(int64_t(i));
            std::sort(pick.begin(), pick.end());
            pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
            while (!pick.empty() && pick.back() >= int64_t(n)) pick.pop_back();
            if (pick.empty()) continue;
            Batch h = *b;
            h.labels.clear();
            for (BatchArray& a : h.arrays) {
                const auto src = slot.cpu.find(int(a.role));
                if (src == slot.cpu.end()) {
                    if (why) *why = path + ": no CPU copy of its " + role_name(a.role);
                    return false;
                }
                const uint32_t row = a.stride ? a.stride : vertex_format_bytes(a.format);
                std::vector<uint8_t> bytes;
                if (a.count == 1) {
                    bytes = src->second;   // one value for all
                } else {
                    bytes.resize(pick.size() * row);
                    for (size_t k = 0; k < pick.size(); ++k)
                        std::memcpy(bytes.data() + k * row, src->second.data() + size_t(pick[k]) * row, row);
                    a.count = pick.size();
                }
                caliper_gfx::BufferDesc d;
                d.size = (bytes.size() + 3) / 4 * 4;
                d.usage = caliper_gfx::kVertex | caliper_gfx::kCopyDst;
                d.label = "scene highlight";
                const Buffer buf = gfx_.create_buffer(d);
                if (!buf || !gfx_.write_buffer(buf, 0, bytes.data(), bytes.size())) {
                    if (why) *why = "the highlight: " + gfx_.last_error();
                    if (buf) gfx_.release(buf);
                    return false;
                }
                hl_buffers_.push_back(buf);
                counts_.bytes_last += bytes.size();
                counts_.bytes_total += bytes.size();
                ++counts_.copies_total;
                a.buffer = buf;
                a.offset = 0;
                a.cpu = nullptr;
            }
            std::string err;
            if (!hl->set_batch(chain(e), h, false, &err)) {
                if (why) *why = path + ": the highlight: " + err;
                return false;
            }
            highlighted_ += pick.size();
        }
    }
    hl->update();
    return true;
}

bool SceneView::in_pixel_space(EntityId e) const {
    for (EntityId p = tree_.parent(e); p; p = tree_.parent(p))
        if (tree_.camera(p)) return true;
    return false;
}

// ---- camera --------------------------------------------------------------------------------------

void SceneView::orbit(float dx, float dy) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->azimuth += double(dx) * 0.01;
        o->elevation = std::clamp(o->elevation + double(dy) * 0.01, -kMaxElevation, kMaxElevation);
        moved_ = true;
        changed();
    } else {
        pan(dx, dy, float(std::max<uint32_t>(height_, 1)) / scale_);
    }
}

void SceneView::pan(float dx, float dy, float view_height) {
    if (!(view_height > 0)) return;
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        const Mat4 v = orbit_view(*o);   // rows 0 and 1 of the view's rotation: right and up
        const Vec3 s{v.at(0, 0), v.at(0, 1), v.at(0, 2)}, up{v.at(1, 0), v.at(1, 1), v.at(1, 2)};
        const double k = 2.0 * o->distance * std::tan(kOrbitFovY * 0.5) / view_height;
        const Vec3 d = (s * -double(dx) + up * double(dy)) * k;
        o->target[0] += d.x;
        o->target[1] += d.y;
        o->target[2] += d.z;
    } else if (PanZoom* p = std::get_if<PanZoom>(&camera_)) {
        const double k = 2.0 * p->half_height / view_height;
        p->center[0] -= double(dx) * k;
        p->center[1] += (p->y_down ? -1.0 : 1.0) * double(dy) * k;
    }
    moved_ = true;
    changed();
}

void SceneView::zoom(float wheel, float mx, float my, float view_w, float view_h) {
    if (wheel == 0.0f) return;
    const double f = std::pow(0.9, double(wheel));
    const Aabb b = tree_.scene_bounds();
    const double r = b.valid ? std::max(b.radius(), 1e-6) : 1.0;
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->distance = std::clamp(o->distance * f, r * 1e-3, r * 1e3);
    } else if (PanZoom* p = std::get_if<PanZoom>(&camera_)) {
        const double aspect = view_h > 0 ? double(view_w) / double(view_h) : 1.0;
        const double sy = p->y_down ? -1.0 : 1.0;
        const double wx = p->center[0] + mx * p->half_height * aspect, wy = p->center[1] + sy * my * p->half_height;
        p->half_height = std::clamp(p->half_height * f, r * 1e-5, r * 1e5);
        p->center[0] = wx - mx * p->half_height * aspect;
        p->center[1] = wy - sy * my * p->half_height;
    }
    moved_ = true;
    changed();
}

void SceneView::fit() {
    look_through("", nullptr);   // a fit leaves a pinhole for the orbit of the fit
    tree_.update();
    const double aspect = (width_ && height_) ? double(width_) / double(height_) : 1.0;
    caliper_scene::fit(camera_, tree_.scene_bounds(), aspect);
    fitted_ = tree_.scene_bounds().valid;
    changed();
}

// ---- drawing ---------------------------------------------------------------------------------------

bool SceneView::render(uint32_t width, uint32_t height, float pixel_scale, uint32_t clear_rgba, std::string* why) {
    if (width == 0 || height == 0) return true;
    if (!res_) {
        if (why) *why = "the scene view has no resources";
        return false;
    }
    const bool depth = tree_.space() == Space::k3D;
    if (!color_ || width != width_ || height != height_ || bool(depth_) != depth) {
        if (color_) gfx_.release(color_);
        if (depth_) gfx_.release(depth_);
        depth_ = {};
        caliper_gfx::TextureDesc d;
        d.width = width;
        d.height = height;
        d.format = caliper_gfx::Format::RGBA8Unorm;
        d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kSampled | caliper_gfx::kCopySrcTex;
        d.label = "scene view colour";
        color_ = gfx_.create_texture(d);
        if (depth) {
            d.format = caliper_gfx::Format::Depth32Float;
            d.usage = caliper_gfx::kRenderTarget;
            d.label = "scene view depth";
            depth_ = gfx_.create_texture(d);
        }
        if (!color_ || (depth && !depth_)) {
            if (why) *why = "no render target of " + std::to_string(width) + " x " + std::to_string(height) + ": " +
                            gfx_.last_error();
            width_ = height_ = 0;
            return false;
        }
        width_ = width;
        height_ = height;
        ++target_creates_;
        changed();
    }
    if (pixel_scale != scale_ || clear_rgba != clear_) {
        scale_ = pixel_scale;
        clear_ = clear_rgba;
        changed();
    }
    if (tree_.revision() != drawn_revision_ || tree_.pose_revision() != drawn_pose_) changed();
    if (!dirty_) return true;

    const auto t0 = std::chrono::steady_clock::now();
    tree_.update();
    // The scene fits once, when it first has bounds, and then only when the person asks.
    if (!fitted_ && tree_.scene_bounds().valid) {
        caliper_scene::fit(camera_, tree_.scene_bounds(), double(width_) / double(height_));
        fitted_ = true;
    }
    if (res_->update_class_table(tree_)) {   // 65,536 texels from the host
        counts_.bytes_last += 65536u * 4u;
        counts_.bytes_total += 65536u * 4u;
        ++counts_.copies_total;
    }
    caliper_gfx::Frame frame;
    compiler_.prepare(tree_, *res_, frame);
    caliper_gfx::RenderTarget target;
    target.color = color_;
    target.depth = depth_;
    target.clear_rgba = clear_;
    caliper_gfx::DrawList list(target);
    CompileOptions opt;
    opt.width = width_;
    opt.height = height_;
    opt.pixel_scale = scale_;
    opt.clear_rgba = clear_;
    DrawMap map;
    if (!through_.empty() && !std::holds_alternative<ThroughPinhole>(effective_camera()))
        note(through_ + " holds no camera now; the view shows the orbit");
    const CompileResult r = compiler_.compile(tree_, effective_camera(), opt, *res_, list, &map);
    for (const std::string& n : r.notes) note(n);
    cost_.compile_ms = ms_since(t0);
    if (!r.ok) {
        if (why) *why = r.notes.empty() ? std::string("the scene did not compile") : r.notes[0];
        compiler_.discard_prepared();   // the frame with its clamps is dropped
        dirty_ = false;   // not retried every frame; the next change tries again
        return false;
    }
    if (keep_draws_) {
        last_list_ = list;
        last_map_ = map;
    }
    cost_.draws = r.draws;
    frame.render(std::move(list));
    // A selection: the picture dimmed towards the background, as the points view dims, then the
    // selected items over it in their own colours, without depth so that none is hidden.
    highlighted_ = 0;
    if (highlight_wanted()) {
        SceneTree hl(tree_.space());
        ViewCamera hl_cam;
        std::string err;
        if (!build_highlight(&hl, &hl_cam, &err)) {
            note(err);
        } else if (highlighted_ > 0) {
            if (!dim_texture_ || dim_rgba_ != clear_) {
                if (!dim_texture_)
                    dim_texture_ = gfx_.create_texture({1, 1, caliper_gfx::Format::RGBA8Unorm,
                                                        caliper_gfx::kSampled | caliper_gfx::kCopyDstTex, "scene dim"});
                const uint32_t ground = clear_ | 0xff000000u;
                if (dim_texture_ && gfx_.write_texture(dim_texture_, &ground, 4)) dim_rgba_ = clear_;
            }
            caliper_gfx::RenderTarget over;
            over.color = color_;
            over.clear = false;
            SceneTree dim(Space::k2D);
            Batch q;
            q.primitive = Primitive::Image;
            q.texture = dim_texture_;
            q.image_width = width_;
            q.image_height = height_;
            q.opacity = 0.72f;   // the points view's dimming: 0.28 of the colour, 0.72 of the ground
            PanZoom pz;
            pz.center[0] = width_ * 0.5;
            pz.center[1] = height_ * 0.5;
            pz.half_height = height_ * 0.5;
            pz.y_down = true;
            caliper_gfx::DrawList dl(over), hll(over);
            if (dim_texture_ && dim.set_batch(dim.root(), q, false, &err)) {
                dim.update();
                CompileOptions dopt = opt;
                if (dim_compiler_.compile(dim, pz, dopt, *res_, dl, nullptr).ok) frame.render(std::move(dl));
            }
            const CompileResult hr = hl_compiler_.compile(hl, hl_cam, opt, *res_, hll, nullptr);
            if (hr.ok) frame.render(std::move(hll));
            else if (!hr.notes.empty()) note("the highlight: " + hr.notes[0]);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const caliper_gfx::Serial s = gfx_.submit(std::move(frame));
    cost_.submit_ms = ms_since(t1);
    dirty_ = false;
    drawn_revision_ = tree_.revision();
    drawn_pose_ = tree_.pose_revision();
    if (s == 0) {
        compiler_.discard_prepared();
        if (why) *why = "the layer refused the frame: " + gfx_.last_error();
        return false;
    }
    ++cost_.frames;
    ++generation_;
    return true;
}

void SceneView::poll() {
    gfx_.poll();
    const caliper_gfx::Stats st = gfx_.take_stats();
    if (st.gpu_ms >= 0) cost_.gpu_ms = st.gpu_ms;
}

uint64_t SceneView::imgui_texture() const { return color_ ? gfx_.imgui_texture_id(color_) : 0; }

bool SceneView::read_pixels(std::vector<uint8_t>* rgba, std::string* why) {
    if (!color_) {
        if (why) *why = "no render target yet";
        return false;
    }
    const caliper_gfx::Readback rb = gfx_.read_texture(color_, 0, 0, width_, height_);
    if (!rb || !gfx_.wait(gfx_.last_use(color_)) || !gfx_.take_readback(rb, rgba)) {
        if (why) *why = "the readback failed: " + gfx_.last_error();
        return false;
    }
    return true;
}

// ---- reading -------------------------------------------------------------------------------------------

SceneCounts SceneView::counts() const {
    SceneCounts c = counts_;
    std::vector<std::string> seen;
    c.batches = 0;
    c.items = 0;
    c.buffers_live = 0;
    c.textures_live = 0;
    for (const auto& kv : slots_) {
        c.buffers_live += kv.second.buffers.size();
        c.textures_live += kv.second.texture ? 1 : 0;
    }
    for (size_t i = 0; i < tree_.entity_count(); ++i) {
        const EntityId e = tree_.entity_at(uint32_t(i));
        bool any = false;
        for (int p = 1; p <= kPrimitiveCount; ++p) {
            const Batch* b = tree_.batch(e, Primitive(p));
            if (!b) continue;
            any = true;
            ++c.batches;
            const BatchArray* idx = b->array(Role::Indices);
            switch (b->primitive) {
                case Primitive::Lines: {
                    const uint64_t v = b->count();
                    c.items += b->line_mode == LineMode::Edges ? (idx ? idx->count : 0)
                               : b->line_mode == LineMode::Strips ? (v ? v - 1 : 0)
                                                                  : v / 2;
                    break;
                }
                case Primitive::Mesh: c.items += idx ? idx->count : b->count() / 3; break;
                case Primitive::Image: c.items += 1; break;
                default: c.items += b->count(); break;
            }
        }
        if (any) ++c.entities;
    }
    return c;
}

std::string SceneView::info() const {
    const SceneCounts c = counts();
    std::string s = count_text(c.entities) + (c.entities == 1 ? " entity  " : " entities  ") + count_text(c.items) +
                    " items";
    if (c.copies_total) s += "  copied " + bytes_text(c.bytes_last);
    if (cost_.gpu_ms >= 0) {
        char t[32];
        std::snprintf(t, sizeof t, "  gpu %.2f ms", cost_.gpu_ms);
        s += t;
    }
    if (!notes_.empty()) s += "  " + std::to_string(notes_.size()) + (notes_.size() == 1 ? " note" : " notes");
    return s;
}

std::string SceneView::camera_text() const {
    char t[256];
    if (const Orbit* o = std::get_if<Orbit>(&camera_)) {
        std::snprintf(t, sizeof t, "orbit target %.9g %.9g %.9g distance %.9g azimuth %.9g elevation %.9g up %s fovy %.9g",
                      o->target[0], o->target[1], o->target[2], o->distance, o->azimuth, o->elevation, axis_name(o->up),
                      kOrbitFovY);
    } else if (const PanZoom* p = std::get_if<PanZoom>(&camera_)) {
        std::snprintf(t, sizeof t, "panzoom center %.9g %.9g half_height %.9g y_down %d", p->center[0], p->center[1],
                      p->half_height, p->y_down ? 1 : 0);
    } else {
        std::snprintf(t, sizeof t, "pinhole");
    }
    std::string out = std::string(t) + " target_px " + std::to_string(width_) + " " + std::to_string(height_);
    if (std::holds_alternative<ThroughPinhole>(effective_camera())) out += " through " + through_;
    return out;
}

}  // namespace caliper_live_views
