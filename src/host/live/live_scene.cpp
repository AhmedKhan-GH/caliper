// live_scene.cpp: the scene view of the shared view layer (live_scene.h).
#include "live/live_scene.h"

#include "gfx/colormap_keys.h"
#include "scene/backend_resources.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

Primitive primitive_of(const std::string& part) {
    if (part == CALIPER_NET_PART_POINTS) return Primitive::Points;
    if (part == CALIPER_NET_PART_LINES) return Primitive::Lines;
    return Primitive::Mesh;
}

double num(const caliper_net::Json& params, const char* key, double fallback) {
    const caliper_net::Json* v = params.find(key);
    return v && v->is_number() ? v->as_double() : fallback;
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
    res_.reset();
    if (color_) gfx_.release(color_);
    if (depth_) gfx_.release(depth_);
}

bool SceneView::init(std::string* why) {
    res_ = std::make_unique<BackendResources>(gfx_, programs_, caliper_gfx::cmap::colormap_source(colormaps_));
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

void SceneView::release(Slot& s) {
    for (auto& kv : s.buffers)
        if (kv.second.buffer) gfx_.release(kv.second.buffer);
    s.buffers.clear();
    s.present = false;
}

void SceneView::reset_scene() {
    for (auto& kv : slots_) release(kv.second);
    slots_.clear();
    tree_.reset();
    changed();
}

// The entity's batches from its slots: both layers of the entity are cleared in the store, then
// the static layer of each part is set again, then the stamped one, which wins where both exist.
void SceneView::rebuild(const std::string& path) {
    const EntityId e = tree_.find(path);
    if (!e) return;
    tree_.clear(e, false, true);
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
    const Primitive prim = primitive_of(u.part);
    const wire::Section* pos = u.array("positions");
    const std::string where = (u.path.empty() ? std::string("the root") : u.path) + " " + u.part;
    auto refuse = [&](const std::string& text) {
        const std::string t = where + ": " + text;
        note(t);
        if (why) *why = t;
        return false;
    };
    if (!pos || pos->shape.size() != 2) return refuse("no positions");
    const uint64_t n = uint64_t(pos->shape[0]);
    const uint32_t dims = uint32_t(pos->shape[1]);
    const uint32_t want = tree_.space() == Space::k2D ? 2u : 3u;
    if (dims != want)
        return refuse(std::to_string(dims) + "-D positions in a " + std::to_string(want) + "-D scene");

    std::string err;
    const EntityId e = tree_.ensure(u.path, &err);
    if (!e) return refuse(err);
    Slot& slot = slots_[Key{u.path, int(prim), u.is_static}];
    if (n == 0) {   // an empty batch removes the part
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
    b.colormap = uint32_t(std::max(0.0, num(params, "colormap", 0)));
    if (!caliper_host::colormap_lut(int32_t(b.colormap))) b.colormap = 0;

    // The colormap's range: the fixed ends, or the producer's stats for the AUTO ones, as in
    // SNAPSHOT; a flat AUTO range widens by 0.5 on each side, as a heatmap's does.
    std::vector<double> stats;
    if (const caliper_net::Json* st = params.find("stats"); st && st->is_array())
        for (const caliper_net::Json& x : st->items()) stats.push_back(x.as_double());
    const uint64_t flags = uint64_t(std::max(0.0, num(params, "flags", 0)));
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

    // Bounds: the producer's, else a scan of the positions as they arrived.
    bool have = stats.size() >= 8;
    for (size_t k = 2; have && k < 8; ++k) have = std::isfinite(stats[k]);
    if (have) {
        b.bounds.add({stats[2], stats[4], dims == 3 ? stats[6] : 0.0});
        b.bounds.add({stats[3], stats[5], dims == 3 ? stats[7] : 0.0});
    } else {
        const float* p = reinterpret_cast<const float*>(pos->data);
        for (uint64_t i = 0; i < n; ++i) {
            const double x = p[i * dims], y = p[i * dims + 1], z = dims == 3 ? p[i * dims + 2] : 0.0;
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) b.bounds.add({x, y, z});
        }
    }

    std::string problem;
    auto add = [&](Role role, const void* bytes, uint64_t rows, uint32_t stride, VertexFormat fmt) {
        Buffer buf;
        if (!upload(slot, role, bytes, rows * (stride ? stride : vertex_format_bytes(fmt)), &buf, &problem))
            return false;
        BatchArray a;
        a.role = role;
        a.buffer = buf;
        a.count = rows;
        a.stride = stride;
        a.format = fmt;
        b.arrays.push_back(a);
        return true;
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

    if (!add(Role::Positions, pos->data, n, dims * 4, dims == 2 ? VertexFormat::F32x2 : VertexFormat::F32x3))
        return failed();
    const wire::Section* colors = u.array("colors");
    const wire::Section* values = u.array("values");
    bool translucent = has_flat && (rgba >> 24) < 255;

    if (prim == Primitive::Lines) {
        const caliper_net::Json* mode = params.find("mode");
        const bool strips = mode && mode->is_string() && mode->as_string() == CALIPER_NET_LINES_STRIPS;
        b.line_mode = strips ? LineMode::Strips : LineMode::Segments;
        b.width_px = float(num(params, "width", 2.0));
        if (!(b.width_px > 0.0f)) b.width_px = 2.0f;
        if (strips) {
            // The break mask from the strip lengths, and per-strip colours or values expanded to
            // their vertices: the scene core reads both per vertex.
            const wire::Section* sl = u.array("strip_lengths");
            if (!sl) {
                problem = "strips without strip_lengths";
                return failed();
            }
            const int32_t* len = reinterpret_cast<const int32_t*>(sl->data);
            const uint64_t strips_n = uint64_t(sl->shape[0]);
            const caliper_net::Json* cp = params.find("color_per");
            const bool per_strip = cp && cp->is_string() && cp->as_string() == CALIPER_NET_COLOR_PER_STRIP;
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
        }
    } else if (prim == Primitive::Mesh) {
        if (const wire::Section* idx = u.array("indices"))
            if (!add(Role::Indices, idx->data, uint64_t(idx->shape[0]), 12, VertexFormat::S32)) return failed();
    } else {
        if (const wire::Section* radii = u.array("radii")) {
            if (!add(Role::Radii, radii->data, n, 4, VertexFormat::F32)) return failed();
        } else if (const caliper_net::Json* r = params.find("radius"); r && r->is_number()) {
            const float one = float(r->as_double());
            if (!add(Role::Radii, &one, 1, 0, VertexFormat::F32)) return failed();
        }
    }
    if (colors) {
        const uint32_t* rows = reinterpret_cast<const uint32_t*>(colors->data);
        const uint64_t m = uint64_t(colors->shape[0]);
        for (uint64_t i = 0; i < m && !translucent; ++i) translucent = (rows[i] >> 24) < 255;
        if (!add(Role::Colors, colors->data, m, 4, VertexFormat::Unorm8x4)) return failed();
    } else if (values) {
        if (!add(Role::Values, values->data, uint64_t(values->shape[0]), 4, VertexFormat::F32)) return failed();
    }
    b.translucent = translucent;

    // Every upload above is in the slot's buffers; the store takes the batch now.
    if (!tree_.set_batch(e, b, u.is_static, &problem)) return failed();
    slot.batch = b;
    slot.present = true;
    // Buffers of roles this batch no longer has go back to the layer.
    for (auto it = slot.buffers.begin(); it != slot.buffers.end();) {
        if (!b.array(Role(it->first))) {
            if (it->second.buffer) gfx_.release(it->second.buffer);
            it = slot.buffers.erase(it);
        } else {
            ++it;
        }
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
    if (c.parts) {
        for (const std::string& p : *c.parts)
            if (p == CALIPER_NET_PART_POINTS || p == CALIPER_NET_PART_LINES || p == CALIPER_NET_PART_MESH)
                parts.push_back(int(primitive_of(p)));
    } else {
        for (int p = 1; p <= kPrimitiveCount; ++p) parts.push_back(p);
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
    (void)why;
    return true;
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
    res_->update_class_table(tree_);
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
    const CompileResult r = compiler_.compile(tree_, camera_, opt, *res_, list, &map);
    for (const std::string& n : r.notes) note(n);
    cost_.compile_ms = ms_since(t0);
    if (!r.ok) {
        if (why) *why = r.notes.empty() ? std::string("the scene did not compile") : r.notes[0];
        dirty_ = false;   // not retried every frame; the next change tries again
        return false;
    }
    if (keep_draws_) {
        last_list_ = list;
        last_map_ = map;
    }
    cost_.draws = r.draws;
    frame.render(std::move(list));
    const auto t1 = std::chrono::steady_clock::now();
    const caliper_gfx::Serial s = gfx_.submit(std::move(frame));
    cost_.submit_ms = ms_since(t1);
    dirty_ = false;
    drawn_revision_ = tree_.revision();
    drawn_pose_ = tree_.pose_revision();
    if (s == 0) {
        if (why) *why = "the layer refused the frame: " + gfx_.last_error();
        return false;
    }
    ++cost_.frames;
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
    for (const auto& kv : slots_) c.buffers_live += kv.second.buffers.size();
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
                    c.items += b->line_mode == LineMode::Strips ? (v ? v - 1 : 0) : v / 2;
                    break;
                }
                case Primitive::Mesh: c.items += idx ? idx->count : b->count() / 3; break;
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
    return std::string(t) + " target_px " + std::to_string(width_) + " " + std::to_string(height_);
}

}  // namespace caliper_live_views
