// live_points.cpp: the points live view as a scene of the scene core (live_points.h).
#include "live/live_points.h"

#include "gfx/colormap_keys.h"
#include "scene/backend_resources.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace caliper_live_views {

using namespace caliper_scene;
namespace gfx = caliper_gfx;

// The colormap source of the view's BackendResources: the keys of a range from the shared cache,
// with the colours of a built-in colormap, or of the palette at every level, dimmed toward the
// ground for the rest of a selection.
struct PointsScene::Tables {
    std::shared_ptr<gfx::cmap::KeyBlockCache> cache = std::make_shared<gfx::cmap::KeyBlockCache>();
    uint32_t palette[256] = {};
    uint32_t ground = 0xff000000u;
};

uint32_t dimmed_rgba(uint32_t c, uint32_t ground) {
    uint32_t out = 0xff000000u;
    for (int k = 0; k < 3; ++k) {
        const float a = (float)((c >> (8 * k)) & 0xff), b = (float)((ground >> (8 * k)) & 0xff);
        out |= (uint32_t)std::lround(a * 0.28f + b * 0.72f) << (8 * k);
    }
    return out;
}

namespace {

constexpr double kOrbitRate = 0.01;        // radians a logical pixel
constexpr double kElevationBound = 1.55;   // the old view's bound, inside the scene core's 89 degrees

std::string chunk_path(const char* base, uint64_t k) {
    if (k == 0) return base;
    char t[32];
    std::snprintf(t, sizeof t, "%s%sc%05llu", base, *base ? "/" : "", (unsigned long long)k);
    return t;
}

}  // namespace

PointsScene::PointsScene(gfx::Backend& backend, const gfx::ProgramDesc* programs)
    : gfx_(backend), programs_(programs), tables_(std::make_shared<Tables>()) {
    for (int i = 0; i < 256; ++i) tables_->palette[i] = kPalette[i % 16];
}

PointsScene::~PointsScene() {
    res_.reset();
    if (color_) gfx_.release(color_);
    if (depth_) gfx_.release(depth_);
}

bool PointsScene::init(std::string* why) {
    std::shared_ptr<Tables> t = tables_;
    auto source = [t](uint32_t colormap, float vmin, float vmax, gfx::shaders::Cmap* out) {
        const uint32_t base = colormap & ~kDimmed;
        t->cache->fill(base == kPaletteTable ? 0 : int32_t(base), vmin, vmax, out);   // the keys depend on the range only
        if (base == kPaletteTable)
            for (int l = 0; l < 256; ++l) out->levels[l][2] = t->palette[l];
        if (colormap & kDimmed)
            for (int l = 0; l < 256; ++l) out->levels[l][2] = dimmed_rgba(out->levels[l][2], t->ground);
    };
    res_ = std::make_unique<BackendResources>(gfx_, programs_, source);
    return res_->init(why);
}

// ---- content ---------------------------------------------------------------------------------------

bool PointsScene::set(const PointsArrays& all, const PointsArrays* selected, const PointsLook& look, std::string* why) {
    if (look.dims != dims_ || tree_.entity_count() == 0) {
        dims_ = look.dims == 2 ? 2 : 3;
        tree_ = SceneTree(dims_ == 2 ? Space::k2D : Space::k3D);
        if (dims_ == 2) {
            PanZoom p;
            p.y_down = false;
            camera_ = p;
        } else {
            camera_ = Orbit{};
        }
    }
    tree_.reset();
    tables_->ground = look.ground;
    const bool dim = selected != nullptr;
    auto add = [&](const PointsArrays& a, const char* base, bool dimmed) {
        for (uint64_t k = 0, first = 0; first < a.count; ++k, first += kMaxItems) {
            const uint64_t m = std::min<uint64_t>(a.count - first, kMaxItems);
            Batch b;
            b.primitive = Primitive::Points;
            BatchArray p;
            p.role = Role::Positions;
            p.buffer = a.pos;
            p.offset = a.pos_offset + first * 12;
            p.count = m;
            p.stride = 12;
            p.format = dims_ == 2 ? gfx::VertexFormat::F32x2 : gfx::VertexFormat::F32x3;
            p.cpu = a.pos_cpu ? static_cast<const uint8_t*>(a.pos_cpu) + first * 12 : nullptr;
            b.arrays.push_back(p);
            if (look.colour != PointsColour::Flat) {
                BatchArray v;
                v.role = Role::Values;
                v.buffer = a.aux;
                v.offset = a.aux_offset + first * 4;
                v.count = m;
                v.stride = 4;
                v.format = gfx::VertexFormat::F32;
                b.arrays.push_back(v);
                b.colormap = (look.colour == PointsColour::Classes ? kPaletteTable : uint32_t(look.colormap)) |
                             (dimmed ? kDimmed : 0u);
                b.vmin = look.colour == PointsColour::Classes ? 0.0f : look.vmin;
                b.vmax = look.colour == PointsColour::Classes ? 255.0f : look.vmax;
            }
            b.flat_rgba = dimmed ? dimmed_rgba(kDefaultRgba, tables_->ground) : kDefaultRgba;
            b.point_shape = PointShape::Square;
            b.size_px = look.size_px;
            b.full_strength = std::string_view(base) == "selection";   // the selected points: no fade
            if (look.has_bbox) {
                b.bounds.add({look.bbox[0], look.bbox[2], dims_ == 3 ? look.bbox[4] : 0.0});
                b.bounds.add({look.bbox[1], look.bbox[3], dims_ == 3 ? look.bbox[5] : 0.0});
            }
            const std::string path = chunk_path(base, k);
            std::string err;
            const EntityId e = tree_.ensure(path, &err);
            if (!e || !tree_.set_batch(e, b, false, &err)) {
                if (why) *why = err;
                return false;
            }
        }
        return true;
    };
    if (!add(all, "", dim)) return false;
    if (selected && selected->count && !add(*selected, "selection", false)) return false;
    tree_.update();
    return true;
}

// ---- the camera ------------------------------------------------------------------------------------

void PointsScene::fit(double aspect) {
    tree_.update();
    const Aabb b = tree_.scene_bounds();
    caliper_scene::fit(camera_, b, aspect);
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {   // the default three-quarter view, as every renderer
        o->azimuth = kDefaultAzimuth;
        o->elevation = kDefaultElevation;
    }
    Aabb u = b;
    if (!u.valid) {
        u.add({-1, -1, -1});
        u.add({1, 1, 1});
    }
    const Vec3 e = u.half_extent();
    if (dims_ == 3) {
        const double r = length(e);
        radius_ = (r > 1e-12 && std::isfinite(r)) ? r : 1.0;
    } else {
        radius_ = std::max(std::max(std::fabs(e.x), std::fabs(e.y)), 1e-6);
    }
}

void PointsScene::orbit(float dx, float dy) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->azimuth += double(dx) * kOrbitRate;
        o->elevation = std::clamp(o->elevation + double(dy) * kOrbitRate, -kElevationBound, kElevationBound);
    }
}

void PointsScene::turn(double radians) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) o->azimuth += radians;
}

void PointsScene::set_angles(double azimuth, double elevation) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->azimuth = azimuth;
        o->elevation = elevation;
    }
}

void PointsScene::cue_values(float* att_ref, float* fade_w0, float* fade_w1) const {
    *att_ref = *fade_w0 = *fade_w1 = 0.0f;
    const Orbit* o = std::get_if<Orbit>(&camera_);
    if (!cues_ || dims_ != 3 || !o) return;
    *att_ref = float(o->distance);
    *fade_w0 = float(o->distance);
    *fade_w1 = float(o->distance + radius_);
}

void PointsScene::pan(float dx, float dy, float view_height) {
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
        p->center[1] += double(dy) * k;
    }
}

void PointsScene::zoom(float wheel, float mx, float my, double aspect) {
    if (wheel == 0.0f) return;
    const double f = std::pow(0.9, double(wheel));
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->distance = std::clamp(o->distance * f, radius_ * 1e-3, radius_ * 1e3);
    } else if (PanZoom* p = std::get_if<PanZoom>(&camera_)) {
        const double wx = p->center[0] + mx * p->half_height * aspect, wy = p->center[1] + my * p->half_height;
        p->half_height = std::clamp(p->half_height * f, radius_ * 1e-5, radius_ * 1e5);
        p->center[0] = wx - mx * p->half_height * aspect;
        p->center[1] = wy - my * p->half_height;
    }
}

bool PointsScene::mvp(uint32_t width, uint32_t height, float out[16]) const {
    const CameraMatrices cm = camera_matrices(camera_, tree_, width, height);
    if (!cm.ok) return false;
    to_f32(cm.proj * cm.view, out);
    return true;
}

// ---- drawing -----------------------------------------------------------------------------------------

bool PointsScene::render(uint32_t width, uint32_t height, float pixel_scale, uint32_t clear_rgba, gfx::Serial* serial,
                         std::string* why) {
    if (serial) *serial = 0;
    if (width == 0 || height == 0) return true;
    if (!res_) {
        if (why) *why = "the points view has no resources";
        return false;
    }
    const bool depth = dims_ == 3;
    if (!color_ || width != width_ || height != height_ || bool(depth_) != depth) {
        if (color_) gfx_.release(color_);
        if (depth_) gfx_.release(depth_);
        depth_ = {};
        gfx::TextureDesc d;
        d.width = width;
        d.height = height;
        d.format = gfx::Format::RGBA8Unorm;
        d.usage = gfx::kRenderTarget | gfx::kSampled | gfx::kCopySrcTex;
        d.label = "points view colour";
        color_ = gfx_.create_texture(d);
        if (depth) {
            d.format = gfx::Format::Depth32Float;
            d.usage = gfx::kRenderTarget;
            d.label = "points view depth";
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
    }
    tree_.update();
    res_->update_class_table(tree_);
    gfx::Frame frame;
    compiler_.prepare(tree_, *res_, frame);
    gfx::RenderTarget target;
    target.color = color_;
    target.depth = depth_;
    target.clear_rgba = clear_rgba;
    gfx::DrawList list(target);
    CompileOptions opt;
    opt.width = width_;
    opt.height = height_;
    opt.pixel_scale = pixel_scale;
    opt.clear_rgba = clear_rgba;
    cue_values(&opt.point_att_ref, &opt.point_fade_w0, &opt.point_fade_w1);
    DrawMap map;
    const CompileResult r = compiler_.compile(tree_, camera_, opt, *res_, list, &map);
    if (!r.ok) {
        if (why) *why = r.notes.empty() ? std::string("the points did not compile") : r.notes[0];
        return false;
    }
    if (keep_draws_) {
        last_list_ = list;
        last_map_ = map;
    }
    frame.render(std::move(list));
    const gfx::Serial s = gfx_.submit(std::move(frame));
    if (s == 0) {
        if (why) *why = "the layer refused the frame: " + gfx_.last_error();
        return false;
    }
    if (serial) *serial = s;
    return true;
}

uint64_t PointsScene::imgui_texture() const { return color_ ? gfx_.imgui_texture_id(color_) : 0; }

bool PointsScene::read_pixels(std::vector<uint8_t>* rgba, std::string* why) {
    if (!color_) {
        if (why) *why = "no render target yet";
        return false;
    }
    const gfx::Readback rb = gfx_.read_texture(color_, 0, 0, width_, height_);
    if (!rb || !gfx_.wait(gfx_.last_use(color_)) || !gfx_.take_readback(rb, rgba)) {
        if (why) *why = "the readback failed: " + gfx_.last_error();
        return false;
    }
    return true;
}

gfx::Stats PointsScene::poll() {
    gfx_.poll();
    return gfx_.take_stats();
}

}  // namespace caliper_live_views
