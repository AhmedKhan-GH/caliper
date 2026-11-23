// pick.cpp: the pick window, the decoding of its readback, and the pick pass on the graphics layer
// (pick.h).
#include "scene/pick.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caliper_scene {

PickWindow pick_window(float x, float y, int radius_px, uint32_t view_width, uint32_t view_height) {
    PickWindow w;
    w.radius = radius_px < 1 ? 1 : radius_px;
    w.width = w.height = 2 * w.radius;
    w.x0 = int32_t(std::floor(x)) - w.radius;
    w.y0 = int32_t(std::floor(y)) - w.radius;
    w.px = x;
    w.py = y;
    w.view_width = view_width;
    w.view_height = view_height;
    return w;
}

namespace {

// The image pixel under NDC (X, Y), from the quad's mvp (the unit square onto clip space).
bool image_pixel(const DrawMapEntry& e, double X, double Y, uint32_t out[2]) {
    const Mat4& m = e.mvp;
    const double a = m.at(0, 0) - X * m.at(3, 0), b = m.at(0, 1) - X * m.at(3, 1);
    const double c = m.at(1, 0) - Y * m.at(3, 0), d = m.at(1, 1) - Y * m.at(3, 1);
    const double r0 = X * m.at(3, 3) - m.at(0, 3), r1 = Y * m.at(3, 3) - m.at(1, 3);
    const double det = a * d - b * c;
    if (!(std::fabs(det) > 0)) return false;
    const double u = (r0 * d - b * r1) / det, v = (a * r1 - c * r0) / det;
    if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) return false;
    out[0] = std::min<uint32_t>(uint32_t(u * e.image_size[0]), e.image_size[0] ? e.image_size[0] - 1 : 0);
    out[1] = std::min<uint32_t>(uint32_t(v * e.image_size[1]), e.image_size[1] ? e.image_size[1] - 1 : 0);
    return true;
}

}  // namespace

std::optional<Hit> decode_pick(const uint32_t* ids, const float* depth, const PickWindow& w, const DrawMap& map) {
    if (!ids) return std::nullopt;
    int best = -1;
    double best_d2 = 0;
    float best_depth = 0;
    const double r2 = double(w.radius) * w.radius;
    for (int32_t j = 0; j < w.height; ++j)
        for (int32_t i = 0; i < w.width; ++i) {
            const int k = j * w.width + i;
            const uint32_t id = ids[2 * k];
            if (id == 0 || !map.find(id)) continue;
            const int32_t vx = w.x0 + i, vy = w.y0 + j;   // the view pixel
            if (vx < 0 || vy < 0 || vx >= int32_t(w.view_width) || vy >= int32_t(w.view_height)) continue;
            const double dx = vx + 0.5 - w.px, dy = vy + 0.5 - w.py;
            const double d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            const float dz = depth ? depth[k] : 0.0f;
            if (best < 0 || d2 < best_d2 || (d2 == best_d2 && dz < best_depth)) {
                best = k;
                best_d2 = d2;
                best_depth = dz;
            }
        }
    if (best < 0) return std::nullopt;
    const DrawMapEntry& e = *map.find(ids[2 * best]);
    Hit h;
    h.entity = e.entity;
    h.primitive = e.primitive;
    h.part = e.part;
    h.item = ids[2 * best + 1];
    h.overlay = e.overlay;
    h.texel[0] = best % w.width;
    h.texel[1] = best / w.width;
    const double cx = w.x0 + h.texel[0] + 0.5, cy = w.y0 + h.texel[1] + 0.5;
    const double X = 2.0 * cx / w.view_width - 1.0, Y = 1.0 - 2.0 * cy / w.view_height;
    if (e.primitive == Primitive::Image && e.part == DrawPart::Batch) h.has_pixel = image_pixel(e, X, Y, h.pixel);
    if (depth) {
        h.depth = depth[best];
        if (map.has_inverse && !e.overlay && h.depth < 1.0f) {
            double p[4];
            transform_h(map.inverse_view_proj, {X, Y, double(h.depth)}, p);
            if (p[3] != 0) {
                h.position = {p[0] / p[3], p[1] / p[3], p[2] / p[3]};
                h.has_position = finite(h.position);
            }
        }
    }
    return h;
}

PickWindow Picker::request(float x, float y, int radius_px, uint32_t view_width, uint32_t view_height) {
    window_ = pick_window(x, y, radius_px, view_width, view_height);
    ++serial_;
    pending_ = true;
    return window_;
}

std::optional<Hit> Picker::deliver(uint64_t serial, uint64_t revision, const std::vector<uint8_t>& ids,
                                   const std::vector<uint8_t>* depth, const DrawMap& map) {
    if (serial != serial_) return std::nullopt;   // superseded
    pending_ = false;
    if (revision != map.revision) return std::nullopt;   // the draw list changed since
    const size_t texels = size_t(window_.width) * window_.height;
    if (ids.size() < texels * 8) return std::nullopt;
    std::vector<uint32_t> id(texels * 2);
    std::memcpy(id.data(), ids.data(), texels * 8);
    std::vector<float> dz;
    if (depth && depth->size() >= texels * 4) {
        dz.resize(texels);
        std::memcpy(dz.data(), depth->data(), texels * 4);
    }
    return decode_pick(id.data(), dz.empty() ? nullptr : dz.data(), window_, map);
}

// ---- The pick pass -------------------------------------------------------------------------------

using namespace caliper_gfx;

PickPass::PickPass(Backend& backend, int radius_px) : gfx_(backend), radius_(radius_px < 1 ? 1 : radius_px) {}

PickPass::~PickPass() {
    cancel();
    if (ids_target_) gfx_.release(ids_target_);
    if (depth_target_) gfx_.release(depth_target_);
}

void PickPass::cancel() {
    if (rb_ids_) gfx_.release(rb_ids_);
    if (rb_depth_) gfx_.release(rb_depth_);
    rb_ids_ = {};
    rb_depth_ = {};
    have_ids_ = have_depth_ = false;
    waiting_ = false;
}

bool PickPass::targets(bool depth, std::string* why) {
    const uint32_t side = uint32_t(2 * radius_);
    if (!ids_target_) {
        ids_target_ = gfx_.create_texture({side, side, Format::RG32Uint, kRenderTarget | kCopySrcTex, "pick ids"});
        if (!ids_target_) {
            if (why) *why = "no pick target: " + gfx_.last_error();
            return false;
        }
    }
    if (depth && !depth_target_) {
        depth_target_ = gfx_.create_texture({side, side, Format::Depth32Float, kRenderTarget | kCopySrcTex, "pick depth"});
        if (!depth_target_) {
            if (why) *why = "no pick depth target: " + gfx_.last_error();
            return false;
        }
    }
    return true;
}

bool PickPass::request(const SceneTree& tree, const ViewCamera& camera, const CompileOptions& options,
                       SceneCompiler& compiler, Resources& res, float x, float y, uint64_t generation,
                       std::string* why) {
    cancel();   // a new request supersedes an unanswered one
    const bool depth = tree.space() == Space::k3D;
    if (!targets(depth, why)) return false;
    window_ = picker_.request(x, y, radius_, options.width, options.height);
    const int32_t window[4] = {window_.x0, window_.y0, window_.width, window_.height};
    RenderTarget rt;
    rt.color = ids_target_;
    rt.depth = depth ? depth_target_ : Texture{};
    rt.clear_rgba = 0;
    DrawList list(rt);
    map_ = DrawMap{};
    const CompileResult r = compiler.compile_pick(tree, camera, options, window, res, list, &map_);
    if (!r.ok) {
        if (why) *why = r.notes.empty() ? std::string("the pick pass did not compile") : r.notes[0];
        return false;
    }
    map_.revision = generation;   // what the person saw when they pointed
    Frame frame;
    frame.render(std::move(list));
    serial_ = gfx_.submit(std::move(frame));
    if (serial_ == 0) {
        if (why) *why = "the layer refused the pick pass: " + gfx_.last_error();
        return false;
    }
    ++submits_;
    rb_ids_ = gfx_.read_texture(ids_target_, 0, 0, uint32_t(window_.width), uint32_t(window_.height));
    want_depth_ = depth;
    if (depth) rb_depth_ = gfx_.read_texture(depth_target_, 0, 0, uint32_t(window_.width), uint32_t(window_.height));
    if (!rb_ids_ || (depth && !rb_depth_)) {
        if (why) *why = "no pick readback: " + gfx_.last_error();
        cancel();
        return false;
    }
    generation_ = generation;
    waiting_ = true;
    return true;
}

Serial PickPass::last_use() const {
    const Serial a = ids_target_ ? gfx_.last_use(ids_target_) : 0;
    const Serial b = depth_target_ ? gfx_.last_use(depth_target_) : 0;
    return a > b ? a : b;
}

PickPass::Result PickPass::poll(uint64_t generation, Hit* out) {
    if (!waiting_) return Result::None;
    if (!have_ids_ && gfx_.take_readback(rb_ids_, &ids_)) {
        have_ids_ = true;
        rb_ids_ = {};
    }
    if (want_depth_ && !have_depth_ && gfx_.take_readback(rb_depth_, &depth_)) {
        have_depth_ = true;
        rb_depth_ = {};
    }
    if (!have_ids_ || (want_depth_ && !have_depth_)) return Result::Pending;
    waiting_ = false;
    have_ids_ = have_depth_ = false;
    if (!want_depth_) depth_.clear();
    if (generation != generation_) {   // the view drew another frame: what was under the pointer moved
        ++dropped_;
        picker_.deliver(picker_.current(), generation, ids_, nullptr, map_);   // retires the request
        return Result::Dropped;
    }
    ++answered_;
    const std::optional<Hit> h = picker_.deliver(picker_.current(), generation, ids_, want_depth_ ? &depth_ : nullptr, map_);
    if (!h) return Result::Miss;
    if (out) *out = *h;
    return Result::Hit;
}

// ---- Picks in a drawn frame ------------------------------------------------------------------------

WindowRegion window_region(const PickWindow& w, uint32_t target_width, uint32_t target_height) {
    WindowRegion r;
    if (w.width <= 0 || w.height <= 0) return r;
    const int64_t x0 = std::max<int64_t>(w.x0, 0), y0 = std::max<int64_t>(w.y0, 0);
    const int64_t x1 = std::min<int64_t>(int64_t(w.x0) + w.width, target_width);
    const int64_t y1 = std::min<int64_t>(int64_t(w.y0) + w.height, target_height);
    if (x1 <= x0 || y1 <= y0) return r;
    r.x = uint32_t(x0);
    r.y = uint32_t(y0);
    r.width = uint32_t(x1 - x0);
    r.height = uint32_t(y1 - y0);
    r.i0 = int32_t(x0 - w.x0);
    r.j0 = int32_t(y0 - w.y0);
    return r;
}

std::optional<FrameHit> decode_frame_pick(const std::vector<uint8_t>& ids, const std::vector<uint8_t>* depth,
                                          const PickWindow& w, const WindowRegion& region,
                                          const std::function<bool(uint32_t)>& is_program) {
    const size_t texels = size_t(region.width) * region.height;
    if (texels == 0 || !is_program || ids.size() / 8 < texels) return std::nullopt;
    const bool have_depth = depth && depth->size() / 4 >= texels;
    const double r2 = double(w.radius) * w.radius;
    bool found = false;
    FrameHit best;
    double best_d2 = 0;
    for (uint32_t j = 0; j < region.height; ++j)
        for (uint32_t i = 0; i < region.width; ++i) {
            const size_t k = size_t(j) * region.width + i;
            uint32_t pair[2];
            std::memcpy(pair, ids.data() + k * 8, 8);
            if (pair[0] == 0 || !is_program(pair[0])) continue;
            const int64_t vx = int64_t(region.x) + i, vy = int64_t(region.y) + j;   // the view pixel
            if (vx >= int64_t(w.view_width) || vy >= int64_t(w.view_height)) continue;
            const double dx = double(vx) + 0.5 - w.px, dy = double(vy) + 0.5 - w.py;
            const double d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            float dz = 1.0f;
            if (have_depth) std::memcpy(&dz, depth->data() + k * 4, 4);
            if (!found || d2 < best_d2 || (d2 == best_d2 && dz < best.depth)) {
                found = true;
                best_d2 = d2;
                best.pick_id = pair[0];
                best.item = pair[1];
                best.depth = dz;
                best.texel[0] = region.i0 + int32_t(i);
                best.texel[1] = region.j0 + int32_t(j);
            }
        }
    if (!found) return std::nullopt;
    return best;
}

bool entity_hit_wins(const Hit& entity, const FrameHit& program) { return entity.depth < program.depth; }

FramePick::FramePick(Backend& backend, int radius_px) : gfx_(backend), radius_(radius_px < 1 ? 1 : radius_px) {}

FramePick::~FramePick() { cancel(); }

void FramePick::cancel() {
    if (rb_ids_) gfx_.release(rb_ids_);
    if (rb_depth_) gfx_.release(rb_depth_);
    rb_ids_ = {};
    rb_depth_ = {};
    have_ids_ = have_depth_ = false;
    waiting_ = false;
}

bool FramePick::request(Texture ids, Texture depth, uint32_t width, uint32_t height, float x, float y,
                        std::string* why) {
    cancel();   // a new request supersedes an unanswered one
    if (!ids) {
        if (why) *why = "the frame has no ID target";
        return false;
    }
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 || x >= float(width) || y >= float(height)) {
        if (why) *why = "the pointer is not on the frame";
        return false;
    }
    window_ = pick_window(x, y, radius_, width, height);
    region_ = window_region(window_, width, height);
    if (region_.width == 0 || region_.height == 0) {
        if (why) *why = "the pick window lies outside the frame";
        return false;
    }
    want_depth_ = bool(depth);
    rb_ids_ = gfx_.read_texture(ids, region_.x, region_.y, region_.width, region_.height);
    if (want_depth_) rb_depth_ = gfx_.read_texture(depth, region_.x, region_.y, region_.width, region_.height);
    if (!rb_ids_ || (want_depth_ && !rb_depth_)) {
        if (why) *why = "no pick readback: " + gfx_.last_error();
        cancel();
        return false;
    }
    waiting_ = true;
    return true;
}

PickPass::Result FramePick::poll(const std::function<bool(uint32_t)>& is_program, FrameHit* out) {
    if (!waiting_) return PickPass::Result::None;
    if (!have_ids_ && gfx_.take_readback(rb_ids_, &ids_)) {
        have_ids_ = true;
        rb_ids_ = {};
    }
    if (want_depth_ && !have_depth_ && gfx_.take_readback(rb_depth_, &depth_)) {
        have_depth_ = true;
        rb_depth_ = {};
    }
    if (!have_ids_ || (want_depth_ && !have_depth_)) return PickPass::Result::Pending;
    waiting_ = false;
    have_ids_ = have_depth_ = false;
    const std::optional<FrameHit> h =
        decode_frame_pick(ids_, want_depth_ ? &depth_ : nullptr, window_, region_, is_program);
    if (!h) return PickPass::Result::Miss;
    if (out) *out = *h;
    return PickPass::Result::Hit;
}

}  // namespace caliper_scene
