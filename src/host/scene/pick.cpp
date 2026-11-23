// pick.cpp: the pick window and the pick of a drawn frame (pick.h).
#include "scene/pick.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caliper_scene {
using namespace caliper_gfx;

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
    // A backend that copies a depth texture only whole (WebGPU) reads no region of it: the pick then has
    // the item of the ID target without its depth, which a program's pick does not need.
    want_depth_ = bool(depth) && !gfx_.caps().has(kCapDepthWholeCopy);
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

PickResult FramePick::poll(const std::function<bool(uint32_t)>& is_program, FrameHit* out) {
    if (!waiting_) return PickResult::None;
    if (!have_ids_ && gfx_.take_readback(rb_ids_, &ids_)) {
        have_ids_ = true;
        rb_ids_ = {};
    }
    if (want_depth_ && !have_depth_ && gfx_.take_readback(rb_depth_, &depth_)) {
        have_depth_ = true;
        rb_depth_ = {};
    }
    if (!have_ids_ || (want_depth_ && !have_depth_)) return PickResult::Pending;
    waiting_ = false;
    have_ids_ = have_depth_ = false;
    const std::optional<FrameHit> h =
        decode_frame_pick(ids_, want_depth_ ? &depth_ : nullptr, window_, region_, is_program);
    if (!h) return PickResult::Miss;
    if (out) *out = *h;
    return PickResult::Hit;
}

}  // namespace caliper_scene
