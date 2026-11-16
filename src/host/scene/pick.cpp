// pick.cpp: the pick window and the decoding of its readback (pick.h).
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

}  // namespace caliper_scene
