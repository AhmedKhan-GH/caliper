// points_parity.cpp: points@1 against the built-in draw of the points view (points_parity.h).
#include "points_parity.h"

#include "live/live_points.h"
#include "scene/scene.h"   // kPalette, default_point_size

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace caliper_web {

namespace {

namespace gfx = caliper_gfx;
namespace lv = caliper_live_views;

constexpr uint32_t kW = 320, kH = 240;
constexpr uint32_t kGround = 0xff201814u;

// One draw of a case: its scene, its readback and what came back.
struct Side {
    std::unique_ptr<lv::PointsScene> scene;
    gfx::Readback rb;
    std::vector<uint8_t> pixels;
    bool done = false;
    bool pieces = false;
    std::string fallback;
};

struct Case {
    std::string name;
    std::vector<float> pos, aux, sel_pos, sel_aux;
    lv::PointsLook look;
    gfx::Buffer bufs[4];
    Side side[2];   // 0: the built-in, 1: points@1
};

struct Run {
    std::vector<std::unique_ptr<Case>> cases;
    gfx::Backend* backend = nullptr;
    bool running = false;
};

Run& run() {
    static Run r;
    return r;
}

// A fixed generator in [-1, 1).
struct Lcg {
    uint32_t s;
    float next() {
        s = s * 1664525u + 1013904223u;
        return float(double(s >> 8) / double(1u << 24)) * 2.0f - 1.0f;
    }
};

gfx::Buffer upload(gfx::Backend& b, const std::vector<float>& v) {
    gfx::BufferDesc d;
    d.size = std::max<uint64_t>(v.size() * 4, 16);
    d.usage = gfx::kVertex | gfx::kCopyDst;
    d.label = "points parity";
    const gfx::Buffer buf = b.create_buffer(d);
    if (buf && !v.empty()) b.write_buffer(buf, 0, v.data(), v.size() * 4);
    return buf;
}

void bound(Case& c, int dims) {
    float bb[6] = {INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY};
    for (size_t i = 0; i + 2 < c.pos.size(); i += 3)
        for (int k = 0; k < 3; ++k)
            if (std::isfinite(c.pos[i + k])) {
                bb[2 * k] = std::min(bb[2 * k], c.pos[i + k]);
                bb[2 * k + 1] = std::max(bb[2 * k + 1], c.pos[i + k]);
            }
    if (dims == 2) bb[4] = bb[5] = 0.0f;
    std::memcpy(c.look.bbox, bb, sizeof bb);
    c.look.has_bbox = true;
    c.look.dims = dims;
    c.look.size_px = caliper_scene::default_point_size(c.pos.size() / 3);
    c.look.ground = kGround;
}

std::vector<std::unique_ptr<Case>> make_cases() {
    std::vector<std::unique_ptr<Case>> out;
    {   // A 2-D lattice of 20 classes, with labels above 255, below 0, at 255, at 256 and NaN.
        auto c = std::make_unique<Case>();
        c->name = "2d-classes";
        for (int i = 0; i < 400; ++i) {
            c->pos.insert(c->pos.end(), {float(i % 20) / 9.5f - 1.0f, float(i / 20) / 9.5f - 1.0f, 0.0f});
            float label = float(i % 20);
            if (i % 61 == 7) label = NAN;
            else if (i % 53 == 3) label = 300.0f;
            else if (i % 59 == 4) label = -3.0f;
            else if (i % 67 == 5) label = 255.0f;
            else if (i % 71 == 6) label = 256.0f;
            c->aux.push_back(label);
        }
        c->look.colour = lv::PointsColour::Classes;
        bound(*c, 2);
        out.push_back(std::move(c));
    }
    {   // 3-D values through viridis with NaN values, under the depth cues.
        auto c = std::make_unique<Case>();
        c->name = "3d-values-cues";
        Lcg r{11u};
        for (int i = 0; i < 3000; ++i) {
            const float x = r.next(), y = r.next(), z = r.next();
            c->pos.insert(c->pos.end(), {x, y * 0.8f, z * 0.6f});
            c->aux.push_back(i % 37 == 0 ? NAN : 0.5f + 0.5f * r.next());
        }
        c->look.colour = lv::PointsColour::Values;
        c->look.colormap = 0;
        c->look.vmin = 0.0f;
        c->look.vmax = 1.0f;
        bound(*c, 3);
        out.push_back(std::move(c));
    }
    {   // A 3-D selection of every third point over the dimmed rest, by class.
        auto c = std::make_unique<Case>();
        c->name = "3d-selection";
        Lcg r{23u};
        for (int i = 0; i < 3000; ++i) {
            c->pos.insert(c->pos.end(), {r.next(), r.next(), r.next()});
            c->aux.push_back(float(i % 10));
        }
        for (int i = 0; i < 3000; i += 3) {
            c->sel_pos.insert(c->sel_pos.end(), c->pos.begin() + 3 * i, c->pos.begin() + 3 * i + 3);
            c->sel_aux.push_back(c->aux[size_t(i)]);
        }
        c->look.colour = lv::PointsColour::Classes;
        bound(*c, 3);
        out.push_back(std::move(c));
    }
    {   // 20,000 flat 2-D points.
        auto c = std::make_unique<Case>();
        c->name = "2d-flat-20000";
        Lcg r{37u};
        for (int i = 0; i < 20000; ++i) c->pos.insert(c->pos.end(), {r.next(), r.next(), 0.0f});
        bound(*c, 2);
        out.push_back(std::move(c));
    }
    return out;
}

lv::PointsArrays arrays(const std::vector<float>& pos, const std::vector<float>& aux, gfx::Buffer pb, gfx::Buffer ab) {
    lv::PointsArrays a;
    a.pos = pb;
    a.pos_cpu = pos.data();
    if (!aux.empty()) {
        a.aux = ab;
        a.aux_cpu = aux.data();
    }
    a.count = pos.size() / 3;
    return a;
}

int gap(uint32_t a, uint32_t b) {
    int d = 0;
    for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return d;
}

// The piece's picture against the built-in's under the match tests' tolerances.
std::string compare(const Case& c) {
    const std::vector<uint8_t>& b = c.side[0].pixels;
    const std::vector<uint8_t>& p = c.side[1].pixels;
    if (b.size() != size_t(kW) * kH * 4 || p.size() != b.size()) return "no picture";
    auto at = [&](const std::vector<uint8_t>& px, int x, int y) {
        uint32_t v;
        std::memcpy(&v, &px[(size_t(std::clamp(y, 0, int(kH) - 1)) * kW + size_t(std::clamp(x, 0, int(kW) - 1))) * 4], 4);
        return v;
    };
    size_t differing = 0, failed = 0, lit = 0;
    int largest = 0;
    char first[96] = "";
    for (int y = 0; y < int(kH); ++y)
        for (int x = 0; x < int(kW); ++x) {
            const uint32_t bc = at(b, x, y), pc = at(p, x, y);
            differing += bc != pc;
            lit += (bc & 0x00ffffffu) != (kGround & 0x00ffffffu);
            largest = std::max(largest, gap(bc, pc));
            bool uniform = true;
            for (int dy = -1; dy <= 1 && uniform; ++dy)
                for (int dx = -1; dx <= 1 && uniform; ++dx) uniform = at(b, x + dx, y + dy) == bc;
            bool ok = uniform ? pc == bc : false;
            for (int dy = -1; dy <= 1 && !ok && !uniform; ++dy)
                for (int dx = -1; dx <= 1 && !ok; ++dx) ok = gap(pc, at(b, x + dx, y + dy)) <= 1;
            if (!ok && failed++ == 0)
                std::snprintf(first, sizeof first, " first=(%d,%d) piece=%08x builtin=%08x", x, y, pc, bc);
        }
    char t[320];
    std::snprintf(t, sizeof t, "%s pieces=%d differing=%zu lit=%zu largest=%d failed=%zu%s%s%s", c.name.c_str(),
                  c.side[1].pieces ? 1 : 0, differing, lit, largest, failed, first,
                  c.side[1].pieces ? "" : " fallback=", c.side[1].pieces ? "" : c.side[1].fallback.c_str());
    return t;
}

}  // namespace

bool points_parity_start(gfx::Backend& b, const gfx::ProgramDesc* programs, std::string* why) {
    points_parity_stop();
    Run& r = run();
    r.backend = &b;
    r.cases = make_cases();
    for (auto& cp : r.cases) {
        Case& c = *cp;
        c.bufs[0] = upload(b, c.pos);
        c.bufs[1] = upload(b, c.aux);
        c.bufs[2] = upload(b, c.sel_pos);
        c.bufs[3] = upload(b, c.sel_aux);
        const lv::PointsArrays all = arrays(c.pos, c.aux, c.bufs[0], c.bufs[1]);
        const lv::PointsArrays sel = arrays(c.sel_pos, c.sel_aux, c.bufs[2], c.bufs[3]);
        for (int k = 0; k < 2; ++k) {
            Side& s = c.side[k];
            s.scene = std::make_unique<lv::PointsScene>(b, programs);
            if (!s.scene->init(why)) return false;
            s.scene->set_draw(k == 0 ? lv::PointsDraw::builtin : lv::PointsDraw::pieces);
            if (!s.scene->set(all, c.sel_pos.empty() ? nullptr : &sel, c.look, why)) return false;
            s.scene->fit(double(kW) / double(kH));
            const uint32_t clear = c.look.dims == 3 ? (kGround & 0x00ffffffu) : kGround;
            gfx::Serial serial = 0;
            if (!s.scene->render(kW, kH, 1.0f, clear, &serial, why)) return false;
            s.pieces = s.scene->drew_pieces();
            s.fallback = s.scene->fallback();
            s.rb = b.read_texture(s.scene->color(), 0, 0, kW, kH);
            if (!s.rb) {
                if (why) *why = "no readback: " + b.last_error();
                return false;
            }
        }
    }
    r.running = true;
    return true;
}

bool points_parity_poll(gfx::Backend& b, std::string* report) {
    Run& r = run();
    if (!r.running) return false;
    bool all = true;
    for (auto& cp : r.cases)
        for (Side& s : cp->side) {
            if (s.done) continue;
            if (b.take_readback(s.rb, &s.pixels)) s.done = true;
            else all = false;
        }
    if (!all) return false;
    bool ok = true;
    std::string text;
    for (auto& cp : r.cases) {
        const std::string clause = compare(*cp);
        ok = ok && cp->side[1].pieces && !cp->side[0].pieces && clause.find(" failed=0") != std::string::npos;
        text += "; " + clause;
    }
    *report = std::string(ok ? "ok" : "fail") + " points@1 against the built-in in " + std::to_string(r.cases.size()) +
              " cases" + text;
    points_parity_stop();
    return true;
}

bool points_parity_running() { return run().running; }

void points_parity_stop() {
    Run& r = run();
    for (auto& cp : r.cases) {
        for (Side& s : cp->side) {
            if (!s.done && s.rb && r.backend) r.backend->release(s.rb);
            s.scene.reset();
        }
        for (gfx::Buffer buf : cp->bufs)
            if (buf && r.backend) r.backend->release(buf);
    }
    r.cases.clear();
    r.running = false;
}

}  // namespace caliper_web
