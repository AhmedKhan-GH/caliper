#pragma once
// The offscreen canvas the GPU tests share, and checks of image properties
// that do not depend on exact pixels: colours present in a region, their
// centroid, and lines of text.
#include "spec_runtime/spec_offscreen.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace spec_test {

// One canvas per test process, made on first use; null without a GPU.
inline caliper_host::SpecOffscreen* canvas() {
    static std::unique_ptr<caliper_host::SpecOffscreen> c;
    static bool tried = false;
    if (!tried) {
        tried = true;
        std::string why;
        c = caliper_host::SpecOffscreen::create(640, 400, 1.0f, &why);
        if (!c) MESSAGE("skipped: no offscreen canvas (" << why << ")");
    }
    return c.get();
}

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> px;   // RGBA8, top row first

    uint32_t rgb(int x, int y) const {
        const uint8_t* p = &px[((size_t)y * w + x) * 4];
        return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16;
    }
    // Pixels in [x0, x1) x [y0, y1) that pass `pred`, with their centroid.
    struct Count {
        size_t n = 0;
        double cx = 0, cy = 0;
    };
    Count count(const std::function<bool(uint32_t)>& pred, int x0 = 0, int y0 = 0, int x1 = -1, int y1 = -1) const {
        if (x1 < 0) x1 = w;
        if (y1 < 0) y1 = h;
        Count c;
        for (int y = std::max(0, y0); y < std::min(h, y1); ++y)
            for (int x = std::max(0, x0); x < std::min(w, x1); ++x)
                if (pred(rgb(x, y))) {
                    ++c.n;
                    c.cx += x;
                    c.cy += y;
                }
        if (c.n) {
            c.cx /= (double)c.n;
            c.cy /= (double)c.n;
        }
        return c;
    }
    // Rows of [x0, x1) that hold a pixel passing `pred`, counted as runs: the
    // lines of text in a column of a table. Runs closer than `gap` rows are
    // one line (a glyph's strokes), and a row whose passing pixels cover
    // more than `max_share` of the width is a rule, not text.
    int runs(const std::function<bool(uint32_t)>& pred, int x0, int x1, int y0, int y1, int gap = 4,
             double max_share = 0.6) const {
        int n = 0, last = -1000;
        x0 = std::max(0, x0);
        x1 = std::min(w, x1);
        for (int y = std::max(0, y0); y < std::min(h, y1); ++y) {
            int hits = 0;
            for (int x = x0; x < x1; ++x) hits += pred(rgb(x, y)) ? 1 : 0;
            if (hits == 0 || hits > max_share * (x1 - x0)) continue;
            if (y - last > gap) ++n;
            last = y;
        }
        return n;
    }
};

// A colour within `tol` of `c` (an ImU32 or LUT entry: r | g<<8 | b<<16) on
// every channel.
inline std::function<bool(uint32_t)> near(uint32_t c, int tol = 2) {
    return [c, tol](uint32_t p) {
        for (int s = 0; s < 24; s += 8)
            if (std::abs((int)((p >> s) & 0xff) - (int)((c >> s) & 0xff)) > tol) return false;
        return true;
    };
}

// Brighter than the faint ink: text in parchment or the secondary ink.
inline bool bright(uint32_t p) {
    const int r = p & 0xff, g = (p >> 8) & 0xff, b = (p >> 16) & 0xff;
    return (r * 3 + g * 6 + b) / 10 > 160;
}

// Draws with `draw` into the shared canvas at w x h and reads it back.
inline Image render(int w, int h, const std::function<void(const ImRect&)>& draw, int frames = 3) {
    Image img;
    caliper_host::SpecOffscreen* c = canvas();
    if (!c) return img;
    REQUIRE(c->resize(w, h));
    std::string why;
    REQUIRE_MESSAGE(c->render(frames, draw, &img.px, &why), why);
    img.w = w;
    img.h = h;
    return img;
}

}  // namespace spec_test
