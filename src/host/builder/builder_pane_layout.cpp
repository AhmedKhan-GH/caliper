// builder_pane_layout.cpp: the builder pane's regions and the canvas's drop
// targets as plain rectangles (builder_pane_layout.h).
#include "builder/builder_pane_layout.h"

#include "builder/builder_pane_edits.h"

#include <algorithm>
#include <cmath>

namespace caliper_host {

BuilderPaneMetrics BuilderPaneMetrics::for_text_scale(float text_scale) {
    const float s = text_scale > 0.0f ? text_scale : 1.0f;
    BuilderPaneMetrics m;
    m.line = 15.0f * s;
    m.rule = 1.0f;
    m.left_min = std::round(14.0f * m.line);
    m.right_min = std::round(18.0f * m.line);
    m.canvas_min_w = std::round(24.0f * m.line);
    m.canvas_min_h = std::round(16.0f * m.line);
    m.strip_min_h = std::round(9.0f * m.line);
    m.header = std::round(24.0f * s);
    m.row = std::round(22.0f * s);
    m.pad = std::round(8.0f * s);
    m.primitives_h = m.header + 7.0f * m.row + 2.0f * m.pad;
    m.sources_min_h = m.header + 5.0f * m.row + m.pad;
    m.properties_min_h = m.header + 6.0f * m.row;
    m.edge_band = std::round(6.0f * m.line);
    m.root_band = std::round(10.0f * s);
    m.slot_row = std::round(30.0f * s);
    m.slot_w = std::round(220.0f * s);
    m.divider_slop = std::max(3.0f, std::round(4.0f * s));
    return m;
}

namespace {

SpecRect snap(const SpecRect& r) {
    const float x0 = std::round(r.x), y0 = std::round(r.y);
    const float x1 = std::round(r.x + r.w), y1 = std::round(r.y + r.h);
    return {x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0)};
}

// The widths of the shown columns and the canvas: each column its share,
// at least its minimum, then the right and the left given back down to
// their minimums while the canvas is below its own. False when the canvas
// still does not fit.
bool fit_columns(float W, bool left, bool right, const BuilderPaneMetrics& m, float* lw, float* rw, float* cw) {
    float l = left ? std::max(m.left_min, std::round(W * m.left_share)) : 0.0f;
    float r = right ? std::max(m.right_min, std::round(W * m.right_share)) : 0.0f;
    const float rules = (left ? m.rule : 0.0f) + (right ? m.rule : 0.0f);
    float c = W - l - r - rules;
    float deficit = m.canvas_min_w - c;
    if (deficit > 0.0f && right) {
        const float take = std::min(deficit, r - m.right_min);
        r -= take;
        deficit -= take;
    }
    if (deficit > 0.0f && left) {
        const float take = std::min(deficit, l - m.left_min);
        l -= take;
        deficit -= take;
    }
    c = W - l - r - rules;
    *lw = l;
    *rw = r;
    *cw = c;
    return c >= m.canvas_min_w;
}

}  // namespace

BuilderPaneLayout compute_builder_pane_layout(const BuilderPaneLayoutInput& in, const BuilderPaneMetrics& m) {
    BuilderPaneLayout out;
    const SpecRect a = snap(in.area);
    if (!in.build) {
        out.canvas = a;
        return out;
    }
    // The columns: both, then without the right, then without the left.
    float lw = 0, rw = 0, cw = 0;
    bool left = true, right = true;
    if (!fit_columns(a.w, true, true, m, &lw, &rw, &cw)) {
        right = false;
        if (!fit_columns(a.w, true, false, m, &lw, &rw, &cw)) {
            left = false;
            fit_columns(a.w, false, false, m, &lw, &rw, &cw);
        }
    }
    // The centre: the canvas over the text strip.
    float strip_h = 0.0f;
    bool strip = in.text_open;
    if (strip) {
        strip_h = std::max(m.strip_min_h, std::round(a.h * m.strip_share));
        if (a.h - strip_h - m.rule < m.canvas_min_h) strip_h = std::max(m.strip_min_h, a.h - m.rule - m.canvas_min_h);
        if (a.h - strip_h - m.rule < m.canvas_min_h) {
            strip = false;
            strip_h = 0.0f;
            out.collapsed.push_back("strip");
        }
    }
    if (!right) out.collapsed.push_back("right");
    if (!left) out.collapsed.push_back("left");

    float x = a.x;
    if (left) {
        out.left = true;
        out.left_column = {x, a.y, lw, a.h};
        // Primitives keep their seven rows; Sources take the rest, and give
        // the Primitives rows back only below their own least.
        float prim = std::min(m.primitives_h, std::max(m.header + 2.0f * m.row, a.h - m.rule - m.sources_min_h));
        prim = std::max(0.0f, std::min(prim, a.h - m.rule));
        const float src = std::max(0.0f, a.h - prim - m.rule);
        out.sources = {x, a.y, lw, src};
        out.rules.push_back({x, a.y + src, lw, m.rule});
        out.primitives = {x, a.y + src + m.rule, lw, prim};
        x += lw;
        out.rules.push_back({x, a.y, m.rule, a.h});
        x += m.rule;
    }
    out.canvas = {x, a.y, cw, a.h};
    if (strip) {
        out.strip = true;
        const float canvas_h = a.h - strip_h - m.rule;
        out.canvas.h = canvas_h;
        out.rules.push_back({x, a.y + canvas_h, cw, m.rule});
        out.strip_rect = {x, a.y + canvas_h + m.rule, cw, strip_h};
    }
    x += cw;
    if (right) {
        out.right = true;
        out.rules.push_back({x, a.y, m.rule, a.h});
        x += m.rule;
        out.right_column = {x, a.y, rw, a.h};
        out.properties = out.right_column;
        if (in.chat) {
            float chat_h = std::round(a.h * m.chat_share);
            chat_h = std::min(chat_h, std::max(0.0f, a.h - m.rule - m.properties_min_h));
            if (chat_h > 0.0f) {
                out.chat = true;
                out.properties.h = a.h - chat_h - m.rule;
                out.rules.push_back({x, a.y + out.properties.h, rw, m.rule});
                out.chat_rect = {x, a.y + out.properties.h + m.rule, rw, chat_h};
            }
        }
    }
    return out;
}

// ---- drop targets -----------------------------------------------------------------------

const char* drop_zone_name(DropZone z) {
    switch (z) {
        case DropZone::none: return "none";
        case DropZone::empty: return "empty";
        case DropZone::root: return "root";
        case DropZone::edge: return "edge";
        case DropZone::centre: return "centre";
        case DropZone::slot: return "slot";
    }
    return "none";
}

bool spec_rect_contains(const SpecRect& r, float x, float y) {
    return x >= r.x && y >= r.y && x < r.right() && y < r.bottom();
}

std::vector<SpecRect> drop_slot_rows(const SpecRect& p, float strip, int slots, const BuilderPaneMetrics& m) {
    std::vector<SpecRect> out;
    if (slots <= 0) return out;
    const float gap = std::round(m.pad * 0.5f);
    const float w = std::min(m.slot_w, p.w - 2.0f * m.pad);
    const float top = p.y + strip + m.pad, bottom = p.bottom() - m.pad;
    if (w <= 0.0f || bottom <= top) return out;
    const int fit = std::max(0, (int)((bottom - top + gap) / (m.slot_row + gap)));
    const int n = std::min(slots, fit);
    if (n == 0) return out;
    const float total = (float)n * m.slot_row + (float)(n - 1) * gap;
    const float x = std::round(p.x + (p.w - w) * 0.5f);
    float y = std::round(top + (bottom - top - total) * 0.5f);
    for (int i = 0; i < n; ++i) {
        out.push_back({x, y, std::round(w), m.slot_row});
        y += m.slot_row + gap;
    }
    return out;
}

DropHit hit_drop(const SpecLayout& layout, const SpecRect& canvas, float x, float y, int slots, float strip,
                 const BuilderPaneMetrics& m) {
    DropHit hit;
    if (!spec_rect_contains(canvas, x, y)) return hit;
    if (layout.panels.empty()) {
        hit.zone = DropZone::empty;
        return hit;
    }
    // The band along the canvas's border: the whole layout's edges.
    {
        const float dl = x - canvas.x, dr = canvas.right() - x, dt = y - canvas.y, db = canvas.bottom() - y;
        const float d = std::min({dl, dr, dt, db});
        if (d < m.root_band) {
            hit.zone = DropZone::root;
            hit.where = d == dl ? "left" : d == dr ? "beside" : d == dt ? "above" : "below";
            return hit;
        }
    }
    for (const auto& p : layout.panels) {
        if (!spec_rect_contains(p.rect, x, y)) continue;
        hit.panel = p.id;
        // A source's slots in the panel's centre.
        const std::vector<SpecRect> rows = drop_slot_rows(p.rect, p.tab ? 0.0f : strip, slots, m);
        for (size_t i = 0; i < rows.size(); ++i)
            if (spec_rect_contains(rows[i], x, y)) {
                hit.zone = DropZone::slot;
                hit.slot = (int)i;
                return hit;
            }
        const float bx = std::max(1.0f, std::min(m.edge_band, 0.25f * p.rect.w));
        const float by = std::max(1.0f, std::min(m.edge_band, 0.25f * p.rect.h));
        const float dl = (x - p.rect.x) / bx, dr = (p.rect.right() - x) / bx;
        const float dt = (y - p.rect.y) / by, db = (p.rect.bottom() - y) / by;
        const float d = std::min({dl, dr, dt, db});
        if (d < 1.0f) {
            hit.zone = DropZone::edge;
            hit.where = d == dl ? "left" : d == dr ? "beside" : d == dt ? "above" : "below";
        } else {
            hit.zone = DropZone::centre;
            hit.where = "tabbed";
        }
        return hit;
    }
    return hit;   // on a rule between panels
}

const SpecLayoutSplit* hit_divider(const SpecLayout& layout, float x, float y, const BuilderPaneMetrics& m) {
    for (auto it = layout.splits.rbegin(); it != layout.splits.rend(); ++it) {
        const SpecLayoutSplit& s = *it;
        if (s.children != 2) continue;
        SpecRect grip = s.divider;
        if (s.horizontal) {
            grip.x -= m.divider_slop;
            grip.w += 2.0f * m.divider_slop;
        } else {
            grip.y -= m.divider_slop;
            grip.h += 2.0f * m.divider_slop;
        }
        if (spec_rect_contains(grip, x, y)) return &s;
    }
    return nullptr;
}

double divider_ratio(const SpecLayoutSplit& s, float x, float y) {
    const float along = s.horizontal ? x - s.rect.x : y - s.rect.y;
    if (s.length <= 0.0f) return 0.5;
    return pane_snap_ratio((double)along / (double)s.length);
}

std::string hit_header(const SpecLayout& layout, float x, float y, float strip) {
    if (strip <= 0.0f) return {};
    for (const auto& p : layout.panels) {
        if (p.tab) continue;
        const SpecRect h{p.rect.x, p.rect.y, p.rect.w, std::min(strip, p.rect.h)};
        if (spec_rect_contains(h, x, y)) return p.id;
    }
    return {};
}

std::string hit_panel(const SpecLayout& layout, float x, float y) {
    for (const auto& p : layout.panels)
        if (spec_rect_contains(p.rect, x, y)) return p.id;
    return {};
}

}  // namespace caliper_host
