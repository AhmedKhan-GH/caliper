// spec_layout.cpp: the layout of a spec's panels as plain rectangles
// (spec_layout.h).
#include "spec_runtime/spec_layout.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace caliper_host {

namespace spec = caliper_builder::spec;

namespace {

// Whole-pixel edges of `r`.
SpecRect snap(const SpecRect& r) {
    const float x0 = std::round(r.x), y0 = std::round(r.y);
    const float x1 = std::round(r.x + r.w), y1 = std::round(r.y + r.h);
    return {x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0)};
}

const spec::Panel* find_panel(const spec::Spec& s, const std::string& id) {
    for (const auto& p : s.panels)
        if (p.id == id) return &p;
    return nullptr;
}

// The first panel id in a node, in tree order, or "".
std::string first_panel(const spec::LayoutNode& n) {
    if (n.kind == spec::LayoutNode::Kind::panel) return n.panel;
    for (const auto& c : n.children) {
        std::string id = first_panel(c);
        if (!id.empty()) return id;
    }
    return {};
}

std::string tab_label(const spec::LayoutNode& n, const spec::Spec& s) {
    if (n.kind != spec::LayoutNode::Kind::panel && n.title && !n.title->empty()) return *n.title;
    const std::string id = first_panel(n);
    if (const spec::Panel* p = find_panel(s, id)) return spec_panel_title(*p);
    return id;
}

std::string child_path(const std::string& path, size_t i) {
    return path.empty() ? std::to_string(i) : path + "." + std::to_string(i);
}

struct Placer {
    const spec::Spec& spec;
    const SpecLayoutMetrics& m;
    const SpecTabChoice& choice;
    SpecLayout out;

    void place(const spec::LayoutNode& n, const SpecRect& r, const std::string& path, int depth) {
        if (depth > 64) return;   // the format allows 8; a guard, not a rule
        switch (n.kind) {
            case spec::LayoutNode::Kind::panel:
                out.panels.push_back({n.panel, path, r});
                return;
            case spec::LayoutNode::Kind::tabs: {
                if (n.children.empty()) return;
                SpecLayoutTabs t;
                t.path = path;
                const float bar_h = std::min(r.h, std::round(m.tab_bar_h));
                t.bar = {r.x, r.y, r.w, bar_h};
                for (const auto& c : n.children) t.labels.push_back(tab_label(c, spec));
                const auto it = choice.find(path);
                t.active = it != choice.end() && it->second >= 0 && it->second < (int)n.children.size() ? it->second : 0;
                const int active = t.active;
                out.tabs.push_back(std::move(t));
                const SpecRect body{r.x, r.y + bar_h, r.w, std::max(0.0f, r.h - bar_h)};
                place(n.children[(size_t)active], body, child_path(path, (size_t)active), depth + 1);
                return;
            }
            case spec::LayoutNode::Kind::split: {
                const size_t k = n.children.size();
                if (k == 0) return;
                const bool across = n.split == spec::SplitDirection::horizontal;
                const float total = across ? r.w : r.h;
                const float gaps = m.gap * (float)(k - 1);
                const float length = std::max(0.0f, total - gaps);
                std::vector<float> mins;
                for (const auto& c : n.children) {
                    const SpecRect ms = spec_layout_min_size(c, m);
                    mins.push_back(across ? ms.w : ms.h);
                }
                const std::vector<float> shares = spec_layout_shares(length, mins, n.ratio.value_or(0.5));
                // With no room for the gaps, the children overlap rather than
                // leave the area.
                const float gap = total >= gaps ? m.gap : 0.0f;
                float at = across ? r.x : r.y;
                for (size_t i = 0; i < k; ++i) {
                    const SpecRect cr = across ? SpecRect{at, r.y, shares[i], r.h} : SpecRect{r.x, at, r.w, shares[i]};
                    place(n.children[i], cr, child_path(path, i), depth + 1);
                    at += shares[i] + gap;
                }
                return;
            }
        }
    }
};

}  // namespace

SpecLayoutMetrics SpecLayoutMetrics::for_text_scale(float text_scale) {
    const float s = text_scale > 0.0f ? text_scale : 1.0f;
    SpecLayoutMetrics m;
    const float line = 15.0f * s;   // the UI face's size at this scale
    m.min_panel_w = std::round(12.0f * line);
    m.min_panel_h = std::round(8.0f * line);
    m.tab_bar_h = std::round(line + 9.0f * s);
    m.gap = 1.0f;
    return m;
}

const SpecLayoutPanel* SpecLayout::find(const std::string& id) const {
    for (const auto& p : panels)
        if (p.id == id) return &p;
    return nullptr;
}

std::string spec_panel_title(const spec::Panel& panel) {
    return panel.title && !panel.title->empty() ? *panel.title : panel.id;
}

SpecRect spec_layout_min_size(const spec::LayoutNode& n, const SpecLayoutMetrics& m) {
    switch (n.kind) {
        case spec::LayoutNode::Kind::panel:
            return {0, 0, m.min_panel_w, m.min_panel_h};
        case spec::LayoutNode::Kind::tabs: {
            float w = 0, h = 0;
            for (const auto& c : n.children) {
                const SpecRect r = spec_layout_min_size(c, m);
                w = std::max(w, r.w);
                h = std::max(h, r.h);
            }
            return {0, 0, w, h + m.tab_bar_h};
        }
        case spec::LayoutNode::Kind::split: {
            const bool across = n.split == spec::SplitDirection::horizontal;
            float along = 0, cross = 0;
            for (const auto& c : n.children) {
                const SpecRect r = spec_layout_min_size(c, m);
                along += across ? r.w : r.h;
                cross = std::max(cross, across ? r.h : r.w);
            }
            if (!n.children.empty()) along += m.gap * (float)(n.children.size() - 1);
            return across ? SpecRect{0, 0, along, cross} : SpecRect{0, 0, cross, along};
        }
    }
    return {};
}

std::vector<float> spec_layout_shares(float length, const std::vector<float>& mins, double ratio) {
    const size_t k = mins.size();
    std::vector<double> share(k, 0.0);
    if (k == 0) return {};
    const double L = std::max(0.0f, std::round(length));
    const double need = std::accumulate(mins.begin(), mins.end(), 0.0);
    if (need >= L) {
        // No room for every minimum: in proportion to them.
        for (size_t i = 0; i < k; ++i) share[i] = need > 0.0 ? L * mins[i] / need : L / (double)k;
    } else if (k == 2) {
        const double r = std::clamp(ratio, 0.0, 1.0);
        share[0] = std::clamp(r * L, (double)mins[0], L - (double)mins[1]);
        share[1] = L - share[0];
    } else {
        // Equal shares, raising each child below its minimum to it and
        // sharing what is left among the others.
        std::vector<bool> fixed(k, false);
        for (bool changed = true; changed;) {
            changed = false;
            double left = L;
            size_t free = 0;
            for (size_t i = 0; i < k; ++i) {
                if (fixed[i]) left -= mins[i];
                else ++free;
            }
            const double each = free ? left / (double)free : 0.0;
            for (size_t i = 0; i < k; ++i) {
                if (fixed[i]) {
                    share[i] = mins[i];
                } else if ((double)mins[i] > each) {
                    fixed[i] = true;
                    changed = true;
                } else {
                    share[i] = each;
                }
            }
        }
    }
    // Whole pixels: round the running edges, so the shares sum to the length.
    std::vector<float> out(k);
    double edge = 0.0;
    float prev = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        edge += share[i];
        const float e = i + 1 == k ? (float)L : (float)std::round(edge);
        out[i] = std::max(0.0f, e - prev);
        prev = e;
    }
    return out;
}

SpecLayout compute_spec_layout(const spec::LayoutNode& root, const spec::Spec& spec, const SpecRect& area,
                               const SpecLayoutMetrics& m, const SpecTabChoice& tabs) {
    Placer p{spec, m, tabs, {}};
    p.place(root, snap(area), "", 0);
    return std::move(p.out);
}

}  // namespace caliper_host
