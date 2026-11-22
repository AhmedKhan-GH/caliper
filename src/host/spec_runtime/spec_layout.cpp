// spec_layout.cpp: the layout of a spec's panels as plain rectangles
// (spec_layout.h).
#include "spec_runtime/spec_layout.h"

#include <algorithm>
#include <cctype>
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
    if (const spec::Panel* p = find_panel(s, id)) return spec_panel_title(s, *p);
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

    void place(const spec::LayoutNode& n, const SpecRect& r, const std::string& path, int depth,
               bool tab_child = false) {
        if (depth > 64) return;   // the format allows 8; a guard, not a rule
        switch (n.kind) {
            case spec::LayoutNode::Kind::panel:
                out.panels.push_back(
                    {n.panel, path, r, tab_child, !spec.find_panel(n.panel) && spec.find_control(n.panel)});
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
                place(n.children[(size_t)active], body, child_path(path, (size_t)active), depth + 1, true);
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
                    const SpecRect ms = spec_layout_min_size(c, m, &spec);
                    mins.push_back(across ? ms.w : ms.h);
                }
                std::vector<float> shares;
                // In a vertical split, a row of controls takes its own height
                // and the other children share the rest.
                std::vector<std::optional<float>> fixed(k);
                size_t n_fixed = 0;
                if (!across)
                    for (size_t i = 0; i < k; ++i)
                        if ((fixed[i] = spec_layout_fixed_height(n.children[i], spec, m))) ++n_fixed;
                if (n_fixed == 0) {
                    shares = spec_layout_shares(length, mins, n.ratio.value_or(0.5), n.shares);
                } else {
                    float left = length;
                    shares.assign(k, 0.0f);
                    for (size_t i = 0; i < k; ++i)
                        if (fixed[i]) {
                            shares[i] = std::min(std::round(*fixed[i]), left);
                            left -= shares[i];
                        }
                    std::vector<float> flex_mins;
                    std::vector<double> flex_weights;
                    for (size_t i = 0; i < k; ++i)
                        if (!fixed[i]) {
                            flex_mins.push_back(mins[i]);
                            if (n.shares.size() == k) flex_weights.push_back(n.shares[i]);
                        }
                    if (!flex_mins.empty()) {
                        const std::vector<float> flex = spec_layout_shares(
                            left, flex_mins, flex_mins.size() == 2 ? n.ratio.value_or(0.5) : 0.5, flex_weights);
                        for (size_t i = 0, j = 0; i < k; ++i)
                            if (!fixed[i]) shares[i] = flex[j++];
                    }
                }
                // With no room for the gaps, the children overlap rather than
                // leave the area.
                const float gap = total >= gaps ? m.gap : 0.0f;
                SpecLayoutSplit split;
                split.path = path;
                split.rect = r;
                split.horizontal = across;
                split.children = k;
                split.length = length;
                split.ratio = n.ratio.value_or(0.5);
                {
                    const float first = (across ? r.x : r.y) + shares[0];
                    split.divider = across ? SpecRect{first, r.y, gap, r.h} : SpecRect{r.x, first, r.w, gap};
                }
                out.splits.push_back(std::move(split));
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
    m.min_control_w = std::round(8.0f * line);
    m.control_row_h = std::round(2.0f * line);
    m.char_w = std::round(0.55f * line);
    m.line_h = std::round(1.2f * line);
    return m;
}

const SpecLayoutPanel* SpecLayout::find(const std::string& id) const {
    for (const auto& p : panels)
        if (p.id == id) return &p;
    return nullptr;
}

std::string spec_binding_name(const spec::Spec& s, const std::string& binding) {
    const size_t dot = binding.find('.');
    if (dot != std::string::npos) return binding.substr(dot + 1);
    for (const auto& src : s.sources) {
        if (src.id != binding) continue;
        if (src.catalog) {
            const size_t colon = src.catalog->find(':');
            return colon == std::string::npos ? *src.catalog : src.catalog->substr(colon + 1);
        }
        if (src.sql) {
            // A query goes by the first table it reads, its FROM's name.
            const std::string& q = *src.sql;
            for (size_t i = 0; i + 4 <= q.size(); ++i) {
                const bool word = (i == 0 || !std::isalnum((unsigned char)q[i - 1])) &&
                                  std::toupper((unsigned char)q[i]) == 'F' && std::toupper((unsigned char)q[i + 1]) == 'R' &&
                                  std::toupper((unsigned char)q[i + 2]) == 'O' && std::toupper((unsigned char)q[i + 3]) == 'M' &&
                                  i + 4 < q.size() && std::isspace((unsigned char)q[i + 4]);
                if (!word) continue;
                size_t a = i + 4;
                while (a < q.size() && std::isspace((unsigned char)q[a])) ++a;
                size_t b = a;
                while (b < q.size() && (std::isalnum((unsigned char)q[b]) || q[b] == '_')) ++b;
                if (b > a) return q.substr(a, b - a);
                break;
            }
        }
        return src.id;   // a derived source goes by its own id
    }
    return binding;
}

std::string spec_panel_title(const spec::Spec& s, const spec::Panel& panel) {
    if (panel.title && !panel.title->empty()) return *panel.title;
    auto bound = [&](const char* slot) -> const spec::SlotValue* {
        const auto it = panel.bind.find(slot);
        return it == panel.bind.end() || it->second.bindings.empty() ? nullptr : &it->second;
    };
    auto name = [&](const char* slot) { return spec_binding_name(s, bound(slot)->bindings.front()); };
    switch (panel.primitive) {
        case spec::Primitive::series:
            if (const spec::SlotValue* y = bound("y")) {
                std::string t;
                for (const auto& b : y->bindings) t += (t.empty() ? "" : ", ") + spec_binding_name(s, b);
                return t;
            }
            break;
        case spec::Primitive::points:
            if (bound("positions")) return name("positions");
            if (bound("x") && bound("y")) return name("y") + " by " + name("x");
            break;
        case spec::Primitive::heatmap:
        case spec::Primitive::histogram:
        case spec::Primitive::image:
            if (bound("values")) return name("values");
            break;
        case spec::Primitive::table:
            if (bound("rows")) return name("rows");
            break;
        case spec::Primitive::text:
            if (bound("value")) return name("value");
            return "Text";
        case spec::Primitive::scene:
            // The live scene's name, as catalog.list names it.
            if (bound(spec::kSceneSlot))
                if (auto scene = spec::live_scene_name(bound(spec::kSceneSlot)->bindings.front())) return *scene;
            break;
    }
    std::string t = spec::to_string(panel.primitive);
    if (!t.empty()) t[0] = (char)(t[0] - 'a' + 'A');
    return t;
}

namespace {

float control_height(const spec::Control& c, const SpecLayoutMetrics& m) {
    return m.control_height ? std::max(1.0f, m.control_height(c)) : m.control_row_h;
}

}  // namespace

std::optional<float> spec_layout_fixed_height(const spec::LayoutNode& n, const spec::Spec& spec,
                                              const SpecLayoutMetrics& m) {
    switch (n.kind) {
        case spec::LayoutNode::Kind::panel: {
            const spec::Control* c = spec.find_panel(n.panel) ? nullptr : spec.find_control(n.panel);
            if (!c) return std::nullopt;
            return control_height(*c, m);
        }
        case spec::LayoutNode::Kind::tabs: return std::nullopt;
        case spec::LayoutNode::Kind::split: {
            if (n.children.empty()) return std::nullopt;
            const bool across = n.split == spec::SplitDirection::horizontal;
            float h = 0.0f;
            for (const auto& c : n.children) {
                const std::optional<float> ch = spec_layout_fixed_height(c, spec, m);
                if (!ch) return std::nullopt;
                h = across ? std::max(h, *ch) : h + *ch;
            }
            if (!across) h += m.gap * (float)(n.children.size() - 1);
            return h;
        }
    }
    return std::nullopt;
}

SpecRect spec_layout_min_size(const spec::LayoutNode& n, const SpecLayoutMetrics& m) {
    return spec_layout_min_size(n, m, nullptr);
}

SpecRect spec_layout_min_size(const spec::LayoutNode& n, const SpecLayoutMetrics& m, const spec::Spec* spec) {
    switch (n.kind) {
        case spec::LayoutNode::Kind::panel:
            if (spec && !spec->find_panel(n.panel))
                if (const spec::Control* c = spec->find_control(n.panel))
                    return {0, 0, n.min ? std::round((float)(*n.min)[0] * m.char_w) : m.min_control_w,
                            control_height(*c, m)};
            if (n.min) return {0, 0, std::round((float)(*n.min)[0] * m.char_w), std::round((float)(*n.min)[1] * m.line_h)};
            return {0, 0, m.min_panel_w, m.min_panel_h};
        case spec::LayoutNode::Kind::tabs: {
            float w = 0, h = 0;
            for (const auto& c : n.children) {
                const SpecRect r = spec_layout_min_size(c, m, spec);
                w = std::max(w, r.w);
                h = std::max(h, r.h);
            }
            return {0, 0, w, h + m.tab_bar_h};
        }
        case spec::LayoutNode::Kind::split: {
            const bool across = n.split == spec::SplitDirection::horizontal;
            float along = 0, cross = 0;
            for (const auto& c : n.children) {
                const SpecRect r = spec_layout_min_size(c, m, spec);
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
    return spec_layout_shares(length, mins, ratio, {});
}

std::vector<float> spec_layout_shares(float length, const std::vector<float>& mins, double ratio,
                                      const std::vector<double>& weights) {
    const size_t k = mins.size();
    std::vector<double> share(k, 0.0);
    if (k == 0) return {};
    const double L = std::max(0.0f, std::round(length));
    const double need = std::accumulate(mins.begin(), mins.end(), 0.0);
    // The weight of each child: its share, or 1 for an even split. A share
    // that is not a positive finite number counts as 1 (the validator
    // refuses it).
    const bool weighted = weights.size() == k;
    std::vector<double> w(k, 1.0);
    if (weighted)
        for (size_t i = 0; i < k; ++i) w[i] = std::isfinite(weights[i]) && weights[i] > 0.0 ? weights[i] : 1.0;
    if (need >= L) {
        // No room for every minimum: in proportion to them.
        for (size_t i = 0; i < k; ++i) share[i] = need > 0.0 ? L * mins[i] / need : L / (double)k;
    } else if (k == 2 && !weighted) {
        const double r = std::clamp(ratio, 0.0, 1.0);
        share[0] = std::clamp(r * L, (double)mins[0], L - (double)mins[1]);
        share[1] = L - share[0];
    } else {
        // Shares by weight (equal without weights), raising each child below
        // its minimum to it and sharing what is left among the others.
        std::vector<bool> fixed(k, false);
        for (bool changed = true; changed;) {
            changed = false;
            double left = L, free_w = 0.0;
            for (size_t i = 0; i < k; ++i) {
                if (fixed[i]) left -= mins[i];
                else free_w += w[i];
            }
            for (size_t i = 0; i < k; ++i) {
                const double part = free_w > 0.0 ? left * w[i] / free_w : 0.0;
                if (fixed[i]) {
                    share[i] = mins[i];
                } else if ((double)mins[i] > part) {
                    fixed[i] = true;
                    changed = true;
                } else {
                    share[i] = part;
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
    SpecRect r = snap(area);
    // A layout of controls only keeps their rows at the top of the area.
    if (const std::optional<float> h = spec_layout_fixed_height(root, spec, m)) r.h = std::min(r.h, std::round(*h));
    p.place(root, r, "", 0);
    return std::move(p.out);
}

}  // namespace caliper_host
