// Annotations, legends and bound texts in the spec runtime
// (spec_annotations.h, design/instrument-interface.md, Annotations and
// Markers, Legends and Scale Bars, Bound Text).
//
// A series panel draws its annotations inside its plot, in data units
// (the rules as filled rectangles of one pixel, so that they stay crisp):
// a threshold as a line across the plot, a band as a shaded span with its
// edges, a marker as a ring, a label as a dot with its text. Every text goes
// through the label placement (label_layout.h) inside the plot's rectangle,
// with the rings as obstacles: labels at a pick or a state first, then the
// others in spec order. A label that finds no free place shows its anchor
// dot, and a hover shows its text. Annotations use the theme's accent, or
// its secondary ink when muted, never a data colour. A click on a label or a
// marker pushes a `pick` event with the key `annotation`.
//
// The option `legend` of a series panel shows or hides the plot's legend
// when it is true or false; `auto` keeps the plot's rule, a legend for two
// series or more.
//
// The bound text helpers below evaluate a bound text's fields at the cursor
// and format them at most ten times a second. They have external linkage so
// that the text panel and the readout can share them once spec_annotations.h
// declares them.
#include "spec_runtime/spec_annotations.h"
#include "ui/sternbild_theme.h"

#include "scene/label_layout.h"

#include <caliper_builder/text/bound_text.h>

#include <implot.h>
#include <implot_internal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace caliper_host {

namespace spec = caliper_builder::spec;
using nlohmann::json;

// ---- Values at the cursor ------------------------------------------------

// The value that `r` reads now: the session's value, else its default, else
// null.
json spec_reference_value(const spec::Reference& r, const SpecLiveHost& host) {
    if (r.kind == spec::Reference::Kind::control && host.spec) {
        const spec::Control* control = host.spec->find_control(r.target);
        const auto target = control ? spec::parse_control_target(control->bind) : std::nullopt;
        if (target && target->kind == spec::ControlTarget::Kind::param) {
            // Unplaced parameter controls still supply current readout values.
            if (!host.run || r.field) return r.fallback;
            const caliper_builder::RunState state = host.run->state();
            if (!state.attached) return r.fallback;
            for (const auto& param : state.params)
                if (param.name == target->name)
                    return param.pending.is_null() ? param.value : param.pending;
            return r.fallback;
        }
    }
    if (host.values) {
        json v = host.values->value(r);
        if (!v.is_null()) return v;
    }
    return r.fallback;
}

namespace {

// A literal or a reference at a value place.
json place_value(const json& v, const SpecLiveHost& host) {
    if (spec::looks_like_reference(v))
        if (std::optional<spec::Reference> r = spec::parse_reference(v)) return spec_reference_value(*r, host);
    return v;
}

std::optional<double> place_number(const json& v, const SpecLiveHost& host) {
    json x = place_value(v, host);
    if (x.is_array() && x.size() == 1) x = x[0];
    if (!x.is_number()) return std::nullopt;
    const double d = x.get<double>();
    return std::isfinite(d) ? std::optional<double>(d) : std::nullopt;
}

// A tensor as a bound text shows it: one element as a number, else its shape.
json tensor_json(const caliper_builder::TensorData& t) {
    if (!t.valid()) return nullptr;
    if (t.numel() == 1) return t.element(0);
    json shape = json::array();
    for (std::int64_t d : t.shape()) shape.push_back(d);
    return json{{"shape", shape}};
}

}  // namespace

// The values of a bound text's fields at the cursor: a reference's value, or
// a source's kept value at the cursor's step. A source that keeps no step
// there gives null, which formats as a dash.
std::map<std::string, json> spec_bound_text_values(const spec::BoundText& text, const SpecLiveHost& host) {
    std::map<std::string, json> out;
    for (const auto& [name, field] : text.fields) {
        if (const spec::Reference* r = std::get_if<spec::Reference>(&field)) {
            out[name] = spec_reference_value(*r, host);
            continue;
        }
        const std::string& source = std::get<std::string>(field);
        json v = nullptr;
        if (host.history) {
            const caliper_builder::CursorState cursor = host.history->cursor();
            if (auto kept = host.history->at(source, cursor.step)) v = tensor_json(kept->second);
        }
        out[name] = std::move(v);
    }
    return out;
}

// The text of a bound text at the cursor, with `extra` fields such as a
// text panel's bound `value`; empty when its format does not parse. The
// format is parsed once, and the text is formatted again only when a value
// changed, so that a new value shows in the same frame.
std::string spec_bound_text_string(const spec::BoundText& text, const SpecLiveHost& host,
                                   const std::map<std::string, json>& extra) {
    struct Kept {
        caliper_builder::text::ParsedText parsed;
        bool ok = false;
        bool parsed_once = false;
        std::map<std::string, json> values;
        std::string out;
    };
    static std::map<std::pair<const spec::BoundText*, std::string>, Kept> cache;
    if (cache.size() > 256) cache.clear();
    Kept& k = cache[{&text, text.format}];
    if (!k.parsed_once) {
        std::vector<std::string> names;
        for (const auto& [name, field] : text.fields) names.push_back(name);
        for (const auto& [name, value] : extra) names.push_back(name);
        k.ok = !caliper_builder::text::parse_bound_text(text.format, names, &k.parsed);
        k.parsed_once = true;
    }
    std::map<std::string, json> values = spec_bound_text_values(text, host);
    for (const auto& [name, value] : extra) values[name] = value;
    if (k.ok && (k.out.empty() || values != k.values)) {
        k.out = caliper_builder::text::format_bound_text(k.parsed, values);
        k.values = std::move(values);
    }
    return k.ok ? k.out : std::string();
}

namespace {

// ---- Annotations of a series panel --------------------------------------

struct Ink {
    ImU32 mark = 0, fill = 0, edge = 0, text = 0, shade = sternbild::col::shade(170);
};

Ink ink_of(const spec::Annotation& a) {
    static const SpecLook look = house_spec_look();
    const bool muted = a.emphasis && *a.emphasis == "muted";
    const ImU32 base = muted ? look.secondary : look.accent;
    Ink k;
    k.mark = base;
    k.fill = sternbild::with_alpha(base, 30);
    k.edge = sternbild::with_alpha(base, 150);
    k.text = muted ? look.secondary : look.text;
    return k;
}

// Whether `at` places the annotation at a pick or at a state, which go first.
bool at_pick_or_state(const json& at) {
    const auto reads = [](const json& v) {
        if (!spec::looks_like_reference(v)) return false;
        const std::optional<spec::Reference> r = spec::parse_reference(v);
        return r && (r->kind == spec::Reference::Kind::selection || r->kind == spec::Reference::Kind::state);
    };
    if (reads(at)) return true;
    if (at.is_array())
        for (const json& v : at)
            if (reads(v)) return true;
    return false;
}

// One text to place, with what it belongs to.
struct PlotLabel {
    const spec::Annotation* annotation = nullptr;
    std::string text;
    LabelBox box;
    bool leader = true;
    double x = 0, y = 0;     // the anchor in data units, for a pick
};

void push_annotation_pick(const spec::Panel& panel, const PlotLabel& l, const SpecLiveHost& host) {
    if (!host.events) return;
    caliper_builder::Event ev;
    ev.kind = caliper_builder::EventKind::pick;
    ev.panel = panel.id;
    ev.origin = "ui";
    ev.fields = json{{"annotation", l.annotation->id}, {"position", json::array({l.x, l.y})}};
    host.events->push(std::move(ev));
}

}  // namespace

void draw_plot_annotations(const spec::Panel& panel, const SpecLiveHost& host) {
    if (panel.primitive == spec::Primitive::series && panel.options.legend.is_boolean()) {
        if (ImPlotPlot* plot = ImPlot::GetCurrentPlot()) {
            if (panel.options.legend.get<bool>()) plot->Flags &= ~ImPlotFlags_NoLegend;
            else plot->Flags |= ImPlotFlags_NoLegend;
        }
    }
    if (panel.annotations.empty()) return;
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    const ImVec2 pos = ImPlot::GetPlotPos(), size = ImPlot::GetPlotSize();
    const LabelRect area{pos.x, pos.y, pos.x + size.x, pos.y + size.y};
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    ImPlot::PushPlotClipRect();

    const float font = ImGui::GetFontSize();
    const float pad = std::round(0.2f * font);
    const float ring = std::max(4.0f, std::round(0.4f * font));
    std::vector<PlotLabel> labels;
    std::vector<PlotLabel> marks;   // the markers, for their picks
    std::vector<LabelRect> rings;
    const auto add_label = [&](const spec::Annotation& a, std::size_t index, std::string text, float ax, float ay,
                               double dx, double dy, bool first) {
        if (text.empty()) return;
        const ImVec2 ts = ImGui::CalcTextSize(text.c_str());
        PlotLabel l;
        l.annotation = &a;
        l.text = std::move(text);
        l.box = {ax, ay, ts.x + 2 * pad, ts.y + 2 * pad, (first ? 0 : 64) + (int)index};
        l.leader = a.leader.value_or(true);
        l.x = dx;
        l.y = dy;
        labels.push_back(std::move(l));
    };

    for (std::size_t i = 0; i < panel.annotations.size(); ++i) {
        const spec::Annotation& a = panel.annotations[i];
        const Ink k = ink_of(a);
        const std::string text = a.text ? spec_bound_text_string(*a.text, host, {}) : std::string();
        const bool vertical = a.axis && *a.axis == "x";   // a line or span across x: vertical
        switch (a.kind) {
            case spec::AnnotationKind::threshold: {
                const std::optional<double> v = place_number(a.value, host);
                if (!v) break;
                const ImVec2 p = ImPlot::PlotToPixels(vertical ? *v : lim.X.Min, vertical ? lim.Y.Max : *v);
                if (vertical) {
                    dl->AddRectFilled(ImVec2(std::round(p.x), area.y0), ImVec2(std::round(p.x) + 1.0f, area.y1), k.mark);
                    add_label(a, i, text, std::round(p.x), area.y0, *v, lim.Y.Max, false);
                } else {
                    dl->AddRectFilled(ImVec2(area.x0, std::round(p.y)), ImVec2(area.x1, std::round(p.y) + 1.0f), k.mark);
                    add_label(a, i, text, area.x1, std::round(p.y), lim.X.Max, *v, false);
                }
                break;
            }
            case spec::AnnotationKind::band: {
                std::optional<double> lo, hi;
                const json r = place_value(a.range, host);
                if (r.is_array() && r.size() == 2) {
                    lo = place_number(r[0], host);
                    hi = place_number(r[1], host);
                }
                if (!lo || !hi) break;
                if (*lo > *hi) std::swap(lo, hi);
                if (vertical) {
                    const float x0 = std::round(ImPlot::PlotToPixels(*lo, lim.Y.Max).x);
                    const float x1 = std::round(ImPlot::PlotToPixels(*hi, lim.Y.Max).x);
                    dl->AddRectFilled(ImVec2(x0, area.y0), ImVec2(x1 + 1.0f, area.y1), k.fill);
                    dl->AddRectFilled(ImVec2(x0, area.y0), ImVec2(x0 + 1.0f, area.y1), k.edge);
                    dl->AddRectFilled(ImVec2(x1, area.y0), ImVec2(x1 + 1.0f, area.y1), k.edge);
                    add_label(a, i, text, std::max(x0, area.x0), area.y0, *lo, lim.Y.Max, false);
                } else {
                    const float y0 = std::round(ImPlot::PlotToPixels(lim.X.Min, *hi).y);
                    const float y1 = std::round(ImPlot::PlotToPixels(lim.X.Min, *lo).y);
                    dl->AddRectFilled(ImVec2(area.x0, y0), ImVec2(area.x1, y1 + 1.0f), k.fill);
                    dl->AddRectFilled(ImVec2(area.x0, y0), ImVec2(area.x1, y0 + 1.0f), k.edge);
                    dl->AddRectFilled(ImVec2(area.x0, y1), ImVec2(area.x1, y1 + 1.0f), k.edge);
                    add_label(a, i, text, area.x0, std::max(y0, area.y0), lim.X.Min, *hi, false);
                }
                break;
            }
            case spec::AnnotationKind::label:
            case spec::AnnotationKind::marker: {
                if (!a.at.is_array() || a.at.size() != 2) break;
                const std::optional<double> x = place_number(a.at[0], host), y = place_number(a.at[1], host);
                if (!x || !y) break;
                const ImVec2 p = ImPlot::PlotToPixels(*x, *y);
                if (a.kind == spec::AnnotationKind::marker) {
                    dl->AddCircle(p, ring, k.mark, 0, 2.0f);
                    rings.push_back({p.x - ring - 1, p.y - ring - 1, p.x + ring + 1, p.y + ring + 1});
                    PlotLabel m;
                    m.annotation = &a;
                    m.box.anchor_x = p.x;
                    m.box.anchor_y = p.y;
                    m.x = *x;
                    m.y = *y;
                    marks.push_back(std::move(m));
                } else {
                    dl->AddCircleFilled(p, std::max(2.0f, 0.15f * font), k.mark);
                }
                add_label(a, i, text, p.x, p.y, *x, *y, at_pick_or_state(a.at));
                break;
            }
        }
    }

    std::vector<LabelBox> boxes;
    boxes.reserve(labels.size());
    for (const PlotLabel& l : labels) boxes.push_back(l.box);
    const std::vector<PlacedLabel> placed = place_labels(boxes, area, rings);
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool hovered = ImPlot::IsPlotHovered();
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const PlotLabel& l = labels[i];
        const PlacedLabel& p = placed[i];
        const Ink k = ink_of(*l.annotation);
        const LabelBox& b = l.box;
        if (!p.shown) {
            dl->AddCircleFilled(ImVec2(b.anchor_x, b.anchor_y), std::max(2.0f, 0.15f * font), k.mark);
            if (hovered && std::fabs(mouse.x - b.anchor_x) <= ring && std::fabs(mouse.y - b.anchor_y) <= ring)
                ImGui::SetTooltip("%s", l.text.c_str());
            continue;
        }
        if (p.leader && l.leader) {
            const float cx = std::clamp(b.anchor_x, p.x, p.x + b.width), cy = std::clamp(b.anchor_y, p.y, p.y + b.height);
            dl->AddLine(ImVec2(b.anchor_x, b.anchor_y), ImVec2(cx, cy), k.edge, 1.0f);
        }
        const ImVec2 at(p.x + pad, p.y + pad);
        dl->AddText(ImVec2(at.x + 1.0f, at.y + 1.0f), k.shade, l.text.c_str());
        dl->AddText(at, k.text, l.text.c_str());
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouse.x >= p.x && mouse.x < p.x + b.width &&
            mouse.y >= p.y && mouse.y < p.y + b.height)
            push_annotation_pick(panel, l, host);
    }
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        for (const PlotLabel& l : marks)
            if (std::hypot(mouse.x - l.box.anchor_x, mouse.y - l.box.anchor_y) <= ring + 2.0f) {
                push_annotation_pick(panel, l, host);
                break;
            }
    ImPlot::PopPlotClipRect();
}

void draw_panel_overlays(const spec::Panel& panel, const ImRect& content, const SpecLook& look,
                         const SpecLiveHost& host) {
    // A 2-D panel draws its annotations inside its plot
    // (draw_plot_annotations), a scene panel through its scene
    // (scene_annotation_marks) and a text panel its bound text; the legend of
    // a scene and the scale bars of image and points panels wait for their
    // panels' drawing to leave them the room (the validator refuses them).
    (void)panel;
    (void)content;
    (void)look;
    (void)host;
}

std::vector<SceneAnnotationMark> scene_annotation_marks(const spec::Panel& panel, const SpecLiveHost& host) {
    std::vector<SceneAnnotationMark> out;
    for (const spec::Annotation& a : panel.annotations) {
        SceneAnnotationMark m;
        m.id = a.id;
        m.kind = a.kind;
        m.muted = a.emphasis && *a.emphasis == "muted";
        m.text = a.text ? spec_bound_text_string(*a.text, host, {}) : std::string();
        const int axis = !a.axis ? 1 : *a.axis == "x" ? 0 : *a.axis == "y" ? 1 : 2;
        switch (a.kind) {
            case spec::AnnotationKind::threshold: {
                const std::optional<double> v = place_number(a.value, host);
                if (!v) continue;
                m.axis = axis;
                m.lo = m.hi = *v;
                break;
            }
            case spec::AnnotationKind::band: {
                json r = place_value(a.range, host);
                if (!r.is_array() || r.size() != 2) continue;
                const std::optional<double> lo = place_number(r[0], host), hi = place_number(r[1], host);
                if (!lo || !hi) continue;
                m.axis = axis;
                m.lo = std::min(*lo, *hi);
                m.hi = std::max(*lo, *hi);
                break;
            }
            case spec::AnnotationKind::label:
            case spec::AnnotationKind::marker: {
                // A pick (its x, y and z), an entity's item, or a point.
                const json at = place_value(a.at, host);
                if (at.is_object() && at.contains("x") && at.contains("y") && at.contains("z")) {
                    if (!at["x"].is_number() || !at["y"].is_number() || !at["z"].is_number()) continue;
                    m.at[0] = at["x"].get<double>();
                    m.at[1] = at["y"].get<double>();
                    m.at[2] = at["z"].get<double>();
                } else if (at.is_object() && at.contains("entity") && at["entity"].is_string()) {
                    const std::optional<double> index =
                        at.contains("index") ? place_number(at["index"], host) : std::nullopt;
                    // An item is a whole number from 0; the cast needs a
                    // value inside the range of int64_t.
                    if (!index || !(*index >= 0.0) || *index > 9007199254740991.0) continue;
                    m.entity = at["entity"].get<std::string>();
                    m.index = static_cast<int64_t>(*index);
                } else if (at.is_array() && (at.size() == 2 || at.size() == 3)) {
                    bool all = true;
                    for (std::size_t k = 0; k < at.size(); ++k) {
                        const std::optional<double> v = place_number(at[k], host);
                        all = all && v.has_value();
                        if (v) m.at[k] = *v;
                    }
                    if (!all) continue;
                } else {
                    continue;
                }
                break;
            }
        }
        out.push_back(std::move(m));
    }
    return out;
}

}  // namespace caliper_host
