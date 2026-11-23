// spec_runtime.cpp: the spec runtime (spec_runtime.h): the look, the
// textures, the drawing of each primitive and the runtime that lays a spec
// out and keeps its panels' data.
#include "spec_runtime/spec_runtime.h"
#include "spec_runtime/spec_annotations.h"
#include "spec_runtime/spec_instruments.h"
#include "spec_runtime/spec_shortcuts.h"
#include "spec_runtime/spec_transport.h"
#include "spec_runtime/spec_stage_host.h"
#include "live/live_views_scene.h"   // a scene drawn into a scene panel's rectangle
#include "live/live_scene.h"         // the scene view of a panel's entities
#include "spec_runtime/spec_entities.h"
#include "spec_runtime/label_layout.h"   // fit_with_ellipsis
#include "draw/draw_host.h"                 // the programs of the scene panels
#include "scene/scene.h"   // kPalette

#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"

#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/colormaps.h>   // colormap_names
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/session/piece_commands.h>   // resolve_pin
#include <caliper_builder/session/history.h>

#include <imgui_internal.h>
#include <implot.h>
#include <implot3d.h>
#include <implot3d_internal.h>   // the plot's rotation, to fit its box

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <limits>

// The live views' categorical palette (the scene core's, scene/scene.h) and
// point sizes (live_views_draw.cpp), so a spec panel and a live view share them.
namespace caliper_live_views {
using caliper_scene::kPalette;
float point_size_px(uint64_t n, float scale);
}  // namespace caliper_live_views

namespace caliper_host {

namespace spec = caliper_builder::spec;
using caliper_builder::PanelData;

namespace {


ImU32 with_alpha(ImU32 c, int a) { return (c & 0x00ffffffu) | ((ImU32)a << 24); }

ImFont* font_or_current(ImFont* f) { return f ? f : ImGui::GetFont(); }

ImVec2 text_size(ImFont* f, float size, const std::string& s) {
    return font_or_current(f)->CalcTextSizeA(size, FLT_MAX, 0.0f, s.c_str());
}

// `s` cut to `w` with an ellipsis, "" when not even that fits (the live
// views' fit_text).
std::string fit_text(ImFont* f, float size, const std::string& s, float w) {
    ImFont* font = font_or_current(f);
    return caliper_host::fit_with_ellipsis(s, w, [font, size](std::string_view t) {
        return font->CalcTextSizeA(size, FLT_MAX, 0.0f, t.data(), t.data() + t.size()).x;
    });
}

void text_at(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 c, const std::string& s) {
    dl->AddText(font_or_current(f), size, ImVec2(std::round(p.x), std::round(p.y)), c, s.c_str());
}

// Evenly spaced round integer ticks in [lo, hi], about n of them (the live
// views' index_ticks).
std::vector<int64_t> index_ticks(double lo, double hi, int n) {
    std::vector<int64_t> out;
    if (!(hi >= lo) || !std::isfinite(lo) || !std::isfinite(hi)) return out;
    const double raw = std::max(hi - lo, 1e-9) / std::max(1, n);
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double r = raw / mag;
    const double step = std::max(1.0, std::round((r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag));
    for (double t = std::ceil(lo / step) * step; t <= hi + step * 1e-9 && out.size() < 64; t += step)
        out.push_back((int64_t)std::llround(t));
    return out;
}

// About n round ticks in [lo, hi] (steps of 1, 2 or 5 times a power of ten).
std::vector<double> nice_ticks(double lo, double hi, int n) {
    std::vector<double> out;
    if (!(hi > lo) || !std::isfinite(lo) || !std::isfinite(hi)) return out;
    const double raw = (hi - lo) / std::max(1, n);
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double r = raw / mag;
    const double step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    for (double t = std::ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9 && out.size() < 32; t += step)
        out.push_back(std::fabs(t) < step * 1e-9 ? 0.0 : t);
    return out;
}

// ImPlot colormaps of the sixteen spec colormaps, added once per ImPlot context, each named
// "caliper.<name>" from caliper_builder::spec::colormap_names().
ImPlotColormap implot_colormap(int cm) {
    static const std::vector<std::string> kNames = [] {
        std::vector<std::string> v;
        for (const std::string& n : caliper_builder::spec::colormap_names()) v.push_back("caliper." + n);
        return v;
    }();
    const int i = std::clamp(cm, 0, int(kNames.size()) - 1);
    ImPlotColormap id = ImPlot::GetColormapIndex(kNames[i].c_str());
    if (id < 0) {
        const uint32_t* lut = spec_colormap_lut(i);
        ImU32 cols[256];
        for (int k = 0; k < 256; ++k) cols[k] = (ImU32)lut[k];
        id = ImPlot::AddColormap(kNames[i].c_str(), cols, 256, false);
    }
    return id;
}

// The house's plot style for one plot, popped by pop_plot_style().
void push_plot_style(const SpecLook& L) {
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(0, 0, 0, 0));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, L.deep);
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, L.rule);
    ImPlot::PushStyleColor(ImPlotCol_LegendBg, with_alpha(L.ground, 230));
    ImPlot::PushStyleColor(ImPlotCol_LegendBorder, L.rule);
    ImPlot::PushStyleColor(ImPlotCol_LegendText, L.secondary);
    ImPlot::PushStyleColor(ImPlotCol_InlayText, L.secondary);
    ImPlot::PushStyleColor(ImPlotCol_AxisText, L.faint);
    ImPlot::PushStyleColor(ImPlotCol_AxisGrid, L.rule_faint);
    ImPlot::PushStyleColor(ImPlotCol_AxisTick, L.rule);
    const float s = L.text_scale;
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(4.0f * s, 6.0f * s));
    ImPlot::PushStyleVar(ImPlotStyleVar_LabelPadding, ImVec2(5.0f * s, 4.0f * s));
    ImPlot::PushStyleVar(ImPlotStyleVar_MajorTickLen, ImVec2(4.0f * s, 4.0f * s));
    ImPlot::PushStyleVar(ImPlotStyleVar_MinorTickLen, ImVec2(2.0f * s, 2.0f * s));
    ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.0f, 0.0f));
    ImPlot::PushStyleVar(ImPlotStyleVar_LegendPadding, ImVec2(8.0f * s, 8.0f * s));
    ImPlot::PushStyleVar(ImPlotStyleVar_LegendInnerPadding, ImVec2(6.0f * s, 4.0f * s));
    ImGui::PushFont(L.mono, L.mono_size - 1.0f);
}

void pop_plot_style() {
    ImGui::PopFont();
    ImPlot::PopStyleVar(7);
    ImPlot::PopStyleColor(10);
}

void push_plot3d_style(const SpecLook& L) {
    ImPlot3D::PushStyleColor(ImPlot3DCol_FrameBg, IM_COL32(0, 0, 0, 0));
    ImPlot3D::PushStyleColor(ImPlot3DCol_PlotBg, L.deep);
    ImPlot3D::PushStyleColor(ImPlot3DCol_PlotBorder, L.rule);
    ImPlot3D::PushStyleColor(ImPlot3DCol_AxisText, L.faint);
    ImPlot3D::PushStyleColor(ImPlot3DCol_AxisGrid, L.rule_faint);
    ImPlot3D::PushStyleColor(ImPlot3DCol_AxisTick, L.rule);
    ImGui::PushFont(L.mono, L.mono_size - 1.0f);
}

void pop_plot3d_style() {
    ImGui::PopFont();
    ImPlot3D::PopStyleColor(6);
}

// Limits [lo, hi] of values with padding: `pad` of the span on a linear
// axis, of the decades on a log axis (the live views' log fit), and a span of
// one around a flat value.
bool fit_limits(const std::vector<double>& v, bool log, double pad, double* lo, double* hi) {
    double a = INFINITY, b = -INFINITY;
    for (double x : v) {
        if (!std::isfinite(x) || (log && x <= 0.0)) continue;
        a = std::min(a, x);
        b = std::max(b, x);
    }
    if (!(b >= a)) return false;
    if (log) {
        const double l = std::log10(a), h = std::log10(b), p = std::max(0.05, (h - l) * pad);
        *lo = std::pow(10.0, l - p);
        *hi = std::pow(10.0, h + p);
    } else if (b == a) {
        *lo = a - 0.5;
        *hi = b + 0.5;
    } else {
        *lo = a - (b - a) * pad;
        *hi = b + (b - a) * pad;
    }
    return true;
}

const char* label_or_null(const std::string& s) { return s.empty() ? nullptr : s.c_str(); }

// Labelled ticks for a log axis over [lo, hi]: 1, 2 and 5 of each decade, or
// only the decades when there are more than four of them. ImPlot's own log
// ticks label only the decades, which leaves a short range with one label.
void setup_log_ticks(ImAxis axis, double lo, double hi) {
    if (!(lo > 0.0) || !(hi > lo)) return;
    const int d0 = (int)std::floor(std::log10(lo)), d1 = (int)std::ceil(std::log10(hi));
    const bool decades_only = d1 - d0 > 4;
    std::vector<double> v;
    std::vector<std::string> text;
    // Every whole multiple of a decade gets a grid line, so the spacing reads
    // as a log axis; 1, 2 and 5 carry labels.
    for (int d = d0; d <= d1 && v.size() < 160; ++d)
        for (int m = 1; m <= 9; ++m) {
            const double t = m * std::pow(10.0, d);
            if (t < lo || t > hi) continue;
            const bool label = decades_only ? m == 1 : (m == 1 || m == 2 || m == 5);
            if (decades_only && d1 - d0 > 12 && m != 1) continue;
            v.push_back(t);
            text.push_back(label ? spec_fmt4(t) : std::string());
        }
    if (v.empty()) return;
    std::vector<const char*> labels;
    for (const auto& t : text) labels.push_back(t.c_str());
    ImPlot::SetupAxisTicks(axis, v.data(), (int)v.size(), labels.data());
}

ImPlotFlags plot_flags(bool legend) {
    return ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText |
           (legend ? 0 : ImPlotFlags_NoLegend);
}

// ---- the selection --------------------------------------------------------------

// Whether a shift-drag may start or runs on this panel's plot: the plot then
// takes no input of its own, so the drag selects instead of panning.
bool selection_input(const SpecPanelView& v, const ImRect& c) {
    if (!v.selection) return false;
    return v.selecting || (ImGui::GetIO().KeyShift && ImGui::IsMouseHoveringRect(c.Min, c.Max, false));
}

// Inside BeginPlot, after the data: the shift-drag that makes a selection
// (`rect`: a rectangle of points; otherwise a range of x), and the
// selection drawn in bronze when it is this panel's.
void selection_gesture(SpecPanelView& v, const SpecLook& L, bool rect) {
    if (!v.selection) return;
    SpecSelection& sel = *v.selection;
    const ImVec2 pp = ImPlot::GetPlotPos(), ps = ImPlot::GetPlotSize();
    const bool over = ImGui::IsMouseHoveringRect(pp, ImVec2(pp.x + ps.x, pp.y + ps.y), false);
    const ImPlotPoint m = ImPlot::GetPlotMousePos();
    if (!v.selecting && over && ImGui::GetIO().KeyShift && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        v.selecting = true;
        v.select_from[0] = m.x;
        v.select_from[1] = m.y;
    }
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    double x0, x1, y0, y1;
    bool show = false;
    if (v.selecting) {
        x0 = std::min(v.select_from[0], m.x);
        x1 = std::max(v.select_from[0], m.x);
        y0 = rect ? std::min(v.select_from[1], m.y) : lim.Y.Min;
        y1 = rect ? std::max(v.select_from[1], m.y) : lim.Y.Max;
        show = true;
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            v.selecting = false;
            const ImVec2 a = ImPlot::PlotToPixels(v.select_from[0], v.select_from[1]);
            const ImVec2 b = ImPlot::PlotToPixels(m.x, m.y);
            if (std::fabs(a.x - b.x) < 3.0f && (!rect || std::fabs(a.y - b.y) < 3.0f)) {
                if (sel.panel == v.data.id) sel = {};   // a shift-click clears it
                show = false;
            } else {
                sel = {};
                sel.kind = rect ? SpecSelection::Kind::points : SpecSelection::Kind::range;
                sel.panel = v.data.id;
                sel.x0 = x0;
                sel.x1 = x1;
                sel.y0 = rect ? y0 : 0.0;
                sel.y1 = rect ? y1 : 0.0;
            }
        }
    } else if (!sel.empty() && sel.panel == v.data.id) {
        x0 = sel.x0;
        x1 = sel.x1;
        y0 = sel.kind == SpecSelection::Kind::points ? sel.y0 : lim.Y.Min;
        y1 = sel.kind == SpecSelection::Kind::points ? sel.y1 : lim.Y.Max;
        show = true;
    }
    if (!show) return;
    ImPlot::PushPlotClipRect();
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    ImVec2 a = ImPlot::PlotToPixels(x0, y1), b = ImPlot::PlotToPixels(x1, y0);
    a = ImVec2(std::round(a.x), std::round(a.y));
    b = ImVec2(std::round(b.x), std::round(b.y));
    dl->AddRectFilled(a, b, sternbild::with_alpha(L.accent, 46));
    if (rect) {
        dl->AddRect(a, b, L.accent, 0.0f, 0, 1.0f);
    } else {
        dl->AddRectFilled(ImVec2(a.x, a.y), ImVec2(a.x + 1.0f, b.y), L.accent);
        dl->AddRectFilled(ImVec2(b.x - 1.0f, a.y), ImVec2(b.x, b.y), L.accent);
    }
    ImPlot::PopPlotClipRect();
}

// ---- the primitives ------------------------------------------------------------

void draw_series(SpecPanelView& v, const ImRect& c, const SpecLook& L) {
    const PreparedPanel& p = v.data;
    const PreparedSeries& S = p.series;
    const bool legend = S.lines.size() > 1;
    push_plot_style(L);
    const ImPlotFlags flags = plot_flags(legend) | (selection_input(v, c) ? ImPlotFlags_NoInputs : 0);
    if (ImPlot::BeginPlot("##spec.series", c.GetSize(), flags)) {
        ImPlot::SetupAxes(label_or_null(p.x_label), label_or_null(p.y_label));
        if (p.x_log) ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        if (p.y_log) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        if (legend) ImPlot::SetupLegend(ImPlotLocation_NorthEast);
        const bool fit = v.fit_frames > 0;
        if (fit) {
            // The data's extents, kept for the log axes' ticks.
            std::vector<double> xs, ys;
            for (const auto& l : S.lines) {
                xs.insert(xs.end(), l.x.begin(), l.x.end());
                ys.insert(ys.end(), l.y.begin(), l.y.end());
            }
            if (!fit_limits(xs, p.x_log, 0.0, &v.x_extent[0], &v.x_extent[1])) v.x_extent[0] = v.x_extent[1] = 0.0;
            if (!fit_limits(ys, p.y_log, 0.05, &v.y_extent[0], &v.y_extent[1])) v.y_extent[0] = v.y_extent[1] = 0.0;
        }
        // A fixed range on a log axis whose end is not above zero takes that
        // end from the data instead.
        double xr[2] = {p.x_range[0], p.x_range[1]}, yr[2] = {p.y_range[0], p.y_range[1]};
        if (p.x_log && p.has_x_range && xr[0] <= 0.0) xr[0] = v.x_extent[0] > 0.0 ? v.x_extent[0] : xr[1] * 1e-3;
        if (p.y_log && p.has_y_range && yr[0] <= 0.0) yr[0] = v.y_extent[0] > 0.0 ? v.y_extent[0] : yr[1] * 1e-3;
        const bool x_fixed = p.has_x_range && xr[1] > xr[0] && (!p.x_log || xr[0] > 0.0);
        const bool y_fixed = p.has_y_range && yr[1] > yr[0] && (!p.y_log || yr[0] > 0.0);
        if (p.x_log) setup_log_ticks(ImAxis_X1, x_fixed ? xr[0] : v.x_extent[0], x_fixed ? xr[1] : v.x_extent[1]);
        if (p.y_log) setup_log_ticks(ImAxis_Y1, y_fixed ? yr[0] : v.y_extent[0], y_fixed ? yr[1] : v.y_extent[1]);
        if (x_fixed) {
            ImPlot::SetupAxisLimits(ImAxis_X1, xr[0], xr[1], ImPlotCond_Always);
        } else if (fit) {
            std::vector<double> xs;
            for (const auto& l : S.lines) xs.insert(xs.end(), l.x.begin(), l.x.end());
            double lo, hi;
            if (fit_limits(xs, p.x_log, 0.0, &lo, &hi)) ImPlot::SetupAxisLimits(ImAxis_X1, lo, hi, ImPlotCond_Always);
        }
        if (y_fixed) {
            ImPlot::SetupAxisLimits(ImAxis_Y1, yr[0], yr[1], ImPlotCond_Always);
        } else if (fit) {
            std::vector<double> ys;
            for (const auto& l : S.lines) ys.insert(ys.end(), l.y.begin(), l.y.end());
            double lo, hi;
            if (fit_limits(ys, p.y_log, 0.05, &lo, &hi)) ImPlot::SetupAxisLimits(ImAxis_Y1, lo, hi, ImPlotCond_Always);
        }
        for (size_t i = 0; i < S.lines.size(); ++i) {
            const SeriesLine& l = S.lines[i];
            if (l.y.empty()) continue;
            ImPlotSpec spec;
            spec.LineColor = ImGui::ColorConvertU32ToFloat4(v.key ? v.key->color(l.name) : spec_palette_color(i));
            spec.LineWeight = 1.25f * L.text_scale;
            ImPlot::PlotLine(l.name.c_str(), l.x.data(), l.y.data(), (int)l.y.size(), spec);
        }
        // Hover: the nearest step of each line, by name and value.
        if (ImPlot::IsPlotHovered() && !S.lines.empty()) {
            const double mx = ImPlot::GetPlotMousePos().x;
            std::string tip;
            for (const auto& l : S.lines) {
                if (l.x.empty()) continue;
                const auto it = std::lower_bound(l.x.begin(), l.x.end(), mx);
                size_t i = (size_t)std::min<ptrdiff_t>(it - l.x.begin(), (ptrdiff_t)l.x.size() - 1);
                if (i > 0 && std::fabs(l.x[i - 1] - mx) < std::fabs(l.x[i] - mx)) --i;
                if (tip.empty()) tip = S.x_name + " " + spec_fmt4(l.x[i]);
                tip += "\n" + l.name + " " + spec_fmt4(l.y[i]);
            }
            if (!tip.empty()) ImGui::SetTooltip("%s", tip.c_str());
        }
        selection_gesture(v, L, false);
        if (v.spec_panel && v.instruments) {   // spec_transport.h, spec_annotations.h
            draw_cursor_line(*v.spec_panel, *v.instruments);
            draw_plot_annotations(*v.spec_panel, *v.instruments);
        }
        ImPlot::EndPlot();
    }
    pop_plot_style();
}

void draw_histogram(SpecPanelView& v, const ImRect& c, const SpecLook& L) {
    const PreparedPanel& p = v.data;
    const PreparedHistogram& H = p.histogram;
    const size_t B = H.counts.size();
    const double w = B ? (H.hi - H.lo) / (double)B : 1.0;
    std::vector<double> xs(B), ys(B);
    for (size_t i = 0; i < B; ++i) {
        xs[i] = H.lo + ((double)i + 0.5) * w;
        ys[i] = H.counts[i];
    }
    push_plot_style(L);
    if (ImPlot::BeginPlot("##spec.histogram", c.GetSize(), plot_flags(false))) {
        ImPlot::SetupAxes(label_or_null(p.x_label), label_or_null(p.y_label));
        if (p.y_log) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        if (v.fit_frames > 0) {
            ImPlot::SetupAxisLimits(ImAxis_X1, H.lo, H.hi, ImPlotCond_Always);
            double lo = 0.0, hi = 1.0;
            if (p.y_log) {
                // Counts on a log axis start under 1, so that a bin of one
                // value shows as a bar; the ticks start at 1.
                double top = 1.0;
                for (double y : ys) top = std::max(top, y);
                lo = 0.5;
                hi = std::pow(10.0, std::log10(top) + std::max(0.05, (std::log10(top) - std::log10(lo)) * 0.05));
                v.y_extent[0] = 1.0;
                v.y_extent[1] = hi;
            } else {
                double top = 0.0;
                for (double y : ys) top = std::max(top, y);
                hi = top > 0.0 ? top * 1.05 : 1.0;
            }
            ImPlot::SetupAxisLimits(ImAxis_Y1, lo, hi, ImPlotCond_Always);
        }
        if (p.y_log) setup_log_ticks(ImAxis_Y1, v.y_extent[0], v.y_extent[1]);
        ImPlotSpec bars;
        bars.FillColor = bars.LineColor = ImGui::ColorConvertU32ToFloat4(L.line);
        bars.LineWeight = 0.0f;
        if (B) ImPlot::PlotBars("count", xs.data(), ys.data(), (int)B, w * 0.92, bars);
        if (ImPlot::IsPlotHovered() && B) {
            const double fi = std::floor((ImPlot::GetPlotMousePos().x - H.lo) / w);
            if (fi >= 0.0 && fi < (double)B) {
                const size_t i = (size_t)fi;
                ImGui::SetTooltip("%s %s\ncount %s", p.x_label.c_str(), spec_fmt4(xs[i]).c_str(),
                                  spec_fmt4(ys[i]).c_str());
            }
        }
        if (v.spec_panel && v.instruments) draw_plot_annotations(*v.spec_panel, *v.instruments);   // spec_annotations.h
        ImPlot::EndPlot();
    }
    pop_plot_style();
}

// A heatmap or an image as the live views draw them on a page: the texture
// over axes of row and column indices in the mono face, the dims' names, and
// a heatmap's colorbar at the right; a heatmap stretched to the area, an
// image at equal units across and down.
void draw_grid(SpecPanelView& v, const ImRect& c, const SpecLook& L, SpecTextures& textures, bool image) {
    const PreparedPanel& p = v.data;
    const PreparedGrid& G = p.grid;
    if (!v.texture && !G.rgba.empty()) v.texture = textures.create(G.cols, G.rows, G.rgba.data());
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float afs = L.mono_size - 1.0f;   // the live views' axes in a window
    const float k = afs / 10.5f;
    // A heatmap's colorbar carries the values of its ends and of round steps
    // between them, right of the bar.
    std::vector<double> bar_ticks;
    std::vector<std::string> bar_labels;
    float bar_label_w = 0.0f;
    if (!image) {
        const float est_h = c.GetHeight() - 48.0f * k;
        bar_ticks = nice_ticks(G.vmin, G.vmax, std::max(2, (int)(est_h / (48.0f * k))));
        for (double t : bar_ticks) {
            bar_labels.push_back(spec_fmt4(t));
            bar_label_w = std::max(bar_label_w, text_size(L.mono, afs, bar_labels.back()).x);
        }
    }
    const float ml = 46.0f * k, mt = 16.0f * k, mb = 32.0f * k;
    const float mr = image ? 10.0f * k : 26.0f * k + (bar_labels.empty() ? 0.0f : 6.0f * k + bar_label_w);
    float ax = c.Min.x + ml, ay = c.Min.y + mt;
    float aw = c.GetWidth() - ml - mr, ah = c.GetHeight() - mt - mb;
    if (aw < 4.0f || ah < 4.0f || G.rows <= 0 || G.cols <= 0) return;
    const double W = G.source_cols, H = G.source_rows;
    if (image) {
        // Equal units, centred in the area.
        const double kf = std::min(aw / W, ah / H);
        const float iw = (float)(W * kf), ih = (float)(H * kf);
        ax += std::floor((aw - iw) * 0.5f);
        ay += std::floor((ah - ih) * 0.5f);
        aw = iw;
        ah = ih;
    }
    ax = std::round(ax);
    ay = std::round(ay);
    aw = std::round(aw);
    ah = std::round(ah);
    if (v.texture) {
        const ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
        const bool magnify = std::min(aw / (float)G.cols, ah / (float)G.rows) * ImGui::GetIO().DisplayFramebufferScale.x >= 1.0f;
        if (magnify && pio.DrawCallback_SetSamplerNearest) dl->AddCallback(pio.DrawCallback_SetSamplerNearest, nullptr);
        dl->AddImage(v.texture->GetTexRef(), ImVec2(ax, ay), ImVec2(ax + aw, ay + ah));
        if (magnify && pio.DrawCallback_SetSamplerLinear) dl->AddCallback(pio.DrawCallback_SetSamplerLinear, nullptr);
    }
    // The frame, the ticks at cell centres, labelled with source indices.
    // The frame on the pixels around the area, never over its cells.
    dl->AddRect(ImVec2(ax - 1.0f, ay - 1.0f), ImVec2(ax + aw + 1.0f, ay + ah + 1.0f), L.rule);
    const float sx = aw / (float)W, sy = ah / (float)H;
    for (int64_t col : index_ticks(0.0, W - 1, std::max(2, (int)(aw / (70 * k))))) {
        const float x = std::round(ax + (float)(col + 0.5) * sx) + 0.5f;
        dl->AddLine(ImVec2(x, ay + ah + 0.5f), ImVec2(x, ay + ah + 4.5f), L.rule);
        const std::string t = std::to_string(col);
        const ImVec2 ts = text_size(L.mono, afs, t);
        text_at(dl, L.mono, afs, ImVec2(x - ts.x * 0.5f, ay + ah + 6.0f * k), L.faint, t);
    }
    for (int64_t row : index_ticks(0.0, H - 1, std::max(2, (int)(ah / (36 * k))))) {
        const float y = std::round(ay + (float)(row + 0.5) * sy) + 0.5f;
        dl->AddLine(ImVec2(ax - 0.5f, y), ImVec2(ax - 4.5f, y), L.rule);
        const std::string t = std::to_string(row);
        const ImVec2 ts = text_size(L.mono, afs, t);
        text_at(dl, L.mono, afs, ImVec2(ax - 6.0f * k - ts.x, y - ts.y * 0.5f), L.faint, t);
    }
    if (!p.x_label.empty()) {
        const ImVec2 ts = text_size(L.mono, afs, p.x_label);
        text_at(dl, L.mono, afs, ImVec2(ax + aw - ts.x, ay + ah + 19.0f * k), L.faint, p.x_label);
    }
    if (!p.y_label.empty()) {
        const ImVec2 ts = text_size(L.mono, afs, p.y_label);
        text_at(dl, L.mono, afs, ImVec2(std::max(c.Min.x, ax - 6.0f * k - ts.x), ay - 4.0f * k - ts.y), L.faint,
                p.y_label);
    }
    if (!image) {
        // The colorbar: the colormap from its top (index 255) to its bottom.
        const uint32_t* lut = spec_colormap_lut(G.colormap);
        const float bx = std::round(ax + aw + 8.0f * k), bw = std::round(8.0f * k);
        const int n = std::max(2, (int)std::floor(ah));
        for (int i = 0; i < n; ++i) {
            const int idx = (int)std::lround(255.0 * (1.0 - (double)i / (n - 1)));
            dl->AddRectFilled(ImVec2(bx, ay + i * ah / n), ImVec2(bx + bw, ay + (i + 1) * ah / n), (ImU32)lut[idx]);
        }
        dl->AddRect(ImVec2(bx - 1.0f, ay - 1.0f), ImVec2(bx + bw + 1.0f, ay + ah + 1.0f), L.rule);
        const double span = (double)G.vmax - (double)G.vmin;
        for (size_t i = 0; i < bar_ticks.size() && span > 0.0; ++i) {
            const float y = std::round(ay + ah * (float)(1.0 - (bar_ticks[i] - G.vmin) / span)) + 0.5f;
            dl->AddLine(ImVec2(bx + bw + 0.5f, y), ImVec2(bx + bw + 3.5f * k, y), L.rule);
            const ImVec2 ts = text_size(L.mono, afs, bar_labels[i]);
            text_at(dl, L.mono, afs, ImVec2(bx + bw + 6.0f * k, std::clamp(y - ts.y * 0.5f, ay - ts.y * 0.5f, ay + ah - ts.y * 0.5f)),
                    L.faint, bar_labels[i]);
        }
    }
    // Hover: the cell under the pointer, by its indices and value.
    const ImVec2 m = ImGui::GetIO().MousePos;
    if (ImGui::IsWindowHovered() && m.x >= ax && m.y >= ay && m.x < ax + aw && m.y < ay + ah) {
        const int col = std::min(G.source_cols - 1, (int)((m.x - ax) / sx));
        const int row = std::min(G.source_rows - 1, (int)((m.y - ay) / sy));
        const int gr = std::min(G.rows - 1, row / G.stride_r), gc = std::min(G.cols - 1, col / G.stride_c);
        if (!G.values.empty())
            ImGui::SetTooltip("(%d, %d)\nvalue %s", row, col, spec_fmt4(G.values[(size_t)gr * G.cols + gc]).c_str());
        else {
            const uint8_t* q = &G.rgba[((size_t)gr * G.cols + gc) * 4];
            ImGui::SetTooltip("(%d, %d)\nrgba %u %u %u %u", row, col, q[0], q[1], q[2], q[3]);
        }
    }
}

// The colour key of a points panel: the colormap's scale, named by the
// colour binding, in a column at the right of the content; for class labels
// a legend over the plot's top right corner instead, as a series' legend
// sits, so the cloud keeps the panel's width.
float points_key_width(const PreparedPoints& P, const SpecLook& L) {
    if (P.rgba.empty() || !P.classes.empty()) return 0.0f;
    const float afs = L.mono_size - 1.0f;
    return std::round(std::max(60.0f, text_size(L.mono, afs, "-0.000e-0").x + 26.0f) * L.text_scale);
}

// The legend of the classes: the colour binding's name over a swatch and the
// label of each class (the live points view's classes), in a box at the top
// right of `r`.
void draw_class_legend(const PreparedPoints& P, const ImRect& r, const SpecLook& L) {
    if (P.classes.empty()) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float afs = L.mono_size - 1.0f, s = L.text_scale;
    const float sw = std::round(7.0f * s), gap = std::round(5.0f * s), pad = std::round(6.0f * s);
    const float line = std::round(L.mono_size + 5.0f * s);
    float w = text_size(L.mono, afs, P.color_name).x;
    for (int c : P.classes) w = std::max(w, sw + gap + text_size(L.mono, L.mono_size, std::to_string(c)).x);
    const size_t rows = std::min<size_t>(P.classes.size(), (size_t)std::max(0.0f, (r.GetHeight() - 2.0f * pad) / line - 1.0f));
    const ImVec2 b1(r.Max.x - std::round(8.0f * s), r.Min.y + std::round(8.0f * s));
    const ImVec2 b0(b1.x - std::round(w + 2.0f * pad), b1.y);
    const ImVec2 b2(b1.x, b1.y + std::round(2.0f * pad + line * (float)(rows + 1)));
    dl->AddRectFilled(b0, b2, with_alpha(L.ground, 230));
    dl->AddRect(b0, b2, L.rule);
    float y = b0.y + pad;
    text_at(dl, L.mono, afs, ImVec2(b0.x + pad, y), L.faint, P.color_name);
    y += line;
    for (size_t i = 0; i < rows; ++i) {
        const std::string t = std::to_string(P.classes[i]);
        const ImVec2 ts = text_size(L.mono, L.mono_size, t);
        const float cy = std::round(y + ts.y * 0.5f);
        dl->AddRectFilled(ImVec2(b0.x + pad, cy - sw * 0.5f), ImVec2(b0.x + pad + sw, cy + sw * 0.5f),
                          (ImU32)spec_class_color(P.classes[i]));
        text_at(dl, L.mono, L.mono_size, ImVec2(b0.x + pad + sw + gap, y), L.secondary, t);
        y += line;
    }
}

void draw_points_key(const PreparedPoints& P, const ImRect& r, const SpecLook& L) {
    if (P.rgba.empty() || !P.classes.empty()) return;
    push_plot_style(L);
    ImGui::SetCursorScreenPos(r.Min);
    ImPlot::ColormapScale(P.color_name.c_str(), P.cmin, P.cmax, r.GetSize(), "%.3g", 0, implot_colormap(P.colormap));
    pop_plot_style();
}

// The view scale that makes the 3-D box, as the current rotation shows it,
// fill `fill` of the plot's width and height: the box's corners projected
// as ImPlot3D projects them (view scale times the rotated corner), against
// ImPlot3D's view scale of min(w, h) / 1.8 at a factor of 1.
float points3d_view_factor(const ImVec2& size, float fill) {
    ImPlot3DPlot* plot = ImPlot3D::GetCurrentPlot();
    if (!plot || size.x <= 0.0f || size.y <= 0.0f) return 1.0f;
    const ImPlot3DPoint scale = plot->GetBoxScale();
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < 8; ++i) {
        const ImPlot3DPoint corner((i & 1 ? 0.5 : -0.5) * scale.x, (i & 2 ? 0.5 : -0.5) * scale.y,
                                   (i & 4 ? 0.5 : -0.5) * scale.z);
        const ImPlot3DPoint q = plot->Rotation * corner;
        mx = std::max(mx, std::fabs(q.x));
        my = std::max(my, std::fabs(q.y));
    }
    if (mx <= 0.0 || my <= 0.0) return 1.0f;
    const double want = std::min(fill * size.x / (2.0 * mx), fill * size.y / (2.0 * my));
    return (float)std::clamp(want / (std::min(size.x, size.y) / 1.8), 0.5, 4.0);
}

void draw_points2d(SpecPanelView& v, const ImRect& c, const SpecLook& L) {
    const PreparedPanel& p = v.data;
    const PreparedPoints& P = p.points;
    const float key = points_key_width(P, L);
    const ImRect plot(c.Min, ImVec2(c.Max.x - key, c.Max.y));
    push_plot_style(L);
    const ImPlotFlags flags = plot_flags(false) | (selection_input(v, plot) ? ImPlotFlags_NoInputs : 0);
    if (ImPlot::BeginPlot("##spec.points", plot.GetSize(), flags)) {
        ImPlot::SetupAxes(label_or_null(p.x_label), label_or_null(p.y_label));
        if (v.fit_frames > 0) {
            double x0, x1, y0, y1;
            const std::vector<double> xs(P.x.begin(), P.x.end()), ys(P.y.begin(), P.y.end());
            if (fit_limits(xs, false, 0.04, &x0, &x1) && fit_limits(ys, false, 0.04, &y0, &y1))
                ImPlot::SetupAxesLimits(x0, x1, y0, y1, ImPlotCond_Always);
        }
        // Squares of the live views' size, drawn over the plot area.
        const float base = caliper_live_views::point_size_px((uint64_t)P.x.size(), L.text_scale);
        ImPlot::PushPlotClipRect();
        ImDrawList* dl = ImPlot::GetPlotDrawList();
        for (size_t i = 0; i < P.x.size(); ++i) {
            const ImVec2 q = ImPlot::PlotToPixels(P.x[i], P.y[i]);
            const float h = 0.5f * base * (P.scale.empty() ? 1.0f : P.scale[i]);
            const ImU32 col = P.rgba.empty() ? L.flat_point : (ImU32)P.rgba[i];
            dl->AddRectFilled(ImVec2(q.x - h, q.y - h), ImVec2(q.x + h, q.y + h), col);
        }
        ImPlot::PopPlotClipRect();
        // A dummy item keeps ImPlot's fit and hover working with the points.
        ImPlotSpec none;
        none.Flags = ImPlotItemFlags_NoLegend | ImPlotItemFlags_NoFit;
        ImPlot::PlotDummy("##points", none);
        selection_gesture(v, L, true);
        if (v.spec_panel && v.instruments) draw_plot_annotations(*v.spec_panel, *v.instruments);   // spec_annotations.h
        ImPlot::EndPlot();
    }
    pop_plot_style();
    if (key > 0.0f) draw_points_key(P, ImRect(ImVec2(plot.Max.x + 4.0f, c.Min.y), c.Max), L);
    draw_class_legend(P, c, L);
}

void draw_points3d(SpecPanelView& v, const ImRect& c, const SpecLook& L) {
    const PreparedPanel& p = v.data;
    const PreparedPoints& P = p.points;
    const float key = points_key_width(P, L);
    const ImRect plot(c.Min, ImVec2(c.Max.x - key, c.Max.y));
    push_plot3d_style(L);
    ImGui::PushFont(L.mono, L.mono_size);   // the box's ticks are drawn at a slant; a step larger
    ImPlot3D::PushStyleVar(ImPlot3DStyleVar_ViewScaleFactor, 1.0f);
    if (ImPlot3D::BeginPlot("##spec.points3d", plot.GetSize(),
                            ImPlot3DFlags_NoTitle | ImPlot3DFlags_NoLegend | ImPlot3DFlags_NoMouseText |
                                ImPlot3DFlags_NoMenus)) {
        ImPlot3D::SetupAxes(label_or_null(p.x_label), label_or_null(p.y_label), label_or_null(p.z_label));
        // The box fills the panel, leaving room for its ticks and labels, as
        // it is turned: the cloud fills its view on opening and on resize.
        ImPlot3D::GetStyle().ViewScaleFactor = points3d_view_factor(plot.GetSize(), 0.84f);
        if (v.fit_frames > 0) {
            double x0, x1, y0, y1, z0, z1;
            const std::vector<double> xs(P.x.begin(), P.x.end()), ys(P.y.begin(), P.y.end()), zs(P.z.begin(), P.z.end());
            if (fit_limits(xs, false, 0.02, &x0, &x1) && fit_limits(ys, false, 0.02, &y0, &y1) &&
                fit_limits(zs, false, 0.02, &z0, &z1))
                ImPlot3D::SetupAxesLimits(x0, x1, y0, y1, z0, z1, ImPlot3DCond_Always);
        }
        const float base = caliper_live_views::point_size_px((uint64_t)P.x.size(), L.text_scale);
        ImPlot3DSpec spec;
        spec.Marker = ImPlot3DMarker_Square;
        spec.MarkerSize = 0.5f * base;
        spec.FillAlpha = 1.0f;
        spec.LineWeight = 0.0f;
        std::vector<ImU32> cols;
        std::vector<float> sizes;
        if (!P.rgba.empty()) {
            cols.assign(P.rgba.begin(), P.rgba.end());
            spec.MarkerFillColors = cols.data();
            spec.MarkerLineColors = cols.data();
        } else {
            spec.MarkerFillColor = spec.MarkerLineColor = ImGui::ColorConvertU32ToFloat4(L.flat_point);
        }
        if (!P.scale.empty()) {
            sizes.resize(P.scale.size());
            for (size_t i = 0; i < sizes.size(); ++i) sizes[i] = 0.5f * base * P.scale[i];
            spec.MarkerSizes = sizes.data();
        }
        ImPlot3D::PlotScatter("##points", P.x.data(), P.y.data(), P.z.data(), (int)P.x.size(), spec);
        ImPlot3D::EndPlot();
    }
    ImPlot3D::PopStyleVar();
    ImGui::PopFont();
    pop_plot3d_style();
    if (key > 0.0f) draw_points_key(P, ImRect(ImVec2(plot.Max.x + 4.0f, c.Min.y), c.Max), L);
    draw_class_legend(P, c, L);
}

void draw_table(SpecPanelView& v, const ImRect& c, const SpecLook& L) {
    const PreparedTable& T = v.data.table;
    const int ncols = (int)std::min<size_t>(T.columns.size(), 64);
    if (ncols <= 0) return;
    // Each column as wide as its head and its widest cell among the first
    // rows, the columns together at the left, and the width they leave empty
    // on the right: a register's columns sit side by side.
    const float pad = std::round(18.0f * L.text_scale);
    const ImVec2 cell = ImGui::GetStyle().CellPadding;
    std::vector<float> widths((size_t)ncols);
    float total = 0.0f;
    for (int i = 0; i < ncols; ++i) {
        float w = text_size(L.ui, L.small_size, T.columns[(size_t)i]).x * 1.25f + 10.0f * L.text_scale;
        for (size_t r = 0; r < std::min<size_t>(T.cells.size(), 200); ++r)
            w = std::max(w, text_size(L.mono, L.mono_size, T.cells[r][(size_t)i]).x);
        widths[(size_t)i] = std::round(w + pad);
        total += widths[(size_t)i] + 2.0f * cell.x;
    }
    // Whole rows: the head and as many rows as fit, so the last row shown is
    // never cut by the panel's edge.
    const float row = sternbild::metric::row;
    const float avail = c.GetHeight();
    const int fit = std::max(0, (int)std::floor((avail - row) / row));
    const bool scrolls = (size_t)fit < T.cells.size();
    const float height = std::min(avail, row * (float)(1 + std::min<size_t>((size_t)fit, T.cells.size())) + 1.0f);
    const float width = std::min(c.GetWidth(), total + (scrolls ? ImGui::GetStyle().ScrollbarSize : 0.0f));
    ImGui::SetCursorScreenPos(c.Min);
    if (sternbild::begin_table("##spec.table", ncols, ImVec2(width, height), ImGuiTableFlags_SizingFixedFit)) {
        for (int i = 0; i < ncols; ++i) {
            const std::string head = (T.numeric[(size_t)i] ? ">" : "") + T.columns[(size_t)i];
            ImGui::TableSetupColumn(head.c_str(), ImGuiTableColumnFlags_WidthFixed, widths[(size_t)i]);
        }
        ImGui::TableSetupScrollFreeze(0, 1);
        sternbild::table_header_row();
        // A column that a series also draws carries its colour key: a swatch
        // beside its head.
        if (v.key) {
            ImGuiTable* table = ImGui::GetCurrentTable();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float sw = std::round(7.0f * L.text_scale);
            for (int i = 0; i < ncols && table; ++i) {
                ImU32 col;
                if (!v.key->find(T.columns[(size_t)i], &col)) continue;
                const ImGuiTableColumn& tc = table->Columns[i];
                std::string upper = T.columns[(size_t)i];
                for (char& ch : upper) ch = (char)std::toupper((unsigned char)ch);
                const float size = sternbild::font_size::ui_small;
                const float lw = sternbild::text_tracked_width(sternbild::fonts().ui, size, upper.c_str(),
                                                               sternbild::metric::tracking * size);
                const float cy = std::round((table->RowPosY1 + table->RowPosY2) * 0.5f);
                const float x = T.numeric[(size_t)i] ? tc.WorkMaxX - lw - sw - 6.0f : tc.WorkMinX + lw + 6.0f;
                dl->AddRectFilled(ImVec2(x, cy - sw * 0.5f), ImVec2(x + sw, cy + sw * 0.5f), col);
            }
        }
        ImGui::PushFont(L.mono, L.mono_size);
        ImGuiListClipper clip;
        clip.Begin((int)T.cells.size(), row);
        while (clip.Step()) {
            for (int r = clip.DisplayStart; r < clip.DisplayEnd; ++r) {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, row);
                const auto& row = T.cells[(size_t)r];
                for (int i = 0; i < ncols; ++i) {
                    ImGui::TableNextColumn();
                    const std::string& s = row[(size_t)i];
                    if (T.numeric[(size_t)i]) {
                        const float w = ImGui::CalcTextSize(s.c_str()).x;
                        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
                    }
                    ImGui::TextUnformatted(s.c_str());
                }
            }
        }
        ImGui::PopFont();
        sternbild::end_table();
    }
}

void draw_text(SpecPanelView& v, const ImRect& c, const SpecLook& L) {
    const PreparedText& X = v.data.text;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float y = c.Min.y;
    // A bound text: formatted at each frame from the values at the cursor
    // (spec_annotations.h), with the bound value as its field `value`.
    if (X.bound && v.spec_panel && v.spec_panel->options.bound_text) {
        static const SpecInstrumentHost no_session;
        const SpecInstrumentHost& host = v.instruments ? *v.instruments : no_session;
        std::map<std::string, nlohmann::json> extra;
        if (v.spec_panel->bind.count("value")) extra["value"] = X.value_json;
        const std::string t = spec_bound_text_string(*v.spec_panel->options.bound_text, host, extra);
        if (!X.text.empty()) {
            text_at(dl, L.ui, L.ui_size, ImVec2(c.Min.x, y), L.secondary, fit_text(L.ui, L.ui_size, X.text, c.GetWidth()));
            y += L.ui_size + 6.0f * L.text_scale;
        }
        const float big = std::round(L.mono_size * 1.6f);
        text_at(dl, L.mono, big, ImVec2(c.Min.x, y), L.text, fit_text(L.mono, big, t, c.GetWidth()));
        return;
    }
    if (!X.text.empty()) {
        ImFont* f = font_or_current(L.ui);
        const float size = std::round(L.ui_size * (X.has_value ? 1.0f : 1.25f));
        const float wrap = std::min(c.GetWidth(), f->CalcTextSizeA(size, FLT_MAX, 0.0f, "0").x * 80.0f);
        dl->AddText(f, size, ImVec2(c.Min.x, y), X.has_value ? L.secondary : L.text, X.text.c_str(), nullptr, wrap);
        y += f->CalcTextSizeA(size, FLT_MAX, wrap, X.text.c_str()).y + 8.0f * L.text_scale;
    }
    if (X.has_value) {
        const float big = std::round(L.mono_size * 2.6f);
        const std::string t = fit_text(L.mono, big, X.value, c.GetWidth());
        text_at(dl, L.mono, big, ImVec2(c.Min.x, y), L.text, t);
    }
}

void draw_state(const PanelState& st, const ImRect& c, const SpecLook& L) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const std::string label = fit_text(L.ui, L.ui_size, st.label, c.GetWidth());
    const std::string detail = fit_text(L.mono, L.mono_size - 1.0f, st.detail, c.GetWidth());
    const ImVec2 a = text_size(L.ui, L.ui_size, label), b = text_size(L.mono, L.mono_size - 1.0f, detail);
    const float gap = detail.empty() ? 0.0f : 4.0f * L.text_scale;
    const float top = c.Min.y + std::max(0.0f, (c.GetHeight() - a.y - gap - b.y) * 0.5f);
    const float cx = (c.Min.x + c.Max.x) * 0.5f;
    text_at(dl, L.ui, L.ui_size, ImVec2(cx - a.x * 0.5f, top), L.secondary, label);
    if (!detail.empty()) text_at(dl, L.mono, L.mono_size - 1.0f, ImVec2(cx - b.x * 0.5f, top + a.y + gap), L.faint, detail);
}

// An empty state with a line of plain words under its label, in the house
// face: how to start, where the builder shows it.
void draw_hint(const std::string& label_in, const std::string& hint_in, const ImRect& c, const SpecLook& L) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const std::string label = fit_text(L.ui, L.ui_size, label_in, c.GetWidth());
    const std::string hint = fit_text(L.ui, L.ui_size, hint_in, c.GetWidth());
    const ImVec2 a = text_size(L.ui, L.ui_size, label), b = text_size(L.ui, L.ui_size, hint);
    const float gap = 6.0f * L.text_scale;
    const float top = c.Min.y + std::max(0.0f, (c.GetHeight() - a.y - gap - b.y) * 0.5f);
    const float cx = (c.Min.x + c.Max.x) * 0.5f;
    text_at(dl, L.ui, L.ui_size, ImVec2(cx - a.x * 0.5f, top), L.secondary, label);
    text_at(dl, L.ui, L.ui_size, ImVec2(cx - b.x * 0.5f, top + a.y + gap), L.faint, hint);
}

}  // namespace

// ---- the look ----------------------------------------------------------------------

SpecLook house_spec_look(float text_scale) {
    SpecLook l;
    const float s = text_scale > 0.0f ? text_scale : 1.0f;
    l.ui = sternbild::fonts().ui;
    l.mono = sternbild::fonts().mono;
    l.ui_size = sternbild::font_size::ui * s;
    l.small_size = sternbild::font_size::ui_small * s;
    l.mono_size = sternbild::font_size::ui_small * s;   // the live views' info and axes size
    l.text_scale = s;
    l.ground = sternbild::col::ground;
    l.deep = sternbild::col::deep;
    l.plate = sternbild::col::plate;
    l.plate_hover = sternbild::col::plate_hover;
    l.rule = sternbild::col::rule;
    l.rule_faint = sternbild::col::rule_faint;
    l.text = sternbild::col::parchment;
    l.secondary = sternbild::col::secondary;
    l.faint = sternbild::col::faint;
    l.accent = sternbild::col::bronze;
    l.line = sternbild::col::parchment;
    return l;
}

ImU32 spec_palette_color(size_t i) {
    static const ImU32 kHouse[] = {sternbild::col::vehicle, sternbild::col::dog, sternbild::col::bird,
                                   sternbild::col::cat, sternbild::col::other};
    constexpr size_t n = sizeof(kHouse) / sizeof(kHouse[0]);
    return i < n ? kHouse[i] : (ImU32)caliper_live_views::kPalette[(i - n) % 16];
}

void SpecColorKey::reset(const spec::Spec& s) {
    index_.clear();
    for (const auto& p : s.panels) {
        if (p.primitive != spec::Primitive::series) continue;
        const auto y = p.bind.find("y");
        if (y == p.bind.end()) continue;
        for (const auto& b : y->second.bindings) color(spec_binding_name(s, b));
    }
}

ImU32 SpecColorKey::color(const std::string& name) {
    auto it = index_.find(name);
    if (it == index_.end()) it = index_.emplace(name, index_.size()).first;
    return spec_palette_color(it->second);
}

bool SpecColorKey::find(const std::string& name, ImU32* out) const {
    const auto it = index_.find(name);
    if (it == index_.end()) return false;
    *out = spec_palette_color(it->second);
    return true;
}

// ---- textures -----------------------------------------------------------------------

SpecTextures::~SpecTextures() {
    // The backend destroys what is still registered when it shuts down; a
    // context that is gone has nothing left to unregister from.
    const bool ctx = ImGui::GetCurrentContext() != nullptr;
    for (ImTextureData* t : live_) {
        if (ctx) ImGui::UnregisterUserTexture(t);
        IM_DELETE(t);
    }
    for (ImTextureData* t : dying_) {
        if (ctx) ImGui::UnregisterUserTexture(t);
        IM_DELETE(t);
    }
}

ImTextureData* SpecTextures::create(int w, int h, const uint8_t* rgba) {
    if (w <= 0 || h <= 0 || !rgba || !ImGui::GetCurrentContext()) return nullptr;
    ImTextureData* t = IM_NEW(ImTextureData)();
    t->Create(ImTextureFormat_RGBA32, w, h);
    std::memcpy(t->GetPixels(), rgba, (size_t)w * (size_t)h * 4);
    t->UseColors = true;
    ImGui::RegisterUserTexture(t);
    live_.push_back(t);
    return t;
}

void SpecTextures::release(ImTextureData* tex) {
    if (!tex) return;
    const auto it = std::find(live_.begin(), live_.end(), tex);
    if (it == live_.end()) return;
    live_.erase(it);
    tex->WantDestroyNextFrame = true;
    dying_.push_back(tex);
}

void SpecTextures::collect() {
    for (size_t i = 0; i < dying_.size();) {
        ImTextureData* t = dying_[i];
        // Destroyed by the backend, or never seen by one.
        const bool gone = t->Status == ImTextureStatus_Destroyed ||
                          (t->TexID == ImTextureID_Invalid && t->Status != ImTextureStatus_OK);
        if (gone && ImGui::GetCurrentContext()) {
            ImGui::UnregisterUserTexture(t);
            IM_DELETE(t);
            dying_.erase(dying_.begin() + (ptrdiff_t)i);
        } else {
            ++i;
        }
    }
}

// ---- sources ----------------------------------------------------------------------------

SessionPanelSource::SessionPanelSource(caliper_builder::Session& session, caliper_builder::ResolveLimits limits)
    : session_(session), limits_(limits) {}

// What references and kept steps read now: the session's values, its kept
// steps and its cursor.
static caliper_builder::ChainContext session_chain(caliper_builder::Session& session) {
    const caliper_builder::CursorState c = session.history().cursor();
    return caliper_builder::ChainContext{&session.values(), &session.history(), c.step, c.live};
}

PanelData SessionPanelSource::resolve(const spec::Spec& s, std::string_view panel_id) {
    const auto open = session_.open_spec();
    if (open && open->spec == s) return session_.resolve(panel_id);
    const caliper_builder::ChainContext chain = session_chain(session_);
    return caliper_builder::resolve_panel(s, panel_id, session_.store(), limits_, &session_.data_sources(), &chain);
}

uint64_t SessionPanelSource::revision() const { return session_.catalog().revision(); }

uint64_t SessionPanelSource::generation(const spec::Spec& s, std::string_view panel_id) {
    const caliper_builder::ChainContext chain = session_chain(session_);
    return caliper_builder::panel_generation(s, panel_id, session_.data_sources(), &chain);
}

// ---- one panel --------------------------------------------------------------------------

float spec_panel_strip_height(const SpecLook& L, bool show_title, float panel_height) {
    return show_title ? std::min(panel_height, std::round(L.ui_size + 9.0f * L.text_scale)) : 0.0f;
}

namespace {

// The mark of a panel drawn from a preview: "PREVIEW" in the small UI face,
// uppercase and tracked like the house's labels, in bronze. Returns its width.
float preview_tag_width(const SpecLook& L) {
    return sternbild::text_tracked_width(L.ui, L.small_size, "PREVIEW", std::round(L.small_size * 0.08f));
}

void preview_tag(ImDrawList* dl, const SpecLook& L, ImVec2 pos) {
    sternbild::text_tracked(dl, L.ui, L.small_size, pos, L.accent, "PREVIEW", std::round(L.small_size * 0.08f));
}

}  // namespace

SpecPanelFrame draw_spec_panel(SpecPanelView& view, const ImRect& rect, const SpecLook& L, SpecTextures& textures,
                               float band) {
    SpecPanelFrame f;
    f.id = view.data.id;
    f.rect = rect;
    f.preview = view.preview;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float s = L.text_scale;
    dl->AddRectFilled(rect.Min, rect.Max, L.ground);

    // The title strip: the title on the left, the info string in mono on the
    // right, cut first when they meet. A preview's strip carries the bronze
    // PREVIEW tag after its title and a bronze rule under it.
    // A tab has no strip: its tab carries the title and the tab bar its info.
    const float hh = spec_panel_strip_height(L, view.show_title, rect.GetHeight());
    f.header = ImRect(rect.Min, ImVec2(rect.Max.x, rect.Min.y + hh));
    const float pad = std::round(8.0f * s);
    if (hh > 0.0f) {
        dl->AddRectFilled(f.header.Min, f.header.Max, L.plate);
        dl->AddRectFilled(ImVec2(rect.Min.x, f.header.Max.y - 1.0f), ImVec2(rect.Max.x, f.header.Max.y),
                          view.preview ? L.accent : L.rule_faint);
    }
    const float tag_w = view.preview && hh > 0.0f ? preview_tag_width(L) + std::round(12.0f * s) : 0.0f;
    const float avail = rect.GetWidth() - 2.0f * pad - tag_w;
    const std::string title = hh > 0.0f ? fit_text(L.ui, L.ui_size, view.data.title, avail) : std::string();
    const ImVec2 ts = text_size(L.ui, L.ui_size, title);
    if (!title.empty())
        text_at(dl, L.ui, L.ui_size, ImVec2(rect.Min.x + pad, rect.Min.y + (hh - ts.y) * 0.5f), L.text, title);
    float title_end = rect.Min.x + pad + ts.x;
    if (tag_w > 0.0f && avail > 0.0f) {
        const float th = text_size(L.ui, L.small_size, "P").y;
        preview_tag(dl, L, ImVec2(std::round(title_end + 12.0f * s), std::round(rect.Min.y + (hh - th) * 0.5f)));
        title_end += tag_w;
    }
    // The header's info, or the error that the panel shows over the last good
    // data it keeps.
    const bool held = !view.data.notice.ready();
    if (hh > 0.0f && view.data.state.ready() && (held || !view.data.info.empty())) {
        const float left = title_end + 14.0f * s, right = rect.Max.x - pad;
        const std::string line = held ? view.data.notice.label + "  " + view.data.notice.detail : view.data.info;
        const std::string info = fit_text(L.mono, L.mono_size, line, right - left);
        if (!info.empty()) {
            const ImVec2 is = text_size(L.mono, L.mono_size, info);
            text_at(dl, L.mono, L.mono_size, ImVec2(right - is.x, rect.Min.y + (hh - is.y) * 0.5f),
                    held ? L.accent : L.faint, info);
        }
    }

    // The band an overlay keeps free under the strip, then the content.
    const float bh = std::max(0.0f, std::min(std::round(band), rect.Max.y - f.header.Max.y));
    f.band = ImRect(ImVec2(rect.Min.x, f.header.Max.y), ImVec2(rect.Max.x, f.header.Max.y + bh));
    const float top = f.band.Max.y;
    const float py = std::round(6.0f * s);
    f.content = ImRect(ImVec2(rect.Min.x + pad, top + py), ImVec2(rect.Max.x - pad, rect.Max.y - py));
    if (view.data.primitive == spec::Primitive::table) {
        // A table runs to the panel's bottom edge, under the title's margin.
        f.content = ImRect(ImVec2(rect.Min.x + pad, top + 1.0f), ImVec2(rect.Max.x - pad, rect.Max.y));
    }
    if (f.content.GetWidth() < 8.0f || f.content.GetHeight() < 8.0f) return f;

    if (!view.data.state.ready()) {
        draw_state(view.data.state, f.content, L);
        ImGui::SetCursorScreenPos(f.content.Min);
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
    } else {
        ImGui::PushClipRect(f.content.Min, f.content.Max, true);
        ImGui::SetCursorScreenPos(f.content.Min);
        switch (view.data.primitive) {
            case spec::Primitive::series: draw_series(view, f.content, L); break;
            case spec::Primitive::histogram: draw_histogram(view, f.content, L); break;
            case spec::Primitive::heatmap: draw_grid(view, f.content, L, textures, false); break;
            case spec::Primitive::image: draw_grid(view, f.content, L, textures, true); break;
            case spec::Primitive::points:
                if (view.data.points.three_d) draw_points3d(view, f.content, L);
                else draw_points2d(view, f.content, L);
                break;
            case spec::Primitive::table: draw_table(view, f.content, L); break;
            case spec::Primitive::text: draw_text(view, f.content, L); break;
            case spec::Primitive::scene:
                // The live scene of the attached script; a scene the script
                // does not send shows as an offline source, as a live: source does.
                if (!view.scene || !view.scene(f.content))
                    draw_state({PanelStateKind::offline, "Source offline",
                                view.data.scene.empty() ? std::string() : "live:" + view.data.scene},
                               f.content, L);
                break;
        }
        ImGui::PopClipRect();
    }
    // An item at the content's corner, so the child window's bounds come from
    // an item and not from a moved cursor (ImGui reports that as an error).
    ImGui::SetCursorScreenPos(f.content.Min);
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
    if (view.fit_frames > 0) --view.fit_frames;
    return f;
}

// ---- the runtime ------------------------------------------------------------------------

bool spec_scene_event(const char* type, const std::string& panel_id, const std::string& scene,
                      const std::string& detail, caliper_builder::Event* out) {
    nlohmann::json fields = nlohmann::json::parse(detail, nullptr, false);
    if (!out || !fields.is_object()) return false;
    caliper_builder::Event ev;
    ev.kind = std::strcmp(type, "pick") == 0 ? caliper_builder::EventKind::pick : caliper_builder::EventKind::select;
    ev.panel = panel_id;
    ev.view = scene;
    ev.origin = "ui";
    if (ev.kind == caliper_builder::EventKind::select && fields.empty())
        fields = {{"entities", nlohmann::json::array()}, {"truncated", false}};   // the selection was cleared
    ev.fields = std::move(fields);
    *out = std::move(ev);
    return true;
}

// The scene of a panel's entities (design/instrument-interface.md, Scene Entities): what they
// built last, and the scene view that the panel owns, on a backend of the graphics layer from the
// runtime's scene layer, made at the first draw.
struct EntityPanel {
    SpecScene scene;
    std::unique_ptr<caliper_gfx::Backend> gfx;   // owned; null when the view borrows a backend
    std::unique_ptr<caliper_live_views::SceneView> view;
    std::vector<std::string> drawn;               // the paths written into the view
    std::vector<std::string> marks_drawn;         // the annotations' paths written into the view
    std::string marks_key;                        // what the annotations drew last
    bool dirty = true;                            // built again since the view took it
    std::string notice;                           // why no view could start
    caliper_scene::StartingCamera start;          // the panel's view_camera, applied when it changes
    bool start_set = false;
};

// The gesture and pick state of each scene panel, and the scenes of panels with entities, by id.
struct SpecRuntime::SceneStates {
    std::map<std::string, caliper_live_views::SceneRectState> by_panel;
    std::map<std::string, std::unique_ptr<EntityPanel>> entities;
    EntityPanel& entity(const std::string& id) {
        std::unique_ptr<EntityPanel>& p = entities[id];
        if (!p) p = std::make_unique<EntityPanel>();
        return *p;
    }
};

namespace {

// The programs of the spec's scene panels (draw/draw_host.h, design/programmable-graphics.md):
// one host for the app, as for stages, with the linked selection that every program reads.
struct ProgramHost {
    caliper_draw::DrawHost host;
    std::map<std::string, caliper_draw::Selected> picked;   // the newest program pick of each panel
    std::int64_t last_step = -1;
    std::uint64_t moves = 0;
    std::function<std::string()> workspace;                 // the app's workspace folder, for its pieces
    std::map<std::string, std::string> seen_picks;          // each panel's last pick that the programs took
    // The runtime whose event sink, values and history the host and the stage host read: the
    // last that gave them. It takes them back when it goes, so that no host of the process reads
    // the history or the values of a session that is gone.
    const void* owner = nullptr;
};

// One status item for each program that the scene panels draw: half "draw" and its draw object.
std::vector<caliper_builder::ProgramStatus> drawn_items(const caliper_draw::DrawHost& host) {
    std::vector<caliper_builder::ProgramStatus> out;
    for (const auto& [id, s] : host.status()) {
        caliper_builder::ProgramStatus st;
        st.id = id;
        st.kind = "piece";
        st.host = "viewer";
        st.half = "draw";
        caliper_builder::DrawStatus d;
        d.state = caliper_draw::draw_status_json(s).value("state", std::string("waiting"));
        if (s.frame_ms >= 0) d.frame_ms = s.frame_ms;
        d.frame_limit_ms = s.frame_limit_ms;
        d.memory_bytes = s.memory_bytes;
        d.memory_limit_bytes = s.memory_limit_bytes;
        if (s.resource_step >= 0) d.resource_step = s.resource_step;
        if (!s.error.ok()) {
            d.error = caliper_builder::ProgramError{s.error.code, s.error.message, s.error.line, s.error.column};
            d.error_where = s.error.where;
        }
        if (s.pass >= 0) st.pass = static_cast<std::uint32_t>(s.pass);
        st.state = d.state;
        st.draw = std::move(d);
        out.push_back(std::move(st));
    }
    return out;
}

ProgramHost& program_host() {
    static ProgramHost* h = [] {
        auto* p = new ProgramHost();
        // A pin resolves to an inline piece of the spec, a piece of the app's workspace or one of
        // the built-in library.
        p->host.set_resolver([p](const spec::Spec& s, const spec::Program& program, spec::Piece* out) {
            caliper_draw::Verdict v;
            if (!program.piece) {
                v.code = caliper_builder::programs::kPieceMissing;
                v.message = "The program " + program.id + " pins no piece.";
                return v;
            }
            caliper_builder::FoundPiece f =
                caliper_builder::resolve_spec_pin(caliper_builder::PieceLibrary::builtin(),
                                             p->workspace ? p->workspace() : std::string(), s, *program.piece);
            if (f.error) {
                v.code = f.error.code;
                v.message = f.error.message;
                v.where = "piece";
                return v;
            }
            *out = std::move(f.piece);
            return v;
        });
        // program.status shows each draw half through the stage host's items.
        app_stage_host_impl()->set_drawn([p] { return drawn_items(p->host); });
        return p;
    }();
    return *h;
}

// The cursor of the step history for the programs, before a panel draws them.
void give_program_cursor(const caliper_builder::StepHistory* history) {
    if (!history) return;
    ProgramHost& p = program_host();
    const caliper_builder::CursorState c = history->cursor();
    if (c.step != p.last_step && !c.live) ++p.moves;
    p.last_step = c.step;
    p.host.set_cursor({c.step, c.latest, c.live, p.moves});
}

// A pick event of a panel: a program's item becomes the panel's entry of the linked selection,
// the newest of all; another pick of the panel, or a cleared selection, removes the entry.
void note_program_pick(const std::string& panel, const caliper_builder::Event& ev) {
    ProgramHost& p = program_host();
    const bool cleared = ev.kind == caliper_builder::EventKind::select && ev.fields.value("entities", nlohmann::json::array()).empty();
    if (ev.kind != caliper_builder::EventKind::pick && !cleared) return;
    p.picked.erase(panel);
    const auto program = ev.fields.find("program");
    if (!cleared && program != ev.fields.end() && program->is_string()) {
        caliper_draw::Selected sel;
        bool found = false;
        if (caliper_draw::DrawFeed* feed = p.host.feed(panel))
            for (const caliper_draw::PanelProgram& pp : feed->programs())
                if (pp.instance.program == program->get<std::string>()) {
                    sel.pick = pp.pick_id;
                    found = true;
                }
        const auto item = ev.fields.find("item");
        if (found && item != ev.fields.end() && item->is_number_integer() && item->get<std::int64_t>() >= 0 &&
            item->get<std::int64_t>() <= 0xffffffffll) {
            sel.item = std::uint32_t(item->get<std::int64_t>());
            if (auto ix = ev.fields.find("index"); ix != ev.fields.end() && ix->is_array() && ix->size() == 4)
                for (std::size_t k = 0; k < 4; ++k)
                    if ((*ix)[k].is_number_integer()) sel.index[k] = std::int32_t((*ix)[k].get<std::int64_t>());
            for (auto& [other, s] : p.picked) s.flags &= ~1u;
            sel.flags = 1u;   // the newest pick
            p.picked[panel] = sel;
        }
    }
    std::vector<caliper_draw::Selected> all;
    for (const auto& [id, s] : p.picked)
        if (all.size() < caliper_builder::programs::kMaxSelected) all.push_back(s);
    p.host.set_selection(std::move(all));
}

// A pick that a command made, as selection.set does, reaches the programs as a click does: the
// panel's last pick in the value store, once each time it changes.
void sync_program_pick(const std::string& panel, const caliper_builder::ValueStore* values) {
    if (!values) return;
    const std::optional<caliper_builder::PickFields> f = values->last_pick(panel);
    const nlohmann::json fields = f ? caliper_builder::pick_fields_json(*f) : nlohmann::json();
    std::string& seen = program_host().seen_picks[panel];
    const std::string key = fields.is_null() ? std::string() : fields.dump();
    if (key == seen) return;
    seen = key;
    caliper_builder::Event ev;
    if (f) {
        ev.kind = caliper_builder::EventKind::pick;
        ev.fields = fields;
    } else {
        ev.kind = caliper_builder::EventKind::select;
        ev.fields = {{"entities", nlohmann::json::array()}};
    }
    note_program_pick(panel, ev);
}

bool same_starting(const caliper_scene::StartingCamera& a, const caliper_scene::StartingCamera& b) {
    auto vec = [](const std::optional<caliper_scene::Vec3>& x, const std::optional<caliper_scene::Vec3>& y) {
        return x.has_value() == y.has_value() && (!x || (x->x == y->x && x->y == y->y && x->z == y->z));
    };
    return a.form == b.form && vec(a.target, b.target) && a.distance == b.distance && a.azimuth_deg == b.azimuth_deg &&
           a.elevation_deg == b.elevation_deg && a.entity == b.entity;
}

}  // namespace

void set_program_workspace(std::function<std::string()> workspace) {
    program_host().workspace = std::move(workspace);
}

caliper_live_views::SceneView* SpecRuntime::entity_view(const std::string& panel_id) {
    auto it = scenes_->entities.find(panel_id);
    if (it == scenes_->entities.end()) return nullptr;
    EntityPanel& ep = *it->second;
    if (!ep.view && ep.notice.empty()) {
        caliper_gfx::Backend* backend = nullptr;
        const caliper_gfx::ProgramDesc* programs = nullptr;
        std::string why;
        bool ok = false;
        if (scene_layer_) {
            ok = scene_layer_(&ep.gfx, &backend, &programs, &why);
        } else {
            CaliperCore* core = core_ ? core_() : nullptr;
            ok = core && caliper_live_views::open_layer(core, "scenes", &ep.gfx, &backend, &programs, &why);
            if (!core) why = "Scene entities need the graphics layer, which starts with a live session";
        }
        if (ok) {
            auto view = std::make_unique<caliper_live_views::SceneView>(*backend, programs);
            if (view->init(&why)) {
                view->set_box_fit(true);
                ep.view = std::move(view);
            }
        }
        if (!ep.view) {
            ep.gfx.reset();
            ep.notice = why.empty() ? "The scene cannot start here" : why;
        }
    }
    if (!ep.view) return nullptr;
    // The panel's starting camera; the walls of the box face away from it, so they follow.
    if (ep.start_set && !same_starting(ep.view->starting(), ep.start)) {
        ep.view->set_starting(ep.start);
        ep.dirty = true;
    }
    const bool rebuilt = ep.dirty;
    if (ep.dirty) {
        apply_entity_scene(*ep.view, spec_scene_content(ep.scene), &ep.drawn);
        ep.dirty = false;
    }
    // The panel's annotations at the cursor, written again when they or the data box change.
    if (spec_)
        if (const spec::Panel* p = spec_->find_panel(panel_id)) {
            const std::vector<SceneAnnotationMark> marks = scene_annotation_marks(*p, instrument_host(&*spec_));
            std::string key;
            char t[160];
            for (const SceneAnnotationMark& m : marks) {
                std::snprintf(t, sizeof t, "|%d %.9g %.9g %.9g %lld %d %.9g %.9g ", int(m.kind), m.at[0], m.at[1],
                              m.at[2], (long long)m.index, m.axis, m.lo, m.hi);
                key += m.id + t + m.entity + "\x1f" + m.text;
            }
            if (rebuilt || key != ep.marks_key) {
                apply_annotation_marks(*ep.view, spec_scene_content(ep.scene), marks, &ep.marks_drawn);
                ep.marks_key = std::move(key);
            }
        }
    return ep.view.get();
}

namespace {

// Whether a scene panel draws a scene of its own, in a view of its own: its
// entities, or nothing but its programs when it binds no live scene
// (design/programmable-graphics.md, Attention Arc Tower). Two panels of one
// live scene share that scene's view, so a panel of programs alone must not.
bool own_scene(const spec::Panel& p) {
    if (p.primitive != spec::Primitive::scene) return false;
    if (!p.entities.empty()) return true;
    const auto it = p.bind.find(spec::kSceneSlot);
    return it == p.bind.end() || it->second.bindings.empty();
}

}  // namespace

caliper_live_views::SceneView* SpecRuntime::scene_view_of(const std::string& panel_id) {
    if (!spec_) return nullptr;
    const spec::Panel* p = spec_->find_panel(panel_id);
    if (!p || p->primitive != spec::Primitive::scene) return nullptr;
    if (own_scene(*p)) return entity_view(panel_id);
    auto it = p->bind.find(spec::kSceneSlot);
    if (it == p->bind.end() || it->second.bindings.empty()) return nullptr;
    const std::optional<std::string> name = spec::live_scene_name(it->second.bindings.front());
    CaliperCore* core = core_ ? core_() : nullptr;
    return name && core ? caliper_live_views::scene_view(core, name->c_str()) : nullptr;
}

bool SpecRuntime::scene_rect(const std::string& panel_id, float rect[4]) const {
    const auto it = scenes_->by_panel.find(panel_id);
    if (it == scenes_->by_panel.end() || !(it->second.rect[2] > 0.0f) || !(it->second.rect[3] > 0.0f)) return false;
    for (int i = 0; i < 4; ++i) rect[i] = it->second.rect[i];
    return true;
}

const SpecScene* SpecRuntime::entity_scene(const std::string& panel_id) const {
    auto it = scenes_->entities.find(panel_id);
    return it == scenes_->entities.end() ? nullptr : &it->second->scene;
}

SpecRuntime::SpecRuntime(SpecTextures& textures) : textures_(textures), scenes_(std::make_unique<SceneStates>()) {}

void SpecRuntime::set_instruments(caliper_builder::EventSink* events, caliper_builder::RunControl* run) {
    events_ = events;
    run_ = run;
    if (events) app_stage_host_impl()->set_events(events);   // the stages' program events
    if (events) program_host().host.set_events(events);       // the programs' draw events
    if (events) program_host().owner = this;
}

void SpecRuntime::set_instrument_state(caliper_builder::ValueStore* values, caliper_builder::StepHistory* history) {
    values_ = values;
    history_ = history;
    if (values || history) program_host().owner = this;
    // A stage's params follow their references at the next frame.
    app_stage_host_impl()->set_evaluate(
        values ? std::function<nlohmann::json(const spec::Reference&)>(
                     [values](const spec::Reference& r) { return values->value(r); })
               : nullptr);
    // The programs' params follow their references, and their resources come from the history.
    caliper_draw::DrawHost& programs = program_host().host;
    programs.set_evaluate(values ? caliper_draw::DrawHost::Evaluate(
                                       [values](const spec::Reference& r) { return values->value(r); })
                                 : nullptr);
    programs.set_resources(
        history ? caliper_draw::DrawHost::Resources(
                      [history, held = std::make_shared<std::shared_ptr<const std::vector<std::uint8_t>>>()](
                          const std::string& program, const std::string& resource, std::int64_t cursor,
                          caliper_draw::ResourceVersion* out, std::uint64_t* epoch) {
                          const std::optional<caliper_builder::ResourceVersion> v =
                              history->resource_at(program + "/" + resource, cursor);
                          if (!v || !v->bytes || !out) return false;
                          *held = v->bytes;   // the bytes stay valid until the next call
                          out->resource = resource;
                          out->step = v->step;
                          out->serial = v->serial;   // a version that replaced another at its step
                          out->count = v->width ? v->width * v->height : v->count;
                          out->bytes = (*held)->data();
                          out->size = (*held)->size();
                          if (epoch) *epoch = 0;
                          return true;
                      })
                : nullptr);
}

caliper_builder::ChainContext SpecRuntime::chain_context() const {
    caliper_builder::ChainContext c{values_, history_, 0, true};
    if (history_) {
        const caliper_builder::CursorState cursor = history_->cursor();
        c.cursor_step = cursor.step;
        c.cursor_live = cursor.live;
    }
    return c;
}

SpecInstrumentHost SpecRuntime::instrument_host(const spec::Spec* shown) const {
    SpecInstrumentHost h;
    h.events = events_;
    h.run = run_;
    h.values = values_;
    h.history = history_;
    h.spec = shown;
    return h;
}

bool SpecRuntime::pick_scene(const std::string& panel_id, float u, float v) {
    if (!spec_) return false;
    const spec::Panel* p = spec_->find_panel(panel_id);
    if (!p || p->primitive != spec::Primitive::scene) return false;
    caliper_live_views::pick_scene_rect(scenes_->by_panel[panel_id], u, v);
    return true;
}

void SpecRuntime::bind_scene(Entry& e) {
    if (e.panel.primitive != spec::Primitive::scene) {
        e.view.scene = nullptr;
        return;
    }
    if (own_scene(e.panel)) {
        bind_entity_scene(e);
        return;
    }
    // The entry moves between maps when the spec changes, so the call keeps
    // copies of what it draws; a change of the panel prepares it again.
    const std::string panel = e.panel.id;
    const std::string scene = e.view.data.scene;
    caliper_live_views::SceneRectOptions o;
    const spec::SceneOptions& so = e.view.data.scene_options;
    o.up = so.up.value_or("");
    o.labels = so.labels.value_or(true);
    o.scale_bar = so.scale_bar.value_or(false);
    // The panel's starting camera and its axes mark. The option box keeps to the entities of a
    // spec, because a live scene keeps its own units (design/instruments.md).
    if (!so.view_camera.is_null()) {
        spec::Panel shown = e.panel;
        shown.options.scene = so;
        o.view_camera = spec_starting_camera(shown);
        o.view_camera_key = so.view_camera.dump();
    }
    o.axes = so.axes.value_or(true);
    o.include = so.include;
    o.exclude = so.exclude;
    o.draw_feed = program_host().host.feed(panel);
    e.view.scene = [this, panel, scene, o](const ImRect& content) {
        CaliperCore* core = core_ ? core_() : nullptr;
        if (!core || scene.empty()) return false;
        give_program_cursor(history_);
        sync_program_pick(panel, values_);
        const std::string id = "##spec.scene." + panel;
        // The panel's stages (spec_stage_host.h), and the line of a refused or stopped one.
        caliper_live_views::SceneRectOptions so = o;
        so.stages = app_stage_host_impl()->source(panel);
        so.stage_message = app_stage_host_impl()->message(panel);
        auto sink = [this, &panel, &scene](const char* type, const std::string& detail) {
            caliper_builder::Event ev;
            if (!events_ || !spec_scene_event(type, panel, scene, detail, &ev)) return;
            const bool pick = ev.kind == caliper_builder::EventKind::pick;
            caliper_builder::Event kept = pick ? ev : caliper_builder::Event{};
            note_program_pick(panel, ev);
            events_->push(std::move(ev));
            if (pick) {   // the panel's last pick and its `select` write (spec_instruments.h)
                SpecInstrumentHost host = instrument_host(spec_ ? &*spec_ : nullptr);
                record_spec_pick(panel, kept, host);
            }
        };
        return caliper_live_views::draw_scene_in_rect(core, scene, id.c_str(), content.Min, content.GetSize(), so,
                                                      scenes_->by_panel[panel],
                                                      sink) == caliper_live_views::SceneRectResult::drawn;
    };
}

void SpecRuntime::bind_entity_scene(Entry& e) {
    const std::string panel = e.panel.id;
    caliper_live_views::SceneRectOptions o;
    o.labels = e.panel.options.scene.labels.value_or(true);
    o.scale_bar = e.panel.options.scene.scale_bar.value_or(false);
    o.draw_feed = program_host().host.feed(panel);
    e.view.scene = [this, panel, o](const ImRect& content) {
        give_program_cursor(history_);
        sync_program_pick(panel, values_);
        caliper_live_views::SceneView* view = entity_view(panel);
        EntityPanel& ep = scenes_->entity(panel);
        if (!view) return false;   // no backend here: the panel shows its source as offline
        caliper_live_views::SceneRectOptions so = o;
        so.stages = app_stage_host_impl()->source(panel);
        so.stage_message = app_stage_host_impl()->message(panel);
        const std::string id = "##spec.entities." + panel;
        auto sink = [this, &panel, &ep](const char* type, const std::string& detail) {
            caliper_builder::Event ev;
            if (!events_ || !spec_scene_event(type, panel, "", detail, &ev)) return;
            const bool pick = ev.kind == caliper_builder::EventKind::pick;
            if (pick && ev.fields.contains("entity") && ev.fields["entity"].is_string() &&
                ev.fields.contains("index") && ev.fields["index"].is_number_integer())
                if (auto f = ep.scene.pick_fields(ev.fields["entity"].get<std::string>(),
                                                  ev.fields["index"].get<std::uint64_t>()))
                    ev.fields["fields"] = caliper_builder::pick_fields_json(*f);
            caliper_builder::Event kept = pick ? ev : caliper_builder::Event{};
            note_program_pick(panel, ev);
            events_->push(std::move(ev));
            if (pick) {
                SpecInstrumentHost host = instrument_host(spec_ ? &*spec_ : nullptr);
                record_spec_pick(panel, kept, host);
            }
        };
        const float scale = ImGui::GetIO().DisplayFramebufferScale.x > 0 ? ImGui::GetIO().DisplayFramebufferScale.x : 1.0f;
        const auto r = caliper_live_views::draw_scene_view_in_rect(*view, scale, id.c_str(), content.Min,
                                                                   content.GetSize(), so, scenes_->by_panel[panel], sink);
        // An entity that draws nothing says why, over the picture: its path and its code.
        for (const EntityDraw& d : spec_scene_content(ep.scene).draws)
            if (d.error && d.error.code != "entity.no_data") {
                const char* label = d.error.code == "entity.limit" ? "Over limit" : "Wrong shape";
                ImGui::GetWindowDrawList()->AddText(ImVec2(content.Min.x + 8, content.Max.y - 22), IM_COL32(230, 120, 90, 255),
                                                    (std::string(label) + " " + d.entity + " (" + d.error.code + ")").c_str());
                break;
            }
        return r != caliper_live_views::SceneRectResult::missing;
    };
}

void SpecRuntime::draw_control(const spec::Control& control, const ImRect& r, const SpecLook& L) {
    // Each control in its own child window, as a panel: its own ids and clipping.
    ImGui::SetCursorScreenPos(r.Min);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    const std::string child = "##spec.control." + control.id;
    if (ImGui::BeginChild(child.c_str(), r.GetSize(), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                              ImGuiWindowFlags_NoBackground)) {
        ImGui::PopStyleVar(2);
        ImGui::GetWindowDrawList()->AddRectFilled(r.Min, r.Max, L.ground);
            SpecInstrumentHost host = instrument_host(spec_ ? &*spec_ : nullptr);
        draw_spec_control(control, r, L, host);
        // An item at the corner, so the child's bounds come from an item.
        ImGui::SetCursorScreenPos(r.Min);
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
    } else {
        ImGui::PopStyleVar(2);
    }
    ImGui::EndChild();
}

SpecRuntime::~SpecRuntime() {
    // The hosts of the process outlive this runtime: they stop reading its sink, values and history.
    if (program_host().owner == this) {
        set_instrument_state(nullptr, nullptr);
        program_host().host.set_events(nullptr);
        app_stage_host_impl()->set_events(nullptr);
        program_host().owner = nullptr;
    }
    for (auto& e : entries_) textures_.release(e.second.view.texture);
    for (auto& e : previews_) textures_.release(e.second.view.texture);
    for (auto& e : spec_preview_entries_) textures_.release(e.second.view.texture);
}

void SpecRuntime::set_spec(const spec::Spec& s) {
    if (spec_ && *spec_ == s) return;
    program_host().host.reconcile(s);   // the programs of its scene panels
    // A panel whose fragment or sources changed is resolved again; sources
    // are shared, so any change of a source marks every panel.
    const bool sources_changed = !spec_ || spec_->sources != s.sources;
    spec_ = s;
    key_.reset(s);
    std::map<std::string, Entry> kept;
    for (const auto& p : s.panels) {
        auto it = entries_.find(p.id);
        if (it != entries_.end()) {
            Entry e = std::move(it->second);
            entries_.erase(it);
            if (sources_changed || e.panel != p) e.stale = true;
            e.panel = p;
            kept.emplace(p.id, std::move(e));
        } else {
            Entry e;
            e.panel = p;
            kept.emplace(p.id, std::move(e));
        }
    }
    for (auto& gone : entries_) textures_.release(gone.second.view.texture);
    entries_ = std::move(kept);
}

void SpecRuntime::clear_spec() {
    for (auto& e : entries_) textures_.release(e.second.view.texture);
    entries_.clear();
    spec_.reset();
    layout_ = {};
}

void SpecRuntime::set_source(SpecPanelSource* source) {
    source_ = source;
    invalidate();
}

void SpecRuntime::invalidate() {
    for (auto& e : entries_) e.second.stale = true;
}

namespace {

bool out_of_range(const caliper_builder::Error& e) { return e && e.code == "resolve.out_of_range"; }

}  // namespace

void SpecRuntime::prepare(Entry& e, const PanelData& resolved, const spec::Spec* spec) {
    textures_.release(e.view.texture);
    e.view.texture = nullptr;
    // A reference whose value is outside the range of its place stops its
    // chain (resolve.out_of_range). The panel, or each entity that the error
    // stopped, then keeps the last good data, and the header shows the error.
    PanelData data = resolved;
    std::optional<caliper_builder::Error> held;
    if (out_of_range(data.error)) {
        held = data.error;
        if (e.last_good) {
            data = *e.last_good;
            data.generation = resolved.generation;
        }
    } else if (!data.error) {
        for (caliper_builder::EntityData& ent : data.entities) {
            if (!out_of_range(ent.error)) continue;
            if (!held) held = ent.error;
            if (!e.last_good) continue;
            for (const caliper_builder::EntityData& old : e.last_good->entities)
                if (old.path == ent.path && !old.error) {
                    ent = old;
                    break;
                }
        }
    }
    if (!data.error) e.last_good = data;
    // The panel as it draws now: its option references applied.
    const caliper_builder::ChainContext chain = chain_context();
    const spec::Panel shown = caliper_builder::panel_with_values(e.panel, &chain);
    e.view.data = prepare_panel(spec ? *spec : spec_ ? *spec_ : spec::Spec{}, shown, data, limits_);
    if (held && e.view.data.state.ready())
        e.view.data.notice = panel_state_for_error(held->code, held->where.empty() ? held->code : held->where);
    e.view.fit_frames = 2;
    ++e.view.version;
    e.stale = false;
    e.generation = data.generation;
    if (own_scene(e.panel)) {
        EntityPanel& ep = scenes_->entity(e.panel.id);
        ep.scene.set_entities(shown, data);
        ep.dirty = true;
        const caliper_scene::StartingCamera start = spec_starting_camera(e.panel);
        if (!ep.start_set || !same_starting(start, ep.start)) {
            ep.start = start;
            ep.start_set = true;
        }
    }
    bind_scene(e);
}

void SpecRuntime::set_preview(const spec::Panel& panel, const PanelData& data, const spec::Spec* spec) {
    Entry& e = previews_[panel.id];
    e.panel = panel;
    prepare(e, data, spec);
}

void SpecRuntime::clear_preview(const std::string& panel_id) {
    auto it = previews_.find(panel_id);
    if (it == previews_.end()) return;
    textures_.release(it->second.view.texture);
    previews_.erase(it);
}

void SpecRuntime::update() { update_only(nullptr); }

void SpecRuntime::update_only(const SpecLayout* shown) {
    if (!spec_) return;
    if (source_) {
        const uint64_t r = source_->revision();
        if (r != revision_) {
            revision_ = r;
            invalidate();
        }
    }
    for (auto& kv : entries_) {
        Entry& e = kv.second;
        if (shown && !shown->find(kv.first)) continue;   // a hidden tab waits until it shows
        // A shown panel whose served data has a new generation is resolved
        // again; the probe reads no data.
        if (e.panel.primitive == spec::Primitive::scene && e.panel.entities.empty()) {
            // A scene panel without entities binds no source: its scene comes from the live core
            // at each draw. One with entities resolves their arrays as any panel resolves its slots.
            if (!e.stale) continue;
            PanelData data;
            data.panel = e.panel.id;
            data.primitive = e.panel.primitive;
            prepare(e, data);
            continue;
        }
        if (!e.stale && source_ && source_->generation(*spec_, e.panel.id) != e.generation) e.stale = true;
        if (!e.stale) continue;
        PanelData data;
        if (source_) {
            data = source_->resolve(*spec_, e.panel.id);
        } else {
            data.panel = e.panel.id;
            data.error = caliper_builder::Error("resolve.unavailable", "No source of data is attached.", "");
        }
        prepare(e, data);
        if (!source_) e.view.data.state = {PanelStateKind::offline, "No data source", ""};
    }
}

const PreparedPanel* SpecRuntime::prepared(const std::string& panel_id) const {
    auto it = entries_.find(panel_id);
    return it == entries_.end() || it->second.stale ? nullptr : &it->second.view.data;
}

void SpecRuntime::draw_tabs(const SpecLayoutTabs& t, const SpecLook& L) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 a(t.bar.x, t.bar.y), b(t.bar.right(), t.bar.bottom());
    dl->AddRectFilled(a, b, L.ground);
    dl->AddRectFilled(ImVec2(a.x, b.y - 1.0f), b, L.rule);
    const float s = L.text_scale, pad = std::round(12.0f * s);
    float x = a.x;
    for (size_t i = 0; i < t.labels.size(); ++i) {
        const std::string label = fit_text(L.ui, L.ui_size, t.labels[i], std::max(0.0f, b.x - x - 2.0f * pad));
        if (label.empty()) break;
        const ImVec2 ts = text_size(L.ui, L.ui_size, label);
        const ImVec2 p0(x, a.y), p1(x + ts.x + 2.0f * pad, b.y);
        const bool active = (int)i == t.active;
        ImGui::SetCursorScreenPos(p0);
        ImGui::PushID(t.path.c_str());
        ImGui::PushID((int)i);
        const bool clicked = ImGui::InvisibleButton("##spec.tab", ImVec2(p1.x - p0.x, std::max(1.0f, p1.y - p0.y)));
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        ImGui::PopID();
        if (clicked) tabs_[t.path] = (int)i;
        if (active) {
            dl->AddRectFilled(p0, ImVec2(p1.x, b.y), L.plate);
            dl->AddRectFilled(p0, ImVec2(p1.x, p0.y + 1.0f), L.accent);   // the bronze overline
        } else if (hovered) {
            dl->AddRectFilled(p0, ImVec2(p1.x, b.y - 1.0f), L.plate_hover);
        }
        dl->AddRectFilled(ImVec2(p1.x, a.y), ImVec2(p1.x + 1.0f, b.y), L.rule);
        text_at(dl, L.ui, L.ui_size, ImVec2(x + pad, a.y + (t.bar.h - ts.y) * 0.5f), active ? L.text : L.secondary,
                label);
        x = p1.x + 1.0f;
    }
    // The shown tab's info string, when the tab is a panel, at the bar's right.
    const std::string shown = t.path.empty() ? std::to_string(t.active) : t.path + "." + std::to_string(t.active);
    for (const auto& lp : layout_.panels) {
        if (lp.path != shown || !lp.tab) continue;
        auto pv = previews_.find(lp.id);
        auto it = entries_.find(lp.id);
        const Entry* e = pv != previews_.end() ? &pv->second : it != entries_.end() ? &it->second : nullptr;
        if (!e) break;
        float left = x + 14.0f * s;
        if (pv != previews_.end()) {
            // A tab drawn from a preview: the PREVIEW tag after the tabs.
            const float th = text_size(L.ui, L.small_size, "P").y;
            if (left + preview_tag_width(L) < b.x - std::round(8.0f * s))
                preview_tag(dl, L, ImVec2(std::round(left), std::round(a.y + (t.bar.h - th) * 0.5f)));
            left += preview_tag_width(L) + 14.0f * s;
        }
        if (!e->view.data.state.ready() || e->view.data.info.empty()) break;
        const float right = b.x - std::round(8.0f * s);
        const std::string info = fit_text(L.mono, L.mono_size, e->view.data.info, right - left);
        if (info.empty()) break;
        const ImVec2 is = text_size(L.mono, L.mono_size, info);
        text_at(dl, L.mono, L.mono_size, ImVec2(right - is.x, a.y + (t.bar.h - is.y) * 0.5f), L.faint, info);
        break;
    }
}

// ---- the selection's rows and the spec's preview ----------------------------------------

SpecSelectionRows spec_selection_rows(const PreparedPanel& p, const SpecSelection& sel, size_t max_rows) {
    SpecSelectionRows out;
    if (sel.empty() || sel.panel != p.id) return out;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (sel.kind == SpecSelection::Kind::range && p.primitive == spec::Primitive::series && !p.series.lines.empty()) {
        const PreparedSeries& S = p.series;
        out.columns.push_back(S.x_name.empty() ? "x" : S.x_name);
        for (const auto& l : S.lines) out.columns.push_back(l.name);
        const SeriesLine& first = S.lines.front();
        for (size_t i = 0; i < first.x.size(); ++i) {
            const double x = first.x[i];
            if (x < sel.x0 || x > sel.x1) continue;
            ++out.total;
            if (out.rows.size() >= max_rows) continue;
            std::vector<double> row{x};
            for (const auto& l : S.lines) {
                double y = nan;
                if (l.x.size() == first.x.size() && i < l.y.size() && l.x[i] == x) {
                    y = l.y[i];
                } else {
                    const auto it = std::lower_bound(l.x.begin(), l.x.end(), x);
                    // A line's y can be shorter than its x: no value there.
                    const size_t k = (size_t)(it - l.x.begin());
                    if (it != l.x.end() && *it == x && k < l.y.size()) y = l.y[k];
                }
                row.push_back(y);
            }
            out.rows.push_back(std::move(row));
        }
    } else if (sel.kind == SpecSelection::Kind::points && p.primitive == spec::Primitive::points && !p.points.three_d) {
        const PreparedPoints& P = p.points;
        out.columns = {p.x_label.empty() ? "x" : p.x_label, p.y_label.empty() ? "y" : p.y_label};
        for (size_t i = 0; i < P.x.size() && i < P.y.size(); ++i) {
            if (P.x[i] < sel.x0 || P.x[i] > sel.x1 || P.y[i] < sel.y0 || P.y[i] > sel.y1) continue;
            ++out.total;
            if (out.rows.size() < max_rows) out.rows.push_back({(double)P.x[i], (double)P.y[i]});
        }
    }
    return out;
}

void SpecRuntime::set_spec_preview(const spec::Spec& s, std::set<std::string> added) {
    if (spec_preview_ && *spec_preview_ == s && spec_preview_added_ == added) return;
    spec_preview_ = s;
    spec_preview_added_ = std::move(added);
    for (auto& e : spec_preview_entries_) {
        e.second.stale = true;
        const spec::Panel* p = s.find_panel(e.first);
        if (p) e.second.panel = *p;
    }
}

void SpecRuntime::clear_spec_preview() {
    for (auto& e : spec_preview_entries_) textures_.release(e.second.view.texture);
    spec_preview_entries_.clear();
    spec_preview_.reset();
    spec_preview_added_.clear();
}

void SpecRuntime::draw(const ImRect& area, const SpecLook& L) {
    textures_.collect();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    // The rules between the panels: the area in the rule colour, which the
    // panels leave showing in the gaps of the layout.
    dl->AddRectFilled(area.Min, area.Max, L.rule);
    layout_ = {};
    if (!spec_) return;
    // The spec shown: the open one, or a preview of the whole spec.
    const spec::Spec& shown = spec_preview_ ? *spec_preview_ : *spec_;
    // The instruments' keys and the transport's play, once a frame
    // (spec_shortcuts.h, spec_transport.h).
    SpecInstrumentHost instruments = instrument_host(&shown);
    handle_instrument_keys(shown, instruments);
    handle_transport_keys(shown, instruments);
    tick_transport(instruments, ImGui::GetIO().DeltaTime);
    const std::optional<spec::LayoutNode> root = caliper_builder::effective_layout(shown);
    if (!root) {
        dl->AddRectFilled(area.Min, area.Max, L.ground);
        if (empty_hint_.empty()) draw_state({PanelStateKind::empty, "No panels", shown.applet.id}, area, L);
        else draw_hint("No panels", empty_hint_, area, L);
        return;
    }
    SpecLayoutMetrics m = SpecLayoutMetrics::for_text_scale(L.text_scale);
    m.control_height = [&L](const spec::Control& c) { return spec_control_height(c, L); };
    layout_ = compute_spec_layout(*root, shown, {area.Min.x, area.Min.y, area.GetWidth(), area.GetHeight()}, m, tabs_);
    // Only the panels shown are resolved; a panel in a hidden tab is resolved
    // when its tab is chosen.
    update_only(&layout_);
    for (const auto& t : layout_.tabs) draw_tabs(t, L);
    for (const auto& lp : layout_.panels) {
        if (lp.control) {
            // A control leaf (spec version 1): its widget, drawn by spec_controls.cpp.
            const spec::Control* c = shown.find_control(lp.id);
            const ImRect r(ImVec2(lp.rect.x, lp.rect.y), ImVec2(lp.rect.right(), lp.rect.bottom()));
            if (c && r.GetWidth() >= 1.0f && r.GetHeight() >= 1.0f) draw_control(*c, r, L);
            continue;
        }
        auto pv = previews_.find(lp.id);
        const bool preview = pv != previews_.end();
        auto it = entries_.find(lp.id);
        // A panel the spec's preview adds or changes: from the preview.
        Entry* whole = nullptr;
        bool added = false;
        if (spec_preview_ && !preview) {
            const spec::Panel* pp = spec_preview_->find_panel(lp.id);
            added = spec_preview_added_.count(lp.id) != 0;
            if (pp && (added || it == entries_.end() || it->second.panel != *pp)) {
                Entry& e = spec_preview_entries_[lp.id];
                if (e.panel != *pp) {
                    e.panel = *pp;
                    e.stale = true;
                }
                if (e.stale) {
                    PanelData data;
                    if (pp->primitive == spec::Primitive::scene && pp->entities.empty()) {
                        data.panel = lp.id;
                        data.primitive = pp->primitive;
                    } else if (source_) {
                        data = source_->resolve(*spec_preview_, lp.id);
                    } else {
                        data.panel = lp.id;
                        data.error = caliper_builder::Error("resolve.unavailable", "No source of data is attached.", "");
                    }
                    prepare(e, data, &*spec_preview_);
                }
                whole = &e;
            }
        }
        if (!preview && !whole && it == entries_.end()) continue;
        Entry& e = preview ? pv->second : whole ? *whole : it->second;
        e.view.key = &key_;
        e.view.show_title = !lp.tab;
        e.view.preview = preview || whole != nullptr;
        e.view.added = added;
        e.view.selection = &selection_;
        const ImRect r(ImVec2(lp.rect.x, lp.rect.y), ImVec2(lp.rect.right(), lp.rect.bottom()));
        if (r.GetWidth() < 1.0f || r.GetHeight() < 1.0f) continue;
        // The band an overlay keeps under the strip, asked before the content.
        float band = 0.0f;
        if (reserve_) {
            SpecPanelFrame pre;
            pre.id = lp.id;
            pre.rect = r;
            pre.header = ImRect(r.Min, ImVec2(r.Max.x, r.Min.y + spec_panel_strip_height(L, e.view.show_title, r.GetHeight())));
            pre.band = ImRect(pre.header.GetBL(), pre.header.GetBR());
            pre.preview = preview;
            band = reserve_(pre);
        }
        // Each panel in its own child window: its own ids, clipping and
        // scrolling, so ImPlot keeps one plot state per panel.
        ImGui::SetCursorScreenPos(r.Min);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        const std::string child = "##spec.panel." + lp.id;
        if (ImGui::BeginChild(child.c_str(), r.GetSize(), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                  ImGuiWindowFlags_NoBackground)) {
            ImGui::PopStyleVar(2);
            e.view.spec_panel = shown.find_panel(lp.id);
            e.view.instruments = &instruments;
            SpecPanelFrame f = draw_spec_panel(e.view, r, L, textures_, band);
            if (e.view.spec_panel) draw_panel_overlays(*e.view.spec_panel, f.content, L, instruments);   // spec_annotations.h
            if (e.view.added) {
                // A panel the preview adds: a dashed bronze frame inside its rectangle.
                ImDrawList* pdl = ImGui::GetWindowDrawList();
                const float dash = std::round(6.0f * L.text_scale), gap = std::round(4.0f * L.text_scale);
                const float x0 = r.Min.x + 0.5f, y0 = r.Min.y + 0.5f, x1 = r.Max.x - 0.5f, y1 = r.Max.y - 0.5f;
                for (float x = x0; x < x1; x += dash + gap) {
                    const float xe = std::min(x + dash, x1);
                    pdl->AddLine(ImVec2(x, y0), ImVec2(xe, y0), L.accent);
                    pdl->AddLine(ImVec2(x, y1), ImVec2(xe, y1), L.accent);
                }
                for (float y = y0; y < y1; y += dash + gap) {
                    const float ye = std::min(y + dash, y1);
                    pdl->AddLine(ImVec2(x0, y), ImVec2(x0, ye), L.accent);
                    pdl->AddLine(ImVec2(x1, y), ImVec2(x1, ye), L.accent);
                }
            }
            if (overlay_) overlay_(f);
        } else {
            ImGui::PopStyleVar(2);
        }
        ImGui::EndChild();
    }
}

}  // namespace caliper_host
