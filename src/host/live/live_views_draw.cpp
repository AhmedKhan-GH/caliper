// live_views_draw.cpp: the windows of the typed live views (caliper/
// live_views.h), drawn by caliper_live_views_frame after the embed v1.2 views.
//
// One ImGui window per view, titled with the view name and keyed
// "<name>###caliper.view.<name>" like the v1.2 views (the layout persists by
// name in <data_dir>/imgui.ini), tiled on first placement with the same cursor
// as the v1.2 views. Every window starts with a Pause/Resume button (one GLOBAL
// pause: caliper_core_live_set_paused) and an info line.
//
//   POINTS     a geometry render target sized to the window's content region
//              (physical pixels; recreated when it drifts more than 8 px),
//              redrawn from the imported payload buffer whenever the payload,
//              the camera or the size changed (zero-copy: the GPU reads the
//              producer's buffer). 3-D: orbit with left/right drag, pan with
//              middle drag or shift+drag, wheel zoom. 2-D: drag pans, wheel
//              zooms about the cursor. Fit button / F key re-fits to the bbox.
//              Opaque points with depth test in 3-D, so categorical colours stay
//              exact under overlap. Hover: nearest point under the cursor from
//              the CPU-visible payload (sampled above 256K points).
//   SERIES     ImPlot line over step index, auto-fit, min/max decimated above
//              8192 values, hover tooltip (step, value), latest value in text.
//   HISTOGRAM  ImPlot bars at bin centres, hover tooltip (bin range, count).
//   HEATMAP / IMAGE  drawn by the v1.2 views; this file appends a one-line
//              overlay (seq, latency, dropped) and the Pause button to that
//              window.
#include "live/live_views_state.h"

#include "renderer/host_renderer.h"

#include <caliper/services/geometry_v1_1.h>

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using caliper_embed_views::TensorView;
using caliper_host::HostGeomDraw;
using caliper_host::HostRenderer;

namespace caliper_live_views {

// Tableau 10 + six lighter companions, packed r | g<<8 | b<<16 | a<<24.
#define LV_RGB(r, g, b) (0xff000000u | ((uint32_t)(b) << 16) | ((uint32_t)(g) << 8) | (uint32_t)(r))
const uint32_t kPalette[16] = {
    LV_RGB(31, 119, 180),  LV_RGB(255, 127, 14),  LV_RGB(44, 160, 44),   LV_RGB(214, 39, 40),
    LV_RGB(148, 103, 189), LV_RGB(140, 86, 75),   LV_RGB(227, 119, 194), LV_RGB(127, 127, 127),
    LV_RGB(188, 189, 34),  LV_RGB(23, 190, 207),  LV_RGB(174, 199, 232), LV_RGB(255, 187, 120),
    LV_RGB(152, 223, 138), LV_RGB(255, 152, 150), LV_RGB(197, 176, 213), LV_RGB(196, 156, 148),
};

namespace {

constexpr uint32_t kClear = LV_RGB(16, 16, 20);    // render-target background
constexpr uint32_t kFlat = LV_RGB(220, 220, 230);  // points without aux
constexpr int      kHysteresisPx = 8;
constexpr int      kMaxTarget = 8192;
constexpr uint64_t kHoverBudget = 1u << 18;
constexpr size_t   kSeriesRawMax = 8192;
constexpr float    kFovY = 45.0f * 3.14159265f / 180.0f;

struct V3 { float x, y, z; };
V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 norm3(V3 a) {
    const float l = std::sqrt(dot(a, a));
    return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : V3{0, 0, 1};
}

// Right-handed lookAt, column-major (flow_scope's).
void look_at(V3 eye, V3 at, V3 up, float* m) {
    const V3 f = norm3(sub(at, eye));
    const V3 s = norm3(cross(f, up));
    const V3 u = cross(s, f);
    const float t[16] = {s.x, u.x, -f.x, 0, s.y, u.y, -f.y, 0, s.z, u.z, -f.z, 0,
                         -dot(s, eye), -dot(u, eye), dot(f, eye), 1};
    std::memcpy(m, t, sizeof(t));
}

// z in [0,1] perspective (Metal NDC), column-major (flow_scope's).
void perspective(float fovy, float aspect, float zn, float zf, float* m) {
    const float f = 1.0f / std::tan(fovy * 0.5f);
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = f / aspect;
    m[5] = f;
    m[10] = zf / (zn - zf);
    m[11] = -1.0f;
    m[14] = (zn * zf) / (zn - zf);
}

V3 eye_of(const Camera& c) {
    const float ce = std::cos(c.el), se = std::sin(c.el);
    const float ca = std::cos(c.az), sa = std::sin(c.az);
    return {c.tx + c.dist * ce * ca, c.ty + c.dist * se, c.tz + c.dist * ce * sa};
}

std::string window_label(const View& v) { return v.name + "###caliper.view." + v.name; }

// First placement, tiled with the v1.2 views' cursor (core->tile_*): the same
// wrap rules as embed_core.cpp's place_view, for a content area `content`.
void place(CaliperCore* core, View& v, ImVec2 content) {
    const float lw = (float)core->canvas_w / core->scale;
    const float lh = (float)core->canvas_h / core->scale;
    const ImGuiStyle& st = ImGui::GetStyle();
    const float chrome_w = st.WindowPadding.x * 2.0f;
    const float chrome_h = ImGui::GetFrameHeight() + st.WindowPadding.y * 2.0f +
                           ImGui::GetFrameHeightWithSpacing();
    const float ww = std::max(200.0f, std::min(content.x + chrome_w, lw * 0.9f));
    const float wh = std::max(120.0f, std::min(content.y + chrome_h, lh * 0.9f));
    const float gap = 8.0f;
    if (core->tile_x > gap && core->tile_x + ww > lw) {
        core->tile_x = gap;
        core->tile_y += core->tile_row_h + gap;
        core->tile_row_h = 0.0f;
    }
    if (core->tile_y > gap && core->tile_y + wh > lh) {
        core->tile_x = gap;
        core->tile_y = gap;
        core->tile_row_h = 0.0f;
    }
    v.pos_x = core->area_x + core->tile_x;   // a hosted core tiles into the host's work area
    v.pos_y = core->area_y + core->tile_y;
    v.size_x = ww;
    v.size_y = wh;
    core->tile_x += ww + gap;
    core->tile_row_h = std::max(core->tile_row_h, wh);
    v.placed = true;
}

bool begin_window(CaliperCore* core, View& v, ImVec2 content) {
    if (!v.placed) place(core, v, content);
    ImGui::SetNextWindowPos(ImVec2(v.pos_x, v.pos_y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(v.size_x, v.size_y), ImGuiCond_FirstUseEver);
    return ImGui::Begin(window_label(v).c_str(), nullptr,
                        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
}

void pause_button(CaliperLiveViewsState* S) {
    if (ImGui::SmallButton(S->paused ? "Resume###caliper.live.pause" : "Pause###caliper.live.pause"))
        S->paused = !S->paused;
}

// "seq 41  lat 2.3 ms  upd 41  drop 0  [paused]"
void telemetry(char* buf, size_t cap, const CaliperLiveViewsState* S, const View& v) {
    std::snprintf(buf, cap, "seq %llu  lat %.1f ms  upd %llu  drop %llu%s",
                  (unsigned long long)v.seq, v.latency_ms, (unsigned long long)v.updates,
                  (unsigned long long)v.dropped, S->paused ? "  [paused]" : "");
}

// ---- points --------------------------------------------------------------
struct Pick {
    int64_t idx = -1;
    float sx = 0.0f, sy = 0.0f;
};

// Nearest point to the cursor among those projected inside the image: a point
// whose drawn square covers the cursor wins (front-most first), else the one
// nearest in screen distance within `radius` px. Sampled above kHoverBudget.
Pick pick_point(const View& v, const float mvp[16], ImVec2 p0, ImVec2 sz, ImVec2 m,
                float half_px, float radius, uint64_t* stride_out) {
    Pick best;
    const float* P = (const float*)v.cur.pos_cpu;
    const uint64_t n = v.cur.n;
    const uint64_t stride = n > kHoverBudget ? (n + kHoverBudget - 1) / kHoverBudget : 1;
    *stride_out = stride;
    int bucket = 2;                 // 0 covering, 1 near
    float key = 0.0f;
    for (uint64_t i = 0; i < n; i += stride) {
        const float x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
        const float cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
        if (!(cw > 1e-9f)) continue;
        const float cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
        const float cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
        const float cz = mvp[2] * x + mvp[6] * y + mvp[10] * z + mvp[14];
        const float nz = cz / cw;
        if (!(nz >= 0.0f && nz <= 1.0f)) continue;
        const float sx = p0.x + (cx / cw * 0.5f + 0.5f) * sz.x;
        const float sy = p0.y + (0.5f - cy / cw * 0.5f) * sz.y;
        const float dx = sx - m.x, dy = sy - m.y;
        if (std::fabs(dx) <= half_px && std::fabs(dy) <= half_px) {
            if (bucket > 0 || nz < key) { bucket = 0; key = nz; best = {(int64_t)i, sx, sy}; }
            continue;
        }
        if (bucket == 0) continue;
        const float d2 = dx * dx + dy * dy;
        if (d2 <= radius * radius && (bucket == 2 || d2 < key)) {
            bucket = 1;
            key = d2;
            best = {(int64_t)i, sx, sy};
        }
    }
    return best;
}

void mul_cm(const float* a, const float* b, float* out) {   // out = a * b
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 4; ++k) acc += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = acc;
        }
}

float point_size_px(uint64_t n, float scale) {
    const float base = n <= 2000 ? 5.0f : n <= 50000 ? 3.0f : 2.0f;
    return base * scale;
}

void draw_points(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    HostRenderer* r = core->renderer.get();
    const bool open = begin_window(core, v, ImVec2(320.0f, 260.0f));
    if (!open) { ImGui::End(); return; }
    if (ImGui::SmallButton("Fit")) v.fit_pending = true;
    ImGui::SameLine();
    pause_button(S);
    ImGui::SameLine();
    char tel[160];
    telemetry(tel, sizeof tel, S, v);
    ImGui::Text("points N=%lld %d-D  %s", (long long)v.count, v.dims, tel);

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 4.0f || avail.y < 4.0f) { ImGui::End(); return; }
    const int want_w = std::clamp((int)(avail.x * core->scale), 1, kMaxTarget);
    const int want_h = std::clamp((int)(avail.y * core->scale), 1, kMaxTarget);
    if (!v.view_rid || std::abs(want_w - v.tex_w) > kHysteresisPx ||
        std::abs(want_h - v.tex_h) > kHysteresisPx) {
        const uint64_t nr = r->geom_create_view_ex(want_w, want_h, CALIPER_GEOM_VIEW_DEPTH);
        if (nr) {
            // The previous target may still be sampled by an in-flight frame:
            // its command buffers retain it, so dropping our reference is safe.
            if (v.view_rid) r->tex_release(v.view_rid);
            v.view_rid = nr;
            v.tex_w = want_w;
            v.tex_h = want_h;
            ++v.creates;
            v.dirty = true;
            v.hover_dirty = true;
        }
    }
    if (!v.view_rid) {
        ImGui::TextUnformatted("render target unavailable");
        ImGui::End();
        return;
    }
    const float aspect = (float)v.tex_w / (float)v.tex_h;
    if (v.fit_pending && v.have_cur) fit_points(v, aspect);

    // ---- input (an invisible button owns the area so drags never move the window)
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##caliper.points", avail,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                               ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    ImGuiIO& io = ImGui::GetIO();
    Camera& c = v.cam;
    if (active && (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f)) {
        const float dx = io.MouseDelta.x, dy = io.MouseDelta.y;
        const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle) || io.KeyShift;
        if (v.dims == 3 && !pan) {
            c.az += dx * 0.01f;
            c.el = std::clamp(c.el + dy * 0.01f, -1.55f, 1.55f);
        } else if (v.dims == 3) {
            const V3 eye = eye_of(c);
            const V3 f = norm3(sub(V3{c.tx, c.ty, c.tz}, eye));
            const V3 s = norm3(cross(f, V3{0, 1, 0}));
            const V3 u = cross(s, f);
            const float k = 2.0f * c.dist * std::tan(kFovY * 0.5f) / avail.y;
            c.tx += (-s.x * dx + u.x * dy) * k;
            c.ty += (-s.y * dx + u.y * dy) * k;
            c.tz += (-s.z * dx + u.z * dy) * k;
        } else {
            const float k = 2.0f * c.half_h / avail.y;
            c.tx -= dx * k;
            c.ty += dy * k;
        }
        v.dirty = true;
        v.hover_dirty = true;
    }
    if (hovered && io.MouseWheel != 0.0f) {
        const float f = std::pow(0.9f, io.MouseWheel);
        if (v.dims == 3) {
            c.dist = std::clamp(c.dist * f, c.radius * 1e-3f, c.radius * 1e3f);
        } else {
            const float mx = (io.MousePos.x - p0.x) / avail.x * 2.0f - 1.0f;
            const float my = 1.0f - (io.MousePos.y - p0.y) / avail.y * 2.0f;
            const float wx = c.tx + mx * c.half_h * aspect, wy = c.ty + my * c.half_h;
            c.half_h = std::clamp(c.half_h * f, c.radius * 1e-5f, c.radius * 1e5f);
            c.tx = wx - mx * c.half_h * aspect;
            c.ty = wy - my * c.half_h;
        }
        v.dirty = true;
        v.hover_dirty = true;
    }
    if ((hovered || ImGui::IsWindowFocused()) && ImGui::IsKeyPressed(ImGuiKey_F, false) && v.have_cur)
        fit_points(v, aspect);

    // ---- GPU: redraw the render target when something changed
    float V[16], P[16];
    camera_matrices(v, aspect, V, P);
    if (v.dirty) {
        bool ok;
        if (v.have_cur) {
            HostGeomDraw d;
            d.pos_alloc = v.cur.pos->rid;
            d.pos_offset = v.cur.pos_off;
            d.vertex_count = v.cur.n;
            d.topology = CALIPER_GEOM_TOPO_POINTS;
            d.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
            d.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
            d.depth_flags = v.dims == 3 ? (CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE) : 0u;
            d.size_px = point_size_px(v.cur.n, core->scale);
            if (v.cur.aux) {
                d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
                d.attr_alloc = v.cur.aux->rid;
                d.attr_offset = v.cur.aux_off;
                d.lut256 = v.lut;
                d.vmin = v.pvmin;
                d.vmax = v.pvmax;
            } else {
                d.color_mode = CALIPER_GEOM_COLOR_FLAT;
                d.flat_rgba = kFlat;
            }
            ok = r->geom_draw_primitives(v.view_rid, V, P, &d, 1, kClear);
            if (ok) {
                void* h = r->geom_draw_retain_last();
                if (v.cur.last_draw) r->device_update_release(v.cur.last_draw);
                v.cur.last_draw = h;
            }
        } else {
            ok = r->geom_draw_primitives(v.view_rid, V, P, nullptr, 0, kClear);
        }
        v.dirty = false;   // a refused draw is not retried every frame (it logs)
        v.draw_failed = !ok;
    }
    if (v.draw_failed) ImGui::TextUnformatted("points draw refused by the renderer (see log)");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)r->tex_imtexture_id(v.view_rid), p0,
                 ImVec2(p0.x + avail.x, p0.y + avail.y));

    // ---- hover
    if (hovered && !active && v.have_cur) {
        if (!v.cur.pos_cpu) {
            ImGui::SetTooltip("hover unavailable: the payload buffer is not CPU-visible");
        } else {
            const ImVec2 m = io.MousePos;
            if (v.hover_dirty || m.x != v.hover_mx || m.y != v.hover_my) {
                float mvp[16];
                mul_cm(P, V, mvp);
                const float half = 0.5f * point_size_px(v.cur.n, core->scale) / core->scale + 1.0f;
                const Pick pk = pick_point(v, mvp, p0, avail, m, half, 8.0f, &v.hover_stride);
                v.hover_idx = pk.idx;
                v.hover_sx = pk.sx;
                v.hover_sy = pk.sy;
                v.hover_mx = m.x;
                v.hover_my = m.y;
                v.hover_dirty = false;
            }
            if (v.hover_idx >= 0 && (uint64_t)v.hover_idx < v.cur.n) {
                const float* q = (const float*)v.cur.pos_cpu + 3 * v.hover_idx;
                dl->AddCircle(ImVec2(v.hover_sx, v.hover_sy), 6.0f, IM_COL32(255, 255, 255, 230), 0, 1.5f);
                char line[96] = "";
                if (v.cur.aux && v.cur.aux_cpu) {
                    const float a = ((const float*)v.cur.aux_cpu)[v.hover_idx];
                    if (v.label_count > 0) std::snprintf(line, sizeof line, "\nlabel %.0f", a);
                    else std::snprintf(line, sizeof line, "\nvalue %.6g", a);
                }
                if (v.dims == 3)
                    ImGui::SetTooltip("#%lld%s\n(%.5g, %.5g, %.5g)%s", (long long)v.hover_idx,
                                      v.hover_stride > 1 ? " (sampled)" : "", q[0], q[1], q[2], line);
                else
                    ImGui::SetTooltip("#%lld%s\n(%.5g, %.5g)%s", (long long)v.hover_idx,
                                      v.hover_stride > 1 ? " (sampled)" : "", q[0], q[1], line);
            }
        }
    }
    ImGui::End();
}

// ---- series ----------------------------------------------------------------
// Min/max envelope of the series into ~2*buckets points (index order), so a
// long ring draws in bounded time and keeps its spikes.
void decimate_series(View& v, size_t buckets) {
    v.xs.clear();
    v.ys.clear();
    const size_t n = v.values.size();
    const size_t per = (n + buckets - 1) / buckets;
    for (size_t b0 = 0; b0 < n; b0 += per) {
        const size_t b1 = std::min(n, b0 + per);
        size_t imin = b0, imax = b0;
        for (size_t i = b0; i < b1; ++i) {
            if (v.values[i] < v.values[imin]) imin = i;
            if (v.values[i] > v.values[imax]) imax = i;
        }
        const size_t a = std::min(imin, imax), b = std::max(imin, imax);
        v.xs.push_back((double)(v.first + a));
        v.ys.push_back(v.values[a]);
        if (b != a) {
            v.xs.push_back((double)(v.first + b));
            v.ys.push_back(v.values[b]);
        }
    }
}

void draw_series(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    const bool open = begin_window(core, v, ImVec2(360.0f, 200.0f));
    if (!open) { ImGui::End(); return; }
    pause_button(S);
    ImGui::SameLine();
    char tel[160];
    telemetry(tel, sizeof tel, S, v);
    const size_t n = v.values.size();
    if (n)
        ImGui::Text("series n=%zu  latest %.6g @ step %llu  %s", n, v.values.back(),
                    (unsigned long long)(v.first + n - 1), tel);
    else
        ImGui::Text("series (no values yet)  %s", tel);
    if (ImPlot::BeginPlot("##caliper.series", ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("step", nullptr, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        if (n) {
            if (n <= kSeriesRawMax) {
                ImPlot::PlotLine("##line", v.values.data(), (int)n, 1.0, (double)v.first);
            } else {
                if (v.draw_dirty) decimate_series(v, 2048);
                ImPlot::PlotLine("##line", v.xs.data(), v.ys.data(), (int)v.xs.size());
            }
            v.draw_dirty = false;
            if (ImPlot::IsPlotHovered()) {
                const ImPlotPoint mp = ImPlot::GetPlotMousePos();
                const double rel = std::round(mp.x - (double)v.first);
                const size_t i = (size_t)std::clamp(rel, 0.0, (double)(n - 1));
                const double x = (double)(v.first + i), y = v.values[i];
                ImPlotSpec spec;
                spec.Marker = ImPlotMarker_Circle;
                spec.MarkerSize = 4.0f;
                ImPlot::PlotScatter("##hover", &x, &y, 1, spec);
                ImGui::SetTooltip("step %llu\nvalue %.6g", (unsigned long long)(v.first + i),
                                  (double)v.values[i]);
            }
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---- histogram -------------------------------------------------------------
void draw_histogram(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    const bool open = begin_window(core, v, ImVec2(360.0f, 200.0f));
    if (!open) { ImGui::End(); return; }
    pause_button(S);
    ImGui::SameLine();
    char tel[160];
    telemetry(tel, sizeof tel, S, v);
    const size_t B = v.values.size();
    ImGui::Text("histogram bins=%zu [%.4g, %.4g]  %s", B, v.lo, v.hi, tel);
    if (ImPlot::BeginPlot("##caliper.histogram", ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes(nullptr, "count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        if (B) {
            const double w = ((double)v.hi - (double)v.lo) / (double)B;
            if (v.draw_dirty || v.xs.size() != B) {
                v.xs.resize(B);
                v.ys.resize(B);
                for (size_t i = 0; i < B; ++i) {
                    v.xs[i] = (double)v.lo + ((double)i + 0.5) * w;
                    v.ys[i] = v.values[i];
                }
                v.draw_dirty = false;
            }
            ImPlot::PlotBars("##bars", v.xs.data(), v.ys.data(), (int)B, w);
            if (ImPlot::IsPlotHovered()) {
                const ImPlotPoint mp = ImPlot::GetPlotMousePos();
                const double fi = std::floor((mp.x - (double)v.lo) / w);
                if (fi >= 0.0 && fi < (double)B) {
                    const size_t i = (size_t)fi;
                    ImGui::SetTooltip("[%.6g, %.6g)\ncount %.6g", (double)v.lo + (double)i * w,
                                      (double)v.lo + (double)(i + 1) * w, (double)v.values[i]);
                }
            }
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---- heatmap / image overlay on the v1.2 window -----------------------------
void draw_texture_overlay(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    const TensorView* tv = nullptr;
    for (const auto& t : core->views)
        if (t.name == v.name) { tv = &t; break; }
    // Only a windowed v1.2 view that views_draw began this frame (placed is set
    // on its first windowed draw); RAW views have no chrome by design.
    if (!tv || !tv->tex || !tv->placed || (tv->flags & CALIPER_VIEW_FLAG_RAW)) return;
    if (ImGui::Begin(window_label(v).c_str())) {   // appends to that window
        char tel[160];
        telemetry(tel, sizeof tel, S, v);
        char line[200];
        std::snprintf(line, sizeof line, "%s  %s", v.kind == CALIPER_LIVE_KIND_HEATMAP ? "heatmap" : "image", tel);
        const ImVec2 wp = ImGui::GetWindowPos(), ws = ImGui::GetWindowSize();
        const ImGuiStyle& st = ImGui::GetStyle();
        const ImVec2 ts = ImGui::CalcTextSize(line);
        const float bh = ImGui::GetFrameHeight();
        const float y = wp.y + ws.y - st.WindowPadding.y - bh;
        const float x = wp.x + st.WindowPadding.x;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(x - 2.0f, y), ImVec2(std::min(wp.x + ws.x - st.WindowPadding.x, x + ts.x + 4.0f), y + bh),
                          IM_COL32(0, 0, 0, 150), 3.0f);
        dl->AddText(ImVec2(x, y + (bh - ts.y) * 0.5f), IM_COL32(230, 230, 235, 255), line);
        const float btn_w = ImGui::CalcTextSize("Resume").x + st.FramePadding.x * 2.0f;
        ImGui::SetCursorScreenPos(ImVec2(wp.x + ws.x - st.WindowPadding.x - btn_w, y));
        pause_button(S);
        const bool on_button = ImGui::IsItemHovered();

        // Hover: the image rect exactly as views_draw lays it out (info line,
        // then the image scaled to fit the rest, centred horizontally).
        const ImVec2 img((float)tv->w / core->scale, (float)tv->h / core->scale);
        const ImVec2 cs = ImGui::GetCursorStartPos();
        const float x_start = wp.x + cs.x;
        const float top = wp.y + cs.y + ImGui::GetTextLineHeightWithSpacing();
        const float aw = wp.x + ws.x - st.WindowPadding.x - x_start;
        const float ah = wp.y + ws.y - st.WindowPadding.y - top;
        const float sc = (img.x > 0 && img.y > 0) ? std::min(aw / img.x, ah / img.y) : 0.0f;
        const ImVec2 m = ImGui::GetIO().MousePos;
        if (sc > 0.0f && !on_button && ImGui::IsWindowHovered()) {
            const float x0 = x_start + std::max(0.0f, (aw - img.x * sc) * 0.5f);
            const float fx = (m.x - x0) / (img.x * sc), fy = (m.y - top) / (img.y * sc);
            if (fx >= 0.0f && fx < 1.0f && fy >= 0.0f && fy < 1.0f) {
                const int col = std::min(tv->w - 1, (int)(fx * (float)tv->w));
                const int row = std::min(tv->h - 1, (int)(fy * (float)tv->h));
                const bool live_buf = v.tex_cpu && tv->alloc && tv->alloc_buf == v.tex_buf;
                const size_t i = (size_t)row * (size_t)tv->w + (size_t)col;
                if (!live_buf) {
                    ImGui::SetTooltip("(%d, %d)\nvalue unavailable (payload not CPU-visible)", row, col);
                } else if (v.kind == CALIPER_LIVE_KIND_HEATMAP) {
                    ImGui::SetTooltip("(%d, %d)\nvalue %.6g", row, col, (double)((const float*)v.tex_cpu)[i]);
                } else {
                    const uint8_t* q = v.tex_cpu + i * 4;
                    ImGui::SetTooltip("(%d, %d)\nrgba %u %u %u %u", row, col, q[0], q[1], q[2], q[3]);
                }
            }
        }
    }
    ImGui::End();
}

}  // namespace

void camera_matrices(const View& v, float aspect, float view16[16], float proj16[16]) {
    const Camera& c = v.cam;
    if (!(aspect > 0.0f)) aspect = 1.0f;
    if (v.dims == 3) {
        look_at(eye_of(c), V3{c.tx, c.ty, c.tz}, V3{0, 1, 0}, view16);
        const float r = std::max(c.radius, 1e-6f);
        float zn = std::max(c.dist - 2.0f * r, std::max(c.dist * 1e-3f, 1e-6f));
        float zf = c.dist + 2.0f * r;
        if (zf <= zn * 1.001f) zf = zn * 2.0f;
        perspective(kFovY, aspect, zn, zf, proj16);
    } else {
        std::memset(view16, 0, 16 * sizeof(float));
        view16[0] = view16[5] = view16[10] = view16[15] = 1.0f;
        const float hh = std::max(c.half_h, 1e-12f), hw = hh * aspect;
        std::memset(proj16, 0, 16 * sizeof(float));
        proj16[0] = 1.0f / hw;
        proj16[5] = 1.0f / hh;
        proj16[10] = 0.0f;           // z ignored: every point at NDC depth 0.5
        proj16[12] = -c.tx / hw;
        proj16[13] = -c.ty / hh;
        proj16[14] = 0.5f;
        proj16[15] = 1.0f;
    }
}

void draw_view(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    switch (v.kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
        case CALIPER_LIVE_KIND_IMAGE:     draw_texture_overlay(core, S, v); break;
        case CALIPER_LIVE_KIND_POINTS:    draw_points(core, S, v); break;
        case CALIPER_LIVE_KIND_SERIES:    draw_series(core, S, v); break;
        case CALIPER_LIVE_KIND_HISTOGRAM: draw_histogram(core, S, v); break;
        default: break;
    }
}

}  // namespace caliper_live_views
