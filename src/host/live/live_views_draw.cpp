// live_views_draw.cpp: the windows of the typed live views (caliper/
// live_views.h), drawn by caliper_live_views_frame after the embed v1.2 views.
//
// One ImGui window per view, titled with the view name and keyed
// "<name>###caliper.view.<name>" like the v1.2 views (the layout persists by
// name in <data_dir>/imgui.ini), tiled on first placement with the same cursor
// as the v1.2 views. Every window starts with a Pause/Resume button (one GLOBAL
// pause: caliper_core_live_set_paused) and an info line.
//
//   POINTS     a scene of the scene core drawn through the graphics layer
//              (live_points.h) into a target sized to the window's content
//              region (physical pixels; recreated when it drifts more than 8
//              px), redrawn from the imported payload buffers whenever the
//              payload, the camera or the size changed (zero-copy on the in-place
//              route: the GPU reads the producer's buffer). 3-D: a drag orbits, a right, middle, alt or
//              ctrl drag pans, the wheel zooms. 2-D: a drag pans, the wheel
//              zooms about the cursor. A double click, the Fit button or the F
//              key fits the camera to the bbox. Opaque points with depth test
//              in 3-D, so categorical colours stay exact under overlap. Hover:
//              nearest point under the cursor from the CPU-visible payload
//              (sampled above 256K points).
//   SERIES     ImPlot line over step index, min/max decimated above 8192
//              values, hover tooltip (step, value), latest value in text.
//   HISTOGRAM  ImPlot bars at bin centres, hover tooltip (bin range, count).
//              Both fit their axes to the data every frame while they follow
//              it; a drag or the wheel in the plot stops following and keeps
//              the view, and a double click follows again.
//   HEATMAP / IMAGE  windows of the v1.2 views; this file draws their image
//              with pan (a drag), zoom about the pointer (the wheel) and a fit
//              (a double click), and appends the telemetry line and the Pause
//              button below it.
//
// The gestures are the browser element's (web/src/views), in the app and in
// the browser alike, and only the plot area takes them, so a drag there never
// moves a floating window; its title bar still does.
//
// EVENTS. When the host set an event sink (set_event_sink: the browser does,
// and sends them to the producer), the selections, picks and hovers of the
// views leave as events with the browser element's details (wire protocol v3
// "Events"), in source indices through the snapshot's payload map: a
// shift-drag selects (points: a lasso, with alt a rectangle; heatmap and
// image: a block of cells; series: a range of steps; histogram: a range of
// bins), a shift-click clears the selection, a click picks, and hovers go out
// at most 30 times a second. Selected points draw over the others, which are
// dimmed. Without a sink a selection stays in the view.
#include "live/live_views_state.h"
#include "live/camera_drag.h"

#include "live/live_points.h"
#include "live/live_scene.h"
#include "live/live_views_scene.h"
#include "live/scene_overlays.h"
#include "embed/render_stats.h"   // copy counters
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"   // colormap_lut

#include <imgui.h>
#include <imgui_internal.h>   // the header's items in the title bar
#include <implot.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using caliper_embed_views::TensorView;
using caliper_host::HostRenderer;

namespace caliper_live_views {

namespace {

constexpr int      kHysteresisPx = 8;
constexpr int      kMaxTarget = 8192;
constexpr uint64_t kHoverBudget = 1u << 18;
constexpr size_t   kSeriesRawMax = 8192;

std::string window_label(const View& v) { return v.name + "###caliper.view." + v.name; }

// ---- events ------------------------------------------------------------------
constexpr float  kClickSlop = 3.0f;              // px a press may move and still be a click
constexpr double kHoverMs = 1000.0 / 30.0;       // at most 30 hovers a second reach the sink
constexpr size_t kMaxEventIndices = 32768;       // wire protocol v3, Events
const ImU32 kSelLine = IM_COL32(0xb3, 0x8b, 0x59, 255);   // bronze, as the browser views
const ImU32 kSelFill = IM_COL32(0xb3, 0x8b, 0x59, 41);
const ImU32 kSelText = kSelLine;

double now_ms() { return (double)now_ns() * 1e-6; }

// ImGui sizes a face by its line height, CSS by its em: the house mono,
// Source Code Pro, is 1.257 em from ascent to descent, so CSS 10.5 px is
// 13.2 ImGui px (sternbild_fonts.h sizes it alike: 15 for CSS 11.9).
constexpr float kMonoEm = 1.257f;

// The host's mono face, found by the file name it was loaded from, or null.
ImFont* mono_face() {
    for (ImFont* f : ImGui::GetIO().Fonts->Fonts)
        if (std::strstr(f->GetDebugName(), "Code") || std::strstr(f->GetDebugName(), "Mono")) return f;
    return nullptr;
}

// A bare view on a page (the browser element's plate): its plots' ticks in
// the element's 10.5 px mono, as the TypeScript views draw them. Returns
// whether a font was pushed (pop it after the plot).
bool push_page_ticks(const CaliperCore* core, const CaliperLiveViewsState* S) {
    if (!core->views_bare || !S->events) return false;
    ImFont* f = mono_face();
    if (!f) return false;
    ImGui::PushFont(f, 10.5f * kMonoEm);
    return true;
}

// The shortest decimal text that reads back as the same double (as
// JavaScript prints numbers), or null for a non-finite one.
std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[40];
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(buf, sizeof buf, "%.*g", p, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    return buf;
}

// A detail object, written in the order the browser views write theirs.
class Detail {
public:
    Detail& num(const char* k, double v) { key(k); s_ += json_number(v); return *this; }
    Detail& integer(const char* k, int64_t v) { key(k); s_ += std::to_string(v); return *this; }
    Detail& pair(const char* k, double a, double b) {
        key(k);
        s_ += "[" + json_number(a) + "," + json_number(b) + "]";
        return *this;
    }
    Detail& ipair(const char* k, int64_t a, int64_t b) {
        key(k);
        s_ += "[" + std::to_string(a) + "," + std::to_string(b) + "]";
        return *this;
    }
    Detail& ints(const char* k, const std::vector<int64_t>& v) {
        key(k);
        s_ += "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s_ += ",";
            s_ += std::to_string(v[i]);
        }
        s_ += "]";
        return *this;
    }
    Detail& flag(const char* k, bool v) { key(k); s_ += v ? "true" : "false"; return *this; }
    Detail& text(const char* k, const char* v) {   // a plain name: no characters to escape
        key(k);
        s_ += "\"";
        s_ += v;
        s_ += "\"";
        return *this;
    }
    std::string str() const { return s_ + "}"; }

private:
    void key(const char* k) {
        if (s_.size() > 1) s_ += ",";
        s_ += "\"";
        s_ += k;
        s_ += "\":";
    }
    std::string s_ = "{";
};

void emit(CaliperLiveViewsState* S, const View& v, const char* type, std::string detail) {
    if (!S->events) return;
    LiveEvent e;
    e.view = v.name;
    e.type = type;
    e.detail = std::move(detail);
    S->events(e);
}

// A hover goes out when the hovered item changed (item -1: nothing hovered,
// which is not reported), and is dropped, not queued, inside 1/30 s of the last.
void emit_hover(CaliperLiveViewsState* S, View& v, int64_t item, const std::function<std::string()>& detail) {
    if (item == v.hover_sent) return;
    v.hover_sent = item;
    if (item < 0 || !S->events) return;
    const double t = now_ms();
    if (t - v.last_hover_ms < kHoverMs) return;
    v.last_hover_ms = t;
    emit(S, v, "hover", detail());
}

// Payload index -> source index (wire protocol v3, Payload axes): the first
// source position a payload index stands for, and the half-open source range
// of payload indices [i0, i1).
int64_t source_index(const caliper_live::PayloadAxis* ax, int64_t i) {
    if (!ax || ax->size <= 0) return i;
    return ax->start + (i * ax->count) / ax->size * ax->step;
}
void source_range(const caliper_live::PayloadAxis* ax, int64_t i0, int64_t i1, int64_t* a, int64_t* b) {
    if (!ax || ax->size <= 0) { *a = i0; *b = i1; return; }
    const int64_t p0 = (i0 * ax->count) / ax->size;
    const int64_t p1 = (i1 * ax->count + ax->size - 1) / ax->size;
    *a = ax->start + p0 * ax->step;
    *b = ax->start + (p1 - 1) * ax->step + 1;
}

// Whether a linked selection holds an item in the source range [a, b).
bool linked_in(const View& v, int64_t a, int64_t b) {
    auto it = std::lower_bound(v.link_items.begin(), v.link_items.end(), a);
    return it != v.link_items.end() && *it < b;
}

// A press, the drag and the release of a gesture on the item drawn just
// before (an InvisibleButton). Returns true on the release frame. In a view
// whose drag moves a camera (`camera`), a press that is not a selection is a
// drag once its pointer went more than kCameraClickTolerance points from the
// press (camera_drag.h), not past kClickSlop; its camera moves from the first
// point of motion all the same.
bool track_gesture(View& v, ImVec2 mouse, bool may_select, bool camera = false) {
    Gesture& g = v.gesture;
    ImGuiIO& io = ImGui::GetIO();
    g.pressed = false;
    if (ImGui::IsItemActivated()) {
        g = Gesture{};
        g.down = true;
        g.pressed = true;
        g.button = ImGui::IsMouseClicked(ImGuiMouseButton_Right) ? 1
                   : ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ? 2 : 0;
        g.shift = io.KeyShift;
        g.alt = io.KeyAlt;
        g.ctrl = io.KeyCtrl || io.KeySuper;
        g.x0 = g.lx = mouse.x;
        g.y0 = g.ly = mouse.y;
        g.selecting = may_select && g.shift && g.button == 0;
        g.rect = g.alt;
    }
    if (g.down) {
        const float ox = mouse.x - g.x0, oy = mouse.y - g.y0;
        if (camera && !g.selecting) {
            if (ImGui::IsMousePosValid(&mouse)) {
                g.max_d2 = std::max(g.max_d2, ox * ox + oy * oy);
                g.moved = !camera_press_is_click(g.max_d2);
            }
        } else if (!g.moved && std::hypot(ox, oy) > kClickSlop) {
            g.moved = true;
        }
    }
    if (g.down && ImGui::IsItemDeactivated()) {
        g.down = false;
        return true;
    }
    return false;
}

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
    if (!window_before(core, v.name)) {
        if (!v.placed) place(core, v, content);
        ImGui::SetNextWindowPos(ImVec2(v.pos_x, v.pos_y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(v.size_x, v.size_y), ImGuiCond_FirstUseEver);
    }
    // The resize grip only under the pointer: a view resizes from its edges.
    ImGui::PushStyleColor(ImGuiCol_ResizeGrip, 0);
    const bool open = ImGui::Begin(window_label(v).c_str(), nullptr,
                                   kWindowFlags | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                       (core->views_bare ? ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                                               ImGuiWindowFlags_NoMove : 0));
    ImGui::PopStyleColor();
    window_after(core, v.name);
    return open;
}

void set_follow(CaliperLiveViewsState* S, View& v, bool on);

// ---- the header ---------------------------------------------------------------------
// What a view's header says about it, as the browser element's header does
// (web/src/views/info.ts): 4 significant digits, an exponent below 1e-3 or
// from 1e5.
std::string fmt4(double v) {
    if (!std::isfinite(v)) return std::isnan(v) ? "NaN" : v > 0 ? "inf" : "-inf";
    if (v == 0.0) return "0";
    char b[48];
    const double a = std::fabs(v);
    if (a >= 1e5 || a < 1e-3) {
        std::snprintf(b, sizeof b, "%.3e", v);
        // "1.234e-04" as JavaScript writes it: "1.234e-4"
        std::string t = b;
        const size_t e = t.find('e');
        std::string mant = t.substr(0, e), ex = t.substr(e + 1);
        const bool neg = !ex.empty() && ex[0] == '-';
        if (!ex.empty() && (ex[0] == '-' || ex[0] == '+')) ex.erase(0, 1);
        while (ex.size() > 1 && ex[0] == '0') ex.erase(0, 1);
        if (mant.find('.') != std::string::npos) {
            while (mant.back() == '0') mant.pop_back();
            if (mant.back() == '.') mant.pop_back();
        }
        return mant + "e" + (neg ? "-" : "") + ex;
    }
    std::snprintf(b, sizeof b, "%.4g", v);
    std::snprintf(b, sizeof b, "%.10g", std::strtod(b, nullptr));   // 1.235e+04 -> 12350
    return b;
}

const char* const kTimes = "\xC3\x97";      // U+00D7
const char* const kEllipsis = "\xE2\x80\xA6";   // U+2026

// `s` cut to `w` with an ellipsis, "" when not even that fits.
std::string fit_text(const std::string& s, float w) {
    if (ImGui::CalcTextSize(s.c_str()).x <= w) return s;
    const float ell = ImGui::CalcTextSize(kEllipsis).x;
    if (ell > w) return {};
    std::string t = s;
    while (!t.empty()) {
        // One whole UTF-8 character off the end.
        while (!t.empty()) {
            const unsigned char c = (unsigned char)t.back();
            t.pop_back();
            if ((c & 0xC0) != 0x80) break;
        }
        while (!t.empty() && t.back() == ' ') t.pop_back();
        if (ImGui::CalcTextSize((t + kEllipsis).c_str()).x <= w) return t + kEllipsis;
    }
    return {};
}

enum Tool : unsigned { kToolPause = 1, kToolLog = 2, kToolFollow = 4, kToolRotate = 8, kToolUp = 16, kToolDisplay = 32 };

// The popup of a scene view's "display" tool (draw_scene opens its content).
constexpr const char* kSceneDisplayMenu = "##caliper.scene.display";

// A scene's up-axis segment: "Y up", "Z up", "-Y up".
std::string up_label(caliper_scene::Axis a) {
    const char* n = caliper_scene::axis_name(a);   // "+y", "-z"
    std::string s = n[0] == '-' ? "-" : "";
    s += char(n[1] - 'a' + 'A');
    return s + " up";
}

// A header tool: a word in a thin outlined box, as the browser element's
// tool buttons; bronze while pressed. True when clicked.
bool tool_button(const Look& L, const char* id, const char* text, bool pressed, ImVec2 pos, float h) {
    ImGuiWindow* w = ImGui::GetCurrentWindow();
    const ImVec2 ts = ImGui::CalcTextSize(text);
    const ImRect bb(pos, ImVec2(pos.x + ts.x + 12.0f, pos.y + h));
    const ImGuiID gid = w->GetID(id);
    if (!ImGui::ItemAdd(bb, gid)) return false;
    bool hovered = false, held = false;
    const bool clicked = ImGui::ButtonBehavior(bb, gid, &hovered, &held);
    ImDrawList* dl = w->DrawList;
    const ImU32 edge = pressed ? L.accent : L.rule;
    dl->AddRect(bb.Min, bb.Max, edge, 3.0f);
    const ImU32 ink = pressed ? L.accent : hovered ? L.text : L.faint;
    dl->AddText(ImVec2(bb.Min.x + 6.0f, bb.Min.y + (h - ts.y) * 0.5f), ink, text);
    return clicked;
}

float tool_width(const char* text) { return ImGui::CalcTextSize(text).x + 12.0f; }

// The view's header, as the browser element's: the name on the left (ImGui's
// title bar draws it), then on the right a compact info string, the tools
// (pause for every view, log for plots, follow for a series, rotate for 3-D
// points, the up-axis segments and the display menu of a scene) and a status
// dot: bronze while the stream is live and the view has data, an outline
// while it waits for its first data, grey once the stream has ended. The
// header lives in the window's title bar; in a docked window, whose tab
// carries the name, it takes the first line of the content. What does not
// fit goes, the info first (cut with an ellipsis), then the tools; nothing
// overlaps the name.
void draw_header(CaliperCore* core, CaliperLiveViewsState* S, View& v, unsigned tools) {
    if (core->views_bare) return;
    const Look& L = S->look;
    ImGuiWindow* w = ImGui::GetCurrentWindow();
    if (w->SkipItems) return;
    const ImGuiStyle& st = ImGui::GetStyle();
    const bool title = !(w->Flags & ImGuiWindowFlags_NoTitleBar) && !w->DockIsActive && w->TitleBarHeight > 0.0f;
    ImRect r;
    float left;
    if (title) {
        r = w->TitleBarRect();
        left = r.Min.x + st.FramePadding.x + ImGui::CalcTextSize(v.name.c_str(), nullptr, true).x + 14.0f;
    } else {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float hh = ImGui::GetFrameHeight();
        r = ImRect(p, ImVec2(p.x + ImGui::GetContentRegionAvail().x, p.y + hh));
        ImGui::Dummy(ImVec2(r.GetWidth(), hh));
        left = r.Min.x;
    }
    const ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImGui::PushClipRect(r.Min, r.Max, false);
    ImDrawList* dl = w->DrawList;
    const float cy = std::floor((r.Min.y + r.Max.y) * 0.5f) + 0.5f;
    float right = r.Max.x - st.FramePadding.x;

    // The dot.
    if (S->stream) {
        const ImVec2 c(right - 4.0f, cy);
        if (v.updates == 0) dl->AddCircle(c, 3.5f, L.faint, 12, 1.0f);
        else dl->AddCircleFilled(c, 4.0f, S->stream == 2 ? L.rule : L.accent, 12);
        right -= 8.0f + 10.0f;
    }

    // The tools, right to left, in the small size.
    if (L.small_size > 0.0f) ImGui::PushFont(nullptr, L.small_size);
    struct T {
        unsigned bit;
        std::string id;
        std::string text;
        bool pressed;
        caliper_scene::Axis axis = caliper_scene::Axis::PosY;   // kToolUp: the segment's axis
    };
    std::vector<T> all = {
        {kToolPause, "##caliper.tool.pause", S->paused ? "paused" : "pause", S->paused},
        {kToolLog, "##caliper.tool.log", "log", v.log_y},
        {kToolFollow, "##caliper.tool.follow", "follow", v.follow},
        {kToolRotate, "##caliper.tool.rotate", "rotate", v.auto_rotate},
    };
    // A scene's up-axis switch, one segment per mode with the active one pressed, and its display menu.
    SceneView* scene = (tools & (kToolUp | kToolDisplay)) ? scene_view(v) : nullptr;
    if (scene && (tools & kToolUp))
        for (caliper_scene::Axis a : scene->up_choices())
            all.push_back({kToolUp, std::string("##caliper.tool.up.") + caliper_scene::axis_name(a), up_label(a),
                           scene->up_mode() == a, a});
    if (scene && (tools & kToolDisplay)) all.push_back({kToolDisplay, "##caliper.tool.display", "display", false});
    float tools_w = 0.0f;
    for (const T& t : all)
        if (tools & t.bit) tools_w += tool_width(t.text.c_str()) + 4.0f;
    const float th = std::min(r.GetHeight() - 4.0f, ImGui::GetFontSize() + 6.0f);
    if (tools_w > 0.0f && right - tools_w >= left) {
        float x = right - tools_w + 4.0f;
        for (const T& t : all) {
            if (!(tools & t.bit)) continue;
            if (tool_button(L, t.id.c_str(), t.text.c_str(), t.pressed, ImVec2(x, std::floor(cy - th * 0.5f)), th)) {
                if (t.bit == kToolPause) S->paused = !S->paused;
                else if (t.bit == kToolUp) scene->switch_up(t.axis);   // the camera only: the producer is not told
                else if (t.bit == kToolDisplay) ImGui::OpenPopup(kSceneDisplayMenu);
                else if (t.bit == kToolLog) {
                    v.log_y = !v.log_y;
                    v.fit_once = true;
                    emit(S, v, "toggle", Detail().text("name", "log").flag("on", v.log_y).str());
                } else if (t.bit == kToolFollow) {
                    set_follow(S, v, !v.follow);
                    if (v.follow) v.fit_once = true;
                } else if (t.bit == kToolRotate) {
                    v.auto_rotate = !v.auto_rotate;
                    emit(S, v, "toggle", Detail().text("name", "rotate").flag("on", v.auto_rotate).str());
                }
            }
            x += tool_width(t.text.c_str()) + 4.0f;
        }
        right -= tools_w + 6.0f;
    }
    if (L.small_size > 0.0f) ImGui::PopFont();

    // The info string, in mono, right-aligned before the tools.
    const std::string info = info_text(core, v);
    if (!info.empty() && right > left) {
        ImGui::PushFont(L.mono, L.mono_size);
        const std::string t = fit_text(info, right - left);
        if (!t.empty()) {
            const ImVec2 ts = ImGui::CalcTextSize(t.c_str());
            dl->AddText(ImVec2(right - ts.x, cy - ts.y * 0.5f), L.faint, t.c_str());
        }
        ImGui::PopFont();
    }
    ImGui::PopClipRect();
    ImGui::SetCursorScreenPos(cursor);
}

// The host's row under the header (a data view picker), at the cursor.
float under_header(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    if (core->views_bare || !S->under_header) return 0.0f;
    return S->under_header(v, ImGui::GetContentRegionAvail().x);
}

// Under the header with CALIPER_VIEWS_DEBUG=1: the view's diagnostics on one
// line, cut to the window's width.
void debug_line(const CaliperCore* core, const std::string& text) {
    if (core->views_bare || !views_debug(core)) return;
    const std::string t = fit_text(text, ImGui::GetContentRegionAvail().x);
    ImGui::TextUnformatted(t.c_str());
}

// "seq 41  lat 2.3 ms  upd 41  drop 0  [paused]"; "rel lat" when the stamps came
// from another clock (CALIPER_LIVE_APPLY_STAMP_RELATIVE).
void telemetry(char* buf, size_t cap, const CaliperLiveViewsState* S, const View& v) {
    std::snprintf(buf, cap, "seq %llu  %s %.1f ms  upd %llu  drop %llu%s",
                  (unsigned long long)v.seq, v.latency_relative ? "rel lat" : "lat", v.latency_ms,
                  (unsigned long long)v.updates,
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
                float half_px, float att_ref, float radius, uint64_t* stride_out) {
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
        // With the depth cues a point is drawn larger or smaller by its distance.
        const float half = att_ref > 0.0f
            ? (half_px - 1.0f) * std::clamp(att_ref / cw, caliper_scene::kPointAttenMin, caliper_scene::kPointAttenMax) + 1.0f
            : half_px;
        if (std::fabs(dx) <= half && std::fabs(dy) <= half) {
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

// Screen positions of the current sample under `mvp` (NaN: behind the camera
// or outside the depth range), as pick_point projects them.
void project_points(const View& v, const float mvp[16], ImVec2 p0, ImVec2 sz, std::vector<float>* out) {
    const float* P = (const float*)v.cur.pos_cpu;
    const uint64_t n = v.cur.n;
    out->assign(2 * n, NAN);
    for (uint64_t i = 0; i < n; ++i) {
        const float x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
        const float cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
        if (!(cw > 1e-9f)) continue;
        const float nz = (mvp[2] * x + mvp[6] * y + mvp[10] * z + mvp[14]) / cw;
        if (!(nz >= 0.0f && nz <= 1.0f)) continue;
        (*out)[2 * i] = p0.x + ((mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12]) / cw * 0.5f + 0.5f) * sz.x;
        (*out)[2 * i + 1] = p0.y + (0.5f - (mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13]) / cw * 0.5f) * sz.y;
    }
}

// The browser element's selection of points (web/src/views/points.ts): the
// points whose screen position is inside the polygon (even-odd rule), reported
// as indices into the sample and their source rows, at most 32,768 of each.
void select_polygon(CaliperLiveViewsState* S, View& v, const float mvp[16], ImVec2 p0, ImVec2 sz,
                    const std::vector<float>& poly) {
    std::vector<float> pr;
    project_points(v, mvp, p0, sz, &pr);
    const size_t m = poly.size() / 2;
    float xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
    for (size_t k = 0; k < m; ++k) {
        xmin = std::min(xmin, poly[2 * k]); xmax = std::max(xmax, poly[2 * k]);
        ymin = std::min(ymin, poly[2 * k + 1]); ymax = std::max(ymax, poly[2 * k + 1]);
    }
    const uint64_t n = v.cur.n;
    std::vector<uint8_t> sel(n, 0);
    std::vector<int64_t> idx;
    for (uint64_t i = 0; i < n; ++i) {
        const float px = pr[2 * i], py = pr[2 * i + 1];
        if (!(px >= xmin && px <= xmax && py >= ymin && py <= ymax)) continue;
        bool inside = false;
        for (size_t a = 0, b = m - 1; a < m; b = a++) {
            const float ax = poly[2 * a], ay = poly[2 * a + 1], bx = poly[2 * b], by = poly[2 * b + 1];
            if ((ay > py) != (by > py) && px < ((bx - ax) * (py - ay)) / (by - ay) + ax) inside = !inside;
        }
        if (inside) {
            sel[i] = 1;
            idx.push_back((int64_t)i);
        }
    }
    v.selected = idx.empty() ? std::vector<uint8_t>() : std::move(sel);
    v.sel_count = idx.size();
    ++v.sel_version;
    v.dirty = true;
    const bool cut = idx.size() > kMaxEventIndices;
    if (cut) idx.resize(kMaxEventIndices);
    const caliper_live::PayloadAxis* ax = v.map.axes.empty() ? nullptr : &v.map.axes.back();
    std::vector<int64_t> rows(idx.size());
    for (size_t k = 0; k < idx.size(); ++k) rows[k] = source_index(ax, idx[k]);
    Detail d;
    d.ints("indices", idx).ints("row_ids", rows);
    if (cut) d.flag("truncated", true);
    emit(S, v, "select", d.str());
}

std::string point_detail(const View& v, int64_t i) {
    const caliper_live::PayloadAxis* ax = v.map.axes.empty() ? nullptr : &v.map.axes.back();
    return Detail().integer("index", i).integer("row_id", source_index(ax, i)).str();
}

// The selected points' positions and aux, staged for the renderer, rebuilt
// when the selection or the payload changed. The one it replaces retires like
// a superseded payload.
bool stage_selection(CaliperCore* core, View& v) {
    if (v.have_sel_draw && v.sel_draw_for == v.sel_version && v.sel_draw_updates == v.updates) return true;
    if (v.have_sel_draw) v.retired.push_back(v.sel_draw);
    v.have_sel_draw = false;
    v.sel_draw = PointsPayload{};
    if (!v.cur.pos_cpu || v.selected.size() != v.cur.n || !v.sel_count) return false;
    std::vector<float> pos, aux;
    pos.reserve(3 * v.sel_count);
    const float* P = (const float*)v.cur.pos_cpu;
    const float* A = (const float*)v.cur.aux_cpu;
    for (uint64_t i = 0; i < v.cur.n; ++i) {
        if (!v.selected[i]) continue;
        pos.insert(pos.end(), P + 3 * i, P + 3 * i + 3);
        if (v.cur.aux && A) aux.push_back(A[i]);
    }
    PointsPayload p;
    p.n = v.sel_count;
    std::string why;
    if (!stage_host(core, v, pos.data(), pos.size() * 4, &p.pos, &p.pos_cpu, &why)) return false;
    if (!aux.empty() && !stage_host(core, v, aux.data(), aux.size() * 4, &p.aux, &p.aux_cpu, &why)) {
        drop_payload(core, v, p);
        return false;
    }
    v.sel_draw = p;
    v.have_sel_draw = true;
    v.sel_draw_for = v.sel_version;
    v.sel_draw_updates = v.updates;
    return true;
}

// The look of the view's points for the scene: its colour source, range, box and size.
PointsLook points_look(const View& v, uint32_t ground) {
    PointsLook l;
    l.dims = v.dims;
    l.colour = !v.cur.aux ? PointsColour::Flat : v.label_count > 0 ? PointsColour::Classes : PointsColour::Values;
    l.colormap = v.colormap;
    l.vmin = v.pvmin;
    l.vmax = v.pvmax;
    l.has_bbox = v.has_bbox;
    std::memcpy(l.bbox, v.bbox, sizeof l.bbox);
    l.size_px = caliper_scene::default_point_size(v.cur.n);
    l.ground = ground;
    return l;
}

PointsArrays points_arrays(const PointsPayload& p) {
    PointsArrays a;
    a.pos = p.pos ? p.pos->buffer : caliper_gfx::Buffer{};
    a.pos_offset = p.pos_off;
    a.pos_cpu = p.pos_cpu;
    a.aux = p.aux ? p.aux->buffer : caliper_gfx::Buffer{};
    a.aux_offset = p.aux_off;
    a.count = p.n;
    return a;
}

// ---- the box around 3-D points ------------------------------------------------
// An ImPlot3D-style frame around the data, so a still image reads in depth:
// the box's twelve edges, tick marks and labels on one silhouette edge of
// each axis with the axis's name, and a faint grid on the floor, the face
// across the up axis that lies behind the data. The back layer (the edges
// between two faces turned away from the camera, and the grid) is drawn
// under the points' image, whose ground is transparent in 3-D, and the front
// layer (every other edge, the ticks and the labels) over it: everything in
// the box is in front of its back faces and behind its front faces, so the
// order gives the depth without a depth buffer. web/src/views/box3d.ts draws
// the same frame for the TypeScript views.
struct BoxInk {
    ImU32 edge, back, grid, tick, text;
    ImFont* font;
    float font_px, k;   // the type's ImGui size; the layout's scale (10.5 px = 1)
};

struct V3 { float x, y, z; };
struct Clip4 { float x, y, z, w; };

Clip4 clip_of(const float m[16], V3 p) {
    return {m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12], m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14], m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15]};
}

ImVec2 screen_of(const Clip4& c, ImVec2 p0, ImVec2 sz) {
    return ImVec2(p0.x + (c.x / c.w * 0.5f + 0.5f) * sz.x, p0.y + (0.5f - c.y / c.w * 0.5f) * sz.y);
}

// A segment's part in front of the near plane (clip z >= 0), on screen.
bool clip_segment(Clip4 a, Clip4 b, ImVec2 p0, ImVec2 sz, ImVec2* sa, ImVec2* sb) {
    if (a.z < 0.0f && b.z < 0.0f) return false;
    auto lerp = [](const Clip4& p, const Clip4& q, float t) {
        return Clip4{p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t, p.z + (q.z - p.z) * t, p.w + (q.w - p.w) * t};
    };
    if (a.z < 0.0f) a = lerp(a, b, a.z / (a.z - b.z));
    else if (b.z < 0.0f) b = lerp(b, a, b.z / (b.z - a.z));
    if (!(a.w > 1e-9f) || !(b.w > 1e-9f)) return false;
    *sa = screen_of(a, p0, sz);
    *sb = screen_of(b, p0, sz);
    return true;
}

// Round ticks in [lo, hi], about n of them (web/src/views/plot.ts, ticks()).
std::vector<double> nice_ticks(double lo, double hi, int n, double* step_out) {
    std::vector<double> out;
    *step_out = 1.0;
    if (!(hi > lo) || !std::isfinite(lo) || !std::isfinite(hi)) return out;
    const double raw = (hi - lo) / std::max(1, n);
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double r = raw / mag;
    const double step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    *step_out = step;
    for (double t = std::ceil(lo / step) * step; t <= hi + step * 1e-9 && out.size() < 32; t += step)
        out.push_back(std::fabs(t) < step * 1e-9 ? 0.0 : t);
    return out;
}

// A tick's label as the browser's axes print it (plot.ts, fmtTick).
std::string tick_text(double v, double step) {
    char b[48];
    if (std::fabs(v) >= 1e5 || (std::fabs(v) < 1e-3 && v != 0.0)) {
        std::snprintf(b, sizeof b, "%.2e", v);   // JavaScript's toExponential(2), "e+05" as "e5"
        const std::string s = b;
        const size_t e = s.find('e');
        if (e == std::string::npos) return s;
        std::string mant = s.substr(0, e), ex = s.substr(e + 1);
        const bool neg = !ex.empty() && ex[0] == '-';
        if (!ex.empty() && (ex[0] == '+' || ex[0] == '-')) ex.erase(0, 1);
        while (ex.size() > 1 && ex[0] == '0') ex.erase(0, 1);
        return mant + "e" + (neg ? "-" : "") + ex;
    }
    const int decimals = std::min(6, std::max(0, (int)-std::floor(std::log10(step) + 1e-9)));
    std::snprintf(b, sizeof b, "%.*f", decimals, v);
    return b;
}

// The box's frame for one frame: its corners in clip space, which faces are
// turned towards the camera, the edge of each axis that carries its labels
// and that axis's ticks.
struct BoxFrame {
    float b[6] = {};
    float m[16] = {};
    Clip4 corner[8] = {};
    bool front[3][2] = {};
    int label_edge[3] = {-1, -1, -1};
    std::vector<double> ticks[3];
    double step[3] = {1.0, 1.0, 1.0};
    int floor_side = 0;
    ImVec2 centre;
};

int corner_index(int a, int at, int a1, int u, int a2, int w) {
    int bits[3];
    bits[a] = at;
    bits[a1] = u;
    bits[a2] = w;
    return bits[0] | (bits[1] << 1) | (bits[2] << 2);
}

V3 box_point(const float b[6], int a, double t, int a1, int u, int a2, int w) {
    float c[3];
    c[a] = (float)t;
    c[a1] = b[2 * a1 + u];
    c[a2] = b[2 * a2 + w];
    return {c[0], c[1], c[2]};
}

BoxFrame box_frame(const float box[6], V3 eye, const float mvp[16], ImVec2 p0, ImVec2 sz) {
    BoxFrame f;
    std::memcpy(f.b, box, sizeof f.b);
    std::memcpy(f.m, mvp, sizeof f.m);
    for (int i = 0; i < 8; ++i)
        f.corner[i] = clip_of(mvp, V3{box[(i & 1) ? 1 : 0], box[(i & 2) ? 3 : 2], box[(i & 4) ? 5 : 4]});
    const float e[3] = {eye.x, eye.y, eye.z};
    for (int a = 0; a < 3; ++a) {
        f.front[a][0] = e[a] < box[2 * a];       // the face at the low end faces -a
        f.front[a][1] = e[a] > box[2 * a + 1];
    }
    // The floor: the low face across the up axis, or the high one when the
    // camera is below the box, whichever is behind the data.
    f.floor_side = eye.y < box[2] ? 1 : 0;
    const Clip4 cc = clip_of(mvp, V3{0.5f * (box[0] + box[1]), 0.5f * (box[2] + box[3]), 0.5f * (box[4] + box[5])});
    f.centre = cc.w > 1e-9f ? screen_of(cc, p0, sz) : ImVec2(p0.x + sz.x * 0.5f, p0.y + sz.y * 0.5f);
    for (int a = 0; a < 3; ++a) {
        const int a1 = (a + 1) % 3, a2 = (a + 2) % 3;
        float best = 0.0f, len = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const int u = k & 1, w = k >> 1;
            if ((int)f.front[a1][u] + (int)f.front[a2][w] != 1) continue;   // a silhouette edge
            const Clip4 A = f.corner[corner_index(a, 0, a1, u, a2, w)], B = f.corner[corner_index(a, 1, a1, u, a2, w)];
            if (A.z < 0.0f || B.z < 0.0f || !(A.w > 1e-9f) || !(B.w > 1e-9f)) continue;
            const ImVec2 sa = screen_of(A, p0, sz), sb = screen_of(B, p0, sz);
            // x and z label their lowest silhouette edge on screen, y its leftmost.
            const float key = a == 1 ? -(sa.x + sb.x) : sa.y + sb.y;
            if (f.label_edge[a] < 0 || key > best) {
                best = key;
                f.label_edge[a] = k;
                len = std::hypot(sb.x - sa.x, sb.y - sa.y);
            }
        }
        if (f.label_edge[a] >= 0 && len < 24.0f) f.label_edge[a] = -1;   // too short to label
        // About one tick every 50 px along the edge.
        const int n = f.label_edge[a] >= 0 ? std::clamp((int)std::floor(len / 50.0f), 3, 8) : 4;
        f.ticks[a] = nice_ticks(box[2 * a], box[2 * a + 1], n, &f.step[a]);
    }
    return f;
}

// The back layer: the grid on the floor at the x and z ticks, and the back edges.
void draw_box_back(ImDrawList* dl, const BoxFrame& f, ImVec2 p0, ImVec2 sz, const BoxInk& ink) {
    ImVec2 sa, sb;
    const int fs = f.floor_side;
    for (int ax : {0, 2}) {
        const int other = ax == 0 ? 2 : 0;
        for (double t : f.ticks[ax]) {
            const V3 a = box_point(f.b, ax, t, 1, fs, other, 0);
            const V3 b = box_point(f.b, ax, t, 1, fs, other, 1);
            if (clip_segment(clip_of(f.m, a), clip_of(f.m, b), p0, sz, &sa, &sb)) dl->AddLine(sa, sb, ink.grid, 1.0f);
        }
    }
    for (int a = 0; a < 3; ++a) {
        const int a1 = (a + 1) % 3, a2 = (a + 2) % 3;
        for (int k = 0; k < 4; ++k) {
            const int u = k & 1, w = k >> 1;
            if (f.front[a1][u] || f.front[a2][w]) continue;
            if (clip_segment(f.corner[corner_index(a, 0, a1, u, a2, w)], f.corner[corner_index(a, 1, a1, u, a2, w)],
                             p0, sz, &sa, &sb))
                dl->AddLine(sa, sb, ink.back, 1.0f);
        }
    }
}

// The front layer: the other edges, the ticks and their labels pointing away
// from the box's centre, and each axis's name beyond them; a label that would
// overlap the one before it is left out. `labels` gets the labels drawn, as
// "x=-4|0|4;y=...;z=..." (box3d.ts reports the same).
void draw_box_front(ImDrawList* dl, const BoxFrame& f, ImVec2 p0, ImVec2 sz, const BoxInk& ink, std::string* labels) {
    ImVec2 sa, sb;
    labels->clear();
    for (int a = 0; a < 3; ++a) {
        const int a1 = (a + 1) % 3, a2 = (a + 2) % 3;
        for (int k = 0; k < 4; ++k) {
            const int u = k & 1, w = k >> 1;
            if (!f.front[a1][u] && !f.front[a2][w]) continue;
            if (clip_segment(f.corner[corner_index(a, 0, a1, u, a2, w)], f.corner[corner_index(a, 1, a1, u, a2, w)],
                             p0, sz, &sa, &sb))
                dl->AddLine(sa, sb, ink.edge, 1.0f);
        }
    }
    auto size_of = [&](const char* t) {
        return ink.font ? ink.font->CalcTextSizeA(ink.font_px, FLT_MAX, 0.0f, t) : ImGui::CalcTextSize(t);
    };
    for (int a = 0; a < 3; ++a) {
        const int k = f.label_edge[a];
        if (k < 0) continue;
        const int a1 = (a + 1) % 3, a2 = (a + 2) % 3, u = k & 1, w = k >> 1;
        const ImVec2 ea = screen_of(f.corner[corner_index(a, 0, a1, u, a2, w)], p0, sz);
        const ImVec2 eb = screen_of(f.corner[corner_index(a, 1, a1, u, a2, w)], p0, sz);
        const ImVec2 mid((ea.x + eb.x) * 0.5f, (ea.y + eb.y) * 0.5f);
        float dx = mid.x - f.centre.x, dy = mid.y - f.centre.y;
        const float dlen = std::hypot(dx, dy);
        if (dlen > 1e-3f) { dx /= dlen; dy /= dlen; } else { dx = 0.0f; dy = 1.0f; }
        const float al_x = 0.5f - 0.5f * dx, al_y = 0.5f - 0.5f * dy;
        float reach = 0.0f;
        ImVec2 last_min(1e30f, 1e30f), last_max(-1e30f, -1e30f);
        std::string drawn;
        for (double t : f.ticks[a]) {
            const Clip4 c = clip_of(f.m, box_point(f.b, a, t, a1, u, a2, w));
            if (!(c.w > 1e-9f) || c.z < 0.0f) continue;
            const ImVec2 s = screen_of(c, p0, sz);
            dl->AddLine(s, ImVec2(s.x + dx * 4.0f * ink.k, s.y + dy * 4.0f * ink.k), ink.tick, 1.0f);
            const std::string text = tick_text(t, f.step[a]);
            const ImVec2 ts = size_of(text.c_str());
            const ImVec2 at(s.x + dx * 7.0f * ink.k - ts.x * al_x, s.y + dy * 7.0f * ink.k - ts.y * al_y);
            const ImVec2 bmax(at.x + ts.x, at.y + ts.y);
            if (at.x < last_max.x + 3.0f && bmax.x > last_min.x - 3.0f && at.y < last_max.y && bmax.y > last_min.y)
                continue;
            dl->AddText(ink.font, ink.font_px, at, ink.text, text.c_str());
            drawn += (drawn.empty() ? "" : "|") + text;
            last_min = at;
            last_max = bmax;
            reach = std::max(reach, std::fabs(dx) * ts.x + std::fabs(dy) * ts.y);
        }
        if (!drawn.empty()) *labels += std::string(labels->empty() ? "" : ";") + "xyz"[a] + "=" + drawn;
        const char name[2] = {"xyz"[a], 0};
        const ImVec2 ns = size_of(name);
        const float off = 7.0f * ink.k + reach + 8.0f * ink.k;
        dl->AddText(ink.font, ink.font_px, ImVec2(mid.x + dx * off - ns.x * al_x, mid.y + dy * off - ns.y * al_y),
                    ink.text, name);
    }
}

// The data's box: the producer's bbox, else the payload's own extent (cached
// per update). False before the first data.
bool points_box(View& v, float out[6]) {
    if (v.has_bbox) {
        std::memcpy(out, v.bbox, 6 * sizeof(float));
        return true;
    }
    if (!v.have_cur || !v.cur.pos_cpu) return false;
    if (v.box_for != v.updates) {
        v.box_for = v.updates;
        float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        const float* P = (const float*)v.cur.pos_cpu;
        const uint64_t n = v.cur.n, stride = n > (4u << 20) ? (n + (4u << 20) - 1) / (4u << 20) : 1;
        for (uint64_t i = 0; i < n; i += stride) {
            const float* q = P + 3 * i;
            if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2])) continue;
            for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], q[k]); hi[k] = std::max(hi[k], q[k]); }
        }
        v.have_box = lo[0] <= hi[0];
        for (int k = 0; k < 3; ++k) { v.box[2 * k] = lo[k]; v.box[2 * k + 1] = hi[k]; }
    }
    if (v.have_box) std::memcpy(out, v.box, 6 * sizeof(float));
    return v.have_box;
}

void draw_points(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    HostRenderer* r = core->renderer.get();
    const bool open = begin_window(core, v, ImVec2(320.0f, 260.0f));
    if (!open) { ImGui::End(); return; }
    draw_header(core, S, v, kToolPause | (v.dims == 3 ? kToolRotate : 0u));
    under_header(core, S, v);
    {
        char tel[160], line[220];
        telemetry(tel, sizeof tel, S, v);
        std::snprintf(line, sizeof line, "points N=%lld %d-D  %s", (long long)v.count, v.dims, tel);
        debug_line(core, line);
    }
    PointsScene* sc = points_scene(core, v, nullptr);
    if (!sc) {
        ImGui::TextWrapped("%s", v.points ? v.points->notice.c_str() : "points cannot be drawn here");
        ImGui::End();
        return;
    }
    // The GPU time of the view's completed draws, for the frame statistics.
    const caliper_gfx::Stats st = sc->poll();
    if (st.gpu_ms > 0.0 && r) r->note_draw_gpu_ms(st.gpu_ms, st.gpu_submits);

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 4.0f || avail.y < 4.0f) { ImGui::End(); return; }
    int want_w = std::clamp((int)(avail.x * core->scale), 1, kMaxTarget);
    int want_h = std::clamp((int)(avail.y * core->scale), 1, kMaxTarget);
    if (sc->has_target() && std::abs(want_w - (int)sc->width()) <= kHysteresisPx &&
        std::abs(want_h - (int)sc->height()) <= kHysteresisPx) {
        want_w = (int)sc->width();
        want_h = (int)sc->height();
    } else {
        ++v.creates;   // a new render target this frame
        v.dirty = true;
        v.hover_dirty = true;
    }
    v.tex_w = want_w;
    v.tex_h = want_h;
    const float aspect = (float)v.tex_w / (float)v.tex_h;
    // The plot clears to the window's ground, as the browser's points view draws on its plate, so
    // the dimmed rest of a selection stays visible.
    const uint32_t clear = ImGui::GetColorU32(ImGuiCol_WindowBg) | 0xff000000u;
    if (clear != v.ground) {
        v.ground = clear;
        v.content_dirty = true;
        v.dirty = true;
    }
    // The scene's batches: the view's points, and the selected points over the dimmed rest.
    const bool ev = true;
    auto set_content = [&]() {
        if (!v.have_cur) return;
        const bool sel = ev && v.sel_count && stage_selection(core, v) && v.have_sel_draw;
        const PointsArrays all = points_arrays(v.cur);
        const PointsArrays chosen = sel ? points_arrays(v.sel_draw) : PointsArrays{};
        std::string why;
        if (!sc->set(all, sel ? &chosen : nullptr, points_look(v, clear), &why))
            core->log(1, "live: points '" + v.name + "': " + why);
        v.content_sel = sel ? v.sel_version * 2 + 1 : 0;
        v.content_updates = v.updates;
        v.content_dirty = false;
        v.dirty = true;
    };
    if (v.content_dirty) set_content();
    if (v.fit_pending && v.have_cur) fit_points(v, aspect);

    // ---- input (an invisible button owns the area so drags never move the window)
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##caliper.points", avail,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                               ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    ImGuiIO& io = ImGui::GetIO();
    // The browser element's gestures, in the app as in the browser; what
    // people do leaves only through an event sink (emit sends nothing without).
    Gesture& g = v.gesture;
    const bool released = ev && track_gesture(v, io.MousePos, v.have_cur && v.cur.pos_cpu, true);
    if (ev && g.down && g.selecting) {
        // The lasso follows the pointer; a rectangle keeps its two corners.
        if (g.rect) g.lasso = {g.x0, g.y0, io.MousePos.x, io.MousePos.y};
        else if (g.lasso.empty()) g.lasso = {g.x0, g.y0};
        else if (g.moved && (io.MousePos.x != g.lasso[g.lasso.size() - 2] ||
                             io.MousePos.y != g.lasso.back()))
            g.lasso.insert(g.lasso.end(), {io.MousePos.x, io.MousePos.y});
    }
    // The camera follows the pointer from the first point of motion after
    // the press, never while selecting, from where it last moved the camera,
    // so a fast drag whose movement and release land in one frame still moves
    // it. A press that ends as a click puts back the camera of the press
    // (camera_drag.h).
    const bool valid = ImGui::IsMousePosValid();
    if (g.pressed) {
        g.press_camera = g.step_camera = sc->camera();
        g.restorable = true;
    }
    const bool drag = (g.down || released) && !g.selecting;
    float dx = 0.0f, dy = 0.0f;
    if (drag && valid) {
        // A fit, a zoom or a turn since the last step: the camera of the press stays away.
        if (!same_camera(sc->camera(), g.step_camera)) g.restorable = false;
        dx = io.MousePos.x - g.lx;
        dy = io.MousePos.y - g.ly;
        g.lx = io.MousePos.x;
        g.ly = io.MousePos.y;
    }
    // Any drag stops the slow turn, which the page hears for its header toggle.
    if (v.auto_rotate && (g.down || released) && g.moved) {
        v.auto_rotate = false;
        emit(S, v, "toggle", Detail().text("name", "rotate").flag("on", false).str());
    }
    // The slow turn waits while a button is held, so that a click's camera comes back as it was.
    if (v.auto_rotate && v.dims == 3 && v.have_cur && !g.down && !released) {
        sc->turn(double(std::min(io.DeltaTime, 0.1f) * kRotateRate));
        v.dirty = true;
        v.hover_dirty = true;
    }
    if (dx != 0.0f || dy != 0.0f) {
        const bool pan = g.button != 0 || g.alt || g.ctrl;
        if (v.dims == 3 && !pan) sc->orbit(dx, dy);
        else sc->pan(dx, dy, avail.y);
        v.dirty = true;
        v.hover_dirty = true;
    }
    if (drag) g.step_camera = sc->camera();
    if (released && !g.selecting && !g.moved && g.restorable) {   // a click: the camera of the press
        sc->set_camera(g.press_camera);
        v.dirty = true;
        v.hover_dirty = true;
    }
    if (hovered && io.MouseWheel != 0.0f) {
        const float mx = (io.MousePos.x - p0.x) / avail.x * 2.0f - 1.0f;
        const float my = 1.0f - (io.MousePos.y - p0.y) / avail.y * 2.0f;
        sc->zoom(io.MouseWheel, mx, my, aspect);
        v.dirty = true;
        v.hover_dirty = true;
    }
    // F, or a double click on the points, fits the camera to them again; the second press of a
    // double click keeps the fit at its release.
    if ((((hovered || ImGui::IsWindowFocused()) && ImGui::IsKeyPressed(ImGuiKey_F, false)) ||
         (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) && v.have_cur) {
        fit_points(v, aspect);
        if (g.down) g.press_camera = g.step_camera = sc->camera();
    }

    float mvp[16];
    if (!sc->mvp((uint32_t)v.tex_w, (uint32_t)v.tex_h, mvp)) std::fill(mvp, mvp + 16, 0.0f);
    const float half_px = 0.5f * point_size_px(v.have_cur ? v.cur.n : 0, core->scale) / core->scale + 1.0f;
    // The depth cues of 3-D points (PointsScene::cue_values): sizes relative to the size at the
    // camera's target, which hover and picks follow, and the points behind the target dimmed
    // towards the far side of the data's bounding sphere.
    sc->set_depth_cues(v.depth_cues);
    float att_ref = 0.0f, fade_w0 = 0.0f, fade_w1 = 0.0f;
    sc->cue_values(&att_ref, &fade_w0, &fade_w1);

    // ---- a finished gesture: a selection, a cleared selection or a pick
    if (released && v.have_cur && v.cur.pos_cpu) {
        if (g.selecting && g.moved && g.lasso.size() >= 4) {
            std::vector<float> poly = g.lasso;
            if (g.rect) {
                const float ax = poly[0], ay = poly[1], bx = poly[2], by = poly[3];
                poly = {ax, ay, bx, ay, bx, by, ax, by};
            }
            select_polygon(S, v, mvp, p0, avail, poly);
        } else {
            if (g.selecting) {
                // A shift-click clears the selection, which an empty detail reports.
                if (v.sel_count) emit(S, v, "select", "{}");
                if (!v.selected.empty()) ++v.sel_version;
                v.selected.clear();
                v.sel_count = 0;
                v.dirty = true;
            }
            if (!g.moved) {   // the point under the press, with the camera of the press
                uint64_t stride = 1;
                const Pick pk = pick_point(v, mvp, p0, avail, ImVec2(g.x0, g.y0), half_px, att_ref, 8.0f, &stride);
                if (pk.idx >= 0) emit(S, v, "pick", point_detail(v, pk.idx));
            }
        }
        g.lasso.clear();
    }
    if (released) g.selecting = false;

    // ---- a linked selection: the points whose source row is a selected item,
    // recomputed for every new link and every new sample.
    if (ev && v.have_cur && v.link_axis == 1 &&
        (v.link_for_version != v.link_version || v.link_for_updates != v.updates)) {
        v.link_for_version = v.link_version;
        v.link_for_updates = v.updates;
        {
            const caliper_live::PayloadAxis* ax = v.map.axes.empty() ? nullptr : &v.map.axes.back();
            std::vector<uint8_t> sel(v.cur.n, 0);
            uint64_t count = 0;
            for (uint64_t i = 0; i < v.cur.n; ++i)
                if (std::binary_search(v.link_items.begin(), v.link_items.end(), source_index(ax, (int64_t)i))) {
                    sel[i] = 1;
                    ++count;
                }
            v.selected = count ? std::move(sel) : std::vector<uint8_t>();
            v.sel_count = count;
            ++v.sel_version;
            v.dirty = true;
        }
    }
    // A selection that changed changes the scene's batches.
    const uint64_t sel_key = (ev && v.sel_count) ? v.sel_version * 2 + 1 : 0;
    if (v.have_cur && (sel_key != v.content_sel || v.content_updates != v.updates)) set_content();

    // ---- GPU: compile and submit when something changed
    // In 3-D the target's ground is transparent, so the box's back layer, drawn before the image,
    // shows behind the points; the dimmed rest of a selection is dimmed toward the window's ground.
    const uint32_t ground = v.dims == 3 ? (clear & 0x00ffffffu) : clear;
    if (v.dirty) {
        const auto t0 = std::chrono::steady_clock::now();
        caliper_gfx::Serial serial = 0;
        std::string why;
        const bool ok = sc->render((uint32_t)v.tex_w, (uint32_t)v.tex_h, core->scale, ground, &serial, &why);
        v.draw_cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++v.redraws;
        if (ok && serial) {
            if (v.have_cur) v.cur.last_draw = serial;
            if (v.have_sel_draw && v.content_sel) v.sel_draw.last_draw = serial;
        }
        if (!ok && !v.draw_failed) core->log(1, "live: points '" + v.name + "': " + why);
        v.dirty = false;   // a refused draw is not retried every frame (it logs)
        v.draw_failed = !ok;
    }
    if (v.draw_failed) ImGui::TextUnformatted("points draw refused by the graphics layer (see log)");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    // 3-D: the box's back layer under the points, its front layer over them.
    float bx[6];
    const bool boxed = v.dims == 3 && points_box(v, bx);
    BoxFrame frame;
    BoxInk ink{};
    const caliper_scene::Orbit* orbit = std::get_if<caliper_scene::Orbit>(&sc->camera());
    if (boxed && orbit) {
        const caliper_scene::Vec3 e = caliper_scene::orbit_eye(*orbit);
        frame = box_frame(bx, V3{float(e.x), float(e.y), float(e.z)}, mvp, p0, avail);
        const bool house = !core->views_bare;
        const Look& L = S->look;
        // The browser element's inks on a page (web/src/views/box3d.ts), the host's in a window.
        const ImU32 rule = house ? L.rule : IM_COL32(0x5a, 0x57, 0x51, 255);
        ink.edge = rule;
        ink.tick = rule;
        ink.back = (rule & 0x00ffffffu) | (150u << 24);
        ink.grid = (rule & 0x00ffffffu) | (70u << 24);
        ink.text = house ? L.faint : IM_COL32(0xa4, 0x9d, 0x8f, 255);
        const float afs = core->views_bare ? 10.5f
                          : L.mono_size > 0.0f ? L.mono_size - 1.0f
                                               : std::max(10.5f, ImGui::GetFontSize() * 0.85f);
        ink.k = afs / 10.5f;
        ink.font = house ? L.mono : mono_face();
        ink.font_px = house ? afs : afs * kMonoEm;
        if (!ink.font) {
            ink.font = ImGui::GetFont();
            ink.font_px = ImGui::GetFontSize();
        }
        dl->PushClipRect(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), true);
        draw_box_back(dl, frame, p0, avail, ink);
        dl->PopClipRect();
    }
    if (const uint64_t tex = sc->imgui_texture())
        dl->AddImage((ImTextureID)tex, p0, ImVec2(p0.x + avail.x, p0.y + avail.y));
    if (boxed && orbit) {
        dl->PushClipRect(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), true);
        draw_box_front(dl, frame, p0, avail, ink, &v.box_labels);
        dl->PopClipRect();
    }

    // Where the data is: the area, and for 2-D the window of x and y it shows.
    v.geo_valid = true;
    v.geo_area[0] = p0.x; v.geo_area[1] = p0.y; v.geo_area[2] = avail.x; v.geo_area[3] = avail.y;
    v.geo_ydown = false;
    if (const caliper_scene::PanZoom* pz = std::get_if<caliper_scene::PanZoom>(&sc->camera())) {
        v.geo_window[0] = pz->center[0] - pz->half_height * aspect;
        v.geo_window[1] = pz->center[0] + pz->half_height * aspect;
        v.geo_window[2] = pz->center[1] - pz->half_height;
        v.geo_window[3] = pz->center[1] + pz->half_height;
    } else {
        std::fill(v.geo_window, v.geo_window + 4, 0.0);
    }
    // ---- the selection being drawn, and how many points are selected
    if (ev && g.down && g.selecting && g.lasso.size() >= 4) {
        dl->PushClipRect(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), true);
        if (g.rect) {
            const ImVec2 a(std::min(g.lasso[0], g.lasso[2]), std::min(g.lasso[1], g.lasso[3]));
            const ImVec2 b(std::max(g.lasso[0], g.lasso[2]), std::max(g.lasso[1], g.lasso[3]));
            dl->AddRectFilled(a, b, kSelFill);
            dl->AddRect(a, b, kSelLine, 0.0f, 0, 1.0f);
        } else {
            std::vector<ImVec2> pts;
            for (size_t k = 0; k + 1 < g.lasso.size(); k += 2) pts.emplace_back(g.lasso[k], g.lasso[k + 1]);
            dl->AddPolyline(pts.data(), (int)pts.size(), kSelLine, ImDrawFlags_Closed, 1.0f);
        }
        dl->PopClipRect();
    }
    if (ev && v.sel_count) {
        char t[48];
        std::snprintf(t, sizeof t, "%llu selected", (unsigned long long)v.sel_count);
        const ImVec2 ts = ImGui::CalcTextSize(t);
        dl->AddText(ImVec2(p0.x + avail.x - ts.x - 10.0f, p0.y + 8.0f), kSelText, t);
    }

    // ---- hover
    if (!(hovered && !active && v.have_cur && v.cur.pos_cpu)) v.hover_item = -1;
    if (hovered && !active && v.have_cur) {
        if (!v.cur.pos_cpu) {
            ImGui::SetTooltip("hover unavailable: the payload buffer is not CPU-visible");
        } else {
            const ImVec2 m = io.MousePos;
            if (v.hover_dirty || m.x != v.hover_mx || m.y != v.hover_my) {
                const Pick pk = pick_point(v, mvp, p0, avail, m, half_px, att_ref, 8.0f, &v.hover_stride);
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
            v.hover_item = v.hover_idx;
            if (ev) {
                const int64_t i = v.hover_idx;
                emit_hover(S, v, i, [&] { return point_detail(v, i); });
            }
        }
    } else if (ev && !active) {
        v.hover_sent = -1;   // left the view: the same point hovered again is reported again
    }
    ImGui::End();
}

// ---- scene -----------------------------------------------------------------
// The scene view's offscreen target shown as an image, with the points view's
// camera gestures: a drag orbits (pans in a 2-D scene), a right, middle, alt or
// ctrl drag pans, the wheel zooms, and F or a double click fits. The view
// compiles and submits only when the scene, the camera or the size changed.
// In 3-D the header's segments, or the key U, switch the orbit between Y up
// and Z up (design/scenes.md, "The up-axis switch"); the header's display menu
// holds the scale bar's option and the reset. The scale bar is drawn here, over
// the picture, as overlay lines and text (scene/scale_bar.h).

// The scale bar in the bottom-right corner of the picture at p0 of `size`
// logical pixels: a bar kScaleBarThickness thick with end ticks kScaleBarTick
// tall, kScaleBarMargin in from the edges, its length above it in the small
// text, over a translucent band of the window's background. Returns the band.
ImRect draw_scale_bar(ImDrawList* dl, const Look& L, ImVec2 p0, ImVec2 size, const caliper_scene::ScaleBar& b) {
    const float th = (float)caliper_scene::kScaleBarThickness, tick = (float)caliper_scene::kScaleBarTick;
    const float x1 = std::floor(p0.x + size.x - (float)caliper_scene::kScaleBarMargin);
    const float y1 = std::floor(p0.y + size.y - (float)caliper_scene::kScaleBarMargin);
    const float x0 = std::round(x1 - (float)b.pixels);
    if (L.small_size > 0.0f) ImGui::PushFont(nullptr, L.small_size);
    const ImVec2 ts = ImGui::CalcTextSize(b.label.c_str());
    // Centred over the bar, and kept from passing the bar's right end.
    const float tx = std::floor(std::min(x1 - ts.x, (x0 + x1 - ts.x) * 0.5f));
    const float ty = std::floor(y1 - tick - 2.0f - ts.y);
    const ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
    const float pad = 4.0f;
    const ImRect band(ImVec2(std::min(x0, tx) - pad, ty - 2.0f), ImVec2(std::max(x1, tx + ts.x) + pad, y1 + pad));
    dl->AddRectFilled(band.Min, band.Max, ImGui::GetColorU32(ImVec4(bg.x, bg.y, bg.z, 0.6f)), 3.0f);
    dl->AddRectFilled(ImVec2(x0, y1 - th), ImVec2(x1, y1), L.text);
    dl->AddRectFilled(ImVec2(x0, y1 - tick), ImVec2(x0 + th, y1), L.text);
    dl->AddRectFilled(ImVec2(x1 - th, y1 - tick), ImVec2(x1, y1), L.text);
    dl->AddText(ImVec2(tx, ty), L.text, b.label.c_str());
    if (L.small_size > 0.0f) ImGui::PopFont();
    return band;
}
}  // namespace

void draw_scene(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    const bool open = begin_window(core, v, ImVec2(360.0f, 300.0f));
    if (!open) { ImGui::End(); return; }
    SceneView* sc = scene_view(v);
    const bool three_d = sc && sc->tree().space() == caliper_scene::Space::k3D;
    draw_header(core, S, v, kToolPause | (three_d ? kToolUp : 0u) | (sc ? kToolDisplay : 0u));
    if (sc && ImGui::BeginPopup(kSceneDisplayMenu)) {
        bool bar = sc->scale_bar_on();
        if (ImGui::MenuItem("Scale bar", nullptr, &bar)) sc->set_scale_bar(bar);
        if (ImGui::MenuItem("Reset view")) sc->reset();
        ImGui::EndPopup();
    }
    under_header(core, S, v);
    if (sc) {
        sc->poll();
        char tel[160], line[320];
        telemetry(tel, sizeof tel, S, v);
        const SceneCost& k = sc->cost();
        std::snprintf(line, sizeof line, "scene %s  compile %.3f ms  submit %.3f ms  %s", tel, k.compile_ms,
                      k.submit_ms, sc->camera_text().c_str());
        debug_line(core, line);
    }
    if (!sc) {
        ImGui::TextWrapped("%s", v.scene && !v.scene->notice.empty() ? v.scene->notice.c_str()
                                                                     : "this scene cannot be drawn here");
        ImGui::End();
        return;
    }
    if (!sc->notes().empty() && !core->views_bare) {
        const Look& L = S->look;
        const std::string t = fit_text(sc->notes().back(), ImGui::GetContentRegionAvail().x);
        ImGui::PushStyleColor(ImGuiCol_Text, L.faint);
        ImGui::TextUnformatted(t.c_str());
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && sc->notes().size() > 1) {
            std::string all;
            for (const std::string& n : sc->notes()) all += n + "\n";
            ImGui::SetTooltip("%s", all.c_str());
        }
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x < 4.0f || avail.y < 4.0f) { ImGui::End(); return; }
    int want_w = std::clamp((int)(avail.x * core->scale), 1, kMaxTarget);
    int want_h = std::clamp((int)(avail.y * core->scale), 1, kMaxTarget);
    if (sc->has_target() && std::abs(want_w - (int)sc->width()) <= kHysteresisPx &&
        std::abs(want_h - (int)sc->height()) <= kHysteresisPx) {
        want_w = (int)sc->width();
        want_h = (int)sc->height();
    }

    // ---- input (an invisible button owns the area so drags never move the window)
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##caliper.scene", avail,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                               ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    ImGuiIO& io = ImGui::GetIO();
    Gesture& g = v.gesture;
    // The points view's gestures: a shift-drag draws a lasso, with alt a rectangle; a shift-click
    // clears the selection; a click picks; a drag without shift orbits or pans.
    const bool released = track_gesture(v, io.MousePos, true, true);
    if (g.down && g.selecting) {
        if (g.rect) g.lasso = {g.x0, g.y0, io.MousePos.x, io.MousePos.y};
        else if (g.lasso.empty()) g.lasso = {g.x0, g.y0};
        else if (g.moved && (io.MousePos.x != g.lasso[g.lasso.size() - 2] || io.MousePos.y != g.lasso.back()))
            g.lasso.insert(g.lasso.end(), {io.MousePos.x, io.MousePos.y});
    }
    // The camera follows the pointer from the first point of motion after the press, as the points
    // view's does, and a press that ends as a click puts back the camera of the press before the
    // frame draws (camera_drag.h).
    const bool valid = ImGui::IsMousePosValid();
    if (g.pressed) {
        g.press_camera = g.step_camera = sc->camera();
        g.press_moved = sc->moved();
        g.restorable = true;
    }
    const bool drag = (g.down || released) && !g.selecting;
    if (drag && valid) {
        // A fit, a zoom or an up switch since the last step: the camera of the press stays away.
        if (!same_camera(sc->camera(), g.step_camera)) g.restorable = false;
        const float dx = io.MousePos.x - g.lx, dy = io.MousePos.y - g.ly;
        g.lx = io.MousePos.x;
        g.ly = io.MousePos.y;
        if (dx != 0.0f || dy != 0.0f) {
            const bool pan = g.button != 0 || g.alt || g.ctrl;
            if (pan) sc->pan(dx, dy, avail.y);
            else sc->orbit(dx, dy);
        }
        g.step_camera = sc->camera();
    }
    if (released && !g.selecting && !g.moved && g.restorable) sc->restore_camera(g.press_camera, g.press_moved);
    if (hovered && io.MouseWheel != 0.0f) {
        const float mx = (io.MousePos.x - p0.x) / avail.x * 2.0f - 1.0f;
        const float my = 1.0f - (io.MousePos.y - p0.y) / avail.y * 2.0f;
        sc->zoom(io.MouseWheel, mx, my, avail.x, avail.y);
    }
    if (((hovered || ImGui::IsWindowFocused()) && ImGui::IsKeyPressed(ImGuiKey_F, false)) ||
        (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) {
        sc->fit();
        if (g.down) {   // the second press of a double click: its release keeps the fit
            g.press_camera = g.step_camera = sc->camera();
            g.press_moved = sc->moved();
        }
    }
    // U steps through the header's up-axis segments.
    if ((hovered || ImGui::IsWindowFocused()) && ImGui::IsKeyPressed(ImGuiKey_U, false)) {
        const std::vector<caliper_scene::Axis> c = sc->up_choices();
        if (!c.empty()) {
            auto at = std::find(c.begin(), c.end(), sc->up_mode());
            sc->switch_up(at == c.end() || at + 1 == c.end() ? c.front() : *(at + 1));
        }
    }

    // ---- GPU: compile and submit when something changed (S->paused freezes the view: the
    // source holds its messages, so nothing changes but the camera)
    const uint32_t clear = ImGui::GetColorU32(ImGuiCol_WindowBg) | 0xff000000u;
    std::string why;
    const uint64_t uploaded = sc->counts().bytes_total;
    sc->set_entity_filter("", nullptr);   // the app's window draws every entity; a panel filters its own draws
    if (!sc->render((uint32_t)want_w, (uint32_t)want_h, core->scale, clear, &why)) sc->note(why);
    if (const uint64_t copied = sc->counts().bytes_total - uploaded)   // the class table's texture
        caliper_stats::note_copy(core, v.name, copied);
    v.tex_w = (int)sc->width();
    v.tex_h = (int)sc->height();
    v.creates = sc->target_creates();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (const uint64_t id = sc->imgui_texture())
        dl->AddImage((ImTextureID)id, p0, ImVec2(p0.x + avail.x, p0.y + avail.y));
    std::string bar_text;   // where the bar is, for the UI driver's `where` and the module's geometry
    std::fill(v.geo_bar, v.geo_bar + 4, 0.0f);
    std::vector<caliper_host::LabelRect> keep_clear;   // the view's own texts, which no label covers
    if (const caliper_scene::ScaleBar bar = sc->scale_bar(avail.x, avail.y); bar.visible) {
        const ImRect band = draw_scale_bar(dl, S->look, p0, avail, bar);
        keep_clear.push_back({band.Min.x, band.Min.y, band.Max.x, band.Max.y});
        char t[96];
        std::snprintf(t, sizeof t, " scale_bar %.2f %.2f %.2f %.2f", band.Min.x, band.Min.y, band.Max.x, band.Max.y);
        bar_text = t;
        v.geo_bar[0] = band.Min.x; v.geo_bar[1] = band.Min.y; v.geo_bar[2] = band.Max.x; v.geo_bar[3] = band.Max.y;
    }

    // ---- picks, hovers and selections (design/scenes.md, "Picking, hover and selection"). Picks run
    // the scene core's pick pass, asked only when the pointer moved or clicked; selections project
    // the CPU copies of the points and boxes. What people do leaves through the event sink.
    const float kx = sc->width() ? float(sc->width()) / avail.x : 1.0f, ky = sc->height() ? float(sc->height()) / avail.y : 1.0f;
    auto target_px = [&](float x, float y) { return ImVec2((x - p0.x) * kx, (y - p0.y) * ky); };
    if (released) {
        if (g.selecting && g.moved && g.lasso.size() >= 4) {
            std::vector<float> poly = g.lasso;
            if (g.rect) poly = {poly[0], poly[1], poly[2], poly[1], poly[2], poly[3], poly[0], poly[3]};
            for (size_t k = 0; k + 1 < poly.size(); k += 2) {
                const ImVec2 t = target_px(poly[k], poly[k + 1]);
                poly[k] = t.x;
                poly[k + 1] = t.y;
            }
            emit(S, v, "select", sc->select_polygon(poly));
        } else if (!g.moved && g.button == 0) {
            if (g.selecting && sc->clear_selection()) emit(S, v, "select", "{}");
            if (!g.selecting) {   // the item under the press, with the camera of the press
                const ImVec2 t = target_px(g.x0, g.y0);
                std::string why;
                v.scene_click = sc->request_pick(t.x, t.y, &why);
            }
        }
        g.lasso.clear();
        g.selecting = false;
    }
    // A hover asks for a pick when the pointer moved over the picture, at most 30 times a second,
    // unless a click's pick is in flight; a still pointer asks nothing.
    const double t_ms = now_ms();
    if (hovered && valid && !g.down && !v.scene_click && (io.MousePos.x != v.scene_hx || io.MousePos.y != v.scene_hy) &&
        t_ms - v.scene_hover_ms >= kHoverMs) {
        v.scene_hx = io.MousePos.x;
        v.scene_hy = io.MousePos.y;
        v.scene_hover_ms = t_ms;
        const ImVec2 t = target_px(io.MousePos.x, io.MousePos.y);
        std::string why;
        sc->request_pick(t.x, t.y, &why);
    }
    if (!hovered && !v.scene_hover.empty()) v.scene_hover.clear();
    caliper_scene::Hit hit;
    const caliper_scene::PickPass::Result pr = sc->poll_pick(&hit);
    if (pr == caliper_scene::PickPass::Result::Hit || pr == caliper_scene::PickPass::Result::Miss ||
        pr == caliper_scene::PickPass::Result::Dropped) {
        const bool click = v.scene_click;
        v.scene_click = false;
        const std::string detail = pr == caliper_scene::PickPass::Result::Hit ? sc->pick_detail(hit) : std::string();
        if (click) {
            if (!detail.empty()) emit(S, v, "pick", detail);
        } else if (pr != caliper_scene::PickPass::Result::Dropped) {
            const int64_t key = detail.empty() ? -1
                                               : (int64_t(hit.entity.index) << 40) ^ (int64_t(hit.primitive) << 36) ^
                                                     (int64_t(hit.part) << 33) ^ int64_t(hit.item);
            v.hover_item = detail.empty() ? -1 : int64_t(hit.item);
            v.scene_hover = detail.empty() ? std::string()
                                           : sc->tree().path(hit.entity) + "  " + caliper_scene::primitive_name(hit.primitive) +
                                                 " " + std::to_string(hit.item);
            emit_hover(S, v, key, [&] { return detail; });
        }
    }
    if (hovered && !v.scene_hover.empty() && !g.down) ImGui::SetTooltip("%s", v.scene_hover.c_str());
    // The lasso or rectangle being drawn, and how many items the highlight shows.
    if (g.down && g.selecting && g.lasso.size() >= 4) {
        dl->PushClipRect(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), true);
        if (g.rect) {
            const ImVec2 a(std::min(g.lasso[0], g.lasso[2]), std::min(g.lasso[1], g.lasso[3]));
            const ImVec2 b(std::max(g.lasso[0], g.lasso[2]), std::max(g.lasso[1], g.lasso[3]));
            dl->AddRectFilled(a, b, kSelFill);
            dl->AddRect(a, b, kSelLine, 0.0f, 0, 1.0f);
        } else {
            std::vector<ImVec2> pts;
            for (size_t k = 0; k + 1 < g.lasso.size(); k += 2) pts.emplace_back(g.lasso[k], g.lasso[k + 1]);
            dl->AddPolyline(pts.data(), (int)pts.size(), kSelLine, ImDrawFlags_Closed, 1.0f);
        }
        dl->PopClipRect();
    }
    if (sc->highlighted()) {
        char t[48];
        std::snprintf(t, sizeof t, "%llu selected", (unsigned long long)sc->highlighted());
        const ImVec2 ts = ImGui::CalcTextSize(t);
        const ImVec2 at(p0.x + avail.x - ts.x - 10.0f, p0.y + 8.0f);
        dl->AddText(at, kSelText, t);
        keep_clear.push_back({at.x - 2.0f, at.y - 2.0f, at.x + ts.x + 2.0f, at.y + ts.y + 2.0f});
    }

    // Labels: text over the picture at anchors the scene core projects, nearest first, at most
    // 512, placed without overlap (scene_overlays.h) and off the scale bar and the selection
    // count; a label without room shows as a dot.
    if (sc->width() && sc->height()) {
        const std::vector<caliper_scene::LabelAnchor> labels = sc->labels();
        caliper_live_views::SceneOverlayInput in;
        in.draw = dl;
        in.origin = p0;
        in.scale_x = avail.x / float(sc->width());
        in.scale_y = avail.y / float(sc->height());
        in.labels = &labels;
        in.ink = ImGui::GetColorU32(ImGuiCol_Text);
        in.shade = IM_COL32(0, 0, 0, 160);
        in.keep_clear = &keep_clear;
        dl->PushClipRect(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), true);
        caliper_live_views::draw_scene_overlays(in);
        dl->PopClipRect();
    }

    // A right click that does not drag offers the scene's cameras to look through.
    if (released && !g.moved && g.button == ImGuiMouseButton_Right && hovered) ImGui::OpenPopup("##caliper.scene.cameras");
    if (ImGui::BeginPopup("##caliper.scene.cameras")) {
        if (ImGui::MenuItem("Orbit", nullptr, sc->looking_through().empty())) sc->look_through("", nullptr);
        const std::vector<std::string> cams = sc->cameras();
        if (cams.empty()) ImGui::TextDisabled("No camera to look through");
        for (const std::string& c : cams)
            if (ImGui::MenuItem(("Look through " + c).c_str(), nullptr, sc->looking_through() == c))
                sc->look_through(c, nullptr);
        ImGui::EndPopup();
    }

    v.geo_valid = true;
    v.geo_area[0] = p0.x; v.geo_area[1] = p0.y; v.geo_area[2] = avail.x; v.geo_area[3] = avail.y;
    v.geo_ydown = false;
    std::fill(v.geo_window, v.geo_window + 4, 0.0);
    v.geo_text = sc->camera_text() + bar_text;
    ImGui::End();
}

namespace {

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

// A plot's axes follow the data (fit every frame) until a pan or a wheel
// over the plot, as the browser element's series does; a double click
// follows again. The page hears every change, for its header toggle.
void set_follow(CaliperLiveViewsState* S, View& v, bool on) {
    if (v.follow == on) return;
    v.follow = on;
    emit(S, v, "toggle", Detail().text("name", "follow").flag("on", on).str());
}

// Before BeginPlot: stop following on a pan or a wheel over last frame's
// plot, so that the pan or zoom shows this frame; fit once if asked. The
// wheel zooms about the pointer by the browser element's factor (ImPlot's own
// wheel zoom, a fixed step per event, is off meanwhile).
void before_plot(CaliperLiveViewsState* S, View& v) {
    const Gesture& g = v.gesture;
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 m = io.MousePos;
    const float* a = v.geo_area;
    const bool over = v.geo_valid && m.x >= a[0] && m.y >= a[1] && m.x < a[0] + a[2] && m.y < a[1] + a[3];
    if (v.follow && ((g.down && !g.selecting && g.moved) || (over && io.MouseWheel != 0.0f)))
        set_follow(S, v, false);
    v.zoom_limits = false;
    if (v.pan_pending && a[2] > 0.0f && a[3] > 0.0f) {
        // The drag that stopped following moved the pointer while the axes
        // still fitted the data: pan by all of it now.
        const double* w = v.geo_window;
        const double dx = v.pan_dx / a[2] * (w[1] - w[0]);
        auto ty = [&](double y) { return v.log_y ? std::log10(std::max(y, 1e-300)) : y; };
        auto uy = [&](double t) { return v.log_y ? std::pow(10.0, t) : t; };
        const double y0 = ty(w[2]), y1 = ty(w[3]), dy = v.pan_dy / a[3] * (y1 - y0);
        v.zoom_lim[0] = w[0] - dx;
        v.zoom_lim[1] = w[1] - dx;
        v.zoom_lim[2] = uy(y0 + dy);
        v.zoom_lim[3] = uy(y1 + dy);
        v.zoom_limits = true;
    }
    v.pan_pending = false;
    if (over && io.MouseWheel != 0.0f && a[2] > 0.0f && a[3] > 0.0f) {
        const double f = std::exp(0.15 * (double)io.MouseWheel);
        const double* w = v.geo_window;
        const double cx = w[0] + (m.x - a[0]) / a[2] * (w[1] - w[0]);
        auto ty = [&](double y) { return v.log_y ? std::log10(std::max(y, 1e-300)) : y; };
        auto uy = [&](double t) { return v.log_y ? std::pow(10.0, t) : t; };
        const double y0 = ty(w[2]), y1 = ty(w[3]);
        const double cy = y1 - (m.y - a[1]) / a[3] * (y1 - y0);
        v.zoom_lim[0] = cx + (w[0] - cx) / f;
        v.zoom_lim[1] = cx + (w[1] - cx) / f;
        v.zoom_lim[2] = uy(cy + (y0 - cy) / f);
        v.zoom_lim[3] = uy(cy + (y1 - cy) / f);
        v.zoom_limits = true;
    }
    v.zoom_rate_saved = ImPlot::GetInputMap().ZoomRate;
    ImPlot::GetInputMap().ZoomRate = 0.0f;
    // A log y axis is fitted here, to the positive values only (ImPlot's fit
    // would reach down to a zero value); x is ImPlot's.
    v.log_fit_now = v.log_y && (v.follow || v.fit_once);
    if (v.fit_once) {
        if (v.log_y) ImPlot::SetNextAxisToFit(ImAxis_X1);
        else ImPlot::SetNextAxesToFit();
        v.fit_once = false;
    }
}

// The axis flags of a plot: fit while following, never fit y on a log scale
// (setup_zoom does), and locked while a selection is dragged.
void plot_flags(const CaliperLiveViewsState* S, const View& v, ImPlotAxisFlags* xf, ImPlotAxisFlags* yf) {
    (void)S;
    const ImPlotAxisFlags lock = v.gesture.down && v.gesture.selecting ? ImPlotAxisFlags_Lock : 0;
    const ImPlotAxisFlags fit = v.follow ? ImPlotAxisFlags_AutoFit : 0;
    *xf = fit | lock;
    *yf = (v.log_y ? 0 : fit) | lock;
}

// After SetupAxes: this frame's wheel zoom, or the fit of a log y axis to
// the positive values with 5% of their decades as padding (the browser views').
void setup_zoom(View& v) {
    if (v.zoom_limits) {
        ImPlot::SetupAxesLimits(v.zoom_lim[0], v.zoom_lim[1], v.zoom_lim[2], v.zoom_lim[3], ImPlotCond_Always);
    } else if (v.log_fit_now) {
        double lo = INFINITY, hi = -INFINITY;
        for (float x : v.values)
            if (std::isfinite(x) && x > 0.0f) { lo = std::min(lo, (double)x); hi = std::max(hi, (double)x); }
        if (!(hi >= lo)) { lo = 0.1; hi = 10.0; }
        const double l = std::log10(lo), h = std::log10(hi), pad = std::max(0.05, (h - l) * 0.05);
        ImPlot::SetupAxisLimits(ImAxis_Y1, std::pow(10.0, l - pad), std::pow(10.0, h + pad), ImPlotCond_Always);
    }
}

// Inside the plot, after its items: a double click follows again.
void after_plot(CaliperLiveViewsState* S, View& v) {
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) set_follow(S, v, true);
}

// After the plot, drawn or not: ImPlot's own wheel zoom back as it was.
void end_plot_input(CaliperLiveViewsState* S, View& v) {
    (void)S;
    ImPlot::GetInputMap().ZoomRate = v.zoom_rate_saved;
}

// The gesture of a plot (series, histogram) while the host listens for
// events: a left press inside the plot starts it, the drag follows the mouse
// in plot x, the release ends it. ImPlot keeps its own pan, zoom and fit; the
// axes are locked while a selection is dragged so they stay put.
bool plot_gesture(CaliperLiveViewsState* S, View& v, double mx) {
    Gesture& g = v.gesture;
    ImGuiIO& io = ImGui::GetIO();
    if (!g.down && ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        g = Gesture{};
        g.down = true;
        g.shift = io.KeyShift;
        g.selecting = g.shift;
        g.x0 = g.lx = io.MousePos.x;
        g.y0 = g.ly = io.MousePos.y;
        g.d0x = g.d1x = mx;
    }
    if (!g.down) return false;
    if (!g.moved && std::hypot(io.MousePos.x - g.x0, io.MousePos.y - g.y0) > kClickSlop) g.moved = true;
    g.d1x = mx;
    if (v.follow && !g.selecting && g.moved) {
        // A pan stops following. ImPlot pans from the next frame on; the
        // movement so far is applied then (pan_pending).
        set_follow(S, v, false);
        v.pan_pending = true;
        v.pan_dx = io.MousePos.x - g.x0;
        v.pan_dy = io.MousePos.y - g.y0;
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        g.down = false;
        return true;
    }
    return false;
}

// A band over the plot's whole height from x = a to x = b (plot units).
void plot_band(double a, double b, bool fill) {
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    ImVec2 p = ImPlot::PlotToPixels(std::min(a, b), lim.Y.Max), q = ImPlot::PlotToPixels(std::max(a, b), lim.Y.Min);
    if (q.x - p.x < 1.0f) q.x = p.x + 1.0f;
    ImPlot::PushPlotClipRect();
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    if (fill) dl->AddRectFilled(p, q, kSelFill);
    dl->AddRect(p, q, kSelLine, 0.0f, 0, 1.0f);
    ImPlot::PopPlotClipRect();
}

void plot_geometry(View& v) {
    const ImVec2 pp = ImPlot::GetPlotPos(), ps = ImPlot::GetPlotSize();
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    v.geo_valid = true;
    v.geo_area[0] = pp.x; v.geo_area[1] = pp.y; v.geo_area[2] = ps.x; v.geo_area[3] = ps.y;
    v.geo_window[0] = lim.X.Min; v.geo_window[1] = lim.X.Max;
    v.geo_window[2] = lim.Y.Min; v.geo_window[3] = lim.Y.Max;
    v.geo_ydown = false;
}

std::string step_detail(const View& v, int64_t step) {
    Detail d;
    d.integer("step", step);
    const int64_t i = step - (int64_t)v.first;
    if (i >= 0 && (size_t)i < v.values.size() && std::isfinite(v.values[(size_t)i])) d.num("value", v.values[(size_t)i]);
    return d.str();
}

void draw_series(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    const bool open = begin_window(core, v, ImVec2(360.0f, 200.0f));
    if (!open) { ImGui::End(); return; }
    const size_t n = v.values.size();
    draw_header(core, S, v, kToolPause | kToolLog | kToolFollow);
    under_header(core, S, v);
    {
        char tel[160], line[260];
        telemetry(tel, sizeof tel, S, v);
        if (n)
            std::snprintf(line, sizeof line, "series n=%zu  latest %.6g @ step %llu  %s", n, v.values.back(),
                          (unsigned long long)(v.first + n - 1), tel);
        else
            std::snprintf(line, sizeof line, "series (no values yet)  %s", tel);
        debug_line(core, line);
    }
    const bool ev = true;   // gestures everywhere; events only through a sink
    Gesture& g = v.gesture;
    before_plot(S, v);
    ImPlotAxisFlags xf, yf;
    plot_flags(S, v, &xf, &yf);
    const bool page_ticks = push_page_ticks(core, S);
    if (ImPlot::BeginPlot("##caliper.series", ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("step", nullptr, xf, yf);
        if (v.log_y) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        setup_zoom(v);
        // The retained steps [lo, hi), as the browser view bounds a selection.
        const int64_t lo = (int64_t)v.first, hi = (int64_t)(v.first + n);
        if (n) {
            ImPlotSpec line;
            if (S->look.plot) line.LineColor = ImGui::ColorConvertU32ToFloat4(S->look.plot);
            if (n <= kSeriesRawMax) {
                ImPlot::PlotLine("##line", v.values.data(), (int)n, 1.0, (double)v.first, line);
            } else {
                if (v.draw_dirty) decimate_series(v, 2048);
                ImPlot::PlotLine("##line", v.xs.data(), v.ys.data(), (int)v.xs.size(), line);
            }
            v.draw_dirty = false;
            if (ImPlot::IsPlotHovered() && !(ev && g.down)) {
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
                v.hover_item = (int64_t)(v.first + i);
                if (ev) {
                    const int64_t step = (int64_t)(v.first + i);
                    emit_hover(S, v, step, [&] { return step_detail(v, step); });
                }
            } else {
                v.hover_item = -1;
                if (ev && !g.down) v.hover_sent = -1;
            }
        }
        if (ev) {
            const double mx = ImPlot::GetPlotMousePos().x;
            if (plot_gesture(S, v, mx)) {
                const bool had = v.sel_s1 > v.sel_s0;
                if (g.selecting && g.moved) {
                    const int64_t s0 = std::max(lo, (int64_t)std::ceil(std::min(g.d0x, g.d1x)));
                    const int64_t s1 = std::min(hi, (int64_t)std::floor(std::max(g.d0x, g.d1x)) + 1);
                    if (s1 > s0) {
                        v.sel_s0 = s0;
                        v.sel_s1 = s1;
                        emit(S, v, "select", Detail().ipair("steps", s0, s1).str());
                    } else if (had) {
                        v.sel_s0 = v.sel_s1 = 0;
                        emit(S, v, "select", "{}");
                    }
                } else if (!g.moved) {
                    if (g.selecting && had) {
                        v.sel_s0 = v.sel_s1 = 0;
                        emit(S, v, "select", "{}");
                    }
                    if (hi > lo) {
                        const int64_t step = std::clamp((int64_t)std::llround(mx), lo, hi - 1);
                        emit(S, v, "pick", step_detail(v, step));
                    }
                }
                g.selecting = false;
            }
            if (v.sel_s1 > v.sel_s0) plot_band((double)v.sel_s0, (double)(v.sel_s1 - 1), true);
            if (g.down && g.selecting && g.moved) plot_band(g.d0x, g.d1x, false);
            // A linked selection: markers at the selected steps that are held
            // (at most 4096, the first ones).
            if (v.link_axis == 4 && n) {
                std::vector<double> mx, my;
                for (int64_t step : v.link_items) {
                    if (step < lo || step >= hi) continue;
                    const float y = v.values[(size_t)(step - lo)];
                    if (!std::isfinite(y)) continue;
                    mx.push_back((double)step);
                    my.push_back(y);
                    if (mx.size() >= 4096) break;
                }
                if (!mx.empty()) {
                    ImPlotSpec spec;
                    spec.Marker = ImPlotMarker_Circle;
                    spec.MarkerSize = 3.5f;
                    spec.MarkerFillColor = ImGui::ColorConvertU32ToFloat4(kSelLine);
                    spec.MarkerLineColor = ImGui::ColorConvertU32ToFloat4(kSelLine);
                    ImPlot::PlotScatter("##linked", mx.data(), my.data(), (int)mx.size(), spec);
                }
            }
        }
        after_plot(S, v);
        plot_geometry(v);
        ImPlot::EndPlot();
    }
    if (page_ticks) ImGui::PopFont();
    end_plot_input(S, v);
    ImGui::End();
}

// ---- histogram -------------------------------------------------------------
std::string bin_detail(const View& v, int64_t b) {
    const double lo = v.lo, w = ((double)v.hi - lo) / (double)std::max<size_t>(1, v.values.size());
    return Detail().integer("bin", b).num("lo", lo + (double)b * w).num("hi", lo + (double)(b + 1) * w)
        .num("count", (double)v.values[(size_t)b]).str();
}

void draw_histogram(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    const bool open = begin_window(core, v, ImVec2(360.0f, 200.0f));
    if (!open) { ImGui::End(); return; }
    const size_t B = v.values.size();
    draw_header(core, S, v, kToolPause | kToolLog);
    under_header(core, S, v);
    {
        char tel[160], line[220];
        telemetry(tel, sizeof tel, S, v);
        std::snprintf(line, sizeof line, "histogram bins=%zu [%.4g, %.4g]  %s", B, v.lo, v.hi, tel);
        debug_line(core, line);
    }
    const bool ev = true;   // gestures everywhere; events only through a sink
    Gesture& g = v.gesture;
    before_plot(S, v);
    ImPlotAxisFlags xf, yf;
    plot_flags(S, v, &xf, &yf);
    const bool page_ticks = push_page_ticks(core, S);
    if (ImPlot::BeginPlot("##caliper.histogram", ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes(nullptr, "count", xf, yf);
        if (v.log_y) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        setup_zoom(v);
        const double w = B ? ((double)v.hi - (double)v.lo) / (double)B : 1.0;
        if (B) {
            if (v.draw_dirty || v.xs.size() != B) {
                v.xs.resize(B);
                v.ys.resize(B);
                for (size_t i = 0; i < B; ++i) {
                    v.xs[i] = (double)v.lo + ((double)i + 0.5) * w;
                    v.ys[i] = v.values[i];
                }
                v.draw_dirty = false;
            }
            ImPlotSpec bars;
            if (S->look.plot) bars.FillColor = bars.LineColor = ImGui::ColorConvertU32ToFloat4(S->look.plot);
            ImPlot::PlotBars("##bars", v.xs.data(), v.ys.data(), (int)B, w, bars);
            bool over = false;
            if (ImPlot::IsPlotHovered() && !(ev && g.down)) {
                const ImPlotPoint mp = ImPlot::GetPlotMousePos();
                const double fi = std::floor((mp.x - (double)v.lo) / w);
                if (fi >= 0.0 && fi < (double)B) {
                    const size_t i = (size_t)fi;
                    over = true;
                    v.hover_item = (int64_t)i;
                    ImGui::SetTooltip("[%.6g, %.6g)\ncount %.6g", (double)v.lo + (double)i * w,
                                      (double)v.lo + (double)(i + 1) * w, (double)v.values[i]);
                    if (ev) emit_hover(S, v, (int64_t)i, [&] { return bin_detail(v, (int64_t)i); });
                }
            }
            v.hover_item = over ? v.hover_item : -1;
            if (ev && !over && !g.down) v.hover_sent = -1;
        }
        if (ev && B) {
            const double mx = ImPlot::GetPlotMousePos().x;
            if (plot_gesture(S, v, mx)) {
                const bool had = v.sel_s1 > v.sel_s0;
                if (g.selecting && g.moved) {
                    const int64_t b0 = std::max<int64_t>(0, (int64_t)std::floor((std::min(g.d0x, g.d1x) - v.lo) / w));
                    const int64_t b1 = std::min<int64_t>((int64_t)B, (int64_t)std::ceil((std::max(g.d0x, g.d1x) - v.lo) / w));
                    if (b1 > b0) {
                        v.sel_s0 = b0;
                        v.sel_s1 = b1;
                        emit(S, v, "select", Detail().pair("range", v.lo + (double)b0 * w, v.lo + (double)b1 * w).str());
                    } else if (had) {
                        v.sel_s0 = v.sel_s1 = 0;
                        emit(S, v, "select", "{}");
                    }
                } else if (!g.moved) {
                    if (g.selecting && had) {
                        v.sel_s0 = v.sel_s1 = 0;
                        emit(S, v, "select", "{}");
                    }
                    const double fi = std::floor((mx - (double)v.lo) / w);
                    if (fi >= 0.0 && fi < (double)B) emit(S, v, "pick", bin_detail(v, (int64_t)fi));
                }
                g.selecting = false;
            }
            if (v.sel_s1 > v.sel_s0) {
                if (v.sel_s1 > (int64_t)B) v.sel_s0 = v.sel_s1 = 0;   // fewer bins now
                else plot_band(v.lo + (double)v.sel_s0 * w, v.lo + (double)v.sel_s1 * w, true);
            }
            if (g.down && g.selecting && g.moved) plot_band(g.d0x, g.d1x, false);
        }
        after_plot(S, v);
        plot_geometry(v);
        ImPlot::EndPlot();
    }
    if (page_ticks) ImGui::PopFont();
    end_plot_input(S, v);
    ImGui::End();
}

// ---- heatmap / image overlay on the v1.2 window -----------------------------
// The payload axes of a heatmap or image: rows, columns and, for a mosaic,
// the tile axis (wire protocol v3, Payload axes).
struct CellAxes {
    const caliper_live::PayloadAxis* row = nullptr;
    const caliper_live::PayloadAxis* col = nullptr;
    const caliper_live::PayloadAxis* tile = nullptr;
};
CellAxes cell_axes(const View& v) {
    const auto& a = v.map.axes;
    CellAxes c;
    if (v.map.tile_cols > 0 && a.size() == 3) {
        c.tile = &a[0]; c.row = &a[1]; c.col = &a[2];
    } else {
        if (a.size() > 0) c.row = &a[0];
        if (a.size() > 1) c.col = &a[1];
    }
    return c;
}

// Where payload cell (r, c) is in the source: the tile (-1: not a mosaic) and
// the row and column inside it.
void locate(const View& v, const CellAxes& ax, int64_t r, int64_t c, int64_t* tile, int64_t* rr, int64_t* cc) {
    if (!ax.tile || !ax.row || !ax.col || ax.row->size <= 0 || ax.col->size <= 0) {
        *tile = -1; *rr = r; *cc = c;
        return;
    }
    *tile = (r / ax.row->size) * v.map.tile_cols + c / ax.col->size;
    *rr = r % ax.row->size;
    *cc = c % ax.col->size;
}

// A pick or hover of a cell: the source indices of its first position, the
// tile of a mosaic, and its value when it is a finite heatmap value.
std::string cell_detail(const View& v, const TensorView& tv, int64_t r, int64_t c, const uint8_t* cpu) {
    const CellAxes ax = cell_axes(v);
    int64_t tile, rr, cc;
    locate(v, ax, r, c, &tile, &rr, &cc);
    Detail d;
    d.integer("row", source_index(ax.row, rr)).integer("col", source_index(ax.col, cc));
    if (tile >= 0) d.integer("tile", source_index(ax.tile, tile));
    if (v.kind == CALIPER_LIVE_KIND_HEATMAP && cpu) {
        const float x = ((const float*)cpu)[(size_t)r * (size_t)tv.w + (size_t)c];
        if (std::isfinite(x)) d.num("value", (double)x);
    }
    return d.str();
}

// The browser heatmap's selection (web/src/views/heatmap.ts): the cells a
// drag from (x0, y0) to (x1, y1) covers (columns x, rows y, in cells), kept
// inside the tile where the drag began in a mosaic. Empty when r1 <= r0.
void select_cells(const View& v, int64_t rows, int64_t cols, double x0, double y0, double x1, double y1,
                  int64_t* r0, int64_t* r1, int64_t* c0, int64_t* c1) {
    const CellAxes ax = cell_axes(v);
    int64_t rlo = 0, rhi = rows, clo = 0, chi = cols;
    if (v.map.tile_cols > 0 && ax.row && ax.col && ax.row->size > 0 && ax.col->size > 0) {
        const int64_t tr = (int64_t)std::floor(std::clamp(y0, 0.0, (double)(rows - 1)) / (double)ax.row->size);
        const int64_t tc = (int64_t)std::floor(std::clamp(x0, 0.0, (double)(cols - 1)) / (double)ax.col->size);
        rlo = tr * ax.row->size; rhi = rlo + ax.row->size;
        clo = tc * ax.col->size; chi = clo + ax.col->size;
    }
    auto cr = [&](double x) { return std::clamp((int64_t)x, rlo, rhi); };
    auto cc = [&](double x) { return std::clamp((int64_t)x, clo, chi); };
    *c0 = cc(std::floor(std::min(x0, x1)));
    *c1 = cc(std::ceil(std::max(x0, x1)));
    *r0 = cr(std::floor(std::min(y0, y1)));
    *r1 = cr(std::ceil(std::max(y0, y1)));
}

std::string cells_detail(const View& v, int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
    const CellAxes ax = cell_axes(v);
    int64_t tile, rr, cc;
    locate(v, ax, r0, c0, &tile, &rr, &cc);
    int64_t ra, rb, ca, cb;
    source_range(ax.row, rr, rr + (r1 - r0), &ra, &rb);
    source_range(ax.col, cc, cc + (c1 - c0), &ca, &cb);
    Detail d;
    d.ipair("rows", ra, rb).ipair("cols", ca, cb);
    if (tile >= 0) d.integer("tile", source_index(ax.tile, tile));
    return d.str();
}

// Evenly spaced round integer ticks in [lo, hi], about n of them (plot.ts).
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

// The browser element's axes around a heatmap or image (web/src/views/plot.ts,
// drawAxes): a frame, ticks at cell centres labelled with source indices
// through the payload axes (a mosaic's per cell), the dims' names, and for a
// heatmap the colorbar at the right. Cells [x0, x1) x [y0, y1) are in view.
void draw_cell_axes(const View& v, const TensorView& tv, ImDrawList* dl, float ax, float ay, float aw, float ah,
                    double x0, double x1, double y0, double y1, float fs, float text_px, ImFont* font,
                    ImU32 faint_ink, ImU32 rule_ink) {
    // The given ink, else the browser element's.
    const ImU32 rule = rule_ink ? rule_ink : IM_COL32(0x5a, 0x57, 0x51, 255);
    const ImU32 faint = faint_ink ? faint_ink : IM_COL32(0xa4, 0x9d, 0x8f, 255);
    ImFont* mono = font ? font : mono_face();   // else the current face
    // fs sets the layout (10.5 is the browser's), text_px the type's ImGui size.
    const float k = fs / 10.5f;
    dl->AddRect(ImVec2(std::round(ax) - 0.5f, std::round(ay) - 0.5f),
                ImVec2(std::round(ax + aw) + 0.5f, std::round(ay + ah) + 0.5f), rule);
    const CellAxes cax = cell_axes(v);
    auto label = [&](const caliper_live::PayloadAxis* a, int64_t i) {
        const int64_t per = cax.tile && a && a->size > 0 ? a->size : INT64_MAX;
        return std::to_string(source_index(a, per == INT64_MAX ? i : i % per));
    };
    auto text = [&](ImVec2 p, const std::string& s, float align_x, float align_y) {
        const ImVec2 ts = mono ? mono->CalcTextSizeA(text_px, FLT_MAX, 0.0f, s.c_str()) : ImGui::CalcTextSize(s.c_str());
        dl->AddText(mono, text_px, ImVec2(p.x - ts.x * align_x, p.y - ts.y * align_y), faint, s.c_str());
    };
    const float sx = aw / (float)(x1 - x0), sy = ah / (float)(y1 - y0);
    for (int64_t c : index_ticks(std::max(x0 - 0.5, 0.0), std::min(x1 - 0.5, (double)tv.w - 1), std::max(2, (int)(aw / (70 * k))))) {
        const float x = std::round(ax + (float)(c + 0.5 - x0) * sx) + 0.5f;
        if (x < ax - 0.5f || x > ax + aw + 0.5f) continue;
        dl->AddLine(ImVec2(x, ay + ah + 0.5f), ImVec2(x, ay + ah + 4.5f), rule);
        text(ImVec2(x, ay + ah + 6.0f * k), label(cax.col, c), 0.5f, 0.0f);
    }
    for (int64_t r : index_ticks(std::max(y0 - 0.5, 0.0), std::min(y1 - 0.5, (double)tv.h - 1), std::max(2, (int)(ah / (36 * k))))) {
        const float y = std::round(ay + (float)(r + 0.5 - y0) * sy) + 0.5f;
        if (y < ay - 0.5f || y > ay + ah + 0.5f) continue;
        dl->AddLine(ImVec2(ax - 0.5f, y), ImVec2(ax - 4.5f, y), rule);
        text(ImVec2(ax - 6.0f * k, y), label(cax.row, r), 1.0f, 0.5f);
    }
    if (cax.col && !cax.col->dim.empty()) text(ImVec2(ax + aw, ay + ah + 19.0f * k), cax.col->dim, 1.0f, 0.0f);
    if (cax.row && !cax.row->dim.empty()) text(ImVec2(ax - 6.0f * k, ay - 4.0f * k), cax.row->dim, 1.0f, 1.0f);
    if (v.kind == CALIPER_LIVE_KIND_HEATMAP) {
        // The colorbar: the colormap from its top (index 255) to its bottom.
        const uint32_t* lut = caliper_host::colormap_lut(tv.colormap);
        const float bx = ax + aw + 8.0f * k, bw = 8.0f * k;
        const int n = std::max(2, (int)std::floor(ah));
        for (int i = 0; lut && i < n; ++i) {
            const int idx = (int)std::lround(255.0 * (1.0 - (double)i / (n - 1)));
            const uint32_t c = lut[idx];
            dl->AddRectFilled(ImVec2(bx, ay + i * ah / n), ImVec2(bx + bw, ay + (i + 1) * ah / n + 0.5f),
                              IM_COL32(c & 0xff, (c >> 8) & 0xff, (c >> 16) & 0xff, 255));
        }
        dl->AddRect(ImVec2(bx - 0.5f, ay - 0.5f), ImVec2(bx + bw + 0.5f, ay + ah + 0.5f), rule);
    }
}

void draw_texture_overlay(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    TensorView* tv = nullptr;
    for (auto& t : core->views)
        if (t.name == v.name) { tv = &t; break; }
    // Only a windowed v1.2 view that views_draw began this frame (placed is set
    // on its first windowed draw); RAW views have no chrome by design.
    if (!tv || !tv->tex || !tv->placed || (tv->flags & CALIPER_VIEW_FLAG_RAW)) return;
    if (ImGui::Begin(window_label(v).c_str())) {   // appends to that window
        const ImVec2 wp = ImGui::GetWindowPos(), ws = ImGui::GetWindowSize();
        const ImGuiStyle& st = ImGui::GetStyle();
        const bool on_button = false;
        draw_header(core, S, v, kToolPause);
        // With CALIPER_VIEWS_DEBUG=1 the diagnostics take a line under the
        // image (the v1.2 window writes its own line above it); nothing is
        // ever drawn over the image.
        const bool debug = !core->views_bare && views_debug(core);
        if (debug) {
            char tel[160], line[200];
            telemetry(tel, sizeof tel, S, v);
            std::snprintf(line, sizeof line, "%s  %s", v.kind == CALIPER_LIVE_KIND_HEATMAP ? "heatmap" : "image", tel);
            const float y = wp.y + ws.y - st.WindowPadding.y - ImGui::GetTextLineHeight();
            const std::string t = fit_text(line, ws.x - 2.0f * st.WindowPadding.x);
            ImGui::GetWindowDrawList()->AddText(ImVec2(wp.x + st.WindowPadding.x, y), ImGui::GetColorU32(ImGuiCol_Text),
                                                t.c_str());
        }

        // The plot area and how cells map onto it: kx and ky screen units per
        // cell across and down, and the cell (wx0, wy0) at the area's top left.
        // The view pans and zooms like the browser element's heatmap, in the
        // app as in the browser, and draws the image itself: the wheel zooms
        // about the pointer, a drag pans, a double click fits it again. The
        // plot area takes the gestures, so a drag there never moves the
        // window; the title bar still does. Every view takes the browser
        // element's layout, in the app as on a page: axes of row and column
        // indices around the plot area, titled with the dims' names, a
        // colorbar beside a heatmap, a heatmap stretched to the area and an
        // image at equal aspect. Only a bare view without an event sink (an
        // embedder that draws its own chrome) keeps the plain fit with square
        // cells and no axes.
        const ImVec2 cs = ImGui::GetCursorStartPos();
        const bool ev = true;
        const bool page = core->views_bare ? (bool)S->events : true;
        const bool image = v.kind == CALIPER_LIVE_KIND_IMAGE;
        // The axes' type: the browser's 10.5 px on a page, the host's small
        // mono in a window; the margins grow with it.
        const float afs = core->views_bare ? 10.5f
                          : S->look.mono_size > 0.0f ? S->look.mono_size - 1.0f
                                                     : std::max(10.5f, ImGui::GetFontSize() * 0.85f);
        const float k = afs / 10.5f;
        // Margins of the element's heatmap view: tick labels left and below, a
        // colorbar right of a heatmap (web/src/views/heatmap.ts).
        const float ml = page ? 46.0f * k : 0.0f, mt = page ? 16.0f * k : 0.0f;
        const float mr = page ? (image ? 10.0f : 26.0f) * k : 0.0f;
        const float mb = (page ? 32.0f * k : 0.0f) + (debug ? ImGui::GetTextLineHeightWithSpacing() : 0.0f);
        // On a page the margins are the element's, from the plate's edges; in
        // a window, inside its padding.
        const bool plate = page && core->views_bare;
        const float pad_x = plate ? 0.0f : st.WindowPadding.x, pad_y = plate ? 0.0f : st.WindowPadding.y;
        const float ax = wp.x + (plate ? 0.0f : cs.x) + ml;
        // Under the header row of a docked window, and the v1.2 window's
        // diagnostics line in debug.
        ImGuiWindow* cw = ImGui::GetCurrentWindow();
        const float head = !core->views_bare && cw->DockIsActive ? ImGui::GetFrameHeight() + st.ItemSpacing.y : 0.0f;
        const float above = head + (debug ? ImGui::GetTextLineHeightWithSpacing() : 0.0f);
        // The host's row (a data view picker) under them, then the image.
        float extra = 0.0f;
        if (!core->views_bare && S->under_header) {
            ImGui::SetCursorScreenPos(ImVec2(wp.x + cs.x, wp.y + cs.y + above));
            extra = under_header(core, S, v);
        }
        const float ay = wp.y + (plate ? 0.0f : cs.y) + above + extra + mt;
        const float aw = wp.x + ws.x - pad_x - ax - mr;
        const float ah = wp.y + ws.y - pad_y - ay - mb;
        const ImVec2 m = ImGui::GetIO().MousePos;
        ImGuiIO& io = ImGui::GetIO();
        // A CPU payload's copy, or the MTLBuffer the v1.2 view still imports.
        const bool live_buf = v.tex_cpu && (v.tex_cpu == v.tex_copy.data() ||
                                            (tv->alloc && tv->alloc_buf == v.tex_buf));
        const uint8_t* cpu = live_buf ? v.tex_cpu : nullptr;
        tv->drawn_by_live = ev;
        if (aw < 1.0f || ah < 1.0f || tv->w <= 0 || tv->h <= 0) { ImGui::End(); return; }
        const double W = tv->w, H = tv->h;
        // The fit: equal units across and down (an image, or a heatmap off a
        // page), or a heatmap stretched to the area on a page.
        const double kfit = std::min(aw / W, ah / H);
        const double kxfit = page && !image ? aw / W : kfit, kyfit = page && !image ? ah / H : kfit;
        if (v.zoom_w != tv->w || v.zoom_h != tv->h) v.zoom_fitted = true;   // another payload shape
        v.zoom_w = tv->w;
        v.zoom_h = tv->h;
        if (v.zoom_fitted) {
            v.zoom_kx = kxfit;
            v.zoom_ky = kyfit;
            v.zoom_cx = W * 0.5;
            v.zoom_cy = H * 0.5;
        }
        double kx = v.zoom_kx, ky = v.zoom_ky;
        double wx0 = v.zoom_cx - aw * 0.5 / kx, wy0 = v.zoom_cy - ah * 0.5 / ky;
        auto at = [&](ImVec2 p, double* cx, double* cy) {
            *cx = wx0 + (p.x - ax) / kx;
            *cy = wy0 + (p.y - ay) / ky;
        };
        auto px = [&](double cx, double cy) { return ImVec2(ax + (float)((cx - wx0) * kx), ay + (float)((cy - wy0) * ky)); };
        auto cell = [&](ImVec2 p, int64_t* r, int64_t* c) {
            if (p.x < ax || p.y < ay || p.x >= ax + aw || p.y >= ay + ah) return false;
            double cx, cy;
            at(p, &cx, &cy);
            if (!(cx >= 0.0 && cx < W && cy >= 0.0 && cy < H)) return false;
            *c = std::min<int64_t>(tv->w - 1, (int64_t)cx);
            *r = std::min<int64_t>(tv->h - 1, (int64_t)cy);
            return true;
        };
        bool active = false;
        if (ev) {
            // An invisible button over the content takes the gestures.
            if (v.sel_r1 > v.sel_r0 && (v.sel_rows != tv->h || v.sel_cols != tv->w)) v.sel_r0 = v.sel_r1 = 0;
            ImGui::SetCursorScreenPos(ImVec2(ax, ay));
            ImGui::InvisibleButton("##caliper.cells", ImVec2(aw, ah),
                                   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            active = ImGui::IsItemActive();
            const bool hovered = ImGui::IsItemHovered();
            Gesture& g = v.gesture;
            const bool released = track_gesture(v, m, true);
            // Pan and zoom (a drag without shift, the wheel), and the fit. The
            // pan follows the pointer from where it last moved the view, so a
            // drag released in the frame it moved still pans.
            if ((g.down || released) && !g.selecting && g.moved && (m.x != g.lx || m.y != g.ly)) {
                v.zoom_fitted = false;
                v.zoom_cx -= (m.x - g.lx) / v.zoom_kx;
                v.zoom_cy -= (m.y - g.ly) / v.zoom_ky;
            }
            g.lx = m.x;
            g.ly = m.y;
            if (hovered && io.MouseWheel != 0.0f) {
                double cx, cy;
                at(m, &cx, &cy);
                const double f = std::clamp(v.zoom_kx * std::exp(0.15 * (double)io.MouseWheel), kxfit * 1e-2, kxfit * 1e4) / v.zoom_kx;
                v.zoom_fitted = false;
                v.zoom_kx *= f;
                v.zoom_ky *= f;
                v.zoom_cx = cx - (m.x - (ax + aw * 0.5)) / v.zoom_kx;
                v.zoom_cy = cy - (m.y - (ay + ah * 0.5)) / v.zoom_ky;
            }
            if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) v.zoom_fitted = true;
            if (v.zoom_fitted) {
                v.zoom_kx = kxfit;
                v.zoom_ky = kyfit;
                v.zoom_cx = W * 0.5;
                v.zoom_cy = H * 0.5;
            }
            kx = v.zoom_kx;
            ky = v.zoom_ky;
            wx0 = v.zoom_cx - aw * 0.5 / kx;
            wy0 = v.zoom_cy - ah * 0.5 / ky;
            if (g.down || released) {
                double cx, cy;
                at(m, &cx, &cy);
                if (ImGui::IsItemActivated()) { g.d0x = cx; g.d0y = cy; }
                g.d1x = cx;
                g.d1y = cy;
            }
            if (released) {
                const bool had = v.sel_r1 > v.sel_r0;
                if (g.selecting && g.moved) {
                    int64_t r0, r1, c0, c1;
                    select_cells(v, tv->h, tv->w, g.d0x, g.d0y, g.d1x, g.d1y, &r0, &r1, &c0, &c1);
                    if (r1 > r0 && c1 > c0) {
                        v.sel_r0 = r0; v.sel_r1 = r1; v.sel_c0 = c0; v.sel_c1 = c1;
                        v.sel_rows = tv->h;
                        v.sel_cols = tv->w;
                        emit(S, v, "select", cells_detail(v, r0, r1, c0, c1));
                    } else if (had) {
                        v.sel_r0 = v.sel_r1 = 0;
                        emit(S, v, "select", "{}");
                    }
                } else if (!g.moved) {
                    // A shift-click clears the selection, which an empty detail reports.
                    if (g.selecting && had) {
                        v.sel_r0 = v.sel_r1 = 0;
                        emit(S, v, "select", "{}");
                    }
                    int64_t r, c;
                    if (cell(m, &r, &c)) emit(S, v, "pick", cell_detail(v, *tv, r, c, cpu));
                }
                g.selecting = false;
            }

            // The image: the visible cells, nearest-sampled once a cell covers
            // a physical pixel or more so that cells read as cells.
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->PushClipRect(ImVec2(ax, ay), ImVec2(ax + aw, ay + ah), true);
            const double c0 = std::max(wx0, 0.0), c1 = std::min(wx0 + aw / kx, W);
            const double r0 = std::max(wy0, 0.0), r1 = std::min(wy0 + ah / ky, H);
            if (c1 > c0 && r1 > r0) {
                const ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
                const bool magnify = std::min(kx, ky) * core->scale >= 1.0;
                if (magnify && pio.DrawCallback_SetSamplerNearest) dl->AddCallback(pio.DrawCallback_SetSamplerNearest, nullptr);
                dl->AddImage((ImTextureID)tv->tex, px(c0, r0), px(c1, r1),
                             ImVec2((float)(c0 / W), (float)(r0 / H)), ImVec2((float)(c1 / W), (float)(r1 / H)));
                if (magnify && pio.DrawCallback_SetSamplerLinear) dl->AddCallback(pio.DrawCallback_SetSamplerLinear, nullptr);
            }
            // The selection, and the drag that makes one.
            if (v.sel_r1 > v.sel_r0) {
                const ImVec2 a = px((double)v.sel_c0, (double)v.sel_r0), b = px((double)v.sel_c1, (double)v.sel_r1);
                dl->AddRectFilled(a, b, kSelFill);
                dl->AddRect(a, b, kSelLine, 0.0f, 0, 1.25f);
            }
            if (g.down && g.selecting && g.moved) {
                const ImVec2 a = px(g.d0x, g.d0y), b = px(g.d1x, g.d1y);
                const ImVec2 lo(std::min(a.x, b.x), std::min(a.y, b.y)), hi(std::max(a.x, b.x), std::max(a.y, b.y));
                dl->AddRectFilled(lo, hi, kSelFill);
                dl->AddRect(lo, hi, kSelLine, 0.0f, 0, 1.0f);
            }
            // A linked selection: each run of rows (or columns) that stand
            // for a selected item gets a band over the image, an outline and
            // a mark at the image's edge.
            if ((v.link_axis == 2 || v.link_axis == 3) && !v.link_items.empty()) {
                const bool rows = v.link_axis == 2;
                const CellAxes cax = cell_axes(v);
                const caliper_live::PayloadAxis* ax = rows ? cax.row : cax.col;
                const int64_t n = rows ? tv->h : tv->w;
                const int64_t per = ax && cax.tile && ax->size > 0 ? ax->size : n;   // a mosaic repeats the axis per cell
                auto marked = [&](int64_t i) {
                    int64_t a, b;
                    source_range(ax, i % per, i % per + 1, &a, &b);
                    return linked_in(v, a, b);
                };
                for (int64_t i = 0; i < n;) {
                    if (!marked(i)) { ++i; continue; }
                    int64_t j = i + 1;
                    while (j < n && marked(j)) ++j;
                    const ImVec2 p0 = rows ? px(0.0, (double)i) : px((double)i, 0.0);
                    const ImVec2 p1 = rows ? px(W, (double)j) : px((double)j, H);
                    dl->AddRectFilled(p0, p1, kSelFill);
                    dl->AddRect(p0, p1, kSelLine, 0.0f, 0, 1.5f);
                    dl->AddRectFilled(p0, rows ? ImVec2(p0.x + 4.0f, p1.y) : ImVec2(p1.x, p0.y + 4.0f), kSelLine);
                    i = j;
                }
            }
            dl->PopClipRect();
            if (page) {
                const bool house = !core->views_bare;
                draw_cell_axes(v, *tv, dl, ax, ay, aw, ah, wx0, wx0 + aw / kx, wy0, wy0 + ah / ky, afs,
                               house ? afs : afs * kMonoEm, house ? S->look.mono : nullptr, house ? S->look.faint : 0,
                               house ? S->look.rule : 0);
            }
        }
        v.geo_valid = true;
        v.geo_area[0] = ax; v.geo_area[1] = ay; v.geo_area[2] = aw; v.geo_area[3] = ah;
        v.geo_window[0] = wx0; v.geo_window[1] = wx0 + aw / kx;
        v.geo_window[2] = wy0; v.geo_window[3] = wy0 + ah / ky;
        v.geo_ydown = true;

        int64_t r = 0, c = 0;
        const bool in = !on_button && !active && ImGui::IsWindowHovered() && cell(m, &r, &c);
        v.hover_item = in ? r * tv->w + c : -1;
        if (in) {
            const int row = (int)r, col = (int)c;
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
        if (ev && !active) emit_hover(S, v, v.hover_item, [&] { return cell_detail(v, *tv, r, c, cpu); });
    }
    ImGui::End();
}

}  // namespace

std::string info_text(const CaliperCore* core, const View& v) {
    char b[200];
    switch (v.kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
        case CALIPER_LIVE_KIND_IMAGE:
            for (const auto& tv : core->views)
                if (tv.name == v.name && tv.tex) {
                    if (v.kind == CALIPER_LIVE_KIND_IMAGE) {
                        std::snprintf(b, sizeof b, "%d%s%d RGBA", tv.h, kTimes, tv.w);
                    } else {
                        std::snprintf(b, sizeof b, "%d%s%d  %s %s %s", tv.h, kTimes, tv.w, fmt4(tv.vmin).c_str(),
                                      kEllipsis, fmt4(tv.vmax).c_str());
                    }
                    return b;
                }
            return {};
        case CALIPER_LIVE_KIND_POINTS: {
            std::string label = v.label_count ? std::to_string(v.label_count) + " classes"
                                : v.cur.aux       ? std::string("values")
                                                  : std::string();
            std::snprintf(b, sizeof b, "%lld points  %d-D%s%s", (long long)v.count, v.dims, label.empty() ? "" : "  ",
                          label.c_str());
            return b;
        }
        case CALIPER_LIVE_KIND_SERIES:
            if (v.values.empty()) return {};
            std::snprintf(b, sizeof b, "%llu steps  last %s", (unsigned long long)(v.first + v.values.size()),
                          fmt4(v.values.back()).c_str());
            return b;
        case kKindScene:
            if (const SceneView* sv = scene_view(v)) return sv->info();
            return {};
        case CALIPER_LIVE_KIND_HISTOGRAM: {
            double total = 0.0;
            for (float c : v.values) total += c;
            std::snprintf(b, sizeof b, "%zu bins  %s %s %s  n %s", v.values.size(), fmt4(v.lo).c_str(), kEllipsis,
                          fmt4(v.hi).c_str(), fmt4(total).c_str());
            return b;
        }
        default:
            return {};
    }
}

// The browser's points view's sizes (web/src/views/points.ts, radius()), which the scene core
// keeps: the side of a point in logical px, from 7 for a few hundred points down to 3 past 50,000.
float point_size_px(uint64_t n, float scale) { return caliper_scene::default_point_size(n) * scale; }

bool points_mvp(const View& v, float out[16]) {
    const PointsScene* sc = v.points ? v.points->scene.get() : nullptr;
    return sc && v.tex_w > 0 && v.tex_h > 0 && sc->mvp((uint32_t)v.tex_w, (uint32_t)v.tex_h, out);
}

void draw_view(CaliperCore* core, CaliperLiveViewsState* S, View& v) {
    switch (v.kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
        case CALIPER_LIVE_KIND_IMAGE:     draw_texture_overlay(core, S, v); break;
        case CALIPER_LIVE_KIND_POINTS:    draw_points(core, S, v); break;
        case CALIPER_LIVE_KIND_SERIES:    draw_series(core, S, v); break;
        case CALIPER_LIVE_KIND_HISTOGRAM: draw_histogram(core, S, v); break;
        case kKindScene:                  draw_scene(core, S, v); break;
        default: break;
    }
}

}  // namespace caliper_live_views
