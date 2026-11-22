// live_views_scene.cpp: the scene view in the typed live views (live_views_scene.h).
#include "live/live_views_scene.h"

#include "caliper/live_net.h"
#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "embed/render_stats.h"   // copy counters
#include "live/live_scene.h"
#include "live/scene_overlays.h"
#include "renderer/host_renderer.h"

#if defined(__EMSCRIPTEN__)
#include "gfx/shaders/generated/programs_webgpu.h"
#include "renderer/webgpu_renderer.h"
#elif defined(__APPLE__)
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace caliper_live_views {

static_assert(kKindScene == CALIPER_NET_KIND_SCENE, "the scene kind of the view registry and of protocol 3.2");

SceneHolder::~SceneHolder() {
    view.reset();   // its resources go back to the layer before the layer goes
    gfx.reset();
}

namespace {

CaliperLiveViewsState* views_state(CaliperCore* core) {
    if (!core->live_views) core->live_views = new CaliperLiveViewsState();
    return core->live_views;
}

View* find_scene(CaliperCore* core, const std::string& name) {
    View* v = find(core, name.c_str());
    return v && v->kind == kKindScene ? v : nullptr;
}

}  // namespace

bool open_layer(CaliperCore* core, const char* what, std::unique_ptr<caliper_gfx::Backend>* owned,
                caliper_gfx::Backend** out, const caliper_gfx::ProgramDesc** programs, std::string* notice) {
    caliper_host::HostRenderer* r = core ? core->renderer.get() : nullptr;
#if defined(__EMSCRIPTEN__)
    (void)owned;
    if (caliper_gfx::Backend* b = r ? static_cast<caliper_host::WebGpuRenderer*>(r)->gfx() : nullptr) {
        *out = b;
        *programs = caliper_gfx::shaders::kWebgpuPrograms;
        return true;
    }
    *notice = std::string(what) + " need the graphics layer's WebGPU backend, which has not started";
#elif defined(__APPLE__)
    if (r && r->native_device() && r->native_queue()) {
        *owned = caliper_gfx::make_metal_backend(r->native_device(), r->native_queue());
        if (*owned) {
            *out = owned->get();
            *programs = caliper_gfx::shaders::kMetalPrograms;
            return true;
        }
    }
    *notice = std::string(what) + " need the Metal renderer";
#else
    (void)r;
    (void)owned;
    *notice = std::string(what) + " need the Metal renderer or a browser with WebGPU";
#endif
    (void)out;
    (void)programs;
    return false;
}

bool scene_open(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneView& info,
                std::string* why) {
    if (!core || name.empty()) {
        if (why) *why = "no core or no name";
        return false;
    }
    if (!core->canvas_ready || !core->renderer) {
        if (why) *why = "attach a canvas first";
        return false;
    }
    View* v = find(core, name.c_str());
    if (v && v->kind != kKindScene) {
        caliper_core_live_view_remove(core, name.c_str());   // another kind: the scene replaces it
        v = nullptr;
    }
    const bool opened = !v;
    if (!v) {
        for (const auto& tv : core->views)
            if (tv.name == name) {
                if (why) *why = "the name is taken by an embed v1.2 view (caliper.show)";
                return false;
            }
        auto nv = std::make_unique<View>();
        nv->name = name;
        nv->kind = kKindScene;
        nv->creates = 0;
        nv->path = caliper_gfx::route_name(caliper_gfx::Route::CopiedFromHost);
        auto h = std::make_shared<SceneHolder>();
        caliper_gfx::Backend* gfx = nullptr;
        const caliper_gfx::ProgramDesc* programs = nullptr;
        if (open_layer(core, "scenes", &h->gfx, &gfx, &programs, &h->notice)) {
            h->view = std::make_unique<SceneView>(*gfx, programs);
            std::string err;
            if (!h->view->init(&err)) {
                h->view.reset();
                h->gfx.reset();
                h->notice = "the scene view could not start: " + err;
            }
        }
        if (!h->notice.empty()) core->log(1, "live: scene '" + name + "': " + h->notice);
        nv->scene = std::move(h);
        v = nv.get();
        views_state(core)->views.push_back(std::move(nv));
    }
    if (SceneView* sv = scene_view(*v)) {
        sv->describe(info);
        CaliperLiveViewsState* S = views_state(core);
        if (auto it = S->scene_settings.find(name); opened && it != S->scene_settings.end()) {
            SceneView::Settings s;
            s.up_chosen = it->second.up_chosen;
            s.up = caliper_scene::Axis(it->second.up);
            s.scale_bar = it->second.scale_bar;
            sv->restore(s);
        }
    }
    return true;
}

void scene_remember(CaliperCore* core, const View& v) {
    const SceneView* sv = scene_view(v);
    if (!core || !sv || !core->live_views) return;
    const SceneView::Settings s = sv->settings();
    core->live_views->scene_settings[v.name] = SceneSettings{s.up_chosen, uint8_t(s.up), s.scale_bar};
}

bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneUpdate& u,
                 std::string* why) {
    View* v = core ? find_scene(core, name) : nullptr;
    if (!v) {
        if (why) *why = "no scene view named '" + name + "'";
        return false;
    }
    SceneView* sv = scene_view(*v);
    if (!sv) return true;   // the notice says why nothing is drawn
    const uint64_t before = sv->counts().bytes_total;
    const bool ok = sv->apply(u, why);
    const uint64_t copied = sv->counts().bytes_total - before;
    if (copied) caliper_stats::note_copy(core, name, copied);   // the host bytes uploaded into its buffers
    ++v->updates;
    v->seq = u.seq;
    v->count = int64_t(sv->counts().items);
    return ok;
}

bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneClear& c,
                 std::string* why) {
    View* v = core ? find_scene(core, name) : nullptr;
    if (!v) {
        if (why) *why = "no scene view named '" + name + "'";
        return false;
    }
    SceneView* sv = scene_view(*v);
    if (!sv) return true;
    const bool ok = sv->apply(c, why);
    v->seq = c.seq;
    v->count = int64_t(sv->counts().items);
    return ok;
}

namespace {

// A message that changes no batch: entity transforms or a class table. A class table's texture is
// uploaded at the next frame, which counts it (draw_scene).
template <class M> bool apply_other(CaliperCore* core, const std::string& name, const M& m, std::string* why) {
    View* v = core ? find_scene(core, name) : nullptr;
    if (!v) {
        if (why) *why = "no scene view named '" + name + "'";
        return false;
    }
    SceneView* sv = scene_view(*v);
    if (!sv) return true;
    const bool ok = sv->apply(m, why);
    v->seq = m.seq;
    return ok;
}

}  // namespace

bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneTransforms& t,
                 std::string* why) {
    return apply_other(core, name, t, why);
}

bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneClasses& c,
                 std::string* why) {
    return apply_other(core, name, c, why);
}

void scene_note(CaliperCore* core, const std::string& name, const std::string& text) {
    View* v = core ? find_scene(core, name) : nullptr;
    if (SceneView* sv = v ? scene_view(*v) : nullptr) sv->note(text);
}

void scene_link(CaliperCore* core, const std::string& name, const std::map<std::string, std::vector<int64_t>>& by_space) {
    View* v = core ? find_scene(core, name) : nullptr;
    if (SceneView* sv = v ? scene_view(*v) : nullptr) sv->set_all_linked(by_space);
}

void scene_begin_round(CaliperCore* core, const std::string& name) {
    View* v = core ? find_scene(core, name) : nullptr;
    if (SceneView* sv = v ? scene_view(*v) : nullptr) sv->begin_round();
    if (v) caliper_stats::begin_update(core, name);
}

SceneView* scene_view(CaliperCore* core, const char* name) {
    View* v = core && name ? find_scene(core, name) : nullptr;
    return v ? scene_view(*v) : nullptr;
}

SceneView* scene_view(View& v) { return v.scene ? v.scene->view.get() : nullptr; }
const SceneView* scene_view(const View& v) { return v.scene ? v.scene->view.get() : nullptr; }

// ---- A scene drawn into a rectangle ----------------------------------------------------------------

void pick_scene_rect(SceneRectState& state, float u, float v) {
    state.seam_u = std::clamp(u, 0.0f, 1.0f);
    state.seam_v = std::clamp(v, 0.0f, 1.0f);
}

SceneRectResult draw_scene_in_rect(CaliperCore* core, const std::string& name, const char* imgui_id, ImVec2 p0,
                                   ImVec2 size, const SceneRectOptions& options, SceneRectState& state,
                                   const SceneRectSink& sink) {
    SceneView* sc = scene_view(core, name.c_str());
    if (!sc) {
        state = SceneRectState{};
        return SceneRectResult::missing;
    }
    return draw_scene_view_in_rect(*sc, core->scale, imgui_id, p0, size, options, state, sink);
}

SceneRectResult draw_scene_view_in_rect(SceneView& view, float pixel_scale, const char* imgui_id, ImVec2 p0,
                                        ImVec2 size, const SceneRectOptions& options, SceneRectState& state,
                                        const SceneRectSink& sink) {
    SceneView* sc = &view;
    if (size.x < 4.0f || size.y < 4.0f) return SceneRectResult::too_small;
    constexpr int kMaxTarget = 8192, kHysteresisPx = 8;
    const float scale = pixel_scale > 0.0f ? pixel_scale : 1.0f;
    int want_w = std::clamp(int(size.x * scale), 1, kMaxTarget);
    int want_h = std::clamp(int(size.y * scale), 1, kMaxTarget);
    if (sc->has_target() && std::abs(want_w - int(sc->width())) <= kHysteresisPx &&
        std::abs(want_h - int(sc->height())) <= kHysteresisPx) {
        want_w = int(sc->width());
        want_h = int(sc->height());
    }

    // The panel's options: the up axis once per value, the scale bar each frame.
    if (!options.up.empty() && options.up != state.up_applied) {
        for (caliper_scene::Axis a : {caliper_scene::Axis::PosX, caliper_scene::Axis::NegX, caliper_scene::Axis::PosY,
                                      caliper_scene::Axis::NegY, caliper_scene::Axis::PosZ, caliper_scene::Axis::NegZ})
            if (options.up == caliper_scene::axis_name(a)) sc->switch_up(a);
        state.up_applied = options.up;
    }
    sc->set_scale_bar(options.scale_bar);
    sc->set_stage_source(options.stages);
    sc->poll();   // the GPU time of completed frames: the info line and the stages' time watchdog

    // ---- input: an invisible button owns the rectangle, so that a drag never moves the window
    ImGui::SetCursorScreenPos(p0);
    ImGui::InvisibleButton(imgui_id, size,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                               ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    ImGuiIO& io = ImGui::GetIO();
    const bool valid = ImGui::IsMousePosValid();
    if (ImGui::IsItemActivated() && valid) {
        state.down = true;
        state.moved = false;
        state.button = ImGui::IsMouseClicked(ImGuiMouseButton_Right)    ? 1
                       : ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ? 2
                                                                        : 0;
        state.selecting = io.KeyShift && state.button == 0;
        state.x0 = state.lx = io.MousePos.x;
        state.y0 = state.ly = io.MousePos.y;
    }
    bool released = false;
    if (state.down) {
        if (valid && (std::fabs(io.MousePos.x - state.x0) > io.MouseDragThreshold ||
                      std::fabs(io.MousePos.y - state.y0) > io.MouseDragThreshold))
            state.moved = true;
        const float dx = valid ? io.MousePos.x - state.lx : 0.0f, dy = valid ? io.MousePos.y - state.ly : 0.0f;
        if (valid) {
            state.lx = io.MousePos.x;
            state.ly = io.MousePos.y;
        }
        if (state.moved && !state.selecting && (dx != 0.0f || dy != 0.0f)) {
            if (state.button != 0 || io.KeyAlt || io.KeyCtrl) sc->pan(dx, dy, size.y);
            else sc->orbit(dx, dy);
        }
        released = !ImGui::IsMouseDown(state.button);
    }
    if (hovered && io.MouseWheel != 0.0f) {
        const float mx = (io.MousePos.x - p0.x) / size.x * 2.0f - 1.0f;
        const float my = 1.0f - (io.MousePos.y - p0.y) / size.y * 2.0f;
        sc->zoom(io.MouseWheel, mx, my, size.x, size.y);
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) sc->fit();

    // ---- GPU: the scene core compiles and submits when something changed
    const uint32_t clear = ImGui::GetColorU32(ImGuiCol_WindowBg) | 0xff000000u;
    std::string why;
    if (!sc->render(uint32_t(want_w), uint32_t(want_h), scale, clear, &why) && !why.empty()) sc->note(why);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + size.x, p0.y + size.y);
    if (const uint64_t tex = sc->imgui_texture()) dl->AddImage((ImTextureID)tex, p0, p1);
    if (!options.stage_message.empty()) {   // a stage's refusal or stop, over the picture's bottom
        const float pad = 4.0f, h = ImGui::GetTextLineHeight() + 2 * pad;
        const ImVec2 a(p0.x, std::max(p0.y, p1.y - h));
        dl->AddRectFilled(a, p1, IM_COL32(0, 0, 0, 170));
        dl->PushClipRect(a, p1, true);
        dl->AddText(ImVec2(a.x + pad, a.y + pad), IM_COL32(255, 196, 120, 255), options.stage_message.c_str());
        dl->PopClipRect();
    }

    // ---- picks and selections, in the render target's pixels
    const float kx = sc->width() ? float(sc->width()) / size.x : 1.0f;
    const float ky = sc->height() ? float(sc->height()) / size.y : 1.0f;
    auto target_px = [&](float x, float y) { return ImVec2((x - p0.x) * kx, (y - p0.y) * ky); };
    if (released) {
        if (state.selecting && state.moved && valid) {
            const ImVec2 a = target_px(state.x0, state.y0), b = target_px(io.MousePos.x, io.MousePos.y);
            const std::vector<float> poly = {a.x, a.y, b.x, a.y, b.x, b.y, a.x, b.y};
            if (sink) sink("select", sc->select_polygon(poly));
        } else if (!state.moved && state.button == 0) {
            if (state.selecting) {
                if (sc->clear_selection() && sink) sink("select", "{}");
            } else if (valid) {
                const ImVec2 t = target_px(io.MousePos.x, io.MousePos.y);
                std::string pick_why;
                state.click = sc->request_pick(t.x, t.y, &pick_why) || state.click;
            }
        }
        state.down = false;
        state.selecting = false;
    }
    // A seam pick stays asked until its own request gives a hit or a miss: a
    // live scene that changes before the pick's readback drops the result, and
    // the pick is then asked again at the next draw.
    if (state.seam_u >= 0.0f && !state.click && sc->width() && sc->height()) {
        std::string pick_why;
        state.seam_asked = sc->request_pick(state.seam_u * float(sc->width()), state.seam_v * float(sc->height()),
                                            &pick_why);
        state.click = state.seam_asked;
    }
    caliper_scene::Hit hit;
    const caliper_scene::PickPass::Result pr = sc->poll_pick(&hit);
    if (pr == caliper_scene::PickPass::Result::Hit || pr == caliper_scene::PickPass::Result::Miss ||
        pr == caliper_scene::PickPass::Result::Dropped) {
        if (state.click && pr == caliper_scene::PickPass::Result::Hit && sink) sink("pick", sc->pick_detail(hit));
        if (state.seam_asked && pr != caliper_scene::PickPass::Result::Dropped) state.seam_u = state.seam_v = -1.0f;
        state.seam_asked = false;
        state.click = false;
    }

    // ---- the selection rectangle being drawn, the scale bar and the labels
    dl->PushClipRect(p0, p1, true);
    if (state.down && state.selecting && state.moved && valid) {
        const ImVec2 a(std::min(state.x0, io.MousePos.x), std::min(state.y0, io.MousePos.y));
        const ImVec2 b(std::max(state.x0, io.MousePos.x), std::max(state.y0, io.MousePos.y));
        dl->AddRectFilled(a, b, IM_COL32(176, 141, 87, 40));
        dl->AddRect(a, b, IM_COL32(176, 141, 87, 220), 0.0f, 0, 1.0f);
    }
    const ImU32 ink = ImGui::GetColorU32(ImGuiCol_Text), shade = IM_COL32(0, 0, 0, 160);
    if (options.scale_bar) {
        if (const caliper_scene::ScaleBar bar = sc->scale_bar(size.x, size.y); bar.visible) {
            const float len = float(bar.pixels), m = 12.0f;
            const ImVec2 a(p0.x + m, p1.y - m), b(p0.x + m + len, p1.y - m);
            dl->AddLine(a, b, ink, 2.0f);
            dl->AddLine(a, ImVec2(a.x, a.y - 5.0f), ink, 2.0f);
            dl->AddLine(b, ImVec2(b.x, b.y - 5.0f), ink, 2.0f);
            dl->AddText(ImVec2(a.x, a.y - 6.0f - ImGui::GetTextLineHeight()), ink, bar.label.c_str());
        }
    }
    if (options.labels && sc->width() && sc->height()) {
        // The item labels and the annotations' labels (scene_overlays.h).
        const std::vector<caliper_scene::LabelAnchor> labels = sc->labels();
        SceneOverlayInput in;
        in.draw = dl;
        in.origin = p0;
        in.scale_x = size.x / float(sc->width());
        in.scale_y = size.y / float(sc->height());
        in.labels = &labels;
        in.ink = ink;
        in.shade = shade;
        draw_scene_overlays(in);
    }
    if (sc->highlighted()) {
        char t[48];
        std::snprintf(t, sizeof t, "%llu selected", (unsigned long long)sc->highlighted());
        const ImVec2 ts = ImGui::CalcTextSize(t);
        dl->AddText(ImVec2(p1.x - ts.x - 10.0f, p0.y + 8.0f), ink, t);
    }
    dl->PopClipRect();
    return SceneRectResult::drawn;
}

}  // namespace caliper_live_views
