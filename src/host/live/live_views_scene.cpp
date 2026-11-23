// live_views_scene.cpp: the scene view in the typed live views (live_views_scene.h).
#include "live/live_views_scene.h"

#include "caliper/live_net.h"
#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "embed/render_stats.h"   // copy counters
#include "live/camera_drag.h"
#include "live/live_scene.h"
#include "live/scene_overlays.h"
#include "scene/pick.h"   // PickPass::Result
#if defined(CALIPER_HAVE_DRAW)
#include "draw/draw_runtime.h"
#include <nlohmann/json.hpp>
#endif
#include "renderer/host_renderer.h"

#if defined(__EMSCRIPTEN__)
#include "gfx/shaders/generated/programs_webgpu.h"
#include "renderer/webgpu_renderer.h"
#elif defined(__APPLE__)
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#endif

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
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

namespace {

// Asks the entities' pick pass and, while the view draws programs, the view's frame pick of its
// ID target for the pixel (x, y); true when either was asked.
bool ask_picks(SceneView& view, SceneRectState& state, float x, float y) {
    std::string why;
    state.entity_pending = view.request_pick(x, y, &why);
    state.entity_dropped = false;
    state.entity_detail.clear();
    state.program_detail.clear();
    state.program_ticket = view.request_program_pick(x, y, &why);
    state.frame_pending = state.program_ticket != 0;
    return state.entity_pending || state.frame_pending;
}

// Takes the frame pick once it answered: the item of the program whose pick id the texel holds,
// read from the version that the program showed in the frame that the readback read. A pick id
// that names no program, or an item outside its item map's valid count, gives nothing.
void poll_program_pick(SceneView& view, SceneRectState& state) {
#if defined(CALIPER_HAVE_DRAW)
    caliper_draw::PickedItem picked;
    float depth = 1.0f;
    const caliper_scene::PickPass::Result r = view.poll_program_pick(state.program_ticket, &picked, &depth);
    if (r == caliper_scene::PickPass::Result::Pending) return;
    state.frame_pending = false;
    state.program_ticket = 0;
    if (r != caliper_scene::PickPass::Result::Hit) return;
    state.program_detail = program_pick_detail(picked);
    state.program_depth = depth;
#else
    (void)view;
    state.frame_pending = false;
    state.program_ticket = 0;
#endif
}

// A state whose picks in flight belong to another view, which a detach, a new attachment or a
// patched panel replaced: they are forgotten, and a seam pick is asked again in this view.
void adopt_view(const SceneView& view, SceneRectState& state) {
    if (state.view == view.serial()) return;
    state.view = view.serial();
    state.entity_pending = state.frame_pending = state.entity_dropped = false;
    state.program_ticket = 0;
    state.entity_detail.clear();
    state.program_detail.clear();
    state.click = false;
    state.seam_asked = false;
}

// The labels of the programs' text tables at the view's size: each anchor projected with the
// programs' camera (clip space for a piece whose camera is none), those in front of the eye and
// inside the picture, the higher priority first, at most kMaxLabels.
std::vector<caliper_scene::LabelAnchor> program_labels(const SceneView& view) {
    std::vector<caliper_scene::LabelAnchor> out;
#if defined(CALIPER_HAVE_DRAW)
    struct Ranked {
        float priority;
        caliper_scene::LabelAnchor anchor;
    };
    std::vector<Ranked> all;
    const caliper_scene::Mat4 identity;
    for (const caliper_draw::DrawRuntime* rt : view.draw_runtimes()) {
        const bool clip_space = rt->camera() == caliper_builder::spec::CameraMode::none;
        const caliper_scene::Mat4& clip = clip_space ? identity : view.program_clip();
        for (caliper_draw::Label& l : rt->labels()) {
            double h[4];
            caliper_scene::transform_h(clip, {l.position[0], l.position[1], l.position[2]}, h);
            if (!(h[3] > 0)) continue;
            const double nx = h[0] / h[3], ny = h[1] / h[3];
            if (!std::isfinite(nx) || !std::isfinite(ny) || std::fabs(nx) > 1.0 || std::fabs(ny) > 1.0) continue;
            caliper_scene::LabelAnchor a;
            a.x = float((nx + 1.0) * 0.5 * view.width());
            a.y = float((1.0 - ny) * 0.5 * view.height());
            a.depth = float(h[3]);
            a.text = std::move(l.text);
            all.push_back({l.priority, std::move(a)});
        }
    }
    std::stable_sort(all.begin(), all.end(), [](const Ranked& a, const Ranked& b) { return a.priority > b.priority; });
    for (std::size_t i = 0; i < all.size() && i < caliper_host::kMaxLabels; ++i) out.push_back(std::move(all[i].anchor));
#else
    (void)view;
#endif
    return out;
}

// The axes mark of a 3-D scene in the corner whose bottom-right is `corner`: an arrow for each of
// x, y and z from one centre, as the camera's rotation turns it, the farthest drawn first, each
// with its letter at its tip. Returns the rectangle it takes; an empty one in a 2-D scene or
// before the view has a camera.
caliper_host::LabelRect draw_axes_mark(ImDrawList* dl, const SceneView& view, ImVec2 corner) {
    const caliper_scene::ViewCamera cam = view.effective_camera();
    if (std::holds_alternative<caliper_scene::PanZoom>(cam) || !view.width() || !view.height()) return {};
    const caliper_scene::CameraMatrices m = caliper_scene::camera_matrices(cam, view.tree(), view.width(), view.height());
    if (!m.ok) return {};
    const float font = ImGui::GetFontSize();
    const float len = std::round(1.6f * font), margin = std::round(0.8f * font);
    const ImVec2 c(corner.x - margin - len - font * 0.5f, corner.y - margin - len - font * 0.5f);
    struct Arrow {
        float dx, dy, depth;
        const char* name;
        ImU32 color;
    };
    Arrow arrows[3] = {{0, 0, 0, "x", IM_COL32(222, 92, 84, 255)},
                       {0, 0, 0, "y", IM_COL32(112, 186, 104, 255)},
                       {0, 0, 0, "z", IM_COL32(96, 146, 232, 255)}};
    for (int i = 0; i < 3; ++i) {
        // Column i of the view's rotation: where world axis i points on the screen (y down there).
        arrows[i].dx = float(m.view.at(0, i));
        arrows[i].dy = -float(m.view.at(1, i));
        arrows[i].depth = float(m.view.at(2, i));
        if (!std::isfinite(arrows[i].dx) || !std::isfinite(arrows[i].dy) || !std::isfinite(arrows[i].depth)) return {};
    }
    std::sort(std::begin(arrows), std::end(arrows), [](const Arrow& a, const Arrow& b) { return a.depth < b.depth; });
    const ImU32 shade = IM_COL32(0, 0, 0, 140);
    dl->AddCircleFilled(c, len + font * 0.5f, IM_COL32(0, 0, 0, 70));
    for (const Arrow& a : arrows) {
        const ImVec2 tip(c.x + a.dx * len, c.y + a.dy * len);
        dl->AddLine(c, tip, shade, 4.0f);
        dl->AddLine(c, tip, a.color, 2.0f);
        const ImVec2 ts = ImGui::CalcTextSize(a.name);
        const ImVec2 at(std::round(c.x + a.dx * (len + font * 0.45f) - ts.x * 0.5f),
                        std::round(c.y + a.dy * (len + font * 0.45f) - ts.y * 0.5f));
        dl->AddText(ImVec2(at.x + 1.0f, at.y + 1.0f), shade, a.name);
        dl->AddText(at, a.color, a.name);
    }
    const float r = len + font;
    return {c.x - r, c.y - r, c.x + r, c.y + r};
}

}  // namespace

bool scene_path_matches(const std::string& pattern, const std::string& path) {
    const auto split = [](const std::string& s) {
        std::vector<std::string> parts;
        std::size_t at = 0;
        while (parts.size() <= 16) {
            const std::size_t next = s.find('/', at);
            parts.push_back(s.substr(at, next == std::string::npos ? std::string::npos : next - at));
            if (next == std::string::npos) break;
            at = next + 1;
        }
        return parts;
    };
    const std::vector<std::string> p = split(pattern), q = split(path);
    if (p.size() > 16 || q.size() > 16 || pattern.empty() || path.empty()) return false;
    // match[i][j]: the first i segments of the pattern match the first j of the path.
    std::vector<std::vector<char>> match(p.size() + 1, std::vector<char>(q.size() + 1, 0));
    match[0][0] = 1;
    for (std::size_t i = 1; i <= p.size(); ++i)
        for (std::size_t j = 0; j <= q.size(); ++j) {
            if (p[i - 1] == "**")
                match[i][j] = match[i - 1][j] || (j > 0 && match[i][j - 1]);
            else if (j > 0)
                match[i][j] = match[i - 1][j - 1] && (p[i - 1] == "*" || p[i - 1] == q[j - 1]);
        }
    return match[p.size()][q.size()] != 0;
}

std::string program_pick_detail(const caliper_draw::PickedItem& picked) {
#if defined(CALIPER_HAVE_DRAW)
    nlohmann::json fields = nlohmann::json::object();
    for (const auto& [name, value] : picked.fields) fields[name] = value;
    const nlohmann::json detail = {{"program", picked.program}, {"item", picked.item}, {"step", picked.step},
                                   {"fields", fields},
                                   {"index", {picked.index[0], picked.index[1], picked.index[2], picked.index[3]}}};
    return detail.dump();
#else
    (void)picked;
    return {};
#endif
}

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
    adopt_view(view, state);
    state.rect[0] = p0.x, state.rect[1] = p0.y, state.rect[2] = size.x, state.rect[3] = size.y;
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
    if (options.view_camera && options.view_camera_key != state.camera_applied) {
        sc->set_starting(*options.view_camera);
        state.camera_applied = options.view_camera_key;
    }
    sc->set_scale_bar(options.scale_bar);
    // The panel's include and exclude, and its programs (draw/draw_host.h): set before each
    // render, because a panel and the app's window can draw one view.
    std::string filter_key;
    std::function<bool(const std::string&)> keep;
    if (!options.include.empty() || !options.exclude.empty()) {
        for (const std::string& pattern : options.include) filter_key += "+" + pattern + "\x1f";
        for (const std::string& pattern : options.exclude) filter_key += "-" + pattern + "\x1f";
        keep = [inc = options.include, exc = options.exclude](const std::string& path) {
            bool in = inc.empty();
            for (const std::string& pattern : inc) in = in || scene_path_matches(pattern, path);
            if (!in) return false;
            for (const std::string& pattern : exc)
                if (scene_path_matches(pattern, path)) return false;
            return true;
        };
    }
    sc->set_entity_filter(filter_key, std::move(keep));
    sc->set_draw_feed(options.draw_feed);
    sc->set_stage_source(options.stages);
    sc->poll();   // the GPU time of completed frames: the info line and the stages' time watchdog

    // ---- input: an invisible button owns the rectangle, so that a drag never moves the window. A
    // control over the picture goes in a child window of its own, as the spec's controls do: that
    // window takes the presses over it, which then move no camera. (An item that allows overlap
    // would lose a press that comes in the same frame as the pointer, as after a click into a
    // window that saw no cursor moves.)
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
        state.max_d2 = 0.0f;
        state.press_camera = state.step_camera = sc->camera();
        state.press_moved = sc->moved();
        state.restorable = true;
    }
    bool released = false;
    if (state.down) {
        // A selection's rectangle starts past ImGui's drag threshold. A camera drag moves the camera
        // from the first point of motion, and the release decides whether it was a click: the pointer
        // stayed within kCameraClickTolerance points of the press all along (camera_drag.h).
        if (valid) {
            const float ox = io.MousePos.x - state.x0, oy = io.MousePos.y - state.y0;
            state.max_d2 = std::max(state.max_d2, ox * ox + oy * oy);
            state.moved = state.selecting ? state.moved || std::fabs(ox) > io.MouseDragThreshold ||
                                                std::fabs(oy) > io.MouseDragThreshold
                                          : !camera_press_is_click(state.max_d2);
        }
        if (!state.selecting && valid) {
            // A fit, a zoom or camera.set since the last step: the camera saved at the press stays away.
            if (!same_camera(sc->camera(), state.step_camera)) state.restorable = false;
            const float dx = io.MousePos.x - state.lx, dy = io.MousePos.y - state.ly;
            state.lx = io.MousePos.x;
            state.ly = io.MousePos.y;
            if (dx != 0.0f || dy != 0.0f) {
                if (state.button != 0 || io.KeyAlt || io.KeyCtrl) sc->pan(dx, dy, size.y);
                else sc->orbit(dx, dy);
            }
            state.step_camera = sc->camera();
        }
        released = !ImGui::IsMouseDown(state.button);
        // A click puts the camera back as it was at the press, before this frame draws, so that the
        // click's wobble is all that was seen to move.
        if (released && !state.selecting && !state.moved && state.restorable)
            sc->restore_camera(state.press_camera, state.press_moved);
    }
    if (hovered && io.MouseWheel != 0.0f) {
        const float mx = (io.MousePos.x - p0.x) / size.x * 2.0f - 1.0f;
        const float my = 1.0f - (io.MousePos.y - p0.y) / size.y * 2.0f;
        sc->zoom(io.MouseWheel, mx, my, size.x, size.y);
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        sc->fit();
        if (state.down) {   // the second press of a double click: its release keeps the fit
            state.press_camera = state.step_camera = sc->camera();
            state.press_moved = sc->moved();
        }
    }
    if (sc->width() && sc->height()) {   // the pointer of the programs' prelude, in physical pixels
        const uint32_t buttons = (io.MouseDown[0] ? 1u : 0u) | (io.MouseDown[1] ? 2u : 0u) | (io.MouseDown[2] ? 4u : 0u);
        sc->set_pointer((io.MousePos.x - p0.x) * float(sc->width()) / size.x,
                        (io.MousePos.y - p0.y) * float(sc->height()) / size.y, hovered ? buttons : 0u, hovered && valid);
    }

    // ---- GPU: the scene core compiles and submits when something changed
    const uint32_t clear = ImGui::GetColorU32(ImGuiCol_WindowBg) | 0xff000000u;
    std::string why;
    if (!sc->render(uint32_t(want_w), uint32_t(want_h), scale, clear, &why) && !why.empty()) sc->note(why);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + size.x, p0.y + size.y);
    if (const uint64_t tex = sc->imgui_texture()) dl->AddImage((ImTextureID)tex, p0, p1);
    std::vector<caliper_host::LabelRect> keep_clear;   // the view's own texts, which no label covers
    if (!options.stage_message.empty()) {   // a stage's refusal or stop, over the picture's bottom
        const float pad = 4.0f, h = ImGui::GetTextLineHeight() + 2 * pad;
        const ImVec2 a(p0.x, std::max(p0.y, p1.y - h));
        dl->AddRectFilled(a, p1, IM_COL32(0, 0, 0, 170));
        keep_clear.push_back({a.x, a.y, p1.x, p1.y});
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
            } else {   // the item under the press, with the camera of the press
                const ImVec2 t = target_px(state.x0, state.y0);
                state.click = ask_picks(*sc, state, t.x, t.y) || state.click;
            }
        }
        state.down = false;
        state.selecting = false;
    }
    // A seam pick stays asked until its own request gives a hit or a miss: a
    // live scene that changes before the pick's readback drops the result, and
    // the pick is then asked again at the next draw.
    if (state.seam_u >= 0.0f && !state.click && sc->width() && sc->height()) {
        state.seam_asked = ask_picks(*sc, state, state.seam_u * float(sc->width()), state.seam_v * float(sc->height()));
        state.click = state.seam_asked;
    }
    // The entities' pick pass and the programs' frame pick answer apart; the pick goes out once
    // both did, the program's item unless the entity's hit is nearer.
    caliper_scene::Hit hit;
    const caliper_scene::PickPass::Result pr = sc->poll_pick(&hit);
    if (state.entity_pending && (pr == caliper_scene::PickPass::Result::Hit ||
                                 pr == caliper_scene::PickPass::Result::Miss ||
                                 pr == caliper_scene::PickPass::Result::Dropped)) {
        state.entity_pending = false;
        state.entity_dropped = pr == caliper_scene::PickPass::Result::Dropped;
        if (pr == caliper_scene::PickPass::Result::Hit) {
            state.entity_detail = sc->pick_detail(hit);
            state.entity_depth = hit.depth;
        }
    }
    if (state.frame_pending) poll_program_pick(*sc, state);
    if (state.click && !state.entity_pending && !state.frame_pending) {
        const bool program = !state.program_detail.empty() &&
                             (state.entity_detail.empty() || !(state.entity_depth < state.program_depth));
        const std::string& detail = program ? state.program_detail : state.entity_detail;
        if (!detail.empty() && sink) sink("pick", detail);
        const bool answered = !detail.empty() || !state.entity_dropped;
        if (state.seam_asked && answered) state.seam_u = state.seam_v = -1.0f;
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
            const ImVec2 at(a.x, a.y - 6.0f - ImGui::GetTextLineHeight());
            dl->AddText(at, ink, bar.label.c_str());
            const float label_w = ImGui::CalcTextSize(bar.label.c_str()).x;
            keep_clear.push_back({a.x - 2.0f, at.y - 2.0f, std::max(b.x, at.x + label_w) + 2.0f, a.y + 2.0f});
        }
    }
    if (options.axes) {
        const float bottom = keep_clear.empty() || options.stage_message.empty() ? p1.y : keep_clear.front().y0;
        if (const caliper_host::LabelRect mark = draw_axes_mark(dl, *sc, ImVec2(p1.x, bottom)); mark.x1 > mark.x0)
            keep_clear.push_back(mark);
    }
    // The selection count at the top right, placed before the labels so that none covers it.
    char count_text[48] = {};
    ImVec2 count_at;
    if (sc->highlighted()) {
        std::snprintf(count_text, sizeof count_text, "%llu selected", (unsigned long long)sc->highlighted());
        const ImVec2 ts = ImGui::CalcTextSize(count_text);
        count_at = ImVec2(p1.x - ts.x - 10.0f, p0.y + 8.0f);
        keep_clear.push_back({count_at.x - 2.0f, count_at.y - 2.0f, count_at.x + ts.x + 2.0f, count_at.y + ts.y + 2.0f});
    }
    if (options.labels && sc->width() && sc->height()) {
        // The annotations' labels, then the programs' labels by their priority, then the item
        // labels nearest first (scene_overlays.h).
        std::vector<caliper_scene::LabelAnchor> labels, items;
        std::vector<bool> annotations, rings;
        for (caliper_scene::LabelAnchor& a : sc->labels()) {
            const std::string& path = sc->tree().path(a.entity);
            if (path.rfind("spec-overlay/", 0) != 0) {
                items.push_back(std::move(a));
                continue;
            }
            rings.push_back(path.rfind("spec-overlay/marker/", 0) == 0);
            labels.push_back(std::move(a));
        }
        std::vector<int> priorities;
        for (std::size_t i = 0; i < labels.size(); ++i) priorities.push_back(int(i));
        for (caliper_scene::LabelAnchor& a : program_labels(*sc)) {
            priorities.push_back(int(caliper_host::kMaxLabels + labels.size()));
            labels.push_back(std::move(a));
        }
        annotations.assign(rings.size(), true);
        for (caliper_scene::LabelAnchor& a : items) {
            priorities.push_back(int(2 * caliper_host::kMaxLabels + labels.size()));
            labels.push_back(std::move(a));
        }
        annotations.resize(labels.size(), false);
        rings.resize(labels.size(), false);
        SceneOverlayInput in;
        in.annotations = &annotations;
        in.rings = &rings;
        in.priorities = &priorities;
        in.accent = IM_COL32(176, 141, 87, 255);
        in.draw = dl;
        in.origin = p0;
        in.scale_x = size.x / float(sc->width());
        in.scale_y = size.y / float(sc->height());
        in.labels = &labels;
        in.ink = ink;
        in.shade = shade;
        in.keep_clear = &keep_clear;
        draw_scene_overlays(in);
    }
    if (count_text[0]) dl->AddText(count_at, ink, count_text);
    dl->PopClipRect();
    return SceneRectResult::drawn;
}

}  // namespace caliper_live_views
