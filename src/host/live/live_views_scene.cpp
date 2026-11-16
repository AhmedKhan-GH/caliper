// live_views_scene.cpp: the scene view in the typed live views (live_views_scene.h).
#include "live/live_views_scene.h"

#include "caliper/live_net.h"
#include "caliper/live_views.h"
#include "embed/embed_core.h"
#include "embed/render_stats.h"   // copy counters
#include "live/live_scene.h"
#include "renderer/host_renderer.h"

#if defined(__EMSCRIPTEN__)
#include "gfx/shaders/generated/programs_webgpu.h"
#include "renderer/webgpu_renderer.h"
#elif defined(__APPLE__)
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#endif

#include <algorithm>

namespace caliper_live_views {

static_assert(kKindScene == CALIPER_NET_KIND_SCENE, "the scene kind of the view registry and of protocol 3.2");

SceneHolder::~SceneHolder() {
    view.reset();   // its resources go back to the layer before the layer goes
    gfx.reset();
}

namespace {

// The layer's backend for a new scene view and the backend's table of programs. On Metal each
// scene view owns a backend of its own on the renderer's device and queue, so that its submits are
// ordered with the renderer's frame by commit order and its statistics (the info line's GPU time,
// the frame's encode time) are its own. In a browser the view borrows the WebGPU renderer's
// backend, which a module instance's views share (an element mounts one view per instance).
bool make_backend(CaliperCore* core, SceneHolder* h, caliper_gfx::Backend** out,
                  const caliper_gfx::ProgramDesc** programs) {
    caliper_host::HostRenderer* r = core->renderer.get();
#if defined(__EMSCRIPTEN__)
    if (caliper_gfx::Backend* b = r ? static_cast<caliper_host::WebGpuRenderer*>(r)->gfx() : nullptr) {
        *out = b;
        *programs = caliper_gfx::shaders::kWebgpuPrograms;
        return true;
    }
    h->notice = "scenes need the graphics layer's WebGPU backend, which has not started";
#elif defined(__APPLE__)
    if (r && r->native_device() && r->native_queue()) {
        h->gfx = caliper_gfx::make_metal_backend(r->native_device(), r->native_queue());
        if (h->gfx) {
            *out = h->gfx.get();
            *programs = caliper_gfx::shaders::kMetalPrograms;
            return true;
        }
    }
    h->notice = "scenes need the Metal renderer";
#else
    (void)r;
    h->notice = "scenes need the Metal renderer or a browser with WebGPU";
#endif
    (void)out;
    (void)programs;
    return false;
}

CaliperLiveViewsState* views_state(CaliperCore* core) {
    if (!core->live_views) core->live_views = new CaliperLiveViewsState();
    return core->live_views;
}

View* find_scene(CaliperCore* core, const std::string& name) {
    View* v = find(core, name.c_str());
    return v && v->kind == kKindScene ? v : nullptr;
}

}  // namespace

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
        if (make_backend(core, h.get(), &gfx, &programs)) {
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
    if (SceneView* sv = scene_view(*v)) sv->describe(info);
    return true;
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

}  // namespace caliper_live_views
