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

void scene_note(CaliperCore* core, const std::string& name, const std::string& text) {
    View* v = core ? find_scene(core, name) : nullptr;
    if (SceneView* sv = v ? scene_view(*v) : nullptr) sv->note(text);
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
