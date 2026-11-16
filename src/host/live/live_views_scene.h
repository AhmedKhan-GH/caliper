// live_views_scene.h: the scene view (live_scene.h) as a typed live view of kind 6, scene.
//
// A source of scene messages (the app's NetSource, the browser's) does not go through
// caliper_core_live_apply, whose snapshots a scene does not have: it opens the view with the
// scene's description from VIEWS and hands it each decoded SCENE_UPDATE and SCENE_CLEAR. The view
// then lives in the core's typed live views like any other: it is laid out on the grid, drawn by
// caliper_live_views_frame with the other windows, removed with caliper_core_live_view_remove, and
// caliper_core_live_view_info and _readback answer for it. Frame thread only.
#pragma once

#include "gfx/gfx.h"
#include "live/live_views_state.h"

#include <memory>
#include <string>

namespace caliper_net::wire {
struct SceneView;
struct SceneUpdate;
struct SceneClear;
}  // namespace caliper_net::wire

namespace caliper_live_views {

class SceneView;

// What a live view of kind scene holds: the backend of the graphics layer it draws through, when
// it owns one, and its scene view. Without a backend (a renderer the layer cannot run on) the view
// shows `notice` in its place and keeps no state.
struct SceneHolder {
    std::unique_ptr<caliper_gfx::Backend> gfx;   // owned; null when the view borrows the renderer's
    std::unique_ptr<SceneView> view;
    std::string notice;
    ~SceneHolder();
};

// Opens the scene view `name` (kind scene) with the scene's description, or describes an open one
// again. A view of another kind with that name is replaced; a name an embed v1.2 view holds is
// refused.
bool scene_open(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneView& info,
                std::string* why);
// A message for the open scene view `name`. False with *why when there is no such view or the
// scene store refused it.
bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneUpdate& u,
                 std::string* why);
bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneClear& c,
                 std::string* why);
// Something the source ignored for this view (a part, mode or section not built), for its info line.
void scene_note(CaliperCore* core, const std::string& name, const std::string& text);
// A new round of messages: the info line's bytes count from here.
void scene_begin_round(CaliperCore* core, const std::string& name);

// The scene view of a live view, or null.
SceneView* scene_view(CaliperCore* core, const char* name);
SceneView* scene_view(View& v);
const SceneView* scene_view(const View& v);

// live_views_draw.cpp: the window of a scene view.
void draw_scene(CaliperCore* core, CaliperLiveViewsState* S, View& v);

}  // namespace caliper_live_views
