// live_views_scene.h: the scene view (live_scene.h) as a typed live view of kind 6, scene.
//
// A source of scene messages (the app's NetSource, the browser's) does not go through
// caliper_core_live_apply, whose snapshots a scene does not have: it opens the view with the
// scene's description from VIEWS and hands it each decoded scene message. The view
// then lives in the core's typed live views like any other: it is laid out on the grid, drawn by
// caliper_live_views_frame with the other windows, removed with caliper_core_live_view_remove, and
// caliper_core_live_view_info and _readback answer for it. Frame thread only.
#pragma once

#include "gfx/gfx.h"
#include "live/live_views_state.h"

#include <map>
#include <memory>
#include <vector>
#include <string>

namespace caliper_net::wire {
struct SceneView;
struct SceneUpdate;
struct SceneClear;
struct SceneTransforms;
struct SceneClasses;
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
bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneTransforms& t,
                 std::string* why);
bool scene_apply(CaliperCore* core, const std::string& name, const caliper_net::wire::SceneClasses& c,
                 std::string* why);
// Something the source ignored for this view (a part, mode or section not built), for its info line.
void scene_note(CaliperCore* core, const std::string& name, const std::string& text);
// The linked selections of the item spaces (wire protocol 3.1) for the scene view `name`: the items of
// every batch whose rows are items of a space are highlighted; spaces absent are cleared.
void scene_link(CaliperCore* core, const std::string& name, const std::map<std::string, std::vector<int64_t>>& by_space);
// A new round of messages: the info line's bytes count from here.
void scene_begin_round(CaliperCore* core, const std::string& name);

// The graphics layer's backend for a new view that draws through the scene core (a scene view, a
// points view) and the backend's table of generated programs. On Metal the view owns a backend of
// its own on the renderer's device and queue (*owned), so that its submits are ordered with the
// renderer's frame by commit order and its statistics are its own; in a browser it borrows the
// WebGPU renderer's (*owned stays null). False with *notice, which names `what` ("scenes",
// "points"), when no backend can start here.
bool open_layer(CaliperCore* core, const char* what, std::unique_ptr<caliper_gfx::Backend>* owned,
                caliper_gfx::Backend** gfx, const caliper_gfx::ProgramDesc** programs, std::string* notice);

// Keeps the view's settings (its up mode and scale bar option) under its name, before it goes;
// scene_open gives them back to a scene view of that name.
void scene_remember(CaliperCore* core, const View& v);

// The scene view of a live view, or null.
SceneView* scene_view(CaliperCore* core, const char* name);
SceneView* scene_view(View& v);
const SceneView* scene_view(const View& v);

// live_views_draw.cpp: the window of a scene view.
void draw_scene(CaliperCore* core, CaliperLiveViewsState* S, View& v);

}  // namespace caliper_live_views
