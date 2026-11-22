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

#include <functional>
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

// ---- A scene drawn into a rectangle (a scene panel of a spec, design/instruments.md) -------------

// The options of a scene drawn into a rectangle: those of a spec's scene panel that the scene view
// takes. An empty `up` keeps the scene's own.
struct SceneRectOptions {
    std::string up;          // "+x", "-x", "+y", "-y", "+z" or "-z"; empty: the scene's own
    bool labels = true;      // the entities' labels over the picture
    bool scale_bar = false;  // a scale bar in the scene's units
};

// What a scene drawn into a rectangle keeps between frames: the gesture in progress, a click's
// pick in flight and a pick that a test seam asked for. The caller keeps one per rectangle.
struct SceneRectState {
    bool down = false;       // a button is held on the picture
    bool moved = false;      // the pointer moved past the drag threshold since the press
    bool selecting = false;  // the press was a shift-press: a rectangle selection
    int button = 0;          // the button held: 0 left, 1 right, 2 middle
    float x0 = 0, y0 = 0;    // where the press was, in screen pixels
    float lx = 0, ly = 0;    // the pointer at the last frame
    bool click = false;      // a click's pick is in flight, so its result is a pick event
    float seam_u = -1.0f;    // a pick asked for by pick_scene_rect(), in 0..1 of the rectangle; -1: none
    float seam_v = -1.0f;
    std::string up_applied;  // the option `up` that was applied, so that it is applied once
};

// Asks for a pick at (u, v), each from 0 to 1 across the rectangle, at the next draw, as if the
// person clicked there. Its result leaves through the sink as a pick event. The app's test seam
// uses it.
void pick_scene_rect(SceneRectState& state, float u, float v);

// What a pick or a selection made in the rectangle sends: `type` is "pick" or "select", `detail`
// the JSON text the scene view gives (live_scene.h: pick_detail(), select_polygon(); "{}" for a
// cleared selection).
using SceneRectSink = std::function<void(const char* type, const std::string& detail)>;

enum class SceneRectResult { drawn, missing, too_small };

// Draws the scene view `name` of the core into the rectangle at `p0` of `size` in the current
// ImGui window, which the caller clips. A drag orbits (pans with the right or middle button, or
// with alt or ctrl), the wheel zooms, a double click fits, a click picks, a shift-drag selects the
// items inside a rectangle and a shift-click clears the selection. Picks and selections leave
// through `sink`; hovers do not. Returns `missing` when the core has no scene view of that name,
// and draws nothing then. Frame thread only.
SceneRectResult draw_scene_in_rect(CaliperCore* core, const std::string& name, const char* imgui_id, ImVec2 p0,
                                   ImVec2 size, const SceneRectOptions& options, SceneRectState& state,
                                   const SceneRectSink& sink);

}  // namespace caliper_live_views
