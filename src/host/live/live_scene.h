// live_scene.h: the scene view of the shared view layer (kind 6, scene; design/sprint-4.md,
// increment 1; design/scenes.md, "Drawing a scene natively").
//
// A scene view owns one scene of the scene core (src/host/scene) and applies to it the decoded
// SCENE_UPDATE and SCENE_CLEAR messages of wire protocol 3.2 (docs/references/wire-protocol-3-2.md):
// the parts points, lines (segments or strips) and mesh, static items, and the reset that opens a
// scene's state for a new subscription. The arrays of a network message arrive as host bytes, so
// every array is uploaded into a buffer of the graphics layer that the view owns, which is one
// copy, reported as such: the info line says "copied" and the bytes, and each upload counts in the
// view's copy counters (embed/render_stats.h). A batch replaced by one of the same or a smaller
// size is written into the buffers it already has; a larger one gets new buffers, with a capacity
// rounded up to a power of two of at least 4 KiB, as web geometry's "Uploads and reuse" sets out.
//
// When something changed, or the camera moved, the view compiles the scene into a draw list of the
// graphics layer and submits it to an offscreen colour target with depth, which the view's window
// shows as an ImGui image; nothing is compiled or submitted while nothing changes. The camera is
// the scene core's orbit about the scene's up axis (pan and zoom in a 2-D scene), fitted once when
// the scene first has bounds and again when the person asks. In 3-D the person can switch the
// orbit between Y up and Z up (design/scenes.md, "The up-axis switch"), which changes only the
// camera, and the view measures the scale bar that the view layer draws over its picture
// (scene/scale_bar.h). The mode the person chose and the scale bar option are the settings the
// view layer keeps when the view closes and gives back when it opens again. A colormapped batch takes its
// colormap block from the view's own cache of key blocks (src/host/gfx/colormap_keys.h), derived
// from the CPU reference once per colormap and range.
//
// Every method runs on the frame thread. The view never waits for the GPU except in
// read_pixels(), which tests call.
#pragma once

#include "gfx/gfx.h"
#include "net/wire.h"
#include "scene/camera.h"
#include "scene/compile.h"
#include "scene/scale_bar.h"
#include "scene/scene.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace caliper_scene { class BackendResources; }
namespace caliper_gfx { namespace cmap { class KeyBlockCache; } }

namespace caliper_live_views {

// What the last drawn frame of a scene view cost.
struct SceneCost {
    double compile_ms = 0;     // the scene core's prepare and compile, CPU
    double submit_ms = 0;      // the layer's validation and encode of the submit, CPU
    double gpu_ms = -1;        // GPU time of the latest completed frames; < 0 when unknown
    uint32_t draws = 0;
    uint64_t frames = 0;       // frames compiled and submitted since the view opened
};

// The counters of a scene view (its info line, its tests and its debug line).
struct SceneCounts {
    uint32_t entities = 0;            // entities holding a batch
    uint32_t batches = 0;
    uint64_t items = 0;               // points, line segments and triangles
    uint64_t messages = 0;            // messages applied
    uint64_t bytes_last = 0;          // bytes copied by the latest round of updates
    uint64_t bytes_total = 0, copies_total = 0;
    uint64_t buffers_created = 0, buffers_reused = 0;
    uint64_t buffers_live = 0;
};

class SceneView {
public:
    // backend: borrowed, it outlives the view. programs: the backend's table of generated programs
    // (kMetalPrograms on Metal, kWebgpuPrograms in a browser).
    SceneView(caliper_gfx::Backend& backend, const caliper_gfx::ProgramDesc* programs);
    ~SceneView();
    SceneView(const SceneView&) = delete;
    SceneView& operator=(const SceneView&) = delete;

    bool init(std::string* why);

    // The scene's description, from its VIEWS entry. A new space empties the scene; a new up axis
    // turns an orbit the person has not moved, unless the person chose an up mode.
    void describe(const caliper_net::wire::SceneView& info);

    // One decoded message. False with *why when the scene store refused it; the view keeps what it
    // drew. A message's sections may be freed once the call returns.
    bool apply(const caliper_net::wire::SceneUpdate& u, std::string* why);
    bool apply(const caliper_net::wire::SceneClear& c, std::string* why);
    // Something the view ignored, as the versioning rules of protocol 3.2 say (a part, a mode or a
    // section that this increment does not build), for the info line.
    void note(const std::string& text);
    // Starts a new round of updates (the bytes of the info line count from here).
    void begin_round();

    // Camera input, in logical pixels of the view.
    void orbit(float dx, float dy);
    void pan(float dx, float dy, float view_height);
    void zoom(float wheel, float mx, float my, float view_w, float view_h);   // m: the pointer, -1 to 1
    void fit();

    // The up-axis switch, in a 3-D scene: the orbit keeps its target, its distance and its eye, and
    // takes its angles again about the new axis; the scene, its batches and the producer are not
    // touched. Looking through a pinhole the mode is kept, and it applies when the view returns to
    // its orbit. A mode the person chose wins over the scene's up until reset(). False in 2-D.
    bool switch_up(caliper_scene::Axis up);
    caliper_scene::Axis up_mode() const { return up_mode_; }
    bool up_chosen() const { return up_chosen_; }
    // The header's segments: Y up and Z up, then the scene's own axis or the mode when either is
    // another one. Empty in 2-D, which has no switch.
    std::vector<caliper_scene::Axis> up_choices() const;
    // The view as it opens: the scene's up, the default angles and the fit. The scale bar option stays.
    void reset();
    // The scale bar of a picture of w x h logical pixels (scene/scale_bar.h); hidden while the
    // option is off.
    caliper_scene::ScaleBar scale_bar(double w, double h) const;
    bool scale_bar_on() const { return scale_bar_on_; }
    void set_scale_bar(bool on) { scale_bar_on_ = on; }
    // What the view keeps of itself across a close and a reopen.
    struct Settings {
        bool up_chosen = false;
        caliper_scene::Axis up = caliper_scene::Axis::PosY;
        bool scale_bar = true;
    };
    Settings settings() const { return {up_chosen_, up_mode_, scale_bar_on_}; }
    // After describe(): a remembered mode wins over the scene's up.
    void restore(const Settings& s);

    // Compiles and submits when something changed since the last frame, into a target of
    // width x height physical pixels. False with *why when the layer refused the frame.
    bool render(uint32_t width, uint32_t height, float pixel_scale, uint32_t clear_rgba, std::string* why);
    // Picks up the GPU time of completed frames; call once per frame.
    void poll();
    bool has_target() const { return bool(color_); }
    uint64_t imgui_texture() const;
    caliper_gfx::Texture color() const { return color_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint64_t target_creates() const { return target_creates_; }
    // The colour target, tightly packed RGBA8 (tests; waits for the GPU).
    bool read_pixels(std::vector<uint8_t>* rgba, std::string* why);

    const caliper_scene::SceneTree& tree() const { return tree_; }
    const caliper_scene::ViewCamera& camera() const { return camera_; }
    caliper_scene::ViewCamera& camera() { return camera_; }
    bool fitted() const { return fitted_; }
    SceneCounts counts() const;
    const SceneCost& cost() const { return cost_; }
    const std::vector<std::string>& notes() const { return notes_; }
    // The header's info line: entities, items, the route and bytes copied, the GPU time.
    std::string info() const;
    // The camera in words (the UI driver's `where`; tests project with it).
    std::string camera_text() const;
    // The draws of the last compiled frame (tests): kept only while keep_draws is on.
    void keep_draws(bool on) { keep_draws_ = on; }
    const caliper_gfx::DrawList& last_draws() const { return last_list_; }
    const caliper_scene::DrawMap& last_map() const { return last_map_; }
    caliper_gfx::Backend& backend() { return gfx_; }

private:
    struct Upload { caliper_gfx::Buffer buffer; uint64_t capacity = 0; };
    // One layer (stamped or static) of one part of one entity: its batch and its buffers by role.
    struct Slot {
        caliper_scene::Batch batch;
        std::map<int, Upload> buffers;
        bool present = false;
    };
    struct Key {
        std::string path;
        int primitive = 0;
        bool fixed = false;
        bool operator<(const Key& o) const {
            if (path != o.path) return path < o.path;
            if (primitive != o.primitive) return primitive < o.primitive;
            return fixed < o.fixed;
        }
    };

    bool upload(Slot& s, caliper_scene::Role role, const void* bytes, uint64_t size, caliper_gfx::Buffer* out,
                std::string* why);
    void release(Slot& s);
    void rebuild(const std::string& path);    // the entity's batches again from the slots
    void reset_scene();
    void changed() { dirty_ = true; }

    caliper_gfx::Backend& gfx_;
    const caliper_gfx::ProgramDesc* programs_;
    std::shared_ptr<caliper_gfx::cmap::KeyBlockCache> colormaps_;   // the view's key blocks
    std::unique_ptr<caliper_scene::BackendResources> res_;
    caliper_scene::SceneTree tree_;
    caliper_scene::SceneCompiler compiler_;
    caliper_scene::ViewCamera camera_ = caliper_scene::Orbit{};
    std::map<Key, Slot> slots_;
    std::vector<uint8_t> scratch_;
    std::vector<std::string> notes_;
    uint64_t seq_ = 0;                 // the store's sequence numbers, one per applied batch
    bool fitted_ = false, moved_ = false, dirty_ = true;
    caliper_scene::Axis up_mode_ = caliper_scene::Axis::PosY;   // the orbit's axis, or the one it returns to
    bool up_chosen_ = false;           // the person switched: the scene's up no longer moves the view
    bool through_ = false;             // the last frame looked through a pinhole
    bool scale_bar_on_ = true;

    caliper_gfx::Texture color_, depth_;
    uint32_t width_ = 0, height_ = 0, clear_ = 0;
    float scale_ = 1.0f;
    uint64_t target_creates_ = 0;
    uint64_t drawn_revision_ = 0, drawn_pose_ = 0;

    SceneCounts counts_;
    SceneCost cost_;
    bool keep_draws_ = false;
    caliper_gfx::DrawList last_list_;
    caliper_scene::DrawMap last_map_;
};

}  // namespace caliper_live_views
