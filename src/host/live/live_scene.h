// live_scene.h: the scene view of the shared view layer (kind 6, scene; design/sprint-4.md,
// increment 1; design/scenes.md, "Drawing a scene natively").
//
// A scene view owns one scene of the scene core (src/host/scene) and applies to it the decoded
// scene messages of wire protocol 3.2 (docs/reference/wire-protocol-3-2.md): SCENE_UPDATE with the
// parts points, lines (segments, strips and index-pair edges), arrows, boxes, mesh (with normals,
// or UVs and a texture), image (colour, grey, depth and segmentation) and camera; SCENE_TRANSFORMS;
// SCENE_CLASSES; SCENE_CLEAR of named parts or all, static items, and the reset that opens a scene's
// state for a new subscription. The arrays of a network message arrive as host bytes, so every
// array is uploaded into a buffer of the graphics layer that the view owns, and every image into a
// texture it owns (a u16 depth scaled to f32 scene units and class ids widened to u32 on the way,
// as the texture's format needs), which is one copy, reported as such: the info line says
// "copied" and the bytes, and each upload counts in the view's copy counters
// (embed/render_stats.h); so does the class table texture, 256 KiB each time the table changes. A
// batch replaced by one of the same or a smaller size is written into the buffers it already has;
// a larger one gets new buffers, with a capacity rounded up to a power of two of at least 4 KiB,
// as web geometry's "Uploads and reuse" sets out, and an image keeps its texture while its size
// and format stay. Encoded (JPEG or PNG) images are not drawn yet.
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
#include "scene/pick.h"
#include "scene/scale_bar.h"
#include "scene/scene.h"
#include "scene/stage_cache.h"

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
    uint64_t textures_created = 0, textures_reused = 0;   // images and textured meshes
    uint64_t textures_live = 0;
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
    bool apply(const caliper_net::wire::SceneTransforms& t, std::string* why);
    bool apply(const caliper_net::wire::SceneClear& c, std::string* why);
    bool apply(const caliper_net::wire::SceneClasses& c, std::string* why);
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
    // Looks through the camera of entity `path` (its pose and intrinsics, its pixel space flat over
    // the picture), or returns to the view's own camera with "". False when the entity holds no
    // camera. The view keeps the path: while the entity holds no camera (a clear, a new state that
    // has not sent it yet) it shows its own camera and says so. A fit returns to the orbit.
    bool look_through(const std::string& path, std::string* why);
    const std::string& looking_through() const { return through_; }
    // The entities that hold a camera, in path order.
    std::vector<std::string> cameras() const;
    // The camera the view draws with: the pinhole looked through, or its own.
    caliper_scene::ViewCamera effective_camera() const;
    // The anchors of the labels at the view's size and camera, nearest first, at most 512.
    std::vector<caliper_scene::LabelAnchor> labels() const;

    // ---- picking (design/scenes.md, "Picking, hover and selection") --------------------------------
    // Asks for a pick at (x, y), in physical pixels of the view's target, through the scene core's
    // pick pass: the pick variants of the drawn scene into a 16 x 16 window around the pointer. The
    // answer comes from poll_pick a frame or two later; a result whose frame the view replaced since,
    // or that a newer request superseded, is dropped. Nothing is submitted while nobody asks.
    bool request_pick(float x, float y, std::string* why);
    caliper_scene::PickPass::Result poll_pick(caliper_scene::Hit* out);
    bool pick_pending() const { return pick_ && pick_->pending(); }
    const caliper_scene::PickPass* pick_pass() const { return pick_.get(); }
    // The frames the view has drawn: the generation a pick belongs to.
    uint64_t generation() const { return generation_; }
    // The detail of a pick or hover event for a hit, as protocol 3.2 gives it: entity, primitive,
    // index (the item in the sent batch), row_id (its source index through the batch's payload axes)
    // and, where the depth gave one, position. "" for a hit on something that is no batch item (a
    // camera's frustum or an entity's axes): a hover there reports nothing.
    std::string pick_detail(const caliper_scene::Hit& h) const;
    // The source index of item `index` of the batch (path, primitive), through its payload axes.
    int64_t row_id(const std::string& path, caliper_scene::Primitive p, int64_t index) const;

    // ---- selections ------------------------------------------------------------------------------
    // Selects the points, and the boxes by their centres, whose projection at the view's camera lies
    // inside the polygon `poly` (x, y pairs in physical pixels of the target, by the even-odd rule),
    // from the CPU copies of their arrays, as the points view selects; occluded items are selected
    // too. It replaces the view's selection and returns the select event's detail: `entities`, each
    // an entity, its primitive, `indices` and `row_ids` (at most 32,768 indices in all, `truncated`
    // when cut), and `space` and `items` when the selected batches map an item space (the first one's,
    // when they map several). "{}" when nothing is inside.
    std::string select_polygon(const std::vector<float>& poly);
    // Clears the view's own selection; true when there was one.
    bool clear_selection();
    // A selection of item space `space` (linked selection, wire protocol 3.1): the items of every
    // batch whose rows are items of that space are highlighted. Empty items clear it.
    void set_linked(const std::string& space, std::vector<int64_t> items);
    // Every linked selection at once (item space to items): spaces absent from `by_space` are cleared.
    void set_all_linked(const std::map<std::string, std::vector<int64_t>>& by_space);
    // The view's own selection: the items selected, in all.
    uint64_t selected_count() const;
    // While anything is selected or linked, the scene is drawn dimmed towards the background (as the
    // points view dims) and the selected points and boxes over it in their own colours. The number of
    // items drawn so now.
    uint64_t highlighted() const { return highlighted_; }

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
    // The view as it opens: the scene's up, the default angles and the fit, on its own camera (a camera
    // looked through is left). The scale bar option stays.
    void reset();
    // The scale bar of a picture of w x h logical pixels (scene/scale_bar.h) at the camera the view
    // draws with; hidden while the option is off and while it looks through a pinhole.
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

    // The stages of the panel that shows the view (scene/stage_cache.h, design/gpu-programs.md):
    // the points and lines of an entity with a stage draw its program, in the colour pass and the
    // pick pass, and the highlight of a selection draws the selected items with them. Null draws
    // the built-in programs. The panel sets it before each render; a new source, a new revision of
    // it, or a stage that reads the time draws the view again. A rectangle selection projects the
    // batches' own positions on the CPU, which a vertex stage does not move.
    void set_stage_source(caliper_scene::StageSource* source);
    // The draws of the last frame that used a stage program.
    uint32_t staged_draws() const { return staged_; }
    // The step the stages read: the first stamp of the latest scene update that carried one.
    int64_t stage_step() const { return stamp_step_; }

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
        caliper_gfx::Texture texture;   // an image's, or a textured mesh's
        uint32_t tex_w = 0, tex_h = 0;
        caliper_gfx::Format tex_format = caliper_gfx::Format::RGBA8Unorm;
        std::vector<uint8_t> cpu_rows, cpu_ids;   // CPU views for the labels: positions or centres, class ids
        std::map<int, std::vector<uint8_t>> cpu;    // points and boxes: every array as it arrived, by role
        // The payload axis of the batch's item dim, when the producer sampled it (the update's `axes`),
        // and the item space its rows are (`items`).
        int64_t ax_start = 0, ax_step = 1, ax_count = 0, ax_size = 0;
        std::string items;
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
    bool upload_texture(Slot& s, caliper_gfx::Format f, uint32_t w, uint32_t h, const void* data,
                        uint32_t row_pitch, std::string* why);
    void release_texture(Slot& s);
    void release(Slot& s);
    bool in_pixel_space(caliper_scene::EntityId e) const;   // under a camera, from the parts now
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
    std::string through_;              // the camera entity looked through, or ""
    std::unique_ptr<caliper_scene::PickPass> pick_;
    uint64_t generation_ = 0;          // colour frames submitted
    // The panel's stages: the programs on this view's backend, the source's revision drawn last,
    // the stage draws of the last frame and the step from the stamps.
    std::unique_ptr<caliper_scene::StageCache> stages_;
    uint64_t stage_revision_ = 0;
    uint32_t staged_ = 0;
    int64_t stamp_step_ = 0;
    caliper_scene::CompileOptions stage_options(caliper_scene::CompileOptions opt) const;

    // Selections and their highlight: the view's own (entity path and primitive to sorted indices),
    // the linked ones (item space to sorted items), and what draws them: a 2-D scene of one quad that
    // dims the picture, and a scene of the selected items alone with the same poses and camera.
    std::map<std::pair<std::string, int>, std::vector<int64_t>> selected_;
    std::map<std::string, std::vector<int64_t>> linked_;
    uint64_t highlighted_ = 0;
    caliper_gfx::Texture dim_texture_;
    uint32_t dim_rgba_ = 0;
    std::vector<caliper_gfx::Buffer> hl_buffers_;
    caliper_scene::SceneCompiler hl_compiler_, dim_compiler_;
    bool build_highlight(caliper_scene::SceneTree* hl, caliper_scene::ViewCamera* cam, std::string* why);
    bool highlight_wanted() const { return !selected_.empty() || !linked_.empty(); }
    std::map<Key, Slot> slots_;
    std::vector<uint8_t> scratch_;
    std::vector<std::string> notes_;
    uint64_t seq_ = 0;                 // the store's sequence numbers, one per applied batch
    bool fitted_ = false, moved_ = false, dirty_ = true;
    caliper_scene::Axis up_mode_ = caliper_scene::Axis::PosY;   // the orbit's axis, or the one it returns to
    bool up_chosen_ = false;           // the person switched: the scene's up no longer moves the view
    bool was_through_ = false;         // the last frame looked through a pinhole
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
