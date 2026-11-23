// live_scene.h: the scene view of the shared view layer (kind 6, scene; design/sprint-4.md,
// increment 1; design/scenes.md, "Drawing a scene natively").
//
// A scene view keeps the entity tree, camera, CPU arrays, and library piece resources.
// Scene messages update that state. The read-only decoder in scene_legacy.cpp also accepts
// saved per-shape streams. DrawRuntime draws each piece into color, ID, and depth targets.
// A frame changes when data, camera, styles, or piece output changes. Otherwise the view keeps
// its previous targets. Camera input, labels, selections, and pick events use the scene tree.
//
// Every method runs on the frame thread. The view never waits for the GPU except in
// read_pixels(), which tests call.
#pragma once

#include "gfx/gfx.h"
#include "net/wire.h"
#include "scene/camera.h"
#include "scene/labels.h"
#include "scene/pick.h"
#include "scene/scale_bar.h"
#include "scene/scene.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace caliper_draw { class DrawFeed; class DrawRuntime; struct PickedItem; class ScenePieces; struct PieceBatch; }

namespace caliper_live_views {

// What the last drawn frame of a scene view cost.
struct SceneCost {
    double compile_ms = 0;     // piece reconciliation, CPU
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
    uint64_t arrays_live = 0;    // CPU arrays for metadata, selections, and saved streams
};

struct SceneShapeCounts {
    uint32_t pieces = 0;
    uint32_t marks = 0;   // the programs of the highlight of a selection, drawn by the pieces
    uint32_t received = 0;   // the shapes drawn from resources that came in the piece form
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
    // An update of the per-shape form, or of the piece form whose `paired` holds its resources.
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
    // Reads a window from the frame's ID and depth targets. A newer request supersedes the
    // current request. A result from an older scene generation is dropped.
    bool request_pick(float x, float y, std::string* why);
    caliper_scene::PickResult poll_pick(caliper_scene::Hit* out);
    bool pick_pending() const { return shape_asked_; }
    // The frames the view has drawn: the generation a pick belongs to.
    uint64_t generation() const { return generation_; }
    // The detail of a pick or hover event for a hit, as protocol 3.2 gives it: entity, primitive,
    // index (the item in the sent batch), row_id (its source index through the batch's payload axes)
    // and, where the depth gave one, position. "" for a hit on something that is no batch item (a
    // camera's frustum or an entity's axes): a hover there reports nothing.
    std::string pick_detail(const caliper_scene::Hit& h) const;
    // The source index of item `index` of the batch (path, primitive), through its payload axes.
    int64_t row_id(const std::string& path, caliper_scene::Primitive p, int64_t index) const;
    // The programs' pick (scene/pick.h FramePick), which the view owns on its own backend: reads
    // the window around (x, y), in physical pixels, of the ID target and the depth that the
    // frames so far drew, and keeps what each program showed in the last frame. Returns the
    // request's ticket, which a newer request supersedes; 0 with *why when nothing was asked, as
    // while the view draws no program.
    uint64_t request_program_pick(float x, float y, std::string* why);
    // The answer to the request `ticket`: Hit fills *out and *depth with the item of the program
    // whose pick id the texel holds, from the version that the program showed in the frame that
    // the readback read, not from a newer one; Miss when no program's texel lies within the
    // radius or that version holds no such item; Pending; Dropped when a newer request superseded
    // it; None when nothing is asked.
    caliper_scene::PickResult poll_program_pick(uint64_t ticket, caliper_draw::PickedItem* out, float* depth);
    // A number that no other view of the process has. A panel's pick state holds it beside its
    // tickets, thus a request made in one view is never polled in another, whose backend may be
    // another, after a detach, a new attachment or a panel that now draws its own scene.
    uint64_t serial() const { return serial_; }

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
    // The view as it opens: the scene's up and its starting camera, filled from the fit, on its own
    // camera (a camera looked through is left). The scale bar option stays.
    void reset();

    // ---- the starting camera (design/scenes.md, "Starting Camera") --------------------------------
    // The camera the view opens on and a reset returns to: an orbit with any of its keys (the rest
    // filled from the fit, the angles read against the scene's up), the fit, or a camera entity
    // to look through. describe() takes it from the VIEWS entry, and a spec panel sets its option
    // view_camera here. A view whose camera the person has not moved takes a new one at once.
    void set_starting(const caliper_scene::StartingCamera& s);
    const caliper_scene::StartingCamera& starting() const { return start_; }
    // Fits the bounds as a data box (caliper_scene::fit_box) instead of a sphere around them: the
    // fit of a spec panel's entities, so that a wide flat box fills the picture's width.
    void set_box_fit(bool on);
    bool box_fit() const { return box_fit_; }
    // Moves the view's own camera to `o` (camera.set): it leaves a camera looked through and stays
    // for the session, as a move of the person's does, until a reset. False in a 2-D scene.
    bool set_orbit(const caliper_scene::Orbit& o);
    // Whether the person orbited, panned or zoomed since the view opened or was reset. Until then
    // the view follows the scene: it frames the bounds again when they grow past what it framed,
    // as when a viewer attaches late and the scene's parts arrive one by one, and when the
    // picture changes shape.
    bool moved() const { return moved_; }
    // Puts back the camera and the moved() state that a press on the picture saved, when the press
    // ends as a click (live/camera_drag.h): exactly as they were, and the view draws again.
    void restore_camera(const caliper_scene::ViewCamera& camera, bool moved) {
        camera_ = camera;
        moved_ = moved;
        changed();
    }
    // What a spec panel sets on the batches of one entity beyond the scene messages: its depth
    // offset, -8 to 8 (scene.h, Batch::depth_offset), and the size and shape of its points.
    struct BatchStyle {
        int depth_offset = 0;
        float size_px = 0;                                    // 0: by count, as the points view
        std::optional<caliper_scene::PointShape> point_shape;
    };
    void set_style(const std::string& path, const BatchStyle& style);
    // The scale bar of a picture of w x h logical pixels (scene/scale_bar.h) at the camera the view
    // draws with; hidden while the option is off and while it looks through a pinhole.
    caliper_scene::ScaleBar scale_bar(double w, double h) const;
    bool scale_bar_on() const { return scale_bar_on_; }
    void set_scale_bar(bool on) { scale_bar_on_ = on; }
    // The entities that the view draws, picks and labels: a spec panel's include and exclude.
    // `keep` answers for an entity's path; an entity it refuses is hidden with its subtree, but
    // for the ancestors of an entity it keeps. `key` names the filter, so that setting the same
    // one each frame costs nothing; an empty key with no function keeps every entity. A panel and
    // the app's window draw one view, thus each sets its own before it renders.
    void set_entity_filter(const std::string& key, std::function<bool(const std::string&)> keep);
    // What the view keeps of itself across a close and a reopen.
    struct Settings {
        bool up_chosen = false;
        caliper_scene::Axis up = caliper_scene::Axis::PosY;
        bool scale_bar = true;
    };
    Settings settings() const { return {up_chosen_, up_mode_, scale_bar_on_}; }
    // After describe(): a remembered mode wins over the scene's up.
    void restore(const Settings& s);

    // ---- programs of pieces (design/programmable-graphics.md, Render Graph, Draw State and Safety
    // of Draws) ------------------------------------------------------------------------------------
    // The draw halves of the programs that the panel shows. The view installs one runtime of the
    // draw layer (draw/draw_runtime.h) for each program of the feed, on its own backend, and draws
    // them in every frame: each program in submits of its own after the scene's frame and before
    // the highlight of a selection, into the view's colour, its ID target (RG32Uint, the program's
    // pick id and the item, cleared to 0 in each frame) and its depth. A panel of programs alone
    // frames the box from -1 to 1 on each axis. The view never waits for a program's GPU
    // work: poll() gives each program the GPU time of its completed submits and the age of its
    // oldest pending one, which stops a program over its budget, and the programs of the panel
    // together over 16.7 ms a frame on 30 polls in a row stop the costliest one. Each change of a
    // program's state goes back to the feed. Null draws no program. Without the draw layer in the
    // build (CALIPER_HAVE_DRAW), the view draws none.
    void set_draw_feed(caliper_draw::DrawFeed* feed);
    // The installed programs of the feed, in draw order.
    std::vector<const caliper_draw::DrawRuntime*> draw_runtimes() const;
    // The frames that the installed programs encoded together, each counted since its install, and
    // the newest step of the resources that their last frames showed (-1 before any).
    uint64_t program_frames() const;
    int64_t program_step() const;
    // The installed pieces of the shapes, in draw order; they draw before the feed's programs.
    std::vector<const caliper_draw::DrawRuntime*> shape_runtimes() const;

    // ---- shapes drawn by library pieces (draw/scene_pieces.h) ---------------------------------------
    // The library pieces and selection marks of the current frame.
    SceneShapeCounts shape_counts() const;
    // A piece that draws the batch (path, p) with resources that the caller writes, in place of the
    // converter's (the spec's surface@1 and path@1); null removes it. Nothing without the pieces.
    void set_piece_batch(const std::string& path, caliper_scene::Primitive p, const caliper_draw::PieceBatch* batch);
    // The programs' ID target; none while the view draws no program.
    caliper_gfx::Texture id_target() const { return id_; }
    // The depth target of the view's frame (Depth32Float, with kCopySrcTex); none in 2-D. A
    // program's pick reads it beside the ID target, so that an entity's hit wins when nearer.
    caliper_gfx::Texture depth_target() const { return depth_; }
    // The clip-from-world matrix of the programs' last frame, with which the panel projects the
    // anchors of their text tables.
    const caliper_scene::Mat4& program_clip() const { return program_clip_; }
    // The pointer over the view, in physical pixels from the top left, its buttons as bits, and
    // whether it is over the view, for the programs' prelude (caliper_view.pointer).
    void set_pointer(float x, float y, uint32_t buttons, bool over);

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
    // The camera of the last frame the view submitted, in the same words (the UI driver's drags).
    std::string drawn_camera_text() const;
    caliper_gfx::Backend& backend() { return gfx_; }

private:
    // One layer of one part: its batch and CPU arrays by role.
    struct Slot {
        caliper_scene::Batch batch;
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

    void release(Slot& s);
    bool in_pixel_space(caliper_scene::EntityId e) const;   // under a camera, from the parts now
    void rebuild(const std::string& path);    // the entity's batches again from the slots
    void apply_start(double aspect);          // the camera from the starting camera and the fit
    bool start_without_bounds() const;        // the starting camera needs no fit: target and distance
    void reset_scene();
    void changed() { dirty_ = true; }

    caliper_gfx::Backend& gfx_;
    caliper_scene::SceneTree tree_;
    caliper_scene::ViewCamera camera_ = caliper_scene::Orbit{};
    std::string through_;              // the camera entity looked through, or ""
    // The programs' pick: its readback, its ticket and what each program showed (live_scene.cpp).
    struct ProgramPick;
    std::unique_ptr<ProgramPick> program_pick_;
    uint64_t serial_ = 0;
    uint64_t generation_ = 0;          // colour frames submitted
    // Selections and their highlight: the view's own (entity path and primitive to sorted indices),
    // the linked ones (item space to sorted items), and what draws them: a 2-D scene of one quad that
    // dims the picture, and a scene of the selected items alone with the same poses and camera.
    std::map<std::pair<std::string, int>, std::vector<int64_t>> selected_;
    std::map<std::string, std::vector<int64_t>> linked_;
    uint64_t highlighted_ = 0;
    // The selected rows of a batch, by the view's selection and the linked one: sorted, unique and
    // below the batch's count; empty when none is selected or the batch has no slot.
    std::vector<int64_t> highlight_rows(caliper_scene::EntityId e, caliper_scene::Primitive p) const;
    bool piece_highlight_ = false;   // the shapes' pieces draw this frame's highlight
    bool highlight_wanted() const { return !selected_.empty() || !linked_.empty(); }
    std::map<Key, Slot> slots_;
    std::vector<uint8_t> scratch_;
    std::vector<std::string> notes_;
    std::vector<std::string> piece_notes_;   // current piece errors, removed when the next sync resolves them
    uint64_t seq_ = 0;                 // the store's sequence numbers, one per applied batch
    bool fitted_ = false, moved_ = false, dirty_ = true;
    caliper_scene::StartingCamera start_;     // where the view opens and resets to
    bool box_fit_ = false;
    caliper_scene::Aabb fitted_bounds_;       // the bounds and the picture's shape of the last fit
    double fitted_aspect_ = 0;
    std::map<std::string, BatchStyle> styles_;   // by entity path
    caliper_scene::Axis up_mode_ = caliper_scene::Axis::PosY;   // the orbit's axis, or the one it returns to
    bool up_chosen_ = false;           // the person switched: the scene's up no longer moves the view
    bool was_through_ = false;         // the last frame looked through a pinhole
    bool scale_bar_on_ = true;
    std::string filter_key_;
    std::function<bool(const std::string&)> filter_;
    uint64_t filtered_rev_ = ~0ull;   // the tree's revision after the filter was applied
    void apply_entity_filter();
    // The update in the per-shape form; `received` holds the piece form's resources, which the pieces draw.
    bool decode_scene_update(const caliper_net::wire::SceneUpdate& u, const std::vector<caliper_net::wire::Section>* received,
                      std::string* why);

    caliper_gfx::Texture color_, depth_;
    uint32_t width_ = 0, height_ = 0, clear_ = 0;
    float scale_ = 1.0f;
    uint64_t target_creates_ = 0;
    uint64_t drawn_revision_ = 0, drawn_pose_ = 0;
    caliper_scene::ViewCamera drawn_camera_;   // the camera of the last submitted frame

    // The programs of pieces: their runtimes with their pending submits, the feed's revision drawn,
    // the ID target, the host's colormaps, the pointer and the polls over the panel's limit.
    struct ProgramSlot;
    std::vector<std::unique_ptr<ProgramSlot>> draws_;
    // The shapes' pieces, their revision drawn, and the pick of their IDs that request_pick() asks
    // beside the pick pass: its readback, whether it and the pass answered, and their hits.
    // Deleted in live_scene.cpp, which knows the type only where the build has the pieces.
    struct ShapesDeleter {
        void operator()(caliper_draw::ScenePieces* p) const;
    };
    std::unique_ptr<caliper_draw::ScenePieces, ShapesDeleter> shapes_;
    uint64_t shape_revision_ = ~0ull, marks_revision_ = ~0ull;
    std::unique_ptr<caliper_scene::FramePick> shape_pick_;
    bool shape_asked_ = false;
    uint64_t shape_generation_ = 0;   // the frame the shapes' pick reads; a newer one drops it
    void sync_shapes();
    void refresh_piece_notes();
    bool external_draws() const;
    caliper_draw::DrawFeed* slot_feed(const ProgramSlot& slot) const;
    caliper_draw::DrawFeed* draw_feed_ = nullptr;
    uint64_t draw_revision_ = ~0ull;
    caliper_gfx::Texture id_, program_colormaps_;
    uint32_t id_w_ = 0, id_h_ = 0;
    caliper_scene::Mat4 program_clip_;
    float pointer_[4] = {0, 0, 0, 0};
    uint32_t panel_over_ = 0;
    std::chrono::steady_clock::time_point last_program_frame_{};
    void reconcile_programs();
    bool prepare_program_targets(std::string* why);
    void draw_programs();
    void poll_programs();

    SceneCounts counts_;
    SceneCost cost_;
};

}  // namespace caliper_live_views
