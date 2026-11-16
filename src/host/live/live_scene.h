// live_scene.h: the scene view of the shared view layer (kind 6, scene; design/sprint-4.md,
// increment 1; design/scenes.md, "Drawing a scene natively").
//
// A scene view owns one scene of the scene core (src/host/scene) and applies to it the decoded
// scene messages of wire protocol 3.2 (docs/references/wire-protocol-3-2.md): SCENE_UPDATE with the
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
// the scene first has bounds and again when the person asks. A colormapped batch takes its
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
    // turns an orbit the person has not moved.
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
    std::map<Key, Slot> slots_;
    std::vector<uint8_t> scratch_;
    std::vector<std::string> notes_;
    uint64_t seq_ = 0;                 // the store's sequence numbers, one per applied batch
    bool fitted_ = false, moved_ = false, dirty_ = true;

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
