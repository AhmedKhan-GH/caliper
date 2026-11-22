// pick.h: picking in a scene (design/renderer-core.md, "Picking"; design/web-geometry.md,
// "Picking").
//
// On a pointer event the view asks the Picker for a window around the pointer, 16 x 16 pixels
// for a radius of 8, and compiles the pick variants of the visible draws into it with
// SceneCompiler::compile_pick: an RG32Uint target (with its own depth target in 3-D), reached by
// zooming clip space through the frame block's pick window. Every pickable draw writes its pick
// identity and the item: the instance, or a vertex of the triangle for a mesh. The layer reads the
// window back asynchronously; the view hands the bytes to Picker::deliver, which drops a result
// that a newer request superseded or that arrives after the draw list changed, and decodes the
// texel nearest the pointer into a hit: the entity, the primitive, the item, and where a depth
// was read, the world position. An image writes no item; its pixel is found from its quad. PickPass
// runs all of this on a backend of the graphics layer.
//
// A program's pick pass writes its ID in the view's own frame instead (design/programmable-graphics.md,
// Picking), thus FramePick reads the same window from the ID and depth targets of a frame that the
// view already drew, and decode_frame_pick takes the texel nearest the pointer among the programs'
// pick ids.
#pragma once

#include "gfx/gfx.h"
#include "scene/compile.h"
#include "scene/resources.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace caliper_scene {

struct PickWindow {
    int32_t x0 = 0, y0 = 0;          // the window's top-left in view pixels (may lie outside the view)
    int32_t width = 16, height = 16;
    float px = 0, py = 0;            // the pointer in view pixels
    uint32_t view_width = 0, view_height = 0;
    int32_t radius = 8;
};
// A window of 2r x 2r pixels whose texel (r, r) holds the pointer's pixel.
PickWindow pick_window(float x, float y, int radius_px, uint32_t view_width, uint32_t view_height);

struct Hit {
    EntityId entity;
    Primitive primitive = Primitive::Points;
    DrawPart part = DrawPart::Batch;
    uint64_t item = 0;               // the instance, the edge, or a vertex of the mesh's triangle
    bool overlay = false;
    float depth = 1.0f;              // the depth texel, when read
    bool has_position = false;
    Vec3 position;                   // the world position, from the depth
    bool has_pixel = false;
    uint32_t pixel[2] = {0, 0};      // images: the column and row under the pointer
    int32_t texel[2] = {0, 0};       // where in the window the hit was read
};

// Decodes a read-back window: ids holds width * height pairs (pick id, item) top row first, as
// read_texture returns an RG32Uint region; depth (or nullptr) holds width * height f32. The hit is
// the texel with a pick id nearest the pointer within the radius and inside the view, the nearer
// depth first between equals.
std::optional<Hit> decode_pick(const uint32_t* ids, const float* depth, const PickWindow& w, const DrawMap& map);

class Picker {
public:
    // A new request supersedes an unanswered one; the view compiles and reads back the window.
    PickWindow request(float x, float y, int radius_px, uint32_t view_width, uint32_t view_height);
    uint64_t current() const { return serial_; }
    bool pending() const { return pending_; }
    // The bytes of request `serial` (read_texture's format: 8 bytes a texel of ids, 4 of depth).
    // Nothing when a newer request superseded it, when the map's revision is not the one the
    // request was compiled from (revision), or when the bytes are short.
    std::optional<Hit> deliver(uint64_t serial, uint64_t revision, const std::vector<uint8_t>& ids,
                               const std::vector<uint8_t>* depth, const DrawMap& map);

private:
    PickWindow window_;
    uint64_t serial_ = 0;
    bool pending_ = false;
};

// ---- The pick pass on the graphics layer --------------------------------------------------------
// On a pointer event the view asks for a pick at a pixel. The pass compiles the pick variants of the
// visible draws (SceneCompiler::compile_pick) into a window of 2r x 2r pixels around the pointer, 16 x 16
// for the default radius of 8: an RG32Uint target with its own Depth32Float target in a 3-D scene, which
// the window reaches by zooming clip space. Every pickable draw writes its pick id and its item (the
// instance; a vertex of the triangle for a mesh; 0 for an image, whose pixel is found from its quad).
// The pass submits that frame and starts the layer's asynchronous readbacks of both targets; poll()
// takes them a frame or two later and decodes the texel nearest the pointer with the Picker. A result
// is dropped when a newer request superseded it or when the view drew another frame since the request
// (its generation changed), because the pixels the person points at have changed. The pass costs
// nothing while the pointer is still: it submits only when asked.
//
// Like BackendResources, it is the scene core's helper over a backend of the layer; the scene core's
// compiler itself never calls one.
class PickPass {
public:
    explicit PickPass(caliper_gfx::Backend& backend, int radius_px = 8);
    ~PickPass();
    PickPass(const PickPass&) = delete;
    PickPass& operator=(const PickPass&) = delete;

    // Compiles and submits the pick pass around (x, y), in physical pixels of the view whose size
    // `options` gives, and starts the readbacks. `generation` names the frame the view shows now. A
    // new request supersedes an unanswered one. False, with why, when the scene did not compile or
    // the layer refused the frame.
    bool request(const SceneTree& tree, const ViewCamera& camera, const CompileOptions& options,
                 SceneCompiler& compiler, Resources& res, float x, float y, uint64_t generation,
                 std::string* why);

    enum class Result { None, Pending, Hit, Miss, Dropped };
    // Takes the readbacks once both are in. Hit fills *out; Miss: no pickable pixel within the
    // radius; Dropped: the view's frame changed since the request; Pending: not yet; None: nothing
    // was asked.
    Result poll(uint64_t generation, Hit* out);
    bool pending() const { return waiting_; }

    // Counters: pick frames submitted, results decoded (hits and misses), results dropped.
    uint64_t submits() const { return submits_; }
    uint64_t answered() const { return answered_; }
    uint64_t dropped() const { return dropped_; }
    const PickWindow& window() const { return window_; }
    caliper_gfx::Serial serial() const { return serial_; }   // the last pick frame's submit
    // The last work on the window's targets, their readbacks included (a test may wait on it).
    caliper_gfx::Serial last_use() const;
    // The window as read back by the last answered request: ids (8 bytes a texel) and depth (4).
    const std::vector<uint8_t>& last_ids() const { return ids_; }
    const std::vector<uint8_t>& last_depth() const { return depth_; }

private:
    bool targets(bool depth, std::string* why);
    void cancel();

    caliper_gfx::Backend& gfx_;
    int radius_;
    caliper_gfx::Texture ids_target_, depth_target_;
    caliper_gfx::Readback rb_ids_, rb_depth_;
    bool have_ids_ = false, have_depth_ = false, want_depth_ = false;
    std::vector<uint8_t> ids_, depth_;
    Picker picker_;
    PickWindow window_;
    DrawMap map_;
    uint64_t generation_ = 0;
    bool waiting_ = false;
    uint64_t submits_ = 0, answered_ = 0, dropped_ = 0;
    caliper_gfx::Serial serial_ = 0;
};

// ---- Picks in a drawn frame (design/programmable-graphics.md, Picking) ---------------------------------
// A program's pick pass writes (its pick id, the item) into the ID target of the view's frame in every
// frame. A pick of a program thus reads the window around the pointer from the newest frame's ID
// target and its depth, with no pass of its own: the same window and the same nearest texel as the
// pick pass, among the texels whose pick id names a program.

// A texel of a program's pick: its pick id, the item that the fragment wrote (unchecked: the caller
// holds it against the item map's valid count), and its depth (1 without a depth target).
struct FrameHit {
    uint32_t pick_id = 0;
    uint32_t item = 0;
    float depth = 1.0f;
    int32_t texel[2] = {0, 0};   // where in the window it was read
};

// The part of window `w` that lies inside a target of width x height, as read_texture takes it, and
// where it starts in the window; width or height 0 when no part does.
struct WindowRegion {
    uint32_t x = 0, y = 0, width = 0, height = 0;
    int32_t i0 = 0, j0 = 0;
};
WindowRegion window_region(const PickWindow& w, uint32_t target_width, uint32_t target_height);

// Decodes a region read back from a frame: `ids` holds region.width * region.height pairs (pick id,
// item), top row first, and `depth` (or null) as many f32. The hit is the texel nearest the pointer
// within the radius whose pick id `is_program` accepts, the nearer depth first between equals.
// Nothing when the bytes are shorter than the region, or when no texel is a program's.
std::optional<FrameHit> decode_frame_pick(const std::vector<uint8_t>& ids, const std::vector<uint8_t>* depth,
                                          const PickWindow& w, const WindowRegion& region,
                                          const std::function<bool(uint32_t)>& is_program);

// Whether the pick pass's hit of a built-in entity wins over a program's texel at the same pointer:
// it wins only when its depth is nearer, because the programs draw after the scene.
bool entity_hit_wins(const Hit& entity, const FrameHit& program);

class FramePick {
public:
    explicit FramePick(caliper_gfx::Backend& backend, int radius_px = 8);
    ~FramePick();
    FramePick(const FramePick&) = delete;
    FramePick& operator=(const FramePick&) = delete;

    // Starts the readbacks of the window around (x, y), in physical pixels, from `ids` (RG32Uint,
    // with kCopySrcTex) and `depth` (Depth32Float, or none in 2-D), a target of width x height that
    // the frames submitted so far drew. A new request supersedes an unanswered one. False, with
    // why, for a pointer that is not finite or not on the target, or when the layer gives no
    // readback.
    bool request(caliper_gfx::Texture ids, caliper_gfx::Texture depth, uint32_t width, uint32_t height, float x,
                 float y, std::string* why);
    // Takes the readbacks once both are in and decodes them with `is_program`: Hit fills *out, Miss
    // is no program's texel within the radius, Pending not yet, None nothing asked. A program draws
    // every frame, thus a newer frame does not drop the result; the caller reads the item in the
    // version of the frame it showed.
    PickPass::Result poll(const std::function<bool(uint32_t)>& is_program, FrameHit* out);
    bool pending() const { return waiting_; }
    const PickWindow& window() const { return window_; }

private:
    void cancel();

    caliper_gfx::Backend& gfx_;
    int radius_;
    PickWindow window_;
    WindowRegion region_;
    caliper_gfx::Readback rb_ids_, rb_depth_;
    bool have_ids_ = false, have_depth_ = false, want_depth_ = false, waiting_ = false;
    std::vector<uint8_t> ids_, depth_;
};

}  // namespace caliper_scene
