// pick.h: picking in a scene (design/renderer-core.md, "Picking"; design/programmable-graphics.md,
// Picking).
//
// Every shape of a scene is drawn by a library piece, whose pick pass writes (its pick id, the item)
// into the ID target of the view's frame. A pick reads the window around the pointer from the newest
// frame's ID and depth targets with FramePick, and decode_frame_pick takes the texel nearest the
// pointer among the programs' pick ids; the view maps it back to a Hit: the entity, the primitive,
// the item, an image's pixel and, where a depth was read, the world position.
#pragma once

#include "gfx/gfx.h"
#include "scene/scene.h"

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

// What a hit is, beyond its batch.
enum class DrawPart : uint8_t {
    Batch = 0,      // the batch's own draw: points, segments, arrows, box edges, mesh, image quad
    BoxFaces = 1,   // boxes with fill
    DepthCloud = 2, // a depth image's back-projected points
    Frustum = 3,    // a camera entity's frustum
    Axes = 4,       // an entity's axes
};

// The state of a pick that the view asked for.
enum class PickResult { None, Pending, Hit, Miss, Dropped };

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
    PickResult poll(const std::function<bool(uint32_t)>& is_program, FrameHit* out);
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
