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
// was read, the world position. An image writes no item; its pixel is found from its quad.
#pragma once

#include "scene/compile.h"

#include <cstdint>
#include <optional>
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

}  // namespace caliper_scene
