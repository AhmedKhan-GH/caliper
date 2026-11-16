// resources.h: what the scene core asks of the graphics layer while it compiles a scene.
//
// The scene core never calls a backend. It compiles into a caliper_gfx::DrawList and a
// caliper_gfx::Frame, and asks a Resources for the handles those records name: pipelines of the
// generated programs, bind groups, the buffer of its constant streams, buffers the layer owns
// for clamped indices and gathered edges, and the colormap blocks. A view implements Resources
// over its Backend (creating and caching each object); the headless tests implement it with
// plain handle values and inspect what was asked.
#pragma once

#include "gfx/gfx.h"
#include "scene/uniform_blocks.h"

#include <cstdint>

namespace caliper_scene {

using caliper_gfx::shaders::ProgramId;

// The scene core's constant streams, uploaded once into the buffer static_geometry() returns:
// what a stream at stride 0 reads when a batch has no such array, and the unit geometry of
// camera frustums and entity axes.
struct StaticGeometry {
    static constexpr uint32_t kIdentityPose = 0;     // 64 bytes: the identity 4 x 4, column-major
    static constexpr uint32_t kIdentityQuat = 48;    // its last column, (0, 0, 0, 1)
    static constexpr uint32_t kZero = 64;            // 16 zero bytes: absent colour sources, radii, normals, UVs
    static constexpr uint32_t kKeep = 80;            // a u32 0xffffffff: the break mask of segments and arrows
    static constexpr uint32_t kFrustum = 96;         // 16 f32x3: the unit frustum's 8 segments
    static constexpr uint32_t kAxes = 288;           // 6 f32x3: three unit segments along x, y and z
    static constexpr uint32_t kAxesColors = 360;     // 6 u32: red, red, green, green, blue, blue
    static constexpr uint32_t kBytes = 384;
    // Fills kBytes bytes. The unit frustum is the centre (0, 0, 0) to the corners (s, t, 1) of
    // the unit square, then the square's four sides; a camera's draw maps (s, t, z) onto its
    // image plane.
    static void fill(uint8_t* out);
};

// Which binding of group 2 a textured draw's image fills.
enum ImageBinding : uint32_t { kImageRgba = 0, kImageClass = 2, kImageValue = 3 };

class Resources {
public:
    virtual ~Resources() = default;
    // A pipeline of the generated program `program` for this descriptor (desc.program is left
    // empty for the implementation to fill). Cached by descriptor in the layer.
    virtual caliper_gfx::Pipeline render_pipeline(ProgramId program, const caliper_gfx::RenderPipelineDesc& desc) = 0;
    virtual caliper_gfx::Pipeline compute_pipeline(ProgramId program, const char* entry) = 0;
    // Group 0 (kArena for the frame block, and the class table), group 1 (kArena for the draw
    // block and the colormap block) or group 2 (image at image_binding, the linear sampler, and
    // placeholders for the other images) of a scene program. Only groups the program declares
    // are asked for.
    virtual caliper_gfx::BindGroup bind_group(ProgramId program, uint32_t group, caliper_gfx::Texture image,
                                              uint32_t image_binding) = 0;
    // A group 0 of the index clamp or the edge gather over these entries, in the program's
    // binding order; binding 0 is kArena for the parameters.
    virtual caliper_gfx::BindGroup compute_group(ProgramId program, caliper_gfx::View<caliper_gfx::BindEntry> entries) = 0;
    // The buffer holding StaticGeometry.
    virtual caliper_gfx::Buffer static_geometry() = 0;
    // A buffer the layer owns, at least `bytes` large, kept under `key`: the same key and size
    // give the same buffer. The scene core keeps clamped indices and gathered edges there.
    virtual caliper_gfx::Buffer owned_buffer(uint64_t key, uint64_t bytes, uint32_t usage) = 0;
    // The colormap block of a colormap and range (renderer core, "Exact colormap output"); the
    // keys come from the CPU reference.
    virtual void colormap_block(uint32_t colormap, float vmin, float vmax, caliper_gfx::shaders::Cmap* out) = 0;
};

}  // namespace caliper_scene
