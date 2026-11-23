// compile.h: compiling a scene into a draw list of the graphics layer
// (design/renderer-core.md, "Compiling primitives into a draw list"; design/scenes.md, "From
// primitives to draws"; design/web-geometry.md, "The design").
//
// For the visible entities, in the byte order of their paths, the compiler chooses each batch's
// program and entries, binds its arrays as vertex or instance-step buffers at their offsets
// (absent arrays read a constant stream at stride 0), writes the frame block, each draw's
// parameter block and the colormap blocks into the draw list's arena, and appends draws: opaque
// first, then translucent, then, when the view looks through a camera, that camera's pixel space
// flat over the picture. prepare() emits the index clamp and the edge gather of batches whose
// indices changed. Labels are not draws: project_labels() gives their anchors on the CPU.
//
// Every draw carries a pick identity; the DrawMap maps it back to the entity, the primitive and
// the part, and compile_pick (pick.h) compiles the same draws with their pick entries.
#pragma once

#include "gfx/gfx.h"
#include "scene/camera.h"
#include "scene/resources.h"
#include "scene/scene.h"
#include "scene/stages.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace caliper_scene {

struct CompileOptions {
    uint32_t width = 0, height = 0;        // the target in physical pixels
    float pixel_scale = 1.0f;              // physical pixels per logical pixel: widths and point sizes
    bool id_target = false;                // an ID attachment beside the colour: refused, see compile()
    uint32_t clear_rgba = 0;
    caliper_gfx::Format color_format = caliper_gfx::Format::RGBA8Unorm;
    bool show_axes = false;                // entity axes for entities with a transform or a camera
    double axes_length = 0;                // scene units; 0 is a tenth of the scene's bounding radius
    uint32_t camera_rgba = 0xff000000u | (230u << 16) | (220u << 8) | 220u;   // frustums
    float camera_width_px = 2.0f;
    // The live points view's depth cues, for points batches drawn at a size in pixels (zero: none,
    // as for every other view). With point_att_ref > 0 a point is its size times
    // clamp(point_att_ref / w, kPointAttenMin, kPointAttenMax) wide, w being its clip-space w, the
    // distance along the view, so that nearer points are larger. With point_fade_w1 >
    // point_fade_w0 its colour is scaled by 1 up to w = point_fade_w0, falling linearly to
    // kPointFadeFloor at point_fade_w1 and beyond, so that farther points are dimmer with their hue
    // and alpha kept; a batch with full_strength is never faded.
    float point_att_ref = 0.0f;
    float point_fade_w0 = 0.0f, point_fade_w1 = 0.0f;
    // The stages of the panel (stages.h): the points and lines batches of an entity that the lookup
    // gives a StageProgram draw with it, in the colour pass and the pick pass. With a lookup, the
    // frame block's pad words carry stage_time_s (as f32 bits) and stage_step, which the stage
    // entries read; without one they stay zero, thus a scene without stages compiles to the same
    // bytes as before.
    const StageLookup* stages = nullptr;
    float stage_time_s = 0.0f;
    uint32_t stage_step = 0;
    // The batches that the caller draws another way (the scene view's library pieces,
    // draw/scene_pieces.h): left out of the colour pass and the pick pass, their bounds and labels
    // kept. Empty: every batch draws here, as before.
    std::function<bool(EntityId, Primitive)> skip;
    // The camera entities whose frustum the caller draws another way (frustum@1).
    std::function<bool(EntityId)> skip_frustum;
};

// The constants of the depth cues above, written out in src/host/gfx/shaders/scene/points.wgsl and
// in the TypeScript points view (web/src/render/points-gpu.ts).
inline constexpr float kPointAttenMin = 0.5f;
inline constexpr float kPointAttenMax = 2.0f;
inline constexpr float kPointFadeFloor = 0.5f;

// What a draw is, beyond its batch.
enum class DrawPart : uint8_t {
    Batch = 0,      // the batch's own draw: points, segments, arrows, box edges, mesh, image quad
    BoxFaces = 1,   // boxes with fill
    DepthCloud = 2, // a depth image's back-projected points
    Frustum = 3,    // a camera entity's frustum
    Axes = 4,       // an entity's axes
};

struct DrawMapEntry {
    EntityId entity;
    Primitive primitive = Primitive::Points;   // meaningless for Frustum and Axes
    DrawPart part = DrawPart::Batch;
    bool overlay = false;                       // drawn flat over a pinhole's picture
    bool translucent = false;                   // alpha blending (in the colour pass)
    bool depth_test = false, depth_write = false;   // the pipeline's depth state in the colour pass
    uint64_t items = 0;                         // instances, or vertices for a mesh
    Mat4 mvp;                                   // images: the unit square onto clip space
    uint32_t image_size[2] = {0, 0};
};

// pick_id to entity, primitive and part. Pick ids start at 1, one per draw, in draw order.
struct DrawMap {
    uint64_t revision = 0;                      // the tree's revision it was compiled from
    std::vector<DrawMapEntry> entries;          // entries[pick_id - 1]
    Mat4 inverse_view_proj;                     // clip to world, for the depth of a pick
    bool has_inverse = false;
    const DrawMapEntry* find(uint32_t pick_id) const {
        return pick_id >= 1 && pick_id <= entries.size() ? &entries[pick_id - 1] : nullptr;
    }
};

struct CompileResult {
    bool ok = false;
    uint32_t draws = 0;
    uint32_t skipped = 0;                       // batches not drawn (reasons in notes)
    uint32_t staged = 0;                        // draws with a stage program
    std::vector<std::string> notes;
};

struct LabelAnchor {
    EntityId entity;
    Primitive primitive = Primitive::Points;
    uint64_t item = 0;
    float x = 0, y = 0;                         // physical pixels, top-left origin
    float depth = 0;                            // distance from the eye; 0 for flat overlays
    std::string text;
    uint8_t pin = 0;                            // 0 placed; pinned at the anchor with the text's 1 start, 2 end, 3 middle
};

class SceneCompiler {
public:
    // Emits into frame the index clamp of each mesh batch and the edge gather of each lines batch
    // of index-pair edges whose arrays changed since the last prepare, into buffers the layer owns.
    void prepare(const SceneTree& tree, Resources& res, caliper_gfx::Frame& frame);
    // The frame of the last prepare() was not submitted (its compile failed or the layer refused
    // it): its clamps and gathers never ran, so the next prepare() emits them again.
    void discard_prepared();
    // Compiles the visible entities (after tree.update()) into out, whose target the caller set.
    // out's depth target decides depth testing; a 2-D scene draws without depth in list order.
    CompileResult compile(const SceneTree& tree, const ViewCamera& camera, const CompileOptions& options,
                          Resources& res, caliper_gfx::DrawList& out, DrawMap* map);

    // The pick pass (pick.h calls it): the same draws with their pick entries into an RG32Uint
    // window of window[2] x window[3] pixels whose top-left is view pixel (window[0], window[1]).
    CompileResult compile_pick(const SceneTree& tree, const ViewCamera& camera, const CompileOptions& options,
                               const int32_t window[4], Resources& res, caliper_gfx::DrawList& out, DrawMap* map);

    // Owned-buffer keys of a batch's clamped indices and gathered edge streams.
    static uint64_t key(EntityId e, Primitive p, uint32_t which);

private:
    struct Prepared { uint64_t seq; uint64_t buffer; uint64_t offset; uint64_t count; };
    CompileResult run(const SceneTree& tree, const ViewCamera& camera, const CompileOptions& options,
                      const int32_t* window, Resources& res, caliper_gfx::DrawList& out, DrawMap* map);
    bool is_prepared(uint64_t key, const BatchArray& a, uint64_t seq) const;
    std::unordered_map<uint64_t, Prepared> prepared_;
    std::vector<uint64_t> fresh_;   // the keys the last prepare() recorded
};

// The label anchors of the visible points and boxes batches whose positions are CPU-visible,
// nearest to the eye first, at most kMaxLabelsDrawn.
std::vector<LabelAnchor> project_labels(const SceneTree& tree, const ViewCamera& camera, uint32_t width,
                                        uint32_t height);

// Checks a pipeline's vertex layouts against the program's description: every vertex input of
// the entry is fed once, by an attribute whose format converts to the shader's (f32x2 into
// vec3 for 2-D positions, u8 into u32 for the break mask).
bool check_vertex_layouts(ProgramId program, const caliper_gfx::RenderPipelineDesc& desc, std::string* error);

}  // namespace caliper_scene
