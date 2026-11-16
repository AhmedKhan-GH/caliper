// scene.h: the scene store of the scene core (design/renderer-core.md, "The scene core";
// design/scenes.md, "The scene model").
//
// A SceneTree holds one scene: its space, up axis and units; its entities by path, stored as
// arrays with each parent before its children; each entity's transform, camera and one batch per
// primitive type; the class table. A batch describes arrays that live in layer buffers (buffer,
// byte offset, count, stride and format, by role) and never copies their data. Every part has a
// stamped layer and a static layer, so that a clear removes stamped parts and keeps the scene's
// setup. update() recomputes, in one forward pass over the dirty subtrees, the world matrices
// (composed in double) and the bounds. Nothing draws here: compile.h turns a tree into a draw
// list of the graphics layer.
//
// Decisions where the design was incomplete (each is in the step's report):
//  - The root takes one of the 1,024 entity rows, because it may hold primitives and a batch
//    names its entity's row; a scene therefore holds the root and 1,023 named entities.
//  - Cameras are refused in a 2-D scene, on the root, inside a camera's pixel space and on an
//    entity whose subtree holds 3-D batches or cameras.
//  - A colour stream is 4 bytes an item (packed RGBA, f32 value bits or an i32 class id), and
//    the break mask of strips is 4 bytes a vertex whose first byte is the flag: WebGPU allows
//    only vertex strides that are multiples of 4, so (N, 3) colours and a one-byte mask are
//    expanded by the producer.
//  - The world bounds of a scene cover its 3-D batches and its cameras' centres. Batches in a
//    camera's pixel space are left out, because their place depends on the plane distance,
//    which depends on the bounds.
//  - Arrays that the index clamp and the edge gather read through storage bindings (a mesh's
//    indices; an edge batch's pairs, positions and per-vertex colours) sit at byte offsets that
//    are multiples of 256, WebGPU's storage offset alignment, and the gather's rows are packed.
//  - An image batch carries the layer texture that the view filled from its buffer with
//    fill_texture_from_buffer, since filling is the view's work on the layer; the store keeps
//    only the handle and the size.
#pragma once

#include "gfx/gfx.h"
#include "scene/path.h"
#include "scene/scene_math.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_scene {

// ---- The scene's properties ----------------------------------------------------------------

enum class Space : uint8_t { k3D = 0, k2D = 1 };
enum class Axis : uint8_t { PosX = 0, NegX, PosY, NegY, PosZ, NegZ };

Vec3 axis_vector(Axis a);
const char* axis_name(Axis a);                         // "+x", "-x", "+y", "-y", "+z", "-z"
bool parse_axis(std::string_view text, Axis* out);

// ---- Limits (design/scenes.md, "Limits") ------------------------------------------------------

inline constexpr uint32_t kMaxEntities = 1024;          // the root included
inline constexpr uint32_t kMaxBatches = 2048;
inline constexpr uint64_t kMaxItems = 4194304;          // points, line vertices, edges, mesh vertices, triangles
inline constexpr uint64_t kMaxShapes = 1048576;         // boxes, arrows, line strips
inline constexpr uint32_t kMaxLabelsPerBatch = 4096;
inline constexpr uint32_t kMaxLabelBytes = 63;
inline constexpr uint32_t kMaxLabelsDrawn = 512;
inline constexpr uint32_t kMaxClasses = 4096;
inline constexpr uint32_t kMaxClassId = 65535;
inline constexpr uint32_t kClassTableSide = 256;        // 256 x 256 texels: ids 0 to 65,535
inline constexpr uint32_t kMaxImageSide = 8192;
inline constexpr uint32_t kMaxUnitsBytes = 15;
// WebGPU's minStorageBufferOffsetAlignment: arrays the index clamp and the edge gather read.
inline constexpr uint32_t kStorageOffsetAlign = 256;

// The points view's flat light grey, packed r | g << 8 | b << 16 | a << 24.
inline constexpr uint32_t kDefaultRgba = 0xff000000u | (230u << 16) | (220u << 8) | 220u;
// The fixed 16-colour categorical palette of the points view (live_views_draw.cpp, kPalette).
extern const uint32_t kPalette[16];

// ---- Entities, bounds, cameras ---------------------------------------------------------------

struct EntityId {
    uint32_t index = 0xffffffffu, generation = 0;
    explicit operator bool() const { return index != 0xffffffffu; }
    friend bool operator==(EntityId a, EntityId b) { return a.index == b.index && a.generation == b.generation; }
    friend bool operator!=(EntityId a, EntityId b) { return !(a == b); }
};

struct Aabb {
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    bool valid = false;
    void add(Vec3 p);
    void merge(const Aabb& b);
    Vec3 center() const { return {(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5}; }
    Vec3 half_extent() const { return {(hi[0] - lo[0]) * 0.5, (hi[1] - lo[1]) * 0.5, (hi[2] - lo[2]) * 0.5}; }
    double radius() const { return length(half_extent()); }   // of the bounding sphere
};
// The bounds of the box's eight corners mapped by m: what a batch's local bounds become in
// world space.
Aabb transform_aabb(const Mat4& m, const Aabb& box);

enum class Convention : uint8_t { Rdf = 0, Rub = 1 };   // x right, y down, z forward; or x right, y up, looking along -z

struct Pinhole {
    double fx = 0, fy = 0, cx = 0, cy = 0;    // pixels
    uint32_t width = 0, height = 0;           // pixels
    Convention convention = Convention::Rdf;
    double plane_distance = 0;                // scene units; 0: a twentieth of the scene's bounding radius
};

// The parts form of an entity transform: T * R * S, the quaternion normalised.
struct Transform {
    double translation[3] = {0, 0, 0};
    double rotation[4] = {0, 0, 0, 1};        // x, y, z, w
    double scale[3] = {1, 1, 1};
};
// The 2-D form: a counter-clockwise angle in radians when y is up.
struct Transform2D {
    double translation[2] = {0, 0};
    double angle = 0;
    double scale[2] = {1, 1};
};
// A zero or non-finite quaternion gives a matrix that is not finite, which hides the entity.
Mat4 transform_matrix(const Transform& t);
Mat4 transform_matrix(const Transform2D& t);

// ---- Batches -----------------------------------------------------------------------------------

enum class Primitive : uint8_t { Points = 1, Lines = 2, Arrows = 3, Boxes = 4, Mesh = 5, Image = 6 };
inline constexpr int kPrimitiveCount = 6;
inline int primitive_slot(Primitive p) { return int(p) - 1; }
const char* primitive_name(Primitive p);

enum class Role : uint8_t {
    Positions,    // f32 (N, D): points, line vertices, mesh vertices
    Colors,       // packed RGBA, 4 bytes an item
    Values,       // f32, through the batch's colormap
    ClassIds,     // i32, through the class table
    Radii,        // f32, scene units
    BreakMask,    // strips: 4 bytes a vertex, first byte 0 at the last vertex of a strip
    Indices,      // i32 rows: (T, 3) triangles of a mesh, (E, 2) pairs of index-pair edges
    Normals,      // f32 (V, 3)
    Uvs,          // f32 (V, 2)
    Origins,      // arrows, f32 (N, D)
    Vectors,      // arrows, f32 (N, D)
    Centers,      // boxes, f32 (N, D)
    HalfSizes,    // boxes, f32 (N, D)
    Rotations,    // boxes, f32 (N, 4) quaternions, 3-D only
};
const char* role_name(Role r);

// One array of a batch, in a layer buffer. count is in rows (items, vertices, triangles or
// pairs); a count of 1 for an optional per-item array means one value for all and binds at
// stride 0. offset and stride are multiples of 4, stride at most 2,048.
struct BatchArray {
    Role role = Role::Positions;
    caliper_gfx::Buffer buffer;
    uint64_t offset = 0;
    uint64_t count = 0;
    uint32_t stride = 0;
    caliper_gfx::VertexFormat format = caliper_gfx::VertexFormat::F32x3;
    const void* cpu = nullptr;   // a CPU view of the rows at offset, for bounds, labels and lasso; or nullptr
};

enum class LineMode : uint8_t { Segments = 0, Strips = 1, Edges = 2 };
enum class ImageForm : uint8_t { Color = 0, Grey = 1, Depth = 2, Segmentation = 3 };
enum class PointShape : uint8_t { Square = 0, Disc = 1 };

struct Batch {
    Primitive primitive = Primitive::Points;
    std::vector<BatchArray> arrays;           // at most 8, one per role

    // Colour: Colors, else Values through the colormap, else ClassIds, else flat_rgba.
    uint32_t flat_rgba = kDefaultRgba;        // one colour for all, or the default
    uint32_t colormap = 0;                    // an id the view's colormap blocks know
    float vmin = 0.0f, vmax = 1.0f;
    bool translucent = false;                 // per-item colours with alpha below 255

    float width_px = 2.0f;                    // lines, arrows, box edges, logical pixels
    float size_px = 0.0f;                     // points: logical pixels; 0 by count, as the points view
    PointShape point_shape = PointShape::Disc;
    bool full_strength = false;               // points: never faded by the view's depth cues (CompileOptions)

    LineMode line_mode = LineMode::Segments;
    bool per_edge = false;                    // index-pair edges: colors and values are per edge, not per vertex
    bool fill = false;                        // boxes: also translucent faces

    ImageForm image_form = ImageForm::Color;  // images; a decoded JPEG or PNG frame is Color
    float opacity = 1.0f;                     // images and segmentation
    bool cloud = false;                       // depth images under a camera: also the back-projected cloud
    caliper_gfx::Texture texture;             // images and textured meshes: the filled layer texture
    uint32_t image_width = 0, image_height = 0;

    Aabb bounds;                              // local, from the producer; else a scan of the CPU view
    std::vector<std::string> labels;          // one per item, or one for all
    uint64_t seq = 0;                         // the producer's sequence number of these arrays

    const BatchArray* array(Role r) const;
    uint64_t count() const;                   // items: points, line vertices, arrows, boxes, mesh vertices
    uint32_t dims() const;                    // 2 or 3, from the positions, origins or centers; 0 for images
};

struct ClassEntry {
    uint32_t id = 0;
    std::string label;
    uint32_t rgba = 0xffffffffu;
};

// ---- The tree ------------------------------------------------------------------------------------

class SceneTree {
public:
    explicit SceneTree(Space space = Space::k3D);
    SceneTree(Space space, Axis up, std::string_view units);

    // The scene's up axis (3-D: any of the six; 2-D: -y or +y) and its unit note.
    bool set_up(Axis up, std::string* error = nullptr);
    bool set_units(std::string_view units, std::string* error = nullptr);
    Space space() const { return space_; }
    Axis up() const { return up_; }
    const std::string& units() const { return units_; }

    // Entities. ensure creates missing parents as empty entities; an invalid id is returned (and
    // *error set) for a path outside the grammar or past the entity limit.
    EntityId root() const { return {0, gens_[0]}; }
    EntityId ensure(std::string_view path, std::string* error = nullptr);
    EntityId find(std::string_view path) const;
    bool valid(EntityId e) const;
    size_t entity_count() const { return ents_.size(); }   // the root included
    EntityId entity_at(uint32_t index) const;
    const std::string& path(EntityId e) const;
    EntityId parent(EntityId e) const;                      // invalid for the root
    // Entity indices in the byte order of their paths, the root first.
    const std::vector<uint32_t>& path_order() const;

    // Parts. static_item writes the static layer, which a non-static clear keeps.
    bool set_transform(EntityId e, const Mat4& m, bool static_item = false, std::string* error = nullptr);
    bool set_transform(EntityId e, const double m[16], bool static_item = false, std::string* error = nullptr);
    bool set_transform(EntityId e, const float m[16], bool static_item = false, std::string* error = nullptr);
    bool set_camera(EntityId e, const Pinhole& c, bool static_item = false, std::string* error = nullptr);
    bool set_batch(EntityId e, const Batch& b, bool static_item = false, std::string* error = nullptr);
    // Removes the entity's transform, camera and batches, and with subtree those of its whole
    // subtree: the stamped layers only, or both layers for a static clear.
    void clear(EntityId e, bool subtree, bool static_clear = false);
    // Empties the scene: every entity but the root, every part, the class table. Old ids die.
    void reset();
    // The person hides an entity (and its subtree) in the view.
    void set_visible(EntityId e, bool visible);

    bool set_classes(const std::vector<ClassEntry>& table, bool static_item = true, std::string* error = nullptr);
    const std::vector<ClassEntry>& classes() const;
    // The class table texture's 65,536 texels: an entry's colour, or palette colour id mod 16.
    void class_texels(uint32_t out[65536]) const;
    std::string class_label(uint32_t id) const;
    uint64_t classes_revision() const { return classes_rev_; }

    // World matrices and bounds of the dirty subtrees. Reading below reflects the last update().
    void update();
    Aabb world_bounds(EntityId subtree) const;              // its batches and cameras' centres
    Aabb scene_bounds() const { return world_bounds(root()); }

    // Reading the effective parts (the stamped layer if present, else the static one).
    const Mat4* transform(EntityId e) const;
    const Pinhole* camera(EntityId e) const;
    const Batch* batch(EntityId e, Primitive p) const;
    bool batch_static(EntityId e, Primitive p) const;       // the effective batch is the static layer
    size_t batch_count() const;                             // entity and primitive pairs that hold one
    const Mat4& world(EntityId e) const;
    // False when the entity's world pose is not finite, or the person hid it or an ancestor.
    bool drawable(EntityId e) const;
    bool pose_finite(EntityId e) const;
    // The camera entity whose pixel space e lives in (its nearest ancestor holding a camera),
    // or an invalid id.
    EntityId pixel_space_of(EntityId e) const;
    // For an entity in a camera's pixel space: the camera's frame (its world matrix times the
    // convention's change to x right, y down, z forward), and the product of the 2-D transforms
    // from the camera's child down to e. world(e) = camera_frame * pixel_to_plane * pixel_local.
    Mat4 camera_frame(EntityId camera) const;
    Mat4 pixel_to_plane(EntityId camera) const;
    const Mat4& pixel_local(EntityId e) const;
    double plane_distance(EntityId camera) const;
    // The local bounds of a batch: the producer's, else a sampled scan of its CPU view.
    Aabb batch_bounds(EntityId e, Primitive p) const;

    // Grows with every change of content (entities, parts, visibility, classes), and with every
    // change of a transform or camera respectively.
    uint64_t revision() const { return rev_; }
    uint64_t pose_revision() const { return pose_rev_; }

private:
    template <class T> struct Part {
        std::optional<T> stamped, fixed;
        const T* get() const { return stamped ? &*stamped : fixed ? &*fixed : nullptr; }
        bool any() const { return stamped.has_value() || fixed.has_value(); }
        bool clear(bool all) {
            const bool had = stamped.has_value() || (all && fixed.has_value());
            stamped.reset();
            if (all) fixed.reset();
            return had;
        }
    };
    struct Entity {
        std::string path;
        uint32_t parent = 0;
        Part<Mat4> transform;
        Part<Pinhole> camera;
        std::array<Part<Batch>, kPrimitiveCount> batches;
        bool visible = true;
        // Derived by update().
        bool dirty = true;              // its transform or camera changed
        bool bounds_dirty = true;
        Mat4 world;
        Mat4 pixel_local;
        bool finite = true;
        bool shown = true;              // visible, and every ancestor visible
        uint32_t pixel_camera = 0xffffffffu;
        Aabb own_bounds;                // world bounds of its own batches (3-D, outside pixel space)
        double plane = 0;               // cameras: the plane distance last used
    };

    bool check(EntityId e, std::string* error) const;
    bool in_pixel_space(uint32_t index) const;            // from the parts now, not the last update
    bool subtree_has(uint32_t index, bool cameras, bool batches3d) const;
    Aabb own_batch_bounds(uint32_t index) const;
    void touch_pose(uint32_t index);
    void rebuild_order() const;

    Space space_ = Space::k3D;
    Axis up_ = Axis::PosY;
    std::string units_ = "m";
    std::vector<Entity> ents_;
    std::vector<uint32_t> gens_;        // per index; the root's is gens_[0]
    uint32_t generation_ = 1;
    Part<std::vector<ClassEntry>> classes_;
    uint64_t rev_ = 1, pose_rev_ = 1, classes_rev_ = 1;
    double scene_radius_ = -1;          // of the last update's 3-D bounds
    Aabb scene_bounds_;
    mutable std::vector<uint32_t> order_;
    mutable bool order_dirty_ = true;
};

// The default point size of the points view by count, in logical pixels
// (live_views_draw.cpp, point_size_px).
float default_point_size(uint64_t count);

}  // namespace caliper_scene
