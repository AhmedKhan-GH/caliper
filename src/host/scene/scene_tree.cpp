// scene_tree.cpp: the scene store, its world matrices and its bounds (scene.h).
#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caliper_scene {

using caliper_gfx::VertexFormat;

#define SCENE_RGB(r, g, b) (0xff000000u | ((uint32_t)(b) << 16) | ((uint32_t)(g) << 8) | (uint32_t)(r))
const uint32_t kPalette[16] = {
    SCENE_RGB(31, 119, 180),  SCENE_RGB(255, 127, 14),  SCENE_RGB(44, 160, 44),   SCENE_RGB(214, 39, 40),
    SCENE_RGB(148, 103, 189), SCENE_RGB(140, 86, 75),   SCENE_RGB(227, 119, 194), SCENE_RGB(127, 127, 127),
    SCENE_RGB(188, 189, 34),  SCENE_RGB(23, 190, 207),  SCENE_RGB(174, 199, 232), SCENE_RGB(255, 187, 120),
    SCENE_RGB(152, 223, 138), SCENE_RGB(255, 152, 150), SCENE_RGB(197, 176, 213), SCENE_RGB(196, 156, 148),
};
#undef SCENE_RGB

namespace {

constexpr uint32_t kNone = 0xffffffffu;

bool fail(std::string* error, std::string text) {
    if (error) *error = std::move(text);
    return false;
}

uint32_t format_bytes(VertexFormat f) {
    switch (f) {
        case VertexFormat::F32: return 4;
        case VertexFormat::F32x2: return 8;
        case VertexFormat::F32x3: return 12;
        case VertexFormat::F32x4: return 16;
        case VertexFormat::Unorm8x4: return 4;
        case VertexFormat::U8: return 1;
        case VertexFormat::U32: return 4;
        case VertexFormat::S32: return 4;
        case VertexFormat::U32x2: return 8;
    }
    return 4;
}

uint32_t format_components(VertexFormat f) {
    switch (f) {
        case VertexFormat::F32x2: return 2;
        case VertexFormat::F32x3: return 3;
        case VertexFormat::F32x4: return 4;
        default: return 1;
    }
}

}  // namespace

// ---- Axes, names --------------------------------------------------------------------------------

Vec3 axis_vector(Axis a) {
    switch (a) {
        case Axis::PosX: return {1, 0, 0};
        case Axis::NegX: return {-1, 0, 0};
        case Axis::PosY: return {0, 1, 0};
        case Axis::NegY: return {0, -1, 0};
        case Axis::PosZ: return {0, 0, 1};
        case Axis::NegZ: return {0, 0, -1};
    }
    return {0, 1, 0};
}

const char* axis_name(Axis a) {
    static const char* names[] = {"+x", "-x", "+y", "-y", "+z", "-z"};
    return names[int(a)];
}

bool parse_axis(std::string_view text, Axis* out) {
    for (int i = 0; i < 6; ++i)
        if (text == axis_name(Axis(i))) {
            *out = Axis(i);
            return true;
        }
    return false;
}

const char* primitive_name(Primitive p) {
    switch (p) {
        case Primitive::Points: return "points";
        case Primitive::Lines: return "lines";
        case Primitive::Arrows: return "arrows";
        case Primitive::Boxes: return "boxes";
        case Primitive::Mesh: return "mesh";
        case Primitive::Image: return "image";
    }
    return "?";
}

const char* role_name(Role r) {
    static const char* names[] = {"positions", "colors",  "values",  "class_ids", "radii",
                                  "break_mask", "indices", "normals", "uvs",       "origins",
                                  "vectors",   "centers", "half_sizes", "rotations"};
    return names[int(r)];
}

float default_point_size(uint64_t n) {
    return n > 50000 ? 3.0f : n > 5000 ? 4.0f : n > 500 ? 5.5f : 7.0f;
}

// ---- Bounds and transforms -------------------------------------------------------------------------

void Aabb::add(Vec3 p) {
    if (!finite(p)) return;
    if (!valid) {
        lo[0] = hi[0] = p.x;
        lo[1] = hi[1] = p.y;
        lo[2] = hi[2] = p.z;
        valid = true;
        return;
    }
    for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], p[k]);
        hi[k] = std::max(hi[k], p[k]);
    }
}

void Aabb::merge(const Aabb& b) {
    if (!b.valid) return;
    add(Vec3{b.lo[0], b.lo[1], b.lo[2]});
    add(Vec3{b.hi[0], b.hi[1], b.hi[2]});
}

Aabb transform_aabb(const Mat4& m, const Aabb& box) {
    Aabb out;
    if (!box.valid) return out;
    for (int c = 0; c < 8; ++c) {
        const Vec3 p{(c & 1) ? box.hi[0] : box.lo[0], (c & 2) ? box.hi[1] : box.lo[1],
                     (c & 4) ? box.hi[2] : box.lo[2]};
        out.add(transform_point(m, p));
    }
    return out;
}

Mat4 transform_matrix(const Transform& t) {
    const double* q = t.rotation;
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    const double k = 1.0 / n;   // infinite for a zero quaternion: the matrix is then not finite
    const Mat4 r = rotation_from_quaternion(q[0] * k, q[1] * k, q[2] * k, q[3] * k);
    return translation({t.translation[0], t.translation[1], t.translation[2]}) * r *
           scaling({t.scale[0], t.scale[1], t.scale[2]});
}

Mat4 transform_matrix(const Transform2D& t) {
    Mat4 r;
    const double c = std::cos(t.angle), s = std::sin(t.angle);
    r.at(0, 0) = c;
    r.at(0, 1) = -s;
    r.at(1, 0) = s;
    r.at(1, 1) = c;
    return translation({t.translation[0], t.translation[1], 0}) * r * scaling({t.scale[0], t.scale[1], 1});
}

// ---- Batches ------------------------------------------------------------------------------------------

const BatchArray* Batch::array(Role r) const {
    for (const BatchArray& a : arrays)
        if (a.role == r) return &a;
    return nullptr;
}

uint64_t Batch::count() const {
    switch (primitive) {
        case Primitive::Points:
        case Primitive::Lines:
        case Primitive::Mesh: {
            const BatchArray* a = array(Role::Positions);
            return a ? a->count : 0;
        }
        case Primitive::Arrows: {
            const BatchArray* a = array(Role::Origins);
            return a ? a->count : 0;
        }
        case Primitive::Boxes: {
            const BatchArray* a = array(Role::Centers);
            return a ? a->count : 0;
        }
        case Primitive::Image: return uint64_t(image_width) * image_height;
    }
    return 0;
}

uint32_t Batch::dims() const {
    const BatchArray* a = array(Role::Positions);
    if (!a) a = array(Role::Origins);
    if (!a) a = array(Role::Centers);
    if (!a) return 0;
    return a->format == VertexFormat::F32x2 ? 2 : a->format == VertexFormat::F32x3 ? 3 : 0;
}

namespace {

struct RoleRule {
    Role role;
    bool required;
    bool may_be_one;   // a count of 1 means one value for all
};

bool check_format(const BatchArray& a, uint32_t dims, std::string* error) {
    const std::string name = role_name(a.role);
    auto want = [&](bool ok, const char* what) {
        return ok ? true : fail(error, name + " must be " + what);
    };
    switch (a.role) {
        case Role::Positions:
        case Role::Origins:
        case Role::Vectors:
        case Role::Centers:
        case Role::HalfSizes:
            return want(a.format == (dims == 2 ? VertexFormat::F32x2 : VertexFormat::F32x3),
                        dims == 2 ? "f32 (N, 2) here, a 2-D space" : "f32 (N, 3) here, a 3-D space");
        case Role::Colors:
            return want(a.format == VertexFormat::Unorm8x4 || a.format == VertexFormat::U32,
                        "4 bytes an item, packed RGBA");
        case Role::Values: return want(a.format == VertexFormat::F32 || a.format == VertexFormat::U32, "f32");
        case Role::ClassIds: return want(a.format == VertexFormat::S32 || a.format == VertexFormat::U32, "i32");
        case Role::Radii: return want(a.format == VertexFormat::F32, "f32");
        case Role::BreakMask:
            return want(a.format == VertexFormat::U8 || a.format == VertexFormat::U32,
                        "4 bytes a vertex whose first byte is the flag");
        case Role::Indices: return want(a.format == VertexFormat::S32 || a.format == VertexFormat::U32, "i32 rows");
        case Role::Normals: return want(a.format == VertexFormat::F32x3, "f32 (V, 3)");
        case Role::Uvs: return want(a.format == VertexFormat::F32x2, "f32 (V, 2)");
        case Role::Rotations: return want(a.format == VertexFormat::F32x4, "f32 (N, 4) quaternions");
    }
    return true;
}

}  // namespace

static bool validate_batch(const Batch& b, uint32_t want_dims, std::string* error) {
    const char* prim = primitive_name(b.primitive);
    if (int(b.primitive) < 1 || int(b.primitive) > kPrimitiveCount) return fail(error, "unknown primitive");
    if (b.arrays.size() > 8) return fail(error, std::string(prim) + ": a batch holds at most 8 arrays");
    // Roles allowed and required per primitive.
    std::vector<RoleRule> rules;
    switch (b.primitive) {
        case Primitive::Points:
            rules = {{Role::Positions, true, false}, {Role::Colors, false, true}, {Role::Values, false, true},
                     {Role::ClassIds, false, true}, {Role::Radii, false, true}};
            break;
        case Primitive::Lines:
            rules = {{Role::Positions, true, false}, {Role::Colors, false, true}, {Role::Values, false, true}};
            if (b.line_mode == LineMode::Strips) rules.push_back({Role::BreakMask, true, false});
            if (b.line_mode == LineMode::Edges) rules.push_back({Role::Indices, true, false});
            break;
        case Primitive::Arrows:
            rules = {{Role::Origins, true, false}, {Role::Vectors, true, false}, {Role::Colors, false, true},
                     {Role::Values, false, true}};
            break;
        case Primitive::Boxes:
            rules = {{Role::Centers, true, false}, {Role::HalfSizes, true, false}, {Role::Rotations, false, true},
                     {Role::Colors, false, true}, {Role::ClassIds, false, true}};
            break;
        case Primitive::Mesh:
            rules = {{Role::Positions, true, false}, {Role::Indices, false, false}, {Role::Normals, false, false},
                     {Role::Colors, false, true}, {Role::Values, false, true}, {Role::Uvs, false, false}};
            break;
        case Primitive::Image: break;
    }
    bool seen[16] = {};
    for (const BatchArray& a : b.arrays) {
        const std::string name = std::string(prim) + " " + role_name(a.role);
        const auto rule = std::find_if(rules.begin(), rules.end(), [&](const RoleRule& r) { return r.role == a.role; });
        if (rule == rules.end()) return fail(error, std::string(prim) + " takes no " + role_name(a.role));
        if (seen[int(a.role)]) return fail(error, name + " is given twice");
        seen[int(a.role)] = true;
        if (!a.buffer) return fail(error, name + " has no buffer");
        if (a.offset % 4 != 0) return fail(error, name + ": the byte offset must be a multiple of 4");
        if (a.stride % 4 != 0 || a.stride > 2048)
            return fail(error, name + ": the stride must be a multiple of 4 and at most 2,048 bytes");
        const uint32_t elem = format_bytes(a.format) *
                              (a.role == Role::Indices ? (b.primitive == Primitive::Mesh ? 3u : 2u) : 1u);
        if (a.role == Role::Indices && a.stride != elem)
            return fail(error, name + ": index rows are packed, " + std::to_string(elem) + " bytes apart");
        if (a.role == Role::BreakMask && a.stride < 4)
            return fail(error, name + ": the mask takes 4 bytes a vertex");
        if (a.stride != 0 && a.role != Role::BreakMask && a.stride < elem)
            return fail(error, name + ": the stride is smaller than one row");
        if (a.stride == 0 && a.count > 1) return fail(error, name + ": stride 0 holds one value for all");
        if (!check_format(a, want_dims, error)) {
            if (error) *error = std::string(prim) + " " + *error;
            return false;
        }
    }
    for (const RoleRule& r : rules)
        if (r.required && !seen[int(r.role)])
            return fail(error, std::string(prim) + " needs " + role_name(r.role));
    if (b.array(Role::Colors) && b.array(Role::Values))
        return fail(error, std::string(prim) + ": colors and values together are refused");

    const uint64_t n = b.count();
    auto per_item = [&](Role role, uint64_t want, const char* what) -> bool {
        const BatchArray* a = b.array(role);
        if (!a) return true;
        const auto rule = std::find_if(rules.begin(), rules.end(), [&](const RoleRule& r) { return r.role == role; });
        if (a->count == want || (rule->may_be_one && a->count == 1)) return true;
        return fail(error, std::string(prim) + " " + role_name(role) + " has " + std::to_string(a->count) +
                               " rows, and needs " + std::to_string(want) + " (" + what + ")" +
                               (rule->may_be_one ? " or 1" : ""));
    };
    switch (b.primitive) {
        case Primitive::Points:
            if (n > kMaxItems) return fail(error, "points: at most 4,194,304 points in a batch");
            for (Role r : {Role::Colors, Role::Values, Role::ClassIds, Role::Radii})
                if (!per_item(r, n, "one per point")) return false;
            break;
        case Primitive::Lines: {
            if (n > kMaxItems) return fail(error, "lines: at most 4,194,304 vertices in a batch");
            if (b.line_mode == LineMode::Edges) {
                const uint64_t e = b.array(Role::Indices)->count;
                if (e > kMaxItems) return fail(error, "lines: at most 4,194,304 index-pair edges in a batch");
                const uint64_t want = b.per_edge ? e : n;
                if (!per_item(Role::Colors, want, b.per_edge ? "one per edge" : "one per vertex")) return false;
                if (!per_item(Role::Values, want, b.per_edge ? "one per edge" : "one per vertex")) return false;
            } else {
                if (!per_item(Role::Colors, n, "one per vertex")) return false;
                if (!per_item(Role::Values, n, "one per vertex")) return false;
                if (b.line_mode == LineMode::Strips && !per_item(Role::BreakMask, n, "one per vertex")) return false;
            }
            break;
        }
        case Primitive::Arrows:
            if (n > kMaxShapes) return fail(error, "arrows: at most 1,048,576 arrows in a batch");
            if (!per_item(Role::Vectors, n, "one per arrow")) return false;
            for (Role r : {Role::Colors, Role::Values})
                if (!per_item(r, n, "one per arrow")) return false;
            break;
        case Primitive::Boxes:
            if (n > kMaxShapes) return fail(error, "boxes: at most 1,048,576 boxes in a batch");
            if (!per_item(Role::HalfSizes, n, "one per box")) return false;
            for (Role r : {Role::Rotations, Role::Colors, Role::ClassIds})
                if (!per_item(r, n, "one per box")) return false;
            if (b.array(Role::Rotations) && want_dims != 3) return fail(error, "boxes: rotations are 3-D only");
            break;
        case Primitive::Mesh: {
            if (n > kMaxItems) return fail(error, "mesh: at most 4,194,304 vertices in a batch");
            if (const BatchArray* t = b.array(Role::Indices); t && t->count > kMaxItems)
                return fail(error, "mesh: at most 4,194,304 triangles in a batch");
            int sources = (b.array(Role::Colors) ? 1 : 0) + (b.array(Role::Values) ? 1 : 0) +
                          (b.array(Role::Uvs) ? 1 : 0);
            if (sources > 1) return fail(error, "mesh: one of colors, values and uvs");
            if (b.array(Role::Uvs) && !b.texture) return fail(error, "mesh: uvs need a texture");
            for (Role r : {Role::Normals, Role::Colors, Role::Values, Role::Uvs})
                if (!per_item(r, n, "one per vertex")) return false;
            break;
        }
        case Primitive::Image:
            if (!b.texture) return fail(error, "image: no texture");
            if (b.image_width < 1 || b.image_height < 1 || b.image_width > kMaxImageSide ||
                b.image_height > kMaxImageSide)
                return fail(error, "image: each side is 1 to 8,192 pixels");
            if (!(b.opacity >= 0.0f && b.opacity <= 1.0f)) return fail(error, "image: opacity is from 0 to 1");
            break;
    }
    if (!b.labels.empty()) {
        if (b.primitive != Primitive::Points && b.primitive != Primitive::Boxes)
            return fail(error, std::string(prim) + " takes no labels");
        if (b.labels.size() > kMaxLabelsPerBatch) return fail(error, "at most 4,096 labels in a batch");
        if (b.labels.size() != 1 && b.labels.size() != n)
            return fail(error, "labels: one per item, or one for all");
        for (const std::string& l : b.labels)
            if (l.size() > kMaxLabelBytes) return fail(error, "a label is at most 63 bytes");
    }
    // The index clamp and the edge gather read these arrays through storage bindings, whose
    // offsets WebGPU aligns to 256 bytes and whose rows the gather reads packed.
    auto storage = [&](Role role, uint32_t packed) -> bool {
        const BatchArray* a = b.array(role);
        if (!a || a->count <= 1) return true;
        if (a->offset % kStorageOffsetAlign != 0)
            return fail(error, std::string(prim) + " " + role_name(role) +
                                   ": read through a storage binding, so its byte offset is a multiple of 256");
        if (packed && a->stride != packed)
            return fail(error, std::string(prim) + " " + role_name(role) + ": the edge gather reads packed rows of " +
                                   std::to_string(packed) + " bytes");
        return true;
    };
    if (b.primitive == Primitive::Mesh && !storage(Role::Indices, 0)) return false;
    if (b.primitive == Primitive::Lines && b.line_mode == LineMode::Edges) {
        if (!storage(Role::Indices, 0) || !storage(Role::Positions, want_dims * 4)) return false;
        if (!b.per_edge && (!storage(Role::Colors, 4) || !storage(Role::Values, 4))) return false;
    }
    if (b.primitive != Primitive::Image && b.dims() != want_dims && n > 0)
        return fail(error, std::string(prim) + ": " + std::to_string(want_dims) + "-D data is needed here");
    if (!(b.width_px > 0.0f) || !(b.size_px >= 0.0f)) return fail(error, std::string(prim) + ": sizes are positive");
    return true;
}

// ---- The tree --------------------------------------------------------------------------------------------

SceneTree::SceneTree(Space space) : SceneTree(space, space == Space::k3D ? Axis::PosY : Axis::NegY, "m") {}

SceneTree::SceneTree(Space space, Axis up, std::string_view units) : space_(space) {
    Entity root;
    root.parent = kNone;
    ents_.push_back(std::move(root));
    gens_.push_back(generation_);
    if (!set_up(up)) up_ = space == Space::k3D ? Axis::PosY : Axis::NegY;
    if (!set_units(units)) units_ = "m";
}

bool SceneTree::set_up(Axis up, std::string* error) {
    if (space_ == Space::k2D && up != Axis::NegY && up != Axis::PosY)
        return fail(error, "a 2-D scene's up is \"-y\" or \"+y\"");
    if (up_ != up) ++rev_;
    up_ = up;
    return true;
}

bool SceneTree::set_units(std::string_view units, std::string* error) {
    if (units.size() > kMaxUnitsBytes) return fail(error, "a unit note is at most 15 bytes");
    units_ = std::string(units);
    return true;
}

bool SceneTree::valid(EntityId e) const {
    return e.index < ents_.size() && gens_[e.index] == e.generation;
}

bool SceneTree::check(EntityId e, std::string* error) const {
    return valid(e) ? true : fail(error, "no such entity (an id from before a reset, or never made)");
}

EntityId SceneTree::entity_at(uint32_t index) const {
    return index < ents_.size() ? EntityId{index, gens_[index]} : EntityId{};
}

const std::string& SceneTree::path(EntityId e) const {
    static const std::string none;
    return valid(e) ? ents_[e.index].path : none;
}

EntityId SceneTree::parent(EntityId e) const {
    if (!valid(e) || e.index == 0) return {};
    return entity_at(ents_[e.index].parent);
}

EntityId SceneTree::find(std::string_view path) const {
    if (path.empty()) return root();
    for (uint32_t i = 1; i < ents_.size(); ++i)
        if (ents_[i].path == path) return entity_at(i);
    return {};
}

EntityId SceneTree::ensure(std::string_view path, std::string* error) {
    std::string why;
    if (!valid_path(path, &why)) {
        fail(error, why);
        return {};
    }
    if (EntityId e = find(path)) return e;
    // The missing ancestors, nearest first.
    std::vector<std::string_view> missing{path};
    std::string_view p = parent_path(path);
    uint32_t parent = 0;
    while (!p.empty()) {
        if (EntityId e = find(p)) {
            parent = e.index;
            break;
        }
        missing.push_back(p);
        p = parent_path(p);
    }
    if (ents_.size() + missing.size() > kMaxEntities) {
        fail(error, "a scene holds at most 1,024 entities, the root included");
        return {};
    }
    for (auto it = missing.rbegin(); it != missing.rend(); ++it) {
        Entity ent;
        ent.path = std::string(*it);
        ent.parent = parent;
        ents_.push_back(std::move(ent));
        gens_.push_back(generation_);
        parent = uint32_t(ents_.size() - 1);
    }
    order_dirty_ = true;
    ++rev_;
    return entity_at(parent);
}

const std::vector<uint32_t>& SceneTree::path_order() const {
    if (order_dirty_) rebuild_order();
    return order_;
}

void SceneTree::rebuild_order() const {
    order_.resize(ents_.size());
    for (uint32_t i = 0; i < ents_.size(); ++i) order_[i] = i;
    // std::string compares bytes as unsigned char, so this is the byte order of the paths; the
    // root's empty path comes first.
    std::sort(order_.begin(), order_.end(),
              [&](uint32_t a, uint32_t b) { return ents_[a].path < ents_[b].path; });
    order_dirty_ = false;
}

bool SceneTree::in_pixel_space(uint32_t index) const {
    for (uint32_t p = ents_[index].parent; p != kNone; p = ents_[p].parent)
        if (ents_[p].camera.any()) return true;
    return false;
}

bool SceneTree::subtree_has(uint32_t index, bool cameras, bool batches3d) const {
    const std::string& base = ents_[index].path;
    for (uint32_t j = 1; j < ents_.size(); ++j) {
        if (j == index) continue;
        const std::string& pj = ents_[j].path;
        const bool below = index == 0 || (pj.size() > base.size() && pj.compare(0, base.size(), base) == 0 &&
                                          pj[base.size()] == '/');
        if (!below) continue;
        if (cameras && ents_[j].camera.any()) return true;
        if (batches3d)
            for (const auto& part : ents_[j].batches) {
                for (const std::optional<Batch>* layer : {&part.stamped, &part.fixed})
                    if (*layer && (*layer)->primitive != Primitive::Image && (*layer)->dims() == 3) return true;
            }
    }
    return false;
}

void SceneTree::touch_pose(uint32_t index) {
    ents_[index].dirty = true;
    ents_[index].bounds_dirty = true;
    ++pose_rev_;
    ++rev_;
}

bool SceneTree::set_transform(EntityId e, const Mat4& m, bool static_item, std::string* error) {
    if (!check(e, error)) return false;
    if (e.index == 0) return fail(error, "the root holds no entity transform");
    Part<Mat4>& part = ents_[e.index].transform;
    (static_item ? part.fixed : part.stamped) = m;
    touch_pose(e.index);
    return true;
}

bool SceneTree::set_transform(EntityId e, const double m[16], bool static_item, std::string* error) {
    return set_transform(e, Mat4::from(m), static_item, error);
}

bool SceneTree::set_transform(EntityId e, const float m[16], bool static_item, std::string* error) {
    return set_transform(e, Mat4::from(m), static_item, error);
}

bool SceneTree::set_camera(EntityId e, const Pinhole& c, bool static_item, std::string* error) {
    if (!check(e, error)) return false;
    if (space_ == Space::k2D) return fail(error, "a 2-D scene holds no camera");
    if (e.index == 0) return fail(error, "the root holds no camera");
    Entity& ent = ents_[e.index];
    for (const auto& part : ent.batches)
        if (part.any()) return fail(error, "an entity with primitives holds no camera");
    if (in_pixel_space(e.index)) return fail(error, "a camera's pixel space holds no camera");
    if (!ent.camera.any() && subtree_has(e.index, true, true))
        return fail(error, "a camera's children live in its 2-D pixel space, and this entity's subtree "
                           "holds 3-D data or a camera");
    if (!(c.fx > 0 && c.fy > 0) || !std::isfinite(c.fx) || !std::isfinite(c.fy) || !std::isfinite(c.cx) ||
        !std::isfinite(c.cy))
        return fail(error, "a camera's focal lengths are finite and above 0, its principal point finite");
    if (c.width < 1 || c.height < 1) return fail(error, "a camera's image size is at least 1 x 1");
    if (!(c.plane_distance >= 0) || !std::isfinite(c.plane_distance))
        return fail(error, "a camera's plane distance is finite and not negative");
    (static_item ? ent.camera.fixed : ent.camera.stamped) = c;
    touch_pose(e.index);
    order_dirty_ = true;
    return true;
}

bool SceneTree::set_batch(EntityId e, const Batch& b, bool static_item, std::string* error) {
    if (!check(e, error)) return false;
    Entity& ent = ents_[e.index];
    if (ent.camera.any()) return fail(error, "an entity that holds a camera holds no primitive");
    const bool pixel = in_pixel_space(e.index);
    const uint32_t dims = (space_ == Space::k2D || pixel) ? 2 : 3;
    if (!validate_batch(b, dims, error)) return false;
    Part<Batch>& part = ent.batches[primitive_slot(b.primitive)];
    if (!part.any() && batch_count() >= kMaxBatches) return fail(error, "a scene holds at most 2,048 batches");
    (static_item ? part.fixed : part.stamped) = b;
    ent.bounds_dirty = true;
    ++rev_;
    return true;
}

void SceneTree::clear(EntityId e, bool subtree, bool static_clear, uint32_t parts) {
    if (!valid(e)) return;
    const std::string base = ents_[e.index].path;
    bool changed = false;
    for (uint32_t j = 0; j < ents_.size(); ++j) {
        const std::string& pj = ents_[j].path;
        const bool hit = j == e.index ||
                         (subtree && (e.index == 0 || (pj.size() > base.size() &&
                                                       pj.compare(0, base.size(), base) == 0 && pj[base.size()] == '/')));
        if (!hit) continue;
        Entity& ent = ents_[j];
        bool pose = (parts & kClearTransform) && ent.transform.clear(static_clear);
        pose = ((parts & kClearCamera) && ent.camera.clear(static_clear)) || pose;
        bool batches = false;
        for (int k = 1; k <= kPrimitiveCount; ++k)
            if (parts & clear_bit(Primitive(k)))
                batches = ent.batches[primitive_slot(Primitive(k))].clear(static_clear) || batches;
        if (pose) touch_pose(j);
        if (batches) ent.bounds_dirty = true;
        changed = changed || pose || batches;
    }
    if (changed) {
        ++rev_;
        order_dirty_ = true;
    }
}

void SceneTree::reset() {
    ++generation_;
    ents_.resize(1);
    gens_.assign(1, generation_);
    Entity root;
    root.parent = kNone;
    ents_[0] = std::move(root);
    if (classes_.any()) ++classes_rev_;   // the texels change only when a table goes
    classes_.stamped.reset();
    classes_.fixed.reset();
    ++rev_;
    ++pose_rev_;
    scene_radius_ = -1;
    scene_bounds_ = Aabb{};
    order_dirty_ = true;
}

void SceneTree::set_visible(EntityId e, bool visible) {
    if (!valid(e) || ents_[e.index].visible == visible) return;
    ents_[e.index].visible = visible;
    ++rev_;
}

bool SceneTree::set_classes(const std::vector<ClassEntry>& table, bool static_item, std::string* error) {
    if (table.size() > kMaxClasses) return fail(error, "a class table has at most 4,096 entries");
    for (const ClassEntry& c : table) {
        if (c.id > kMaxClassId) return fail(error, "a class id is 0 to 65,535");
        if (c.label.size() > kMaxLabelBytes) return fail(error, "a class label is at most 63 bytes");
    }
    (static_item ? classes_.fixed : classes_.stamped) = table;
    ++classes_rev_;
    ++rev_;
    return true;
}

const std::vector<ClassEntry>& SceneTree::classes() const {
    static const std::vector<ClassEntry> none;
    const auto* t = classes_.get();
    return t ? *t : none;
}

void SceneTree::class_texels(uint32_t out[65536]) const {
    for (uint32_t id = 0; id < 65536; ++id) out[id] = kPalette[id % 16];
    for (const ClassEntry& c : classes()) out[c.id] = c.rgba;
}

std::string SceneTree::class_label(uint32_t id) const {
    for (const ClassEntry& c : classes())
        if (c.id == id) return c.label;
    return std::to_string(id);
}

// ---- Reading -------------------------------------------------------------------------------------------------

const Mat4* SceneTree::transform(EntityId e) const { return valid(e) ? ents_[e.index].transform.get() : nullptr; }
const Pinhole* SceneTree::camera(EntityId e) const { return valid(e) ? ents_[e.index].camera.get() : nullptr; }

const Batch* SceneTree::batch(EntityId e, Primitive p) const {
    if (!valid(e) || int(p) < 1 || int(p) > kPrimitiveCount) return nullptr;
    return ents_[e.index].batches[primitive_slot(p)].get();
}

bool SceneTree::batch_static(EntityId e, Primitive p) const {
    if (!valid(e)) return false;
    const auto& part = ents_[e.index].batches[primitive_slot(p)];
    return !part.stamped && part.fixed;
}

size_t SceneTree::batch_count() const {
    size_t n = 0;
    for (const Entity& ent : ents_)
        for (const auto& part : ent.batches) n += part.any() ? 1 : 0;
    return n;
}

const Mat4& SceneTree::world(EntityId e) const {
    static const Mat4 id;
    return valid(e) ? ents_[e.index].world : id;
}

bool SceneTree::drawable(EntityId e) const {
    return valid(e) && ents_[e.index].finite && ents_[e.index].shown;
}

bool SceneTree::pose_finite(EntityId e) const { return valid(e) && ents_[e.index].finite; }

EntityId SceneTree::pixel_space_of(EntityId e) const {
    if (!valid(e) || ents_[e.index].pixel_camera == kNone) return {};
    return entity_at(ents_[e.index].pixel_camera);
}

const Mat4& SceneTree::pixel_local(EntityId e) const {
    static const Mat4 id;
    return valid(e) ? ents_[e.index].pixel_local : id;
}

double SceneTree::plane_distance(EntityId camera) const {
    const Pinhole* c = this->camera(camera);
    if (!c) return 0;
    if (c->plane_distance > 0) return c->plane_distance;
    const double r = scene_bounds_.valid && scene_bounds_.radius() > 0 ? scene_bounds_.radius() : 1.0;
    return r / 20.0;
}

Mat4 SceneTree::camera_frame(EntityId camera) const {
    const Pinhole* c = this->camera(camera);
    Mat4 m = world(camera);
    if (c && c->convention == Convention::Rub) m = m * scaling({1, -1, -1});
    return m;
}

Mat4 SceneTree::pixel_to_plane(EntityId camera) const {
    const Pinhole* c = this->camera(camera);
    if (!c) return Mat4{};
    const double d = plane_distance(camera);
    Mat4 m;
    m.at(0, 0) = d / c->fx;
    m.at(0, 3) = -c->cx * d / c->fx;
    m.at(1, 1) = d / c->fy;
    m.at(1, 3) = -c->cy * d / c->fy;
    m.at(2, 2) = d / c->fx;   // z of pixel-space content is 0; kept invertible for the normal matrix
    m.at(2, 3) = d;
    return m;
}

Aabb SceneTree::batch_bounds(EntityId e, Primitive p) const {
    const Batch* b = batch(e, p);
    Aabb out;
    if (!b) return out;
    if (b->bounds.valid) return b->bounds;
    auto scan = [&](const BatchArray* a, const BatchArray* plus, const BatchArray* extent) {
        if (!a || !a->cpu || a->count == 0) return;
        const uint32_t comps = format_components(a->format);
        const uint64_t n = a->count;
        const uint64_t step = n > (4u << 20) ? (n + (4u << 20) - 1) / (4u << 20) : 1;
        auto row = [&](const BatchArray* arr, uint64_t i, double v[3]) {
            const auto* base = static_cast<const uint8_t*>(arr->cpu) + (arr->stride ? i * arr->stride : 0);
            float f[3] = {0, 0, 0};
            std::memcpy(f, base, std::min<uint32_t>(comps, 3) * sizeof(float));
            v[0] = f[0];
            v[1] = f[1];
            v[2] = comps >= 3 ? f[2] : 0.0;
        };
        for (uint64_t i = 0; i < n; i += step) {
            double v[3];
            row(a, i, v);
            Vec3 c{v[0], v[1], v[2]};
            if (plus && plus->cpu && i < plus->count) {
                double w[3];
                row(plus, i, w);
                out.add(c);
                out.add(c + Vec3{w[0], w[1], w[2]});
            } else if (extent && extent->cpu) {
                double h[3];
                row(extent, extent->count == 1 ? 0 : i, h);
                Vec3 hv{std::fabs(h[0]), std::fabs(h[1]), std::fabs(h[2])};
                if (b->array(Role::Rotations)) {
                    const double l = length(hv);
                    hv = {l, l, comps >= 3 ? l : 0.0};
                }
                out.add(c - hv);
                out.add(c + hv);
            } else {
                out.add(c);
            }
        }
    };
    switch (p) {
        case Primitive::Points:
        case Primitive::Lines:
        case Primitive::Mesh: scan(b->array(Role::Positions), nullptr, nullptr); break;
        case Primitive::Arrows: scan(b->array(Role::Origins), b->array(Role::Vectors), nullptr); break;
        case Primitive::Boxes: scan(b->array(Role::Centers), nullptr, b->array(Role::HalfSizes)); break;
        case Primitive::Image:
            out.add(Vec3{0, 0, 0});
            out.add(Vec3{double(b->image_width), double(b->image_height), 0});
            break;
    }
    return out;
}

Aabb SceneTree::own_batch_bounds(uint32_t index) const {
    Aabb local;
    const EntityId e = entity_at(index);
    for (int k = 1; k <= kPrimitiveCount; ++k) {
        const Primitive p = Primitive(k);
        if (!batch(e, p)) continue;
        if (p == Primitive::Image && space_ == Space::k3D) continue;   // drawn only in a pixel space
        local.merge(batch_bounds(e, p));
    }
    Aabb out = transform_aabb(ents_[index].world, local);
    if (ents_[index].camera.any()) out.add(transform_point(ents_[index].world, {0, 0, 0}));
    return out;
}

Aabb SceneTree::world_bounds(EntityId subtree) const {
    Aabb out;
    if (!valid(subtree)) return out;
    const std::string& base = ents_[subtree.index].path;
    for (uint32_t j = 0; j < ents_.size(); ++j) {
        const Entity& ent = ents_[j];
        if (ent.pixel_camera != kNone || !ent.finite || !ent.shown) continue;
        const std::string& pj = ent.path;
        const bool below = subtree.index == 0 || j == subtree.index ||
                           (pj.size() > base.size() && pj.compare(0, base.size(), base) == 0 && pj[base.size()] == '/');
        if (below) out.merge(ent.own_bounds);
    }
    return out;
}

// ---- update(): one forward pass over the dirty subtrees ------------------------------------------------------

void SceneTree::update() {
    const size_t n = ents_.size();
    std::vector<char> changed(n, 0);
    // Visibility and pixel spaces follow the parents, which come first.
    for (uint32_t i = 0; i < n; ++i) {
        Entity& ent = ents_[i];
        if (i == 0) {
            ent.shown = ent.visible;
            ent.pixel_camera = kNone;
            continue;
        }
        const Entity& par = ents_[ent.parent];
        ent.shown = par.shown && ent.visible;
        const uint32_t pc = par.camera.any() ? ent.parent : par.pixel_camera;
        if (pc != ent.pixel_camera) {
            ent.pixel_camera = pc;
            ent.dirty = true;
            ent.bounds_dirty = true;
        }
    }
    // Pass 1: the entities outside every pixel space.
    for (uint32_t i = 0; i < n; ++i) {
        Entity& ent = ents_[i];
        if (ent.pixel_camera != kNone) continue;
        const bool parent_changed = i > 0 && changed[ent.parent];
        if (!ent.dirty && !parent_changed) continue;
        const Mat4* local = ent.transform.get();
        if (i == 0) {
            ent.world = Mat4{};
            ent.finite = true;
        } else {
            ent.world = local ? ents_[ent.parent].world * *local : ents_[ent.parent].world;
            ent.finite = ents_[ent.parent].finite && finite(ent.world);
        }
        ent.pixel_local = Mat4{};
        ent.bounds_dirty = true;
        changed[i] = 1;
    }
    // Bounds of the 3-D content, and the scene's.
    for (uint32_t i = 0; i < n; ++i) {
        Entity& ent = ents_[i];
        if (ent.pixel_camera != kNone || !ent.bounds_dirty) continue;
        ent.own_bounds = ent.finite ? own_batch_bounds(i) : Aabb{};
        ent.bounds_dirty = false;
    }
    scene_bounds_ = world_bounds(root());
    scene_radius_ = scene_bounds_.valid ? scene_bounds_.radius() : 0;
    // A camera whose plane distance moved (it follows the bounds by default) moves its pixel space.
    for (uint32_t i = 1; i < n; ++i) {
        Entity& ent = ents_[i];
        if (!ent.camera.any()) continue;
        const double d = plane_distance(entity_at(i));
        if (d != ent.plane) {
            ent.plane = d;
            changed[i] = 1;
        }
    }
    // Pass 2: the pixel spaces, parents first.
    for (uint32_t i = 1; i < n; ++i) {
        Entity& ent = ents_[i];
        if (ent.pixel_camera == kNone) continue;
        if (!ent.dirty && !changed[ent.parent]) continue;
        const Mat4* local = ent.transform.get();
        const Mat4 base = ent.parent == ent.pixel_camera ? Mat4{} : ents_[ent.parent].pixel_local;
        ent.pixel_local = local ? base * *local : base;
        const EntityId cam = entity_at(ent.pixel_camera);
        ent.world = camera_frame(cam) * pixel_to_plane(cam) * ent.pixel_local;
        ent.finite = ents_[ent.parent].finite && finite(ent.world);
        ent.own_bounds = Aabb{};
        ent.bounds_dirty = false;
        changed[i] = 1;
    }
    for (Entity& ent : ents_) ent.dirty = false;
}

}  // namespace caliper_scene
