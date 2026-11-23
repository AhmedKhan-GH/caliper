#pragma once
// caliper_draw: the shapes of a scene view drawn by the library pieces
// (design/programmable-graphics.md, Library Pieces).
//
// ScenePieces is the scene view's own feed of draw halves (live/live_scene.h): one program for
// each batch of the scene tree that a library piece can draw, an instance of points@1, lines@1,
// arrows@1, boxes@1, mesh@1, image@1, depth@1 or segmentation@1, whose resources the host writes
// from the part's arrays as the piece's derive half would write them. A spec can give a batch
// another piece (surface@1 over a surface's grid, path@1 over the rows of a path), with its own
// resources and params. The scene core then leaves those batches out of its colour pass and its
// pick pass, and a pick of a piece's ID maps back to the batch's entity, primitive and item, so
// that the pick's event is the one the built-in draw gives.
//
// The view keeps the built-in draw of every batch beside it: a batch that no piece can draw yet,
// such as one above the piece's capacity, a textured mesh or the overlay of a camera looked
// through, a piece that its runtime refused or stopped, and every batch while the view draws
// with ShapeDraw::builtin, draws with the scene core as before. Each such fallback is counted by
// its kind.
//
// The params of a batch, its styles, its pose and the scene's class table are read at each sync
// from the scene tree; the arrays are read once, when the part arrives, and every size of a
// part's array is checked against its shape and dtype before a byte is read. Everything runs on
// the scene view's frame thread.
#include "draw/draw_host.h"
#include "net/wire.h"
#include "scene/camera.h"
#include "scene/compile.h"
#include "scene/pick.h"
#include "scene/scene.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caliper_draw {

// A piece that draws one batch in place of the one the converter chooses, with resources that
// the caller writes: the spec's surface (surface@1 over its grid) and path (path@1 over its rows).
// `piece` names a library piece of version 1; params left out take the manifest's defaults.
struct PieceBatch {
    std::string piece;
    std::vector<std::pair<std::string, double>> params;
    struct Resource {
        std::string id;
        std::uint32_t count = 0;
        std::vector<std::uint8_t> bytes;
    };
    std::vector<Resource> resources;
};

// The selected rows of one batch, which the highlight of a selection draws over the dimmed picture.
struct HighlightRows {
    std::string path;
    caliper_scene::Primitive primitive = caliper_scene::Primitive::Points;
    std::vector<std::int64_t> rows;   // sorted, unique, each below the batch's count
};

// Why a shape keeps the built-in draw: an input that the piece does not take (another dtype or
// shape), a part above the piece's capacity, a textured mesh, a stage program, a param outside
// the piece's range, a pixel space the piece cannot place, or a piece that is missing or stopped.
// A test holds this list (tests/gfx/gfx_scene_view.cpp), thus a new kind fails it until it is
// listed there.
enum class Fallback : std::uint8_t {
    points_shape,
    lines_shape,
    arrows_shape,
    boxes_shape,
    mesh_shape,
    image_shape,
    depth_shape,
    mask_shape,
    points_capacity,
    segments_capacity,
    arrows_capacity,
    boxes_capacity,
    mesh_vertices_capacity,
    mesh_triangles_capacity,
    image_pixels_capacity,
    depth_pixels_capacity,
    mask_pixels_capacity,
    textured_mesh,
    stage,
    point_size,
    line_width,
    box_width,
    depth_cloud_moved,
    mask_class,
    no_arrays,
    stopped,
    no_piece,
    count,
    none = 255,
};
const char* fallback_name(Fallback kind);
// The names of every kind, in the order of Fallback; ShapeCounts::fallbacks counts by these names.
std::vector<std::string> fallback_kinds();

// The shapes that pieces draw and those that fall back to the built-in draw, by kind.
struct ShapeCounts {
    std::uint32_t pieces = 0;
    std::uint32_t builtin = 0;
    std::map<std::string, std::uint32_t> fallbacks;
    std::uint32_t marks = 0;   // the highlight's programs
    std::uint32_t received = 0;   // the shapes whose resources came in the piece form, drawn as they came
};

class ScenePieces final : public DrawFeed {
public:
    ScenePieces();
    ~ScenePieces() override;

    // One part of one layer of an entity arrived (SceneView::apply), after the view stored `batch`:
    // its arrays become the piece's resources. An update of the piece form brings the resources
    // themselves in `received` (one section each, named by its id), which the piece draws as they are.
    void take(const std::string& path, caliper_scene::Primitive p, bool fixed, const caliper_net::wire::SceneUpdate& u,
              const caliper_scene::Batch& batch, const std::vector<caliper_net::wire::Section>* received = nullptr);
    // The part of that layer is gone.
    void drop(const std::string& path, caliper_scene::Primitive p, bool fixed);
    void clear();
    // A piece that draws the batch of (path, p) in either layer, or none again.
    void set_override(const std::string& path, caliper_scene::Primitive p, PieceBatch batch);
    void clear_override(const std::string& path, caliper_scene::Primitive p);

    // Before a frame, after the tree's update(): the programs from the tree's effective batches,
    // with each batch's params and pose. `through` is the camera looked through, `stage` whether a
    // batch draws a stage program, and `three_d` whether the scene has depth.
    struct SyncState {
        caliper_scene::EntityId through;
        bool three_d = true;
        std::function<bool(const std::string& path, caliper_scene::Primitive p)> stage;
    };
    void sync(const caliper_scene::SceneTree& tree, const SyncState& state);
    // Whether a piece draws the batch (e, p) in this frame; the scene core leaves it out.
    bool draws(caliper_scene::EntityId e, caliper_scene::Primitive p) const;
    // Whether the batch (e, p) lies in the pixel space of the camera looked through and keeps the
    // built-in draw: the view leaves it out of its first frame and draws it with the scene core
    // among the shapes' pieces, in the scene core's order of that flat layer. The batches of one
    // segment draw together just before the program late_segments()[segment] ("" after the last
    // shape's program).
    bool late(caliper_scene::EntityId e, caliper_scene::Primitive p) const;
    std::size_t late_segment(caliper_scene::EntityId e, caliper_scene::Primitive p) const;
    const std::vector<std::string>& late_segments() const { return late_before_; }
    // Whether the built-in batch of the flat layer that `builtin` hit draws over the piece that
    // `piece` hit, both in that layer.
    bool late_over(const caliper_scene::Hit& builtin, const caliper_scene::Hit& piece) const;
    // Whether frustum@1 draws the frustum of the camera entity e in this frame.
    bool draws_frustum(caliper_scene::EntityId e) const;
    bool any() const;
    // The matrices of the frame: the scene core's, with which each program gets its batch's model.
    void set_camera(const caliper_scene::CameraMatrices& cm);
    // Whether `pick_id` is one of these programs', and the hit of its item as the scene core gives
    // it: the batch's entity, primitive and item, an image's pixel, and the position from the
    // depth at the window's texel through the inverse of the frame's camera.
    bool owns(std::uint32_t pick_id) const;
    bool hit(const caliper_scene::FrameHit& fh, const caliper_scene::PickWindow& w, caliper_scene::Hit* out) const;
    ShapeCounts counts() const;
    // The view installed a new runtime for `program`: the next frame gives it every resource.
    void installed(const std::string& program);

    // The highlight of a selection, after sync(): the picture dimmed toward the ground `ground`
    // (image@1 over the whole target, 0.72 of the ground over 0.28 of the picture, as the points
    // view dims), then the selected rows of each batch in a second instance of its piece, with
    // the batch's params and matrix. marks() draws them into the colour alone, without depth and
    // ID, so that none is hidden and a pick reads the shapes under them. False, with no marks,
    // when a selected batch has no piece in this frame: the view then keeps the built-in
    // highlight. No rows clear the marks. The rows drawn are counted in `highlighted`.
    bool set_highlight(const std::vector<HighlightRows>& rows, std::uint32_t ground, std::uint32_t width,
                       std::uint32_t height, std::uint64_t* highlighted);
    // The feed of the highlight's programs, which the view draws after every other program.
    DrawFeed& marks();

    // DrawFeed
    std::uint64_t revision() const override { return revision_; }
    std::vector<PanelProgram> programs() const override;
    void frame(DrawRuntime& runtime, FrameInputs* in) override;
    void status(const DrawRuntime& runtime) override;

    // The library piece `name` of version 1, with its files and hash; null when the build has none.
    static const caliper_builder::spec::Piece* library_piece(const std::string& name);

    struct Converted;
    struct Program;
    class Marks;

private:
    using Key = std::pair<std::string, int>;   // path, primitive
    const caliper_scene::SceneTree* tree_ = nullptr;
    std::map<std::pair<Key, bool>, std::shared_ptr<Converted>> parts_;   // by (path, primitive), layer
    std::map<Key, std::shared_ptr<Converted>> overrides_;
    std::map<Key, std::size_t> late_;      // overlay batches that keep the built-in, by segment
    std::vector<std::string> late_before_;   // each segment's next program, "" for none
    std::shared_ptr<Converted> frustum_;   // frustum@1's resource: the identity, element 0
    std::vector<std::unique_ptr<Program>> programs_;     // in the tree's path order
    std::map<std::string, std::uint32_t> pick_ids_;      // by program id, stable while it lives
    std::map<std::string, int> refused_;                 // program id: its generation when it stopped
    std::uint32_t next_pick_ = 0x40000000u;
    std::uint64_t revision_ = 1;
    std::uint64_t generation_ = 0;
    caliper_scene::CameraMatrices cm_;
    ShapeCounts counts_;
    // The highlight's programs (marks()): the dim, then the selected rows of each batch.
    std::unique_ptr<Marks> marks_feed_;
    std::vector<std::unique_ptr<Program>> marks_;
    std::uint64_t marks_revision_ = 1;
    std::uint32_t next_mark_pick_ = 0x60000000u;
    void frame_of(Program& pr, DrawRuntime& runtime, FrameInputs* in);
};

}  // namespace caliper_draw
