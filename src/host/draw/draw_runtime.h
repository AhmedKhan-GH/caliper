#pragma once
// caliper_draw: the draw half of a program in a viewer
// (design/programmable-graphics.md, Render Graph, Draw State, Safety of
// Draws, Picking and Text).
//
// One DrawRuntime runs the draw half of one program, an instance of a piece,
// in one scene view, on that view's backend of the graphics layer
// (gfx/gfx.h). It decodes the piece's manifest and its resolved files from a
// PROGRAM message, checks the piece's hash against the pin, composes each
// pass with programs/pieces.h, creates the pipelines, allocates the resources,
// the draw buffers and the offscreen targets, and records the passes of one
// frame in their declared order into a frame of their own, so that the
// layer's GPU time of a submit is the program's time. The app and the
// browser module compile this same file: on Metal the composed WGSL goes
// through caliper-wgsl to MSL, and on WebGPU it goes to the device as it is.
//
// The scene view owns the camera, the host target (color, ID and depth),
// the pick readback, the labels and the step cursor; it gives each frame's
// values in FrameInputs. Every function runs on the view's frame thread.
// Params that follow controls or state reach it as the packed bytes of the
// struct Params; a piece reads no state variable by name.
// Until the draw half is built, install() answers program.not_built and the
// runtime draws nothing. Nothing here throws.
#include "gfx/gfx.h"

#include <caliper_builder/programs/pieces.h>

#include <array>
#include <map>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace caliper_draw {

using caliper_builder::programs::Verdict;

// The instance that a viewer draws: the program's id, its bindings and the
// scene panel of its draw half. The piece holds the manifest and the
// resolved files (common.wgsl and draw.wgsl, imports included).
struct Instance {
    std::string program;
    std::map<std::string, std::string> bind;   // input name to program:<id>/<resource> for a buffer input
    std::string panel;
};

// One version of a published resource: the bytes of a RESOURCE message, or
// a version that the step history kept. The bytes stay the caller's.
struct ResourceVersion {
    std::string resource;            // the resource's id, or a buffer input's name
    std::int64_t step = -1;          // the step of the derive pass that wrote it
    std::uint32_t count = 0;         // valid elements; a texture's width * height
    const std::uint8_t* bytes = nullptr;
    std::size_t size = 0;            // count * stride, or width * height * 4
};

// One entry of the linked selection (CaliperSelected of the prelude).
struct Selected {
    std::uint32_t pick = 0;          // the pick id of the program that wrote it
    std::uint32_t item = 0;
    std::uint32_t flags = 0;         // bit 0: the newest pick
    std::array<std::int32_t, 4> index{-1, -1, -1, -1};
};

// What the scene view gives one frame of a program: the values of
// CaliperView, the selection, and the packed bytes of `params`.
struct FrameInputs {
    std::array<float, 16> clip_from_world{};
    std::array<float, 16> world_from_clip{};
    std::array<float, 16> view_from_world{};
    std::array<float, 4> eye{};
    std::array<float, 4> viewport{};
    std::array<float, 4> pointer{};
    double time_s = 0;               // seconds since the install in this view
    double dt_s = 0;                 // seconds since the previous frame
    float pixel_ratio = 1;
    std::int64_t cursor_step = -1;
    std::int64_t latest_step = -1;
    bool live = true;
    bool cursor_moved = false;       // a command, a key or the transport moved the cursor
    std::vector<Selected> selection; // at most kMaxSelected
    std::vector<std::uint8_t> params;   // the generated struct Params, in the manifest's order
};

// The host target of the scene view, which the program's passes with the
// target "host" write: color RGBA8Unorm, ID RG32Uint (pick_id, item) and
// depth Depth32Float. The ID and the depth may be absent, as in a 2-D panel.
struct HostTarget {
    caliper_gfx::Texture color;
    caliper_gfx::Texture id;
    caliper_gfx::Texture depth;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// A picked element of the item map: the resource's element `item` at the
// step that the frame showed, with the fields that the map names.
struct PickedItem {
    std::string program;
    std::uint32_t item = 0;
    std::int64_t step = -1;
    std::vector<std::pair<std::string, double>> fields;
};

// A label of the text table: a world position, its text and its priority,
// which the scene view places with place_labels() (spec_runtime/label_layout.h).
struct Label {
    std::array<float, 3> position{};
    std::string text;
    float priority = 0;
};

// The state of the program in this view (design/programmable-graphics.md, Status).
enum class DrawState { waiting, compiled, running, refused, stopped };

struct DrawStatus {
    DrawState state = DrawState::waiting;
    double frame_ms = -1;            // the median GPU time of the last 30 frames; < 0 when unknown
    double frame_limit_ms = 0;       // draw.frame_ms
    std::uint64_t memory_bytes = 0;  // resources, draw buffers and targets
    std::uint64_t memory_limit_bytes = 0;
    std::int64_t resource_step = -1; // the step of the resources of the last frame
    std::uint64_t frames = 0;
    Verdict error;                   // the refusal or the stop; `where` names the pass, e.g. "draw.passes[2]"
};

class DrawRuntime {
public:
    // `webgpu`: the backend takes WGSL as it is (the browser module); else
    // the runtime translates each composed module with caliper-wgsl to MSL.
    DrawRuntime(caliper_gfx::Backend& gfx, bool webgpu);
    ~DrawRuntime();
    DrawRuntime(const DrawRuntime&) = delete;
    DrawRuntime& operator=(const DrawRuntime&) = delete;

    // Decodes the piece and the instance from the JSON metadata "declaration"
    // of a PROGRAM message and its sections "common" and "draw", and checks
    // the piece's hash against the pin (piece.hash_mismatch). The app decodes
    // the JSON of its own spec with the same function, so that both viewers
    // read one form.
    static Verdict decode(std::string_view declaration_json, std::string_view common_wgsl, std::string_view draw_wgsl,
                          caliper_builder::spec::Piece* piece, Instance* instance);

    // Checks, composes and compiles each pass, then allocates the draw
    // buffers and the targets. `pick_id` is the program's id in the ID
    // target, which the scene view gives out. A program that is installed
    // again replaces this one and starts with zeroed buffers.
    Verdict install(const caliper_builder::spec::Piece& piece, const Instance& instance, std::uint32_t pick_id);
    // Releases every object of the program.
    void remove();

    // Gives the version of one resource that the next frame shows: the
    // newest version at or before the cursor. A version larger than the
    // resource's declared bytes is program.resource_too_large.
    Verdict set_resource(const ResourceVersion& version);

    // Records the program's passes of one frame into `frame`, which the scene
    // view submits on its own after the scene's frame; false when the program
    // draws nothing in this frame. Applies the reset rules of the draw buffers.
    bool encode(caliper_gfx::Frame& frame, const FrameInputs& in, const HostTarget& target);
    // Reports the GPU time of the program's submit (< 0 when the backend has
    // no timestamps) and whether the frame completed. Over draw.frame_ms on
    // 30 frames in a row, or a frame still pending after 1 s, stops the
    // program with program.draw_over_time or program.draw_timeout.
    void frame_done(double gpu_ms);
    void frame_pending(double waited_ms);

    // The item map's fields of `item` in the version of the last frame, or
    // nothing when the program has no map or no such element.
    std::optional<PickedItem> pick(std::uint32_t item) const;
    // The labels of the text table in the version of the last frame.
    std::vector<Label> labels() const;

    DrawStatus status() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_draw
