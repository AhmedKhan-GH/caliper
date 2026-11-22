#pragma once
// caliper_draw: the draw half of a program in a viewer
// (design/programmable-graphics.md, Render Graph, Draw State, Safety of
// Draws, Picking and Text).
//
// One DrawRuntime runs the draw half of one program, an instance of a piece,
// in one scene view, on that view's backend of the graphics layer
// (gfx/gfx.h). It decodes the piece's manifest and its files from a PROGRAM
// message, checks the piece's hash against the pin, runs the checks of the
// sandbox before it makes any program or pipeline, composes each pass with
// programs/pieces.h, creates the pipelines, allocates the resources, the draw
// buffers and the offscreen targets, and records the passes of one frame in
// their declared order into a frame of their own, so that the layer's GPU
// time of a submit is the program's time. The app and the browser module
// compile this same file: on Metal the composed WGSL goes through caliper-wgsl
// with its bounds checks to MSL, and on WebGPU it goes to the device as it is.
//
// The scene view owns the camera, the host target (color, ID and depth),
// the pick readback, the labels and the step cursor; it gives each frame's
// values in FrameInputs. Every function runs on the view's frame thread.
// Params that follow controls or state reach it as the packed bytes of the
// struct Params; a piece reads no state variable by name.
//
// Safety (design/programmable-graphics.md, Safety of Draws): the host cannot
// stop a pass on the GPU, thus the runtime never waits for one. The host
// reports each submit's GPU time with frame_done() and the age of the oldest
// pending one with frame_pending(). A program over draw.frame_ms on 30 frames
// in a row records one more frame with each pass in a submit of its own, and
// then stops with program.draw_over_time, naming its costliest pass; a frame
// still pending after 1 s stops it at once with program.draw_timeout; a
// resource, draw buffer or target over draw.memory_mb stops it with
// program.draw_over_memory. A stopped program releases its objects and draws
// nothing; the other programs of the panel keep drawing. Nothing here throws.
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

// The checks and the helpers of programs/pieces.h that the runtime runs
// before it makes anything (Check Order 2, 4 and 5, and the layouts). The
// app and the browser give pieces(); a test may give others, to reach the
// render graph in a tree whose piece checks are not built.
struct DrawChecks {
    Verdict (*manifest)(const caliper_builder::spec::Piece&) = nullptr;
    Verdict (*source)(std::string_view, std::size_t) = nullptr;
    caliper_builder::programs::Resolved (*resolve)(const caliper_builder::spec::Piece&) = nullptr;
    Verdict (*layout)(const caliper_builder::spec::Layout&, caliper_builder::programs::ElementLayout*) = nullptr;
    std::string (*hash)(const caliper_builder::spec::Piece&) = nullptr;
    // The functions of programs/pieces.h.
    static DrawChecks pieces();
};

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
    // The attachment and the epoch of the resources: a change (a new attachment, or a version
    // at an earlier step than the newest of the same attachment) zeroes every draw buffer.
    std::uint64_t epoch = 0;
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
    // The host's colormaps, caliper_colormaps of the prelude: RGBA8, 256 texels wide, one row
    // for each colormap id of the graphics layer (gfx/colormap_tables.h), sampled. Absent, the
    // runtime binds grey ramps.
    caliper_gfx::Texture colormaps;
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
    int pass = -1;                   // the index of the pass that the error names, or -1
    std::string entry;               // that pass's entry point (its fragment for a render pass)
};

class DrawRuntime {
public:
    // `webgpu`: the backend takes WGSL as it is (the browser module); else
    // the runtime translates each composed module with caliper-wgsl to MSL.
    DrawRuntime(caliper_gfx::Backend& gfx, bool webgpu, DrawChecks checks = DrawChecks::pieces());
    ~DrawRuntime();
    DrawRuntime(const DrawRuntime&) = delete;
    DrawRuntime& operator=(const DrawRuntime&) = delete;

    // Decodes the piece and the instance from the JSON metadata "declaration"
    // of a PROGRAM message and its sections "common" and "draw", and checks
    // the piece's hash against the pin (piece.hash_mismatch). The app decodes
    // the JSON of its own spec with the same function, so that both viewers
    // read one form.
    //
    // The declaration holds "id" (or "program"), "piece" (the pin: name, version, hash),
    // "manifest" (the keys of piece.toml), "bind" and "panel". The sections may hold the
    // author's files or the files with the imports resolved (the lib modules before
    // common.wgsl, and "::" rewritten to "__"); the decoder returns the author's files, whose
    // hash is the pin's. The pin's hash covers derive.wgsl too, thus a viewer checks it only
    // with that section (`derive_wgsl`); without it every hash differs.
    static Verdict decode(std::string_view declaration_json, std::string_view common_wgsl, std::string_view draw_wgsl,
                          caliper_builder::spec::Piece* piece, Instance* instance,
                          const DrawChecks& checks = DrawChecks::pieces(), std::string_view derive_wgsl = {});

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

    // Records the program's passes of one frame into frames that it appends to
    // `frames`: one frame, or one for each pass while it measures its passes.
    // The scene view submits each on its own, in order, after the scene's
    // frame. False when the program draws nothing in this frame (not
    // installed, stopped, a resource without a version, or a refusal, which
    // stops it). Applies the reset rules of the draw buffers.
    bool encode(std::vector<caliper_gfx::Frame>* frames, const FrameInputs& in, const HostTarget& target);
    // Reports the GPU time of one submitted frame, in the order of encode()'s
    // frames: from the layer's timestamps, or the time from the submit until
    // the work was done where the backend has none. Over draw.frame_ms on 30
    // frames in a row stops the program with program.draw_over_time.
    void frame_done(double gpu_ms);
    // The oldest submitted frame of the program that is not done has waited
    // this long; after 1 s the program stops at once with program.draw_timeout.
    void frame_pending(double waited_ms);
    // The layer refused a submitted frame (Backend::last_error()): the program
    // stops with program.layer_refused; the panel's other programs go on.
    void frame_refused(const std::string& why);
    // Stops the program after one more frame that measures each pass, naming
    // the costliest: the scene view's rule for the programs of one panel
    // together over 16.7 ms a frame. `message` says why.
    void request_stop(const char* code, const std::string& message);
    // The camera that the piece selects, the instance and the pick id.
    caliper_builder::spec::CameraMode camera() const;
    const Instance& instance() const;
    std::uint32_t pick_id() const;

    // The item map's fields of `item` in the version of the last frame, or
    // nothing when the program has no map or no such element.
    std::optional<PickedItem> pick(std::uint32_t item) const;
    // The labels of the text table in the version of the last frame.
    std::vector<Label> labels() const;

    DrawStatus status() const;

    // The runtime's state, defined in draw_runtime_impl.h and internal to src/host/draw; the
    // tests read it through internals().
    struct Impl;
    const Impl& internals() const { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_draw
