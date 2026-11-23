#pragma once
// The state of a DrawRuntime (draw_runtime.h), which draw_runtime.cpp (the
// render graph), draw_pick.cpp (the item map) and draw_text.cpp (the labels)
// share. Internal to src/host/draw.
//
// The item map and the text table read `slots`: the slot of a resource that
// the piece's pick or text names keeps a CPU copy of the version that the
// last frame showed (`shown`), with its element layout. The copy of a shown
// version never changes, thus a pick that read an earlier frame keeps it.
#include "draw/draw_runtime.h"

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace caliper_draw {

struct DrawRuntime::Impl {
    caliper_gfx::Backend* gfx = nullptr;
    bool webgpu = false;
    DrawChecks checks;
    DrawStatus status;

    // ---- the installed program
    bool installed = false;
    caliper_builder::spec::Piece piece;
    Instance instance;
    std::uint32_t pick_id = 0;

    // A version of a resource as one frame showed it, for the item map and the text table. It
    // never changes once made: a newer version makes a new one.
    struct Shown {
        std::int64_t step = -1;
        std::uint32_t count = 0;
        std::vector<std::uint8_t> bytes;   // count * stride bytes; empty unless the slot keeps a copy
    };
    // One published resource or buffer input of the draw half: a read-only binding of group 1.
    struct Slot {
        std::string id;                              // the resource's id, or the buffer input's name
        bool texture = false;
        caliper_builder::programs::ElementLayout layout;   // buffers
        std::uint32_t declared_count = 0;            // elements, or a texture's width * height
        std::uint32_t width = 0, height = 0;         // textures
        std::uint64_t declared_bytes = 0;            // count * stride, or width * height * 4
        caliper_gfx::Buffer buffer;                  // buffers: grown up to declared_bytes
        std::uint64_t capacity = 0;
        caliper_gfx::Texture tex;                    // textures
        bool has_version = false;
        std::int64_t step = -1;                      // the version that the next frame shows
        std::uint32_t count = 0;
        // The number of that version among all that set_resource() took, which differs for a
        // version that replaces another at the same step; 0 before the first.
        std::uint64_t serial = 0;
        bool keep_cpu = false;                       // the item map or the text table reads it
        std::vector<std::uint8_t> cpu;               // the bytes of that version, when keep_cpu
        std::uint64_t drawn_serial = 0;              // the version that the last frame showed
        std::shared_ptr<const Shown> shown;          // its copy; null before a frame showed one
    };
    std::vector<Slot> slots;                         // published resources in order, then buffer inputs
    std::uint64_t versions = 0;                      // the versions set_resource() took, for Slot::serial

    // One draw buffer (`draw.buffers`), kept between frames.
    struct DrawBufferSlot {
        std::string id;
        caliper_gfx::Buffer buffer;
        std::uint64_t bytes = 0;                     // count * stride, and at least one element
        std::uint32_t stride = 0;
        caliper_builder::spec::BufferReset reset = caliper_builder::spec::BufferReset::install;
    };
    std::vector<DrawBufferSlot> buffers;

    // One offscreen target (`draw.targets`) at its scale of the host target.
    struct TargetSlot {
        std::string id;
        float scale = 1;
        std::uint32_t clear_rgba = 0;
        caliper_gfx::Texture tex;
        std::uint32_t width = 0, height = 0;
    };
    std::vector<TargetSlot> targets;

    // One pass: its program, its pipelines by host variant (bit 0 depth, bit 1 ID), its
    // bindings and its bind groups.
    struct PassSlot {
        std::string label;
        std::string wgsl, msl;
        std::vector<std::string> names;              // keeps the entry names of `entries` alive
        std::vector<caliper_gfx::EntryDesc> entries;
        std::vector<caliper_gfx::BindingDesc> bindings;
        caliper_gfx::Program program;
        std::array<caliper_gfx::Pipeline, 4> pipelines{};
        std::array<caliper_gfx::BindGroup, 3> groups{};
        std::string vertex, fragment, compute;
    };
    std::vector<PassSlot> passes;

    // The prelude's objects and the uniforms of group 0 and 1.
    caliper_gfx::Buffer view, selection, params, counts;
    std::uint32_t params_bytes = 0, counts_bytes = 0;
    std::vector<std::uint8_t> params_default;
    caliper_gfx::Sampler linear, nearest;
    caliper_gfx::Texture colormaps;                  // the fallback rows, grey ramps
    caliper_gfx::Texture bound_colormaps;            // the host's (HostTarget::colormaps) or the fallback
    bool groups_dirty = true;

    // ---- draw state
    std::uint64_t epoch = 0;
    bool epoch_known = false;
    std::uint64_t frames_since_reset = 0;
    bool first_after_reset = true;
    std::uint64_t memory_bytes = 0;

    // ---- timing: the frames submitted and not reported yet, oldest first; pass -1 is a
    // whole frame, else one pass of a measuring frame.
    struct Expected {
        std::uint64_t frame = 0;
        int pass = -1;
        bool last = false;                           // the last pass of a measuring frame
    };
    std::deque<Expected> expected;
    std::uint64_t frame_no = 0;
    std::deque<double> window;                       // the last 30 whole-frame times
    std::uint32_t over = 0;                          // frames over draw.frame_ms in a row
    bool measure = false;                            // the next frame measures each pass
    std::string measure_code;                        // what a measured frame stops with
    std::string measure_message;
    std::vector<double> measured;                    // per pass of the measuring frame; < 0 none
    // A frame that is not done after kTimeoutMs by the wall clock is late: the program submits
    // no frame until it is done, and is then judged by its GPU time, or, where the backend gives
    // none, by kLateFrames late frames in a row (draw_runtime.cpp).
    bool late = false;
    std::uint32_t late_frames = 0;
    // Frames over kHeavyMs of GPU time in a row, and their GPU time together.
    std::uint32_t heavy_frames = 0;
    double heavy_ms = 0;
};

}  // namespace caliper_draw
