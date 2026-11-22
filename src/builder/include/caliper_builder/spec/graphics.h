#pragma once
// caliper_builder::spec: the value types of programmable graphics
// (design/programmable-graphics.md, Declaration).
//
// A program of the graphics form is one declaration with two halves. Its
// derive half is compute passes that run in the training script at
// caliper.step(), read tensors and publish named, typed resources. Its draw
// half is compute and render passes that run in the viewer every frame, read
// the published resources, keep buffers between frames and write the host's
// color, ID and depth targets. Program (program.h) holds these values beside
// the one-pass form of design/gpu-programs.md; a program uses one form only.
//
// Everything here is a plain value with inline helpers and no dependency but
// the standard library, because the composer (programs/graphics.h), the
// producer runtime, the app's viewer and the browser module all read these
// types. The spec codec reads and writes them; until it does, it refuses the
// keys resources, common, derive and draw with spec.not_built. Nothing here
// throws.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::spec {

// Limits of the graphics form (design/programmable-graphics.md, Limits).
// The browser's WebGPU defaults set the binding counts.
namespace graphics_limits {
inline constexpr std::size_t kResources = 8;                     // published or private resources of a program
inline constexpr std::size_t kFields = 16;                       // fields of one layout
inline constexpr std::uint32_t kStrideMax = 256;                 // bytes of one element
inline constexpr std::uint32_t kCountMax = 4'194'304;            // elements of one buffer
inline constexpr std::uint32_t kTextureSideMax = 4096;           // pixels of one texture side
inline constexpr std::uint64_t kResourceBytesMax = 64ull << 20;  // bytes of one resource, as CALIPER_NET_RESOURCE_MAX_BYTES
inline constexpr std::uint32_t kKeepDefault = 256;               // kept versions of a resource in the step history
inline constexpr std::uint32_t kKeepMax = 4096;
inline constexpr std::size_t kInputs = 8;                        // tensors that the derive half reads
inline constexpr std::size_t kDerivePasses = 8;
inline constexpr std::size_t kDrawPasses = 16;
inline constexpr std::size_t kDrawBuffers = 4;                   // buffers that the draw half keeps between frames
inline constexpr std::size_t kTargets = 4;                       // offscreen targets
inline constexpr std::size_t kStateReads = 16;                   // state variables that the draw half reads
inline constexpr std::size_t kSourceBytes = 65536;               // each of derive.wgsl and draw.wgsl
inline constexpr std::size_t kCommonBytes = 16384;               // common, which both halves include
inline constexpr std::uint32_t kInstancesMax = 4'194'304;
inline constexpr std::uint32_t kVerticesMax = 65'536;            // vertices of one instance
inline constexpr std::uint32_t kWorkgroupsMax = 65'535;          // each of three dispatch counts
inline constexpr double kFrameMsDefault = 4.0;                   // GPU time of the draw half in one frame
inline constexpr double kFrameMsMax = 16.0;
inline constexpr std::uint32_t kDrawMemoryMbDefault = 64;        // resources, draw buffers and targets in the viewer
inline constexpr std::uint32_t kDrawMemoryMbMax = 256;
inline constexpr std::size_t kPickFields = 8;
inline constexpr std::size_t kStrings = 4096;                    // strings of the text table
inline constexpr std::size_t kStringBytes = 64;
inline constexpr std::size_t kLabels = 512;                      // labels drawn in one frame, as place_labels
}  // namespace graphics_limits

// The kind of a resource.
enum class ResourceKind { buffer, texture };
// What a pass of the draw half is.
enum class PassKind { compute, render };
// The primitive topology of a render pass.
enum class DrawTopology { triangle_list, triangle_strip, line_list, line_strip, point_list };
// The blend of a render pass's color output; the ID output never blends.
enum class DrawBlend { opaque, alpha, additive };
// The depth of a render pass into the host target: none, a less-or-equal
// test without a write, or the test with a write. Offscreen targets have no depth.
enum class DrawDepth { none, test, write };
// The host camera that a program selects.
enum class CameraMode { orbit, pan_zoom, none };
// When the draw half's buffers return to zero besides an install.
enum class BufferReset { install, cursor, resource };

inline const char* resource_kind_name(ResourceKind k) { return k == ResourceKind::buffer ? "buffer" : "texture"; }
inline const char* pass_kind_name(PassKind k) { return k == PassKind::compute ? "compute" : "render"; }
inline const char* draw_topology_name(DrawTopology t) {
    switch (t) {
        case DrawTopology::triangle_list: return "triangle_list";
        case DrawTopology::triangle_strip: return "triangle_strip";
        case DrawTopology::line_list: return "line_list";
        case DrawTopology::line_strip: return "line_strip";
        case DrawTopology::point_list: return "point_list";
    }
    return "triangle_list";
}
inline const char* draw_blend_name(DrawBlend b) {
    return b == DrawBlend::opaque ? "opaque" : b == DrawBlend::alpha ? "alpha" : "additive";
}
inline const char* draw_depth_name(DrawDepth d) { return d == DrawDepth::none ? "none" : d == DrawDepth::test ? "test" : "write"; }
inline const char* camera_mode_name(CameraMode c) {
    return c == CameraMode::orbit ? "orbit" : c == CameraMode::pan_zoom ? "pan_zoom" : "none";
}
inline const char* buffer_reset_name(BufferReset r) {
    return r == BufferReset::install ? "install" : r == BufferReset::cursor ? "cursor" : "resource";
}

inline std::optional<ResourceKind> resource_kind_from_name(std::string_view s) {
    if (s == "buffer") return ResourceKind::buffer;
    if (s == "texture") return ResourceKind::texture;
    return std::nullopt;
}
inline std::optional<PassKind> pass_kind_from_name(std::string_view s) {
    if (s == "compute") return PassKind::compute;
    if (s == "render") return PassKind::render;
    return std::nullopt;
}
inline std::optional<DrawTopology> draw_topology_from_name(std::string_view s) {
    for (auto t : {DrawTopology::triangle_list, DrawTopology::triangle_strip, DrawTopology::line_list,
                   DrawTopology::line_strip, DrawTopology::point_list})
        if (s == draw_topology_name(t)) return t;
    return std::nullopt;
}
inline std::optional<DrawBlend> draw_blend_from_name(std::string_view s) {
    for (auto b : {DrawBlend::opaque, DrawBlend::alpha, DrawBlend::additive})
        if (s == draw_blend_name(b)) return b;
    return std::nullopt;
}
inline std::optional<DrawDepth> draw_depth_from_name(std::string_view s) {
    for (auto d : {DrawDepth::none, DrawDepth::test, DrawDepth::write})
        if (s == draw_depth_name(d)) return d;
    return std::nullopt;
}
inline std::optional<CameraMode> camera_mode_from_name(std::string_view s) {
    for (auto c : {CameraMode::orbit, CameraMode::pan_zoom, CameraMode::none})
        if (s == camera_mode_name(c)) return c;
    return std::nullopt;
}
inline std::optional<BufferReset> buffer_reset_from_name(std::string_view s) {
    for (auto r : {BufferReset::install, BufferReset::cursor, BufferReset::resource})
        if (s == buffer_reset_name(r)) return r;
    return std::nullopt;
}

// One field of a layout, written "name: type" in the spec. The types are
// f32, i32, u32, vec2<f32>, vec4<f32>, vec2<u32> and vec4<u32>; vec3 is left
// out because its alignment differs from its size.
struct Field {
    std::string name;
    std::string type;
};
inline bool operator==(const Field& a, const Field& b) { return a.name == b.name && a.type == b.type; }

// The element of a buffer: either one plain type (`element`), or a struct
// (`struct` names it, `fields` lists its fields in order). The composer
// generates the struct in both halves, thus they agree on its bytes.
struct Layout {
    std::optional<std::string> element;  // "f32", "u32", "vec4<f32>", ...
    std::optional<std::string> type_name;  // the key `struct`: a WGSL struct name
    std::vector<Field> fields;
};
inline bool operator==(const Layout& a, const Layout& b) {
    return a.element == b.element && a.type_name == b.type_name && a.fields == b.fields;
}

// One entry of `resources`. A buffer holds `count` elements of its layout;
// with `counted`, the derive half also gets the counter <id>_count, and the
// valid elements are the first min(counter, count). A texture is rgba8unorm,
// `width` by `height`, which the derive half writes as packed array<u32> and
// the draw half samples as texture_2d<f32>.
struct Resource {
    std::string id;                          // unique among the program's resources and draw buffers
    ResourceKind kind = ResourceKind::buffer;
    Layout layout;                           // buffer
    std::uint32_t count = 0;                 // buffer: elements
    bool counted = false;                    // buffer: the derive half counts the valid elements
    std::uint32_t width = 0, height = 0;     // texture
    bool publish = true;                     // false: a private buffer of the derive half, never sent
    bool clear = false;                      // true: zero before each run of the derive passes
    std::optional<std::uint32_t> keep;       // kept versions in the step history, kKeepDefault when absent
};
inline bool operator==(const Resource& a, const Resource& b) {
    return a.id == b.id && a.kind == b.kind && a.layout == b.layout && a.count == b.count && a.counted == b.counted &&
           a.width == b.width && a.height == b.height && a.publish == b.publish && a.clear == b.clear &&
           a.keep == b.keep;
}

// One tensor that the derive half reads, read-only and in place. `type` is
// the element type of the generated array: f32 (default), i32 or u32.
struct DeriveInput {
    std::string var;
    std::string source;                      // live:<name>
    std::optional<std::string> type;
};
inline bool operator==(const DeriveInput& a, const DeriveInput& b) {
    return a.var == b.var && a.source == b.source && a.type == b.type;
}

// One compute pass of the derive half. Passes run in their order in one
// submit; `writes` names the resources that the pass may change.
struct DerivePass {
    std::string entry;
    std::array<std::uint32_t, 3> workgroups{1, 1, 1};
    std::vector<std::string> writes;
};
inline bool operator==(const DerivePass& a, const DerivePass& b) {
    return a.entry == b.entry && a.workgroups == b.workgroups && a.writes == b.writes;
}

// The key `derive`. Its limits are those of design/gpu-programs.md:
// time_ms for all passes of one run, memory_mb, and every.
struct Derive {
    std::string wgsl;                        // at most kSourceBytes, no binding and no directive
    std::vector<DeriveInput> inputs;         // at most kInputs
    std::vector<DerivePass> passes;          // 1 to kDerivePasses
    std::optional<double> time_ms;
    std::optional<std::uint32_t> memory_mb;
    std::optional<std::uint32_t> every;
};
inline bool operator==(const Derive& a, const Derive& b) {
    return a.wgsl == b.wgsl && a.inputs == b.inputs && a.passes == b.passes && a.time_ms == b.time_ms &&
           a.memory_mb == b.memory_mb && a.every == b.every;
}

// One buffer that the draw half keeps between frames (the key `buffers`).
struct DrawBuffer {
    std::string id;
    Layout layout;
    std::uint32_t count = 0;
    BufferReset reset = BufferReset::install;
};
inline bool operator==(const DrawBuffer& a, const DrawBuffer& b) {
    return a.id == b.id && a.layout == b.layout && a.count == b.count && a.reset == b.reset;
}

// One offscreen target: rgba8unorm at `scale` of the host target's size,
// cleared to `clear` at the start of each frame.
struct DrawTarget {
    std::string id;
    float scale = 1.0f;                      // 1, 0.5 or 0.25
    std::array<float, 4> clear{0.0f, 0.0f, 0.0f, 0.0f};
};
inline bool operator==(const DrawTarget& a, const DrawTarget& b) {
    return a.id == b.id && a.scale == b.scale && a.clear == b.clear;
}

// One pass of the draw half. A compute pass has `entry`, `workgroups` and
// `writes` (draw buffers). A render pass has `fragment`, and `vertex` unless
// it is a full-screen triangle; `instances` is a count or, through
// `instances_of`, the valid count of a resource at the cursor.
struct DrawPass {
    PassKind kind = PassKind::render;
    std::string entry;                                   // compute
    std::array<std::uint32_t, 3> workgroups{1, 1, 1};    // compute
    std::vector<std::string> writes;                     // compute: draw buffers
    std::optional<std::string> vertex;                   // render: absent for the full-screen triangle
    std::string fragment;                                // render
    std::string target = "host";                         // render: "host" or an offscreen target
    std::vector<std::string> reads;                      // offscreen targets that the pass samples
    std::uint32_t instances = 1;
    std::optional<std::string> instances_of;             // a buffer resource
    std::optional<std::uint32_t> vertices;               // render: 3 for the full-screen triangle
    DrawTopology topology = DrawTopology::triangle_list;
    std::optional<DrawBlend> blend;                      // opaque on the host target, additive offscreen
    std::optional<DrawDepth> depth;                      // write on the host target with a vertex entry, else none
    bool pick = false;                                   // the fragment also writes the ID at location 1
};
inline bool operator==(const DrawPass& a, const DrawPass& b) {
    return a.kind == b.kind && a.entry == b.entry && a.workgroups == b.workgroups && a.writes == b.writes &&
           a.vertex == b.vertex && a.fragment == b.fragment && a.target == b.target && a.reads == b.reads &&
           a.instances == b.instances && a.instances_of == b.instances_of && a.vertices == b.vertices &&
           a.topology == b.topology && a.blend == b.blend && a.depth == b.depth && a.pick == b.pick;
}

// The item map: a picked item is the index of an element of `resource`, and
// the pick event carries the named `fields` of that element at the cursor.
struct PickMap {
    std::string resource;
    std::vector<std::string> fields;         // 1 to kPickFields fields of the resource's layout
};
inline bool operator==(const PickMap& a, const PickMap& b) { return a.resource == b.resource && a.fields == b.fields; }

// Text that the host draws: one label at each valid element of `anchors`,
// whose `position` field (vec4<f32>, world xyz) places it and whose `string`
// field (u32) indexes `strings`. An optional `priority` field (f32) orders
// the labels when they overlap.
struct TextTable {
    std::string anchors;                     // a buffer resource
    std::string position = "position";
    std::string string = "string";
    std::optional<std::string> priority;
    std::vector<std::string> strings;        // at most kStrings, each at most kStringBytes
};
inline bool operator==(const TextTable& a, const TextTable& b) {
    return a.anchors == b.anchors && a.position == b.position && a.string == b.string && a.priority == b.priority &&
           a.strings == b.strings;
}

// The key `draw`.
struct Draw {
    std::string panel;                       // a scene panel of the spec
    std::string wgsl;                        // at most kSourceBytes, no binding and no directive
    CameraMode camera = CameraMode::orbit;
    std::vector<std::string> state;          // state variables that the draw half reads as `state.<id>`
    std::vector<DrawBuffer> buffers;
    std::vector<DrawTarget> targets;
    std::vector<DrawPass> passes;            // 1 to kDrawPasses
    std::optional<PickMap> pick;
    std::optional<TextTable> text;
    std::optional<double> frame_ms;          // kFrameMsDefault when absent
    std::optional<std::uint32_t> memory_mb;  // kDrawMemoryMbDefault when absent
};
inline bool operator==(const Draw& a, const Draw& b) {
    return a.panel == b.panel && a.wgsl == b.wgsl && a.camera == b.camera && a.state == b.state &&
           a.buffers == b.buffers && a.targets == b.targets && a.passes == b.passes && a.pick == b.pick &&
           a.text == b.text && a.frame_ms == b.frame_ms && a.memory_mb == b.memory_mb;
}

// The graphics form of a program: what Program holds beside its id, title
// and params. A program of this form has no wgsl, entry, inputs, output,
// limits or workgroups of the one-pass form.
struct Graphics {
    std::vector<Resource> resources;
    std::optional<std::string> common;       // WGSL that both halves include: structs, constants, functions
    std::optional<Derive> derive;
    std::optional<Draw> draw;

    bool empty() const { return resources.empty() && !common && !derive && !draw; }
};
inline bool operator==(const Graphics& a, const Graphics& b) {
    return a.resources == b.resources && a.common == b.common && a.derive == b.derive && a.draw == b.draw;
}

// The keys of a program table that belong to the graphics form.
inline constexpr std::array<const char*, 4> kGraphicsKeys{"resources", "common", "derive", "draw"};

}  // namespace caliper_builder::spec
