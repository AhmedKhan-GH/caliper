#pragma once
// caliper_builder::spec: the value types of pieces, the unit of programmable
// graphics (design/programmable-graphics.md, Pieces and Piece Manifest).
//
// A piece is a folder: piece.toml (its manifest) and up to three WGSL files,
// common.wgsl, derive.wgsl and draw.wgsl. Its derive half is compute passes
// that run in the training script at caliper.step(), read tensors and publish
// named, typed resources. Its draw half is compute and render passes that run
// in the viewer every frame, read the published resources, keep buffers
// between frames and write the host's color, ID and depth targets. A piece
// declares typed inputs, parameters that become controls, its resources,
// the lib modules it imports, the pieces it depends on, an optional sample
// input and its hash. Library pieces, workspace pieces and inline pieces of
// a spec have this one format.
//
// A program of a spec (program.h) is an instance of a piece: it pins the
// piece by name, version and hash, binds the piece's inputs, sets its
// parameters and names the scene panel of its draw half.
//
// Everything here is a plain value with inline helpers and no dependency but
// the standard library, because the composer (programs/pieces.h), the
// producer runtime, the app's viewer and the browser module all read these
// types. The spec codec reads and writes them (spec.h, format.h), and
// programs/pieces.h checks them. Nothing here throws.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::spec {

// Limits of pieces (design/programmable-graphics.md, Limits). The browser's
// WebGPU defaults set the binding counts.
namespace piece_limits {
inline constexpr std::size_t kNameMax = 32;                      // [a-z][a-z0-9_]{0,31}
inline constexpr std::uint32_t kVersionMax = 1'000'000;
inline constexpr std::size_t kDescriptionMax = 256;
inline constexpr std::size_t kInputs = 8;                        // tensors and buffers that a piece reads
inline constexpr std::size_t kParams = 16;
inline constexpr std::size_t kResources = 8;                     // published or private resources of a piece
inline constexpr std::size_t kImports = 8;                       // lib modules
inline constexpr std::size_t kDepends = 8;
inline constexpr std::size_t kFields = 16;                       // fields of one layout
inline constexpr std::uint32_t kStrideMax = 256;                 // bytes of one element
inline constexpr std::uint32_t kCountMax = 33'554'432;           // elements of one buffer: 128 MiB of 4-byte elements
inline constexpr std::uint32_t kTextureSideMax = 4096;           // pixels of one texture side
inline constexpr std::uint64_t kResourceBytesMax = 64ull << 20;  // bytes of one private resource or draw buffer
// Bytes of one published resource, 128 MiB: the largest storage binding of WebGPU's default limits, which
// the app's graphics layer and the browser module both hold to, and the bound of a version that goes to a
// viewer in RESOURCE parts (CALIPER_NET_RESOURCE_VERSION_MAX_BYTES). The step history keeps at most this
// much of one resource as well.
inline constexpr std::uint64_t kPublishedResourceBytesMax = 128ull << 20;
// Bytes of all published resources of a piece together, 256 MiB: one version of each fits the step
// history's 256 MiB of resources. A version above 8 MiB goes to a viewer in parts of 8 MiB, each while
// the viewer's queue holds at most 8 MiB and 64 KiB, half of the 16 MiB control queue of either sender
// (python/caliper/_programs.py, PUBLISHED_BYTES_MAX, PART_BYTES and RESOURCE_BACKLOG_BYTES).
inline constexpr std::uint64_t kPublishedBytesMax = 256ull << 20;
inline constexpr std::uint32_t kKeepDefault = 256;               // kept versions of a resource in the step history
inline constexpr std::uint32_t kKeepMax = 4096;
inline constexpr std::size_t kDerivePasses = 8;
inline constexpr std::size_t kDrawPasses = 16;
inline constexpr std::size_t kDrawBuffers = 4;                   // buffers that the draw half keeps between frames
inline constexpr std::size_t kTargets = 4;                       // offscreen targets
inline constexpr std::size_t kFileBytes = 65536;                 // each WGSL file
inline constexpr std::size_t kSampleBytes = 1u << 20;            // all sample files of a piece
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
}  // namespace piece_limits

// The only file names of a piece folder besides piece.toml and sample/.
inline constexpr std::array<const char*, 3> kPieceFiles{"common.wgsl", "derive.wgsl", "draw.wgsl"};

// What an input binds: a tensor of the script (derive half), or a published
// buffer resource of another program (draw half).
enum class InputKind { tensor, buffer };
// The type of a parameter's field in the generated struct Params.
enum class ParamType { f32, i32, u32 };
// Where a pinned piece comes from. A workspace piece with the name of a
// library piece replaces it only when the program says workspace.
enum class PieceFrom { library, workspace, inline_piece };
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
// The host camera that a piece selects.
enum class CameraMode { orbit, pan_zoom, none };
// When the draw half's buffers return to zero besides an install.
enum class BufferReset { install, cursor, resource };
// Where the host puts a label: `placed` lets the placer choose one of eight
// places around the anchor, as for the other labels of the panel; `left`,
// `right` and `center` pin the label at its anchor, with the start, the end or
// the middle of its text there and the text centered on it vertically.
enum class TextAlign { placed, left, right, center };

inline const char* input_kind_name(InputKind k) { return k == InputKind::tensor ? "tensor" : "buffer"; }
inline const char* param_type_name(ParamType t) { return t == ParamType::f32 ? "f32" : t == ParamType::i32 ? "i32" : "u32"; }
inline const char* piece_from_name(PieceFrom f) {
    return f == PieceFrom::library ? "library" : f == PieceFrom::workspace ? "workspace" : "inline";
}
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
inline const char* text_align_name(TextAlign a) {
    return a == TextAlign::placed ? "placed" : a == TextAlign::left ? "left" : a == TextAlign::right ? "right" : "center";
}

namespace detail {
template <typename E, std::size_t N>
std::optional<E> from_name(std::string_view s, const E (&all)[N], const char* (*name)(E)) {
    for (E e : all)
        if (s == name(e)) return e;
    return std::nullopt;
}
}  // namespace detail

inline std::optional<InputKind> input_kind_from_name(std::string_view s) {
    static const InputKind all[] = {InputKind::tensor, InputKind::buffer};
    return detail::from_name(s, all, input_kind_name);
}
inline std::optional<ParamType> param_type_from_name(std::string_view s) {
    static const ParamType all[] = {ParamType::f32, ParamType::i32, ParamType::u32};
    return detail::from_name(s, all, param_type_name);
}
inline std::optional<PieceFrom> piece_from_from_name(std::string_view s) {
    static const PieceFrom all[] = {PieceFrom::library, PieceFrom::workspace, PieceFrom::inline_piece};
    return detail::from_name(s, all, piece_from_name);
}
inline std::optional<ResourceKind> resource_kind_from_name(std::string_view s) {
    static const ResourceKind all[] = {ResourceKind::buffer, ResourceKind::texture};
    return detail::from_name(s, all, resource_kind_name);
}
inline std::optional<PassKind> pass_kind_from_name(std::string_view s) {
    static const PassKind all[] = {PassKind::compute, PassKind::render};
    return detail::from_name(s, all, pass_kind_name);
}
inline std::optional<DrawTopology> draw_topology_from_name(std::string_view s) {
    static const DrawTopology all[] = {DrawTopology::triangle_list, DrawTopology::triangle_strip,
                                       DrawTopology::line_list, DrawTopology::line_strip, DrawTopology::point_list};
    return detail::from_name(s, all, draw_topology_name);
}
inline std::optional<DrawBlend> draw_blend_from_name(std::string_view s) {
    static const DrawBlend all[] = {DrawBlend::opaque, DrawBlend::alpha, DrawBlend::additive};
    return detail::from_name(s, all, draw_blend_name);
}
inline std::optional<DrawDepth> draw_depth_from_name(std::string_view s) {
    static const DrawDepth all[] = {DrawDepth::none, DrawDepth::test, DrawDepth::write};
    return detail::from_name(s, all, draw_depth_name);
}
inline std::optional<CameraMode> camera_mode_from_name(std::string_view s) {
    static const CameraMode all[] = {CameraMode::orbit, CameraMode::pan_zoom, CameraMode::none};
    return detail::from_name(s, all, camera_mode_name);
}
inline std::optional<BufferReset> buffer_reset_from_name(std::string_view s) {
    static const BufferReset all[] = {BufferReset::install, BufferReset::cursor, BufferReset::resource};
    return detail::from_name(s, all, buffer_reset_name);
}
inline std::optional<TextAlign> text_align_from_name(std::string_view s) {
    static const TextAlign all[] = {TextAlign::placed, TextAlign::left, TextAlign::right, TextAlign::center};
    return detail::from_name(s, all, text_align_name);
}

// One field of a layout, written "name: type" in the manifest. The types are
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
    std::optional<std::string> element;    // "f32", "u32", "vec4<f32>", ...
    std::optional<std::string> type_name;  // the key `struct`: a WGSL struct name
    std::vector<Field> fields;
};
inline bool operator==(const Layout& a, const Layout& b) {
    return a.element == b.element && a.type_name == b.type_name && a.fields == b.fields;
}

// One entry of `inputs`. A tensor input is a read-only array of `dtype` in
// the derive half, with its shape in the generated uniform `shapes`; its rank
// must be from rank_min to rank_max. A buffer input is a read-only array of
// `layout` in the draw half, bound to another program's published buffer
// resource with the same layout. An optional input may stay unbound: a tensor
// input then reads as an empty array with the shape (0, 0, 0, 0), and a buffer
// input as an empty array with the count 0.
struct PieceInput {
    std::string name;                        // the WGSL name of the generated binding
    InputKind kind = InputKind::tensor;
    std::string dtype = "f32";               // tensor: f32, i32 or u32
    std::uint32_t rank_min = 1;              // tensor
    std::uint32_t rank_max = 8;
    Layout layout;                           // buffer
    std::optional<std::string> title;
    bool optional = false;                   // a program may leave it unbound
};
inline bool operator==(const PieceInput& a, const PieceInput& b) {
    return a.name == b.name && a.kind == b.kind && a.dtype == b.dtype && a.rank_min == b.rank_min &&
           a.rank_max == b.rank_max && a.layout == b.layout && a.title == b.title && a.optional == b.optional;
}

// One entry of `params`: a field of the generated struct Params, which both
// halves read as params.<name>. An instance gives it a number or a binding
// reference; without one, the app places a control with the range.
struct PieceParam {
    std::string name;
    ParamType type = ParamType::f32;
    double default_value = 0;                // the key `default`
    std::optional<double> min;
    std::optional<double> max;
    std::optional<std::string> title;
};
inline bool operator==(const PieceParam& a, const PieceParam& b) {
    return a.name == b.name && a.type == b.type && a.default_value == b.default_value && a.min == b.min &&
           a.max == b.max && a.title == b.title;
}

// One entry of `resources`. A buffer holds `count` elements of its layout;
// with `counted`, the derive half also gets the counter <id>_count, and the
// valid elements are the first min(counter, count). A texture is rgba8unorm,
// `width` by `height`, which the derive half writes as packed array<u32> and
// the draw half samples as texture_2d<f32>. A published resource is an
// output that the viewer and other pieces read.
struct Resource {
    std::string id;                          // unique among the piece's inputs, resources and draw buffers
    ResourceKind kind = ResourceKind::buffer;
    Layout layout;                           // buffer
    std::uint32_t count = 0;                 // buffer: elements
    bool counted = false;                    // buffer: the derive half counts the valid elements
    std::uint32_t width = 0, height = 0;     // texture
    bool publish = true;                     // false: a private resource of the derive half, never sent
    bool clear = false;                      // true: zero before each run of the derive passes
    std::optional<std::uint32_t> keep;       // kept versions in the step history, kKeepDefault when absent
};
inline bool operator==(const Resource& a, const Resource& b) {
    return a.id == b.id && a.kind == b.kind && a.layout == b.layout && a.count == b.count && a.counted == b.counted &&
           a.width == b.width && a.height == b.height && a.publish == b.publish && a.clear == b.clear &&
           a.keep == b.keep;
}

// One compute pass of the derive half. Passes run in their order in one
// submit. `writes` names the resources that the pass changes; a published
// resource that no pass names is sent after the first run only.
struct DerivePass {
    std::string entry;
    std::array<std::uint32_t, 3> workgroups{1, 1, 1};
    std::vector<std::string> writes;
};
inline bool operator==(const DerivePass& a, const DerivePass& b) {
    return a.entry == b.entry && a.workgroups == b.workgroups && a.writes == b.writes;
}

// The table `derive`; its source is derive.wgsl. Its limits are those of
// design/gpu-programs.md for one run of all passes.
struct Derive {
    std::vector<DerivePass> passes;          // 1 to kDerivePasses
    std::optional<double> time_ms;
    std::optional<std::uint32_t> memory_mb;
    std::optional<std::uint32_t> every;
};
inline bool operator==(const Derive& a, const Derive& b) {
    return a.passes == b.passes && a.time_ms == b.time_ms && a.memory_mb == b.memory_mb && a.every == b.every;
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
// `instances_of`, the valid count of a buffer resource or input at the cursor.
// With `indices`, a render pass draws the valid elements of a u32 buffer
// resource or input as an index buffer: each index is the vertex_index of the
// vertex entry, and each instance draws all of them.
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
    std::optional<std::string> instances_of;             // a buffer resource or buffer input
    std::optional<std::uint32_t> vertices;               // render: 3 for the full-screen triangle
    std::optional<std::string> indices;                  // render: a u32 buffer resource or input of indices
    DrawTopology topology = DrawTopology::triangle_list;
    std::optional<DrawBlend> blend;                      // opaque on the host target, additive offscreen
    std::optional<DrawDepth> depth;                      // write on the host target with a vertex entry, else none
    bool pick = false;                                   // the fragment also writes the ID at location 1
};
inline bool operator==(const DrawPass& a, const DrawPass& b) {
    return a.kind == b.kind && a.entry == b.entry && a.workgroups == b.workgroups && a.writes == b.writes &&
           a.vertex == b.vertex && a.fragment == b.fragment && a.target == b.target && a.reads == b.reads &&
           a.instances == b.instances && a.instances_of == b.instances_of && a.vertices == b.vertices &&
           a.topology == b.topology && a.blend == b.blend && a.depth == b.depth && a.pick == b.pick &&
           a.indices == b.indices;
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
// the labels when they overlap. With `strings_from`, the strings come from a
// published buffer resource or buffer input instead: each valid element is one
// string of UTF-8 bytes, ended by its stride (at most kStringBytes) or its
// first zero byte.
struct TextTable {
    std::string anchors;                     // a buffer resource
    std::string position = "position";
    std::string string = "string";
    std::optional<std::string> priority;
    std::vector<std::string> strings;        // at most kStrings, each at most kStringBytes
    std::optional<std::string> strings_from; // a buffer of at most kStrings elements of at most kStringBytes
    TextAlign align = TextAlign::placed;
};
inline bool operator==(const TextTable& a, const TextTable& b) {
    return a.anchors == b.anchors && a.position == b.position && a.string == b.string && a.priority == b.priority &&
           a.strings == b.strings && a.strings_from == b.strings_from && a.align == b.align;
}

// The table `draw`; its source is draw.wgsl.
struct Draw {
    CameraMode camera = CameraMode::orbit;
    std::vector<DrawBuffer> buffers;
    std::vector<DrawTarget> targets;
    std::vector<DrawPass> passes;            // 1 to kDrawPasses
    std::optional<PickMap> pick;
    std::optional<TextTable> text;
    std::optional<double> frame_ms;          // kFrameMsDefault when absent
    std::optional<std::uint32_t> memory_mb;  // kDrawMemoryMbDefault when absent
};
inline bool operator==(const Draw& a, const Draw& b) {
    return a.camera == b.camera && a.buffers == b.buffers && a.targets == b.targets && a.passes == b.passes &&
           a.pick == b.pick && a.text == b.text && a.frame_ms == b.frame_ms && a.memory_mb == b.memory_mb;
}

// A piece that this one depends on: its output feeds one of this piece's
// buffer inputs, or the piece is drawn with it.
struct PieceDependency {
    std::string name;
    std::uint32_t version = 1;
};
inline bool operator==(const PieceDependency& a, const PieceDependency& b) {
    return a.name == b.name && a.version == b.version;
}

// A sample of one input for a preview: a file under sample/ in the piece
// folder with the raw little-endian elements, and its shape.
struct SampleInput {
    std::string file;                        // sample/<input>.bin
    std::vector<std::uint32_t> shape;
};
inline bool operator==(const SampleInput& a, const SampleInput& b) { return a.file == b.file && a.shape == b.shape; }

// A piece: its manifest (piece.toml) and its WGSL files by name. An inline
// piece of a spec holds the same keys in a [[piece]] table, with the files
// in its table `files`.
struct Piece {
    std::string name;                        // [a-z][a-z0-9_]{0,31}
    std::uint32_t version = 1;
    std::string description;                 // one line
    std::vector<PieceInput> inputs;
    std::vector<PieceParam> params;
    std::vector<Resource> resources;
    std::vector<std::string> imports;        // lib modules, such as "colormap"
    std::vector<PieceDependency> depends;
    std::optional<Derive> derive;
    std::optional<Draw> draw;
    std::map<std::string, SampleInput> sample;   // by input name
    std::string hash;                        // "sha256:<64 hex>", over the manifest and the resolved WGSL
    std::map<std::string, std::string> files;    // common.wgsl, derive.wgsl and draw.wgsl
};
inline bool operator==(const Piece& a, const Piece& b) {
    return a.name == b.name && a.version == b.version && a.description == b.description && a.inputs == b.inputs &&
           a.params == b.params && a.resources == b.resources && a.imports == b.imports && a.depends == b.depends &&
           a.derive == b.derive && a.draw == b.draw && a.sample == b.sample && a.hash == b.hash && a.files == b.files;
}

// The pin of a piece in a program: its name, version, hash and source.
struct PieceRef {
    std::string name;
    std::uint32_t version = 1;
    std::string hash;                        // "sha256:<64 hex>"
    PieceFrom from = PieceFrom::library;
};
inline bool operator==(const PieceRef& a, const PieceRef& b) {
    return a.name == b.name && a.version == b.version && a.hash == b.hash && a.from == b.from;
}

// The program keys of an instance of a piece, and the top-level key of the
// inline pieces of a spec.
inline constexpr std::array<const char*, 3> kInstanceKeys{"piece", "bind", "panel"};
inline constexpr const char* kPiecesKey = "piece";

}  // namespace caliper_builder::spec
