#pragma once
// caliper_builder::spec: the value types that spec version 1 adds for GPU
// programs (design/gpu-programs.md, Spec Additions): the top-level arrays
// `program` and `stage`, and the object `provenance`.
//
// A program is WGSL compute source with a declaration of its inputs (live
// tensors of the script by name, read-only), its one output (geometry that a
// scene panel draws as one entity, or pixels that the live view
// program.<id> carries) and its limits. It runs in the script's process on
// the producer runtime. A stage is WGSL that replaces the material or the
// vertex function of one entity of a scene panel, in the viewer. Provenance
// marks a spec from someone else, whose programs and stages run only after a
// person reviewed their source.
//
// Spec::programs, Spec::stages and Spec::provenance hold them (spec.h). The
// spec library reads, writes and validates them; the sandbox
// (caliper_builder/programs/sandbox.h), the producer runtime and the viewer
// check and run them. Everything here is a plain value with inline helpers,
// so that the producer runtime can include this header without the spec
// library. Nothing here throws.
#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <caliper_builder/spec/reference.h>

namespace caliper_builder::spec {

namespace program_limits {
inline constexpr std::size_t kPrograms = 16;                    // `program` items; spec.too_many
// A program's id: a pixels output is the live view `program.<id>`, whose name
// holds at most 63 bytes (CALIPER_NET_NAME_MAX), thus the wire's PROGRAM takes
// an id of at most 55 bytes (CALIPER_NET_PROGRAM_ID_MAX); spec.bad_value.
inline constexpr std::size_t kProgramIdChars = 55;
inline constexpr std::size_t kStages = 32;                      // `stage` items; spec.too_many
inline constexpr std::size_t kProgramSourceBytes = 64 * 1024;   // a program's `wgsl`
inline constexpr std::size_t kStageSourceBytes = 16 * 1024;     // a stage's `wgsl`
inline constexpr std::size_t kInputs = 8;                       // a program's `inputs`
inline constexpr std::uint32_t kMaxCount = 1000000;             // geometry `max_count`
inline constexpr std::uint32_t kPixelsSide = 2048;              // pixels `width` and `height`
inline constexpr double kTimeMsDefault = 4.0;                   // `limits.time_ms`: GPU time of one pass
inline constexpr double kTimeMsMax = 50.0;
inline constexpr std::uint32_t kMemoryMbDefault = 64;           // `limits.memory_mb`: outputs and bound tensors
inline constexpr std::uint32_t kMemoryMbMax = 512;
inline constexpr std::uint32_t kEveryMax = 10000;               // `limits.every`, and the cadence N
inline constexpr std::uint32_t kWorkgroupInvocations = 256;     // x * y * z of the entry's workgroup size
inline constexpr std::uint32_t kWorkgroupsPerDimension = 65535; // `workgroups`, each
inline constexpr std::uint32_t kFrameUniformBytes = 32;         // the uniform `caliper` (CaliperFrame)
inline constexpr std::uint32_t kOverTimePasses = 3;             // passes over time_ms in a row that stop a program
inline constexpr double kRuntimeShare = 0.04;                   // the runtime's share of the step time (Cadence)
}  // namespace program_limits

// The names that a program's WGSL uses for its fixed bindings.
inline constexpr const char* kCountVar = "count";        // geometry: var<storage, read_write> count: atomic<u32>
inline constexpr const char* kPixelsVar = "pixels";      // pixels: var<storage, read_write> pixels: array<u32>
inline constexpr const char* kFrameVar = "caliper";      // optional: var<uniform> caliper: CaliperFrame
inline constexpr const char* kLiveInputPrefix = "live:"; // an input's `source`

// The output kinds of a program. `tensor` publishes the first elements of
// `out` as the live view "program.<id>" (design/instrument-interface.md,
// Tensor Output); operators use it, and agents' programs may.
enum class OutputKind { geometry, pixels, tensor };
// The scene parts a geometry output draws. `mesh` is reserved.
enum class GeometryPart { points, lines };
// The arrays of a geometry output, by their WGSL names (Geometry Output):
//   positions  array<f32>, 3 for each item (necessary)
//   colors     array<u32>, RGBA8 packed with pack4x8unorm
//   values     array<f32>, mapped through the style's colormap
//   radii      array<f32>, points only
enum class GeometryArray { positions, colors, values, radii };
// The pixel formats of a pixels output.
enum class PixelFormat { rgba8unorm };
// The kinds of a stage. The `compute` kind of earlier plans is a program with
// a geometry output.
enum class StageKind { material, vertex };

inline const char* output_kind_name(OutputKind k) {
    return k == OutputKind::pixels ? "pixels" : k == OutputKind::tensor ? "tensor" : "geometry";
}
inline std::optional<OutputKind> output_kind_from_name(std::string_view n) {
    if (n == "geometry") return OutputKind::geometry;
    if (n == "pixels") return OutputKind::pixels;
    if (n == "tensor") return OutputKind::tensor;
    return std::nullopt;
}
inline const char* geometry_part_name(GeometryPart p) { return p == GeometryPart::lines ? "lines" : "points"; }
inline std::optional<GeometryPart> geometry_part_from_name(std::string_view n) {
    if (n == "points") return GeometryPart::points;
    if (n == "lines") return GeometryPart::lines;
    return std::nullopt;
}
inline const char* geometry_array_name(GeometryArray a) {
    switch (a) {
        case GeometryArray::positions: return "positions";
        case GeometryArray::colors: return "colors";
        case GeometryArray::values: return "values";
        case GeometryArray::radii: return "radii";
    }
    return "positions";
}
inline std::optional<GeometryArray> geometry_array_from_name(std::string_view n) {
    for (GeometryArray a : {GeometryArray::positions, GeometryArray::colors, GeometryArray::values, GeometryArray::radii})
        if (n == geometry_array_name(a)) return a;
    return std::nullopt;
}
// The 32-bit words of one item of an array: 3 for positions, else 1.
inline std::uint32_t geometry_array_words(GeometryArray a) { return a == GeometryArray::positions ? 3u : 1u; }
inline const char* pixel_format_name(PixelFormat) { return "rgba8unorm"; }
inline std::optional<PixelFormat> pixel_format_from_name(std::string_view n) {
    if (n == "rgba8unorm") return PixelFormat::rgba8unorm;
    return std::nullopt;
}
inline const char* stage_kind_name(StageKind k) { return k == StageKind::vertex ? "vertex" : "material"; }
inline std::optional<StageKind> stage_kind_from_name(std::string_view n) {
    if (n == "material") return StageKind::material;
    if (n == "vertex") return StageKind::vertex;
    return std::nullopt;
}

// One input: the WGSL variable `var`, declared var<storage, read>, bound to
// the script's tensor `source`, "live:<name>". The names are the parameters
// of model.named_parameters(), "<parameter>.grad" and "<module>.act".
struct ProgramInput {
    std::string var;
    std::string source;
};
inline bool operator==(const ProgramInput& a, const ProgramInput& b) { return a.var == b.var && a.source == b.source; }

// How a geometry output's values map to colors.
struct GeometryStyle {
    std::optional<std::string> colormap;   // viridis, magma, rdbu or greys; viridis when absent
    std::optional<double> vmin;
    std::optional<double> vmax;
};
inline bool operator==(const GeometryStyle& a, const GeometryStyle& b) {
    return a.colormap == b.colormap && a.vmin == b.vmin && a.vmax == b.vmax;
}

// The one output of a program. The keys of the other kind stay empty.
struct ProgramOutput {
    OutputKind kind = OutputKind::geometry;
    // geometry
    std::string panel;                     // a scene panel's id
    std::optional<std::string> entity;     // the entity path; default_entity(id) when absent
    GeometryPart part = GeometryPart::points;
    std::uint32_t max_count = 0;           // 1 to kMaxCount
    std::vector<GeometryArray> arrays;     // in the order written, positions among them
    GeometryStyle style;
    // pixels
    std::uint32_t width = 0;               // 1 to kPixelsSide
    std::uint32_t height = 0;
    PixelFormat format = PixelFormat::rgba8unorm;
    // tensor: 1 to 4 axes, at most 1,000,000 elements
    std::vector<std::int64_t> shape;
};
inline bool operator==(const ProgramOutput& a, const ProgramOutput& b) {
    return a.kind == b.kind && a.panel == b.panel && a.entity == b.entity && a.part == b.part &&
           a.max_count == b.max_count && a.arrays == b.arrays && a.style == b.style && a.width == b.width &&
           a.height == b.height && a.format == b.format && a.shape == b.shape;
}

// The named numbers of a program or a stage (`params`): at most kParams
// names, each a number or a reference. The WGSL reads them in the uniform
// struct `Params` (design/instrument-interface.md, Program Parameters).
using Params = std::map<std::string, ValueOr<double>>;
inline constexpr std::size_t kParams = 16;
inline constexpr const char* kParamsVar = "params";

// The limits of a program. An absent limit takes its default.
struct ProgramLimits {
    std::optional<double> time_ms;           // GPU time of one pass, in ms
    std::optional<std::uint32_t> memory_mb;  // output buffers and bound tensors, in MiB
    std::optional<std::uint32_t> every;      // the fewest steps between two passes

    double time_ms_or_default() const { return time_ms.value_or(program_limits::kTimeMsDefault); }
    std::uint64_t memory_bytes_or_default() const {
        return std::uint64_t(memory_mb.value_or(program_limits::kMemoryMbDefault)) << 20;
    }
    std::uint32_t every_or_default() const { return every.value_or(1u); }
};
inline bool operator==(const ProgramLimits& a, const ProgramLimits& b) {
    return a.time_ms == b.time_ms && a.memory_mb == b.memory_mb && a.every == b.every;
}

// One entry of the `program` array.
struct Program {
    std::string id;                                       // unique among programs
    std::optional<std::string> title;                     // the id when absent
    std::string wgsl;                                     // at most kProgramSourceBytes
    std::optional<std::string> entry;                     // the one compute entry point when absent
    std::vector<ProgramInput> inputs;                     // at most kInputs
    ProgramOutput output;
    ProgramLimits limits;
    std::optional<std::array<std::uint32_t, 3>> workgroups;  // derived from the output when absent
    Params params;                                        // version 1: `params`
    // The operator's name and version on a program that the app generates
    // for an operator ("surface@1"); never read from or written to a spec.
    std::optional<std::string> builtin;
    nlohmann::json x;                                     // the extension table, null when absent
};
inline bool operator==(const Program& a, const Program& b) {
    return a.id == b.id && a.title == b.title && a.wgsl == b.wgsl && a.entry == b.entry && a.inputs == b.inputs &&
           a.output == b.output && a.limits == b.limits && a.workgroups == b.workgroups && a.params == b.params &&
           a.builtin == b.builtin && a.x == b.x;
}
inline bool operator!=(const Program& a, const Program& b) { return !(a == b); }

// One entry of the `stage` array. The triple panel, entity and kind is unique.
struct Stage {
    std::string panel;                  // a scene panel's id
    std::string entity;                 // an entity path of the panel's scene
    StageKind kind = StageKind::material;
    std::string wgsl;                   // at most kStageSourceBytes: functions and constants only
    Params params;                      // version 1: `params`, in the stage's uniform
    nlohmann::json x;                   // the extension table, null when absent
};
inline bool operator==(const Stage& a, const Stage& b) {
    return a.panel == b.panel && a.entity == b.entity && a.kind == b.kind && a.wgsl == b.wgsl && a.params == b.params &&
           a.x == b.x;
}
inline bool operator!=(const Stage& a, const Stage& b) { return !(a == b); }

// A reviewed source of a foreign spec: the target "program:<id>" or
// "stage:<panel>/<entity>/<kind>", and the SHA-256 of its WGSL in lowercase hex.
struct ReviewedSource {
    std::string target;
    std::string sha256;
};
inline bool operator==(const ReviewedSource& a, const ReviewedSource& b) {
    return a.target == b.target && a.sha256 == b.sha256;
}

// The `provenance` object. No command writes it: the app sets it when it opens
// a spec from outside the workspace, and the edits of a command keep it. A spec
// whose provenance is empty is the person's own, and its programs run at once.
struct Provenance {
    bool foreign = false;
    std::optional<std::string> from;    // a free text that names the origin
    std::vector<ReviewedSource> reviewed;
    nlohmann::json x;                   // the extension table, null when absent

    bool empty() const { return !foreign && !from && reviewed.empty() && x.is_null(); }
};
inline bool operator==(const Provenance& a, const Provenance& b) {
    return a.foreign == b.foreign && a.from == b.from && a.reviewed == b.reviewed && a.x == b.x;
}

// The entity path of a geometry output without `entity`: "programs/<id>".
inline std::string default_entity(std::string_view program_id) { return "programs/" + std::string(program_id); }
// The live view that carries a pixels output: "program.<id>".
inline std::string program_view_name(std::string_view program_id) { return "program." + std::string(program_id); }
// The id of a stage in program.status and in program events: "<panel>/<entity>/<kind>".
inline std::string stage_id(const Stage& s) { return s.panel + "/" + s.entity + "/" + stage_kind_name(s.kind); }
// The targets of provenance.reviewed.
inline std::string program_target(const Program& p) { return "program:" + p.id; }
inline std::string stage_target(const Stage& s) { return "stage:" + stage_id(s); }

}  // namespace caliper_builder::spec
