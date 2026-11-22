#pragma once
// caliper_builder::programs: the composer and the checks of the graphics
// form of a program (design/programmable-graphics.md, Prelude, Generated
// Header and Sandbox).
//
// A program's source declares no binding. The composer writes every binding
// of both halves from the declaration, so that the app, the script and the
// browser module bind the same names at the same places without the
// compiler's reflection; the browser has none. A composed module is, in this
// order: the draw prelude (draw half only), the generated header, the
// program's `common` source and the half's own source.
//
// The checks run in the order of design/programmable-graphics.md, Check
// Order. check_graphics_declaration() and check_graphics_source() run
// everywhere, in the browser module too; check_graphics_reflection() needs
// the compiler's reflection and runs in the app and in the script.
//
// graphics.cpp needs only the standard library and spec/graphics.h, because
// the browser module compiles it beside the app and the script's native
// module. Until the graphics form is built, every function but
// draw_prelude() answers program.not_built. Nothing here throws. Every
// function is safe from any thread.
#include <caliper_builder/programs/reflection.h>
#include <caliper_builder/spec/graphics.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::programs {

// What the composer and the checks read of a program: its id, the names of
// its params (whose types the struct Params of `common` gives) and its
// graphics form. The editor and the script fill it from spec::Program with
// graphics_program() of sandbox.h; the viewers fill it from the declaration
// of a PROGRAM message.
struct GraphicsProgram {
    std::string id;
    std::vector<std::string> params;
    spec::Graphics graphics;
};

// ---- codes (design/programmable-graphics.md, Check Order and Host Errors)
// A part of programmable graphics that this build does not have yet.
inline constexpr const char* kProgramNotBuilt = "program.not_built";
// The editor's codes, which the script and the browser module also give.
inline constexpr const char* kProgramFormMixed = "program.form_mixed";             // keys of both forms
inline constexpr const char* kProgramBadResource = "program.bad_resource";         // kind, count, size or a duplicate id
inline constexpr const char* kProgramBadLayout = "program.bad_layout";             // a field's name or type, or a stride
inline constexpr const char* kProgramUnknownResource = "program.unknown_resource"; // a pass, a pick or a text names none
inline constexpr const char* kProgramBadPass = "program.bad_pass";                 // a pass's kind, entry, target or values
inline constexpr const char* kProgramTargetCycle = "program.target_cycle";         // a pass reads its own target, or before a write
inline constexpr const char* kProgramBadPick = "program.bad_pick";                 // the item map names no field or no pick pass
inline constexpr const char* kProgramBadText = "program.bad_text";                 // the text table's anchors or strings
inline constexpr const char* kProgramUnknownState = "program.unknown_state";       // draw.state names no state variable
inline constexpr const char* kProgramCameraConflict = "program.camera_conflict";   // two programs of a panel name two cameras
inline constexpr const char* kProgramBindingNotAllowed = "program.binding_not_allowed";  // the source declares @group or @binding
inline constexpr const char* kProgramReservedName = "program.reserved_name";       // the source declares a caliper_ or Caliper name
inline constexpr const char* kProgramEntryStage = "program.entry_stage";           // an entry point of the wrong stage, or none
inline constexpr const char* kProgramInterfaceMismatch = "program.interface_mismatch";  // stage outputs and inputs disagree
inline constexpr const char* kProgramOverWebLimits = "program.over_web_limits";    // over a default limit of WebGPU
// The hosts' codes.
inline constexpr const char* kProgramResourcesUnavailable = "program.resources_unavailable";  // no capability "resources"
inline constexpr const char* kProgramResourceTooLarge = "program.resource_too_large";        // a version over its bytes
inline constexpr const char* kProgramDrawOverTime = "program.draw_over_time";      // over frame_ms for 30 frames in a row
inline constexpr const char* kProgramDrawTimeout = "program.draw_timeout";         // a frame not done after 1 s
inline constexpr const char* kProgramDrawOverMemory = "program.draw_over_memory";  // over draw.memory_mb
inline constexpr const char* kProgramBrowserRefused = "program.browser_refused";   // WebGPU refused a module or a pipeline

// ---- bind groups of a composed module (design/programmable-graphics.md, Prelude)
inline constexpr std::uint32_t kPreludeVersion = 1;
inline constexpr std::uint32_t kGroupPrelude = 0;    // draw: the prelude, params, state; derive: frame, params, shapes
inline constexpr std::uint32_t kGroupInputs = 1;     // derive: the tensors; draw: the counts and the resources
inline constexpr std::uint32_t kGroupResources = 2;  // derive: the resources; draw: draw buffers and read targets
// Group 0 of the draw half: the prelude's five bindings, then the generated ones.
inline constexpr std::uint32_t kBindingView = 0;        // var<uniform> caliper_view: CaliperView
inline constexpr std::uint32_t kBindingSelection = 1;   // var<storage, read> caliper_selection
inline constexpr std::uint32_t kBindingLinear = 2;      // caliper_linear: sampler
inline constexpr std::uint32_t kBindingNearest = 3;     // caliper_nearest: sampler
inline constexpr std::uint32_t kBindingColormaps = 4;   // caliper_colormaps: texture_2d<f32>
inline constexpr std::uint32_t kBindingParams = 5;      // var<uniform> params: Params, when the program has params
inline constexpr std::uint32_t kBindingState = 6;       // var<uniform> state: CaliperState, when draw.state is set
inline constexpr std::uint32_t kViewBytes = 288;        // the size of CaliperView
inline constexpr std::uint32_t kSelectedBytes = 32;     // the size of CaliperSelected
inline constexpr std::uint32_t kMaxSelected = 256;      // entries of caliper_selection

// The text of src/builder/programs/draw_prelude.wgsl, which the build embeds
// here and in the browser module from the same file.
std::string_view draw_prelude();

// ---- layout
// A field of an element after the storage layout rules of WGSL.
struct FieldLayout {
    std::string name;
    std::string type;
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
};
// The bytes of one element: its stride (the struct's size, rounded up to its
// alignment), its alignment and its fields.
struct ElementLayout {
    std::uint32_t stride = 0;
    std::uint32_t align = 0;
    std::vector<FieldLayout> fields;
};
// The layout of a buffer's element; a bad field type or a stride over
// kStrideMax is program.bad_layout.
Verdict element_layout(const spec::Layout& layout, ElementLayout* out);

// ---- composition
enum class Half { derive, draw };

// The WGSL type of a state variable as the draw half reads it: integer i32,
// number f32, boolean u32, text u32 (the index of its choice) and range
// vec2<f32>. A list is not readable; the linked selection carries picks.
struct StateType {
    std::string id;
    std::string wgsl_type;
};

// A composed module and where its parts start, so that a compiler message's
// line maps back to the part the agent wrote.
struct Composed {
    Verdict verdict;                 // not ok: nothing was composed
    std::string text;
    std::uint32_t header_line = 1;   // the first line of the generated header
    std::uint32_t common_line = 0;   // the first line of `common`, 0 without it
    std::uint32_t source_line = 0;   // the first line of the half's source
};
// A line of a composed module in the part that holds it: "prelude",
// "header", "common", "derive" or "draw", and the line in that part from 1.
struct SourcePlace {
    std::string part;
    std::uint32_t line = 0;
};
SourcePlace source_place(const Composed& composed, Half half, std::uint32_t line);

// The derive half's module: generated header (group 0 the frame uniform
// `caliper`, `params` and the input shapes; group 1 the inputs; group 2 the
// resources and their counters), `common`, then derive.wgsl.
Composed compose_derive(const GraphicsProgram& program);
// The module of one pass of the draw half: the prelude, the generated
// header (group 0 `params` and `state`; group 1 the counts and the resources,
// read-only; group 2 the draw buffers, read_write only in a compute pass that
// writes them, and the targets the pass reads), `common`, then draw.wgsl.
Composed compose_draw(const GraphicsProgram& program, std::size_t pass, const std::vector<StateType>& state);

// ---- checks, in the order of Check Order
// 1. The declaration: the form, ids, kinds, layouts, counts, sizes, limits,
//    the passes' names and values, the targets' order, the item map, the
//    text table and the state reads (with the spec's state variables).
Verdict check_graphics_declaration(const GraphicsProgram& program, const std::vector<StateType>& state);
// 3. One source (common, derive or draw): its size limit, no directive, no
//    @group or @binding, and no declaration of a caliper_ or Caliper name.
Verdict check_graphics_source(std::string_view wgsl, std::size_t max_bytes);
// 5. and 6. A compiled pass: its entry points and their stages, workgroup
//    sizes, the vertex outputs against the fragment inputs, the fragment
//    outputs against the pass's target and pick, and each binding of the
//    reflection against the generated header.
Verdict check_graphics_reflection(const GraphicsProgram& program, Half half, std::size_t pass,
                                  const Reflection& reflection);
// 7. The default limits of WebGPU that a pass would pass in a browser:
//    storage buffers and sampled textures in one stage, bind groups,
//    inter-stage variables, workgroup sizes and buffer binding sizes.
Verdict check_web_limits(const GraphicsProgram& program);

}  // namespace caliper_builder::programs
