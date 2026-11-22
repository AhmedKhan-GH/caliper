#pragma once
// caliper_builder::programs: names, files, imports, hash, composition and
// checks of pieces (design/programmable-graphics.md, Pieces, Imports, Hash,
// Prelude, Generated Header and Sandbox).
//
// A piece's WGSL declares no binding. The composer writes every binding of
// both halves from the manifest, so that the app, the script and the browser
// module bind the same names at the same places without the compiler's
// reflection; the browser has none. A piece imports lib modules by name and
// version (imports = ["colormap@1"]) and calls their items as
// colormap::magma. Resolution checks each such name against the imported
// modules, rewrites "::" to "__" (the same length, thus every line and column
// of the author's files stays) and puts the modules' text before the piece's
// own. A lib module declares its items with the prefix <module>__ already, and
// a piece may not declare a name with "__", so that nothing collides.
//
// A composed module is, in this order: the draw prelude (draw half only), the
// generated header, the imported lib modules, common.wgsl and the half's own
// file. Its segments map a compiler message's line back to the file that
// holds it, with the column unchanged.
//
// pieces.cpp needs only the standard library and spec/piece.h, because the
// browser module compiles it beside the app and the script's native module.
// Until pieces are built, every function but draw_prelude(), lib_module()
// and lib_modules() answers program.not_built. Nothing here throws. Every
// function is safe from any thread.
#include <caliper_builder/programs/reflection.h>
#include <caliper_builder/spec/piece.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::programs {

// ---- codes (design/programmable-graphics.md, Check Codes and Host Errors)
// A part of programmable graphics that this build does not have yet.
inline constexpr const char* kProgramNotBuilt = "program.not_built";
// Pieces: names, files, sources and pins.
inline constexpr const char* kPieceBadName = "piece.bad_name";             // a name or version outside its form
inline constexpr const char* kPieceBadManifest = "piece.bad_manifest";     // a key, count or value of piece.toml
inline constexpr const char* kPieceBadPath = "piece.bad_path";             // a file outside its folder, a link, or another name
inline constexpr const char* kPieceUnknownImport = "piece.unknown_import"; // no such lib module, or m::item names none
inline constexpr const char* kPieceNotFound = "piece.not_found";           // a command names no such piece
inline constexpr const char* kPieceMissing = "piece.missing";              // a program pins a piece that its source lacks
inline constexpr const char* kPieceHashMismatch = "piece.hash_mismatch";   // the pinned hash differs from the piece's
inline constexpr const char* kPieceExists = "piece.exists";                // that name and version exist in the workspace
inline constexpr const char* kPieceReadOnly = "piece.read_only";           // a library piece
inline constexpr const char* kPieceInUse = "piece.in_use";                 // the open spec pins that version
inline constexpr const char* kPieceBadBinding = "piece.bad_binding";       // an input unbound, or bound to another dtype, rank or layout
inline constexpr const char* kPieceCycle = "piece.cycle";                  // buffer inputs of programs form a cycle
// The manifest and the sources; the editor, the script and the browser give them.
inline constexpr const char* kProgramBadResource = "program.bad_resource";         // kind, count, size or a duplicate id
inline constexpr const char* kProgramBadLayout = "program.bad_layout";             // a field's name or type, or a stride
inline constexpr const char* kProgramUnknownResource = "program.unknown_resource"; // a pass, a pick or a text names none
inline constexpr const char* kProgramBadPass = "program.bad_pass";                 // a pass's kind, entry, target or values
inline constexpr const char* kProgramTargetCycle = "program.target_cycle";         // a pass reads its own target, or before a write
inline constexpr const char* kProgramBadPick = "program.bad_pick";                 // the item map names no field or no pick pass
inline constexpr const char* kProgramBadText = "program.bad_text";                 // the text table's anchors or strings
inline constexpr const char* kProgramCameraConflict = "program.camera_conflict";   // two programs of a panel select two cameras
inline constexpr const char* kProgramBindingNotAllowed = "program.binding_not_allowed";  // a file declares @group or @binding
inline constexpr const char* kProgramReservedName = "program.reserved_name";       // a caliper_ or Caliper name, or a name with "__"
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

// ---- bind groups of a composed module (design/programmable-graphics.md, Generated Header)
inline constexpr std::uint32_t kPreludeVersion = 1;
inline constexpr std::uint32_t kGroupPrelude = 0;    // draw: the prelude and params; derive: caliper, params, shapes
inline constexpr std::uint32_t kGroupInputs = 1;     // derive: the tensors; draw: counts, resources, buffer inputs
inline constexpr std::uint32_t kGroupResources = 2;  // derive: resources and counters; draw: draw buffers, read targets
// Group 0 of the draw half: the prelude's five bindings, then params.
inline constexpr std::uint32_t kBindingView = 0;        // var<uniform> caliper_view: CaliperView
inline constexpr std::uint32_t kBindingSelection = 1;   // var<storage, read> caliper_selection
inline constexpr std::uint32_t kBindingLinear = 2;      // caliper_linear: sampler
inline constexpr std::uint32_t kBindingNearest = 3;     // caliper_nearest: sampler
inline constexpr std::uint32_t kBindingColormaps = 4;   // caliper_colormaps: texture_2d<f32>
inline constexpr std::uint32_t kBindingParams = 5;      // var<uniform> params: Params, when the piece has params
inline constexpr std::uint32_t kViewBytes = 288;        // the size of CaliperView
inline constexpr std::uint32_t kSelectedBytes = 32;     // the size of CaliperSelected
inline constexpr std::uint32_t kMaxSelected = 256;      // entries of caliper_selection

// The text of src/builder/programs/draw_prelude.wgsl, which the build embeds
// here and in the browser module from the same file.
std::string_view draw_prelude();
// The built-in lib modules, as "<name>@<version>" in order, and the text of
// one (src/builder/programs/library/lib/<name>/<version>.wgsl), empty when
// there is none. A released version never changes.
std::vector<std::string_view> lib_modules();
std::string_view lib_module(std::string_view name_at_version);

// ---- names and paths (design/programmable-graphics.md, Piece Files)
// A piece name is [a-z][a-z0-9_]{0,31} and a version 1 to kVersionMax, so
// that <workspace>/pieces/<name>/<version>/ never leaves pieces/.
Verdict check_piece_name(std::string_view name, std::uint32_t version);
// A file of a piece folder relative to it: piece.toml, one of kPieceFiles, or
// sample/<input>.bin for an input of the manifest. Anything else, an
// absolute path, a "..", a link or a path whose real path leaves the folder
// is piece.bad_path; the caller checks links and real paths on disk.
Verdict check_piece_file(const spec::Piece& piece, std::string_view relative_path);

// ---- imports and hash
// The text of one segment of a composed module: its part ("prelude",
// "header", "lib/colormap@1", "common.wgsl", "derive.wgsl" or "draw.wgsl") and
// its first line in the composed text, from 1.
struct Segment {
    std::string part;
    std::uint32_t first_line = 1;
    std::uint32_t lines = 0;
};
// A piece after its imports: the imported modules' text in order, and the
// author's files with "::" rewritten to "__".
struct Resolved {
    Verdict verdict;                         // not ok: an unknown module, or m::item names nothing
    std::vector<std::string> modules;        // "<name>@<version>", in the manifest's order
    std::string lib_text;                    // the modules' texts, one after another
    std::vector<Segment> lib_segments;       // within lib_text
    std::map<std::string, std::string> files;
};
Resolved resolve_imports(const spec::Piece& piece);

// The manifest as RFC 8785 canonical JSON, without `hash` and without the
// files: the bytes that the hash covers, the same on every platform.
std::string canonical_manifest(const spec::Piece& piece);
// "sha256:<64 lowercase hex>" over "caliper-piece/1\n", the canonical
// manifest, and for each imported module and then each file of kPieceFiles
// that the piece has, a line "--- <part> <bytes>\n" and its text with LF line
// ends. Sample files are not covered.
std::string piece_hash(const spec::Piece& piece);

// ---- layout
struct FieldLayout {
    std::string name;
    std::string type;
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
};
// The bytes of one element after the storage layout rules of WGSL: its
// stride (the struct's size, rounded up to its alignment) and its fields.
struct ElementLayout {
    std::uint32_t stride = 0;
    std::uint32_t align = 0;
    std::vector<FieldLayout> fields;
};
// A bad field type or a stride over kStrideMax is program.bad_layout.
Verdict element_layout(const spec::Layout& layout, ElementLayout* out);

// ---- composition
enum class Half { derive, draw };

// A composed module and its segments.
struct Composed {
    Verdict verdict;                         // not ok: nothing was composed
    std::string text;
    std::vector<Segment> segments;
};
// The file, line and column of a composed module's line and column.
struct SourcePlace {
    std::string part;
    std::uint32_t line = 0;
    std::uint32_t column = 0;
};
SourcePlace source_place(const Composed& composed, std::uint32_t line, std::uint32_t column);

// The derive half's module: the generated header (group 0 the frame uniform
// `caliper`, `params` and `shapes`; group 1 the tensor inputs; group 2 the
// resources and their counters), the lib modules, common.wgsl, derive.wgsl.
Composed compose_derive(const spec::Piece& piece, const Resolved& resolved);
// The module of one pass of the draw half: the prelude, the generated header
// (group 0 `params`; group 1 the counts, the published resources and the
// buffer inputs, read-only; group 2 the draw buffers, read_write only in a
// compute pass that writes them, and the targets that the pass reads), the
// lib modules, common.wgsl, draw.wgsl.
Composed compose_draw(const spec::Piece& piece, const Resolved& resolved, std::size_t pass);

// ---- checks, in the order of Check Order
// 1. The manifest: name and version, the kinds, layouts, counts, sizes and
//    limits, the passes' names and values, the targets' order, the item map,
//    the text table, the params' ranges, and the WebGPU limits of 7.
Verdict check_piece_manifest(const spec::Piece& piece);
// 3. One file: its size limit, no directive, no @group or @binding, and no
//    declaration of a caliper_, Caliper or CALIPER_ name or of a name with "__".
Verdict check_piece_source(std::string_view wgsl, std::size_t max_bytes);
// 6. A compiled pass: its entry points and their stages, workgroup sizes, the
//    vertex outputs against the fragment inputs, the fragment outputs against
//    the pass's target and pick, and each binding against the header.
Verdict check_piece_reflection(const spec::Piece& piece, Half half, std::size_t pass, const Reflection& reflection);
// 7. The default limits of WebGPU that a pass must keep to run in a browser:
//    storage buffers and sampled textures in one stage, bind groups,
//    inter-stage variables, workgroup sizes and buffer binding sizes.
Verdict check_web_limits(const spec::Piece& piece);
// An instance's binding of one input: a tensor's dtype and rank, or the
// layout of another program's published buffer.
Verdict check_tensor_binding(const spec::PieceInput& input, std::string_view dtype, std::uint32_t rank);
Verdict check_buffer_binding(const spec::PieceInput& input, const spec::Layout& published);

}  // namespace caliper_builder::programs
