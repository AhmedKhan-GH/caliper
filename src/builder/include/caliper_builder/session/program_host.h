#pragma once
// The hosts of GPU programs and stages (design/gpu-programs.md, Program Status
// and Attachment): where the session meets the parts of the app that run the
// spec's programs in the attached script and its stages in the viewer.
//
// The app gives the session a ProgramHost and a StageHost, as it gives its
// RunControl (Session::set_program_host(), Session::set_stage_host()), and
// takes them back with null before it destroys them. The session calls
// reconcile() on the frame thread after each change of the open spec (an
// edit, an undo, a redo, an open), and the host then makes what it runs equal
// to the spec: the program host sends PROGRAM set and remove to the attached
// script, and the stage host compiles the stages. program.status reads
// status() of both. Each host pushes its own `program` events
// (push_program_event()). Without a host, program.status reports each item
// with host none and state waiting.
#include <caliper_builder/session/events.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

// The last error of a program or a stage: a code of sandbox.h, one plain
// sentence, and the line and column in its source (0 when none).
struct ProgramError {
    std::string code;
    std::string message;
    std::uint32_t line = 0;
    std::uint32_t column = 0;
};

// The draw half of an instance of a piece in the viewer
// (design/programmable-graphics.md, Program Commands): program.status's
// `draw` object.
struct DrawStatus {
    std::string state = "waiting";                // "waiting", "review", "compiled", "running", "refused" or "stopped"
    std::optional<double> frame_ms;               // GPU time of the last frame; null when unknown
    double frame_limit_ms = 4;                    // the piece's draw.frame_ms
    std::uint64_t memory_bytes = 0;               // resources, buffer inputs, draw buffers and targets
    std::uint64_t memory_limit_bytes = 0;         // the piece's draw.memory_mb in bytes
    std::optional<std::int64_t> resource_step;    // the step of the newest resource version that a frame drew
    std::optional<ProgramError> error;            // the last refusal or stop
    std::string error_where;                      // the pass of the error, such as "draw.passes[1]"; empty when none
};

// One status item (the command table's program_status, and PROGRAM_STATE's
// programs).
struct ProgramStatus {
    std::string id;                               // a program's id, or spec::stage_id() of a stage
    std::string kind;                             // "geometry", "pixels", "material" or "vertex"
    std::string host = "none";                    // "script", "viewer" or "none"
    std::string state = "waiting";                // "waiting", "review", "compiled", "running", "refused", "stopped"
    std::string panel;                            // the target scene panel of a geometry output or a stage; empty otherwise
    // The cadence (design/gpu-programs.md, Cadence) as the script measures it
    // on the run's wall clock; null for a stage and before it is measured.
    std::optional<std::uint32_t> every;           // steps between passes: N, or a multiple of N
    std::optional<std::uint32_t> every_min;       // the declared limits.every
    std::optional<double> last_pass_ms;           // GPU time of the last pass or frame
    std::optional<double> last_cost_ms;           // the overhead D of a pass, shared by all programs
    std::optional<double> step_ms;                // the base step time b
    std::optional<double> overhead_share;         // D / (N b): the share of the training time
    std::optional<double> passes_per_s;           // passes a second over the program's last passes
    std::optional<std::int64_t> last_step;        // the script's step of the last pass
    std::uint64_t runs = 0;                       // passes, or frames of a stage
    std::uint64_t memory_bytes = 0;               // output buffers and bound tensors
    std::uint64_t memory_limit_bytes = 0;         // from limits.memory_mb; 0 for a stage
    std::optional<ProgramError> error;            // the last error
    std::string sha256;                           // of the source the host runs; empty when none
    // An instance of a piece (design/programmable-graphics.md, Status and
    // Events). The item's `piece` and `draw` keys, and the event's `half` and
    // `pass`, come from these; each stays absent for another program.
    std::optional<spec::PieceRef> piece;          // the pin
    std::string half;                             // "derive" or "draw": the half that the host runs; empty otherwise
    std::optional<std::uint32_t> pass;            // the pass that the error names; null for the whole half
    std::optional<DrawStatus> draw;               // the draw half, when a viewer's host reports it
};

// The status item in the command table's form (program_status).
nlohmann::json program_status_json(const ProgramStatus& status);

// Pushes one `program` event: change "compiled", "refused" or "stopped", with
// the item's id, host, panel and, when it has one, its error.
void push_program_event(EventSink& sink, const ProgramStatus& status, std::string_view change);

// The programs in the attached script. The app's implementation owns the
// script side of the wire path (PROGRAM and PROGRAM_STATE).
class ProgramHost {
public:
    virtual ~ProgramHost() = default;
    // Makes the attached script's programs equal to spec.programs, by id and
    // hash: set for a new or changed program, remove for a removed one. Frame
    // thread.
    virtual void reconcile(const spec::Spec& spec) = 0;
    // One item for each program of the last reconciled spec, in its order. Any
    // thread.
    virtual std::vector<ProgramStatus> status() const = 0;
};

// The stages in the viewer. The app's implementation compiles them for the
// scene panels of the spec runtime and runs their time watchdog.
class StageHost {
public:
    virtual ~StageHost() = default;
    // Compiles the new and changed stages of spec.stages and drops the removed
    // ones. Frame thread.
    virtual void reconcile(const spec::Spec& spec) = 0;
    // One item for each stage of the last reconciled spec, in its order. Any
    // thread. A host that also draws instances of pieces in the viewer adds
    // one item for each, with the program's id, half "draw" and `draw` set;
    // program.status takes the instance's draw object from it.
    virtual std::vector<ProgramStatus> status() const = 0;
};

}  // namespace caliper_builder
