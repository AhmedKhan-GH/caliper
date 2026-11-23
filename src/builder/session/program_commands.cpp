// program.status (design/gpu-programs.md, Commands), and the status item and
// the `program` event that the hosts share (program_host.h).
//
// program.status lists the open spec's programs in order, each with the item
// its host reports. An item that no host reports has host none, and its state
// is review in a foreign spec whose provenance lacks its hash, else waiting.
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/session/program_commands.h>
#include <caliper_builder/session/chain_hooks.h>

#include <string>
#include <utility>

namespace caliper_builder {

using nlohmann::json;

namespace {

// program.status's `draw` object of an instance.
json draw_status_json(const DrawStatus& d) {
    auto opt = [](const auto& v) { return v ? json(*v) : json(nullptr); };
    json out = {{"state", d.state},
                {"frame_ms", opt(d.frame_ms)},
                {"frame_limit_ms", d.frame_limit_ms},
                {"memory_bytes", d.memory_bytes},
                {"memory_limit_bytes", d.memory_limit_bytes},
                {"resource_step", opt(d.resource_step)},
                {"error", nullptr}};
    if (d.error) out["error"] = {{"code", d.error->code}, {"message", d.error->message}, {"where", d.error_where}};
    return out;
}

}  // namespace

json program_status_json(const ProgramStatus& s) {
    auto opt = [](const auto& v) { return v ? json(*v) : json(nullptr); };
    json out = {{"id", s.id},
                {"kind", s.kind},
                {"host", s.host},
                {"state", s.state},
                {"every", opt(s.every)},
                {"every_min", opt(s.every_min)},
                {"last_pass_ms", opt(s.last_pass_ms)},
                {"last_cost_ms", opt(s.last_cost_ms)},
                {"step_ms", opt(s.step_ms)},
                {"overhead_share", opt(s.overhead_share)},
                {"passes_per_s", opt(s.passes_per_s)},
                {"last_step", opt(s.last_step)},
                {"runs", s.runs},
                {"memory_bytes", s.memory_bytes},
                {"memory_limit_bytes", s.memory_limit_bytes},
                {"error", nullptr},
                {"sha256", s.sha256}};
    if (s.error)
        out["error"] = {{"code", s.error->code},
                        {"message", s.error->message},
                        {"line", s.error->line},
                        {"column", s.error->column}};
    if (s.piece) out["piece"] = spec::to_json(*s.piece);
    if (s.draw) out["draw"] = draw_status_json(*s.draw);
    return out;
}

void push_program_event(EventSink& sink, const ProgramStatus& s, std::string_view change) {
    Event e;
    e.kind = EventKind::program;
    e.panel = s.panel;
    e.origin = s.host == "script" ? "script" : "ui";
    e.fields = {{"program", s.id}, {"change", std::string(change)}, {"host", s.host == "none" ? "viewer" : s.host}};
    if (s.error) {
        e.fields["code"] = s.error->code;
        e.fields["message"] = s.error->message;
        e.fields["line"] = s.error->line;
        e.fields["column"] = s.error->column;
    }
    // An instance's half and its pass (design/programmable-graphics.md, Status and Events).
    if (!s.half.empty()) {
        e.fields["half"] = s.half;
        e.fields["pass"] = s.pass ? json(*s.pass) : json(nullptr);
    }
    sink.push(std::move(e));
}

namespace {

std::string arg_string(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

// The item of a program that no host reports.
ProgramStatus unhosted(const spec::Spec& s, std::string id, std::string kind, std::string panel,
                       std::string_view target, const std::string& wgsl) {
    ProgramStatus st;
    st.id = std::move(id);
    st.kind = std::move(kind);
    st.panel = std::move(panel);
    st.host = "none";
    const bool may_run = programs::check_provenance(s.provenance, target, programs::sha256_hex(wgsl)).ok();
    st.state = may_run ? "waiting" : "review";
    return st;
}

// The status item of an instance of a piece (design/programmable-graphics.md,
// Program Commands): the kind piece, the pin, and the draw half's object when
// the piece draws. An item that no host reports is refused when its pin has no
// hash or does not resolve, in review when a foreign spec's provenance lacks
// the piece's hash, and waiting otherwise.
json instance_status(const spec::Spec& s, const spec::Program& p, const ProgramStatus* hosted,
                     const ProgramStatus* drawn, const std::string& workspace) {
    const FoundPiece found = resolve_spec_pin(PieceLibrary::builtin(), workspace, s, *p.piece);
    json out;
    if (hosted) {
        out = program_status_json(*hosted);
    } else {
        ProgramStatus st;
        st.id = p.id;
        st.panel = p.panel.value_or("");
        if (found.error) {
            st.state = "refused";
            st.error = ProgramError{found.error.code, found.error.message, 0, 0};
        } else {
            const std::string hex = found.piece.hash.substr(found.piece.hash.find(':') + 1);
            st.state = programs::check_provenance(s.provenance, spec::program_target(p), hex).ok() ? "waiting" : "review";
            st.sha256 = hex;
        }
        if (!found.error && found.piece.derive) {
            st.every_min = found.piece.derive->every;
            st.memory_limit_bytes = std::uint64_t(found.piece.derive->memory_mb.value_or(64)) << 20;
        }
        out = program_status_json(st);
    }
    out["kind"] = "piece";
    out["piece"] = spec::to_json(*p.piece);
    // The viewer's host that draws the instance reports its draw object (the
    // draw status host's item of the same id); else the manifest gives its limits,
    // and the state follows the pin.
    if (drawn && drawn->draw) out["draw"] = draw_status_json(*drawn->draw);
    if (!found.error && found.piece.draw && !out.contains("draw")) {
        const spec::Draw& d = *found.piece.draw;
        const std::string state = out["state"] == "refused" ? "refused" : out["state"] == "review" ? "review" : "waiting";
        out["draw"] = {{"state", state},
                       {"frame_ms", nullptr},
                       {"frame_limit_ms", d.frame_ms.value_or(spec::piece_limits::kFrameMsDefault)},
                       {"memory_bytes", 0},
                       {"memory_limit_bytes",
                        std::uint64_t(d.memory_mb.value_or(spec::piece_limits::kDrawMemoryMbDefault)) << 20},
                       {"resource_step", nullptr},
                       {"error", nullptr}};
    }
    return out;
}

const ProgramStatus* reported(const std::vector<ProgramStatus>& list, const std::string& id) {
    for (const ProgramStatus& st : list)
        if (st.id == id) return &st;
    return nullptr;
}

CommandOutput program_status(const json& args, const spec::Spec& s, const ProgramHost* program_host,
                             const DrawStatusHost* draw_host, const std::string& workspace) {
    const std::string only = arg_string(args, "id");
    const std::vector<ProgramStatus> hosted_programs =
        program_host ? program_host->status() : std::vector<ProgramStatus>{};
    const std::vector<ProgramStatus> hosted_draws = draw_host ? draw_host->status() : std::vector<ProgramStatus>{};
    json programs_out = json::array();
    for (const spec::Program& p : s.programs) {
        if (!only.empty() && p.id != only) continue;
        if (p.piece) {
            programs_out.push_back(
                instance_status(s, p, reported(hosted_programs, p.id), reported(hosted_draws, p.id), workspace));
            continue;
        }
        if (const ProgramStatus* st = reported(hosted_programs, p.id)) {
            programs_out.push_back(program_status_json(*st));
            continue;
        }
        const bool geometry = p.output.kind == spec::OutputKind::geometry;
        ProgramStatus st = unhosted(s, p.id, spec::output_kind_name(p.output.kind), geometry ? p.output.panel : "",
                                    spec::program_target(p), p.wgsl);
        st.every_min = p.limits.every;
        st.memory_limit_bytes = p.limits.memory_bytes_or_default();
        programs_out.push_back(program_status_json(st));
    }
    // The program of each operator that runs in the script
    // (design/instrument-interface.md, Operator Programs): kind operator, and
    // the source's id in its own.
    for (const spec::Source& source : s.sources) {
        if (!operator_runs_in_script(s, source, 0)) continue;
        const std::string id = "operator." + source.id + ".0";
        if (!only.empty() && id != only) continue;
        if (const ProgramStatus* st = reported(hosted_programs, id)) {
            programs_out.push_back(program_status_json(*st));
            continue;
        }
        ProgramStatus st;
        st.id = id;
        st.kind = "operator";
        programs_out.push_back(program_status_json(st));
    }
    if (!only.empty() && programs_out.empty())
        return CommandOutput::fail(Error(programs::kProgramNotFound,
                                         "The open spec has no program " + only + ".",
                                         "args.id"));
    return CommandOutput::ok({{"programs", std::move(programs_out)}});
}

}  // namespace

CommandOutput program_command(const CommandContext& context, const spec::Spec& spec, const ProgramHost* programs,
                              const DrawStatusHost* draws, const std::string& workspace) {
    const std::string& name = context.invocation.name;
    if (name == "program.status") return program_status(context.invocation.args, spec, programs, draws, workspace);
    return CommandOutput::fail(Error("command.unknown", name + " is not a program read command."));
}

}  // namespace caliper_builder
