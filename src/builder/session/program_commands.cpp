// stage.check and program.status (design/gpu-programs.md, Commands), and the
// status item and the `program` event that the hosts share (program_host.h).
//
// stage.check runs the editor's sandbox checks (program_checks.h) on a source
// without a declaration and lists what the compiler reflects; without the
// compiler library it answers program.compiler_unavailable. program.status
// lists the open spec's programs and stages in order, each with the item its
// host reports. An item that no host reports has host none, and its state is
// review in a foreign spec whose provenance lacks its hash, else waiting.
#include "program_checks.h"

#include <caliper_builder/session/program_commands.h>

#include <string>
#include <utility>

namespace caliper_builder {

using nlohmann::json;

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
    sink.push(std::move(e));
}

namespace {

std::string arg_string(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

const char* binding_kind_name(programs::BindingKind k) {
    switch (k) {
        case programs::BindingKind::uniform_buffer: return "uniform";
        case programs::BindingKind::storage_buffer: return "storage";
        case programs::BindingKind::texture: return "texture";
        case programs::BindingKind::uint_texture: return "uint_texture";
        case programs::BindingKind::storage_texture: return "storage_texture";
        case programs::BindingKind::sampler: return "sampler";
    }
    return "storage";
}

const char* entry_stage_name(programs::EntryStage s) {
    switch (s) {
        case programs::EntryStage::vertex: return "vertex";
        case programs::EntryStage::fragment: return "fragment";
        case programs::EntryStage::compute: return "compute";
    }
    return "compute";
}

CommandOutput stage_check(const json& args) {
    if (!programs::compiler_available())
        return CommandOutput::fail(Error(programs::kProgramCompilerUnavailable,
                                         "This build has no WGSL compiler, so stage.check cannot compile the source.",
                                         "args.wgsl"));
    const std::string kind = arg_string(args, "kind");
    const std::string wgsl = arg_string(args, "wgsl");
    program_checks::Outcome o;
    if (kind == "compute") {
        std::optional<std::string> entry;
        if (args.contains("entry") && args["entry"].is_string()) entry = args["entry"].get<std::string>();
        o = program_checks::check_compute_source(wgsl, entry);
    } else {
        const auto stage_kind = spec::stage_kind_from_name(kind);
        if (!stage_kind)
            return CommandOutput::fail(Error("spec.bad_value", "kind is material, vertex or compute.", "args.kind"));
        spec::Stage st;
        st.kind = *stage_kind;
        st.wgsl = wgsl;
        o = program_checks::check_stage_source(st);
    }
    json messages = json::array();
    if (!o.error.ok()) messages.push_back(program_checks::message_json("error", o.error));
    json entries = json::array();
    json bindings = json::array();
    for (const programs::ReflectedEntry& e : o.reflection.entries)
        entries.push_back({{"name", e.name},
                           {"stage", entry_stage_name(e.stage)},
                           {"workgroup_size", {e.workgroup_size[0], e.workgroup_size[1], e.workgroup_size[2]}}});
    for (const programs::ReflectedBinding& b : o.reflection.bindings)
        bindings.push_back({{"group", b.group},
                            {"binding", b.binding},
                            {"name", b.name},
                            {"kind", binding_kind_name(b.kind)},
                            {"access", b.write ? (b.read ? "read_write" : "write") : "read"}});
    return CommandOutput::ok({{"ok", o.error.ok()},
                              {"messages", std::move(messages)},
                              {"entry_points", std::move(entries)},
                              {"bindings", std::move(bindings)}});
}

// The item of a program or a stage that no host reports.
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

const ProgramStatus* reported(const std::vector<ProgramStatus>& list, const std::string& id) {
    for (const ProgramStatus& st : list)
        if (st.id == id) return &st;
    return nullptr;
}

CommandOutput program_status(const json& args, const spec::Spec& s, const ProgramHost* program_host,
                             const StageHost* stage_host) {
    const std::string only = arg_string(args, "id");
    const std::vector<ProgramStatus> hosted_programs =
        program_host ? program_host->status() : std::vector<ProgramStatus>{};
    const std::vector<ProgramStatus> hosted_stages = stage_host ? stage_host->status() : std::vector<ProgramStatus>{};
    json programs_out = json::array();
    for (const spec::Program& p : s.programs) {
        if (!only.empty() && p.id != only) continue;
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
    json stages_out = json::array();
    for (const spec::Stage& stage : s.stages) {
        const std::string id = spec::stage_id(stage);
        if (!only.empty() && id != only) continue;
        if (const ProgramStatus* st = reported(hosted_stages, id)) {
            stages_out.push_back(program_status_json(*st));
            continue;
        }
        stages_out.push_back(program_status_json(
            unhosted(s, id, spec::stage_kind_name(stage.kind), stage.panel, spec::stage_target(stage), stage.wgsl)));
    }
    if (!only.empty() && programs_out.empty() && stages_out.empty())
        return CommandOutput::fail(Error(programs::kProgramNotFound,
                                         "The open spec has no program " + only +
                                             " and no stage of that id; a stage's id is <panel>/<entity>/<kind>.",
                                         "args.id"));
    return CommandOutput::ok({{"programs", std::move(programs_out)}, {"stages", std::move(stages_out)}});
}

}  // namespace

CommandOutput program_command(const CommandContext& context, const spec::Spec& spec, const ProgramHost* programs,
                              const StageHost* stages) {
    const std::string& name = context.invocation.name;
    if (name == "stage.check") return stage_check(context.invocation.args);
    if (name == "program.status") return program_status(context.invocation.args, spec, programs, stages);
    return CommandOutput::fail(Error("command.unknown", name + " is not a stage or program read command."));
}

}  // namespace caliper_builder
