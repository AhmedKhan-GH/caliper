// The stage and program edits of the open spec (design/gpu-programs.md, Stage
// Commands and Program Commands): stage.set, stage.remove, program.create,
// program.patch and program.remove, and the editor's sandbox checks that they
// and stage.check share (program_checks.h).
//
// The edits follow the rules of the panel and control edits: they work on a
// copy of the spec, refuse an edit whose own rule fails with the codes of
// programs/sandbox.h (program.exists, program.not_found, stage.not_found,
// stage.not_scene_panel, program.bad_target, program.over_limits and the
// sandbox's checks), then hand the result to the validator, whose first error
// refuses the edit and whose warnings are kept. An edit that adds a program
// or a stage to a version 0 spec raises the version to 1 in the same edit.
// A refusal of the sandbox carries its line and column in the message and in
// `where` (args.wgsl:<line>:<column>). Without the compiler library an edit
// applies with the warning program.not_checked, in `messages` and in
// `warnings`, because the hosts check each source again before it runs.
#include "program_checks.h"

#include <caliper_builder/session/program_commands.h>
#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/patch.h>

#include <algorithm>
#include <utility>

namespace caliper_builder {

// ---- the editor's checks ---------------------------------------------------------

namespace program_checks {

namespace {

namespace lim = spec::program_limits;
using programs::Verdict;

// The entry point the editor appends to a stage to check its function's
// signature; it calls the function with the contract's input struct.
std::string check_entry(spec::StageKind kind) {
    if (kind == spec::StageKind::material)
        return "@fragment fn caliper_stage_check() -> @location(0) vec4<f32> {\n"
               "    return material(MaterialIn(vec4<f32>(0.0), 0.0, 0u, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0u));\n"
               "}\n";
    return "@vertex fn caliper_stage_check() -> @builtin(position) vec4<f32> {\n"
           "    let o: VertexOut = vertex(VertexIn(vec3<f32>(0.0), 0.0, 0.0, 0u, 0.0, 0u));\n"
           "    return vec4<f32>(o.position, o.radius);\n"
           "}\n";
}

std::uint32_t line_count(std::string_view s) {
    return static_cast<std::uint32_t>(std::count(s.begin(), s.end(), '\n')) + 1;
}

// A compiler verdict of a composed stage, moved to the lines of the stage's
// own source: a place in the prelude or after the source has none.
Verdict in_stage_source(Verdict v, std::uint32_t source_lines) {
    if (v.code == programs::kProgramCompileFailed) v.code = programs::kStageCompileFailed;
    const std::uint32_t first = programs::stage_prelude_lines() + 1;
    if (v.line >= first && v.line < first + source_lines) {
        v.line -= programs::stage_prelude_lines();
    } else {
        v.line = 0;
        v.column = 0;
    }
    return v;
}

}  // namespace

Verdict not_checked_warning() {
    Verdict v;
    v.code = programs::kProgramNotChecked;
    v.message = "This build has no WGSL compiler, so the source was not compiled here; the script or the viewer "
                "checks it again before it runs.";
    v.where = "wgsl";
    return v;
}

nlohmann::json message_json(std::string_view severity, const Verdict& v) {
    return {{"severity", std::string(severity)},
            {"code", v.code},
            {"message", v.message},
            {"line", v.line},
            {"column", v.column}};
}

Outcome check_program_source(const spec::Program& p) {
    Outcome out;
    out.error = programs::check_source(p.wgsl, lim::kProgramSourceBytes);
    if (!out.error.ok() || !programs::compiler_available()) return out;
    programs::Compiled c = programs::compile_and_reflect(p.wgsl);
    if (!c.verdict.ok()) {
        out.error = std::move(c.verdict);
        return out;
    }
    out.compiled = true;
    out.reflection = std::move(c.reflection);
    out.error = programs::check_program(p, out.reflection);
    out.checked = out.error.ok();
    return out;
}

Outcome check_stage_source(const spec::Stage& s) {
    Outcome out;
    out.error = programs::check_source(s.wgsl, lim::kStageSourceBytes);
    if (!out.error.ok()) return out;
    // Without the compiler, the scan of the source for a binding or an entry
    // point still runs.
    if (!programs::compiler_available()) {
        out.error = programs::check_stage(s, programs::Reflection{});
        return out;
    }
    const std::uint32_t lines = line_count(s.wgsl);
    std::string composed(programs::stage_prelude());
    composed += s.wgsl;
    programs::Compiled c = programs::compile_and_reflect(composed);
    if (!c.verdict.ok()) {
        out.error = in_stage_source(std::move(c.verdict), lines);
        return out;
    }
    out.compiled = true;
    out.reflection = std::move(c.reflection);
    out.error = programs::check_stage(s, out.reflection);
    if (!out.error.ok()) return out;
    composed += "\n";
    composed += check_entry(s.kind);
    programs::Compiled call = programs::compile_and_reflect(composed);
    if (!call.verdict.ok()) {
        Verdict v = in_stage_source(std::move(call.verdict), lines);
        if (v.line == 0)
            v.message = std::string("The stage must define ") +
                        (s.kind == spec::StageKind::material ? "fn material(in: MaterialIn) -> vec4<f32>"
                                                             : "fn vertex(in: VertexIn) -> VertexOut") +
                        ", and the call of it does not compile: " + v.message;
        out.error = std::move(v);
        return out;
    }
    out.checked = true;
    return out;
}

Outcome check_compute_source(std::string_view wgsl, const std::optional<std::string>& entry) {
    Outcome out;
    out.error = programs::check_source(wgsl, lim::kProgramSourceBytes);
    if (!out.error.ok() || !programs::compiler_available()) return out;
    programs::Compiled c = programs::compile_and_reflect(wgsl);
    if (!c.verdict.ok()) {
        out.error = std::move(c.verdict);
        return out;
    }
    out.compiled = true;
    out.reflection = std::move(c.reflection);
    std::vector<const programs::ReflectedEntry*> compute;
    for (const programs::ReflectedEntry& e : out.reflection.entries)
        if (e.stage == programs::EntryStage::compute && (!entry || e.name == *entry)) compute.push_back(&e);
    if (compute.size() != 1) {
        out.error.code = programs::kProgramNoEntry;
        out.error.where = "entry";
        out.error.message = entry ? "The source has no compute entry point " + *entry + "."
                            : compute.empty()
                                ? std::string("The source has no compute entry point; add a function with "
                                              "@compute @workgroup_size(...).")
                                : "The source has " + std::to_string(compute.size()) +
                                      " compute entry points; name the one to check with entry.";
        return out;
    }
    const auto& ws = compute.front()->workgroup_size;
    const std::uint64_t n = std::uint64_t(ws[0]) * ws[1] * ws[2];
    if (n == 0 || n > lim::kWorkgroupInvocations) {
        out.error.code = programs::kProgramOverLimits;
        out.error.where = "entry";
        out.error.message = "The entry point " + compute.front()->name + " has a workgroup of " + std::to_string(n) +
                            " invocations; the limit is " + std::to_string(lim::kWorkgroupInvocations) + ".";
        return out;
    }
    out.checked = true;
    return out;
}

}  // namespace program_checks

// ---- the edits -------------------------------------------------------------------

namespace {

using nlohmann::json;
using programs::Verdict;
namespace lim = spec::program_limits;

Error fail(std::string code, std::string message, std::string where = {}) {
    return Error(std::move(code), std::move(message), std::move(where));
}

SpecEdit refuse(Error err) {
    SpecEdit out;
    out.error = std::move(err);
    return out;
}

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

// A declaration key of a verdict's `where`, as opposed to a WGSL variable.
bool declaration_key(const std::string& where) {
    for (const char* key : {"id", "title", "wgsl", "entry", "inputs", "output", "limits", "workgroups", "panel",
                            "entity", "kind"}) {
        const std::size_t n = std::char_traits<char>::length(key);
        if (where.compare(0, n, key) == 0 && (where.size() == n || where[n] == '.' || where[n] == '['))
            return true;
    }
    return false;
}

// A refusal of the sandbox as the command's error: the line and column go
// into the message and into `where`, else the declaration key under `base`.
Error verdict_error(const Verdict& v, const std::string& base) {
    if (v.line > 0)
        return fail(v.code, v.message + " (line " + std::to_string(v.line) + ", column " + std::to_string(v.column) + ")",
                    base + ".wgsl:" + std::to_string(v.line) + ":" + std::to_string(v.column));
    return fail(v.code, v.message, declaration_key(v.where) ? base + "." + v.where : base + ".wgsl");
}

// Validates the edited spec: the first error refuses the edit; otherwise the
// warnings are kept, and an unchecked source adds program.not_checked.
SpecEdit finish(spec::Spec next, const spec::SourceResolver& resolver, json result, bool checked,
                const std::string& source_path) {
    SpecEdit out;
    spec::ValidationReport report = spec::validate(next, resolver);
    if (Error err = first_spec_error(report.diagnostics)) {
        out.error = std::move(err);
        return out;
    }
    for (auto& d : report.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    json messages = json::array();
    if (!checked && !source_path.empty()) {
        const Verdict w = program_checks::not_checked_warning();
        messages.push_back(program_checks::message_json("warning", w));
        spec::Diagnostic d;
        d.severity = spec::Severity::warning;
        d.code = w.code;
        d.message = w.message;
        d.path = source_path;
        out.warnings.push_back(std::move(d));
    }
    if (!source_path.empty()) result["messages"] = std::move(messages);
    out.spec = std::move(next);
    out.result = std::move(result);
    return out;
}

void raise_version(spec::Spec& s) {
    if (s.version < spec::kInstrumentVersion) s.version = spec::kInstrumentVersion;
}

std::size_t program_index(const spec::Spec& s, const std::string& id) {
    for (std::size_t i = 0; i < s.programs.size(); ++i)
        if (s.programs[i].id == id) return i;
    return s.programs.size();
}

std::size_t stage_index(const spec::Spec& s, const std::string& panel, const std::string& entity,
                        spec::StageKind kind) {
    for (std::size_t i = 0; i < s.stages.size(); ++i)
        if (s.stages[i].panel == panel && s.stages[i].entity == entity && s.stages[i].kind == kind) return i;
    return s.stages.size();
}

// The scene panel an item draws in: an error with the item's code when the
// spec has no such panel or it is of another primitive.
Error scene_panel(const spec::Spec& s, const std::string& panel, const char* code, const std::string& where) {
    const spec::Panel* p = s.find_panel(panel);
    if (!p) return fail(code, "The spec has no panel " + panel + "; name a scene panel of the open spec.", where);
    if (p->primitive != spec::Primitive::scene)
        return fail(code,
                    "The panel " + panel + " is a " + spec::to_string(p->primitive) + " panel; name a scene panel.",
                    where);
    return {};
}

// The program as the result of program.patch shows it: its canonical JSON
// without the extension table, which tool schemas cannot carry.
json program_result_json(const spec::Program& p) {
    json j = spec::to_json(p);
    j.erase("x");
    return j;
}

// The editor's checks of a program before it enters the spec: the
// declaration, the target panel and the source.
Error check_program_edit(const spec::Spec& s, const spec::Program& p, const std::string& base, bool* checked) {
    if (Verdict v = programs::check_declaration(p); !v.ok()) return verdict_error(v, base);
    if (p.output.kind == spec::OutputKind::geometry)
        if (Error e = scene_panel(s, p.output.panel, programs::kProgramBadTarget, base + ".output.panel")) return e;
    program_checks::Outcome o = program_checks::check_program_source(p);
    if (!o.error.ok()) return verdict_error(o.error, base);
    *checked = o.checked;
    return {};
}

SpecEdit stage_set(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    spec::Stage st;
    st.panel = str_arg(args, "panel");
    st.entity = str_arg(args, "entity");
    const auto kind = spec::stage_kind_from_name(str_arg(args, "kind"));
    if (!kind) return refuse(fail("spec.bad_value", "A stage's kind is material or vertex.", "args.kind"));
    st.kind = *kind;
    st.wgsl = str_arg(args, "wgsl");
    if (Verdict v = programs::check_stage_declaration(st); !v.ok()) return refuse(verdict_error(v, "args"));
    if (Error e = scene_panel(s, st.panel, programs::kStageNotScenePanel, "args.panel")) return refuse(std::move(e));
    const std::size_t index = stage_index(s, st.panel, st.entity, st.kind);
    if (index == s.stages.size() && s.stages.size() >= lim::kStages)
        return refuse(fail(programs::kProgramOverLimits,
                           "The spec has " + std::to_string(s.stages.size()) + " stages, the most it holds; remove "
                           "one first.",
                           "args"));
    program_checks::Outcome o = program_checks::check_stage_source(st);
    if (!o.error.ok()) return refuse(verdict_error(o.error, "args"));

    spec::Spec next = s;
    raise_version(next);
    if (index < next.stages.size()) {
        st.x = next.stages[index].x;
        next.stages[index] = std::move(st);
    } else {
        next.stages.push_back(std::move(st));
    }
    const std::size_t at = index < s.stages.size() ? index : next.stages.size() - 1;
    const spec::Stage& kept = next.stages[at];
    json result{{"panel", kept.panel}, {"entity", kept.entity}, {"kind", spec::stage_kind_name(kept.kind)}};
    return finish(std::move(next), resolver, std::move(result), o.checked, "stage[" + std::to_string(at) + "].wgsl");
}

SpecEdit stage_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string panel = str_arg(args, "panel");
    const std::string entity = str_arg(args, "entity");
    const auto kind = spec::stage_kind_from_name(str_arg(args, "kind"));
    const std::size_t index = kind ? stage_index(s, panel, entity, *kind) : s.stages.size();
    if (index == s.stages.size())
        return refuse(fail(programs::kStageNotFound,
                           "The panel " + panel + " has no " + str_arg(args, "kind") + " stage on the entity " +
                               entity + ".",
                           "args"));
    spec::Spec next = s;
    next.stages.erase(next.stages.begin() + static_cast<std::ptrdiff_t>(index));
    json result{{"panel", panel}, {"entity", entity}, {"kind", spec::stage_kind_name(*kind)}};
    return finish(std::move(next), resolver, std::move(result), true, "");
}

SpecEdit program_create(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    json item = args;
    item.erase("dry_run");
    spec::Decoded<spec::Program> d = spec::program_from_json(item, "args");
    if (Error e = first_spec_error(d.diagnostics)) return refuse(std::move(e));
    spec::Program p = std::move(d.value);
    if (s.find_program(p.id))
        return refuse(fail(programs::kProgramExists, "The spec already has a program " + p.id + "; choose another id.",
                           "args.id"));
    if (s.programs.size() >= lim::kPrograms)
        return refuse(fail(programs::kProgramOverLimits,
                           "The spec has " + std::to_string(s.programs.size()) + " programs, the most it holds; "
                           "remove one first.",
                           "args"));
    bool checked = false;
    if (Error e = check_program_edit(s, p, "args", &checked)) return refuse(std::move(e));
    spec::Spec next = s;
    raise_version(next);
    const std::string id = p.id;
    next.programs.push_back(std::move(p));
    return finish(std::move(next), resolver, json{{"id", id}}, checked,
                  "program[" + std::to_string(next.programs.size() - 1) + "].wgsl");
}

SpecEdit program_patch(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const std::size_t index = program_index(s, id);
    if (index == s.programs.size())
        return refuse(fail(programs::kProgramNotFound, "The spec has no program " + id + ".", "args.id"));
    const json patch = args.contains("patch") && args["patch"].is_object() ? args["patch"] : json::object();
    if (patch.contains("id"))
        return refuse(fail("spec.bad_patch", "A program's id cannot change; remove it and create another.",
                           "args.patch.id"));
    json current = spec::to_json(s.programs[index]);
    json effective = patch;
    // A patch that changes the output's kind replaces the output, because the
    // keys of one kind do not fit the other.
    if (patch.contains("output") && patch["output"].is_object() && patch["output"].contains("kind") &&
        patch["output"]["kind"] != current["output"]["kind"])
        current.erase("output");
    spec::Decoded<spec::Program> d = spec::program_from_json(spec::merge_patch(current, effective), "args.patch");
    if (Error e = first_spec_error(d.diagnostics)) return refuse(std::move(e));
    spec::Program p = std::move(d.value);
    p.id = id;
    bool checked = false;
    if (Error e = check_program_edit(s, p, "args.patch", &checked)) return refuse(std::move(e));
    spec::Spec next = s;
    raise_version(next);
    next.programs[index] = p;
    // An unchanged source needs no new warning.
    const bool same_source = s.programs[index].wgsl == p.wgsl;
    return finish(std::move(next), resolver, json{{"program", program_result_json(p)}}, checked || same_source,
                  "program[" + std::to_string(index) + "].wgsl");
}

SpecEdit program_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const std::size_t index = program_index(s, id);
    if (index == s.programs.size())
        return refuse(fail(programs::kProgramNotFound, "The spec has no program " + id + ".", "args.id"));
    spec::Spec next = s;
    next.programs.erase(next.programs.begin() + static_cast<std::ptrdiff_t>(index));
    return finish(std::move(next), resolver, json{{"id", id}}, true, "");
}

}  // namespace

SpecEdit apply_program_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                            const spec::SourceResolver& resolver) {
    if (name == "stage.set") return stage_set(spec, args, resolver);
    if (name == "stage.remove") return stage_remove(spec, args, resolver);
    if (name == "program.create") return program_create(spec, args, resolver);
    if (name == "program.patch") return program_patch(spec, args, resolver);
    if (name == "program.remove") return program_remove(spec, args, resolver);
    return refuse(fail("command.unknown", "This is not a command that edits the spec's programs or stages.",
                       std::string(name)));
}

}  // namespace caliper_builder
