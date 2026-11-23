// The program edits of the open spec (design/gpu-programs.md, Program
// Commands): program.create, program.patch and program.remove, and the
// editor's sandbox checks that they use (program_checks.h).
//
// The edits follow the rules of the panel and control edits: they work on a
// copy of the spec, refuse an edit whose own rule fails with the codes of
// programs/sandbox.h (program.exists, program.not_found, program.bad_target, program.over_limits and the
// sandbox's checks), then hand the result to the validator, whose first error
// refuses the edit and whose warnings are kept. An edit that adds a program
// to a version 0 spec raises the version to 1 in the same edit.
// A refusal of the sandbox carries its line and column in the message and in
// `where` (args.wgsl:<line>:<column>). Without the compiler library an edit
// applies with the warning program.not_checked, in `messages` and in
// `warnings`, because the hosts check each source again before it runs.
#include <caliper_builder/programs/pieces.h>
#include "program_checks.h"

#include <caliper_builder/session/program_commands.h>
#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/patch.h>

#include <utility>

namespace caliper_builder {

// ---- the editor's checks ---------------------------------------------------------

namespace program_checks {

namespace lim = spec::program_limits;
using programs::Verdict;

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
    // An instance of a piece: its piece is checked alone, and the session
    // resolves its pin and its bindings after the edit (piece_commands.h).
    if (spec::is_piece_instance(p)) {
        *checked = true;
        if (p.panel) return scene_panel(s, *p.panel, programs::kProgramBadTarget, base + ".panel");
        return {};
    }
    if (Verdict v = programs::check_declaration(p); !v.ok()) return verdict_error(v, base);
    if (p.output.kind == spec::OutputKind::geometry)
        if (Error e = scene_panel(s, p.output.panel, programs::kProgramBadTarget, base + ".output.panel")) return e;
    program_checks::Outcome o = program_checks::check_program_source(p);
    if (!o.error.ok()) return verdict_error(o.error, base);
    *checked = o.checked;
    return {};
}

// The inline piece of program.create's `inline`: it joins the spec's array
// `piece` in the same step, and a program without `piece` pins it. The same
// name and version with other contents is piece.exists.
Error add_inline_piece(spec::Spec& next, const json& args, json& item) {
    const auto it = args.find("inline");
    if (it == args.end()) return {};
    spec::PieceParse parsed = spec::piece_from_json_value(*it, "args.inline");
    if (Error e = first_spec_error(parsed.diagnostics)) return e;
    spec::Piece piece = std::move(parsed.piece);
    if (!item.contains("piece"))
        item["piece"] = {{"name", piece.name}, {"version", piece.version}, {"from", "inline"}};
    const json& pin = item["piece"];
    if (!pin.is_object() || pin.value("name", "") != piece.name || pin.value("version", std::int64_t(1)) != std::int64_t(piece.version) ||
        pin.value("from", "library") != "inline")
        return fail("spec.bad_value", "A program with inline pins that piece: from inline, with its name and version.",
                    "args.piece");
    for (const spec::Piece& p : next.pieces) {
        if (p.name != piece.name || p.version != piece.version) continue;
        if (p == piece) return {};
        return fail(programs::kPieceExists,
                    "The spec already has the inline piece " + piece.name + " version " +
                        std::to_string(piece.version) + " with other contents; give the new one another version.",
                    "args.inline.version");
    }
    next.pieces.push_back(std::move(piece));
    return {};
}

SpecEdit program_create(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    json item = args;
    item.erase("dry_run");
    item.erase("inline");
    spec::Spec with_inline = s;
    if (Error e = add_inline_piece(with_inline, args, item)) return refuse(std::move(e));
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
    spec::Spec next = std::move(with_inline);
    raise_version(next);
    const std::string id = p.id;
    next.programs.push_back(std::move(p));
    // The path is made before the call, which moves `next`.
    const std::string source_path = "program[" + std::to_string(next.programs.size() - 1) + "].wgsl";
    return finish(std::move(next), resolver, json{{"id", id}}, checked, source_path);
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
    // A patch that pins a piece replaces the pin, so that an old hash never
    // stays beside a new version; the session writes the new hash.
    if (patch.contains("piece") && patch["piece"].is_object()) current.erase("piece");
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
    if (name == "program.create") return program_create(spec, args, resolver);
    if (name == "program.patch") return program_patch(spec, args, resolver);
    if (name == "program.remove") return program_remove(spec, args, resolver);
    return refuse(fail("command.unknown", "This is not a command that edits the spec's programs.",
                       std::string(name)));
}

}  // namespace caliper_builder
