// The annotation commands (design/instrument-interface.md, Commands):
// annotation.set adds an annotation to a panel or replaces the one with the
// same id in its place, and annotation.remove takes one out.
//
// Both follow the rules of the panel edits: they work on a copy of the spec,
// refuse an edit whose own rule fails (panel.not_found,
// annotation.not_found, or the decoder's first error for a malformed
// annotation table) and hand the result to the validator, whose first error
// refuses the edit and whose warnings are kept. An edit that adds an
// annotation to a version 0 spec raises the version to 1 in the same edit.
#include <caliper_builder/session/interactive_commands.h>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/validate.h>

#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

SpecEdit refuse(Error err) {
    SpecEdit out;
    out.error = std::move(err);
    return out;
}

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

// The index of panel `id` in `s`, or s.panels.size().
std::size_t panel_index(const spec::Spec& s, const std::string& id) {
    for (std::size_t i = 0; i < s.panels.size(); ++i)
        if (s.panels[i].id == id) return i;
    return s.panels.size();
}

SpecEdit finish(spec::Spec next, const spec::SourceResolver& resolver, json result) {
    SpecEdit out;
    spec::ValidationReport report = spec::validate(next, resolver);
    if (Error err = first_spec_error(report.diagnostics)) {
        out.error = std::move(err);
        return out;
    }
    for (auto& d : report.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    out.spec = std::move(next);
    out.result = std::move(result);
    return out;
}

SpecEdit annotation_set(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string panel = str_arg(args, "panel");
    const std::size_t pi = panel_index(s, panel);
    if (pi == s.panels.size())
        return refuse(Error("panel.not_found", "The spec has no panel " + panel + ".", "args.panel"));
    const json item = args.contains("annotation") ? args["annotation"] : json();
    if (!item.is_object())
        return refuse(Error("spec.wrong_type", "The annotation must be a JSON object with an id, a kind and its keys.",
                            "args.annotation"));
    spec::Decoded<spec::Annotation> decoded = spec::annotation_from_json(item, "args.annotation");
    if (Error err = first_spec_error(decoded.diagnostics)) return refuse(std::move(err));

    spec::Spec next = s;
    next.version = std::max(next.version, spec::kFormatVersion);
    std::vector<spec::Annotation>& list = next.panels[pi].annotations;
    bool replaced = false;
    for (spec::Annotation& a : list)
        if (a.id == decoded.value.id) {
            a = decoded.value;
            replaced = true;
            break;
        }
    if (!replaced) list.push_back(std::move(decoded.value));
    return finish(std::move(next), resolver, json{{"panel", panel}});
}

SpecEdit annotation_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string panel = str_arg(args, "panel");
    const std::string id = str_arg(args, "id");
    const std::size_t pi = panel_index(s, panel);
    if (pi == s.panels.size())
        return refuse(Error("panel.not_found", "The spec has no panel " + panel + ".", "args.panel"));
    spec::Spec next = s;
    std::vector<spec::Annotation>& list = next.panels[pi].annotations;
    for (std::size_t i = 0; i < list.size(); ++i)
        if (list[i].id == id) {
            list.erase(list.begin() + static_cast<std::ptrdiff_t>(i));
            return finish(std::move(next), resolver, json{{"panel", panel}});
        }
    return refuse(Error("annotation.not_found", "The panel " + panel + " has no annotation " + id + ".", "args.id"));
}

}  // namespace

SpecEdit apply_annotation_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                              const spec::SourceResolver& resolver) {
    if (name == "annotation.set") return annotation_set(spec, args, resolver);
    if (name == "annotation.remove") return annotation_remove(spec, args, resolver);
    SpecEdit out;
    out.error = not_implemented(name);
    return out;
}

}  // namespace caliper_builder
