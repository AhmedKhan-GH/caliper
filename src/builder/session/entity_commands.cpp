// The entity commands, entity.set and entity.remove
// (design/instrument-interface.md, Commands). Each one works on a copy of the
// spec, refuses an edit whose own rule fails (panel.not_found,
// entity.not_scene_panel, entity.not_found or the codec's diagnostic of a
// malformed entity), and hands the result to the validator, whose first error
// refuses the edit and whose warnings are kept.
#include <caliper_builder/session/interactive_commands.h>

#include <caliper_builder/spec/format.h>

#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

std::string text_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

SpecEdit refuse_with(Error err) {
    SpecEdit out;
    out.error = std::move(err);
    return out;
}

// Validates the edited spec: the first error refuses the edit, and the
// warnings stay with the result.
SpecEdit finish_edit(spec::Spec next, const spec::SourceResolver& resolver, json result) {
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

// The scene panel that the command names, or the command's refusal when the
// spec has no such panel or the panel is of another primitive.
Error find_scene_panel(const spec::Spec& s, const std::string& id) {
    const spec::Panel* p = s.find_panel(id);
    if (!p) return Error("panel.not_found", "The spec has no panel " + id + ".", "args.panel");
    if (p->primitive != spec::Primitive::scene)
        return Error("entity.not_scene_panel",
                     "The panel " + id + " is a " + spec::to_string(p->primitive) + " panel; name a scene panel.",
                     "args.panel");
    return {};
}

std::size_t panel_index(const spec::Spec& s, const std::string& id) {
    for (std::size_t i = 0; i < s.panels.size(); ++i)
        if (s.panels[i].id == id) return i;
    return s.panels.size();
}

// entity.set: replaces the entity of the same path in place, or appends it.
SpecEdit entity_set(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = text_arg(args, "panel");
    if (Error e = find_scene_panel(s, id)) return refuse_with(std::move(e));
    const auto it = args.find("entity");
    if (it == args.end() || !it->is_object())
        return refuse_with(Error("spec.bad_value", "entity.set needs an entity table.", "args.entity"));
    auto decoded = spec::entity_from_json(*it, "args.entity");
    if (!decoded.ok()) return refuse_with(first_spec_error(decoded.diagnostics));

    spec::Spec next = s;
    next.version = std::max(next.version, spec::kFormatVersion);
    spec::Panel& panel = next.panels[panel_index(next, id)];
    bool replaced = false;
    for (spec::Entity& e : panel.entities)
        if (e.path == decoded.value.path) {
            e = std::move(decoded.value);
            replaced = true;
            break;
        }
    if (!replaced) panel.entities.push_back(std::move(decoded.value));
    return finish_edit(std::move(next), resolver, json{{"panel", id}});
}

// entity.remove: takes out the entity of the path, or refuses with entity.not_found.
SpecEdit entity_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = text_arg(args, "panel");
    const std::string path = text_arg(args, "path");
    if (Error e = find_scene_panel(s, id)) return refuse_with(std::move(e));
    spec::Spec next = s;
    spec::Panel& panel = next.panels[panel_index(next, id)];
    for (std::size_t i = 0; i < panel.entities.size(); ++i)
        if (panel.entities[i].path == path) {
            panel.entities.erase(panel.entities.begin() + static_cast<std::ptrdiff_t>(i));
            return finish_edit(std::move(next), resolver, json{{"panel", id}});
        }
    return refuse_with(Error("entity.not_found", "The panel " + id + " has no entity " + path + ".", "args.path"));
}

}  // namespace

// Runs entity.set or entity.remove on a copy of the spec; any other name
// answers command.not_implemented.
SpecEdit apply_entity_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                          const spec::SourceResolver& resolver) {
    if (name == "entity.set") return entity_set(spec, args, resolver);
    if (name == "entity.remove") return entity_remove(spec, args, resolver);
    return refuse_with(not_implemented(name));
}

}  // namespace caliper_builder
