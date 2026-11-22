// camera.save, the one camera command that edits the spec
// (design/instrument-interface.md, Starting Camera and Saved Views). The
// session adds the argument `orbit` from the app before the edit runs; without
// it, as in a replay of the journal with no app, the edit is refused with
// camera.no_orbit. The validator refuses a 17th view.
#include <caliper_builder/session/instrument_commands.h>

#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

SpecEdit refuse_with(Error err) {
    SpecEdit out;
    out.error = std::move(err);
    return out;
}

// A number member of an object, or `fallback` when it is missing or not a number.
double number_member(const json& object, const char* key, double fallback) {
    const auto it = object.find(key);
    return it != object.end() && it->is_number() ? it->get<double>() : fallback;
}

// camera.save: writes the view into the panel's saved views, replacing a view
// of the same name in place or appending it.
SpecEdit camera_save(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const auto pit = args.find("panel");
    const std::string id = pit != args.end() && pit->is_string() ? pit->get<std::string>() : std::string();
    const auto nit = args.find("name");
    const std::string view_name = nit != args.end() && nit->is_string() ? nit->get<std::string>() : std::string();
    const spec::Panel* found = s.find_panel(id);
    if (!found) return refuse_with(Error("panel.not_found", "The spec has no panel " + id + ".", "args.panel"));
    if (found->primitive != spec::Primitive::scene)
        return refuse_with(Error("camera.not_scene_panel",
                                 "The panel " + id + " is a " + spec::to_string(found->primitive) +
                                     " panel; name a scene panel.",
                                 "args.panel"));
    const auto oit = args.find("orbit");
    if (oit == args.end() || !oit->is_object())
        return refuse_with(Error("camera.no_orbit", "camera.save needs the app's current orbit.", "args.orbit"));

    spec::SavedView view;
    view.name = view_name;
    const auto tit = oit->find("target");
    if (tit != oit->end() && tit->is_array() && tit->size() == 3)
        for (std::size_t i = 0; i < 3; ++i)
            view.target[i] = (*tit)[i].is_number() ? (*tit)[i].get<double>() : 0.0;
    view.distance = number_member(*oit, "distance", view.distance);
    view.azimuth = number_member(*oit, "azimuth", view.azimuth);
    view.elevation = number_member(*oit, "elevation", view.elevation);

    spec::Spec next = s;
    // Saved views are a key of version 1, as the other instrument edits raise it.
    if (next.version < spec::kInstrumentVersion) next.version = spec::kInstrumentVersion;
    spec::Panel* panel = nullptr;
    for (spec::Panel& p : next.panels)
        if (p.id == id) panel = &p;
    bool replaced = false;
    for (spec::SavedView& v : panel->options.views)
        if (v.name == view_name) {
            v = view;
            replaced = true;
            break;
        }
    if (!replaced) panel->options.views.push_back(std::move(view));

    SpecEdit out;
    spec::ValidationReport report = spec::validate(next, resolver);
    if (Error err = first_spec_error(report.diagnostics)) {
        out.error = std::move(err);
        return out;
    }
    for (auto& d : report.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    out.spec = std::move(next);
    out.result = json{{"panel", id}};
    return out;
}

}  // namespace

// Runs camera.save on a copy of the spec; any other name answers
// command.not_implemented.
SpecEdit apply_camera_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                          const spec::SourceResolver& resolver) {
    if (name == "camera.save") return camera_save(spec, args, resolver);
    return refuse_with(not_implemented(name));
}

}  // namespace caliper_builder
