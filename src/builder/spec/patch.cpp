// JSON merge patches (RFC 7386) of one panel or one source of a spec.
#include <caliper_builder/spec/patch.h>

#include <algorithm>

#include <caliper_builder/spec/format.h>

#include "internal.h"

namespace caliper_builder::spec {

namespace {

using detail::json;

enum class Item { panel, source, control };

PatchResult patch_item(const Spec& spec, Item item, std::string_view id, const json& patch,
                       const SourceResolver& resolver) {
    PatchResult r;
    r.spec = spec;
    try {
        const bool panel = item == Item::panel;
        const char* noun = panel ? "panel" : item == Item::control ? "control" : "source";
        std::size_t index = 0;
        bool found = false;
        const std::size_t count = panel                   ? spec.panels.size()
                                  : item == Item::control ? spec.controls.size()
                                                          : spec.sources.size();
        for (std::size_t i = 0; i < count && !found; ++i) {
            const std::string& item_id = panel                   ? spec.panels[i].id
                                         : item == Item::control ? spec.controls[i].id
                                                                 : spec.sources[i].id;
            if (item_id == id) {
                index = i;
                found = true;
            }
        }
        if (!found) {
            r.diagnostics.push_back({Severity::error, "spec.not_found",
                                     std::string("The spec has no ") + noun + " " + detail::quote(id) + ".", ""});
            return r;
        }
        const std::string path = detail::index_path(noun, index);
        if (!patch.is_object()) {
            r.diagnostics.push_back(
                {Severity::error, "spec.bad_patch", "A patch of one " + std::string(noun) + " must be a JSON object.",
                 path});
            return r;
        }
        {
            std::vector<Diagnostic> found_errors;
            detail::Sink sink(found_errors);
            std::size_t values = 0;
            if (!detail::check_values(patch, path, 2, values, sink, /*allow_null=*/true)) {
                for (Diagnostic& d : found_errors) {
                    d.code = d.code == "spec.too_deep" || d.code == "spec.too_many" ? std::string("spec.bad_patch")
                                                                                    : d.code;
                    r.diagnostics.push_back(std::move(d));
                }
                return r;
            }
        }
        const json target = panel                   ? to_json(spec.panels[index])
                            : item == Item::control ? to_json(spec.controls[index])
                                                    : to_json(spec.sources[index]);
        const json merged = merge_patch(target, patch);
        auto new_id = merged.find("id");
        if (new_id == merged.end() || !new_id->is_string() || new_id->get<std::string>() != id) {
            r.diagnostics.push_back({Severity::error, "spec.bad_patch",
                                     std::string("A patch cannot change or remove the id of a ") + noun +
                                         "; bindings and the layout name it.",
                                     path + ".id"});
            return r;
        }
        Spec next = spec;
        if (panel) {
            Decoded<Panel> decoded = panel_from_json(merged, path);
            if (!decoded.ok()) {
                r.diagnostics = std::move(decoded.diagnostics);
                return r;
            }
            next.panels[index] = std::move(decoded.value);
        } else if (item == Item::control) {
            Decoded<Control> decoded = control_from_json(merged, path);
            if (!decoded.ok()) {
                r.diagnostics = std::move(decoded.diagnostics);
                return r;
            }
            next.controls[index] = std::move(decoded.value);
        } else {
            Decoded<Source> decoded = source_from_json(merged, path);
            if (!decoded.ok()) {
                r.diagnostics = std::move(decoded.diagnostics);
                return r;
            }
            next.sources[index] = std::move(decoded.value);
        }
        // A patch that makes a panel a scene raises a version 0 spec to version 1 in the same step.
        next.version = std::max(next.version, required_version(next));
        ValidationReport report = validate(next, resolver);
        r.applied = true;
        r.spec = std::move(next);
        r.diagnostics = std::move(report.diagnostics);
    } catch (...) {
        r.applied = false;
        r.spec = spec;
        r.diagnostics.push_back({Severity::error, "spec.internal", "The spec library failed unexpectedly.", ""});
    }
    return r;
}

}  // namespace

json merge_patch(const json& target, const json& patch) {
    json result = target;
    result.merge_patch(patch);
    return result;
}

PatchResult patch_panel(const Spec& spec, std::string_view panel_id, const json& patch,
                        const SourceResolver& resolver) {
    return patch_item(spec, Item::panel, panel_id, patch, resolver);
}

PatchResult patch_source(const Spec& spec, std::string_view source_id, const json& patch,
                         const SourceResolver& resolver) {
    return patch_item(spec, Item::source, source_id, patch, resolver);
}

PatchResult patch_control(const Spec& spec, std::string_view control_id, const json& patch,
                          const SourceResolver& resolver) {
    return patch_item(spec, Item::control, control_id, patch, resolver);
}

}  // namespace caliper_builder::spec
