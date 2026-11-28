#pragma once
// JSON merge patches (RFC 7386) of one panel, one source or one control of a
// spec, which the command layer's panel.patch, source.patch and control.patch
// call.
//
// The patch applies to the item's canonical JSON form (format.h): a key set to
// null is removed, an object merges key by key, and any other value replaces
// what was there. The patched item is decoded with the checks of a whole
// document, and then the whole spec is validated again.
//
// A patch is refused, with `applied` false and the spec unchanged, when the
// item does not exist (spec.not_found), when the patch is not a JSON object,
// nests more than limits::kDepth levels or holds more than limits::kValues
// values, or would change or remove the item's id (spec.bad_patch), or when the
// patched item does not decode. Otherwise `applied` is true, `spec` holds the
// patched spec and `diagnostics` its full validation; whether errors elsewhere
// in the spec refuse the change is the command layer's choice. Nothing here
// throws.
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>

namespace caliper_builder::spec {

struct PatchResult {
    bool applied = false;
    Spec spec;
    std::vector<Diagnostic> diagnostics;
    bool ok() const { return applied && !has_errors(diagnostics); }
};

PatchResult patch_panel(const Spec& spec, std::string_view panel_id, const nlohmann::json& patch,
                        const SourceResolver& resolver = {});
PatchResult patch_source(const Spec& spec, std::string_view source_id, const nlohmann::json& patch,
                         const SourceResolver& resolver = {});
// A patch of one control (spec version 1), which control.patch calls: the
// same rules as a panel's, with the control's canonical JSON form.
PatchResult patch_control(const Spec& spec, std::string_view control_id, const nlohmann::json& patch,
                          const SourceResolver& resolver = {});

// RFC 7386 applied to any JSON value. The caller bounds the patch's depth.
nlohmann::json merge_patch(const nlohmann::json& target, const nlohmann::json& patch);

}  // namespace caliper_builder::spec
