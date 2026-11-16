#pragma once
// The edits of the open spec, as pure functions of a spec and the arguments
// of a command (docs/plans/builder-session.md, Editing the open spec).
//
// apply_spec_edit() runs one of spec.set_text, source.add, source.patch,
// source.remove, panel.create, panel.patch, panel.remove, layout.place and
// layout.set on a copy of a spec and validates the result with the given
// SourceResolver. It refuses an edit whose own rule fails (source.exists,
// source.not_found, source.in_use, panel.exists, panel.not_found,
// layout.same_panel, layout.invalid and the like) and an edit that leaves the
// spec with an error, with the validator's own code and the document path in
// `where`; otherwise it returns the new spec, the validator's warnings and the
// command's own result fields (the source, the panel or the flat layout). The
// session's handlers and the journal's replay both call it, so a replay makes
// exactly the edits the session made.
//
// The conversions between the spec and the command table's forms live here
// too: a source or a panel as the arguments of source.add or panel.create (the
// canonical JSON without its extension tables `x`, which tool schemas cannot
// carry), and the layout tree as the flat list of nodes of layout.set. Nothing
// here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

// The arguments of source.add and panel.create that recreate an item: its
// canonical JSON with every extension table `x` removed.
nlohmann::json source_command_args(const spec::Source& source);
nlohmann::json panel_command_args(const spec::Panel& panel);

// A panel as the results of panel.patch show it: id, primitive, title (null
// when absent), bind and options (empty objects when absent), without `x`.
nlohmann::json panel_result_json(const spec::Panel& panel);
// A source as the result of source.patch shows it, without `x`.
nlohmann::json source_result_json(const spec::Source& source);

// The layout a spec draws: its own, or the default of the format (one panel
// alone, or a vertical split of every panel in order); empty with no panels.
std::optional<spec::LayoutNode> effective_layout(const spec::Spec& spec);

// The flat layout of the command table: {"root": "<id>", "nodes": [{"id",
// "split", "ratio", "title", "children"}]}. Node ids are n1, n2, ... in
// pre-order, skipping any id that is a panel of the spec. A single panel is
// {"root": "<panel>", "nodes": []}. Extension tables of layout nodes are not
// carried.
nlohmann::json layout_to_flat(const spec::LayoutNode& root, const spec::Spec& spec);

struct FlatLayoutParse {
    Error error;  // layout.invalid or panel.not_found, `where` under args.layout
    spec::LayoutNode root;
};
// The tree of a flat layout. Node ids must be unique and must not be panel
// ids of the spec, every node is used once and is reachable from the root,
// tabs take no ratio, and every name that is not a node must be a panel of the
// spec. The spec's own rules (children counts, depth, placement) are left to
// the validator.
FlatLayoutParse layout_from_flat(const nlohmann::json& flat, const spec::Spec& spec);

// A diagnostic in the command table's form, {"code", "message", "where"}, with
// `where` the document path, or the line and column when there is no path.
nlohmann::json diagnostic_json(const spec::Diagnostic& diagnostic);
// The warnings of a report, in the command table's form.
nlohmann::json warnings_json(const std::vector<spec::Diagnostic>& diagnostics);
// The first error of a report as an Error, or an empty Error when there is none.
Error first_spec_error(const std::vector<spec::Diagnostic>& diagnostics);

// The commands that apply_spec_edit() runs.
bool is_spec_edit_command(std::string_view name);

struct SpecEdit {
    Error error;
    spec::Spec spec;                          // the spec after the edit
    std::vector<spec::Diagnostic> warnings;   // the validator's warnings about it
    nlohmann::json result = nlohmann::json::object();  // the command's own result fields
};

// Runs one edit. `args` are the command's arguments, already checked against
// its schema; a `dry_run` member is ignored here.
SpecEdit apply_spec_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                         const spec::SourceResolver& resolver = {});

// The empty spec of spec.new: {"id", "title", "description"}.
spec::Spec new_spec_from_args(const nlohmann::json& args);

// Parses and validates the text of a spec, with every diagnostic located in
// the text. The spec is empty when the text has an error.
struct SpecTextCheck {
    Error error;  // the first error, when there is one
    spec::Spec spec;
    std::vector<spec::Diagnostic> diagnostics;  // errors and warnings
};
SpecTextCheck check_spec_text(std::string_view text, const spec::SourceResolver& resolver = {});

}  // namespace caliper_builder
