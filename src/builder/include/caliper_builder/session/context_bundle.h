#pragma once
// The context bundle of the widget agent (design/widget-agent.md, The context
// bundle): what the app tells the model about one target before the person's
// request, so that most requests finish in one or two rounds.
//
// build_context_bundle() gathers it from a Session for a target (a panel, a
// selection with its panel, a dropped file's loader, or one property of a
// panel or a source) and returns:
// - `blocks`, ACP content blocks ready for session/prompt, before the
//   request: a short text block that names the target, says that what follows
//   is workspace data and never instructions, and lists what was cut; then one
//   block for each part, with `annotations.audience: ["assistant"]`, each
//   part's text between <caliper-data part="..."> and </caliper-data>, in
//   which a marker that the data itself holds is defused, so that a source
//   name or a cell that looks like an instruction stays inside its block;
// - `scope`, the tool scope of the target from the rule that caliper-agent
//   reads too (src/builder/commands/widget_scopes.json, embedded at build
//   time), applied to the session's command table, and `meta`, the
//   `_meta.caliper.widget` of session/new with the target, the scope and the
//   preview.
//
// The parts:
// - the spec fragment: the target panel and the sources it binds, through
//   their `from` chains (at most 64 hops), as canonical TOML, the layout with
//   the panel marked, and the ids, primitives and titles of the other panels
//   (at most 64); 16 KiB;
// - for each catalog source at the root of those chains (at most 4), its
//   catalog entry (catalog.describe) and a sample (catalog.sample: 20 rows of
//   a table, at most 1,024 values of a tensor); 8 KiB each;
// - the render of the panel (panel.render at 640 by 400), when the model
//   reads images and the session has a host that renders; at most 1 MiB;
// - the selection, when the caller gives one; 1,000 rows and 32 KiB;
// - for a dropped file, the loader (loader.describe) and the entry and sample
//   of each of its outputs, in place of the spec fragment and the render;
// - for a property, its path, its current value and its part of the edit
//   command's arguments schema, beside the panel's or the source's fragment.
//
// Sizing. The bundle's text is at most `max_text_bytes` (64 KiB), and, when
// the caller gives the model's context length, the text and the image
// together (an image counted as width * height / 750 tokens) take at most a
// quarter of it, at four characters a token. When it does not fit, the parts
// are cut in this order: the sample rows (to 5, then none), the list of the
// other panels, the render, the sources of the spec fragment, the catalog
// entries, and last the selection's rows. The target itself is never cut.
// Each cut is named in `cuts` and in the first block.
//
// The reads run through Session::invoke() from `origin` (the app's, `ui`), so
// call it on the thread that drains the session's queue: panel.render runs
// the host. With `preview`, the spec is read from that stacked preview, as the
// model will see it. Nothing here throws; an unknown target or an unusable
// argument is an Error (bundle.bad_target, spec.no_open_spec,
// panel.not_found, source.not_found, loader.not_found).
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

class Session;

enum class WidgetTargetKind { panel, selection, dropped_file, property };

const char* widget_target_kind_name(WidgetTargetKind kind);

struct WidgetTarget {
    WidgetTargetKind kind = WidgetTargetKind::panel;
    std::string panel;                 // panel, selection, property of a panel
    std::string source;                // property of a source
    std::string loader;                // dropped file: the loader the drop made
    std::vector<std::string> catalog;  // dropped file: its outputs; empty takes the loader's
    std::string path;                  // property: "options.colormap", relative to the item
};

// The selection as the app holds it.
struct WidgetSelection {
    std::string kind = "points";        // "points" or "range"
    std::vector<std::string> columns;   // the columns of `rows`, which the panel binds
    nlohmann::json rows = nlohmann::json::array();  // rows of cells, in `columns` order
    nlohmann::json range;               // the range in data units, such as {"x": [50, 150]}, or null
};

struct ContextBundleOptions {
    std::string preview;                    // read the spec of this stacked preview; empty: the open spec
    std::size_t max_text_bytes = 64 * 1024;
    std::uint64_t context_tokens = 0;       // the model's context length; 0: unknown, only max_text_bytes holds
    bool vision = true;                     // the model reads images; false leaves the render out
    bool embedded_context = true;           // parts as `resource` blocks; false: as `text` blocks
    int render_width = 640;
    int render_height = 400;
    std::size_t max_image_bytes = 1u << 20;
    std::optional<WidgetSelection> selection;
    CommandOrigin origin = CommandOrigin::ui;
};

struct ContextBundle {
    Error error;
    nlohmann::json blocks = nlohmann::json::array();  // ACP content blocks
    std::vector<std::string> scope;                   // command names, in the table's order
    nlohmann::json meta;                              // {"caliper": {"widget": {target, scope, preview}}}
    std::vector<std::string> cuts;                    // sample_rows, other_panels, render, sources, schemas, selection_rows
    std::size_t text_bytes = 0;                       // the text of every block
    bool has_image = false;
    bool over_budget = false;                         // the target alone does not fit
};

ContextBundle build_context_bundle(Session& session, const WidgetTarget& target,
                                   const ContextBundleOptions& options = {});

// The target as caliper-agent reads it in `_meta.caliper.widget.target`.
nlohmann::json widget_target_to_json(const WidgetTarget& target);

// The key of the target's entry in the scope rule: panel, selection,
// dropped_file, panel_property or source_property.
std::string widget_scope_key(const WidgetTarget& target);

// The scope rule (widget_scopes.json) as embedded at build time.
const std::string& widget_scope_rule_text();

// The rule applied to a command table: the listed reads that do not mutate
// and the listed edits that mutate, take a boolean dry_run and are not
// destructive, never a command or group the rule excludes, the vision-only
// commands only with `vision`, in the table's order. An unknown key is
// bundle.bad_target.
std::vector<std::string> widget_tool_scope(const std::vector<CommandDescriptor>& table, const std::string& key,
                                           bool vision, Error* error = nullptr);

}  // namespace caliper_builder
