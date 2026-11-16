#pragma once
// ============================================================================
// builder_pane_edits.h: the pure parts of the builder pane (builder_pane.h),
// functions of a spec, a resolver and the catalog that need no window and no
// session, so that the tests check them directly.
//
//   - The shapes of sources, written in the terse style of the panel headers
//     ("4,000 × 3", "f32 4×8×48×48"), and the shape after each transform of
//     a spec source as the spec validator computes it.
//   - The palette's primitives, the slots and options of each, and the
//     bindings a primitive takes for a dropped source or column (auto_bind),
//     with the primitive a dropped source makes when none is chosen.
//   - Ids: the id a catalog source or a new panel gets, and the renames of a
//     panel or a source with every reference to it (the format has no rename
//     command, so the pane sends a rename as spec.set_text of the renamed
//     spec's canonical text).
//   - The spec text: the line the read-only strip follows after a change.
//   - The layout tree: a node by its path, a panel placed at an edge of the
//     whole layout or next to another panel (also on its left, which
//     layout.place does not offer), and a split's ratio set and snapped.
//
// Nothing here throws.
// ============================================================================
#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace caliper_host {

// ---- shapes -----------------------------------------------------------------------------

struct PaneColumn {
    std::string name;
    std::string type;       // a DuckDB type name
    bool numeric = false;   // a number or a boolean, as the spec format counts it
};

// The data of a source, when known: columns for a tabular source, a dtype
// and dims for a tensor.
struct PaneShape {
    bool known = false;
    bool tabular = false;
    std::vector<PaneColumn> columns;
    std::optional<int64_t> rows;   // tabular, when the catalog knows it
    std::string dtype;             // tensor; empty when not known
    std::vector<int64_t> dims;     // tensor; -1 for a size that is not fixed
    int64_t elements() const;      // the product of dims, or -1
};

// True for the DuckDB types the spec format counts as numeric.
bool pane_type_numeric(const std::string& duckdb_type);
// "4,000 × 3" for a table with its rows known, "3 columns" otherwise;
// "f32 4×8×48×48" for a tensor; "" when nothing is known.
std::string pane_shape_text(const PaneShape& shape);
// "4×8×48×48", "scalar" for rank 0.
std::string pane_dims_text(const std::vector<int64_t>& dims);
// A whole number with thousands separated by commas.
std::string pane_count_text(int64_t n);

PaneShape pane_shape_from(const caliper_builder::spec::ResolvedSource& r);
PaneShape pane_shape_from(const caliper_builder::spec::SourceInfo& info);

// The shape of every source of `spec` after its transforms, by its id.
std::map<std::string, PaneShape> pane_source_shapes(const caliper_builder::spec::Spec& spec,
                                                    const caliper_builder::spec::SourceResolver& resolver);

// The shape of the source `source_id` before its transforms (index 0) and
// after each (index i + 1), as the validator computes them on the spec with
// the source's first i transforms. Empty when the spec has no such source.
std::vector<PaneShape> pane_transform_shapes(const caliper_builder::spec::Spec& spec, const std::string& source_id,
                                             const caliper_builder::spec::SourceResolver& resolver);

// ---- primitives, slots and options ------------------------------------------------------

// The seven primitives in the palette's order.
const std::vector<caliper_builder::spec::Primitive>& pane_primitives();
// The noun shown for a primitive: "Series", "Heatmap", ...
const char* pane_primitive_noun(caliper_builder::spec::Primitive p);

struct PaneSlot {
    const char* name;
    bool required;   // required as the format lists it (points: x and y unless positions)
    bool list;       // takes a list of bindings (series.y)
};
// A primitive's slots, in the inspector's order.
const std::vector<PaneSlot>& pane_slots(caliper_builder::spec::Primitive p);

enum class PaneOptionKind { text, longtext, range, flag, colormap, bins, columns };
struct PaneOption {
    const char* name;    // the key in [panel.options]
    PaneOptionKind kind;
    const char* label;   // the noun shown
};
// Exactly the options a primitive takes, in the inspector's order.
const std::vector<PaneOption>& pane_options(caliper_builder::spec::Primitive p);
// The four colormaps of the format, in their order.
const std::vector<std::string>& pane_colormaps();

// The bindings of a panel of `p` over a source (`column` empty) or one of
// its columns, by the pane's rules: a series takes the first column as x
// and the other numeric columns as y (a column alone: y, with the first
// column as x when it is another numeric one), a rank-1 or rank-2 tensor as
// y; a heatmap a rank-2 tensor; an image [H, W], [H, W, 3] or [H, W, 4]; points
// a tensor [n, 2] or [n, 3] as positions, or the first two numeric columns;
// a histogram a column, the first numeric column or a tensor; a table the
// whole source; a text panel a column or a one-element tensor. Nothing
// when the data does not fit by these rules; the validator has the last
// word on what does.
std::optional<std::map<std::string, caliper_builder::spec::SlotValue>> pane_auto_bind(
    caliper_builder::spec::Primitive p, const std::string& source_id, const PaneShape& shape,
    const std::string& column = {});

// Why a source does not fit a primitive by these rules, in the source's own
// terms, for the refusal of a panel that pane_auto_bind() binds nothing for:
// what the source is, what the primitive takes, and for a tensor with more
// dimensions than the primitive draws, that Properties slices or reduces it
// ("attn is f32 4×4×32×32, and a heatmap takes a tensor of two dimensions:
// slice or reduce it in Properties."). Empty for a text panel.
std::string pane_fit_hint(caliper_builder::spec::Primitive p, const std::string& source_id, const PaneShape& shape);

// The primitive a dropped source makes when none was chosen: a series for a
// column, a table with numeric columns after its first, a rank-1 tensor; a
// table for other tables; a heatmap for a rank-2 tensor, points for
// [n, 2] and [n, 3] with n of at least 16, an image for [H, W, 3] and
// [H, W, 4]; a histogram otherwise.
caliper_builder::spec::Primitive pane_default_primitive(const PaneShape& shape, const std::string& column = {});

// The arguments of the bind patch or of panel.create's bind for `bind`.
nlohmann::json pane_bind_json(const std::map<std::string, caliper_builder::spec::SlotValue>& bind);

// ---- ids --------------------------------------------------------------------------------

bool pane_valid_id(const std::string& id);
// A spec id from a catalog name ("run42_metrics", "attn.weights" ->
// "attn_weights"), unique among the spec's sources.
std::string pane_source_id(const std::string& catalog_name, const caliper_builder::spec::Spec& spec);
// The applet id of a new spec titled `title`: its letters and digits in
// lower case, every other run of characters one '-', "dashboard" when
// nothing is left, "d-" before an id that does not begin with a letter, and
// then "-2", "-3", ... while `taken` says the id is used.
std::string pane_new_spec_id(const std::string& title, const std::function<bool(const std::string&)>& taken);
// The id of a new panel of `p`: its primitive's name, then "-2", "-3", ...
std::string pane_panel_id(caliper_builder::spec::Primitive p, const caliper_builder::spec::Spec& spec);

// The spec with the panel `from` named `to`, in its table and the layout.
caliper_builder::spec::Spec pane_rename_panel(const caliper_builder::spec::Spec& spec, const std::string& from,
                                              const std::string& to);
// The spec with the source `from` named `to`, in its table, every binding
// that names it and every `from` that derives from it.
caliper_builder::spec::Spec pane_rename_source(const caliper_builder::spec::Spec& spec, const std::string& from,
                                               const std::string& to);

// ---- the spec text ----------------------------------------------------------------------

// The line of `after` that the read-only text strip brings into view after
// the open spec's text changed from `before`: the last line that differs,
// 0-based, found by the lines the two share at their start and at their
// end; where lines were only taken out, the line now in their place. -1
// when the texts are equal.
int pane_text_follow_line(const std::string& before, const std::string& after);

// ---- the layout tree --------------------------------------------------------------------

// The node at a path of child indices joined by '.' ("" is the root), as
// SpecLayoutPanel::path and SpecLayoutSplit::path name them; null when the
// path leads nowhere.
const caliper_builder::spec::LayoutNode* pane_layout_node(const caliper_builder::spec::LayoutNode& root,
                                                          const std::string& path);

// Where a panel goes: "left" and "beside" (right) make a horizontal split,
// "above" and "below" a vertical one, "tabbed" tabs.
// The layout with `panel` placed at an edge of the whole layout `root`
// (taken out first when it is already in it). With no layout, the panel
// alone.
caliper_builder::spec::LayoutNode pane_place_at_root(const std::optional<caliper_builder::spec::LayoutNode>& root,
                                                     const std::string& panel, const std::string& where);
// The layout with `panel` placed next to the panel `target` (taken out
// first when it is already in it); nothing when `target` is not placed or
// is `panel`. The new split takes no ratio, so the two share it equally.
std::optional<caliper_builder::spec::LayoutNode> pane_place_next_to(
    const std::optional<caliper_builder::spec::LayoutNode>& root, const std::string& panel, const std::string& target,
    const std::string& where);
// The layout with the split at `path` given `ratio` (written only when it
// is not 0.5, as the session writes it); the same layout when the path is
// not a split of two children.
caliper_builder::spec::LayoutNode pane_with_ratio(const caliper_builder::spec::LayoutNode& root,
                                                  const std::string& path, double ratio);
// A dragged ratio in hundredths, from 0.05 to 0.95.
double pane_snap_ratio(double ratio);

}  // namespace caliper_host
