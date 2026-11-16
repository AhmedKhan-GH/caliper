#pragma once
// ============================================================================
// builder_pane.h: the WYSIWYG builder pane (design/builder-pane.md;
// design/builder-overview.md, item 7 and the placement sketch). Someone
// builds a dashboard by hand: picks a source in the Sources palette, picks a
// primitive, gets a panel on the canvas, arranges the panels, edits their
// properties, and sees the spec text follow. Every edit is a command from the
// `builder` origin through the session's command layer, so it is journaled
// and can be undone, and one gesture is one undo step: the commands of a
// gesture share a group, and a value that is dragged or typed sends each
// change with a merge key under the group of its gesture.
//
// Build mode adds two columns and a strip beside the page's own canvas:
//   - Sources (left, upper): the catalog's sources grouped by kind, and the
//     open spec's own sources, each with its name and its shape or columns in
//     the terse style of the panel headers; a filter; a tabular source's
//     columns as children. Dragging a source or a column onto the canvas or
//     onto a panel's slot is the main gesture; select, then Add (⏎) is the
//     other way.
//   - Primitives (left, lower): the seven primitives as nouns. Adding one
//     makes a panel bound to the selected source by the pane's rules
//     (pane_auto_bind()); one the selection does not fit is faint, with the
//     validator's message in its tooltip.
//   - The canvas is the spec runtime's own drawing. A click focuses a panel;
//     its title strip drags it onto another panel's edge or centre
//     (layout.place, or layout.set for a place layout.place cannot name); a
//     divider drags its split's ratio (layout.set); ⌫ removes the focused
//     panel. While something is dragged, the rectangle where it would land
//     is outlined, computed with the layout of the spec it would make.
//   - Properties (right, upper): the focused panel's id, primitive, title,
//     bindings (one row per slot; each a choice among the spec's sources and
//     columns, the ones that do not fit faint with the validator's message
//     in a tooltip and refused before anything is sent) and exactly the
//     options its primitive takes; or the focused source's id, catalog id,
//     shape and transforms, with the shape after each step as the validator
//     computes it; or, with no focus, the spec's own title and description.
//   - The spec text (under the canvas, toggled): the open spec's canonical
//     TOML, read-only, or a draft with Edit that is applied by spec.set_text
//     whenever it validates, with the validator's diagnostics at their lines.
//
// Text is the source of truth, and the pane keeps no spec of its own: the
// canvas, the Properties and the strip read one OpenSpec of the session (the
// spec and its canonical text, read under the session's lock), read again
// after every command the pane sends and whenever the port's version moves
// (the journal, in the app). Every change from any of them is a command on
// the session. A draft is the one thing that can differ from the open text:
// it is applied the moment it validates, and while it does not, it carries
// its diagnostics and the pane refuses every other edit, so no edit is lost
// between the two; when the open spec changes under an applied draft, the
// draft becomes the new canonical text.
//
// Three parts, so that each is tested without a window:
//   - BuilderPane's state and gestures (this file), over a BuilderPort;
//   - the pure helpers (builder_pane_edits.h) and the regions and drop
//     targets as plain rectangles (builder_pane_layout.h);
//   - the ImGui side (builder_pane_draw.cpp): draw() on the frame thread in
//     the house style, which turns the mouse and the keys into the gestures.
//
// Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include "builder/builder_pane_edits.h"
#include "builder/builder_pane_layout.h"

#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

struct ImRect;

namespace caliper_host {

class SpecRuntime;
struct SpecLook;

// ---- what the pane needs of the app -----------------------------------------------------

// The session as the pane uses it on the frame thread.
class BuilderPort {
public:
    virtual ~BuilderPort() = default;
    // False while the session must be left alone (a long command runs on the
    // app's worker): the pane then sends nothing and computes nothing new.
    virtual bool ready() const { return true; }
    // Runs one invocation on the calling thread (the frame thread).
    virtual caliper_builder::CommandResult run(caliper_builder::CommandInvocation inv) = 0;
    // Submits a long command (data.import) through the session's queue, which
    // the app drains and runs on its worker.
    virtual std::future<caliper_builder::CommandResult> submit(caliper_builder::CommandInvocation inv) = 0;
    virtual std::optional<caliper_builder::OpenSpec> open_spec() const = 0;
    virtual caliper_builder::spec::SourceResolver resolver() const = 0;
    virtual uint64_t catalog_revision() const = 0;
    // Grows when the workspace may have changed from elsewhere; the pane reads
    // the open spec again when it moves. The default moves on every call.
    virtual uint64_t version() const;
    // The undo and redo stacks are not empty.
    virtual bool can_undo() const { return true; }
    virtual bool can_redo() const { return true; }

private:
    mutable uint64_t calls_ = 0;
};

// A BuilderPort over a builder Session: run() is Session::invoke(), submit()
// the session's queue; `busy` says when ready() is false
// (AppSession::worker_busy()) and `version` what version() is
// (AppSession::journaled()).
class SessionBuilderPort : public BuilderPort {
public:
    SessionBuilderPort(caliper_builder::Session& session, std::function<bool()> busy = {},
                       std::function<uint64_t()> version = {});
    bool ready() const override;
    caliper_builder::CommandResult run(caliper_builder::CommandInvocation inv) override;
    std::future<caliper_builder::CommandResult> submit(caliper_builder::CommandInvocation inv) override;
    std::optional<caliper_builder::OpenSpec> open_spec() const override;
    caliper_builder::spec::SourceResolver resolver() const override;
    uint64_t catalog_revision() const override;
    uint64_t version() const override;
    bool can_undo() const override;
    bool can_redo() const override;
    caliper_builder::Session& session() { return session_; }

private:
    caliper_builder::Session& session_;
    std::function<bool()> busy_;
    std::function<uint64_t()> version_;
};

// ---- what the pane shows ----------------------------------------------------------------

// One row of the Sources palette: a catalog source, or a source of the open
// spec (the group "Spec"), which carries its transforms.
struct PaneSource {
    enum class Group { spec, table, tensor, live, feed, metric };
    Group group = Group::table;
    std::string key;          // the catalog id ("table:run42_metrics"), or "spec:<id>"
    std::string name;         // the catalog name, or the spec source's id
    std::string catalog;      // the catalog id it reads; "" for a spec source over SQL or another source
    PaneShape shape;          // after its transforms, for a spec source
    std::string shape_text;   // pane_shape_text(shape)
    std::string spec_id;      // the spec source's id; for a catalog row, the spec source that reads it unchanged
    int transforms = 0;       // a spec source's transforms
};
const char* pane_group_label(PaneSource::Group g);   // "Spec", "Tables", "Tensors", "Live", "Feeds", "Metrics"

// What is dragged.
struct DragPayload {
    enum class Kind { none, source, column, primitive, panel };
    Kind kind = Kind::none;
    std::string source;   // source and column: the row's key
    std::string column;   // column
    caliper_builder::spec::Primitive primitive = caliper_builder::spec::Primitive::series;
    std::string panel;    // panel
    bool operator==(const DragPayload& o) const {
        return kind == o.kind && source == o.source && column == o.column && primitive == o.primitive &&
               panel == o.panel;
    }
};

// What a gesture would send, and whether the open spec takes it: every
// command is applied in order to the open spec with apply_spec_edit() and
// the session's resolver, as the session will apply it, so a gesture that
// does not fit is refused before anything is sent, with the validator's (or
// the command's) message.
struct PanePlan {
    std::vector<caliper_builder::CommandInvocation> commands;   // origin, group and merge key are set when sent
    caliper_builder::Error refusal;                              // empty when it fits
    std::optional<caliper_builder::spec::Spec> after;           // the open spec it would make
    std::string placed;                                          // the panel it makes or moves
    std::string focus_panel;                                     // the focus after it
    std::string focus_source;
    bool fits() const { return !refusal && !commands.empty(); }
};

// A slot's row in Properties: one per slot, and for series.y one per
// binding and one more to add a line.
struct PaneBindingRow {
    std::string slot;
    size_t index = 0;
    bool required = false;
    bool list = false;
    bool add = false;        // the row that adds a binding to a list slot
    std::string binding;     // "" when unbound
};

// A binding a slot can take, and whether it fits.
struct PaneBindingChoice {
    std::string binding;     // "metrics.loss"; "" unbinds
    std::string source;      // the spec source
    std::string column;      // "" for the whole source
    bool fits = false;
    std::string message;     // why it does not fit, from the validator
};

class BuilderPane {
public:
    struct Options {
        std::function<std::chrono::steady_clock::time_point()> now;
        // The prefix of the pane's groups; "builder-<hex>" by default. Groups
        // match [A-Za-z0-9._:/-]{1,128}.
        std::string group_prefix;
    };
    explicit BuilderPane(std::unique_ptr<BuilderPort> port, Options options = {});
    ~BuilderPane();
    BuilderPane(const BuilderPane&) = delete;
    BuilderPane& operator=(const BuilderPane&) = delete;

    BuilderPort& port() { return *port_; }

    // ---- the page's switches
    bool build_mode() const { return build_; }
    void set_build_mode(bool on);
    bool text_open() const { return text_open_; }
    void set_text_open(bool on) { text_open_ = on; }

    // ---- per frame: the import's answer, the catalog when its revision
    // moved, the open spec when the port's version moved.
    void update();

    // ---- the open spec as the pane last read it
    const std::optional<caliper_builder::OpenSpec>& open_spec() const { return open_; }

    // ---- Sources
    const std::vector<PaneSource>& sources() const { return rows_; }
    std::vector<const PaneSource*> visible_sources() const;   // the filter applied
    const PaneSource* find_source(const std::string& key) const;
    const std::string& filter() const { return filter_; }
    void set_filter(std::string text) { filter_ = std::move(text); }
    bool expanded(const std::string& key) const { return expanded_.count(key) != 0; }
    void toggle_expanded(const std::string& key);
    // The selection: one row, or one column of a tabular row. Selecting a
    // row focuses it in Properties.
    const std::string& selected() const { return selected_; }
    const std::string& selected_column() const { return selected_column_; }
    void select(const std::string& key, const std::string& column = {});
    // The rows the last file drop made.
    const std::vector<std::string>& fresh() const { return fresh_; }

    // ---- the focus: one panel of the canvas, or one source
    const std::string& focused_panel() const { return focus_panel_; }
    const std::string& focused_source() const { return focus_source_; }   // a row's key
    void focus_panel(const std::string& id);
    void focus_source(const std::string& key);
    void clear_focus();

    // ---- gestures: each sends one command or a group of commands from the
    // builder origin, one undo step, and returns false with error() when it
    // was refused (before anything was sent) or failed.
    // Sources' Add: a panel of the selection's default primitive.
    bool add_selected();
    // Primitives' Add: a panel of `p` bound to the selection.
    bool add_primitive(caliper_builder::spec::Primitive p);
    // What Add of `p` would do; its refusal is the faint primitive's tooltip.
    const PanePlan& primitive_plan(caliper_builder::spec::Primitive p);
    // A drop on the canvas.
    PanePlan plan_drop(const DragPayload& payload, const DropHit& hit);
    bool drop(const DragPayload& payload, const DropHit& hit);
    // The slots a dragged source offers in a panel's centre: the panel's
    // slots, in the inspector's order.
    std::vector<PaneSlot> drop_slots(const std::string& panel) const;

    // The focused panel's properties.
    bool set_panel_id(const std::string& id);       // a rename, sent as spec.set_text
    bool set_title(const std::string& title);       // "" removes it
    bool set_primitive(caliper_builder::spec::Primitive p);
    std::vector<PaneBindingRow> binding_rows() const;
    const std::vector<PaneBindingChoice>& binding_choices(const std::string& slot, size_t index);
    bool set_binding(const std::string& slot, size_t index, const std::string& binding);   // "" unbinds
    bool remove_binding(const std::string& slot, size_t index);
    bool set_option(const std::string& name, const nlohmann::json& value);   // null removes it
    // The focused source's properties. A catalog source not in the spec
    // yet takes an id before it is added; its first transform adds it.
    std::string source_id() const;                  // of the focused source, or the id it will get
    bool set_source_id(const std::string& id);      // before it is added: kept; after: a rename
    std::vector<PaneShape> transform_shapes();      // before and after each transform
    bool add_transform(caliper_builder::spec::TransformOp op);
    bool set_transform(size_t index, const nlohmann::json& transform);   // the whole transform, as in source.add
    bool remove_transform(size_t index);
    // The spec's own applet id, title and description (no focus), sent as
    // spec.set_text.
    bool set_spec_id(const std::string& id);
    bool set_spec_title(const std::string& title);
    bool set_spec_description(const std::string& description);
    // Removes the focused panel, or the focused spec source.
    bool remove_focused();
    // The ratio of the split at `path` (SpecLayoutSplit::path).
    bool set_ratio(const std::string& path, double ratio);

    // A value dragged or typed: the commands until end_gesture() share one
    // group and one merge key, so the gesture is one undo step.
    void begin_gesture(const std::string& what);
    void end_gesture();
    bool in_gesture() const { return !gesture_group_.empty(); }

    // ---- the spec text
    bool editing() const { return editing_; }
    void set_editing(bool on);
    // The draft while editing, else the open spec's canonical text.
    const std::string& text() const;
    // A new draft: validated, and applied by spec.set_text when it validates.
    bool set_draft(std::string text);
    // The draft's diagnostics, located at their lines; empty when it applies.
    const std::vector<caliper_builder::spec::Diagnostic>& diagnostics() const { return diagnostics_; }
    // The draft does not validate, so it is not applied.
    bool draft_pending() const { return editing_ && draft_pending_; }

    // ---- history and saving
    bool undo();
    bool redo();
    bool save();
    bool can_undo() const;
    bool can_redo() const;

    // ---- files dropped on the window: data.import of the paths, from the
    // ui origin (a file the person dropped), through the session's queue,
    // which the app runs on its worker with their loaders; the sources they
    // make are selected in the palette once it answers.
    bool drop_files(const std::vector<std::string>& paths,
                    caliper_builder::CommandOrigin origin = caliper_builder::CommandOrigin::ui);
    bool importing() const { return import_.has_value(); }
    // The files the last import refused, each with its path, the loader
    // store's code (loader.pickle_refused, loader.no_reader, ...) and its
    // reason; the Sources palette lists them until the next drop.
    struct Refused {
        std::string path, code, message;
    };
    const std::vector<Refused>& refused() const { return refused_; }

    // The last gesture's error: refused before anything was sent, or the
    // command's own; empty after one that succeeded.
    const caliper_builder::Error& error() const { return error_; }
    // The commands the pane has sent, newest last (at most 256), for tests.
    const std::vector<caliper_builder::CommandInvocation>& sent() const { return sent_; }

    // ---- the ImGui side (frame thread; builder_pane_draw.cpp)
    // The page bar's items: ⌘B Build, and in build mode ⌘T Text, ⌘Z Undo,
    // ⇧⌘Z Redo and ⌘S Save as key buttons, each doing what its key does
    // (Ctrl off macOS), then the last refusal's code. With `max_width` and
    // less room than page_bar_width(), only Build is drawn.
    void page_bar(float max_width = 0.0f);
    float page_bar_width() const;
    // The keys of build mode, before the page's own: ⌘B, ⌘T, ⌘Z, ⇧⌘Z, ⌘S,
    // ⌫ on the focused panel, Escape (a drag, a draft, the focus). True when
    // the pane took Escape.
    bool keys(bool escape);
    // The regions of `area` this frame: the layout rule's, which is
    // compute_builder_pane_layout() unless the page gives its own
    // (layout_spec_page(), which sizes the columns as the side chat does).
    BuilderPaneLayout layout(const ImRect& area, float text_scale) const;
    using LayoutRule = std::function<BuilderPaneLayout(const BuilderPaneLayoutInput& in, float text_scale)>;
    void set_layout_rule(LayoutRule rule) { layout_rule_ = std::move(rule); }
    // In build mode, inside a window over the work area: lays the regions
    // out, draws the runtime on the canvas, then the columns, the strip and
    // the canvas's overlays, and turns the mouse into gestures. Outside build
    // mode it draws the runtime over the whole area. `focus`, the page's
    // focused panel, is taken when the page changed it since the last call
    // and holds the pane's focused panel after it.
    void draw(const ImRect& area, SpecRuntime& runtime, const SpecLook& look, std::string* focus = nullptr);
    // The side chat in the right column's lower part while it is open in
    // build mode: `draw` is called with its rectangle inside the page's
    // window (ChatPane::draw_in()). Closed, or without `draw`, Properties
    // takes the whole column.
    void set_chat(bool open, std::function<void(const SpecRect&)> draw = {});
    // Named rectangles of the last draw ("source:table:run42_metrics",
    // "primitive:series", "prop:title", "panel:curves", "divider:", ...), for
    // drivers and tests; false when the name was not drawn.
    bool item_rect(const std::string& name, SpecRect* out) const;
    std::vector<std::string> item_names() const;
    // The drag in progress, and where it would land.
    const DragPayload& dragging() const { return drag_; }
    const DropHit& drop_hit() const { return hit_; }

    struct Draw;   // the ImGui side's state
    struct Rows;   // Properties' rows

private:
    friend struct Draw;
    void draw_body(const ImRect& area, SpecRuntime& runtime, const SpecLook& look);
    void draw_sources(const SpecRect& rect);
    void draw_primitives(const SpecRect& rect);
    void draw_properties(const SpecRect& rect, SpecRuntime& runtime);
    void draw_panel_properties(Rows& rows, const caliper_builder::spec::Panel& panel, SpecRuntime& runtime);
    void draw_source_properties(Rows& rows);
    void draw_spec_properties(Rows& rows);
    void draw_text_strip(const SpecRect& rect);
    bool run_plan(PanePlan plan, const std::string& merge = {});
    PanePlan simulate(std::vector<caliper_builder::CommandInvocation> commands) const;
    caliper_builder::CommandResult send(caliper_builder::CommandInvocation inv, const std::string& group,
                                        const std::string& merge);
    bool refuse(caliper_builder::Error e);
    bool blocked();
    void reread();
    void refresh_catalog();
    void rebuild_rows();
    void invalidate_caches();
    std::string next_group();
    std::string merge_key(const std::string& what) const;
    // The spec source a row binds as (existing or to add) and its shape.
    struct Resolved {
        std::string id;                                   // the spec source id
        std::optional<caliper_builder::CommandInvocation> add;   // source.add when it is not in the spec
        PaneShape shape;
        caliper_builder::Error error;
    };
    Resolved resolve_row(const std::string& key) const;
    PanePlan plan_new_panel(const std::string& key, const std::string& column,
                            std::optional<caliper_builder::spec::Primitive> primitive, const DropHit& hit) const;
    PanePlan plan_spec_text(const caliper_builder::spec::Spec& spec) const;
    const caliper_builder::spec::Panel* panel() const;          // the focused panel
    const caliper_builder::spec::Source* spec_source() const;   // the focused spec source
    std::string panel_title_text() const;

    std::unique_ptr<BuilderPort> port_;
    Options options_;
    bool build_ = false;
    bool text_open_ = false;
    bool chat_open_ = false;
    std::function<void(const SpecRect&)> chat_draw_;
    LayoutRule layout_rule_;

    std::optional<caliper_builder::OpenSpec> open_;
    uint64_t seen_version_ = ~uint64_t(0);
    uint64_t seen_catalog_ = ~uint64_t(0);
    nlohmann::json catalog_ = nlohmann::json::array();   // catalog.list's entries
    std::vector<PaneSource> rows_;
    std::string filter_;
    std::set<std::string> expanded_;
    std::string selected_, selected_column_;
    std::vector<std::string> fresh_;
    std::string focus_panel_, focus_source_;
    std::string synced_focus_;   // the page's focus as draw() last left it
    std::map<std::string, std::string> pending_ids_;   // a catalog row's id before it is added

    std::string gesture_group_, gesture_what_;
    int groups_ = 0;
    std::string prefix_;

    bool editing_ = false;
    std::string draft_;
    bool draft_pending_ = false;
    std::vector<caliper_builder::spec::Diagnostic> diagnostics_;

    std::optional<std::future<caliper_builder::CommandResult>> import_;
    std::vector<Refused> refused_;

    caliper_builder::Error error_;
    std::vector<caliper_builder::CommandInvocation> sent_;

    // Caches, dropped whenever the open spec, the catalog or the selection change.
    std::map<std::string, PanePlan> primitive_plans_;
    std::map<std::string, std::vector<PaneBindingChoice>> choices_;
    std::optional<std::vector<PaneShape>> shapes_;
    std::map<std::string, PaneShape> spec_shapes_;

    DragPayload drag_;
    DropHit hit_;
    std::unique_ptr<Draw> draw_;
};

}  // namespace caliper_host
