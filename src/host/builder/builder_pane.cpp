// builder_pane.cpp: the builder pane's state and gestures (builder_pane.h).
// The ImGui side is in builder_pane_draw.cpp.
#include "builder/builder_pane.h"

#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
#include <chrono>
#include <random>

namespace caliper_host {

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using cb::CommandInvocation;
using cb::CommandOrigin;
using cb::CommandResult;
using cb::Error;
using nlohmann::json;
using spec::Primitive;

// ---- ports ------------------------------------------------------------------------------

uint64_t BuilderPort::version() const { return ++calls_; }

SessionBuilderPort::SessionBuilderPort(cb::Session& session, std::function<bool()> busy,
                                       std::function<uint64_t()> version)
    : session_(session), busy_(std::move(busy)), version_(std::move(version)) {}

bool SessionBuilderPort::ready() const { return !busy_ || !busy_(); }

CommandResult SessionBuilderPort::run(CommandInvocation inv) { return session_.invoke(std::move(inv)); }

std::future<CommandResult> SessionBuilderPort::submit(CommandInvocation inv) {
    return session_.queue().submit(std::move(inv));
}

std::optional<cb::OpenSpec> SessionBuilderPort::open_spec() const { return session_.open_spec(); }

spec::SourceResolver SessionBuilderPort::resolver() const { return session_.source_resolver(); }

uint64_t SessionBuilderPort::catalog_revision() const { return session_.catalog().revision(); }

uint64_t SessionBuilderPort::version() const { return version_ ? version_() : BuilderPort::version(); }

bool SessionBuilderPort::can_undo() const { return session_.registry().history_state(1).can_undo; }

bool SessionBuilderPort::can_redo() const { return session_.registry().history_state(1).can_redo; }

// ---- rows -------------------------------------------------------------------------------

const char* pane_group_label(PaneSource::Group g) {
    switch (g) {
        case PaneSource::Group::spec: return "Spec";
        case PaneSource::Group::table: return "Tables";
        case PaneSource::Group::tensor: return "Tensors";
        case PaneSource::Group::live: return "Live";
        case PaneSource::Group::feed: return "Feeds";
        case PaneSource::Group::metric: return "Metrics";
    }
    return "";
}

namespace {

CommandInvocation invocation(const std::string& name, json args) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = CommandOrigin::builder;
    return inv;
}

PaneSource::Group group_of_kind(const std::string& kind) {
    if (kind == "tensor") return PaneSource::Group::tensor;
    if (kind == "live") return PaneSource::Group::live;
    if (kind == "feed") return PaneSource::Group::feed;
    if (kind == "metric") return PaneSource::Group::metric;
    return PaneSource::Group::table;
}

PaneShape shape_of_entry(const json& e) {
    PaneShape s;
    if (e.contains("columns") && e["columns"].is_array()) {
        s.known = true;
        s.tabular = true;
        for (const auto& c : e["columns"]) {
            const std::string name = c.value("name", ""), type = c.value("type", "");
            s.columns.push_back({name, type, pane_type_numeric(type)});
        }
        if (e.contains("rows") && e["rows"].is_number_integer()) s.rows = e["rows"].get<int64_t>();
    } else if (e.contains("shape") && e["shape"].is_array()) {
        s.known = true;
        for (const auto& d : e["shape"]) s.dims.push_back(d.is_number_integer() ? d.get<int64_t>() : -1);
        if (e.contains("dtype") && e["dtype"].is_string()) s.dtype = e["dtype"].get<std::string>();
    }
    return s;
}

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Whether a whole spec source is bound by a panel of `s` (directly or
// through `from`).
bool slot_takes_list(Primitive p, const std::string& slot) {
    for (const auto& s : pane_slots(p))
        if (slot == s.name) return s.list;
    return false;
}

json transform_json(const spec::Transform& t) {
    json j = {{"op", spec::to_string(t.op)}};
    if (!t.axis.empty()) {
        if (t.axis_list || t.axis.size() > 1) j["axis"] = t.axis;
        else j["axis"] = t.axis.front();
    }
    if (t.index) j["index"] = *t.index;
    if (t.start) j["start"] = *t.start;
    if (t.stop) j["stop"] = *t.stop;
    if (t.step) j["step"] = *t.step;
    if (t.fn) j["fn"] = *t.fn;
    if (t.keepdims) j["keepdims"] = *t.keepdims;
    if (t.mode) j["mode"] = *t.mode;
    return j;
}

}  // namespace

// ---- the pane ---------------------------------------------------------------------------

// The constructor and the destructor are in builder_pane_draw.cpp, where
// the ImGui side's state is a complete type.

void BuilderPane::set_build_mode(bool on) {
    build_ = on;
    if (!on) {
        drag_ = {};
        hit_ = {};
        end_gesture();
    }
}

std::string BuilderPane::next_group() { return prefix_ + "-" + std::to_string(++groups_); }

std::string BuilderPane::merge_key(const std::string& what) const { return prefix_ + "." + what; }

void BuilderPane::begin_gesture(const std::string& what) {
    if (gesture_what_ == what && !gesture_group_.empty()) return;
    gesture_group_ = next_group();
    gesture_what_ = what;
}

void BuilderPane::end_gesture() {
    gesture_group_.clear();
    gesture_what_.clear();
}

void BuilderPane::invalidate_caches() {
    primitive_plans_.clear();
    choices_.clear();
    shapes_.reset();
}

void BuilderPane::update() {
    if (!port_->ready()) return;
    // The import's answer: the sources it made are selected.
    if (import_ && import_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const CommandResult r = import_->get();
        import_.reset();
        if (!r.ok) {
            error_ = r.error;
        } else {
            fresh_.clear();
            for (const auto& l : r.result.value("loaders", json::array()))
                for (const auto& o : l.value("outputs", json::array()))
                    if (o.is_string()) fresh_.push_back(o.get<std::string>());
            if (!r.result.value("refused", json::array()).empty()) {
                const json& first = r.result["refused"][0];
                error_ = Error(first.value("code", "loader.refused"), first.value("message", ""), first.value("path", ""));
            } else {
                error_ = {};
            }
            refresh_catalog();
            if (!fresh_.empty()) select(fresh_.front());
        }
    }
    const uint64_t rev = port_->catalog_revision();
    if (rev != seen_catalog_) {
        seen_catalog_ = rev;
        refresh_catalog();
    }
    const uint64_t v = port_->version();
    if (v != seen_version_) {
        seen_version_ = v;
        reread();
    }
}

void BuilderPane::refresh_catalog() {
    CommandInvocation inv = invocation("catalog.list", {{"limit", 1000}});
    const CommandResult r = port_->run(std::move(inv));
    catalog_ = r.ok ? r.result.value("entries", json::array()) : json::array();
    rebuild_rows();
    invalidate_caches();
}

void BuilderPane::reread() {
    std::optional<cb::OpenSpec> next = port_->open_spec();
    const bool changed = !next != !open_ || (next && open_ && next->text != open_->text);
    const std::string before = open_ ? open_->text : std::string();
    open_ = std::move(next);
    if (!changed) return;
    invalidate_caches();
    spec_shapes_ = open_ ? pane_source_shapes(open_->spec, port_->resolver()) : std::map<std::string, PaneShape>{};
    rebuild_rows();
    // The focus follows what is still there.
    if (!focus_panel_.empty() && (!open_ || !open_->spec.find_panel(focus_panel_))) focus_panel_.clear();
    if (focus_source_.rfind("spec:", 0) == 0 && (!open_ || !open_->spec.find_source(focus_source_.substr(5))))
        focus_source_.clear();
    if (selected_.rfind("spec:", 0) == 0 && !find_source(selected_)) selected_.clear(), selected_column_.clear();
    // An applied draft follows the open spec; a pending one is kept.
    if (editing_ && !draft_pending_) {
        draft_ = open_ ? open_->text : std::string();
        diagnostics_.clear();
    }
    (void)before;
}

void BuilderPane::rebuild_rows() {
    rows_.clear();
    const spec::Spec* s = open_ ? &open_->spec : nullptr;
    if (s) {
        for (const auto& src : s->sources) {
            PaneSource r;
            r.group = PaneSource::Group::spec;
            r.key = "spec:" + src.id;
            r.name = src.id;
            r.catalog = src.catalog.value_or("");
            r.spec_id = src.id;
            r.transforms = (int)src.transforms.size();
            const auto it = spec_shapes_.find(src.id);
            if (it != spec_shapes_.end()) r.shape = it->second;
            // A table's rows, from its catalog entry.
            if (r.shape.tabular && src.catalog && src.transforms.empty() && !src.from)
                for (const auto& e : catalog_)
                    if (e.value("id", "") == *src.catalog) r.shape.rows = shape_of_entry(e).rows;
            r.shape_text = pane_shape_text(r.shape);
            rows_.push_back(std::move(r));
        }
    }
    std::vector<PaneSource> cat;
    for (const auto& e : catalog_) {
        PaneSource r;
        r.key = e.value("id", "");
        r.group = group_of_kind(e.value("kind", "table"));
        r.name = e.value("name", r.key);
        r.catalog = r.key;
        r.shape = shape_of_entry(e);
        r.shape_text = pane_shape_text(r.shape);
        if (s)
            for (const auto& src : s->sources)
                if (src.catalog && *src.catalog == r.key && src.transforms.empty() && !src.from) {
                    r.spec_id = src.id;
                    break;
                }
        cat.push_back(std::move(r));
    }
    std::stable_sort(cat.begin(), cat.end(), [](const PaneSource& a, const PaneSource& b) {
        return a.group != b.group ? a.group < b.group : a.name < b.name;
    });
    for (auto& r : cat) rows_.push_back(std::move(r));
}

std::vector<const PaneSource*> BuilderPane::visible_sources() const {
    std::vector<const PaneSource*> out;
    const std::string f = lower(filter_);
    for (const auto& r : rows_) {
        if (!f.empty()) {
            bool match = lower(r.name).find(f) != std::string::npos || lower(r.key).find(f) != std::string::npos;
            for (const auto& c : r.shape.columns)
                if (lower(c.name).find(f) != std::string::npos) match = true;
            if (!match) continue;
        }
        out.push_back(&r);
    }
    return out;
}

const PaneSource* BuilderPane::find_source(const std::string& key) const {
    for (const auto& r : rows_)
        if (r.key == key) return &r;
    return nullptr;
}

void BuilderPane::toggle_expanded(const std::string& key) {
    if (!expanded_.erase(key)) expanded_.insert(key);
}

void BuilderPane::select(const std::string& key, const std::string& column) {
    if (key != selected_ || column != selected_column_) primitive_plans_.clear();
    selected_ = key;
    selected_column_ = column;
    if (!key.empty()) focus_source(key);
}

void BuilderPane::focus_panel(const std::string& id) {
    if (id != focus_panel_) choices_.clear();
    focus_panel_ = id;
    if (!id.empty()) focus_source_.clear();
    shapes_.reset();
}

void BuilderPane::focus_source(const std::string& key) {
    focus_source_ = key;
    if (!key.empty()) focus_panel_.clear();
    shapes_.reset();
}

void BuilderPane::clear_focus() {
    focus_panel_.clear();
    focus_source_.clear();
    shapes_.reset();
}

const spec::Panel* BuilderPane::panel() const {
    if (!open_ || focus_panel_.empty()) return nullptr;
    return open_->spec.find_panel(focus_panel_);
}

const spec::Source* BuilderPane::spec_source() const {
    if (!open_ || focus_source_.empty()) return nullptr;
    if (focus_source_.rfind("spec:", 0) == 0) return open_->spec.find_source(focus_source_.substr(5));
    const PaneSource* r = find_source(focus_source_);
    if (r && !r->spec_id.empty()) return open_->spec.find_source(r->spec_id);
    return nullptr;
}

// ---- sending ----------------------------------------------------------------------------

bool BuilderPane::refuse(Error e) {
    error_ = std::move(e);
    return false;
}

// True, with error(), when no gesture may run now.
bool BuilderPane::blocked() {
    if (!port_->ready()) {
        refuse(Error("builder.busy", "A loader is running; the change waits until it is done.", ""));
        return true;
    }
    if (!open_) {
        update();
        if (!open_) {
            refuse(Error("spec.no_open_spec", "No spec is open.", ""));
            return true;
        }
    }
    if (draft_pending()) {
        refuse(Error("builder.draft", "The spec text has an edit that does not validate.", "text"));
        return true;
    }
    return false;
}

CommandResult BuilderPane::send(CommandInvocation inv, const std::string& group, const std::string& merge) {
    inv.origin = CommandOrigin::builder;
    inv.group = group;
    inv.merge_key = merge;
    sent_.push_back(inv);
    if (sent_.size() > 256) sent_.erase(sent_.begin());
    return port_->run(std::move(inv));
}

PanePlan BuilderPane::simulate(std::vector<CommandInvocation> commands) const {
    PanePlan plan;
    plan.commands = std::move(commands);
    if (!open_) {
        plan.refusal = Error("spec.no_open_spec", "No spec is open.", "");
        return plan;
    }
    spec::Spec s = open_->spec;
    const spec::SourceResolver resolver = port_->resolver();
    for (const auto& c : plan.commands) {
        if (!cb::is_spec_edit_command(c.name)) continue;
        cb::SpecEdit e = cb::apply_spec_edit(s, c.name, c.args, resolver);
        if (e.error) {
            plan.refusal = e.error;
            return plan;
        }
        s = std::move(e.spec);
    }
    plan.after = std::move(s);
    return plan;
}

bool BuilderPane::run_plan(PanePlan plan, const std::string& merge) {
    if (blocked()) return false;
    if (plan.refusal) return refuse(plan.refusal);
    if (plan.commands.empty()) return refuse(Error("builder.nothing", "The gesture changes nothing.", ""));
    if (plan.after && open_ && *plan.after == open_->spec && !in_gesture()) {
        error_ = {};
        return true;   // nothing to change
    }
    const std::string group = in_gesture() ? gesture_group_ : next_group();
    const std::string key = in_gesture() && !merge.empty() ? merge_key(gesture_what_ + "." + merge) : std::string();
    size_t ran = 0;
    for (auto& c : plan.commands) {
        const CommandResult r = send(c, group, plan.commands.size() == 1 ? key : std::string());
        if (!r.ok) {
            // Take back what ran of the group, so a gesture is all or nothing.
            if (ran > 0) send(invocation("history.undo", json::object()), std::string(), std::string());
            error_ = r.error;
            reread();
            return false;
        }
        ++ran;
    }
    error_ = {};
    reread();
    // A catalog row that the gesture added to the spec is selected as the
    // spec's source from now on.
    for (const auto& c : plan.commands) {
        if (c.name != "source.add" || !c.args.contains("catalog")) continue;
        const std::string catalog = c.args["catalog"].get<std::string>(), id = c.args["id"].get<std::string>();
        pending_ids_.erase(catalog);
        if (selected_ == catalog) selected_ = "spec:" + id;
        if (focus_source_ == catalog) focus_source_ = "spec:" + id;
    }
    if (!plan.focus_panel.empty()) focus_panel(plan.focus_panel);
    if (!plan.focus_source.empty()) focus_source(plan.focus_source);
    return true;
}

// ---- making panels ----------------------------------------------------------------------

BuilderPane::Resolved BuilderPane::resolve_row(const std::string& key) const {
    Resolved out;
    const PaneSource* row = find_source(key);
    if (!row) {
        out.error = Error("builder.no_source", "Select a source first.", "");
        return out;
    }
    if (!row->spec_id.empty()) {
        out.id = row->spec_id;
        const auto it = spec_shapes_.find(out.id);
        out.shape = row->group == PaneSource::Group::spec ? row->shape
                    : it != spec_shapes_.end()               ? it->second
                                                              : row->shape;
        if (row->shape.rows) out.shape.rows = row->shape.rows;
        return out;
    }
    // A catalog source that the spec does not read yet: added in the same gesture.
    const auto pending = pending_ids_.find(key);
    out.id = pending != pending_ids_.end() ? pending->second
                                            : pane_source_id(row->name, open_ ? open_->spec : spec::Spec{});
    out.add = invocation("source.add", {{"id", out.id}, {"catalog", row->catalog}});
    out.shape = row->shape;
    return out;
}

PanePlan BuilderPane::plan_new_panel(const std::string& key, const std::string& column,
                                     std::optional<Primitive> primitive, const DropHit& hit) const {
    PanePlan plan;
    if (!open_) {
        plan.refusal = Error("spec.no_open_spec", "No spec is open.", "");
        return plan;
    }
    const spec::Spec& s = open_->spec;
    std::vector<CommandInvocation> cmds;
    json create;
    Primitive p = primitive.value_or(Primitive::series);
    if (key.empty()) {
        // No source: only a text panel stands alone, with its text.
        create = {{"primitive", spec::to_string(p)}};
        if (p == Primitive::text) create["options"] = {{"text", "Text"}};
    } else {
        Resolved r = resolve_row(key);
        if (r.error) {
            plan.refusal = r.error;
            return plan;
        }
        if (r.add) cmds.push_back(*r.add);
        if (!primitive) p = pane_default_primitive(r.shape, column);
        create = {{"primitive", spec::to_string(p)}};
        const auto bind = pane_auto_bind(p, r.id, r.shape, column);
        if (bind) create["bind"] = pane_bind_json(*bind);
        else if (p == Primitive::text) create["options"] = {{"text", "Text"}};
    }
    // The new panel's id, unique also against what the gesture adds.
    const std::string id = pane_panel_id(p, s);
    create["id"] = id;
    // Where it goes.
    const std::optional<spec::LayoutNode> root = cb::effective_layout(s);
    std::optional<spec::LayoutNode> after_layout;
    if (hit.zone == DropZone::edge || hit.zone == DropZone::centre) {
        if (hit.where == "left") {
            // layout.place has no left: the panel is made, then placed.
            after_layout = pane_place_next_to(root ? std::optional<spec::LayoutNode>(*root) : std::nullopt, id,
                                              hit.panel, "left");
            if (!after_layout) {
                plan.refusal = Error("panel.not_found", "The layout has no panel " + hit.panel + ".", "");
                return plan;
            }
        } else {
            create["place"] = {{"target", hit.panel}, {"where", hit.where}};
        }
    } else if (hit.zone == DropZone::root && hit.where != "beside") {
        after_layout = pane_place_at_root(root, id, hit.where);
    } else if (hit.zone == DropZone::none && !focus_panel_.empty() && s.find_panel(focus_panel_)) {
        // An Add without a drop: beside the focused panel.
        create["place"] = {{"target", focus_panel_}, {"where", "beside"}};
    }
    cmds.push_back(invocation("panel.create", create));
    plan = simulate(cmds);
    // A place that panel.create cannot name: the panel is made, then the
    // layout is set, flattened against the spec with the new panel.
    if (after_layout && !plan.refusal && plan.after) {
        cmds.push_back(invocation("layout.set", {{"layout", cb::layout_to_flat(*after_layout, *plan.after)}}));
        plan = simulate(cmds);
    }
    plan.placed = id;
    plan.focus_panel = id;
    return plan;
}

const PanePlan& BuilderPane::primitive_plan(Primitive p) {
    const std::string k = std::string(spec::to_string(p)) + "|" + selected_ + "|" + selected_column_ + "|" +
                          focus_panel_;
    auto it = primitive_plans_.find(k);
    if (it == primitive_plans_.end())
        it = primitive_plans_.emplace(k, plan_new_panel(selected_, selected_column_, p, DropHit{})).first;
    return it->second;
}

bool BuilderPane::add_primitive(Primitive p) {
    if (blocked()) return false;
    return run_plan(plan_new_panel(selected_, selected_column_, p, DropHit{}));
}

bool BuilderPane::add_selected() {
    if (blocked()) return false;
    if (selected_.empty()) return refuse(Error("builder.no_source", "Select a source first.", ""));
    return run_plan(plan_new_panel(selected_, selected_column_, std::nullopt, DropHit{}));
}

std::vector<PaneSlot> BuilderPane::drop_slots(const std::string& panel_id) const {
    if (!open_) return {};
    const spec::Panel* p = open_->spec.find_panel(panel_id);
    if (!p) return {};
    return pane_slots(p->primitive);
}

PanePlan BuilderPane::plan_drop(const DragPayload& payload, const DropHit& hit) {
    PanePlan plan;
    if (!open_) {
        plan.refusal = Error("spec.no_open_spec", "No spec is open.", "");
        return plan;
    }
    const spec::Spec& s = open_->spec;
    if (hit.zone == DropZone::none) {
        plan.refusal = Error("builder.no_target", "Nothing to drop on there.", "");
        return plan;
    }
    switch (payload.kind) {
        case DragPayload::Kind::none:
            plan.refusal = Error("builder.nothing", "Nothing is dragged.", "");
            return plan;
        case DragPayload::Kind::source:
        case DragPayload::Kind::column: {
            if (hit.zone != DropZone::slot) return plan_new_panel(payload.source, payload.column, std::nullopt, hit);
            const spec::Panel* p = s.find_panel(hit.panel);
            const std::vector<PaneSlot> slots = drop_slots(hit.panel);
            if (!p || hit.slot < 0 || hit.slot >= (int)slots.size()) {
                plan.refusal = Error("panel.not_found", "The spec has no panel " + hit.panel + ".", "");
                return plan;
            }
            Resolved r = resolve_row(payload.source);
            if (r.error) {
                plan.refusal = r.error;
                return plan;
            }
            const PaneSlot& slot = slots[(size_t)hit.slot];
            const std::string binding = payload.column.empty() ? r.id : r.id + "." + payload.column;
            json value = binding;
            if (slot.list) {
                std::vector<std::string> list;
                const auto it = p->bind.find(slot.name);
                if (it != p->bind.end()) list = it->second.bindings;
                if (std::find(list.begin(), list.end(), binding) == list.end()) list.push_back(binding);
                value = list.size() == 1 && (it == p->bind.end() || !it->second.list) ? json(list.front()) : json(list);
            }
            std::vector<CommandInvocation> cmds;
            if (r.add) cmds.push_back(*r.add);
            cmds.push_back(invocation("panel.patch", {{"id", p->id}, {"patch", {{"bind", {{slot.name, value}}}}}}));
            plan = simulate(cmds);
            plan.placed = p->id;
            plan.focus_panel = p->id;
            return plan;
        }
        case DragPayload::Kind::primitive:
            return plan_new_panel(selected_, selected_column_, payload.primitive, hit);
        case DragPayload::Kind::panel: {
            const std::string& moved = payload.panel;
            std::vector<CommandInvocation> cmds;
            const std::optional<spec::LayoutNode> root = cb::effective_layout(s);
            if ((hit.zone == DropZone::edge || hit.zone == DropZone::centre || hit.zone == DropZone::slot) &&
                hit.panel == moved) {
                plan.refusal = Error("layout.same_panel", "A panel cannot be placed next to itself.", "");
                return plan;
            }
            if ((hit.zone == DropZone::edge || hit.zone == DropZone::centre) && hit.where != "left") {
                cmds.push_back(invocation("layout.place", {{"panel", moved}, {"target", hit.panel}, {"where", hit.where}}));
            } else {
                std::optional<spec::LayoutNode> next;
                if (hit.zone == DropZone::root) next = pane_place_at_root(root, moved, hit.where);
                else if (hit.zone == DropZone::edge) next = pane_place_next_to(root, moved, hit.panel, "left");
                if (!next) {
                    plan.refusal = Error("layout.invalid", "The panel cannot go there.", "");
                    return plan;
                }
                cmds.push_back(invocation("layout.set", {{"layout", cb::layout_to_flat(*next, s)}}));
            }
            plan = simulate(cmds);
            plan.placed = moved;
            plan.focus_panel = moved;
            return plan;
        }
    }
    return plan;
}

bool BuilderPane::drop(const DragPayload& payload, const DropHit& hit) {
    if (blocked()) return false;
    PanePlan plan = plan_drop(payload, hit);
    if (payload.kind == DragPayload::Kind::panel && plan.after && open_ && *plan.after == open_->spec) {
        error_ = {};
        return true;   // dropped where it was
    }
    return run_plan(std::move(plan));
}

// ---- the focused panel ------------------------------------------------------------------

PanePlan BuilderPane::plan_spec_text(const spec::Spec& s) const {
    return simulate({invocation("spec.set_text", {{"text", spec::to_toml(s)}})});
}

bool BuilderPane::set_panel_id(const std::string& id) {
    if (blocked()) return false;
    const spec::Panel* p = panel();
    if (!p) return refuse(Error("panel.not_found", "No panel is focused.", ""));
    if (id == p->id) return true;
    PanePlan plan = plan_spec_text(pane_rename_panel(open_->spec, p->id, id));
    plan.focus_panel = id;
    return run_plan(std::move(plan));
}

bool BuilderPane::set_title(const std::string& title) {
    if (blocked()) return false;
    const spec::Panel* p = panel();
    if (!p) return refuse(Error("panel.not_found", "No panel is focused.", ""));
    const json value = title.empty() ? json(nullptr) : json(title);
    return run_plan(simulate({invocation("panel.patch", {{"id", p->id}, {"patch", {{"title", value}}}})}), "title");
}

bool BuilderPane::set_primitive(Primitive prim) {
    if (blocked()) return false;
    const spec::Panel* p = panel();
    if (!p) return refuse(Error("panel.not_found", "No panel is focused.", ""));
    if (p->primitive == prim) return true;
    json patch = {{"primitive", spec::to_string(prim)}};
    // The slots and options the new primitive does not take are dropped;
    // with its required slots unbound, the panel's first source is bound by
    // the pane's rules.
    json bind = json::object();
    std::set<std::string> keep;
    for (const auto& s : pane_slots(prim)) keep.insert(s.name);
    std::string first_source;
    for (const auto& [slot, v] : p->bind) {
        if (first_source.empty() && !v.bindings.empty()) {
            const std::string& b = v.bindings.front();
            first_source = b.substr(0, b.find('.'));
        }
        if (!keep.count(slot)) bind[slot] = nullptr;
    }
    {
        PanePlan trial = simulate({invocation("panel.patch", {{"id", p->id}, {"patch", {{"primitive", spec::to_string(prim)},
                                                                                    {"bind", bind}}}})});
        if (trial.refusal && !first_source.empty()) {
            const auto it = spec_shapes_.find(first_source);
            if (it != spec_shapes_.end()) {
                const auto auto_bind = pane_auto_bind(prim, first_source, it->second);
                if (auto_bind)
                    for (const auto& [slot, v] : *auto_bind) bind[slot] = v.list || v.bindings.size() != 1
                                                                        ? json(v.bindings)
                                                                        : json(v.bindings.front());
            }
        }
    }
    json options = json::object();
    std::set<std::string> takes;
    for (const auto& o : pane_options(prim)) takes.insert(o.name);
    for (const auto& name : p->options.set_names())
        if (!takes.count(name)) options[name] = nullptr;
    if (!bind.empty()) patch["bind"] = bind;
    if (!options.empty()) patch["options"] = options;
    return run_plan(simulate({invocation("panel.patch", {{"id", p->id}, {"patch", patch}})}));
}

std::vector<PaneBindingRow> BuilderPane::binding_rows() const {
    std::vector<PaneBindingRow> rows;
    const spec::Panel* p = panel();
    if (!p) return rows;
    for (const auto& slot : pane_slots(p->primitive)) {
        const auto it = p->bind.find(slot.name);
        const std::vector<std::string> bound = it == p->bind.end() ? std::vector<std::string>{} : it->second.bindings;
        if (slot.list) {
            for (size_t i = 0; i < bound.size(); ++i) rows.push_back({slot.name, i, slot.required, true, false, bound[i]});
            if (bound.size() < 16) rows.push_back({slot.name, bound.size(), slot.required && bound.empty(), true, true, ""});
        } else {
            rows.push_back({slot.name, 0, slot.required, false, false, bound.empty() ? std::string() : bound.front()});
        }
    }
    return rows;
}

const std::vector<PaneBindingChoice>& BuilderPane::binding_choices(const std::string& slot, size_t index) {
    const std::string k = focus_panel_ + "|" + slot + "|" + std::to_string(index);
    auto it = choices_.find(k);
    if (it != choices_.end()) return it->second;
    std::vector<PaneBindingChoice> out;
    const spec::Panel* p = panel();
    if (p && open_) {
        const bool list = slot_takes_list(p->primitive, slot);
        const auto bound_it = p->bind.find(slot);
        std::vector<std::string> bound = bound_it == p->bind.end() ? std::vector<std::string>{} : bound_it->second.bindings;
        const bool was_list = bound_it != p->bind.end() && bound_it->second.list;
        auto value_for = [&](const std::string& binding) -> json {
            if (!list) return binding.empty() ? json(nullptr) : json(binding);
            std::vector<std::string> next = bound;
            if (binding.empty()) {
                if (index < next.size()) next.erase(next.begin() + (std::ptrdiff_t)index);
            } else if (index < next.size()) {
                next[index] = binding;
            } else {
                next.push_back(binding);
            }
            if (next.empty()) return json(nullptr);
            if (next.size() == 1 && !was_list) return json(next.front());
            return json(next);
        };
        auto check = [&](PaneBindingChoice c) {
            PanePlan plan = simulate({invocation("panel.patch", {{"id", p->id}, {"patch", {{"bind", {{slot, value_for(c.binding)}}}}}})});
            c.fits = !plan.refusal;
            c.message = plan.refusal.message;
            out.push_back(std::move(c));
        };
        for (const auto& src : open_->spec.sources) {
            const auto sh = spec_shapes_.find(src.id);
            const bool tabular = sh != spec_shapes_.end() && sh->second.tabular;
            check({src.id, src.id, "", false, ""});
            if (tabular)
                for (const auto& c : sh->second.columns) check({src.id + "." + c.name, src.id, c.name, false, ""});
        }
        // Unbinding, for a row that holds a binding.
        if (index < bound.size()) check({"", "", "", false, ""});
    }
    return choices_.emplace(k, std::move(out)).first->second;
}

bool BuilderPane::set_binding(const std::string& slot, size_t index, const std::string& binding) {
    if (blocked()) return false;
    const spec::Panel* p = panel();
    if (!p) return refuse(Error("panel.not_found", "No panel is focused.", ""));
    for (const auto& c : binding_choices(slot, index)) {
        if (c.binding != binding) continue;
        if (!c.fits) return refuse(Error("spec.slot_mismatch", c.message, "bind." + slot));
        break;
    }
    const bool list = slot_takes_list(p->primitive, slot);
    const auto it = p->bind.find(slot);
    std::vector<std::string> bound = it == p->bind.end() ? std::vector<std::string>{} : it->second.bindings;
    json value;
    if (!list) {
        value = binding.empty() ? json(nullptr) : json(binding);
    } else {
        if (binding.empty()) {
            if (index < bound.size()) bound.erase(bound.begin() + (std::ptrdiff_t)index);
        } else if (index < bound.size()) {
            bound[index] = binding;
        } else {
            bound.push_back(binding);
        }
        const bool was_list = it != p->bind.end() && it->second.list;
        value = bound.empty() ? json(nullptr) : bound.size() == 1 && !was_list ? json(bound.front()) : json(bound);
    }
    return run_plan(simulate({invocation("panel.patch", {{"id", p->id}, {"patch", {{"bind", {{slot, value}}}}}})}));
}

bool BuilderPane::remove_binding(const std::string& slot, size_t index) { return set_binding(slot, index, ""); }

bool BuilderPane::set_option(const std::string& name, const json& value) {
    if (blocked()) return false;
    const spec::Panel* p = panel();
    if (!p) return refuse(Error("panel.not_found", "No panel is focused.", ""));
    return run_plan(simulate({invocation("panel.patch", {{"id", p->id}, {"patch", {{"options", {{name, value}}}}}})}),
                    "option." + name);
}

// ---- the focused source -----------------------------------------------------------------

std::string BuilderPane::source_id() const {
    if (const spec::Source* s = spec_source()) return s->id;
    const PaneSource* r = find_source(focus_source_);
    if (!r) return {};
    const auto it = pending_ids_.find(r->key);
    return it != pending_ids_.end() ? it->second : pane_source_id(r->name, open_ ? open_->spec : spec::Spec{});
}

bool BuilderPane::set_source_id(const std::string& id) {
    if (const spec::Source* s = spec_source()) {
        if (blocked()) return false;
        if (s->id == id) return true;
        PanePlan plan = plan_spec_text(pane_rename_source(open_->spec, s->id, id));
        plan.focus_source = "spec:" + id;
        const std::string old = "spec:" + s->id;
        const bool ok = run_plan(std::move(plan));
        if (ok && selected_ == old) selected_ = "spec:" + id;
        return ok;
    }
    const PaneSource* r = find_source(focus_source_);
    if (!r) return refuse(Error("source.not_found", "No source is focused.", ""));
    if (!pane_valid_id(id)) {
        spec::Spec trial = open_ ? open_->spec : spec::Spec{};
        spec::Source probe;
        probe.id = id;
        probe.catalog = r->catalog;
        trial.sources.push_back(probe);
        const spec::ValidationReport rep = spec::validate(trial);
        for (const auto& d : rep.diagnostics)
            if (d.code == "spec.bad_id") return refuse(d.to_error());
        return refuse(Error("spec.bad_id", "The id " + id + " is not an id.", ""));
    }
    if (open_ && open_->spec.find_source(id))
        return refuse(Error("source.exists", "The spec already has a source " + id + "; choose another id.", ""));
    pending_ids_[r->key] = id;
    primitive_plans_.clear();
    error_ = {};
    return true;
}

std::vector<PaneShape> BuilderPane::transform_shapes() {
    if (shapes_) return *shapes_;
    std::vector<PaneShape> out;
    if (const spec::Source* s = spec_source()) {
        out = pane_transform_shapes(open_->spec, s->id, port_->resolver());
    } else if (const PaneSource* r = find_source(focus_source_)) {
        out.push_back(r->shape);
    }
    shapes_ = out;
    return out;
}

bool BuilderPane::add_transform(spec::TransformOp op) {
    if (blocked()) return false;
    json t;
    switch (op) {
        case spec::TransformOp::slice: t = {{"op", "slice"}, {"axis", 0}, {"index", 0}}; break;
        case spec::TransformOp::reduce: t = {{"op", "reduce"}, {"axis", 0}, {"fn", "mean"}}; break;
        case spec::TransformOp::normalise: t = {{"op", "normalise"}, {"mode", "minmax"}}; break;
    }
    if (const spec::Source* s = spec_source()) {
        json all = json::array();
        for (const auto& x : s->transforms) all.push_back(transform_json(x));
        all.push_back(t);
        PanePlan plan = simulate({invocation("source.patch", {{"id", s->id}, {"patch", {{"transform", all}}}})});
        plan.focus_source = "spec:" + s->id;
        return run_plan(std::move(plan));
    }
    // A catalog source not in the spec: its first transform adds it.
    const PaneSource* r = find_source(focus_source_);
    if (!r || r->catalog.empty()) return refuse(Error("source.not_found", "No source is focused.", ""));
    const std::string id = source_id();
    PanePlan plan = simulate({invocation("source.add", {{"id", id}, {"catalog", r->catalog}, {"transform", json::array({t})}})});
    plan.focus_source = "spec:" + id;
    const std::string was = r->key;
    const bool ok = run_plan(std::move(plan));
    if (ok) {
        pending_ids_.erase(was);
        if (selected_ == was) selected_ = "spec:" + id, selected_column_.clear();
    }
    return ok;
}

bool BuilderPane::set_transform(size_t index, const json& transform) {
    if (blocked()) return false;
    const spec::Source* s = spec_source();
    if (!s || index >= s->transforms.size()) return refuse(Error("source.not_found", "No transform there.", ""));
    json all = json::array();
    for (size_t i = 0; i < s->transforms.size(); ++i) all.push_back(i == index ? transform : transform_json(s->transforms[i]));
    PanePlan plan = simulate({invocation("source.patch", {{"id", s->id}, {"patch", {{"transform", all}}}})});
    plan.focus_source = "spec:" + s->id;
    return run_plan(std::move(plan), "transform." + std::to_string(index));
}

bool BuilderPane::remove_transform(size_t index) {
    if (blocked()) return false;
    const spec::Source* s = spec_source();
    if (!s || index >= s->transforms.size()) return refuse(Error("source.not_found", "No transform there.", ""));
    json all = json::array();
    for (size_t i = 0; i < s->transforms.size(); ++i)
        if (i != index) all.push_back(transform_json(s->transforms[i]));
    PanePlan plan = simulate({invocation("source.patch", {{"id", s->id}, {"patch", {{"transform", all.empty() ? json(nullptr) : all}}}})});
    plan.focus_source = "spec:" + s->id;
    return run_plan(std::move(plan));
}

bool BuilderPane::set_spec_title(const std::string& title) {
    if (blocked()) return false;
    spec::Spec s = open_->spec;
    if (title.empty()) s.applet.title.reset();
    else s.applet.title = title;
    return run_plan(plan_spec_text(s), "spec.title");
}

bool BuilderPane::set_spec_description(const std::string& description) {
    if (blocked()) return false;
    spec::Spec s = open_->spec;
    if (description.empty()) s.applet.description.reset();
    else s.applet.description = description;
    return run_plan(plan_spec_text(s), "spec.description");
}

bool BuilderPane::remove_focused() {
    if (blocked()) return false;
    if (const spec::Panel* p = panel())
        return run_plan(simulate({invocation("panel.remove", {{"id", p->id}})}));
    if (const spec::Source* s = spec_source()) {
        // source.remove's own rule (source.in_use) is the session's; it is
        // checked here the same way.
        for (const auto& pp : open_->spec.panels)
            for (const auto& [slot, v] : pp.bind)
                for (const auto& b : v.bindings)
                    if (b.substr(0, b.find('.')) == s->id)
                        return refuse(Error("source.in_use",
                                            "The panel " + pp.id + " binds the source " + s->id +
                                                "; change or remove that panel first.",
                                            "bind." + slot));
        PanePlan plan = simulate({invocation("source.remove", {{"id", s->id}})});
        const bool ok = run_plan(std::move(plan));
        if (ok) clear_focus();
        return ok;
    }
    return refuse(Error("builder.nothing", "Nothing is focused.", ""));
}

bool BuilderPane::set_ratio(const std::string& path, double ratio) {
    if (blocked()) return false;
    const std::optional<spec::LayoutNode> root = cb::effective_layout(open_->spec);
    if (!root) return refuse(Error("layout.invalid", "The spec has no layout.", ""));
    const spec::LayoutNode* n = pane_layout_node(*root, path);
    if (!n || n->kind != spec::LayoutNode::Kind::split || n->children.size() != 2)
        return refuse(Error("layout.invalid", "No split of two panels there.", path));
    const spec::LayoutNode next = pane_with_ratio(*root, path, pane_snap_ratio(ratio));
    if (open_->spec.layout && next == *open_->spec.layout) return true;
    return run_plan(simulate({invocation("layout.set", {{"layout", cb::layout_to_flat(next, open_->spec)}})}),
                    "ratio." + path);
}

// ---- the spec text ----------------------------------------------------------------------

void BuilderPane::set_editing(bool on) {
    if (on == editing_) return;
    editing_ = on;
    end_gesture();
    draft_ = open_ ? open_->text : std::string();
    draft_pending_ = false;
    diagnostics_.clear();
}

const std::string& BuilderPane::text() const {
    static const std::string none;
    if (editing_) return draft_;
    return open_ ? open_->text : none;
}

bool BuilderPane::set_draft(std::string text) {
    if (!editing_) set_editing(true);
    draft_ = std::move(text);
    if (!port_->ready()) {
        draft_pending_ = true;
        return refuse(Error("builder.busy", "A loader is running; the change waits until it is done.", ""));
    }
    const cb::SpecTextCheck check = cb::check_spec_text(draft_, port_->resolver());
    diagnostics_.clear();
    if (check.error) {
        for (const auto& d : check.diagnostics)
            if (d.is_error()) diagnostics_.push_back(d);
        if (diagnostics_.empty()) {
            spec::Diagnostic d;
            d.code = check.error.code;
            d.message = check.error.message;
            diagnostics_.push_back(d);
        }
        draft_pending_ = true;
        error_ = check.error;
        return false;
    }
    draft_pending_ = false;
    if (open_ && check.spec == open_->spec) {
        error_ = {};
        return true;   // the same spec, written another way
    }
    const bool own = in_gesture();
    if (!own) begin_gesture("text");
    PanePlan plan = simulate({invocation("spec.set_text", {{"text", draft_}})});
    // The draft stays as typed while the person types; reread() keeps it,
    // since it is applied.
    const std::string typed = draft_;
    const bool ok = run_plan(std::move(plan), "text");
    if (!own) end_gesture();
    draft_ = typed;
    return ok;
}

// ---- history and saving -----------------------------------------------------------------

bool BuilderPane::undo() {
    if (!port_->ready()) return refuse(Error("builder.busy", "A loader is running.", ""));
    end_gesture();
    if (editing_ && draft_pending_) set_editing(false);
    const CommandResult r = send(invocation("history.undo", json::object()), {}, {});
    if (!r.ok) return refuse(r.error);
    error_ = {};
    reread();
    if (editing_) draft_ = open_ ? open_->text : std::string();
    return true;
}

bool BuilderPane::redo() {
    if (!port_->ready()) return refuse(Error("builder.busy", "A loader is running.", ""));
    end_gesture();
    if (editing_ && draft_pending_) set_editing(false);
    const CommandResult r = send(invocation("history.redo", json::object()), {}, {});
    if (!r.ok) return refuse(r.error);
    error_ = {};
    reread();
    if (editing_) draft_ = open_ ? open_->text : std::string();
    return true;
}

bool BuilderPane::can_undo() const { return port_->ready() && port_->can_undo(); }

bool BuilderPane::can_redo() const { return port_->ready() && port_->can_redo(); }

bool BuilderPane::save() {
    if (blocked()) return false;
    json args = json::object();
    // The open spec's own applet is saved over; a new one is written new.
    if (open_->path) args["overwrite"] = true;
    const CommandResult r = send(invocation("applet.save", args), {}, {});
    if (!r.ok) return refuse(r.error);
    error_ = {};
    reread();
    return true;
}

// ---- files ------------------------------------------------------------------------------

bool BuilderPane::drop_files(const std::vector<std::string>& paths) {
    if (paths.empty()) return refuse(Error("builder.nothing", "No files.", ""));
    if (import_) return refuse(Error("builder.busy", "An import is running.", ""));
    json list = json::array();
    for (const auto& p : paths) list.push_back(p);
    CommandInvocation inv = invocation("data.import", {{"paths", list}, {"run", true}});
    inv.group = next_group();
    sent_.push_back(inv);
    import_ = port_->submit(std::move(inv));
    error_ = {};
    return true;
}

}  // namespace caliper_host
