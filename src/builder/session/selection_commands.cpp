// The selection commands (design/instrument-interface.md, Commands and Pick
// Fields).
//
// selection.set makes the pick that a person's click on an item of an entity
// makes: it resolves the panel, reads the item's fields from the entity's
// arrays, pushes the `pick` event with those fields and the caller's origin,
// and records the pick, which runs the panel's `select` write. With `clear` it
// clears the panel's selection and pushes a `select` event that names no
// entity. selection.get gives the fields of the panel's last pick.
#include <caliper_builder/session/instrument_commands.h>

#include <cmath>
#include <cstdint>
#include <optional>

namespace caliper_builder {

namespace {

using nlohmann::json;

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

Error bad_item(const std::string& message) { return Error("selection.bad_item", message, "args"); }

// The array `name` of a resolved entity: a slot of that name, or a geometry
// array of that name in any of its slots. Null when there is none.
const TensorData* array_of(const EntityData& e, const std::string& name) {
    for (const SlotData& s : e.slots)
        if (s.slot == name && s.geometry.empty() && s.tensor.numel() > 0) return &s.tensor;
    for (const SlotData& s : e.slots) {
        const auto it = s.geometry.find(name);
        if (it != s.geometry.end()) return &it->second;
    }
    return nullptr;
}

// Whether the chain of the source that `e` binds starts with a window, so
// that the rows of a surface stand for steps.
bool windowed(const spec::Spec& s, const spec::Entity& e) {
    std::string id = e.source.value_or(std::string());
    for (int depth = 0; depth < 64 && !id.empty(); ++depth) {
        const spec::Source* src = s.find_source(id);
        if (!src) return false;
        for (const auto& t : src->transforms)
            if (t.op == spec::TransformOp::window) return true;
        id = src->from.value_or(std::string());
    }
    return false;
}

// The fields of item `index` of a surface: the cell (r, c) of the pooled
// grid, given as the row and the column of the input before pooling. The
// index comes from the caller, so each array is read through element_at().
Error surface_fields(const EntityData& e, std::int64_t index, bool steps, PickFields* out) {
    const TensorData* h = array_of(e, "height");
    if (!h || h->rank() != 2 || h->shape()[1] < 1) return bad_item("The surface " + e.path + " has no grid of heights yet.");
    const std::int64_t rows = h->shape()[0], cols = h->shape()[1];
    if (index < 0 || index >= rows * cols)
        return bad_item("The surface " + e.path + " has " + std::to_string(rows * cols) + " cells, and the index is " +
                        std::to_string(index) + ".");
    const std::int64_t r = index / cols, c = index % cols;
    const TensorData* x = array_of(e, "x");
    const TensorData* y = array_of(e, "y");
    const TensorData* values = array_of(e, "values");
    auto at = [](const TensorData* t, std::int64_t i) { return t ? t->element_at(i) : std::nullopt; };
    // The pool factor: the distance of two pooled columns, as the operator's
    // `y` holds the mean column index of each block.
    std::int64_t f = 1;
    const std::optional<double> y0 = at(y, 0), y1 = at(y, 1);
    if (y0 && y1 && std::isfinite(*y1 - *y0) && std::fabs(*y1 - *y0) < 1e15)
        f = std::max<std::int64_t>(1, std::llround(*y1 - *y0));
    else if (y0 && !y1 && std::isfinite(*y0) && std::fabs(*y0) < 1e15)
        f = std::max<std::int64_t>(1, std::llround(2 * *y0 + 1));
    out->row = r * f + (f - 1) / 2;
    const std::optional<double> yc = at(y, c), xr = at(x, r);
    out->column = yc && std::isfinite(*yc) && std::fabs(*yc) < 1e15 ? static_cast<std::int64_t>(std::floor(*yc))
                                                                     : c * f + (f - 1) / 2;
    const double px = xr ? *xr : static_cast<double>(r);
    const double py = yc ? *yc : static_cast<double>(c);
    const double pz = h->element_at(index).value_or(0.0);
    out->position = std::array<double, 3>{px, py, pz};
    out->value = values && values->numel() == h->numel() ? values->element_at(index).value_or(pz) : pz;
    if (steps && xr && std::isfinite(px) && std::fabs(px) < 1e15) out->step = std::llround(px);
    return {};
}

// The fields of item `index` of a points or lines entity, read through
// element_at() because the index comes from the caller.
Error item_fields(const EntityData& e, std::int64_t index, PickFields* out) {
    std::int64_t n = -1;
    std::array<double, 3> p{0, 0, 0};
    const TensorData* pos = array_of(e, "positions");
    const bool rows = pos && pos->rank() == 2 && pos->shape()[1] >= 2;
    const char* axes[] = {"x", "y", "z"};
    if (rows) {
        n = pos->shape()[0];
    } else {
        for (int k = 0; k < 3; ++k)
            if (const TensorData* a = array_of(e, axes[k]))
                n = n < 0 ? static_cast<std::int64_t>(a->numel()) : std::min<std::int64_t>(n, a->numel());
    }
    if (n < 0) return bad_item("The entity " + e.path + " has no positions yet.");
    if (index < 0 || index >= n)
        return bad_item("The entity " + e.path + " has " + std::to_string(n) + " items, and the index is " +
                        std::to_string(index) + ".");
    if (rows) {
        for (std::int64_t k = 0; k < std::min<std::int64_t>(3, pos->shape()[1]); ++k)
            p[static_cast<std::size_t>(k)] = pos->element_at(index, k).value_or(0.0);
    } else {
        for (int k = 0; k < 3; ++k)
            if (const TensorData* a = array_of(e, axes[k])) p[static_cast<std::size_t>(k)] = a->element_at(index).value_or(0.0);
    }
    out->index = index;
    out->position = p;
    if (const TensorData* v = array_of(e, "values"); v && static_cast<std::int64_t>(v->numel()) == n)
        out->value = v->element_at(index);
    return {};
}

CommandOutput selection_set(const CommandContext& ctx, InstrumentContext& ic) {
    const json& args = ctx.invocation.args;
    const std::string panel_id = str_arg(args, "panel");
    const spec::Panel* panel = ic.spec.find_panel(panel_id);
    if (!panel) return CommandOutput::fail(Error("panel.not_found", "The open spec has no panel " + panel_id + ".", "args.panel"));
    if (args.value("clear", false)) {
        ic.values.clear_pick(panel_id);
        if (ic.events) {
            Event e;
            e.kind = EventKind::select;
            e.panel = panel_id;
            e.origin = command_origin_name(ctx.invocation.origin);
            e.fields = json{{"entities", json::array()}};
            ic.events->push(std::move(e));
        }
        return CommandOutput::ok(json{{"fields", nullptr}});
    }
    if (panel->primitive != spec::Primitive::scene)
        return CommandOutput::fail(bad_item("The panel " + panel_id + " is a " + spec::to_string(panel->primitive) +
                                            " panel; only the items of a scene panel's entities are picked."));
    const std::string entity_path = str_arg(args, "entity");
    if (entity_path.empty() || !args.contains("index") || !args["index"].is_number_integer())
        return CommandOutput::fail(bad_item("selection.set names an entity and the index of an item, or clear."));
    // An item index is at least 0; an unsigned value past the signed range
    // would wrap to a negative one.
    const json& given = args["index"];
    if ((given.is_number_unsigned() && given.get<std::uint64_t>() > std::uint64_t(INT64_MAX)) ||
        (!given.is_number_unsigned() && given.get<std::int64_t>() < 0))
        return CommandOutput::fail(Error("selection.bad_item",
                                         "The index of an item is a whole number from 0, and it is " + given.dump() + ".",
                                         "args.index"));
    const std::int64_t index = given.get<std::int64_t>();
    const spec::Entity* entity = nullptr;
    for (const auto& e : panel->entities)
        if (e.path == entity_path) entity = &e;
    if (!entity)
        return CommandOutput::fail(bad_item("The panel " + panel_id + " has no entity " + entity_path + "."));
    if (!ic.store) return CommandOutput::fail(bad_item("The panel's data cannot be read here."));
    // The panel as it shows now: its references' values and the cursor.
    const CursorState cursor = ic.history.cursor();
    const ChainContext chain{&ic.values, &ic.history, cursor.step, cursor.live};
    const PanelData data = resolve_panel(ic.spec, panel_id, *ic.store, ic.limits, ic.sources, &chain);
    const EntityData* resolved = nullptr;
    for (const auto& e : data.entities)
        if (e.path == entity_path) resolved = &e;
    if (!resolved || resolved->error)
        return CommandOutput::fail(bad_item("The entity " + entity_path + " has no data to pick yet" +
                                            (resolved && resolved->error ? ": " + resolved->error.message : ".")));
    PickFields fields;
    fields.entity = entity_path;
    Error err = entity->draw == spec::DrawKind::surface ? surface_fields(*resolved, index, windowed(ic.spec, *entity), &fields)
                                                        : item_fields(*resolved, index, &fields);
    if (err) return CommandOutput::fail(std::move(err));
    if (entity->draw == spec::DrawKind::surface) fields.index = index;
    const json all = pick_fields_json(fields);
    if (ic.events) {
        Event e;
        e.kind = EventKind::pick;
        e.panel = panel_id;
        e.origin = command_origin_name(ctx.invocation.origin);
        e.fields = json{{"entity", entity_path}, {"primitive", spec::draw_kind_name(entity->draw)}, {"index", index},
                        {"fields", all}};
        if (fields.position) e.fields["position"] = *fields.position;
        ic.events->push(std::move(e));
    }
    ic.values.record_pick(panel_id, fields, ctx.invocation.origin, ic.events);
    return CommandOutput::ok(json{{"fields", all}});
}

CommandOutput selection_get(const CommandContext& ctx, InstrumentContext& ic) {
    const std::string panel_id = str_arg(ctx.invocation.args, "panel");
    if (!ic.spec.find_panel(panel_id))
        return CommandOutput::fail(Error("panel.not_found", "The open spec has no panel " + panel_id + ".", "args.panel"));
    const auto pick = ic.values.last_pick(panel_id);
    return CommandOutput::ok(json{{"fields", pick ? pick_fields_json(*pick) : json()}});
}

}  // namespace

CommandOutput selection_command(const CommandContext& ctx, InstrumentContext& ic) {
    if (!ic.spec_open) return CommandOutput::fail(Error("spec.no_open_spec", "No spec is open."));
    if (ctx.invocation.name == "selection.set") return selection_set(ctx, ic);
    if (ctx.invocation.name == "selection.get") return selection_get(ctx, ic);
    return CommandOutput::fail(not_implemented(ctx.invocation.name));
}

}  // namespace caliper_builder
