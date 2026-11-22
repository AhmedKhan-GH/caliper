// The entities of a scene panel in panel.data (panel_data.h). Each entity of
// the panel, in spec order, gives its path, its draw kind and either an error
// or its slots with their shapes and the generation of its data. The shape
// check follows the plan's Array Resolution: a wrong shape or length is the
// entity's error and does not stop the panel.
#include <caliper_builder/session/panel_data.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_builder {

namespace {

using nlohmann::json;

// The shape of a slot: the tensor's shape, or [n] for a column or a table's rows.
std::vector<std::int64_t> slot_shape(const SlotData& s) {
    if (s.kind == SlotData::Kind::tensor) return s.tensor.shape();
    return {static_cast<std::int64_t>(s.values.size())};
}

const SlotData* find_slot(const EntityData& d, const char* name) {
    for (const SlotData& s : d.slots)
        if (s.slot == name) return &s;
    return nullptr;
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
    std::string out = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) out += (i ? ", " : "") + std::to_string(shape[i]);
    return out + ")";
}

// The shape or length error of an entity's slots, or an empty Error.
Error check_entity_shape(const spec::Entity& entity, const EntityData& data) {
    const std::string where = "entity " + entity.path;
    if (entity.draw == spec::DrawKind::surface) {
        const SlotData* height = find_slot(data, "height");
        if (!height) return {};
        const auto h = slot_shape(*height);
        if (h.size() != 2)
            return Error("entity.shape", "The height of " + where + " is " + shape_text(h) + "; it must be rank 2.");
        const std::int64_t R = h[0], C = h[1];
        if (const SlotData* x = find_slot(data, "x"); x && slot_shape(*x) != std::vector<std::int64_t>{R})
            return Error("entity.length", "The x of " + where + " must have " + std::to_string(R) + " items, one per row.");
        if (const SlotData* y = find_slot(data, "y"); y && slot_shape(*y) != std::vector<std::int64_t>{C})
            return Error("entity.length",
                         "The y of " + where + " must have " + std::to_string(C) + " items, one per column.");
        if (const SlotData* v = find_slot(data, "values"); v && slot_shape(*v) != std::vector<std::int64_t>{R, C})
            return Error("entity.length", "The values of " + where + " must be " + shape_text({R, C}) + ".");
        return {};
    }
    if (const SlotData* pos = find_slot(data, "positions")) {
        const auto p = slot_shape(*pos);
        if (p.size() != 2 || p[1] != 3)
            return Error("entity.shape", "The positions of " + where + " are " + shape_text(p) + "; they must be (N, 3).");
    }
    const SlotData* first = nullptr;
    std::int64_t n = 0;
    for (const SlotData& s : data.slots) {
        const auto shape = slot_shape(s);
        if (shape.empty()) continue;
        if (!first) {
            first = &s;
            n = shape[0];
        } else if (shape[0] != n) {
            return Error("entity.length", "The slots of " + where + " have different lengths: " + first->slot + " has " +
                                              std::to_string(n) + " items and " + s.slot + " has " +
                                              std::to_string(shape[0]) + ".");
        }
    }
    return {};
}

json error_json(const Error& e) { return {{"code", e.code}, {"message", e.message}}; }

// The values of a tensor slot in row-major order, NaN and infinities as
// null; the smallest and largest finite value beside them.
void add_values(const SlotData& s, bool with_values, json* item) {
    if (s.kind != SlotData::Kind::tensor || !s.tensor.valid()) return;
    const std::uint64_t n = s.tensor.numel();
    double lo = 0, hi = 0;
    bool any = false;
    json values = json::array();
    for (std::uint64_t i = 0; i < n; ++i) {
        const double v = s.tensor.element(i);
        const bool finite = std::isfinite(v);
        if (finite) {
            lo = any ? std::min(lo, v) : v;
            hi = any ? std::max(hi, v) : v;
            any = true;
        }
        if (with_values) values.push_back(finite ? json(v) : json(nullptr));
    }
    (*item)["min"] = any ? json(lo) : json(nullptr);
    (*item)["max"] = any ? json(hi) : json(nullptr);
    if (with_values) (*item)["values"] = std::move(values);
}

}  // namespace

// Sets (*out)["entities"] to one object per entity of the panel, in spec order.
void encode_scene_entities(const spec::Panel& panel, const PanelData& data, const PanelDataLimits& limits,
                           nlohmann::json* out) {
    json list = json::array();
    for (const spec::Entity& entity : panel.entities) {
        json item{{"path", entity.path}, {"draw", spec::draw_kind_name(entity.draw)}};
        const EntityData* found = nullptr;
        for (const EntityData& d : data.entities)
            if (d.path == entity.path) found = &d;
        if (!found) {
            item["error"] = {{"code", "entity.no_data"}, {"message", "The entity " + entity.path + " has no data."}};
        } else if (found->error) {
            item["error"] = error_json(found->error);
        } else if (Error shape = check_entity_shape(entity, *found)) {
            item["error"] = error_json(shape);
        } else {
            // Each slot's shape, its range, and its values when it is one axis
            // (x, y, steps, a column) or small; a grid of heights gives its range only.
            json slots = json::array();
            for (const SlotData& s : found->slots) {
                json one{{"slot", s.slot}, {"binding", s.binding}, {"shape", slot_shape(s)}};
                const auto shape = slot_shape(s);
                const std::uint64_t n = s.kind == SlotData::Kind::tensor && s.tensor.valid() ? s.tensor.numel() : 0;
                add_values(s, (shape.size() == 1 && n <= limits.max_points) || n <= 4096, &one);
                slots.push_back(std::move(one));
            }
            item["slots"] = std::move(slots);
            item["generation"] = found->generation;
        }
        list.push_back(std::move(item));
    }
    (*out)["entities"] = std::move(list);
}

}  // namespace caliper_builder
