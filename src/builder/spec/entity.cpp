// The entities of a scene panel (entity.h): the names of the draw kinds and
// the comparisons.
#include <caliper_builder/spec/entity.h>

namespace caliper_builder::spec {

const char* draw_kind_name(DrawKind kind) {
    switch (kind) {
        case DrawKind::points: return "points";
        case DrawKind::lines: return "lines";
        case DrawKind::surface: return "surface";
    }
    return "points";
}

std::optional<DrawKind> draw_kind_from_name(std::string_view name) {
    for (DrawKind k : {DrawKind::points, DrawKind::lines, DrawKind::surface})
        if (name == draw_kind_name(k)) return k;
    return std::nullopt;
}

bool operator==(const Pose& a, const Pose& b) {
    return a.position == b.position && a.rotation == b.rotation && a.scale == b.scale;
}

bool operator==(const Entity& a, const Entity& b) {
    return a.path == b.path && a.draw == b.draw && a.title == b.title && a.source == b.source && a.slots == b.slots &&
           a.options == b.options && a.pose == b.pose && a.x == b.x;
}

namespace {

using nlohmann::json;

// A number, or a reference that stays for the app to evaluate.
std::optional<ValueOr<double>> number_or_reference(const json& v) {
    if (v.is_number()) return ValueOr<double>(v.get<double>());
    if (auto r = parse_reference(v)) return ValueOr<double>(*r);
    return std::nullopt;
}

std::optional<AxisPlace> axis_place(const json& v) {
    AxisPlace a;
    if (v.is_number()) {
        a.value = v.get<double>();
        return a;
    }
    if (v == "min") {
        a.kind = AxisPlace::Kind::min;
        return a;
    }
    if (v == "max") {
        a.kind = AxisPlace::Kind::max;
        return a;
    }
    return std::nullopt;
}

}  // namespace

EntityStyle entity_style(const Entity& entity) {
    EntityStyle s;
    const json& o = entity.options;
    if (!o.is_object()) return s;
    if (auto it = o.find("colormap"); it != o.end()) {
        if (it->is_string())
            s.colormap = ValueOr<std::string>(it->get<std::string>());
        else if (auto r = parse_reference(*it))
            s.colormap = ValueOr<std::string>(*r);
    }
    if (auto it = o.find("vmin"); it != o.end()) s.vmin = number_or_reference(*it);
    if (auto it = o.find("vmax"); it != o.end()) s.vmax = number_or_reference(*it);
    if (auto it = o.find("size_px"); it != o.end() && it->is_number()) s.size_px = it->get<double>();
    if (auto it = o.find("shape"); it != o.end() && it->is_string()) s.shape = it->get<std::string>();
    if (auto it = o.find("width_px"); it != o.end() && it->is_number()) s.width_px = it->get<double>();
    if (auto it = o.find("mode"); it != o.end() && it->is_string()) s.mode = it->get<std::string>();
    if (auto it = o.find("x"); it != o.end()) s.x = axis_place(*it);
    if (auto it = o.find("y"); it != o.end()) s.y = axis_place(*it);
    if (auto it = o.find("z"); it != o.end()) s.z = axis_place(*it);
    if (auto it = o.find("own"); it != o.end() && it->is_array())
        for (const json& a : *it)
            if (a.is_string()) s.own.push_back(a.get<std::string>());
    if (auto it = o.find("depth_offset"); it != o.end() && it->is_number_integer())
        s.depth_offset = it->get<std::int64_t>();
    // `highlight` is a reference with the extra key `along`.
    if (auto it = o.find("highlight"); it != o.end() && it->is_object()) {
        json ref = *it;
        if (auto a = ref.find("along"); a != ref.end()) {
            if (a->is_string()) s.highlight_along = a->get<std::string>();
            ref.erase(a);
        }
        s.highlight = parse_reference(ref);
        if (!s.highlight) s.highlight_along.reset();
    }
    if (auto it = o.find("visible"); it != o.end()) {
        if (it->is_boolean())
            s.visible = ValueOr<bool>(it->get<bool>());
        else if (auto r = parse_reference(*it))
            s.visible = ValueOr<bool>(*r);
    }
    return s;
}

}  // namespace caliper_builder::spec
