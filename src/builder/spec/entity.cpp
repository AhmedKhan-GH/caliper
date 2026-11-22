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

EntityStyle entity_style(const Entity&) {
    // The scene entities work fills this: it reads each option of
    // `entity.options` into the typed view.
    return EntityStyle{};
}

}  // namespace caliper_builder::spec
