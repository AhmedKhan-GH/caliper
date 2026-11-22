#pragma once
// caliper_builder::spec: the entities of a scene panel
// (design/instrument-interface.md, section Scene Entities).
//
// An entity is a batch that a scene panel builds from the spec's own
// sources: points, lines or a surface. The app writes it under the reserved
// root `spec/` of the panel's scene, so that a stage names it as
// "spec/<path>". A scene panel with at least one entity needs no `scene`
// slot.
//
// The codec reads and writes the array `entity` of a panel into
// Panel::entities. The options keep their JSON, because each draw kind takes
// a subset and several of them take references; entity_style() gives the
// typed view, and validate_entities.cpp checks them. Nothing here throws.
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/spec/reference.h>

namespace caliper_builder::spec {

namespace limits {
inline constexpr std::size_t kEntities = 64;                    // in one panel, spec.too_many
inline constexpr std::size_t kEntityPathSegments = 8;
inline constexpr std::int64_t kEntityItems = 1048576;           // points and lines
inline constexpr std::int64_t kSurfaceCells = 262144;
inline constexpr std::int64_t kSurfaceSide = 4096;
inline constexpr std::int64_t kPanelArrayBytes = 64ll << 20;
}  // namespace limits

// The root of the entities in a panel's scene.
inline constexpr const char* kSpecEntityRoot = "spec/";

enum class DrawKind { points, lines, surface };

// "points", "lines", "surface".
const char* draw_kind_name(DrawKind kind);
std::optional<DrawKind> draw_kind_from_name(std::string_view name);

// A literal pose: a translation, a quaternion [x, y, z, w] and a scale.
struct Pose {
    std::array<double, 3> position{0, 0, 0};
    std::array<double, 4> rotation{0, 0, 0, 1};
    std::array<double, 3> scale{1, 1, 1};
};
bool operator==(const Pose& a, const Pose& b);

// One entry of a scene panel's `entity` array. `bind` is either a source id,
// which binds each named array of a geometry source to the slot of the same
// name, or a table of slots.
struct Entity {
    std::string path;                                    // 1 to 8 segments of [a-z0-9_-], joined by "/"
    DrawKind draw = DrawKind::points;
    std::optional<std::string> title;                    // the path when absent
    std::optional<std::string> source;                   // `bind` written as a source id
    std::map<std::string, ValueOr<std::string>> slots;   // `bind` written as a table: slot to a binding or a reference
    nlohmann::json options;                              // the draw kind's options; null when absent
    std::optional<Pose> pose;
    nlohmann::json x;                                    // the extension table, null when absent
};
bool operator==(const Entity& a, const Entity& b);
inline bool operator!=(const Entity& a, const Entity& b) { return !(a == b); }

// An axis position of an entity without a slot for it: a number in data
// units, or the low or high end of the data box.
struct AxisPlace {
    enum class Kind { value, min, max };
    Kind kind = Kind::value;
    double value = 0;
};

// The typed view of an entity's options. An option that is not set is
// empty; a reference stays a reference until the app evaluates it.
struct EntityStyle {
    std::optional<ValueOr<std::string>> colormap;
    std::optional<ValueOr<double>> vmin;
    std::optional<ValueOr<double>> vmax;
    std::optional<double> size_px;                  // points
    std::optional<std::string> shape;               // points
    std::optional<double> width_px;                 // lines
    std::optional<std::string> mode;                // lines: "strip" or "segments"
    std::optional<AxisPlace> x, y, z;
    std::vector<std::string> own;                   // axes mapped onto the box by the entity's own range
    std::optional<std::int64_t> depth_offset;       // -8 to 8; 1 for points and lines, 0 for a surface
    std::optional<Reference> highlight;             // with highlight_along
    std::optional<std::string> highlight_along;     // "row", "column" or "index"
    std::optional<ValueOr<bool>> visible;
};

// The typed view of `entity.options`, or empty fields where an option is
// missing or has the wrong form (the validator reports those).
EntityStyle entity_style(const Entity& entity);

}  // namespace caliper_builder::spec
