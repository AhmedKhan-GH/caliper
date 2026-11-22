#pragma once
// caliper_builder::spec: the colormap names of the format
// (design/instrument-interface.md, section Colormaps).
//
// One table holds the names, in id order: spec panels, entities, program
// outputs and the scene core read the same list, and the graphics layer
// holds the colors of each id. Version 0 has the first four names; the
// others join as the graphics layer gains their tables. Nothing here throws.
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::spec {

// The names that a spec may use, in id order.
const std::vector<std::string>& colormap_names();

// The id of a name (its index in colormap_names()), or empty.
std::optional<int> colormap_id(std::string_view name);

}  // namespace caliper_builder::spec
