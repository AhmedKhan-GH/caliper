// The colormap names of the format (colormaps.h).
#include <caliper_builder/spec/colormaps.h>

namespace caliper_builder::spec {

const std::vector<std::string>& colormap_names() {
    // These are the four names of format version 0. The other twelve
    // (cividis, inferno, plasma, turbo, twilight, coolwarm, batlow, vik, roma,
    // oslo, lajolla and berlin) join with their color tables in the graphics
    // layer, so that a name never exists without its colors.
    static const std::vector<std::string> names = {"viridis", "magma", "rdbu", "greys"};
    return names;
}

std::optional<int> colormap_id(std::string_view name) {
    const std::vector<std::string>& names = colormap_names();
    for (std::size_t i = 0; i < names.size(); ++i)
        if (names[i] == name) return static_cast<int>(i);
    return std::nullopt;
}

}  // namespace caliper_builder::spec
