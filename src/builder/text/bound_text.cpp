// The format language of bound texts (caliper_builder/text/bound_text.h).
// Not built yet: every format is refused with spec.bad_format, and the
// validator refuses bound texts (spec.not_built).
#include <caliper_builder/text/bound_text.h>

namespace caliper_builder::text {

Error parse_bound_text(std::string_view format, const std::vector<std::string>& field_names, ParsedText* out) {
    (void)format;
    (void)field_names;
    *out = {};
    return Error("spec.bad_format", "Bound texts are not built yet.");
}

std::string format_bound_text(const ParsedText& text, const std::map<std::string, nlohmann::json>& values) {
    (void)text;
    (void)values;
    return {};
}

}  // namespace caliper_builder::text
