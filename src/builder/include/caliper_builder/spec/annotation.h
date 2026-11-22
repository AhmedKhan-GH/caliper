#pragma once
// caliper_builder::spec: bound texts and annotations
// (design/instrument-interface.md, sections Bound Text and Annotations and
// Markers).
//
// A bound text is a format with named fields, each a source id or a
// reference; it stands in a text panel (options `format` and `fields`), in a
// readout control and in each annotation. An annotation marks a value in a
// panel: a label, a threshold line, a band or a marker. It is an item of its
// panel (Panel::annotations, the array `annotation`).
//
// The codec reads and writes these keys. The grammar of a format is in
// caliper_builder/text/bound_text.h, and the validator's checks are in
// validate_bound_text.cpp and validate_annotations.cpp. Nothing here throws.
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include <nlohmann/json.hpp>

#include <caliper_builder/spec/reference.h>

namespace caliper_builder::spec {

namespace limits {
inline constexpr std::size_t kAnnotations = 64;       // in one panel, spec.too_many
inline constexpr std::size_t kFormatChars = 1024;
inline constexpr std::size_t kFormatFields = 16;
}  // namespace limits

// A format and its fields. A field is a source id (its value at the cursor)
// or a reference.
struct BoundText {
    std::string format;
    std::map<std::string, std::variant<std::string, Reference>> fields;
};
bool operator==(const BoundText& a, const BoundText& b);
inline bool operator!=(const BoundText& a, const BoundText& b) { return !(a == b); }

enum class AnnotationKind { label, threshold, band, marker };

// "label", "threshold", "band", "marker".
const char* annotation_kind_name(AnnotationKind kind);
std::optional<AnnotationKind> annotation_kind_from_name(std::string_view name);

// One entry of a panel's `annotation` array. Which keys apply depends on
// `kind`; the validator reports a key that does not.
struct Annotation {
    std::string id;                          // unique in the panel
    AnnotationKind kind = AnnotationKind::label;
    std::optional<std::string> axis;         // threshold and band: "x", "y" or "z"
    nlohmann::json value;                    // threshold: a number or a reference; null when absent
    nlohmann::json range;                    // band: [lo, hi] of numbers or references, or a range reference
    nlohmann::json at;                       // label and marker: a point, { entity, index } or a selection reference
    std::optional<BoundText> text;           // the keys `text` (the format) and `fields`
    std::optional<bool> leader;              // label: a leader line, default true
    std::optional<std::string> emphasis;     // "accent" (the default) or "muted"
    nlohmann::json x;                        // the extension table, null when absent
};
bool operator==(const Annotation& a, const Annotation& b);
inline bool operator!=(const Annotation& a, const Annotation& b) { return !(a == b); }

}  // namespace caliper_builder::spec
