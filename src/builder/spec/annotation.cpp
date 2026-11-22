// Bound texts and annotations (annotation.h): the names of the kinds and the
// comparisons.
#include <caliper_builder/spec/annotation.h>

namespace caliper_builder::spec {

bool operator==(const BoundText& a, const BoundText& b) { return a.format == b.format && a.fields == b.fields; }

const char* annotation_kind_name(AnnotationKind kind) {
    switch (kind) {
        case AnnotationKind::label: return "label";
        case AnnotationKind::threshold: return "threshold";
        case AnnotationKind::band: return "band";
        case AnnotationKind::marker: return "marker";
    }
    return "label";
}

std::optional<AnnotationKind> annotation_kind_from_name(std::string_view name) {
    for (AnnotationKind k :
         {AnnotationKind::label, AnnotationKind::threshold, AnnotationKind::band, AnnotationKind::marker})
        if (name == annotation_kind_name(k)) return k;
    return std::nullopt;
}

bool operator==(const Annotation& a, const Annotation& b) {
    return a.id == b.id && a.kind == b.kind && a.axis == b.axis && a.value == b.value && a.range == b.range &&
           a.at == b.at && a.text == b.text && a.leader == b.leader && a.emphasis == b.emphasis && a.x == b.x;
}

}  // namespace caliper_builder::spec
