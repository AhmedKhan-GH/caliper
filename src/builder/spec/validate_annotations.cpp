// The validator's checks of annotations, legends and scale bars
// (design/instrument-interface.md, Annotations and Markers, Legends and
// Scale Bars).
//
// Each annotation has a unique id in its panel and the keys of its kind:
// a label `at` and `text`, a threshold `axis` and `value`, a band `axis`
// and `range`, a marker `at`. A key that the kind does not take is
// spec.unknown_option and a missing key spec.missing_key. A value place
// holds a literal or a reference, checked as validate_bound_text.cpp checks
// one. The spec runtime draws annotations in series, histogram, 2-D points
// and scene panels; heatmaps and images still refuse them with
// spec.not_built, as every panel but a series refuses a legend set to true or
// false and every panel but a scene a scale bar, which their drawing does not
// read yet.
#include "validate_additions.h"

#include <caliper_builder/spec/annotation.h>

#include <cmath>

namespace caliper_builder::spec::detail {

// validate_bound_text.cpp: the checks that bound texts and annotations share.
std::optional<Reference> read_place_reference(const AdditionsCheck& c, const nlohmann::json& v, const std::string& path);
std::string reference_value_type(const AdditionsCheck& c, const Reference& r, const std::string& path, bool* ok);
void check_bound_text(const AdditionsCheck& c, const BoundText& t, const std::string& path, const char* format_key,
                      const std::vector<std::string>& extra);

namespace {

bool number_type(const std::string& t) { return t == "number" || t == "integer" || t.empty(); }

// One number of a value place: a number literal, or a reference that reads
// a number. `what` names the place in the message.
void check_number(const AdditionsCheck& c, const nlohmann::json& v, const std::string& path, const std::string& what) {
    if (v.is_number()) return;
    if (looks_like_reference(v)) {
        const std::optional<Reference> r = read_place_reference(c, v, path);
        if (!r) return;
        bool ok = true;
        const std::string type = reference_value_type(c, *r, path, &ok);
        if (!ok) return;
        if (!number_type(type))
            c.sink.error("spec.reference_type", path + " reads a " + type + ", and " + what + " is a number.", path);
        else if (!r->fallback.is_null() && !r->fallback.is_number())
            c.sink.error("spec.bad_reference", path + ".default must be a number.", key_path(path, "default"));
        return;
    }
    c.sink.error("spec.wrong_type", path + " must be a number or a reference, the " + what + ".", path);
}

// The key of an annotation that the kind does not take.
void refuse_key(const AdditionsCheck& c, bool set, const char* key, const Annotation& a, const std::string& path) {
    if (!set) return;
    c.sink.error("spec.unknown_option",
                 std::string("A ") + annotation_kind_name(a.kind) + " takes no " + key + "; " + path + "." + key +
                     " does not apply.",
                 key_path(path, key));
}

void require_key(const AdditionsCheck& c, bool set, const char* key, const Annotation& a, const std::string& path) {
    if (set) return;
    c.sink.error("spec.missing_key", std::string("A ") + annotation_kind_name(a.kind) + " needs the key " + key + ".",
                 key_path(path, key));
}

// `range`: [lo, hi] of numbers or references, or a reference that reads a pair.
void check_range(const AdditionsCheck& c, const nlohmann::json& v, const std::string& path) {
    if (looks_like_reference(v)) {
        const std::optional<Reference> r = read_place_reference(c, v, path);
        if (!r) return;
        bool ok = true;
        const std::string type = reference_value_type(c, *r, path, &ok);
        if (ok && !type.empty() && type != "pair")
            c.sink.error("spec.reference_type",
                         path + " reads a " + type + "; a band's range reads a pair, such as a range control.", path);
        if (ok && !r->fallback.is_null() &&
            !(r->fallback.is_array() && r->fallback.size() == 2 && r->fallback[0].is_number() && r->fallback[1].is_number()))
            c.sink.error("spec.bad_reference", path + ".default must be [lo, hi].", key_path(path, "default"));
        return;
    }
    if (!v.is_array() || v.size() != 2) {
        c.sink.error("spec.wrong_type", path + " must be [lo, hi] or a reference to a range.", path);
        return;
    }
    check_number(c, v[0], index_path(path, 0), "low end of the band");
    check_number(c, v[1], index_path(path, 1), "high end of the band");
    if (v[0].is_number() && v[1].is_number() && v[0].get<double>() > v[1].get<double>())
        c.sink.error("spec.bad_value", path + " has its low end above its high end.", path);
}

// `at`: a point in data units, an item { entity, index } of a scene, or a
// selection reference, which places the annotation at the pick.
void check_at(const AdditionsCheck& c, const Panel& p, const nlohmann::json& v, const std::string& path) {
    const bool scene = p.primitive == Primitive::scene;
    if (v.is_object() && !looks_like_reference(v)) {
        if (!scene) {
            c.sink.error("spec.wrong_type", path + " must be [x, y] in data units, or a reference per value.", path);
            return;
        }
        for (auto it = v.begin(); it != v.end(); ++it)
            if (it.key() != "entity" && it.key() != "index")
                c.sink.error("spec.unknown_option", path + " takes entity and index, not " + quote(it.key()) + ".",
                             key_path(path, it.key()));
        if (!v.contains("entity") || !v["entity"].is_string()) {
            c.sink.error("spec.missing_key", path + " needs the entity's path as a text.", key_path(path, "entity"));
        } else {
            const std::string e = v["entity"].get<std::string>();
            bool found = false;
            for (const Entity& en : p.entities) found = found || en.path == e;
            if (!found) c.sink.error("spec.bad_value", path + " names the entity " + quote(e) + ", which the panel does not draw.", key_path(path, "entity"));
        }
        if (!v.contains("index")) {
            c.sink.error("spec.missing_key", path + " needs the item's index.", key_path(path, "index"));
        } else if (v["index"].is_number_integer()) {
            if (v["index"].get<std::int64_t>() < 0) c.sink.error("spec.out_of_range", path + ".index must be 0 or more.", key_path(path, "index"));
        } else {
            check_number(c, v["index"], key_path(path, "index"), "item's index");
        }
        return;
    }
    if (looks_like_reference(v)) {
        const std::optional<Reference> r = read_place_reference(c, v, path);
        if (!r) return;
        if (r->kind != Reference::Kind::selection || (r->field && *r->field != "position")) {
            c.sink.error("spec.reference_type",
                         path + " places the annotation at a pick, so it reads a selection without a field, or its "
                                "position.",
                         path);
            return;
        }
        bool ok = true;
        reference_value_type(c, *r, path, &ok);
        if (!ok) return;
        if (!scene || c.spec.find_panel(r->target)->primitive != Primitive::scene)
            c.sink.error("spec.reference_type",
                         path + " reads a pick, which only a scene panel gives and only a scene panel places.", path);
        return;
    }
    const std::size_t n = scene ? 3 : 2;
    if (!v.is_array() || v.size() != n) {
        c.sink.error("spec.wrong_type",
                     path + (scene ? " must be [x, y, z], { entity, index } or a selection reference."
                                   : " must be [x, y] in data units."),
                     path);
        return;
    }
    for (std::size_t i = 0; i < n; ++i) check_number(c, v[i], index_path(path, i), "position");
}

void check_annotation(const AdditionsCheck& c, const Panel& p, const Annotation& a, const std::string& path) {
    const bool scene = p.primitive == Primitive::scene;
    const bool has_axis = a.axis.has_value(), has_value = !a.value.is_null(), has_range = !a.range.is_null();
    const bool has_at = !a.at.is_null(), has_text = a.text.has_value(), has_leader = a.leader.has_value();
    switch (a.kind) {
        case AnnotationKind::label:
            refuse_key(c, has_axis, "axis", a, path);
            refuse_key(c, has_value, "value", a, path);
            refuse_key(c, has_range, "range", a, path);
            require_key(c, has_at, "at", a, path);
            require_key(c, has_text, "text", a, path);
            break;
        case AnnotationKind::threshold:
            refuse_key(c, has_at, "at", a, path);
            refuse_key(c, has_range, "range", a, path);
            refuse_key(c, has_leader, "leader", a, path);
            require_key(c, has_axis, "axis", a, path);
            require_key(c, has_value, "value", a, path);
            break;
        case AnnotationKind::band:
            refuse_key(c, has_at, "at", a, path);
            refuse_key(c, has_value, "value", a, path);
            refuse_key(c, has_leader, "leader", a, path);
            require_key(c, has_axis, "axis", a, path);
            require_key(c, has_range, "range", a, path);
            break;
        case AnnotationKind::marker:
            refuse_key(c, has_axis, "axis", a, path);
            refuse_key(c, has_value, "value", a, path);
            refuse_key(c, has_range, "range", a, path);
            refuse_key(c, has_leader, "leader", a, path);
            require_key(c, has_at, "at", a, path);
            break;
    }
    if (has_axis && (a.kind == AnnotationKind::threshold || a.kind == AnnotationKind::band)) {
        const std::string& x = *a.axis;
        if (!(x == "x" || x == "y" || (scene && x == "z")))
            c.sink.error("spec.bad_value",
                         quote(x) + " is not an axis of this panel; the axes are " + (scene ? "x, y and z." : "x and y."),
                         key_path(path, "axis"));
    }
    if (has_value && a.kind == AnnotationKind::threshold) check_number(c, a.value, key_path(path, "value"), "threshold");
    if (has_range && a.kind == AnnotationKind::band) check_range(c, a.range, key_path(path, "range"));
    if (has_at && (a.kind == AnnotationKind::label || a.kind == AnnotationKind::marker)) check_at(c, p, a.at, key_path(path, "at"));
    if (a.emphasis && *a.emphasis != "accent" && *a.emphasis != "muted")
        c.sink.error("spec.bad_value", quote(*a.emphasis) + " is not an emphasis; it is accent or muted.",
                     key_path(path, "emphasis"));
    if (has_text) check_bound_text(c, *a.text, path, "text", {});
}

}  // namespace

void validate_annotations(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i) {
        const Panel& p = c.spec.panels[i];
        const std::string path = index_path("panel", i);
        const std::string op = path + ".options";
        if (!p.options.legend.is_null()) {
            const nlohmann::json& l = p.options.legend;
            if (!(l.is_boolean() || (l.is_string() && l.get<std::string>() == "auto")))
                c.sink.error("spec.bad_value", op + ".legend must be auto, true or false.", op + ".legend");
            else if (l.is_boolean() && p.primitive != Primitive::series)
                not_built(c.sink, "the option legend set to true or false outside a series panel", op + ".legend");
        }
        if (p.primitive != Primitive::scene && p.options.scene.scale_bar)
            not_built(c.sink, "the option scale_bar outside a scene", op + ".scale_bar");
        if (p.annotations.empty()) continue;
        const std::string ap = path + ".annotation";
        if (p.primitive == Primitive::table || p.primitive == Primitive::text) {
            c.sink.error("spec.unknown_option", std::string("A ") + to_string(p.primitive) + " panel takes no annotations.",
                         ap);
            continue;
        }
        // Series, histogram and 2-D points panels draw annotations inside
        // their plots, and a scene panel through its projection; a heatmap
        // and an image do not draw them yet. Their annotations are still
        // checked below.
        if (p.primitive == Primitive::heatmap || p.primitive == Primitive::image)
            not_built(c.sink, std::string("annotations on a ") + to_string(p.primitive) + " panel", ap);
        if (p.annotations.size() > limits::kAnnotations)
            c.sink.error("spec.too_many",
                         ap + " has " + std::to_string(p.annotations.size()) + " annotations, and a panel has at most 64.",
                         ap);
        std::vector<std::string> ids;
        for (std::size_t j = 0; j < p.annotations.size(); ++j) {
            const Annotation& a = p.annotations[j];
            const std::string at = index_path(ap, j);
            if (!valid_id(a.id)) {
                c.sink.error("spec.bad_id",
                             quote(a.id) + " is not an id; an id is a lowercase letter, then lowercase letters, digits, "
                                           "_ and -.",
                             key_path(at, "id"));
            } else if (contains(ids, a.id)) {
                c.sink.error("spec.duplicate_id", "Two annotations of the panel have the id " + quote(a.id) + ".",
                             key_path(at, "id"));
            }
            ids.push_back(a.id);
            check_annotation(c, p, a, at);
        }
    }
}

}  // namespace caliper_builder::spec::detail
