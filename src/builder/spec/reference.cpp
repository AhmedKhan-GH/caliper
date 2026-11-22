// Binding references (reference.h): the comparison, the names, the form test
// and the table of a reference.
#include <caliper_builder/spec/reference.h>

#include <string_view>

namespace caliper_builder::spec {

namespace {

bool is_input_key(std::string_view key) {
    for (const char* k : kReferenceInputKeys)
        if (key == k) return true;
    return false;
}

std::optional<Reference::Kind> kind_of_key(std::string_view key) {
    if (key == "control") return Reference::Kind::control;
    if (key == "selection") return Reference::Kind::selection;
    if (key == "state") return Reference::Kind::state;
    return std::nullopt;
}

}  // namespace

bool operator==(const Reference& a, const Reference& b) {
    return a.kind == b.kind && a.target == b.target && a.field == b.field && a.entity == b.entity &&
           a.fallback == b.fallback;
}

const char* reference_kind_name(Reference::Kind kind) {
    switch (kind) {
        case Reference::Kind::control: return "control";
        case Reference::Kind::selection: return "selection";
        case Reference::Kind::state: return "state";
    }
    return "state";
}

bool looks_like_reference(const nlohmann::json& v) {
    if (!v.is_object()) return false;
    for (auto it = v.begin(); it != v.end(); ++it)
        if (is_input_key(it.key())) return true;
    return false;
}

std::optional<Reference> parse_reference(const nlohmann::json& v) {
    if (!v.is_object()) return std::nullopt;
    std::optional<Reference::Kind> kind;
    std::string target;
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (!is_input_key(it.key())) continue;
        if (kind || !it.value().is_string()) return std::nullopt;  // two input keys, or one that is not text
        kind = kind_of_key(it.key());
        target = it.value().get<std::string>();
    }
    if (!kind) return std::nullopt;
    Reference r;
    r.kind = *kind;
    r.target = std::move(target);
    // The keys `field` and `entity` hold text. A value of another type is not a
    // reference that can be written back.
    for (const char* key : {"field", "entity"}) {
        auto it = v.find(key);
        if (it == v.end()) continue;
        if (!it->is_string()) return std::nullopt;
        (std::string_view(key) == "field" ? r.field : r.entity) = it->get<std::string>();
    }
    if (auto it = v.find("default"); it != v.end()) r.fallback = *it;
    return r;
}

nlohmann::json reference_json(const Reference& r) {
    nlohmann::json j = nlohmann::json::object();
    j[reference_kind_name(r.kind)] = r.target;
    if (r.field) j["field"] = *r.field;
    if (r.entity) j["entity"] = *r.entity;
    if (!r.fallback.is_null()) j["default"] = r.fallback;
    return j;
}

}  // namespace caliper_builder::spec
