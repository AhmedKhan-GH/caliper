// The validator's checks of bound texts in text panels and readouts
// (design/instrument-interface.md, Bound Text), and the checks of a
// reference that bound texts and annotations share.
//
// A bound text's format must parse against the names of its fields table
// (caliper_builder/text/bound_text.h), each field must name a source of the
// spec or hold a reference whose target and field exist, and a number format
// must not read a value that the validator knows to be a text or a boolean.
// The reference checks here cover the places of this file and of
// validate_annotations.cpp, which declares the two shared functions below.
#include "validate_additions.h"

#include <caliper_builder/spec/annotation.h>
#include <caliper_builder/text/bound_text.h>

namespace caliper_builder::spec::detail {

namespace {

// The names a field may have: a letter or an underscore, then letters,
// digits and underscores, at most 64 characters.
bool valid_field_name(const std::string& s) {
    if (s.empty() || s.size() > 64) return false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        const bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        if (!letter && !(i > 0 && c >= '0' && c <= '9')) return false;
    }
    return true;
}

// The type of a state variable's value.
std::string state_value_type(const StateVar& s) {
    switch (s.type) {
        case StateType::number: return "number";
        case StateType::integer: return "integer";
        case StateType::boolean: return "boolean";
        case StateType::text: return "text";
        case StateType::range: return "pair";
        case StateType::list: return "list";
    }
    return "";
}

// The type of a control's value, or "none" for a control that holds none.
std::string control_value_type(const Control& c) {
    switch (c.kind) {
        case ControlKind::slider: return "number";
        case ControlKind::range: return "pair";
        case ControlKind::toggle: return "boolean";
        case ControlKind::picker:
        case ControlKind::text:
        case ControlKind::segmented: return "";
        case ControlKind::button:
        case ControlKind::readout:
        case ControlKind::transport: return "none";
    }
    return "";
}

}  // namespace

// Reads the reference table `v` at a value place, reporting each fault of
// its form with spec.bad_reference: no input key or two, an input or a field
// that is not a text, `entity` outside a selection, a key outside the form
// or a `default` that is a reference. Empty when `v` is not a valid form.
std::optional<Reference> read_place_reference(const AdditionsCheck& c, const nlohmann::json& v,
                                              const std::string& path) {
    if (!v.is_object()) return std::nullopt;
    int inputs = 0;
    for (const char* k : kReferenceInputKeys) inputs += v.contains(k) ? 1 : 0;
    const auto refuse = [&](const std::string& why) {
        c.sink.error("spec.bad_reference", path + " is not a reference: " + why, path);
        return std::nullopt;
    };
    if (inputs != 1) return refuse("it needs exactly one of control, selection and state.");
    for (auto it = v.begin(); it != v.end(); ++it) {
        bool known = false;
        for (const char* k : kReferenceInputKeys) known = known || it.key() == k;
        for (const char* k : kReferenceOtherKeys) known = known || it.key() == k;
        if (!known) return refuse("the key " + quote(it.key()) + " is not part of a reference.");
    }
    for (const char* k : {"field", "entity"})
        if (v.contains(k) && !v[k].is_string()) return refuse(std::string("its ") + k + " must be a text.");
    if (v.contains("default") && looks_like_reference(v["default"])) return refuse("its default holds another reference.");
    std::optional<Reference> r = parse_reference(v);
    if (!r) return refuse("its input must name an id as a text.");
    if (r->entity && r->kind != Reference::Kind::selection) return refuse("only a selection takes an entity.");
    return r;
}

// Checks the target and the field of `r` and gives the type of the value it
// reads: "number", "integer", "boolean", "text", "pair", "list", "position",
// "pick", or "" when the type is not known. Reports spec.unknown_control,
// spec.unknown_state, spec.unknown_panel, spec.unknown_field and, for a
// control that holds no value, spec.reference_type; *ok is then false.
std::string reference_value_type(const AdditionsCheck& c, const Reference& r, const std::string& path, bool* ok) {
    *ok = true;
    const auto fail = [&](const char* code, const std::string& message) {
        c.sink.error(code, message, path);
        *ok = false;
        return std::string();
    };
    switch (r.kind) {
        case Reference::Kind::control: {
            const Control* ctl = c.spec.find_control(r.target);
            if (!ctl) return fail("spec.unknown_control", path + " reads the control " + quote(r.target) + ", which the spec does not have.");
            const std::string type = control_value_type(*ctl);
            if (type == "none")
                return fail("spec.reference_type", path + " reads the control " + quote(r.target) + ", a " +
                                                       control_kind_name(ctl->kind) + ", which holds no value.");
            if (r.field) {
                if (ctl->kind != ControlKind::range || (*r.field != "lo" && *r.field != "hi"))
                    return fail("spec.unknown_field", path + " reads the field " + quote(*r.field) + " of the control " +
                                                          quote(r.target) + "; only a range has the fields lo and hi.");
                return "number";
            }
            return type;
        }
        case Reference::Kind::state: {
            if (r.target == "cursor") {
                const std::string f = r.field.value_or("step");
                if (f == "step" || f == "first" || f == "latest") return "integer";
                if (f == "live" || f == "playing") return "boolean";
                if (f == "speed") return "number";
                return fail("spec.unknown_field", path + " reads the cursor's field " + quote(f) +
                                                      "; the fields are step, first, latest, live, playing and speed.");
            }
            const StateVar* s = c.spec.find_state(r.target);
            if (!s) return fail("spec.unknown_state", path + " reads the state " + quote(r.target) + ", which the spec does not have.");
            if (r.field) {
                if (s->type != StateType::range || (*r.field != "lo" && *r.field != "hi"))
                    return fail("spec.unknown_field", path + " reads the field " + quote(*r.field) + " of the state " +
                                                          quote(r.target) + "; only a range state has the fields lo and hi.");
                return "number";
            }
            return state_value_type(*s);
        }
        case Reference::Kind::selection: {
            const Panel* p = c.spec.find_panel(r.target);
            if (!p) return fail("spec.unknown_panel", path + " reads a pick of the panel " + quote(r.target) + ", which the spec does not have.");
            if (r.entity) {
                bool found = false;
                for (const Entity& e : p->entities) found = found || e.path == *r.entity;
                if (!found)
                    return fail("spec.unknown_field", path + " counts picks on the entity " + quote(*r.entity) +
                                                          ", which the panel " + quote(r.target) + " does not draw.");
            }
            if (!r.field) return "pick";
            const std::string& f = *r.field;
            if (f == "entity" || f == "annotation") return "text";
            if (f == "index" || f == "row_id" || f == "row" || f == "column" || f == "step") return "integer";
            if (f == "value") return "number";
            if (f == "position") return "position";
            return fail("spec.unknown_field", path + " reads the pick field " + quote(f) +
                                                  "; the fields are entity, index, row_id, row, column, position, value, "
                                                  "step and annotation.");
        }
    }
    return "";
}

// Checks one bound text at `path` (the table that holds `format_key` and
// `fields`): the size of the fields table, the names, each field's source
// or reference, the format's grammar and the fit of each number format.
// `extra` names fields that the place adds, such as a text panel's bound
// `value`.
void check_bound_text(const AdditionsCheck& c, const BoundText& t, const std::string& path, const char* format_key,
                      const std::vector<std::string>& extra) {
    const std::string fp = key_path(path, "fields");
    if (t.fields.size() > limits::kFormatFields)
        c.sink.error("spec.bad_format",
                     fp + " has " + std::to_string(t.fields.size()) + " fields, and a bound text has at most 16.", fp);
    std::vector<std::string> names = extra;
    std::map<std::string, std::string> types;
    for (const auto& [name, field] : t.fields) {
        const std::string ep = key_path(fp, name);
        if (!valid_field_name(name)) {
            c.sink.error("spec.bad_value",
                         quote(name) + " is not a field name; a name is a letter or an underscore, then letters, digits "
                                       "and underscores.",
                         ep);
            continue;
        }
        if (contains(extra, name)) {
            c.sink.error("spec.bad_value", ep + " repeats the field " + quote(name) + ", which the panel's bound slot gives.", ep);
            continue;
        }
        names.push_back(name);
        if (const std::string* source = std::get_if<std::string>(&field)) {
            // A source id, or "<source>.<column>" for one column of a table.
            const std::size_t dot = source->find('.');
            if (!c.spec.find_source(*source) && (dot == std::string::npos || !c.spec.find_source(source->substr(0, dot))))
                c.sink.error("spec.unknown_source", ep + " names the source " + quote(*source) + ", which the spec does not have.", ep);
            continue;
        }
        const Reference& r = std::get<Reference>(field);
        bool ok = true;
        const std::string type = reference_value_type(c, r, ep, &ok);
        if (!ok) continue;
        if (!r.fallback.is_null() && (r.fallback.is_object() || r.fallback.is_array()) && type != "pair" &&
            type != "list" && type != "position")
            c.sink.error("spec.bad_reference", ep + ".default must be a number, a text or a boolean.", key_path(ep, "default"));
        types[name] = type;
    }
    text::ParsedText parsed;   // the type, from bound_text.h
    const std::string formp = key_path(path, format_key);
    if (Error e = text::parse_bound_text(t.format, names, &parsed)) {
        c.sink.error(e.code, formp + ": " + e.message, formp);
        return;
    }
    for (const text::FieldFormat& f : parsed.fields) {
        if (f.type == 's') continue;
        const auto it = types.find(f.name);
        if (it == types.end()) continue;
        if (it->second == "text" || it->second == "boolean")
            c.sink.error("spec.bad_format",
                         formp + " formats the field " + quote(f.name) + " as a number, and it reads a " + it->second +
                             "; use the format s.",
                         formp);
    }
}

void validate_bound_text(const AdditionsCheck& c) {
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i) {
        const Panel& p = c.spec.panels[i];
        if (!p.options.bound_text) continue;
        const std::string op = index_path("panel", i) + ".options";
        if (p.options.text)
            c.sink.error("spec.bad_value",
                         op + " has both text and format; a text panel shows a static text or a bound text.",
                         op + ".format");
        std::vector<std::string> extra;
        if (p.bind.count("value")) extra.push_back("value");
        check_bound_text(c, *p.options.bound_text, op, "format", extra);
    }
    for (std::size_t i = 0; i < c.spec.controls.size(); ++i) {
        const Control& ctl = c.spec.controls[i];
        const std::string op = index_path("control", i) + ".options";
        if (ctl.kind == ControlKind::readout && !ctl.options.bound_text) {
            c.sink.error("spec.missing_key", "A readout needs the option format.", op + ".format");
            continue;
        }
        if (ctl.options.bound_text) check_bound_text(c, *ctl.options.bound_text, op, "format", {});
    }
}

}  // namespace caliper_builder::spec::detail
