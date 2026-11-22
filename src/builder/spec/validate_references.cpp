// The validator's checks of binding references
// (design/instrument-interface.md, Binding References).
//
// Each reference at a value place names a known control, panel or state, a
// field that its input has, and gives a value whose type fits the place; its
// default is a literal of that type. The graph from each input to the
// controls whose options read it, and from each control to the state it
// writes, has no cycle. A window never reads a source whose own steps read a
// reference.
//
// Every place is checked here: the steps of sources, the slots and options of
// panels, entities, annotations, bound texts and the min and max of controls;
// the features that read them refuse what they do not build yet.
#include "validate_additions.h"

#include <caliper_builder/spec/colormaps.h>
#include <caliper_builder/spec/state.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>

namespace caliper_builder::spec::detail {

namespace {

using json = nlohmann::json;

// The type that a place takes, or that a reference gives.
enum class VT {
    none,      // gives no value (a button, a readout)
    integer,
    number,
    boolean,
    text,
    pair,      // [lo, hi]
    list,      // a list of texts
    pick,      // a whole pick: a selection without a field
    // Places only:
    colormap,  // a text whose values are colormap names
    source,    // a text whose values are source ids
    any,
};

const char* vt_name(VT t) {
    switch (t) {
        case VT::none: return "no value";
        case VT::integer: return "an integer";
        case VT::number: return "a number";
        case VT::boolean: return "true or false";
        case VT::text: return "a text";
        case VT::pair: return "a pair [lo, hi]";
        case VT::list: return "a list";
        case VT::pick: return "a pick";
        case VT::colormap: return "a colormap name";
        case VT::source: return "a source id";
        case VT::any: return "any value";
    }
    return "a value";
}

// What a reference gives: its type and, for a text or an integer, the values
// it can take when they are known.
struct Given {
    VT type = VT::none;
    const std::vector<json>* choices = nullptr;   // a state's choices
    std::vector<json> own;                        // a control's choices
    bool known_choices = false;
};

bool is_integral(const json& v) {
    if (v.is_number_integer()) return true;
    if (!v.is_number_float()) return false;
    const double d = v.get<double>();
    return std::isfinite(d) && std::floor(d) == d;
}

bool is_pair(const json& v) {
    return v.is_array() && v.size() == 2 && v[0].is_number() && v[1].is_number();
}

class References {
public:
    explicit References(const AdditionsCheck& c) : c_(c) {
        for (const auto& s : c.spec.states) states_.emplace(s.id, &s);
        for (const auto& ctl : c.spec.controls) controls_.emplace(ctl.id, &ctl);
    }

    void run() {
        sources();
        panels();
        controls();
        params();
        cycles();
        history();
    }

private:
    const AdditionsCheck& c_;
    std::map<std::string, const StateVar*> states_;
    std::map<std::string, const Control*> controls_;

    const StateVar* state_of(const std::string& id) const {
        const auto it = states_.find(id);
        return it == states_.end() ? nullptr : it->second;
    }
    const Control* control_of(const std::string& id) const {
        const auto it = controls_.find(id);
        return it == controls_.end() ? nullptr : it->second;
    }
    const StateVar* bound_state(const Control& ctl) const {
        const auto t = parse_control_target(ctl.bind);
        return t && t->kind == ControlTarget::Kind::state ? state_of(t->name) : nullptr;
    }

    static VT state_vt(StateType t) {
        switch (t) {
            case StateType::number: return VT::number;
            case StateType::integer: return VT::integer;
            case StateType::boolean: return VT::boolean;
            case StateType::text: return VT::text;
            case StateType::range: return VT::pair;
            case StateType::list: return VT::list;
        }
        return VT::none;
    }

    // The field of `r` after its input is known; reports spec.unknown_field
    // and returns false when the input has no such field.
    bool field_of_pair(const Reference& r, Given& g, const std::string& path, const std::string& what) {
        if (!r.field) return true;
        if (g.type == VT::pair && (*r.field == "lo" || *r.field == "hi")) {
            g.type = VT::number;
            return true;
        }
        c_.sink.error("spec.unknown_field",
                      quote(*r.field) + " is not a field of " + what +
                          (g.type == VT::pair ? "; a range gives the fields lo and hi." : ", which has no fields."),
                      path + ".field");
        return false;
    }

    // What `r` gives, or empty after a reported error.
    std::optional<Given> given(const Reference& r, const std::string& path) {
        if (r.entity && r.kind != Reference::Kind::selection) {
            c_.sink.error("spec.bad_reference", "entity belongs to a selection reference only.", path + ".entity");
            return std::nullopt;
        }
        Given g;
        switch (r.kind) {
            case Reference::Kind::state: {
                if (r.target == kCursorState) {
                    static const std::map<std::string, VT> fields = {
                        {"step", VT::integer},    {"first", VT::integer}, {"latest", VT::integer},
                        {"live", VT::boolean}, {"playing", VT::boolean},  {"speed", VT::number}};
                    const auto it = fields.find(r.field.value_or("step"));
                    if (it == fields.end()) {
                        c_.sink.error("spec.unknown_field",
                                      quote(*r.field) + " is not a field of the cursor; its fields are step, first, "
                                                        "latest, live, playing and speed.",
                                      path + ".field");
                        return std::nullopt;
                    }
                    g.type = it->second;
                    return g;
                }
                const StateVar* s = state_of(r.target);
                if (!s) {
                    c_.sink.error("spec.unknown_state",
                                  quote(r.target) + " is not a state variable of this spec, and not cursor.",
                                  path + ".state");
                    return std::nullopt;
                }
                g.type = state_vt(s->type);
                g.choices = &s->choices;
                g.known_choices = !s->choices.empty();
                if (!field_of_pair(r, g, path, "the state " + quote(s->id))) return std::nullopt;
                return g;
            }
            case Reference::Kind::control: {
                const Control* ctl = control_of(r.target);
                if (!ctl) {
                    c_.sink.error("spec.unknown_control", quote(r.target) + " is not a control of this spec.",
                                  path + ".control");
                    return std::nullopt;
                }
                if (const StateVar* s = bound_state(*ctl)) {
                    g.type = state_vt(s->type);
                    g.choices = &s->choices;
                    g.known_choices = !s->choices.empty();
                } else {
                    switch (ctl->kind) {
                        case ControlKind::slider: g.type = VT::number; break;
                        case ControlKind::range: g.type = VT::pair; break;
                        case ControlKind::toggle: g.type = VT::boolean; break;
                        case ControlKind::text: g.type = VT::text; break;
                        case ControlKind::picker:
                        case ControlKind::segmented: {
                            bool integers = !ctl->options.choices.empty();
                            for (const auto& ch : ctl->options.choices) {
                                integers = integers && is_integral(ch.value);
                                g.own.push_back(ch.value);
                            }
                            g.type = ctl->options.multiple && *ctl->options.multiple ? VT::list
                                     : integers                                         ? VT::integer
                                                                                        : VT::text;
                            g.known_choices = !ctl->options.choices.empty();
                            break;
                        }
                        default: g.type = VT::none; break;
                    }
                }
                if (!field_of_pair(r, g, path, "the control " + quote(ctl->id))) return std::nullopt;
                return g;
            }
            case Reference::Kind::selection: {
                const Panel* p = c_.spec.find_panel(r.target);
                if (!p) {
                    c_.sink.error("spec.unknown_panel", quote(r.target) + " is not a panel of this spec.",
                                  path + ".selection");
                    return std::nullopt;
                }
                if (p->primitive != Primitive::scene) {
                    c_.sink.error("spec.reference_type",
                                  "The panel " + quote(r.target) + " is a " + to_string(p->primitive) +
                                      " panel, and only a scene panel has picks.",
                                  path + ".selection");
                    return std::nullopt;
                }
                if (!r.field) {
                    g.type = VT::pick;
                    return g;
                }
                static const std::map<std::string, VT> fields = {
                    {"entity", VT::text},   {"index", VT::integer}, {"row_id", VT::integer}, {"row", VT::integer},
                    {"column", VT::integer}, {"x", VT::number},      {"y", VT::number},       {"z", VT::number},
                    {"value", VT::number},  {"step", VT::integer},  {"annotation", VT::text}};
                const auto it = fields.find(*r.field);
                if (it == fields.end()) {
                    // A field of the item map of a program that draws in the
                    // panel (design/programmable-graphics.md, Picking): a number.
                    for (const auto& prog : c_.spec.programs)
                        if (is_piece_instance(prog) && prog.panel && *prog.panel == p->id) {
                            g.type = VT::number;
                            return g;
                        }
                    c_.sink.error("spec.unknown_field",
                                  quote(*r.field) + " is not a pick field; the fields are entity, index, row_id, row, "
                                                    "column, x, y, z, value and step, and the item map's fields of a "
                                                    "program that draws in the panel.",
                                  path + ".field");
                    return std::nullopt;
                }
                g.type = it->second;
                return g;
            }
        }
        return std::nullopt;
    }

    // Whether a literal fits a place of type `place`.
    bool literal_fits(VT place, const json& v) const {
        switch (place) {
            case VT::integer: return is_integral(v);
            case VT::number: return v.is_number() && std::isfinite(v.get<double>());
            case VT::boolean: return v.is_boolean();
            case VT::text: return v.is_string();
            case VT::pair: return is_pair(v);
            case VT::list: return v.is_array();
            case VT::pick: return v.is_object() || v.is_array();
            case VT::colormap: return v.is_string() && colormap_id(v.get<std::string>()).has_value();
            case VT::source: return v.is_string() && c_.spec.find_source(v.get<std::string>()) != nullptr;
            case VT::any: return true;
            case VT::none: return false;
        }
        return false;
    }

    // Whether what a reference gives fits the place; `why` says what the
    // values must be when a text does not fit.
    bool fits(VT place, const Given& g, std::string* why) const {
        switch (place) {
            case VT::integer:
            case VT::number: return g.type == VT::integer || g.type == VT::number;
            case VT::colormap:
            case VT::source: {
                if (g.type != VT::text || !g.known_choices) {
                    *why = place == VT::colormap ? "a text state whose choices are colormap names"
                                                 : "a text state whose choices are source ids";
                    return false;
                }
                const std::vector<json>& values = g.choices ? *g.choices : g.own;
                for (const auto& v : values)
                    if (!literal_fits(place, v)) {
                        *why = place == VT::colormap ? "a text whose every choice is a colormap name"
                                                     : "a text whose every choice is a source id of this spec";
                        return false;
                    }
                return true;
            }
            case VT::any: return g.type != VT::none;
            default: return g.type == place;
        }
    }

    // Checks one reference that the codec read, at `path`, for a place of
    // type `place`.
    void check(const Reference& r, VT place, const std::string& path) {
        const std::optional<Given> g = given(r, path);
        if (!r.fallback.is_null() && !literal_fits(place, r.fallback))
            c_.sink.error("spec.bad_reference",
                          path + ".default must be " + vt_name(place) + ", the type of the place.", path + ".default");
        if (!g) return;
        std::string why;
        if (!fits(place, *g, &why))
            c_.sink.error("spec.reference_type",
                          "The reference gives " + std::string(vt_name(g->type)) + ", and " + path + " takes " +
                              (why.empty() ? vt_name(place) : why) + ".",
                          path);
    }

    // A value kept as written (entity options, annotations): a reference
    // when it has the form of one, else a literal. `extra` names one more key
    // the table may hold (highlight's `along`).
    bool check_json(const json& v, VT place, const std::string& path, const char* extra = nullptr) {
        if (!looks_like_reference(v)) return false;
        std::optional<Reference> r = parse_reference(v);
        bool ok = r.has_value() && !looks_like_reference(r->fallback);
        for (auto it = v.begin(); ok && it != v.end(); ++it) {
            bool known = extra && it.key() == extra;
            for (const char* k : kReferenceInputKeys) known = known || it.key() == k;
            for (const char* k : kReferenceOtherKeys) known = known || it.key() == k;
            ok = known;
        }
        if (!ok) {
            c_.sink.error("spec.bad_reference",
                          path + " is not a binding reference; it needs one of control, selection and state with a "
                                 "text value, and may add field, entity and default.",
                          path);
            return true;
        }
        check(*r, place, path);
        return true;
    }

    void sources() {
        for (std::size_t i = 0; i < c_.spec.sources.size(); ++i) {
            const auto& transforms = c_.spec.sources[i].transforms;
            for (std::size_t j = 0; j < transforms.size(); ++j) {
                const Transform& t = transforms[j];
                const std::string path = index_path(index_path("source", i) + ".transform", j);
                for (const auto& [key, ref] : t.references) {
                    // The places of integer_places() (spec.h), whose range the
                    // resolver checks again for each value a reference gives.
                    if (!integer_place(t.op, key))
                        c_.sink.error("spec.reference_not_allowed",
                                      key + " of a " + to_string(t.op) + " step takes a literal only.",
                                      path + "." + key);
                    else
                        check(ref, VT::integer, path + "." + key);
                }
            }
        }
    }

    void bound_text(const BoundText& text, const std::string& path) {
        for (const auto& [name, field] : text.fields)
            if (std::holds_alternative<Reference>(field))
                check(std::get<Reference>(field), VT::any, path + ".fields." + name);
    }

    void entity(const Entity& e, const std::string& path) {
        for (const auto& [slot, value] : e.slots)
            if (is_reference(value)) check(std::get<Reference>(value), VT::source, path + ".bind." + slot);
        if (!e.options.is_object()) return;
        static const std::map<std::string, VT> places = {{"colormap", VT::colormap}, {"vmin", VT::number},
                                                         {"vmax", VT::number},       {"highlight", VT::integer},
                                                         {"visible", VT::boolean}};
        for (auto it = e.options.begin(); it != e.options.end(); ++it) {
            const std::string op = path + ".options." + it.key();
            const auto place = places.find(it.key());
            if (place == places.end()) {
                if (looks_like_reference(it.value()))
                    c_.sink.error("spec.reference_not_allowed", it.key() + " of an entity takes a literal only.", op);
                continue;
            }
            check_json(it.value(), place->second, op, it.key() == "highlight" ? "along" : nullptr);
        }
    }

    void annotation(const Annotation& a, const std::string& path) {
        if (!a.value.is_null()) check_json(a.value, VT::number, path + ".value");
        if (!a.range.is_null() && !check_json(a.range, VT::pair, path + ".range") && a.range.is_array())
            for (std::size_t k = 0; k < a.range.size(); ++k)
                check_json(a.range[k], VT::number, index_path(path + ".range", k));
        if (!a.at.is_null()) {
            if (looks_like_reference(a.at)) {
                std::optional<Reference> r = parse_reference(a.at);
                if (r && r->kind != Reference::Kind::selection)
                    c_.sink.error("spec.reference_type",
                                  path + ".at takes a selection reference, a point or an entity's item.", path + ".at");
                else
                    check_json(a.at, VT::pick, path + ".at");
            } else if (a.at.is_array()) {
                for (std::size_t k = 0; k < a.at.size(); ++k)
                    check_json(a.at[k], VT::number, index_path(path + ".at", k));
            }
        }
        if (a.text) bound_text(*a.text, path);
    }

    void panels() {
        for (std::size_t i = 0; i < c_.spec.panels.size(); ++i) {
            const Panel& p = c_.spec.panels[i];
            const std::string path = index_path("panel", i);
            // The resolver applies these (resolve.h, panel_with_values and the
            // slot's binding read now).
            for (const auto& [key, ref] : p.options.references)
                check(ref, key == "colormap" ? VT::colormap : VT::pair, path + ".options." + key);
            for (const auto& [slot, value] : p.bind)
                if (value.reference) check(*value.reference, VT::source, path + ".bind." + slot);
            if (p.options.bound_text) bound_text(*p.options.bound_text, path + ".options");
            for (std::size_t k = 0; k < p.entities.size(); ++k) entity(p.entities[k], index_path(path + ".entity", k));
            for (std::size_t k = 0; k < p.annotations.size(); ++k)
                annotation(p.annotations[k], index_path(path + ".annotation", k));
        }
    }

    void controls() {
        for (std::size_t i = 0; i < c_.spec.controls.size(); ++i) {
            const ControlOptions& o = c_.spec.controls[i].options;
            const std::string path = index_path("control", i) + ".options";
            for (const auto& [key, ref] : o.references) {
                if (key == "min" || key == "max")
                    check(ref, VT::number, path + "." + key);
                else
                    c_.sink.error("spec.reference_not_allowed", key + " of a control takes a literal only.",
                                  path + "." + key);
            }
            if (o.bound_text) bound_text(*o.bound_text, path);
        }
    }

    void params() {
        auto each = [&](const Params& params, const std::string& path) {
            for (const auto& [name, value] : params)
                if (is_reference(value)) check(std::get<Reference>(value), VT::number, path + ".params." + name);
        };
        for (std::size_t i = 0; i < c_.spec.programs.size(); ++i)
            each(c_.spec.programs[i].params, index_path("program", i));
        for (std::size_t i = 0; i < c_.spec.stages.size(); ++i) each(c_.spec.stages[i].params, index_path("stage", i));
    }

    // ---- spec.reference_cycle -------------------------------------------------
    // Nodes are "control:<id>" and "state:<id>". A control's options that read
    // an input give an edge from the input to the control; a control bound to
    // a state gives an edge from the control to the state. A reference to a
    // control bound to a state reads the state.

    std::string node_of(const Reference& r) const {
        if (r.kind == Reference::Kind::state) return r.target == kCursorState ? std::string() : "state:" + r.target;
        if (r.kind != Reference::Kind::control) return {};
        const Control* ctl = control_of(r.target);
        if (!ctl) return {};
        if (const StateVar* s = bound_state(*ctl)) return "state:" + s->id;
        return "control:" + ctl->id;
    }

    static std::string node_name(const std::string& node) {
        return node.rfind("control:", 0) == 0 ? "the control " + quote(node.substr(8))
                                               : "the state " + quote(node.substr(6));
    }

    void cycles() {
        std::map<std::string, std::set<std::string>> edges;
        std::map<std::string, std::size_t> index_of;   // control node to its index
        for (std::size_t i = 0; i < c_.spec.controls.size(); ++i) {
            const Control& ctl = c_.spec.controls[i];
            const std::string self = "control:" + ctl.id;
            index_of.emplace(self, i);
            for (const auto& [key, ref] : ctl.options.references) {
                (void)key;
                const std::string from = node_of(ref);
                if (!from.empty()) edges[from].insert(self);
            }
            if (const StateVar* s = bound_state(ctl)) edges[self].insert("state:" + s->id);
        }
        // Depth-first search; a back edge closes a cycle, which is reported
        // once, at the first control on it.
        std::map<std::string, int> mark;   // 1 on the stack, 2 done
        std::vector<std::string> stack;
        std::set<std::string> reported;
        std::function<void(const std::string&)> visit = [&](const std::string& n) {
            mark[n] = 1;
            stack.push_back(n);
            for (const std::string& next : edges[n]) {
                if (mark[next] == 1) {
                    auto start = std::find(stack.begin(), stack.end(), next);
                    std::vector<std::string> cycle(start, stack.end());
                    std::vector<std::string> sorted = cycle;
                    std::sort(sorted.begin(), sorted.end());
                    std::string id;
                    for (const auto& s : sorted) id += s + "\n";
                    if (!reported.insert(id).second) continue;
                    // Each step of the cycle in words: a control writes its
                    // state, and the options of a control read an input.
                    std::string text;
                    std::size_t at = c_.spec.controls.size();
                    for (std::size_t k = 0; k < cycle.size(); ++k) {
                        const std::string& a = cycle[k];
                        const std::string& b = cycle[(k + 1) % cycle.size()];
                        if (!text.empty()) text += ", and ";
                        const bool writes = a.rfind("control:", 0) == 0 && b.rfind("state:", 0) == 0;
                        text += writes ? node_name(a) + " writes " + node_name(b).substr(4)
                                       : "the options of " + node_name(b).substr(4) + " read " + node_name(a).substr(4);
                        if (auto it = index_of.find(a); it != index_of.end()) at = std::min(at, it->second);
                    }
                    c_.sink.error("spec.reference_cycle",
                                  "The options of a control read the value that it writes: " + text + ".",
                                  at < c_.spec.controls.size() ? index_path("control", at) + ".options" : "control");
                } else if (mark[next] == 0) {
                    visit(next);
                }
            }
            stack.pop_back();
            mark[n] = 2;
        };
        std::vector<std::string> nodes;
        for (const auto& [n, out] : edges) nodes.push_back(n);
        for (const auto& n : nodes)
            if (mark[n] == 0) visit(n);
    }

    // ---- spec.reference_in_history ----------------------------------------------

    void history() {
        for (std::size_t i = 0; i < c_.spec.sources.size(); ++i) {
            const Source& s = c_.spec.sources[i];
            if (s.transforms.empty() || s.transforms.front().op != TransformOp::window || !s.from) continue;
            const Source* from = c_.spec.find_source(*s.from);
            if (!from) continue;
            for (const Transform& t : from->transforms)
                if (!t.references.empty()) {
                    c_.sink.error("spec.reference_in_history",
                                  "The window reads the kept steps of " + quote(from->id) +
                                      ", whose own steps read a reference; read the source before those steps.",
                                  index_path(index_path("source", i) + ".transform", 0));
                    break;
                }
        }
    }
};

}  // namespace

void validate_references(const AdditionsCheck& c) { References(c).run(); }

}  // namespace caliper_builder::spec::detail
