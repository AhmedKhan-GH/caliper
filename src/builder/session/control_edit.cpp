// The edits of the spec's controls (spec_edit.h, design/instruments.md,
// Control Commands and Control Layout).
//
// control.create, control.patch and control.remove follow the rules of the
// panel edits: they work on a copy of the spec, refuse an edit whose own rule
// fails (control.exists, control.not_found) and hand the result to the
// validator, whose first error refuses the edit and whose warnings are kept.
// A control is a leaf of the layout, as a panel is. Without `place`, a new
// control goes into the row of controls under the whole layout: the last
// child of a vertical root split that holds only control leaves, or a new
// row when there is none. An edit that adds a control to a version 0 spec
// raises the version to 1 in the same edit, so the edit and the number are
// one undo step.
#include <caliper_builder/session/spec_edit.h>

#include <caliper_builder/spec/instrument.h>

#include <cmath>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;
using spec::Control;
using spec::LayoutNode;

// Layout trees of a valid spec are at most 8 nodes deep; this bounds the
// recursion of the helpers below for any tree.
constexpr int kMaxWalkDepth = 300;

Error fail(std::string code, std::string message, std::string where = {}) {
    return Error(std::move(code), std::move(message), std::move(where));
}

SpecEdit refuse(Error err) {
    SpecEdit out;
    out.error = std::move(err);
    return out;
}

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

// Rounds a share to 12 decimals, so that 1 - 0.4 is written as 0.6.
double tidy_ratio(double r) { return std::round(r * 1e12) / 1e12; }

bool is_control_leaf(const LayoutNode& node, const spec::Spec& s) {
    return node.kind == LayoutNode::Kind::panel && s.find_control(node.panel) != nullptr;
}

bool layout_places(const LayoutNode& node, const std::string& id, int depth = 0) {
    if (depth > kMaxWalkDepth) return false;
    if (node.kind == LayoutNode::Kind::panel) return node.panel == id;
    for (const auto& c : node.children)
        if (layout_places(c, id, depth + 1)) return true;
    return false;
}

// Removes the leaf of `id`. A split left with one child is replaced by that
// child, and a split or tabs left with none is removed itself.
bool remove_leaf(LayoutNode& node, const std::string& id, bool* remove_self, int depth = 0) {
    *remove_self = false;
    if (depth > kMaxWalkDepth) return false;
    if (node.kind == LayoutNode::Kind::panel) {
        if (node.panel != id) return false;
        *remove_self = true;
        return true;
    }
    for (std::size_t i = 0; i < node.children.size(); ++i) {
        bool drop = false;
        if (!remove_leaf(node.children[i], id, &drop, depth + 1)) continue;
        if (drop) node.children.erase(node.children.begin() + static_cast<std::ptrdiff_t>(i));
        if (node.children.empty()) {
            *remove_self = true;
        } else if (node.kind == LayoutNode::Kind::split && node.children.size() == 1) {
            LayoutNode only = std::move(node.children[0]);
            if (only.kind != LayoutNode::Kind::panel && !only.title && node.title) only.title = node.title;
            node = std::move(only);
        } else if (node.kind == LayoutNode::Kind::split && node.children.size() != 2) {
            node.ratio.reset();
        }
        return true;
    }
    return false;
}

// Puts a new leaf for `id` next to the leaf of `target`, as panel.create's
// `place` does.
bool place_next_to(LayoutNode& node, const std::string& target, const std::string& id, const std::string& where,
                   double share, int depth = 0) {
    if (depth > kMaxWalkDepth) return false;
    if (node.kind == LayoutNode::Kind::panel) {
        if (node.panel != target) return false;
        LayoutNode old = std::move(node);
        LayoutNode wrap;
        if (where == "tabbed") {
            wrap.kind = LayoutNode::Kind::tabs;
            wrap.children.push_back(std::move(old));
            wrap.children.push_back(LayoutNode::of_panel(id));
        } else {
            wrap.kind = LayoutNode::Kind::split;
            wrap.split = where == "beside" ? spec::SplitDirection::horizontal : spec::SplitDirection::vertical;
            const bool new_first = where == "above";
            if (new_first) {
                wrap.children.push_back(LayoutNode::of_panel(id));
                wrap.children.push_back(std::move(old));
            } else {
                wrap.children.push_back(std::move(old));
                wrap.children.push_back(LayoutNode::of_panel(id));
            }
            const double first = tidy_ratio(new_first ? share : 1.0 - share);
            if (std::fabs(first - 0.5) > 1e-12) wrap.ratio = first;
        }
        node = std::move(wrap);
        return true;
    }
    if (node.kind == LayoutNode::Kind::tabs && where == "tabbed") {
        for (std::size_t i = 0; i < node.children.size(); ++i) {
            const LayoutNode& c = node.children[i];
            if (c.kind == LayoutNode::Kind::panel && c.panel == target) {
                node.children.insert(node.children.begin() + static_cast<std::ptrdiff_t>(i + 1),
                                     LayoutNode::of_panel(id));
                return true;
            }
        }
    }
    for (auto& c : node.children)
        if (place_next_to(c, target, id, where, share, depth + 1)) return true;
    return false;
}

// Whether `node` is a row of controls: a control leaf, or a horizontal split
// whose children are all control leaves.
bool is_control_row(const LayoutNode& node, const spec::Spec& s) {
    if (is_control_leaf(node, s)) return true;
    if (node.kind != LayoutNode::Kind::split || node.split != spec::SplitDirection::horizontal) return false;
    for (const auto& c : node.children)
        if (!is_control_leaf(c, s)) return false;
    return !node.children.empty();
}

}  // namespace

// The layout steps of control.create, with external linkage so that
// tests/test_control_edit.cpp can check them on their own; no header declares
// them.
namespace control_layout {

// The layout the spec draws, with its controls: its own, or the default of
// version 1 (design/instruments.md, Control Layout): the panels as version 0
// places them, then one horizontal split of the controls in declaration
// order, under them in a vertical split. A single control is a leaf.
std::optional<LayoutNode> layout_with_controls(const spec::Spec& s) {
    if (s.layout) return s.layout;
    std::optional<LayoutNode> panels = effective_layout(s);
    // A default of the spec library that already places the controls is the
    // one the runtime draws.
    if (panels)
        for (const auto& c : s.controls)
            if (layout_places(*panels, c.id)) return panels;
    std::optional<LayoutNode> row;
    if (s.controls.size() == 1) {
        row = LayoutNode::of_panel(s.controls[0].id);
    } else if (s.controls.size() > 1) {
        LayoutNode split;
        split.kind = LayoutNode::Kind::split;
        split.split = spec::SplitDirection::horizontal;
        for (const auto& c : s.controls) split.children.push_back(LayoutNode::of_panel(c.id));
        row = std::move(split);
    }
    if (!panels) return row;
    if (!row) return panels;
    LayoutNode root;
    root.kind = LayoutNode::Kind::split;
    root.split = spec::SplitDirection::vertical;
    root.children.push_back(std::move(*panels));
    root.children.push_back(std::move(*row));
    return root;
}

// Puts the leaf of `id` into the row of controls under the whole layout.
LayoutNode add_to_control_row(std::optional<LayoutNode> layout, const spec::Spec& s, const std::string& id) {
    if (!layout) return LayoutNode::of_panel(id);
    LayoutNode root = std::move(*layout);
    LayoutNode* row = nullptr;
    if (is_control_row(root, s)) {
        row = &root;
    } else if (root.kind == LayoutNode::Kind::split && root.split == spec::SplitDirection::vertical &&
               root.children.size() >= 2 && is_control_row(root.children.back(), s)) {
        row = &root.children.back();
    }
    if (row) {
        if (row->kind == LayoutNode::Kind::panel) {
            LayoutNode split;
            split.kind = LayoutNode::Kind::split;
            split.split = spec::SplitDirection::horizontal;
            split.children.push_back(std::move(*row));
            split.children.push_back(LayoutNode::of_panel(id));
            *row = std::move(split);
        } else {
            row->children.push_back(LayoutNode::of_panel(id));
            if (row->children.size() != 2) row->ratio.reset();
        }
        return root;
    }
    LayoutNode wrap;
    wrap.kind = LayoutNode::Kind::split;
    wrap.split = spec::SplitDirection::vertical;
    wrap.children.push_back(std::move(root));
    wrap.children.push_back(LayoutNode::of_panel(id));
    return wrap;
}

}  // namespace control_layout

namespace {

using control_layout::add_to_control_row;
using control_layout::layout_with_controls;

// Validates the edited spec: the first error refuses the edit; otherwise the
// warnings are kept.
SpecEdit finish(spec::Spec next, const spec::SourceResolver& resolver, json result) {
    SpecEdit out;
    spec::ValidationReport report = spec::validate(next, resolver);
    if (Error err = first_spec_error(report.diagnostics)) {
        out.error = std::move(err);
        return out;
    }
    for (auto& d : report.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    out.spec = std::move(next);
    out.result = std::move(result);
    return out;
}

// Sets one option from a JSON value: a number or a boolean sets it, null
// clears it. False for any other value.
bool set_number(std::optional<double>& slot, const json& v) {
    if (v.is_null()) {
        slot.reset();
        return true;
    }
    if (!v.is_number()) return false;
    slot = v.get<double>();
    return true;
}

bool set_bool(std::optional<bool>& slot, const json& v) {
    if (v.is_null()) {
        slot.reset();
        return true;
    }
    if (!v.is_boolean()) return false;
    slot = v.get<bool>();
    return true;
}

// Applies an options object to `options`: each key sets or, with null,
// clears its option. Refuses an unknown key or a value of the wrong type.
Error merge_options(spec::ControlOptions& options, const json& patch, const std::string& where) {
    if (patch.is_null()) {
        options = spec::ControlOptions{};
        return {};
    }
    if (!patch.is_object()) return fail("spec.wrong_type", "options is a table.", where);
    for (const auto& [key, v] : patch.items()) {
        bool ok = false;
        if (key == "min") ok = set_number(options.min, v);
        else if (key == "max") ok = set_number(options.max, v);
        else if (key == "step") ok = set_number(options.step, v);
        else if (key == "value") ok = set_number(options.value, v);
        else if (key == "log") ok = set_bool(options.log, v);
        else return fail("spec.unknown_option", "A control has no option " + key + ".", where + "." + key);
        if (!ok)
            return fail("spec.wrong_type",
                        "The option " + key + (key == "log" ? " is true or false." : " is a number."), where + "." + key);
    }
    return {};
}

Error set_kind(Control& c, const json& v, const std::string& where) {
    const auto kind = v.is_string() ? spec::control_kind_from_name(v.get<std::string>()) : std::nullopt;
    if (!kind)
        return fail("spec.unknown_control_kind",
                    "A control's kind is slider, button, range, toggle, picker or text.", where);
    c.kind = *kind;
    return {};
}

std::size_t control_index(const spec::Spec& s, const std::string& id) {
    for (std::size_t i = 0; i < s.controls.size(); ++i)
        if (s.controls[i].id == id) return i;
    return s.controls.size();
}

void raise_version(spec::Spec& s) {
    if (s.version < spec::kInstrumentVersion) s.version = spec::kInstrumentVersion;
}

// A control as the result of control.patch shows it: id, kind, title (null
// when absent), bind and options (an empty object when none), without `x`.
json control_result_json(const Control& c) {
    json options = json::object();
    if (c.options.min) options["min"] = *c.options.min;
    if (c.options.max) options["max"] = *c.options.max;
    if (c.options.step) options["step"] = *c.options.step;
    if (c.options.log) options["log"] = *c.options.log;
    if (c.options.value) options["value"] = *c.options.value;
    return json{{"id", c.id},
                {"kind", spec::control_kind_name(c.kind)},
                {"title", c.title ? json(*c.title) : json()},
                {"bind", c.bind},
                {"options", std::move(options)}};
}

SpecEdit control_create(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    if (s.find_control(id))
        return refuse(fail("control.exists", "The spec already has a control " + id + "; choose another id.", "args.id"));
    if (s.find_panel(id))
        return refuse(fail("spec.duplicate_id",
                           "The spec has a panel " + id + ", and panels and controls share one set of ids; choose "
                           "another id.",
                           "args.id"));
    Control c;
    c.id = id;
    if (Error e = set_kind(c, args.contains("kind") ? args["kind"] : json(), "args.kind")) return refuse(std::move(e));
    c.bind = str_arg(args, "bind");
    if (args.contains("title") && args["title"].is_string()) c.title = args["title"].get<std::string>();
    if (args.contains("options"))
        if (Error e = merge_options(c.options, args["options"], "args.options")) return refuse(std::move(e));

    spec::Spec next = s;
    raise_version(next);
    std::optional<LayoutNode> layout = layout_with_controls(s);
    next.controls.push_back(std::move(c));
    if (args.contains("place") && args["place"].is_object()) {
        const json& place = args["place"];
        const std::string target = str_arg(place, "target");
        const std::string where = str_arg(place, "where");
        const double share = place.contains("ratio") && place["ratio"].is_number() ? place["ratio"].get<double>() : 0.5;
        if (!layout || !layout_places(*layout, target))
            return refuse(fail(s.find_control(target) ? "control.not_found" : "panel.not_found",
                               "The layout has no panel or control " + target + " to place the new control next to.",
                               "args.place.target"));
        place_next_to(*layout, target, id, where, share);
        next.layout = std::move(layout);
    } else {
        next.layout = add_to_control_row(std::move(layout), next, id);
    }
    return finish(std::move(next), resolver, json{{"id", id}});
}

SpecEdit control_patch(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const std::size_t index = control_index(s, id);
    if (index == s.controls.size()) return refuse(fail("control.not_found", "The spec has no control " + id + ".", "args.id"));
    const json patch = args.contains("patch") && args["patch"].is_object() ? args["patch"] : json::object();
    spec::Spec next = s;
    Control& c = next.controls[index];
    for (const auto& [key, v] : patch.items()) {
        const std::string where = "args.patch." + key;
        if (key == "kind") {
            if (Error e = set_kind(c, v, where)) return refuse(std::move(e));
        } else if (key == "title") {
            if (v.is_null()) c.title.reset();
            else if (v.is_string()) c.title = v.get<std::string>();
            else return refuse(fail("spec.wrong_type", "A control's title is text.", where));
        } else if (key == "bind") {
            if (!v.is_string()) return refuse(fail("spec.wrong_type", "A control's bind is a target string.", where));
            c.bind = v.get<std::string>();
        } else if (key == "options") {
            if (Error e = merge_options(c.options, v, where)) return refuse(std::move(e));
        } else if (key == "id") {
            return refuse(fail("control.id_immutable", "A control's id cannot change; remove it and create another.", where));
        } else {
            return refuse(fail("spec.unknown_key", "A control has no key " + key + ".", where));
        }
    }
    raise_version(next);
    json result{{"control", control_result_json(c)}};
    return finish(std::move(next), resolver, std::move(result));
}

SpecEdit control_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const std::size_t index = control_index(s, id);
    if (index == s.controls.size()) return refuse(fail("control.not_found", "The spec has no control " + id + ".", "args.id"));
    spec::Spec next = s;
    next.controls.erase(next.controls.begin() + static_cast<std::ptrdiff_t>(index));
    // A spec without a layout keeps the default, which places what is left.
    if (next.layout) {
        bool drop = false;
        remove_leaf(*next.layout, id, &drop);
        if (drop) next.layout.reset();
    }
    return finish(std::move(next), resolver, json{{"id", id}});
}

}  // namespace

SpecEdit apply_control_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                            const spec::SourceResolver& resolver) {
    if (name == "control.create") return control_create(spec, args, resolver);
    if (name == "control.patch") return control_patch(spec, args, resolver);
    if (name == "control.remove") return control_remove(spec, args, resolver);
    return refuse(fail("command.unknown", "This is not a command that edits the spec's controls.", std::string(name)));
}

}  // namespace caliper_builder
